"""Build each WR's movement fingerprint and test whether movement style is real.

1. Validation: a classifier guesses WHICH receiver ran a route from that
   route's movement features alone. Train/test is split by game
   (GroupShuffleSplit on gameId), so no game appears on both sides. The split
   is repeated 5 times so every WR gets enough test routes.
2. Signature Score: per-WR recall on unseen games (how often his routes are
   correctly attributed to him), pooled over the 5 splits. High = distinctive.
3. Teammates-only test: among WRs on the same team (same QB, same playbook),
   how often is the right one ranked first? Separates player from offense.
4. Feature importance: permutation importance on the test set.
5. Fingerprint: per-WR mean, 25th and 75th percentile of each feature (NaN
   ignored) plus cut_rate, standardized with StandardScaler.

Input:  outputs/route_features.parquet, data/data/players.csv
Output: outputs/wr_fingerprints.csv       standardized fingerprint (for matching)
        outputs/wr_fingerprints_raw.csv   same, in real units (for charts)
        outputs/signature_scores.csv      per-WR recall
        outputs/classifier_summary.csv    accuracy vs chance
        outputs/feature_importance.csv

Run:  .venv/bin/python src/fingerprint.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
FEATURES_FILE = OUT_DIR / "route_features.parquet"
PLAYERS_FILE = ROOT / "data" / "data" / "players.csv"

MIN_ROUTES = 80
TEST_SIZE = 0.25
N_SPLITS = 5             # repeated game splits; one split leaves ~20 WRs with no test games
MIN_TEST_ROUTES = 20     # Signature Score needs enough test routes to mean something
TOP_K = 5
SEED = 0

FEATURES = ["release_speed", "release_accel", "top_speed", "avg_speed", "max_decel",
            "cut_sharpness", "speed_kept_in_cut", "has_cut", "straightness", "route_time"]
# has_cut is 0/1, so its percentiles are meaningless; its mean becomes cut_rate.
CONTINUOUS = [c for c in FEATURES if c != "has_cut"]


def load_routes():
    """Route features for WRs with at least MIN_ROUTES routes, with names attached."""
    f = pd.read_parquet(FEATURES_FILE)
    f["team"] = f["team"].astype(str)
    counts = f["nflId"].value_counts()
    f = f[f["nflId"].isin(counts[counts >= MIN_ROUTES].index)].reset_index(drop=True)
    names = pd.read_csv(PLAYERS_FILE, usecols=["nflId", "displayName"])
    return f.merge(names, on="nflId", how="left")


# ---------------------------------------------------------------- validation
def game_splits(f):
    """N_SPLITS random 75/25 train/test splits, each with no game on both sides."""
    split = GroupShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=SEED)
    for tr, te in split.split(f, groups=f["gameId"]):
        assert not set(f.loc[tr, "gameId"]) & set(f.loc[te, "gameId"])
        yield f.loc[tr].reset_index(drop=True), f.loc[te].reset_index(drop=True)


def train_model(train):
    model = HistGradientBoostingClassifier(
        learning_rate=0.1, max_iter=200, max_leaf_nodes=15, l2_regularization=1.0,
        early_stopping=True, random_state=SEED,
        class_weight="balanced")  # otherwise it wins by guessing high-volume WRs
    return model.fit(train[FEATURES], train["nflId"])


def score_routes(model, test):
    """Per test route: top-1 hit, top-5 hit, teammates-only hit, teammate count."""
    proba = model.predict_proba(test[FEATURES])
    classes = model.classes_
    y_idx = np.searchsorted(classes, test["nflId"].to_numpy())
    true_p = proba[np.arange(len(test)), y_idx]
    rank = (proba > true_p[:, None]).sum(axis=1)          # 0 = ranked first
    top1, top5 = rank == 0, rank < TOP_K

    # Teammates only: rank only the WRs who played for this route's team.
    team_hit = np.zeros(len(test), dtype=bool)
    team_n = np.zeros(len(test))
    for team, ids in test.groupby("team")["nflId"].unique().items():
        rows = np.flatnonzero(test["team"].to_numpy() == team)
        cand = np.searchsorted(classes, np.sort(ids))
        team_hit[rows] = cand[proba[np.ix_(rows, cand)].argmax(axis=1)] == y_idx[rows]
        team_n[rows] = len(cand)

    return pd.DataFrame({"nflId": test["nflId"], "displayName": test["displayName"],
                         "team": test["team"], "hit": top1, "hit5": top5,
                         "team_hit": team_hit, "team_chance": 1 / team_n})


def summarize(results, n_players):
    """Accuracy vs chance, averaged over the repeated splits."""
    per_split = results.groupby("split").agg(top1=("hit", "mean"), top5=("hit5", "mean"),
                                             team=("team_hit", "mean"),
                                             team_chance=("team_chance", "mean"))
    m, s = per_split.mean(), per_split.std()
    summary = pd.DataFrame([
        {"test": "All WRs, top-1", "accuracy": m["top1"], "sd": s["top1"],
         "chance": 1 / n_players},
        {"test": f"All WRs, top-{TOP_K}", "accuracy": m["top5"], "sd": s["top5"],
         "chance": TOP_K / n_players},
        {"test": "Teammates only, top-1", "accuracy": m["team"], "sd": s["team"],
         "chance": m["team_chance"]},
    ])
    summary["x_chance"] = summary["accuracy"] / summary["chance"]
    return summary


def signature_scores(results):
    """Per-WR recall pooled over all splits in which he was in the test set."""
    scores = (results.groupby(["nflId", "displayName"], as_index=False)
              .agg(team=("team", lambda t: t.mode().iat[0]),
                   test_routes=("hit", "size"), signature_score=("hit", "mean"),
                   top5_recall=("hit5", "mean"), teammate_recall=("team_hit", "mean"),
                   teammate_chance=("team_chance", "mean")))
    scores["reliable"] = scores["test_routes"] >= MIN_TEST_ROUTES
    return scores.sort_values(["reliable", "signature_score", "top5_recall"],
                              ascending=False)


def feature_importance(model, test):
    """Rise in test log-loss when one feature is shuffled (higher = more useful)."""
    def neg_log_loss(est, X, y):
        return -log_loss(y, est.predict_proba(X), labels=est.classes_)

    r = permutation_importance(model, test[FEATURES], test["nflId"], scoring=neg_log_loss,
                               n_repeats=5, random_state=SEED, n_jobs=-1)
    return (pd.DataFrame({"feature": FEATURES, "importance": r.importances_mean,
                          "std": r.importances_std})
            .sort_values("importance", ascending=False))


# --------------------------------------------------------------- fingerprint
def build_fingerprints(f):
    """Per WR: mean, p25, p75 of each continuous feature (NaN skipped) + cut_rate."""
    g = f.groupby("nflId")
    parts = [g[CONTINUOUS].mean().add_suffix("_mean"),
             g[CONTINUOUS].quantile(0.25).add_suffix("_p25"),
             g[CONTINUOUS].quantile(0.75).add_suffix("_p75"),
             g["has_cut"].mean().rename("cut_rate")]
    raw = pd.concat(parts, axis=1)
    raw = raw[sorted(raw.columns, key=lambda c: (c.rsplit("_", 1)[0], c))]
    info = g.agg(displayName=("displayName", "first"),
                 team=("team", lambda t: t.mode().iat[0]),
                 n_routes=("playId", "size"))

    z = pd.DataFrame(StandardScaler().fit_transform(raw), index=raw.index, columns=raw.columns)
    raw = info.join(raw).reset_index().sort_values("n_routes", ascending=False)
    z = info.join(z).reset_index().sort_values("n_routes", ascending=False)
    return raw, z


def main():
    pd.set_option("display.width", 200)
    fmt = lambda v: f"{v:.3f}"
    f = load_routes()
    print(f"WRs with >= {MIN_ROUTES} routes: {f['nflId'].nunique()}   routes: {len(f):,}")

    results, first = [], None
    for i, (train, test) in enumerate(game_splits(f), 1):
        model = train_model(train)
        results.append(score_routes(model, test).assign(split=i))
        print(f"  split {i}/{N_SPLITS}: train {train['gameId'].nunique()} games / "
              f"{len(train):,} routes, test {test['gameId'].nunique()} games / "
              f"{len(test):,} routes, {model.n_iter_} boosting rounds")
        if first is None:
            first = (model, test)
    results = pd.concat(results, ignore_index=True)

    summary = summarize(results, f["nflId"].nunique())
    print(f"\nWho ran this route? (unseen games, mean of {N_SPLITS} game splits)")
    print(summary.to_string(index=False, float_format=fmt))

    scores = signature_scores(results)
    ok = scores[scores["reliable"]]
    cols = ["displayName", "team", "test_routes", "signature_score", "top5_recall",
            "teammate_recall", "teammate_chance"]
    print(f"\nSignature Score = per-WR recall on unseen games "
          f"({len(ok)} WRs with >= {MIN_TEST_ROUTES} test routes; chance = {1/len(scores):.3f})")
    print("10 MOST identifiable WRs")
    print(ok[cols].head(10).to_string(index=False, float_format=fmt))
    print("10 LEAST identifiable WRs")
    print(ok[cols].tail(10).to_string(index=False, float_format=fmt))

    imp = feature_importance(*first)
    print("\nPermutation importance (test log-loss increase when shuffled):")
    print(imp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    raw, z = build_fingerprints(f)
    z.to_csv(OUT_DIR / "wr_fingerprints.csv", index=False)
    raw.to_csv(OUT_DIR / "wr_fingerprints_raw.csv", index=False)
    scores.to_csv(OUT_DIR / "signature_scores.csv", index=False)
    summary.to_csv(OUT_DIR / "classifier_summary.csv", index=False)
    imp.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    print(f"\nFingerprint: {len(z)} WRs x {z.shape[1] - 4} standardized columns")
    print("Saved -> outputs/wr_fingerprints.csv, wr_fingerprints_raw.csv, "
          "signature_scores.csv, classifier_summary.csv, feature_importance.csv")


if __name__ == "__main__":
    main()
