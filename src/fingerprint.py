"""Build each WR's movement fingerprint and test whether movement style is real.

1. Who ran this route? A classifier guesses WHICH receiver ran a route from that
   route's movement features alone. Train/test is split by game
   (GroupShuffleSplit on gameId), repeated 5 times, so no test game is ever seen
   in training.
2. Signature Score: per-WR top-5 recall on unseen games, averaged over the 5
   splits. Players are grouped into terciles (distinctive / typical / generic);
   the exact order inside a tier is noise and should not be read as a ranking.
3. Teammates-only test: among WRs on the same team (same QB, same playbook),
   how often is the right one ranked first? Separates player from offense.
4. Split-half fingerprint test (the headline): build each WR's fingerprint from
   a random half of his games and from the other half. Is the nearest half-B
   fingerprint (cosine) to his half-A fingerprint his own?
5. Feature importance: permutation importance on held-out games.
6. Fingerprint: per-WR mean, 25th and 75th percentile of each feature (NaN
   ignored) plus cut_rate, standardized with StandardScaler.

Input:  outputs/route_features.parquet, data/data/players.csv
Output: outputs/wr_fingerprints.csv       standardized fingerprint (for matching)
        outputs/wr_fingerprints_raw.csv   same, in real units (for charts)
        outputs/signature_scores.csv      per-WR Signature Score + tier
        outputs/classifier_summary.csv    every test vs chance
        outputs/feature_importance.csv

Run:  .venv/bin/python src/fingerprint.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
FEATURES_FILE = OUT_DIR / "route_features.parquet"
PLAYERS_FILE = ROOT / "data" / "data" / "players.csv"

MIN_ROUTES = 80
TEST_SIZE = 0.25
N_SPLITS = 5             # repeated game splits for the classifier
MIN_TEST_ROUTES = 20     # a Signature Score needs enough test routes to mean anything
HALF_REPEATS = 20        # random game halves for the split-half test
MIN_HALF_ROUTES = 20     # a half-season fingerprint needs at least this many routes
TOP_K = 5
SEED = 0
TIERS = ["generic", "typical", "distinctive"]

FEATURES = ["release_speed", "release_accel", "top_speed", "avg_speed", "max_decel",
            "cut_sharpness", "speed_kept_in_cut", "has_cut", "straightness", "route_time"]
# has_cut is 0/1, so its percentiles are meaningless; its mean becomes cut_rate.
CONTINUOUS = [c for c in FEATURES if c != "has_cut"]
INFO = ["nflId", "displayName", "team", "n_routes"]


def load_routes():
    """Route features for WRs with at least MIN_ROUTES routes, with names attached."""
    f = pd.read_parquet(FEATURES_FILE)
    f["team"] = f["team"].astype(str)
    counts = f["nflId"].value_counts()
    f = f[f["nflId"].isin(counts[counts >= MIN_ROUTES].index)].reset_index(drop=True)
    names = pd.read_csv(PLAYERS_FILE, usecols=["nflId", "displayName"])
    return f.merge(names, on="nflId", how="left")


# --------------------------------------------------------------- fingerprint
def raw_fingerprint(f):
    """Per WR, in real units: mean, p25, p75 of each continuous feature + cut_rate."""
    g = f.groupby("nflId")
    raw = pd.concat([g[CONTINUOUS].mean().add_suffix("_mean"),
                     g[CONTINUOUS].quantile(0.25).add_suffix("_p25"),
                     g[CONTINUOUS].quantile(0.75).add_suffix("_p75"),
                     g["has_cut"].mean().rename("cut_rate")], axis=1)
    return raw[sorted(raw.columns, key=lambda c: (c.rsplit("_", 1)[0], c))]


def standardize(*raws):
    """Z-score one or more fingerprint tables with one shared scaler.

    A missing value (e.g. no cut routes in a half) becomes 0 = league average.
    """
    scaler = StandardScaler().fit(pd.concat(raws))
    out = [pd.DataFrame(scaler.transform(r), index=r.index, columns=r.columns).fillna(0)
           for r in raws]
    return out if len(out) > 1 else out[0]


def player_info(f):
    return f.groupby("nflId").agg(displayName=("displayName", "first"),
                                  team=("team", lambda t: t.mode().iat[0]),
                                  n_routes=("playId", "size"))


def build_fingerprints(f):
    raw = raw_fingerprint(f)
    z = standardize(raw)
    info = player_info(f)
    order = lambda t: info.join(t).reset_index().sort_values("n_routes", ascending=False)
    return order(raw), order(z)


def split_half_fingerprints(f, rng):
    """Standardized fingerprints from two random, disjoint halves of each WR's games."""
    games = f[["nflId", "gameId"]].drop_duplicates()
    games["half"] = rng.integers(0, 2, len(games))
    x = f.merge(games, on=["nflId", "gameId"])
    n = x.groupby(["nflId", "half"]).size().unstack(fill_value=0)
    ids = n[(n[0] >= MIN_HALF_ROUTES) & (n[1] >= MIN_HALF_ROUTES)].index
    x = x[x["nflId"].isin(ids)]
    return standardize(raw_fingerprint(x[x["half"] == 0]), raw_fingerprint(x[x["half"] == 1]))


def split_half_test(f, drop=()):
    """Is a WR's half-A fingerprint closest (cosine) to his own half-B fingerprint?

    drop: feature names to leave out of the fingerprint (ablation), e.g. ["route_time"].
    "has_cut" removes cut_rate. The default uses every column.
    """
    rng = np.random.default_rng(SEED)
    team = f.groupby("nflId")["team"].agg(lambda t: t.mode().iat[0])
    drop = {"cut" if c == "has_cut" else c for c in drop}
    rows = []
    for _ in range(HALF_REPEATS):
        a, b = split_half_fingerprints(f, rng)
        keep = [c for c in a.columns if c.rsplit("_", 1)[0] not in drop]
        a, b = a[keep], b[keep]
        sim = cosine_similarity(a, b)                         # rows: A, cols: B
        rank = (sim > np.diag(sim)[:, None]).sum(axis=1)      # 0 = himself is closest
        t = team.loc[a.index].to_numpy()
        same = t[:, None] == t[None]
        team_pick = np.where(same, sim, -np.inf).argmax(axis=1)
        rows.append({"n": len(a), "top1": (rank == 0).mean(), "top5": (rank < TOP_K).mean(),
                     "team": (team_pick == np.arange(len(a))).mean(),
                     "team_chance": (1 / same.sum(axis=1)).mean()})
    r = pd.DataFrame(rows)
    m, s = r.mean(), r.std()
    out = pd.DataFrame([
        {"test": "Split-half fingerprint, top-1", "accuracy": m["top1"], "sd": s["top1"],
         "chance": 1 / m["n"]},
        {"test": f"Split-half fingerprint, top-{TOP_K}", "accuracy": m["top5"],
         "sd": s["top5"], "chance": TOP_K / m["n"]},
        {"test": "Split-half fingerprint, teammates only", "accuracy": m["team"],
         "sd": s["team"], "chance": m["team_chance"]},
    ])
    out["x_chance"] = out["accuracy"] / out["chance"]
    return out, int(round(m["n"]))


# ---------------------------------------------------------------- classifier
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
    """Per test route: top-1 hit, top-5 hit, teammates-only hit, teammate chance."""
    proba = model.predict_proba(test[FEATURES])
    classes = model.classes_
    y_idx = np.searchsorted(classes, test["nflId"].to_numpy())
    true_p = proba[np.arange(len(test)), y_idx]
    rank = (proba > true_p[:, None]).sum(axis=1)          # 0 = ranked first

    # Teammates only: rank only the WRs who played for this route's team.
    team_hit = np.zeros(len(test), dtype=bool)
    team_n = np.zeros(len(test))
    for team, ids in test.groupby("team")["nflId"].unique().items():
        rows = np.flatnonzero(test["team"].to_numpy() == team)
        cand = np.searchsorted(classes, np.sort(ids))
        team_hit[rows] = cand[proba[np.ix_(rows, cand)].argmax(axis=1)] == y_idx[rows]
        team_n[rows] = len(cand)

    return pd.DataFrame({"nflId": test["nflId"], "displayName": test["displayName"],
                         "team": test["team"], "hit": rank == 0, "hit5": rank < TOP_K,
                         "team_hit": team_hit, "team_chance": 1 / team_n})


def summarize(results, n_players):
    """Single-route accuracy vs chance, averaged over the repeated splits."""
    per_split = results.groupby("split").agg(top1=("hit", "mean"), top5=("hit5", "mean"),
                                             team=("team_hit", "mean"),
                                             team_chance=("team_chance", "mean"))
    m, s = per_split.mean(), per_split.std()
    summary = pd.DataFrame([
        {"test": "Single route, all WRs, top-1", "accuracy": m["top1"], "sd": s["top1"],
         "chance": 1 / n_players},
        {"test": f"Single route, all WRs, top-{TOP_K}", "accuracy": m["top5"],
         "sd": s["top5"], "chance": TOP_K / n_players},
        {"test": "Single route, teammates only", "accuracy": m["team"], "sd": s["team"],
         "chance": m["team_chance"]},
    ])
    summary["x_chance"] = summary["accuracy"] / summary["chance"]
    return summary


def signature_scores(results, n_players):
    """Signature Score = top-5 recall on unseen games, averaged over the splits."""
    per_split = (results.groupby(["nflId", "split"])
                 .agg(routes=("hit5", "size"), top5=("hit5", "mean")).reset_index())
    scores = (per_split.groupby("nflId")
              .agg(test_routes=("routes", "sum"), signature_score=("top5", "mean"),
                   splits=("split", "nunique")))
    info = results.groupby("nflId").agg(displayName=("displayName", "first"),
                                        team=("team", lambda t: t.mode().iat[0]))
    scores = info.join(scores).reset_index()
    scores["chance"] = TOP_K / n_players
    scores["x_chance"] = scores["signature_score"] / scores["chance"]

    ok = scores["test_routes"] >= MIN_TEST_ROUTES
    scores["tier"] = "too few routes"
    scores.loc[ok, "tier"] = pd.qcut(scores.loc[ok, "signature_score"].rank(method="first"),
                                     3, labels=TIERS).astype(str)
    return scores.sort_values(["tier", "displayName"])


def feature_importance(model, test):
    """Rise in test log-loss when one feature is shuffled (higher = more useful)."""
    def neg_log_loss(est, X, y):
        return -log_loss(y, est.predict_proba(X), labels=est.classes_)

    r = permutation_importance(model, test[FEATURES], test["nflId"], scoring=neg_log_loss,
                               n_repeats=5, random_state=SEED, n_jobs=-1)
    return (pd.DataFrame({"feature": FEATURES, "importance": r.importances_mean,
                          "std": r.importances_std})
            .sort_values("importance", ascending=False))


# ---------------------------------------------------------------------- main
def print_tiers(scores):
    for tier in reversed(TIERS):
        t = scores[scores["tier"] == tier]
        lo, hi = t["x_chance"].min(), t["x_chance"].max()
        print(f"\n{tier.upper()} ({len(t)} WRs, {lo:.1f}x-{hi:.1f}x chance), alphabetical:")
        names = [f"{r.displayName} ({r.team})" for r in t.itertuples()]
        for i in range(0, len(names), 6):
            print("  " + ", ".join(names[i:i + 6]))


def main():
    pd.set_option("display.width", 200)
    fmt = lambda v: f"{v:.3f}"
    f = load_routes()
    n_players = f["nflId"].nunique()
    print(f"WRs with >= {MIN_ROUTES} routes: {n_players}   routes: {len(f):,}")

    raw, z = build_fingerprints(f)

    print(f"\nSplit-half fingerprint test ({HALF_REPEATS} random game halves)")
    half, n_half = split_half_test(f)
    print(f"  WRs with >= {MIN_HALF_ROUTES} routes in both halves: ~{n_half}")

    print(f"\nWho ran this route? ({N_SPLITS} game splits)")
    results, first = [], None
    for i, (train, test) in enumerate(game_splits(f), 1):
        model = train_model(train)
        results.append(score_routes(model, test).assign(split=i))
        print(f"  split {i}: train {train['gameId'].nunique()} games, "
              f"test {test['gameId'].nunique()} games / {len(test):,} routes")
        first = first or (model, test)
    results = pd.concat(results, ignore_index=True)

    summary = pd.concat([half, summarize(results, n_players)], ignore_index=True)
    print("\nAll tests on unseen games vs chance:")
    print(summary.to_string(index=False, float_format=fmt))

    scores = signature_scores(results, n_players)
    print(f"\nSignature Score = top-5 recall on unseen games, mean of {N_SPLITS} splits "
          f"(chance {TOP_K / n_players:.3f}).")
    print("Tiers are terciles. Order inside a tier is noise, so names are alphabetical.")
    print_tiers(scores)

    imp = feature_importance(*first)
    print("\nPermutation importance (test log-loss increase when shuffled):")
    print(imp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    z.to_csv(OUT_DIR / "wr_fingerprints.csv", index=False)
    raw.to_csv(OUT_DIR / "wr_fingerprints_raw.csv", index=False)
    scores.to_csv(OUT_DIR / "signature_scores.csv", index=False)
    summary.to_csv(OUT_DIR / "classifier_summary.csv", index=False)
    imp.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    print(f"\nFingerprint: {len(z)} WRs x {z.shape[1] - len(INFO)} standardized columns")
    print("Saved -> outputs/wr_fingerprints.csv, wr_fingerprints_raw.csv, "
          "signature_scores.csv, classifier_summary.csv, feature_importance.csv")


if __name__ == "__main__":
    main()
