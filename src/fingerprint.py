"""Build each WR's movement fingerprint and test whether it is real.

1. Fingerprint: per-WR average of the route features (NaN-aware), plus
   cut_rate = share of routes with a real cut.
2. "Who ran this route?": a classifier guesses the receiver from one route's
   movement alone. Cross-validation is split by game, so the model never sees
   the test game. Scored against chance.
3. Teammates-only test: among WRs on the same team (same QB, same playbook),
   how often is the right one picked? This separates player style from the
   team's offense.
4. Split-half test: a fingerprint built from half of a WR's games should be
   closest to the fingerprint from his other half.

Input:  outputs/route_features.parquet, data/data/players.csv
Output: outputs/wr_fingerprints.csv, outputs/wr_identifiability.csv,
        outputs/classifier_summary.csv

Run:  .venv/bin/python src/fingerprint.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
FEATURES_FILE = OUT_DIR / "route_features.parquet"
PLAYERS_FILE = ROOT / "data" / "data" / "players.csv"

MIN_ROUTES = 80
N_FOLDS = 5
TOP_K = 5
SEED = 0

FEATURES = ["release_speed", "release_accel", "top_speed", "avg_speed", "max_decel",
            "cut_sharpness", "speed_kept_in_cut", "has_cut", "straightness", "route_time"]
FINGERPRINT = [c for c in FEATURES if c != "has_cut"] + ["cut_rate"]


def load_routes():
    """Route features for WRs with at least MIN_ROUTES routes, with names attached."""
    f = pd.read_parquet(FEATURES_FILE)
    f["team"] = f["team"].astype(str)
    counts = f["nflId"].value_counts()
    f = f[f["nflId"].isin(counts[counts >= MIN_ROUTES].index)].reset_index(drop=True)
    names = pd.read_csv(PLAYERS_FILE, usecols=["nflId", "displayName"])
    return f.merge(names, on="nflId", how="left")


def build_fingerprints(f):
    """One row per WR: NaN-aware mean of each feature, cut_rate, route count, team."""
    fp = f.groupby("nflId")[FEATURES].mean()          # pandas mean skips NaN
    fp = fp.rename(columns={"has_cut": "cut_rate"})
    info = f.groupby("nflId").agg(displayName=("displayName", "first"),
                                  team=("team", lambda t: t.mode().iat[0]),
                                  n_routes=("playId", "size"))
    return info.join(fp[FINGERPRINT]).reset_index().sort_values("n_routes", ascending=False)


def out_of_fold_proba(f):
    """Probability of each WR for every route, from a model that never saw that game."""
    X = f[FEATURES].to_numpy(dtype=float)
    y = f["nflId"].to_numpy()
    classes = np.sort(np.unique(y))
    proba = np.zeros((len(f), len(classes)))

    for fold, (tr, te) in enumerate(GroupKFold(n_splits=N_FOLDS).split(X, y, f["gameId"]), 1):
        model = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=200,
                                               max_leaf_nodes=15, l2_regularization=1.0,
                                               early_stopping=True, random_state=SEED,
                                               # Without this the model wins by guessing
                                               # WRs with the most routes.
                                               class_weight="balanced")
        model.fit(X[tr], y[tr])
        cols = np.searchsorted(classes, model.classes_)
        proba[np.ix_(te, cols)] = model.predict_proba(X[te])
        print(f"  fold {fold}/{N_FOLDS}: {len(te):,} test routes, "
              f"{model.n_iter_} boosting rounds")
    return proba, classes


def evaluate(f, proba, classes):
    """All-WR and teammates-only accuracy vs chance."""
    y_idx = np.searchsorted(classes, f["nflId"].to_numpy())
    rows = np.arange(len(f))

    # All WRs: is the true player ranked 1st / in the top 5?
    true_p = proba[rows, y_idx]
    rank = (proba > true_p[:, None]).sum(axis=1)       # 0 = ranked first
    top1, topk = rank == 0, rank < TOP_K

    # Teammates only: candidates = model WRs who ran routes for this route's team.
    team_of = f.groupby("team")["nflId"].unique()
    team_hit = np.zeros(len(f), dtype=bool)
    team_size = np.zeros(len(f))
    for team, ids in team_of.items():
        r = np.flatnonzero(f["team"].to_numpy() == team)
        cand = np.searchsorted(classes, np.sort(ids))
        best = cand[proba[np.ix_(r, cand)].argmax(axis=1)]
        team_hit[r] = best == y_idx[r]
        team_size[r] = len(cand)

    # Chance levels: uniform guess, and always guessing the busiest WR.
    share = f["nflId"].value_counts(normalize=True)
    busiest_teammate = f.groupby("team")["nflId"].agg(lambda s: s.value_counts().iat[0] / len(s))
    k = len(classes)

    summary = pd.DataFrame([
        {"test": "All WRs, top-1", "accuracy": top1.mean(),
         "chance_uniform": 1 / k, "chance_busiest_wr": share.max()},
        {"test": f"All WRs, top-{TOP_K}", "accuracy": topk.mean(),
         "chance_uniform": TOP_K / k, "chance_busiest_wr": share.nlargest(TOP_K).sum()},
        {"test": "Teammates only, top-1", "accuracy": team_hit.mean(),
         "chance_uniform": (1 / team_size).mean(),
         "chance_busiest_wr": (f["team"].map(busiest_teammate)).mean()},
    ])
    summary["lift_vs_uniform"] = summary["accuracy"] / summary["chance_uniform"]

    per_wr = (pd.DataFrame({"nflId": f["nflId"], "displayName": f["displayName"],
                            "team": f["team"], "top1": top1, f"top{TOP_K}": topk,
                            "teammate_hit": team_hit, "teammates": team_size})
              .groupby(["nflId", "displayName"])
              .agg(team=("team", lambda t: t.mode().iat[0]), n_routes=("top1", "size"),
                   top1=("top1", "mean"), **{f"top{TOP_K}": (f"top{TOP_K}", "mean")},
                   teammate_hit=("teammate_hit", "mean"), teammates=("teammates", "mean"))
              .reset_index())
    per_wr["teammate_chance"] = 1 / per_wr["teammates"]
    per_wr["teammate_lift"] = per_wr["teammate_hit"] / per_wr["teammate_chance"]

    # Same tests averaged per WR, so high-volume stars can't carry the score.
    balanced = pd.DataFrame([
        {"test": "All WRs, top-1 (avg per WR)", "accuracy": per_wr["top1"].mean(),
         "chance_uniform": 1 / k, "chance_busiest_wr": 1 / k},
        {"test": f"All WRs, top-{TOP_K} (avg per WR)", "accuracy": per_wr[f"top{TOP_K}"].mean(),
         "chance_uniform": TOP_K / k, "chance_busiest_wr": TOP_K / k},
        {"test": "Teammates only, top-1 (avg per WR)", "accuracy": per_wr["teammate_hit"].mean(),
         "chance_uniform": per_wr["teammate_chance"].mean(),
         "chance_busiest_wr": per_wr["teammate_chance"].mean()},
    ])
    balanced["lift_vs_uniform"] = balanced["accuracy"] / balanced["chance_uniform"]
    summary = pd.concat([summary, balanced], ignore_index=True)
    return summary, per_wr.sort_values("teammate_lift", ascending=False), team_size


def split_half_test(f, repeats=20):
    """Does a WR's fingerprint from half his games find his other half?

    Each WR's games are split at random into two halves. A fingerprint is
    built from each half, standardized, and half A is matched to the nearest
    half B (Euclidean). Repeated `repeats` times with different splits.
    This tests the fingerprint itself, the thing the doppelganger finder uses.
    """
    rng = np.random.default_rng(SEED)
    team = f.groupby("nflId")["team"].agg(lambda t: t.mode().iat[0])
    rows = []
    for _ in range(repeats):
        games = f[["nflId", "gameId"]].drop_duplicates()
        games["half"] = rng.integers(0, 2, len(games))
        x = f.merge(games, on=["nflId", "gameId"])
        a = x[x["half"] == 0].groupby("nflId")[FEATURES].mean()
        b = x[x["half"] == 1].groupby("nflId")[FEATURES].mean()
        ids = a.index.intersection(b.index)
        both = pd.concat([a.loc[ids], b.loc[ids]])
        mu, sd = both.mean(), both.std()
        za = ((a.loc[ids] - mu) / sd).fillna(0).to_numpy()
        zb = ((b.loc[ids] - mu) / sd).fillna(0).to_numpy()

        dist = np.linalg.norm(za[:, None] - zb[None], axis=2)
        rank = (dist < np.diag(dist)[:, None]).sum(axis=1)       # 0 = self is nearest
        t = team.loc[ids].to_numpy()
        same_team = t[:, None] == t[None]
        team_pick = np.where(same_team, dist, np.inf).argmin(axis=1)
        rows.append({"n": len(ids), "top1": (rank == 0).mean(), "top5": (rank < TOP_K).mean(),
                     "team_top1": (team_pick == np.arange(len(ids))).mean(),
                     "team_chance": (1 / same_team.sum(axis=1)).mean()})
    r = pd.DataFrame(rows).mean()
    out = pd.DataFrame([
        {"test": "Split-half fingerprint, top-1", "accuracy": r["top1"],
         "chance_uniform": 1 / r["n"], "chance_busiest_wr": 1 / r["n"]},
        {"test": f"Split-half fingerprint, top-{TOP_K}", "accuracy": r["top5"],
         "chance_uniform": TOP_K / r["n"], "chance_busiest_wr": TOP_K / r["n"]},
        {"test": "Split-half fingerprint, teammates only", "accuracy": r["team_top1"],
         "chance_uniform": r["team_chance"], "chance_busiest_wr": r["team_chance"]},
    ])
    out["lift_vs_uniform"] = out["accuracy"] / out["chance_uniform"]
    return out


def main():
    f = load_routes()
    print(f"WRs with >= {MIN_ROUTES} routes: {f['nflId'].nunique()}   "
          f"routes: {len(f):,}   games: {f['gameId'].nunique()}")

    fp = build_fingerprints(f)
    fp.to_csv(OUT_DIR / "wr_fingerprints.csv", index=False)

    print(f"\nTraining 'who ran this route?' ({N_FOLDS}-fold, split by game)")
    proba, classes = out_of_fold_proba(f)
    summary, per_wr, team_size = evaluate(f, proba, classes)
    summary = pd.concat([summary, split_half_test(f)], ignore_index=True)
    summary.to_csv(OUT_DIR / "classifier_summary.csv", index=False)
    per_wr.to_csv(OUT_DIR / "wr_identifiability.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nResults (out-of-fold, unseen games):")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nTeammate candidates per route: median {np.median(team_size):.0f}, "
          f"range {team_size.min():.0f}-{team_size.max():.0f}")

    cols = ["displayName", "team", "n_routes", "top1", f"top{TOP_K}",
            "teammate_hit", "teammate_chance", "teammate_lift"]
    fmt = lambda v: f"{v:.2f}"
    print("\nMost identifiable WRs (teammates-only lift over chance):")
    print(per_wr[cols].head(10).to_string(index=False, float_format=fmt))
    print("\nLeast identifiable WRs:")
    print(per_wr[cols].tail(5).to_string(index=False, float_format=fmt))
    print(f"\nSaved -> outputs/wr_fingerprints.csv ({len(fp)} WRs), "
          "outputs/wr_identifiability.csv, outputs/classifier_summary.csv")


if __name__ == "__main__":
    main()
