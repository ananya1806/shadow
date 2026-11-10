"""A small "who ran this route?" model that can run in a web browser.

The main model (fingerprint.py) uses the tracking chip's own speed and
acceleration. A route drawn on the website has only positions, so this module:

1. computes movement features from POSITIONS ONLY (10 frames per second),
   with the exact same steps as docs/assets/route_features.js;
2. trains a multinomial logistic regression on those features for the 110
   fingerprinted WRs (simple enough to export as weights);
3. tests it on unseen games, so the website can state its real accuracy.

route_time is left out: on the website it is a slider, not the player's movement.

Run:  .venv/bin/python src/route_model.py      (prints accuracy vs chance)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import fingerprint as fpr

ROUTES_FILE = fpr.OUT_DIR / "wr_routes.parquet"
FEATURES = ["release_speed", "release_accel", "top_speed", "avg_speed", "max_decel",
            "cut_sharpness", "speed_kept_in_cut", "has_cut", "straightness"]
FPS = 10
RELEASE = 10          # frame 10 = 1.0 s after the snap
WINDOW = 10           # a cut is measured over 1.0 s
CUT_MIN_SPEED = 3.0
CUT_DEGREES = 45
DECEL_FLOOR = -10.0
N_SPLITS, SEED = 5, fpr.SEED


def xy_features(p):
    """Movement features of one route from an (n, 2) array of positions in yards.

    Frame 0 is the snap. Mirrors routeFeatures() in route_features.js line by line.
    """
    n = len(p)
    seg = np.hypot(*np.diff(p, axis=0).T)                      # distance moved each frame
    vec = np.empty((n, 2))                                     # direction of travel
    vec[0], vec[-1], vec[1:-1] = p[1] - p[0], p[-1] - p[-2], (p[2:] - p[:-2]) / 2
    s = np.hypot(*vec.T) * FPS                                 # speed, yd/s
    heading = np.degrees(np.arctan2(vec[:, 1], vec[:, 0]))
    ds = np.diff(s) * FPS                                      # change in speed, yd/s^2

    best, at = -1.0, -1
    for t in range(RELEASE, n - WINDOW):
        if s[t] > CUT_MIN_SPEED:
            turn = abs((heading[t + WINDOW] - heading[t] + 180) % 360 - 180)
            if turn > best:
                best, at = turn, t
    has_cut = float(best > CUT_DEGREES)
    kept = np.nan
    if has_cut:
        kept = min(max(s[at:at + WINDOW + 1].min() / max(s[at - 5], 0.1), 0.0), 1.2)
    path = seg.sum()
    straight = 1.0 if path == 0 else min(np.hypot(*(p[-1] - p[0])) / path, 1.0)
    return [s[RELEASE], ds[:RELEASE].max(), s.max(), s.mean(), max(ds.min(), DECEL_FLOOR),
            best if at >= 0 else np.nan, kept, has_cut, straight]


def route_positions(keys=None):
    """{(gameId, playId, nflId): (n, 2) positions from the snap to the throw}."""
    r = pd.read_parquet(ROUTES_FILE, columns=["gameId", "playId", "nflId", "frameId",
                                               "snapFrame", "endFrame", "x", "y"])
    if keys is not None:
        r = r.merge(keys, on=["gameId", "playId", "nflId"])
    r = r[(r["frameId"] >= r["snapFrame"]) & (r["frameId"] <= r["endFrame"])]
    r = r.sort_values(["gameId", "playId", "nflId", "frameId"])
    return {k: g[["x", "y"]].to_numpy(float) for k, g in r.groupby(["gameId", "playId", "nflId"])}


def build_table():
    """One row of position-only features per route, for the 110 fingerprinted WRs."""
    f = fpr.load_routes()
    pos = route_positions(f[["gameId", "playId", "nflId"]])
    rows = [[*k, *xy_features(p)] for k, p in pos.items()]
    t = pd.DataFrame(rows, columns=["gameId", "playId", "nflId"] + FEATURES)
    return t.merge(f[["gameId", "playId", "nflId", "displayName", "team"]],
                   on=["gameId", "playId", "nflId"])


def prepare(t, prep):
    """Fill missing cut values, then clip to the training range (drawn routes can be extreme)."""
    x = t[FEATURES].copy()
    x["cut_sharpness"] = x["cut_sharpness"].fillna(0.0)
    x["speed_kept_in_cut"] = x["speed_kept_in_cut"].fillna(prep["kept_median"])
    return np.clip(x.to_numpy(float), prep["lo"], prep["hi"])


def fit(t):
    prep = {"kept_median": float(t["speed_kept_in_cut"].median()), "lo": -np.inf, "hi": np.inf}
    x = prepare(t, prep)
    prep["lo"], prep["hi"] = np.percentile(x, 1, axis=0), np.percentile(x, 99, axis=0)
    x = np.clip(x, prep["lo"], prep["hi"])
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    model.fit(scaler.transform(x), t["displayName"])
    return model, scaler, prep


def evaluate(t):
    """Top-1 / top-5 on unseen games, mean of N_SPLITS game splits."""
    split = GroupShuffleSplit(n_splits=N_SPLITS, test_size=0.25, random_state=SEED)
    res = []
    for tr, te in split.split(t, groups=t["gameId"]):
        model, scaler, prep = fit(t.iloc[tr])
        proba = model.predict_proba(scaler.transform(prepare(t.iloc[te], prep)))
        truth = np.searchsorted(model.classes_, t.iloc[te]["displayName"])
        rank = (proba > proba[np.arange(len(truth)), truth][:, None]).sum(axis=1)
        res.append({"top1": (rank == 0).mean(), "top5": (rank < 5).mean()})
    r = pd.DataFrame(res).mean()
    n = t["displayName"].nunique()
    return {"top1": r["top1"], "top5": r["top5"], "chance1": 1 / n, "chance5": 5 / n, "n_wrs": n}


def export(t):
    """Weights for the browser: clip, z = (x - mean) / sd, logits = W z + b."""
    model, scaler, prep = fit(t)
    return {"features": FEATURES, "classes": model.classes_.tolist(),
            "mean": scaler.mean_.round(5).tolist(), "sd": scaler.scale_.round(5).tolist(),
            "W": model.coef_.round(4).tolist(), "b": model.intercept_.round(4).tolist(),
            "kept_median": round(prep["kept_median"], 4),
            "lo": np.round(prep["lo"], 4).tolist(), "hi": np.round(prep["hi"], 4).tolist()}


def main():
    t = build_table()
    print(f"Routes: {len(t):,}   WRs: {t['displayName'].nunique()}")
    print(t[FEATURES].describe().T[["mean", "50%", "min", "max"]].round(2))
    e = evaluate(t)
    print(f"\nUnseen games (mean of {N_SPLITS} splits): top-1 {e['top1']:.1%} "
          f"({e['top1'] / e['chance1']:.1f}x chance)   top-5 {e['top5']:.1%} "
          f"({e['top5'] / e['chance5']:.1f}x chance)")


if __name__ == "__main__":
    main()
