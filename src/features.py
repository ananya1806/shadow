"""Turn WR route frames into one row of movement-style features per route.

Input:  outputs/wr_routes.parquet  (from load_data.py)
Output: outputs/route_features.parquet

Features describe HOW a receiver moves, never WHERE he lines up (no x/y
alignment features). Frame 0 is the snap; tracking is 10 frames per second.

Run:  python3 src/features.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_FILE = ROOT / "outputs" / "wr_routes.parquet"
OUT_FILE = ROOT / "outputs" / "route_features.parquet"

ROUTE = ["gameId", "playId", "nflId"]
FPS = 10
RELEASE_FRAMES = 10      # first second after the snap
CUT_MIN_SPEED = 3.0      # yd/s; ignore direction changes while nearly standing
CUT_WINDOW = 3           # frames before/after the sharpest cut
FEATURES = ["release_speed", "release_accel", "top_speed", "avg_speed", "max_decel",
            "cut_sharpness", "speed_kept_in_cut", "straightness", "route_time"]


def wrap_angle(d):
    """Map an angle difference to [-180, 180) so 359 -> 1 counts as 2 degrees."""
    return ((d + 180) % 360) - 180


def build_features(d):
    d = d.sort_values(ROUTE + ["frameId"]).reset_index(drop=True)
    g = d.groupby(ROUTE, sort=False)
    k = g.cumcount()                       # frame index since snap (0 = snap)
    n = g["frameId"].transform("size")     # frames in this route

    f = g.agg(team=("team", "first"),
              end_event=("event", "last"),
              top_speed=("s", "max"),
              avg_speed=("s", "mean"),
              frames=("frameId", "size"))

    # Release: speed one second after the snap, peak acceleration in that second.
    f["release_speed"] = d[k == RELEASE_FRAMES].set_index(ROUTE)["s"]
    f["release_accel"] = d[k < RELEASE_FRAMES].groupby(ROUTE)["a"].max()

    # Deceleration: 'a' is unsigned, so use the signed change in speed.
    d["ds"] = g["s"].diff() * FPS
    f["max_decel"] = d.groupby(ROUTE)["ds"].min()

    # Cuts: frame-to-frame turn in degrees, only while moving faster than 3 yd/s.
    s_prev = g["s"].shift()
    turn = wrap_angle(g["dir"].diff()).abs()
    moving = (k > 0) & (d["s"] > CUT_MIN_SPEED) & (s_prev > CUT_MIN_SPEED)
    d["turn"] = turn.where(moving)
    f["cut_sharpness"] = d.groupby(ROUTE)["turn"].max()

    # Speed kept through the sharpest cut: speed 3 frames after / 3 frames before.
    cut_rows = d.dropna(subset=["turn"]).groupby(ROUTE)["turn"].idxmax()
    i = cut_rows.to_numpy()
    in_route = (k.to_numpy()[i] >= CUT_WINDOW) & (k.to_numpy()[i] + CUT_WINDOW < n.to_numpy()[i])
    s = d["s"].to_numpy()
    kept = np.full(len(i), np.nan)
    ok = i[in_route]
    kept[in_route] = s[ok + CUT_WINDOW] / s[ok - CUT_WINDOW]
    f["speed_kept_in_cut"] = pd.Series(np.clip(kept, 0, 1.5), index=cut_rows.index)

    # Straightness: start-to-end distance / distance actually run.
    # Frame 0's 'dis' is movement from the pre-snap frame, so skip it.
    ends = g.agg(x0=("x", "first"), y0=("y", "first"), x1=("x", "last"), y1=("y", "last"))
    line = np.hypot(ends["x1"] - ends["x0"], ends["y1"] - ends["y0"])
    path = d[k > 0].groupby(ROUTE)["dis"].sum()
    # 'dis' is rounded to 0.01 yd, which can push the ratio a hair above 1.
    f["straightness"] = (line / path).clip(upper=1.0)

    f["route_time"] = f["frames"] / FPS

    f = f.drop(columns="frames").reset_index()
    f["end_event"] = f["end_event"].astype(str)
    return f[ROUTE + ["team", "end_event"] + FEATURES]


def check(f):
    print(f"Routes: {len(f):,}   WRs: {f['nflId'].nunique()}")
    print("\nNaN per feature:")
    print(f[FEATURES].isna().sum().to_string())
    print("\nImpossible / suspicious values:")
    tests = {
        "top_speed > 12 yd/s": f["top_speed"] > 12,
        "release_speed > 12 yd/s": f["release_speed"] > 12,
        "release_accel > 15 yd/s^2": f["release_accel"] > 15,
        "max_decel < -15 yd/s^2": f["max_decel"] < -15,
        "max_decel > 0 (never slowed)": f["max_decel"] > 0,
        "cut_sharpness > 180 deg": f["cut_sharpness"] > 180,
        "straightness > 1": f["straightness"] > 1.0001,
        "straightness <= 0": f["straightness"] <= 0,
        "avg_speed > top_speed": f["avg_speed"] > f["top_speed"] + 1e-6,
    }
    for name, mask in tests.items():
        print(f"  {name:<30} {int(mask.sum()):>6}")


def main():
    d = pd.read_parquet(IN_FILE)
    f = build_features(d)
    f.to_parquet(OUT_FILE, index=False)

    pd.set_option("display.width", 200)
    print(f[FEATURES].describe().round(2).to_string())
    print()
    check(f)
    print(f"\nSaved -> {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
