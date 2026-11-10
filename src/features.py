"""Turn WR route frames into one row of movement-style features per route.

Input:  outputs/wr_routes.parquet  (from load_data.py)
Output: outputs/route_features.parquet

Features describe HOW a receiver moves, never WHERE he lines up (no x/y
alignment features). Frame 0 is the snap; tracking is 10 frames per second.
Missing values are kept (e.g. no cut before the throw); later steps ignore
them with nanmean or models that handle NaN.

Run:  .venv/bin/python src/features.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_FILE = ROOT / "outputs" / "wr_routes.parquet"
OUT_FILE = ROOT / "outputs" / "route_features.parquet"

ROUTE = ["gameId", "playId", "nflId"]
FPS = 10
RELEASE_FRAMES = 10      # first second after the snap; cuts are searched after this
CUT_MIN_SPEED = 3.0      # yd/s; ignore direction changes while nearly standing
CUT_WINDOW = 10          # frames (1.0 s, i.e. t-5 to t+5) over which a turn is measured.
                         # Tracking smooths 'dir', so a real break spreads over ~1 s;
                         # a 0.5 s window gave a median "sharpest cut" of only 16 deg.
PRE_CUT = 5              # speed_kept_in_cut compares against speed 5 frames before the cut
CUT_DEGREES = 45         # a windowed turn above this counts as a real cut
MAX_DECEL_FLOOR = -10.0  # yd/s^2; anything harsher is contact or a tracking blip
KEPT_CLIP = (0.0, 1.2)
DROP_END_EVENTS = ["run"]  # QB took off: not a real pass route. Sacks are kept.
MAX_SNAP_SPEED = 2.0     # yd/s; faster = pre-snap motion, which fakes the release features

FEATURES = ["release_speed", "release_accel", "top_speed", "avg_speed", "max_decel",
            "cut_sharpness", "speed_kept_in_cut", "has_cut", "straightness", "route_time"]


def wrap_angle(d):
    """Map an angle difference to [-180, 180) so 359 -> 1 counts as 2 degrees."""
    return ((d + 180) % 360) - 180


def drop_qb_runs(d):
    """Remove routes whose play ended with the QB running."""
    end = d.groupby(ROUTE)["event"].last().astype(str).rename("end_event")
    keep = end[~end.isin(DROP_END_EVENTS)].index
    return d.set_index(ROUTE).loc[keep].reset_index(), int((~end.index.isin(keep)).sum())


def drop_presnap_motion(d):
    """Remove routes where the WR was already moving faster than 2 yd/s at the snap."""
    snap_speed = d[d["frameId"] == d["snapFrame"]].set_index(ROUTE)["s"]
    keep = snap_speed[snap_speed <= MAX_SNAP_SPEED].index
    return d.set_index(ROUTE).loc[keep].reset_index(), int((snap_speed > MAX_SNAP_SPEED).sum())


def cut_features(d, g, k):
    """Sharpest 1.0 s turn after the release, and speed kept through it.

    A window starts at frame t (t >= 10, after the release) and ends at t+10.
    Its turn is the wrapped change in 'dir' from start to end. The WR must
    enter the window faster than 3 yd/s; he may slow below that inside it
    (curls and comebacks nearly stop, and those are real cuts).
    speed_kept_in_cut = min speed inside the window / speed 5 frames before it.
    """
    dir_ahead = g["dir"].shift(-CUT_WINDOW)
    valid = (k >= RELEASE_FRAMES) & (d["s"] > CUT_MIN_SPEED) & dir_ahead.notna()
    d["turn"] = wrap_angle(dir_ahead - d["dir"]).abs().where(valid)

    cut_start = d.dropna(subset=["turn"]).groupby(ROUTE)["turn"].idxmax()
    i = cut_start.to_numpy()
    s = d["s"].to_numpy()
    window_min = np.min([s[i + j] for j in range(CUT_WINDOW + 1)], axis=0)
    before = s[i - PRE_CUT]               # i >= 10, so this stays inside the route
    kept = np.clip(window_min / before, *KEPT_CLIP)

    return pd.DataFrame({"cut_sharpness": d.loc[i, "turn"].to_numpy(),
                         "speed_kept_in_cut": kept}, index=cut_start.index)


def build_features(d):
    d = d.sort_values(ROUTE + ["frameId"]).reset_index(drop=True)
    g = d.groupby(ROUTE, sort=False)
    k = g.cumcount()                       # frame index since snap (0 = snap)

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
    f["max_decel"] = d.groupby(ROUTE)["ds"].min().clip(lower=MAX_DECEL_FLOOR)

    # Cuts.
    f = f.join(cut_features(d, g, k))
    f["has_cut"] = (f["cut_sharpness"] > CUT_DEGREES).astype("int8")  # no window -> 0
    f.loc[f["has_cut"] == 0, "speed_kept_in_cut"] = np.nan

    # Straightness: start-to-end distance / distance actually run.
    # Frame 0's 'dis' is movement from the pre-snap frame, so skip it.
    # 'dis' is rounded to 0.01 yd, which can push the ratio a hair above 1.
    ends = g.agg(x0=("x", "first"), y0=("y", "first"), x1=("x", "last"), y1=("y", "last"))
    line = np.hypot(ends["x1"] - ends["x0"], ends["y1"] - ends["y0"])
    path = d[k > 0].groupby(ROUTE)["dis"].sum()
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
        "max_decel < -10 yd/s^2": f["max_decel"] < MAX_DECEL_FLOOR,
        "cut_sharpness > 180 deg": f["cut_sharpness"] > 180,
        "straightness > 1": f["straightness"] > 1,
        "straightness <= 0": f["straightness"] <= 0,
        "avg_speed > top_speed": f["avg_speed"] > f["top_speed"] + 1e-6,
        "kept outside 0-1.2": (f["speed_kept_in_cut"] < 0) | (f["speed_kept_in_cut"] > 1.2),
    }
    for name, mask in tests.items():
        print(f"  {name:<30} {int(mask.sum()):>6}")

    c = f["cut_sharpness"].dropna()
    print("\ncut_sharpness (deg, 1.0 s window):")
    print(f"  median {c.median():.1f}   p75 {c.quantile(.75):.1f}   p90 {c.quantile(.9):.1f}"
          f"   max {c.max():.1f}")
    print(f"  has_cut rate: {f['has_cut'].mean():.1%}")
    bins = pd.cut(c, [0, 15, 30, 45, 60, 90, 135, 180], include_lowest=True)
    print(bins.value_counts(sort=False).to_string())


def main():
    d = pd.read_parquet(IN_FILE)
    d, n_runs = drop_qb_runs(d)
    print(f"Dropped {n_runs:,} routes on QB-run plays")
    d, n_motion = drop_presnap_motion(d)
    print(f"Dropped {n_motion:,} routes with pre-snap motion (> {MAX_SNAP_SPEED} yd/s at snap)")
    f = build_features(d)
    f.to_parquet(OUT_FILE, index=False)

    pd.set_option("display.width", 200)
    print(f[FEATURES].describe().round(2).to_string())
    print()
    check(f)
    print(f"\nSaved -> {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
