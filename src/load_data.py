"""Load the Big Data Bowl files and build one clean table of WR route frames.

Output: outputs/wr_routes.parquet, one row per WR per frame, from the snap to
the throw (or sack / run), with every play normalized to go left to right.

Run:  python3 src/load_data.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "data"
OUT_DIR = ROOT / "outputs"
OUT_FILE = OUT_DIR / "wr_routes.parquet"

FIELD_LENGTH = 120.0
FIELD_WIDTH = 53.3
MIN_FRAMES = 15

SNAP_EVENTS = ["ball_snap", "autoevent_ballsnap"]
END_EVENTS = ["pass_forward", "autoevent_passforward", "qb_sack", "qb_strip_sack", "run"]

TRACK_COLS = ["gameId", "playId", "nflId", "frameId", "team", "playDirection",
              "x", "y", "s", "a", "dis", "o", "dir", "event"]
FLOAT_COLS = ["x", "y", "s", "a", "dis", "o", "dir"]
TRACK_DTYPES = {c: "float32" for c in FLOAT_COLS} | {"nflId": "float64"}


def load_tables():
    """Return the four small tables: players, plays, games, pff."""
    players = pd.read_csv(DATA_DIR / "players.csv")
    plays = pd.read_csv(DATA_DIR / "plays.csv")
    games = pd.read_csv(DATA_DIR / "games.csv")
    pff = pd.read_csv(DATA_DIR / "pffScoutingData.csv")
    return players, plays, games, pff


def wr_route_keys(players, pff):
    """(gameId, playId, nflId) for every WR who ran a route on a play."""
    wr_ids = players.loc[players["officialPosition"] == "WR", "nflId"]
    routes = pff[(pff["pff_role"] == "Pass Route") & pff["nflId"].isin(wr_ids)]
    return routes[["gameId", "playId", "nflId"]].drop_duplicates()


def normalize_direction(t):
    """Flip plays going left so every play moves left to right."""
    left = t["playDirection"] == "left"
    t.loc[left, "x"] = FIELD_LENGTH - t.loc[left, "x"]
    t.loc[left, "y"] = FIELD_WIDTH - t.loc[left, "y"]
    t.loc[left, "o"] = (t.loc[left, "o"] + 180) % 360
    t.loc[left, "dir"] = (t.loc[left, "dir"] + 180) % 360
    return t


def play_windows(t):
    """Snap frame and end frame for each play.

    Snap: first ball_snap. If a play has no ball_snap, first autoevent_ballsnap.
    (The auto event sometimes fires 2-14 s before the real snap, so it is only
    a fallback.)
    End: first throw / sack / run event after the snap.
    """
    ev = t.loc[t["event"].notna(), ["gameId", "playId", "frameId", "event"]].drop_duplicates()
    keys = ["gameId", "playId"]

    manual = ev[ev["event"] == "ball_snap"].groupby(keys)["frameId"].min()
    auto = ev[ev["event"] == "autoevent_ballsnap"].groupby(keys)["frameId"].min()
    snap = manual.combine_first(auto).rename("snapFrame")

    ends = ev[ev["event"].isin(END_EVENTS)].merge(snap.reset_index(), on=keys)
    ends = ends[ends["frameId"] > ends["snapFrame"]]
    end = ends.groupby(keys)["frameId"].min().rename("endFrame")

    return pd.concat([snap, end], axis=1).dropna().astype("int32").reset_index()


def load_game(path, keys):
    """Read one tracking file and keep only WR route frames from snap to end."""
    t = pd.read_csv(path, usecols=TRACK_COLS, dtype=TRACK_DTYPES)
    t = t.dropna(subset=["nflId"])            # drop football rows
    t["nflId"] = t["nflId"].astype("int64")
    t = t.merge(keys, on=["gameId", "playId", "nflId"], how="inner")
    if t.empty:
        return t, 0

    t = normalize_direction(t)
    windows = play_windows(t)
    n_plays = t[["gameId", "playId"]].drop_duplicates().shape[0]
    t = t.merge(windows, on=["gameId", "playId"], how="inner")
    t = t[(t["frameId"] >= t["snapFrame"]) & (t["frameId"] <= t["endFrame"])]
    return t, n_plays - len(windows)


def build_wr_routes():
    players, plays, games, pff = load_tables()
    keys = wr_route_keys(players, pff)

    frames, no_window = [], 0
    files = sorted((DATA_DIR / "tracking").glob("tracking_*.csv"))
    for i, path in enumerate(files, 1):
        game_keys = keys[keys["gameId"] == int(path.stem.split("_")[1])]
        t, dropped = load_game(path, game_keys)
        frames.append(t)
        no_window += dropped
        if i % 20 == 0 or i == len(files):
            print(f"  loaded {i}/{len(files)} games")

    df = pd.concat(frames, ignore_index=True)
    route_id = ["gameId", "playId", "nflId"]
    n_frames = df.groupby(route_id)["frameId"].transform("size")
    short = df.loc[n_frames < MIN_FRAMES, route_id].drop_duplicates().shape[0]
    df = df[n_frames >= MIN_FRAMES]

    df = df.sort_values(route_id + ["frameId"]).reset_index(drop=True)
    for col in ["team", "playDirection", "event"]:
        df[col] = df[col].astype("category")

    OUT_DIR.mkdir(exist_ok=True)
    df.to_parquet(OUT_FILE, index=False)

    per_route = df.groupby(route_id).size()
    print(f"\nWR route rows in PFF:          {len(keys):,}")
    print(f"Plays dropped (no snap/end):   {no_window:,}")
    print(f"Routes dropped (<{MIN_FRAMES} frames):  {short:,}")
    print(f"Routes kept:                   {len(per_route):,}")
    print(f"WRs:                           {df['nflId'].nunique():,}")
    print(f"Frame rows:                    {len(df):,}")
    print("Frames per route:")
    print(per_route.describe().round(1).to_string())
    print(f"Duplicate rows (full):         {df.duplicated().sum():,}")
    print(f"Duplicate rows (route+frame):  {df.duplicated(route_id + ['frameId']).sum():,}")
    print(f"Saved -> {OUT_FILE.relative_to(ROOT)}")
    return df


if __name__ == "__main__":
    build_wr_routes()
