"""Find each WR's movement doppelgangers and check which matches are stable.

1. Cosine similarity between every pair of standardized WR fingerprints.
2. find_doppelgangers(name, k=5): the k WRs who move most like him.
3. Stability: rebuild fingerprints from two random, disjoint halves of each
   WR's games (20 times). A pair is "stable" when B is in A's top 5 in BOTH
   halves in at least half of those repeats. Only stable pairs are headline
   material; the rest are flagged.
4. Sanity checks: are matches alike in speed (in the fingerprint) and in body
   size (NOT in the fingerprint)? Are teammates over-matched? Are a few
   "average" WRs matched to everyone?

Input:  outputs/wr_fingerprints.csv, wr_fingerprints_raw.csv, route_features.parquet
Output: outputs/doppelgangers.csv  every WR -> every other WR, with rank,
        similarity, stability rate and stable flag

Run:  .venv/bin/python src/doppelganger.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import fingerprint as fpr

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
K = 5
STABLE_RATE = 0.5      # share of split-half repeats where the pair survives in both halves
STARS = ["Tyreek Hill", "Cooper Kupp", "Davante Adams", "Justin Jefferson",
         "Deebo Samuel", "Ja'Marr Chase", "Stefon Diggs"]


# ----------------------------------------------------------------- matching
def load_fingerprints():
    z = pd.read_csv(OUT_DIR / "wr_fingerprints.csv").set_index("nflId")
    cols = [c for c in z.columns if c not in fpr.INFO]
    return z, cols


def similarity_matrix(z, cols):
    sim = cosine_similarity(z[cols])
    np.fill_diagonal(sim, np.nan)                     # a WR is not his own doppelganger
    return pd.DataFrame(sim, index=z.index, columns=z.index)


def top_k(sim, k=K):
    """{nflId: [k most similar nflIds]} from a similarity matrix with NaN diagonal."""
    vals = np.nan_to_num(sim.to_numpy(), nan=-np.inf)
    order = np.argsort(-vals, axis=1)[:, :k]
    return {wr: list(sim.columns[row]) for wr, row in zip(sim.index, order)}


def all_pairs(z, sim):
    """Long table: every WR -> every other WR, ranked by similarity."""
    pairs = (sim.stack().rename("similarity").rename_axis(["nflId", "match_nflId"])
             .reset_index())
    pairs["rank"] = (pairs.groupby("nflId")["similarity"]
                     .rank(ascending=False, method="first").astype(int))
    names = z[["displayName", "team"]]
    pairs = (pairs.join(names, on="nflId")
             .join(names.add_prefix("match_"), on="match_nflId"))
    return pairs.sort_values(["displayName", "rank"])


def find_doppelgangers(name, k=K, pairs=None):
    """The k WRs whose movement fingerprint is most similar to `name`'s."""
    pairs = pairs if pairs is not None else build()[0]
    hits = pairs[pairs["displayName"].str.lower() == name.lower()]
    if hits.empty:
        raise ValueError(f"{name!r} not found (needs >= {fpr.MIN_ROUTES} routes)")
    cols = ["match_displayName", "match_team", "similarity", "rank"]
    cols += [c for c in ["stable_rate", "stable"] if c in hits.columns]
    return hits.nsmallest(k, "rank")[cols].reset_index(drop=True)


# ---------------------------------------------------------------- stability
def stability(f, k=K, repeats=fpr.HALF_REPEATS):
    """How often each top-k pair shows up in both halves, and per-WR top-k overlap."""
    rng = np.random.default_rng(fpr.SEED)
    both, seen, overlaps = {}, {}, []
    for _ in range(repeats):
        a, b = fpr.split_half_fingerprints(f, rng)
        sims = []
        for half in (a, b):
            s = cosine_similarity(half)
            np.fill_diagonal(s, np.nan)
            sims.append(top_k(pd.DataFrame(s, index=half.index, columns=half.index), k))
        top_a, top_b = sims
        for wr in top_a:
            common = set(top_a[wr]) & set(top_b[wr])
            overlaps.append({"nflId": wr, "overlap": len(common) / k, "pool": len(a) - 1})
            for m in set(top_a[wr]) | set(top_b[wr]):
                seen[(wr, m)] = seen.get((wr, m), 0) + 1
            for m in common:
                both[(wr, m)] = both.get((wr, m), 0) + 1
    rate = pd.Series({p: both.get(p, 0) / repeats for p in seen}, name="stable_rate")
    rate.index = pd.MultiIndex.from_tuples(rate.index, names=["nflId", "match_nflId"])
    return rate, pd.DataFrame(overlaps)


def build():
    z, cols = load_fingerprints()
    sim = similarity_matrix(z, cols)
    pairs = all_pairs(z, sim)
    rate, overlaps = stability(fpr.load_routes())
    pairs = pairs.join(rate, on=["nflId", "match_nflId"])
    pairs["stable_rate"] = pairs["stable_rate"].fillna(0.0)
    pairs["stable"] = pairs["stable_rate"] >= STABLE_RATE
    return pairs, overlaps, z


# ------------------------------------------------------------ sanity checks
def body_size():
    p = pd.read_csv(fpr.PLAYERS_FILE, usecols=["nflId", "height", "weight"]).set_index("nflId")
    ft_in = p["height"].str.split("-", expand=True).astype(float)
    p["height_in"] = ft_in[0] * 12 + ft_in[1]
    return p[["height_in", "weight"]]


def sanity_checks(pairs):
    raw = pd.read_csv(OUT_DIR / "wr_fingerprints_raw.csv").set_index("nflId")
    raw = raw.join(body_size())
    checks = {"top_speed_mean": "yd/s", "release_speed_mean": "yd/s",
              "release_accel_mean": "yd/s^2", "cut_rate": "share",
              "weight": "lb (not in fingerprint)", "height_in": "in (not in fingerprint)"}

    top = pairs[pairs["rank"] <= K]
    rest = pairs                                       # every possible pair = random baseline
    print(f"\nSANITY 1: average gap between a WR and his top-{K} matches vs a random WR")
    print(f"  {'feature':<20}{'top-5 gap':>10}{'random gap':>12}{'ratio':>8}  unit")
    for col, unit in checks.items():
        v = raw[col]
        gap = lambda p: (v.reindex(p["nflId"]).to_numpy()
                         - v.reindex(p["match_nflId"]).to_numpy())
        g_top, g_all = np.nanmean(np.abs(gap(top))), np.nanmean(np.abs(gap(rest)))
        print(f"  {col:<20}{g_top:>10.3f}{g_all:>12.3f}{g_top / g_all:>8.2f}  {unit}")

    team_rate = (top["team"] == top["match_team"]).mean()
    team_base = (rest["team"] == rest["match_team"]).mean()
    print(f"\nSANITY 2: teammates in top-{K}: {team_rate:.1%} vs {team_base:.1%} for random pairs"
          f" ({team_rate / team_base:.1f}x)")

    hubs = top["match_displayName"].value_counts()
    print(f"\nSANITY 3: hubs (WRs in many top-{K} lists; expected {K} each)")
    print("  " + ", ".join(f"{n} {c}" for n, c in hubs.head(6).items()))
    print(f"  WRs who are nobody's top-{K}: {len(set(pairs['displayName']) - set(hubs.index))}")
    return raw


def flag_weird(name, matches, raw, ids):
    """Notes on matches that a scout might question."""
    me = raw.loc[ids[name]]
    notes = []
    for r in matches.itertuples():
        o = raw.loc[ids[r.match_displayName]]
        if abs(o["weight"] - me["weight"]) >= 30:
            notes.append(f"{r.match_displayName}: {o['weight'] - me['weight']:+.0f} lb")
        if abs(o["top_speed_mean"] - me["top_speed_mean"]) >= 0.5:
            notes.append(f"{r.match_displayName}: top speed "
                         f"{o['top_speed_mean'] - me['top_speed_mean']:+.2f} yd/s")
    return notes


def main():
    pd.set_option("display.width", 200)
    pairs, overlaps, z = build()

    pool = overlaps["pool"].mean()
    chance = K / pool
    print(f"STABILITY ({fpr.HALF_REPEATS} random game halves, ~{pool + 1:.0f} WRs each)")
    print(f"  top-{K} overlap between halves: {overlaps['overlap'].mean():.1%} "
          f"(chance {chance:.1%}, {overlaps['overlap'].mean() / chance:.1f}x)")
    top = pairs[pairs["rank"] <= K]
    print(f"  full-season top-{K} pairs that are stable (in both halves in >= "
          f"{STABLE_RATE:.0%} of repeats): {top['stable'].sum()} of {len(top)} "
          f"({top['stable'].mean():.1%})")
    print(f"  WRs with at least one stable doppelganger: "
          f"{top.loc[top['stable'], 'nflId'].nunique()} of {top['nflId'].nunique()}")

    raw = sanity_checks(pairs)
    ids = z.reset_index().set_index("displayName")["nflId"]

    print("\nSTAR DOPPELGANGERS (stable = match survives in both halves of the season)")
    for name in STARS:
        try:
            m = find_doppelgangers(name, pairs=pairs)
        except ValueError as e:
            print(f"\n{e}")
            continue
        team = z.loc[ids[name], "team"]
        print(f"\n{name} ({team})")
        for r in m.itertuples():
            tag = "STABLE" if r.stable else "      "
            print(f"  {r.rank}. {r.match_displayName:<24}{r.match_team:<5}"
                  f"sim {r.similarity:.2f}   in both halves {r.stable_rate:.0%}  {tag}")
        for note in flag_weird(name, m, raw, ids):
            print(f"     flag: {note}")

    stable = (top[top["stable"]]
              .sort_values("stable_rate", ascending=False)
              [["displayName", "team", "match_displayName", "match_team",
                "similarity", "stable_rate"]])
    print(f"\nMOST STABLE DOPPELGANGER PAIRS (top 15 of {len(stable)})")
    print(stable.head(15).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    out = pairs[["nflId", "displayName", "team", "rank", "match_nflId", "match_displayName",
                 "match_team", "similarity", "stable_rate", "stable"]]
    out.to_csv(OUT_DIR / "doppelgangers.csv", index=False, float_format="%.4f")
    print(f"\nSaved -> outputs/doppelgangers.csv ({len(out):,} pairs)")


if __name__ == "__main__":
    main()
