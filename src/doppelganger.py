"""Find each WR's movement doppelgangers and check which matches are stable.

1. Cosine similarity between every pair of standardized WR fingerprints.
   Teammates are excluded: a GM replacing a player shops on other teams, and
   teammates share a scheme and QB, which inflates their similarity.
2. find_doppelgangers(name, k=5): the k non-teammates who move most like him.
   built_only=True keeps only matches who are also "built like" him
   (weight within 15 lb and height within 2 in, from players.csv).
3. Stability: rebuild fingerprints from two random, disjoint halves of each
   WR's games (20 times). A pair is "stable" when B is in A's top 5 in BOTH
   halves in at least half of those repeats.
4. Hubs: WRs who land in 10+ top-5 lists (plus Keenan Allen and Keelan Cole).
   They have a common profile that many WRs resemble, so being "his
   doppelganger" says little. (They are not near league average: list count
   correlates +0.42 with distance from average.) Flagged, never headlined.

Input:  outputs/wr_fingerprints.csv, wr_fingerprints_raw.csv, route_features.parquet
Output: outputs/doppelgangers.csv  every WR -> every non-teammate, with rank,
        similarity, stability, built_like and hub flags

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
HUB_MIN_LISTS = 10     # in this many top-5 lists or more = hub (common profile)
FORCED_HUBS = ["Keenan Allen", "Keelan Cole"]
BUILT_LB, BUILT_IN = 15, 2
STARS = ["Tyreek Hill", "Cooper Kupp", "Davante Adams", "Justin Jefferson",
         "Deebo Samuel", "Ja'Marr Chase", "Stefon Diggs"]


# ----------------------------------------------------------------- matching
def load_fingerprints():
    z = pd.read_csv(OUT_DIR / "wr_fingerprints.csv").set_index("nflId")
    cols = [c for c in z.columns if c not in fpr.INFO]
    return z, cols


def similarity(z, cols, team):
    """Cosine similarity with self-pairs and teammates set to NaN."""
    sim = pd.DataFrame(cosine_similarity(z[cols]), index=z.index, columns=z.index)
    t = team.reindex(z.index).to_numpy()
    sim = sim.mask(t[:, None] == t[None])            # also masks the diagonal
    return sim


def top_k(sim, k=K):
    """{nflId: [k most similar nflIds]}; NaN entries are never picked."""
    vals = np.nan_to_num(sim.to_numpy(), nan=-np.inf)
    order = np.argsort(-vals, axis=1)[:, :k]
    return {wr: list(sim.columns[row]) for wr, row in zip(sim.index, order)}


def body_size():
    p = pd.read_csv(fpr.PLAYERS_FILE, usecols=["nflId", "height", "weight"]).set_index("nflId")
    ft_in = p["height"].str.split("-", expand=True).astype(float)
    p["height_in"] = ft_in[0] * 12 + ft_in[1]
    return p[["height_in", "weight"]]


def all_pairs(z, sim):
    """Long table: every WR -> every non-teammate, ranked by similarity."""
    pairs = (sim.stack().rename("similarity").rename_axis(["nflId", "match_nflId"])
             .reset_index())
    pairs["rank"] = (pairs.groupby("nflId")["similarity"]
                     .rank(ascending=False, method="first").astype(int))
    names = z[["displayName", "team"]].join(body_size())
    pairs = pairs.join(names, on="nflId").join(names.add_prefix("match_"), on="match_nflId")
    pairs["built_like"] = ((pairs["weight"] - pairs["match_weight"]).abs() <= BUILT_LB) & \
                          ((pairs["height_in"] - pairs["match_height_in"]).abs() <= BUILT_IN)
    return pairs.sort_values(["displayName", "rank"])


def find_doppelgangers(name, k=K, pairs=None, built_only=False):
    """The k non-teammates whose movement fingerprint is most like `name`'s."""
    pairs = pairs if pairs is not None else build()[0]
    hits = pairs[pairs["displayName"].str.lower() == name.lower()]
    if hits.empty:
        raise ValueError(f"{name!r} not found (needs >= {fpr.MIN_ROUTES} routes)")
    if built_only:
        hits = hits[hits["built_like"]]
    cols = ["match_displayName", "match_team", "similarity", "rank", "built_like"]
    cols += [c for c in ["stable_rate", "stable", "match_is_hub"] if c in hits.columns]
    return hits.nsmallest(k, "rank")[cols].reset_index(drop=True)


# ---------------------------------------------------------------- stability
def stability(f, team, k=K, repeats=fpr.HALF_REPEATS):
    """Share of repeats where a pair is in the top k in both halves (teammates excluded)."""
    rng = np.random.default_rng(fpr.SEED)
    cols = None
    both, seen, overlaps = {}, {}, []
    for _ in range(repeats):
        a, b = fpr.split_half_fingerprints(f, rng)
        cols = list(a.columns)
        top_a, top_b = (top_k(similarity(h, cols, team), k) for h in (a, b))
        for wr in top_a:
            common = set(top_a[wr]) & set(top_b[wr])
            pool = int((team.reindex(a.index) != team.get(wr)).sum())
            overlaps.append({"nflId": wr, "overlap": len(common) / k, "pool": pool})
            for m in set(top_a[wr]) | set(top_b[wr]):
                seen[(wr, m)] = seen.get((wr, m), 0) + 1
            for m in common:
                both[(wr, m)] = both.get((wr, m), 0) + 1
    rate = pd.Series({p: both.get(p, 0) / repeats for p in seen}, name="stable_rate")
    rate.index = pd.MultiIndex.from_tuples(rate.index, names=["nflId", "match_nflId"])
    return rate, pd.DataFrame(overlaps)


def build():
    z, cols = load_fingerprints()
    team = z["team"]
    pairs = all_pairs(z, similarity(z, cols, team))
    rate, overlaps = stability(fpr.load_routes(), team)
    pairs = pairs.join(rate, on=["nflId", "match_nflId"])
    pairs["stable_rate"] = pairs["stable_rate"].fillna(0.0)
    pairs["stable"] = pairs["stable_rate"] >= STABLE_RATE

    lists = pairs.loc[pairs["rank"] <= K, "match_displayName"].value_counts()
    hubs = set(lists[lists >= HUB_MIN_LISTS].index) | set(FORCED_HUBS)
    pairs["is_hub"] = pairs["displayName"].isin(hubs)
    pairs["match_is_hub"] = pairs["match_displayName"].isin(hubs)
    pairs["match_top5_lists"] = pairs["match_displayName"].map(lists).fillna(0).astype(int)
    return pairs, overlaps, z


# ------------------------------------------------------------ sanity checks
def sanity_checks(pairs):
    raw = pd.read_csv(OUT_DIR / "wr_fingerprints_raw.csv").set_index("nflId").join(body_size())
    checks = {"top_speed_mean": "yd/s", "release_speed_mean": "yd/s",
              "release_accel_mean": "yd/s^2", "cut_rate": "share",
              "weight": "lb (not in fingerprint)", "height_in": "in (not in fingerprint)"}
    top = pairs[pairs["rank"] <= K]
    print(f"\nSANITY: average gap to top-{K} matches vs to a random non-teammate")
    print(f"  {'feature':<20}{'top-5 gap':>10}{'random gap':>12}{'ratio':>8}  unit")
    for col, unit in checks.items():
        v = raw[col]
        gap = lambda p: np.abs(v.reindex(p["nflId"]).to_numpy()
                               - v.reindex(p["match_nflId"]).to_numpy())
        g_top, g_all = np.nanmean(gap(top)), np.nanmean(gap(pairs))
        print(f"  {col:<20}{g_top:>10.3f}{g_all:>12.3f}{g_top / g_all:>8.2f}  {unit}")
    print(f"  built-like share: top-{K} {top['built_like'].mean():.1%} vs random "
          f"{pairs['built_like'].mean():.1%}")


def print_matches(title, m):
    print(f"  {title}")
    if m.empty:
        print("     (none)")
    for r in m.itertuples():
        tags = []
        if r.stable:
            tags.append("STABLE")
        if r.match_is_hub:
            tags.append("hub")
        if r.built_like:
            tags.append("built like")
        print(f"     {r.rank:>2}. {r.match_displayName:<24}{r.match_team:<5}sim {r.similarity:.2f}"
              f"  both halves {r.stable_rate:>4.0%}  {', '.join(tags)}")


def main():
    pd.set_option("display.width", 200)
    pairs, overlaps, z = build()
    top = pairs[pairs["rank"] <= K]

    chance = (K / overlaps["pool"]).mean()
    print(f"STABILITY, teammates excluded ({fpr.HALF_REPEATS} random game halves)")
    print(f"  top-{K} overlap between halves: {overlaps['overlap'].mean():.1%} "
          f"(chance {chance:.1%}, {overlaps['overlap'].mean() / chance:.1f}x)")
    print(f"  stable top-{K} pairs (both halves in >= {STABLE_RATE:.0%} of repeats): "
          f"{top['stable'].sum()} of {len(top)}")

    hubs = sorted(top.loc[top["match_is_hub"], "match_displayName"].unique())
    counts = top.drop_duplicates("match_displayName").set_index("match_displayName")
    print(f"\nHUBS = common profile (in >= {HUB_MIN_LISTS} top-{K} lists, or named):")
    print("  " + ", ".join(f"{h} ({counts.loc[h, 'match_top5_lists']})" for h in hubs))

    sanity_checks(pairs)

    print("\nSTAR DOPPELGANGERS (teammates excluded)")
    for name in STARS:
        try:
            moves = find_doppelgangers(name, pairs=pairs)
            built = find_doppelgangers(name, pairs=pairs, built_only=True)
        except ValueError as e:
            print(f"\n{e}")
            continue
        print(f"\n{name} ({z.loc[z['displayName'] == name, 'team'].iat[0]})")
        print_matches("moves like:", moves)
        print_matches("moves + built like:", built)

    headline = top[top["stable"] & ~top["match_is_hub"] & ~top["is_hub"]]
    headline = headline.assign(pair=headline.apply(
        lambda r: " & ".join(sorted([r["displayName"], r["match_displayName"]])), axis=1))
    headline = (headline.sort_values("stable_rate", ascending=False)
                .drop_duplicates("pair"))
    print(f"\nSTABLE NON-TEAMMATE PAIRS ({len(headline)} unique, hubs removed)")
    print(headline[["displayName", "team", "match_displayName", "match_team", "similarity",
                    "stable_rate", "built_like"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    out = pairs[["nflId", "displayName", "team", "is_hub", "rank", "match_nflId",
                 "match_displayName", "match_team", "similarity", "stable_rate", "stable",
                 "built_like", "match_is_hub", "match_top5_lists"]]
    out.to_csv(OUT_DIR / "doppelgangers.csv", index=False, float_format="%.4f")
    print(f"\nSaved -> outputs/doppelgangers.csv ({len(out):,} pairs)")


if __name__ == "__main__":
    main()
