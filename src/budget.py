"""Budget doppelgangers: cheaper non-teammates who move like an expensive star.

Salary: nflverse contracts (OverTheCap data, via nfl_data_py; see
fetch_external.py). For each WR we take the contract in force in 2021: the most
recently signed contract with year_signed <= 2021. APY = average per year, $M.
Matching is by normalized name, with birth date used to split same-name players.

Availability (weeks 1-8):
  weeks_missed  = his team's games - games he appears in (PFF data, any role)
  weeks_injured = weeks he is on the official injury report (any status)
  weeks_out     = weeks he is listed Out

Budget doppelganger for each star with APY > $10M: the most similar
non-teammate with similarity >= 0.70 and APY under half the star's. Hubs
(common profiles) are never picked. savings = star APY - doppelganger APY.

Input:  outputs/doppelgangers.csv, outputs/external/*.csv, data/data/*.csv
Output: outputs/wr_salary_availability.csv, outputs/budget_doppelgangers.csv

Run:  .venv/bin/python src/budget.py
"""
import re
from pathlib import Path

import pandas as pd

import fingerprint as fpr

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
EXT = OUT_DIR / "external"
DATA = ROOT / "data" / "data"

SEASON = 2021
STAR_APY = 10.0          # $M
MIN_SIM = 0.70
STRICT_SIM = 0.85        # flagged as a near-identical mover
MAX_SHARE = 0.5          # doppelganger APY must be under half the star's
# Same player, different name in the contracts data.
ALIASES = {"robby anderson": "robbie chosen", "gabe davis": "gabriel davis"}


def norm_name(s):
    """'D.J. Moore' -> 'dj moore', 'Henry Ruggs III' -> 'henry ruggs'."""
    s = re.sub(r"[^a-z ]", "", str(s).lower().replace("-", " "))
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return " ".join(s.split())


def contracts_2021():
    """One APY per (name, birth date): latest contract signed on or before 2021."""
    c = pd.read_csv(EXT / "contracts_wr.csv",
                    usecols=["player", "team", "year_signed", "years", "apy", "date_of_birth"])
    c = c[(c["year_signed"] > 0) & (c["year_signed"] <= SEASON)].copy()
    c["key"] = c["player"].map(norm_name)
    # "March 1, 1994" -> "1994-03-01", the format players.csv uses.
    c["date_of_birth"] = (pd.to_datetime(c["date_of_birth"], format="%B %d, %Y", errors="coerce")
                          .dt.strftime("%Y-%m-%d"))
    c = c.sort_values(["year_signed", "apy"]).groupby(["key", "date_of_birth"],
                                                       dropna=False).tail(1)
    return c.rename(columns={"team": "contract_team", "year_signed": "contract_signed",
                             "years": "contract_years"})


def attach_salary(wrs):
    """Add apy to the fingerprinted WRs; returns the table and the match rate."""
    players = pd.read_csv(DATA / "players.csv", usecols=["nflId", "birthDate"])
    w = wrs.merge(players, on="nflId", how="left")
    w["key"] = w["displayName"].map(norm_name).replace(ALIASES)
    c = contracts_2021()

    # Same name + birth date first; else name only if the name is unique.
    by_dob = c.dropna(subset=["date_of_birth"]).set_index(["key", "date_of_birth"])
    unique = c.groupby("key").filter(lambda g: len(g) == 1).set_index("key")
    cols = ["apy", "contract_team", "contract_signed", "contract_years"]
    rows = []
    for r in w.itertuples():
        if (r.key, r.birthDate) in by_dob.index:
            hit = by_dob.loc[(r.key, r.birthDate)]
            hit = hit.iloc[-1] if isinstance(hit, pd.DataFrame) else hit
            rows.append({**hit[cols].to_dict(), "salary_match": "name+dob"})
        elif r.key in unique.index:
            rows.append({**unique.loc[r.key, cols].to_dict(), "salary_match": "name"})
        else:
            rows.append({c_: None for c_ in cols} | {"salary_match": "none"})
    w = pd.concat([w.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    w["apy"] = pd.to_numeric(w["apy"])
    return w.drop(columns=["key", "birthDate"])


def attach_availability(w):
    games = pd.read_csv(DATA / "games.csv")
    pff = pd.read_csv(DATA / "pffScoutingData.csv", usecols=["gameId", "nflId"])
    team_games = pd.concat([games[["gameId", "homeTeamAbbr"]].rename(columns={"homeTeamAbbr": "team"}),
                            games[["gameId", "visitorTeamAbbr"]].rename(columns={"visitorTeamAbbr": "team"})])
    played = pff.drop_duplicates().groupby("nflId")["gameId"].nunique().rename("games_played")
    team_n = team_games.groupby("team")["gameId"].nunique().rename("team_games")
    w = w.join(played, on="nflId").join(team_n, on="team")
    w["weeks_missed"] = (w["team_games"] - w["games_played"]).clip(lower=0)

    inj = pd.read_csv(EXT / "injuries_2021.csv",
                      usecols=["season", "game_type", "week", "position", "full_name",
                               "report_status"])
    inj = inj[(inj["game_type"] == "REG") & (inj["week"] <= 8) & (inj["position"] == "WR")]
    inj["key"] = inj["full_name"].map(norm_name)
    on_report = inj.groupby("key")["week"].nunique().rename("weeks_injured")
    out = inj[inj["report_status"] == "Out"].groupby("key")["week"].nunique().rename("weeks_out")
    w["key"] = w["displayName"].map(norm_name)
    w = w.join(on_report, on="key").join(out, on="key").drop(columns="key")
    w[["weeks_injured", "weeks_out"]] = w[["weeks_injured", "weeks_out"]].fillna(0).astype(int)
    return w


def budget_table(w, pairs):
    apy = w.set_index("nflId")["apy"]
    p = pairs.assign(star_apy=pairs["nflId"].map(apy), match_apy=pairs["match_nflId"].map(apy))
    stars = p[p["star_apy"] > STAR_APY]
    cand = stars[(stars["similarity"] >= MIN_SIM) & stars["match_apy"].notna()
                 & (stars["match_apy"] < MAX_SHARE * stars["star_apy"])
                 & ~stars["match_is_hub"]]
    best = cand.sort_values("similarity", ascending=False).drop_duplicates("nflId")

    avail = w.set_index("nflId")[["weeks_missed", "weeks_injured", "weeks_out"]]
    best = best.join(avail, on="nflId")
    best["savings"] = best["star_apy"] - best["match_apy"]
    best["near_identical"] = best["similarity"] >= STRICT_SIM
    t = best.rename(columns={"displayName": "star", "team": "star_team",
                             "match_displayName": "doppelganger", "match_team": "dopp_team",
                             "match_apy": "dopp_apy"})
    cols = ["star", "star_team", "star_apy", "weeks_missed", "weeks_injured", "weeks_out",
            "doppelganger", "dopp_team", "dopp_apy", "similarity", "rank", "stable_rate",
            "stable", "built_like", "near_identical", "savings"]
    no_match = sorted(set(stars["displayName"]) - set(t["star"]))
    return t[cols].sort_values("savings", ascending=False), no_match


def main():
    pd.set_option("display.width", 220)
    fp = pd.read_csv(OUT_DIR / "wr_fingerprints.csv", usecols=fpr.INFO)
    w = attach_availability(attach_salary(fp))
    rate = (w["salary_match"] != "none").mean()
    print(f"Salary match: {rate:.1%} of {len(w)} WRs "
          f"({(w['salary_match'] == 'name+dob').sum()} by name+birth date, "
          f"{(w['salary_match'] == 'name').sum()} by unique name)")
    print("Unmatched:", ", ".join(w.loc[w["salary_match"] == "none", "displayName"]) or "none")
    print(f"WRs with 2021 APY > ${STAR_APY:.0f}M: {(w['apy'] > STAR_APY).sum()}")

    check = ["Tyreek Hill", "Cooper Kupp", "Davante Adams", "Stefon Diggs", "Mike Williams",
             "Ja'Marr Chase", "Hunter Renfrow"]
    print("\nSpot check (APY $M, contract signed, weeks missed / on injury report / out):")
    print(w[w["displayName"].isin(check)][["displayName", "team", "apy", "contract_team",
                                           "contract_signed", "weeks_missed", "weeks_injured",
                                           "weeks_out"]].to_string(index=False))

    pairs = pd.read_csv(OUT_DIR / "doppelgangers.csv")
    t, no_match = budget_table(w, pairs)
    print(f"\nBUDGET DOPPELGANGERS (star APY > ${STAR_APY:.0f}M; best non-teammate, non-hub "
          f"match with sim >= {MIN_SIM} and APY < {MAX_SHARE:.0%} of his)")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nStars with no qualifying match ({len(no_match)}): {', '.join(no_match)}")
    print(f"Matches with sim >= {STRICT_SIM}: {int(t['near_identical'].sum())}")

    w.to_csv(OUT_DIR / "wr_salary_availability.csv", index=False)
    t.to_csv(OUT_DIR / "budget_doppelgangers.csv", index=False, float_format="%.3f")
    print("\nSaved -> outputs/wr_salary_availability.csv, outputs/budget_doppelgangers.csv")


if __name__ == "__main__":
    main()
