# What is not in this repo, and how to rebuild it

Some files are git-ignored because they are large, rebuildable, or machine-specific.
This page lists each one and the exact steps to reproduce every output on a new machine.

## Not in the repo

| Path | What it is | Size | How to get it back |
|---|---|---|---|
| `data/` | Raw Big Data Bowl data (games, plays, players, PFF, 122 tracking files) | ~1 GB | Clone the source repo (step 2) |
| `outputs/wr_routes.parquet` | WR route frames, snap to throw, normalized | 8 MB | `src/load_data.py` (~7 s) |
| `outputs/route_features.parquet` | One row of movement features per route | 0.5 MB | `src/features.py` (~1 s) |
| `.venv/` | Python env for the pipeline | - | `requirements.txt` (step 1) |
| `.venv-nfl/` | Separate env for `nfl_data_py` (it needs pandas < 2) | - | `requirements-nfl.txt` (step 1) |
| `scratch/` | Throwaway checks, not part of the pipeline (see below) | - | Not needed to reproduce outputs |

**In the repo** (no rebuild needed): all code in `src/`, every CSV in `outputs/`, and
`outputs/external/` (snapshots of the public contract and injury data, so the salary
step runs without `nfl_data_py`).

## Reproduce from scratch

Tested on macOS with Python 3.9.6. Run everything from the repo root.

**1. Environments**
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Optional: only if you want to re-download contracts/injuries
python3 -m venv .venv-nfl
.venv-nfl/bin/pip install -r requirements-nfl.txt
```

**2. Raw data.** The code expects the CSVs at `data/data/` (the source repo's own layout).
```bash
git clone --depth 1 https://github.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data.git data
```
Source commit used: `85da22eeff2f1d5be106faa9dfe06a1205f2defd`.
Check: `data/data/` should contain `games.csv`, `plays.csv`, `players.csv`,
`pffScoutingData.csv` and `tracking/` with 122 `tracking_*.csv` files.

**3. Pipeline, in order**
```bash
.venv/bin/python src/load_data.py        # -> outputs/wr_routes.parquet        (~7 s)
.venv/bin/python src/features.py         # -> outputs/route_features.parquet   (~1 s)
.venv/bin/python src/fingerprint.py      # -> wr_fingerprints*.csv, signature_scores.csv,
                                         #    classifier_summary.csv, feature_importance.csv (~1 min)
.venv/bin/python src/doppelganger.py     # -> doppelgangers.csv                 (~1 min)
.venv/bin/python src/budget.py           # -> wr_salary_availability.csv, budget_doppelgangers.csv
```

**4. Optional: refresh the public salary and injury data**
```bash
.venv-nfl/bin/python src/fetch_external.py   # -> outputs/external/*.csv
```
Skip this to get identical results: nflverse updates these files over time, and the
committed snapshots are what the current outputs were built from.

## Expected numbers (to confirm the rebuild worked)

| Step | Check |
|---|---|
| `load_data.py` | 22,900 routes kept, 210 WRs, 0 duplicate rows |
| `features.py` | 1,235 QB-run routes and 649 pre-snap-motion routes dropped; 21,016 routes |
| `fingerprint.py` | 110 WRs with >= 80 routes; split-half top-1 about 11x chance |
| `doppelganger.py` | 13 hubs; 5 stable non-teammate pairs after removing hubs |
| `budget.py` | Salary match 100% of 110 WRs; 15 budget doppelgangers |

All random steps use `SEED = 0`, so the numbers should match exactly with the
library versions in `requirements.txt`. Other versions may shift the classifier
and stability numbers slightly.

## What is in `scratch/` (not needed to reproduce)

One-off checks whose results are quoted in our notes. They read the pipeline outputs.

| File | What it checked |
|---|---|
| `stress_test.py` | Pre-snap motion, slot vs outside, what drives the Signature Score, and Signature Score stability across game halves |
| `stability_dims.py` | Whether a smaller fingerprint (10 means vs 28 columns) gives more stable doppelgangers (it does not) |
| `hub_check.py` | Whether hub WRs are near league average (they are not; +0.42 correlation with distance from average) |
| `peek_contracts.py`, `peek_names.py` | Contract fields, birth-date formats, and name mismatches for the salary join |

## Charts, notebook and website

```bash
.venv/bin/python src/visuals.py          # -> outputs/*.png, outputs/wr_map.html
.venv/bin/python src/website.py          # -> docs/ (the GitHub Pages site)
.venv/bin/jupyter nbconvert --to notebook --execute demo.ipynb --output demo.ipynb
```
`demo.ipynb` runs every step above in order (about 3 minutes).

## Known gaps

- `budget.py` uses the most recent contract signed in or before 2021, so fifth-year
  options are missed (Mike Williams shows $4.9M instead of about $15.7M).
