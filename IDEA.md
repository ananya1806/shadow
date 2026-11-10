# WR Doppelgänger: Project Idea and Data Notes

NFL Big Data Bowl, NYU Edition. 3-hour hackathon, due 2:00 PM.

## The idea

Scouts compare wide receivers by eye ("he moves like Cooper Kupp"). We do it with tracking data. Every WR has a movement style: how explosive his release is, how hard he slows into a cut, how much speed he keeps through a break, and how straight or wavy his paths are. We turn each WR's routes into a **movement fingerprint**, then:

1. **Prove the fingerprint is real.** Train a model that guesses *which* receiver ran a route from movement alone. If it beats random chance by a wide margin, movement style is a real, measurable trait.
2. **Find doppelgängers.** For any WR, find the receivers who move most like him (nearest neighbors in fingerprint space).
3. **Find budget doppelgängers.** For expensive or injury-prone star WRs, find cheaper players who move almost the same. A GM can use this to sign a cheaper backup or replacement.

### Who uses it

- **Scouts and GMs:** player comps and cheaper replacements in free agency or trades.
- **Coaches:** if a starter gets hurt, which backup can run his role the same way.
- **Broadcasters:** "this rookie moves like Davante Adams."

## Key rules

- Only WRs (`officialPosition == 'WR'`) with `pff_role == 'Pass Route'`.
- Normalize so every play goes left to right.
- The fingerprint describes **how** a player moves, not **where** he lines up.
- Python only: pandas, numpy, scikit-learn, scipy, matplotlib, plotly. Keep code simple and readable.

## Repo structure

```
src/load_data.py      load + filter + normalize tracking
src/features.py       per-route movement features
src/fingerprint.py    per-WR fingerprint + "who ran this route?" model
src/doppelganger.py   nearest neighbors + budget comps
src/visuals.py        radar charts, similarity maps
demo.ipynb            runs the whole pipeline end to end
outputs/              PNGs, HTML, CSVs
README.md
```

Data lives in `./data/data/` (git-ignored, 1 GB). Source: https://github.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data

## The data

2021 season, weeks 1–8, 122 games, passing plays only.

| File | One row = | Key | Rows |
|---|---|---|---|
| `games.csv` | one game | `gameId` | 122 |
| `plays.csv` | one pass play | `gameId + playId` | 8,557 |
| `players.csv` | one player | `nflId` | 1,679 (210 WRs) |
| `pffScoutingData.csv` | one player on one play | `gameId + playId + nflId` | 188,254 |
| `tracking/tracking_[gameId].csv` | one player (or the ball) at one frame (0.1 s) | `gameId + playId + nflId + frameId` | ~811 MB total |

**Joins:** `pff` → `players` on `nflId`, filter WR + `Pass Route`, then → `tracking` on `gameId + playId + nflId`. `plays` and `games` join on `gameId + playId` / `gameId` for context. Every key matches cleanly; no orphans, no duplicate keys, no fully duplicated rows in any file.

**WR route sample:** 23,033 WR route rows across 210 WRs. 103 WRs have ≥100 routes, 44 have ≥200. Leader: Chris Godwin (288).

## Data traps

Tracking checks were run on 15 games spread across the 8 weeks (1,065 plays).

1. **Duplicate snap and throw events.** About 40% of plays have both `ball_snap` and `autoevent_ballsnap`, usually 1–2 frames apart, but the auto event sometimes fires 20–140 frames early. `pass_forward` and `autoevent_passforward` overlap the same way; 8 sampled plays have only the auto version. **Rule:** use the manual event, fall back to the auto event only if the manual one is missing. 5 of 1,065 sampled plays have no snap event; drop them.
2. **Tracking stops 0.5 s after the throw, sack, or run on every play.** Snap to throw: median 2.8 s (middle half 2.3–3.6 s), about 28 frames per route. Deep routes are often thrown before the break, so break features (e.g. speed kept through the cut) are computed only when a break happens inside the window.
3. **Almost no pre-snap data.** The snap is usually at frame 6 (~0.5 s in). Fine, since the fingerprint is post-snap only.
4. **Ball rows.** `nflId`, `jerseyNumber`, `o`, `dir` are NaN and `team == 'football'`. `nflId` loads as float; cast to int before joining.
5. **`time` only changes once per second.** Use `frameId / 10` for timing.
6. **Otherwise clean.** No duplicate frames, no frame gaps, 22 players in every frame, no player speed above 12 yd/s (the 28 yd/s max is the ball).
7. **Unusual WR rows.** 44 WR rows are `Pass Block` and 1 is `Pass` (trick play); the role filter drops them. ~90 route rows have the WR lined up at TE or HB; kept by default.
8. **Direction normalization.** For `playDirection == 'left'`: `x → 120 − x`, `y → 53.3 − y`, `o, dir → (o + 180) mod 360`. To keep alignment out of the fingerprint, also mirror `y` so every route starts on the same side, and use only shape and motion features (speed, acceleration, turning, straightness), never field position.
9. **The "who ran this route?" model can cheat** by learning the QB, team scheme, or slot vs. outside alignment instead of personal style, and teammates share routes. To keep it honest: split train/test **by game**, and also report accuracy among teammates only.
10. **No salary or injury data in the dataset.** Budget doppelgängers need an outside public source (e.g. 2021 cap hits from Spotrac or OverTheCap) in a small hand-built CSV for the featured WRs.
11. **Environment.** System Python 3.9 has pandas, numpy, matplotlib; scikit-learn, scipy, plotly need installing. `data/` is 1 GB with its own `.git`, so it stays in `.gitignore`.

## Plan

- **Load:** read only WR `Pass Route` rows from all 122 tracking files. Cut each route from the snap to the throw (or sack or run), normalize direction, mirror sides.
- **Fingerprint:** ~15 movement-only features per route: release (time to 5 yd/s, peak acceleration in the first second), cuts (peak deceleration, sharpest turn rate, speed kept through the hardest cut), path (straightness, side-to-side wobble, top speed). Average per WR, using WRs with ≥100 routes.
- **Prove it's real:** classifier (random forest or logistic regression) predicts which WR ran a route, split by game. Report top-1 and top-5 accuracy vs. 1/103 chance (~1%), plus the teammates-only test.
- **Doppelgängers:** nearest neighbors on the standardized fingerprint (cosine distance); radar charts comparing a WR with his top match.
- **Budget doppelgänger + deliverables:** join a small public salary CSV to rank "same movement, lower cost" matches. Ship `demo.ipynb`, PNG/HTML outputs, and the README.

## Open decisions

1. Install missing libraries in a venv: `python3 -m venv .venv && .venv/bin/pip install scikit-learn scipy plotly jupyter`.
2. Salaries: build a 2021 cap-hit CSV for ~25 notable WRs from a cited public source, or drop "budget" and keep the pure doppelgänger finder.
