"""Build the project website as one static page in docs/ (ready for GitHub Pages).

Every number on the page is read from outputs/ (or computed here from them), so
rerunning the pipeline and then this script keeps the site in sync. There is no
build step and no server: data is embedded as JavaScript files, so the page also
works when docs/index.html is opened straight from disk.

Interactive parts (src/web/app.js):
  - Finder: pick a WR, click a match, compare radar + speed curves side by side.
  - Watch them move: real routes of two WRs animated from the same start point.
  - Draw a route: the browser computes movement features from your drawing and a
    small model (src/route_model.py, exported as weights) guesses who runs like that.

Input:  outputs/*.csv, outputs/*.png, outputs/wr_map.html, outputs/wr_routes.parquet
Output: docs/index.html, docs/wr_map.html, docs/assets/*, docs/.nojekyll

Run:  .venv/bin/python src/website.py
Host: GitHub repo Settings -> Pages -> Deploy from branch: main, folder /docs
"""
import html
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import fingerprint as fpr
import route_model as rm
import visuals

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
WEB = ROOT / "src" / "web"
REPO_URL = "https://github.com/chirantan2003/shadow"
MATCHES_PER_WR = 25        # enough for the "built like" filter to still find 5
ROUTES_PER_WR = 10         # real routes embedded per WR for "Watch them move"
FEATURED = ("Cooper Kupp", "K.J. Osborn")
T_MAX = 3.0                # speed curves: seconds after the snap
SEED = fpr.SEED


def esc(v):
    return html.escape(str(v))


def money(m):
    return "n/a" if pd.isna(m) else (f"${m:.1f}M" if m >= 1 else f"${m * 1000:.0f}K")


def load():
    return {name: pd.read_csv(OUT / f"{name}.csv") for name in [
        "classifier_summary", "signature_scores", "doppelgangers", "budget_doppelgangers",
        "wr_salary_availability", "feature_importance", "wr_fingerprints", "ablation"]}


def finder_data(d):
    """Compact JSON for the in-page doppelganger finder."""
    pairs = d["doppelgangers"]
    sal = d["wr_salary_availability"].set_index("nflId")
    tier = d["signature_scores"].set_index("nflId")["tier"]
    hubs = set(pairs.loc[pairs["is_hub"], "nflId"])
    wrs = []
    for wr_id, g in pairs.groupby("nflId"):
        g = g.nsmallest(MATCHES_PER_WR, "rank")
        wrs.append({
            "name": g["displayName"].iat[0], "team": g["team"].iat[0],
            "apy": money(sal.loc[wr_id, "apy"]), "routes": int(sal.loc[wr_id, "n_routes"]),
            "tier": tier.get(wr_id, "n/a"), "hub": wr_id in hubs,
            "m": [[r.match_displayName, r.match_team, round(r.similarity, 2),
                   round(r.stable_rate * 100), int(r.stable), int(r.built_like),
                   int(r.match_is_hub), money(sal["apy"].get(r.match_nflId))]
                  for r in g.itertuples()],
        })
    return sorted(wrs, key=lambda w: w["name"])


def movement_data():
    """window.MOVE: speed curves, radar percentiles, sample routes and the draw model."""
    raw = pd.read_csv(OUT / "wr_fingerprints_raw.csv").set_index("nflId")
    pct = pd.DataFrame({k: (sign * raw[col]).rank(pct=True) * 100
                        for k, (col, sign) in visuals.RADAR.items()})
    f = fpr.load_routes()
    keys = f[["gameId", "playId", "nflId"]]

    # Average speed every 0.1 s after the snap (tracking speed), per WR and league quartiles.
    r = pd.read_parquet(rm.ROUTES_FILE, columns=["gameId", "playId", "nflId", "frameId",
                                                 "snapFrame", "s"]).merge(keys)
    r["t"] = ((r["frameId"] - r["snapFrame"]) / 10).round(1)
    r = r[(r["t"] >= 0) & (r["t"] <= T_MAX)]
    league = r.groupby("t")["s"].quantile([0.25, 0.5, 0.75]).unstack().round(2)
    speed = r.groupby(["nflId", "t"])["s"].mean().unstack().round(2)

    # A random sample of real routes per WR, moved to start at (0, 0).
    # x = yards toward the middle of the field, y = yards downfield; stored in 0.1 yd.
    sample = f.groupby("nflId", group_keys=False).sample(ROUTES_PER_WR, random_state=SEED)
    pos = rm.route_positions(sample[["gameId", "playId", "nflId"]])
    routes = {}
    for (g, p, wr), xy in pos.items():
        side = 1 if xy[0, 1] < 53.3 / 2 else -1             # + = toward the middle of the field
        inside = (xy[:, 1] - xy[0, 1]) * side
        down = xy[:, 0] - xy[0, 0]
        routes.setdefault(wr, []).append(
            np.round(np.column_stack([inside, down]) * 10).astype(int).ravel().tolist())

    # The browser model, trained on position-only features of every route.
    table = rm.build_table()
    model = rm.export(table)
    ev = rm.evaluate(table)
    q = table[rm.FEATURES].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    typical = [{f"p{int(k * 100):02d}": round(float(q.loc[k, c]), 3) for k in q.index}
               for c in rm.FEATURES]
    typical[rm.FEATURES.index("has_cut")]["p50"] = round(float(table["has_cut"].mean()), 3)

    names = f.groupby("nflId").agg(name=("displayName", "first"),
                                   team=("team", lambda t: t.mode().iat[0]))
    wr = {}
    for wid, row in names.iterrows():
        wr[row["name"]] = {"team": row["team"], "pct": pct.loc[wid].round(1).tolist(),
                           "speed": speed.loc[wid].tolist(), "routes": routes[wid]}
    move = {"t": league.index.tolist(),
            "league": {"q25": league[0.25].tolist(), "q50": league[0.5].tolist(),
                       "q75": league[0.75].tolist()},
            "traits": [k.split("\n")[0] for k in visuals.RADAR], "wr": wr,
            "model": model, "typical": typical}
    return move, ev


def stable_pairs(d):
    p = d["doppelgangers"]
    p = p[(p["rank"] <= 5) & p["stable"] & ~p["is_hub"] & ~p["match_is_hub"]].copy()
    p["pair"] = p.apply(lambda r: tuple(sorted([r["displayName"], r["match_displayName"]])),
                        axis=1)
    return p.sort_values("stable_rate", ascending=False).drop_duplicates("pair")


def budget_rows(d):
    rows = []
    for r in d["budget_doppelgangers"].sort_values("savings", ascending=False).itertuples():
        badge = ('<span class="badge ok">Stable</span>' if r.stable
                 else '<span class="badge">Tentative</span>')
        built = "Yes" if r.built_like else "No"
        rows.append(
            f"<tr data-a='{esc(r.star)}' data-b='{esc(r.doppelganger)}' "
            f"title='Compare {esc(r.star)} with {esc(r.doppelganger)}'>"
            f"<td class='star'>{esc(r.star)} <span class='team'>{esc(r.star_team)}</span></td>"
            f"<td class='num'>{money(r.star_apy)}</td><td class='num'>{int(r.weeks_injured)}</td>"
            f"<td class='dbl'>{esc(r.doppelganger)} <span class='team'>{esc(r.dopp_team)}</span></td>"
            f"<td class='num'>{money(r.dopp_apy)}</td><td class='num'>{r.similarity:.2f}</td>"
            f"<td>{badge}</td><td>{built}</td><td class='num'><b>{money(r.savings)}</b></td></tr>")
    return "\n".join(rows)


def stable_rows(d):
    return "\n".join(
        f"<tr data-a='{esc(r.displayName)}' data-b='{esc(r.match_displayName)}' "
        f"title='Compare {esc(r.displayName)} with {esc(r.match_displayName)}'>"
        f"<td class='star'>{esc(r.displayName)} <span class='team'>{esc(r.team)}</span></td>"
        f"<td class='dbl'>{esc(r.match_displayName)} <span class='team'>{esc(r.match_team)}</span></td>"
        f"<td class='num'>{r.similarity:.2f}</td><td class='num'>{r.stable_rate:.0%}</td>"
        f"<td>{'Yes' if r.built_like else 'No'}</td></tr>"
        for r in stable_pairs(d).itertuples())


def tier_lists(d):
    s = d["signature_scores"]
    out = []
    for tier, label in [("distinctive", "Distinctive movers"), ("generic", "Generic movers")]:
        names = ", ".join(f"{esc(r.displayName)}" for r in s[s["tier"] == tier]
                          .sort_values("displayName").itertuples())
        out.append(f"<div class='tier'><h4>{label}</h4><p>{names}</p></div>")
    return "\n".join(out)


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WR Doppelgänger: movement fingerprints from NFL tracking data</title>
<meta name="description" content="Every WR moves in a measurable way. We fingerprint 110 receivers from 2021 Next Gen Stats tracking data and find who moves like whom, and who does it for less.">
<style>
  :root { --star:#2a78d6; --dbl:#eb6834; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
          --grid:#e1e0d9; --bg:#fcfcfb; --card:#ffffff; --soft:#f3f6f1; --hl:#eef4fc; }
  * { box-sizing:border-box; }
  html { scroll-behavior:smooth; scroll-padding-top:56px; }
  body { margin:0; background:var(--bg); color:var(--ink); line-height:1.55;
         font:16px/1.55 -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif; }
  header { background:#0d1b2a; color:#fff; padding:52px 20px 40px; }
  .wrap { max-width:1200px; margin:0 auto; padding:0 20px; }
  header .wrap { padding:0; }
  .kicker { text-transform:uppercase; letter-spacing:.12em; font-size:13px; color:#9ec5f4; }
  h1 { font-size:44px; margin:8px 0 12px; line-height:1.1; }
  header p.lede { font-size:20px; max-width:780px; color:#d6dde6; margin:0 0 26px; }
  .stats { display:grid; grid-template-columns:repeat(4, 1fr); gap:14px; }
  .stat { background:rgba(255,255,255,.08); border-radius:10px; padding:14px 18px; }
  .stat b { display:block; font-size:28px; color:#fff; }
  .stat span { font-size:14px; color:#c9d2dc; }
  .cta { margin-top:22px; display:flex; gap:10px; flex-wrap:wrap; }
  .cta a { color:#fff; text-decoration:none; border:1px solid rgba(255,255,255,.35); border-radius:999px;
           padding:8px 16px; font-size:15px; }
  .cta a.primary { background:var(--star); border-color:var(--star); }
  .cta a:hover { border-color:#fff; }
  nav { position:sticky; top:0; background:rgba(252,252,251,.96); backdrop-filter:blur(6px);
        border-bottom:1px solid var(--grid); z-index:5; }
  nav .wrap { display:flex; gap:2px; overflow-x:auto; white-space:nowrap; }
  nav a { display:inline-block; padding:13px 12px 11px; color:var(--ink2); text-decoration:none; font-size:15px;
          border-bottom:2px solid transparent; }
  nav a:hover { color:var(--star); }
  nav a.on { color:var(--star); border-bottom-color:var(--star); }
  nav a.new::after { content:"new"; font-size:10px; margin-left:5px; padding:1px 5px; border-radius:6px;
                     background:#e3eefc; color:#184f95; vertical-align:2px; }
  section { padding:48px 0 8px; }
  h2 { font-size:28px; margin:0 0 6px; }
  h3 { font-size:19px; margin:18px 0 8px; }
  p.sub { color:var(--ink2); margin-top:0; max-width:860px; }
  .card { background:var(--card); border:1px solid var(--grid); border-radius:12px; padding:18px; margin:16px 0; }
  img.fig { width:100%; height:auto; display:block; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:16px; align-items:start; }
  .finder-grid { display:grid; grid-template-columns:1.1fr 1fr; gap:16px; align-items:start; }
  .sticky { position:sticky; top:64px; }
  .play-grid { display:grid; grid-template-columns:1.55fr 1fr; gap:16px; align-items:start; }
  @media (max-width:960px) {
    .grid2, .finder-grid, .play-grid { grid-template-columns:1fr; }
    .stats { grid-template-columns:1fr 1fr; } .sticky { position:static; }
    h1 { font-size:34px; }
  }
  iframe { width:100%; height:820px; border:0; }
  table { border-collapse:collapse; width:100%; font-size:15px; }
  th { text-align:left; color:var(--ink2); font-weight:600; border-bottom:2px solid var(--muted); padding:8px 8px; }
  td { border-bottom:1px solid var(--grid); padding:8px 8px; vertical-align:middle; }
  td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
  td.star { color:var(--star); font-weight:600; } td.dbl { color:var(--dbl); font-weight:600; }
  tr.pick { cursor:pointer; } tr.pick:hover td { background:#f7f8f6; }
  tr.pick:focus { outline:2px solid var(--star); outline-offset:-2px; }
  tr.sel td { background:var(--hl); } tr.sel td:first-child { box-shadow:inset 3px 0 0 var(--dbl); }
  .team { color:var(--muted); font-weight:400; font-size:13px; }
  .badge { font-size:12px; padding:2px 8px; border-radius:20px; background:#eee; color:var(--ink2); white-space:nowrap; margin-right:4px; }
  .badge.ok { background:#dff3e4; color:#17602c; font-weight:600; }
  .badge.hub { background:#f6ead8; color:#7a4a00; }
  .badge.built { background:#e3eefc; color:#184f95; }
  .scroll { overflow-x:auto; }
  .controls { display:flex; gap:10px 14px; flex-wrap:wrap; align-items:center; margin-bottom:12px; }
  .controls label { font-size:14px; color:var(--ink2); display:flex; align-items:center; gap:6px; }
  select, input[type=text] { font:inherit; font-size:15px; padding:7px 10px; border:1px solid var(--grid); border-radius:8px; background:#fff; color:var(--ink); }
  #wr-search { width:200px; } #wr-select { min-width:240px; }
  .btn { font:inherit; font-size:15px; padding:7px 14px; border-radius:8px; border:1px solid var(--grid);
         background:#fff; color:var(--ink); cursor:pointer; }
  .btn:hover { border-color:var(--muted); }
  .btn.primary { background:var(--star); border-color:var(--star); color:#fff; }
  .btn.primary:hover { background:#256abf; }
  .btn:focus-visible, select:focus-visible, input:focus-visible { outline:2px solid var(--star); outline-offset:1px; }
  .mini { font:inherit; font-size:12px; padding:2px 8px; border-radius:6px; border:1px solid var(--grid); background:#fff; cursor:pointer; color:var(--ink2); }
  .who { font-size:15px; color:var(--ink2); margin:4px 0 12px; }
  .bar { display:inline-block; height:8px; background:var(--dbl); border-radius:4px; vertical-align:middle; margin-right:6px; }
  .tier h4 { margin:6px 0 2px; } .tier p { margin:0 0 10px; color:var(--ink2); font-size:15px; }
  .note { background:#fff8ec; border-left:4px solid #e0a43a; padding:10px 14px; border-radius:6px; color:#4a3b20; font-size:15px; }
  ul.tight li, ol.tight li { margin-bottom:6px; }
  .vs { font-size:20px; font-weight:600; margin:0 0 2px; }
  .vs .a { color:var(--star); } .vs .b { color:var(--dbl); } .vs .x { color:var(--muted); font-weight:400; margin:0 6px; }
  .legend { display:flex; gap:16px; flex-wrap:wrap; font-size:13px; color:var(--ink2); margin:6px 0 2px; }
  .legend i { display:inline-block; width:14px; height:4px; border-radius:2px; vertical-align:middle; margin-right:6px; }
  .tabs { display:flex; gap:4px; border-bottom:1px solid var(--grid); margin-bottom:14px; overflow-x:auto; }
  .tabs button { font:inherit; font-size:15px; background:none; border:0; padding:8px 12px; color:var(--ink2);
                 cursor:pointer; border-bottom:2px solid transparent; white-space:nowrap; }
  .tabs button[aria-selected=true] { color:var(--ink); border-bottom-color:var(--star); font-weight:600; }
  .tabcap { color:var(--ink2); font-size:15px; margin:10px 0 0; }
  canvas { display:block; width:100%; border-radius:10px; }
  #watch-canvas { height:520px; }
  @media (max-width:600px) { #watch-canvas { height:420px; } }
  #draw-canvas { aspect-ratio:30 / 28; height:auto; cursor:crosshair; touch-action:none; }
  input[type=range] { width:180px; accent-color:var(--star); }
  .meter { height:10px; background:#f0efec; border-radius:5px; overflow:hidden; margin:6px 0 2px; }
  .meter span { display:block; height:100%; border-radius:5px; transition:width .08s linear; }
  .live { padding:10px 0; border-bottom:1px solid var(--grid); }
  .live:last-of-type { border-bottom:0; }
  .live .nm { font-weight:600; }
  ol.preds { list-style:none; padding:0; margin:8px 0 12px; }
  ol.preds li { display:grid; grid-template-columns:1fr auto auto; gap:2px 10px; align-items:center; padding:7px 0;
                border-bottom:1px solid var(--grid); }
  ol.preds li.truth .pname { color:var(--star); }
  ol.preds .prow { grid-column:1; } ol.preds .pname { font-weight:600; }
  ol.preds .pbar { grid-column:1 / 2; grid-row:2; height:6px; background:var(--star); border-radius:3px; }
  ol.preds .pval { grid-column:2; grid-row:1 / 3; font-size:13px; color:var(--ink2); font-variant-numeric:tabular-nums; }
  ol.preds .mini { grid-column:3; grid-row:1 / 3; }
  .dblc { color:var(--dbl); }
  table.feat td { padding:6px 6px; font-size:14px; } table.feat th { font-size:13px; padding:6px; }
  .range { width:40%; font-size:13px; color:var(--muted); }
  .track { position:relative; display:block; height:8px; background:#f0efec; border-radius:4px; }
  .track .band { position:absolute; top:0; bottom:0; background:#cdd9ea; border-radius:4px; }
  .track .mark { position:absolute; top:-4px; width:4px; height:16px; margin-left:-2px; border-radius:2px; background:var(--star); }
  .seg { display:inline-flex; border:1px solid var(--grid); border-radius:8px; overflow:hidden; }
  .seg label { padding:6px 10px; cursor:pointer; color:var(--ink2); }
  .seg input { position:absolute; opacity:0; }
  .seg label:has(input:checked) { background:var(--hl); color:var(--ink); font-weight:600; }
  footer { padding:36px 0 50px; color:var(--muted); font-size:14px; }
  a { color:var(--star); }
</style>
</head>
<body>
<header><div class="wrap">
  <div class="kicker">NFL Big Data Bowl · NYU Edition · 2021 Next Gen Stats, weeks 1–8</div>
  <h1>WR Doppelgänger</h1>
  <p class="lede">Scouts say "he moves like Cooper Kupp." We measured it. Every receiver's routes
  become a movement fingerprint: how he releases, how hard he brakes, how much speed he keeps
  through a cut. Then we find who moves like whom, and who does it for less money.</p>
  <div class="stats">
    <div class="stat"><b>{{X_TOP1}}× chance</b><span>picks a WR out of {{N_WRS}} from half his games</span></div>
    <div class="stat"><b>{{N_ROUTES}}</b><span>routes fingerprinted, snap to throw</span></div>
    <div class="stat"><b>{{N_STABLE}} stable pairs</b><span>doppelgängers that hold up across the season</span></div>
    <div class="stat"><b>{{TOP_SAVINGS}}</b><span>per year: {{FEAT_STAR}} → {{FEAT_DBL}}</span></div>
  </div>
  <div class="cta"><a class="primary" href="#finder">Find a doppelgänger</a><a href="#watch">Watch them move</a>
    <a href="#draw">Draw a route</a></div>
</div></header>

<nav aria-label="Sections"><div class="wrap">
  <a href="#fingerprint">Fingerprint</a><a href="#finder">Find a doppelgänger</a>
  <a href="#watch" class="new">Watch them move</a><a href="#draw" class="new">Draw a route</a>
  <a href="#map">Map</a><a href="#budget">Budget</a><a href="#pair">Featured pair</a>
  <a href="#method">Method</a><a href="#limits">Limits</a><a href="{{REPO}}">Code</a>
</div></nav>

<main class="wrap">

<section id="fingerprint">
  <h2>A receiver's movement is a fingerprint</h2>
  <p class="sub">We build each WR's fingerprint from a random half of his games, then ask: is the
  closest fingerprint from his <em>other</em> half his own? Out of {{N_WRS}} receivers it is
  {{TOP1}} of the time, against {{CHANCE1}} by chance. One route alone is too short to tell
  (about 2.8 s of tracking before the throw); a season of routes is not.</p>
  <div class="card">
    <div class="tabs" role="tablist">
      <button role="tab" aria-selected="true" data-tab="fp-acc">Accuracy vs chance</button>
      <button role="tab" aria-selected="false" data-tab="fp-qb">Player or QB?</button>
      <button role="tab" aria-selected="false" data-tab="fp-route">One route → numbers</button>
      <button role="tab" aria-selected="false" data-tab="fp-data">Who has enough data</button>
    </div>
    <div id="fp-acc" role="tabpanel"><img class="fig" src="assets/identification.png" alt="Bar chart: identification accuracy vs chance for season fingerprints and single routes"></div>
    <div id="fp-qb" role="tabpanel" hidden><img class="fig" src="assets/ablation.png" alt="Bar chart: accuracy with groups of features removed">
      <p class="tabcap">Route time depends on when the QB throws. Without it the fingerprint still finds
      the player at {{X_NO_RT}}× chance, and the teammates-only score rises. It's the player, not his QB.</p></div>
    <div id="fp-route" role="tabpanel" hidden><img class="fig" src="assets/route_anatomy.png" alt="One route: path, speed over time, and its row of features"></div>
    <div id="fp-data" role="tabpanel" hidden><img class="fig" src="assets/routes_per_wr.png" alt="Histogram of routes per receiver with the 80-route cutoff"></div>
  </div>
  <p class="sub">The release decides most of it. The features that most help the model are
  <b>{{IMP1}}</b>, <b>{{IMP2}}</b> and <b>{{IMP3}}</b>, the same "get-off" and "long speed"
  that scouts grade.</p>
  <div class="card">{{TIERS}}
    <p class="who">Tiers are terciles of top-5 recall on unseen games. The order inside a tier is
    noise, so names are alphabetical.</p></div>
</section>

<section id="finder">
  <h2>Find a doppelgänger</h2>
  <p class="sub">Pick any of the {{N_WRS}} receivers, then <b>click a match</b> to compare them side by side.
  Matches come from other teams only, because a GM replacing a player shops elsewhere and teammates share
  a scheme. <b>Stable</b> means the match shows up in both halves of the season in at least half of 20
  random splits.</p>
  <div class="finder-grid">
    <div class="card">
      <div class="controls">
        <input type="text" id="wr-search" placeholder="Type a name…" aria-label="Search receivers">
        <select id="wr-select" aria-label="Receiver"></select>
        <label><input type="checkbox" id="built-only"> Also built like him (±15 lb, ±2 in)</label>
      </div>
      <div class="who" id="wr-who"></div>
      <div class="scroll"><table>
        <thead><tr><th>#</th><th>Moves like</th><th class="num">Similarity</th>
          <th class="num">Both halves</th><th>Flags</th><th class="num">2021 APY</th></tr></thead>
        <tbody id="wr-rows"></tbody>
      </table></div>
      <p class="who"><span class="badge hub">Hub</span> = a common profile that 10+ receivers resemble,
      so the match says little. <span class="badge built">Built like</span> = similar size, which the
      fingerprint does not use.</p>
    </div>
    <div class="card sticky" aria-live="polite">
      <p class="vs"><span class="a" id="cmp-a"></span><span class="x">vs</span><span class="b" id="cmp-b"></span></p>
      <div class="who" id="cmp-meta"></div>
      <div id="cmp-radar"></div>
      <h3 style="margin-top:4px">Average speed after the snap</h3>
      <div id="cmp-speed"></div>
      <div class="legend"><span><i style="background:var(--star)"></i>Picked WR</span>
        <span><i style="background:var(--dbl)"></i>Match</span>
        <span><i style="background:#ebeae4;height:10px"></i>Middle 50% of WRs</span></div>
      <div class="controls" style="margin:12px 0 0">
        <button class="btn primary" id="cmp-watch">▶ Watch them move</button>
        <button class="btn" id="cmp-draw">Test the model on his routes</button>
      </div>
    </div>
  </div>
  <h3>The doppelgängers that hold up</h3>
  <div class="card scroll"><table>
    <thead><tr><th>Receiver</th><th>Moves like</th><th class="num">Similarity</th>
      <th class="num">Both halves</th><th>Built like</th></tr></thead>
    <tbody>{{STABLE_ROWS}}</tbody></table>
    <p class="who">Click a row to compare the pair above.</p></div>
</section>

<section id="watch">
  <h2>Watch them move</h2>
  <p class="sub">Real routes from the tracking data, played back in real time from the same starting
  point. Routes differ, so look at <b>how</b> they move: how fast they get going, where they slow down,
  how sharply they turn. Dots mark every 0.5 s, so wider gaps mean faster running; an open circle is where
  the ball was thrown.</p>
  <div class="play-grid">
    <div class="card">
      <div class="controls">
        <label><span style="color:var(--star);font-weight:600">Blue</span><select id="watch-a" aria-label="First receiver"></select></label>
        <label><span style="color:var(--dbl);font-weight:600">Orange</span><select id="watch-b" aria-label="Second receiver"></select></label>
        <button class="btn" id="watch-swap" title="Set orange to blue's closest non-teammate">Closest match</button>
        <label>Routes <select id="watch-n"><option>3</option><option>5</option><option selected>10</option></select></label>
      </div>
      <canvas id="watch-canvas" aria-label="Animated routes on a field"></canvas>
      <div class="controls" style="margin:12px 0 0">
        <button class="btn primary" id="watch-play" aria-pressed="false">▶ Play</button>
        <button class="btn" id="watch-restart">↺ Restart</button>
        <input type="range" id="watch-scrub" min="0" max="3" step="0.1" value="0" aria-label="Time since snap">
        <span id="watch-time" style="font-variant-numeric:tabular-nums;min-width:48px">0.0 s</span>
        <label>Speed <select id="watch-rate"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label>
      </div>
    </div>
    <div class="card sticky">
      <h3 style="margin-top:0">Speed right now</h3>
      <p class="who">Average speed of each player's routes still running at this moment.</p>
      <div class="live"><div class="nm" style="color:var(--star)" id="watch-a-name"></div>
        <div class="meter"><span id="watch-a-bar" style="background:var(--star)"></span></div>
        <div class="who" id="watch-a-speed" style="margin:0"></div></div>
      <div class="live" id="watch-b-row"><div class="nm" style="color:var(--dbl)" id="watch-b-name"></div>
        <div class="meter"><span id="watch-b-bar" style="background:var(--dbl)"></span></div>
        <div class="who" id="watch-b-speed" style="margin:0"></div></div>
      <p class="who" style="margin-top:12px">Scale: 0 to 10 yd/s. Elite WRs top out near 10 yd/s (about 20 mph).
      These are {{ROUTES_PER_WR}} random routes per player, not hand-picked.</p>
    </div>
  </div>
</section>

<section id="draw">
  <h2>Draw a route: who runs it like that?</h2>
  <p class="sub">Draw a route on the field. We time it like a real play, turn it into the same movement
  numbers we use for real routes, and a model trained on {{N_DRAW_ROUTES}} real routes guesses which of the
  {{N_WRS}} receivers moves most like that. <b>Your pace matters, not just the shape:</b> slow down where a
  receiver would brake into a cut.</p>
  <div class="play-grid">
    <div class="card">
      <div class="controls">
        <label>Route time <input type="range" id="draw-time" min="1.5" max="5" step="0.1" value="2.8" aria-label="Route time in seconds"><span id="draw-time-label" style="font-variant-numeric:tabular-nums">2.8 s</span></label>
        <span class="seg" role="radiogroup" aria-label="Pace">
          <label><input type="radio" name="pace" value="mine" checked>My drawing pace</label>
          <label><input type="radio" name="pace" value="even">Even pace</label></span>
      </div>
      <canvas id="draw-canvas" aria-label="Field to draw a route on"></canvas>
      <div class="controls" style="margin:12px 0 0">
        <button class="btn" id="draw-replay">▶ Replay</button>
        <button class="btn" id="draw-clear">Clear</button>
        <span style="color:var(--muted)">or</span>
        <select id="draw-real" aria-label="Receiver for a real route"></select>
        <button class="btn primary" id="draw-load">Load one of his real routes</button>
      </div>
    </div>
    <div class="card" id="draw-results" aria-live="polite">
      <p class="who">Draw a route on the field to see who runs it most like that.</p>
    </div>
  </div>
  <p class="note">How much to trust it: one route is a weak clue. On games it never saw, this model's top guess
  is right {{DRAW_TOP1}} of the time ({{DRAW_X1}}× chance) and the right player is in its top 5
  {{DRAW_TOP5}} of the time ({{DRAW_X5}}× chance). That is the same as our main single-route model;
  it takes a season of routes to really pin a player down.</p>
</section>

<section id="map">
  <h2>Every receiver on one map</h2>
  <p class="sub">Nearby dots move alike. Color is 2021 pay. Pay doesn't follow movement:
  $15M stars and minimum-salary receivers share the same neighborhoods. Hover a dot for his top
  matches.</p>
  <div class="card"><iframe src="wr_map.html" title="Interactive map of WR movement fingerprints" loading="lazy"></iframe></div>
</section>

<section id="budget">
  <h2>Budget doppelgängers</h2>
  <p class="sub">For each star paid over $10M a year: the most similar receiver on another team
  with similarity of at least 0.70 who costs under half as much. Hubs are never picked.
  <b>Only stable matches are firm.</b> Treat the rest as names to scout, not answers.
  Click a row to compare the pair.</p>
  <div class="card scroll"><table>
    <thead><tr><th>Star</th><th class="num">APY</th><th class="num">Wks on injury report</th>
      <th>Moves like</th><th class="num">APY</th><th class="num">Similarity</th>
      <th>Split-half</th><th>Built like</th><th class="num">Saves / yr</th></tr></thead>
    <tbody>{{BUDGET_ROWS}}</tbody></table>
    <p class="who">APY = average per year of the contract in force in 2021 (OverTheCap via nflverse).
    Injury weeks = weeks 1–8 on the official report, any status.</p></div>
</section>

<section id="pair">
  <h2>{{FEAT_STAR}} → {{FEAT_DBL}}</h2>
  <p class="sub">The only budget match that is stable <em>and</em> built alike. Different routes,
  same motion: the same release speed and nearly the same top speed, at a fraction of the cost.</p>
  <div class="card">
    <div class="tabs" role="tablist">
      <button role="tab" aria-selected="true" data-tab="pr-speed">Speed over time</button>
      <button role="tab" aria-selected="false" data-tab="pr-radar">Six traits</button>
      <button role="tab" aria-selected="false" data-tab="pr-routes">Their routes</button>
      <button role="tab" aria-selected="false" data-tab="pr-dist">In the training data</button>
      <button role="tab" aria-selected="false" data-tab="pr-top6">His 6 closest movers</button>
    </div>
    <div id="pr-speed" role="tabpanel"><img class="fig" src="assets/speed_profile.png" alt="Average speed by time since the snap for both players"></div>
    <div id="pr-radar" role="tabpanel" hidden><div style="max-width:620px;margin:0 auto"><img class="fig" src="assets/radar.png" alt="Radar chart comparing movement percentiles"></div></div>
    <div id="pr-routes" role="tabpanel" hidden><img class="fig" src="assets/route_overlay.png" alt="Ten routes from each receiver overlaid from the same start point"></div>
    <div id="pr-dist" role="tabpanel" hidden><img class="fig" src="assets/feature_distributions.png" alt="Histograms of every feature with both players' averages"></div>
    <div id="pr-top6" role="tabpanel" hidden><img class="fig" src="assets/compare_top6.png" alt="Small radar charts: the star against his six closest movers"></div>
  </div>
</section>

<section id="method">
  <h2>How it works</h2>
  <div class="card"><ol class="tight">
    <li><b>Routes.</b> Every WR route with PFF role "Pass Route", from the snap to the throw (or sack),
      flipped so all plays go left to right. QB scrambles and routes with pre-snap motion are dropped.
      Player tracking only; the ball is never used.</li>
    <li><b>Movement features per route.</b> Release speed and burst in the first second, top and
      average speed, hardest braking, sharpest 1-second turn after the release, speed kept through
      that cut, path straightness, route time. Nothing about where he lines up.</li>
    <li><b>Fingerprint per WR.</b> The average, 25th and 75th percentile of each feature, plus how often
      he makes a real cut (28 numbers), standardized. Receivers with 80+ routes ({{N_WRS}}).</li>
    <li><b>Is it real?</b> Split-half test (above), plus a gradient-boosting model that guesses the
      receiver from one route, tested on games it never saw.</li>
    <li><b>Doppelgängers.</b> Cosine similarity between fingerprints, teammates excluded, with a
      stability check across 20 random season splits.</li>
    <li><b>Draw a route.</b> The same features computed from positions only (10 per second), and a
      logistic regression on the {{N_WRS}} receivers, exported as weights so it runs in your browser.
      Python and the browser give identical numbers for the same route.</li>
  </ol>
  <p>Data: <a href="https://github.com/ThompsonJamesBliss/nfl-big-data-bowl-regional-event-data">Big Data Bowl regional event data</a>
  (NFL Next Gen Stats, PFF). Contracts and injury reports: <a href="https://github.com/nflverse/nflverse-data">nflverse</a>.
  Code, notebook and every output: <a href="{{REPO}}">{{REPO}}</a>.</p></div>
</section>

<section id="limits">
  <h2>Limits</h2>
  <div class="card"><ul class="tight">
    <li><b>Most doppelgänger lists are not stable.</b> A WR's top 5 overlaps only about 2× more than
      chance between season halves, because each half has only ~40 routes. Headline only the stable pairs.</li>
    <li><b>One route is a weak clue.</b> The draw tool and the single-route model are right about
      {{DRAW_X1}}× more often than chance, not more. A season fingerprint is far stronger.</li>
    <li><b>Eight weeks, passing plays only,</b> and tracking stops at the throw, so deep breaks after the
      throw are never seen.</li>
    <li><b>Role, not just style.</b> The teammates test controls for QB and scheme but not slot vs.
      outside roles. Several Giants receivers are among the hardest to identify, which may be a team or
      role effect we did not investigate.</li>
    <li><b>Salary is approximate.</b> We use the most recent contract signed in or before 2021, so
      fifth-year options are missed (Mike Williams shows $4.9M, not about $15.7M), and rookie deals make
      young stars look cheap.</li>
    <li><b>"Moves like" is not "built like" or "produces like".</b> Size and catch skills are not in the fingerprint.</li>
  </ul></div>
</section>
</main>

<footer><div class="wrap">Built at the NFL Big Data Bowl NYU Edition hackathon with Python
(pandas, scikit-learn, matplotlib, plotly) and Kiro. <a href="{{REPO}}">Source on GitHub</a>.</div></footer>

<script>
var WRS = {{DATA}};
var SITE = {"star": "{{FEAT_STAR_JS}}", "dbl": "{{FEAT_DBL_JS}}"};
</script>
<script src="assets/movement.js"></script>
<script src="assets/route_features.js"></script>
<script src="assets/app.js"></script>
</body>
</html>
"""


def to_js(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")


def main():
    d = load()
    DOCS.mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    star, dbl = FEATURED
    s1, s2 = visuals.slug(star), visuals.slug(dbl)
    for src, dst in [("identification.png", "identification.png"),
                     ("ablation.png", "ablation.png"),
                     ("routes_per_wr.png", "routes_per_wr.png"),
                     (f"route_anatomy_{s1}.png", "route_anatomy.png"),
                     (f"radar_{s1}_vs_{s2}.png", "radar.png"),
                     (f"route_overlay_{s1}_vs_{s2}.png", "route_overlay.png"),
                     (f"speed_profile_{s1}_vs_{s2}.png", "speed_profile.png"),
                     (f"feature_distributions_{s1}_vs_{s2}.png", "feature_distributions.png"),
                     (f"compare_{s1}_top6.png", "compare_top6.png")]:
        shutil.copy(OUT / src, ASSETS / dst)
    for js in ["route_features.js", "app.js"]:
        shutil.copy(WEB / js, ASSETS / js)
    shutil.copy(OUT / "wr_map.html", DOCS / "wr_map.html")
    (DOCS / ".nojekyll").touch()

    move, ev = movement_data()
    (ASSETS / "movement.js").write_text("window.MOVE = " + to_js(move) + ";\n", encoding="utf-8")

    clf = d["classifier_summary"].set_index("test")
    top1 = clf.loc["Split-half fingerprint, top-1"]
    no_rt = d["ablation"].set_index("fingerprint").loc["Without route time (QB timing)"]
    imp = d["feature_importance"]["feature"].str.replace("_", " ").tolist()
    bud = d["budget_doppelgangers"]
    feat = bud[(bud["star"] == star) & (bud["doppelganger"] == dbl)].iloc[0]

    values = {
        "X_TOP1": f"{top1['x_chance']:.0f}", "TOP1": f"{top1['accuracy']:.1%}",
        "CHANCE1": f"{top1['chance']:.1%}", "N_WRS": str(len(d["wr_fingerprints"])),
        "N_ROUTES": f"{int(d['wr_fingerprints']['n_routes'].sum()):,}",
        "N_STABLE": str(len(stable_pairs(d))), "TOP_SAVINGS": money(feat["savings"]),
        "X_NO_RT": f"{no_rt['top1_x']:.0f}",
        "FEAT_STAR": esc(star), "FEAT_DBL": esc(dbl),
        "FEAT_STAR_JS": star.replace('"', ""), "FEAT_DBL_JS": dbl.replace('"', ""),
        "IMP1": esc(imp[0]), "IMP2": esc(imp[1]), "IMP3": esc(imp[2]),
        "TIERS": tier_lists(d), "STABLE_ROWS": stable_rows(d), "BUDGET_ROWS": budget_rows(d),
        "ROUTES_PER_WR": str(ROUTES_PER_WR), "N_DRAW_ROUTES": f"{int(d['wr_fingerprints']['n_routes'].sum()):,}",
        "DRAW_TOP1": f"{ev['top1']:.1%}", "DRAW_X1": f"{ev['top1'] / ev['chance1']:.1f}",
        "DRAW_TOP5": f"{ev['top5']:.1%}", "DRAW_X5": f"{ev['top5'] / ev['chance5']:.1f}",
        "REPO": REPO_URL, "DATA": to_js(finder_data(d)),
    }
    page = PAGE
    for key, val in values.items():
        page = page.replace("{{" + key + "}}", val)
    assert "{{" not in page, "unfilled placeholder in page"
    (DOCS / "index.html").write_text(page, encoding="utf-8")
    kb = lambda p: p.stat().st_size / 1024
    print(f"Draw model on unseen games: top-1 {ev['top1']:.1%}, top-5 {ev['top5']:.1%}")
    print(f"Saved -> docs/index.html ({kb(DOCS / 'index.html'):.0f} KB), "
          f"docs/assets/movement.js ({kb(ASSETS / 'movement.js'):.0f} KB), app.js, route_features.js, "
          f"9 images, docs/wr_map.html")


if __name__ == "__main__":
    main()
