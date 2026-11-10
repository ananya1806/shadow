"""Build the project website as one static page in docs/ (ready for GitHub Pages).

Every number on the page is read from outputs/*.csv, so rerunning the pipeline
and then this script keeps the site in sync. The page has no build step and no
server: data for the doppelganger finder is embedded as JSON, so it also works
when docs/index.html is opened straight from disk.

Input:  outputs/*.csv, outputs/*.png, outputs/wr_map.html
Output: docs/index.html, docs/wr_map.html, docs/assets/*.png, docs/.nojekyll

Run:  .venv/bin/python src/website.py
Host: GitHub repo Settings -> Pages -> Deploy from branch: main, folder /docs
"""
import html
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
REPO_URL = "https://github.com/chirantan2003/shadow"
MATCHES_PER_WR = 25        # enough for the "built like" filter to still find 5
FEATURED = ("Cooper Kupp", "K.J. Osborn")


def esc(v):
    return html.escape(str(v))


def money(m):
    return "n/a" if pd.isna(m) else (f"${m:.1f}M" if m >= 1 else f"${m * 1000:.0f}K")


def load():
    return {name: pd.read_csv(OUT / f"{name}.csv") for name in [
        "classifier_summary", "signature_scores", "doppelgangers", "budget_doppelgangers",
        "wr_salary_availability", "feature_importance", "wr_fingerprints"]}


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
            f"<tr><td class='star'>{esc(r.star)} <span class='team'>{esc(r.star_team)}</span></td>"
            f"<td class='num'>{money(r.star_apy)}</td><td class='num'>{int(r.weeks_injured)}</td>"
            f"<td class='dbl'>{esc(r.doppelganger)} <span class='team'>{esc(r.dopp_team)}</span></td>"
            f"<td class='num'>{money(r.dopp_apy)}</td><td class='num'>{r.similarity:.2f}</td>"
            f"<td>{badge}</td><td>{built}</td><td class='num'><b>{money(r.savings)}</b></td></tr>")
    return "\n".join(rows)


def stable_rows(d):
    return "\n".join(
        f"<tr><td class='star'>{esc(r.displayName)} <span class='team'>{esc(r.team)}</span></td>"
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
          --grid:#e1e0d9; --bg:#fcfcfb; --card:#ffffff; --soft:#f3f6f1; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); line-height:1.55;
         font:16px/1.55 -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif; }
  header { background:#0d1b2a; color:#fff; padding:56px 20px 44px; }
  .wrap { max-width:1080px; margin:0 auto; padding:0 20px; }
  header .wrap { padding:0; }
  .kicker { text-transform:uppercase; letter-spacing:.12em; font-size:13px; color:#9ec5f4; }
  h1 { font-size:44px; margin:8px 0 12px; line-height:1.1; }
  header p.lede { font-size:20px; max-width:760px; color:#d6dde6; margin:0 0 28px; }
  .stats { display:flex; gap:16px; flex-wrap:wrap; }
  .stat { background:rgba(255,255,255,.08); border-radius:10px; padding:14px 18px; min-width:200px; }
  .stat b { display:block; font-size:30px; color:#fff; }
  .stat span { font-size:14px; color:#c9d2dc; }
  nav { position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--grid); z-index:5; }
  nav a { display:inline-block; padding:12px 12px; color:var(--ink2); text-decoration:none; font-size:15px; }
  nav a:hover { color:var(--star); }
  section { padding:44px 0 8px; }
  h2 { font-size:28px; margin:0 0 6px; }
  h2 + p.sub { color:var(--ink2); margin-top:0; max-width:820px; }
  .card { background:var(--card); border:1px solid var(--grid); border-radius:12px; padding:18px; margin:18px 0; }
  img.fig { width:100%; height:auto; display:block; }
  .two { display:grid; grid-template-columns:1fr 1.6fr; gap:18px; align-items:start; }
  @media (max-width:860px) { .two { grid-template-columns:1fr; } h1 { font-size:34px; } }
  iframe { width:100%; height:820px; border:0; }
  table { border-collapse:collapse; width:100%; font-size:15px; }
  th { text-align:left; color:var(--ink2); font-weight:600; border-bottom:2px solid var(--muted); padding:8px 8px; }
  td { border-bottom:1px solid var(--grid); padding:8px 8px; vertical-align:top; }
  td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
  td.star { color:var(--star); font-weight:600; } td.dbl { color:var(--dbl); font-weight:600; }
  .team { color:var(--muted); font-weight:400; font-size:13px; }
  .badge { font-size:12px; padding:2px 8px; border-radius:20px; background:#eee; color:var(--ink2); white-space:nowrap; }
  .badge.ok { background:#dff3e4; color:#17602c; font-weight:600; }
  .badge.hub { background:#f6ead8; color:#7a4a00; }
  .badge.built { background:#e3eefc; color:#184f95; }
  .scroll { overflow-x:auto; }
  .finder-controls { display:flex; gap:14px; flex-wrap:wrap; align-items:center; margin-bottom:12px; }
  select, input[type=text] { font-size:16px; padding:8px 10px; border:1px solid var(--grid); border-radius:8px; background:#fff; }
  #wr-search { width:220px; }
  #wr-select { min-width:280px; }
  .who { font-size:15px; color:var(--ink2); margin:4px 0 12px; }
  .bar { display:inline-block; height:8px; background:var(--dbl); border-radius:4px; vertical-align:middle; margin-right:6px; }
  .tier h4 { margin:6px 0 2px; } .tier p { margin:0 0 10px; color:var(--ink2); font-size:15px; }
  .note { background:#fff8ec; border-left:4px solid #e0a43a; padding:10px 14px; border-radius:6px; color:#4a3b20; font-size:15px; }
  ul.tight li { margin-bottom:6px; }
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
</div></header>

<nav><div class="wrap">
  <a href="#fingerprint">Fingerprint</a><a href="#finder">Find a doppelgänger</a>
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
  <div class="card"><img class="fig" src="assets/identification.png" alt="Bar chart: identification accuracy vs chance for season fingerprints and single routes"></div>
  <p class="sub">The release decides most of it. The features that most help the model are
  <b>{{IMP1}}</b>, <b>{{IMP2}}</b> and <b>{{IMP3}}</b>, the same "get-off" and "long speed"
  that scouts grade.</p>
  <div class="card">{{TIERS}}
    <p class="who">Tiers are terciles of top-5 recall on unseen games. The order inside a tier is
    noise, so names are alphabetical.</p></div>
</section>

<section id="finder">
  <h2>Find a doppelgänger</h2>
  <p class="sub">Pick any of the {{N_WRS}} receivers. Matches come from other teams only, because a
  GM replacing a player shops elsewhere and teammates share a scheme. <b>Stable</b> means the match
  shows up in both halves of the season in at least half of 20 random splits.</p>
  <div class="card">
    <div class="finder-controls">
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
  <h3>The doppelgängers that hold up</h3>
  <div class="card scroll"><table>
    <thead><tr><th>Receiver</th><th>Moves like</th><th class="num">Similarity</th>
      <th class="num">Both halves</th><th>Built like</th></tr></thead>
    <tbody>{{STABLE_ROWS}}</tbody></table></div>
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
  <b>Only stable matches are firm.</b> Treat the rest as names to scout, not answers.</p>
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
  <div class="two">
    <div class="card"><img class="fig" src="assets/radar.png" alt="Radar chart comparing movement percentiles"></div>
    <div class="card"><img class="fig" src="assets/route_overlay.png" alt="Ten routes from each receiver overlaid from the same start point"></div>
  </div>
</section>

<section id="method">
  <h2>How it works</h2>
  <div class="card"><ol class="tight">
    <li><b>Routes.</b> Every WR route with PFF role "Pass Route", from the snap to the throw (or sack),
      flipped so all plays go left to right. QB scrambles and routes with pre-snap motion are dropped.</li>
    <li><b>Movement features per route.</b> Release speed and burst in the first second, top and
      average speed, hardest braking, sharpest 1-second turn after the release, speed kept through
      that cut, path straightness, route time. Nothing about where he lines up.</li>
    <li><b>Fingerprint per WR.</b> The average, 25th and 75th percentile of each feature, plus how often
      he makes a real cut (28 numbers), standardized. Receivers with 80+ routes ({{N_WRS}}).</li>
    <li><b>Is it real?</b> Split-half test (above), plus a gradient-boosting model that guesses the
      receiver from one route, tested on games it never saw.</li>
    <li><b>Doppelgängers.</b> Cosine similarity between fingerprints, teammates excluded, with a
      stability check across 20 random season splits.</li>
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
(function () {
  var sel = document.getElementById("wr-select"), search = document.getElementById("wr-search"),
      built = document.getElementById("built-only"), rows = document.getElementById("wr-rows"),
      who = document.getElementById("wr-who");

  function el(tag, text, cls) {
    var e = document.createElement(tag);
    if (text !== undefined) e.textContent = text;
    if (cls) e.className = cls;
    return e;
  }
  function fill(filter) {
    var f = (filter || "").toLowerCase(), keep = sel.value;
    sel.innerHTML = "";
    WRS.forEach(function (w, i) {
      if (f && w.name.toLowerCase().indexOf(f) < 0) return;
      var o = el("option", w.name + " (" + w.team + ")");
      o.value = i;
      sel.appendChild(o);
    });
    if (keep && sel.querySelector('option[value="' + keep + '"]')) sel.value = keep;
  }
  function show() {
    var w = WRS[sel.value];
    rows.innerHTML = "";
    if (!w) { who.textContent = "No receiver matches that name."; return; }
    who.textContent = w.name + ", " + w.team + " · " + w.routes + " routes · 2021 APY " + w.apy +
      " · Signature tier: " + w.tier + (w.hub ? " · hub (common profile)" : "");
    var list = w.m.filter(function (m) { return !built.checked || m[5]; }).slice(0, 5);
    if (!list.length) {
      var tr = el("tr"), td = el("td", "No built-alike match in his top " + w.m.length + ".");
      td.colSpan = 6; tr.appendChild(td); rows.appendChild(tr); return;
    }
    list.forEach(function (m, i) {
      var tr = el("tr"), name = el("td", null, "dbl");
      name.appendChild(document.createTextNode(m[0] + " "));
      name.appendChild(el("span", m[1], "team"));
      var sim = el("td", null, "num"), bar = el("span", null, "bar");
      bar.style.width = Math.max(4, Math.round(m[2] * 60)) + "px";
      sim.appendChild(bar); sim.appendChild(document.createTextNode(m[2].toFixed(2)));
      var flags = el("td");
      if (m[4]) flags.appendChild(el("span", "Stable", "badge ok"));
      if (m[6]) flags.appendChild(el("span", "Hub", "badge hub"));
      if (m[5]) flags.appendChild(el("span", "Built like", "badge built"));
      [el("td", String(i + 1)), name, sim, el("td", m[3] + "%", "num"), flags,
       el("td", m[7], "num")].forEach(function (c) { tr.appendChild(c); });
      rows.appendChild(tr);
    });
  }
  fill();
  for (var i = 0; i < WRS.length; i++) if (WRS[i].name === "{{FEAT_STAR}}") sel.value = i;
  search.addEventListener("input", function () { fill(search.value); show(); });
  sel.addEventListener("change", show);
  built.addEventListener("change", show);
  show();
})();
</script>
</body>
</html>
"""


def main():
    d = load()
    DOCS.mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    star, dbl = FEATURED
    s1, s2 = (x.lower().replace(".", "").replace(" ", "_") for x in FEATURED)
    for src, dst in [("identification.png", "identification.png"),
                     (f"radar_{s1}_vs_{s2}.png", "radar.png"),
                     (f"route_overlay_{s1}_vs_{s2}.png", "route_overlay.png")]:
        shutil.copy(OUT / src, ASSETS / dst)
    shutil.copy(OUT / "wr_map.html", DOCS / "wr_map.html")
    (DOCS / ".nojekyll").touch()

    clf = d["classifier_summary"].set_index("test")
    top1 = clf.loc["Split-half fingerprint, top-1"]
    imp = d["feature_importance"]["feature"].str.replace("_", " ").tolist()
    bud = d["budget_doppelgangers"]
    feat = bud[(bud["star"] == star) & (bud["doppelganger"] == dbl)].iloc[0]
    data = json.dumps(finder_data(d), separators=(",", ":"), ensure_ascii=False)
    data = data.replace("</", "<\\/")                      # never close the <script> early

    values = {
        "X_TOP1": f"{top1['x_chance']:.0f}", "TOP1": f"{top1['accuracy']:.1%}",
        "CHANCE1": f"{top1['chance']:.1%}", "N_WRS": str(len(d["wr_fingerprints"])),
        "N_ROUTES": f"{int(d['wr_fingerprints']['n_routes'].sum()):,}",
        "N_STABLE": str(len(stable_pairs(d))), "TOP_SAVINGS": money(feat["savings"]),
        "FEAT_STAR": esc(star), "FEAT_DBL": esc(dbl),
        "IMP1": esc(imp[0]), "IMP2": esc(imp[1]), "IMP3": esc(imp[2]),
        "TIERS": tier_lists(d), "STABLE_ROWS": stable_rows(d), "BUDGET_ROWS": budget_rows(d),
        "REPO": REPO_URL, "DATA": data,
    }
    page = PAGE
    for key, val in values.items():
        page = page.replace("{{" + key + "}}", val)
    assert "{{" not in page, "unfilled placeholder in page"
    (DOCS / "index.html").write_text(page, encoding="utf-8")
    size = (DOCS / "index.html").stat().st_size / 1024
    print(f"Saved -> docs/index.html ({size:.0f} KB), docs/wr_map.html, docs/assets/ (3 images)")


if __name__ == "__main__":
    main()
