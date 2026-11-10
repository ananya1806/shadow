"""Charts for the WR Doppelganger story. Everything is saved to outputs/.

1. wr_map.html                      PCA map of all WR fingerprints (interactive)
2. radar_<star>_vs_<double>.png     star vs his budget doppelganger, percentiles 0-100
3. budget_table.png                 top 10 budget doppelgangers by savings
4. identification.png               "who ran this route?" accuracy vs chance
5. route_overlay_<star>_vs_<double>.png   10 routes each, same starting point

Color scheme (same everywhere): star = blue, doppelganger = orange,
chance / reference = gray. Titles state the finding, not just the variable.

Input:  outputs/*.csv from fingerprint.py, doppelganger.py, budget.py,
        outputs/wr_routes.parquet, outputs/route_features.parquet
Run:    .venv/bin/python src/visuals.py            (Kupp vs Osborn by default)
        .venv/bin/python src/visuals.py "Stefon Diggs"   (any star in budget_doppelgangers.csv)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

import fingerprint as fpr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SEED = 0

# ---- palette (validated: blue/orange pass CVD + contrast on the light surface)
STAR = "#2a78d6"        # blue   = the star
DOUBLE = "#eb6834"      # orange = his doppelganger
CHANCE = "#c3c2b7"      # gray   = random chance / reference
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

_installed = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
plt.rcParams.update({
    "font.family": [f for f in ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]
                    if f in _installed] or ["DejaVu Sans"],
    "font.size": 12, "axes.titlesize": 15, "axes.labelsize": 12,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 100, "savefig.dpi": 160,
    "text.parse_math": False,          # titles contain "$" amounts, not math
})

FAMOUS = ["Cooper Kupp", "Davante Adams", "Tyreek Hill", "Justin Jefferson", "Stefon Diggs",
          "DeAndre Hopkins", "Ja'Marr Chase", "Mike Evans", "Keenan Allen", "CeeDee Lamb"]

# Radar axes: label -> (raw fingerprint column, sign). sign=-1 flips a column
# so that "higher percentile" always means "more of the trait".
RADAR = {
    "Release burst\n(accel, 1st sec)": ("release_accel_mean", 1),
    "Top speed": ("top_speed_mean", 1),
    "Hard braking\n(peak decel)": ("max_decel_mean", -1),
    "Cut sharpness": ("cut_sharpness_mean", 1),
    "Speed kept\nin cut": ("speed_kept_in_cut_mean", 1),
    "Straight paths": ("straightness_mean", 1),
}


# ------------------------------------------------------------------ loading
def load_training():
    """Just the per-route training table (needs only load_data.py + features.py)."""
    names = pd.read_csv(fpr.PLAYERS_FILE, usecols=["nflId", "displayName"])
    return {"feats": pd.read_parquet(OUT / "route_features.parquet").merge(names, on="nflId")}


def load():
    d = {
        "z": pd.read_csv(OUT / "wr_fingerprints.csv"),
        "raw": pd.read_csv(OUT / "wr_fingerprints_raw.csv"),
        "pairs": pd.read_csv(OUT / "doppelgangers.csv"),
        "salary": pd.read_csv(OUT / "wr_salary_availability.csv"),
        "budget": pd.read_csv(OUT / "budget_doppelgangers.csv"),
        "clf": pd.read_csv(OUT / "classifier_summary.csv"),
    }
    d["ids"] = d["z"].set_index("displayName")["nflId"]
    names = pd.read_csv(fpr.PLAYERS_FILE, usecols=["nflId", "displayName"])
    d["feats"] = pd.read_parquet(OUT / "route_features.parquet").merge(names, on="nflId")
    return d


def slug(name):
    return "".join(c for c in name.lower().replace(" ", "_") if c.isalnum() or c == "_")


def money(m):
    return f"${m:.1f}M" if m >= 1 else f"${m * 1000:.0f}K"


# ------------------------------------------------------------------ 1. map
def wr_map(d):
    z = d["z"]
    cols = [c for c in z.columns if c not in ("nflId", "displayName", "team", "n_routes")]
    pca = PCA(n_components=2, random_state=SEED)
    xy = pca.fit_transform(z[cols])
    df = z[["nflId", "displayName", "team", "n_routes"]].copy()
    df["pc1"], df["pc2"] = xy[:, 0], xy[:, 1]
    df = df.merge(d["salary"][["nflId", "apy"]], on="nflId", how="left")

    top3 = (d["pairs"][d["pairs"]["rank"] <= 3].sort_values(["nflId", "rank"])
            .groupby("nflId").apply(lambda g: "<br>".join(
                f"  {r.rank}. {r.match_displayName} ({r.match_team}) · sim {r.similarity:.2f}"
                for r in g.itertuples()), include_groups=False))
    df["top3"] = df["nflId"].map(top3)

    # Name each axis by the trait that loads on it most.
    def axis_name(i):
        load = pd.Series(pca.components_[i], index=cols)
        base = load.abs().groupby(load.index.str.rsplit("_", n=1).str[0]).sum()
        lead = base.sort_values(ascending=False).index[:2]
        return (f"PC{i + 1} ({pca.explained_variance_ratio_[i]:.0%} of variance): "
                f"mostly {lead[0].replace('_', ' ')}, {lead[1].replace('_', ' ')}")

    rho = max(abs(spearmanr(df["apy"], df["pc1"], nan_policy="omit")[0]),
              abs(spearmanr(df["apy"], df["pc2"], nan_policy="omit")[0]))
    if rho < 0.3:
        title = ("Pay doesn't follow movement: $15M+ stars and minimum-salary WRs "
                 "share the same neighborhoods")
    else:
        title = "Similar movers cluster together, and pay partly follows the map"
    sub = (f"Each dot is one WR (≥80 routes, 2021 wks 1–8). Nearby = similar movement fingerprint. "
           f"Rank correlation of APY with either axis: |ρ| ≤ {rho:.2f}. Hover for top-3 doppelgängers.")

    hover = ("<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
             "APY: %{customdata[2]}<br>Routes: %{customdata[3]}<br>"
             "<b>Moves most like</b> (other teams):<br>%{customdata[4]}<extra></extra>")
    custom = np.stack([df["displayName"], df["team"],
                       df["apy"].map(lambda v: money(v) if pd.notna(v) else "n/a"),
                       df["n_routes"], df["top3"].fillna("")], axis=1)
    scale = [[i / (len(BLUE_RAMP) - 1), c] for i, c in enumerate(BLUE_RAMP)]

    fig = go.Figure(go.Scatter(
        x=df["pc1"], y=df["pc2"], mode="markers", customdata=custom, hovertemplate=hover,
        marker=dict(size=12, color=np.log10(df["apy"]), colorscale=scale,
                    line=dict(width=1.5, color=SURFACE),
                    colorbar=dict(title=dict(text="2021 APY", font=dict(color=INK_2)),
                                  tickvals=np.log10([0.75, 1, 2, 5, 10, 20]),
                                  ticktext=["$0.75M", "$1M", "$2M", "$5M", "$10M", "$20M"],
                                  outlinewidth=0, thickness=12, tickfont=dict(color=MUTED)))))

    fam = df[df["displayName"].isin(FAMOUS)]
    missing = sorted(set(FAMOUS) - set(fam["displayName"]))
    if missing:
        print(f"  map: not in the 110-WR fingerprint set, so not labeled: {', '.join(missing)}")
    placed = []                        # (x, y, ay) of labels so far; flip below if crowded
    for r in fam.sort_values("pc1").itertuples():
        crowded = any(abs(r.pc1 - x) < 2.2 and abs(r.pc2 - y) < 0.6 and a < 0 for x, y, a in placed)
        ay = 24 if crowded else -22
        placed.append((r.pc1, r.pc2, ay))
        fig.add_annotation(x=r.pc1, y=r.pc2, text=f"<b>{r.displayName}</b>", showarrow=True,
                           arrowhead=0, arrowwidth=1, arrowcolor=MUTED, ax=0, ay=ay,
                           font=dict(size=12, color=INK), bgcolor="rgba(252,252,251,0.85)")

    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><span style='font-size:13px;color:{INK_2}'>{sub}</span>",
                   x=0.01, xanchor="left", font=dict(size=19, color=INK)),
        font=dict(family="Segoe UI, -apple-system, system-ui, sans-serif", color=INK),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, width=1150, height=780,
        margin=dict(t=110, l=70, r=40, b=70),
        xaxis=dict(title=axis_name(0), gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(color=MUTED)),
        yaxis=dict(title=axis_name(1), gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(color=MUTED)),
        hoverlabel=dict(bgcolor="white", font=dict(size=13, color=INK), bordercolor=GRID),
    )
    path = OUT / "wr_map.html"
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"Saved -> {path.relative_to(ROOT)}")
    return fig


# ---------------------------------------------------------------- 2. radar
def percentiles(d):
    """Radar traits as percentiles (0-100) among the 110 fingerprinted WRs."""
    raw = d["raw"].set_index("displayName")
    return pd.DataFrame({label: (sign * raw[col]).rank(pct=True) * 100
                         for label, (col, sign) in RADAR.items()})


def pair_sim(d, star, other):
    hit = d["pairs"].query("displayName == @star and match_displayName == @other")
    return hit["similarity"].iloc[0] if len(hit) else np.nan


def draw_radar(ax, a, b, star, other, small=False):
    """Star (blue) vs another WR (orange) on one polar axis."""
    labels = list(RADAR)
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    ang_c = np.append(ang, ang[0])
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    for vals, color, name in [(a, STAR, star), (b, DOUBLE, other)]:
        v = np.append(vals, vals[0])
        ax.plot(ang_c, v, color=color, lw=1.8 if small else 2.2, label=name)
        ax.fill(ang_c, v, color=color, alpha=0.12)
        ax.scatter(ang, vals, s=24 if small else 46, color=color, edgecolor=SURFACE,
                   linewidth=1.5, zorder=3)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels([] if small else ["25", "50", "75", "100th pct"], color=MUTED, fontsize=9)
    ax.set_rlabel_position(180 / len(labels))
    ax.set_xticks(ang)
    short = [l.split("\n")[0] for l in labels] if small else labels
    ax.set_xticklabels(short, fontsize=9.5 if small else 12, color=INK_2 if small else INK)
    ax.tick_params(axis="x", pad=4 if small else 14)
    ax.grid(color=GRID, lw=0.8)
    ax.spines["polar"].set_color(GRID)
    return int((np.abs(a - b) <= 15).sum())


def radar(d, star, double):
    pct = percentiles(d)
    a, b = pct.loc[star].to_numpy(), pct.loc[double].to_numpy()
    sim = pair_sim(d, star, double)

    fig, ax = plt.subplots(figsize=(8.6, 8.4), subplot_kw=dict(polar=True))
    close = draw_radar(ax, a, b, star, double)
    fig.suptitle(f"{double} matches {star} on {close} of {len(RADAR)} movement traits",
                 fontsize=16, fontweight="bold", x=0.03, ha="left", y=0.985)
    fig.text(0.03, 0.925, f"Percentile among 110 WRs (0–100). Cosine similarity of full fingerprints: "
             f"{sim:.2f}. 'Match' = within 15 percentile points.", color=INK_2, fontsize=11)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=12)
    fig.subplots_adjust(top=0.82, bottom=0.12)
    return save(fig, f"radar_{slug(star)}_vs_{slug(double)}.png")


def save(fig, name):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path.relative_to(ROOT)}")
    return path


# ---------------------------------------------------------- 3. budget table
def budget_table(d, n=10):
    t = d["budget"].sort_values("savings", ascending=False).head(n)
    rows = [[r.star, r.star_team, money(r.star_apy), r.doppelganger, r.dopp_team,
             money(r.dopp_apy), f"{r.similarity:.2f}",
             "Stable" if r.stable else "Tentative", f"{money(r.savings)}"]
            for r in t.itertuples()]
    header = ["Star", "Team", "Star APY", "Moves like", "Team", "APY", "Sim.",
              "Split-half", "Saves / yr"]
    widths = [0.15, 0.06, 0.09, 0.19, 0.06, 0.08, 0.06, 0.12, 0.09]

    fig, ax = plt.subplots(figsize=(13, 0.48 * (n + 1) + 1.7))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=header, colWidths=widths, loc="upper left",
                   cellLoc="left", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    for (i, j), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.visible_edges = "B"
        cell.PAD = 0.04
        align = "right" if j in (2, 5, 6, 8) else "center" if j == 7 else "left"
        cell._loc = align
        cell.get_text().set_horizontalalignment(align)
        if i == 0:
            cell.set_text_props(color=INK_2, fontweight="bold", fontsize=11)
            cell.set_facecolor(SURFACE)
            cell.set_edgecolor(MUTED)
            continue
        cell.set_facecolor(SURFACE)
        if j == 0:
            cell.set_text_props(color=STAR, fontweight="bold")
        elif j == 3:
            cell.set_text_props(color=DOUBLE, fontweight="bold")
        elif j == 8:
            cell.set_text_props(fontweight="bold")
        elif j == 7:
            stable = rows[i - 1][7] == "Stable"
            cell.get_text().set_text("✓ Stable" if stable else "Tentative")
            cell.set_text_props(color=INK if stable else MUTED,
                                fontweight="bold" if stable else "normal")
        elif j in (1, 4):
            cell.set_text_props(color=INK_2)

    total = t["savings"].sum()
    n_stable = int(t["stable"].sum())
    fig.suptitle(f"Ten cheaper WRs who move like stars: {money(t['savings'].min())}–"
                 f"{money(t['savings'].max())} saved per year each", fontsize=16,
                 fontweight="bold", x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.905, f"Best non-teammate match with similarity ≥ 0.70 at under half the star's 2021 APY "
             f"(OverTheCap via nflverse). Only {n_stable} of {len(t)} stay a top-5 match in both halves "
             f"of the games; treat the rest as leads to scout, not answers.",
             color=INK_2, fontsize=10.5, wrap=True)
    fig.subplots_adjust(top=0.84, bottom=0.02, left=0.01, right=0.99)
    path = OUT / "budget_table.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path.relative_to(ROOT)}  (top {n} total savings {money(total)})")


# --------------------------------------------------------- 4. identification
def identification(d):
    c = d["clf"].copy()
    nice = {
        "Split-half fingerprint, top-1": "Season fingerprint · top-1",
        "Split-half fingerprint, top-5": "Season fingerprint · top-5",
        "Split-half fingerprint, teammates only": "Season fingerprint · vs. teammates",
        "Single route, all WRs, top-1": "Single route · top-1",
        "Single route, all WRs, top-5": "Single route · top-5",
        "Single route, teammates only": "Single route · vs. teammates",
    }
    c["label"] = c["test"].map(nice)
    c = c.iloc[::-1].reset_index(drop=True)          # first test on top
    y = np.arange(len(c))
    h = 0.36

    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.barh(y + h / 2 + 0.01, c["accuracy"] * 100, height=h, color=STAR, label="Our model")
    ax.errorbar(c["accuracy"] * 100, y + h / 2 + 0.01, xerr=c["sd"] * 100, fmt="none",
                ecolor=INK_2, elinewidth=1, capsize=3)
    ax.barh(y - h / 2 - 0.01, c["chance"] * 100, height=h, color=CHANCE, label="Random guess")
    for i, r in c.iterrows():
        ax.text(r.accuracy * 100 + r.sd * 100 + 0.8, i + h / 2, f"{r.accuracy:.1%}  ·  "
                f"{r.x_chance:.1f}× chance", va="center", fontsize=11,
                color=INK, fontweight="bold" if r.x_chance >= 5 else "normal")
        ax.text(r.chance * 100 + 0.8, i - h / 2, f"{r.chance:.1%}", va="center",
                fontsize=10, color=MUTED)
    ax.axhline(2.5, color=GRID, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(c["label"], fontsize=12, color=INK)
    ax.set_xlabel("Accuracy on games the model never saw (%)")
    ax.set_xlim(0, max(c["accuracy"].max(), c["chance"].max()) * 100 * 1.32)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=11)

    top1 = c.loc[c["test"] == "Split-half fingerprint, top-1"].iloc[0]
    fig.suptitle(f"Movement is a fingerprint: from half his games, we pick a WR out of 110 "
                 f"at {top1.x_chance:.0f}× chance", fontsize=15.5, fontweight="bold",
                 x=0.01, ha="left", y=0.99)
    fig.text(0.01, 0.925, "One route is too short to tell (≈2.8 s of tracking), but a season of "
             "routes is. 'vs. teammates' = same QB and scheme, so scheme alone can't explain it.",
             color=INK_2, fontsize=11)
    fig.subplots_adjust(top=0.86, left=0.25, right=0.98, bottom=0.1)
    path = OUT / "identification.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path.relative_to(ROOT)}")


# --------------------------------------------------------- 5. route overlay
def route_paths(nflId, n=10, seed=SEED):
    """n random routes for one WR (from the feature set, so QB-run and motion
    routes are excluded), shifted to start at (0, 0). Lateral is flipped so
    + always means 'toward the middle of the field'."""
    feats = pd.read_parquet(OUT / "route_features.parquet", columns=["gameId", "playId", "nflId"])
    keys = feats[feats["nflId"] == nflId].sample(n, random_state=seed)
    r = pd.read_parquet(OUT / "wr_routes.parquet").merge(keys, on=["gameId", "playId", "nflId"])
    r = r[(r["frameId"] >= r["snapFrame"]) & (r["frameId"] <= r["endFrame"])]
    out = []
    for _, g in r.sort_values("frameId").groupby(["gameId", "playId"]):
        x0, y0 = g["x"].iloc[0], g["y"].iloc[0]
        side = 1 if y0 < 53.3 / 2 else -1
        out.append(pd.DataFrame({"down": g["x"].to_numpy() - x0,
                                 "inside": (g["y"].to_numpy() - y0) * side,
                                 "t": (g["frameId"].to_numpy() - g["snapFrame"].iloc[0]) / 10}))
    return out


def route_overlay(d, star, double):
    paths = {star: route_paths(d["ids"][star]), double: route_paths(d["ids"][double])}
    raw = d["raw"].set_index("displayName")
    allp = pd.concat([p for v in paths.values() for p in v])
    xlim = (min(-6, allp["inside"].min() - 2), max(6, allp["inside"].max() + 2))
    ylim = (min(-2, allp["down"].min() - 1), allp["down"].max() + 3)

    fig, axes = plt.subplots(1, 3, figsize=(15, 7.4), sharex=True, sharey=True)
    panels = [(star, [star]), (double, [double]), ("Both, overlaid", [star, double])]
    for ax, (title, who) in zip(axes, panels):
        ax.set_facecolor("#f3f6f1")
        for yd in range(0, int(ylim[1]) + 1, 5):
            ax.axhline(yd, color="#dfe5da", lw=1, zorder=0)
            ax.text(xlim[0] + 0.4, yd + 0.3, f"{yd} yd", color=MUTED, fontsize=8)
        ax.axhline(0, color=MUTED, lw=1.2, zorder=0)
        for name in who:
            color = STAR if name == star else DOUBLE
            for p in paths[name]:
                ax.plot(p["inside"], p["down"], color=color, lw=1.6, alpha=0.85)
                half = p[np.isclose(p["t"] % 0.5, 0) & (p["t"] > 0)]
                ax.scatter(half["inside"], half["down"], s=10, color=color, zorder=3,
                           edgecolor="white", linewidth=0.6)
                ax.scatter(p["inside"].iloc[-1], p["down"].iloc[-1], s=34, marker="o",
                           facecolor="white", edgecolor=color, linewidth=1.5, zorder=4)
        ax.scatter([0], [0], s=60, color=INK, zorder=5)
        if len(who) == 1:
            r = raw.loc[who[0]]
            ax.set_title(f"{who[0]}\nrelease {r.release_speed_mean:.1f} yd/s · top {r.top_speed_mean:.1f} yd/s",
                         color=STAR if who[0] == star else DOUBLE, fontsize=13, fontweight="bold")
        else:
            ax.set_title(f"{title}\n", fontsize=13, fontweight="bold", color=INK)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("← sideline      yards      middle of field →")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Yards downfield from the snap")

    sim = d["pairs"].query("displayName == @star and match_displayName == @double")["similarity"].iloc[0]
    fig.suptitle(f"Different routes, same motion: {double} accelerates and runs like {star}",
                 fontsize=16, fontweight="bold", x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.925, f"10 random routes each (seed {SEED}), snap to throw, every route moved to start "
             f"at the black dot. Small dots = every 0.5 s (wider gaps = faster); open circle = "
             f"ball thrown. Fingerprint similarity {sim:.2f}.", color=INK_2, fontsize=11)
    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.05, right=0.99, wspace=0.08)
    path = OUT / f"route_overlay_{slug(star)}_vs_{slug(double)}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path.relative_to(ROOT)}")


# ------------------------------------------------------- 6. training data
FEATURE_LABELS = {
    "release_speed": "Release speed (yd/s at 1.0 s)",
    "release_accel": "Release burst (peak accel, yd/s²)",
    "top_speed": "Top speed (yd/s)",
    "avg_speed": "Average speed (yd/s)",
    "max_decel": "Hardest braking (yd/s²)",
    "cut_sharpness": "Sharpest cut (degrees in 1 s)",
    "speed_kept_in_cut": "Speed kept through cut (share)",
    "straightness": "Straightness (1 = straight line)",
    "route_time": "Route time, snap to throw (s)",
}


def routes_per_wr(d):
    """How many routes each WR ran; only WRs past the cutoff get a fingerprint."""
    n = d["feats"].groupby("nflId").size()
    cut = fpr.MIN_ROUTES
    kept, total = int((n >= cut).sum()), len(n)
    share = n[n >= cut].sum() / n.sum()
    bins = np.arange(0, n.max() + 20, 10)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.hist(n[n < cut], bins=bins, color=CHANCE, rwidth=0.88, label=f"Under {cut} routes: left out")
    ax.hist(n[n >= cut], bins=bins, color=STAR, rwidth=0.88, label=f"{cut}+ routes: fingerprinted")
    ax.axvline(cut, color=INK_2, lw=1.2)
    ax.text(cut + 2, ax.get_ylim()[1] * 0.92, f"cutoff = {cut} routes", color=INK_2, fontsize=11)
    ax.set_xlabel("Routes run, weeks 1–8 (after cleaning)")
    ax.set_ylabel("Number of WRs")
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.suptitle(f"{kept} of {total} WRs ran {cut}+ routes, enough for a reliable fingerprint",
                 fontsize=15.5, fontweight="bold", x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.92, f"Those {kept} WRs ran {share:.0%} of all {n.sum():,} routes. "
             f"Fewer routes = a noisy average, so the rest are left out.", color=INK_2, fontsize=11)
    fig.subplots_adjust(top=0.84)
    return save(fig, "routes_per_wr.png")


def _one_route(d, star):
    """A typical route with a real cut for this WR, as tracking frames."""
    f = d["feats"]
    wid = int(f.loc[f["displayName"] == star, "nflId"].iloc[0])
    mine = f[(f["nflId"] == wid) & (f["has_cut"] == 1) & f["route_time"].between(2.6, 3.8)]
    allr = pd.read_parquet(OUT / "wr_routes.parquet", filters=[("nflId", "==", wid)])
    allr = allr[(allr["frameId"] >= allr["snapFrame"]) & (allr["frameId"] <= allr["endFrame"])]
    depth = allr.groupby(["gameId", "playId"])["x"].agg(lambda x: x.max() - x.iloc[0]).rename("depth")
    mine = mine.join(depth, on=["gameId", "playId"])
    mine = mine[mine["depth"] >= 8]
    # Most typical of his routes: closest to his own average on every continuous feature.
    cols = list(FEATURE_LABELS)
    z = (mine[cols] - f[cols].mean()) / f[cols].std()
    zbar = (f.loc[f["nflId"] == wid, cols].mean() - f[cols].mean()) / f[cols].std()
    pick = mine.loc[((z - zbar) ** 2).sum(axis=1).idxmin()]
    r = allr[(allr["gameId"] == pick.gameId) & (allr["playId"] == pick.playId)]
    r = r.sort_values("frameId").reset_index(drop=True)
    x0, y0 = r["x"].iloc[0], r["y"].iloc[0]
    side = 1 if y0 < 53.3 / 2 else -1
    r["down"], r["inside"] = r["x"] - x0, (r["y"] - y0) * side
    r["t"] = np.arange(len(r)) / 10
    r["ds"] = r["s"].diff() * 10
    ahead = r["dir"].shift(-10)
    turn = (((ahead - r["dir"]) + 180) % 360 - 180).abs()
    r["turn"] = turn.where((r.index >= 10) & (r["s"] > 3) & ahead.notna())
    return r, pick


def route_anatomy(d, star):
    """Raw tracking for one route -> the features the model sees."""
    r, pick = _one_route(d, star)
    c = int(r["turn"].idxmax())                       # cut window = frames c .. c+10
    rel, top = 10, int(r["s"].idxmax())
    brake = int(r.loc[1:, "ds"].idxmin())

    fig = plt.figure(figsize=(15.5, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.45, 0.95], wspace=0.28)
    ax0, ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    # 1. Path
    ax0.plot(r["inside"], r["down"], color=STAR, lw=2.2)
    w = r.loc[c:c + 10]
    ax0.plot(w["inside"], w["down"], color=INK, lw=7, alpha=0.15, solid_capstyle="round")
    ax0.scatter(r["inside"], r["down"], s=12, color=STAR, edgecolor="white", linewidth=0.5, zorder=3)
    ax0.scatter([0], [0], s=60, color=INK, zorder=4)
    ax0.scatter(r["inside"].iloc[-1], r["down"].iloc[-1], s=60, facecolor="white", edgecolor=STAR,
                lw=2, zorder=4)
    ax0.annotate("snap", (0, 0), (8, -12), textcoords="offset points", fontsize=10, color=INK_2)
    ax0.annotate("1.0 s: release", (r["inside"][rel], r["down"][rel]), (10, 0),
                 textcoords="offset points", fontsize=10, color=INK_2, va="center")
    ax0.annotate(f"sharpest cut\n{pick.cut_sharpness:.0f}° in 1 s", (w["inside"].mean(), w["down"].mean()),
                 (14, -16), textcoords="offset points", fontsize=10, color=INK)
    ax0.annotate("throw", (r["inside"].iloc[-1], r["down"].iloc[-1]), (-10, 8), ha="right",
                 textcoords="offset points", fontsize=10, color=INK_2)
    ax0.set_aspect("equal")
    pad = 3
    ax0.set_xlim(min(-pad, r["inside"].min() - pad), max(pad, r["inside"].max() + pad + 3))
    ax0.set_ylim(min(-1, r["down"].min() - 1), r["down"].max() + 2)
    ax0.set_xlabel("yards (+ = toward middle)")
    ax0.set_ylabel("yards downfield")
    ax0.set_title("1. The path (dots every 0.1 s)", loc="left", fontsize=12.5, color=INK)
    ax0.grid(color=GRID, lw=0.8)

    # 2. Speed over time
    ax1.axvspan(r["t"][c], r["t"][min(c + 10, len(r) - 1)], color=GRID, alpha=0.7, lw=0)
    ax1.plot(r["t"], r["s"], color=STAR, lw=2.2)
    ax1.scatter(r["t"], r["s"], s=12, color=STAR, edgecolor="white", linewidth=0.5, zorder=3)
    marks = [(rel, f"release speed {r['s'][rel]:.1f} yd/s", (10, -18)),
             (top, f"top speed {r['s'][top]:.1f} yd/s", (-30, 12)),
             (brake, f"hardest braking {pick.max_decel:.1f} yd/s²", (10, -22))]
    for i, text, off in marks:
        ax1.scatter(r["t"][i], r["s"][i], s=60, facecolor="white", edgecolor=INK, lw=1.5, zorder=4)
        ax1.annotate(text, (r["t"][i], r["s"][i]), off, textcoords="offset points", fontsize=10.5,
                     color=INK)
    ax1.text(r["t"][c] + 0.03, 0.3, f"cut window\nspeed kept {pick.speed_kept_in_cut:.0%}",
             fontsize=10, color=INK_2)
    ax1.set_xlabel("seconds since snap")
    ax1.set_ylabel("speed (yd/s)")
    ax1.set_ylim(0, r["s"].max() + 1.5)
    ax1.set_title("2. Speed every 0.1 s", loc="left", fontsize=12.5, color=INK)
    ax1.grid(color=GRID, lw=0.8)

    # 3. The row of numbers the model trains on
    ax2.axis("off")
    ax2.set_title("3. One training row", loc="left", fontsize=12.5, color=INK)
    rows = [(FEATURE_LABELS[k].split(" (")[0], pick[k]) for k in FEATURE_LABELS]
    rows.insert(6, ("Has a cut (>45°)", pick.has_cut))
    fmt = {"Speed kept through cut": "{:.0%}", "Sharpest cut": "{:.0f}°", "Has a cut (>45°)": "{:.0f}",
           "Straightness": "{:.2f}"}
    y = 0.95
    for name, val in rows:
        ax2.text(0.0, y, name, fontsize=11, color=INK_2, transform=ax2.transAxes)
        ax2.text(1.0, y, fmt.get(name, "{:.1f}").format(val), fontsize=11, color=INK, ha="right",
                 fontweight="bold", transform=ax2.transAxes)
        ax2.plot([0, 1], [y - 0.025, y - 0.025], color=GRID, lw=0.8, transform=ax2.transAxes)
        y -= 0.088
    ax2.text(0.0, y - 0.02, f"label = {star}", fontsize=11, color=STAR, fontweight="bold",
             transform=ax2.transAxes)

    for ax in (ax0, ax1):
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle(f"How one route becomes training data: a typical {star} route, {pick.route_time:.1f} s from snap to throw",
                 fontsize=16, fontweight="bold", x=0.01, ha="left", y=1.0)
    fig.text(0.01, 0.925, "Tracking gives position and speed 10 times a second. We boil each route down to "
             "10 numbers about HOW he moves, never WHERE he lines up.", color=INK_2, fontsize=11)
    fig.subplots_adjust(top=0.83, left=0.05, right=0.98)
    return save(fig, f"route_anatomy_{slug(star)}.png")


def feature_distributions(d, star, double):
    """Every route's features (gray) with the two players' averages marked."""
    f, raw = d["feats"], d["raw"].set_index("displayName")
    feats = list(FEATURE_LABELS)
    close = 0
    fig, axes = plt.subplots(3, 3, figsize=(14, 9.6))
    for ax, k in zip(axes.flat, feats):
        v = f[k].dropna()
        lo, hi = v.quantile([0.01, 0.99])
        ax.hist(v.clip(lo, hi), bins=40, color=CHANCE, rwidth=0.9)
        a, b = raw.loc[star, f"{k}_mean"], raw.loc[double, f"{k}_mean"]
        ax.axvline(a, color=STAR, lw=2.4)
        ax.axvline(b, color=DOUBLE, lw=2.4)
        same = abs(a - b) <= 0.5 * raw[f"{k}_mean"].std()     # within half a player-to-player SD
        close += same
        ax.set_title(FEATURE_LABELS[k] + ("   ✓ close" if same else ""), fontsize=11.5, loc="left",
                     color=INK if same else INK_2, fontweight="bold" if same else "normal")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(length=0, labelsize=10)
    handles = [plt.Line2D([], [], color=STAR, lw=2.4), plt.Line2D([], [], color=DOUBLE, lw=2.4),
               plt.Rectangle((0, 0), 1, 1, color=CHANCE)]
    fig.legend(handles, [f"{star} average", f"{double} average", f"All {len(f):,} routes"],
               loc="upper right", bbox_to_anchor=(0.99, 0.95), ncol=3, frameon=False, fontsize=11)
    fig.suptitle(f"The training data: {double}'s average route sits next to {star}'s on {close} of "
                 f"{len(feats)} features", fontsize=16, fontweight="bold", x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.955, "Gray = every route in the data. '✓ close' = averages within half the "
             "typical gap between two WRs.", color=INK_2, fontsize=11)
    fig.subplots_adjust(top=0.88, hspace=0.5, wspace=0.12, left=0.02, right=0.99)
    return save(fig, f"feature_distributions_{slug(star)}_vs_{slug(double)}.png")


def speed_profile(d, star, double, t_max=3.0):
    """Average speed by time since snap: star, his match, his least similar WR, and the league."""
    keys = d["feats"][["gameId", "playId", "nflId"]]
    keys = keys[keys["nflId"].isin(d["z"]["nflId"])]
    r = pd.read_parquet(OUT / "wr_routes.parquet",
                        columns=["gameId", "playId", "nflId", "frameId", "snapFrame", "s"]).merge(keys)
    r["t"] = ((r["frameId"] - r["snapFrame"]) / 10).round(1)
    r = r[r["t"] <= t_max]
    league = r.groupby("t")["s"].quantile([0.25, 0.5, 0.75]).unstack()
    mine = r.groupby(["nflId", "t"])["s"].mean().unstack(0)

    p = d["pairs"][d["pairs"]["displayName"] == star]
    unlike = p.loc[p["similarity"].idxmin(), "match_displayName"]
    curve = lambda name: mine[d["ids"][name]]

    fig, ax = plt.subplots(figsize=(11.5, 6))
    ax.fill_between(league.index, league[0.25], league[0.75], color=GRID, alpha=0.8, lw=0,
                    label="Middle 50% of all fingerprinted WRs")
    ax.plot(league.index, league[0.5], color=MUTED, lw=1.2, label="Median WR")
    lines = [(star, STAR, 2.6, ""), (double, DOUBLE, 2.6, ""), (unlike, INK_2, 1.6, "  (least similar)")]
    ends = sorted((curve(n).iloc[-1], i) for i, (n, *_) in enumerate(lines))
    label_y, last = {}, -np.inf
    for y_end, i in ends:                                   # keep end labels >= 0.3 yd/s apart
        last = max(y_end, last + 0.3)
        label_y[i] = last
    for i, (name, color, lw, tag) in enumerate(lines):
        y = curve(name)
        ax.plot(y.index, y, color=color, lw=lw)
        ax.text(t_max + 0.04, label_y[i], name + tag, color=color, fontsize=11, va="center",
                fontweight="bold")
    gap = (curve(star) - curve(double)).abs().max()
    gap_u = (curve(star) - curve(unlike)).abs().max()
    ax.set_xlim(0, t_max + 1.1)
    ax.set_xticks(np.arange(0, t_max + 0.01, 0.5))
    ax.set_xlabel("seconds since snap")
    ax.set_ylabel("average speed (yd/s)")
    ax.grid(color=GRID, lw=0.8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.legend(loc="lower right", frameon=False, fontsize=10.5)
    diff = curve(double) - curve(star)
    early, late = diff[diff.index <= 1.2].abs().max(), diff[diff.index > 1.2]
    if gap <= 0.4:
        title = f"{double} speeds up like {star}: their curves never differ by more than {gap:.1f} yd/s"
    else:
        faster = "faster" if late.mean() > 0 else "slower"
        title = (f"{double} matches {star} off the line (within {early:.1f} yd/s in the first 1.2 s), "
                 f"then runs {faster}")
    fig.suptitle(title, fontsize=15.5, fontweight="bold", x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.925, f"Average speed every 0.1 s after the snap, over all their routes. For contrast, "
             f"his least similar WR ({unlike}) differs by up to {gap_u:.1f} yd/s.", color=INK_2, fontsize=11)
    fig.subplots_adjust(top=0.85, right=0.97)
    return save(fig, f"speed_profile_{slug(star)}_vs_{slug(double)}.png")


def compare_matches(d, star, k=6):
    """Star vs each of his k closest non-teammates, one small radar each."""
    pct = percentiles(d)
    apy = d["salary"].set_index("displayName")["apy"]
    top = d["pairs"][d["pairs"]["displayName"] == star].nsmallest(k, "rank").reset_index(drop=True)
    a = pct.loc[star].to_numpy()

    ncol = 3
    nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 5.6 * nrow), subplot_kw=dict(polar=True))
    for ax, m in zip(np.ravel(axes), top.itertuples()):
        close = draw_radar(ax, a, pct.loc[m.match_displayName].to_numpy(), star, m.match_displayName,
                           small=True)
        tags = ["✓ stable" if m.stable else "not stable"] + (["hub"] if m.match_is_hub else [])
        ax.set_title(f"{m.rank}. {m.match_displayName} ({m.match_team})\n"
                     f"sim {m.similarity:.2f} · {money(apy[m.match_displayName])} · "
                     f"{' · '.join(tags)} · {close}/6 traits",
                     fontsize=11.5, color=INK, pad=22, fontweight="bold" if m.stable else "normal")
    for ax in np.ravel(axes)[len(top):]:
        ax.set_visible(False)

    stable = top.loc[top["stable"], "match_displayName"].tolist()
    if len(stable) == 1:
        verdict = f"only {stable[0]} holds up across the season"
    elif stable:
        verdict = f"{len(stable)} of them hold up across the season"
    else:
        verdict = "none holds up across the season, so treat them as leads"
    fig.suptitle(f"The {k} non-teammates who move most like {star}: {verdict}",
                 fontsize=16, fontweight="bold", x=0.01, ha="left", y=1.0)
    fig.text(0.01, 0.955 if nrow > 1 else 0.92,
             f"Blue = {star} ({money(apy[star])} APY); orange = the match. Percentiles among 110 WRs. "
             f"'stable' = in his top 5 in both halves of the games in at least half of 20 random splits. "
             f"'hub' = resembles many WRs.", color=INK_2, fontsize=10.5)
    fig.subplots_adjust(top=0.86 if nrow > 1 else 0.75, hspace=0.55, wspace=0.35)
    return save(fig, f"compare_{slug(star)}_top{k}.png")


def ablation(d):
    """Split-half test with feature groups removed: is it the player, or his QB?"""
    f = fpr.load_routes()
    speed = ["release_speed", "release_accel", "top_speed", "avg_speed"]
    path = ["max_decel", "cut_sharpness", "speed_kept_in_cut", "has_cut", "straightness"]
    variants = [("All features", []),
                ("Without route time (QB timing)", ["route_time"]),
                ("Speed features only", path + ["route_time"]),
                ("Cut and path features only", speed + ["route_time"])]
    rows = []
    for name, drop in variants:
        r, _ = fpr.split_half_test(f, drop)
        rows.append({"fingerprint": name, "top1": r["accuracy"][0], "top1_x": r["x_chance"][0],
                     "top5": r["accuracy"][1], "team": r["accuracy"][2], "team_x": r["x_chance"][2]})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "ablation.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 4.6))
    y = np.arange(len(t))[::-1]
    ax.barh(y, t["top1_x"], height=0.55, color=STAR)
    ax.axvline(1, color=MUTED, lw=1.2)
    ax.text(1.1, y[0] + 0.42, "random = 1×", color=MUTED, fontsize=10)
    for yi, rr in zip(y, t.itertuples()):
        ax.text(rr.top1_x + 0.2, yi, f"{rr.top1_x:.1f}× chance  ·  vs. teammates {rr.team_x:.2f}×",
                va="center", fontsize=11, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(t["fingerprint"], fontsize=12, color=INK)
    ax.set_xlim(0, t["top1_x"].max() * 1.55)
    ax.set_xlabel("Top-1 accuracy as a multiple of chance (split-half test, unseen games)")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)
    no_rt = t.iloc[1]
    fig.suptitle(f"It's the player, not his QB: without route time we still find him at "
                 f"{no_rt.top1_x:.0f}× chance", fontsize=15.5, fontweight="bold", x=0.01, ha="left", y=1.0)
    fig.text(0.01, 0.885, f"Route time depends on when the QB throws. Dropping it barely moves accuracy, and the "
             f"teammates-only score rises ({t.team_x[0]:.2f}× → {no_rt.team_x:.2f}×).\nSpeed carries most "
             f"of the signal; cuts and paths add a real but smaller part.", color=INK_2, fontsize=10.5)
    fig.subplots_adjust(top=0.76, left=0.26)
    save(fig, "ablation.png")
    return t


# ------------------------------------------------------------------- main
def main(star="Cooper Kupp"):
    d = load()
    row = d["budget"].loc[d["budget"]["star"] == star]
    if row.empty:
        sys.exit(f"{star} has no budget doppelganger. Pick one of: {', '.join(d['budget']['star'])}")
    double = row["doppelganger"].iloc[0]
    print(f"Featured pair: {star} -> {double}")
    wr_map(d)
    radar(d, star, double)
    budget_table(d)
    identification(d)
    route_overlay(d, star, double)
    routes_per_wr(d)
    route_anatomy(d, star)
    feature_distributions(d, star, double)
    speed_profile(d, star, double)
    compare_matches(d, star)
    ablation(d)


if __name__ == "__main__":
    main(*sys.argv[1:2])
