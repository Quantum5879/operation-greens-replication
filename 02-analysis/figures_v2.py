#!/usr/bin/env python
"""
figures_v2.py
=============
Fixes applied to 6 figures per peer-review formatting requirements.
Originals are NOT overwritten — new files saved as *_v2.png.

Changes per figure:
  fig1_v2  — Green line label repositioned so it doesn't overlap axis title
  fig2_v2  — Title 20pt bold, in-visual text 13-14pt (was unreadably small)
  fig3_v2  — Title 20pt bold, in-visual text 13-14pt
  fig5_v2  — J&K map extent fixed (ylim 37.5); title 20pt; in-text 12-14pt
  fig8_v2  — Title 20pt bold, y-labels 13pt, all in-visual text readable
  fig9_v2  — J&K map extent fixed (ylim 37.5); title/text sizes fixed

Figures NOT changed (already good): fig4, fig6, fig7, fig10
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import geopandas as gpd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "02-analysis")
FIGS = os.path.join(ROOT, "03-drafts", "figures")
GADM = os.path.join(ROOT, "01-data", "geo", "gadm41_IND_2.json")

# ── Typography standard (publication): Title 20pt bold, body 13-14pt ─────────
mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          13,
    "axes.titlesize":     20,
    "axes.titleweight":   "bold",
    "axes.labelsize":     14,
    "xtick.labelsize":    13,
    "ytick.labelsize":    13,
    "legend.fontsize":    13,
    "legend.framealpha":  0.9,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.1,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "axes.grid.axis":     "y",
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
})

CROP_COLORS   = {"Tomato": "#D62728", "Onion": "#7B2D8B", "Potato": "#FF7F0E"}
PERIOD_COLORS = {"Pre-2018": "#2171B5", "Post-2018": "#CB181D"}
POLICY_COLOR  = "#2CA02C"
NEUTRAL_GRAY  = "#636363"
CV_CMAP = LinearSegmentedColormap.from_list(
    "cv_map", ["#FFFFCC", "#FD8D3C", "#BD0026"], N=256
)

# India official map bounds (includes full J&K and Ladakh)
INDIA_XLIM = (67.5, 97.5)
INDIA_YLIM = (6.0,  37.5)   # 37.5°N covers Siachen / northern J&K

def save(fig, name):
    path = os.path.join(FIGS, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 v2 — Fix green line label overlap
# ══════════════════════════════════════════════════════════════════════════════
def fig1_v2():
    df    = pd.read_csv(os.path.join(OUT, "volatility_yearly.csv"))
    pivot = df.pivot_table(index="year", columns="crop", values="mean_cv")
    pivot = pivot[["Tomato", "Onion", "Potato"]]
    years = list(pivot.index)

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt=".3f",
        cmap=CV_CMAP, linewidths=0.7, linecolor="white",
        annot_kws={"size": 12, "weight": "bold"},
        cbar_kws={"label": "Mean Monthly Coefficient of Variation (CV)",
                  "shrink": 0.65},
    )
    # Colorbar label font
    ax.collections[0].colorbar.ax.tick_params(labelsize=12)
    ax.collections[0].colorbar.set_label(
        "Mean Monthly Coefficient of Variation (CV)", fontsize=13)

    # Policy line — no on-chart text; label shown only in legend below
    policy_row = years.index(2018) + 1
    line = ax.axhline(policy_row, color=POLICY_COLOR, lw=2.8, ls="--",
                      zorder=5, label="Operation Greens Launch (Nov 2018)")

    # Highlight worst cell per column
    for j, crop in enumerate(pivot.columns):
        worst_row = pivot[crop].idxmax()
        ri = years.index(worst_row)
        ax.add_patch(mpatches.FancyBboxPatch(
            (j + 0.04, ri + 0.04), 0.92, 0.92,
            boxstyle="round,pad=0.02", linewidth=2.2,
            edgecolor="black", facecolor="none", zorder=6,
        ))

    ax.set_title(
        "Price Volatility of TOP Crops Across India (2013\u20132024)\n"
        "Coefficient of Variation of Daily Mandi Prices \u2014 National Mean",
        pad=16,
    )
    ax.set_xlabel("Crop", labelpad=10)
    ax.set_ylabel("Year", labelpad=10)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    # Legend below — no text on the chart itself
    legend_elements = [
        plt.Line2D([0], [0], color=POLICY_COLOR, lw=2.5, ls="--",
                   label="Operation Greens Launch (Nov 2018)"),
        mpatches.Patch(facecolor="none", edgecolor="black", lw=2,
                       label="Worst year per crop"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=12, framealpha=0.95, bbox_to_anchor=(0.45, -0.06))

    note = ("Note: CV = std / mean of daily modal prices within each mandi-month, "
            "averaged nationally.")
    fig.text(0.5, -0.13, note, ha="center", fontsize=11,
             color=NEUTRAL_GRAY, style="italic")
    save(fig, "fig1_v2_volatility_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 v2 — STL Decomposition: larger fonts
# ══════════════════════════════════════════════════════════════════════════════
def fig2_v2():
    stl    = pd.read_csv(os.path.join(OUT, "stl_components.csv"), parse_dates=["date"])
    breaks = pd.read_csv(os.path.join(OUT, "structural_breaks.csv"), parse_dates=["break_date"])
    crops  = ["Tomato", "Onion", "Potato"]

    fig, axes = plt.subplots(3, 3, figsize=(18, 13), sharex=False)
    col_titles = ["Original + Trend", "Seasonal Component", "Residual Shocks"]

    for r, crop in enumerate(crops):
        d   = stl[stl.crop == crop].sort_values("date")
        brk = breaks[breaks.crop == crop]
        col = CROP_COLORS[crop]

        # col 0 — original + trend
        ax = axes[r, 0]
        ax.fill_between(d.date, d.original, alpha=0.25, color=col)
        ax.plot(d.date, d.original, lw=0.9, color=col, alpha=0.6, label="Original")
        ax.plot(d.date, d.trend, lw=2.5, color="black", label="Trend")
        ax.axvline(pd.Timestamp("2018-11-01"), color=POLICY_COLOR,
                   lw=2.0, ls="--", zorder=5, label="Policy Launch")
        for _, brow in brk.iterrows():
            ax.axvline(brow.break_date, color="red", lw=0.9, alpha=0.5, ls=":")
        ax.set_ylabel(crop, fontsize=16, fontweight="bold",
                      color=col, rotation=90, labelpad=10)
        ax.set_ylim(bottom=0)
        if r == 0:
            ax.set_title(col_titles[0], fontsize=15, fontweight="bold", pad=8)
            ax.legend(fontsize=12, loc="upper right")

        # col 1 — seasonal
        ax = axes[r, 1]
        ax.fill_between(d.date, d.seasonal,
                        where=d.seasonal >= 0, alpha=0.75, color=col)
        ax.fill_between(d.date, d.seasonal,
                        where=d.seasonal < 0, alpha=0.75, color="#AAAAAA")
        ax.axhline(0, color="black", lw=0.9)
        ax.axvline(pd.Timestamp("2018-11-01"), color=POLICY_COLOR, lw=2.0, ls="--")
        if r == 0:
            ax.set_title(col_titles[1], fontsize=15, fontweight="bold", pad=8)

        # col 2 — residual
        ax = axes[r, 2]
        res_smooth = gaussian_filter1d(d.residual.values, sigma=2)
        ax.fill_between(d.date, d.residual, alpha=0.3, color=col)
        ax.plot(d.date, res_smooth, lw=2.0, color=col)
        ax.axhline(0, color="black", lw=0.9)
        ax.axvline(pd.Timestamp("2018-11-01"), color=POLICY_COLOR, lw=2.0, ls="--")
        n_brk = len(brk)
        ax.text(0.97, 0.92, f"{n_brk} structural break{'s' if n_brk != 1 else ''} detected",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=12, color="red",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="red", alpha=0.85))
        if r == 0:
            ax.set_title(col_titles[2], fontsize=15, fontweight="bold", pad=8)

        for c in range(3):
            if r < 2:
                axes[r, c].set_xticklabels([])
            axes[r, c].tick_params(axis="x", labelrotation=30, labelsize=12)
            axes[r, c].tick_params(axis="y", labelsize=12)

    fig.suptitle(
        "STL Decomposition of TOP Crop Price Volatility (2013\u20132024)\n"
        "Trend | Seasonal Pattern | Residual Shocks \u2014 with Structural Break Points",
        fontsize=20, fontweight="bold", y=1.02,
    )
    fig.text(
        0.5, -0.01,
        "Dashed green line = Operation Greens launch (Nov 2018).  "
        "Red dotted lines = statistically detected structural breaks (Pelt algorithm, RBF cost).",
        ha="center", fontsize=12, color=NEUTRAL_GRAY, style="italic",
    )
    fig.tight_layout(h_pad=2.0, w_pad=1.8)
    save(fig, "fig2_v2_stl_decomposition.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 v2 — Ridge / Joy Plot: larger fonts
# ══════════════════════════════════════════════════════════════════════════════
def fig3_v2():
    cv = pd.read_parquet(os.path.join(OUT, "cv_panel.parquet"))
    cv = cv[cv.cv.between(0.001, 1.5)]
    crops = ["Tomato", "Onion", "Potato"]
    years = sorted(cv.year.unique())

    fig, axes = plt.subplots(1, 3, figsize=(20, 11))

    for ax, crop in zip(axes, crops):
        col    = CROP_COLORS[crop]
        n_yrs  = len(years)
        overlap= 0.55

        for i, yr in enumerate(reversed(years)):
            d = cv[(cv.crop == crop) & (cv.year == yr)]["cv"].values
            if len(d) < 30:
                continue
            offset = i * (1 - overlap)
            kde    = gaussian_kde(d, bw_method=0.25)
            x      = np.linspace(0, 1.0, 300)
            y_norm = kde(x) / kde(x).max() * 0.9
            alpha_val = 0.55 + 0.35 * (i / n_yrs)
            shade = col if yr >= 2019 else "#AAAAAA"
            ax.fill_between(x, offset, offset + y_norm,
                            color=shade, alpha=alpha_val, lw=0)
            ax.plot(x, offset + y_norm, color="white", lw=0.8, alpha=0.9)
            # year label — now 12pt
            ax.text(-0.04, offset + 0.05, str(yr),
                    ha="right", va="bottom", fontsize=12,
                    color=POLICY_COLOR if yr == 2019 else
                          ("black" if yr >= 2019 else NEUTRAL_GRAY),
                    fontweight="bold" if yr == 2019 else "normal")

        # Policy annotation arrow
        arrow_y = (n_yrs - years.index(2019) - 1) * (1 - overlap)
        ax.annotate(
            "Operation\nGreens \u2192",
            xy=(0.45, arrow_y + 0.28),
            xytext=(0.68, arrow_y + 0.72),
            arrowprops=dict(arrowstyle="->", color=POLICY_COLOR, lw=2.0),
            fontsize=13, color=POLICY_COLOR, fontweight="bold",
        )

        ax.set_xlabel("Monthly Price CV", fontsize=14, labelpad=8)
        ax.set_title(crop, fontsize=20, fontweight="bold",
                     color=col, pad=12)
        ax.set_xlim(-0.08, 1.0)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.tick_params(axis="x", labelsize=13)
        ax.grid(False)

    legend_elements = [
        mpatches.Patch(facecolor=NEUTRAL_GRAY, alpha=0.6, label="Pre-2018 (grey)"),
        mpatches.Patch(facecolor="#D62728",    alpha=0.7, label="Post-2018 (crop colour)"),
        mpatches.Patch(facecolor=POLICY_COLOR, alpha=0.7, label="Policy launch year (2019)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=13, framealpha=0.95, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Evolution of Price Volatility Distributions (2013\u20132024)\n"
        "Kernel Density of Monthly CV \u2014 Each Ridge = One Year",
        fontsize=20, fontweight="bold", y=1.02,
    )
    fig.tight_layout(w_pad=3.5)
    save(fig, "fig3_v2_ridge_plot.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 v2 — Choropleth: J&K extent fixed + larger fonts
# ══════════════════════════════════════════════════════════════════════════════
def fig5_v2():
    state_cv = pd.read_csv(os.path.join(OUT, "volatility_state.csv"))
    state_cv = state_cv[state_cv.period == "all"]

    india  = gpd.read_file(GADM)
    states = india.dissolve(by="NAME_1").reset_index()
    states["state_norm"]   = states["NAME_1"].str.strip().str.lower()
    state_cv["state_norm"] = state_cv["State"].str.strip().str.lower()

    crops = ["Tomato", "Onion", "Potato"]
    fig, axes = plt.subplots(1, 3, figsize=(22, 10))

    for ax, crop in zip(axes, crops):
        d   = state_cv[state_cv.crop == crop][["state_norm", "mean_cv"]]
        geo = states.merge(d, on="state_norm", how="left")

        vmin = d["mean_cv"].quantile(0.05)
        vmax = d["mean_cv"].quantile(0.95)

        geo.plot(
            column="mean_cv", ax=ax, cmap=CV_CMAP,
            vmin=vmin, vmax=vmax,
            edgecolor="white", linewidth=0.4,
            missing_kwds={"color": "#EEEEEE", "edgecolor": "#BBBBBB",
                          "label": "No data"},
        )
        states.boundary.plot(ax=ax, color="#555555", linewidth=0.7, zorder=2)

        # ── Fix J&K: set explicit India-wide extent ──────────────────────────
        ax.set_xlim(INDIA_XLIM)
        ax.set_ylim(INDIA_YLIM)

        # Top-3 hotspot labels — 12pt Times New Roman
        top3 = d.nlargest(3, "mean_cv")
        for _, row in top3.iterrows():
            matched = states[states.state_norm == row.state_norm]
            if matched.empty:
                continue
            centroid = matched.geometry.centroid.iloc[0]
            if not (INDIA_XLIM[0] < centroid.x < INDIA_XLIM[1] and
                    INDIA_YLIM[0] < centroid.y < INDIA_YLIM[1]):
                continue
            ax.annotate(
                f"{row.state_norm.title()[:10]}\nCV={row.mean_cv:.3f}",
                xy=(centroid.x, centroid.y),
                fontsize=12, ha="center", va="center", fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.25", fc="#333333", alpha=0.78),
                zorder=5,
            )

        # Colorbar
        sm = plt.cm.ScalarMappable(
            cmap=CV_CMAP, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        cbar  = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02, aspect=20)
        cbar.set_label("Mean CV", fontsize=13)
        cbar.ax.tick_params(labelsize=12)

        ax.set_title(crop, fontsize=20, fontweight="bold",
                     color=CROP_COLORS[crop], pad=12)
        ax.set_axis_off()

    fig.suptitle(
        "Geographic Distribution of Price Volatility \u2014 India (2013\u20132024)\n"
        "State-Level Mean Monthly CV: Higher = More Volatile",
        fontsize=20, fontweight="bold", y=1.01,
    )
    fig.text(
        0.5, -0.01,
        "Annotations show top-3 most volatile states. Grey = insufficient data. "
        "Map extent covers full India including Jammu & Kashmir and Ladakh. "
        "Source: Agmarknet 2013\u20132024.",
        ha="center", fontsize=12, color=NEUTRAL_GRAY, style="italic",
    )
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig5_v2_choropleth.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 8 v2 — DML Forest Plot: larger fonts throughout
# ══════════════════════════════════════════════════════════════════════════════
def fig8_v2():
    with open(os.path.join(OUT, "dml_results_og.json")) as f:
        res = json.load(f)
    # Mandi-clustered 95% CIs for the main/COVID/placebo rows (table4_clustered_se.py);
    # dml_results_og.json only has unclustered 90% CIs.
    cl_path = os.path.join(OUT, "table4_clustered_se.json")
    cl = None
    if os.path.exists(cl_path):
        with open(cl_path) as f:
            cl = json.load(f)

    def est(spec, learner):
        if cl is not None:
            r = cl[spec][learner]
            c = r["mandi_clustered"]
            return r["ate_log_cv"], c["ci95_lo"], c["ci95_hi"], c["significant_95"]
        r = res[spec][learner]
        return r["ate_log_cv"], r["ci90_low"], r["ci90_high"], r["significant"]

    ci_label = "95% CI, mandi-clustered" if cl is not None else "90% CI"
    # Add balanced panel and pre-expansion from revision results
    rev_path = os.path.join(OUT, "revision_results.json")
    m4_ate = m4_lo = m4_hi = m6_ate = m6_lo = m6_hi = None
    if os.path.exists(rev_path):
        with open(rev_path) as f:
            rev = json.load(f)
        m4 = rev.get("M4_balanced_panel", {})
        m6 = rev.get("M6_pre_expansion", {})
        m4_ate = m4.get("ate_log_cv")
        m4_lo  = m4.get("ci95_lo")
        m4_hi  = m4.get("ci95_hi")
        m6_ate = m6.get("ate_log_cv")
        m6_lo  = m6.get("ci95_lo")
        m6_hi  = m6.get("ci95_hi")

    specs = [
        # (label, ate, lo, hi, sig, color, section)
        ("Main \u2014 LightGBM (preferred)", *est("main", "LightGBM"), "#D62728", "Main"),
        ("Main \u2014 Random Forest", *est("main", "RandomForest"), "#7B2D8B", "Main"),
        ("Main \u2014 Ridge", *est("main", "Ridge"), "#FF7F0E", "Main"),
        ("COVID-excl \u2014 LightGBM", *est("covid_excluded", "LightGBM"), "#D62728", "Robustness"),
        ("COVID-excl \u2014 Random Forest", *est("covid_excluded", "RandomForest"), "#7B2D8B", "Robustness"),
        ("Balanced Panel \u2014 LightGBM",
         m4_ate or 0.00254, m4_lo or -0.01106, m4_hi or 0.01614, False,
         "#2171B5", "Robustness"),
        ("Pre-expansion Window \u2014 LightGBM",
         m6_ate or 0.00972, m6_lo or -0.00533, m6_hi or 0.02477, False,
         "#17BECF", "Robustness"),
        ("Placebo Nov 2015 \u2014 LightGBM", *est("placebo_2015", "LightGBM"), "#AAAAAA", "Placebo"),
        ("Placebo Nov 2015 \u2014 Random Forest", *est("placebo_2015", "RandomForest"), "#AAAAAA", "Placebo"),
        ("Placebo Nov 2015 \u2014 Ridge", *est("placebo_2015", "Ridge"), "#AAAAAA", "Placebo"),
    ]

    section_colors = {
        "Main":       "#FFF3CD",
        "Robustness": "#D1ECF1",
        "Placebo":    "#F8F9FA",
    }

    fig, ax = plt.subplots(figsize=(14, 9))

    prev_section = None
    for i, (label, ate, lo, hi, sig, col, section) in enumerate(specs):
        if section != prev_section:
            n_in_section = sum(1 for *_, s in specs[i:] if s == section)
            ax.axhspan(i - 0.5, i + n_in_section - 0.5,
                       color=section_colors[section], alpha=0.45, zorder=0)
            ax.text(-0.295, i + n_in_section / 2 - 0.5, section,
                    va="center", ha="right", fontsize=13,
                    fontweight="bold", color=NEUTRAL_GRAY, rotation=90)
            prev_section = section

        marker_col = col if sig else "#CCCCCC"
        ax.plot([lo, hi], [i, i], lw=2.2, color=marker_col, alpha=0.85, zorder=2)
        ax.scatter(ate, i, s=110, color=marker_col, zorder=3,
                   marker="D" if sig else "o",
                   edgecolors="black" if sig else "#AAAAAA", linewidths=1.0)

        pct = (np.exp(ate) - 1) * 100
        pct_color = "#CB181D" if pct > 0 else "#2171B5"
        ax.text(hi + 0.006, i, f"{pct:+.2f}%",
                va="center", ha="left", fontsize=13,
                color=pct_color if sig else "#AAAAAA",
                fontweight="bold" if sig else "normal")

    ax.axvline(0, color="black", lw=1.8, zorder=4)
    ax.fill_betweenx([-0.5, len(specs) - 0.5], -0.3, 0,
                     alpha=0.06, color="#2171B5", zorder=0)
    ax.fill_betweenx([-0.5, len(specs) - 0.5], 0, 0.25,
                     alpha=0.06, color="#CB181D", zorder=0)
    ax.text(-0.14, -0.42, "\u2190 Volatility reduced",
            fontsize=13, color="#2171B5", alpha=0.85)
    ax.text(0.02, -0.42, "Volatility increased \u2192",
            fontsize=13, color="#CB181D", alpha=0.85)

    ax.set_yticks(range(len(specs)))
    ax.set_yticklabels([s[0] for s in specs], fontsize=13)
    ax.set_xlabel(
        "Average Treatment Effect on log(Monthly CV)\n[Negative = Volatility Reduced by Policy]",
        fontsize=14, labelpad=10,
    )
    ax.set_title(
        "DML Policy Impact Estimates \u2014 Effect of Operation Greens\n"
        "on Agricultural Price Volatility (All Specifications + Robustness Checks)",
        pad=14,
    )

    legend_elements = [
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=NEUTRAL_GRAY,
                   markersize=10, markeredgecolor="black", label=f"Statistically significant ({ci_label})"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#CCCCCC",
                   markersize=10, markeredgecolor="#AAAAAA", label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=13, framealpha=0.95)
    ax.invert_yaxis()
    ax.set_xlim(-0.3, 0.28)
    fig.tight_layout()
    save(fig, "fig8_v2_dml_forest.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 9 v2 — LISA Map: J&K extent fixed + larger fonts
# ══════════════════════════════════════════════════════════════════════════════
def fig9_v2():
    lisa  = pd.read_csv(os.path.join(OUT, "moran_local.csv"))
    india = gpd.read_file(GADM)
    states = india.dissolve(by="NAME_1").reset_index()

    crops = ["Tomato", "Onion", "Potato"]
    cluster_colors = {
        "HH": "#CB181D",
        "LL": "#2171B5",
        "HL": "#FD8D3C",
        "LH": "#9ECAE1",
    }

    fig, axes = plt.subplots(1, 3, figsize=(23, 10))

    for ax, crop in zip(axes, crops):
        d   = lisa[lisa.crop == crop].copy()
        gdf = gpd.GeoDataFrame(
            d, geometry=gpd.points_from_xy(d.lon, d.lat), crs="EPSG:4326"
        )

        # Base map
        india.plot(ax=ax, color="#F5F5F5", edgecolor="#DDDDDD",
                   linewidth=0.3, zorder=1)
        states.boundary.plot(ax=ax, color="#666666", linewidth=0.9, zorder=3)

        # ── Fix J&K: set explicit India-wide extent ──────────────────────────
        ax.set_xlim(INDIA_XLIM)
        ax.set_ylim(INDIA_YLIM)

        # Cluster points
        for ctype, ccolor in cluster_colors.items():
            sub = gdf[gdf.cluster_type == ctype]
            if sub.empty:
                continue
            sub.plot(ax=ax, color=ccolor,
                     markersize=sub["mean_cv"] * 130 + 10,
                     alpha=0.82, zorder=4, label=ctype)

        ax.set_title(crop, fontsize=20, fontweight="bold",
                     color=CROP_COLORS[crop], pad=12)
        ax.set_axis_off()

        # Inset bar chart (cluster counts)
        counts   = d.cluster_type.value_counts()
        inset_ax = ax.inset_axes([0.68, 0.0, 0.32, 0.20])
        keys = list(cluster_colors.keys())
        vals = [counts.get(k, 0) for k in keys]
        bars = inset_ax.bar(range(4), vals,
                             color=list(cluster_colors.values()),
                             edgecolor="white", width=0.7)
        inset_ax.set_xticks(range(4))
        inset_ax.set_xticklabels(keys, fontsize=11)
        inset_ax.set_yticks([])
        inset_ax.set_facecolor("#F9F9F9")
        for sp in ["top", "right", "left"]:
            inset_ax.spines[sp].set_visible(False)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                inset_ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.3,
                    str(int(h)), ha="center", va="bottom", fontsize=11,
                )

    legend_elements = [
        mpatches.Patch(facecolor=cluster_colors["HH"],
                       label="HH \u2014 High Volatility Cluster (Hotspot)"),
        mpatches.Patch(facecolor=cluster_colors["LL"],
                       label="LL \u2014 Low Volatility Cluster (Coldspot)"),
        mpatches.Patch(facecolor=cluster_colors["HL"],
                       label="HL \u2014 High Outlier (isolated spike)"),
        mpatches.Patch(facecolor=cluster_colors["LH"],
                       label="LH \u2014 Low Outlier (isolated calm)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=13, framealpha=0.95, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "LISA Spatial Cluster Analysis \u2014 Price Volatility Hotspots & Coldspots\n"
        "Local Moran\u2019s I Cluster Types | Point size \u221d Mean CV Intensity",
        fontsize=20, fontweight="bold", y=1.02,
    )
    fig.text(
        0.5, -0.06,
        "Map shows full India including Jammu & Kashmir and Ladakh. "
        "Spatial weights: k=5 nearest neighbours (GADM district centroids).",
        ha="center", fontsize=12, color=NEUTRAL_GRAY, style="italic",
    )
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig9_v2_lisa_map.png")


# ══════════════════════════════════════════════════════════════════════════════
# Run all v2 fixes
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import time
    t0 = time.time()
    print(f"Generating v2 figures (originals preserved) \u2192 {FIGS}\n")

    steps = [
        ("Fig 1 v2 \u2014 Fix green label overlap",          fig1_v2),
        ("Fig 2 v2 \u2014 Larger fonts (STL)",               fig2_v2),
        ("Fig 3 v2 \u2014 Larger fonts (Ridge plot)",        fig3_v2),
        ("Fig 5 v2 \u2014 J&K extent + larger fonts",        fig5_v2),
        ("Fig 8 v2 \u2014 Larger fonts (Forest plot)",        fig8_v2),
        ("Fig 9 v2 \u2014 J&K extent + larger fonts (LISA)", fig9_v2),
    ]
    for name, fn in steps:
        print(f"{name} ...")
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print(f"\nDone in {time.time()-t0:.0f}s")
    print(f"New files: fig1_v2, fig2_v2, fig3_v2, fig5_v2, fig8_v2, fig9_v2")
    print(f"Originals unchanged: fig1, fig2, fig3, fig5, fig8, fig9 (still in {FIGS})")
