#!/usr/bin/env python
"""
Publication-Ready Figures — Operation Greens Policy Evaluation
==============================================================
All figures: 300 DPI, serif font, print-safe color palette.

Figures produced:
  fig1_volatility_heatmap.png      — Crop × Year CV heatmap (annotated)
  fig2_stl_decomposition.png       — STL trend + structural breaks, all crops
  fig3_ridge_plot.png              — Temporal evolution of CV distribution (joy plot)
  fig4_violin_prepost.png          — Pre vs Post 2018 violin + strip comparison
  fig5_choropleth.png              — India district-level mean CV maps (3 crops)
  fig6_morans_trend.png            — Moran's I over time + spatial clustering
  fig7_seasonal_heatmap.png        — Month × Crop volatility heatmap
  fig8_dml_forest.png              — DML ATE forest plot (all specs + robustness)
  fig9_lisa_map.png                — LISA HH/LL hotspot cluster map
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import geopandas as gpd
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT   = os.path.join(ROOT, "02-analysis")
FIGS  = os.path.join(ROOT, "03-drafts", "figures")
GADM  = os.path.join(ROOT, "01-data", "geo", "gadm41_IND_2.json")
os.makedirs(FIGS, exist_ok=True)

# ── Global publication style ──────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "legend.framealpha":  0.9,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "axes.grid.axis":     "y",
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
})

# ── Colour palette ────────────────────────────────────────────────────────────
CROP_COLORS = {
    "Tomato":  "#D62728",   # crimson
    "Onion":   "#7B2D8B",   # deep purple
    "Potato":  "#FF7F0E",   # amber
}
PERIOD_COLORS = {"Pre-2018":  "#2171B5", "Post-2018": "#CB181D"}
POLICY_COLOR  = "#2CA02C"
NEUTRAL_GRAY  = "#636363"

CV_CMAP = LinearSegmentedColormap.from_list(
    "cv_map", ["#FFFFCC","#FD8D3C","#BD0026"], N=256
)

def save(fig, name, tight=True):
    path = os.path.join(FIGS, name)
    fig.savefig(path, dpi=300, bbox_inches="tight" if tight else None)
    plt.close(fig)
    print(f"  Saved → {name}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Crop × Year Volatility Heatmap (annotated)
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_heatmap():
    df = pd.read_csv(os.path.join(OUT, "volatility_yearly.csv"))
    pivot = df.pivot_table(index="year", columns="crop", values="mean_cv")
    pivot = pivot[["Tomato","Onion","Potato"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt=".3f",
        cmap=CV_CMAP, linewidths=0.6, linecolor="white",
        annot_kws={"size": 10, "weight": "bold"},
        cbar_kws={"label": "Mean Monthly Coefficient of Variation (CV)", "shrink": 0.7},
    )
    # policy line — between 2018 and 2019 (row index 5 = 2018, so after row 6)
    years = list(pivot.index)
    policy_row = years.index(2018) + 1
    ax.axhline(policy_row, color=POLICY_COLOR, lw=2.5, ls="--", zorder=5)
    ax.text(3.05, policy_row - 0.15, "← Operation Greens\n     Launch (Nov 2018)",
            color=POLICY_COLOR, fontsize=9.5, fontweight="bold", va="top")

    # highlight worst cell per column
    for j, crop in enumerate(pivot.columns):
        worst_row = pivot[crop].idxmax()
        ri = years.index(worst_row)
        ax.add_patch(mpatches.FancyBboxPatch(
            (j + 0.04, ri + 0.04), 0.92, 0.92,
            boxstyle="round,pad=0.02", linewidth=2,
            edgecolor="black", facecolor="none", zorder=6,
        ))

    ax.set_title(
        "Price Volatility of TOP Crops Across India (2013–2024)\n"
        "Coefficient of Variation of Daily Mandi Prices — National Mean",
        pad=14,
    )
    ax.set_xlabel("Crop", labelpad=8)
    ax.set_ylabel("Year", labelpad=8)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    note = ("Note: CV = std/mean of daily modal prices within each mandi-month, "
            "averaged nationally. Black box = worst year per crop.")
    fig.text(0.5, -0.02, note, ha="center", fontsize=8.5, color=NEUTRAL_GRAY,
             style="italic")
    save(fig, "fig1_volatility_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — STL Decomposition with structural breaks
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_stl():
    stl     = pd.read_csv(os.path.join(OUT, "stl_components.csv"), parse_dates=["date"])
    breaks  = pd.read_csv(os.path.join(OUT, "structural_breaks.csv"), parse_dates=["break_date"])
    crops   = ["Tomato","Onion","Potato"]

    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=False)
    col_titles = ["Original + Trend", "Seasonal Component", "Residual"]

    for r, crop in enumerate(crops):
        d   = stl[stl.crop == crop].sort_values("date")
        brk = breaks[breaks.crop == crop]
        col = CROP_COLORS[crop]

        # col 0 — original + trend
        ax = axes[r, 0]
        ax.fill_between(d.date, d.original, alpha=0.25, color=col, label="Original")
        ax.plot(d.date, d.original, lw=0.8, color=col, alpha=0.6)
        ax.plot(d.date, d.trend, lw=2.2, color="black", label="Trend")
        ax.axvline(pd.Timestamp("2018-11-01"), color=POLICY_COLOR,
                   lw=1.8, ls="--", zorder=5, label="Policy Launch")
        for _, brow in brk.iterrows():
            ax.axvline(brow.break_date, color="red", lw=0.8, alpha=0.45, ls=":")
        ax.set_ylabel(crop, fontsize=12, fontweight="bold", color=col, rotation=90, labelpad=8)
        ax.set_ylim(bottom=0)
        if r == 0:
            ax.set_title(col_titles[0], fontsize=12, fontweight="bold")
            ax.legend(fontsize=8.5, loc="upper right")

        # col 1 — seasonal
        ax = axes[r, 1]
        ax.fill_between(d.date, d.seasonal, where=d.seasonal >= 0,
                        alpha=0.7, color=col, label="Positive seasonal")
        ax.fill_between(d.date, d.seasonal, where=d.seasonal < 0,
                        alpha=0.7, color="#AAAAAA", label="Negative seasonal")
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(pd.Timestamp("2018-11-01"), color=POLICY_COLOR, lw=1.8, ls="--")
        if r == 0:
            ax.set_title(col_titles[1], fontsize=12, fontweight="bold")

        # col 2 — residual + strength annotation
        ax = axes[r, 2]
        residual_smoothed = gaussian_filter1d(d.residual.values, sigma=2)
        ax.fill_between(d.date, d.residual, alpha=0.3, color=col)
        ax.plot(d.date, residual_smoothed, lw=1.8, color=col)
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(pd.Timestamp("2018-11-01"), color=POLICY_COLOR, lw=1.8, ls="--")
        # annotate n breaks
        n_brk = len(brk)
        ax.text(0.97, 0.93, f"{n_brk} breaks detected",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="red",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
        if r == 0:
            ax.set_title(col_titles[2], fontsize=12, fontweight="bold")

        # x-axis only on bottom row
        for c in range(3):
            if r < 2:
                axes[r, c].set_xticklabels([])
            axes[r, c].tick_params(axis="x", labelrotation=30)

    fig.suptitle(
        "STL Decomposition of TOP Crop Price Volatility (2013–2024)\n"
        "Trend | Seasonal Pattern | Residual Shocks — with Structural Break Points",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.text(0.5, -0.01,
             "Dashed green line = Operation Greens launch (Nov 2018). "
             "Red dotted lines = statistically detected structural breaks (Pelt algorithm).",
             ha="center", fontsize=8.5, color=NEUTRAL_GRAY, style="italic")
    fig.tight_layout(h_pad=1.5, w_pad=1.5)
    save(fig, "fig2_stl_decomposition.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Ridge / Joy Plot: temporal evolution of CV distribution
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_ridge():
    cv = pd.read_parquet(os.path.join(OUT, "cv_panel.parquet"))
    cv = cv[cv.cv.between(0.001, 1.5)]
    crops = ["Tomato","Onion","Potato"]
    years = sorted(cv.year.unique())

    fig, axes = plt.subplots(1, 3, figsize=(18, 10))

    for ax, crop in zip(axes, crops):
        col = CROP_COLORS[crop]
        n_years = len(years)
        overlap = 0.55

        for i, yr in enumerate(reversed(years)):
            d = cv[(cv.crop == crop) & (cv.year == yr)]["cv"].values
            if len(d) < 30:
                continue
            offset = i * (1 - overlap)
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(d, bw_method=0.25)
            x = np.linspace(0, 1.0, 300)
            y = kde(x)
            y_norm = y / y.max() * 0.9

            alpha_val = 0.55 + 0.35 * (i / n_years)
            shade = col if yr >= 2019 else "#AAAAAA"
            ax.fill_between(x, offset, offset + y_norm,
                            color=shade, alpha=alpha_val, lw=0)
            ax.plot(x, offset + y_norm, color="white", lw=0.7, alpha=0.9)

            # year label
            ax.text(-0.03, offset + 0.05, str(yr),
                    ha="right", va="bottom", fontsize=9,
                    color=POLICY_COLOR if yr == 2019 else
                          ("black" if yr >= 2019 else NEUTRAL_GRAY),
                    fontweight="bold" if yr == 2019 else "normal")

        # policy annotation arrow
        ax.annotate(
            "Operation\nGreens →",
            xy=(0.5, (n_years - years.index(2019) - 1) * (1 - overlap) + 0.3),
            xytext=(0.72, (n_years - years.index(2019) - 1) * (1 - overlap) + 0.7),
            arrowprops=dict(arrowstyle="->", color=POLICY_COLOR, lw=1.5),
            fontsize=8.5, color=POLICY_COLOR, fontweight="bold",
        )

        ax.set_xlabel("Monthly Price CV", fontsize=11)
        ax.set_title(crop, fontsize=13, fontweight="bold",
                     color=CROP_COLORS[crop], pad=10)
        ax.set_xlim(-0.05, 1.0)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.grid(False)

    # legend
    legend_elements = [
        mpatches.Patch(facecolor=NEUTRAL_GRAY, alpha=0.6, label="Pre-2018"),
        mpatches.Patch(facecolor="#D62728", alpha=0.7, label="Post-2018 (TOP crop colour)"),
        mpatches.Patch(facecolor=POLICY_COLOR, alpha=0.7, label="Operation Greens year (2019)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Evolution of Price Volatility Distributions (2013–2024)\n"
        "Kernel Density of Monthly CV — Each Ridge = One Year",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout(w_pad=3)
    save(fig, "fig3_ridge_plot.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Pre vs Post 2018 Split Violin Plot
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_violin():
    cv = pd.read_parquet(os.path.join(OUT, "cv_panel.parquet"))
    cv = cv[cv.cv.between(0.001, 1.2)]
    cv["Period"] = cv["post"].map({0: "Pre-2018", 1: "Post-2018"})
    crops = ["Tomato","Onion","Potato"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=False)

    for ax, crop in zip(axes, crops):
        d = cv[cv.crop == crop]
        col = CROP_COLORS[crop]

        # violin
        parts = ax.violinplot(
            [d[d.Period == "Pre-2018"]["cv"].values,
             d[d.Period == "Post-2018"]["cv"].values],
            positions=[1, 2], widths=0.6, showmedians=False,
        )
        for i, (pc, period) in enumerate(zip(parts["bodies"],
                                             ["Pre-2018","Post-2018"])):
            pc.set_facecolor(PERIOD_COLORS[period])
            pc.set_alpha(0.65)
            pc.set_edgecolor("white")
        for part in ["cmins","cmaxes","cbars"]:
            parts[part].set_color("black")
            parts[part].set_linewidth(1.2)

        # box stats overlay
        for pos, period in [(1, "Pre-2018"), (2, "Post-2018")]:
            vals  = d[d.Period == period]["cv"].values
            q25, med, q75 = np.percentile(vals, [25, 50, 75])
            mean  = vals.mean()
            ax.vlines(pos, q25, q75, lw=5, color="white", alpha=0.9, zorder=4)
            ax.scatter(pos, med, s=80, color="black", zorder=5)
            ax.scatter(pos, mean, s=60, marker="D", color=PERIOD_COLORS[period],
                       edgecolors="black", linewidths=0.8, zorder=6)
            ax.text(pos + 0.22, med + 0.005, f"med={med:.3f}",
                    fontsize=8.5, va="center", color="black")
            ax.text(pos + 0.22, mean - 0.018, f"μ={mean:.3f}",
                    fontsize=8.5, va="center", color=PERIOD_COLORS[period])

        # % change annotation
        pre_m  = d[d.Period == "Pre-2018"]["cv"].mean()
        post_m = d[d.Period == "Post-2018"]["cv"].mean()
        delta  = (post_m - pre_m) / pre_m * 100
        sign   = "↑" if delta > 0 else "↓"
        arrow_col = "#CB181D" if delta > 0 else "#2171B5"
        ax.text(1.5, ax.get_ylim()[1] * 0.93 if ax.get_ylim()[1] > 0 else 0.9,
                f"{sign} {abs(delta):.1f}%",
                ha="center", fontsize=14, fontweight="bold", color=arrow_col,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=arrow_col, alpha=0.9))

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Pre-2018\n(Jan 2013–\nOct 2018)",
                             "Post-2018\n(Nov 2018–\nDec 2024)"],
                           fontsize=10)
        ax.set_title(crop, fontsize=13, fontweight="bold", color=col, pad=10)
        ax.set_ylabel("Monthly Price CV" if crop == "Tomato" else "", fontsize=11)
        ax.set_ylim(bottom=0)

    # shared legend
    legend_elements = [
        mpatches.Patch(color=PERIOD_COLORS["Pre-2018"], alpha=0.7,
                       label="Pre-Policy (2013–Oct 2018)"),
        mpatches.Patch(color=PERIOD_COLORS["Post-2018"], alpha=0.7,
                       label="Post-Policy (Nov 2018–2024)"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="black",
                   markersize=7, label="Median"),
        plt.Line2D([0],[0], marker="D", color="w", markerfacecolor=NEUTRAL_GRAY,
                   markersize=7, label="Mean"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, -0.07), framealpha=0.95)

    fig.suptitle(
        "Price Volatility Before and After Operation Greens (2018)\n"
        "Distribution of Monthly CV Across All Mandis",
        fontsize=14, fontweight="bold", y=1.03,
    )
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig4_violin_prepost.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — India Choropleth: District-level Mean CV
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_choropleth():
    state_cv = pd.read_csv(os.path.join(OUT, "volatility_state.csv"))
    state_cv = state_cv[state_cv.period == "all"]

    india = gpd.read_file(GADM)
    india["state_norm"] = india["NAME_1"].str.strip().str.lower()
    state_cv["state_norm"] = state_cv["State"].str.strip().str.lower()

    # state-level dissolve
    states = india.dissolve(by="NAME_1").reset_index()
    states["state_norm"] = states["NAME_1"].str.strip().str.lower()

    crops = ["Tomato","Onion","Potato"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 9))

    for ax, crop in zip(axes, crops):
        d = state_cv[state_cv.crop == crop][["state_norm","mean_cv","n_mandis"]]
        geo = states.merge(d, on="state_norm", how="left")

        vmin, vmax = d["mean_cv"].quantile(0.05), d["mean_cv"].quantile(0.95)

        geo.plot(
            column="mean_cv", ax=ax, cmap=CV_CMAP,
            vmin=vmin, vmax=vmax,
            edgecolor="white", linewidth=0.4,
            missing_kwds={"color": "#EEEEEE", "edgecolor": "#CCCCCC",
                          "label": "No data"},
        )
        # state boundaries
        states.boundary.plot(ax=ax, color="#555555", linewidth=0.6, zorder=2)

        # top 3 hotspot labels
        top3 = d.nlargest(3, "mean_cv")
        for _, row in top3.iterrows():
            matched = states[states.state_norm == row.state_norm]
            if matched.empty:
                continue
            centroid = matched.geometry.centroid.iloc[0]
            ax.annotate(
                f"{row.state_norm.title()[:8]}\n{row.mean_cv:.3f}",
                xy=(centroid.x, centroid.y),
                fontsize=7.5, ha="center", va="center", fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="#333333", alpha=0.75),
                zorder=5,
            )

        # colorbar
        sm = plt.cm.ScalarMappable(cmap=CV_CMAP,
                                    norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02, aspect=20)
        cbar.set_label("Mean CV", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        ax.set_title(crop, fontsize=14, fontweight="bold",
                     color=CROP_COLORS[crop], pad=10)
        ax.set_axis_off()

    fig.suptitle(
        "Geographic Distribution of Price Volatility — India (2013–2024)\n"
        "State-Level Mean Monthly CV: Higher = More Volatile",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.text(0.5, -0.01,
             "Annotations show top-3 most volatile states. "
             "Grey = insufficient data. Source: Agmarknet 2013–2024.",
             ha="center", fontsize=9, color=NEUTRAL_GRAY, style="italic")
    fig.tight_layout(w_pad=1)
    save(fig, "fig5_choropleth.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Moran's I Trend: spatial clustering over time
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_morans():
    mi = pd.read_csv(os.path.join(OUT, "moran_global.csv"))
    crops = ["Tomato","Onion","Potato"]

    fig, (ax_main, ax_bar) = plt.subplots(
        1, 2, figsize=(15, 6),
        gridspec_kw={"width_ratios": [2, 1]}
    )

    # — Left: Moran's I trend lines
    for crop in crops:
        d = mi[mi.crop == crop].sort_values("year")
        col = CROP_COLORS[crop]
        ax_main.plot(d.year, d.morans_I, lw=2.5, color=col,
                     marker="o", ms=6, label=crop, zorder=3)
        ax_main.fill_between(d.year,
                              d.morans_I - 1.96 / np.sqrt(d.n_districts),
                              d.morans_I + 1.96 / np.sqrt(d.n_districts),
                              alpha=0.12, color=col)
        # significance stars
        for _, row in d.iterrows():
            if row.p_value < 0.01:
                ax_main.text(row.year, row.morans_I + 0.012, "***",
                             ha="center", fontsize=7.5, color=col)
            elif row.p_value < 0.05:
                ax_main.text(row.year, row.morans_I + 0.012, "**",
                             ha="center", fontsize=7.5, color=col)

    ax_main.axvline(2018.83, color=POLICY_COLOR, lw=2.0, ls="--", zorder=4,
                    label="Operation Greens (Nov 2018)")
    ax_main.axhline(0, color="black", lw=0.8, ls="-")
    ax_main.fill_betweenx([-0.1, 0.6], 2018.83, 2025,
                           alpha=0.05, color=POLICY_COLOR)
    ax_main.text(2019.1, 0.53, "Post-Policy\nPeriod",
                 color=POLICY_COLOR, fontsize=9.5, alpha=0.8)

    ax_main.set_xlabel("Year", fontsize=11)
    ax_main.set_ylabel("Global Moran's I", fontsize=11)
    ax_main.set_title(
        "Spatial Clustering of Price Volatility Over Time\n"
        "Moran's I > 0 → High-volatility mandis cluster geographically",
        fontsize=12, fontweight="bold",
    )
    ax_main.legend(fontsize=10, loc="upper right")
    ax_main.set_xlim(2012.5, 2025)
    ax_main.set_ylim(-0.05, 0.62)
    ax_main.xaxis.set_major_locator(mpl.ticker.MultipleLocator(2))

    # — Right: avg Moran's I comparison bar chart
    summary = mi.groupby("crop").agg(
        pre=("morans_I", lambda x: x[mi.loc[x.index, "year"] <= 2018].mean()),
        post=("morans_I", lambda x: x[mi.loc[x.index, "year"] > 2018].mean()),
    ).reset_index()

    x = np.arange(len(crops))
    w = 0.35
    ax_bar.bar(x - w/2, summary["pre"],  width=w, label="Pre-2018",
               color=[CROP_COLORS[c] for c in summary.crop], alpha=0.55, edgecolor="white")
    ax_bar.bar(x + w/2, summary["post"], width=w, label="Post-2018",
               color=[CROP_COLORS[c] for c in summary.crop], alpha=0.9, edgecolor="white")

    for i, row in summary.iterrows():
        delta = row.post - row.pre
        sign  = "↑" if delta > 0 else "↓"
        color = "#CB181D" if delta > 0 else "#2171B5"
        ax_bar.text(i, max(row.pre, row.post) + 0.012,
                    f"{sign}{abs(delta):.3f}",
                    ha="center", fontsize=9.5, fontweight="bold", color=color)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(crops, fontsize=11)
    ax_bar.set_ylabel("Average Moran's I", fontsize=11)
    ax_bar.set_title("Pre vs Post 2018\nSpatial Clustering", fontsize=12,
                     fontweight="bold")
    ax_bar.legend(fontsize=9)
    ax_bar.set_ylim(0, 0.55)
    ax_bar.axhline(0, color="black", lw=0.8)

    fig.tight_layout(w_pad=3)
    save(fig, "fig6_morans_trend.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Seasonal Heatmap: worst months
# ═══════════════════════════════════════════════════════════════════════════════
def fig7_seasonal():
    seasonal = pd.read_csv(os.path.join(OUT, "volatility_monthly.csv"))
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    pivot = seasonal.pivot_table(index="month", columns="crop", values="mean_cv")
    pivot = pivot[["Tomato","Onion","Potato"]]
    pivot.index = month_names

    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                              gridspec_kw={"width_ratios": [2, 1]})

    # — Left: heatmap
    ax = axes[0]
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt=".3f", cmap=CV_CMAP,
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 10},
        cbar_kws={"label": "Mean Monthly CV", "shrink": 0.75},
    )
    # circle worst month per crop
    for j, crop in enumerate(pivot.columns):
        worst = pivot[crop].idxmax()
        ri = list(pivot.index).index(worst)
        ax.add_patch(mpatches.FancyBboxPatch(
            (j + 0.05, ri + 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.03", lw=2.5,
            edgecolor="black", facecolor="none", zorder=5,
        ))
    ax.set_title("Seasonal Volatility Pattern by Crop\n"
                 "(Black box = worst month per crop)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Crop", fontsize=11)
    ax.set_ylabel("Month", fontsize=11)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    # — Right: polar chart of seasonal pattern
    ax2 = fig.add_subplot(1, 2, 2, polar=True)
    theta = np.linspace(0, 2*np.pi, 12, endpoint=False)
    width = 2 * np.pi / 12

    for crop in ["Tomato","Onion","Potato"]:
        vals = seasonal[seasonal.crop == crop].sort_values("month")["mean_cv"].values
        ax2.bar(theta, vals, width=width * 0.8, alpha=0.55,
                color=CROP_COLORS[crop], label=crop, bottom=0.05)
        ax2.plot(np.append(theta, theta[0]),
                 np.append(vals, vals[0]),
                 lw=2, color=CROP_COLORS[crop])

    ax2.set_xticks(theta)
    ax2.set_xticklabels(month_names, fontsize=9.5)
    ax2.set_yticklabels([])
    ax2.set_title("Seasonal Polar Chart\n(radius = volatility)",
                  fontsize=11, fontweight="bold", pad=20)
    ax2.legend(loc="lower right", bbox_to_anchor=(1.3, -0.05), fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Within-Year Seasonality of Price Volatility\n"
        "Which Months are Worst for Each TOP Crop?",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout(w_pad=3)
    save(fig, "fig7_seasonal_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — DML Forest Plot: policy impact estimates
# ═══════════════════════════════════════════════════════════════════════════════
def fig8_forest():
    with open(os.path.join(OUT, "dml_results_og.json")) as f:
        res = json.load(f)

    specs = [
        ("Main — LightGBM",    res["main"]["LightGBM"],         "#D62728", "Main spec"),
        ("Main — Random Forest", res["main"]["RandomForest"],   "#7B2D8B", "Main spec"),
        ("Main — Ridge",       res["main"]["Ridge"],             "#FF7F0E", "Main spec"),
        ("COVID-excl — LightGBM", res["covid_excluded"]["LightGBM"], "#D62728", "Robustness"),
        ("COVID-excl — RF",    res["covid_excluded"]["RandomForest"], "#7B2D8B","Robustness"),
        ("COVID-excl — Ridge", res["covid_excluded"]["Ridge"],   "#FF7F0E", "Robustness"),
        ("Placebo 2015 — LightGBM", res["placebo_2015"]["LightGBM"], "#AAAAAA","Placebo"),
        ("Placebo 2015 — RF",  res["placebo_2015"]["RandomForest"], "#AAAAAA","Placebo"),
        ("Placebo 2015 — Ridge", res["placebo_2015"]["Ridge"],   "#AAAAAA", "Placebo"),
    ]

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = list(range(len(specs)))

    section_colors = {"Main spec": "#FFF3CD", "Robustness": "#D1ECF1", "Placebo": "#F8F9FA"}
    prev_section = None
    for i, (label, r, col, section) in enumerate(specs):
        if section != prev_section:
            ystart = i - 0.5
            yend   = i + (sum(1 for *_, s in specs[i:] if s == section)) - 0.5
            ax.axhspan(ystart, yend, color=section_colors[section],
                       alpha=0.4, zorder=0)
            ax.text(-0.28, (ystart + yend) / 2, section,
                    va="center", ha="right", fontsize=9.5,
                    fontweight="bold", color=NEUTRAL_GRAY,
                    rotation=90)
            prev_section = section

        if "error" in r:
            continue
        ate = r["ate_log_cv"]
        lo  = r["ci90_low"]
        hi  = r["ci90_high"]
        sig = r["significant"]

        marker_col = col if sig else "#CCCCCC"
        ax.plot([lo, hi], [i, i], lw=2.0, color=marker_col, alpha=0.85, zorder=2)
        ax.scatter(ate, i, s=90, color=marker_col, zorder=3,
                   marker="D" if sig else "o",
                   edgecolors="black" if sig else "#AAAAAA", linewidths=0.8)

        # pct change label
        pct = r["pct_change_cv"]
        label_col = "#CB181D" if pct > 0 else "#2171B5"
        ax.text(hi + 0.005, i, f"{pct:+.1f}%",
                va="center", ha="left", fontsize=9,
                color=label_col if sig else "#AAAAAA",
                fontweight="bold" if sig else "normal")

    ax.axvline(0, color="black", lw=1.5, ls="-", zorder=4)
    ax.axvline(0, color="#CB181D", lw=0.5, ls="--", alpha=0.4, zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([s[0] for s in specs], fontsize=10)
    ax.set_xlabel("ATE on log(Monthly CV)  [Negative = Volatility Reduced]",
                  fontsize=11, labelpad=8)
    ax.set_title(
        "DML Policy Impact Estimates — Effect of Operation Greens\n"
        "on Agricultural Price Volatility (All Specifications)",
        fontsize=13, fontweight="bold", pad=12,
    )

    # reference regions
    ax.fill_betweenx([-0.5, len(specs)-0.5], -0.3, 0,
                     alpha=0.05, color="#2171B5", zorder=0)
    ax.fill_betweenx([-0.5, len(specs)-0.5], 0, 0.25,
                     alpha=0.05, color="#CB181D", zorder=0)
    ax.text(-0.15, -0.4, "← Volatility\n   reduced",
            fontsize=8.5, color="#2171B5", alpha=0.8)
    ax.text(0.02, -0.4, "Volatility →\n increased",
            fontsize=8.5, color="#CB181D", alpha=0.8)

    # legend
    legend_elements = [
        plt.Line2D([0],[0], marker="D", color="w", markerfacecolor=NEUTRAL_GRAY,
                   markersize=8, markeredgecolor="black", label="Significant (90% CI)"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#CCCCCC",
                   markersize=8, markeredgecolor="#AAAAAA", label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10,
              framealpha=0.95)
    ax.invert_yaxis()
    ax.set_xlim(-0.3, 0.25)
    fig.tight_layout()
    save(fig, "fig8_dml_forest.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 9 — LISA Hotspot Cluster Map
# ═══════════════════════════════════════════════════════════════════════════════
def fig9_lisa():
    lisa = pd.read_csv(os.path.join(OUT, "moran_local.csv"))
    india = gpd.read_file(GADM)
    states = india.dissolve(by="NAME_1").reset_index()

    crops = ["Tomato","Onion","Potato"]
    cluster_colors = {
        "HH": "#CB181D",   # high-high = hotspot (red)
        "LL": "#2171B5",   # low-low = coldspot (blue)
        "HL": "#FD8D3C",   # high outlier (orange)
        "LH": "#9ECAE1",   # low outlier (light blue)
    }

    fig, axes = plt.subplots(1, 3, figsize=(21, 9))

    for ax, crop in zip(axes, crops):
        d = lisa[lisa.crop == crop].copy()
        # create geodataframe from lat/lon points
        gdf = gpd.GeoDataFrame(
            d, geometry=gpd.points_from_xy(d.lon, d.lat), crs="EPSG:4326"
        )

        # base map (all districts, light gray)
        india.plot(ax=ax, color="#F5F5F5", edgecolor="#DDDDDD",
                   linewidth=0.3, zorder=1)
        states.boundary.plot(ax=ax, color="#666666", linewidth=0.8, zorder=3)

        # plot each cluster type
        for ctype, ccolor in cluster_colors.items():
            sub = gdf[gdf.cluster_type == ctype]
            if sub.empty:
                continue
            sub.plot(ax=ax, color=ccolor, markersize=sub["mean_cv"] * 120 + 8,
                     alpha=0.8, zorder=4, label=ctype)

        ax.set_title(crop, fontsize=14, fontweight="bold",
                     color=CROP_COLORS[crop], pad=10)
        ax.set_axis_off()

        # inset: count per cluster type
        counts = d.cluster_type.value_counts()
        inset_ax = ax.inset_axes([0.65, 0.0, 0.35, 0.22])
        keys = list(cluster_colors.keys())
        vals = [counts.get(k, 0) for k in keys]
        bars = inset_ax.bar(range(4), vals,
                             color=list(cluster_colors.values()),
                             edgecolor="white", width=0.7)
        inset_ax.set_xticks(range(4))
        inset_ax.set_xticklabels(keys, fontsize=8)
        inset_ax.set_yticks([])
        inset_ax.set_facecolor("#F9F9F9")
        inset_ax.spines["top"].set_visible(False)
        inset_ax.spines["right"].set_visible(False)
        inset_ax.spines["left"].set_visible(False)
        for bar, key in zip(bars, cluster_colors):
            h = bar.get_height()
            if h > 0:
                inset_ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                              str(int(h)), ha="center", va="bottom", fontsize=7.5)

    # shared legend
    legend_elements = [
        mpatches.Patch(facecolor=cluster_colors["HH"], label="HH — High Volatility Cluster (Hotspot)"),
        mpatches.Patch(facecolor=cluster_colors["LL"], label="LL — Low Volatility Cluster (Coldspot)"),
        mpatches.Patch(facecolor=cluster_colors["HL"], label="HL — High Outlier (isolated spike)"),
        mpatches.Patch(facecolor=cluster_colors["LH"], label="LH — Low Outlier (isolated calm)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=10.5, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "LISA Spatial Cluster Analysis — Price Volatility Hotspots & Coldspots\n"
        "Local Moran's I Cluster Types | Point size ∝ Mean CV Intensity",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig9_lisa_map.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import time
    t0 = time.time()
    print(f"Generating publication-ready figures → {FIGS}\n")

    steps = [
        ("Fig 1 — Volatility Heatmap",      fig1_heatmap),
        ("Fig 2 — STL Decomposition",        fig2_stl),
        ("Fig 3 — Ridge/Joy Plot",           fig3_ridge),
        ("Fig 4 — Pre/Post Violin",          fig4_violin),
        ("Fig 5 — Choropleth Maps",          fig5_choropleth),
        ("Fig 6 — Moran's I Trend",          fig6_morans),
        ("Fig 7 — Seasonal Heatmap",         fig7_seasonal),
        ("Fig 8 — DML Forest Plot",          fig8_forest),
        ("Fig 9 — LISA Cluster Map",         fig9_lisa),
    ]

    for name, fn in steps:
        print(f"{name} ...")
        try:
            fn()
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nAll figures done in {time.time()-t0:.0f}s")
    print(f"Location: {FIGS}")
