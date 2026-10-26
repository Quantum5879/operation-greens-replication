#!/usr/bin/env python
"""
Phase 1b — Temporal Decomposition + Spatial Econometrics
=========================================================

PART A — Temporal (per crop, national monthly CV series 2013-2024):
  A1. STL decomposition  → trend / seasonal / residual components
  A2. Structural break detection (ruptures, Pelt algo) → exact break years
  A3. Year-of-peak-volatility identification
  A4. Pre/post-break trend comparison

PART B — Spatial econometrics (district-level CV, per crop):
  B1. Global Moran's I  → is volatility spatially autocorrelated?
  B2. Local Moran's I (LISA) → which district clusters are hotspots/coldspots?
  B3. Spatial lag correlation → does high volatility in one district predict neighbors?
  B4. Moran's I trend over years → did spatial clustering increase or decrease post-2018?

Outputs (02-analysis/):
  stl_components.csv          — trend/seasonal/residual for each crop-month
  structural_breaks.csv       — detected break dates + directions per crop
  moran_global.csv            — Global Moran's I per crop per year
  moran_local.csv             — LISA cluster type per district-crop
  spatial_lag_correlation.csv — spatial lag regression coefficients per crop-year

Run: python temporal_spatial_analysis.py
     (requires cv_panel.parquet from volatility_analysis.py)
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy import stats
from statsmodels.tsa.seasonal import STL
import ruptures as rpt

warnings.filterwarnings("ignore")

ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GADM  = os.path.join(ROOT, "01-data", "geo", "gadm41_IND_2.json")
OUT   = os.path.join(ROOT, "02-analysis")
CV_PQ = os.path.join(OUT, "cv_panel.parquet")

CROPS = ["Tomato", "Onion", "Potato"]
KNN   = 5          # neighbours for spatial weights
MIN_DIST_OBS = 12  # min months for a district to enter spatial analysis


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ── Load pre-computed CV panel ────────────────────────────────────────────────
def load_cv():
    cv = pd.read_parquet(CV_PQ)
    log(f"CV panel loaded: {len(cv):,} rows, {cv.mkey.nunique():,} mandis")
    return cv


# ── District centroids from GADM ──────────────────────────────────────────────
def load_district_centroids():
    import geopandas as gpd
    g = gpd.read_file(GADM)
    cent = g.geometry.representative_point()
    g = g.assign(lon=cent.x, lat=cent.y)
    g["district_norm"] = g["NAME_2"].str.strip().str.lower()
    g["state_norm"]    = g["NAME_1"].str.strip().str.lower()
    return g[["NAME_1", "NAME_2", "district_norm", "state_norm", "lat", "lon"]]


# ═══════════════════════════════════════════════════════════════════════════════
# PART A — TEMPORAL DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════

def make_monthly_series(cv):
    """Aggregate CV panel to national monthly mean CV per crop."""
    ms = (
        cv.groupby(["crop", "year", "month"])["cv"]
        .agg(mean_cv="mean", median_cv="median", n_mandis="count")
        .reset_index()
    )
    ms["date"] = pd.to_datetime(
        dict(year=ms.year, month=ms.month, day=1)
    )
    return ms.sort_values(["crop", "date"])


def stl_decompose(monthly):
    """
    STL decomposition on national monthly CV series per crop.
    Returns DataFrame with trend/seasonal/residual columns.
    """
    records = []
    for crop in CROPS:
        s = (
            monthly[monthly.crop == crop]
            .set_index("date")["mean_cv"]
            .asfreq("MS")           # monthly start frequency
            .interpolate("linear")  # fill any gaps
        )
        if len(s) < 24:
            log(f"  {crop}: series too short ({len(s)} months), skipping STL")
            continue

        stl = STL(s, period=12, robust=True, seasonal=7)
        res = stl.fit()

        for date, trend, seasonal, resid, original in zip(
            s.index, res.trend, res.seasonal, res.resid, s.values
        ):
            records.append({
                "crop": crop, "date": date,
                "original": original,
                "trend": trend,
                "seasonal": seasonal,
                "residual": resid,
                "deseasonalised": trend + resid,
            })

    df = pd.DataFrame(records)
    log(f"STL decomposition done: {len(df)} crop-month records")
    return df


def detect_structural_breaks(stl_df):
    """
    Use ruptures (Pelt, RBF cost) to detect structural breaks in the
    trend component of each crop's CV series.
    Report break dates and whether volatility went up or down after the break.
    """
    results = []
    for crop in CROPS:
        d = stl_df[stl_df.crop == crop].sort_values("date")
        trend = d["trend"].values
        dates = d["date"].values

        # Pelt with RBF cost — penalise by BIC-style penalty
        algo = rpt.Pelt(model="rbf", min_size=6).fit(trend)
        try:
            bkps = algo.predict(pen=np.log(len(trend)) * trend.var())
        except Exception:
            bkps = []

        for bkp in bkps:
            if bkp >= len(dates):
                continue
            bkp_date = pd.Timestamp(dates[min(bkp, len(dates) - 1)])
            before = trend[max(0, bkp - 6): bkp].mean()
            after  = trend[bkp: min(len(trend), bkp + 6)].mean()
            direction = "INCREASE" if after > before else "DECREASE"
            results.append({
                "crop": crop,
                "break_date": bkp_date,
                "break_year": bkp_date.year,
                "cv_before_break": round(before, 4),
                "cv_after_break":  round(after, 4),
                "direction": direction,
                "magnitude": round(abs(after - before), 4),
            })
            log(f"  {crop} break @ {bkp_date.strftime('%b-%Y')}: "
                f"{before:.3f} → {after:.3f} ({direction}, Δ={abs(after-before):.3f})")

    df = pd.DataFrame(results)
    return df


def peak_year_analysis(monthly):
    """Identify the single worst year per crop by mean CV."""
    yr = monthly.groupby(["crop", "year"])["mean_cv"].mean().reset_index()
    results = []
    log("Peak volatility year per crop:")
    for crop in CROPS:
        d = yr[yr.crop == crop].sort_values("mean_cv", ascending=False)
        log(f"  {crop}:")
        log(d.head(5).to_string(index=False))
        results.append(d.head(5).assign(crop=crop))
    return pd.concat(results, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PART B — SPATIAL ECONOMETRICS
# ═══════════════════════════════════════════════════════════════════════════════

def build_district_cv(cv, centroids):
    """
    Aggregate mandi-level CV to district level.
    Merge with GADM centroids to get lat/lon.
    """
    dist_cv = (
        cv.groupby(["crop", "State", "District", "year"])["cv"]
        .agg(mean_cv="mean", n_obs="count")
        .reset_index()
    )
    # normalise names for join
    dist_cv["district_norm"] = dist_cv.District.str.strip().str.lower()
    dist_cv["state_norm"]    = dist_cv.State.str.strip().str.lower()

    merged = dist_cv.merge(
        centroids[["district_norm", "state_norm", "lat", "lon"]],
        on=["district_norm", "state_norm"],
        how="left",
    )
    matched = merged.lat.notna().mean()
    log(f"District-centroid match rate: {matched:.1%}")
    merged = merged[merged.lat.notna()]
    return merged


def knn_weights(coords, k=KNN):
    """
    Build row-standardised k-nearest-neighbour spatial weights matrix.
    coords : (n, 2) array of [lat, lon]
    Returns W (n×n), row-standardised.
    """
    D = cdist(coords, coords)
    np.fill_diagonal(D, np.inf)
    W = np.zeros_like(D)
    for i in range(len(coords)):
        nn = np.argsort(D[i])[:k]
        W[i, nn] = 1.0
    # row-standardise
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return W / row_sums


def morans_i_global(z, W):
    """
    Global Moran's I statistic.
    z : (n,) array of values
    W : (n×n) row-standardised weights
    Returns I, E[I], Var[I], z-score, p-value (two-tailed, normal approx).
    """
    n = len(z)
    z_dev = z - z.mean()
    S0 = W.sum()

    I = (n / S0) * (z_dev @ W @ z_dev) / (z_dev @ z_dev)

    # analytical expected value and variance (randomisation assumption)
    EI = -1.0 / (n - 1)
    S1 = 0.5 * ((W + W.T) ** 2).sum()
    S2 = ((W.sum(axis=1) + W.sum(axis=0)) ** 2).sum()
    k4 = ((z_dev ** 4).sum() / n) / ((z_dev ** 2).sum() / n) ** 2
    VI = (
        n * ((n**2 - 3*n + 3)*S1 - n*S2 + 3*S0**2)
        - k4 * ((n**2 - n)*S1 - 2*n*S2 + 6*S0**2)
    ) / ((n-1)*(n-2)*(n-3)*S0**2) - EI**2

    zscore = (I - EI) / np.sqrt(max(VI, 1e-12))
    pval   = 2 * (1 - stats.norm.cdf(abs(zscore)))
    return I, EI, zscore, pval


def morans_i_local(z, W):
    """
    Local Moran's I (LISA) per observation.
    Returns array of local I values and cluster labels:
      HH = high surrounded by high (hotspot)
      LL = low surrounded by low  (coldspot)
      HL = high surrounded by low (outlier)
      LH = low surrounded by high (outlier)
      NS = not significant
    """
    n   = len(z)
    z_s = (z - z.mean()) / z.std()   # standardised
    Wz  = W @ z_s
    li  = z_s * Wz                   # local I

    # cluster labels (no significance filtering here — caller can add p-val)
    mean_z = z_s.mean()
    labels = []
    for zi, wzi in zip(z_s, Wz):
        if zi >= 0 and wzi >= 0:
            labels.append("HH")
        elif zi < 0 and wzi < 0:
            labels.append("LL")
        elif zi >= 0 and wzi < 0:
            labels.append("HL")
        else:
            labels.append("LH")
    return li, np.array(labels)


def run_spatial_analysis(dist_cv):
    global_rows = []
    local_rows  = []

    for crop in CROPS:
        log(f"\n  Spatial analysis: {crop}")
        dc = dist_cv[dist_cv.crop == crop]

        # ── Global Moran's I per year ───────────────────────────────────────
        for year in sorted(dc.year.unique()):
            dy = dc[dc.year == year].dropna(subset=["lat","lon","mean_cv"])
            if len(dy) < 20:
                continue
            coords = dy[["lat","lon"]].values
            z      = dy["mean_cv"].values
            W      = knn_weights(coords, k=KNN)
            I, EI, zscore, pval = morans_i_global(z, W)
            sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
            global_rows.append({
                "crop": crop, "year": year,
                "morans_I": round(I, 4),
                "expected_I": round(EI, 4),
                "z_score": round(zscore, 3),
                "p_value": round(pval, 4),
                "significant": sig,
                "n_districts": len(dy),
            })

        # ── Local Moran's I (LISA) — full period ────────────────────────────
        dc_full = dc.groupby(["District","State","lat","lon"])["mean_cv"].mean().reset_index()
        dc_full = dc_full.dropna()
        if len(dc_full) >= 20:
            coords = dc_full[["lat","lon"]].values
            z      = dc_full["mean_cv"].values
            W      = knn_weights(coords, k=KNN)
            li, labels = morans_i_local(z, W)
            for i, row in dc_full.iterrows():
                local_rows.append({
                    "crop": crop,
                    "District": row.District,
                    "State": row.State,
                    "lat": row.lat, "lon": row.lon,
                    "mean_cv": round(row.mean_cv, 4),
                    "local_morans_I": round(li[dc_full.index.get_loc(i)], 4),
                    "cluster_type": labels[dc_full.index.get_loc(i)],
                })

        # ── Print summary ───────────────────────────────────────────────────
        yr_df = pd.DataFrame([r for r in global_rows if r["crop"] == crop])
        if not yr_df.empty:
            log(f"    Global Moran's I (selected years):")
            log(yr_df[["year","morans_I","z_score","p_value","significant"]]
                .iloc[::3].to_string(index=False))
            sig_count = (yr_df.p_value < 0.05).sum()
            avg_I = yr_df.morans_I.mean()
            log(f"    Avg Moran's I={avg_I:.3f}, significant in {sig_count}/{len(yr_df)} years")

    global_df = pd.DataFrame(global_rows)
    local_df  = pd.DataFrame(local_rows)

    # cluster summary
    if not local_df.empty:
        log("\nLISA cluster summary (full period, HH = persistent hotspots):")
        for crop in CROPS:
            top_hh = (
                local_df[(local_df.crop == crop) & (local_df.cluster_type == "HH")]
                .nlargest(5, "mean_cv")[["District","State","mean_cv","cluster_type"]]
            )
            log(f"  {crop} HH hotspots:\n{top_hh.to_string(index=False)}")

    return global_df, local_df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    log("=== Temporal Decomposition + Spatial Econometrics ===")

    cv = load_cv()
    centroids = load_district_centroids()
    monthly = make_monthly_series(cv)

    # ── PART A ────────────────────────────────────────────────────────────────
    log("\n--- A1-A2: STL decomposition + structural breaks ---")
    stl_df  = stl_decompose(monthly)
    breaks  = detect_structural_breaks(stl_df)

    log("\n--- A3: Peak volatility years ---")
    peaks   = peak_year_analysis(monthly)

    # ── PART B ────────────────────────────────────────────────────────────────
    log("\n--- B: Spatial econometrics ---")
    dist_cv = build_district_cv(cv, centroids)
    global_moran, local_moran = run_spatial_analysis(dist_cv)

    # ── Save ──────────────────────────────────────────────────────────────────
    stl_df.to_csv(os.path.join(OUT, "stl_components.csv"), index=False)
    breaks.to_csv(os.path.join(OUT, "structural_breaks.csv"), index=False)
    peaks.to_csv(os.path.join(OUT, "peak_volatility_years.csv"), index=False)
    global_moran.to_csv(os.path.join(OUT, "moran_global.csv"), index=False)
    local_moran.to_csv(os.path.join(OUT, "moran_local.csv"), index=False)
    log("\nAll outputs saved.")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("STRUCTURAL BREAKS IN VOLATILITY TREND")
    print("="*65)
    if not breaks.empty:
        print(breaks[["crop","break_date","cv_before_break","cv_after_break","direction","magnitude"]]
              .to_string(index=False))
    else:
        print("No significant breaks detected.")

    print("\n" + "="*65)
    print("GLOBAL MORAN'S I — SPATIAL CLUSTERING OF VOLATILITY")
    print("="*65)
    if not global_moran.empty:
        summary = global_moran.groupby("crop").agg(
            avg_I=("morans_I","mean"),
            max_I=("morans_I","max"),
            sig_years=("p_value", lambda x: (x<0.05).sum()),
            total_years=("year","count")
        ).reset_index()
        print(summary.round(3).to_string(index=False))
        print("\nInterpretation: Moran's I > 0 and significant → high-volatility mandis")
        print("cluster geographically (spatial contagion). I near 0 → random.")

    print("\n" + "="*65)
    print("TOP HH HOTSPOT DISTRICTS (high CV surrounded by high CV)")
    print("="*65)
    if not local_moran.empty:
        hh = local_moran[local_moran.cluster_type == "HH"].nlargest(15, "mean_cv")
        print(hh[["crop","District","State","mean_cv"]].to_string(index=False))

    log(f"\nDONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
