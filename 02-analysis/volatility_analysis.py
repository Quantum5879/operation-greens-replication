#!/usr/bin/env python
"""
Phase 1 Descriptive Analysis — Operation Greens study.

Answers three questions:
  1. TEMPORAL : Which years had highest price volatility for TOP crops (2013-2024)?
               Is volatility increasing over time?
  2. SPATIAL  : Which states are worst-hit? Which specific mandis are hotspots?
  3. SEASONAL : Which months are worst within a year (for policy targeting)?

Outputs (all in 02-analysis/):
  volatility_yearly.csv        — mean annual CV per crop × year
  volatility_state.csv         — mean CV per crop × state (full period + pre/post 2018)
  volatility_mandi_top50.csv   — 50 worst mandis per crop by mean annual CV
  volatility_monthly.csv       — mean CV per crop × calendar month (seasonality)
  summary_stats.csv            — quick pre/post 2018 comparison table

Volatility measure: CV = std(daily modal price) / mean(daily modal price)
computed within each mandi × crop × year-month cell,
then averaged up to year / state / mandi as needed.

Run: python volatility_analysis.py
"""

import os
import re
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQ  = os.path.join(ROOT, "01-data", "archive (4)", "parquet")
OUT   = os.path.join(ROOT, "02-analysis")

STUDY_YEARS = list(range(2013, 2025))   # 2013-2024
POLICY_DATE = pd.Timestamp("2018-11-01")

CROPS = {
    "Tomato": re.compile(r"^tomato", re.I),
    "Onion":  re.compile(r"^onion$", re.I),
    "Potato": re.compile(r"^potato", re.I),
}

norm = lambda s: re.sub(r"\s+", " ", str(s).strip())


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ── Load price panel ──────────────────────────────────────────────────────────
def load_panel(years):
    frames = []
    for y in years:
        p = os.path.join(PARQ, f"{y}.parquet")
        if not os.path.exists(p):
            log(f"  MISSING: {y}.parquet — skip")
            continue
        df = pd.read_parquet(
            p,
            columns=["State", "District", "Market", "Commodity",
                     "Arrival_Date", "Min_Price", "Max_Price", "Modal_Price"],
        )
        # tag crop
        df["crop"] = None
        com = df["Commodity"].astype(str).str.strip()
        for c, rx in CROPS.items():
            df.loc[com.str.match(rx), "crop"] = c
        df = df[df.crop.notna()].copy()

        # basic QC
        df = df[(df.Modal_Price > 0) & (df.Min_Price <= df.Max_Price)]

        df["date"]    = pd.to_datetime(df["Arrival_Date"], errors="coerce")
        df["year"]    = df.date.dt.year
        df["month"]   = df.date.dt.month
        df["ym"]      = df.date.dt.to_period("M")
        df["State"]   = df.State.map(norm)
        df["District"]= df.District.map(norm)
        df["Market"]  = df.Market.map(norm)
        df["mkey"]    = df.State + " | " + df.District + " | " + df.Market

        frames.append(df)
        log(f"  {y}: {len(df):,} rows ({', '.join(df.crop.value_counts().to_dict().__str__()[1:-1].split(',')[:3])}...)")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel[panel.date.notna() & panel.year.between(2013, 2024)]
    log(f"Panel loaded: {len(panel):,} rows | {panel.mkey.nunique():,} mandis | {panel.State.nunique()} states")
    return panel


# ── Compute monthly CV per mandi-crop ────────────────────────────────────────
def compute_cv(panel):
    """
    CV = std / mean of daily Modal_Price within each (mkey, crop, year-month).
    Require ≥ 5 trading days in the month for a valid CV.
    """
    grp = panel.groupby(["mkey", "State", "District", "Market", "crop", "year", "month"])
    agg = grp["Modal_Price"].agg(
        price_mean="mean",
        price_std="std",
        n_days="count",
    ).reset_index()
    agg = agg[agg.n_days >= 5].copy()
    agg["cv"] = agg.price_std / agg.price_mean          # monthly CV
    agg["log_cv"] = np.log(agg.cv.clip(lower=1e-6))     # log CV for regressions
    agg["post"] = (
        pd.to_datetime(dict(year=agg.year, month=agg.month, day=1)) >= POLICY_DATE
    ).astype(int)
    log(f"Monthly CV computed: {len(agg):,} mandi-crop-month cells  |  median CV={agg.cv.median():.3f}")
    return agg


# ── Q1: Temporal — year-by-year volatility ────────────────────────────────────
def temporal_analysis(cv):
    yearly = (
        cv.groupby(["crop", "year"])
        .agg(
            mean_cv=("cv", "mean"),
            median_cv=("cv", "median"),
            p90_cv=("cv", lambda x: x.quantile(0.90)),
            n_mandis=("mkey", "nunique"),
        )
        .reset_index()
    )
    yearly["pct_mean_cv"] = yearly["mean_cv"] * 100

    # simple trend: slope of mean_cv on year, per crop
    for crop in CROPS:
        d = yearly[yearly.crop == crop].sort_values("year")
        if len(d) >= 3:
            slope = np.polyfit(d.year, d.mean_cv, 1)[0]
            direction = "INCREASING" if slope > 0 else "DECREASING"
            log(f"  {crop}: trend {direction}  (slope={slope:+.4f} CV/yr)")

    return yearly


# ── Q2: Spatial — state and mandi hotspots ───────────────────────────────────
def spatial_analysis(cv):
    # state-level: mean CV across all years + split pre/post
    state_all = (
        cv.groupby(["crop", "State"])
        .agg(mean_cv=("cv", "mean"), median_cv=("cv", "median"), n_mandis=("mkey", "nunique"))
        .reset_index()
        .assign(period="all")
    )
    state_pre = (
        cv[cv.post == 0].groupby(["crop", "State"])
        .agg(mean_cv=("cv", "mean"), median_cv=("cv", "median"), n_mandis=("mkey", "nunique"))
        .reset_index()
        .assign(period="pre_2018")
    )
    state_post = (
        cv[cv.post == 1].groupby(["crop", "State"])
        .agg(mean_cv=("cv", "mean"), median_cv=("cv", "median"), n_mandis=("mkey", "nunique"))
        .reset_index()
        .assign(period="post_2018")
    )
    state_cv = pd.concat([state_all, state_pre, state_post], ignore_index=True)

    # change in volatility post-policy (simple before/after delta)
    pivot = state_cv[state_cv.period.isin(["pre_2018", "post_2018"])].pivot_table(
        index=["crop", "State"], columns="period", values="mean_cv"
    ).reset_index()
    pivot.columns.name = None
    pivot["delta_cv"] = pivot["post_2018"] - pivot["pre_2018"]
    pivot["pct_change"] = pivot["delta_cv"] / pivot["pre_2018"] * 100

    log("Top 5 worst states (highest mean CV, full period):")
    for crop in CROPS:
        top = state_all[state_all.crop == crop].nlargest(5, "mean_cv")[["State","mean_cv","n_mandis"]]
        log(f"  {crop}: {top.to_string(index=False)}")

    return state_cv, pivot


# ── Q3: Mandi hotspots ────────────────────────────────────────────────────────
def mandi_hotspots(cv):
    # compute pre/post CV separately then merge
    base = (
        cv.groupby(["crop", "mkey", "State", "District", "Market"])
        .agg(
            mean_cv=("cv", "mean"),
            median_cv=("cv", "median"),
            p90_cv=("cv", lambda x: x.quantile(0.90)),
            n_months=("cv", "count"),
        )
        .reset_index()
    )
    pre = (
        cv[cv.post == 0]
        .groupby(["crop", "mkey"])["cv"].mean()
        .rename("pre_cv").reset_index()
    )
    post = (
        cv[cv.post == 1]
        .groupby(["crop", "mkey"])["cv"].mean()
        .rename("post_cv").reset_index()
    )
    mandi_cv = base.merge(pre, on=["crop", "mkey"], how="left")
    mandi_cv = mandi_cv.merge(post, on=["crop", "mkey"], how="left")
    mandi_cv["delta_cv"] = mandi_cv["post_cv"] - mandi_cv["pre_cv"]

    # require ≥ 24 months of data (2 years) to be in the ranking
    mandi_cv = mandi_cv[mandi_cv.n_months >= 24]

    top50 = pd.concat(
        [mandi_cv[mandi_cv["crop"] == crop].nlargest(50, "mean_cv") for crop in CROPS],
        ignore_index=True,
    )

    log("Top 5 worst mandis (highest mean CV):")
    for crop in CROPS:
        t = top50[top50["crop"] == crop].head(5)[["Market", "District", "State", "mean_cv"]]
        log(f"  {crop}:\n{t.to_string(index=False)}")

    return top50, mandi_cv


# ── Q4: Seasonal — worst months ──────────────────────────────────────────────
def seasonal_analysis(cv):
    seasonal = (
        cv.groupby(["crop", "month"])
        .agg(mean_cv=("cv", "mean"), median_cv=("cv", "median"))
        .reset_index()
    )
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    seasonal["month_name"] = seasonal.month.map(month_names)

    log("Worst 3 months per crop:")
    for crop in CROPS:
        top = seasonal[seasonal.crop == crop].nlargest(3, "mean_cv")[["month_name","mean_cv"]]
        log(f"  {crop}: {list(top.itertuples(index=False, name=None))}")

    return seasonal


# ── Summary: pre vs post 2018 ─────────────────────────────────────────────────
def summary_stats(cv):
    s = (
        cv.groupby(["crop", "post"])
        .agg(
            mean_cv=("cv", "mean"),
            median_cv=("cv", "median"),
            p90_cv=("cv", lambda x: x.quantile(0.90)),
            n_obs=("cv", "count"),
            n_mandis=("mkey", "nunique"),
        )
        .reset_index()
    )
    s["period"] = s.post.map({0: "Pre-2018 (2013-Oct 2018)", 1: "Post-2018 (Nov 2018-2024)"})

    log("\n=== Pre vs Post 2018 Summary ===")
    for crop in CROPS:
        d = s[s.crop == crop][["period","mean_cv","median_cv","p90_cv","n_mandis"]]
        log(f"\n  {crop}:\n{d.to_string(index=False)}")

    return s


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    log("=== Phase 1: Volatility Analysis — TOP Crops 2013-2024 ===")

    log("Loading price panel...")
    panel = load_panel(STUDY_YEARS)

    log("Computing monthly CV...")
    cv = compute_cv(panel)

    log("\n--- Q1: Temporal trend ---")
    yearly = temporal_analysis(cv)

    log("\n--- Q2: Spatial hotspots ---")
    state_cv, state_delta = spatial_analysis(cv)

    log("\n--- Q3: Mandi hotspots ---")
    top50, mandi_cv = mandi_hotspots(cv)

    log("\n--- Q4: Seasonal patterns ---")
    seasonal = seasonal_analysis(cv)

    log("\n--- Summary: Pre vs Post 2018 ---")
    summary = summary_stats(cv)

    # ── Save outputs ──────────────────────────────────────────────────────────
    yearly.to_csv(os.path.join(OUT, "volatility_yearly.csv"), index=False)
    state_cv.to_csv(os.path.join(OUT, "volatility_state.csv"), index=False)
    state_delta.to_csv(os.path.join(OUT, "volatility_state_delta.csv"), index=False)
    top50.to_csv(os.path.join(OUT, "volatility_mandi_top50.csv"), index=False)
    mandi_cv.to_csv(os.path.join(OUT, "volatility_mandi_all.csv"), index=False)
    seasonal.to_csv(os.path.join(OUT, "volatility_monthly.csv"), index=False)
    summary.to_csv(os.path.join(OUT, "summary_stats.csv"), index=False)
    cv.to_parquet(os.path.join(OUT, "cv_panel.parquet"), index=False)

    log(f"\nAll outputs saved to 02-analysis/")
    log(f"DONE in {time.time()-t0:.0f}s")

    # ── Quick console table ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("YEAR-BY-YEAR MEAN CV (price volatility index)")
    print("="*60)
    pivot_yr = yearly.pivot_table(index="year", columns="crop", values="mean_cv")
    print(pivot_yr.round(3).to_string())

    print("\n" + "="*60)
    print("PRE vs POST 2018 — did Operation Greens reduce volatility?")
    print("="*60)
    print(summary[["crop","period","mean_cv","median_cv","p90_cv"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
