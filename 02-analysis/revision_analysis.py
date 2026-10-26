#!/usr/bin/env python
"""
revision_analysis.py
====================
Implements the 6 mandatory peer-review revisions:

  M1 — Event study (year-specific DiD coefficients, Figure 10)
  M2 — Control group CV comparison table + KS test
  M4 — Balanced panel robustness (DML on mandis in both periods)
  M5 — Moran's I of DML residuals (spatial autocorrelation check)
  M6 — Pre-expansion robustness (post restricted to Nov 2018 – Mar 2020)

Also adds:
  S1 — Minimum Detectable Effect (MDE) computation
  S8 — 95% confidence intervals for all estimates

Outputs (02-analysis/):
  revision_results.json       — all numerical results for paper
  fig10_event_study.png       — Figure 10: event study plot (300 DPI)

Run: python revision_analysis.py
"""

import os, re, json, time, warnings
import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

warnings.filterwarnings("ignore")

ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQ  = os.path.join(ROOT, "01-data", "archive (4)", "parquet")
GEO   = os.path.join(ROOT, "01-data", "geo", "gadm41_IND_2.json")
OUT   = os.path.join(ROOT, "02-analysis")
FIG   = os.path.join(ROOT, "03-drafts", "figures")
CV_PQ = os.path.join(OUT, "cv_panel.parquet")

STUDY_YEARS = list(range(2013, 2025))
POLICY_DATE = pd.Timestamp("2018-11-01")
N_FOLDS     = 5
TRIM_EPS    = 0.01
KNN         = 5

TOP_CROPS = {
    "Tomato": re.compile(r"^tomato", re.I),
    "Onion":  re.compile(r"^onion$", re.I),
    "Potato": re.compile(r"^potato", re.I),
}
CTRL_CROPS = {
    "Brinjal":    re.compile(r"^brinjal", re.I),
    "Cauliflower":re.compile(r"^cauliflower", re.I),
    "Bhindi":     re.compile(r"^bhindi|ladies finger", re.I),
}
EXPORT_BANS = [
    ("2019-09-13", "2020-03-08"),
    ("2023-08-19", "2024-05-04"),
]

norm = lambda s: re.sub(r"\s+", " ", str(s).strip())
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ── Data loading (mirrors dml_operation_greens.py) ────────────────────────────
def load_top_cv():
    cv = pd.read_parquet(CV_PQ)
    cv["TOP_c"] = 1
    return cv

def compute_ctrl_cv():
    frames = []
    for y in STUDY_YEARS:
        p = os.path.join(PARQ, f"{y}.parquet")
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(
            p, columns=["State","District","Market","Commodity",
                         "Arrival_Date","Min_Price","Max_Price","Modal_Price"],
        )
        df["crop"] = None
        com = df["Commodity"].astype(str).str.strip()
        for c, rx in CTRL_CROPS.items():
            df.loc[com.str.match(rx), "crop"] = c
        df = df[df.crop.notna()]
        df = df[(df.Modal_Price > 0) & (df.Min_Price <= df.Max_Price)]
        df["date"]  = pd.to_datetime(df["Arrival_Date"], errors="coerce")
        df["year"]  = df.date.dt.year
        df["month"] = df.date.dt.month
        df["State"]    = df.State.map(norm)
        df["District"] = df.District.map(norm)
        df["Market"]   = df.Market.map(norm)
        df["mkey"]     = df.State + " | " + df.District + " | " + df.Market
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw.date.notna() & raw.year.between(2013, 2024)]
    grp = raw.groupby(["mkey","State","District","Market","crop","year","month"])
    agg = grp["Modal_Price"].agg(price_mean="mean", price_std="std", n_days="count").reset_index()
    agg = agg[agg.n_days >= 5]
    agg["cv"]     = agg.price_std / agg.price_mean
    agg["log_cv"] = np.log(agg.cv.clip(lower=1e-6))
    agg["TOP_c"]  = 0
    agg["post"]   = (pd.to_datetime(dict(year=agg.year, month=agg.month, day=1)) >= POLICY_DATE).astype(int)
    return agg

def build_panel(top_cv, ctrl_cv):
    panel = pd.concat(
        [top_cv[["mkey","State","District","Market","crop","year","month","cv","log_cv","n_days","post","TOP_c"]],
         ctrl_cv[["mkey","State","District","Market","crop","year","month","cv","log_cv","n_days","post","TOP_c"]]],
        ignore_index=True,
    )
    panel = panel.dropna(subset=["log_cv","cv"])
    panel = panel[panel.cv > 0]
    panel["D"] = panel["TOP_c"] * panel["post"]
    panel["date_d1"] = pd.to_datetime(dict(year=panel.year, month=panel.month, day=1))
    panel["export_ban_onion"] = 0
    for s, e in EXPORT_BANS:
        mask = (panel.date_d1 >= s) & (panel.date_d1 <= e)
        panel.loc[mask, "export_ban_onion"] = 1
    panel["log_n_days"] = np.log(panel.n_days.clip(lower=1))
    panel["year_trend"] = panel.year - 2018
    panel["mandi_fe"]   = panel["mkey"].astype("category").cat.codes
    panel["crop_fe"]    = panel["crop"].astype("category").cat.codes
    return panel

def make_folds(panel):
    unique_mk  = np.unique(panel["mkey"].values)
    np.random.seed(42)
    mandi_fold = {m: i % N_FOLDS for i, m in enumerate(np.random.permutation(unique_mk))}
    return np.array([mandi_fold[m] for m in panel["mkey"].values])

def build_features(panel):
    month_dummies = pd.get_dummies(panel["month"], prefix="m", drop_first=False)
    month_dummies = month_dummies.drop(
        columns=[c for c in month_dummies.columns if c == "m_12"], errors="ignore"
    )
    shared = pd.concat([month_dummies, panel[["year_trend","export_ban_onion","log_n_days","mandi_fe"]]], axis=1).astype(float)
    Z_y = pd.concat([shared, panel[["crop_fe"]].astype(float)], axis=1)
    Z_t = shared.copy()
    return Z_y, Z_t

def cross_fit_lgbm(Y, Z_y, fold_ids):
    """Cross-fit LightGBM outcome model; return Y_tilde."""
    lgbm = lgb.LGBMRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05, num_leaves=63,
        min_child_samples=20, colsample_bytree=0.8, subsample=0.8,
        verbosity=-1, random_state=0,
    )
    Y_tilde = np.zeros(len(Y))
    mspe_parts = []
    for k in range(N_FOLDS):
        tr = np.where(fold_ids != k)[0]
        te = np.where(fold_ids == k)[0]
        lgbm.fit(Z_y.values[tr], Y[tr])
        pred = lgbm.predict(Z_y.values[te])
        Y_tilde[te] = Y[te] - pred
        mspe_parts.append(np.mean((Y[te] - pred)**2))
    log(f"  LightGBM MSPE(Y): {np.mean(mspe_parts):.5f}")
    return Y_tilde

def cross_fit_ridge_treatment(T, Z_t, fold_ids):
    """Cross-fit Ridge treatment model; return T_tilde."""
    ridge = Pipeline([("sc", StandardScaler()), ("r", Ridge(alpha=1.0))])
    T_tilde = np.zeros(len(T))
    mspe_parts = []
    for k in range(N_FOLDS):
        tr = np.where(fold_ids != k)[0]
        te = np.where(fold_ids == k)[0]
        ridge.fit(Z_t.values[tr], T[tr])
        pred = ridge.predict(Z_t.values[te])
        T_tilde[te] = T[te] - pred
        mspe_parts.append(np.mean((T[te] - pred)**2))
    log(f"  Ridge MSPE(T): {np.mean(mspe_parts):.5f}")
    return T_tilde

def plr_estimate(Y_tilde, T_tilde, clusters, alpha=0.05):
    """PLR point estimate + clustered SE + CI at level (1-alpha)."""
    theta = np.sum(Y_tilde * T_tilde) / np.sum(T_tilde**2)
    psi   = (Y_tilde - theta * T_tilde) * T_tilde
    # clustered sandwich
    unique_cl = np.unique(clusters)
    score_cl  = np.array([psi[clusters == c].sum() for c in unique_cl])
    V_cl      = (len(unique_cl) / (len(unique_cl)-1)) * np.sum(score_cl**2)
    denom     = (np.sum(T_tilde**2))**2
    V_if      = V_cl / denom
    se        = np.sqrt(V_if)
    z         = sps.norm.ppf(1 - alpha/2)
    ci_lo     = theta - z * se
    ci_hi     = theta + z * se
    return theta, se, ci_lo, ci_hi


# ═════════════════════════════════════════════════════════════════
# M2 — Control group comparison
# ═════════════════════════════════════════════════════════════════
def run_m2(top_cv, ctrl_cv):
    log("\n=== M2: Control Group CV Comparison ===")
    pre_top  = top_cv[top_cv.year < 2018]["cv"].dropna()
    pre_ctrl = ctrl_cv[ctrl_cv.year < 2018]["cv"].dropna()

    stats_top  = {"mean": pre_top.mean(), "median": pre_top.median(),
                  "std": pre_top.std(),   "p90": pre_top.quantile(0.90),
                  "n": len(pre_top)}
    stats_ctrl = {"mean": pre_ctrl.mean(), "median": pre_ctrl.median(),
                  "std": pre_ctrl.std(),   "p90": pre_ctrl.quantile(0.90),
                  "n": len(pre_ctrl)}

    ks_stat, ks_p = sps.ks_2samp(pre_top.values, pre_ctrl.values)

    # Per-crop pre-period mean CV
    per_crop = {}
    for c, df in [("Tomato", top_cv), ("Onion", top_cv), ("Potato", top_cv)]:
        vals = df[(df.crop == c) & (df.year < 2018)]["cv"].dropna()
        per_crop[c] = {"mean": float(vals.mean()), "median": float(vals.median()), "std": float(vals.std())}
    for c, df in [("Brinjal", ctrl_cv), ("Cauliflower", ctrl_cv), ("Bhindi", ctrl_cv)]:
        vals = df[(df.crop == c) & (df.year < 2018)]["cv"].dropna()
        per_crop[c] = {"mean": float(vals.mean()), "median": float(vals.median()), "std": float(vals.std())}

    log(f"  TOP pre-2018 CV: mean={stats_top['mean']:.4f}  median={stats_top['median']:.4f}  std={stats_top['std']:.4f}")
    log(f"  Control pre-2018 CV: mean={stats_ctrl['mean']:.4f}  median={stats_ctrl['median']:.4f}  std={stats_ctrl['std']:.4f}")
    log(f"  KS test: D={ks_stat:.4f}  p={ks_p:.4f}")

    return {
        "top_pre2018":  {k: round(v, 5) for k, v in stats_top.items()},
        "ctrl_pre2018": {k: round(v, 5) for k, v in stats_ctrl.items()},
        "ks_statistic": round(float(ks_stat), 5),
        "ks_pvalue":    round(float(ks_p), 5),
        "per_crop":     {k: {m: round(v, 5) for m, v in vals.items()} for k, vals in per_crop.items()},
    }


# ═════════════════════════════════════════════════════════════════
# M1 — Event study
# ═════════════════════════════════════════════════════════════════
def run_m1_event_study(panel, Y_tilde):
    """
    Event study using year-demeaned Y_tilde to avoid collinearity.
    Y_tilde absorbs linear year_trend; demeaning removes remaining year-level shocks.
    OLS: Y_dem ~ TOP_c × year_dummy (reference=2017), no separate year FEs.
    Mandi-clustered SEs.
    """
    log("\n=== M1: Event Study ===")
    panel = panel.copy().reset_index(drop=True)

    # Step 1: year-demean Y_tilde to remove non-linear year aggregate shocks
    yr_means = pd.Series(Y_tilde).groupby(panel["year"].values).transform("mean").values
    Y_dem = Y_tilde - yr_means

    # Step 2: TOP_c × year interactions; reference year = 2017
    ref_year  = 2017
    event_years = [y for y in range(2013, 2025) if y != ref_year]
    event_vars  = []
    for y in event_years:
        colname = f"top_yr_{y}"
        panel[colname] = (panel["TOP_c"] * (panel["year"] == y)).astype(float)
        event_vars.append(colname)

    import statsmodels.api as sm
    X = panel[event_vars].astype(float)
    X = sm.add_constant(X)

    clusters = panel["mkey"].astype("category").cat.codes.values
    model    = sm.OLS(Y_dem, X).fit(cov_type="cluster",
                                     cov_kwds={"groups": clusters})

    coefs = model.params[event_vars]
    ci95  = model.conf_int(alpha=0.05).loc[event_vars]
    ci90  = model.conf_int(alpha=0.10).loc[event_vars]

    results = {}
    for y, var in zip(event_years, event_vars):
        results[y] = {
            "coef":    round(float(coefs[var]), 6),
            "ci95_lo": round(float(ci95.loc[var, 0]), 6),
            "ci95_hi": round(float(ci95.loc[var, 1]), 6),
            "ci90_lo": round(float(ci90.loc[var, 0]), 6),
            "ci90_hi": round(float(ci90.loc[var, 1]), 6),
        }
        log(f"  {y}: β={results[y]['coef']:+.4f}  95%CI[{results[y]['ci95_lo']:+.4f}, {results[y]['ci95_hi']:+.4f}]")

    # Pre-period Wald test (joint significance of years 2013-2016)
    pre_vars = [f"top_yr_{y}" for y in range(2013, 2017)]
    try:
        wald     = model.wald_test([f"{v} = 0" for v in pre_vars], use_f=False)
        wald_p   = float(wald.pvalue)
        wald_stat = float(wald.statistic.flat[0])
    except Exception:
        # fallback: manual chi2 from the cluster-robust variance
        b_pre = np.array([float(coefs[v]) for v in pre_vars])
        V_pre = model.cov_params().loc[pre_vars, pre_vars].values
        wald_stat = float(b_pre @ np.linalg.pinv(V_pre) @ b_pre)
        wald_p    = float(1 - sps.chi2.cdf(wald_stat, df=len(pre_vars)))
    log(f"  Pre-period Wald test (2013-2016): chi2={wald_stat:.3f}  p={wald_p:.4f}")
    log(f"  {'PARALLEL TRENDS HOLD' if wald_p > 0.1 else 'Pre-period variation detected (assess magnitude)'}")

    # ── Figure 10 ────────────────────────────────────────────────
    years_plot  = list(results.keys()) + [ref_year]
    coefs_plot  = [results[y]["coef"] for y in event_years] + [0.0]
    lo95_plot   = [results[y]["ci95_lo"] for y in event_years] + [0.0]
    hi95_plot   = [results[y]["ci95_hi"] for y in event_years] + [0.0]
    lo90_plot   = [results[y]["ci90_lo"] for y in event_years] + [0.0]
    hi90_plot   = [results[y]["ci90_hi"] for y in event_years] + [0.0]

    sort_idx   = np.argsort(years_plot)
    yrs_sorted = np.array(years_plot)[sort_idx]
    co_sorted  = np.array(coefs_plot)[sort_idx]
    lo95_s     = np.array(lo95_plot)[sort_idx]
    hi95_s     = np.array(hi95_plot)[sort_idx]
    lo90_s     = np.array(lo90_plot)[sort_idx]
    hi90_s     = np.array(hi90_plot)[sort_idx]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(2018, color="#D62728", linewidth=1.2, linestyle=":", alpha=0.8, label="Operation Greens launch")
    ax.axvspan(2013, 2017.5, alpha=0.04, color="grey", label="Pre-policy window")

    # 95% CI band
    ax.fill_between(yrs_sorted, lo95_s, hi95_s, alpha=0.15, color="#1F77B4", label="95% CI")
    # 90% CI band
    ax.fill_between(yrs_sorted, lo90_s, hi90_s, alpha=0.25, color="#1F77B4", label="90% CI")
    # point estimates
    ax.plot(yrs_sorted, co_sorted, "o-", color="#1F77B4", linewidth=2,
            markersize=7, markerfacecolor="white", markeredgewidth=2)
    # reference point
    ref_idx = list(yrs_sorted).index(ref_year)
    ax.plot(ref_year, 0, "s", color="grey", markersize=8,
            markerfacecolor="grey", label=f"Reference year ({ref_year})")

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("DiD coefficient on log(CV)\n(TOP vs. non-TOP crops)", fontsize=11)
    ax.set_title("Figure 10: Event Study — Operation Greens Effect on Price Volatility\n"
                 "(relative to 2017 baseline, TOP vs. control crops, mandi-clustered 95% CI)", fontsize=12)
    ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(2012.5, 2024.5)

    # Wald annotation top-right (keep clear of legend)
    ax.text(0.98, 0.97,
            f"Pre-trend Wald test (2013–2016):\n$\\chi^2$={wald_stat:.2f}, p={wald_p:.3f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="grey", alpha=0.95))

    plt.tight_layout()
    out_path = os.path.join(FIG, "fig10_event_study.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"  Figure 10 saved → {out_path}")

    return results, {"wald_chi2": wald_stat, "wald_pvalue": wald_p}


# ═════════════════════════════════════════════════════════════════
# M4 — Balanced panel
# ═════════════════════════════════════════════════════════════════
def run_m4_balanced(panel):
    log("\n=== M4: Balanced Panel DML ===")
    pre_keys  = set(panel.loc[panel.year <= 2018, "mkey"].unique())
    post_keys = set(panel.loc[panel.year > 2018,  "mkey"].unique())
    balanced_keys = pre_keys & post_keys

    bp = panel[panel["mkey"].isin(balanced_keys)].copy()
    log(f"  Full panel: {panel.mkey.nunique():,} mandis  | Balanced: {len(balanced_keys):,} mandis  ({len(bp):,} obs)")

    Z_y, Z_t = build_features(bp)
    folds     = make_folds(bp)
    Y  = bp["log_cv"].values
    T  = bp["D"].values

    Y_tilde = cross_fit_lgbm(Y, Z_y, folds)
    T_tilde = cross_fit_ridge_treatment(T, Z_t, folds)

    clusters = bp["mkey"].astype("category").cat.codes.values
    theta95, se95, lo95, hi95 = plr_estimate(Y_tilde, T_tilde, clusters, alpha=0.05)
    theta90, se90, lo90, hi90 = plr_estimate(Y_tilde, T_tilde, clusters, alpha=0.10)

    pct = (np.exp(theta95) - 1) * 100
    log(f"  Balanced ATE={theta95:+.5f} ({pct:+.2f}% CV)  95%CI[{lo95:+.5f}, {hi95:+.5f}]")

    return {
        "n_mandis_full": int(panel.mkey.nunique()),
        "n_mandis_balanced": int(len(balanced_keys)),
        "n_obs": int(len(bp)),
        "ate_log_cv": round(float(theta95), 5),
        "se": round(float(se95), 6),
        "ci95_lo": round(float(lo95), 5),
        "ci95_hi": round(float(hi95), 5),
        "ci90_lo": round(float(lo90), 5),
        "ci90_hi": round(float(hi90), 5),
        "pct_change_cv": round(float(pct), 3),
        "significant_95": bool(lo95 > 0 or hi95 < 0),
    }, Y_tilde, T_tilde


# ═════════════════════════════════════════════════════════════════
# M6 — Pre-expansion robustness (Nov 2018 – Mar 2020)
# ═════════════════════════════════════════════════════════════════
def run_m6_pre_expansion(panel):
    log("\n=== M6: Pre-expansion robustness (Nov 2018 – Mar 2020) ===")
    p = panel.copy()
    # Keep all pre-2018 obs + post obs only up to March 2020
    post_cutoff = pd.Timestamp("2020-04-01")
    mask = (p.date_d1 < POLICY_DATE) | (p.date_d1 < post_cutoff)
    p = p[mask].copy()
    # Redefine D using the same post flag but now restricted to this window
    p["D"] = p["TOP_c"] * (p.date_d1 >= POLICY_DATE).astype(int)
    log(f"  Pre-expansion panel: {len(p):,} obs  D=1: {p.D.mean():.1%}")

    Z_y, Z_t = build_features(p)
    folds     = make_folds(p)
    Y  = p["log_cv"].values
    T  = p["D"].values

    Y_tilde = cross_fit_lgbm(Y, Z_y, folds)
    T_tilde = cross_fit_ridge_treatment(T, Z_t, folds)

    clusters = p["mkey"].astype("category").cat.codes.values
    theta95, se95, lo95, hi95 = plr_estimate(Y_tilde, T_tilde, clusters, alpha=0.05)
    theta90, se90, lo90, hi90 = plr_estimate(Y_tilde, T_tilde, clusters, alpha=0.10)

    pct = (np.exp(theta95) - 1) * 100
    log(f"  Pre-expansion ATE={theta95:+.5f} ({pct:+.2f}% CV)  95%CI[{lo95:+.5f}, {hi95:+.5f}]")

    return {
        "period": "Nov 2018 – Mar 2020",
        "n_obs": int(len(p)),
        "ate_log_cv": round(float(theta95), 5),
        "se": round(float(se95), 6),
        "ci95_lo": round(float(lo95), 5),
        "ci95_hi": round(float(hi95), 5),
        "ci90_lo": round(float(lo90), 5),
        "ci90_hi": round(float(hi90), 5),
        "pct_change_cv": round(float(pct), 3),
        "significant_95": bool(lo95 > 0 or hi95 < 0),
    }


# ═════════════════════════════════════════════════════════════════
# M5 — Moran's I of DML residuals
# ═════════════════════════════════════════════════════════════════
def run_m5_moran_residuals(panel, Y_tilde, T_tilde, theta):
    log("\n=== M5: Moran's I of DML residuals ===")
    psi = Y_tilde - theta * T_tilde  # PLR residual

    # Aggregate to district level
    meta = panel[["State","District"]].copy().reset_index(drop=True)
    meta["psi"] = psi
    district_psi = meta.groupby(["State","District"])["psi"].mean().reset_index()
    district_psi.columns = ["State","District","psi"]

    # Try to load GADM for centroid-based spatial weights
    try:
        import geopandas as gpd
        g = gpd.read_file(GEO)
        cent = g.geometry.representative_point()
        g = g.assign(lon=cent.x, lat=cent.y)
        g["dist_norm"]  = g["NAME_2"].str.strip().str.lower()
        g["state_norm"] = g["NAME_1"].str.strip().str.lower()

        district_psi["dist_norm"]  = district_psi["District"].str.lower().str.strip()
        district_psi["state_norm"] = district_psi["State"].str.lower().str.strip()

        merged = district_psi.merge(
            g[["dist_norm","state_norm","lat","lon"]], on=["dist_norm","state_norm"], how="inner"
        )
        log(f"  Matched {len(merged)} / {len(district_psi)} districts to GADM centroids")

        if len(merged) < 20:
            raise ValueError("Too few matches — falling back to state-level")

        coords = merged[["lat","lon"]].values
        from scipy.spatial.distance import cdist
        dists  = cdist(coords, coords)
        np.fill_diagonal(dists, np.inf)

        # k-NN spatial weights
        W = np.zeros_like(dists)
        for i in range(len(coords)):
            nn = np.argsort(dists[i])[:KNN]
            W[i, nn] = 1.0
        # row-standardize
        W = W / W.sum(axis=1, keepdims=True)

        x    = merged["psi"].values
        xbar = x.mean()
        n    = len(x)
        z    = x - xbar
        Wz   = W @ z
        I    = (n * np.dot(z, Wz)) / (np.dot(z, z) * W.sum())

        # permutation test (999 iterations)
        np.random.seed(42)
        I_perm = []
        for _ in range(999):
            xp = np.random.permutation(x)
            zp = xp - xp.mean()
            I_perm.append((n * np.dot(zp, W @ zp)) / (np.dot(zp, zp) * W.sum()))
        p_val = (np.sum(np.abs(I_perm) >= abs(I)) + 1) / 1000

        log(f"  Moran's I (DML residuals): I={I:.4f}  p={p_val:.4f}  n_districts={n}")
        spatial_concern = bool(p_val < 0.05)
        log(f"  {'SPATIAL AUTOCORRELATION DETECTED — Conley SEs recommended' if spatial_concern else 'No significant spatial autocorrelation in residuals'}")

        return {
            "morans_I": round(float(I), 5),
            "p_value": round(float(p_val), 4),
            "n_districts": int(n),
            "spatial_concern": spatial_concern,
            "weights": f"k={KNN} nearest neighbours, row-standardised",
            "method": "GADM district centroids, permutation test 999 iter",
        }

    except Exception as e:
        log(f"  GADM match failed ({e}) — using state-level fallback")
        # State-level Moran's I using approximate state centroids
        state_psi = meta.groupby("State")["psi"].mean().reset_index()

        # Approximate state centroids (major Indian states)
        state_coords = {
            "Andhra Pradesh": (15.9, 79.7), "Arunachal Pradesh": (28.0, 94.7),
            "Assam": (26.2, 92.9), "Bihar": (25.7, 85.9), "Chhattisgarh": (21.3, 81.9),
            "Delhi": (28.7, 77.1), "Goa": (15.3, 74.0), "Gujarat": (22.3, 71.2),
            "Haryana": (29.1, 76.1), "Himachal Pradesh": (31.1, 77.2),
            "Jharkhand": (23.6, 85.3), "Karnataka": (15.3, 75.7),
            "Kerala": (10.5, 76.2), "Madhya Pradesh": (23.5, 77.4),
            "Maharashtra": (19.7, 75.7), "Manipur": (24.7, 94.0),
            "Meghalaya": (25.5, 91.4), "Mizoram": (23.2, 92.9),
            "Nagaland": (26.2, 94.6), "Odisha": (20.9, 84.8),
            "Punjab": (31.1, 75.3), "Rajasthan": (27.0, 74.2),
            "Sikkim": (27.5, 88.5), "Tamil Nadu": (11.1, 78.7),
            "Telangana": (17.9, 79.2), "Tripura": (23.7, 91.7),
            "Uttar Pradesh": (26.8, 80.9), "Uttarakhand": (30.1, 79.3),
            "West Bengal": (22.8, 87.8), "Jammu And Kashmir": (33.7, 75.4),
            "Himachal Pradesh": (31.1, 77.2),
        }
        state_psi["lat"] = state_psi["State"].map(lambda s: state_coords.get(s, (None, None))[0])
        state_psi["lon"] = state_psi["State"].map(lambda s: state_coords.get(s, (None, None))[1])
        state_psi = state_psi.dropna(subset=["lat","lon"])

        coords = state_psi[["lat","lon"]].values
        from scipy.spatial.distance import cdist
        dists  = cdist(coords, coords)
        np.fill_diagonal(dists, np.inf)
        W = np.zeros_like(dists)
        for i in range(len(coords)):
            nn = np.argsort(dists[i])[:min(KNN, len(coords)-1)]
            W[i, nn] = 1.0
        W = W / W.sum(axis=1, keepdims=True)

        x    = state_psi["psi"].values
        xbar = x.mean()
        n    = len(x)
        z    = x - xbar
        Wz   = W @ z
        I    = (n * np.dot(z, Wz)) / (np.dot(z, z) * W.sum())

        np.random.seed(42)
        I_perm = []
        for _ in range(999):
            xp = np.random.permutation(x)
            zp = xp - xp.mean()
            I_perm.append((n * np.dot(zp, W @ zp)) / (np.dot(zp, zp) * W.sum()))
        p_val = (np.sum(np.abs(I_perm) >= abs(I)) + 1) / 1000

        log(f"  State-level Moran's I: I={I:.4f}  p={p_val:.4f}  n_states={n}")
        return {
            "morans_I": round(float(I), 5),
            "p_value": round(float(p_val), 4),
            "n_districts": int(n),
            "spatial_concern": bool(p_val < 0.05),
            "weights": "state-level centroids (GADM match failed)",
            "method": "Approximate state centroids, permutation test 999 iter",
        }


# ═════════════════════════════════════════════════════════════════
# S1 — Minimum Detectable Effect
# ═════════════════════════════════════════════════════════════════
def compute_mde(Y_tilde, T_tilde, alpha=0.05, power=0.80):
    """MDE = smallest |theta| detectable at given alpha and power."""
    n_eff  = np.sum(T_tilde**2)**2 / np.sum(T_tilde**4)  # effective n
    sigma2 = np.var(Y_tilde)
    se_ref = np.sqrt(sigma2 / (n_eff * np.var(T_tilde)))
    z_a    = sps.norm.ppf(1 - alpha/2)
    z_b    = sps.norm.ppf(power)
    mde    = (z_a + z_b) * se_ref
    mde_pct = (np.exp(mde) - 1) * 100
    return mde, mde_pct


# ═════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    log("=== REVISION ANALYSIS — Operation Greens ===\n")

    # ── Load data ─────────────────────────────────────────────────
    log("Loading TOP crop CV...")
    top_cv = load_top_cv()
    log("Computing control crop CV...")
    ctrl_cv = compute_ctrl_cv()
    panel = build_panel(top_cv, ctrl_cv)

    # ── Main cross-fit (full panel) for M1 + M5 ──────────────────
    log("\n--- Full panel cross-fit (LightGBM) ---")
    Z_y, Z_t = build_features(panel)
    folds_main = make_folds(panel)
    Y_main = panel["log_cv"].values
    T_main = panel["D"].values

    Y_tilde_main = cross_fit_lgbm(Y_main, Z_y, folds_main)
    T_tilde_main = cross_fit_ridge_treatment(T_main, Z_t, folds_main)

    # Main ATE (with 95% CI this time)
    clusters_main = panel["mkey"].astype("category").cat.codes.values
    theta_main, se_main, lo95_main, hi95_main = plr_estimate(
        Y_tilde_main, T_tilde_main, clusters_main, alpha=0.05
    )
    _, _, lo90_main, hi90_main = plr_estimate(
        Y_tilde_main, T_tilde_main, clusters_main, alpha=0.10
    )
    pct_main = (np.exp(theta_main) - 1) * 100
    log(f"\nMain ATE (LightGBM): {theta_main:+.5f}  ({pct_main:+.2f}% CV)")
    log(f"  90% CI: [{lo90_main:+.5f}, {hi90_main:+.5f}]")
    log(f"  95% CI: [{lo95_main:+.5f}, {hi95_main:+.5f}]")

    # MDE
    mde, mde_pct = compute_mde(Y_tilde_main, T_tilde_main)
    log(f"\n  MDE (alpha=5%, power=80%): {mde:.5f} log-CV units = {mde_pct:.2f}% CV reduction")

    # ── Run revisions ─────────────────────────────────────────────
    m2_results  = run_m2(top_cv, ctrl_cv)
    es_results, wald = run_m1_event_study(panel.copy(), Y_tilde_main)
    m4_results, _, _ = run_m4_balanced(panel)
    m5_results  = run_m5_moran_residuals(panel, Y_tilde_main, T_tilde_main, theta_main)
    m6_results  = run_m6_pre_expansion(panel)

    # ── Compile and save ──────────────────────────────────────────
    all_results = {
        "main_ate_full_95ci": {
            "ate_log_cv": round(float(theta_main), 5),
            "se": round(float(se_main), 6),
            "ci95_lo": round(float(lo95_main), 5),
            "ci95_hi": round(float(hi95_main), 5),
            "ci90_lo": round(float(lo90_main), 5),
            "ci90_hi": round(float(hi90_main), 5),
            "pct_change_cv": round(float(pct_main), 3),
        },
        "mde": {
            "mde_log_cv": round(float(mde), 5),
            "mde_pct_cv": round(float(mde_pct), 3),
            "alpha": 0.05, "power": 0.80,
        },
        "M2_control_comparison": m2_results,
        "M1_event_study": {"coefficients": {str(k): v for k, v in es_results.items()}, "wald_pretrend": wald},
        "M4_balanced_panel": m4_results,
        "M5_moran_residuals": m5_results,
        "M6_pre_expansion": m6_results,
    }

    out_path = os.path.join(OUT, "revision_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    log(f"\n=== DONE in {time.time()-t0:.0f}s ===")
    log(f"Results saved → {out_path}")

    print("\n" + "="*65)
    print("REVISION RESULTS SUMMARY")
    print("="*65)
    print(f"Main ATE (LightGBM, 95% CI): {theta_main:+.5f} [{lo95_main:+.5f}, {hi95_main:+.5f}]")
    print(f"  = {pct_main:+.2f}% change in CV  |  {'significant' if (lo95_main>0 or hi95_main<0) else 'NOT significant'}")
    print(f"  MDE: can detect effects >= {mde_pct:.2f}% CV reduction at 80% power")
    print(f"\nM2 — Control group KS test: D={m2_results['ks_statistic']:.4f}  p={m2_results['ks_pvalue']:.4f}")
    print(f"     TOP mean CV={m2_results['top_pre2018']['mean']:.4f}  Control mean CV={m2_results['ctrl_pre2018']['mean']:.4f}")
    print(f"\nM1 — Pre-trend Wald: chi2={wald['wald_chi2']:.3f}  p={wald['wald_pvalue']:.4f}")
    print(f"     {'Pre-trends hold (p > 0.10)' if wald['wald_pvalue'] > 0.10 else 'WARNING: Pre-trend detected'}")
    print(f"\nM4 — Balanced panel ATE: {m4_results['ate_log_cv']:+.5f}  95%CI[{m4_results['ci95_lo']:+.5f}, {m4_results['ci95_hi']:+.5f}]")
    print(f"     N mandis balanced: {m4_results['n_mandis_balanced']:,} / {m4_results['n_mandis_full']:,}")
    print(f"\nM5 — Moran's I of residuals: I={m5_results['morans_I']:.4f}  p={m5_results['p_value']:.4f}")
    print(f"     {'Spatial autocorrelation detected — Conley SEs needed' if m5_results['spatial_concern'] else 'No significant spatial autocorrelation'}")
    print(f"\nM6 — Pre-expansion ATE: {m6_results['ate_log_cv']:+.5f}  95%CI[{m6_results['ci95_lo']:+.5f}, {m6_results['ci95_hi']:+.5f}]")
    print(f"     Period: {m6_results['period']}")


if __name__ == "__main__":
    main()
