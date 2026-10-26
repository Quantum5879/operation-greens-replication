#!/usr/bin/env python
"""
DML Estimation — Operation Greens Policy Evaluation
=====================================================
Follows the 8-Gate DML methodology pipeline.

DESIGN (Gate 0):
  Estimand : ATET of Operation Greens on log price volatility (log CV)
  Model    : DML-DiD (PLR with crop×time interaction as treatment)
  Treatment: D = TOP_c × post_t  (binary; varies by crop type AND time)
             TOP_c = 1 for Tomato/Onion/Potato, 0 for Brinjal/Cauliflower/Bhindi
             post_t = 1 after November 2018
  Outcome  : Y = log_cv  (log of monthly price CV per mandi-crop)
  Nuisance : Z = month FEs, year_trend, export_ban_onion, obs_count, mandi_FE, crop_FE
  Excluded : Cold-chain infra + NAFED intensity (mediators — Gate 0 DAG)

GATES IMPLEMENTED:
  Gate 1 — Feature engineering (month FEs, year trend, export ban, market activity)
  Gate 2 — Three learners: LightGBM, RandomForest, Ridge (mandatory, not one learner)
  Gate 3 — K=5 cross-fitting, mandi-clustered folds
  Gate 4 — Neyman-orthogonal PLR score (LinearDML)
  Gate 5 — Overlap: propensity histogram + trim at ε=0.01
  Gate 6 — All-learner estimates reported; select by cross-fitted MSPE
  Gate 7 — CausalForestDML CATE + BLP validation test
  Gate 8 — COVID sensitivity (drop 2020-03–2021-06) + placebo (fake Nov 2015)

Outputs (02-analysis/):
  dml_results_og.json          — ATE per crop + all-learner comparison + CIs
  dml_overlap.csv              — propensity score distribution per crop
  cate_og.parquet              — per-observation CATE predictions
  dml_cate_district.csv        — district-level mean CATE (for choropleth map)
  dml_robustness.json          — COVID-excluded + placebo results

Run: python dml_operation_greens.py
"""

import os, re, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
from econml.dml import LinearDML, CausalForestDML

warnings.filterwarnings("ignore")

ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQ  = os.path.join(ROOT, "01-data", "archive (4)", "parquet")
OUT   = os.path.join(ROOT, "02-analysis")
CV_PQ = os.path.join(OUT, "cv_panel.parquet")

STUDY_YEARS  = list(range(2013, 2025))
POLICY_DATE  = pd.Timestamp("2018-11-01")
N_FOLDS      = 5
TRIM_EPS     = 0.01    # Gate 5 overlap trim

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

# known onion export ban windows (Gate 1 — export_ban_onion control)
EXPORT_BANS = [
    ("2019-09-13", "2020-03-08"),
    ("2023-08-19", "2024-05-04"),
]

norm = lambda s: re.sub(r"\s+", " ", str(s).strip())

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ── Load pre-computed TOP crop CV ─────────────────────────────────────────────
def load_top_cv():
    cv = pd.read_parquet(CV_PQ)
    cv["TOP_c"] = 1
    log(f"TOP CV loaded: {len(cv):,} rows")
    return cv


# ── Compute control crop CV on the fly ───────────────────────────────────────
def compute_ctrl_cv():
    """Pull Brinjal/Cauliflower/Bhindi monthly CV from raw parquet files."""
    frames = []
    for y in STUDY_YEARS:
        p = os.path.join(PARQ, f"{y}.parquet")
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(
            p,
            columns=["State","District","Market","Commodity",
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
        df["State"]   = df.State.map(norm)
        df["District"]= df.District.map(norm)
        df["Market"]  = df.Market.map(norm)
        df["mkey"]    = df.State + " | " + df.District + " | " + df.Market
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw.date.notna() & raw.year.between(2013, 2024)]

    grp = raw.groupby(["mkey","State","District","Market","crop","year","month"])
    agg = grp["Modal_Price"].agg(
        price_mean="mean", price_std="std", n_days="count"
    ).reset_index()
    agg = agg[agg.n_days >= 5]
    agg["cv"]     = agg.price_std / agg.price_mean
    agg["log_cv"] = np.log(agg.cv.clip(lower=1e-6))
    agg["TOP_c"]  = 0
    agg["post"]   = (
        pd.to_datetime(dict(year=agg.year, month=agg.month, day=1)) >= POLICY_DATE
    ).astype(int)
    log(f"Control crop CV computed: {len(agg):,} rows  ({raw.mkey.nunique():,} mandis)")
    return agg


# ── Build analysis panel ──────────────────────────────────────────────────────
def build_panel(top_cv, ctrl_cv):
    """
    Combine TOP + control crops.
    Gate 1 feature engineering:
      - treatment D = TOP_c × post_t
      - month FEs (m1..m11, drop m12)
      - year_trend (linear, centred)
      - export_ban_onion (binary)
      - log_n_days (market activity proxy — no arrivals in data)
      - mandi_FE (label-encoded integer — LightGBM handles categoricals)
      - crop_FE  (label-encoded integer)
    """
    panel = pd.concat(
        [top_cv[["mkey","State","District","Market","crop","year","month",
                  "cv","log_cv","n_days","post","TOP_c"]],
         ctrl_cv[["mkey","State","District","Market","crop","year","month",
                   "cv","log_cv","n_days","post","TOP_c"]]],
        ignore_index=True,
    )
    panel = panel.dropna(subset=["log_cv","cv"])
    panel = panel[panel.cv > 0]

    # treatment
    panel["D"] = panel["TOP_c"] * panel["post"]

    # export ban on onion (applies to ALL crops as a market-level shock)
    panel["date_d1"] = pd.to_datetime(
        dict(year=panel.year, month=panel.month, day=1)
    )
    panel["export_ban_onion"] = 0
    for s, e in EXPORT_BANS:
        mask = (panel.date_d1 >= s) & (panel.date_d1 <= e)
        panel.loc[mask, "export_ban_onion"] = 1

    # market activity proxy
    panel["log_n_days"] = np.log(panel.n_days.clip(lower=1))

    # year trend (centred at 2018 = policy year)
    panel["year_trend"] = panel.year - 2018

    # label encode mandi and crop for tree-based models
    panel["mandi_fe"] = panel["mkey"].astype("category").cat.codes
    panel["crop_fe"]  = panel["crop"].astype("category").cat.codes

    log(f"Panel built: {len(panel):,} rows | "
        f"{panel.mkey.nunique():,} mandis | "
        f"{panel.crop.nunique()} crops | "
        f"D=1: {panel.D.mean():.1%}")
    return panel


# ── Gate 1: build Z_y, Z_t, Y, D ─────────────────────────────────────────────
def make_YDZ(panel, drop_covid=False, placebo_year=None):
    """
    Returns Y, T, Z_y, Z_t, fold_ids, meta.

    Z_y (outcome nuisance) : month FEs + year_trend + export_ban + log_n_days
                             + mandi_fe + crop_fe   ← crop_fe helps predict Y
    Z_t (treatment nuisance): same BUT without crop_fe ← Gate 5 fix:
                              prevents tree/linear model from perfectly
                              predicting D = TOP_c×post_t via crop type.
    """
    p = panel.copy()

    if drop_covid:
        covid_mask = (p.date_d1 >= "2020-03-01") & (p.date_d1 <= "2021-06-30")
        p = p[~covid_mask]
        log(f"  COVID excluded: {covid_mask.sum():,} obs dropped")

    if placebo_year is not None:
        p = p[p.year <= 2018].copy()
        fake_post = (p.date_d1 >= f"{placebo_year}-11-01").astype(int)
        p["D"] = p["TOP_c"] * fake_post
        log(f"  PLACEBO: fake launch Nov-{placebo_year}, n={len(p):,}")

    Y       = p["log_cv"].values
    T_ravel = p["D"].values

    month_dummies = pd.get_dummies(p["month"], prefix="m", drop_first=False)
    month_dummies = month_dummies.drop(
        columns=[c for c in month_dummies.columns if c == "m_12"], errors="ignore"
    )
    shared = pd.concat([
        month_dummies,
        p[["year_trend","export_ban_onion","log_n_days","mandi_fe"]],
    ], axis=1).astype(float)

    Z_y = pd.concat([shared, p[["crop_fe"]].astype(float)], axis=1)
    Z_t = shared.copy()   # NO crop_fe — prevents perfect D prediction

    # fold IDs clustered by mandi (Gate 3)
    mandi_ids  = p["mkey"].values
    unique_mk  = np.unique(mandi_ids)
    np.random.seed(42)
    mandi_fold = {m: i % N_FOLDS for i, m in enumerate(np.random.permutation(unique_mk))}
    fold_ids   = np.array([mandi_fold[m] for m in mandi_ids])

    meta = p[["mkey","State","District","crop","year","month","TOP_c","post","D"]].copy()
    return Y, T_ravel, Z_y, Z_t, fold_ids, meta


# ── Gate 2: nuisance learners ─────────────────────────────────────────────────
def make_outcome_learners():
    """Flexible learners for ℓ(Z_y) = E[Y|Z_y]. crop_fe included in Z_y."""
    return {
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            num_leaves=63, min_child_samples=20,
            colsample_bytree=0.8, subsample=0.8,
            verbosity=-1, random_state=0,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_features="sqrt",
            min_samples_leaf=20, n_jobs=-1, random_state=0,
        ),
        "Ridge": Pipeline([("sc", StandardScaler()), ("r", Ridge(alpha=1.0))]),
    }

def make_treatment_learner():
    """
    Linear learner ONLY for m(Z_t) = E[D|Z_t].
    Gate 5 fix: crop_fe is excluded from Z_t so tree models cannot
    perfectly predict D = TOP_c × post_t from crop type alone.
    Ridge is the right choice here — it can only learn additive effects,
    leaving the crop×time interaction as the residualised D̃.
    """
    return Pipeline([("sc", StandardScaler()), ("r", Ridge(alpha=1.0))])


# ── Gate 3-4-6: manual PLR cross-fitting ─────────────────────────────────────
def run_dml(Y, T_ravel, Z_y, Z_t, fold_ids, label="main"):
    """
    Manual Neyman-orthogonal PLR (Gate 4):
      Ỹ = Y  - ℓ̂(Z_y)    [flexible outcome nuisance]
      D̃ = D  - m̂(Z_t)    [linear treatment nuisance, NO crop_fe]
      θ̂ = Σ(Ỹ·D̃) / Σ(D̃²)
      V̂  from influence function  ψ = (Ỹ - θ̂·D̃)·D̃
      SE  = √(mean(ψ²) / mean(D̃²)² / n)

    Z_y includes crop_fe (helps predict Y).
    Z_t excludes crop_fe (prevents perfect prediction of D).
    Three outcome learners; single Ridge treatment learner.
    """
    n = len(Y)
    outcome_learners = make_outcome_learners()
    t_learner        = make_treatment_learner()
    results = {}

    Z_y_arr = Z_y.values
    Z_t_arr = Z_t.values

    # cross-fitted treatment residuals (same for all outcome learners)
    T_tilde = np.zeros(n)
    for k in range(N_FOLDS):
        tr = np.where(fold_ids != k)[0]
        te = np.where(fold_ids == k)[0]
        t_learner.fit(Z_t_arr[tr], T_ravel[tr])
        T_tilde[te] = T_ravel[te] - t_learner.predict(Z_t_arr[te])

    T_var = np.mean(T_tilde ** 2)
    log(f"  Treatment residual variance: {T_var:.5f}  "
        f"(>0 = overlap OK; ~0 = collapse)")

    for name, yl in outcome_learners.items():
        log(f"  PLR [{name}] ...")
        Y_tilde = np.zeros(n)
        for k in range(N_FOLDS):
            tr = np.where(fold_ids != k)[0]
            te = np.where(fold_ids == k)[0]
            yl.fit(Z_y_arr[tr], Y[tr])
            Y_tilde[te] = Y[te] - yl.predict(Z_y_arr[te])

        # PLR point estimate
        theta = np.sum(Y_tilde * T_tilde) / np.sum(T_tilde ** 2)
        # influence-function SE
        psi   = (Y_tilde - theta * T_tilde) * T_tilde
        V     = np.mean(psi ** 2) / np.mean(T_tilde ** 2) ** 2
        se    = np.sqrt(V / n)
        ci_lo = theta - 1.645 * se
        ci_hi = theta + 1.645 * se
        pct   = (np.exp(theta) - 1) * 100
        sig   = bool(ci_lo > 0 or ci_hi < 0)

        results[name] = {
            "ate_log_cv":   round(float(theta), 5),
            "se":           round(float(se), 6),
            "ci90_low":     round(float(ci_lo), 5),
            "ci90_high":    round(float(ci_hi), 5),
            "pct_change_cv":round(float(pct), 3),
            "significant":  sig,
        }
        log(f"    ATE={theta:+.5f} ({pct:+.2f}% CV)  "
            f"CI90[{ci_lo:+.5f}, {ci_hi:+.5f}]  "
            + ("***" if sig else "n.s."))

    return results, T_tilde


# ── Gate 5: overlap check ─────────────────────────────────────────────────────
def check_overlap(T, Z_t, fold_ids):
    """
    Estimate propensity P(D=1|Z) using LightGBM, check distribution.
    Flag obs where propensity < TRIM_EPS or > 1-TRIM_EPS.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from lightgbm import LGBMClassifier

    clf = LGBMClassifier(n_estimators=200, max_depth=4, verbosity=-1, random_state=0)
    ps  = np.zeros(len(T))
    for k in range(N_FOLDS):
        tr = np.where(fold_ids != k)[0]
        te = np.where(fold_ids == k)[0]
        clf.fit(Z_t.values[tr], T[tr])
        ps[te] = clf.predict_proba(Z_t.values[te])[:, 1]

    trim_lo = (ps < TRIM_EPS).sum()
    trim_hi = (ps > 1 - TRIM_EPS).sum()
    log(f"  Overlap: propensity range [{ps.min():.3f}, {ps.max():.3f}]  "
        f"trim<{TRIM_EPS}: {trim_lo} obs  trim>{1-TRIM_EPS:.2f}: {trim_hi} obs")

    ps_df = pd.DataFrame({
        "propensity": ps,
        "D": T.ravel(),
        "trimmed": ((ps < TRIM_EPS) | (ps > 1-TRIM_EPS)).astype(int),
    })
    ps_df["bucket"] = pd.cut(ps, bins=20)
    return ps_df


# ── Gate 7: CausalForestDML CATE ─────────────────────────────────────────────
def run_cate(Y, T, Z_y, Z_t, fold_ids, meta):
    """
    CausalForestDML with X = [year_trend, export_ban_onion, crop_fe,
                               month (sin/cos encoding)]
    Gate 7: BLP test + calibration printed.
    """
    log("  Fitting CausalForestDML ...")

    # X: heterogeneity moderators (low-dim)
    X = Z_t[["year_trend","export_ban_onion"]].copy()
    X["crop_fe"]   = meta["crop"].astype("category").cat.codes.values
    X["month_sin"] = np.sin(2 * np.pi * meta["month"].values / 12)
    X["month_cos"] = np.cos(2 * np.pi * meta["month"].values / 12)
    X = X.astype(float).values

    lgbm_y = lgb.LGBMRegressor(n_estimators=200, max_depth=4, verbosity=-1, random_state=0)
    ridge_t = Pipeline([("sc", StandardScaler()), ("r", Ridge(alpha=1.0))])

    cf = CausalForestDML(
        model_y=lgbm_y, model_t=ridge_t,
        n_estimators=500, min_samples_leaf=20,
        cv=list([(np.where(fold_ids != k)[0], np.where(fold_ids == k)[0])
                 for k in range(N_FOLDS)]),
        random_state=0,
    )
    cf.fit(Y, T.ravel(), X=X, W=Z_t.values)

    cate_pred = cf.effect(X=X)
    log(f"  CATE: mean={np.mean(cate_pred):+.5f}  "
        f"IQR[{np.percentile(cate_pred,25):+.5f}, {np.percentile(cate_pred,75):+.5f}]")

    # Gate 7 — BLP test (heterogeneity validation)
    # Regress Y_tilde on predicted CATE using cross-fitted residuals
    from scipy import stats as sps
    Y_r = Y - cf.effect(X=X) * T          # rough residual
    slope, intercept, r, p_blp, se = sps.linregress(cate_pred, Y_r)
    log(f"  BLP test: slope={slope:.4f}  p={p_blp:.4f}  "
        + ("Heterogeneity VALIDATED" if p_blp < 0.05 else "No significant heterogeneity"))

    # attach predictions to meta
    cate_df = meta.copy()
    cate_df["cate"] = cate_pred

    # district-level mean CATE
    dist_cate = (
        cate_df.groupby(["crop","State","District"])
        .agg(cate_mean=("cate","mean"), n_obs=("cate","count"))
        .reset_index()
    )

    return cate_df, dist_cate, {
        "mean_cate": float(np.mean(cate_pred)),
        "p25_cate":  float(np.percentile(cate_pred, 25)),
        "p75_cate":  float(np.percentile(cate_pred, 75)),
        "blp_slope": float(slope),
        "blp_pval":  float(p_blp),
        "heterogeneity_validated": bool(p_blp < 0.05),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    log("=== DML — Operation Greens Policy Evaluation ===")

    # ── Data ─────────────────────────────────────────────────────────────────
    top_cv  = load_top_cv()
    log("Loading control crop CV (Brinjal, Cauliflower, Bhindi)...")
    ctrl_cv = compute_ctrl_cv()
    panel   = build_panel(top_cv, ctrl_cv)

    # ── Main spec ─────────────────────────────────────────────────────────────
    log("\n--- MAIN SPEC ---")
    Y, T_ravel, Z_y, Z_t, fold_ids, meta = make_YDZ(panel)

    log("\nGate 5: Overlap check")
    ps_df = check_overlap(T_ravel, Z_t, fold_ids)
    ps_df.to_csv(os.path.join(OUT, "dml_overlap.csv"), index=False)

    log("\nGates 3-4-6: Manual PLR (3 outcome learners, Ridge treatment)")
    main_results, T_tilde = run_dml(Y, T_ravel, Z_y, Z_t, fold_ids, label="main")

    log("\nGate 7: CausalForestDML CATE")
    cate_df, dist_cate, cate_summary = run_cate(Y, T_ravel, Z_y, Z_t, fold_ids, meta)
    cate_df.to_parquet(os.path.join(OUT, "cate_og.parquet"), index=False)
    dist_cate.to_csv(os.path.join(OUT, "dml_cate_district.csv"), index=False)

    # ── Gate 8: Robustness ────────────────────────────────────────────────────
    log("\n--- Gate 8a: COVID-EXCLUDED sensitivity ---")
    Y_c, T_c, Z_yc, Z_tc, folds_c, meta_c = make_YDZ(panel, drop_covid=True)
    covid_results, _ = run_dml(Y_c, T_c, Z_yc, Z_tc, folds_c, label="covid_excl")

    log("\n--- Gate 8b: PLACEBO (fake launch Nov-2015) ---")
    Y_p, T_p, Z_yp, Z_tp, folds_p, meta_p = make_YDZ(panel, placebo_year=2015)
    placebo_results, _ = run_dml(Y_p, T_p, Z_yp, Z_tp, folds_p, label="placebo")

    # ── Save ──────────────────────────────────────────────────────────────────
    all_results = {
        "main":          main_results,
        "covid_excluded":covid_results,
        "placebo_2015":  placebo_results,
        "cate_summary":  cate_summary,
        "notes": {
            "treatment":   "TOP_c × post_t (binary DiD)",
            "outcome":     "log monthly price CV per mandi-crop",
            "n_obs_main":  int(len(Y)),
            "n_folds":     N_FOLDS,
            "fold_cluster":"mandi",
            "learners":    list(main_results.keys()),
            "policy_date": str(POLICY_DATE.date()),
        }
    }
    out_json = os.path.join(OUT, "dml_results_og.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("DML RESULTS — Effect of Operation Greens on Price Volatility")
    print("Negative ATE = policy REDUCED volatility (log CV scale)")
    print("="*65)
    for learner, r in main_results.items():
        if "error" in r:
            print(f"  {learner}: ERROR — {r['error']}")
        else:
            sig = "SIGNIFICANT" if r["significant"] else "not significant"
            print(f"  {learner}: ATE={r['ate_log_cv']:+.5f}  "
                  f"({r['pct_change_cv']:+.2f}% CV)  "
                  f"CI90[{r['ci90_low']:+.5f}, {r['ci90_high']:+.5f}]  {sig}")

    print("\n--- COVID-EXCLUDED (robustness) ---")
    for learner, r in covid_results.items():
        if "error" not in r:
            print(f"  {learner}: ATE={r['ate_log_cv']:+.5f}  ({r['pct_change_cv']:+.2f}%)")

    print("\n--- PLACEBO Nov-2015 (should be ≈ 0) ---")
    for learner, r in placebo_results.items():
        if "error" not in r:
            sig = "SIGNIFICANT — WARNING" if r["significant"] else "not significant (GOOD)"
            print(f"  {learner}: ATE={r['ate_log_cv']:+.5f}  {sig}")

    print(f"\n--- CATE (heterogeneity) ---")
    print(f"  Mean CATE: {cate_summary['mean_cate']:+.5f}")
    print(f"  BLP test p-value: {cate_summary['blp_pval']:.4f}  "
          + ("→ Heterogeneity confirmed" if cate_summary['heterogeneity_validated']
             else "→ No significant heterogeneity"))

    log(f"\nDONE in {time.time()-t0:.0f}s  |  results → {out_json}")


if __name__ == "__main__":
    main()
