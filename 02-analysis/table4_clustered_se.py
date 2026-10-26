#!/usr/bin/env python
"""
table4_clustered_se.py — mandi-clustered inference for every Table 4 row.

dml_operation_greens.py reports influence-function SEs that treat each
mandi-crop-month as independent, while revision_analysis.py reports
mandi-clustered SEs. Table 4 mixed the two. This script re-runs the main,
COVID-excluded, and placebo specifications for all three outcome learners with
the original code path (same features, folds, learners, seeds) and reports both
SE types side by side, so the table can use one consistent standard.

Writes table4_clustered_se.json. Existing result files are not modified.

Run: python table4_clustered_se.py
"""

import json
import os
import time

import numpy as np
from scipy import stats as sps

from dml_operation_greens import (
    N_FOLDS, OUT, build_panel, compute_ctrl_cv, load_top_cv, log,
    make_outcome_learners, make_treatment_learner, make_YDZ,
)

Z95 = sps.norm.ppf(0.975)


def inference(Y_tilde, T_tilde, clusters):
    """Unclustered (as in dml_operation_greens.py) and mandi-clustered
    (same sandwich as revision_analysis.plr_estimate) SEs for the PLR score."""
    n = len(Y_tilde)
    theta = np.sum(Y_tilde * T_tilde) / np.sum(T_tilde ** 2)
    psi = (Y_tilde - theta * T_tilde) * T_tilde
    se_iid = np.sqrt(np.mean(psi ** 2) / np.mean(T_tilde ** 2) ** 2 / n)
    _, codes = np.unique(clusters, return_inverse=True)
    score_cl = np.bincount(codes, weights=psi)
    g = len(score_cl)
    se_cl = np.sqrt((g / (g - 1)) * np.sum(score_cl ** 2) / np.sum(T_tilde ** 2) ** 2)

    def row(se):
        lo, hi = theta - Z95 * se, theta + Z95 * se
        return {"se": round(float(se), 6), "ci95_lo": round(float(lo), 5),
                "ci95_hi": round(float(hi), 5), "significant_95": bool(lo > 0 or hi < 0)}

    return {"ate_log_cv": round(float(theta), 5),
            "pct_change_cv": round(float((np.exp(theta) - 1) * 100), 3),
            "n_obs": int(n), "n_mandis": int(g),
            "unclustered": row(se_iid), "mandi_clustered": row(se_cl),
            "se_ratio_cl_over_iid": round(float(se_cl / se_iid), 3)}


def run_spec(panel, label, **kw):
    log(f"\n--- {label} ---")
    Y, T, Z_y, Z_t, folds, meta = make_YDZ(panel, **kw)
    clusters = meta["mkey"].values
    Zy, Zt = Z_y.values, Z_t.values
    T_tilde = np.zeros(len(Y))
    tl = make_treatment_learner()
    for k in range(N_FOLDS):
        tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
        tl.fit(Zt[tr], T[tr])
        T_tilde[te] = T[te] - tl.predict(Zt[te])
    out = {}
    for name, yl in make_outcome_learners().items():
        Y_tilde = np.zeros(len(Y))
        for k in range(N_FOLDS):
            tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
            yl.fit(Zy[tr], Y[tr])
            Y_tilde[te] = Y[te] - yl.predict(Zy[te])
        out[name] = r = inference(Y_tilde, T_tilde, clusters)
        c = r["mandi_clustered"]
        log(f"  {name:12s} ATE={r['ate_log_cv']:+.5f}  clustered 95% CI "
            f"[{c['ci95_lo']:+.5f}, {c['ci95_hi']:+.5f}]  sig={c['significant_95']}  "
            f"SE ratio={r['se_ratio_cl_over_iid']}")
    return out


def main():
    t0 = time.time()
    panel = build_panel(load_top_cv(), compute_ctrl_cv())
    results = {
        "main": run_spec(panel, "MAIN"),
        "covid_excluded": run_spec(panel, "COVID-EXCLUDED", drop_covid=True),
        "placebo_2015": run_spec(panel, "PLACEBO Nov-2015", placebo_year=2015),
    }
    ref = json.load(open(os.path.join(OUT, "dml_results_og.json")))
    keymap = {"main": "main", "covid_excluded": "covid_excluded", "placebo_2015": "placebo_2015"}
    check = {s: {l: round(results[s][l]["ate_log_cv"] - ref[keymap[s]][l]["ate_log_cv"], 5)
                 for l in results[s]} for s in results if keymap[s] in ref}
    results["point_estimate_diff_vs_dml_results_og"] = check
    results["runtime_min"] = round((time.time() - t0) / 60, 1)
    path = os.path.join(OUT, "table4_clustered_se.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nPoint-estimate differences vs dml_results_og.json: {check}")
    log(f"Saved {path} ({results['runtime_min']} min)")


if __name__ == "__main__":
    main()
