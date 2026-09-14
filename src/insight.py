"""Two real supplementary analyses added for the CREDENCE dashboard audit:

1. Global permutation feature importance of the ACTUAL trained model (model.joblib)
   on the out-of-time test split — a real answer to "what drives the model", not a
   per-loan SHAP explainer (out of scope here), but a genuine, reproducible global
   attribution computed against the shipped estimator.

2. LGD segmented by grade (same method as lgd.py — 1 minus recovery rate on charged-off
   TRAIN loans — grouped by Lending Club grade) so Expected Loss is legible as a range
   instead of one flat portfolio-wide number. This is additive: it does NOT change the
   headline LGD/EL figures already written by lgd.py / marts, which still use the
   single exposure-weighted LGD (documented limitation, now made visible instead of
   silently flat).

Reads  : artifacts/model.joblib, v_model_frame (test split), mart_loan_outcomes,
         mart_loan_benchmark (grade)
Writes : artifacts/insight.json
"""
from __future__ import annotations

import json
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from src import features
from src.config import load
from src.db import connect

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Human-readable labels for the allowlisted features (spec §2.2) — used only for display.
LABEL = {
    "fico_mid": "Credit score (FICO)",
    "dti": "Debt-to-income ratio",
    "revol_util": "Revolving utilisation",
    "loan_to_income": "Loan-to-income ratio",
    "log_annual_inc": "Annual income (log)",
    "inq_last_6mths": "Inquiries, last 6 months",
    "delinq_2yrs": "Delinquencies, last 2 years",
    "bc_util": "Bankcard utilisation",
    "num_tl_90g_dpd_24m": "Trades 90+ DPD, last 24m",
    "pct_tl_nvr_dlq": "% trades never delinquent",
    "mo_sin_old_rev_tl_op": "Age of oldest revolving trade",
    "acc_open_past_24mths": "Accounts opened, last 24m",
    "mths_since_recent_inq": "Months since last inquiry",
    "mths_since_recent_bc": "Months since newest bankcard",
    "revol_bal": "Revolving balance",
    "total_bc_limit": "Total bankcard limit",
    "num_actv_bc_tl": "Active bankcard trades",
    "bc_open_to_buy": "Unused bankcard credit",
    "num_tl_op_past_12m": "Trades opened, last 12m",
    "annual_inc": "Annual income",
    "num_accts_ever_120_pd": "Accounts ever 120+ DPD",
    "percent_bc_gt_75": "% bankcards over 75% utilised",
    "total_bal_ex_mort": "Total balance ex-mortgage",
    "credit_history_months": "Credit history length",
    "emp_length_years": "Employment length",
    "pub_rec": "Public records",
    "pub_rec_bankruptcies": "Public-record bankruptcies",
    "open_acc": "Open accounts",
    "total_acc": "Total accounts",
    "mort_acc": "Mortgage accounts",
    "loan_amnt": "Loan amount",
    "home_ownership": "Home ownership",
    "purpose": "Loan purpose",
    "verification_status": "Income verification",
    "application_type": "Application type",
}


def _feature_importance(con, cfg, sample_n: int = 15_000, n_repeats: int = 5) -> dict:
    bundle = joblib.load(cfg["paths"]["model"])
    model, feat_cols = bundle["model"], bundle["feature_columns"]

    frame = con.execute("SELECT * FROM v_model_frame WHERE split_set = 'test'").fetchdf()
    frame = features.engineer(frame)
    features.assert_no_leakage(frame.drop(columns=["default_flag"]))

    rng = np.random.default_rng(42)
    idx = rng.choice(len(frame), size=min(sample_n, len(frame)), replace=False)
    sample = frame.iloc[idx]
    X, y = sample[feat_cols], sample["default_flag"].to_numpy()

    baseline_auc = float(roc_auc_score(y, model.predict_proba(X)[:, 1]))

    result = permutation_importance(
        model, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=42, n_jobs=-1,
    )
    rows = [
        {
            "feature": col,
            "label": LABEL.get(col, col),
            "importance_mean": float(result.importances_mean[i]),
            "importance_std": float(result.importances_std[i]),
        }
        for i, col in enumerate(feat_cols)
    ]
    rows.sort(key=lambda r: r["importance_mean"], reverse=True)
    return {
        "method": "sklearn.inspection.permutation_importance, scoring=roc_auc",
        "sample_n": int(len(sample)),
        "n_repeats": n_repeats,
        "baseline_auc_on_sample": baseline_auc,
        "top": rows[:12],
        "note": (
            "Global attribution on the shipped model, not per-loan SHAP — mean AUC drop when a "
            "feature's values are shuffled, averaged over 5 repeats on a 15,000-loan sample of the "
            "out-of-time test set. Per-decision reason codes (ECOA/Reg B adverse-action) would need "
            "a real per-loan explainer; not built here."
        ),
    }


def _lgd_by_grade(con, cfg) -> dict:
    """Reads the canonical segmented-LGD estimate — same function src/lgd.py uses to build
    the `lgd_by_grade` table that 06_marts.sql joins for Expected Loss. Single source of
    truth: this module reports it, it does not independently recompute it."""
    from src import lgd as lgd_mod

    flat = lgd_mod.resolve(cfg, con)
    by_grade = lgd_mod.estimate_by_grade(con, cfg["split"]["oot_cutoff"], flat)
    return {
        "method": (
            "1 − (principal repaid + recoveries) / funded, per grade, train charged-off loans "
            f"only, Buhlmann credibility-weighted toward the flat LGD (full_credibility_n="
            f"{lgd_mod.FULL_CREDIBILITY_N})."
        ),
        "by_grade": by_grade,
        "note": (
            "This is the LGD actually used in Expected Loss (mart_loan_el joins this table by "
            "grade, falling back to the flat portfolio LGD for any grade not present)."
        ),
    }


def main() -> dict:
    cfg = load()
    con = connect(cfg["paths"]["duckdb"])
    out = {
        "feature_importance": _feature_importance(con, cfg),
        "lgd_by_grade": _lgd_by_grade(con, cfg),
    }
    con.close()
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "insight.json").write_text(json.dumps(out, indent=2))
    print("Top features by permutation importance (AUC drop):")
    for r in out["feature_importance"]["top"][:8]:
        print(f"   {r['label']:<32} {r['importance_mean']:+.4f}")
    print("LGD by grade:")
    for r in out["lgd_by_grade"]["by_grade"]:
        print(f"   {r['grade']}: {r['lgd']}")
    return out


if __name__ == "__main__":
    main()
