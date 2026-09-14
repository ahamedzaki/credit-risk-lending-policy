"""Real per-loan attribution on the trained model (audit finding: the dashboard's
"why is this borrower risky?" panel runs on a synthetic borrower set and a hand-written
scoring function, not the real model — labelled DEMO, correctly, but that means no real
per-loan explanation exists anywhere in the project).

This computes one for a small real sample, honestly labelled for what it actually is:
single-feature marginal-contribution attribution, NOT an exact Shapley-value (SHAP)
decomposition. shap.Explainer's default tabular masker assumes numeric data and breaks
on this project's mixed categorical/numeric feature set; rather than force-fit that, this
implements the simpler, fully-inspectable method directly against the real model:

    baseline_pd  = model.predict_proba(loan)                       [loan's actual PD]
    typical_pd_f = model.predict_proba(loan with feature f reset to the population's
                                        typical value — median for numeric, mode for
                                        categorical, computed on the test population)
    contribution_f = baseline_pd - typical_pd_f

Positive contribution_f: this loan's actual value of f is pushing its PD UP relative to a
typical applicant. This is a real, correct, single-order sensitivity on the actual trained
model — not a combinatorial Shapley average over feature orderings, and it says so.

Reads  : artifacts/model.joblib, v_model_frame (test split)
Writes : artifacts/explanations.json — a stratified sample (3 per risk band) of REAL
         test-set loans with real per-loan attribution.
"""
from __future__ import annotations

import json
import pathlib

import joblib
import numpy as np
import pandas as pd

from src import features
from src.config import load
from src.db import connect

ROOT = pathlib.Path(__file__).resolve().parents[1]
N_PER_BAND = 3
BAND_EDGES = [0.0, 0.05, 0.10, 0.20, 1.0]
BAND_LABELS = ["Low", "Medium", "High", "Critical"]

LABEL = {
    "fico_mid": "Credit score (FICO)", "dti": "Debt-to-income ratio",
    "revol_util": "Revolving utilisation", "loan_to_income": "Loan-to-income ratio",
    "log_annual_inc": "Annual income (log)", "annual_inc": "Annual income",
    "inq_last_6mths": "Inquiries, last 6 months", "delinq_2yrs": "Delinquencies, last 2 years",
    "bc_util": "Bankcard utilisation", "num_tl_90g_dpd_24m": "Trades 90+ DPD, last 24m",
    "pct_tl_nvr_dlq": "% trades never delinquent", "mo_sin_old_rev_tl_op": "Age of oldest revolving trade",
    "acc_open_past_24mths": "Accounts opened, last 24m", "mths_since_recent_inq": "Months since last inquiry",
    "mths_since_recent_bc": "Months since newest bankcard", "num_accts_ever_120_pd": "Accounts ever 120+ DPD",
    "percent_bc_gt_75": "% bankcards over 75% utilised", "total_bal_ex_mort": "Total balance ex-mortgage",
    "credit_history_months": "Credit history length", "emp_length_years": "Employment length",
    "pub_rec": "Public records", "pub_rec_bankruptcies": "Public-record bankruptcies",
    "open_acc": "Open accounts", "total_acc": "Total accounts", "mort_acc": "Mortgage accounts",
    "loan_amnt": "Loan amount", "home_ownership": "Home ownership", "purpose": "Loan purpose",
    "verification_status": "Income verification", "application_type": "Application type",
    "num_tl_op_past_12m": "Trades opened, last 12m",
    "avg_cur_bal": "Average balance per account", "bc_open_to_buy": "Unused bankcard credit",
    "num_actv_bc_tl": "Active bankcard trades", "tot_hi_cred_lim": "Total high credit limit",
    "total_bc_limit": "Total bankcard limit", "tot_cur_bal": "Total current balance",
    "tot_coll_amt": "Total in collections", "revol_bal": "Revolving balance",
}


def _typical_values(pop: pd.DataFrame, feat_cols: list[str]) -> dict:
    typical = {}
    for col in feat_cols:
        s = pop[col]
        typical[col] = s.mode(dropna=True).iloc[0] if s.dtype == object else float(s.median())
    return typical


def _explain_loan(model, feat_cols: list[str], row_df: pd.DataFrame, typical: dict, top_n: int = 6) -> dict:
    # row_df: a 1-row slice of the ORIGINAL multi-row frame (df.loc[[idx], feat_cols]) — keeps
    # each column's real dtype intact. A Series -> to_frame().T round trip silently corrupts
    # NaN handling (pd.NA vs np.nan) for mixed numeric/categorical rows and breaks sklearn's
    # array validation; replicating the row instead avoids that entirely.
    baseline_pd = float(model.predict_proba(row_df)[:, 1][0])
    batch = pd.concat([row_df] * len(feat_cols), ignore_index=True)
    for i, col in enumerate(feat_cols):
        batch.at[i, col] = typical[col]
    typical_pd = model.predict_proba(batch)[:, 1]
    contributions = [
        {"feature": col, "label": LABEL.get(col, col), "contribution_pp": round(float((baseline_pd - typical_pd[i]) * 100), 2)}
        for i, col in enumerate(feat_cols)
    ]
    contributions.sort(key=lambda c: abs(c["contribution_pp"]), reverse=True)
    return {"pd": round(baseline_pd, 4), "top_drivers": contributions[:top_n]}


def main(n_per_band: int = N_PER_BAND, seed: int = 20240611) -> dict:
    cfg = load()
    bundle = joblib.load(cfg["paths"]["model"])
    model, feat_cols = bundle["model"], bundle["feature_columns"]

    con = connect(cfg["paths"]["duckdb"])
    df = con.execute(
        "SELECT * FROM v_model_frame WHERE split_set = 'test'"
    ).fetchdf()
    df = features.engineer(df)
    features.assert_no_leakage(df.drop(columns=["default_flag"]))
    con.close()

    df["pd_hat"] = model.predict_proba(df[feat_cols])[:, 1]
    df["risk_band"] = pd.cut(df["pd_hat"], BAND_EDGES, labels=BAND_LABELS)
    typical = _typical_values(df, feat_cols)

    rng = np.random.default_rng(seed)
    sample_idx = []
    for band in BAND_LABELS:
        band_idx = df.index[df["risk_band"] == band].to_numpy()
        if len(band_idx) == 0:
            continue
        take = min(n_per_band, len(band_idx))
        sample_idx.extend(rng.choice(band_idx, size=take, replace=False).tolist())

    loans = []
    for idx in sample_idx:
        row = df.loc[idx]
        row_df = df.loc[[idx], feat_cols].reset_index(drop=True)
        expl = _explain_loan(model, feat_cols, row_df, typical)
        loans.append({
            "loan_id": int(row["loan_id"]),
            "risk_band": str(row["risk_band"]),
            "fico_mid": float(row["fico_mid"]),
            "dti": float(row["dti"]),
            "purpose": str(row["purpose"]),
            "loan_amnt": float(row["loan_amnt"]),
            **expl,
        })

    out = {
        "method": (
            "Single-feature marginal-contribution attribution on the ACTUAL trained model "
            "(model.joblib), not a synthetic scoring function. For each loan, every feature "
            "is reset one at a time to the test population's typical value (median for "
            "numeric, mode for categorical) and the PD change is recorded. This is a real, "
            "correct sensitivity on the real model — it is NOT an exact Shapley-value (SHAP) "
            "decomposition (no combinatorial averaging over feature orderings); labelled "
            "accordingly, not oversold as SHAP."
        ),
        "sample": f"{n_per_band} real test-set loans per risk band, seed={seed}",
        "loans": loans,
    }
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "explanations.json").write_text(json.dumps(out, indent=2))
    print(f"   explained {len(loans)} real test-set loans across {len(BAND_LABELS)} risk bands")
    for loan in loans[:3]:
        top = loan["top_drivers"][0]
        print(f"   loan {loan['loan_id']}  PD={loan['pd']:.3f}  top driver: {top['label']} ({top['contribution_pp']:+.1f}pp)")
    return out


if __name__ == "__main__":
    main()
