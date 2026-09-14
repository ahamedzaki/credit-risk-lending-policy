"""Real population-stability (PSI) check — audit finding fixed.

The dashboard's "Monitoring" page previously showed PSI as a wholly synthetic monthly
series (no production traffic exists to compute a live one against). What we DO have,
for real, is two genuinely different historical populations already built into the
pipeline: the train cohort (loans issued 2012-01 .. 2014-12) and the test cohort (issued
2015-01 .. 2016-02) — a real, if retrospective, population-shift comparison, not a live
monitoring feed. This module computes it honestly and labels it as exactly that: a
train-vs-test population stability check, not a production drift monitor.

PSI, standard credit-risk definition, deciles cut on the REFERENCE (train) population:
    PSI = sum over deciles of (pct_test - pct_train) * ln(pct_test / pct_train)
Conventional read: <0.10 no material shift, 0.10-0.25 moderate, >0.25 material shift —
these thresholds are cited as-is from standard risk-monitoring practice, not derived here.

Computes PSI for:
  - the model's own predicted-PD score (the single number a real risk team watches first)
  - a handful of the model's strongest inputs (src/insight.py's feature-importance ranking)

Reads  : v_model_frame, mart_scores
Writes : artifacts/monitor.json
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from src.config import load
from src.db import connect

ROOT = pathlib.Path(__file__).resolve().parents[1]
N_BINS = 10
# The features actually worth watching — top of src/insight.py's permutation-importance
# ranking, i.e. the inputs the model leans on most, not an arbitrary list.
WATCH_FEATURES = ["fico_mid", "acc_open_past_24mths", "loan_to_income", "dti", "annual_inc"]


def _psi(reference: pd.Series, comparison: pd.Series, n_bins: int = N_BINS) -> dict:
    ref = reference.dropna()
    cmp_ = comparison.dropna()
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:  # degenerate (near-constant) feature — PSI is meaningless
        return {"psi": None, "n_bins_effective": len(edges) - 1, "note": "too few distinct values for decile binning"}
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = pd.cut(ref, edges, include_lowest=True).value_counts(sort=False)
    cmp_counts = pd.cut(cmp_, edges, include_lowest=True).value_counts(sort=False)
    ref_pct = (ref_counts / ref_counts.sum()).clip(lower=1e-6)
    cmp_pct = (cmp_counts / cmp_counts.sum()).clip(lower=1e-6)
    psi = float(((cmp_pct - ref_pct) * np.log(cmp_pct / ref_pct)).sum())
    return {"psi": round(psi, 4), "n_bins_effective": len(edges) - 1}


def _band(psi: float | None) -> str | None:
    if psi is None:
        return None
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "moderate shift"
    return "material shift"


def main() -> dict:
    cfg = load()
    con = connect(cfg["paths"]["duckdb"])

    scores = con.execute(
        "SELECT f.loan_id, f.split_set, s.pd_hat, f.fico_mid, f.acc_open_past_24mths, "
        "       f.dti, f.annual_inc, f.loan_amnt "
        "FROM v_model_frame f JOIN mart_scores s USING (loan_id)"
    ).fetchdf()
    scores["loan_to_income"] = scores["loan_amnt"] / scores["annual_inc"].where(scores["annual_inc"] > 0)
    con.close()

    train = scores[scores["split_set"] == "train"]
    test = scores[scores["split_set"] == "test"]

    score_psi = _psi(train["pd_hat"], test["pd_hat"])
    feature_psi = {}
    for feat in WATCH_FEATURES:
        r = _psi(train[feat], test[feat])
        r["band"] = _band(r["psi"])
        feature_psi[feat] = r

    out = {
        "scope": (
            "Retrospective train-vs-test population stability — NOT a live production drift "
            "monitor (there is no scored production traffic in this project). train = loans "
            "issued 2012-01..2014-12 (306,462); test = 2015-01..2016-02 (333,721)."
        ),
        "method": "Decile PSI, deciles cut on the train (reference) population, standard formula.",
        "thresholds": {"stable": "< 0.10", "moderate shift": "0.10-0.25", "material shift": ">= 0.25"},
        "score_psi": {**score_psi, "band": _band(score_psi["psi"])},
        "feature_psi": feature_psi,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    }
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "monitor.json").write_text(json.dumps(out, indent=2))

    print(f"   score PSI (predicted PD, train vs test) : {out['score_psi']['psi']}  ({out['score_psi']['band']})")
    for feat, r in feature_psi.items():
        print(f"   {feat:<24s} PSI = {r['psi']}  ({r['band']})")
    return out


if __name__ == "__main__":
    main()
