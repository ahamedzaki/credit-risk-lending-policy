"""Sanity tests for the simulator math. Run: python -m pytest -q  (or python tests/test_sim_core.py)"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from app.sim_core import AMORT_FACTOR, SERVICING_COST, approved_metrics, best_policy, money, profit_curve

DF = pd.DataFrame(
    {
        "loan_id": range(6),
        "loan_amnt": [10_000.0] * 6,
        "fico_mid": [640, 660, 680, 700, 720, 740],
        "int_rate": [0.15] * 6,
        "pd_hat": [0.30, 0.20, 0.12, 0.08, 0.04, 0.02],
        "ead": [10_000.0] * 6,
        "expected_loss": [1350, 900, 540, 360, 180, 90],
        "lc_grade": list("EDCBAA"),
        "default_flag": [1, 1, 0, 0, 0, 0],
    }
)


def test_thresholds_filter():
    m = approved_metrics(DF, pd_cut=0.10, fico_min=660, cost_of_funds=0.04, horizon=3.0)
    assert m["n"] == 3                                  # pd 0.08, 0.04, 0.02 and fico >= 660
    assert m["approval_rate"] == 0.5
    assert m["volume"] == 30_000.0


def test_profit_identity_and_components():
    m = approved_metrics(DF, 0.40, 600, cost_of_funds=0.04, horizon=3.0)
    # all six approved
    b, s, T = AMORT_FACTOR, SERVICING_COST, 3.0
    surv = sum(1 - p for p in DF["pd_hat"])             # 5.24
    exp_interest = 10_000 * 0.15 * T * b * surv
    exp_funding = 6 * 10_000 * 0.04 * T * b
    exp_serv = 6 * 10_000 * s * T
    assert abs(m["interest_income"] - exp_interest) < 1e-6
    assert abs(m["funding_cost"] - exp_funding) < 1e-6
    assert abs(m["servicing_cost"] - exp_serv) < 1e-6
    assert abs(m["exp_profit"]
               - (m["interest_income"] - m["funding_cost"] - m["servicing_cost"] - m["exp_loss"])) < 1e-6


def test_curve_has_interior_optimum():
    # amortisation factor + (1 - PD) haircut + servicing must make the curve turn over:
    # the profit-maximising cut-off is NOT the loosest one.
    c = profit_curve(DF, fico_min=600, cost_of_funds=0.04, horizon=3.0, n_points=40)
    assert c["exp_profit"].idxmax() < len(c) - 1
    assert c["exp_profit"].iloc[-1] < c["exp_profit"].max()


def test_empty_approved_set():
    m = approved_metrics(DF, pd_cut=0.001, fico_min=600, cost_of_funds=0.04, horizon=3.0)
    assert m == dict(approval_rate=0.0, volume=0.0, exp_default_rate=0.0, exp_loss=0.0,
                     interest_income=0.0, funding_cost=0.0, servicing_cost=0.0,
                     exp_profit=0.0, n=0)


def test_curve_and_best():
    c = profit_curve(DF, fico_min=600, cost_of_funds=0.04, horizon=3.0, n_points=20)
    assert len(c) == 20 and {"approval_rate", "exp_profit", "pd_cut"} <= set(c.columns)
    assert c["approval_rate"].is_monotonic_increasing
    bp = best_policy(c)
    assert bp["exp_profit"] == c["exp_profit"].max()


def test_money():
    assert money(1_500_000) == "$1.5M"
    assert money(2_400) == "$2.4k"
    assert money(90) == "$90"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all simulator-core tests passed")
