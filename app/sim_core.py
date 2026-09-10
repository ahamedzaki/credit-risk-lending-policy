"""Pure simulator math — no Streamlit, so it can be unit-tested and reused.

Policy = (PD cut-off, minimum FICO). Expected-value P&L on horizon T (spec §5.6):

    interest income (A) = Σ  loan_amnt · int_rate · T · b · (1 − PD)
    funding cost   (A)  = Σ  loan_amnt · r_f     · T · b
    servicing cost (A)  = Σ  loan_amnt · s       · T
    expected loss  (A)  = Σ  PD · EAD · LGD                      (the EL already in the mart)
    expected profit(A)  = interest − funding − servicing − expected loss

Two corrections keep the profit-vs-approval curve from rising forever (the naive-simulator
failure mode):
  * (1 − PD) haircut on interest — defaulters stop paying coupons.
  * amortisation factor b (~0.52) — a level-payment 36-month loan carries, on average,
    only ~half its original principal, so interest earned and funding cost both accrue on
    b · principal, not the full principal × T.
Still an approximation: no cash-flow discounting, no prepayment, no recovery lag, flat
per-loan servicing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AMORT_FACTOR = 0.52       # avg outstanding balance / original principal, 36-month level-pay
SERVICING_COST = 0.004    # annual, fraction of original principal (config.yaml is authoritative)


def approved_metrics(
    df: pd.DataFrame,
    pd_cut: float,
    fico_min: float,
    cost_of_funds: float,
    horizon: float,
    amort_factor: float = AMORT_FACTOR,
    servicing_cost: float = SERVICING_COST,
) -> dict:
    a = df[(df["pd_hat"] < pd_cut) & (df["fico_mid"] >= fico_min)]
    n_all = len(df)
    if len(a) == 0 or n_all == 0:
        return dict(approval_rate=0.0, volume=0.0, exp_default_rate=0.0, exp_loss=0.0,
                    interest_income=0.0, funding_cost=0.0, servicing_cost=0.0,
                    exp_profit=0.0, n=0)
    principal = a["loan_amnt"]
    interest_income = float((principal * a["int_rate"] * horizon * amort_factor * (1.0 - a["pd_hat"])).sum())
    funding_cost = float((principal * cost_of_funds * horizon * amort_factor).sum())
    servicing = float((principal * servicing_cost * horizon).sum())
    exp_loss = float(a["expected_loss"].sum())
    return dict(
        approval_rate=len(a) / n_all,
        volume=float(principal.sum()),
        exp_default_rate=float(a["pd_hat"].mean()),
        exp_loss=exp_loss,
        interest_income=interest_income,
        funding_cost=funding_cost,
        servicing_cost=servicing,
        exp_profit=interest_income - funding_cost - servicing - exp_loss,
        n=int(len(a)),
    )


def profit_curve(
    df: pd.DataFrame,
    fico_min: float,
    cost_of_funds: float,
    horizon: float,
    n_points: int = 60,
    amort_factor: float = AMORT_FACTOR,
    servicing_cost: float = SERVICING_COST,
) -> pd.DataFrame:
    cuts = np.linspace(0.01, 0.40, n_points)
    recs = [
        approved_metrics(df, t, fico_min, cost_of_funds, horizon, amort_factor, servicing_cost)
        | {"pd_cut": float(t)}
        for t in cuts
    ]
    return pd.DataFrame(recs)


def best_policy(curve: pd.DataFrame) -> pd.Series:
    return curve.sort_values("exp_profit", ascending=False).iloc[0]


def money(x: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= div:
            return f"${x / div:,.1f}{unit}"
    return f"${x:,.0f}"
