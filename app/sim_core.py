"""Pure simulator math — no Streamlit, so it can be unit-tested and reused.

Policy = (PD cut-off, minimum FICO). All money terms are put on the same horizon T
(spec §5.6): interest income and funding cost are annual rates x T; expected loss is the
lifetime EL already in the mart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def approved_metrics(
    df: pd.DataFrame,
    pd_cut: float,
    fico_min: float,
    cost_of_funds: float,
    horizon: float,
) -> dict:
    a = df[(df["pd_hat"] < pd_cut) & (df["fico_mid"] >= fico_min)]
    n_all = len(df)
    if len(a) == 0 or n_all == 0:
        return dict(approval_rate=0.0, volume=0.0, exp_default_rate=0.0, exp_loss=0.0,
                    interest_income=0.0, funding_cost=0.0, exp_profit=0.0, n=0)
    interest_income = float((a["loan_amnt"] * a["int_rate"] * horizon).sum())
    funding_cost = float((a["loan_amnt"] * cost_of_funds * horizon).sum())
    exp_loss = float(a["expected_loss"].sum())
    return dict(
        approval_rate=len(a) / n_all,
        volume=float(a["loan_amnt"].sum()),
        exp_default_rate=float(a["pd_hat"].mean()),
        exp_loss=exp_loss,
        interest_income=interest_income,
        funding_cost=funding_cost,
        exp_profit=interest_income - funding_cost - exp_loss,
        n=int(len(a)),
    )


def profit_curve(
    df: pd.DataFrame,
    fico_min: float,
    cost_of_funds: float,
    horizon: float,
    n_points: int = 60,
) -> pd.DataFrame:
    cuts = np.linspace(0.01, 0.40, n_points)
    recs = [approved_metrics(df, t, fico_min, cost_of_funds, horizon) | {"pd_cut": float(t)}
            for t in cuts]
    return pd.DataFrame(recs)


def best_policy(curve: pd.DataFrame) -> pd.Series:
    return curve.sort_values("exp_profit", ascending=False).iloc[0]


def money(x: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= div:
            return f"${x / div:,.1f}{unit}"
    return f"${x:,.0f}"
