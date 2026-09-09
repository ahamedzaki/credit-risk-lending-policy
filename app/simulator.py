"""Lending Policy Simulator (spec §5.6).

Two levers — PD cut-off and minimum FICO. Recomputes approval rate, loan volume,
expected default rate, expected loss and expected profit on the approved set, and
draws the profit-vs-approval curve. Every money term is on the same horizon T.

Run:  streamlit run app/simulator.py
Reads exports/simulator_base.parquet (falls back to the committed sample).
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import streamlit as st
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())["simulator"]
LGD = yaml.safe_load((ROOT / "config.yaml").read_text())["expected_loss"]["lgd"]

st.set_page_config(page_title="Lending Policy Simulator", layout="wide")


@st.cache_data
def load_base() -> tuple[pd.DataFrame, str]:
    for name in ("simulator_base.parquet", "simulator_base_sample.parquet"):
        p = ROOT / "exports" / name
        if p.exists():
            return pd.read_parquet(p), name
    st.error("No exports/simulator_base*.parquet found. Run `python run_pipeline.py all`.")
    st.stop()


def approved_metrics(df: pd.DataFrame, pd_cut: float, fico_min: float,
                     cost_of_funds: float, horizon: float) -> dict:
    a = df[(df["pd_hat"] < pd_cut) & (df["fico_mid"] >= fico_min)]
    n_all = len(df)
    if len(a) == 0:
        return dict(approval_rate=0.0, volume=0.0, exp_default_rate=0.0,
                    exp_loss=0.0, interest_income=0.0, funding_cost=0.0, exp_profit=0.0, n=0)
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
        n=len(a),
    )


def money(x: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= div:
            return f"${x / div:,.1f}{unit}"
    return f"${x:,.0f}"


df, src = load_base()

st.title("Lending Policy Simulator")
st.caption(
    f"Source: `{src}` · {len(df):,} loans · "
    f"assumptions — LGD {LGD}, cost of funds {CFG['cost_of_funds']:.0%}, "
    f"horizon T = {CFG['horizon_years']:g} yrs, simple interest, no prepayment / servicing cost."
)

with st.sidebar:
    st.header("Policy levers")
    pd_cut = st.slider("Approve if PD <", 0.01, 0.40, float(CFG["default_pd_cutoff"]), 0.005)
    fico_min = st.slider("Minimum FICO", 600, 780, int(CFG["default_min_fico"]), 5)
    st.divider()
    st.subheader("Assumptions (config.yaml)")
    cost_of_funds = st.number_input("Cost of funds (annual)", 0.0, 0.20, float(CFG["cost_of_funds"]), 0.005)
    horizon = st.number_input("Horizon T (years)", 1.0, 5.0, float(CFG["horizon_years"]), 0.5)
    st.caption("Expected profit (A) = interest income − funding cost − expected loss, "
               "all over T years on the approved set A.")

m = approved_metrics(df, pd_cut, fico_min, cost_of_funds, horizon)

c = st.columns(5)
c[0].metric("Approval rate", f"{m['approval_rate']:.1%}")
c[1].metric("Loan volume", money(m["volume"]))
c[2].metric("Expected default rate", f"{m['exp_default_rate']:.2%}")
c[3].metric("Expected loss", money(m["exp_loss"]))
c[4].metric("Expected profit", money(m["exp_profit"]))

st.divider()
left, right = st.columns([3, 2])

with left:
    st.subheader("Profit vs approval rate")
    cuts = np.linspace(0.01, 0.40, 60)
    curve = pd.DataFrame(
        [
            {
                "pd_cut": t,
                **{k: v for k, v in approved_metrics(df, t, fico_min, cost_of_funds, horizon).items()
                   if k in ("approval_rate", "exp_profit", "exp_loss")},
            }
            for t in cuts
        ]
    )
    curve = curve.set_index("approval_rate")[["exp_profit", "exp_loss"]]
    st.line_chart(curve)
    best = (
        pd.DataFrame([approved_metrics(df, t, fico_min, cost_of_funds, horizon) | {"pd_cut": t}
                      for t in cuts])
        .sort_values("exp_profit", ascending=False)
        .iloc[0]
    )
    st.info(f"Profit-maximising cut-off at this FICO floor: **PD < {best['pd_cut']:.3f}** "
            f"→ approval {best['approval_rate']:.1%}, profit {money(best['exp_profit'])}.")

with right:
    st.subheader("Approved book by grade")
    a = df[(df["pd_hat"] < pd_cut) & (df["fico_mid"] >= fico_min)]
    if len(a):
        by_grade = a.groupby("lc_grade").agg(
            loans=("loan_id", "size"), volume=("loan_amnt", "sum"),
            avg_pd=("pd_hat", "mean"), exp_loss=("expected_loss", "sum"),
        )
        st.dataframe(by_grade.style.format({"volume": "{:,.0f}", "avg_pd": "{:.2%}",
                                            "exp_loss": "{:,.0f}"}))

if "default_flag" in df.columns and df["default_flag"].notna().any():
    st.caption(
        "Back-test check (terminal loans only): observed default rate on the approved set = "
        f"{a.loc[a['default_flag'].notna(), 'default_flag'].mean():.2%} "
        f"vs predicted {m['exp_default_rate']:.2%}."
    )
