"""Lending Policy Simulator (spec §5.6).

Two levers — PD cut-off and minimum FICO. Shows approval rate, loan volume, expected
default rate, expected loss and expected profit on the approved set, plus the
profit-vs-approval curve. Pure math lives in app/sim_core.py.

Run:  streamlit run app/simulator.py
Reads exports/simulator_base.parquet (falls back to the committed sample).
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd
import streamlit as st
import yaml

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from app.sim_core import approved_metrics, best_policy, money, profit_curve

ROOT = pathlib.Path(__file__).resolve().parents[1]
_CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
SIM = _CFG["simulator"]
LGD = _CFG["expected_loss"]["lgd"]
AMORT_FACTOR = SIM["amort_factor"]
SERVICING_COST = SIM["servicing_cost"]

st.set_page_config(page_title="Lending Policy Simulator", layout="wide")


@st.cache_data
def load_base() -> tuple[pd.DataFrame, str]:
    for name in ("simulator_base.parquet", "simulator_base_sample.parquet"):
        p = ROOT / "exports" / name
        if p.exists():
            return pd.read_parquet(p), name
    st.error("No exports/simulator_base*.parquet found. Run `python run_pipeline.py all`.")
    st.stop()


df, src = load_base()

st.title("Lending Policy Simulator")
st.caption(
    f"Source: `{src}` · {len(df):,} loans · assumptions — LGD {LGD}, "
    f"cost of funds {SIM['cost_of_funds']:.0%}, horizon T = {SIM['horizon_years']:g} yrs, "
    f"amortisation factor {AMORT_FACTOR}, servicing {SERVICING_COST:.1%}/yr. "
    f"No cash-flow discounting or prepayment."
)

with st.sidebar:
    st.header("Policy levers")
    pd_cut = st.slider("Approve if PD <", 0.01, 0.40, float(SIM["default_pd_cutoff"]), 0.005)
    fico_min = st.slider("Minimum FICO", 600, 780, int(SIM["default_min_fico"]), 5)
    st.divider()
    st.subheader("Assumptions (config.yaml)")
    cof = st.number_input("Cost of funds (annual)", 0.0, 0.20, float(SIM["cost_of_funds"]), 0.005)
    horizon = st.number_input("Horizon T (years)", 1.0, 5.0, float(SIM["horizon_years"]), 0.5)
    st.caption("Expected profit (A) = interest income − funding cost − servicing − expected loss, "
               "over T years on the approved set A. Interest and funding accrue on "
               "b · principal (b = amortisation factor); interest is also × (1 − PD).")

m = approved_metrics(df, pd_cut, fico_min, cof, horizon, AMORT_FACTOR, SERVICING_COST)

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
    curve = profit_curve(df, fico_min, cof, horizon, amort_factor=AMORT_FACTOR,
                         servicing_cost=SERVICING_COST)
    st.line_chart(curve.set_index("approval_rate")[["exp_profit", "exp_loss"]])
    bp = best_policy(curve)
    st.info(f"Profit-maximising cut-off at this FICO floor: **PD < {bp['pd_cut']:.3f}** "
            f"→ approval {bp['approval_rate']:.1%}, profit {money(bp['exp_profit'])}.")

with right:
    st.subheader("Approved book by grade")
    a = df[(df["pd_hat"] < pd_cut) & (df["fico_mid"] >= fico_min)]
    if len(a):
        by_grade = a.groupby("lc_grade").agg(
            loans=("loan_id", "size"), volume=("loan_amnt", "sum"),
            avg_pd=("pd_hat", "mean"), exp_loss=("expected_loss", "sum"),
        )
        st.dataframe(by_grade.style.format(
            {"volume": "{:,.0f}", "avg_pd": "{:.2%}", "exp_loss": "{:,.0f}"}))

if "default_flag" in df.columns and df["default_flag"].notna().any():
    a_term = a.loc[a["default_flag"].notna(), "default_flag"] if len(a) else pd.Series(dtype=float)
    if len(a_term):
        st.caption(
            f"Back-test (terminal loans only): observed default rate on the approved set = "
            f"{a_term.mean():.2%} vs predicted {m['exp_default_rate']:.2%}."
        )
