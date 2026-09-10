"""Page 3 — Lending Strategy: the policy simulator (PD cut-off + min FICO -> P&L)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dash_data import CONFIG, data_source_note, money, page_header, simulator_base
from sim_core import approved_metrics, best_policy, profit_curve

st.set_page_config(page_title="Lending Strategy", layout="wide")

SIM = CONFIG["simulator"]
LGD = CONFIG["expected_loss"]

df, src = simulator_base()

page_header("Lending Strategy",
            "Move the levers — see approval, volume, expected loss and risk-adjusted profit")
data_source_note(src)
st.caption(
    f"Assumptions — LGD {LGD.get('lgd', 0.5)} (data-estimated), cost of funds "
    f"{SIM['cost_of_funds']:.0%}/yr, servicing {SIM['servicing_cost']:.1%}/yr, "
    f"amortisation factor {SIM['amort_factor']}, T = {SIM['horizon_years']:g} yr. "
    "Interest is haircut by (1 − PD). Directional, not P&L-grade — see Backtest & Validation."
)

with st.sidebar:
    st.header("Policy levers")
    pd_cut = st.slider("Approve if PD <", 0.02, 0.40, float(SIM["default_pd_cutoff"]), 0.005)
    fico_min = st.slider("Minimum FICO", 600, 780, int(SIM["default_min_fico"]), 5)
    cof = st.number_input("Cost of funds (annual)", 0.0, 0.2, float(SIM["cost_of_funds"]), 0.005)
    horizon = st.number_input("Horizon T (years)", 1.0, 5.0, float(SIM["horizon_years"]), 0.5)

m = approved_metrics(df, pd_cut, fico_min, cof, horizon)
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
    curve = profit_curve(df, fico_min, cof, horizon)
    st.line_chart(curve.set_index("approval_rate")[["exp_profit", "exp_loss"]])
    bp = best_policy(curve)
    st.info(f"Profit-maximising cut-off at this FICO floor: **PD < {bp['pd_cut']:.3f}** → "
            f"approval {bp['approval_rate']:.1%}, model profit {money(bp['exp_profit'])}. "
            "The Backtest page confirms this cut-off against realized outcomes.")

with right:
    st.subheader("Approved book by grade")
    a = df[(df["pd_hat"] < pd_cut) & (df["fico_mid"] >= fico_min)]
    if len(a):
        by_grade = a.groupby("lc_grade").agg(
            loans=("loan_id", "size"), volume=("loan_amnt", "sum"),
            avg_pd=("pd_hat", "mean"), exp_loss=("expected_loss", "sum"))
        st.dataframe(by_grade.style.format({"loans": "{:,}", "volume": lambda v: money(v),
                                            "avg_pd": "{:.2%}", "exp_loss": lambda v: money(v)}),
                     use_container_width=True)

st.divider()
st.subheader("Policy comparison")
rows = []
for label, cut in [("Conservative", 0.08), ("Current-equivalent", 0.15),
                   ("Profit-maximising", float(best_policy(profit_curve(df, fico_min, cof, horizon))["pd_cut"])),
                   ("Growth", 0.20)]:
    mm = approved_metrics(df, cut, fico_min, cof, horizon)
    rows.append({"Policy": label, "PD <": round(cut, 3),
                 "Approval": f"{mm['approval_rate']:.0%}", "Volume": money(mm["volume"]),
                 "Exp. default": f"{mm['exp_default_rate']:.1%}", "Exp. loss": money(mm["exp_loss"]),
                 "Exp. profit": money(mm["exp_profit"])})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
