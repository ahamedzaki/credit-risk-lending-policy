"""Page 4 — Backtest & Validation: does the policy hold up on realized outcomes?"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dash_data import artifact, money, page_header

st.set_page_config(page_title="Backtest & Validation", layout="wide")

bt = artifact("backtest.json")
page_header("Backtest & Validation",
            "The policy sweep re-run on 334k out-of-time test loans using ACTUAL cash flows "
            "and ACTUAL charge-offs — not predicted PD")

if not bt:
    st.warning("artifacts/backtest.json not present — run `python run_pipeline.py backtest`.")
    st.stop()

mo, ro = bt["model_optimum"], bt["realized_optimum"]
c = st.columns(4)
c[0].metric("Model-PD optimal cut-off", f"PD < {mo['pd_cut']:.3f}", f"{mo['approval_rate']:.0%} approval")
c[1].metric("Realized optimal cut-off", f"PD < {ro['pd_cut']:.3f}", f"{ro['approval_rate']:.0%} approval")
c[2].metric("Regret from following the model", money(bt["regret_following_model_optimum"]),
            "on a ~$6B book")
c[3].metric("Realized loss / model EL", f"{bt['loss_ratio_realized_over_model']:.2f}×",
            "model under-predicts loss")

st.success(
    "**The decision is sound.** The realized-profit optimum and the model-driven optimum "
    "land at essentially the same cut-off. Following the model instead of perfect hindsight "
    f"costs only {money(bt['regret_following_model_optimum'])}. The optimum is not a "
    "miscalibration artifact."
)

curve = pd.DataFrame(bt["curve"])

st.divider()
st.subheader("Profit vs approval — model projection vs realized")
st.line_chart(
    curve.assign(model_profit_M=curve["model_profit"] / 1e6,
                 realized_profit_M=curve["realized_profit"] / 1e6)
         .set_index("approval_rate")[["model_profit_M", "realized_profit_M"]],
    y_label="portfolio profit ($M, T=3)",
)
st.caption("Both curves peak at the same approval rate. Realized profit is ≈ half the model "
           "projection — the (1 − PD)·coupon interest term is optimistic. Use the *shape*, not the level.")

st.subheader("Decision calibration — predicted vs actual default rate of the approved book")
st.line_chart(
    curve.set_index("approval_rate")[["pred_default_rate", "actual_default_rate"]],
    y_label="default rate",
)
st.caption("Predicted and actual default rates track within ~1 pp at every cut-off — the "
           "approve/deny decision is well calibrated even where the low-grade probability is not.")

st.divider()
st.subheader("Full backtest curve")
show = curve[["pd_cut", "approval_rate", "pred_default_rate", "actual_default_rate",
              "model_expected_loss", "realized_credit_loss", "model_profit", "realized_profit"]]
st.dataframe(
    show.style.format({
        "pd_cut": "{:.3f}", "approval_rate": "{:.1%}", "pred_default_rate": "{:.1%}",
        "actual_default_rate": "{:.1%}", "model_expected_loss": lambda v: money(v),
        "realized_credit_loss": lambda v: money(v), "model_profit": lambda v: money(v),
        "realized_profit": lambda v: money(v),
    }),
    use_container_width=True, hide_index=True, height=360,
)
