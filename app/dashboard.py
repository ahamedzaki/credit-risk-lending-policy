"""Credit Risk & Lending Policy — dashboard (page 1: Executive Overview).

Multi-page Streamlit app. Other pages live in app/pages/.
Run:  streamlit run app/dashboard.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dash_data import (artifact, data_source_note, money, page_header,
                           portfolio_summary, simulator_base)

st.set_page_config(page_title="Credit Risk & Lending Policy", layout="wide")

df, src = simulator_base()
metrics = artifact("metrics.json")
lgd = artifact("lgd.json")
bt = artifact("backtest.json")

page_header(
    "Executive Risk Overview",
    "Lending Club accepted loans · 36-month term · issued 2012-01 – 2016-02 · "
    f"{len(df):,} loans in view",
)
data_source_note(src)

terminal = df[df["is_terminal"]]
total_funded = df["loan_amnt"].sum()
total_el = df["expected_loss"].sum()
obs_dr = terminal["default_flag"].mean()
high_risk_expo = df.loc[df["pd_hat"] >= 0.20, "loan_amnt"].sum()
lgd_val = (lgd or {}).get("lgd_data", 0.45)

c = st.columns(5)
c[0].metric("Portfolio funded", money(total_funded))
c[1].metric("Expected loss", money(total_el), f"{total_el / total_funded:.1%} of funded")
c[2].metric("Observed default rate", f"{obs_dr:.1%}", "terminal loans")
c[3].metric("High-risk exposure", money(high_risk_expo), "PD ≥ 20%")
c[4].metric("LGD (data-estimated)", f"{lgd_val:.0%}", "from charged-off recoveries")

st.divider()
left, right = st.columns(2)

with left:
    st.subheader("Expected loss by vintage")
    v = (df.assign(year=df["vintage_year"].astype(str).str[:4])
           .groupby("year").agg(expected_loss=("expected_loss", "sum"),
                                funded=("loan_amnt", "sum")))
    v["loss_rate"] = v["expected_loss"] / v["funded"]
    st.bar_chart(v["expected_loss"], y_label="expected loss ($)")
    st.caption("Loss-rate by vintage is flat (~"
               f"{v['loss_rate'].min():.1%}–{v['loss_rate'].max():.1%}) — no vintage drift.")

with right:
    st.subheader("Risk-band distribution")
    band = pd.cut(df["pd_hat"], [0, 0.05, 0.10, 0.20, 1.0],
                  labels=["Low <5%", "Moderate 5-10%", "Elevated 10-20%", "High 20%+"])
    bd = (df.groupby(band.rename("risk band"), observed=True)
            .agg(loans=("loan_id", "size"), exposure=("loan_amnt", "sum"),
                 exp_loss=("expected_loss", "sum")))
    st.dataframe(
        bd.style.format({"loans": "{:,}", "exposure": lambda v: money(v),
                         "exp_loss": lambda v: money(v)}),
        use_container_width=True,
    )

st.divider()
st.subheader("Where risk concentrates")
ps = portfolio_summary()
if ps is not None:
    g = ps[ps["dimension"] == "grade"].copy()
    g["exposure_share"] = g["funded"] / g["funded"].sum()
    g["el_share"] = g["total_el"] / g["total_el"].sum()
    g["el_to_exposure"] = (g["el_share"] / g["exposure_share"]).round(2)
    st.dataframe(
        g[["bucket", "n_loans", "funded", "avg_pd", "total_el",
           "exposure_share", "el_share", "el_to_exposure"]]
        .rename(columns={"bucket": "grade"})
        .style.format({"n_loans": "{:,}", "funded": lambda v: money(v),
                       "avg_pd": "{:.1%}", "total_el": lambda v: money(v),
                       "exposure_share": "{:.1%}", "el_share": "{:.1%}"}),
        use_container_width=True,
    )
    st.caption("`el_to_exposure` > 1 means the grade carries more loss than its share of "
               "the book. Grades C–D are the concentration.")
else:
    st.info("portfolio_summary.parquet not present — grade concentration hidden.")

if metrics:
    st.divider()
    mt = metrics["model_test"]
    dl = metrics["delong_primary_vs_grade"]
    st.subheader("Model at a glance (out-of-time test)")
    m = st.columns(4)
    m[0].metric("AUC", f"{mt['auc']:.3f}", f"vs grade {metrics['benchmark_grade_test']['auc']:.3f}")
    m[1].metric("Gini", f"{mt['gini']:.3f}")
    m[2].metric("KS", f"{mt['ks']:.3f}")
    m[3].metric("Brier", f"{mt['brier']:.3f}")
    st.caption(f"Beats Lending Club grade by {dl['auc_diff']:+.3f} AUC — DeLong z={dl['z']:.1f}, "
               f"p={dl['p_value']:.1e}. Detail on the **Credit Risk** page.")

st.divider()
st.caption("Pages: Executive Overview · Credit Risk · Lending Strategy · Backtest & Validation · Fairness")
