"""Page 2 — Credit Risk: PD distribution, calibration, discrimination, benchmark, segments."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dash_data import artifact, data_source_note, money, page_header, simulator_base

st.set_page_config(page_title="Credit Risk", layout="wide")

df, src = simulator_base()
metrics = artifact("metrics.json")

page_header("Credit Risk", "Probability of default — distribution, calibration, and how it beats Lending Club grade")
data_source_note(src)

st.subheader("PD distribution")
hist = np.histogram(df["pd_hat"], bins=40, range=(0, 0.6))
st.bar_chart(pd.Series(hist[0], index=np.round(hist[1][:-1], 3)), y_label="loans")

if metrics:
    mt, mtr = metrics["model_test"], metrics["model_train"]
    cmp = metrics["comparison_test_auc"]
    dl = metrics["delong_primary_vs_grade"]

    st.divider()
    st.subheader("Discrimination & calibration (out-of-time test)")
    c = st.columns(4)
    c[0].metric("AUC", f"{mt['auc']:.3f}", f"train {mtr['auc']:.3f}")
    c[1].metric("Gini", f"{mt['gini']:.3f}")
    c[2].metric("KS", f"{mt['ks']:.3f}")
    c[3].metric("Brier", f"{mt['brier']:.3f}", f"base rate {mt['base_rate']:.3f}")

    st.divider()
    st.subheader("Benchmark — model vs Lending Club grade")
    b = pd.DataFrame(
        {"model": ["HGB (primary)", "Logistic", "Grade only"],
         "test AUC": [cmp.get("hgb"), cmp.get("logistic"), cmp["grade_only"]]}
    ).dropna()
    st.dataframe(b.style.format({"test AUC": "{:.4f}"}), use_container_width=True, hide_index=True)
    st.success(
        f"HGB beats grade by **{dl['auc_diff']:+.4f} AUC** — DeLong z = **{dl['z']:.1f}**, "
        f"p = **{dl['p_value']:.1e}** on {mt['n']:,} test loans. Not noise."
    )

    st.divider()
    st.subheader("Calibration by PD decile (out-of-time test)")
    dec = pd.DataFrame(metrics["decile_table_test"])
    show = dec[["bucket", "n", "mean_pd", "obs_rate", "lift"]].rename(
        columns={"bucket": "decile", "mean_pd": "predicted PD", "obs_rate": "observed default rate"})
    st.dataframe(show.style.format({"n": "{:,}", "predicted PD": "{:.3f}",
                                    "observed default rate": "{:.3f}", "lift": "{:.2f}"}),
                 use_container_width=True, hide_index=True)
    st.bar_chart(dec.set_index("bucket")[["mean_pd", "obs_rate"]],
                 y_label="default rate")
    st.caption("Predicted PD tracks the observed rate within ~1–2 pp across the book. "
               "The gap widens in the top decile — the low-grade tail is under-predicted "
               "(documented limitation).")

st.divider()
st.subheader("Risk & expected loss by grade")
g = df.groupby("lc_grade").agg(
    loans=("loan_id", "size"), exposure=("loan_amnt", "sum"),
    avg_pd=("pd_hat", "mean"), exp_loss=("expected_loss", "sum"),
    obs_dr=("default_flag", "mean"),
)
st.dataframe(
    g.style.format({"loans": "{:,}", "exposure": lambda v: money(v), "avg_pd": "{:.1%}",
                    "exp_loss": lambda v: money(v), "obs_dr": "{:.1%}"}),
    use_container_width=True,
)

st.subheader("Expected loss by loan purpose")
p = (df.groupby("purpose").agg(exposure=("loan_amnt", "sum"), exp_loss=("expected_loss", "sum"))
       .sort_values("exp_loss", ascending=False).head(10))
st.bar_chart(p["exp_loss"], y_label="expected loss ($)")
