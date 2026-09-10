"""Page 5 — Fairness & Adverse Action: disparate-impact proxy check of the recommended policy."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dash_data import artifact, page_header

st.set_page_config(page_title="Fairness", layout="wide")

fair = artifact("fairness.json")
page_header("Fairness & Adverse Action",
            "4/5ths-rule disparate-impact check of the recommended policy across observable proxies")

if not fair:
    st.warning("artifacts/fairness.json not present — run `python run_pipeline.py fairness`.")
    st.stop()

st.warning(fair["note"])
st.metric("Overall approval rate (recommended policy)", f"{fair['overall_approval_rate']:.1%}",
          f"PD < {fair['policy']['pd_cutoff']}, FICO ≥ {fair['policy']['min_fico']}")

_ORDER = {
    "by_income_band": ["<40k", "40-70k", "70-120k", "120k+"],
    "by_region": ["Northeast", "Midwest", "South", "West"],
    "by_home_ownership": ["RENT", "OWN", "MORTGAGE"],
}
for key, label in [("by_income_band", "Income band"), ("by_region", "US region"),
                   ("by_home_ownership", "Home ownership")]:
    d = fair[key]
    st.divider()
    verdict = "✅ passes 4/5ths rule" if d["passes_4_5ths_rule"] else "🚩 FLAGS 4/5ths rule"
    st.subheader(f"{label} — adverse-impact ratio {d['adverse_impact_ratio']}  ·  {verdict}")
    rows = []
    for g in d["approval_rate"]:
        n = d.get("group_n", {}).get(g)
        if n is not None and n < 500:            # tiny groups (e.g. home_ownership 'ANY', n=1) are noise
            continue
        rows.append({
            "group": g,
            "n": n,
            "approval rate": d["approval_rate"][g],
            "approved-book default rate": d["approved_book_default_rate"].get(g),
        })
    t = pd.DataFrame(rows)
    order = [g for g in _ORDER.get(key, []) if g in set(t["group"])]
    if order:
        t = t.set_index("group").loc[order].reset_index()
    st.dataframe(
        t.style.format({"n": "{:,.0f}", "approval rate": "{:.1%}",
                        "approved-book default rate": "{:.1%}"}),
        use_container_width=True, hide_index=True,
    )
    st.bar_chart(t.set_index("group")["approval rate"], y_label="approval rate")

st.divider()
st.info(
    "Where a group is approved less often, its **approved-book default rate is still "
    "comparable** — the model is not letting through worse risks in the lower-approval "
    "groups. The disparity is in *access*, driven by FICO and PD correlating with income "
    "and housing tenure. In production this goes to a fair-lending review, and every "
    "decline owes an ECOA / Reg B reason code (per-decision SHAP) — noted, not built."
)
