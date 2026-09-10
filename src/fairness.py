"""Fairness & adverse-action check (council rec #3).

Lending Club data contains no protected-class attributes (race, sex, age, national
origin), so a true fair-lending test is impossible here — that is itself a finding, and
it is stated in the memo. What we CAN do is check the recommended policy for disparate
impact across the segments we do observe (income band, region, home ownership): the
"4/5ths rule" adverse-impact ratio on approval rates, plus outcome parity (realized
default rate of the approved book by group).

A production model would also owe ECOA / Reg B adverse-action reason codes — per-decision
SHAP attributions. That is noted, not built.
"""
from __future__ import annotations

import json
import pathlib

import pandas as pd

from src.config import load
from src.db import connect

ROOT = pathlib.Path(__file__).resolve().parents[1]

_REGION = {
    "Northeast": ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"],
    "Midwest": ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "South": ["DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV", "AL", "KY", "MS",
              "TN", "AR", "LA", "OK", "TX"],
    "West": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"],
}
_STATE2REGION = {s: r for r, ss in _REGION.items() for s in ss}


def _load(con) -> pd.DataFrame:
    df = con.execute(
        """
        SELECT s.loan_id, s.pd_hat, s.fico_mid,
               f.annual_inc, f.home_ownership, f.addr_state,
               o.default_flag, o.is_terminal
        FROM mart_simulator_base s
        JOIN mart_loan_features  f USING (loan_id)
        LEFT JOIN mart_loan_outcomes o USING (loan_id)
        WHERE s.split_set = 'test'
        """
    ).fetchdf()
    df["income_band"] = pd.cut(
        df["annual_inc"], [-1, 40_000, 70_000, 120_000, 1e12],
        labels=["<40k", "40-70k", "70-120k", "120k+"],
    )
    df["region"] = df["addr_state"].map(_STATE2REGION).fillna("Other")
    return df


def _by(df: pd.DataFrame, col: str, approved: pd.Series, min_group: int = 500) -> dict:
    grp = df.assign(approved=approved).groupby(col, observed=True)
    sizes = grp.size()
    keep = sizes[sizes >= min_group].index          # ignore tiny groups when scoring the rule
    appr_rate = grp["approved"].mean()
    appr_rate_scored = appr_rate.loc[appr_rate.index.isin(keep)]
    term = df["is_terminal"] & (df["default_flag"].notna())
    dd = df[approved & term]
    obs_dr = dd.groupby(col, observed=True)["default_flag"].mean() if len(dd) else pd.Series(dtype=float)
    air = (float(appr_rate_scored.min() / appr_rate_scored.max())
           if len(appr_rate_scored) and appr_rate_scored.max() > 0 else None)
    return {
        "group_n": {str(k): int(v) for k, v in sizes.items()},
        "approval_rate": {str(k): round(float(v), 4) for k, v in appr_rate.items()},
        "adverse_impact_ratio": round(air, 3) if air is not None else None,
        "passes_4_5ths_rule": (air is not None and air >= 0.8),
        "approved_book_default_rate": {str(k): round(float(v), 4) for k, v in obs_dr.items()},
    }


def main() -> dict:
    cfg = load()
    sim = cfg["simulator"]
    con = connect(cfg["paths"]["duckdb"])
    df = _load(con)
    con.close()

    approved = (df["pd_hat"] < sim["default_pd_cutoff"]) & (df["fico_mid"] >= sim["default_min_fico"])
    out = {
        "policy": {"pd_cutoff": sim["default_pd_cutoff"], "min_fico": sim["default_min_fico"]},
        "note": ("No protected-class attributes in Lending Club data — this is a disparate-impact "
                 "proxy check, not a fair-lending test. A production model owes ECOA/Reg B "
                 "adverse-action reason codes (per-decision SHAP); not built here."),
        "overall_approval_rate": round(float(approved.mean()), 4),
        "by_income_band": _by(df, "income_band", approved),
        "by_region": _by(df, "region", approved),
        "by_home_ownership": _by(df, "home_ownership", approved),
    }
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "fairness.json").write_text(json.dumps(out, indent=2))

    for dim in ("by_income_band", "by_region", "by_home_ownership"):
        d = out[dim]
        print(f"   {dim:18s} AIR = {d['adverse_impact_ratio']}  "
              f"(4/5ths rule: {'pass' if d['passes_4_5ths_rule'] else 'FLAG'})")
    return out


if __name__ == "__main__":
    main()
