"""Fairness & adverse-action check (council rec #3).

Lending Club data contains no protected-class attributes (race, sex, age, national
origin), so a true fair-lending test is impossible here — that is itself a finding, and
it is stated in the memo. What we CAN do is check the recommended policy for disparate
impact across the segments we do observe (income band, region, home ownership): the
"4/5ths rule" adverse-impact ratio on approval rates, plus outcome parity (realized
default rate of the approved book by group).

A production model would also owe ECOA / Reg B adverse-action reason codes — per-decision
SHAP attributions. That is noted, not built (see src/explain.py for a real GLOBAL
attribution and a small real per-loan sample — still not a full production reason-code
system).

Audit finding (fixed): the AIR was previously a bare point estimate with no uncertainty
and no correction for testing three dimensions at once. Both are now handled:
  - a nonparametric bootstrap (resample approved/declined indicators within each group,
    B=2000) gives a 95% CI on the AIR;
  - a Bonferroni correction (alpha=0.05 / 3 dimensions tested) tightens the pass/fail
    bar so "FLAG" isn't just noise from testing three things and expecting one to wobble.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from src.config import load
from src.db import connect

ROOT = pathlib.Path(__file__).resolve().parents[1]
N_BOOTSTRAP = 2000
N_DIMENSIONS_TESTED = 3          # income_band, region, home_ownership
ALPHA = 0.05
ALPHA_BONFERRONI = ALPHA / N_DIMENSIONS_TESTED
RNG_SEED = 42

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


def _bootstrap_air_ci(
    df: pd.DataFrame, col: str, approved: pd.Series, keep_groups, alpha: float,
    n_boot: int = N_BOOTSTRAP, seed: int = RNG_SEED,
) -> tuple[float, float] | None:
    """Nonparametric bootstrap CI on the adverse-impact ratio: resample the approved/declined
    indicator WITHIN each group (stratified — preserves group sizes), recompute AIR, repeat
    n_boot times, take the percentile interval at the given alpha. Answers: given sampling
    noise, how confident are we the AIR is really below 0.80, not just below it by chance?"""
    sub = df[[col]].copy()
    sub["approved"] = approved.to_numpy()
    sub = sub[sub[col].isin(keep_groups)]
    if sub[col].nunique() < 2:
        return None
    rng = np.random.default_rng(seed)
    group_arrays = [g["approved"].to_numpy() for _, g in sub.groupby(col, observed=True)]
    boot = np.empty(n_boot)
    for i in range(n_boot):
        rates = np.array([
            arr[rng.integers(0, len(arr), len(arr))].mean() for arr in group_arrays
        ])
        boot[i] = rates.min() / rates.max() if rates.max() > 0 else np.nan
    boot = boot[~np.isnan(boot)]
    if len(boot) == 0:
        return None
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return lo, hi


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

    ci = _bootstrap_air_ci(df, col, approved, keep, ALPHA_BONFERRONI) if air is not None else None
    # "flagged" = we are CONFIDENT (at the Bonferroni-corrected level, accounting for testing
    # 3 dimensions at once) the true AIR sits below 0.80 — not just that the point estimate
    # does. A dimension can fail the raw 4/5ths point estimate but not be statistically
    # distinguishable from 0.80 at this sample size; that distinction is the point of the CI.
    flagged_significant = ci is not None and ci[1] < 0.8

    return {
        "group_n": {str(k): int(v) for k, v in sizes.items()},
        "approval_rate": {str(k): round(float(v), 4) for k, v in appr_rate.items()},
        "adverse_impact_ratio": round(air, 3) if air is not None else None,
        "air_ci": [round(ci[0], 3), round(ci[1], 3)] if ci is not None else None,
        "air_ci_method": (
            f"nonparametric stratified bootstrap, n={N_BOOTSTRAP}, "
            f"{100 * (1 - ALPHA_BONFERRONI):.2f}% CI (Bonferroni-corrected for "
            f"{N_DIMENSIONS_TESTED} dimensions tested)"
        ),
        "passes_4_5ths_rule": (air is not None and air >= 0.8),
        "flagged_significant": flagged_significant,
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
                 "adverse-action reason codes (per-decision SHAP); not built here. "
                 f"AIR confidence intervals are Bonferroni-corrected for testing "
                 f"{N_DIMENSIONS_TESTED} dimensions at once (alpha={ALPHA} -> "
                 f"{ALPHA_BONFERRONI:.4f} per dimension)."),
        "overall_approval_rate": round(float(approved.mean()), 4),
        "by_income_band": _by(df, "income_band", approved),
        "by_region": _by(df, "region", approved),
        "by_home_ownership": _by(df, "home_ownership", approved),
    }
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "fairness.json").write_text(json.dumps(out, indent=2))

    for dim in ("by_income_band", "by_region", "by_home_ownership"):
        d = out[dim]
        sig = "significant FLAG" if d["flagged_significant"] else ("pass" if d["passes_4_5ths_rule"] else "flag not significant at Bonferroni level")
        print(f"   {dim:18s} AIR = {d['adverse_impact_ratio']}  CI {d['air_ci']}  ({sig})")
    return out


if __name__ == "__main__":
    main()
