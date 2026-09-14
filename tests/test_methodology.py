"""Tests for the audit-driven fixes: credibility-weighted LGD, PSI, and the fairness
bootstrap CI. No DB dependency — these exercise the pure functions directly on small
synthetic frames, matching the existing lightweight test style. Run: python tests/test_methodology.py"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from src.fairness import _bootstrap_air_ci, _by
from src.monitor import _psi


def test_lgd_credibility_shrinks_thin_segments_to_flat():
    # Reimplements the exact formula in src/lgd.py::estimate_by_grade — kept independent
    # of the DB-backed function so this test needs no fixture data.
    FULL_CREDIBILITY_N = 2000

    def credible(n, lgd_raw, lgd_flat):
        credibility = n / (n + FULL_CREDIBILITY_N)
        return credibility * lgd_raw + (1 - credibility) * lgd_flat

    lgd_flat = 0.50
    # a grade with almost no charged-off history should land close to the flat portfolio LGD,
    # even if its own raw estimate is wildly different
    thin = credible(n=5, lgd_raw=0.90, lgd_flat=lgd_flat)
    assert abs(thin - lgd_flat) < 0.02, f"thin segment should shrink to ~flat, got {thin}"
    # a grade with abundant history should land close to its OWN raw estimate
    thick = credible(n=50_000, lgd_raw=0.90, lgd_flat=lgd_flat)
    assert abs(thick - 0.90) < 0.02, f"thick segment should trust its own data, got {thick}"


def test_psi_identical_distributions_near_zero():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0, 1, 5000))
    y = pd.Series(rng.normal(0, 1, 5000))  # same distribution, different draw
    result = _psi(x, y)
    assert result["psi"] is not None
    assert result["psi"] < 0.02, f"same-distribution PSI should be ~0, got {result['psi']}"


def test_psi_shifted_distribution_is_large():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0, 1, 5000))
    y = pd.Series(rng.normal(3, 1, 5000))  # a large, genuine population shift
    result = _psi(x, y)
    assert result["psi"] > 0.25, f"a 3-sigma population shift should read as material, got {result['psi']}"


def test_psi_degenerate_feature_does_not_crash():
    x = pd.Series([5.0] * 100)  # constant — no real distribution to bin
    y = pd.Series([5.0] * 100)
    result = _psi(x, y)
    assert result["psi"] is None  # can't be legitimately computed; must say so, not fabricate a number


def _synthetic_fairness_frame(n_per_group=2000, gap=True):
    """Two groups with a real, deliberate approval-rate gap (or none, if gap=False)."""
    rng = np.random.default_rng(1)
    rows = []
    for group, base_rate in [("low", 0.30 if gap else 0.60), ("high", 0.60)]:
        approved = rng.random(n_per_group) < base_rate
        rows.append(pd.DataFrame({
            "seg": group,
            "default_flag": rng.integers(0, 2, n_per_group).astype(float),
            "is_terminal": True,
        }).assign(_approved=approved))
    df = pd.concat(rows, ignore_index=True)
    approved = df.pop("_approved")
    return df, approved


def test_fairness_ci_flags_a_real_gap():
    df, approved = _synthetic_fairness_frame(gap=True)
    result = _by(df, "seg", approved, min_group=100)
    assert result["adverse_impact_ratio"] < 0.8
    assert result["air_ci"] is not None
    lo, hi = result["air_ci"]
    assert hi < 0.8, "a real, large gap (0.30 vs 0.60 approval) should stay below 0.8 even at the CI's upper bound"
    assert result["flagged_significant"] is True


def test_fairness_ci_does_not_flag_when_there_is_no_gap():
    df, approved = _synthetic_fairness_frame(gap=False)
    result = _by(df, "seg", approved, min_group=100)
    assert result["adverse_impact_ratio"] > 0.85
    assert result["flagged_significant"] is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all methodology tests passed")
