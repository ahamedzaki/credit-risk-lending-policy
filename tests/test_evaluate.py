"""Sanity tests for the DeLong test and KS. Run: python tests/test_evaluate.py"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from src.evaluate import delong_roc_test, ks_statistic

rng = np.random.default_rng(0)
n = 4000
y = rng.integers(0, 2, n)
# a good score and a slightly noisier version of it
good = y + rng.normal(0, 0.8, n)
noisier = y + rng.normal(0, 1.4, n)
random_score = rng.normal(0, 1, n)


def test_identical_scores_not_significant():
    r = delong_roc_test(y, good, good.copy())
    assert abs(r["auc_diff"]) < 1e-9
    assert r["p_value"] > 0.99
    assert not r["significant_at_0.05"]


def test_better_vs_random_is_significant():
    r = delong_roc_test(y, good, random_score)
    assert r["auc_a"] > r["auc_b"]
    assert r["p_value"] < 1e-6
    assert r["significant_at_0.05"]


def test_small_edge_direction():
    r = delong_roc_test(y, good, noisier)
    assert r["auc_diff"] > 0            # the less noisy score ranks better
    assert 0.0 <= r["p_value"] <= 1.0


def test_ks_bounds():
    assert 0.0 <= ks_statistic(y, good) <= 1.0
    assert ks_statistic(y, random_score) < ks_statistic(y, good)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all evaluate tests passed")
