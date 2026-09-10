"""Discrimination + calibration metrics and figures (spec §5.2, §5.7)."""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)
    y = np.asarray(y_true)[order]
    pos = np.cumsum(y) / max(y.sum(), 1)
    neg = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(pos - neg)))


def decile_table(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y_true), "pd": np.asarray(y_score)})
    df["bucket"] = pd.qcut(df["pd"].rank(method="first"), k, labels=False)
    g = df.groupby("bucket").agg(n=("y", "size"), mean_pd=("pd", "mean"), obs_rate=("y", "mean"))
    g["lift"] = g["obs_rate"] / df["y"].mean()
    return g.reset_index()


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x_sorted = x[order]
    n = len(x)
    tr = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        tr[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = tr
    return out


def delong_roc_test(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict:
    """Fast DeLong test for two correlated ROC AUCs on the same sample (Sun & Xu 2014).

    Returns both AUCs, the AUC difference, the DeLong z-statistic and a two-sided p-value
    for H0: AUC_a == AUC_b. Use it so "model beats grade" is a tested claim, not a point
    estimate.
    """
    from scipy import stats

    y = np.asarray(y_true).astype(int)
    pos = np.c_[score_a, score_b][y == 1].T          # 2 x m
    neg = np.c_[score_a, score_b][y == 0].T          # 2 x n
    m, n = pos.shape[1], neg.shape[1]
    k = 2

    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    txy = np.array([_midrank(np.r_[pos[r], neg[r]]) for r in range(k)])

    aucs = txy[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2.0 * n)
    v01 = (txy[:, :m] - tx) / n
    v10 = 1.0 - (txy[:, m:] - ty) / m
    s01 = np.cov(v01)
    s10 = np.cov(v10)
    cov = s01 / m + s10 / n
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = aucs[0] - aucs[1]
    z = float(diff / np.sqrt(var_diff)) if var_diff > 0 else 0.0
    p = float(2 * stats.norm.sf(abs(z)))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "auc_diff": float(diff),
        "z": z,
        "p_value": p,
        "significant_at_0.05": p < 0.05,
    }


def summary(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "gini": float(2 * roc_auc_score(y_true, y_score) - 1),
        "ks": ks_statistic(y_true, y_score),
        "brier": float(brier_score_loss(y_true, y_score)),
        "base_rate": float(np.mean(y_true)),
        "n": int(len(y_true)),
    }


def plot_calibration(y_true: np.ndarray, y_score: np.ndarray, path: str, title: str) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1, label="perfect")
    ax.plot(mean_pred, frac_pos, "o-", color="#1f4e79", label="model")
    ax.set_xlabel("mean predicted PD")
    ax.set_ylabel("observed default rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def plot_decile_lift(dec: pd.DataFrame, path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(dec["bucket"].astype(str), dec["obs_rate"], color="#1f4e79")
    ax.plot(dec["bucket"].astype(str), dec["mean_pd"], "o-", color="#c17f00", label="predicted PD")
    ax.set_xlabel("PD decile (0 = safest)")
    ax.set_ylabel("default rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def _save(fig, path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_metrics(metrics: dict, path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
