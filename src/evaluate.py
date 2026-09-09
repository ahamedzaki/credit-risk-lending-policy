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
