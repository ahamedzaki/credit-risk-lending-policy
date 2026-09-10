"""Fit the PD model, calibrate it, benchmark it against Lending Club grade, write artefacts.

Reads  : v_model_frame (features + label, terminal loans, split_set column)
         mart_loan_benchmark (grade — benchmark only)
Writes : artifacts/model.joblib, artifacts/metrics.json, reports/figures/*.png
Returns: test AUC (float) — used by the pipeline's reproducibility assertion.
"""
from __future__ import annotations

import contextlib
import pathlib
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src import evaluate, features
from src.config import load
from src.db import connect

ROOT = pathlib.Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def _quiet_blas():
    """Some OpenBLAS builds emit spurious divide/overflow RuntimeWarnings from `matmul`
    inside the LBFGS solver. They do not affect the fitted coefficients — silence them."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
        yield


def _load_frame(con) -> pd.DataFrame:
    df = con.execute("SELECT * FROM v_model_frame").fetchdf()
    df = features.engineer(df)
    features.assert_no_leakage(df.drop(columns=["default_flag"]))
    return df


def _fit_pd_model(X: pd.DataFrame, y: np.ndarray, cfg: dict) -> Pipeline:
    # NO class_weight: this is a probability-of-default model and calibration is the
    # point (spec §5.2). Class balancing inflates the raw scores and forces the
    # calibrator to undo its own distortion. Default ~15-20% is not rare enough to need it.
    base = Pipeline(
        [
            ("prep", features.build_preprocessor()),
            (
                "clf",
                LogisticRegression(
                    C=cfg["model"]["logistic_C"],
                    max_iter=2000,
                    random_state=cfg["model"]["random_state"],
                ),
            ),
        ]
    )
    method = cfg["model"]["calibration_method"]
    with _quiet_blas():
        if method == "none":
            base.fit(X, y)
            return base
        calibrated = CalibratedClassifierCV(base, method=method, cv=3)
        calibrated.fit(X, y)
    return calibrated


def _grade_benchmark(bench_train: pd.DataFrame, bench_test: pd.DataFrame,
                     y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Baseline PD = historical default rate of the loan's grade, learned on train."""
    rate_by_grade = (
        pd.DataFrame({"g": bench_train["lc_grade"].values, "y": y_train})
        .groupby("g")["y"].mean()
    )
    overall = float(np.mean(y_train))
    pd_test = bench_test["lc_grade"].map(rate_by_grade).fillna(overall).values
    return {
        "auc": float(roc_auc_score(y_test, pd_test)),
        "rate_by_grade": {k: float(v) for k, v in rate_by_grade.sort_index().items()},
    }


def _lightgbm_delta(X_train, y_train, X_test, y_test, cfg) -> dict:
    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "note": f"lightgbm not usable: {exc}"}
    pipe = Pipeline(
        [("prep", features.build_preprocessor()),
         ("clf", LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=cfg["model"]["random_state"], verbose=-1))]
    )
    pipe.fit(X_train, y_train)
    p = pipe.predict_proba(X_test)[:, 1]
    return {"available": True, "auc": float(roc_auc_score(y_test, p))}


def main() -> float:
    cfg = load()
    con = connect(cfg["paths"]["duckdb"])

    frame = _load_frame(con)
    bench = con.execute("SELECT loan_id, lc_grade FROM mart_loan_benchmark").fetchdf()
    frame = frame.merge(bench, on="loan_id", how="left")

    train = frame[frame["split_set"] == "train"].reset_index(drop=True)
    test = frame[frame["split_set"] == "test"].reset_index(drop=True)
    if len(train) == 0 or len(test) == 0:
        raise SystemExit("Empty train or test split — check config.split.oot_cutoff vs the mature window.")

    feat_cols = features.ALLOWLIST + features.DERIVED_NUMERIC
    Xtr, ytr = train[feat_cols], train["default_flag"].to_numpy()
    Xte, yte = test[feat_cols], test["default_flag"].to_numpy()

    model = _fit_pd_model(Xtr, ytr, cfg)
    with _quiet_blas():
        p_tr = model.predict_proba(Xtr)[:, 1]
        p_te = model.predict_proba(Xte)[:, 1]

    fig_dir = cfg["paths"]["figures_dir"]
    evaluate.plot_calibration(yte, p_te, f"{fig_dir}/calibration_test.png",
                              "Calibration — out-of-time test")
    dec = evaluate.decile_table(yte, p_te)
    evaluate.plot_decile_lift(dec, f"{fig_dir}/decile_lift_test.png",
                              "PD deciles — out-of-time test")

    grade_bm = _grade_benchmark(train[["lc_grade"]], test[["lc_grade"]], ytr, yte)
    lgbm = _lightgbm_delta(Xtr, ytr, Xte, yte, cfg)

    metrics = {
        "dataset": {
            "snapshot": cfg["data"]["snapshot_label"],
            "oot_cutoff": cfg["split"]["oot_cutoff"],
            "n_train": int(len(train)), "n_test": int(len(test)),
            "train_default_rate": float(ytr.mean()), "test_default_rate": float(yte.mean()),
        },
        "model_train": evaluate.summary(ytr, p_tr),
        "model_test": evaluate.summary(yte, p_te),
        "benchmark_grade_test": grade_bm,
        "delta_auc_vs_grade": float(roc_auc_score(yte, p_te) - grade_bm["auc"]),
        "lightgbm_test": lgbm,
        "decile_table_test": dec.to_dict(orient="records"),
        "config": {"calibration_method": cfg["model"]["calibration_method"],
                   "logistic_C": cfg["model"]["logistic_C"]},
    }

    joblib.dump({"model": model, "feature_columns": feat_cols}, cfg["paths"]["model"])
    evaluate.write_metrics(metrics, cfg["paths"]["metrics"])

    print(f"   test AUC        : {metrics['model_test']['auc']:.4f}")
    print(f"   grade AUC       : {grade_bm['auc']:.4f}")
    print(f"   delta AUC       : {metrics['delta_auc_vs_grade']:+.4f}")
    if lgbm.get("available"):
        print(f"   lightgbm AUC    : {lgbm['auc']:.4f}")
    print(f"   test KS / Brier : {metrics['model_test']['ks']:.4f} / {metrics['model_test']['brier']:.4f}")
    con.close()
    return metrics["model_test"]["auc"]


if __name__ == "__main__":
    main()
