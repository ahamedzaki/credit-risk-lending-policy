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
from sklearn.ensemble import HistGradientBoostingClassifier
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


def _estimator(kind: str, cfg: dict) -> Pipeline:
    """Unfitted preprocessing + classifier pipeline for the requested model type.

    No class_weight anywhere: this is a probability-of-default model and calibration is
    the point (spec §5.2). A ~13-15% positive rate is not rare enough to need balancing,
    and balancing forces the calibrator to undo its own distortion.
    """
    if kind == "logistic":
        clf = LogisticRegression(
            C=cfg["model"]["logistic_C"], max_iter=2000,
            random_state=cfg["model"]["random_state"],
        )
    elif kind == "hgb":
        clf = HistGradientBoostingClassifier(
            max_iter=cfg["model"]["hgb_max_iter"],
            learning_rate=cfg["model"]["hgb_learning_rate"],
            max_leaf_nodes=cfg["model"]["hgb_max_leaf_nodes"],
            early_stopping=True, validation_fraction=0.1,
            random_state=cfg["model"]["random_state"],
        )
    else:
        raise SystemExit(f"config.model.type must be 'hgb' or 'logistic', got {kind!r}")
    return Pipeline([("prep", features.build_preprocessor()), ("clf", clf)])


def _fit_calibrated(kind: str, X: pd.DataFrame, y: np.ndarray, cfg: dict):
    base = _estimator(kind, cfg)
    method = cfg["model"]["calibration_method"]
    with _quiet_blas():
        if method == "none":
            return base.fit(X, y)
        model = CalibratedClassifierCV(base, method=method, cv=3)
        model.fit(X, y)
    return model


def _auc_of(kind: str, Xtr, ytr, Xte, yte, cfg) -> tuple[float, object]:
    m = _fit_calibrated(kind, Xtr, ytr, cfg)
    with _quiet_blas():
        p = m.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, p)), m


def _grade_benchmark(bench_train: pd.DataFrame, bench_test: pd.DataFrame,
                     y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Baseline PD = historical default rate of the loan's grade, learned on train."""
    rate_by_grade = (
        pd.DataFrame({"g": bench_train["lc_grade"].values, "y": y_train})
        .groupby("g")["y"].mean()
    )
    overall = float(np.mean(y_train))
    pd_test = bench_test["lc_grade"].map(rate_by_grade).fillna(overall).to_numpy(dtype=float)
    return {
        "auc": float(roc_auc_score(y_test, pd_test)),
        "rate_by_grade": {k: float(v) for k, v in rate_by_grade.sort_index().items()},
        "pd_test": pd_test,
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

    primary_kind = cfg["model"]["type"]
    secondary_kind = "logistic" if primary_kind == "hgb" else "hgb"

    model = _fit_calibrated(primary_kind, Xtr, ytr, cfg)
    with _quiet_blas():
        p_tr = model.predict_proba(Xtr)[:, 1]
        p_te = model.predict_proba(Xte)[:, 1]

    # three-way comparison (spec §2.4): primary vs the other model type vs grade-alone
    secondary_auc, _ = _auc_of(secondary_kind, Xtr, ytr, Xte, yte, cfg)
    grade_bm = _grade_benchmark(train[["lc_grade"]], test[["lc_grade"]], ytr, yte)
    lgbm = _lightgbm_delta(Xtr, ytr, Xte, yte, cfg)

    fig_dir = cfg["paths"]["figures_dir"]
    evaluate.plot_calibration(yte, p_te, f"{fig_dir}/calibration_test.png",
                              "Calibration — out-of-time test")
    dec = evaluate.decile_table(yte, p_te)
    evaluate.plot_decile_lift(dec, f"{fig_dir}/decile_lift_test.png",
                              "PD deciles — out-of-time test")

    primary_auc = float(roc_auc_score(yte, p_te))
    delong = evaluate.delong_roc_test(yte, p_te, grade_bm["pd_test"])
    grade_bm = {k: v for k, v in grade_bm.items() if k != "pd_test"}  # don't serialise the vector
    metrics = {
        "dataset": {
            "snapshot": cfg["data"]["snapshot_label"],
            "oot_cutoff": cfg["split"]["oot_cutoff"],
            "n_train": int(len(train)), "n_test": int(len(test)),
            "train_default_rate": float(ytr.mean()), "test_default_rate": float(yte.mean()),
        },
        "primary_model": primary_kind,
        "model_train": evaluate.summary(ytr, p_tr),
        "model_test": evaluate.summary(yte, p_te),
        "comparison_test_auc": {
            primary_kind: primary_auc,
            secondary_kind: secondary_auc,
            "grade_only": grade_bm["auc"],
        },
        "benchmark_grade_test": grade_bm,
        "delta_auc_vs_grade": primary_auc - grade_bm["auc"],
        "delong_primary_vs_grade": delong,
        "lightgbm_test": lgbm,
        "decile_table_test": dec.to_dict(orient="records"),
        "config": {"type": primary_kind,
                   "calibration_method": cfg["model"]["calibration_method"],
                   "logistic_C": cfg["model"]["logistic_C"]},
    }

    joblib.dump({"model": model, "feature_columns": feat_cols}, cfg["paths"]["model"])
    evaluate.write_metrics(metrics, cfg["paths"]["metrics"])

    print(f"   primary ({primary_kind}) AUC : {primary_auc:.4f}")
    print(f"   {secondary_kind:<14} AUC : {secondary_auc:.4f}")
    print(f"   grade-only     AUC : {grade_bm['auc']:.4f}")
    print(f"   delta vs grade     : {metrics['delta_auc_vs_grade']:+.4f}  "
          f"(DeLong z={delong['z']:.1f}, p={delong['p_value']:.2e})")
    if lgbm.get("available"):
        print(f"   lightgbm       AUC : {lgbm['auc']:.4f}")
    print(f"   test KS / Brier    : {metrics['model_test']['ks']:.4f} / {metrics['model_test']['brier']:.4f}")
    con.close()
    return primary_auc


if __name__ == "__main__":
    main()
