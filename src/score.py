"""Batch-score every loan in the mature window and write mart_scores.

Scores ALL loans in mart_loan_features (terminal or not) so the portfolio view and the
simulator cover the whole book. Expected Loss is then built in sql/06_marts.sql.
"""
from __future__ import annotations

import pathlib

import joblib

from src import features
from src.config import load
from src.db import connect

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    cfg = load()
    bundle = joblib.load(cfg["paths"]["model"])
    model, feat_cols = bundle["model"], bundle["feature_columns"]

    con = connect(cfg["paths"]["duckdb"])
    df = con.execute("SELECT * FROM mart_loan_features").fetchdf()
    df = features.engineer(df)

    df["pd_hat"] = model.predict_proba(df[feat_cols])[:, 1]

    con.execute("DROP TABLE IF EXISTS mart_scores")
    con.execute("CREATE TABLE mart_scores (loan_id BIGINT, pd_hat DOUBLE)")
    con.register("_scores_df", df[["loan_id", "pd_hat"]])
    con.execute("INSERT INTO mart_scores SELECT loan_id, pd_hat FROM _scores_df")
    n = con.execute("SELECT count(*), round(avg(pd_hat), 4) FROM mart_scores").fetchone()
    print(f"   scored {n[0]} loans, mean PD {n[1]}")
    con.close()
    return int(n[0])


if __name__ == "__main__":
    main()
