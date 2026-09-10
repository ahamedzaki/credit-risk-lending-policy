#!/usr/bin/env python3
"""Stage runner for the Credit Risk & Lending Policy pipeline.

Usage:
    python run_pipeline.py preflight     # step 0 — decide the maturity window
    python run_pipeline.py all           # raw -> stage -> features -> train -> score -> marts -> export -> check
    python run_pipeline.py train         # any single stage
    python run_pipeline.py stage features marts export   # a subset, in order

Stages: raw, stage, features, train, score, marts, export, check
"""
from __future__ import annotations

import json
import pathlib
import sys
import warnings

import numpy as np

# Some OpenBLAS builds emit spurious FP RuntimeWarnings from `matmul` inside the LBFGS
# solver. They do not affect results; keep the CLI output clean.
np.seterr(all="ignore")
for _m in ("invalid value encountered", "divide by zero encountered", "overflow encountered"):
    warnings.filterwarnings("ignore", message=_m, category=RuntimeWarning)

from src.config import load, sql_params
from src.db import connect, run_sql_file, show

ROOT = pathlib.Path(__file__).resolve().parent
STAGES = ["raw", "stage", "features", "train", "score", "marts", "export", "check"]


def _banner(name: str) -> None:
    print(f"\n=== {name} ===")


def stage_raw(cfg, params, con):
    csv = pathlib.Path(cfg["data"]["raw_csv"])
    if not csv.exists():
        raise SystemExit(
            f"Raw CSV not found: {csv}\n"
            "Download 'wordsforthewise/lending-club' from Kaggle, unzip "
            "accepted_2007_to_2018Q4.csv into data/, or edit config.yaml -> data.raw_csv."
        )
    show(run_sql_file(con, "01_raw.sql", params))


def stage_stage(cfg, params, con):
    show(run_sql_file(con, "02_staging.sql", params))


def stage_features(cfg, params, con):
    show(run_sql_file(con, "03_features_outcome.sql", params))


def stage_train(cfg, params, con):
    from src.train import main as train_main

    auc = train_main()
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "_last_train_auc.json").write_text(json.dumps({"test_auc": auc}))


def stage_score(cfg, params, con):
    from src.score import main as score_main

    score_main()


def stage_marts(cfg, params, con):
    show(run_sql_file(con, "06_marts.sql", params))
    # star-schema views (BI exhibit) — safe to rebuild every run
    show(run_sql_file(con, "../schema/star_schema.sql", params))


def stage_export(cfg, params, con):
    out = pathlib.Path(cfg["paths"]["exports_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for tbl, fname in [
        ("mart_simulator_base", "simulator_base.parquet"),
        ("mart_portfolio_summary", "portfolio_summary.parquet"),
        ("mart_loan_el", "loan_el.parquet"),
    ]:
        con.execute(f"COPY {tbl} TO '{out / fname}' (FORMAT PARQUET)")
        print(f"   wrote {out / fname}")
    # small committed sample for the hosted Streamlit app
    con.execute(
        f"COPY (SELECT * FROM mart_simulator_base USING SAMPLE 50000 ROWS (reservoir, 42)) "
        f"TO '{out / 'simulator_base_sample.parquet'}' (FORMAT PARQUET)"
    )
    print(f"   wrote {out / 'simulator_base_sample.parquet'}")


def stage_check(cfg, params, con):
    """Retrain and assert the test AUC reproduces — guards against a stale committed model."""
    from src.train import main as train_main

    prev = json.loads((ROOT / "artifacts" / "_last_train_auc.json").read_text())["test_auc"]
    auc = train_main()
    tol = cfg["model"]["auc_repro_tolerance"]
    delta = abs(auc - prev)
    print(f"   reproduced test AUC {auc:.4f} vs {prev:.4f} (delta {delta:.4f}, tol {tol})")
    if delta > tol:
        raise SystemExit(f"AUC reproducibility FAILED: delta {delta:.4f} > tol {tol}")
    print("   reproducibility OK")


DISPATCH = {
    "raw": stage_raw, "stage": stage_stage, "features": stage_features,
    "train": stage_train, "score": stage_score, "marts": stage_marts,
    "export": stage_export, "check": stage_check,
}


def run_preflight(cfg, params):
    con = connect(cfg["paths"]["duckdb"])
    stage_raw(cfg, params, con)
    _banner("preflight — maturity by issue month (36-month loans)")
    rows = run_sql_file(con, "preflight_maturity.sql", params)
    print("   issue_month | n | frac_terminal | frac_charged_off")
    show(rows)
    thr = cfg["maturity"]["terminal_fraction_threshold"]
    mature = [r for r in rows if r[2] is not None and r[2] >= thr]
    if mature:
        print(f"\n   newest month with frac_terminal >= {thr}: {mature[-1][0]}")
        print("   -> set config.split.oot_cutoff a few months before this, and")
        print("      config.maturity.issue_year_max to that year (or earlier).")
    con.close()


def main(argv: list[str]) -> None:
    cfg = load()
    params = sql_params(cfg)
    args = argv or ["all"]

    if args == ["preflight"]:
        run_preflight(cfg, params)
        return

    stages = STAGES if args == ["all"] else args
    unknown = [s for s in stages if s not in DISPATCH]
    if unknown:
        raise SystemExit(f"Unknown stage(s): {unknown}\nValid: {STAGES}")

    con = connect(cfg["paths"]["duckdb"])
    for s in stages:
        _banner(s)
        DISPATCH[s](cfg, params, con)
    con.close()
    print("\nDone.")


if __name__ == "__main__":
    main(sys.argv[1:])
