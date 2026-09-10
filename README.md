# Credit Risk & Lending Policy Analysis

End-to-end analysis that answers one business question:
**which loan applicants should be approved, and what does each policy choice cost in
expected loss versus earn in risk-adjusted profit?**

Scoped v1, hardened per design review. See `reports/foundation_architecture.md` (the spec)
for the full rationale. This README is the operational guide.

---

## What it produces

| Artefact | Path | Description |
|---|---|---|
| Cleaned analytical DB | `data/credit.duckdb` | raw → staging → feature/outcome → marts |
| PD model | `artifacts/model.joblib` | calibrated PD model (`config.model.type`: `hgb` default, or `logistic`) |
| Metrics | `artifacts/metrics.json` | AUC / KS / Brier, calibration, benchmark ΔAUC |
| Figures | `reports/figures/*.png` | calibration curve, decile lift, profit-vs-approval |
| BI marts | `exports/*.parquet` | `portfolio_summary`, `simulator_base` for Power BI |
| Simulator | `app/simulator.py` | Streamlit policy simulator |
| Memo | `reports/memo.md` | one-page recommendation + Assumptions & Limitations |

Nothing runs as a service. One command rebuilds everything.

---

## Prerequisites

- **Python 3.11** (match this exactly on both machines).
- **libomp** for LightGBM on macOS: `brew install libomp`.
- The **Lending Club accepted-loans CSV**. This repo does **not** ship it (~1.6 GB).

### Get the data

1. Kaggle dataset: `wordsforthewise/lending-club` → file `accepted_2007_to_2018Q4.csv.gz`.
2. Unzip and place at: `data/accepted_2007_to_2018Q4.csv`
   (or point `config.yaml → data.raw_csv` at wherever you put it).
3. Record the mirror + snapshot label in `config.yaml` (already set to `2018Q4`).

> Column names vary across Kaggle mirrors. This pipeline targets the `wordsforthewise`
> mirror. If you use another, check `sql/02_staging.sql` and `src/features.py`.

---

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Run

### Step 0 — pre-flight (decides the maturity window)

Before modelling, find the newest issue month where ~all 36-month loans have finished:

```bash
python run_pipeline.py preflight
```

This prints a table of `issue_month | n | frac_terminal`. Pick the newest month with
`frac_terminal >= config.maturity.terminal_fraction_threshold` (0.98) and, if it differs
from the default, set `config.maturity.issue_year_max` / `config.split.oot_cutoff`
accordingly. The window is also auto-enforced in staging, but you should confirm it.

### Full pipeline

```bash
python run_pipeline.py all
```

Stages (each runnable on its own, e.g. `python run_pipeline.py train`):

| Stage | Does |
|---|---|
| `raw`      | load CSV → `raw_loans` (all-varchar, no typing) |
| `stage`    | `sql/02_staging.sql` → typed, de-duped, maturity-windowed `stg_loans` |
| `features` | `sql/03_features_outcome.sql` → `mart_loan_features` (allowlist) + `mart_loan_outcomes` |
| `train`    | `src/train.py` → `model.joblib`, `metrics.json`, calibration + lift figures, grade benchmark |
| `score`    | `src/score.py` → `pd_hat` per loan, `mart_loan_el` (EL = PD·EAD·LGD) |
| `marts`    | `sql/06_marts.sql` → `mart_portfolio_summary`, `mart_simulator_base` |
| `export`   | write `exports/*.parquet` |
| `check`    | retrain and assert test AUC reproduces within `config.model.auc_repro_tolerance` |

### Simulator

```bash
streamlit run app/simulator.py
```

Reads `exports/simulator_base.parquet` (falls back to the committed `*_sample.parquet`).
Two levers (PD cut-off, min FICO). Shows approval rate, volume, default rate, expected
loss, expected profit, and the profit-vs-approval curve. Assumptions (LGD, cost of funds,
horizon) are shown in the sidebar and live in `config.yaml`.

Profit model (single-period, spec §5.6): `interest income = Σ loan_amnt · int_rate · T ·
(1 − PD)`, `funding cost = Σ loan_amnt · r_f · T`, `expected loss = Σ PD · EAD · LGD`,
`profit = interest − funding − loss`. The `(1 − PD)` haircut on interest is what lets the
profit curve turn over. Still an approximation — no cash-flow timing, prepayment or
servicing cost.

### Power BI (Windows contributor)

Open `exports/portfolio_summary.parquet` and `exports/simulator_base.parquet` in Power BI
Desktop (import mode). Two pages: Portfolio Risk, Lending Strategy. Export a PDF to
`reports/powerbi_dashboard.pdf`. Only the Windows contributor edits `*.pbix`.

---

## Foundation rules (do not break these)

1. **Maturity window** — 36-month loans only, mature vintages only. Enforced in `02_staging.sql`.
2. **Feature allowlist** — only `src/features.py::ALLOWLIST` enters the model. `grade`,
   `sub_grade`, `int_rate` are **excluded** (they are Lending Club's own risk model) and
   kept only for the benchmark and the profit calc.
3. **Out-of-time split** — by `issue_d`, never random. Set in `config.yaml`.
4. **Benchmark** — model vs `grade`-alone must be reported (`metrics.json`).
5. **Reproducibility** — commit code, not artefacts, as source of truth. `check` stage guards
   against a stale committed model.

## Repo layout

```
config.yaml              all assumptions and paths
run_pipeline.py           stage runner + AUC-repro assert
sql/                      01_raw · 02_staging · 03_features_outcome · 06_marts · preflight
src/features.py           the allowlist + transforms (shared)
src/train.py              fit + calibrate + grade benchmark + metrics
src/score.py              batch predict + expected loss
src/evaluate.py           calibration curve, KS, decile lift
schema/star_schema.sql    DIM_/FACT_ views (BI exhibit)  ·  schema/er_diagram.md
app/simulator.py          Streamlit UI  ·  app/sim_core.py = pure policy/profit math
scripts/make_synthetic_data.py   LC-schema synthetic generator (local testing / demo)
tests/test_sim_core.py    unit tests for the simulator math
notebooks/                01_eda · 02_features_model · 03_expected_loss  (# %% scripts)
reports/memo.md           one-pager + Assumptions & Limitations
Makefile                  make install | synth | preflight | pipeline | app | test
```
