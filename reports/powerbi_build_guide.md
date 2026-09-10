# Power BI build guide (Windows contributor)

You are building `reports/powerbi_dashboard.pdf` — a **2-page** report over the parquet
marts the pipeline produced. Import mode only. Do not connect to DuckDB.

Estimated time: ~half a day. Only you edit the `.pbix`; everyone else consumes the PDF.

---

## 1. Get the data

From the repo, after someone has run `python run_pipeline.py all`:

```
exports/simulator_base.parquet     640,919 rows — the fact table (one row per loan)
exports/portfolio_summary.parquet  30 rows — pre-aggregated concentration (optional convenience)
```

`simulator_base.parquet` has everything both pages need. `portfolio_summary` is only a
shortcut for the concentration matrix.

### Import

1. **Home → Get data → Parquet** → pick `exports/simulator_base.parquet`.
2. In Power Query, confirm column types:
   | column | type |
   |---|---|
   | `loan_id` | Whole number |
   | `issue_d`, `vintage_year` | Date |
   | `split_set`, `purpose`, `lc_grade` | Text |
   | `loan_amnt`, `fico_mid`, `int_rate`, `pd_hat`, `ead`, `expected_loss` | Decimal number |
   | `default_flag` | Whole number |
   | `is_terminal` | True/False |
3. Rename the query to `loans`. **Close & Apply.**
4. (Optional) repeat for `portfolio_summary.parquet` → query `concentration`.
   No relationship needed — it is a standalone summary table.
5. Add a small **Policy** parameter table for the Lending Strategy page (What-if
   parameter, below).

There is **no star schema in Power BI** here — `loans` is a single flat fact table, which
is the right shape at this size. (The DuckDB `schema/star_schema.sql` views are a modelling
exhibit for the repo, not for the report.)

---

## 2. Measures (DAX)

Create these in the `loans` table. `[...]` are new measures.

```DAX
Loans              = COUNTROWS ( loans )
Funded             = SUM ( loans[loan_amnt] )
Expected Loss      = SUM ( loans[expected_loss] )
Avg PD             = AVERAGE ( loans[pd_hat] )
EL Rate            = DIVIDE ( [Expected Loss], [Funded] )

Terminal Loans     = CALCULATE ( [Loans], loans[is_terminal] = TRUE )
Observed Defaults  = CALCULATE ( SUM ( loans[default_flag] ), loans[is_terminal] = TRUE )
Observed DR        = DIVIDE ( [Observed Defaults], [Terminal Loans] )

High-Risk Exposure = CALCULATE ( [Funded], loans[pd_hat] >= 0.20 )
High-Risk Share    = DIVIDE ( [High-Risk Exposure], [Funded] )

-- concentration: EL share vs exposure share within the current visual's grouping
Exposure Share     = DIVIDE ( [Funded], CALCULATE ( [Funded], ALLSELECTED ( loans ) ) )
EL Share           = DIVIDE ( [Expected Loss], CALCULATE ( [Expected Loss], ALLSELECTED ( loans ) ) )
EL-to-Exposure     = DIVIDE ( [EL Share], [Exposure Share] )   -- > 1 = over-represented in losses
```

### Lending Strategy — What-if parameters + policy measures

Modeling → **New parameter → Numeric range**:
- `PD Cutoff` : 0.02 … 0.40, step 0.005, default 0.15
- `Min FICO`  : 600 … 780, step 5, default 660

Then:

```DAX
Approved Loans =
    CALCULATE ( [Loans],
        FILTER ( loans, loans[pd_hat] < 'PD Cutoff'[PD Cutoff Value]
                     && loans[fico_mid] >= 'Min FICO'[Min FICO Value] ) )

Approval Rate = DIVIDE ( [Approved Loans], [Loans] )

Approved Volume =
    CALCULATE ( [Funded],
        FILTER ( loans, loans[pd_hat] < 'PD Cutoff'[PD Cutoff Value]
                     && loans[fico_mid] >= 'Min FICO'[Min FICO Value] ) )

Approved Expected Loss =
    CALCULATE ( [Expected Loss],
        FILTER ( loans, loans[pd_hat] < 'PD Cutoff'[PD Cutoff Value]
                     && loans[fico_mid] >= 'Min FICO'[Min FICO Value] ) )

-- profit model must match app/sim_core.py: b = 0.52, r_f = 4%, s = 1.2%, T = 3
Approved Interest =
    SUMX (
        FILTER ( loans, loans[pd_hat] < 'PD Cutoff'[PD Cutoff Value]
                     && loans[fico_mid] >= 'Min FICO'[Min FICO Value] ),
        loans[loan_amnt] * loans[int_rate] * 3 * 0.52 * ( 1 - loans[pd_hat] ) )

Approved Funding =
    CALCULATE ( SUMX ( loans, loans[loan_amnt] * 0.04 * 3 * 0.52 ),
        FILTER ( loans, loans[pd_hat] < 'PD Cutoff'[PD Cutoff Value]
                     && loans[fico_mid] >= 'Min FICO'[Min FICO Value] ) )

Approved Servicing =
    CALCULATE ( SUMX ( loans, loans[loan_amnt] * 0.012 * 3 ),
        FILTER ( loans, loans[pd_hat] < 'PD Cutoff'[PD Cutoff Value]
                     && loans[fico_mid] >= 'Min FICO'[Min FICO Value] ) )

Approved Profit = [Approved Interest] - [Approved Funding] - [Approved Servicing] - [Approved Expected Loss]
```

> Keep the constants (0.52, 0.04, 0.012, 3) identical to `config.yaml → simulator` and
> `app/sim_core.py`. If those change, update here too.

---

## 3. Page 1 — Portfolio Risk Overview

Layout: a KPI row across the top, then concentration below.

**KPI cards (row of 5):** `Funded`, `Expected Loss`, `EL Rate` (%), `Observed DR` (%),
`High-Risk Exposure`.

**Visuals:**
1. **Clustered bar — Expected Loss by `lc_grade`** (A→G). Add `EL-to-Exposure` as a line
   on a secondary axis, or a second bar. Title: "Losses concentrate in C–D".
2. **Matrix** — rows `lc_grade`, values `Funded`, `Exposure Share`, `Expected Loss`,
   `EL Share`, `EL-to-Exposure`. Conditional-format `EL-to-Exposure` (red > 1).
3. **Clustered bar — `Observed DR` vs `Avg PD` by `lc_grade`** — shows the model
   under-predicts the low grades (calibration story).
4. **Bar — Expected Loss by `purpose`** (top 8). Call out `small_business`
   (1.1% of exposure, 1.9% of EL).
5. **Column — `Funded` and `EL Rate` by `vintage_year`** — flat ≈ no vintage bias.
6. **Slicers:** `lc_grade`, `purpose`, `split_set`.

Headline callouts (text boxes): "Grade A = 25.6% of exposure but 12.9% of expected loss";
"Grades C+D = 36.5% of exposure, 49.4% of expected loss".

---

## 4. Page 2 — Lending Strategy

**Slicers (top):** `PD Cutoff` and `Min FICO` numeric-range slicers (from the What-if
parameters).

**KPI cards (row of 5):** `Approval Rate` (%), `Approved Volume`, `Approved Expected Loss`,
`Approved Profit`, and a card showing `Approved Loans`.

**Visuals:**
1. **Line chart — profit vs approval (the centrepiece).** Build a disconnected
   `PD grid` table (0.02, 0.025, … 0.40) with Enter Data or a calculated table. X-axis =
   an "Approval Rate at grid" measure; Y = a "Profit at grid" measure that reuses the
   profit formula with `[PD grid value]` instead of the parameter. Mark the peak
   (≈ PD < 0.20, ~83% approval on the full book; ~PD 0.18 on the out-of-time test —
   `artifacts/backtest.json`) with a constant line or annotation.
   *Simpler fallback:* a table with rows Conservative 0.08 / Current 0.15 /
   Profit-max ~0.18 / Growth 0.20 and the profit for each.
2. **Waterfall — Approved Profit bridge:** `Approved Interest` → `- Approved Funding`
   → `- Approved Servicing` → `- Approved Expected Loss` → `Approved Profit`.
3. **Matrix — approved book by `lc_grade`:** `Approved Loans`, `Approved Volume`,
   `Avg PD`, `Approved Expected Loss`.
4. **Card + text:** the recommendation — "Move cut-off 0.15 → ~0.18: +~13 pts approval,
   +$0.9B volume. Optimal cut-off confirmed on out-of-time data; realized profit ≈ half
   the model projection. Do not go past ~0.20 — profit falls."

---

## 5. Export

**File → Export → Export to PDF.** Save as `reports/powerbi_dashboard.pdf`, commit it.
Also drop 2 PNG screenshots into `reports/figures/` if you want them in the README.

## 6. Sanity checks before exporting

Numbers below are the **full scored book** (what `mart_simulator_base` / the Streamlit
app show). LGD is now data-estimated at **0.50** (was 0.45) and servicing is **0.4%/yr**.

- KPI `Expected Loss` on Page 1 (no filters) ≈ **$524M**; `Funded` ≈ **$8.2B**;
  `Observed DR` ≈ **14.1%**.
- Page 2 at defaults (PD < 0.15, FICO ≥ 660): `Approval Rate` ≈ **64%**,
  `Approved Volume` ≈ **$5.4B**, `Approved Expected Loss` ≈ **$228M**,
  `Approved Profit` ≈ **$163M**; profit-max ≈ PD < 0.20 / 83%.
- These are **model projections on the whole book**. The recommendation in
  `reports/memo.md` is validated separately on the out-of-time test set
  (`artifacts/backtest.json`): same optimal cut-off (~PD 0.18), but realized profit ≈
  half the model figure and realized loss ≈ 1.12× model EL. If Page 2 shows a
  model-vs-realized comparison, pull it from `artifacts/backtest.json`.
- Disparate-impact flags for the recommendation card come from `artifacts/fairness.json`
  (income-band AIR 0.42, home-ownership AIR 0.68 — both flag the 4/5ths rule).
- `reports/memo.md` is the source of truth — re-check after any pipeline change.
