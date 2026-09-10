# Data-quality log

Numbers below are from `notebooks/01_eda.ipynb` on the real dataset. Interview talking point.

## Source
- Mirror: `wordsforthewise/lending-club` (Kaggle)
- File: `accepted_2007_to_2018Q4.csv.gz` (read directly by DuckDB)
- Snapshot label: 2018Q4 (data max `issue_d` = 2018-12)
- **Raw row count: 2,260,701**

## Maturity pre-flight result (`sql/preflight_maturity.sql`)
- 36-month loans are ≥ 98% resolved (`frac_terminal`) through **2016-02**; 2016-03 drops
  to ~0.87 and it falls steadily after (2017-11 ≈ 0.30, 2018-12 ≈ 0.03).
- Data max issue = 2018-12 → **`min_months_since_issue = 34`** lands the tail on 2016-02.
- **Window: issued 2012-01 … 2016-02.** `issue_year_max = 2016`, `oot_cutoff = 2015-01-01`.

## Filters applied in `sql/02_staging.sql`
Raw 2,260,701 → **stg_loans 640,919 (28.4% of raw)**. The big cuts are the 60-month term
and the immaturity tail (2016-03 onward), as intended.

| Rule | Reason |
|---|---|
| `loan_status LIKE 'Does not meet the credit policy%'` dropped | pre-2009 non-standard underwriting |
| `term <> 36 months` dropped | maturity window needs a fixed horizon (spec §2.1) |
| `issue_year` outside [2012, 2016] | thin early vintages / immature tail |
| `date_diff(month, issue_d, max_issue) < 34` | loan not old enough to have resolved |
| `loan_amnt <= 0`; `dti` outside [-1, 1000]; `revol_util` outside [0, 1000] | impossible values |
| duplicate `id` | keep earliest `issue_d` (no dupes found in this mirror) |

**Residual extremes kept** (not filtered, < 0.01% of rows, tree model handles them):
`dti` max = 999 (a sentinel), `revol_util` max ≈ 892%, `annual_inc` max = 9,000,000.
Flagged rather than clipped so the choice is visible; revisit if a linear model is used.

## Outcome mix (stg_loans)
| loan_status | n |
|---|---|
| Fully Paid | 549,933 |
| Charged Off | 90,250 |
| Current | 378 |
| Late (31-120 days) | 294 |
| In Grace Period / Late (16-30) | 64 |

`default_flag = 1` for Charged Off + Default; `= 0` for Fully Paid; **NULL for the ~740
still-open loans** → dropped from the modelling frame (`v_model_frame` = 640,183).
Overall observed default rate ≈ **14.1%** (train 13.2%, test 14.9%).

## Default rate by vintage — window is benign but usable
| issue_year | n | default_rate |
|---|---|---|
| 2012 | 43,470 | 0.136 |
| 2013 | 100,422 | 0.123 |
| 2014 | 162,570 | 0.137 |
| 2015 | 283,173 | 0.149 |
| 2016 | 51,284 | 0.148 |

No recession in the window (spec §6 limitation). El-per-exposure is flat across vintages
(~1.0), so there is no vintage bias in the portfolio marts.

## Missingness (median-imputed inside the model pipeline, not in staging)
| Column | null_frac | handling |
|---|---|---|
| emp_length_years | 0.061 | median impute ('n/a' / '' → NULL, `< 1 year` → 0, `10+ years` → 10) |
| mort_acc | 0.0094 | median impute |
| revol_util | 0.0005 | median impute |
| dti | ~0.000005 (3 rows) | median impute |
| everything else | 0.0000 | — |

## Categorical sanity (stg_loans)
- `home_ownership`: MORTGAGE 299,890 / RENT 273,925 / OWN 67,027 / OTHER 40 / NONE 35 / ANY 2
  (the 77 OTHER/NONE/ANY rows are folded by the one-hot `min_frequency` = 0.01).
- `purpose`: debt_consolidation 365,917 (57%), credit_card 159,351 (25%), then a long tail;
  `educational` has 1 row.
- `verification_status`: Source Verified / Not Verified / Verified — 3 clean levels.

## Expanded bureau feature block (added after the minimal-allowlist baseline)
The first pass used 17 core application fields; a linear model on those *lost* to LC's
grade (AUC 0.658 vs 0.669). Added the credit-bureau attributes below — all reported at
the bureau pull, i.e. knowable at underwriting, so no leakage. Null fractions measured on
the modelling window (36-month, 2012-01 … 2016-02):

| Feature | null_frac | Feature | null_frac |
|---|---|---|---|
| `acc_open_past_24mths` | 0.009 | `num_accts_ever_120_pd` | 0.035 |
| `total_bc_limit` | 0.009 | `num_tl_90g_dpd_24m` | 0.035 |
| `total_bal_ex_mort` | 0.009 | `pct_tl_nvr_dlq` | 0.035 |
| `pub_rec_bankruptcies` | 0.000 | `percent_bc_gt_75` | 0.020 |
| `application_type` | 0.000 | `tot_hi_cred_lim` | 0.035 |
| `bc_util` | 0.020 | `tot_cur_bal` | 0.035 |
| `bc_open_to_buy` | 0.019 | `avg_cur_bal` | 0.035 |
| `mths_since_recent_bc` | 0.018 | `tot_coll_amt` | 0.035 |
| `num_actv_bc_tl` | 0.035 | `mo_sin_old_rev_tl_op` | 0.035 |
| `num_tl_op_past_12m` | 0.035 | `mths_since_recent_inq` | 0.116 |

**Excluded despite being origination-time:** the `open_il_*` / `open_rv_*` / `il_util` /
`all_util` / `inq_last_12m` family (only collected from 2015-12, ~99% null in the window)
and `mths_since_last_delinq` (50% null — "never" and "long ago" are not separable under
median imputation without a companion flag).

**Result:** with the expanded ~33-field allowlist, logistic AUC 0.680, HGB 0.691,
grade-alone 0.669 — both models now beat grade (HGB by +0.022). See `artifacts/metrics.json`.

## LGD estimated from recoveries (not assumed)
`src/lgd.py`, on the **40,596 charged-off training-period loans**:
`recovery_rate = Σ(total_rec_prncp + recoveries) / Σ(funded_amnt) = 0.500`
→ **LGD = 0.50** (exposure-weighted; equal-weighted also 0.50). The textbook 0.45 was
optimistic. `config.expected_loss.lgd_mode: data` makes the `lgd` stage compute this and
feed it to `06_marts.sql`; set `fixed` to fall back to `lgd_fixed`. (`artifacts/lgd.json`)

## Realized-outcome backtest (`src/backtest.py`)
Re-runs the policy sweep on the 334k out-of-time **test** loans using actual cash
(`total_pymnt + recoveries − funded_amnt`) and actual charge-offs, not predicted PD:
- realized-profit optimum PD < 0.181 vs model optimum PD < 0.187 — **$0.1M regret** on a ~$6B book
- predicted vs actual default rate track within ~1 pp at every cut-off
- realized credit loss ≈ **1.12×** model EL; realized profit ≈ half the model projection
  (the `(1−PD)·coupon` interest term is optimistic)

## Columns excluded from features
- **Post-origination (leakage):** `last_pymnt_d`, `last_pymnt_amnt`, `total_pymnt`,
  `total_rec_prncp`, `total_rec_int`, `recoveries`, `collection_recovery_fee`, `out_prncp`,
  `out_prncp_inv`, `last_fico_range_*`, `next_pymnt_d`, `settlement_*`, `hardship_*`,
  `debt_settlement_flag`. Kept in `mart_loan_outcomes` for EAD / LGD only.
- **Circular (LC's own risk model):** `grade`, `sub_grade`, `int_rate` → `mart_loan_benchmark`
  (benchmark + the simulator's interest term only).
- **Fairness:** `addr_state` held out of the model by default (spec §6).
