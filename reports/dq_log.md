# Data-quality log

Record every cleaning / exclusion decision here as you make it. This is an interview
talking point — keep it specific.

## Source
- Mirror: `wordsforthewise/lending-club` (Kaggle)
- File: `accepted_2007_to_2018Q4.csv`
- Snapshot label: 2018Q4
- Raw row count: [fill after `run_pipeline.py raw`]

## Filters applied in `sql/02_staging.sql`
| Rule | Reason | Rows removed |
|---|---|---|
| `loan_status LIKE 'Does not meet the credit policy%'` dropped | pre-2009 non-standard underwriting | [ ] |
| `term <> 36 months` dropped | maturity window needs a fixed horizon (spec §2.1) | [ ] |
| `issue_year` outside [2012, 2016] | thin/immature vintages | [ ] |
| `date_diff(month, issue_d, max_issue) < 39` | loan not old enough to have resolved | [ ] |
| `loan_amnt <= 0`, `dti` outside [-1, 1000], `revol_util` outside [0, 1000] | impossible values | [ ] |
| duplicate `id` | keep earliest `issue_d` | [ ] |

## Columns excluded from features
- **Post-origination (leakage):** `last_pymnt_d`, `last_pymnt_amnt`, `total_pymnt`,
  `total_rec_prncp`, `total_rec_int`, `recoveries`, `collection_recovery_fee`,
  `out_prncp`, `out_prncp_inv`, `last_fico_range_*`, `next_pymnt_d`, `settlement_*`,
  `hardship_*`, `debt_settlement_flag`.
- **Circular (LC's own risk model):** `grade`, `sub_grade`, `int_rate` — moved to
  `mart_loan_benchmark`.

## Missingness (fill after `run_pipeline.py stage`)
| Column | null_frac | handling |
|---|---|---|
| annual_inc | [ ] | median impute |
| dti | [ ] | median impute |
| revol_util | [ ] | median impute |
| emp_length_years | [ ] | median impute (kept as numeric; 'n/a' → null) |
| mort_acc | [ ] | median impute (absent in some mirrors) |

## Maturity pre-flight result
Newest issue month with `frac_terminal >= 0.98`: [fill]
→ `oot_cutoff` set to [fill]; `issue_year_max` set to [fill].
