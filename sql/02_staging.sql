-- Stage 2 (silver): typed, de-duplicated, quality-filtered, maturity-windowed.
-- One clean row per loan. Applies the foundation rules from spec §2.1–§2.3.
--
-- Placeholders substituted by run_pipeline.py:
--   ${term_months} ${issue_year_min} ${issue_year_max} ${min_months_since_issue}

DROP TABLE IF EXISTS stg_loans;

CREATE TABLE stg_loans AS
WITH typed AS (
    SELECT
        TRY_CAST(id AS BIGINT)                                              AS loan_id,
        try_strptime(trim(issue_d), '%b-%Y')::DATE                          AS issue_d,
        TRY_CAST(regexp_extract(trim(term), '[0-9]+') AS INTEGER)           AS term_months,
        trim(loan_status)                                                   AS loan_status,

        -- ---- origination-time features (allowlist; see src/features.py) ----
        TRY_CAST(loan_amnt AS DOUBLE)                                       AS loan_amnt,
        TRY_CAST(annual_inc AS DOUBLE)                                      AS annual_inc,
        TRY_CAST(dti AS DOUBLE)                                             AS dti,
        CASE
            WHEN emp_length IS NULL OR trim(emp_length) IN ('', 'n/a') THEN NULL
            WHEN trim(emp_length) = '< 1 year' THEN 0
            ELSE TRY_CAST(regexp_extract(trim(emp_length), '[0-9]+') AS INTEGER)
        END                                                                AS emp_length_years,
        upper(trim(home_ownership))                                         AS home_ownership,
        lower(trim(purpose))                                               AS purpose,
        upper(trim(addr_state))                                            AS addr_state,
        trim(verification_status)                                          AS verification_status,
        TRY_CAST(fico_range_low AS DOUBLE)                                  AS fico_range_low,
        TRY_CAST(fico_range_high AS DOUBLE)                                 AS fico_range_high,
        try_strptime(trim(earliest_cr_line), '%b-%Y')::DATE                 AS earliest_cr_line,
        TRY_CAST(open_acc AS DOUBLE)                                        AS open_acc,
        TRY_CAST(total_acc AS DOUBLE)                                       AS total_acc,
        TRY_CAST(revol_bal AS DOUBLE)                                       AS revol_bal,
        TRY_CAST(replace(trim(revol_util), '%', '') AS DOUBLE)             AS revol_util,
        TRY_CAST(delinq_2yrs AS DOUBLE)                                     AS delinq_2yrs,
        TRY_CAST(inq_last_6mths AS DOUBLE)                                  AS inq_last_6mths,
        TRY_CAST(pub_rec AS DOUBLE)                                         AS pub_rec,
        TRY_CAST(mort_acc AS DOUBLE)                                        AS mort_acc,

        -- ---- expanded bureau attributes (credit-bureau pull at application; <4% null
        --      in the 2012-01..2016-02 window; all origination-time, no leakage) ----
        TRY_CAST(acc_open_past_24mths AS DOUBLE)                            AS acc_open_past_24mths,
        TRY_CAST(replace(trim(bc_util), '%', '') AS DOUBLE)               AS bc_util,
        TRY_CAST(bc_open_to_buy AS DOUBLE)                                  AS bc_open_to_buy,
        TRY_CAST(mo_sin_old_rev_tl_op AS DOUBLE)                            AS mo_sin_old_rev_tl_op,
        TRY_CAST(mths_since_recent_inq AS DOUBLE)                           AS mths_since_recent_inq,
        TRY_CAST(mths_since_recent_bc AS DOUBLE)                            AS mths_since_recent_bc,
        TRY_CAST(num_actv_bc_tl AS DOUBLE)                                  AS num_actv_bc_tl,
        TRY_CAST(num_tl_op_past_12m AS DOUBLE)                              AS num_tl_op_past_12m,
        TRY_CAST(num_accts_ever_120_pd AS DOUBLE)                           AS num_accts_ever_120_pd,
        TRY_CAST(num_tl_90g_dpd_24m AS DOUBLE)                              AS num_tl_90g_dpd_24m,
        TRY_CAST(replace(trim(pct_tl_nvr_dlq), '%', '') AS DOUBLE)        AS pct_tl_nvr_dlq,
        TRY_CAST(replace(trim(percent_bc_gt_75), '%', '') AS DOUBLE)      AS percent_bc_gt_75,
        TRY_CAST(tot_hi_cred_lim AS DOUBLE)                                 AS tot_hi_cred_lim,
        TRY_CAST(total_bal_ex_mort AS DOUBLE)                               AS total_bal_ex_mort,
        TRY_CAST(total_bc_limit AS DOUBLE)                                  AS total_bc_limit,
        TRY_CAST(tot_cur_bal AS DOUBLE)                                     AS tot_cur_bal,
        TRY_CAST(avg_cur_bal AS DOUBLE)                                     AS avg_cur_bal,
        TRY_CAST(tot_coll_amt AS DOUBLE)                                    AS tot_coll_amt,
        TRY_CAST(pub_rec_bankruptcies AS DOUBLE)                            AS pub_rec_bankruptcies,
        nullif(trim(application_type), '')                                 AS application_type,

        -- ---- benchmark-only (NOT features): Lending Club's own risk model ----
        trim(grade)                                                        AS lc_grade,
        trim(sub_grade)                                                    AS lc_sub_grade,
        TRY_CAST(replace(trim(int_rate), '%', '') AS DOUBLE)              AS int_rate,

        -- ---- post-origination fields (outcomes / LGD / EAD only; never features) ----
        TRY_CAST(funded_amnt AS DOUBLE)                                    AS funded_amnt,
        TRY_CAST(out_prncp AS DOUBLE)                                      AS out_prncp,
        TRY_CAST(total_pymnt AS DOUBLE)                                    AS total_pymnt,
        TRY_CAST(total_rec_prncp AS DOUBLE)                               AS total_rec_prncp,
        TRY_CAST(recoveries AS DOUBLE)                                    AS recoveries,
        try_strptime(trim(last_pymnt_d), '%b-%Y')::DATE                    AS last_pymnt_d
    FROM raw_loans
    WHERE loan_status IS NOT NULL
      AND trim(loan_status) NOT LIKE 'Does not meet the credit policy%'
),
flagged AS (
    SELECT
        *,
        extract('year' FROM issue_d)                                       AS issue_year,
        (CASE WHEN loan_status IN ('Fully Paid', 'Charged Off', 'Default') THEN TRUE ELSE FALSE END)
                                                                          AS is_terminal
    FROM typed
    WHERE loan_id IS NOT NULL
      AND issue_d IS NOT NULL
      AND term_months = ${term_months}
),
maturity AS (
    -- an issue month is usable only if the whole cohort has had time to resolve
    SELECT date_diff('month', min_issue, max_issue) AS _ignore,
           max_issue
    FROM (SELECT min(issue_d) AS min_issue, max(issue_d) AS max_issue FROM flagged)
),
windowed AS (
    SELECT f.*
    FROM flagged f, maturity m
    WHERE f.issue_year BETWEEN ${issue_year_min} AND ${issue_year_max}
      AND date_diff('month', f.issue_d, m.max_issue) >= ${min_months_since_issue}
      -- impossible-value guards
      AND f.loan_amnt > 0
      AND (f.annual_inc IS NULL OR f.annual_inc >= 0)
      AND (f.dti IS NULL OR f.dti BETWEEN -1 AND 1000)
      AND (f.revol_util IS NULL OR f.revol_util BETWEEN 0 AND 1000)
)
SELECT * EXCLUDE (issue_year)
FROM windowed
QUALIFY row_number() OVER (PARTITION BY loan_id ORDER BY issue_d) = 1;

-- data-quality summary (printed by the runner + logged to reports/dq_log.md by src)
SELECT
    count(*)                                                    AS n_loans,
    min(issue_d)                                                AS first_issue,
    max(issue_d)                                                AS last_issue,
    round(avg(CASE WHEN is_terminal THEN 1.0 ELSE 0.0 END), 4)  AS frac_terminal,
    round(avg(CASE WHEN is_terminal AND loan_status <> 'Fully Paid' THEN 1.0
                   WHEN is_terminal THEN 0.0 END), 4)           AS observed_default_rate,
    sum(CASE WHEN annual_inc IS NULL THEN 1 ELSE 0 END)         AS null_annual_inc,
    sum(CASE WHEN dti IS NULL THEN 1 ELSE 0 END)                AS null_dti,
    sum(CASE WHEN revol_util IS NULL THEN 1 ELSE 0 END)         AS null_revol_util
FROM stg_loans;
