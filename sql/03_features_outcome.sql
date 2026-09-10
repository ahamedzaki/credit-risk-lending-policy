-- Stage 3 (gold, part 1): split the clean table into three disjoint marts so a
-- post-origination or benchmark column can NEVER reach the model by accident.
--
--   mart_loan_features    origination-time ALLOWLIST only  (spec §2.2, §4)
--   mart_loan_benchmark   Lending Club grade / int_rate    (benchmark + profit calc only)
--   mart_loan_outcomes    default flag + fields for EAD / LGD
--
-- Placeholder: ${oot_cutoff}  (YYYY-MM-DD)

DROP TABLE IF EXISTS mart_loan_features;
CREATE TABLE mart_loan_features AS
SELECT
    loan_id,
    issue_d,
    CASE WHEN issue_d < DATE '${oot_cutoff}' THEN 'train' ELSE 'test' END AS split_set,

    loan_amnt,
    annual_inc,
    dti,
    emp_length_years,
    home_ownership,
    purpose,
    addr_state,
    verification_status,
    (fico_range_low + fico_range_high) / 2.0                              AS fico_mid,
    date_diff('month', earliest_cr_line, issue_d)                         AS credit_history_months,
    open_acc,
    total_acc,
    revol_bal,
    revol_util,
    delinq_2yrs,
    inq_last_6mths,
    pub_rec,
    mort_acc,

    -- expanded bureau attributes (spec §4; see 02_staging.sql)
    acc_open_past_24mths,
    bc_util,
    bc_open_to_buy,
    mo_sin_old_rev_tl_op,
    mths_since_recent_inq,
    mths_since_recent_bc,
    num_actv_bc_tl,
    num_tl_op_past_12m,
    num_accts_ever_120_pd,
    num_tl_90g_dpd_24m,
    pct_tl_nvr_dlq,
    percent_bc_gt_75,
    tot_hi_cred_lim,
    total_bal_ex_mort,
    total_bc_limit,
    tot_cur_bal,
    avg_cur_bal,
    tot_coll_amt,
    pub_rec_bankruptcies,
    application_type
FROM stg_loans;

DROP TABLE IF EXISTS mart_loan_benchmark;
CREATE TABLE mart_loan_benchmark AS
SELECT loan_id, issue_d, lc_grade, lc_sub_grade, int_rate
FROM stg_loans;

DROP TABLE IF EXISTS mart_loan_outcomes;
CREATE TABLE mart_loan_outcomes AS
SELECT
    loan_id,
    issue_d,
    loan_status,
    is_terminal,
    CASE
        WHEN loan_status IN ('Charged Off', 'Default') THEN 1
        WHEN loan_status = 'Fully Paid'                 THEN 0
        ELSE NULL                                          -- non-terminal: excluded from modelling
    END                                                                  AS default_flag,
    funded_amnt,
    loan_amnt,
    out_prncp,
    total_rec_prncp,
    recoveries
FROM stg_loans;

-- modelling frame = features + label, terminal loans only
DROP VIEW IF EXISTS v_model_frame;
CREATE VIEW v_model_frame AS
SELECT f.*, o.default_flag
FROM mart_loan_features f
JOIN mart_loan_outcomes o USING (loan_id)
WHERE o.default_flag IS NOT NULL;

SELECT split_set, count(*) AS n, round(avg(default_flag), 4) AS default_rate
FROM v_model_frame GROUP BY 1 ORDER BY 1;
