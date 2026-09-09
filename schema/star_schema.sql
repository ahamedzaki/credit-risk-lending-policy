-- Dimensional model (BI exhibit, spec §3). Views over the clean marts — no data moves.
-- Grain of FACT_LOAN: one row per loan.

CREATE OR REPLACE VIEW DIM_DATE AS
SELECT DISTINCT
    issue_d                                   AS date_key,
    extract('year'  FROM issue_d)             AS year,
    extract('quarter' FROM issue_d)           AS quarter,
    extract('month' FROM issue_d)             AS month,
    strftime(issue_d, '%Y-%m')               AS year_month
FROM mart_loan_features;

CREATE OR REPLACE VIEW DIM_BORROWER AS
SELECT
    loan_id                                   AS loan_key,      -- 1:1 with loan here (no customer id in LC)
    home_ownership,
    verification_status,
    addr_state,
    CASE WHEN annual_inc < 40000 THEN '1 <40k'
         WHEN annual_inc < 70000 THEN '2 40-70k'
         WHEN annual_inc < 120000 THEN '3 70-120k'
         ELSE '4 120k+' END                   AS income_band,
    emp_length_years
FROM mart_loan_features;

CREATE OR REPLACE VIEW DIM_LOAN_PRODUCT AS
SELECT DISTINCT
    md5(purpose || '|' || COALESCE(b.lc_grade, '?'))  AS product_key,
    f.purpose,
    b.lc_grade,
    36                                        AS term_months
FROM mart_loan_features f
JOIN mart_loan_benchmark b USING (loan_id);

CREATE OR REPLACE VIEW FACT_LOAN AS
SELECT
    f.loan_id                                 AS loan_key,
    f.issue_d                                 AS date_key,
    md5(f.purpose || '|' || COALESCE(b.lc_grade, '?')) AS product_key,
    f.split_set,
    f.loan_amnt,
    f.fico_mid,
    f.dti,
    b.int_rate / 100.0                        AS int_rate,
    s.pd_hat,
    el.ead,
    el.lgd,
    el.expected_loss,
    o.default_flag,
    o.is_terminal
FROM mart_loan_features f
JOIN mart_loan_benchmark b USING (loan_id)
LEFT JOIN mart_scores    s USING (loan_id)
LEFT JOIN mart_loan_el   el USING (loan_id)
LEFT JOIN mart_loan_outcomes o USING (loan_id);

SELECT 'FACT_LOAN' AS view, count(*) AS rows FROM FACT_LOAN;
