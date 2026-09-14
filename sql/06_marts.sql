-- Stage 3 (gold, part 2): serving marts, built AFTER scoring writes mart_scores.
--   mart_loan_el          per-loan expected loss
--   mart_simulator_base   loan-level input for the Streamlit simulator
--   mart_portfolio_summary aggregated view for Power BI page 1
--
-- Placeholders: ${lgd}  ${ead_mode}   (ead_mode: 'funded' | 'outstanding')
-- ${lgd} is the flat, exposure-weighted portfolio LGD (fallback only, e.g. a grade with
-- no charged-off history). Real LGD is segmented by grade via `lgd_by_grade` (written by
-- src/lgd.py: estimate_by_grade + write_grade_table, credibility-weighted toward the flat
-- value for thin grades) — this is what actually feeds Expected Loss below. Flat LGD
-- alone hid a real 44%-vs-60% spread by grade; see artifacts/lgd.json's "by_grade" key.

DROP TABLE IF EXISTS mart_loan_el;
CREATE TABLE mart_loan_el AS
SELECT
    s.loan_id,
    s.pd_hat,
    CASE '${ead_mode}'
        WHEN 'outstanding' THEN COALESCE(o.out_prncp, o.funded_amnt, o.loan_amnt)
        ELSE COALESCE(o.funded_amnt, o.loan_amnt)
    END                                             AS ead,
    COALESCE(g.lgd, ${lgd})                          AS lgd,
    s.pd_hat *
      (CASE '${ead_mode}'
           WHEN 'outstanding' THEN COALESCE(o.out_prncp, o.funded_amnt, o.loan_amnt)
           ELSE COALESCE(o.funded_amnt, o.loan_amnt)
       END) * COALESCE(g.lgd, ${lgd})                AS expected_loss
FROM mart_scores s
JOIN mart_loan_outcomes o USING (loan_id)
LEFT JOIN mart_loan_benchmark b USING (loan_id)
LEFT JOIN lgd_by_grade g ON g.grade = b.lc_grade;

DROP TABLE IF EXISTS mart_simulator_base;
CREATE TABLE mart_simulator_base AS
SELECT
    f.loan_id,
    f.issue_d,
    date_trunc('year', f.issue_d)                    AS vintage_year,
    f.split_set,
    f.loan_amnt,
    f.fico_mid,
    f.purpose,
    b.lc_grade,
    b.int_rate / 100.0                               AS int_rate,          -- stored as %, expose as fraction
    s.pd_hat,
    el.ead,
    el.expected_loss,
    o.default_flag,
    o.is_terminal
FROM mart_loan_features f
JOIN mart_loan_benchmark b USING (loan_id)
JOIN mart_scores         s USING (loan_id)
JOIN mart_loan_el        el USING (loan_id)
JOIN mart_loan_outcomes  o USING (loan_id);

DROP TABLE IF EXISTS mart_portfolio_summary;
CREATE TABLE mart_portfolio_summary AS
WITH banded AS (
    SELECT
        *,
        CASE
            WHEN pd_hat < 0.05 THEN '1 Low (<5%)'
            WHEN pd_hat < 0.10 THEN '2 Moderate (5-10%)'
            WHEN pd_hat < 0.20 THEN '3 Elevated (10-20%)'
            ELSE                    '4 High (20%+)'
        END AS risk_band
    FROM mart_simulator_base
)
SELECT
    'grade'      AS dimension, lc_grade                    AS bucket,
    count(*) AS n_loans, sum(loan_amnt) AS funded,
    round(avg(pd_hat), 4) AS avg_pd, sum(expected_loss) AS total_el,
    round(avg(CASE WHEN is_terminal THEN default_flag END), 4) AS observed_default_rate,
    sum(CASE WHEN pd_hat >= 0.20 THEN loan_amnt ELSE 0 END)    AS high_risk_exposure
FROM banded GROUP BY 2
UNION ALL SELECT 'purpose', purpose, count(*), sum(loan_amnt), round(avg(pd_hat),4),
    sum(expected_loss), round(avg(CASE WHEN is_terminal THEN default_flag END),4),
    sum(CASE WHEN pd_hat >= 0.20 THEN loan_amnt ELSE 0 END)
FROM banded GROUP BY 2
UNION ALL SELECT 'vintage', CAST(extract('year' FROM vintage_year) AS VARCHAR), count(*), sum(loan_amnt),
    round(avg(pd_hat),4), sum(expected_loss), round(avg(CASE WHEN is_terminal THEN default_flag END),4),
    sum(CASE WHEN pd_hat >= 0.20 THEN loan_amnt ELSE 0 END)
FROM banded GROUP BY 2
UNION ALL SELECT 'risk_band', risk_band, count(*), sum(loan_amnt), round(avg(pd_hat),4),
    sum(expected_loss), round(avg(CASE WHEN is_terminal THEN default_flag END),4),
    sum(CASE WHEN pd_hat >= 0.20 THEN loan_amnt ELSE 0 END)
FROM banded GROUP BY 2
ORDER BY dimension, bucket;

SELECT dimension, count(*) AS buckets, sum(n_loans) AS loans FROM mart_portfolio_summary GROUP BY 1;
