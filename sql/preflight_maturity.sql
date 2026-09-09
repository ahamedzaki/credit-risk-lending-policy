-- Step 0 (spec §2.1): which issue months are mature enough to label?
-- Pick the newest issue_month whose frac_terminal >= maturity.terminal_fraction_threshold,
-- then set config.maturity.issue_year_max / config.split.oot_cutoff accordingly.

SELECT
    strftime(try_strptime(trim(issue_d), '%b-%Y'), '%Y-%m')                       AS issue_month,
    count(*)                                                                       AS n,
    round(avg(CASE WHEN trim(loan_status) IN ('Fully Paid', 'Charged Off', 'Default')
                   THEN 1.0 ELSE 0.0 END), 4)                                      AS frac_terminal,
    round(avg(CASE WHEN trim(loan_status) = 'Charged Off' THEN 1.0 ELSE 0.0 END), 4) AS frac_charged_off
FROM raw_loans
WHERE trim(term) = '${term_months} months'
  AND try_strptime(trim(issue_d), '%b-%Y') IS NOT NULL
GROUP BY 1
ORDER BY 1;
