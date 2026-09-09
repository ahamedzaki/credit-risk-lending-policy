-- Stage 1 (bronze): load the Lending Club accepted-loans CSV verbatim.
-- Everything stays VARCHAR here; typing happens in 02_staging.sql.

DROP TABLE IF EXISTS raw_loans;

CREATE TABLE raw_loans AS
SELECT *
FROM read_csv(
    '${raw_csv}',
    all_varchar = true,
    header      = true,
    sample_size = -1,
    ignore_errors = true,
    null_padding = true
);

-- Quick shape check (printed by the runner).
SELECT count(*) AS raw_row_count FROM raw_loans;
