# Star schema — ER diagram

Dimensional model built by `schema/star_schema.sql` as DuckDB **views** over the clean
marts (no data is copied). Grain of `FACT_LOAN`: one row per loan.

```mermaid
erDiagram
    DIM_DATE ||--o{ FACT_LOAN : "issued on"
    DIM_BORROWER ||--o{ FACT_LOAN : "borrowed by"
    DIM_LOAN_PRODUCT ||--o{ FACT_LOAN : "is a"

    DIM_DATE {
        date   date_key PK
        int    year
        int    quarter
        int    month
        string year_month
    }
    DIM_BORROWER {
        bigint loan_key PK
        string home_ownership
        string verification_status
        string addr_state
        string income_band
        int    emp_length_years
    }
    DIM_LOAN_PRODUCT {
        string product_key PK
        string purpose
        string lc_grade
        int    term_months
    }
    FACT_LOAN {
        bigint loan_key PK
        date   date_key FK
        string product_key FK
        string split_set
        double loan_amnt
        double fico_mid
        double dti
        double int_rate
        double pd_hat
        double ead
        double lgd
        double expected_loss
        int    default_flag
        bool   is_terminal
    }
```

## Source lineage

| Star object | Built from |
|---|---|
| `DIM_DATE` | distinct `issue_d` in `mart_loan_features` |
| `DIM_BORROWER` | `mart_loan_features` (one row per loan — Lending Club has no customer id) |
| `DIM_LOAN_PRODUCT` | `mart_loan_features.purpose` × `mart_loan_benchmark.lc_grade` |
| `FACT_LOAN` | `mart_loan_features` + `mart_loan_benchmark` + `mart_scores` + `mart_loan_el` + `mart_loan_outcomes` |

> Note: because Lending Club data has no borrower identifier, `DIM_BORROWER` is 1:1 with
> `FACT_LOAN`. In a real lender this dimension would be keyed on `customer_id` and a
> borrower could appear on many loans.
