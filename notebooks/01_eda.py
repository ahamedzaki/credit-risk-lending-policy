# %% [markdown]
# # 01 — EDA & data-quality decisions
# Run as a script or open with Jupyter (`# %%` cells). Decisions made here are promoted
# into `sql/02_staging.sql`; this notebook is the narrative, not the source of truth.

# %%
import duckdb
import pandas as pd

from src.config import load, sql_params
from src.db import run_sql_file, connect

cfg = load()
params = sql_params(cfg)
con = connect(cfg["paths"]["duckdb"])

# %% [markdown]
# ## Raw shape & the maturity pre-flight
# (assumes `python run_pipeline.py raw` has been run)

# %%
print(con.execute("SELECT count(*) FROM raw_loans").fetchone())
preflight = con.execute(open("sql/preflight_maturity.sql").read()
                        .replace("${term_months}", params["term_months"])).fetchdf()
preflight.tail(30)

# %% [markdown]
# **Decision:** pick the newest `issue_month` with `frac_terminal >= 0.98`; set
# `config.split.oot_cutoff` a few months earlier and `config.maturity.issue_year_max`
# to that year. Record the choice in `reports/memo.md`.

# %% [markdown]
# ## Column profiling on the staged table
# (assumes `python run_pipeline.py stage` has been run)

# %%
stg = con.execute("SELECT * FROM stg_loans").fetchdf()
profile = pd.DataFrame({
    "dtype": stg.dtypes.astype(str),
    "null_frac": stg.isna().mean().round(4),
    "n_unique": stg.nunique(),
})
profile.sort_values("null_frac", ascending=False)

# %%
# categorical sanity
for c in ["home_ownership", "purpose", "verification_status", "loan_status"]:
    print(c)
    print(stg[c].value_counts(dropna=False).head(12), "\n")

# %%
# numeric ranges — look for impossible values that staging did NOT already filter
stg[["loan_amnt", "annual_inc", "dti", "revol_util", "fico_mid" if "fico_mid" in stg else "fico_range_low"]].describe()

# %% [markdown]
# ## Default rate by vintage — confirm the window is benign but usable
# %%
con.execute("""
    SELECT extract('year' FROM issue_d) AS yr,
           count(*) n,
           round(avg(CASE WHEN is_terminal AND loan_status <> 'Fully Paid' THEN 1.0
                          WHEN is_terminal THEN 0.0 END), 4) AS default_rate
    FROM stg_loans GROUP BY 1 ORDER BY 1
""").fetchdf()

# %%
con.close()
