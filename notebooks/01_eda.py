# %% [markdown]
# # 01 — EDA & data-quality decisions
# Narrative around `sql/01_raw.sql` + `sql/02_staging.sql`. Decisions made here are
# promoted into the SQL; this notebook is the record of *why*.
#
# Prereq: `python run_pipeline.py raw stage` has been run.

# %%
import pathlib, sys
ROOT = pathlib.Path.cwd()
if not (ROOT / "config.yaml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.config import load, sql_params
from src.db import connect

cfg = load()
params = sql_params(cfg)
con = connect(cfg["paths"]["duckdb"], read_only=True)

# %% [markdown]
# ## Raw shape

# %%
raw_n = con.execute("SELECT count(*) FROM raw_loans").fetchone()[0]
print(f"raw_loans: {raw_n:,} rows")

# %% [markdown]
# ## Maturity pre-flight (36-month loans) — how the window was chosen
# Newest `issue_month` with `frac_terminal >= 0.98` sets the end of the modelling window.

# %%
sql = (ROOT / "sql" / "preflight_maturity.sql").read_text().replace("${term_months}", params["term_months"])
preflight = con.execute(sql).fetchdf()
preflight.tail(24)

# %%
thr = cfg["maturity"]["terminal_fraction_threshold"]
mature = preflight[preflight.frac_terminal >= thr]
print(f"newest month with frac_terminal >= {thr}: {mature.issue_month.iloc[-1]}")
print(f"config: oot_cutoff={cfg['split']['oot_cutoff']}  issue_year_max={cfg['maturity']['issue_year_max']}"
      f"  min_months_since_issue={cfg['maturity']['min_months_since_issue']}")

# %% [markdown]
# ## Staged table — column profile

# %%
stg = con.execute("SELECT * FROM stg_loans").fetchdf()
print(f"stg_loans: {len(stg):,} rows  ({len(stg)/raw_n:.1%} of raw)")
pd.DataFrame({
    "dtype": stg.dtypes.astype(str),
    "null_frac": stg.isna().mean().round(4),
    "n_unique": stg.nunique(),
}).sort_values("null_frac", ascending=False)

# %% [markdown]
# ## Categorical sanity — standardised values only?

# %%
for c in ["home_ownership", "purpose", "verification_status", "loan_status"]:
    print(f"--- {c} ---")
    print(stg[c].value_counts(dropna=False).head(10).to_string(), "\n")

# %% [markdown]
# ## Numeric ranges — any impossible values staging did not catch?

# %%
stg[["loan_amnt", "annual_inc", "dti", "revol_util", "fico_range_low", "fico_range_high"]].describe().round(1)

# %% [markdown]
# ## Default rate by vintage — is the window benign but usable?

# %%
by_vintage = con.execute("""
    SELECT extract('year' FROM issue_d) AS issue_year,
           count(*) AS n,
           round(avg(CASE WHEN is_terminal AND loan_status <> 'Fully Paid' THEN 1.0
                          WHEN is_terminal THEN 0.0 END), 4) AS default_rate
    FROM stg_loans GROUP BY 1 ORDER BY 1
""").fetchdf()
by_vintage

# %% [markdown]
# **DQ decisions promoted to `sql/02_staging.sql`** (fill `reports/dq_log.md` from the
# numbers above): 36-month filter, issue-year window, `min_months_since_issue`, impossible-
# value guards on `loan_amnt` / `dti` / `revol_util`, de-dupe on `id`, drop
# `Does not meet the credit policy%`. Median imputation for the nulls shown above happens
# inside the model pipeline, not staging.

# %%
con.close()
