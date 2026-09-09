# %% [markdown]
# # 03 — Expected loss & portfolio concentration
# Narrative around `sql/06_marts.sql`. Where is risk concentrated?

# %%
import pandas as pd
from src.config import load
from src.db import connect

cfg = load()
con = connect(cfg["paths"]["duckdb"])

# %%
el = con.execute("SELECT * FROM mart_loan_el").fetchdf()
el[["pd_hat", "ead", "expected_loss"]].describe()

# %% [markdown]
# ## Concentration: share of exposure vs share of expected loss
# The headline finding for the memo looks like "segment X is A% of exposure but B% of EL".
# %%
base = con.execute("SELECT * FROM mart_simulator_base").fetchdf()
def concentration(col):
    g = base.groupby(col).agg(exposure=("loan_amnt", "sum"), el=("expected_loss", "sum"),
                              n=("loan_id", "size"), avg_pd=("pd_hat", "mean"))
    g["exposure_share"] = g.exposure / g.exposure.sum()
    g["el_share"] = g.el / g.el.sum()
    g["el_to_exposure"] = g.el_share / g.exposure_share
    return g.sort_values("el_to_exposure", ascending=False).round(3)

concentration("lc_grade")
# %%
concentration("purpose")
# %%
concentration("vintage_year")

# %% [markdown]
# ## Simple scenario stress (deferred layer — one table only)
# Multiply PD by a factor, re-aggregate EL and profit.
# %%
COF, T, LGD = cfg["simulator"]["cost_of_funds"], cfg["simulator"]["horizon_years"], cfg["expected_loss"]["lgd"]
rows = []
for f in (1.0, 1.2, 1.5):
    pdf = (base.pd_hat * f).clip(upper=1.0)
    el_s = (pdf * base.ead * LGD).sum()
    interest = (base.loan_amnt * base.int_rate * T).sum()
    funding = (base.loan_amnt * COF * T).sum()
    rows.append({"pd_factor": f, "expected_loss": el_s, "expected_profit": interest - funding - el_s})
pd.DataFrame(rows)

# %%
con.close()
