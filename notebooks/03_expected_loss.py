# %% [markdown]
# # 03 — Expected loss & portfolio concentration
# Narrative around `sql/06_marts.sql` and `app/sim_core.py`. Where is risk concentrated,
# and what does the policy trade-off look like?
#
# Prereq: `python run_pipeline.py all` has been run.

# %%
import pathlib, sys
ROOT = pathlib.Path.cwd()
if not (ROOT / "config.yaml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
np.seterr(all="ignore")
from src.config import load
from src.db import connect
from app.sim_core import approved_metrics, profit_curve, best_policy, money

cfg = load()
con = connect(cfg["paths"]["duckdb"], read_only=True)

# %%
el = con.execute("SELECT * FROM mart_loan_el").fetchdf()
print(f"{len(el):,} loans")
el[["pd_hat", "ead", "expected_loss"]].describe().round(2)

# %% [markdown]
# ## Concentration: share of exposure vs share of expected loss
# The memo headline: "segment X is A% of exposure but B% of expected loss".

# %%
base = con.execute("SELECT * FROM mart_simulator_base").fetchdf()

def concentration(col):
    g = base.groupby(col).agg(n=("loan_id", "size"), exposure=("loan_amnt", "sum"),
                              el=("expected_loss", "sum"), avg_pd=("pd_hat", "mean"),
                              obs_dr=("default_flag", "mean"))
    g["exposure_share"] = g.exposure / g.exposure.sum()
    g["el_share"] = g.el / g.el.sum()
    g["el_per_exposure"] = (g.el_share / g.exposure_share).round(2)
    return g.sort_values("el_per_exposure", ascending=False).round(3)

concentration("lc_grade")
# %%
concentration("purpose")
# %%
concentration("vintage_year")

# %% [markdown]
# ## Policy trade-off — the same math the simulator uses (`app/sim_core.py`)

# %%
SIM = cfg["simulator"]
args = dict(fico_min=SIM["default_min_fico"], cost_of_funds=SIM["cost_of_funds"],
            horizon=SIM["horizon_years"], amort_factor=SIM["amort_factor"],
            servicing_cost=SIM["servicing_cost"])
rows = []
for name, cut in [("Conservative", 0.08), ("Current", 0.15), ("Growth", 0.20)]:
    m = approved_metrics(base, cut, **args)
    rows.append({"policy": name, "pd_cut": cut, **{k: m[k] for k in
                 ("approval_rate", "volume", "exp_default_rate", "exp_loss", "exp_profit")}})
curve = profit_curve(base, args["fico_min"], args["cost_of_funds"], args["horizon"],
                     n_points=120, amort_factor=args["amort_factor"], servicing_cost=args["servicing_cost"])
bp = best_policy(curve)
m = approved_metrics(base, bp["pd_cut"], **args)
rows.append({"policy": "Profit-max", "pd_cut": round(bp["pd_cut"], 3), **{k: m[k] for k in
             ("approval_rate", "volume", "exp_default_rate", "exp_loss", "exp_profit")}})
pd.DataFrame(rows).assign(
    volume=lambda d: d.volume.map(money), exp_loss=lambda d: d.exp_loss.map(money),
    exp_profit=lambda d: d.exp_profit.map(money),
    approval_rate=lambda d: (d.approval_rate * 100).round(1),
    exp_default_rate=lambda d: (d.exp_default_rate * 100).round(2))

# %%
imax = curve.exp_profit.idxmax()
print(f"profit peaks at PD<{curve.pd_cut[imax]:.3f} / approval {curve.approval_rate[imax]:.1%} "
      f"= {money(curve.exp_profit.max())}")
print(f"at 100% approval profit falls to {money(curve.exp_profit.iloc[-1])} "
      f"-> the curve has a real interior optimum")

# %% [markdown]
# ## Scenario stress (a single table — the full layer is deferred, spec §1)
# Take the book approved under the CURRENT policy (PD<0.15), then stress: multiply every
# PD by a factor and re-price *that same population* (no re-underwriting).

# %%
LGD = cfg["expected_loss"]["lgd"]
booked = base[(base.pd_hat < 0.15) & (base.fico_mid >= args["fico_min"])].copy()
stress = []
for f in (1.0, 1.2, 1.5):
    b = booked.copy()
    b["pd_hat"] = (b.pd_hat * f).clip(upper=1.0)
    b["expected_loss"] = b.pd_hat * b.ead * LGD
    # approve everyone in the fixed book (cut-off 1.0), keep all other assumptions
    m = approved_metrics(b, 1.0, fico_min=0, cost_of_funds=args["cost_of_funds"],
                         horizon=args["horizon"], amort_factor=args["amort_factor"],
                         servicing_cost=args["servicing_cost"])
    stress.append({"pd_x": f, "mean_pd": round(b.pd_hat.mean(), 3),
                   "exp_loss": money(m["exp_loss"]), "exp_profit": money(m["exp_profit"])})
pd.DataFrame(stress)

# %%
con.close()
