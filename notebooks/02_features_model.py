# %% [markdown]
# # 02 — Features & PD model
# Narrative around `src/train.py`. Shows the leakage guard, the out-of-time split,
# calibration, and the grade benchmark. Re-uses the exact pipeline the runner uses.

# %%
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import evaluate, features
from src.config import load
from src.db import connect

cfg = load()
con = connect(cfg["paths"]["duckdb"])

# %%
frame = con.execute("SELECT * FROM v_model_frame").fetchdf()
frame = features.engineer(frame)
features.assert_no_leakage(frame.drop(columns=["default_flag"]))
frame["split_set"].value_counts()

# %% [markdown]
# ## Out-of-time split (never random)
# %%
train = frame[frame.split_set == "train"]
test = frame[frame.split_set == "test"]
print(len(train), len(test), train.default_flag.mean().round(4), test.default_flag.mean().round(4))

# %% [markdown]
# ## Fit + inspect calibration
# %%
from src.train import _fit_pd_model

cols = features.ALLOWLIST + features.DERIVED_NUMERIC
model = _fit_pd_model(train[cols], train.default_flag.to_numpy(), cfg)
p_te = model.predict_proba(test[cols])[:, 1]
evaluate.summary(test.default_flag.to_numpy(), p_te)

# %%
evaluate.plot_calibration(test.default_flag.to_numpy(), p_te,
                          f"{cfg['paths']['figures_dir']}/nb_calibration.png", "Calibration (nb)")
evaluate.decile_table(test.default_flag.to_numpy(), p_te)

# %% [markdown]
# ## Grade benchmark — does the model re-rank within grade?
# %%
bench = con.execute("SELECT loan_id, lc_grade FROM mart_loan_benchmark").fetchdf()
t = test.merge(bench, on="loan_id")
t["pd_hat"] = p_te
(t.assign(decile=pd.qcut(t.pd_hat.rank(method="first"), 10, labels=False))
   .groupby(["lc_grade", "decile"]).default_flag.mean().unstack().round(3))

# %%
con.close()
