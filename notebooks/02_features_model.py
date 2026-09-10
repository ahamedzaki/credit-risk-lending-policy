# %% [markdown]
# # 02 — Features & PD model
# Narrative around `src/train.py`: the leakage guard, the out-of-time split, calibration,
# the three-way benchmark (HGB vs logistic vs grade-alone), and within-grade re-ranking.
# Uses the exact pipeline objects the runner uses.
#
# Prereq: `python run_pipeline.py features` has been run.

# %%
import pathlib, sys
ROOT = pathlib.Path.cwd()
if not (ROOT / "config.yaml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
np.seterr(all="ignore")
from sklearn.metrics import roc_auc_score

from src import evaluate, features
from src.config import load
from src.db import connect
from src.train import _fit_calibrated, _grade_benchmark

cfg = load()
con = connect(cfg["paths"]["duckdb"], read_only=True)

# %% [markdown]
# ## Modelling frame + leakage guard

# %%
frame = con.execute("SELECT * FROM v_model_frame").fetchdf()
frame = features.engineer(frame)
features.assert_no_leakage(frame.drop(columns=["default_flag"]))          # raises if a forbidden col slipped in
bench = con.execute("SELECT loan_id, lc_grade FROM mart_loan_benchmark").fetchdf()
frame = frame.merge(bench, on="loan_id", how="left")
frame["split_set"].value_counts()

# %% [markdown]
# ## Out-of-time split (by issue_d, never random)

# %%
train = frame[frame.split_set == "train"].reset_index(drop=True)
test = frame[frame.split_set == "test"].reset_index(drop=True)
cols = features.ALLOWLIST + features.DERIVED_NUMERIC
ytr, yte = train.default_flag.to_numpy(), test.default_flag.to_numpy()
print(f"train {len(train):,} (DR {ytr.mean():.3f})   test {len(test):,} (DR {yte.mean():.3f})")
print("features:", cols)

# %% [markdown]
# ## Fit the primary model + inspect calibration

# %%
model = _fit_calibrated(cfg["model"]["type"], train[cols], ytr, cfg)
p_te = model.predict_proba(test[cols])[:, 1]
evaluate.summary(yte, p_te)

# %%
fig_dir = cfg["paths"]["figures_dir"]
evaluate.plot_calibration(yte, p_te, f"{fig_dir}/nb_calibration.png", "Calibration — out-of-time test")
dec = evaluate.decile_table(yte, p_te)
dec.round(4)

# %% [markdown]
# ## Three-way comparison: primary vs the other model type vs grade-alone

# %%
prim = cfg["model"]["type"]
sec = "logistic" if prim == "hgb" else "hgb"
sec_model = _fit_calibrated(sec, train[cols], ytr, cfg)
p_sec = sec_model.predict_proba(test[cols])[:, 1]
grade_bm = _grade_benchmark(train[["lc_grade"]], test[["lc_grade"]], ytr, yte)
pd.Series({
    f"{prim} (primary)": roc_auc_score(yte, p_te),
    f"{sec}": roc_auc_score(yte, p_sec),
    "grade-only": grade_bm["auc"],
}).round(4)

# %% [markdown]
# ## Does the model re-rank *within* a grade?
# Observed default rate by grade x model-PD-decile. If the model adds nothing over grade,
# rows are flat.

# %%
t = test.copy()
t["pd_hat"] = p_te
t["pd_decile"] = pd.qcut(t.pd_hat.rank(method="first"), 10, labels=False)
(t.groupby(["lc_grade", "pd_decile"]).default_flag.mean().unstack().round(3))

# %% [markdown]
# ## Calibration by grade — where does PD miss?

# %%
(t.groupby("lc_grade")
   .agg(n=("loan_id", "size"), pred_pd=("pd_hat", "mean"), obs_dr=("default_flag", "mean"))
   .round(3))

# %%
con.close()
