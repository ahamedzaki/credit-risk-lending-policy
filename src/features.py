"""The feature contract (spec §2.2, §4).

ALLOWLIST is the single source of truth for what enters the PD model. Anything not
listed here is excluded by construction. `grade`, `sub_grade`, `int_rate` are NOT here
on purpose — they are Lending Club's own risk-model output (circularity) and live in
mart_loan_benchmark, used only for the benchmark and the profit calculation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC: list[str] = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "emp_length_years",
    "fico_mid",
    "credit_history_months",
    "open_acc",
    "total_acc",
    "revol_bal",
    "revol_util",
    "delinq_2yrs",
    "inq_last_6mths",
    "pub_rec",
    "mort_acc",
]

CATEGORICAL: list[str] = [
    "home_ownership",
    "purpose",
    "verification_status",
    # addr_state is intentionally omitted by default (fairness, spec §6).
    # Add it here only if you also add the disparate-impact check.
]

ALLOWLIST: list[str] = NUMERIC + CATEGORICAL

# columns that must never appear in the modelling frame — asserted at train time
FORBIDDEN = {
    "grade", "sub_grade", "lc_grade", "lc_sub_grade", "int_rate",
    "last_pymnt_d", "last_pymnt_amnt", "total_pymnt", "total_rec_prncp", "total_rec_int",
    "recoveries", "collection_recovery_fee", "out_prncp", "out_prncp_inv",
    "last_fico_range_high", "last_fico_range_low", "next_pymnt_d",
    "loan_status", "default_flag", "funded_amnt",
}


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Light, leakage-safe derived features. Everything here uses only allowlist inputs."""
    out = df.copy()
    if "annual_inc" in out.columns:
        out["annual_inc"] = out["annual_inc"].clip(lower=0)
        out["log_annual_inc"] = np.log1p(out["annual_inc"].fillna(0.0))
    if {"loan_amnt", "annual_inc"}.issubset(out.columns):
        inc = out["annual_inc"].replace(0, np.nan)
        out["loan_to_income"] = out["loan_amnt"] / inc
    return out


DERIVED_NUMERIC = ["log_annual_inc", "loan_to_income"]


def build_preprocessor() -> ColumnTransformer:
    num_cols = NUMERIC + DERIVED_NUMERIC
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, num_cols), ("cat", categorical, CATEGORICAL)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def assert_no_leakage(df: pd.DataFrame) -> None:
    bad = FORBIDDEN.intersection(df.columns)
    if bad:
        raise AssertionError(f"Leakage guard: forbidden columns in modelling frame: {sorted(bad)}")
