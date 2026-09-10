"""Shared data loading for the multi-page dashboard.

Reads the pipeline's Parquet marts and JSON artefacts. On Streamlit Community Cloud the
full marts are absent, so it falls back to the committed 50k-row sample and the committed
artefact JSONs.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd
import streamlit as st
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())


@st.cache_data
def simulator_base() -> tuple[pd.DataFrame, str]:
    for name in ("simulator_base.parquet", "simulator_base_sample.parquet"):
        p = ROOT / "exports" / name
        if p.exists():
            return pd.read_parquet(p), name
    st.error("No exports/simulator_base*.parquet — run `python run_pipeline.py all`.")
    st.stop()


@st.cache_data
def portfolio_summary() -> pd.DataFrame | None:
    p = ROOT / "exports" / "portfolio_summary.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data
def artifact(name: str) -> dict | None:
    p = ROOT / "artifacts" / name
    return json.loads(p.read_text()) if p.exists() else None


def money(x: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= div:
            return f"${x / div:,.1f}{unit}"
    return f"${x:,.0f}"


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def data_source_note(src: str) -> None:
    if src.endswith("sample.parquet"):
        st.info("Running on the committed 50k-row sample (full marts not present in this "
                "environment). Figures are directional; rebuild the pipeline for exact totals.")
