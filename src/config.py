"""Load config.yaml and expose it as a plain dict + a flat placeholder map for SQL."""
from __future__ import annotations

import pathlib
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


def load() -> dict:
    with open(CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh)
    # resolve paths relative to repo root
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = str((ROOT / v).resolve())
    cfg["data"]["raw_csv"] = str((ROOT / cfg["data"]["raw_csv"]).resolve())
    return cfg


def sql_params(cfg: dict) -> dict[str, str]:
    """Flat string map for ${...} substitution in the .sql files."""
    return {
        "raw_csv": cfg["data"]["raw_csv"],
        "term_months": str(cfg["maturity"]["term_months"]),
        "issue_year_min": str(cfg["maturity"]["issue_year_min"]),
        "issue_year_max": str(cfg["maturity"]["issue_year_max"]),
        "min_months_since_issue": str(cfg["maturity"]["min_months_since_issue"]),
        "oot_cutoff": cfg["split"]["oot_cutoff"],
        # placeholder default; the `lgd` / `marts` stages overwrite it with the effective
        # value (data-estimate when expected_loss.lgd_mode == "data").
        "lgd": str(cfg["expected_loss"].get("lgd_fixed", cfg["expected_loss"].get("lgd", 0.45))),
        "ead_mode": cfg["expected_loss"]["ead_mode"],
    }
