"""Estimate LGD from the data instead of assuming it (council rec #2).

For unsecured consumer loans the textbook ~45% LGD is optimistic. Lending Club reports,
for each charged-off loan, the principal repaid before charge-off (`total_rec_prncp`) and
any post-charge-off recovery (`recoveries`). So on the CHARGED-OFF loans in the training
period:

    recovery_rate = sum(total_rec_prncp + recoveries) / sum(funded_amnt)
    LGD           = 1 - recovery_rate

Train period only, so the number that feeds Expected Loss is not estimated on the
out-of-time test set.
"""
from __future__ import annotations

import json
import pathlib

from src.config import load
from src.db import connect

ROOT = pathlib.Path(__file__).resolve().parents[1]
LGD_JSON = ROOT / "artifacts" / "lgd.json"


def estimate(con, oot_cutoff: str) -> dict:
    row = con.execute(
        f"""
        SELECT count(*)                                   AS n_charged_off,
               sum(funded_amnt)                           AS ead_sum,
               sum(coalesce(total_rec_prncp, 0)
                   + coalesce(recoveries, 0))             AS recovered_sum,
               avg(1.0 - (coalesce(total_rec_prncp, 0) + coalesce(recoveries, 0))
                         / nullif(funded_amnt, 0))        AS lgd_loan_mean
        FROM mart_loan_outcomes
        WHERE default_flag = 1
          AND issue_d < DATE '{oot_cutoff}'
        """
    ).fetchone()
    n, ead_sum, rec_sum, lgd_loan_mean = row
    recovery_rate = float(rec_sum / ead_sum)
    return {
        "lgd_data": round(1.0 - recovery_rate, 4),          # exposure-weighted (feeds EL)
        "lgd_loan_mean": round(float(lgd_loan_mean), 4),     # equal-weighted, for reference
        "recovery_rate": round(recovery_rate, 4),
        "n_charged_off_train": int(n),
        "method": "1 - sum(total_rec_prncp + recoveries) / sum(funded_amnt), charged-off train loans",
    }


def resolve(cfg: dict, con=None) -> float:
    """Effective LGD for this run: data-estimate (cached in artifacts/lgd.json) or the fixed value."""
    el = cfg["expected_loss"]
    if el["lgd_mode"] != "data":
        return float(el["lgd_fixed"])
    if LGD_JSON.exists():
        return float(json.loads(LGD_JSON.read_text())["lgd_data"])
    if con is None:
        con = connect(cfg["paths"]["duckdb"])
    return float(estimate(con, cfg["split"]["oot_cutoff"])["lgd_data"])


def main() -> float:
    cfg = load()
    con = connect(cfg["paths"]["duckdb"])
    if cfg["expected_loss"]["lgd_mode"] != "data":
        lgd = float(cfg["expected_loss"]["lgd_fixed"])
        print(f"   lgd_mode=fixed -> LGD = {lgd}")
        con.close()
        return lgd
    info = estimate(con, cfg["split"]["oot_cutoff"])
    LGD_JSON.parent.mkdir(parents=True, exist_ok=True)
    LGD_JSON.write_text(json.dumps(info, indent=2))
    print(f"   charged-off train loans : {info['n_charged_off_train']:,}")
    print(f"   recovery rate           : {info['recovery_rate']:.3f}")
    print(f"   LGD (exposure-weighted) : {info['lgd_data']:.3f}   [fixed assumption was {cfg['expected_loss']['lgd_fixed']}]")
    con.close()
    return info["lgd_data"]


if __name__ == "__main__":
    main()
