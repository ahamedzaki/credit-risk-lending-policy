"""Estimate LGD from the data instead of assuming it (council rec #2).

For unsecured consumer loans the textbook ~45% LGD is optimistic. Lending Club reports,
for each charged-off loan, the principal repaid before charge-off (`total_rec_prncp`) and
any post-charge-off recovery (`recoveries`). So on the CHARGED-OFF loans in the training
period:

    recovery_rate = sum(total_rec_prncp + recoveries) / sum(funded_amnt)
    LGD           = 1 - recovery_rate

Train period only, so the number that feeds Expected Loss is not estimated on the
out-of-time test set.

Expected Loss is segmented by grade (audit finding: a single flat LGD hides real
dispersion — 44% on grade A vs 60% on grade G). Grade-level LGD uses the same formula
above, restricted to that grade, then Buhlmann credibility-weighted toward the flat
portfolio LGD so thin tail grades (e.g. G, ~70 charged-off loans) don't get a noisy raw
point estimate:

    credibility_g   = n_g / (n_g + K)          K = full_credibility_n (config)
    lgd_credible_g  = credibility_g * lgd_raw_g + (1 - credibility_g) * lgd_flat

This is standard actuarial credibility theory, not an arbitrary cutoff — it shrinks
sparse segments toward the population estimate in proportion to how much data actually
supports them, rather than a hard include/exclude threshold.
"""
from __future__ import annotations

import json
import pathlib

from src.config import load
from src.db import connect

ROOT = pathlib.Path(__file__).resolve().parents[1]
LGD_JSON = ROOT / "artifacts" / "lgd.json"
FULL_CREDIBILITY_N = 2000  # charged-off loans at which a grade's own LGD gets ~full weight


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


def estimate_by_grade(con, oot_cutoff: str, lgd_flat: float) -> list[dict]:
    """Per-grade LGD (same formula as `estimate`), credibility-weighted toward `lgd_flat`."""
    rows = con.execute(
        f"""
        SELECT b.lc_grade                                              AS grade,
               count(*)                                                AS n_charged_off,
               sum(o.funded_amnt)                                      AS ead_sum,
               sum(coalesce(o.total_rec_prncp, 0) + coalesce(o.recoveries, 0)) AS recovered_sum
        FROM mart_loan_outcomes o
        JOIN mart_loan_benchmark b USING (loan_id)
        WHERE o.default_flag = 1 AND o.issue_d < DATE '{oot_cutoff}'
        GROUP BY b.lc_grade
        ORDER BY b.lc_grade
        """
    ).fetchdf()
    out = []
    for _, r in rows.iterrows():
        n = int(r["n_charged_off"])
        lgd_raw = 1.0 - float(r["recovered_sum"] / r["ead_sum"]) if r["ead_sum"] else lgd_flat
        credibility = n / (n + FULL_CREDIBILITY_N)
        lgd_credible = credibility * lgd_raw + (1 - credibility) * lgd_flat
        out.append({
            "grade": r["grade"],
            "n_charged_off_train": n,
            "lgd_raw": round(lgd_raw, 4),
            "credibility": round(credibility, 3),
            "lgd": round(lgd_credible, 4),  # credibility-weighted — this is what feeds EL
        })
    return out


def write_grade_table(con, by_grade: list[dict]) -> None:
    """Materialize a small `lgd_by_grade(grade, lgd)` table for 06_marts.sql to join against."""
    con.execute("DROP TABLE IF EXISTS lgd_by_grade")
    import pandas as pd  # local import: only needed here, keeps module import light
    df = pd.DataFrame(by_grade)[["grade", "lgd", "n_charged_off_train"]]
    con.register("_lgd_by_grade_df", df)
    con.execute("CREATE TABLE lgd_by_grade AS SELECT grade, lgd, n_charged_off_train FROM _lgd_by_grade_df")
    con.unregister("_lgd_by_grade_df")


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
    by_grade = estimate_by_grade(con, cfg["split"]["oot_cutoff"], info["lgd_data"])
    write_grade_table(con, by_grade)
    info["by_grade"] = by_grade
    info["by_grade_method"] = (
        f"Buhlmann credibility toward the flat LGD, full_credibility_n={FULL_CREDIBILITY_N} "
        "charged-off loans. This segmented table feeds Expected Loss (06_marts.sql); "
        "`lgd_data` above remains the flat fallback for any grade not in the table."
    )
    LGD_JSON.parent.mkdir(parents=True, exist_ok=True)
    LGD_JSON.write_text(json.dumps(info, indent=2))
    print(f"   charged-off train loans : {info['n_charged_off_train']:,}")
    print(f"   recovery rate           : {info['recovery_rate']:.3f}")
    print(f"   LGD (exposure-weighted, flat fallback) : {info['lgd_data']:.3f}   [fixed assumption was {cfg['expected_loss']['lgd_fixed']}]")
    print("   LGD by grade (credibility-weighted, feeds EL):")
    for g in by_grade:
        print(f"      {g['grade']}: {g['lgd']:.3f}  (raw {g['lgd_raw']:.3f}, credibility {g['credibility']:.2f}, n={g['n_charged_off_train']:,})")
    con.close()
    return info["lgd_data"]


if __name__ == "__main__":
    main()
