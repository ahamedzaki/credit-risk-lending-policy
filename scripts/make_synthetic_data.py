#!/usr/bin/env python3
"""Generate a synthetic Lending Club "accepted loans" CSV in the real column schema,
with a genuine latent default process so the PD model has signal to learn.

NOT real data. Use only for: local pipeline verification, CI, demoing the app before the
Kaggle download, and building the committed Streamlit sample.

    python scripts/make_synthetic_data.py --rows 8000 --out data/synthetic_accepted.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import random

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
GRADES = list("ABCDEFG")
GRADE_BASE_RATE = dict(zip(GRADES, [0.03, 0.06, 0.10, 0.15, 0.21, 0.28, 0.34]))
GRADE_RATE = dict(zip(GRADES, [6.5, 9.5, 12.5, 15.5, 18.5, 22.0, 26.0]))
PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "car",
            "major_purchase", "medical", "small_business", "other"]
HOME = ["RENT", "OWN", "MORTGAGE"]
VERIF = ["Verified", "Source Verified", "Not Verified"]
COLS = ["id", "issue_d", "term", "loan_status", "loan_amnt", "funded_amnt", "int_rate",
        "grade", "sub_grade", "emp_length", "home_ownership", "annual_inc",
        "verification_status", "purpose", "addr_state", "dti", "earliest_cr_line",
        "fico_range_low", "fico_range_high", "open_acc", "total_acc", "revol_bal",
        "revol_util", "delinq_2yrs", "inq_last_6mths", "pub_rec", "mort_acc",
        "out_prncp", "total_pymnt", "total_rec_prncp", "recoveries", "last_pymnt_d"]


def _logit(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def one_row(i: int, rng: random.Random, year: int) -> dict:
    grade = rng.choices(GRADES, [0.16, 0.24, 0.24, 0.16, 0.11, 0.06, 0.03])[0]
    fico = int(rng.gauss(710 - GRADES.index(grade) * 12, 25))
    fico = max(660, min(845, fico))
    annual_inc = round(max(12000, rng.lognormvariate(11.0, 0.55)), 0)
    loan_amnt = float(rng.choice([5000, 8000, 10000, 12000, 15000, 20000, 25000, 30000, 35000]))
    dti = round(min(45, max(0, rng.gauss(18 + GRADES.index(grade) * 1.5, 8)), 45), 2)
    revol_util = round(min(140, max(0, rng.gauss(45 + GRADES.index(grade) * 4, 22))), 1)
    emp_years = rng.choices([0, 1, 2, 3, 5, 7, 10, None], [0.08, 0.09, 0.1, 0.12, 0.15, 0.12, 0.28, 0.06])[0]
    inq6 = rng.choices([0, 1, 2, 3, 4], [0.55, 0.25, 0.12, 0.05, 0.03])[0]
    delinq = rng.choices([0, 1, 2, 3], [0.82, 0.12, 0.04, 0.02])[0]
    pub_rec = rng.choices([0, 1, 2], [0.87, 0.11, 0.02])[0]
    term = rng.choices([36, 60], [0.72, 0.28])[0]
    lti = loan_amnt / annual_inc

    # latent default process (kept modest so AUC lands in a realistic ~0.68-0.72 band)
    z = (
        -2.15
        + 3.4 * GRADE_BASE_RATE[grade]
        - 0.010 * (fico - 700)
        + 0.022 * (dti - 18)
        + 0.010 * (revol_util - 45)
        + 0.18 * inq6
        + 0.30 * delinq
        + 0.25 * pub_rec
        + 1.1 * lti
        - 0.02 * (emp_years or 0)
        + (0.15 if term == 60 else 0.0)
        + rng.gauss(0, 0.55)
    )
    pd_true = _logit(z)
    roll = rng.random()
    if roll < 0.06:                       # small slice still open
        status = "Current"
    elif rng.random() < pd_true:
        status = "Charged Off"
    else:
        status = "Fully Paid"

    funded = loan_amnt
    if status == "Charged Off":
        total_rec_prncp = round(funded * rng.uniform(0.1, 0.6), 2)
        recoveries = round((funded - total_rec_prncp) * rng.uniform(0.0, 0.35), 2)
        total_pymnt = round(total_rec_prncp + funded * rng.uniform(0.05, 0.3), 2)
        out_prncp = 0.0
    elif status == "Fully Paid":
        total_rec_prncp = funded
        recoveries = 0.0
        total_pymnt = round(funded * (1 + GRADE_RATE[grade] / 100 * (term / 24)), 2)
        out_prncp = 0.0
    else:
        total_rec_prncp = round(funded * rng.uniform(0.1, 0.5), 2)
        recoveries = 0.0
        total_pymnt = round(total_rec_prncp * 1.2, 2)
        out_prncp = round(funded - total_rec_prncp, 2)

    return {
        "id": 100000 + i,
        "issue_d": f"{rng.choice(MONTHS)}-{year}",
        "term": f" {term} months",
        "loan_status": status,
        "loan_amnt": loan_amnt,
        "funded_amnt": funded,
        "int_rate": f"{GRADE_RATE[grade] + rng.uniform(-1.5, 1.5):.2f}%",
        "grade": grade,
        "sub_grade": f"{grade}{rng.randint(1, 5)}",
        "emp_length": "n/a" if emp_years is None else ("< 1 year" if emp_years == 0
                      else f"{emp_years} years" if emp_years < 10 else "10+ years"),
        "home_ownership": rng.choice(HOME),
        "annual_inc": annual_inc,
        "verification_status": rng.choice(VERIF),
        "purpose": rng.choices(PURPOSES, [0.45, 0.22, 0.1, 0.07, 0.05, 0.04, 0.04, 0.03])[0],
        "addr_state": rng.choice(["CA", "TX", "NY", "FL", "IL", "NJ", "PA", "OH", "GA", "NC"]),
        "dti": dti,
        "earliest_cr_line": f"{rng.choice(MONTHS)}-{rng.randint(1985, 2008)}",
        "fico_range_low": fico,
        "fico_range_high": fico + 4,
        "open_acc": rng.randint(3, 22),
        "total_acc": rng.randint(8, 45),
        "revol_bal": rng.randint(0, 55000),
        "revol_util": f"{revol_util}%",
        "delinq_2yrs": delinq,
        "inq_last_6mths": inq6,
        "pub_rec": pub_rec,
        "mort_acc": rng.randint(0, 5),
        "out_prncp": out_prncp,
        "total_pymnt": total_pymnt,
        "total_rec_prncp": total_rec_prncp,
        "recoveries": recoveries,
        "last_pymnt_d": f"{rng.choice(MONTHS)}-{min(year + rng.randint(1, 3), 2018)}",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8000)
    ap.add_argument("--out", default="data/synthetic_accepted.csv")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    records = []
    # main body: issue years 2012-2016
    for i in range(args.rows):
        yr = rng.choices([2012, 2013, 2014, 2015, 2016], [0.15, 0.2, 0.22, 0.22, 0.21])[0]
        records.append(one_row(i, rng, yr))
    # snapshot anchor: a few 2019 rows (filtered out by issue_year_max, but they set
    # max(issue_d) so the maturity window mimics a real 2018Q4+ snapshot)
    for j in range(60):
        records.append(one_row(args.rows + j, rng, 2019))

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(records)
    print(f"wrote {len(records)} rows -> {args.out}")


if __name__ == "__main__":
    main()
