# Recommendation memo — Credit Risk & Lending Policy Analysis

> One page. Fill the bracketed values from `artifacts/metrics.json` and the simulator
> once the pipeline has run on real data.

## Question
Which applicants should we approve, and what does each policy choice cost in expected
loss versus earn in risk-adjusted profit?

## Data & method (one paragraph)
Lending Club accepted loans, 36-month term, mature vintages only ([issue window] — see
Limitations). Origination-time features only; `grade` / `sub_grade` / `int_rate` excluded
from the model. Out-of-time split: train issued before [oot_cutoff], test on/after.
Calibrated logistic regression for PD; Expected Loss = PD × EAD × LGD with EAD = funded
amount and LGD = 0.45 (assumption).

## Results
- **Model:** test AUC [x.xx], KS [x.xx], Brier [x.xx]. Calibration: see
  `reports/figures/calibration_test.png`.
- **Benchmark:** grade-alone AUC [x.xx] → the model adds **[+x.xx] AUC** and re-ranks
  within grade (worst vs best decile of grade B: [x]% vs [x]% observed default).
- **Concentration:** [segment] is [A]% of exposure but [B]% of expected loss.

## Policy recommendation
| Policy | Approve if PD < | Approval rate | Volume | Exp. default rate | Exp. loss | Exp. profit (T=3) |
|---|---|---|---|---|---|---|
| Conservative | [0.08] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Current-equivalent | [0.15] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Growth | [0.20] | [ ] | [ ] | [ ] | [ ] | [ ] |
| **Profit-maximising** | **[best]** | [ ] | [ ] | [ ] | [ ] | **[ ]** |

**Recommendation:** [e.g. move the cut-off from 0.15 to 0.[xx]; this trades $[x]M more
expected loss for $[x]M more expected profit at +[x] pts approval].

## Assumptions & limitations
- **Label maturity** — 36-month loans, issue date ≤ [cutoff]; loans still *Current* after
  the window are dropped. Removes right-censoring bias; tilts the sample to earlier vintages.
- **Reject inference** — data is *accepted, funded* loans only. PD is calibrated to approved
  borrowers, not the through-the-door population; loosening the cut-off in the simulator
  extrapolates. Not fixable with this data.
- **Macro regime** — the issue window contains no recession; out-of-regime calibration is
  fragile. This is why multi-scenario stress testing is deferred, not faked.
- **LGD** fixed at 0.45 → Expected Loss is a rescaling of PD; segment analysis still valid.
- **EAD** = funded amount, no amortisation → overstates exposure for seasoned loans.
- **Profit model** — simple interest over T = 3 years, cost of funds 4%, no prepayment or
  servicing cost. Directional, not P&L-grade.
- **Circularity** — `grade` / `sub_grade` / `int_rate` excluded from features; `int_rate`
  used only in the profit calculation.
- **Fairness** — `addr_state` excluded from the model by default; a disparate-impact check
  by state / income band is [done / not done].
- **Dataset** — Kaggle mirror `wordsforthewise/lending-club`, snapshot 2018Q4.

## Deferred (future work)
Early-warning system for post-origination deterioration; risk-state migration matrices;
multi-scenario macro stress testing.
