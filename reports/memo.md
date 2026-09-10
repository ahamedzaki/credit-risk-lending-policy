# Recommendation memo — Credit Risk & Lending Policy Analysis

> One page. Fill the bracketed values from `artifacts/metrics.json` and the simulator
> once the pipeline has run on real data.

## Question
Which applicants should we approve, and what does each policy choice cost in expected
loss versus earn in risk-adjusted profit?

## Data & method (one paragraph)
Lending Club accepted loans (`wordsforthewise` mirror, 2.26M rows), 36-month term,
mature vintages only (**issued 2012-01 … 2016-02**; ~641k terminal loans, default rate
14.9%). Origination-time features only; `grade` / `sub_grade` / `int_rate` excluded from
the model. Out-of-time split: train issued before **2015-01**, test on/after
(306k / 334k). Primary PD model: calibrated **HistGradientBoosting**; Expected Loss =
PD × EAD × LGD with EAD = funded amount and LGD = 0.45 (assumption).

## Results
- **Model (HGB):** test AUC **0.670**, KS **0.247**, Brier **0.121**, mean PD 0.133 vs
  base rate 0.132 (well calibrated in the large). Calibration curve:
  `reports/figures/calibration_test.png`.
- **Benchmark:** grade-alone AUC **0.669**; a linear model on the same features tops out
  at **0.658**. So HGB **edges past grade (+0.001 AUC)**; logistic **loses to it**. The
  honest read: application-time features carry only slightly more separable signal than
  LC's grade already encodes.
- **Segment calibration:** PD is under-predicted in the low grades (grade G: predicted
  24.5% vs observed 41.0%; grade F: 21.9% vs 33.8%) — the model compresses the risk tail.
  Isotonic calibration is global and cannot fix a segment-specific bias. Small books
  (E–G together are 3.9% of loans) so portfolio impact is limited; flagged in Limitations.
- **Concentration:** grade A is **25.6%** of exposure but only **14.2%** of expected loss;
  grades **C + D are 36.5% of exposure but 47.4% of expected loss** — risk is concentrated
  one notch below the middle of the book.

## Policy recommendation
Min FICO 660, LGD 0.45, cost of funds 4%, T = 3 yr, amort factor 0.52, servicing 1.2%/yr.

| Policy | Approve if PD < | Approval rate | Volume | Exp. default rate | Exp. loss | Exp. profit (T=3) |
|---|---|---|---|---|---|---|
| Conservative | 0.08 | 24.9% | $2.4B | 5.5% | $57.9M | $12.6M |
| Current-equivalent | 0.15 | 63.3% | $5.4B | 9.1% | $214.3M | $49.5M |
| **Profit-maximising** | **0.17** | **72.1%** | **$6.1B** | **10.0%** | **$262.1M** | **$51.6M** |
| Growth | 0.20 | 83.0% | $6.9B | 11.1% | $328.9M | $47.5M |

**Recommendation:** loosen the PD cut-off from **0.15 → ~0.17** (+~9 pts approval,
+$0.7B volume). Expected profit rises ~$2M while expected loss rises ~$48M — the extra
margin on the newly-approved band still clears its expected loss. Do **not** go to 0.20:
approval keeps climbing but expected profit *falls* (~‑$4M vs the 0.17 optimum) because
the marginal loans past ~0.17 lose money. The profit-vs-approval curve peaks and turns
over — see `reports/figures/` / the simulator.

## Assumptions & limitations
- **Label maturity** — 36-month loans, issue date ≤ [cutoff]; loans still *Current* after
  the window are dropped. Removes right-censoring bias; tilts the sample to earlier vintages.
- **Reject inference** — data is *accepted, funded* loans only. PD is calibrated to approved
  borrowers, not the through-the-door population; loosening the cut-off in the simulator
  extrapolates. Not fixable with this data.
- **Macro regime** — the issue window contains no recession; out-of-regime calibration is
  fragile. This is why multi-scenario stress testing is deferred, not faked.
- **Model** — HGB beats grade by only ~0.001 AUC; a linear model loses to grade. The
  low grades (E–G) are under-predicted (risk-tail compression) and global isotonic
  calibration does not fix it. Treat PD as a ranking tool within the bulk of the book,
  not a precise low-grade probability.
- **LGD** fixed at 0.45 → Expected Loss is a rescaling of PD; segment analysis still valid.
- **EAD** = funded amount, no amortisation → overstates exposure for seasoned loans.
- **Profit model** — expected value over T = 3 yr: interest and funding accrue on
  `b · principal` (b = 0.52 amortisation factor), interest also × `(1 − PD)`; plus a flat
  1.2%/yr servicing charge; loss = `PD · EAD · LGD`. This is enough to give the curve a
  real interior optimum, but it has no cash-flow discounting, no prepayment, and a single
  blended amortisation factor. Directional, not P&L-grade.
- **Circularity** — `grade` / `sub_grade` / `int_rate` excluded from features; `int_rate`
  used only in the profit calculation.
- **Fairness** — `addr_state` excluded from the model by default; a disparate-impact check
  by state / income band is [done / not done].
- **Dataset** — Kaggle mirror `wordsforthewise/lending-club`, snapshot 2018Q4.

## Deferred (future work)
Early-warning system for post-origination deterioration; risk-state migration matrices;
multi-scenario macro stress testing.
