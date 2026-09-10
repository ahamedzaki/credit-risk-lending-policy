# Recommendation memo — Credit Risk & Lending Policy Analysis

> One page. Fill the bracketed values from `artifacts/metrics.json` and the simulator
> once the pipeline has run on real data.

## Question
Which applicants should we approve, and what does each policy choice cost in expected
loss versus earn in risk-adjusted profit?

## Data & method (one paragraph)
Lending Club accepted loans (`wordsforthewise` mirror, 2.26M rows), 36-month term,
mature vintages only (**issued 2012-01 … 2016-02**; ~641k terminal loans, default rate
14.9%). **~33 origination-time features** — core application fields plus credit-bureau
attributes available at underwriting (utilisation, trade-line counts and ages, recent
inquiries, prior serious delinquencies, total limits and balances). `grade` / `sub_grade`
/ `int_rate` excluded from the model. Out-of-time split: train issued before **2015-01**,
test on/after (306k / 334k). Primary PD model: calibrated **HistGradientBoosting**;
Expected Loss = PD × EAD × LGD with EAD = funded amount and LGD = 0.45 (assumption).

## Results
- **Model (HGB):** test AUC **0.691**, Gini **0.382**, KS **0.278**, Brier **0.119**,
  mean PD 0.135 vs base rate 0.132 (well calibrated in the large — deciles track observed
  rates within ~1–2 pp). Calibration curve: `reports/figures/calibration_test.png`.
- **Benchmark (spec §2.4):** grade-alone AUC **0.669**; logistic on the same features
  **0.680**; HGB **0.691**. **Both models now beat grade** — HGB by **+0.022 AUC**. With
  a minimal 17-field allowlist the linear model *lost* to grade (0.658); adding bona-fide
  origination-time bureau attributes closed and reversed the gap. Headline: with public
  application + bureau fields only, the model out-ranks Lending Club's own grade.
- **Segment calibration:** PD is still under-predicted in the low grades (grade G:
  predicted 30.5% vs observed 46.0%; grade F: 27.4% vs 42.3%; grade D: 20.7% vs 26.6%) —
  the model compresses the risk tail, and the expanded features did not fix it. Global
  isotonic calibration cannot correct a segment-specific bias; a production system would
  calibrate within risk bands. E–G together are ~4% of loans, so portfolio impact is
  limited; flagged in Limitations.
- **Concentration:** grade A is **25.6%** of exposure but only **12.9%** of expected loss;
  grades **C + D are 36.5% of exposure but 49.4% of expected loss** — risk is concentrated
  one notch below the middle of the book.

## Policy recommendation
Min FICO 660, LGD 0.45, cost of funds 4%, T = 3 yr, amort factor 0.52, servicing 1.2%/yr.

| Policy | Approve if PD < | Approval rate | Volume | Exp. default rate | Exp. loss | Exp. profit (T=3) |
|---|---|---|---|---|---|---|
| Conservative | 0.08 | 27.0% | $2.5B | 5.2% | $58.2M | $17.8M |
| Current-equivalent | 0.15 | 64.0% | $5.4B | 8.8% | $207.4M | $54.0M |
| **Profit-maximising** | **0.173** | **72.9%** | **$6.1B** | **9.7%** | **$255.5M** | **$56.8M** |
| Growth | 0.20 | 80.8% | $6.7B | 10.6% | $304.7M | $54.0M |

**Recommendation:** loosen the PD cut-off from **0.15 → ~0.17** (+~9 pts approval,
+$0.7B volume). Expected profit rises ~$3M while expected loss rises ~$48M — the extra
margin on the newly-approved band still clears its expected loss. Do **not** go to 0.20:
approval keeps climbing but expected profit *falls back* (~‑$3M vs the 0.173 optimum)
because the marginal loans past ~0.17 lose money. The profit-vs-approval curve peaks and
turns over — see `reports/figures/` / the simulator.

## Assumptions & limitations
- **Label maturity** — 36-month loans, issue date ≤ 2016-02; loans still *Current* after
  the window are dropped. Removes right-censoring bias; tilts the sample to earlier vintages.
- **Reject inference** — data is *accepted, funded* loans only. PD is calibrated to approved
  borrowers, not the through-the-door population; loosening the cut-off in the simulator
  extrapolates. Not fixable with this data.
- **Macro regime** — the issue window contains no recession; out-of-regime calibration is
  fragile. This is why multi-scenario stress testing is deferred, not faked.
- **Model** — HGB beats grade by +0.022 AUC (logistic by +0.011) on ~33 origination-time
  fields. The low grades (D–G) are under-predicted (risk-tail compression) and global
  isotonic calibration does not fix it. Treat PD as a ranking tool and a well-calibrated
  probability in the bulk of the book, but not a precise low-grade probability.
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
  by state / income band is not done (future work).
- **Dataset** — Kaggle mirror `wordsforthewise/lending-club`, snapshot 2018Q4.

## Deferred (future work)
Early-warning system for post-origination deterioration; risk-state migration matrices;
multi-scenario macro stress testing.
