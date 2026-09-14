# Recommendation memo — Credit Risk & Lending Policy Analysis

## Question
Which applicants should we approve, and what does each policy choice cost in expected
loss versus earn in risk-adjusted profit?

## Data & method
Lending Club accepted loans (`wordsforthewise` mirror, 2.26M rows), 36-month term,
mature vintages only (**issued 2012-01 … 2016-02**; ~641k terminal loans, default rate
14.9%). **~33 origination-time features** — core application fields plus credit-bureau
attributes available at underwriting (utilisation, trade-line counts and ages, recent
inquiries, prior serious delinquencies, total limits and balances). `grade` / `sub_grade`
/ `int_rate` excluded from the model. Out-of-time split: train issued before **2015-01**,
test on/after (306k / 334k). Primary PD model: calibrated **HistGradientBoosting**.
Expected Loss = PD × EAD × LGD with EAD = funded amount and **LGD estimated from the data
(0.50)** — see below.

## Results

**Model (HGB).** Out-of-time test AUC **0.691**, Gini **0.382**, KS **0.278**, Brier
**0.119**; mean PD 0.135 vs base rate 0.132 (well calibrated in the large — deciles track
observed rates within ~1–2 pp). Calibration curve: `reports/figures/calibration_test.png`.

**Benchmark, with significance test.** grade-alone AUC 0.669; logistic on the same
features 0.680; HGB 0.691. **HGB beats grade by +0.0215 AUC**, and a **DeLong test on the
334k test loans gives z ≈ 19.3, p < 1e-80** — the gap is not noise. With a minimal
17-field allowlist the linear model *lost* to grade (0.658); adding bona-fide
origination-time bureau attributes reversed it. Headline: with public application +
bureau fields only, the model out-ranks Lending Club's own grade at a statistically
decisive margin.

**LGD estimated, not assumed.** On the 40,596 charged-off training loans,
`recovery_rate = Σ(principal repaid + post-charge-off recoveries) / Σ(funded amount) =
0.50`, so **LGD = 0.50** (exposure-weighted; equal-weighted 0.50 too). The textbook 0.45
was close but optimistic; 0.50 now feeds Expected Loss. LGD is still a single number, so
EL remains a monotone rescaling of PD — but it is now grounded in this book's own
recoveries, not an assumption. (`artifacts/lgd.json`)

**Segment calibration.** PD is still under-predicted in the low grades (grade G: predicted
30.5% vs observed 46.0%; F: 27.4% vs 42.3%; D: 20.7% vs 26.6%) — the model compresses the
risk tail, and the expanded features did not fix it. Global isotonic calibration cannot
correct a segment-specific bias; a production system would calibrate within risk bands.
E–G together are ~4% of loans.

**Concentration.** Grade A is 25.6% of exposure but only 12.9% of expected loss; grades
**C + D are 36.5% of exposure but 49.4% of expected loss** — risk is one notch below the
middle of the book.

## Policy — validated against realized outcomes

The simulator's profit curve is built from the model's own predicted PD, so the optimum
could be an artifact. `src/backtest.py` re-runs the same policy sweep on the **334k
out-of-time test loans using their actual cash flows and actual charge-offs**. Min FICO
660, cost of funds 4%/yr, servicing 0.4%/yr, T = 3 yr, amortisation factor 0.52.

| Approve if PD < | Approval | Pred. DR | **Actual DR** | Model EL | **Realized loss** | Model profit | **Realized profit** |
|---|---|---|---|---|---|---|---|
| 0.08 | 28% | 5.1% | 5.4% | $34M | $35M | $28M | $11M |
| 0.15 | 63% | 8.6% | 9.4% | $118M | $127M | $57M | $27M |
| **0.18 (model opt)** | **76%** | **10.0%** | **10.9%** | **$161M** | **$177M** | **$61M** | **$29M** |
| 0.18 (realized opt) | 74% | 9.8% | 10.7% | $153M | $169M | $60M | $29M |
| 0.20 | 80% | 10.4% | 11.4% | $175M | $194M | $60M | $28M |
| 0.25 | 90% | 11.8% | 13.0% | $222M | $250M | $54M | $21M |

**What the backtest shows:**
1. **The decision is sound.** The realized-profit optimum (PD ≈ 0.17–0.19 across runs) is
   essentially the same as the model-driven one (PD ≈ 0.19). Following the model's cut-off
   instead of the perfect-hindsight one costs **≈ $0.1–0.2M** on a ~$6B book. The optimum
   is *not* a miscalibration artifact.
2. **Predicted vs actual default rate track within ~1 pp** at every cut-off — the model's
   approve/deny decision is well calibrated even where the low-grade *probability* is not.
3. **Realized loss runs ~1.12× model EL**, and realized profit is ~half the model
   projection — the `(1 − PD)·coupon` interest approximation is optimistic and the tail
   loses more than predicted. Read the model's dollar figures as directional and biased
   high; read the *shape* and the *optimal cut-off* as reliable.

**Recommendation:** move the PD cut-off from 0.15 to **~0.18** (+~13 pts approval,
+$0.9B volume). Both the model and the realized backtest put the profit peak there.
Do not go past ~0.20 — realized profit falls as the marginal loans stop clearing their
loss. Expect ~$29M realized profit at the optimum, not the $61M the model projects.

## Fairness & adverse action

Lending Club data has **no protected-class attributes** (race, sex, age, national
origin), so a true fair-lending test is impossible here — that itself is a finding. What
`src/fairness.py` checks is disparate impact of the recommended policy across observable
proxies (4/5ths-rule adverse-impact ratio on approval rate; n ≥ 500 groups):

| Dimension | AIR | 4/5ths | Notes |
|---|---|---|---|
| Region (Census) | 0.96 | **pass** | approval 62–65% across all four regions |
| Income band | **0.42** | **FLAG** | <$40k approved 36% vs $120k+ 86% |
| Home ownership | **0.68** | **FLAG** | renters approved 51% vs mortgage-holders 75% |

The approved-book **default rate is roughly flat** across income bands (11.0% → 8.2%) and
tenure (10.6% renter → 8.6% mortgage) — the model is not letting through worse risks in
the lower-approval groups; the disparity is in *access*, driven by FICO and PD correlating
with income and housing tenure. In a real deployment this would go to a fair-lending
review, and the model would owe **ECOA / Reg B adverse-action reason codes** (per-decision
SHAP attributions) — noted, not built. (`artifacts/fairness.json`)

## Assumptions & limitations
- **Label maturity** — 36-month loans, issue date ≤ 2016-02; loans still *Current* after
  the window are dropped. Removes most right-censoring; slightly tilts the sample to
  earlier vintages.
- **Reject inference** — data is *accepted, funded* loans only. PD is calibrated to
  approved borrowers, not the through-the-door population; loosening the cut-off in the
  simulator extrapolates. Not fixable with this data.
- **Macro regime** — the issue window contains no recession; out-of-regime calibration is
  fragile. This is why multi-scenario stress testing is deferred, not faked.
- **Model** — HGB beats grade by +0.0215 AUC (DeLong p < 1e-80). The low grades (D–G) are
  under-predicted (risk-tail compression); global isotonic calibration does not fix it.
  Treat PD as a ranking tool and a calibrated probability in the bulk of the book, not a
  precise low-grade probability.
- **LGD** — estimated at 0.50 from this book's charged-off recoveries (was assumed 0.45),
  now segmented by grade (44% on A to 60% on G) and credibility-weighted toward that flat
  figure for thin grades (Buhlmann credibility, `src/lgd.py`) — feeds Expected Loss
  directly rather than sitting as a supplementary side stat. Still not segmented by
  vintage or macro regime, and still a point estimate with no confidence interval.
- **EAD** = funded amount, no amortisation → overstates exposure for seasoned loans.
- **Profit model** — expected value over T = 3 yr: interest and funding accrue on
  `b · principal` (b = 0.52), interest also × `(1 − PD)`; flat 0.4%/yr servicing;
  loss = `PD · EAD · LGD`. The realized backtest shows this **over-states profit ~2×** and
  is highly sensitive to the servicing assumption (an earlier 1.2%/yr made even prime
  lending unprofitable). No cash-flow discounting, no prepayment. Use the *shape*, not the
  level.
- **Circularity** — `grade` / `sub_grade` / `int_rate` excluded from features; `int_rate`
  used only in the profit calculation.
- **Fairness** — no protected-class data; the checks above are proxy disparate-impact
  only. Both flagged dimensions (income band, home ownership) are now confirmed
  statistically significant via a stratified bootstrap CI, Bonferroni-corrected for
  testing 3 dimensions at once (`src/fairness.py`) — not just a point estimate below
  0.80. Per-decision (ECOA/Reg B) adverse-action reason codes are still not built; what
  exists instead is a real global feature-importance ranking (`src/insight.py`) and a
  real per-loan marginal-contribution attribution on the actual trained model for a small
  sample of test loans (`src/explain.py`) — a genuine sensitivity on the real model, but
  explicitly not an exact Shapley-value decomposition, and not a production reason-code
  system generated per live decision.
- **Temporal validity** — 2012–2016 originations from a now-defunct retail platform in a
  zero-rate regime. Directionally useful; not a 2026 production scorecard.
- **Dataset** — Kaggle mirror `wordsforthewise/lending-club`, snapshot 2018Q4.

## What would be needed for real use
Reject-inference modelling; an EAD model (not a constant); recession stress scenarios;
LIVE population-stability monitoring against scored production traffic (see below — a
real but retrospective version now exists); per-decision reason codes at production
grade; a fair-lending review with real protected-class data; and independent model
validation (SR 11-7). This project is a decision-support analysis and an honest map of
those gaps — not a deployable underwriting model.

## Post-audit fixes (this pass)
A ruthless audit of this exact codebase found several gaps between what was claimed and
what was implemented. Fixed, with evidence in `src/`:
- **LGD segmented by grade**, credibility-weighted, now feeding Expected Loss directly
  (`src/lgd.py`, `sql/06_marts.sql`) — was previously a single flat number.
- **Fairness AIR given a real confidence interval**, Bonferroni-corrected for testing 3
  dimensions (`src/fairness.py`) — was previously a bare point estimate.
- **A real population-stability (PSI) check** between the pipeline's own train (2012-14)
  and test (2015-16) cohorts, on the model's own score and its top features
  (`src/monitor.py`). This is retrospective, not a live monitor — stated as such, see
  "Deferred" below.
- **A real per-loan attribution sample** on the actual trained model, not a synthetic
  scoring function (`src/explain.py`) — labelled precisely as marginal-contribution
  attribution, not SHAP, and not a production per-decision reason-code system.
- **CI added** (`.github/workflows/ci.yml`) running both real test suites plus a new one
  covering the LGD/PSI/fairness logic (`tests/test_methodology.py`) on every push —
  there was previously no CI at all.
- Two dead/misleading config keys corrected (`config.yaml`'s `target_positive_statuses`
  previously listed a status the SQL never actually treats as positive); two verified
  dashboard bugs fixed in CREDENCE (a tautological "high-risk approval" formula, a
  hardcoded average risk score that didn't match its own generator).

## Deferred (future work) — and why
- **A true, LIVE early-warning / drift-monitoring system** — not built, because there is
  no scored production traffic to monitor. What exists instead (`src/monitor.py`) is a
  real, retrospective PSI comparison between the two cohorts the pipeline already has;
  it is explicitly labelled as that, not as a live monitor, in its own output.
- **Risk-state migration matrices** — genuinely not buildable with this dataset: Lending
  Club's accepted-loans data has no month-by-month delinquency/DPD panel, only
  origination-time fields plus a single terminal outcome per loan. A migration matrix
  needs a loan's risk classification tracked over time; that data does not exist here.
  Faking one from origination-time data alone would be worse than not having it.
- **Multi-scenario macro stress testing** — not built: the training/test window (2012-16)
  contains no recession, so any stress scenario would be pure extrapolation with nothing
  to validate it against. Deferred deliberately, not from lack of time.
