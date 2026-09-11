# CREDENCE — build plan

Pre-implementation architecture. Aesthetic + product truth live in `PRODUCT.md` and
`.impeccable/surfaces/src-app-tsx.md` (direction contract). This file is IA, hierarchy,
design system, data model, and flows.

---

## 1. Information architecture

```
CREDENCE
├── Overview        Executive Risk Overview — portfolio health in one screen
├── Portfolio       Portfolio Analytics — how exposure & risk are distributed
├── Borrowers       Borrower Risk Explorer — investigate individual credit risk
│   └── (drawer)    Borrower detail — profile, risk drivers, decision explanation
├── Decisions       Credit Decision Analysis — approval / rejection / threshold trade-off
├── Monitoring      Risk Monitoring — detect change; includes Model Risk
├── ─────
├── Settings        display density, risk-band thresholds, reduced motion
└── Data / Model    dataset provenance, model card, what is real vs DEMO
```

Left rail, 200px, fixed. Wordmark top; 5 primary items with 20px line icons; Settings +
Data/Model pinned to the foot. No secondary nav — depth is via drawer / in-page tabs only.

## 2. Page hierarchy

**Overview** (single reading column)
1. Title "Credit Risk" + one supporting line
2. KPI row (5): Portfolio Exposure · Default Rate · High-Risk Accounts · Expected Loss · Risk Trend — figure / label / period delta / trend caret. Not cards.
3. **Portfolio Risk Distribution** (dominant) — horizontal segmented risk spectrum, 4 bands, proportional width, share %, one-line read-out
4. **Risk Over Time** — line chart, switchable metric (Default rate / Delinquency / Expected loss), range 1M·3M·6M·1Y·All *(DEMO series, badged)*
5. **Where risk is concentrated** — 5 compact horizontal bar strips: Grade · Loan purpose · FICO band · Income band · Vintage (real)
6. **Needs attention** — 3 quiet alert rows (High / Medium / Low emphasis) *(DEMO)*

**Portfolio**
- Metric line: Total Loans · Total Exposure · Avg Loan Size · Avg Interest Rate · Default Rate · Expected Loss
- Charts: Exposure by purpose · Default rate by purpose · Risk distribution by FICO band · Default rate by income band (real) · Delinquency trend *(DEMO)* · Exposure by geography (US region, real approval/DR proxy)
- Filter bar (unobtrusive, collapsible): Date · Purpose · FICO · Income · Employment · Loan amount · Risk category · Loan status *(filters act on DEMO borrower rows + re-weight aggregates where possible; otherwise annotate)*

**Borrowers**
- Search + filter chips
- Table: Borrower ID · Credit Score · Income · Loan Amount · DTI · Employment · Purpose · Risk Score · Risk Category · Default Probability · Decision  *(DEMO rows, badged)*
- Row → **detail drawer** (slide-over, right, ~520px):
  - Header: Borrower ID · Risk Score · Risk Category · Default Probability
  - Sections: Financial Profile · Credit Profile · Loan Information
  - **"Why is this borrower risky?"** — diverging horizontal contribution bars (DTI +18, Credit Score +14, Recent Delinquencies +21, Income Stability −7, Loan Amount +9 …), summing to the score; each row expandable to a plain-language note
  - Decision Explanation — rule + threshold + what would flip it

**Decisions**
- Metric line: Approval Rate · Rejection Rate · High-Risk Approval Rate · Avg Risk Score · Expected Loss
- Charts: Approval rate over time *(DEMO)* · Approval vs rejection by FICO band · Approval vs rejection by income band (real, from fairness AIR data) · Default rate by decision · Expected loss by decision
- **Threshold simulator** (centerpiece, real): slider over PD cut-off; reads the pipeline's 60-row out-of-time backtest curve; outputs Approved / Rejected / Expected defaults / Expected loss (model) / Realized loss / Expected profit (model vs realized) / Portfolio exposure. Shows the Growth ↔ Risk trade and marks the realized-profit optimum (PD≈0.17).

**Monitoring**
- Trend tiles: Risk trend · Default trend · Delinquency trend · Population shift (PSI) · Model performance — sparkline + current + delta *(trends DEMO except train→test drift, which is real)*
- **Emerging Risk** — alert timeline (vertical), each entry: severity dot, metric, magnitude, window, affected segment *(DEMO)*
- **Model Risk** section:
  - Model Performance: AUC · Precision · Recall · F1 · KS · Calibration (AUC/KS/Brier real; P/R/F1 computed at PD≥0.20 flag from real aggregates, labelled). Calibration = real 10-decile predicted-vs-observed chart.
  - Model Stability: train (2012–14) vs test (2015–16) — AUC 0.715 → 0.690, default rate 13.2% → 14.9% (real, labelled as the OOT time split)

**Data / Model**
- Dataset card: source (Lending Club accepted loans, `wordsforthewise`), window 2012-01…2016-02, 640,919 terminal 36-mo loans, snapshot 2018Q4
- Model card: HistGradientBoosting, isotonic-calibrated, ~33 origination-time features, LGD 0.50 data-estimated, EAD = funded amount
- Real-vs-DEMO ledger: exactly which panels are pipeline-real and which are synthetic

**Settings** — density (comfortable / compact), risk-band cut-offs (0.05 / 0.10 / 0.20 default), reduced motion, currency display note.

## 3. Component hierarchy

```
AppShell
├── Rail (Nav)                     Wordmark, NavItem×5, footer NavItem×2
└── Main
    ├── PageHeader                 title, supporting line, optional right-side control slot
    └── <page>

Primitives
  Stat            figure + label + Delta (+ optional Sparkline)
  StatRow         horizontal group of Stat, baseline-aligned
  Card            hairline container, radius, padding; variant: plain | inset
  Section         heading + description + body + optional control (MetricToggle / TimeRange)
  Delta           signed value + TrendCaret, tinted (risk | neutral | positive)
  DemoBadge       small "DEMO DATA" chip with tooltip
  Icon            inline SVG set, 1.5px stroke, 20/16px

Charts (hand-built SVG, shared <ChartFrame> + scales.ts)
  RiskSpectrum    segmented proportional horizontal bar, 4 bands, labels + %
  LineChart       1–3 series, minimal axes, no gridlines, hover crosshair + tooltip
  BarStrip        compact horizontal bars for "where risk is concentrated"
  BarPair         grouped/diverging bars (approval vs rejection, default rate by decision)
  Distribution    histogram (FICO / PD distribution)
  ContributionBars diverging bars from a zero baseline, ± labels (risk drivers)
  CalibrationChart predicted vs observed dots + 45° reference
  Sparkline       tiny real-data line, no axis

Composite
  FilterBar       chips + popovers; controlled filter state; "clear all"
  DataTable       column defs, sort, search, sticky header, row click; keyboard nav
  Drawer          right slide-over, focus-trap, esc-close, backdrop
  AlertRow        severity dot + text + magnitude; no colored bar
  AlertTimeline   vertical rail of AlertRow with date ticks
  ThresholdSimulator  Slider + live Stat grid + mini trade-off chart, bound to backtest curve
  MetricToggle    segmented control (small)
  TimeRange       1M 3M 6M 1Y All segmented control
```

State: React context for `settings` (density, thresholds, reducedMotion) and `filters`
(Portfolio/Borrowers). Everything else local. Routing: `react-router-dom`, layout route +
5 pages + 2 footer pages; borrower detail is `?borrower=<id>` search param → Drawer.

Separation: `data/` (real.ts, demo.ts, types.ts) · `lib/` (format, scales, risk, curve
interpolation) · `components/` (UI) · `pages/` (composition). No fetching.

## 4. Design system (tokens — full set in `src/styles/tokens.css`)

- **Ground** `--bg` #FBFBFC · **surface** #FFFFFF · **surface-inset** #F6F7F9
- **Ink** `--ink` #17191F · `--ink-2` #5C6270 · `--ink-3` #8B909C · `--ink-disabled` #B9BDC6
- **Hairline** `--line` #E8E9ED · `--line-strong` #D8DAE0
- **Interactive** `--accent` #33528C · `--accent-weak` #ECF0F7 · focus ring `--accent` @ 40% + 2px
- **Risk** `--risk-low` #2F7D57 · `--risk-med` #B07D1A · `--risk-high` #C0552C · `--risk-crit` #9E2B22; each has a `-wash` at ~10% for fills
- **Positive/neutral delta** `--pos` #2F7D57 · `--neg` #C0552C · `--flat` #8B909C
- **Type** family `"Public Sans Variable", system-ui, sans-serif`; scale (px): 34/28/22/17/15/13/12/11; weights 400/500/600; tabular lining numerals default (`font-variant-numeric: tabular-nums`); tracking −0.011em on ≥22px, −0.006em body, 0 on ≤12 labels; line-height 1.2 display / 1.5 body
- **Space** 4 8 12 16 20 24 32 40 48 64 · radius 6 (control) / 12 (card) / 999 (pill) · rail 200 · content max 1160, reading blocks 720–920
- **Motion** 120ms control, 180ms fade, 240ms drawer; ease `cubic-bezier(.22,1,.36,1)`; one authored moment = drawer slide + content stagger; all gated by `prefers-reduced-motion`
- **Browser surfaces** themed: `::selection` accent-weak, `::-webkit-scrollbar` 10px line/tertiary, `:focus-visible` accent ring, `caret-color` accent, tabular-nums on all data.
- **Elevation** cards = hairline only; drawer/popover = `0 1px 2px rgba(20,22,30,.06), 0 8px 24px rgba(20,22,30,.10)`; no other shadow.

## 5. Data model

```ts
Loan / Application (DEMO rows)
  loan_id, borrower_id, application_date
  loan_amount, annual_income, employment_length, employment_type
  credit_score, debt_to_income, credit_history_length, home_ownership
  loan_purpose, interest_rate, term
  delinquencies_2y
  risk_score (0–100), probability_of_default (0–1), risk_category (Low|Medium|High|Critical)
  expected_loss
  decision (Approved|Declined|Manual review), loan_status (Current|Delinquent|Default|Paid)
  drivers: { factor, contribution(±), note }[]   // sums to risk_score offset from base

PortfolioAggregate (REAL, from pipeline)
  totals { loans, exposure, expected_loss, observed_default_rate, lgd }
  byRiskBand[] { band, loans, exposure, expected_loss, observed_default_rate }
  byGrade[]   { grade, loans, exposure, avg_pd, expected_loss, observed_default_rate, high_risk_exposure }
  byPurpose[] { purpose, loans, exposure, avg_pd, expected_loss, observed_default_rate }
  byFicoBand[]{ band, loans, exposure, avg_pd, observed_default_rate }
  byVintage[] { year, loans, exposure, avg_pd, observed_default_rate }

ModelCard (REAL)
  test { auc, gini, ks, brier, n, base_rate }   train { … }
  benchmark { hgb, logistic, grade_only }        delong { z, p_value, auc_diff }
  decile[] { bucket, n, mean_pd, obs_rate, lift }
  lgd { value, recovery_rate, n_charged_off }
  derivedOperatingPoint { cutoff: 0.20, precision, recall, f1, tp, fp, fn }   // computed, labelled

BacktestCurve (REAL, 60 rows)
  point[] { pd_cut, approval_rate, pred_default_rate, actual_default_rate,
            model_expected_loss, realized_credit_loss, model_profit, realized_profit, n }
  model_optimum { … }  realized_optimum { … }  regret  loss_ratio

Fairness (REAL)
  overall_approval_rate
  byIncomeBand / byRegion / byHomeOwnership { group, approval_rate, approved_book_default_rate }, air, passes

TimeSeries (DEMO, seeded)
  monthly[] { month, default_rate, delinquency_rate, expected_loss, avg_credit_score, avg_dti, psi }
  anchored so the latest month ≈ the real headline values

Alert (DEMO)  { id, severity: High|Medium|Low, title, detail, metric, magnitude, window, segment, date }
```

`lib/curve.ts` — monotone interpolation along `BacktestCurve.point` by `pd_cut` for the
threshold simulator; clamps to the real data range, never extrapolates.

## 6. User flows

1. **10-second health check** — open → Overview → read KPI row + risk spectrum + "needs attention". Done without a click.
2. **Investigate concentration** — Overview "Where risk is concentrated" → notice Grade C–D → click → Portfolio pre-filtered to that grade → read exposure vs EL share.
3. **Investigate a borrower** — Borrowers → search / sort by Default Probability desc → click row → drawer → read "Why is this borrower risky?" contribution bars → expand a driver for the plain-language note → read Decision Explanation.
4. **Policy trade-off** — Decisions → drag threshold slider → watch Approved ↔ Expected loss ↔ Profit move along the real backtest curve → see the realized-profit optimum marker → read the Growth ↔ Risk read-out.
5. **Catch emerging risk** — Monitoring → scan trend tiles → open Emerging Risk timeline → follow an entry to the affected segment.
6. **Check the model** — Monitoring → Model Risk → read AUC/KS/Brier + calibration chart + train→test stability; or Data/Model for provenance and the real-vs-DEMO ledger.

## Real vs DEMO ledger (also surfaced in-app on Data/Model)

REAL: all Overview KPIs except Risk-Trend delta; risk spectrum; "where risk is concentrated" (all 5); Portfolio metric line + purpose/FICO/income/geography charts; Decisions income-band approval split; the entire threshold simulator; all of Model Risk (P/R/F1 computed + labelled).
DEMO (badged): borrower table + drawer + risk drivers; all monthly time series (Risk Over Time, delinquency trend, Monitoring trend tiles, PSI); "Needs attention" + Emerging Risk timeline; "approval rate over time".
