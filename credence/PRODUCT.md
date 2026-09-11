# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

React + Vite + TypeScript (delegated within the user's stated choice of "multi-file React app in the repo"). Hand-built SVG chart components, no charting library. Self-hosted variable font. Client-only; no backend.

## Users

Primary: **credit-risk analysts and risk managers** at a consumer lender, at a desk, mid-workday, reviewing portfolio health before a risk-committee readout or investigating a flagged segment / application. Secondary: lending-team leads who consume the executive view and the decision trade-off.

## Product Purpose

**CREDENCE — Credit Risk Intelligence Platform.** A decision-support dashboard that lets a risk team, within ~10 seconds of opening it, understand: how healthy the portfolio is, whether risk is rising or falling, where it is concentrated, which borrowers/segments need attention, why they are risky, and what policy action to consider. It communicates complex credit-risk information without overwhelming the reader.

## Positioning

Built on a real, reproducible credit-risk pipeline (`../` in this repo): 640,919 US consumer loans, a calibrated gradient-boosted PD model, an expected-loss engine, and a realized-outcome backtest. Every executive figure traces to that pipeline; the decision simulator is driven by the pipeline's actual out-of-time backtest curve, not a toy model. Where the pipeline has no row-level or longitudinal data, the surface uses clearly-labelled synthetic DEMO data and never presents it as a real result.

## Operating Context

Desktop-first analyst tool (also usable on laptop and tablet; reflow, not shrink). Five areas: Executive Overview, Portfolio Analytics, Borrower Risk Explorer (+ borrower detail drill-down), Credit Decision Analysis (threshold simulator), Risk Monitoring (+ Model Risk). Minimal left navigation: Overview / Portfolio / Borrowers / Decisions / Monitoring, with Settings and Data / Model Information at the foot.

## Capabilities and Constraints

- Real aggregates available: portfolio exposure ($8.16B), expected loss ($524M, LGD 0.50 data-estimated), observed default rate (14.1%), risk-band / grade / purpose / FICO-band / vintage breakdowns, model metrics (test AUC 0.690, Gini 0.381, KS 0.275, Brier 0.119; train AUC 0.715), grade benchmark + DeLong test (z=19.4, p≈2e-83), 10-row calibration decile table, 60-row realized-outcome backtest curve, fairness 4/5ths results (income-band AIR 0.42 fail, region 0.96 pass).
- No real data for: individual borrower rows, per-decision model attributions (SHAP), monthly time series, population-stability index, alerts. These are synthetic DEMO data, generated from a seeded transparent function and badged on every surface that shows them.
- Precision / Recall / F1 are not in the pipeline artifacts; where shown they are computed at a stated operating point (PD ≥ 0.20 high-risk flag) from real aggregates and labelled as such.
- No login, no persistence, no external calls. Terminology: PD (probability of default), EAD, LGD, EL (expected loss = PD×EAD×LGD), AIR (adverse-impact ratio), OOT (out-of-time), vintage.

## Brand Commitments

- Product name **CREDENCE**, subtitle **Credit Risk Intelligence Platform**. (Inferred from brief; binding.)
- Visual direction pinned by the brief: interpret Apple's design *principles* — extreme simplicity, generous whitespace, strong typographic hierarchy, restrained near-white palette, colour used only to encode risk semantically, elegant subtle cards, progressive disclosure, calm and trustworthy. Must not resemble a generic Power BI / Bootstrap / admin-template dashboard. Do not copy Apple's proprietary UI, icons, product screens, or branding.
- Semantic risk scale: Low → muted green, Medium → amber, High → orange-red, Critical → deep red. No neon, no rainbow charts, no heavy gradients or shadows.

## Evidence on Hand

- `../artifacts/metrics.json`, `../artifacts/lgd.json`, `../artifacts/backtest.json`, `../artifacts/fairness.json`, `../exports/portfolio_summary.parquet`, `../reports/memo.md` — the real pipeline outputs the dashboard's executive figures are transcribed from.
- No real customer data, no real financial-institution identity. CREDENCE is presented as an analytics product over the (public) Lending Club dataset; no claim is made about any real bank, customer, or portfolio.

## Product Principles

1. **Ten-second comprehension.** The Overview answers the six questions (health / direction / concentration / who / why / what action) without opening another page.
2. **Colour is a risk signal, not decoration.** A screen with no elevated risk is nearly monochrome.
3. **Progressive disclosure.** Headline first; drivers, tables, and methodology one interaction deeper.
4. **Traceable numbers.** Real figures cite the pipeline; synthetic figures are badged DEMO and never dressed as results.
5. **Calm density.** Institutional seriousness through precision and restraint, not through filling space.

## Accessibility & Inclusion

Desktop analyst tool: keyboard-operable navigation, filters, table, drawer, and simulator; visible focus; WCAG AA contrast for text and for risk colours against their ground; risk never encoded by colour alone (always paired with a label or value); `prefers-reduced-motion` respected.
