import { PageHead } from "../components/AppShell";
import { Card, Section } from "../components/primitives";
import * as R from "../data/real";
import { count, pct } from "../lib/format";

export function DataModel() {
  return (
    <div className="page">
      <PageHead
        title="Data & Model"
        lede="Where the numbers come from, and which panels are measured results versus illustrative demo data."
      />

      <div className="cardgrid cardgrid--2">
        <Card title="Dataset">
          <dl className="deflist">
            <div><dt>Source</dt><dd>{R.PROVENANCE.dataset}</dd></div>
            <div><dt>Window</dt><dd>{R.PROVENANCE.window}</dd></div>
            <div><dt>Loans</dt><dd>{count(R.PROVENANCE.loans)} terminal</dd></div>
            <div><dt>Term</dt><dd>{R.PROVENANCE.term}</dd></div>
            <div><dt>Snapshot</dt><dd>{R.PROVENANCE.snapshot}</dd></div>
            <div><dt>Split</dt><dd>{R.PROVENANCE.oot}</dd></div>
          </dl>
        </Card>
        <Card title="Model">
          <dl className="deflist">
            <div><dt>Estimator</dt><dd>{R.PROVENANCE.model}</dd></div>
            <div><dt>Test AUC / Gini</dt><dd>{R.modelTest.auc.toFixed(3)} / {R.modelTest.gini.toFixed(3)}</dd></div>
            <div><dt>KS / Brier</dt><dd>{R.modelTest.ks.toFixed(3)} / {R.modelTest.brier.toFixed(3)}</dd></div>
            <div><dt>vs risk grade</dt><dd>+{R.delong.auc_diff.toFixed(3)} AUC (DeLong p ≈ {R.delong.p_value.toExponential(0)})</dd></div>
            <div><dt>LGD</dt><dd>{pct(R.lgd.value, 0)} — {R.PROVENANCE.lgdBasis}</dd></div>
            <div><dt>EAD</dt><dd>{R.PROVENANCE.eadBasis}</dd></div>
          </dl>
        </Card>
      </div>

      <Section title="Real vs demo — panel ledger" desc="CREDENCE presents pipeline results and clearly-labelled synthetic data side by side. It never dresses synthetic data as a measured result.">
        <Card variant="flush">
          <table className="ledger">
            <tbody>
              <tr>
                <td>Executive KPIs, risk spectrum, concentration</td>
                <td><span className="tag tag--real">REAL</span></td>
                <td>Exposure, expected loss, default rate, risk-band / grade / purpose / FICO / vintage breakdowns — from <code>portfolio_summary.parquet</code> and <code>metrics.json</code>.</td>
              </tr>
              <tr>
                <td>Threshold simulator</td>
                <td><span className="tag tag--real">REAL</span></td>
                <td>Runs along the 60-point out-of-time backtest curve in <code>backtest.json</code> — 333,721 test loans, actual repayment outcomes. Interpolated, never extrapolated past the measured range.</td>
              </tr>
              <tr>
                <td>Model Risk section</td>
                <td><span className="tag tag--real">REAL</span></td>
                <td>AUC, Gini, KS, Brier, calibration deciles, train→test stability, DeLong benchmark. Precision / Recall / F1 are computed at the PD ≥ 20% flag and labelled as such.</td>
              </tr>
              <tr>
                <td>Decisions — income-band approval split, Overview fair-lending signal</td>
                <td><span className="tag tag--real">REAL</span></td>
                <td>Approval rates and adverse-impact ratios from the pipeline's fairness analysis (<code>fairness.json</code>). Income band and home-ownership both fail the 4/5ths rule (AIR 0.42 and 0.68, both statistically significant via a Bonferroni-corrected bootstrap CI); region passes (0.96).</td>
              </tr>
              <tr>
                <td>Monitoring — "What drives the model"</td>
                <td><span className="tag tag--real">REAL</span></td>
                <td>Global permutation feature importance computed directly on the shipped estimator (<code>model.joblib</code>) against a 15,000-loan out-of-time test sample — a real attribution, not an assumption.</td>
              </tr>
              <tr>
                <td>Monitoring — "Population stability (PSI)"</td>
                <td><span className="tag tag--real">REAL</span></td>
                <td>A genuine PSI comparison between the pipeline's own train (2012–14) and test (2015–16) cohorts on the model's score and top features — retrospective, not a live production feed (see Known limitations).</td>
              </tr>
              <tr>
                <td>Monitoring — "Real per-loan sample"</td>
                <td><span className="tag tag--real">REAL</span></td>
                <td>12 real out-of-time test loans, explained by perturbing the actual trained model one feature at a time — a genuine per-loan sensitivity, not synthetic. Not an exact SHAP decomposition (see Known limitations).</td>
              </tr>
              <tr>
                <td>Portfolio — loss given default by grade</td>
                <td><span className="tag tag--real">REAL</span></td>
                <td>Recovery-rate LGD, credibility-weighted by grade (Buhlmann), 46.4% (Grade A) to 53.5% (Grade E) — thin grades (F, G) shrink toward the flat portfolio figure rather than trusting a small sample. This is what actually feeds Expected Loss now, not a supplementary side stat.</td>
              </tr>
              <tr>
                <td>Borrower table &amp; risk-driver panel</td>
                <td><span className="tag tag--demo">DEMO</span></td>
                <td>220 synthetic applications from a seeded generator; risk drivers are the terms of a transparent scoring function, not the pipeline's model attributions.</td>
              </tr>
              <tr>
                <td>All monthly time series</td>
                <td><span className="tag tag--demo">DEMO</span></td>
                <td>Risk-over-time, delinquency trend, monitoring trend tiles, PSI — seeded synthetic series, anchored so the latest month matches the real headline values. The train→test drift shown in Model Risk is the one real time comparison.</td>
              </tr>
              <tr>
                <td>Alerts &amp; emerging-risk timeline</td>
                <td><span className="tag tag--demo">DEMO</span></td>
                <td>Plausible monitoring signals for the layout; not generated from live data.</td>
              </tr>
            </tbody>
          </table>
        </Card>
        <p className="note">
          CREDENCE is a dashboard built over the public Lending Club dataset and this repository's
          reproducible pipeline. No claim is made about any real bank, customer, or portfolio.
        </p>
      </Section>

      <Section
        title="Known limitations"
        desc="What this build deliberately does not fake. Each of these needs infrastructure a static, single-user dashboard doesn't have — building a fake version would be worse than stating the gap."
      >
        <Card variant="flush">
          <table className="ledger">
            <tbody>
              <tr>
                <td>Access control</td>
                <td>No authentication or per-role permissions — anyone with the URL sees every panel. Fine for a portfolio demo over public data; required before this points at anything real.</td>
              </tr>
              <tr>
                <td>Live model monitoring</td>
                <td>The monthly trend tiles on Monitoring are still seeded synthetic series — there is no scored production traffic to plot month by month. What's real now is a retrospective PSI check between the pipeline's own train and test cohorts ("Population stability (PSI)" above) — a genuine population-stability computation, just not a live, continuously-updating one.</td>
              </tr>
              <tr>
                <td>Per-loan reason codes</td>
                <td>"What drives the model" is a real global attribution; "Real per-loan sample" (above) is now a real per-loan sensitivity on the actual trained model for 12 sampled loans. Neither is a production ECOA/Reg B system: the per-loan version is marginal-contribution attribution (one feature perturbed at a time), not an exact Shapley-value decomposition, and it covers a small fixed sample, not every live decision.</td>
              </tr>
              <tr>
                <td>Protected-class fair lending</td>
                <td>The fairness checks are a disparate-impact proxy on income, region, and home ownership. Lending Club data has no race, sex, age, or national-origin fields, so a true ECOA fair-lending test isn't possible on this dataset.</td>
              </tr>
              <tr>
                <td>Macro / stress scenarios</td>
                <td>The model and backtest reflect one historical window (2012–2016 originations, scored through 2018). There's no scenario engine for a rate shock or a recession — a real deployment would need one.</td>
              </tr>
            </tbody>
          </table>
        </Card>
      </Section>
    </div>
  );
}
