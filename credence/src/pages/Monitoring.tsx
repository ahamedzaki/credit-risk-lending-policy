import { PageHead } from "../components/AppShell";
import { AlertTimeline } from "../components/blocks";
import { BarStrip, CalibrationChart, Sparkline } from "../components/charts";
import { Card, DemoBadge, Delta, Section, Stat } from "../components/primitives";
import { emerging, monthly } from "../data/demo";
import * as R from "../data/real";
import { fixed, pct, usdCompact } from "../lib/format";
import { RISK_HEX } from "../lib/risk";

function TrendTile({
  label,
  values,
  current,
  delta,
  fmt,
  invert = true,
  demo = true,
}: {
  label: string;
  values: number[];
  current: string;
  delta: number;
  fmt: (v: number) => string;
  invert?: boolean;
  demo?: boolean;
}) {
  return (
    <div className="card">
      <div className="stat__label">{label}</div>
      <div className="stat__figure" style={{ fontSize: 22 }}>{current}</div>
      <div className="stat__sub">
        <Delta value={delta} invert={invert} format={fmt} />
        {demo && <DemoBadge what="monthly series" />}
      </div>
      <div className="stat__spark">
        <Sparkline values={values} width={160} height={30} color={delta > 0 === invert ? RISK_HEX.Low : RISK_HEX.High} />
      </div>
    </div>
  );
}

export function Monitoring() {
  const last12 = monthly.slice(-12);
  const dr = last12.map((m) => m.default_rate);
  const dq = last12.map((m) => m.delinquency_rate);
  const el = last12.map((m) => m.expected_loss);
  const psi = last12.map((m) => m.psi);
  const cs = last12.map((m) => m.avg_credit_score);
  const d = (a: number[]) => a[a.length - 1] - a[0];

  const calib = R.decile.map((r) => ({ pred: r.mean_pd, obs: r.obs_rate, n: r.n }));

  return (
    <div className="page">
      <PageHead
        title="Monitoring"
        lede="Detect changes in credit quality before they become losses."
      />

      <Section title="Trends" desc="Twelve-month direction on the metrics that move first.">
        <div className="cardgrid cardgrid--3">
          <TrendTile label="Default rate" values={dr} current={pct(dr[dr.length - 1], 1)} delta={d(dr)} fmt={(v) => `${v > 0 ? "+" : ""}${(v * 100).toFixed(1)} pts`} />
          <TrendTile label="Delinquency rate" values={dq} current={pct(dq[dq.length - 1], 1)} delta={d(dq)} fmt={(v) => `${v > 0 ? "+" : ""}${(v * 100).toFixed(1)} pts`} />
          <TrendTile label="Expected loss" values={el} current={usdCompact(el[el.length - 1])} delta={d(el) / el[0]} fmt={(v) => `${v > 0 ? "+" : ""}${(v * 100).toFixed(0)}%`} />
          <TrendTile label="Population shift (PSI)" values={psi} current={fixed(psi[psi.length - 1], 2)} delta={d(psi)} fmt={(v) => `${v > 0 ? "+" : ""}${v.toFixed(2)}`} />
          <TrendTile label="Applicant credit score" values={cs} current={fixed(cs[cs.length - 1], 0)} delta={d(cs)} fmt={(v) => `${v > 0 ? "+" : ""}${v.toFixed(0)} pts`} invert={false} />
          <div className="card">
            <div className="stat__label">Model performance</div>
            <div className="stat__figure" style={{ fontSize: 22 }}>AUC {fixed(R.modelTest.auc, 3)}</div>
            <div className="stat__sub">
              <Delta value={R.modelTest.auc - R.modelTrain.auc} format={(v) => `${(v).toFixed(3)} vs train`} />
            </div>
            <p className="mini" style={{ marginTop: 10 }}>
              Real out-of-time result — train 2012–14 → test 2015–16.
            </p>
          </div>
        </div>
      </Section>

      <Section title="Emerging risk" desc="Signals that crossed a threshold, newest first." aside={<DemoBadge what="monitoring signals" />}>
        <Card variant="flush">
          <div style={{ padding: "var(--s-4) var(--s-6) var(--s-6)" }}>
            <AlertTimeline items={emerging} />
          </div>
        </Card>
      </Section>

      <Section
        title="Model risk"
        desc="How well the probability-of-default model discriminates and how stable it is over time."
      >
        <div className="statrow statrow--six">
          <Stat label="AUC" figure={fixed(R.modelTest.auc, 3)} sub={<span>Gini {fixed(R.modelTest.gini, 3)}</span>} />
          <Stat label="KS statistic" figure={fixed(R.modelTest.ks, 3)} />
          <Stat label="Brier score" figure={fixed(R.modelTest.brier, 3)} sub={<span>base rate {pct(R.modelTest.base_rate, 1)}</span>} />
          <Stat label="Precision" figure={pct(R.operatingPoint.precision, 0)} sub={<span>at PD ≥ 20% flag</span>} />
          <Stat label="Recall" figure={pct(R.operatingPoint.recall, 0)} sub={<span>at PD ≥ 20% flag</span>} />
          <Stat label="F1" figure={fixed(R.operatingPoint.f1, 2)} sub={<span>at PD ≥ 20% flag</span>} />
        </div>
        <p className="note">
          AUC, KS and Brier are the pipeline's real out-of-time figures. Precision / Recall / F1
          are computed at the PD ≥ 20% high-risk flag from the real risk-band aggregates
          (positives = Σ loans · observed default rate); they depend on that operating point and
          are labelled accordingly.
        </p>

        <div className="cardgrid cardgrid--2" style={{ marginTop: "var(--s-6)" }}>
          <Card title="Calibration" note="Predicted PD vs observed default rate, by decile — out-of-time test.">
            <CalibrationChart points={calib} />
            <p className="note">
              Predicted and observed track within ~1–2 points across the book; the top decile
              runs light — the low-grade tail is under-predicted, a documented limitation.
            </p>
          </Card>
          <Card title="Model stability — train → test" note="The out-of-time time split: older vintages (train) vs newer (test).">
            <div className="stripset">
              {[
                { k: "AUC", tr: R.modelTrain.auc, te: R.modelTest.auc, f: (v: number) => v.toFixed(3), inv: true },
                { k: "Gini", tr: R.modelTrain.gini, te: R.modelTest.gini, f: (v: number) => v.toFixed(3), inv: true },
                { k: "KS", tr: R.modelTrain.ks, te: R.modelTest.ks, f: (v: number) => v.toFixed(3), inv: true },
                { k: "Default rate", tr: R.modelTrain.base_rate, te: R.modelTest.base_rate, f: (v: number) => pct(v, 1), inv: false },
              ].map((r) => (
                <div key={r.k} className="grade-conc">
                  <span style={{ fontWeight: 600 }}>{r.k}</span>
                  <div className="dualbar">
                    <i style={{ width: `${(r.tr / Math.max(r.tr, r.te)) * 100}%`, background: "var(--ink-3)" }} />
                  </div>
                  <span>
                    train {r.f(r.tr)} → test {r.f(r.te)}{" "}
                    <Delta value={r.te - r.tr} invert={!r.inv} format={(v) => `${v > 0 ? "+" : ""}${r.f(Math.abs(v))}`} />
                  </span>
                </div>
              ))}
            </div>
            <p className="note">
              Discrimination softens on newer vintages (AUC 0.715 → 0.690) and the default rate
              drifts up 1.6 points — expected as the book seasons into a slightly worse cohort;
              well within tolerance, but the direction is worth watching.
            </p>
          </Card>
        </div>

        <Card
          title="What drives the model"
          note="Global permutation importance on the shipped model — 15,000 out-of-time test loans, 5 repeats."
        >
          <BarStrip
            rows={R.featureImportance.map((f) => ({ name: f.label, value: f.importance }))}
            valueFormat={(v) => `−${v.toFixed(3)} AUC`}
            color="var(--accent)"
          />
          <p className="note">
            Bar length is the mean AUC drop when that feature's values are shuffled — the model's real
            attribution, not an assumption. Credit score, recent credit-seeking (accounts opened in 24
            months), and loan-to-income dominate; this is a global ranking, not a per-loan reason code.
            A production deployment would owe ECOA/Reg B adverse-action reasons per decision (a real
            per-loan explainer) — noted as a limitation on Data/Model, not built here.
          </p>
        </Card>

        <Card title="Benchmark" note="Does the model beat the lender's own risk grade?" aside={undefined}>
          <div className="stripset">
            {[
              { k: "Gradient-boosted PD model", v: R.benchmark.hgb, c: "var(--accent)" },
              { k: "Logistic regression", v: R.benchmark.logistic, c: "var(--ink-2)" },
              { k: "Risk grade alone", v: R.benchmark.grade_only, c: "var(--ink-3)" },
            ].map((r) => (
              <div key={r.k} className="grade-conc" style={{ gridTemplateColumns: "220px 1fr 70px" }}>
                <span>{r.k}</span>
                <div className="dualbar">
                  <i style={{ width: `${((r.v - 0.6) / 0.12) * 100}%`, background: r.c }} />
                </div>
                <span>AUC {r.v.toFixed(3)}</span>
              </div>
            ))}
          </div>
          <p className="note">
            The model beats grade by {R.delong.auc_diff.toFixed(3)} AUC. A DeLong test on the
            333,721 test loans gives z ≈ {R.delong.z.toFixed(1)}, p ≈ {R.delong.p_value.toExponential(0)} —
            the gap is not noise.
          </p>
        </Card>
      </Section>
    </div>
  );
}
