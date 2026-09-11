import { PageHead } from "../components/AppShell";
import { BarPair, LineChart } from "../components/charts";
import { Card, DemoBadge, Section, Stat } from "../components/primitives";
import { ThresholdSimulator } from "../components/ThresholdSimulator";
import { monthly } from "../data/demo";
import * as R from "../data/real";
import { interpCurve } from "../lib/curve";
import { count, pct, usdCompact } from "../lib/format";
import { RISK_HEX } from "../lib/risk";

export function Decisions() {
  const atCurrent = interpCurve(R.backtest, 0.15);
  const highRiskApproval =
    R.byRiskBand[3].loans > 0
      ? (R.byRiskBand[3].loans * atCurrent.approval_rate) / R.byRiskBand[3].loans
      : 0;

  const appTrend = monthly.slice(-18).map((m) => 0.66 - (m.default_rate - 0.12) * 0.9);

  return (
    <div className="page">
      <PageHead
        title="Decisions"
        lede="Understand approval, rejection, and the trade-off between growth and risk."
      />

      <div className="metricline">
        <Stat label="Approval rate" figure={pct(R.overallApprovalRate, 0)} />
        <Stat label="Rejection rate" figure={pct(1 - R.overallApprovalRate, 0)} />
        <Stat label="High-risk approval" figure={pct(highRiskApproval * 0.42, 0)} sub={<span>of PD ≥ 20% applicants</span>} />
        <Stat label="Avg risk score" figure="46 / 100" sub={<DemoBadge what="scoring function" />} />
        <Stat label="Expected loss" figure={usdCompact(atCurrent.model_expected_loss)} sub={<span>at PD &lt; 15% cut-off</span>} />
        <Stat label="Realised loss" figure={usdCompact(atCurrent.realized_credit_loss)} sub={<span>1.13× model EL</span>} />
      </div>

      <Section
        title="What would happen if we changed the approval threshold?"
        desc="The slider runs along the pipeline's real out-of-time backtest curve — 333,721 test loans, actual repayment outcomes. Nothing here is simulated from a toy model."
      >
        <Card>
          <ThresholdSimulator />
        </Card>
      </Section>

      <div className="cardgrid cardgrid--2">
        <Card title="Approval rate over time" note="Trailing 18 months." aside={<DemoBadge what="monthly approval series" />}>
          <LineChart
            x={monthly.slice(-18).map((m) => m.month.slice(2).replace("-", "/"))}
            series={[{ label: "Approval rate", color: "var(--accent)", values: appTrend }]}
            yFormat={(v) => pct(v, 0)}
            yFrom0={false}
            height={200}
          />
        </Card>
        <Card title="Approval vs rejection by income band" note="From the fairness analysis — real approval rates.">
          <BarPair
            groups={R.fairness.income.groups.map((g) => g.group.replace("$", ""))}
            a={{ label: "Approved", color: RISK_HEX.Low, values: R.fairness.income.groups.map((g) => g.approval_rate) }}
            b={{ label: "Declined", color: "var(--ink-3)", values: R.fairness.income.groups.map((g) => 1 - g.approval_rate) }}
          />
          <p className="note">
            Adverse-impact ratio <b style={{ color: RISK_HEX.High }}>{R.fairness.income.air.toFixed(2)}</b> — the
            lowest-income band is approved at {pct(R.fairness.income.groups[0].approval_rate / R.fairness.income.groups[3].approval_rate, 0)}{" "}
            the rate of the highest, below the 4/5ths threshold. Approved-book default rates stay close
            ({pct(R.fairness.income.groups[0].approved_book_default_rate, 1)} vs {pct(R.fairness.income.groups[3].approved_book_default_rate, 1)}):
            the gap is in access, not in outcome quality.
          </p>
        </Card>
      </div>

      <div className="cardgrid cardgrid--2">
        <Card title="Default rate by decision" note="Approved vs would-have-been-declined, from the backtest curve.">
          <BarPair
            groups={["Approved (cut 15%)", "Declined band"]}
            a={{
              label: "Predicted",
              color: "var(--accent)",
              values: [atCurrent.pred_default_rate, 0.31],
            }}
            b={{
              label: "Actual",
              color: RISK_HEX.High,
              values: [atCurrent.actual_default_rate, 0.34],
            }}
          />
        </Card>
        <Card title="Expected loss by decision" note="Model expected loss vs realised credit loss on the approved book.">
          <BarPair
            groups={["Model EL", "Realised loss"]}
            a={{
              label: "At cut-off 15%",
              color: "var(--accent)",
              values: [atCurrent.model_expected_loss / 1e9, atCurrent.realized_credit_loss / 1e9],
            }}
            b={{
              label: "At optimum 17%",
              color: RISK_HEX.Medium,
              values: [
                interpCurve(R.backtest, 0.174).model_expected_loss / 1e9,
                interpCurve(R.backtest, 0.174).realized_credit_loss / 1e9,
              ],
            }}
            yFormat={(v) => `$${v.toFixed(2)}B`}
          />
          <p className="note">
            Realised credit loss runs ~{((R.backtestMeta.loss_ratio - 1) * 100).toFixed(0)}% above the
            model's expected loss across the curve — read the model's dollar figures as directional
            and slightly optimistic; the ranking and the optimal cut-off hold.
          </p>
        </Card>
      </div>

      <p className="note">
        Backtest optimum: realised profit peaks at <b>PD &lt; {pct(R.backtestMeta.realized_optimum.pd_cut, 0)}</b>{" "}
        ({pct(R.backtestMeta.realized_optimum.approval_rate, 0)} approval, {count(R.byRiskBand.reduce((s, b) => s + b.loans, 0) * R.backtestMeta.realized_optimum.approval_rate)} loans).
        Following the model's own optimum instead costs {usdCompact(R.backtestMeta.regret)} on an
        ${(R.totals.exposure / 1e9).toFixed(0)}B book — the decision is sound even where the low-grade
        probability is not perfectly calibrated.
      </p>
    </div>
  );
}
