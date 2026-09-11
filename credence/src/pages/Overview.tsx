import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { PageHead } from "../components/AppShell";
import { AlertList, FairnessSignal, RiskSpectrum } from "../components/blocks";
import { LineChart, Sparkline } from "../components/charts";
import { Card, Delta, Seg, Section, Stat } from "../components/primitives";
import { DemoBadge } from "../components/primitives";
import { alerts, monthly } from "../data/demo";
import * as R from "../data/real";
import { count, pct, usdCompact } from "../lib/format";
import { RISK_HEX } from "../lib/risk";

const METRICS = ["Default rate", "Delinquency", "Expected loss"] as const;
const RANGES = ["3M", "6M", "1Y", "All"] as const;

export function Overview() {
  const nav = useNavigate();
  const [metric, setMetric] = useState<(typeof METRICS)[number]>("Default rate");
  const [range, setRange] = useState<(typeof RANGES)[number]>("1Y");

  const criticalLoans = R.byRiskBand[3].loans;
  const totalExposure = R.totals.exposure;

  const bands = R.byRiskBand.map((b) => ({
    band: b.band,
    range: b.range,
    exposure: b.exposure,
    count: b.loans,
    share: b.exposure / totalExposure,
  }));

  const series = useMemo(() => {
    const n = range === "3M" ? 3 : range === "6M" ? 6 : range === "1Y" ? 12 : monthly.length;
    const slice = monthly.slice(-n);
    const key =
      metric === "Default rate" ? "default_rate" : metric === "Delinquency" ? "delinquency_rate" : "expected_loss";
    return {
      x: slice.map((m) => m.month.slice(2).replace("-", "/")),
      values: slice.map((m) => m[key as "default_rate"]),
    };
  }, [metric, range]);

  const drTrend = monthly.slice(-12).map((m) => m.default_rate);
  const drDelta = drTrend[drTrend.length - 1] - drTrend[0];

  const fairnessDims = [
    { key: "income", label: "Income band", dim: R.fairness.income },
    { key: "region", label: "Region", dim: R.fairness.region },
    { key: "homeOwnership", label: "Home ownership", dim: R.fairness.homeOwnership },
  ];
  const fairnessFailing = fairnessDims.filter((d) => d.dim.air < 0.8).length;

  return (
    <div className="page">
      <PageHead title="Credit Risk" lede="Portfolio health and risk intelligence — the whole book at a glance." />

      <div className="statrow">
        <Stat
          label="Portfolio exposure"
          figure={usdCompact(totalExposure)}
          sub={<span>{count(R.totals.loans)} loans · 2012–2016 vintages</span>}
        />
        <Stat
          label="Default rate"
          figure={pct(R.totals.observed_default_rate, 1)}
          sub={<Delta value={drDelta} invert format={(v) => `${v > 0 ? "+" : ""}${(v * 100).toFixed(1)} pts / 12m`} />}
        />
        <Stat
          label="High-risk accounts"
          figure={count(criticalLoans)}
          sub={<span>PD ≥ 20% · {usdCompact(R.byRiskBand[3].exposure)} exposure</span>}
        />
        <Stat
          label="Expected loss"
          figure={usdCompact(R.totals.expected_loss)}
          sub={<span>{pct(R.totals.expected_loss / totalExposure, 1)} of exposure · LGD {pct(R.totals.lgd, 0)}</span>}
        />
        <Stat
          label="Risk trend"
          figure={
            <span style={{ color: drDelta > 0 ? RISK_HEX.High : RISK_HEX.Low, fontSize: 22 }}>
              {drDelta > 0 ? "↑" : "↓"} {Math.abs(drDelta * 100).toFixed(1)} pts
            </span>
          }
          sub={<span>default rate, trailing 12 months</span>}
          spark={<Sparkline values={drTrend} color={drDelta > 0 ? RISK_HEX.High : RISK_HEX.Low} />}
        />
      </div>

      <Section
        title="Portfolio risk distribution"
        desc="Funded exposure by model risk band. The single question this answers: how risky is the book right now?"
      >
        <RiskSpectrum bands={bands} />
      </Section>

      {fairnessFailing > 0 && (
        <Section
          title="Fair-lending signal"
          desc="Adverse-impact check on the approval policy, against the pipeline's real fairness analysis."
          aside={
            <>
              <span className="tag tag--real">REAL</span>
              <Link to="/decisions" className="pill" style={{ marginLeft: 8 }}>
                Full breakdown →
              </Link>
            </>
          }
        >
          <FairnessSignal dims={fairnessDims} />
        </Section>
      )}

      <Section
        title="Risk over time"
        desc="Direction of travel for the metrics a committee asks about first."
        aside={
          <>
            <DemoBadge what="monthly trend series" />
            <Seg options={METRICS} value={metric} onChange={setMetric} />
            <Seg options={RANGES} value={range} onChange={setRange} />
          </>
        }
      >
        <LineChart
          x={series.x}
          series={[
            {
              label: metric,
              color: metric === "Expected loss" ? "var(--accent)" : RISK_HEX.High,
              values: series.values,
            },
          ]}
          yFormat={(v) =>
            metric === "Expected loss" ? usdCompact(v, { decimals: 0 }) : `${(v * 100).toFixed(1)}%`
          }
          yFrom0={metric === "Expected loss"}
          height={230}
        />
      </Section>

      <Section
        title="Where risk is concentrated"
        desc="Expected-loss share against each dimension. Click a grade or purpose to open it in Portfolio."
      >
        <div className="stripset stripset--split">
          <div>
            <div className="strip__head">By grade — click to drill in</div>
            <ConcentrationStrips
              rows={R.byGrade.map((g) => ({ name: g.label, el: g.expected_loss, expo: g.exposure }))}
              onSelect={(name) => nav(`/portfolio?grade=${name.replace("Grade ", "")}`)}
            />
          </div>
          <div>
            <div className="strip__head">By loan purpose</div>
            <ConcentrationStrips
              rows={R.byPurpose.slice(0, 7).map((p) => ({ name: p.label, el: p.expected_loss, expo: p.exposure }))}
              onSelect={(name) => nav(`/portfolio?purpose=${encodeURIComponent(name)}`)}
            />
          </div>
          <div>
            <div className="strip__head">By credit-score band</div>
            <ConcentrationStrips
              rows={R.byFicoBand.map((f) => ({ name: f.label, el: f.exposure * f.avg_pd * 0.5, expo: f.exposure }))}
            />
          </div>
          <div>
            <div className="strip__head">By origination vintage</div>
            <ConcentrationStrips
              rows={R.byVintage.map((v) => ({ name: v.label, el: v.expected_loss, expo: v.exposure }))}
            />
          </div>
        </div>
        <p className="note">
          Grades <b>C and D</b> carry {pct((R.byGrade[2].expected_loss + R.byGrade[3].expected_loss) / R.totals.expected_loss, 0)}{" "}
          of expected loss on {pct((R.byGrade[2].exposure + R.byGrade[3].exposure) / totalExposure, 0)} of exposure — the book's
          risk sits one notch below the middle.
        </p>
      </Section>

      <Section title="Needs attention" desc="Movements outside the expected range, ordered by urgency." aside={<DemoBadge what="monitoring alerts" />}>
        <Card variant="flush">
          <div style={{ padding: "0 var(--s-6)" }}>
            <AlertList items={alerts} />
          </div>
        </Card>
      </Section>
    </div>
  );
}

function ConcentrationStrips({
  rows,
  onSelect,
}: {
  rows: { name: string; el: number; expo: number }[];
  onSelect?: (name: string) => void;
}) {
  const totalEl = rows.reduce((s, r) => s + r.el, 0);
  const totalExpo = rows.reduce((s, r) => s + r.expo, 0);
  const max = Math.max(...rows.map((r) => r.el / totalEl));
  return (
    <div>
      {rows.map((r) => {
        const elShare = r.el / totalEl;
        const expoShare = r.expo / totalExpo;
        const ratio = elShare / expoShare;
        return (
          <div
            key={r.name}
            className={`strip__row${onSelect ? " strip__row--link" : ""}`}
            onClick={onSelect ? () => onSelect(r.name) : undefined}
            role={onSelect ? "button" : undefined}
            tabIndex={onSelect ? 0 : undefined}
            onKeyDown={onSelect ? (e) => e.key === "Enter" && onSelect(r.name) : undefined}
          >
            <span className="strip__name">{r.name}</span>
            <span className="strip__track">
              <span
                className="strip__fill"
                style={{
                  width: `${(elShare / max) * 100}%`,
                  background: ratio > 1.15 ? "var(--risk-high)" : ratio > 0.9 ? "var(--risk-med)" : "var(--ink-3)",
                }}
              />
            </span>
            <span className="strip__val">
              {pct(elShare, 0)} of loss · {ratio.toFixed(2)}×
            </span>
          </div>
        );
      })}
    </div>
  );
}
