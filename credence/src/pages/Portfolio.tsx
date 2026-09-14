import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { PageHead } from "../components/AppShell";
import { BarStrip, Distribution, LineChart } from "../components/charts";
import { Card, Delta, Section, Stat } from "../components/primitives";
import { DemoBadge } from "../components/primitives";
import { monthly } from "../data/demo";
import * as R from "../data/real";
import { count, pct, usdCompact } from "../lib/format";
import { RISK_HEX } from "../lib/risk";

type FilterDim = "grade" | "purpose" | "ficoBand" | "incomeBand" | "vintage" | "region";

const FILTER_DEFS: { key: FilterDim; label: string; options: string[] }[] = [
  { key: "grade", label: "Grade", options: R.byGrade.map((g) => g.label) },
  { key: "purpose", label: "Purpose", options: R.byPurpose.map((p) => p.label) },
  { key: "ficoBand", label: "Credit score", options: R.byFicoBand.map((f) => f.label) },
  { key: "incomeBand", label: "Income", options: R.byIncomeBand.map((b) => b.label) },
  { key: "vintage", label: "Vintage", options: R.byVintage.map((v) => v.label) },
  { key: "region", label: "Region", options: R.fairness.region.groups.map((g) => g.group) },
];

export function Portfolio() {
  const [sp, setSp] = useSearchParams();
  const urlGrade = sp.get("grade");
  const urlPurpose = sp.get("purpose");
  const [filter, setFilter] = useState<{ dim: FilterDim; value: string } | null>(() => {
    if (urlGrade) return { dim: "grade", value: `Grade ${urlGrade}` };
    if (urlPurpose) return { dim: "purpose", value: urlPurpose };
    return null;
  });

  const totalExposure = R.totals.exposure;
  const dq = monthly.slice(-18);
  const maxGradeExposure = Math.max(...R.byGrade.map((g) => g.exposure));

  const purposeRows = useMemo(
    () => R.byPurpose.map((p) => ({ name: p.label, value: p.exposure, sub: pct(p.observed_default_rate, 1) })),
    [],
  );

  function pick(dim: FilterDim, value: string) {
    setFilter(value ? { dim, value } : null);
    if (sp.size) setSp({});
  }
  function clear() {
    setFilter(null);
    setSp({});
  }
  const hl = (dim: FilterDim): string | null => (filter?.dim === dim ? filter.value : null);
  const dim = (dim: FilterDim, value: string) => (hl(dim) && hl(dim) !== value ? 0.35 : 1);

  return (
    <div className="page">
      <PageHead
        title="Portfolio"
        lede="Understand how exposure and risk are distributed across the book."
      />

      <div className="metricline">
        <Stat label="Total loans" figure={count(R.totals.loans)} />
        <Stat label="Total exposure" figure={usdCompact(totalExposure)} />
        <Stat label="Avg loan size" figure={usdCompact(R.derived.avgLoanSize, { decimals: 1 })} />
        <Stat label="Avg interest rate" figure={pct(R.derived.avgInterestRate, 1)} />
        <Stat label="Default rate" figure={pct(R.totals.observed_default_rate, 1)} />
        <Stat label="Expected loss" figure={usdCompact(R.totals.expected_loss)} />
      </div>

      <div className="section" style={{ marginTop: 0 }}>
        <div className="section__head">
          <div>
            <h2>Filters</h2>
            <p>Pick a dimension and a value to highlight it across every breakdown below.</p>
          </div>
          {filter && (
            <button className="pill" style={{ cursor: "pointer" }} onClick={clear}>
              Clear · {filter.value}
            </button>
          )}
        </div>
        <div className="tabletools" style={{ border: "1px solid var(--line)", borderRadius: "var(--r-card)" }}>
          {FILTER_DEFS.map((f) => (
            <select
              key={f.key}
              className="filterselect"
              value={filter?.dim === f.key ? filter.value : ""}
              onChange={(e) => pick(f.key, e.target.value)}
              aria-label={`Filter by ${f.label}`}
            >
              <option value="">{f.label}</option>
              {f.options.map((o) => (
                <option key={o} value={o}>
                  {o}
                </option>
              ))}
            </select>
          ))}
        </div>
      </div>

      <Section title="Exposure by loan purpose" desc="Funded amount; label shows the observed default rate.">
        <BarStrip rows={purposeRows} valueFormat={(v) => usdCompact(v)} color="var(--ink-2)" highlight={hl("purpose")} />
      </Section>

      <Section title="Default rate by loan purpose" desc="Observed default rate on terminal loans — where the book actually loses money.">
        <BarStrip
          rows={R.byPurpose
            .slice()
            .sort((a, b) => b.observed_default_rate - a.observed_default_rate)
            .map((p) => ({ name: p.label, value: p.observed_default_rate }))}
          valueFormat={(v) => pct(v, 1)}
          color={RISK_HEX.High}
          highlight={hl("purpose")}
        />
      </Section>

      <Section title="Risk by grade" desc="Exposure, expected loss, and observed default rate by Lending Club grade.">
        <div>
          {R.byGrade.map((g) => (
            <div key={g.key} className="grade-conc" style={{ opacity: dim("grade", g.label), transition: "opacity var(--t-fade) var(--ease)" }}>
              <span style={{ fontWeight: 600 }}>{g.key}</span>
              <div className="dualbar">
                <i style={{ width: `${(g.exposure / maxGradeExposure) * 100}%`, background: "var(--ink-2)" }} />
              </div>
              <span>
                {usdCompact(g.exposure)} · EL {usdCompact(g.expected_loss)} · default {pct(g.observed_default_rate, 1)}
              </span>
            </div>
          ))}
        </div>
      </Section>

      <div className="cardgrid cardgrid--2">
        <Card title="Risk distribution by credit-score band" note="Loan count; tint by average PD.">
          <Distribution
            bins={R.byFicoBand.map((f) => ({
              label: f.label,
              value: f.loans,
              tint:
                f.avg_pd >= 0.15 ? RISK_HEX.High : f.avg_pd >= 0.1 ? RISK_HEX.Medium : RISK_HEX.Low,
            }))}
            highlight={hl("ficoBand")}
          />
        </Card>
        <Card title="Default rate by income band" note="Approved-book default rate (from the fairness analysis).">
          <BarStrip
            rows={R.byIncomeBand.map((b) => ({ name: b.label, value: b.observed_default_rate }))}
            valueFormat={(v) => pct(v, 1)}
            color={RISK_HEX.Medium}
            highlight={hl("incomeBand")}
          />
        </Card>
      </div>

      <Card
        title="Loss given default by grade"
        note="Same recovery-based method as the portfolio LGD, segmented by grade and credibility-weighted toward that flat figure for thin grades (Buhlmann) — this is what actually feeds Expected Loss, not a flat multiplier."
      >
        <BarStrip
          rows={R.lgdByGrade.map((r) => ({ name: `Grade ${r.grade}`, value: r.lgd, sub: count(r.n_charged_off_train) + " charge-offs" }))}
          valueFormat={(v) => pct(v, 0)}
          color={RISK_HEX.Medium}
          highlight={hl("grade")}
        />
        <p className="note">
          LGD runs from {pct(Math.min(...R.lgdByGrade.map((r) => r.lgd)), 0)} (Grade A) to{" "}
          {pct(Math.max(...R.lgdByGrade.map((r) => r.lgd)), 0)} (Grade E) — riskier grades recover less,
          not just default more. Grades F and G shrink back toward the flat {pct(R.totals.lgd, 0)}{" "}
          portfolio figure rather than trusting a raw estimate from a few hundred charged-off loans —
          without that credibility weighting, Grade G's own recovery data alone would say 60%.
        </p>
      </Card>

      <Section
        title="Delinquency trend"
        desc="Early-stage delinquency rate, trailing 18 months."
        aside={<DemoBadge what="monthly delinquency series" />}
      >
        <LineChart
          x={dq.map((m) => m.month.slice(2).replace("-", "/"))}
          series={[{ label: "Delinquency rate", color: RISK_HEX.Medium, values: dq.map((m) => m.delinquency_rate) }]}
          yFormat={(v) => `${(v * 100).toFixed(1)}%`}
          yFrom0={false}
          height={210}
        />
      </Section>

      <Section title="Exposure by geography" desc="US census region — funded exposure and approved-book default rate.">
        <div className="stripset">
          {R.fairness.region.groups.map((g) => (
            <div
              key={g.group}
              className="grade-conc"
              style={{ opacity: dim("region", g.group), transition: "opacity var(--t-fade) var(--ease)" }}
            >
              <span style={{ fontWeight: 600 }}>{g.group}</span>
              <div className="dualbar">
                <i style={{ width: `${g.approval_rate * 100}%`, background: "var(--ink-2)" }} />
              </div>
              <span>
                approval {pct(g.approval_rate, 0)} · default {pct(g.approved_book_default_rate, 1)}
              </span>
            </div>
          ))}
        </div>
        <p className="note">
          Approval rates are even across regions (adverse-impact ratio {R.fairness.region.air.toFixed(2)},
          within the 4/5ths rule) — geography is not where the risk differences sit.
        </p>
      </Section>

      <Section title="Vintage quality" desc="Average model PD and observed default rate by origination year.">
        <div className="stripset">
          {R.byVintage.map((v) => (
            <div
              key={v.key}
              className="grade-conc"
              style={{ opacity: dim("vintage", v.label), transition: "opacity var(--t-fade) var(--ease)" }}
            >
              <span style={{ fontWeight: 600 }}>{v.label}</span>
              <div className="dualbar">
                <i style={{ width: `${(v.avg_pd / 0.2) * 100}%`, background: "var(--accent)" }} />
              </div>
              <span>
                PD {pct(v.avg_pd, 1)} · default {pct(v.observed_default_rate, 1)} ·{" "}
                {usdCompact(v.exposure)}
                {v.observed_default_rate > v.avg_pd + 0.008 && (
                  <> · <Delta value={v.observed_default_rate - v.avg_pd} invert format={(x) => `${(x * 100).toFixed(1)}pt over model`} /></>
                )}
              </span>
            </div>
          ))}
        </div>
      </Section>
    </div>
  );
}
