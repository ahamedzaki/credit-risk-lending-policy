import { useState } from "react";
import { backtest, backtestMeta, totals } from "../data/real";
import { interpCurve } from "../lib/curve";
import { linear, niceMax, ticks } from "../lib/scales";
import { count, pct, usdCompact } from "../lib/format";

const APPROVED_EXPOSURE_AT_FULL = totals.exposure; // exposure scales with approval share

export function ThresholdSimulator() {
  const [cut, setCut] = useState(0.15);
  const p = interpCurve(backtest, cut);
  const optimum = backtestMeta.realized_optimum;

  const approvedExposure = APPROVED_EXPOSURE_AT_FULL * (p.approval_rate / 0.9956);
  const approvedLoans = totals.loans * p.approval_rate;
  const rejectedLoans = totals.loans * (1 - p.approval_rate);
  const expectedDefaults = approvedLoans * p.actual_default_rate;

  // mini trade-off chart: approval_rate (x) vs realized_profit (y), marker at cut
  const W = 520;
  const H = 200;
  const m = { t: 12, r: 14, b: 26, l: 46 };
  const xs = backtest.map((d) => d.approval_rate);
  const ys = backtest.map((d) => d.realized_profit);
  const x = linear([Math.min(...xs), Math.max(...xs)], [m.l, W - m.r]);
  const yMax = niceMax(Math.max(...ys) * 1.1);
  const y = linear([0, yMax], [H - m.b, m.t]);
  const path = backtest.map((d, i) => `${i ? "L" : "M"} ${x(d.approval_rate)} ${y(d.realized_profit)}`).join(" ");
  const cur = { x: x(p.approval_rate), y: y(p.realized_profit) };
  const opt = { x: x(optimum.approval_rate), y: y(optimum.realized_profit) };

  return (
    <div className="sim">
      <div className="sim__control">
        <label htmlFor="thr">Approve if PD is below</label>
        <div className="sim__value">{pct(cut, 1)}</div>
        <div className="sim__hint">
          {cut < optimum.pd_cut - 0.005
            ? "Tighter than the profit-maximising cut-off — leaving good loans on the table."
            : cut > optimum.pd_cut + 0.02
              ? "Looser than the optimum — the marginal loans no longer clear their loss."
              : "Near the realised-profit optimum."}
        </div>
        <input
          id="thr"
          className="sim__slider"
          type="range"
          min={backtestMeta.min_cut}
          max={backtestMeta.max_cut}
          step={0.005}
          value={cut}
          onChange={(e) => setCut(Number(e.target.value))}
        />
        <div className="sim__ticks">
          <span>{pct(backtestMeta.min_cut, 0)}</span>
          <span>optimum {pct(optimum.pd_cut, 0)}</span>
          <span>{pct(backtestMeta.max_cut, 0)}</span>
        </div>

        <div className="sim__grid">
          <div className="sim__cell">
            <b>{pct(p.approval_rate, 0)}</b>
            <span>Approved</span>
            <br />
            <em>{count(approvedLoans)} loans</em>
          </div>
          <div className="sim__cell">
            <b>{pct(1 - p.approval_rate, 0)}</b>
            <span>Rejected</span>
            <br />
            <em>{count(rejectedLoans)} loans</em>
          </div>
          <div className="sim__cell">
            <b>{pct(p.actual_default_rate, 1)}</b>
            <span>Actual default rate</span>
            <br />
            <em>{count(expectedDefaults)} defaults</em>
          </div>
          <div className="sim__cell">
            <b>{usdCompact(approvedExposure)}</b>
            <span>Portfolio exposure</span>
          </div>
          <div className="sim__cell">
            <b>{usdCompact(p.realized_credit_loss)}</b>
            <span>Realised credit loss</span>
            <br />
            <em>model EL {usdCompact(p.model_expected_loss)}</em>
          </div>
          <div className="sim__cell">
            <b>{usdCompact(p.realized_profit)}</b>
            <span>Realised profit (T=3)</span>
            <br />
            <em>model projects {usdCompact(p.model_profit)}</em>
          </div>
        </div>

        <p className="tradeoff">
          Growth ↔ risk: moving the cut-off from {pct(cut, 0)} toward{" "}
          {pct(Math.min(cut + 0.05, backtestMeta.max_cut), 0)} would approve{" "}
          <b>
            {count(
              totals.loans *
                (interpCurve(backtest, Math.min(cut + 0.05, backtestMeta.max_cut)).approval_rate -
                  p.approval_rate),
            )}
          </b>{" "}
          more loans and add{" "}
          <b>
            {usdCompact(
              interpCurve(backtest, Math.min(cut + 0.05, backtestMeta.max_cut)).realized_credit_loss -
                p.realized_credit_loss,
            )}
          </b>{" "}
          of realised loss. The realised-profit optimum is <b>PD &lt; {pct(optimum.pd_cut, 0)}</b>{" "}
          ({pct(optimum.approval_rate, 0)} approval); following the model's own optimum instead
          costs only {usdCompact(backtestMeta.regret)} on a ${(totals.exposure / 1e9).toFixed(0)}B book.
        </p>
      </div>

      <div>
        <div className="card__title">Realised profit vs approval rate</div>
        <div className="card__note">
          Every point is the pipeline's out-of-time backtest on 333,721 test loans. The
          curve turns over — approving past the optimum trades profit for volume.
        </div>
        <div className="chart" style={{ position: "relative", marginTop: 16 }}>
          <svg viewBox={`0 0 ${W} ${H}`}>
            {ticks(0, yMax, 4).map((t) => (
              <g key={t}>
                <line x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} className="chart__hairline" />
                <text x={m.l - 6} y={y(t) + 3} textAnchor="end" className="chart__axis">
                  {usdCompact(t, { decimals: 0 })}
                </text>
              </g>
            ))}
            <line x1={m.l} x2={W - m.r} y1={y(0)} y2={y(0)} className="chart__baseline" />
            {[0.2, 0.4, 0.6, 0.8, 1].map((t) => (
              <text key={t} x={x(t)} y={H - 8} textAnchor="middle" className="chart__axis">
                {pct(t, 0)}
              </text>
            ))}
            <path d={path} className="chart__series" stroke="var(--accent)" />
            <line x1={opt.x} x2={opt.x} y1={m.t} y2={H - m.b} stroke="var(--line-strong)" strokeDasharray="3 3" />
            <circle cx={opt.x} cy={opt.y} r={3} fill="var(--ink-3)" />
            <text x={opt.x} y={m.t} textAnchor="middle" className="chart__label" dy={-1}>
              optimum
            </text>
            <circle cx={cur.x} cy={cur.y} r={5} fill="var(--accent)" className="chart__dot" />
          </svg>
        </div>
        <div style={{ display: "flex", gap: 18, marginTop: 6 }}>
          <span className="spectrum__key"><i style={{ background: "var(--accent)" }} /> Realised profit</span>
          <span className="spectrum__key"><i style={{ background: "var(--ink-3)" }} /> Current cut-off</span>
        </div>
      </div>
    </div>
  );
}
