import { useRef, useState } from "react";
import { linear, smoothPath, ticks } from "../lib/scales";

function niceMax(v: number): number {
  if (v <= 0) return 1;
  const mag = Math.pow(10, Math.floor(Math.log10(v)));
  const n = v / mag;
  const step = n <= 1 ? 1 : n <= 1.5 ? 1.5 : n <= 2 ? 2 : n <= 3 ? 3 : n <= 5 ? 5 : n <= 7.5 ? 7.5 : 10;
  return step * mag;
}

/* ------------------------------------------------------------------ */
/*  Sparkline — tiny real-data line, no axis                          */
/* ------------------------------------------------------------------ */
export function Sparkline({
  values,
  width = 96,
  height = 26,
  color = "var(--ink-3)",
}: {
  values: number[];
  width?: number;
  height?: number;
  color?: string;
}) {
  if (values.length < 2) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const x = linear([0, values.length - 1], [1, width - 1]);
  const y = linear([min, max], [height - 2, 2]);
  const pts = values.map((v, i) => [x(i), y(v)] as [number, number]);
  return (
    <svg width={width} height={height} aria-hidden="true">
      <path d={smoothPath(pts)} fill="none" stroke={color} strokeWidth={1.5} strokeLinecap="round" />
      <circle cx={pts[pts.length - 1][0]} cy={pts[pts.length - 1][1]} r={2} fill={color} />
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/*  LineChart — 1..3 series, minimal axes, hover crosshair            */
/* ------------------------------------------------------------------ */
export interface Series {
  label: string;
  color: string;
  values: number[];
}
export function LineChart({
  x: xLabels,
  series,
  height = 240,
  yFormat = (v) => `${v}`,
  yFrom0 = true,
  markers,
  smooth = true,
}: {
  x: string[];
  series: Series[];
  height?: number;
  yFormat?: (v: number) => string;
  yFrom0?: boolean;
  markers?: { at: number; label: string }[];
  smooth?: boolean;
}) {
  const W = 800;
  const H = height;
  const m = { t: 12, r: 16, b: 26, l: 52 };
  const [hi, setHi] = useState<number | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  const all = series.flatMap((s) => s.values);
  const rawMin = Math.min(...all);
  const rawMax = Math.max(...all);
  let yMin: number;
  let yMax: number;
  if (yFrom0) {
    yMin = 0;
    yMax = niceMax(rawMax * 1.06);
  } else {
    const pad = (rawMax - rawMin) * 0.28 || rawMax * 0.08 || 1;
    yMin = Math.max(0, rawMin - pad);
    yMax = rawMax + pad;
  }
  const x = linear([0, xLabels.length - 1], [m.l, W - m.r]);
  const y = linear([yMin, yMax], [H - m.b, m.t]);
  const yTicks = ticks(yMin, yMax, 4);

  function onMove(e: React.PointerEvent<SVGSVGElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const idx = Math.round(x.invert(px));
    setHi(Math.max(0, Math.min(xLabels.length - 1, idx)));
  }

  const step = Math.ceil(xLabels.length / 7);

  return (
    <div className="chart" ref={wrapRef} style={{ position: "relative" }}>
      <svg viewBox={`0 0 ${W} ${H}`} onPointerMove={onMove} onPointerLeave={() => setHi(null)}>
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} className="chart__hairline" />
            <text x={m.l - 8} y={y(t) + 3} textAnchor="end" className="chart__axis">
              {yFormat(t)}
            </text>
          </g>
        ))}
        <line x1={m.l} x2={W - m.r} y1={y(yMin)} y2={y(yMin)} className="chart__baseline" />
        {xLabels.map((lab, i) =>
          i % step === 0 || i === xLabels.length - 1 ? (
            <text key={i} x={x(i)} y={H - 8} textAnchor="middle" className="chart__axis">
              {lab}
            </text>
          ) : null,
        )}
        {markers?.map((mk) => (
          <g key={mk.label}>
            <line x1={x(mk.at)} x2={x(mk.at)} y1={m.t} y2={H - m.b} stroke="var(--line-strong)" strokeDasharray="3 3" />
            <text x={x(mk.at)} y={m.t + 2} textAnchor="middle" className="chart__label" dy={-2}>
              {mk.label}
            </text>
          </g>
        ))}
        {series.map((s) => {
          const pts = s.values.map((v, i) => [x(i), y(v)] as [number, number]);
          return (
            <path
              key={s.label}
              d={smooth ? smoothPath(pts) : pts.map((p, i) => `${i ? "L" : "M"} ${p[0]} ${p[1]}`).join(" ")}
              className="chart__series"
              stroke={s.color}
            />
          );
        })}
        {hi !== null && (
          <>
            <line x1={x(hi)} x2={x(hi)} y1={m.t} y2={H - m.b} className="chart__crosshair" />
            {series.map((s) => (
              <circle key={s.label} cx={x(hi)} cy={y(s.values[hi])} r={3.5} fill={s.color} className="chart__dot" />
            ))}
          </>
        )}
      </svg>
      {hi !== null && (
        <div className="tooltip" style={{ left: `${(x(hi) / W) * 100}%`, top: `${(y(Math.max(...series.map((s) => s.values[hi]))) / H) * 100}%` }}>
          <b>{xLabels[hi]}</b>
          {series.map((s) => (
            <div className="tooltip__row" key={s.label}>
              <i style={{ background: s.color }} />
              {s.label} {yFormat(s.values[hi])}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  BarStrip — compact horizontal bars (where risk concentrates)      */
/* ------------------------------------------------------------------ */
export function BarStrip({
  rows,
  valueFormat,
  color = "var(--ink-2)",
  onSelect,
  highlight,
}: {
  rows: { name: string; value: number; sub?: string }[];
  valueFormat: (v: number) => string;
  color?: string;
  onSelect?: (name: string) => void;
  /** when set, dims every row that doesn't match (case-insensitive substring) */
  highlight?: string | null;
}) {
  const max = Math.max(...rows.map((r) => r.value));
  const hl = highlight?.toLowerCase();
  return (
    <div>
      {rows.map((r) => {
        const matched = !hl || r.name.toLowerCase().includes(hl);
        return (
          <div
            key={r.name}
            className={`strip__row${onSelect ? " strip__row--link" : ""}`}
            onClick={onSelect ? () => onSelect(r.name) : undefined}
            role={onSelect ? "button" : undefined}
            tabIndex={onSelect ? 0 : undefined}
            onKeyDown={onSelect ? (e) => e.key === "Enter" && onSelect(r.name) : undefined}
            style={{ opacity: matched ? 1 : 0.35, transition: "opacity var(--t-fade) var(--ease)" }}
          >
            <span className="strip__name" style={{ fontWeight: hl && matched ? 600 : undefined }}>
              {r.name}
            </span>
            <span className="strip__track">
              <span className="strip__fill" style={{ width: `${(r.value / max) * 100}%`, background: color }} />
            </span>
            <span className="strip__val">
              {valueFormat(r.value)}
              {r.sub ? ` · ${r.sub}` : ""}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  BarPair — grouped vertical bars (approval vs rejection etc.)      */
/* ------------------------------------------------------------------ */
export function BarPair({
  groups,
  a,
  b,
  yFormat = (v) => `${(v * 100).toFixed(0)}%`,
  height = 220,
}: {
  groups: string[];
  a: { label: string; color: string; values: number[] };
  b: { label: string; color: string; values: number[] };
  yFormat?: (v: number) => string;
  height?: number;
}) {
  const W = 760;
  const H = height;
  const m = { t: 12, r: 12, b: 30, l: 46 };
  const [hi, setHi] = useState<number | null>(null);
  const yMax = niceMax(Math.max(...a.values, ...b.values) * 1.05);
  const y = linear([0, yMax], [H - m.b, m.t]);
  const bandW = (W - m.l - m.r) / groups.length;
  const barW = Math.min(26, bandW * 0.3);
  const yTicks = ticks(0, yMax, 4);
  return (
    <div className="chart" style={{ position: "relative" }}>
      <svg viewBox={`0 0 ${W} ${H}`}>
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} className="chart__hairline" />
            <text x={m.l - 8} y={y(t) + 3} textAnchor="end" className="chart__axis">
              {yFormat(t)}
            </text>
          </g>
        ))}
        <line x1={m.l} x2={W - m.r} y1={y(0)} y2={y(0)} className="chart__baseline" />
        {groups.map((g, i) => {
          const cx = m.l + bandW * i + bandW / 2;
          return (
            <g key={g} onPointerEnter={() => setHi(i)} onPointerLeave={() => setHi(null)}>
              <rect
                x={cx - barW - 2}
                y={y(a.values[i])}
                width={barW}
                height={y(0) - y(a.values[i])}
                fill={a.color}
                rx={2}
                opacity={hi === null || hi === i ? 1 : 0.4}
              />
              <rect
                x={cx + 2}
                y={y(b.values[i])}
                width={barW}
                height={y(0) - y(b.values[i])}
                fill={b.color}
                rx={2}
                opacity={hi === null || hi === i ? 1 : 0.4}
              />
              <text x={cx} y={H - 10} textAnchor="middle" className="chart__axis">
                {g}
              </text>
            </g>
          );
        })}
      </svg>
      <div style={{ display: "flex", gap: 18, marginTop: 8 }}>
        {[a, b].map((s) => (
          <span key={s.label} className="spectrum__key">
            <i style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
      {hi !== null && (
        <div
          className="tooltip"
          style={{ left: `${((m.l + bandW * hi + bandW / 2) / W) * 100}%`, top: "6%" }}
        >
          <b>{groups[hi]}</b>
          <div className="tooltip__row">
            <i style={{ background: a.color }} />
            {a.label} {yFormat(a.values[hi])}
          </div>
          <div className="tooltip__row">
            <i style={{ background: b.color }} />
            {b.label} {yFormat(b.values[hi])}
          </div>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  CalibrationChart — predicted vs observed dots + 45° reference     */
/* ------------------------------------------------------------------ */
export function CalibrationChart({
  points,
  height = 300,
}: {
  points: { pred: number; obs: number; n: number }[];
  height?: number;
}) {
  const W = 480;
  const H = height;
  const m = { t: 14, r: 14, b: 40, l: 44 };
  const max = Math.max(...points.flatMap((p) => [p.pred, p.obs])) * 1.08;
  const x = linear([0, max], [m.l, W - m.r]);
  const y = linear([0, max], [H - m.b, m.t]);
  const t = ticks(0, max, 4);
  const [hi, setHi] = useState<number | null>(null);
  return (
    <div className="chart" style={{ position: "relative", maxWidth: W }}>
      <svg viewBox={`0 0 ${W} ${H}`}>
        {t.map((v) => (
          <g key={v}>
            <line x1={m.l} x2={W - m.r} y1={y(v)} y2={y(v)} className="chart__hairline" />
            <text x={m.l - 6} y={y(v) + 3} textAnchor="end" className="chart__axis">
              {(v * 100).toFixed(0)}%
            </text>
            <text x={x(v)} y={H - 22} textAnchor="middle" className="chart__axis">
              {(v * 100).toFixed(0)}%
            </text>
          </g>
        ))}
        <line x1={x(0)} y1={y(0)} x2={x(max)} y2={y(max)} stroke="var(--line-strong)" strokeDasharray="4 4" />
        <path
          d={points.map((p, i) => `${i ? "L" : "M"} ${x(p.pred)} ${y(p.obs)}`).join(" ")}
          fill="none"
          stroke="var(--accent)"
          strokeWidth={1.5}
          opacity={0.5}
        />
        {points.map((p, i) => (
          <circle
            key={i}
            cx={x(p.pred)}
            cy={y(p.obs)}
            r={hi === i ? 5 : 4}
            fill="var(--accent)"
            className="chart__dot"
            onPointerEnter={() => setHi(i)}
            onPointerLeave={() => setHi(null)}
          />
        ))}
        <text x={(W) / 2} y={H - 4} textAnchor="middle" className="chart__label">
          Predicted default probability
        </text>
        <text
          x={12}
          y={H / 2}
          textAnchor="middle"
          className="chart__label"
          transform={`rotate(-90 12 ${H / 2})`}
        >
          Observed default rate
        </text>
      </svg>
      {hi !== null && (
        <div className="tooltip" style={{ left: `${(x(points[hi].pred) / W) * 100}%`, top: `${(y(points[hi].obs) / H) * 100}%` }}>
          <b>Decile {hi + 1}</b>
          <div>predicted {(points[hi].pred * 100).toFixed(1)}%</div>
          <div>observed {(points[hi].obs * 100).toFixed(1)}%</div>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  ContributionBars — diverging ± from a zero baseline               */
/* ------------------------------------------------------------------ */
export function ContributionBars({
  rows,
  expanded,
  onToggle,
}: {
  rows: { factor: string; contribution: number; note: string }[];
  expanded: string | null;
  onToggle: (f: string) => void;
}) {
  const maxAbs = Math.max(...rows.map((r) => Math.abs(r.contribution)), 1);
  return (
    <div>
      {rows.map((r) => {
        const pos = r.contribution >= 0;
        const w = (Math.abs(r.contribution) / maxAbs) * 50;
        const open = expanded === r.factor;
        return (
          <div className="contrib__row" key={r.factor}>
            <div
              className="contrib__top"
              role="button"
              tabIndex={0}
              aria-expanded={open}
              onClick={() => onToggle(r.factor)}
              onKeyDown={(e) => e.key === "Enter" && onToggle(r.factor)}
            >
              <span className="contrib__name">{r.factor}</span>
              <span className={`contrib__val contrib__val--${pos ? "pos" : "neg"}`}>
                {pos ? "+" : "−"}
                {Math.abs(r.contribution)}
              </span>
            </div>
            <div className="contrib__bar">
              <span className="contrib__zero" style={{ left: "50%" }} />
              <i
                style={{
                  left: pos ? "50%" : `${50 - w}%`,
                  width: `${w}%`,
                  background: pos ? "var(--risk-high)" : "var(--risk-low)",
                }}
              />
            </div>
            {open && <div className="contrib__note">{r.note}</div>}
          </div>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Distribution — histogram (loan count by band)                     */
/* ------------------------------------------------------------------ */
export function Distribution({
  bins,
  color = "var(--ink-2)",
  height = 200,
  valueFormat = (v) => v.toLocaleString("en-US"),
  highlight,
}: {
  bins: { label: string; value: number; tint?: string }[];
  color?: string;
  height?: number;
  valueFormat?: (v: number) => string;
  /** when set, dims every bin whose label doesn't match */
  highlight?: string | null;
}) {
  const W = 760;
  const H = height;
  const m = { t: 10, r: 8, b: 28, l: 8 };
  const max = niceMax(Math.max(...bins.map((b) => b.value)));
  const bw = (W - m.l - m.r) / bins.length;
  const y = linear([0, max], [H - m.b, m.t]);
  const [hi, setHi] = useState<number | null>(null);
  return (
    <div className="chart" style={{ position: "relative" }}>
      <svg viewBox={`0 0 ${W} ${H}`}>
        <line x1={m.l} x2={W - m.r} y1={y(0)} y2={y(0)} className="chart__baseline" />
        {bins.map((b, i) => (
          <g key={b.label} onPointerEnter={() => setHi(i)} onPointerLeave={() => setHi(null)}>
            <rect
              x={m.l + bw * i + bw * 0.14}
              y={y(b.value)}
              width={bw * 0.72}
              height={y(0) - y(b.value)}
              fill={b.tint ?? color}
              rx={2}
              opacity={
                (hi === null || hi === i) && (!highlight || b.label === highlight)
                  ? 1
                  : hi === i
                    ? 1
                    : 0.3
              }
            />
            <text x={m.l + bw * i + bw / 2} y={H - 10} textAnchor="middle" className="chart__axis">
              {b.label}
            </text>
          </g>
        ))}
      </svg>
      {hi !== null && (
        <div className="tooltip" style={{ left: `${((m.l + bw * hi + bw / 2) / W) * 100}%`, top: `${(y(bins[hi].value) / H) * 100}%` }}>
          <b>{bins[hi].label}</b> {valueFormat(bins[hi].value)}
        </div>
      )}
    </div>
  );
}
