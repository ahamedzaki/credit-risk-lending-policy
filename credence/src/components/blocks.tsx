import type { Alert, FairnessDim } from "../data/types";
import { RISK_HEX } from "../lib/risk";
import type { RiskBand } from "../lib/risk";
import { count, pct } from "../lib/format";

/* ---------- RiskSpectrum — the hero ---------- */
export function RiskSpectrum({
  bands,
}: {
  bands: { band: RiskBand; range: string; exposure: number; share: number; count: number }[];
}) {
  const elevated = bands.filter((b) => b.band === "High" || b.band === "Critical");
  const elevatedShare = elevated.reduce((s, b) => s + b.share, 0);
  const critical = bands.find((b) => b.band === "Critical");
  return (
    <div>
      <div className="spectrum" role="img" aria-label="Portfolio risk distribution by exposure">
        {bands.map((b) => (
          <div
            key={b.band}
            className="spectrum__seg"
            style={{ flex: `${b.share} 0 0`, background: RISK_HEX[b.band] }}
            title={`${b.band} — ${pct(b.share)} of exposure`}
          >
            <b>{b.band}</b>
            <span>{pct(b.share, 0)}</span>
          </div>
        ))}
      </div>
      <div className="spectrum__legend">
        {bands.map((b) => (
          <span className="spectrum__key" key={b.band}>
            <i style={{ background: RISK_HEX[b.band] }} />
            {b.band} · {b.range} · {count(b.count)} loans
          </span>
        ))}
      </div>
      <p className="spectrum__readout">
        <b>{pct(elevatedShare, 0)}</b> of funded exposure sits in <b>Elevated</b> risk or
        higher, and <b>{pct(critical?.share ?? 0, 0)}</b> ({count(critical?.count ?? 0)}{" "}
        loans) is <b>Critical</b> — PD at or above 20%, where the observed default rate is{" "}
        {pct(0.286, 0)}.
      </p>
    </div>
  );
}

/* ---------- FairnessSignal — real 4/5ths-rule adverse-impact flag ---------- */
const AIR_THRESHOLD = 0.8;

export function FairnessSignal({
  dims,
}: {
  dims: { key: string; label: string; dim: FairnessDim }[];
}) {
  const failing = dims
    .filter((d) => d.dim.air < AIR_THRESHOLD)
    .map((d) => {
      const lo = d.dim.groups.reduce((a, b) => (a.approval_rate < b.approval_rate ? a : b));
      const hi = d.dim.groups.reduce((a, b) => (a.approval_rate > b.approval_rate ? a : b));
      return { ...d, lo, hi };
    });
  if (!failing.length) return null;

  return (
    <div>
      {failing.map((f) => {
        const critical = f.dim.air < 0.5;
        const color = critical ? RISK_HEX.Critical : RISK_HEX.High;
        return (
          <div key={f.key} className="grade-conc" style={{ gridTemplateColumns: "128px 1fr 1fr" }}>
            <span style={{ fontWeight: 600 }}>{f.label}</span>
            <div className="dualbar">
              <i style={{ width: `${Math.min(100, (f.dim.air / AIR_THRESHOLD) * 100)}%`, background: color }} />
            </div>
            <span>
              AIR <b style={{ color }}>{f.dim.air.toFixed(2)}</b> — {f.lo.group} approved at{" "}
              {pct(f.lo.approval_rate / f.hi.approval_rate, 0)} the rate of {f.hi.group}, below the
              0.80 threshold
            </span>
          </div>
        );
      })}
      <p className="note">
        {failing.length} of {dims.length} disparate-impact checks fail the 4/5ths rule (adverse-impact
        ratio &lt; 0.80). Approved-book default rates stay within a few points across every group in
        each check — the gap sits in who gets approved, not in how those loans perform. This is a
        proxy check on income, region and home ownership, not a protected-class fair-lending test —
        Lending Club data carries no race, sex, age, or national-origin fields.
      </p>
    </div>
  );
}

/* ---------- AlertList — "needs attention" ---------- */
function sevHex(s: Alert["severity"]) {
  return s === "High" ? RISK_HEX.High : s === "Medium" ? RISK_HEX.Medium : RISK_HEX.Low;
}
export function AlertList({ items }: { items: Alert[] }) {
  return (
    <div className="alerts">
      {items.map((a) => (
        <div className="alert" key={a.id}>
          <span className="alert__dot" style={{ background: sevHex(a.severity) }} />
          <div>
            <div className="alert__title">{a.title}</div>
            <div className="alert__detail">{a.detail}</div>
            <div className="alert__meta">
              <span style={{ color: sevHex(a.severity) }} className="alert__sev">
                {a.severity}
              </span>
              <span>{a.segment}</span>
              <span>{a.window}</span>
            </div>
          </div>
          <div className="alert__mag">{a.magnitude}</div>
        </div>
      ))}
    </div>
  );
}

/* ---------- AlertTimeline — emerging risk ---------- */
export function AlertTimeline({ items }: { items: Alert[] }) {
  return (
    <div className="timeline">
      {items.map((a) => (
        <div className="timeline__item" key={a.id} style={{ color: sevHex(a.severity) }}>
          <div className="timeline__date">
            {new Date(a.date).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
          </div>
          <div className="alert__title" style={{ marginTop: 2, color: "var(--ink)" }}>
            {a.title}
          </div>
          <div className="alert__detail">{a.detail}</div>
          <div className="alert__meta">
            <span className="alert__sev" style={{ color: sevHex(a.severity) }}>
              {a.severity}
            </span>
            <span>{a.segment}</span>
            <span>{a.window}</span>
            <span style={{ color: "var(--ink)", fontWeight: 600 }}>{a.magnitude}</span>
          </div>
        </div>
      ))}
    </div>
  );
}
