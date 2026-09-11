import type { ReactNode } from "react";
import { Icon } from "./Icon";

/* ---------- Delta ---------- */
export function Delta({
  value,
  format = (v) => `${v > 0 ? "+" : ""}${(v * 100).toFixed(1)}%`,
  /** does an increase read as bad? (risk metrics: yes) */
  invert = false,
  neutral = false,
}: {
  value: number;
  format?: (v: number) => string;
  invert?: boolean;
  neutral?: boolean;
}) {
  const tone = neutral
    ? "flat"
    : value === 0
      ? "flat"
      : (value > 0) !== invert
        ? "pos"
        : "neg";
  const up = value > 0;
  return (
    <span className={`delta delta--${tone}`}>
      {value !== 0 && <Icon name={up ? "arrowUp" : "arrowDown"} size={12} strokeWidth={2} />}
      {format(value)}
    </span>
  );
}

/* ---------- Stat ---------- */
export function Stat({
  label,
  figure,
  sub,
  spark,
}: {
  label: string;
  figure: ReactNode;
  sub?: ReactNode;
  spark?: ReactNode;
}) {
  return (
    <div>
      <div className="stat__label">{label}</div>
      <div className="stat__figure">{figure}</div>
      {sub && <div className="stat__sub">{sub}</div>}
      {spark && <div className="stat__spark">{spark}</div>}
    </div>
  );
}

/* ---------- Section ---------- */
export function Section({
  title,
  desc,
  aside,
  children,
}: {
  title: string;
  desc?: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="section">
      <div className="section__head">
        <div>
          <h2>{title}</h2>
          {desc && <p>{desc}</p>}
        </div>
        {aside && <div className="section__aside">{aside}</div>}
      </div>
      {children}
    </section>
  );
}

/* ---------- Card ---------- */
export function Card({
  title,
  note,
  aside,
  variant,
  children,
}: {
  title?: string;
  note?: ReactNode;
  aside?: ReactNode;
  variant?: "inset" | "flush";
  children: ReactNode;
}) {
  return (
    <div className={`card${variant ? ` card--${variant}` : ""}`}>
      {(title || aside) && (
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12 }}>
          <div>
            {title && <div className="card__title">{title}</div>}
            {note && <div className="card__note">{note}</div>}
          </div>
          {aside}
        </div>
      )}
      <div className={title ? "card__body" : undefined}>{children}</div>
    </div>
  );
}

/* ---------- Segmented control ---------- */
export function Seg<T extends string>({
  options,
  value,
  onChange,
  labels,
}: {
  options: readonly T[];
  value: T;
  onChange: (v: T) => void;
  labels?: Partial<Record<T, string>>;
}) {
  return (
    <div className="seg" role="group">
      {options.map((o) => (
        <button key={o} aria-pressed={o === value} onClick={() => onChange(o)}>
          {labels?.[o] ?? o}
        </button>
      ))}
    </div>
  );
}

/* ---------- DemoBadge ---------- */
export function DemoBadge({ what = "synthetic data" }: { what?: string }) {
  return (
    <span
      className="demobadge"
      title={`Demo data — ${what}. Not a measured result from the pipeline.`}
    >
      <i /> Demo data
    </span>
  );
}

/* ---------- Switch ---------- */
export function Switch({
  on,
  onToggle,
  label,
}: {
  on: boolean;
  onToggle: () => void;
  label: string;
}) {
  return (
    <button
      className="switch"
      role="switch"
      aria-checked={on}
      aria-pressed={on}
      aria-label={label}
      onClick={onToggle}
    />
  );
}

/* ---------- RiskPill ---------- */
import { RISK_HEX } from "../lib/risk";
import type { RiskBand } from "../lib/risk";
export function RiskPill({ band }: { band: RiskBand }) {
  return (
    <span className="pill">
      <i style={{ background: RISK_HEX[band] }} />
      {band}
    </span>
  );
}

export function Crumb({ children }: { children: ReactNode }) {
  return (
    <div className="crumb">
      <Icon name="chevronRight" size={12} /> {children}
    </div>
  );
}
