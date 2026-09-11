export type RiskBand = "Low" | "Medium" | "High" | "Critical";

export const BANDS: RiskBand[] = ["Low", "Medium", "High", "Critical"];

/** Default PD cut-offs; overridable in Settings. */
export const DEFAULT_CUTS = { medium: 0.05, high: 0.1, critical: 0.2 };

export function bandFor(
  pd: number,
  cuts: { medium: number; high: number; critical: number } = DEFAULT_CUTS,
): RiskBand {
  if (pd < cuts.medium) return "Low";
  if (pd < cuts.high) return "Medium";
  if (pd < cuts.critical) return "High";
  return "Critical";
}

export const RISK_VAR: Record<RiskBand, string> = {
  Low: "var(--risk-low)",
  Medium: "var(--risk-med)",
  High: "var(--risk-high)",
  Critical: "var(--risk-crit)",
};

export const RISK_WASH: Record<RiskBand, string> = {
  Low: "var(--risk-low-wash)",
  Medium: "var(--risk-med-wash)",
  High: "var(--risk-high-wash)",
  Critical: "var(--risk-crit-wash)",
};

export const RISK_HEX: Record<RiskBand, string> = {
  Low: "#3fa277",
  Medium: "#c79238",
  High: "#d2673b",
  Critical: "#c6453a",
};

export const BAND_RANGE_LABEL: Record<RiskBand, string> = {
  Low: "PD < 5%",
  Medium: "5–10%",
  High: "10–20%",
  Critical: "20%+",
};
