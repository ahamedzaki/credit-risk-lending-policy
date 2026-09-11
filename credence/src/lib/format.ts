/** Formatting helpers. Currency is USD (the pipeline's unit). */

export function usdCompact(n: number, opts: { decimals?: number } = {}): string {
  const d = opts.decimals ?? 1;
  const abs = Math.abs(n);
  if (abs >= 1e9) return `$${(n / 1e9).toFixed(d)}B`;
  if (abs >= 1e6) return `$${(n / 1e6).toFixed(d)}M`;
  if (abs >= 1e3) return `$${(n / 1e3).toFixed(abs >= 1e5 ? 0 : d)}K`;
  return `$${Math.round(n).toLocaleString("en-US")}`;
}

export function usd(n: number): string {
  return `$${Math.round(n).toLocaleString("en-US")}`;
}

export function count(n: number): string {
  return Math.round(n).toLocaleString("en-US");
}

export function pct(n: number, decimals = 1): string {
  return `${(n * 100).toFixed(decimals)}%`;
}

export function pp(n: number, decimals = 1): string {
  // percentage points, signed
  const v = (n * 100).toFixed(decimals);
  return `${n > 0 ? "+" : ""}${v} pp`;
}

export function signedPct(n: number, decimals = 1): string {
  return `${n > 0 ? "+" : ""}${(n * 100).toFixed(decimals)}%`;
}

export function fixed(n: number, decimals = 2): string {
  return n.toFixed(decimals);
}

export function ordinalDecile(i: number): string {
  return ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"][i] ?? `${i + 1}th`;
}
