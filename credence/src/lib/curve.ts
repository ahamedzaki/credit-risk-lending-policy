import type { BacktestPoint } from "../data/types";

/**
 * Linear interpolation along the real out-of-time backtest curve by pd_cut.
 * Clamps to the observed range — never extrapolates past what the pipeline measured.
 */
export function interpCurve(points: BacktestPoint[], pdCut: number): BacktestPoint {
  const sorted = [...points].sort((a, b) => a.pd_cut - b.pd_cut);
  const lo = sorted[0];
  const hi = sorted[sorted.length - 1];
  if (pdCut <= lo.pd_cut) return lo;
  if (pdCut >= hi.pd_cut) return hi;
  let a = lo;
  let b = hi;
  for (let i = 0; i < sorted.length - 1; i++) {
    if (sorted[i].pd_cut <= pdCut && sorted[i + 1].pd_cut >= pdCut) {
      a = sorted[i];
      b = sorted[i + 1];
      break;
    }
  }
  const t = (pdCut - a.pd_cut) / (b.pd_cut - a.pd_cut || 1);
  const lerp = (k: keyof BacktestPoint) =>
    (a[k] as number) + t * ((b[k] as number) - (a[k] as number));
  return {
    pd_cut: pdCut,
    approval_rate: lerp("approval_rate"),
    pred_default_rate: lerp("pred_default_rate"),
    actual_default_rate: lerp("actual_default_rate"),
    model_expected_loss: lerp("model_expected_loss"),
    realized_credit_loss: lerp("realized_credit_loss"),
    model_profit: lerp("model_profit"),
    realized_profit: lerp("realized_profit"),
    n: Math.round(lerp("n")),
  };
}

export function curveExtent(points: BacktestPoint[], key: keyof BacktestPoint): [number, number] {
  const vals = points.map((p) => p[key] as number);
  return [Math.min(...vals), Math.max(...vals)];
}
