"""Realized-outcome backtest of the lending-policy curve (council rec #1).

The simulator's profit-vs-approval curve is built from the model's OWN predicted PD, so a
sceptic can call the optimum circular. This re-runs the same policy sweep on the
out-of-time TEST loans using their ACTUAL outcomes — realized cash flows and realized
charge-offs — and checks whether the profit peak lands in the same place.

Per loan (test, terminal only):
    realized_net_cash = total_pymnt + recoveries - funded_amnt      # actual interest/fees earned, net of actual credit loss
    funding           = funded_amnt * r_f * T * b
    servicing         = funded_amnt * s   * T
    realized_profit   = realized_net_cash - funding - servicing
    realized_credit_loss = max(funded_amnt - total_rec_prncp - recoveries, 0)  if charged off else 0

Model side at each cut-off uses the same P&L skeleton but with predicted PD and the
data-estimated LGD already baked into mart_simulator_base.expected_loss.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load
from src.db import connect

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_test(con) -> pd.DataFrame:
    return con.execute(
        """
        SELECT s.loan_id, s.pd_hat, s.fico_mid, s.loan_amnt, s.int_rate, s.expected_loss,
               o.default_flag, o.funded_amnt, o.total_pymnt, o.total_rec_prncp, o.recoveries
        FROM mart_simulator_base s
        JOIN mart_loan_outcomes  o USING (loan_id)
        WHERE s.split_set = 'test' AND o.is_terminal AND o.default_flag IS NOT NULL
        """
    ).fetchdf()


def _curves(df: pd.DataFrame, fico_min: float, rf: float, s: float, T: float, b: float,
            n_points: int = 60) -> pd.DataFrame:
    fa = df["funded_amnt"].to_numpy()
    df = df.assign(
        realized_net_cash=df["total_pymnt"] + df["recoveries"] - fa,
        funding=fa * rf * T * b,
        servicing=fa * s * T,
        realized_credit_loss=np.where(
            df["default_flag"] == 1,
            np.clip(fa - df["total_rec_prncp"] - df["recoveries"], 0, None),
            0.0,
        ),
    )
    df["realized_profit"] = df["realized_net_cash"] - df["funding"] - df["servicing"]
    df["model_interest"] = df["loan_amnt"] * df["int_rate"] * T * b * (1.0 - df["pd_hat"])
    df["model_profit"] = df["model_interest"] - df["funding"] - df["servicing"] - df["expected_loss"]

    n_all = len(df)
    rows = []
    for cut in np.linspace(0.03, 0.40, n_points):
        a = df[(df["pd_hat"] < cut) & (df["fico_mid"] >= fico_min)]
        if len(a) == 0:
            continue
        rows.append(
            dict(
                pd_cut=float(cut),
                approval_rate=len(a) / n_all,
                pred_default_rate=float(a["pd_hat"].mean()),
                actual_default_rate=float(a["default_flag"].mean()),
                model_expected_loss=float(a["expected_loss"].sum()),
                realized_credit_loss=float(a["realized_credit_loss"].sum()),
                model_profit=float(a["model_profit"].sum()),
                realized_profit=float(a["realized_profit"].sum()),
                n=int(len(a)),
            )
        )
    return pd.DataFrame(rows)


def _plot(curve: pd.DataFrame, path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(curve["approval_rate"], curve["model_profit"] / 1e6, "o-", color="#c17f00", label="model-PD profit")
    ax1.plot(curve["approval_rate"], curve["realized_profit"] / 1e6, "o-", color="#1f4e79", label="realized profit")
    for c, col in [("model_profit", "#c17f00"), ("realized_profit", "#1f4e79")]:
        r = curve.loc[curve[c].idxmax()]
        ax1.axvline(r["approval_rate"], color=col, ls="--", lw=1)
    ax1.set_xlabel("approval rate")
    ax1.set_ylabel("portfolio profit ($M, T=3)")
    ax1.set_title("Profit vs approval — model PD vs realized outcomes")
    ax1.legend()

    ax2.plot(curve["approval_rate"], curve["pred_default_rate"], "o-", color="#c17f00", label="predicted default rate")
    ax2.plot(curve["approval_rate"], curve["actual_default_rate"], "o-", color="#1f4e79", label="actual default rate")
    ax2.set_xlabel("approval rate")
    ax2.set_ylabel("default rate of the approved book")
    ax2.set_title("Decision calibration: predicted vs actual")
    ax2.legend()
    fig.tight_layout()
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> dict:
    cfg = load()
    sim = cfg["simulator"]
    con = connect(cfg["paths"]["duckdb"])
    df = _load_test(con)
    curve = _curves(
        df, sim["default_min_fico"], sim["cost_of_funds"], sim["servicing_cost"],
        sim["horizon_years"], sim["amort_factor"],
    )
    con.close()

    model_opt = curve.loc[curve["model_profit"].idxmax()]
    realized_opt = curve.loc[curve["realized_profit"].idxmax()]
    # profit left on the table by following the model optimum instead of the realized one
    at_model = curve.iloc[(curve["pd_cut"] - model_opt["pd_cut"]).abs().idxmin()]
    regret = float(realized_opt["realized_profit"] - at_model["realized_profit"])

    out = {
        "fico_min": sim["default_min_fico"],
        "model_optimum": {k: float(model_opt[k]) for k in
                          ["pd_cut", "approval_rate", "pred_default_rate", "actual_default_rate",
                           "model_profit", "realized_profit"]},
        "realized_optimum": {k: float(realized_opt[k]) for k in
                             ["pd_cut", "approval_rate", "actual_default_rate",
                              "realized_credit_loss", "realized_profit"]},
        "regret_following_model_optimum": regret,
        "loss_ratio_realized_over_model": float(
            curve["realized_credit_loss"].sum() / max(curve["model_expected_loss"].sum(), 1)
        ),
        "curve": curve.to_dict(orient="records"),
    }
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "backtest.json").write_text(json.dumps(out, indent=2))
    _plot(curve, f"{cfg['paths']['figures_dir']}/backtest_profit.png")

    print(f"   model optimum    : PD<{model_opt['pd_cut']:.3f}  approval {model_opt['approval_rate']:.1%}")
    print(f"   realized optimum : PD<{realized_opt['pd_cut']:.3f}  approval {realized_opt['approval_rate']:.1%}")
    print(f"   regret (following model opt) : ${regret/1e6:,.1f}M")
    print(f"   realized loss / model EL     : {out['loss_ratio_realized_over_model']:.2f}x")
    return out


if __name__ == "__main__":
    main()
