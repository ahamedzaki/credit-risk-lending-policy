import { useEffect, useState } from "react";
import type { Borrower } from "../data/types";
import { titleFor } from "../data/demo";
import { RISK_HEX } from "../lib/risk";
import { count, fixed, pct, usd } from "../lib/format";
import { Icon } from "./Icon";
import { ContributionBars } from "./charts";
import { DemoBadge, RiskPill } from "./primitives";

function Def({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

export function BorrowerDrawer({ borrower, onClose }: { borrower: Borrower; onClose: () => void }) {
  const [driver, setDriver] = useState<string | null>(borrower.drivers[0]?.factor ?? null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [onClose]);

  const b = borrower;
  return (
    <>
      <div className="scrim" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-modal="true" aria-label={`Borrower ${b.borrower_id}`}>
        <div className="drawer__head">
          <div>
            <div className="drawer__id">{b.borrower_id} · {b.loan_id}</div>
            <div className="drawer__name">{titleFor(b)}</div>
            <div className="drawer__score">
              <b>{b.risk_score}</b>
              <RiskPill band={b.risk_category} />
              <span style={{ fontSize: 13, color: "var(--ink-2)" }}>
                PD {pct(b.probability_of_default)}
              </span>
            </div>
          </div>
          <button className="iconbtn" onClick={onClose} aria-label="Close">
            <Icon name="close" size={18} />
          </button>
        </div>

        <div className="drawer__body">
          <section>
            <div className="subhead">Financial profile</div>
            <dl className="deflist">
              <Def label="Annual income" value={usd(b.annual_income)} />
              <Def label="Debt-to-income" value={pct(b.debt_to_income, 0)} />
              <Def label="Employment" value={`${b.employment_type} · ${fixed(b.employment_length, 1)} yr`} />
              <Def label="Home" value={b.home_ownership} />
            </dl>
          </section>

          <section>
            <div className="subhead">Credit profile</div>
            <dl className="deflist">
              <Def label="Credit score" value={count(b.credit_score)} />
              <Def label="Credit history" value={`${b.credit_history_length} yr`} />
              <Def label="Delinquencies (24m)" value={count(b.delinquencies_2y)} />
              <Def label="Model PD" value={pct(b.probability_of_default)} />
            </dl>
          </section>

          <section>
            <div className="subhead">Loan information</div>
            <dl className="deflist">
              <Def label="Amount" value={usd(b.loan_amount)} />
              <Def label="Purpose" value={b.loan_purpose} />
              <Def label="Rate" value={pct(b.interest_rate)} />
              <Def label="Term" value={`${b.term} months`} />
              <Def label="Expected loss" value={usd(b.expected_loss)} />
              <Def label="Status" value={b.loan_status} />
            </dl>
          </section>

          <section>
            <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 4 }}>
              <div className="subhead" style={{ marginBottom: 0 }}>Why is this borrower risky?</div>
              <DemoBadge what="illustrative model attribution" />
            </div>
            <p style={{ fontSize: 12.5, color: "var(--ink-2)", margin: "0 0 12px", lineHeight: 1.55 }}>
              Points each factor adds to (orange) or removes from (green) the risk score of{" "}
              <b style={{ color: "var(--ink)" }}>{b.risk_score}</b>. Select a factor for the reasoning.
            </p>
            <ContributionBars
              rows={b.drivers}
              expanded={driver}
              onToggle={(f) => setDriver((d) => (d === f ? null : f))}
            />
          </section>

          <section>
            <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 4 }}>
              <div className="subhead" style={{ marginBottom: 0 }}>Decision explanation</div>
              <DemoBadge what="synthetic-borrower decision rule" />
            </div>
            <p style={{ fontSize: 13.5, lineHeight: 1.6, color: "var(--ink)" }}>
              Demo policy for this synthetic set: approve at <b>PD &lt; 12%</b>, decline at{" "}
              <b>PD ≥ 20%</b>, route the band between to manual review — illustrative, distinct
              from the pipeline's real backtested underwriting cut-off (PD &lt; 15%, see
              Decisions). This application scores{" "}
              <b>PD {pct(b.probability_of_default)}</b> →{" "}
              <b style={{ color: b.decision === "Declined" ? RISK_HEX.High : b.decision === "Approved" ? RISK_HEX.Low : RISK_HEX.Medium }}>
                {b.decision}
              </b>
              .{" "}
              {b.decision !== "Approved" &&
                `Reducing DTI to the book median or clearing the recent delinquency would move it toward approval.`}
            </p>
          </section>
        </div>
      </aside>
    </>
  );
}
