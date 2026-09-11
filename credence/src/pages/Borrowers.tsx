import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { PageHead } from "../components/AppShell";
import { BorrowerDrawer } from "../components/BorrowerDrawer";
import { DataTable, type Column } from "../components/DataTable";
import { DemoBadge, RiskPill, Seg, Stat } from "../components/primitives";
import { borrowers } from "../data/demo";
import type { Borrower } from "../data/types";
import { count, pct, usd, usdCompact } from "../lib/format";
import { RISK_HEX } from "../lib/risk";
import type { RiskBand } from "../lib/risk";

const BAND_FILTER = ["All", "Low", "Medium", "High", "Critical"] as const;

export function Borrowers() {
  const [sp, setSp] = useSearchParams();
  const [band, setBand] = useState<(typeof BAND_FILTER)[number]>("All");
  const selectedId = sp.get("id");
  const selected = borrowers.find((b) => b.borrower_id === selectedId) ?? null;

  const rows = useMemo(
    () => (band === "All" ? borrowers : borrowers.filter((b) => b.risk_category === band)),
    [band],
  );

  const declined = borrowers.filter((b) => b.decision === "Declined").length;
  const review = borrowers.filter((b) => b.decision === "Manual review").length;
  const avgPd = borrowers.reduce((s, b) => s + b.probability_of_default, 0) / borrowers.length;

  const columns: Column<Borrower>[] = [
    { key: "borrower_id", header: "Borrower", render: (r) => r.borrower_id, sortValue: (r) => r.borrower_id },
    { key: "credit_score", header: "Score", num: true, render: (r) => count(r.credit_score), sortValue: (r) => r.credit_score },
    { key: "annual_income", header: "Income", num: true, render: (r) => usdCompact(r.annual_income, { decimals: 0 }), sortValue: (r) => r.annual_income },
    { key: "loan_amount", header: "Amount", num: true, render: (r) => usdCompact(r.loan_amount, { decimals: 0 }), sortValue: (r) => r.loan_amount },
    { key: "dti", header: "DTI", num: true, render: (r) => pct(r.debt_to_income, 0), sortValue: (r) => r.debt_to_income },
    { key: "employment_type", header: "Employment", render: (r) => r.employment_type, sortValue: (r) => r.employment_type },
    { key: "loan_purpose", header: "Purpose", render: (r) => r.loan_purpose, sortValue: (r) => r.loan_purpose },
    {
      key: "risk_score",
      header: "Risk score",
      num: true,
      render: (r) => <b style={{ color: RISK_HEX[r.risk_category] }}>{r.risk_score}</b>,
      sortValue: (r) => r.risk_score,
    },
    { key: "risk_category", header: "Category", render: (r) => <RiskPill band={r.risk_category} />, sortValue: (r) => ["Low", "Medium", "High", "Critical"].indexOf(r.risk_category) },
    { key: "pd", header: "Default prob.", num: true, render: (r) => pct(r.probability_of_default, 1), sortValue: (r) => r.probability_of_default },
    { key: "decision", header: "Decision", render: (r) => <span className={`decision decision--${r.decision}`}>{r.decision}</span>, sortValue: (r) => r.decision },
  ];

  return (
    <div className="page">
      <PageHead
        title="Borrowers"
        lede="Investigate individual credit risk. Sort by default probability, then open a borrower for the model's reasoning."
        aside={<DemoBadge what="220 synthetic applications" />}
      />

      <div className="statrow">
        <Stat label="Applications" figure={count(borrowers.length)} />
        <Stat label="Avg default probability" figure={pct(avgPd, 1)} />
        <Stat label="Manual review" figure={count(review)} sub={<span>{pct(review / borrowers.length, 0)} of pipeline</span>} />
        <Stat label="Declined" figure={count(declined)} sub={<span>PD ≥ 20%</span>} />
        <Stat
          label="Critical band"
          figure={count(borrowers.filter((b) => b.risk_category === "Critical").length)}
          sub={<span style={{ color: RISK_HEX.Critical }}>needs review</span>}
        />
      </div>

      <div className="section">
        <div className="section__head">
          <div>
            <h2>Applications</h2>
            <p>Every row is clickable. {selected ? "" : "Click a borrower to open the risk profile."}</p>
          </div>
          <Seg options={BAND_FILTER} value={band} onChange={setBand} />
        </div>
        <DataTable
          rows={rows}
          columns={columns}
          searchKeys={(r) => `${r.borrower_id} ${r.loan_id} ${r.loan_purpose} ${r.decision} ${r.employment_type}`}
          onRowClick={(r) => setSp({ id: r.borrower_id })}
          initialSort={{ key: "pd", dir: "desc" }}
        />
        <p className="note">
          Risk score and probability come from a transparent scoring function over DTI, credit
          score, delinquencies, tenure, loan-to-income and utilisation — the same terms shown in
          each borrower's "why is this risky" panel. It is illustrative, not the pipeline's model.
        </p>
      </div>

      {selected && <BorrowerDrawer borrower={selected} onClose={() => setSp({})} />}
    </div>
  );
}

export type { RiskBand };
export { usd };
