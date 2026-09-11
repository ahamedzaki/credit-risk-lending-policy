/**
 * DEMO data — synthetic, seeded, deterministic. Used only where the pipeline has no
 * row-level or longitudinal data: individual borrower applications, per-decision risk
 * attributions, and monthly time series. Every surface that renders this is badged
 * "DEMO DATA". Endpoints of the time series are anchored near the real headline values
 * so the picture stays coherent; nothing here is a measured result.
 */
import { bandFor } from "../lib/risk";
import type { Alert, Borrower, EmploymentType, MonthPoint, RiskDriver } from "./types";

/* ---- seeded RNG (mulberry32) ---- */
function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const pick = <T>(r: () => number, arr: T[]): T => arr[Math.floor(r() * arr.length)];
const between = (r: () => number, lo: number, hi: number) => lo + r() * (hi - lo);
const gauss = (r: () => number) => (r() + r() + r() + r() - 2) / 2; // ~N(0,1)-ish, bounded

const PURPOSES = [
  "Debt consolidation",
  "Credit card",
  "Home improvement",
  "Major purchase",
  "Medical",
  "Small business",
  "Car",
  "Moving",
];
const EMPLOYMENT: EmploymentType[] = [
  "Full-time",
  "Full-time",
  "Full-time",
  "Part-time",
  "Self-employed",
  "Contract",
  "Retired",
];
const TITLES = [
  "Operations Manager",
  "Registered Nurse",
  "Software Engineer",
  "Retail Supervisor",
  "Logistics Coordinator",
  "Account Executive",
  "Electrician",
  "Teacher",
  "Project Manager",
  "Customer Success Lead",
];

/**
 * Transparent scoring function. risk_score ≈ base 22 plus driver contributions;
 * PD is a squashed function of the score. Drivers are the SAME terms, so the
 * "why is this borrower risky" panel is internally consistent with the score.
 */
function score(b: {
  dti: number;
  credit_score: number;
  delinq: number;
  emp_years: number;
  loan_to_income: number;
  util: number;
}): { risk_score: number; pd: number; drivers: RiskDriver[] } {
  const dtiC = Math.round((b.dti - 0.18) * 62); // ±
  const scoreC = Math.round((690 - b.credit_score) * 0.13);
  const delinqC = Math.round(b.delinq * 7);
  const empC = Math.round((3 - b.emp_years) * 1.6);
  const ltiC = Math.round((b.loan_to_income - 0.28) * 28);
  const utilC = Math.round((b.util - 0.4) * 18);
  const drivers: RiskDriver[] = [
    { factor: "Debt-to-income", contribution: dtiC, note: `DTI of ${(b.dti * 100).toFixed(0)}% against a book median near 18%.` },
    { factor: "Credit score", contribution: scoreC, note: `FICO ${b.credit_score}; each 100 points below 690 adds ~13 points of risk.` },
    { factor: "Recent delinquencies", contribution: delinqC, note: `${b.delinq} delinquency event(s) reported in the last 24 months.` },
    { factor: "Employment tenure", contribution: empC, note: `${b.emp_years.toFixed(1)} years in current employment; short tenure raises risk.` },
    { factor: "Loan-to-income", contribution: ltiC, note: `Requested amount is ${(b.loan_to_income * 100).toFixed(0)}% of annual income.` },
    { factor: "Revolving utilisation", contribution: utilC, note: `${(b.util * 100).toFixed(0)}% of revolving credit in use.` },
  ].sort((a, z) => Math.abs(z.contribution) - Math.abs(a.contribution));
  const raw = 24 + dtiC + scoreC + delinqC + empC + ltiC + utilC;
  const risk_score = Math.max(2, Math.min(99, Math.round(raw)));
  const pd = Math.max(0.01, Math.min(0.92, 1 / (1 + Math.exp(-(risk_score - 50) / 11))));
  return { risk_score, pd: Math.round(pd * 1000) / 1000, drivers };
}

export function makeBorrowers(seed = 20240611, n = 220): Borrower[] {
  const r = rng(seed);
  const out: Borrower[] = [];
  for (let i = 0; i < n; i++) {
    const credit_score = Math.round(Math.max(620, Math.min(820, 690 + gauss(r) * 46)));
    const annual_income = Math.round(
      Math.max(24000, Math.min(260000, Math.exp(between(r, 10.7, 11.9)) * (0.8 + r() * 0.5))),
    );
    const loan_amount = Math.round(
      Math.max(2000, Math.min(40000, annual_income * between(r, 0.08, 0.55))) / 500,
    ) * 500;
    const debt_to_income = Math.round(Math.max(2, Math.min(46, 16 + gauss(r) * 12)) * 10) / 10 / 100;
    const emp_years = Math.round(Math.max(0, Math.min(12, 5 + gauss(r) * 4)) * 10) / 10;
    const employment_type = pick(r, EMPLOYMENT);
    const delinquencies_2y = r() < 0.7 ? 0 : r() < 0.85 ? 1 : r() < 0.95 ? 2 : 3;
    const util = Math.max(0.02, Math.min(1.1, 0.42 + gauss(r) * 0.24));
    const credit_history_length = Math.round(Math.max(2, Math.min(32, 12 + gauss(r) * 7)));
    const home_ownership = pick(r, ["Rent", "Rent", "Mortgage", "Mortgage", "Own"] as const);
    const loan_purpose = pick(r, PURPOSES);
    const term = (r() < 0.72 ? 36 : 60) as 36 | 60;
    const loan_to_income = loan_amount / annual_income;
    const { risk_score, pd, drivers } = score({
      dti: debt_to_income,
      credit_score,
      delinq: delinquencies_2y,
      emp_years,
      loan_to_income,
      util,
    });
    const interest_rate =
      Math.round((0.062 + pd * 0.42 + (term === 60 ? 0.012 : 0)) * 1000) / 1000;
    const risk_category = bandFor(pd);
    const expected_loss = Math.round(pd * loan_amount * 0.5);
    const decision =
      pd < 0.12 ? "Approved" : pd < 0.2 ? "Manual review" : ("Declined" as const);
    const month = 1 + Math.floor(r() * 12);
    const application_date = `2018-${String(month).padStart(2, "0")}-${String(
      1 + Math.floor(r() * 27),
    ).padStart(2, "0")}`;
    const loan_status =
      decision === "Declined"
        ? "Current"
        : r() < 0.06 + pd * 0.4
          ? r() < 0.5
            ? "Delinquent"
            : "Default"
          : r() < 0.25
            ? "Paid"
            : ("Current" as const);
    out.push({
      loan_id: `LN-${(4820100 + i * 37).toString(36).toUpperCase()}`,
      borrower_id: `BR-${(9310 + i).toString().padStart(5, "0")}`,
      application_date,
      loan_amount,
      annual_income,
      employment_length: emp_years,
      employment_type,
      credit_score,
      debt_to_income,
      credit_history_length,
      home_ownership,
      loan_purpose,
      interest_rate,
      term,
      delinquencies_2y,
      risk_score,
      probability_of_default: pd,
      risk_category,
      expected_loss,
      decision: decision as Borrower["decision"],
      loan_status: loan_status as Borrower["loan_status"],
      drivers,
    });
  }
  return out;
}

export const borrowers = makeBorrowers();

/* pull one memorable title per borrower for the detail header, deterministic */
export function titleFor(b: Borrower): string {
  const idx = parseInt(b.borrower_id.slice(3), 10) % TITLES.length;
  return `${TITLES[idx]} · ${b.employment_type}`;
}

/* ---- monthly time series (36 months to 2018-12), anchored near real headlines ---- */
export function makeMonthly(seed = 71119): MonthPoint[] {
  const r = rng(seed);
  const out: MonthPoint[] = [];
  let dr = 0.126;
  let delq = 0.03;
  let el = 452_000_000;
  let cs = 703;
  let dti = 0.169;
  let psi = 0.03;
  for (let i = 0; i < 36; i++) {
    const y = 2016 + Math.floor(i / 12);
    const m = (i % 12) + 1;
    // a mild, broad deterioration through the last ~9 months
    const late = i > 26 ? (i - 26) / 18 : 0;
    dr = Math.max(0.1, dr + 0.00018 + late * 0.0009 + gauss(r) * 0.0012);
    delq = Math.max(0.02, delq + 0.00012 + late * 0.0005 + gauss(r) * 0.0008);
    el = Math.max(3e8, el + 1.1e6 + late * 2.6e6 + gauss(r) * 3e6);
    cs = cs - 0.04 - late * 0.4 + gauss(r) * 0.55;
    dti = dti + 0.00028 + late * 0.001 + gauss(r) * 0.0007;
    psi = Math.max(0.01, psi + late * 0.011 + Math.abs(gauss(r)) * 0.0035);
    out.push({
      month: `${y}-${String(m).padStart(2, "0")}`,
      default_rate: Math.round(dr * 10000) / 10000,
      delinquency_rate: Math.round(delq * 10000) / 10000,
      expected_loss: Math.round(el),
      avg_credit_score: Math.round(cs * 10) / 10,
      avg_dti: Math.round(dti * 10000) / 10000,
      psi: Math.round(psi * 1000) / 1000,
    });
  }
  // ease the final three months onto the real book-level headline (14.1% / $524.2M)
  const targets = { dr: 0.1409, el: 524_155_631 };
  for (let k = 3; k >= 1; k--) {
    const row = out[out.length - k];
    const w = (4 - k) / 4;
    row.default_rate = Math.round((row.default_rate * (1 - w) + targets.dr * w) * 10000) / 10000;
    row.expected_loss = Math.round(row.expected_loss * (1 - w) + targets.el * w);
  }
  return out;
}

export const monthly = makeMonthly();

/* ---- alerts (DEMO) ---- */
export const alerts: Alert[] = [
  {
    id: "a1",
    severity: "High",
    title: "High-risk borrower share rising",
    detail:
      "Applications scoring PD ≥ 20% grew for a fourth straight month, concentrated in 36-month debt-consolidation loans.",
    magnitude: "+12.0% MoM",
    window: "last 30 days",
    segment: "Debt consolidation · 36-month",
    date: "2018-12-04",
  },
  {
    id: "a2",
    severity: "Medium",
    title: "Auto-loan defaults above range",
    detail:
      "Realised default rate on car loans ran 8.4% above its trailing-12-month average this month.",
    magnitude: "+8.4% vs T12M avg",
    window: "December",
    segment: "Car",
    date: "2018-12-02",
  },
  {
    id: "a3",
    severity: "Medium",
    title: "DTI drifting upward",
    detail:
      "Average applicant debt-to-income has climbed 1.6 points over the quarter; the shift is broad-based across income bands.",
    magnitude: "+1.6 pts QoQ",
    window: "Q4",
    segment: "All segments",
    date: "2018-11-27",
  },
  {
    id: "a4",
    severity: "Low",
    title: "Portfolio exposure within expected range",
    detail:
      "Total funded exposure held near plan; new originations are pacing to the quarterly target.",
    magnitude: "−0.3% vs plan",
    window: "December",
    segment: "Portfolio",
    date: "2018-12-05",
  },
];

/* emerging-risk timeline entries (DEMO) */
export const emerging: Alert[] = [
  {
    id: "e1",
    severity: "High",
    title: "Credit-score distribution deteriorating",
    detail: "The 25th percentile of applicant FICO fell 11 points since September; population stability index crossed 0.10.",
    magnitude: "PSI 0.12",
    window: "Sep – Dec",
    segment: "New applications",
    date: "2018-12-03",
  },
  {
    id: "e2",
    severity: "Medium",
    title: "Debt-to-income increasing",
    detail: "Mean DTI up 1.6 points over the quarter, steepening in November.",
    magnitude: "+1.6 pts",
    window: "Q4",
    segment: "All",
    date: "2018-11-20",
  },
  {
    id: "e3",
    severity: "Medium",
    title: "Predicted default probability rising",
    detail: "Mean model PD on the incoming pipeline moved from 13.1% to 14.4%.",
    magnitude: "+1.3 pts",
    window: "Oct – Dec",
    segment: "Pipeline",
    date: "2018-11-14",
  },
  {
    id: "e4",
    severity: "Low",
    title: "Small-business segment: abnormal delinquency",
    detail: "Early-stage delinquency on small-business loans ran two standard deviations above its mean for one month, then normalised.",
    magnitude: "2.1σ, one month",
    window: "October",
    segment: "Small business",
    date: "2018-10-30",
  },
];
