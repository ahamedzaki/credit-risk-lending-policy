import type { RiskBand } from "../lib/risk";

/* ---------- REAL: pipeline aggregates ---------- */

export interface PortfolioTotals {
  loans: number;
  exposure: number;
  expected_loss: number;
  observed_default_rate: number;
  lgd: number;
}

export interface BandRow {
  band: RiskBand;
  range: string;
  loans: number;
  exposure: number;
  expected_loss: number;
  observed_default_rate: number;
}

export interface SegmentRow {
  key: string;
  label: string;
  loans: number;
  exposure: number;
  avg_pd: number;
  expected_loss: number;
  observed_default_rate: number;
}

export interface ModelSplit {
  auc: number;
  gini: number;
  ks: number;
  brier: number;
  n: number;
  base_rate: number;
}

export interface DecileRow {
  bucket: number;
  n: number;
  mean_pd: number;
  obs_rate: number;
  lift: number;
}

export interface BacktestPoint {
  pd_cut: number;
  approval_rate: number;
  pred_default_rate: number;
  actual_default_rate: number;
  model_expected_loss: number;
  realized_credit_loss: number;
  model_profit: number;
  realized_profit: number;
  n: number;
}

export interface FairnessGroup {
  group: string;
  approval_rate: number;
  approved_book_default_rate: number;
}

export interface FairnessDim {
  groups: FairnessGroup[];
  air: number;
  passes: boolean;
}

export interface FeatureImportanceRow {
  feature: string;
  label: string;
  importance: number; // mean AUC drop when the feature is shuffled (permutation importance)
}

export interface LgdByGradeRow {
  grade: string;
  lgd: number;
  n_charged_off_train: number;
}

/* ---------- DEMO: synthetic ---------- */

export type Decision = "Approved" | "Declined" | "Manual review";
export type LoanStatus = "Current" | "Delinquent" | "Default" | "Paid";
export type EmploymentType =
  | "Full-time"
  | "Part-time"
  | "Self-employed"
  | "Contract"
  | "Retired";

export interface RiskDriver {
  factor: string;
  contribution: number; // ± points
  note: string;
}

export interface Borrower {
  loan_id: string;
  borrower_id: string;
  application_date: string;
  loan_amount: number;
  annual_income: number;
  employment_length: number;
  employment_type: EmploymentType;
  credit_score: number;
  debt_to_income: number;
  credit_history_length: number;
  home_ownership: "Rent" | "Mortgage" | "Own";
  loan_purpose: string;
  interest_rate: number;
  term: 36 | 60;
  delinquencies_2y: number;
  risk_score: number; // 0–100
  probability_of_default: number;
  risk_category: RiskBand;
  expected_loss: number;
  decision: Decision;
  loan_status: LoanStatus;
  drivers: RiskDriver[];
}

export interface MonthPoint {
  month: string; // YYYY-MM
  default_rate: number;
  delinquency_rate: number;
  expected_loss: number;
  avg_credit_score: number;
  avg_dti: number;
  psi: number;
}

export type Severity = "High" | "Medium" | "Low";

export interface Alert {
  id: string;
  severity: Severity;
  title: string;
  detail: string;
  magnitude: string;
  window: string;
  segment: string;
  date: string;
}
