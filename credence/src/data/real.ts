/**
 * REAL data — transcribed verbatim from the credit-risk-lending-policy pipeline outputs
 * (../artifacts/metrics.json, lgd.json, backtest.json, fairness.json, monitor.json,
 *  insight.json, explanations.json, ../exports/portfolio_summary.parquet). Dataset:
 * Lending Club accepted loans (wordsforthewise mirror), 36-month term, issued
 * 2012-01…2016-02, 640,919 terminal loans, snapshot 2018Q4. Model: isotonic-calibrated
 * HistGradientBoosting on ~33 origination-time features. EAD = funded amount.
 *
 * Post-audit pass: LGD is now segmented by grade (Buhlmann credibility-weighted) and
 * feeds Expected Loss directly — not a single flat number. Fairness AIR carries a real
 * bootstrap confidence interval, Bonferroni-corrected for testing 3 dimensions. A real
 * (retrospective) population-stability check and a real per-loan attribution sample on
 * the actual trained model are new exports below (psi, explainedLoans).
 */
import type {
  BacktestPoint,
  BandRow,
  DecileRow,
  ExplainedLoan,
  FairnessDim,
  FeatureImportanceRow,
  LgdByGradeRow,
  ModelSplit,
  PortfolioTotals,
  PsiResult,
  SegmentRow,
} from "./types";

export const PROVENANCE = {
  dataset: "Lending Club accepted loans (wordsforthewise mirror)",
  window: "issued 2012-01 – 2016-02",
  loans: 640919,
  term: "36-month",
  snapshot: "2018Q4",
  model: "HistGradientBoosting, isotonic-calibrated, ~33 origination-time features",
  lgdBasis: "1 − (principal repaid + recoveries) / funded, over 40,596 charged-off training loans, segmented by grade (Buhlmann credibility)",
  eadBasis: "funded amount at origination",
  oot: "train issued < 2015-01 · test issued ≥ 2015-01",
} as const;

export const totals: PortfolioTotals = {
  loans: 640919,
  exposure: 8_157_055_225,
  expected_loss: 519_948_331,
  observed_default_rate: 0.141,
  lgd: 0.4995,
};

export const lgd = {
  value: 0.4995,
  recovery_rate: 0.5005,
  n_charged_off_train: 40596,
};

/* portfolio_summary.parquet — risk_band rows (EL now uses grade-segmented LGD) */
export const byRiskBand: BandRow[] = [
  { band: "Low", range: "PD < 5%", loans: 79767, exposure: 1_231_269_725, expected_loss: 19_982_689, observed_default_rate: 0.0304 },
  { band: "Medium", range: "5–10%", loans: 168911, exposure: 2_268_782_350, expected_loss: 81_881_569, observed_default_rate: 0.0741 },
  { band: "High", range: "10–20%", loans: 268899, exposure: 3_203_741_775, expected_loss: 226_915_840, observed_default_rate: 0.1493 },
  { band: "Critical", range: "20%+", loans: 123342, exposure: 1_453_261_375, expected_loss: 191_168_231, observed_default_rate: 0.286 },
];

/* portfolio_summary.parquet — grade rows (EL now uses grade-segmented LGD) */
export const byGrade: SegmentRow[] = [
  { key: "A", label: "Grade A", loans: 146017, exposure: 2_092_034_900, avg_pd: 0.0671, expected_loss: 62_327_743, observed_default_rate: 0.0545 },
  { key: "B", label: "Grade B", loans: 221171, exposure: 2_775_406_775, avg_pd: 0.1192, expected_loss: 152_389_162, observed_default_rate: 0.1127 },
  { key: "C", label: "Grade C", loans: 169208, exposure: 2_045_113_700, avg_pd: 0.1663, expected_loss: 166_660_574, observed_default_rate: 0.1818 },
  { key: "D", label: "Grade D", loans: 77437, exposure: 928_241_475, avg_pd: 0.1998, expected_loss: 98_111_177, observed_default_rate: 0.2388 },
  { key: "E", label: "Grade E", loans: 22124, exposure: 265_840_800, avg_pd: 0.2289, expected_loss: 33_431_105, observed_default_rate: 0.2942 },
  { key: "F", label: "Grade F", loans: 4416, exposure: 43_697_875, avg_pd: 0.2515, expected_loss: 6_053_023, observed_default_rate: 0.3384 },
  { key: "G", label: "Grade G", loans: 546, exposure: 6_719_700, avg_pd: 0.2902, expected_loss: 975_542, observed_default_rate: 0.4096 },
];

/* portfolio_summary.parquet — purpose rows (top by exposure; EL uses grade-segmented LGD) */
export const byPurpose: SegmentRow[] = [
  { key: "debt_consolidation", label: "Debt consolidation", loans: 365917, exposure: 4_850_745_075, avg_pd: 0.1412, expected_loss: 326_008_320, observed_default_rate: 0.1477 },
  { key: "credit_card", label: "Credit card", loans: 159351, exposure: 2_163_730_075, avg_pd: 0.1139, expected_loss: 113_892_500, observed_default_rate: 0.1179 },
  { key: "home_improvement", label: "Home improvement", loans: 36689, exposure: 440_613_200, avg_pd: 0.118, expected_loss: 25_406_572, observed_default_rate: 0.1255 },
  { key: "other", label: "Other", loans: 33517, exposure: 281_027_950, avg_pd: 0.1536, expected_loss: 21_752_441, observed_default_rate: 0.1645 },
  { key: "major_purchase", label: "Major purchase", loans: 12311, exposure: 121_259_825, avg_pd: 0.1258, expected_loss: 7_652_082, observed_default_rate: 0.1317 },
  { key: "small_business", label: "Small business", loans: 6714, exposure: 93_400_000, avg_pd: 0.2064, expected_loss: 9_976_482, observed_default_rate: 0.2242 },
  { key: "medical", label: "Medical", loans: 7036, exposure: 52_142_175, avg_pd: 0.1525, expected_loss: 3_987_156, observed_default_rate: 0.173 },
  { key: "car", label: "Car", loans: 6461, exposure: 51_794_650, avg_pd: 0.1247, expected_loss: 3_262_496, observed_default_rate: 0.1195 },
  { key: "house", label: "House", loans: 2516, exposure: 32_039_725, avg_pd: 0.1611, expected_loss: 2_515_064, observed_default_rate: 0.1883 },
  { key: "moving", label: "Moving", loans: 4491, exposure: 30_776_275, avg_pd: 0.1542, expected_loss: 2_439_518, observed_default_rate: 0.196 },
  { key: "vacation", label: "Vacation", loans: 4310, exposure: 24_306_175, avg_pd: 0.1473, expected_loss: 1_896_626, observed_default_rate: 0.1607 },
];

/* simulator_base FICO bands (min FICO in data ≈ 660) */
export const byFicoBand: SegmentRow[] = [
  { key: "660-690", label: "660–690", loans: 326260, exposure: 3_762_044_000, avg_pd: 0.1695, expected_loss: 0, observed_default_rate: 0.1766 },
  { key: "690-720", label: "690–720", loans: 196461, exposure: 2_678_040_000, avg_pd: 0.118, expected_loss: 0, observed_default_rate: 0.1236 },
  { key: "720-750", label: "720–750", loans: 74725, exposure: 1_098_972_000, avg_pd: 0.075, expected_loss: 0, observed_default_rate: 0.0823 },
  { key: "750+", label: "750+", loans: 43473, exposure: 617_999_000, avg_pd: 0.046, expected_loss: 0, observed_default_rate: 0.0532 },
];

/* fairness.json — income-band approval split (real) */
export const byIncomeBand: SegmentRow[] = [
  { key: "<40k", label: "Under $40k", loans: 69083, exposure: 0, avg_pd: 0, expected_loss: 0, observed_default_rate: 0.1105 },
  { key: "40-70k", label: "$40k–70k", loans: 128838, exposure: 0, avg_pd: 0, expected_loss: 0, observed_default_rate: 0.0974 },
  { key: "70-120k", label: "$70k–120k", loans: 99411, exposure: 0, avg_pd: 0, expected_loss: 0, observed_default_rate: 0.0911 },
  { key: "120k+", label: "$120k+", loans: 37125, exposure: 0, avg_pd: 0, expected_loss: 0, observed_default_rate: 0.0836 },
];

/* portfolio_summary.parquet — vintage rows (EL uses grade-segmented LGD) */
export const byVintage: SegmentRow[] = [
  { key: "2012", label: "2012", loans: 43470, exposure: 507_799_125, avg_pd: 0.1353, expected_loss: 33_272_352, observed_default_rate: 0.1358 },
  { key: "2013", label: "2013", loans: 100422, exposure: 1_272_091_475, avg_pd: 0.1265, expected_loss: 76_549_011, observed_default_rate: 0.1233 },
  { key: "2014", label: "2014", loans: 162570, exposure: 2_046_040_750, avg_pd: 0.1356, expected_loss: 131_149_184, observed_default_rate: 0.1373 },
  { key: "2015", label: "2015", loans: 283173, exposure: 3_626_461_100, avg_pd: 0.1366, expected_loss: 234_475_061, observed_default_rate: 0.1489 },
  { key: "2016", label: "2016", loans: 51284, exposure: 704_662_775, avg_pd: 0.1322, expected_loss: 44_502_719, observed_default_rate: 0.1484 },
];

/* metrics.json */
export const modelTest: ModelSplit = { auc: 0.6907, gini: 0.3813, ks: 0.2782, brier: 0.1194, n: 333721, base_rate: 0.1488 };
export const modelTrain: ModelSplit = { auc: 0.7137, gini: 0.4275, ks: 0.308, brier: 0.1068, n: 306462, base_rate: 0.1325 };

export const benchmark = { hgb: 0.6907, logistic: 0.6804, grade_only: 0.6688 };
export const delong = { z: 19.57, p_value: 3.02e-85, auc_diff: 0.0218 };

export const decile: DecileRow[] = [
  { bucket: 0, n: 33373, mean_pd: 0.031, obs_rate: 0.0319, lift: 0.214 },
  { bucket: 1, n: 33372, mean_pd: 0.0557, obs_rate: 0.0582, lift: 0.391 },
  { bucket: 2, n: 33372, mean_pd: 0.0742, obs_rate: 0.0822, lift: 0.552 },
  { bucket: 3, n: 33372, mean_pd: 0.0924, obs_rate: 0.103, lift: 0.692 },
  { bucket: 4, n: 33372, mean_pd: 0.1119, obs_rate: 0.1232, lift: 0.828 },
  { bucket: 5, n: 33372, mean_pd: 0.1329, obs_rate: 0.1468, lift: 0.986 },
  { bucket: 6, n: 33371, mean_pd: 0.1533, obs_rate: 0.1728, lift: 1.161 },
  { bucket: 7, n: 33373, mean_pd: 0.1833, obs_rate: 0.2025, lift: 1.361 },
  { bucket: 8, n: 33372, mean_pd: 0.2227, obs_rate: 0.2432, lift: 1.635 },
  { bucket: 9, n: 33372, mean_pd: 0.2977, obs_rate: 0.3242, lift: 2.179 },
];

/**
 * Precision / Recall / F1 are NOT in the pipeline artifacts (they need a decision
 * threshold). These are computed at the PD ≥ 0.20 "high-risk flag" from the real
 * risk-band aggregates: positives = Σ loans·observed_default_rate.
 */
const _pos = byRiskBand.reduce((s, b) => s + b.loans * b.observed_default_rate, 0);
const _crit = byRiskBand[3];
const _tp = _crit.loans * _crit.observed_default_rate;
const _fp = _crit.loans - _tp;
const _fn = _pos - _tp;
export const operatingPoint = {
  cutoff: 0.2,
  precision: _tp / _crit.loans,
  recall: _tp / _pos,
  f1: (2 * _tp) / (2 * _tp + _fp + _fn),
  tp: Math.round(_tp),
  fp: Math.round(_fp),
  fn: Math.round(_fn),
};

/* backtest.json — 60-row realized-outcome curve (model_expected_loss / model_profit now
   reflect grade-segmented LGD, not a flat multiplier)
   tuple: [pd_cut, approval_rate, pred_dr, actual_dr, model_EL, realized_loss, model_profit, realized_profit, n] */
const _curve: number[][] = [
  [0.03,0.043,0.021,0.0205,2268500,2140956,3525600,170809,14338],[0.0363,0.0643,0.0251,0.025,4075700,3991518,5685946,865750,21448],[0.0425,0.0879,0.029,0.03,6362680,6579019,8310880,1555362,29340],[0.0488,0.122,0.0336,0.0347,10038485,10348727,12190368,3299673,40725],[0.0551,0.1505,0.0371,0.0387,13532641,14217627,15605806,4714453,50212],[0.0614,0.1778,0.0403,0.0426,17197981,18329069,18823219,6018301,59348],[0.0676,0.2101,0.044,0.0468,21832825,23604986,22528433,7547695,70122],[0.0739,0.2451,0.0478,0.051,27243294,29480676,26525951,9603010,81780],[0.0802,0.2811,0.0516,0.0549,33247697,36040683,30789724,11904117,93813],[0.0864,0.3138,0.0549,0.059,39034435,42985636,34491781,13384336,104735],[0.0927,0.3468,0.0582,0.063,45202220,50279950,38030033,15011317,115743],[0.099,0.3802,0.0615,0.0669,51780563,57829916,41639867,16743820,126877],[0.1053,0.4138,0.0648,0.0706,58823639,65698960,45051800,18697571,138110],[0.1115,0.4408,0.0675,0.0736,64733878,72411774,47614636,20030766,147112],[0.1178,0.4729,0.0706,0.0767,72139994,80453836,50572283,22013824,157828],[0.1241,0.5044,0.0738,0.0803,79787534,89405392,53258881,23208563,168336],[0.1303,0.5342,0.0768,0.0837,87357600,98132973,55563515,24252785,178277],[0.1366,0.5638,0.0798,0.0869,95117175,106760814,57641730,25576286,188139],[0.1429,0.5953,0.0829,0.0908,103899256,117186651,59625590,26172155,198670],[0.1492,0.6248,0.0859,0.0941,112272133,127175430,61335322,26695699,208494],[0.1554,0.6536,0.0889,0.0972,120922350,136927470,62764363,27404493,218116],[0.1617,0.6783,0.0914,0.0999,128525100,145714095,63892927,28045368,226351],[0.168,0.7015,0.0938,0.1027,135987059,154160814,64689040,28435228,234118],[0.1742,0.7234,0.0962,0.1052,143268129,162658950,65162530,28521794,241425],[0.1805,0.7433,0.0983,0.1077,150229111,170687866,65448059,28451459,248068],[0.1868,0.7615,0.1004,0.1097,156753272,178155776,65514500,28341273,254144],[0.1931,0.7818,0.1027,0.1125,164420163,187467655,65417846,27675637,260918],[0.1993,0.7949,0.1042,0.1143,169545990,193585688,65242884,27293097,265289],[0.2056,0.8126,0.1064,0.1166,176568552,201825097,64767908,26882651,271182],[0.2119,0.8248,0.1079,0.1184,181591477,208121514,64314934,26034680,275241],[0.2181,0.8393,0.1097,0.1204,187717949,214997754,63580208,25640935,280097],[0.2244,0.8511,0.1113,0.1221,192848725,221057582,62843062,24966549,284024],[0.2307,0.8657,0.1132,0.1241,199412211,228813992,61670325,23797535,288888],[0.2369,0.8786,0.115,0.1261,205388230,235686254,60504885,22906042,293213],[0.2432,0.8928,0.117,0.1281,212026908,243152329,59096668,22167637,297951],[0.2495,0.9047,0.1187,0.1299,217826866,249993168,57808303,21148882,301912],[0.2558,0.9155,0.1203,0.1318,223322778,256762938,56469991,19674065,305525],[0.262,0.9249,0.1217,0.1333,228154550,262369548,55200181,18774105,308642],[0.2683,0.9336,0.123,0.1348,232909668,268134174,53863606,17620042,311553],[0.2746,0.9424,0.1244,0.1363,237794279,274079902,52417502,16301386,314513],[0.2808,0.9493,0.1255,0.1376,241758149,279186143,51176794,14921840,316792],[0.2871,0.9546,0.1264,0.1385,244838525,282762389,50182962,14272428,318562],[0.2934,0.9593,0.1272,0.1393,247782517,286008490,49219249,13905520,320142],[0.2997,0.9642,0.1281,0.1403,250775890,289539326,48156131,13113756,321776],[0.3059,0.9686,0.1289,0.1412,253538899,292915993,47128539,12280121,323227],[0.3122,0.9713,0.1294,0.1418,255421825,295090717,46410232,11783080,324154],[0.3185,0.9745,0.13,0.1424,257578849,297559515,45558815,11263630,325209],[0.3247,0.9786,0.1308,0.1433,260428529,300856562,44394908,10519133,326574],[0.331,0.9804,0.1311,0.1437,261798333,302601027,43813850,9929043,327184],[0.3373,0.9827,0.1316,0.1443,263571835,304691624,43034019,9339558,327961],[0.3436,0.9861,0.1323,0.145,265921866,307405534,41935117,8646030,329070],[0.3498,0.9878,0.1327,0.1454,267259036,308958587,41288709,8257840,329640],[0.3561,0.9895,0.1331,0.1458,268542966,310511644,40667245,7764383,330217],[0.3624,0.9907,0.1334,0.1462,269507955,311706104,40190775,7349883,330624],[0.3686,0.9919,0.1336,0.1465,270437169,312874407,39715684,6985474,331010],[0.3749,0.993,0.1339,0.1467,271348348,313825714,39222557,6721101,331391],[0.3812,0.9936,0.1341,0.1468,271888874,314451287,38940705,6528929,331599],[0.3875,0.9945,0.1343,0.1471,272655751,315491418,38526282,6136823,331893],[0.3937,0.9949,0.1344,0.1472,272977871,315813476,38344381,6045411,332025],[0.4,0.9956,0.1346,0.1474,273612203,316647463,37983670,5686030,332262],
];

export const backtest: BacktestPoint[] = _curve.map((r) => ({
  pd_cut: r[0],
  approval_rate: r[1],
  pred_default_rate: r[2],
  actual_default_rate: r[3],
  model_expected_loss: r[4],
  realized_credit_loss: r[5],
  model_profit: r[6],
  realized_profit: r[7],
  n: r[8],
}));

export const backtestMeta = {
  model_optimum: { pd_cut: 0.187, approval_rate: 0.762, actual_default_rate: 0.11, model_profit: 65_514_500, realized_profit: 28_341_273 },
  realized_optimum: { pd_cut: 0.174, approval_rate: 0.723, actual_default_rate: 0.105, realized_credit_loss: 162_658_950, realized_profit: 28_521_794 },
  regret: 180_521,
  loss_ratio: 1.148,
  min_cut: 0.03,
  max_cut: 0.4,
};

/* fairness.json — AIR now carries a real bootstrap 98.33% CI (Bonferroni-corrected for
   testing 3 dimensions at once, n=2000 resamples) instead of a bare point estimate. */
export const fairness: Record<"income" | "region" | "homeOwnership", FairnessDim> = {
  income: {
    groups: [
      { group: "Under $40k", approval_rate: 0.3581, approved_book_default_rate: 0.1105 },
      { group: "$40k–70k", approval_rate: 0.595, approved_book_default_rate: 0.0974 },
      { group: "$70k–120k", approval_rate: 0.7719, approved_book_default_rate: 0.0911 },
      { group: "$120k+", approval_rate: 0.8612, approved_book_default_rate: 0.0836 },
    ],
    air: 0.416,
    airCi: [0.41, 0.421],
    passes: false,
    flaggedSignificant: true,
  },
  region: {
    groups: [
      { group: "Northeast", approval_rate: 0.647, approved_book_default_rate: 0.0994 },
      { group: "Midwest", approval_rate: 0.6195, approved_book_default_rate: 0.0886 },
      { group: "South", approval_rate: 0.6196, approved_book_default_rate: 0.0976 },
      { group: "West", approval_rate: 0.6313, approved_book_default_rate: 0.0906 },
    ],
    air: 0.958,
    airCi: [0.947, 0.964],
    passes: true,
    flaggedSignificant: false,
  },
  homeOwnership: {
    groups: [
      { group: "Rent", approval_rate: 0.5107, approved_book_default_rate: 0.1056 },
      { group: "Own", approval_rate: 0.5955, approved_book_default_rate: 0.1033 },
      { group: "Mortgage", approval_rate: 0.7464, approved_book_default_rate: 0.0855 },
    ],
    air: 0.684,
    airCi: [0.679, 0.689],
    passes: false,
    flaggedSignificant: true,
  },
};

export const overallApprovalRate = 0.6282;

/**
 * insight.json — global permutation feature importance of the shipped model
 * (model.joblib), computed on a 15,000-loan sample of the out-of-time test split,
 * 5 repeats, scoring = ROC AUC. This is a real attribution of the actual estimator —
 * not per-loan SHAP. See `explainedLoans` below for a real per-loan sample instead.
 */
export const featureImportance: FeatureImportanceRow[] = [
  { feature: "fico_mid", label: "Credit score (FICO)", importance: 0.02 },
  { feature: "acc_open_past_24mths", label: "Accounts opened, last 24m", importance: 0.0181 },
  { feature: "dti", label: "Debt-to-income ratio", importance: 0.0119 },
  { feature: "loan_to_income", label: "Loan-to-income ratio", importance: 0.0119 },
  { feature: "mths_since_recent_bc", label: "Months since newest bankcard", importance: 0.0052 },
  { feature: "purpose", label: "Loan purpose", importance: 0.0052 },
  { feature: "num_tl_op_past_12m", label: "Trades opened, last 12m", importance: 0.0051 },
  { feature: "inq_last_6mths", label: "Inquiries, last 6 months", importance: 0.0039 },
  { feature: "percent_bc_gt_75", label: "% bankcards over 75% utilised", importance: 0.0033 },
  { feature: "loan_amnt", label: "Loan amount", importance: 0.0033 },
  { feature: "num_actv_bc_tl", label: "Active bankcard trades", importance: 0.0031 },
  { feature: "mths_since_recent_inq", label: "Months since last inquiry", importance: 0.0028 },
];
export const featureImportanceMeta = {
  method: "sklearn permutation_importance, scoring=roc_auc",
  sampleN: 15000,
  nRepeats: 5,
  baselineAuc: 0.68,
};

/**
 * insight.json / lgd.json — LGD segmented by grade, Buhlmann credibility-weighted toward
 * the flat portfolio LGD for thin grades. This is what actually feeds Expected Loss
 * (sql/06_marts.sql joins this by grade) — not a supplementary side stat.
 */
export const lgdByGrade: LgdByGradeRow[] = [
  { grade: "A", lgd: 0.4641, n_charged_off_train: 3458 },
  { grade: "B", lgd: 0.4766, n_charged_off_train: 11790 },
  { grade: "C", lgd: 0.5002, n_charged_off_train: 13090 },
  { grade: "D", lgd: 0.5269, n_charged_off_train: 8582 },
  { grade: "E", lgd: 0.5351, n_charged_off_train: 2837 },
  { grade: "F", lgd: 0.5258, n_charged_off_train: 766 },
  { grade: "G", lgd: 0.503, n_charged_off_train: 73 },
];

/**
 * monitor.json — a REAL, but retrospective, population-stability check between the
 * pipeline's own train (2012-14) and test (2015-16) cohorts. Not a live production
 * drift monitor — there is no scored production traffic in this project — and it says
 * so plainly wherever it's shown.
 */
export const psi: PsiResult = {
  scope: "Retrospective train-vs-test population stability, not a live production monitor.",
  scorePsi: 0.0065,
  scoreBand: "stable",
  features: [
    { feature: "fico_mid", label: "Credit score (FICO)", psi: 0.0038, band: "stable" },
    { feature: "acc_open_past_24mths", label: "Accounts opened, last 24m", psi: 0.0247, band: "stable" },
    { feature: "loan_to_income", label: "Loan-to-income ratio", psi: 0.004, band: "stable" },
    { feature: "dti", label: "Debt-to-income ratio", psi: 0.03, band: "stable" },
    { feature: "annual_inc", label: "Annual income", psi: 0.0067, band: "stable" },
  ],
  nTrain: 306462,
  nTest: 333721,
};

/**
 * explanations.json — a real per-loan attribution sample: 12 real out-of-time test loans
 * (3 per risk band), each explained by resetting one feature at a time to the test
 * population's typical value and measuring the real PD change on the actual trained
 * model. This is a genuine sensitivity on the real model — explicitly NOT an exact
 * Shapley-value (SHAP) decomposition, and not a full per-decision reason-code system.
 */
export const explainedLoans: ExplainedLoan[] = [
  { loanId: 65097230, riskBand: "Low", pd: 0.0328, ficoMid: 737, dti: 30.22, purpose: "credit_card", loanAmnt: 8000, topDrivers: [{ feature: "fico_mid", label: "Credit score (FICO)", contributionPp: -3.33 }, { feature: "bc_open_to_buy", label: "Unused bankcard credit", contributionPp: -0.77 }, { feature: "purpose", label: "Loan purpose", contributionPp: -0.5 }] },
  { loanId: 59120926, riskBand: "Low", pd: 0.014, ficoMid: 797, dti: 14.19, purpose: "debt_consolidation", loanAmnt: 12000, topDrivers: [{ feature: "fico_mid", label: "Credit score (FICO)", contributionPp: -2.46 }, { feature: "revol_bal", label: "Revolving balance", contributionPp: 0.19 }, { feature: "mths_since_recent_inq", label: "Months since last inquiry", contributionPp: 0.19 }] },
  { loanId: 60800555, riskBand: "Low", pd: 0.0342, ficoMid: 697, dti: 10.79, purpose: "debt_consolidation", loanAmnt: 12000, topDrivers: [{ feature: "bc_open_to_buy", label: "Unused bankcard credit", contributionPp: 0.42 }, { feature: "total_bal_ex_mort", label: "Total balance ex-mortgage", contributionPp: 0.42 }, { feature: "avg_cur_bal", label: "Average balance per account", contributionPp: -0.41 }] },
  { loanId: 59312381, riskBand: "Medium", pd: 0.0769, ficoMid: 662, dti: 5.0, purpose: "home_improvement", loanAmnt: 5000, topDrivers: [{ feature: "loan_to_income", label: "Loan-to-income ratio", contributionPp: -2.6 }, { feature: "dti", label: "Debt-to-income ratio", contributionPp: -2.44 }, { feature: "percent_bc_gt_75", label: "% bankcards over 75% utilised", contributionPp: -1.46 }] },
  { loanId: 60415164, riskBand: "Medium", pd: 0.0834, ficoMid: 697, dti: 22.25, purpose: "debt_consolidation", loanAmnt: 14500, topDrivers: [{ feature: "num_tl_op_past_12m", label: "Trades opened, last 12m", contributionPp: -1.94 }, { feature: "fico_mid", label: "Credit score (FICO)", contributionPp: -1.5 }, { feature: "delinq_2yrs", label: "Delinquencies, last 2 years", contributionPp: 1.38 }] },
  { loanId: 50353355, riskBand: "Medium", pd: 0.0807, ficoMid: 697, dti: 29.92, purpose: "credit_card", loanAmnt: 6000, topDrivers: [{ feature: "dti", label: "Debt-to-income ratio", contributionPp: 1.7 }, { feature: "purpose", label: "Loan purpose", contributionPp: -1.53 }, { feature: "loan_to_income", label: "Loan-to-income ratio", contributionPp: -1.02 }] },
  { loanId: 41183324, riskBand: "High", pd: 0.1368, ficoMid: 677, dti: 32.12, purpose: "credit_card", loanAmnt: 10000, topDrivers: [{ feature: "dti", label: "Debt-to-income ratio", contributionPp: 3.98 }, { feature: "acc_open_past_24mths", label: "Accounts opened, last 24m", contributionPp: 3.33 }, { feature: "purpose", label: "Loan purpose", contributionPp: -2.84 }] },
  { loanId: 58632562, riskBand: "High", pd: 0.1118, ficoMid: 662, dti: 12.92, purpose: "credit_card", loanAmnt: 7700, topDrivers: [{ feature: "purpose", label: "Loan purpose", contributionPp: -2.59 }, { feature: "total_acc", label: "Total accounts", contributionPp: 1.75 }, { feature: "num_tl_op_past_12m", label: "Trades opened, last 12m", contributionPp: 1.54 }] },
  { loanId: 52446317, riskBand: "High", pd: 0.1039, ficoMid: 667, dti: 8.34, purpose: "credit_card", loanAmnt: 18000, topDrivers: [{ feature: "fico_mid", label: "Credit score (FICO)", contributionPp: 2.18 }, { feature: "acc_open_past_24mths", label: "Accounts opened, last 24m", contributionPp: 1.86 }, { feature: "mort_acc", label: "Mortgage accounts", contributionPp: 1.54 }] },
  { loanId: 45414433, riskBand: "Critical", pd: 0.2019, ficoMid: 677, dti: 18.24, purpose: "debt_consolidation", loanAmnt: 6000, topDrivers: [{ feature: "acc_open_past_24mths", label: "Accounts opened, last 24m", contributionPp: 3.91 }, { feature: "num_tl_op_past_12m", label: "Trades opened, last 12m", contributionPp: 2.25 }, { feature: "mths_since_recent_inq", label: "Months since last inquiry", contributionPp: 1.58 }] },
  { loanId: 61349835, riskBand: "Critical", pd: 0.2151, ficoMid: 697, dti: 27.14, purpose: "debt_consolidation", loanAmnt: 10975, topDrivers: [{ feature: "loan_to_income", label: "Loan-to-income ratio", contributionPp: 4.43 }, { feature: "emp_length_years", label: "Employment length", contributionPp: -4.12 }, { feature: "mths_since_recent_inq", label: "Months since last inquiry", contributionPp: 4.11 }] },
  { loanId: 55960490, riskBand: "Critical", pd: 0.2039, ficoMid: 682, dti: 19.98, purpose: "credit_card", loanAmnt: 7200, topDrivers: [{ feature: "tot_hi_cred_lim", label: "Total high credit limit", contributionPp: 3.25 }, { feature: "purpose", label: "Loan purpose", contributionPp: -3.04 }, { feature: "dti", label: "Debt-to-income ratio", contributionPp: 2.98 }] },
];

/* averages derived from the real book for the Portfolio metric line */
export const derived = {
  avgLoanSize: totals.exposure / totals.loans,
  avgInterestRate: 0.1199, // simulator_base mean int_rate
  totalLoans: totals.loans,
};
