/**
 * REAL data — transcribed verbatim from the credit-risk-lending-policy pipeline outputs
 * (../artifacts/metrics.json, lgd.json, backtest.json, fairness.json,
 *  ../exports/portfolio_summary.parquet). Dataset: Lending Club accepted loans
 * (wordsforthewise mirror), 36-month term, issued 2012-01…2016-02, 640,919 terminal
 * loans, snapshot 2018Q4. Model: isotonic-calibrated HistGradientBoosting on ~33
 * origination-time features. EAD = funded amount. LGD = 0.50 (data-estimated).
 */
import type {
  BacktestPoint,
  BandRow,
  DecileRow,
  FairnessDim,
  FeatureImportanceRow,
  LgdByGradeRow,
  ModelSplit,
  PortfolioTotals,
  SegmentRow,
} from "./types";

export const PROVENANCE = {
  dataset: "Lending Club accepted loans (wordsforthewise mirror)",
  window: "issued 2012-01 – 2016-02",
  loans: 640919,
  term: "36-month",
  snapshot: "2018Q4",
  model: "HistGradientBoosting, isotonic-calibrated, ~33 origination-time features",
  lgdBasis: "1 − (principal repaid + recoveries) / funded, over 40,596 charged-off training loans",
  eadBasis: "funded amount at origination",
  oot: "train issued < 2015-01 · test issued ≥ 2015-01",
} as const;

export const totals: PortfolioTotals = {
  loans: 640919,
  exposure: 8_157_055_225,
  expected_loss: 524_155_631,
  observed_default_rate: 0.1409,
  lgd: 0.4995,
};

export const lgd = {
  value: 0.4995,
  recovery_rate: 0.5005,
  n_charged_off_train: 40596,
};

/* portfolio_summary.parquet — risk_band rows */
export const byRiskBand: BandRow[] = [
  { band: "Low", range: "PD < 5%", loans: 79767, exposure: 1_231_270_000, expected_loss: 21_305_600, observed_default_rate: 0.0304 },
  { band: "Medium", range: "5–10%", loans: 168911, exposure: 2_268_782_000, expected_loss: 85_544_140, observed_default_rate: 0.0741 },
  { band: "High", range: "10–20%", loans: 268899, exposure: 3_203_742_000, expected_loss: 230_014_400, observed_default_rate: 0.1493 },
  { band: "Critical", range: "20%+", loans: 123342, exposure: 1_453_261_000, expected_loss: 187_291_500, observed_default_rate: 0.286 },
];

/* portfolio_summary.parquet — grade rows */
export const byGrade: SegmentRow[] = [
  { key: "A", label: "Grade A", loans: 146017, exposure: 2_092_035_000, avg_pd: 0.0671, expected_loss: 67_081_900, observed_default_rate: 0.0545 },
  { key: "B", label: "Grade B", loans: 221171, exposure: 2_775_407_000, avg_pd: 0.1192, expected_loss: 159_711_300, observed_default_rate: 0.1127 },
  { key: "C", label: "Grade C", loans: 169208, exposure: 2_045_114_000, avg_pd: 0.1663, expected_loss: 166_427_300, observed_default_rate: 0.1818 },
  { key: "D", label: "Grade D", loans: 77437, exposure: 928_241_500, avg_pd: 0.1998, expected_loss: 93_009_170, observed_default_rate: 0.2388 },
  { key: "E", label: "Grade E", loans: 22124, exposure: 265_840_800, avg_pd: 0.2289, expected_loss: 31_206_950, observed_default_rate: 0.2942 },
  { key: "F", label: "Grade F", loans: 4416, exposure: 43_697_880, avg_pd: 0.2515, expected_loss: 5_750_258, observed_default_rate: 0.3384 },
  { key: "G", label: "Grade G", loans: 546, exposure: 6_719_700, avg_pd: 0.2902, expected_loss: 968_755, observed_default_rate: 0.4096 },
];

/* portfolio_summary.parquet — purpose rows (top by exposure) */
export const byPurpose: SegmentRow[] = [
  { key: "debt_consolidation", label: "Debt consolidation", loans: 365917, exposure: 4_850_745_000, avg_pd: 0.1412, expected_loss: 327_855_100, observed_default_rate: 0.1477 },
  { key: "credit_card", label: "Credit card", loans: 159351, exposure: 2_163_730_000, avg_pd: 0.1139, expected_loss: 116_848_800, observed_default_rate: 0.1179 },
  { key: "home_improvement", label: "Home improvement", loans: 36689, exposure: 440_613_200, avg_pd: 0.118, expected_loss: 25_536_860, observed_default_rate: 0.1255 },
  { key: "other", label: "Other", loans: 33517, exposure: 281_028_000, avg_pd: 0.1536, expected_loss: 21_405_280, observed_default_rate: 0.1645 },
  { key: "major_purchase", label: "Major purchase", loans: 12311, exposure: 121_259_800, avg_pd: 0.1258, expected_loss: 7_682_926, observed_default_rate: 0.1317 },
  { key: "small_business", label: "Small business", loans: 6714, exposure: 93_400_000, avg_pd: 0.2064, expected_loss: 9_736_143, observed_default_rate: 0.2242 },
  { key: "medical", label: "Medical", loans: 7036, exposure: 52_142_180, avg_pd: 0.1525, expected_loss: 3_942_889, observed_default_rate: 0.173 },
  { key: "car", label: "Car", loans: 6461, exposure: 51_794_650, avg_pd: 0.1247, expected_loss: 3_291_156, observed_default_rate: 0.1195 },
  { key: "house", label: "House", loans: 2516, exposure: 32_039_720, avg_pd: 0.1611, expected_loss: 2_463_194, observed_default_rate: 0.1883 },
  { key: "moving", label: "Moving", loans: 4491, exposure: 30_776_280, avg_pd: 0.1542, expected_loss: 2_381_193, observed_default_rate: 0.196 },
  { key: "vacation", label: "Vacation", loans: 4310, exposure: 24_306_180, avg_pd: 0.1473, expected_loss: 1_870_528, observed_default_rate: 0.1607 },
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

/* portfolio_summary.parquet — vintage rows */
export const byVintage: SegmentRow[] = [
  { key: "2012", label: "2012", loans: 43470, exposure: 507_799_100, avg_pd: 0.1353, expected_loss: 33_597_730, observed_default_rate: 0.1358 },
  { key: "2013", label: "2013", loans: 100422, exposure: 1_272_091_000, avg_pd: 0.1265, expected_loss: 77_156_230, observed_default_rate: 0.1233 },
  { key: "2014", label: "2014", loans: 162570, exposure: 2_046_041_000, avg_pd: 0.1356, expected_loss: 132_038_800, observed_default_rate: 0.1373 },
  { key: "2015", label: "2015", loans: 283173, exposure: 3_626_461_000, avg_pd: 0.1366, expected_loss: 236_390_200, observed_default_rate: 0.1489 },
  { key: "2016", label: "2016", loans: 51284, exposure: 704_662_800, avg_pd: 0.1322, expected_loss: 44_972_640, observed_default_rate: 0.1484 },
];

/* metrics.json */
export const modelTest: ModelSplit = { auc: 0.6904, gini: 0.3808, ks: 0.2751, brier: 0.1194, n: 333721, base_rate: 0.1488 };
export const modelTrain: ModelSplit = { auc: 0.7152, gini: 0.4304, ks: 0.3114, brier: 0.1067, n: 306462, base_rate: 0.1325 };

export const benchmark = { hgb: 0.6904, logistic: 0.6803, grade_only: 0.6688 };
export const delong = { z: 19.35, p_value: 1.95e-83, auc_diff: 0.0216 };

export const decile: DecileRow[] = [
  { bucket: 0, n: 33373, mean_pd: 0.0312, obs_rate: 0.0314, lift: 0.211 },
  { bucket: 1, n: 33372, mean_pd: 0.056, obs_rate: 0.0596, lift: 0.4 },
  { bucket: 2, n: 33372, mean_pd: 0.0751, obs_rate: 0.0818, lift: 0.55 },
  { bucket: 3, n: 33372, mean_pd: 0.093, obs_rate: 0.1026, lift: 0.69 },
  { bucket: 4, n: 33372, mean_pd: 0.1127, obs_rate: 0.1242, lift: 0.835 },
  { bucket: 5, n: 33372, mean_pd: 0.1326, obs_rate: 0.1462, lift: 0.983 },
  { bucket: 6, n: 33371, mean_pd: 0.1548, obs_rate: 0.1711, lift: 1.15 },
  { bucket: 7, n: 33373, mean_pd: 0.1833, obs_rate: 0.2027, lift: 1.362 },
  { bucket: 8, n: 33372, mean_pd: 0.2237, obs_rate: 0.2453, lift: 1.649 },
  { bucket: 9, n: 33372, mean_pd: 0.2985, obs_rate: 0.323, lift: 2.171 },
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

/* backtest.json — 60-row realized-outcome curve
   tuple: [pd_cut, approval_rate, pred_dr, actual_dr, model_EL, realized_loss, model_profit, realized_profit, n] */
const _curve: number[][] = [
  [0.03,0.043,0.021,0.0205,2433547,2140956,3360553,170809,14338],[0.0363,0.0643,0.0251,0.025,4366113,3991518,5395533,865750,21448],[0.0425,0.0879,0.029,0.03,6803941,6579019,7869620,1555362,29340],[0.0488,0.122,0.0336,0.0347,10711108,10348727,11517745,3299673,40725],[0.0551,0.1505,0.0371,0.0387,14413073,14217627,14725374,4714453,50212],[0.0614,0.1778,0.0403,0.0426,18287823,18329069,17733377,6018301,59348],[0.0676,0.2101,0.044,0.0468,23172120,23604986,21189138,7547695,70122],[0.0739,0.2451,0.0478,0.051,28856958,29480676,24912287,9603010,81780],[0.0802,0.2811,0.0516,0.0549,35140615,36040683,28896806,11904117,93813],[0.0864,0.3138,0.0549,0.059,41180212,42985636,32346004,13384336,104735],[0.0927,0.3468,0.0582,0.063,47602493,50279950,35629760,15011317,115743],[0.099,0.3802,0.0615,0.0669,54423084,57829916,38997347,16743820,126877],[0.1053,0.4138,0.0648,0.0706,61711195,65698960,42164244,18697571,138110],[0.1115,0.4408,0.0675,0.0736,67811807,72411774,44536707,20030766,147112],[0.1178,0.4729,0.0706,0.0767,75432541,80453836,47279736,22013824,157828],[0.1241,0.5044,0.0738,0.0803,83281466,89405392,49764949,23208563,168336],[0.1303,0.5342,0.0768,0.0837,91034200,98132973,51886915,24252785,178277],[0.1366,0.5638,0.0798,0.0869,98959608,106760814,53799296,25576286,188139],[0.1429,0.5953,0.0829,0.0908,107911764,117186651,55613083,26172155,198670],[0.1492,0.6248,0.0859,0.0941,116415987,127175430,57191468,26695699,208494],[0.1554,0.6536,0.0889,0.0972,125184853,136927470,58501859,27404493,218116],[0.1617,0.6783,0.0914,0.0999,132867853,145714095,59550174,28045368,226351],[0.168,0.7015,0.0938,0.1027,140392424,154160814,60283675,28435228,234118],[0.1742,0.7234,0.0962,0.1052,147725972,162658950,60704687,28521794,241425],[0.1805,0.7433,0.0983,0.1077,154720552,170687866,60956617,28451459,248068],[0.1868,0.7615,0.1004,0.1097,161264781,178155776,61002991,28341273,254144],[0.1931,0.7818,0.1027,0.1125,168932419,187467655,60905590,27675637,260918],[0.1993,0.7949,0.1042,0.1143,174046994,193585688,60741880,27293097,265289],[0.2056,0.8126,0.1064,0.1166,181049262,201825097,60287198,26882651,271182],[0.2119,0.8248,0.1079,0.1184,186045111,208121514,59861299,26034680,275241],[0.2181,0.8393,0.1097,0.1204,192129628,214997754,59168529,25640935,280097],[0.2244,0.8511,0.1113,0.1221,197220197,221057582,58471589,24966549,284024],[0.2307,0.8657,0.1132,0.1241,203732736,228813992,57349799,23797535,288888],[0.2369,0.8786,0.115,0.1261,209648906,235686254,56244210,22906042,293213],[0.2432,0.8928,0.117,0.1281,216202756,243152329,54920820,22167637,297951],[0.2495,0.9047,0.1187,0.1299,221911425,249993168,53723744,21148882,301912],[0.2558,0.9155,0.1203,0.1318,227315258,256762938,52477511,19674065,305525],[0.262,0.9249,0.1217,0.1333,232055544,262369548,51299187,18774105,308642],[0.2683,0.9336,0.123,0.1348,236718772,268134174,50054502,17620042,311553],[0.2746,0.9424,0.1244,0.1363,241498425,274079902,48713356,16301386,314513],[0.2808,0.9493,0.1255,0.1376,245372484,279186143,47562459,14921840,316792],[0.2871,0.9546,0.1264,0.1385,248373969,282762389,46647518,14272428,318562],[0.2934,0.9593,0.1272,0.1393,251233101,286008490,45768665,13905520,320142],[0.2997,0.9642,0.1281,0.1403,254145562,289539326,44786460,13113756,321776],[0.3059,0.9686,0.1289,0.1412,256830047,292915993,43837391,12280121,323227],[0.3122,0.9713,0.1294,0.1418,258656050,295090717,43176007,11783080,324154],[0.3185,0.9745,0.13,0.1424,260743774,297559515,42393891,11263630,325209],[0.3247,0.9786,0.1308,0.1433,263500959,300856562,41322479,10519133,326574],[0.331,0.9804,0.1311,0.1437,264821785,302601027,40790398,9929043,327184],[0.3373,0.9827,0.1316,0.1443,266534151,304691624,40071702,9339558,327961],[0.3436,0.9861,0.1323,0.145,268807065,307405534,39049917,8646030,329070],[0.3498,0.9878,0.1327,0.1454,270099874,308958587,38447871,8257840,329640],[0.3561,0.9895,0.1331,0.1458,271338918,310511644,37871293,7764383,330217],[0.3624,0.9907,0.1334,0.1462,272265375,311706104,37433355,7349883,330624],[0.3686,0.9919,0.1336,0.1465,273158627,312874407,36994226,6985474,331010],[0.3749,0.993,0.1339,0.1467,274035622,313825714,36535284,6721101,331391],[0.3812,0.9936,0.1341,0.1468,274554580,314451287,36274999,6528929,331599],[0.3875,0.9945,0.1343,0.1471,275289302,315491418,35892731,6136823,331893],[0.3937,0.9949,0.1344,0.1472,275598347,315813476,35723906,6045411,332025],[0.4,0.9956,0.1346,0.1474,276205595,316647463,35390278,5686030,332262],
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
  model_optimum: { pd_cut: 0.187, approval_rate: 0.762, actual_default_rate: 0.11, model_profit: 61_002_991, realized_profit: 28_341_273 },
  realized_optimum: { pd_cut: 0.174, approval_rate: 0.723, actual_default_rate: 0.105, realized_credit_loss: 162_658_950, realized_profit: 28_521_794 },
  regret: 180_521,
  loss_ratio: 1.126,
  min_cut: 0.03,
  max_cut: 0.4,
};

/* fairness.json */
export const fairness: Record<"income" | "region" | "homeOwnership", FairnessDim> = {
  income: {
    groups: [
      { group: "Under $40k", approval_rate: 0.3581, approved_book_default_rate: 0.1105 },
      { group: "$40k–70k", approval_rate: 0.595, approved_book_default_rate: 0.0974 },
      { group: "$70k–120k", approval_rate: 0.7719, approved_book_default_rate: 0.0911 },
      { group: "$120k+", approval_rate: 0.8612, approved_book_default_rate: 0.0836 },
    ],
    air: 0.416,
    passes: false,
  },
  region: {
    groups: [
      { group: "Northeast", approval_rate: 0.647, approved_book_default_rate: 0.0994 },
      { group: "Midwest", approval_rate: 0.6195, approved_book_default_rate: 0.0886 },
      { group: "South", approval_rate: 0.6196, approved_book_default_rate: 0.0976 },
      { group: "West", approval_rate: 0.6313, approved_book_default_rate: 0.0906 },
    ],
    air: 0.958,
    passes: true,
  },
  homeOwnership: {
    groups: [
      { group: "Rent", approval_rate: 0.5107, approved_book_default_rate: 0.1056 },
      { group: "Own", approval_rate: 0.5955, approved_book_default_rate: 0.1033 },
      { group: "Mortgage", approval_rate: 0.7464, approved_book_default_rate: 0.0855 },
    ],
    air: 0.684,
    passes: false,
  },
};

export const overallApprovalRate = 0.6282;

/**
 * insight.json — global permutation feature importance of the shipped model
 * (model.joblib), computed on a 15,000-loan sample of the out-of-time test split,
 * 5 repeats, scoring = ROC AUC. This is a real attribution of the actual estimator —
 * not per-loan SHAP (out of scope; see DataModel for what that would take).
 */
export const featureImportance: FeatureImportanceRow[] = [
  { feature: "fico_mid", label: "Credit score (FICO)", importance: 0.0228 },
  { feature: "acc_open_past_24mths", label: "Accounts opened, last 24m", importance: 0.0182 },
  { feature: "loan_to_income", label: "Loan-to-income ratio", importance: 0.0131 },
  { feature: "dti", label: "Debt-to-income ratio", importance: 0.0091 },
  { feature: "num_tl_op_past_12m", label: "Trades opened, last 12m", importance: 0.005 },
  { feature: "purpose", label: "Loan purpose", importance: 0.0049 },
  { feature: "annual_inc", label: "Annual income", importance: 0.0047 },
  { feature: "mo_sin_old_rev_tl_op", label: "Age of oldest revolving trade", importance: 0.004 },
  { feature: "loan_amnt", label: "Loan amount", importance: 0.0031 },
  { feature: "home_ownership", label: "Home ownership", importance: 0.0025 },
  { feature: "mths_since_recent_bc", label: "Months since newest bankcard", importance: 0.0024 },
  { feature: "total_acc", label: "Total accounts", importance: 0.0024 },
];
export const featureImportanceMeta = {
  method: "sklearn permutation_importance, scoring=roc_auc",
  sampleN: 15000,
  nRepeats: 5,
  baselineAuc: 0.687,
};

/**
 * insight.json — LGD estimated the same way as the headline figure (1 − recovery rate
 * on charged-off TRAIN loans), but segmented by Lending Club grade instead of pooled.
 * Additive: the headline Expected Loss figures still use the single 49.95% portfolio
 * LGD — this shows the range that flat number hides.
 */
export const lgdByGrade: LgdByGradeRow[] = [
  { grade: "A", lgd: 0.4436, n_charged_off_train: 3458 },
  { grade: "B", lgd: 0.4727, n_charged_off_train: 11790 },
  { grade: "C", lgd: 0.5003, n_charged_off_train: 13090 },
  { grade: "D", lgd: 0.5333, n_charged_off_train: 8582 },
  { grade: "E", lgd: 0.5602, n_charged_off_train: 2837 },
  { grade: "F", lgd: 0.5946, n_charged_off_train: 766 },
  { grade: "G", lgd: 0.6, n_charged_off_train: 73 },
];

/* averages derived from the real book for the Portfolio metric line */
export const derived = {
  avgLoanSize: totals.exposure / totals.loans,
  avgInterestRate: 0.1199, // simulator_base mean int_rate
  totalLoans: totals.loans,
};
