/**
 * TypeScript formulas for every field in your schema.
 * Assumes you have aligned 5m candles for btc/eth/sol/x at the same index i (same open/close timestamps).
 * If you *don’t* have perfect alignment, you must resample/join first (by ts_close_ms).
 */

const EPS = 1e-12;

type Asset = "btc" | "eth" | "sol" | "x";

export type Candle = {
  ts_open_ms: number;
  ts_close_ms: number;

  open: number;
  high: number;
  low: number;
  close: number;

  volume: number;
  quote_volume: number;
  count: number;

  taker_buy_volume: number;
  taker_buy_quote_volume: number;
};

export type RowMeta = {
  ts_open_ms: number;
  ts_close_ms: number;
  tf_minutes: number;   // e.g. 5
  exchange: string;     // e.g. "binance"
  x_symbol: string;     // e.g. "HYPE"
};

/* ----------------------------- basic math helpers ----------------------------- */

function ln(x: number): number {
  return Math.log(x);
}

function logRet(close_t: number, close_prev: number): number {
  // ln(close_t / close_{t-1})
  return ln(close_t / (close_prev + EPS));
}

function trailRet(close_t: number, close_lag: number): number {
  // ln(close_t / close_{t-lag})
  return ln(close_t / (close_lag + EPS));
}

function candleRangePct(high: number, low: number, close: number): number {
  return (high - low) / (close + EPS);
}

function candleBodyPct(open: number, close: number): number {
  return (close - open) / (close + EPS);
}

function upperWickPct(open: number, high: number, close: number): number {
  return (high - Math.max(open, close)) / (close + EPS);
}

function lowerWickPct(open: number, low: number, close: number): number {
  return (Math.min(open, close) - low) / (close + EPS);
}

function log1p(x: number): number {
  return Math.log(x + 1);
}

/* ------------------------- rolling stats (formulas) -------------------------- */
/**
 * These are straightforward window formulas (O(window)).
 * In production you’ll likely replace them with rolling sums/variance for speed.
 */

function rollingMean(series: number[], i: number, window: number): number {
  // mean(series[i-window+1 .. i])
  const start = i - window + 1;
  if (start < 0) return NaN;
  let s = 0;
  for (let k = start; k <= i; k++) s += series[k];
  return s / window;
}

function rollingVar(series: number[], i: number, window: number): number {
  const start = i - window + 1;
  if (start < 0) return NaN;
  const mu = rollingMean(series, i, window);
  let s2 = 0;
  for (let k = start; k <= i; k++) {
    const d = series[k] - mu;
    s2 += d * d;
  }
  // population variance (divide by window). You can use (window-1) if you prefer sample var.
  return s2 / window;
}

function rollingStd(series: number[], i: number, window: number): number {
  const v = rollingVar(series, i, window);
  return Math.sqrt(v);
}

function zScore(x: number, mu: number, sigma: number): number {
  return (x - mu) / (sigma + EPS);
}

/* --------------------------- correlation / beta ------------------------------ */

function rollingCov(a: number[], b: number[], i: number, window: number): number {
  const start = i - window + 1;
  if (start < 0) return NaN;
  const muA = rollingMean(a, i, window);
  const muB = rollingMean(b, i, window);
  let s = 0;
  for (let k = start; k <= i; k++) {
    s += (a[k] - muA) * (b[k] - muB);
  }
  return s / window;
}

function rollingCorr(a: number[], b: number[], i: number, window: number): number {
  const cov = rollingCov(a, b, i, window);
  const stdA = rollingStd(a, i, window);
  const stdB = rollingStd(b, i, window);
  return cov / (stdA * stdB + EPS);
}

function rollingBeta(xRet: number[], mktRet: number[], i: number, window: number): number {
  // beta = cov(x, mkt) / var(mkt)
  const cov = rollingCov(xRet, mktRet, i, window);
  const v = rollingVar(mktRet, i, window);
  return cov / (v + EPS);
}

/* ----------------------------- flow ratios ---------------------------------- */

function takerBuyRatioVol(taker_buy_volume: number, volume: number): number {
  return taker_buy_volume / (volume + EPS);
}

function takerBuyRatioQv(taker_buy_quote_volume: number, quote_volume: number): number {
  return taker_buy_quote_volume / (quote_volume + EPS);
}

function rollingMeanOnTheFly(series: number[], i: number, window: number): number {
  return rollingMean(series, i, window);
}

/* -------------------------- core per-asset features -------------------------- */
/**
 * To compute rolling features, you typically precompute base series arrays:
 * closes[], volumes[], quoteVolumes[], counts[], logret1[], tbrVol[], tbrQv[]
 */

export type AssetSeries = {
  candles: Candle[];

  close: number[];
  volume: number[];
  quote_volume: number[];
  count: number[];
  logret1: number[];

  tbrVol: number[]; // taker buy ratio by volume
  tbrQv: number[];  // taker buy ratio by quote volume
};

export function buildAssetSeries(candles: Candle[]): AssetSeries {
  const close = candles.map(c => c.close);
  const volume = candles.map(c => c.volume);
  const quote_volume = candles.map(c => c.quote_volume);
  const count = candles.map(c => c.count);

  const logret1 = candles.map((c, i) => {
    if (i === 0) return NaN;
    return logRet(close[i], close[i - 1]);
  });

  const tbrVol = candles.map(c => takerBuyRatioVol(c.taker_buy_volume, c.volume));
  const tbrQv  = candles.map(c => takerBuyRatioQv(c.taker_buy_quote_volume, c.quote_volume));

  return { candles, close, volume, quote_volume, count, logret1, tbrVol, tbrQv };
}

/* -------------------- formulas for YOUR computed columns -------------------- */

export type PerAssetComputed = {
  // Price / trend
  logret_1: number;

  trail_ret_4: number;
  trail_ret_12: number;
  trail_ret_36: number;
  trail_ret_72: number;

  roll_mean_logret_12: number;
  roll_mean_logret_36: number;
  roll_mean_logret_72: number;

  roll_vol_logret_576: number;

  candle_range_pct: number;
  candle_body_pct: number;
  upper_wick_pct: number;
  lower_wick_pct: number;

  // Activity / liquidity
  volume_log: number;
  quote_volume_log: number;
  count_log: number;

  volume_z_576: number;
  quote_volume_z_576: number;
  count_z_576: number;

  // Flow / aggression
  taker_buy_ratio_vol: number;
  taker_buy_ratio_qv: number;

  taker_buy_ratio_vol_ma12: number;
  taker_buy_ratio_qv_ma12: number;

  taker_buy_ratio_vol_ma36: number;
  taker_buy_ratio_qv_ma36: number;
};

export function computePerAssetComputed(s: AssetSeries, i: number): PerAssetComputed {
  const c = s.candles[i];

  // -------- Price / trend --------
  const logret_1 = s.logret1[i];

  const trail_ret_4  = i >= 4  ? trailRet(s.close[i], s.close[i - 4])  : NaN;
  const trail_ret_12 = i >= 12 ? trailRet(s.close[i], s.close[i - 12]) : NaN;
  const trail_ret_36 = i >= 36 ? trailRet(s.close[i], s.close[i - 36]) : NaN;
  const trail_ret_72 = i >= 72 ? trailRet(s.close[i], s.close[i - 72]) : NaN;

  const roll_mean_logret_12 = rollingMean(s.logret1, i, 12);
  const roll_mean_logret_36 = rollingMean(s.logret1, i, 36);
  const roll_mean_logret_72 = rollingMean(s.logret1, i, 72);

  const roll_vol_logret_576 = rollingStd(s.logret1, i, 576);

  const candle_range_pct = candleRangePct(c.high, c.low, c.close);
  const candle_body_pct  = candleBodyPct(c.open, c.close);
  const upper_wick_pct   = upperWickPct(c.open, c.high, c.close);
  const lower_wick_pct   = lowerWickPct(c.open, c.low,  c.close);

  // -------- Activity / liquidity --------
  const volume_log       = log1p(c.volume);
  const quote_volume_log = log1p(c.quote_volume);
  const count_log        = log1p(c.count);

  const muVol  = rollingMean(s.volume, i, 576);
  const sdVol  = rollingStd(s.volume, i, 576);
  const volume_z_576 = zScore(c.volume, muVol, sdVol);

  const muQv = rollingMean(s.quote_volume, i, 576);
  const sdQv = rollingStd(s.quote_volume, i, 576);
  const quote_volume_z_576 = zScore(c.quote_volume, muQv, sdQv);

  const muCnt = rollingMean(s.count, i, 576);
  const sdCnt = rollingStd(s.count, i, 576);
  const count_z_576 = zScore(c.count, muCnt, sdCnt);

  // -------- Flow / aggression --------
  const taker_buy_ratio_vol = s.tbrVol[i]; // taker_buy_volume / volume
  const taker_buy_ratio_qv  = s.tbrQv[i];  // taker_buy_quote_volume / quote_volume

  const taker_buy_ratio_vol_ma12 = rollingMean(s.tbrVol, i, 12);
  const taker_buy_ratio_qv_ma12  = rollingMean(s.tbrQv,  i, 12);

  const taker_buy_ratio_vol_ma36 = rollingMean(s.tbrVol, i, 36);
  const taker_buy_ratio_qv_ma36  = rollingMean(s.tbrQv,  i, 36);

  return {
    logret_1,
    trail_ret_4, trail_ret_12, trail_ret_36, trail_ret_72,
    roll_mean_logret_12, roll_mean_logret_36, roll_mean_logret_72,
    roll_vol_logret_576,
    candle_range_pct, candle_body_pct, upper_wick_pct, lower_wick_pct,
    volume_log, quote_volume_log, count_log,
    volume_z_576, quote_volume_z_576, count_z_576,
    taker_buy_ratio_vol, taker_buy_ratio_qv,
    taker_buy_ratio_vol_ma12, taker_buy_ratio_qv_ma12,
    taker_buy_ratio_vol_ma36, taker_buy_ratio_qv_ma36
  };
}

/* ---------------------- cross-asset features for X ---------------------- */

export type CrossForX = {
  x_minus_btc_trail_ret_12: number;
  x_minus_btc_trail_ret_36: number;
  x_minus_btc_trail_ret_72: number;

  x_minus_eth_trail_ret_12: number;
  x_minus_sol_trail_ret_12: number;

  corr_x_btc_logret_72: number;
  corr_x_eth_logret_72: number;
  corr_x_sol_logret_72: number;

  beta_x_btc_288: number;
  beta_x_eth_288: number;
  beta_x_sol_288: number;
};

export function computeCrossForX(
  btc: AssetSeries,
  eth: AssetSeries,
  sol: AssetSeries,
  x: AssetSeries,
  i: number
): CrossForX {
  // Relative strength = difference of trailing returns
  const x_minus_btc_trail_ret_12 = (i >= 12) ? (trailRet(x.close[i], x.close[i-12]) - trailRet(btc.close[i], btc.close[i-12])) : NaN;
  const x_minus_btc_trail_ret_36 = (i >= 36) ? (trailRet(x.close[i], x.close[i-36]) - trailRet(btc.close[i], btc.close[i-36])) : NaN;
  const x_minus_btc_trail_ret_72 = (i >= 72) ? (trailRet(x.close[i], x.close[i-72]) - trailRet(btc.close[i], btc.close[i-72])) : NaN;

  const x_minus_eth_trail_ret_12 = (i >= 12) ? (trailRet(x.close[i], x.close[i-12]) - trailRet(eth.close[i], eth.close[i-12])) : NaN;
  const x_minus_sol_trail_ret_12 = (i >= 12) ? (trailRet(x.close[i], x.close[i-12]) - trailRet(sol.close[i], sol.close[i-12])) : NaN;

  // Rolling correlation on 1-bar log returns
  const corr_x_btc_logret_72 = rollingCorr(x.logret1, btc.logret1, i, 72);
  const corr_x_eth_logret_72 = rollingCorr(x.logret1, eth.logret1, i, 72);
  const corr_x_sol_logret_72 = rollingCorr(x.logret1, sol.logret1, i, 72);

  // Rolling beta on 1-bar log returns
  const beta_x_btc_288 = rollingBeta(x.logret1, btc.logret1, i, 288);
  const beta_x_eth_288 = rollingBeta(x.logret1, eth.logret1, i, 288);
  const beta_x_sol_288 = rollingBeta(x.logret1, sol.logret1, i, 288);

  return {
    x_minus_btc_trail_ret_12,
    x_minus_btc_trail_ret_36,
    x_minus_btc_trail_ret_72,
    x_minus_eth_trail_ret_12,
    x_minus_sol_trail_ret_12,
    corr_x_btc_logret_72,
    corr_x_eth_logret_72,
    corr_x_sol_logret_72,
    beta_x_btc_288,
    beta_x_eth_288,
    beta_x_sol_288
  };
}

/* -------------------------- targets for X (H=4/12/36/72) -------------------------- */

export type TargetH = {
  x_fwd_logret_H: number;
  x_sigma_H: number;
  x_cost_H: number;
  x_z_H: number;
  x_score_H: number;
};

export type CostModel = {
  /**
   * Return expected ROUND-TRIP cost for horizon H, in the same units as log return (approx).
   * E.g. constant bps: cost = (feeBps + slippageBps) / 10000
   */
  costLogReturnH: (H: number) => number;
};

/**
 * Example constant cost model (you should tune these to your venue/fees/slippage).
 */
export function makeConstantCostModel(totalRoundTripBps: number): CostModel {
  const cost = totalRoundTripBps / 10000;
  return { costLogReturnH: (_H: number) => cost };
}

export function computeTargetsForXAtH(
  x: AssetSeries,
  i: number,
  H: number,
  costModel: CostModel
): TargetH {
  // forward log return: ln(x_close_{t+H} / x_close_t)
  const x_fwd_logret_H = (i + H < x.close.length) ? ln(x.close[i + H] / (x.close[i] + EPS)) : NaN;

  // base vol (2 days) already computed as x_roll_vol_logret_576 in per-asset features,
  // but we can recompute here from the series:
  const sigma_base = rollingStd(x.logret1, i, 576);          // std of 1-bar (5m) log returns
  const x_sigma_H = sigma_base * Math.sqrt(H);               // scale vol to horizon H

  const x_cost_H = costModel.costLogReturnH(H);

  const x_fwd_logret_adj_H = x_fwd_logret_H - x_cost_H;

  const x_z_H = x_fwd_logret_adj_H / (x_sigma_H + EPS);

  // bounded score in [-1, +1]
  const x_score_H = Math.tanh(x_z_H);

  return { x_fwd_logret_H, x_sigma_H, x_cost_H, x_z_H, x_score_H };
}

/* -------------------------- putting it all together (one row) -------------------------- */

export type FeatureRow = RowMeta & {
  // raw columns (prefixed)
  btc_open: number; btc_high: number; btc_low: number; btc_close: number;
  btc_volume: number; btc_quote_volume: number; btc_count: number;
  btc_taker_buy_volume: number; btc_taker_buy_quote_volume: number;

  eth_open: number; eth_high: number; eth_low: number; eth_close: number;
  eth_volume: number; eth_quote_volume: number; eth_count: number;
  eth_taker_buy_volume: number; eth_taker_buy_quote_volume: number;

  sol_open: number; sol_high: number; sol_low: number; sol_close: number;
  sol_volume: number; sol_quote_volume: number; sol_count: number;
  sol_taker_buy_volume: number; sol_taker_buy_quote_volume: number;

  x_open: number; x_high: number; x_low: number; x_close: number;
  x_volume: number; x_quote_volume: number; x_count: number;
  x_taker_buy_volume: number; x_taker_buy_quote_volume: number;

  // computed columns for each asset (prefixed)
  // (I include them as nested objects here; when writing CSV you’ll flatten keys.)
  btc_comp: PerAssetComputed;
  eth_comp: PerAssetComputed;
  sol_comp: PerAssetComputed;
  x_comp: PerAssetComputed;

  x_cross: CrossForX;

  // targets (for training labels)
  tgt_4: TargetH;
  tgt_12: TargetH;
  tgt_36: TargetH;
  tgt_72: TargetH;
};

export function computeRow(
  meta: RowMeta,
  i: number,
  btc: AssetSeries,
  eth: AssetSeries,
  sol: AssetSeries,
  x: AssetSeries,
  costModel: CostModel
): FeatureRow {
  const btcC = btc.candles[i];
  const ethC = eth.candles[i];
  const solC = sol.candles[i];
  const xC   = x.candles[i];

  const btc_comp = computePerAssetComputed(btc, i);
  const eth_comp = computePerAssetComputed(eth, i);
  const sol_comp = computePerAssetComputed(sol, i);
  const x_comp   = computePerAssetComputed(x, i);

  const x_cross = computeCrossForX(btc, eth, sol, x, i);

  const tgt_4  = computeTargetsForXAtH(x, i, 4,  costModel);
  const tgt_12 = computeTargetsForXAtH(x, i, 12, costModel);
  const tgt_36 = computeTargetsForXAtH(x, i, 36, costModel);
  const tgt_72 = computeTargetsForXAtH(x, i, 72, costModel);

  return {
    ...meta,

    btc_open: btcC.open, btc_high: btcC.high, btc_low: btcC.low, btc_close: btcC.close,
    btc_volume: btcC.volume, btc_quote_volume: btcC.quote_volume, btc_count: btcC.count,
    btc_taker_buy_volume: btcC.taker_buy_volume, btc_taker_buy_quote_volume: btcC.taker_buy_quote_volume,

    eth_open: ethC.open, eth_high: ethC.high, eth_low: ethC.low, eth_close: ethC.close,
    eth_volume: ethC.volume, eth_quote_volume: ethC.quote_volume, eth_count: ethC.count,
    eth_taker_buy_volume: ethC.taker_buy_volume, eth_taker_buy_quote_volume: ethC.taker_buy_quote_volume,

    sol_open: solC.open, sol_high: solC.high, sol_low: solC.low, sol_close: solC.close,
    sol_volume: solC.volume, sol_quote_volume: solC.quote_volume, sol_count: solC.count,
    sol_taker_buy_volume: solC.taker_buy_volume, sol_taker_buy_quote_volume: solC.taker_buy_quote_volume,

    x_open: xC.open, x_high: xC.high, x_low: xC.low, x_close: xC.close,
    x_volume: xC.volume, x_quote_volume: xC.quote_volume, x_count: xC.count,
    x_taker_buy_volume: xC.taker_buy_volume, x_taker_buy_quote_volume: xC.taker_buy_quote_volume,

    btc_comp, eth_comp, sol_comp, x_comp,
    x_cross,
    tgt_4, tgt_12, tgt_36, tgt_72
  };
}

/**
 * Notes:
 * - Any field returning NaN means “insufficient history or future data” at index i.
 *   You typically drop those rows when training.
 * - For training, you must ensure i+H exists (otherwise targets are NaN at the end).
 * - For live inference, you compute ONLY feature columns (NOT targets).
 */
