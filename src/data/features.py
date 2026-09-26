import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONTEXT, DATA_DIR, HORIZONS, TARGETS
from src.data.bars import BAR_FIELDS

log = logging.getLogger(__name__)

EPS = 1e-12
SCHEMA_ASSETS = (*CONTEXT, "x")
INPUT_ASSETS = ("x", *CONTEXT)  # own inputs first, so best_feature ties go to the coin

PRICE = (
    "logret_1",
    "trail_ret_4",
    "trail_ret_12",
    "trail_ret_36",
    "trail_ret_72",
    "roll_mean_logret_12",
    "roll_mean_logret_36",
    "roll_mean_logret_72",
    "roll_vol_logret_576",
    "candle_range_pct",
    "candle_body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
)
ACTIVITY = (
    "volume_log",
    "quote_volume_log",
    "count_log",
    "volume_z_576",
    "quote_volume_z_576",
    "count_z_576",
)
FLOW = (
    "taker_buy_ratio_vol",
    "taker_buy_ratio_qv",
    "taker_buy_ratio_vol_ma12",
    "taker_buy_ratio_qv_ma12",
    "taker_buy_ratio_vol_ma36",
    "taker_buy_ratio_qv_ma36",
)
CROSS = (
    "x_minus_btc_trail_ret_12",
    "x_minus_btc_trail_ret_36",
    "x_minus_btc_trail_ret_72",
    "x_minus_eth_trail_ret_12",
    "x_minus_sol_trail_ret_12",
    "corr_x_btc_logret_72",
    "corr_x_eth_logret_72",
    "corr_x_sol_logret_72",
    "beta_x_btc_288",
    "beta_x_eth_288",
    "beta_x_sol_288",
)
PER_ASSET = PRICE + ACTIVITY + FLOW

# levels drift over months, the other six copy columns that stay
NOT_INPUTS = {
    "volume_log",
    "quote_volume_log",
    "count_log",
    "roll_mean_logret_12",
    "roll_mean_logret_36",
    "roll_mean_logret_72",
    "taker_buy_ratio_qv",
    "taker_buy_ratio_qv_ma12",
    "taker_buy_ratio_qv_ma36",
}

FEATURES = [
    f"{asset}_{name}"
    for group in (PRICE, ACTIVITY, FLOW)
    for asset in SCHEMA_ASSETS
    for name in group
] + list(CROSS)
MODEL_INPUTS = [
    f"{asset}_{name}" for asset in INPUT_ASSETS for name in PER_ASSET if name not in NOT_INPUTS
] + list(CROSS)
OWN_INPUTS = [column for column in MODEL_INPUTS if column.startswith("x_") and column not in CROSS]
IMPORTANCE_GROUPS = {"x": OWN_INPUTS} | {
    asset: [c for c in MODEL_INPUTS if c.startswith(f"{asset}_") or f"_{asset}_" in c]
    for asset in CONTEXT
}


def log_return(close: pd.Series, lag: int) -> pd.Series:
    # complete grid, so shift(lag) is a time lag
    return np.log(close / (close.shift(lag) + EPS))


def rolling_mean(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window).mean()


def rolling_std(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window).std(ddof=0)


def rolling_cov(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    return a.rolling(window).cov(b, ddof=0)


def rolling_zscore(values: pd.Series, window: int) -> pd.Series:
    return (values - rolling_mean(values, window)) / (rolling_std(values, window) + EPS)


def rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    return rolling_cov(a, b, window) / (rolling_std(a, window) * rolling_std(b, window) + EPS)


def rolling_beta(returns: pd.Series, market: pd.Series, window: int) -> pd.Series:
    return rolling_cov(returns, market, window) / (rolling_std(market, window) ** 2 + EPS)


def asset_features(bars: pd.DataFrame, asset: str) -> pd.DataFrame:
    bar = {field: bars[f"{asset}_{field}"] for field in BAR_FIELDS}
    close = bar["close"]
    logret = log_return(close, 1)
    activity = ("volume", "quote_volume", "count")
    buy_ratio = {
        "vol": bar["taker_buy_volume"] / (bar["volume"] + EPS),
        "qv": bar["taker_buy_quote_volume"] / (bar["quote_volume"] + EPS),
    }
    columns = {
        "logret_1": logret,
        **{f"trail_ret_{lag}": log_return(close, lag) for lag in (4, 12, 36, 72)},
        **{f"roll_mean_logret_{window}": rolling_mean(logret, window) for window in (12, 36, 72)},
        "roll_vol_logret_576": rolling_std(logret, 576),
        "candle_range_pct": (bar["high"] - bar["low"]) / (close + EPS),
        "candle_body_pct": (close - bar["open"]) / (close + EPS),
        "upper_wick_pct": (bar["high"] - np.maximum(bar["open"], close)) / (close + EPS),
        "lower_wick_pct": (np.minimum(bar["open"], close) - bar["low"]) / (close + EPS),
        **{f"{field}_log": np.log1p(bar[field]) for field in activity},
        **{f"{field}_z_576": rolling_zscore(bar[field], 576) for field in activity},
        **{f"taker_buy_ratio_{kind}": ratio for kind, ratio in buy_ratio.items()},
        **{
            f"taker_buy_ratio_{kind}_ma{window}": rolling_mean(ratio, window)
            for window in (12, 36)
            for kind, ratio in buy_ratio.items()
        },
    }
    return pd.DataFrame(columns).add_prefix(f"{asset}_")


def cross_features(bars: pd.DataFrame) -> pd.DataFrame:
    close = {asset: bars[f"{asset}_close"] for asset in ("x", *CONTEXT)}
    logret = {asset: log_return(series, 1) for asset, series in close.items()}
    columns = {}
    for asset, lag in (("btc", 12), ("btc", 36), ("btc", 72), ("eth", 12), ("sol", 12)):
        excess = log_return(close["x"], lag) - log_return(close[asset], lag)
        columns[f"x_minus_{asset}_trail_ret_{lag}"] = excess
    for asset in CONTEXT:
        columns[f"corr_x_{asset}_logret_72"] = rolling_corr(logret["x"], logret[asset], 72)
        columns[f"beta_x_{asset}_288"] = rolling_beta(logret["x"], logret[asset], 288)
    return pd.DataFrame(columns)[list(CROSS)]


def labels(bars: pd.DataFrame, horizons: Iterable[int] = HORIZONS) -> pd.DataFrame:
    close = bars["x_close"]
    bar_volatility = rolling_std(log_return(close, 1), 576)
    columns = {}
    for horizon in horizons:
        forward = log_return(close, horizon).shift(-horizon)
        sigma = bar_volatility * np.sqrt(horizon)
        columns[f"fwd_logret_{horizon}"] = forward
        columns[f"sigma_{horizon}"] = sigma
        columns[f"y_{horizon}"] = np.tanh(forward / (sigma + EPS))
    return pd.DataFrame(columns)


def build_features(bars: pd.DataFrame, coin: str) -> pd.DataFrame:
    frame = bars.rename(columns={f"{coin}_{field}": f"x_{field}" for field in BAR_FIELDS})
    features = pd.concat(
        [*(asset_features(frame, asset) for asset in SCHEMA_ASSETS), cross_features(frame)],
        axis=1,
    )
    return pd.concat([frame["x_close"], features[FEATURES], labels(frame)], axis=1)


def build(data_dir: Path = DATA_DIR) -> None:
    bars = pd.read_parquet(data_dir / "bars.parquet")
    folder = data_dir / "features"
    folder.mkdir(exist_ok=True)
    for coin in TARGETS:
        build_features(bars, coin).to_parquet(folder / f"{coin}.parquet")
        log.debug("built features for %s", coin)
    log.info("wrote features for %s", ", ".join(TARGETS))
