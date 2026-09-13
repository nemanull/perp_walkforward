import logging

import numpy as np
import pandas as pd

from walkforward import config
from walkforward.data import bars
from walkforward.experiments import common

log = logging.getLogger(__name__)


def five_minute_log_returns(start: str, end: str) -> pd.DataFrame:
    close = common.load_bars()[[f"{asset}_close" for asset in config.PAIRS]]
    returns = np.log(close).diff().set_axis(list(config.PAIRS), axis=1)
    return common.between(returns, start, end)


def extreme_bars_table() -> pd.DataFrame:
    tables = []
    for asset in config.PAIRS:
        close = common.load_bars()[f"{asset}_close"]
        close = common.between(close, config.GRID_START, config.RESEARCH_END)
        moves = np.log(close).diff()[bars.is_extreme_return(close)]
        tables.append(pd.DataFrame({"asset": asset, "open_time": moves.index, "log_return": moves}))
    return pd.concat(tables)


def rolling_btc_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    window, filled = 30 * bars.BARS_PER_DAY, 20 * bars.BARS_PER_DAY
    rolling = returns[list(config.TARGETS)].rolling(window, min_periods=filled)
    daily = rolling.corr(returns["btc"]).resample("1D").last().dropna(how="all")
    return daily.rename_axis("day").reset_index()


def lead_lag_table(returns: pd.DataFrame) -> pd.DataFrame:
    calm = returns.copy()
    calm.loc[common.in_crash(calm.index)] = np.nan
    lags = range(-6, 7)
    table = pd.DataFrame(
        {
            coin: [calm[coin].corr(calm["btc"].shift(lag)) for lag in lags]
            for coin in config.TARGETS
        },
        index=pd.Index(lags, name="btc_lead_bars"),
    )
    return table.reset_index()


def autocorrelation_table(returns: pd.DataFrame) -> pd.DataFrame:
    calm = returns[~common.in_crash(returns.index)]
    assets = list(config.PAIRS)
    return pd.DataFrame(
        {
            "asset": assets,
            "lag1": [returns[asset].autocorr() for asset in assets],
            "lag1_without_crash": [calm[asset].autocorr() for asset in assets],
            "lag1_rank_without_crash": [calm[asset].rank().autocorr() for asset in assets],
        }
    )


def daily_volatility_table(returns: pd.DataFrame) -> pd.DataFrame:
    volatility = returns[list(config.TARGETS)].resample("1D").std() * np.sqrt(bars.BARS_PER_DAY)
    return volatility.dropna(how="all").rename_axis("day").reset_index()


def run_audit() -> None:
    out = common.output_dir("audit")
    # research rows only
    end = common.utc(config.RESEARCH_END) - bars.BAR
    frames = {asset: bars.read_klines(asset).loc[:end] for asset in config.PAIRS}
    seams = bars.find_seams(frames)
    common.write_table(bars.audit_bars(frames), out / "bars.csv")
    common.write_table(seams, out / "seams.csv")
    late = seams[seams["first_bar_after"] >= common.utc(config.TRAIN_START)]
    if len(late):
        raise RuntimeError(f"price seams inside the study window:\n{late}")

    returns = five_minute_log_returns(config.GRID_START, config.RESEARCH_END)
    first = common.between(returns, *common.FIRST_WINDOW)
    correlation = returns.corr().rename_axis("asset").reset_index()
    common.write_table(extreme_bars_table(), out / "extreme_bars.csv")
    common.write_table(correlation, out / "return_correlation.csv", digits=4)
    rolling = rolling_btc_correlation(returns)
    common.write_table(rolling, out / "rolling_btc_correlation.csv", digits=4)
    common.write_table(lead_lag_table(first), out / "lead_lag.csv", digits=4)
    common.write_table(autocorrelation_table(first), out / "autocorrelation.csv", digits=4)
    common.write_table(daily_volatility_table(returns), out / "daily_volatility.csv", digits=5)
    log.info("audit written, %d seams", len(seams))


EXPERIMENTS = {
    "audit": run_audit,
}
