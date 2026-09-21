import warnings

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from walkforward.config import BOOTSTRAP_DRAWS, BOOTSTRAP_MEAN_BLOCK_DAYS, SEED
from walkforward.metrics import stationary_bootstrap_indices

DAYS_PER_YEAR = 365


def signals(pred: ArrayLike, q_lo: ArrayLike, q_hi: ArrayLike) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    return np.select([pred > q_hi, pred < q_lo], [1, -1], default=0)


def tranche_position(signal: ArrayLike, horizon: int, delay: int = 0) -> np.ndarray:
    signal = np.nan_to_num(np.asarray(signal, dtype=float))
    opened = np.cumsum(np.concatenate([np.zeros(horizon + delay), signal]))
    bars = len(signal)
    return (opened[horizon : horizon + bars] - opened[:bars]) / horizon


def bar_pnl(
    position: ArrayLike, close: pd.Series, fee_bps: float, funding: pd.Series
) -> pd.DataFrame:
    position = np.asarray(position, dtype=float)
    held = np.concatenate([[0.0], position[:-1]])
    turnover = np.abs(position - held)
    turnover[-1] += abs(position[-1])
    # ffill books a halt's move on the next traded bar
    bar_return = close.ffill().pct_change().fillna(0.0).to_numpy()
    rate = funding.reindex(close.index).fillna(0.0).to_numpy()
    gross = held * bar_return
    fees = fee_bps / 1e4 * turnover
    funding_paid = held * rate
    return pd.DataFrame(
        {
            "gross": gross,
            "fees": fees,
            "funding": funding_paid,
            "net": gross - fees - funding_paid,
            "turnover": turnover,
        },
        index=close.index,
    )


def daily_pnl(pnl: pd.DataFrame) -> pd.DataFrame:
    return pnl.resample("D").sum()


def sharpe(daily_net: pd.Series) -> float:
    spread = daily_net.std()
    return float(np.sqrt(DAYS_PER_YEAR) * daily_net.mean() / spread) if spread > 0 else np.nan


def sharpe_interval(daily_net: pd.Series, level: float = 0.9) -> tuple[float, float]:
    values = daily_net.to_numpy()
    indices = stationary_bootstrap_indices(
        len(values), BOOTSTRAP_MEAN_BLOCK_DAYS, BOOTSTRAP_DRAWS, SEED
    )
    resampled = values[indices]
    # draws without a trading day give NaN, which nanquantile skips
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.sqrt(DAYS_PER_YEAR) * resampled.mean(axis=1) / resampled.std(axis=1, ddof=1)
    if np.isnan(ratios).all():
        return np.nan, np.nan
    tail = (1 - level) / 2
    low, high = np.nanquantile(ratios, [tail, 1 - tail])
    return float(low), float(high)


def max_drawdown(daily_net: pd.Series) -> float:
    equity = daily_net.cumsum()
    # equity starts at zero, so a first-day loss is a drawdown
    return float((equity.cummax().clip(lower=0.0) - equity).max())


def breakeven_fee_bps(pnl: pd.DataFrame) -> float:
    turnover = pnl["turnover"].sum()
    edge = pnl["gross"].sum() - pnl["funding"].sum()
    return float(edge / turnover * 1e4) if turnover > 0 else np.nan


def time_in_market(position: ArrayLike) -> float:
    return float(np.mean(np.asarray(position) != 0))


def long_share(position: ArrayLike) -> float:
    position = np.asarray(position, dtype=float)
    exposure = np.abs(position).sum()
    return float(position.clip(min=0).sum() / exposure) if exposure > 0 else np.nan


def beta_to_target(daily_net: pd.Series, daily_target_return: pd.Series) -> float:
    return float(daily_net.cov(daily_target_return) / daily_target_return.var())


def buy_and_hold(close: pd.Series) -> pd.Series:
    price = close.ffill()
    day_close = price.resample("D").last()
    return day_close / day_close.shift(fill_value=price.iloc[0]) - 1
