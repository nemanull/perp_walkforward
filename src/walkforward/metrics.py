import math

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from walkforward.config import BOOTSTRAP_DRAWS, BOOTSTRAP_MEAN_BLOCK_DAYS, SEED


def standardised_ranks(values: pd.Series) -> pd.Series:
    ranks = values.rank()
    return (ranks - ranks.mean()) / ranks.std(ddof=0)


def rank_products(pred: pd.Series, realised: pd.Series) -> pd.Series:
    both = pd.DataFrame({"pred": pred, "realised": realised}).dropna()
    return standardised_ranks(both["pred"]) * standardised_ranks(both["realised"])


def daily_rank_ic(pred: pd.Series, realised: pd.Series) -> pd.Series:
    product = rank_products(pred, realised)
    return product.groupby(product.index.normalize().rename("day")).mean()


def equal_weight(by_coin: dict[str, pd.Series] | dict[str, pd.DataFrame]) -> pd.Series:
    return pd.concat(by_coin.values()).groupby(level=0).mean()


def stationary_bootstrap_indices(n: int, mean_block: float, draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    new_block = rng.random((draws, n)) < 1 / mean_block
    new_block[:, 0] = True
    position = np.arange(n)
    block_start = np.maximum.accumulate(np.where(new_block, position, 0), axis=1)
    first_index = rng.integers(0, n, size=(draws, n))
    return (np.take_along_axis(first_index, block_start, axis=1) + position - block_start) % n


def stationary_bootstrap_se(
    daily: pd.Series,
    mean_block: float = BOOTSTRAP_MEAN_BLOCK_DAYS,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
) -> float:
    values = np.asarray(daily, dtype=float)
    indices = stationary_bootstrap_indices(len(values), mean_block, draws, seed)
    return float(values[indices].mean(axis=1).std())


def bootstrap_t(daily: pd.Series) -> float:
    se = stationary_bootstrap_se(daily)
    return float(daily.mean() / se) if se > 0 else float("nan")


def naive_t(daily: pd.Series) -> float:
    return float(daily.mean() / daily.std() * math.sqrt(len(daily)))


def ic_summary(daily: pd.Series) -> dict[str, float]:
    ic = float(daily.mean())
    se = stationary_bootstrap_se(daily)
    return {"ic": ic, "se": se, "t": ic / se if se > 0 else float("nan"), "days": len(daily)}


def pooled_ic_summary(daily_by_coin: dict[str, pd.Series]) -> dict[str, float]:
    return ic_summary(equal_weight(daily_by_coin))


def paired_ic_difference(daily_a: pd.Series, daily_b: pd.Series) -> dict[str, float]:
    return ic_summary((daily_a - daily_b).dropna())


def pooled_ic_difference(
    daily_a: dict[str, pd.Series], daily_b: dict[str, pd.Series]
) -> dict[str, float]:
    return pooled_ic_summary({coin: (daily_a[coin] - daily_b[coin]).dropna() for coin in daily_a})


def ic_by_month(daily: pd.Series) -> pd.Series:
    return daily.groupby(daily.index.strftime("%Y-%m").rename("month")).mean()


def hit_rate(side: ArrayLike, realised: ArrayLike) -> float:
    agreement = np.asarray(side) * np.asarray(realised)
    return float((agreement > 0).sum() / (agreement != 0).sum())


def oos_r2(pred: pd.Series, y: pd.Series) -> float:
    return float(1 - ((y - pred) ** 2).sum() / (y**2).sum())


def within_month_spearman(a: pd.Series, b: pd.Series, month: pd.Series) -> float:
    both = pd.concat({"a": a, "b": b, "month": month}, axis=1).dropna()
    by_month = both.groupby("month")[["a", "b"]].corr("spearman").xs("a", level=1)["b"]
    return float(by_month.mean())


def decile_returns(pred: pd.Series, realised: pd.Series, month: pd.Series) -> pd.DataFrame:
    decile = np.ceil(10 * pred.groupby(month).rank(pct=True)).astype(int).rename("decile")
    return realised.groupby([month.rename("month"), decile]).mean().unstack("decile")
