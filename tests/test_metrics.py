import numpy as np
import pandas as pd
import pytest
from conftest import bar_index, day_index
from scipy.stats import rankdata, spearmanr

from walkforward.config import SEED
from walkforward.data.bars import BARS_PER_DAY
from walkforward.metrics import (
    bootstrap_t,
    daily_rank_ic,
    decile_returns,
    equal_weight,
    hit_rate,
    ic_summary,
    paired_ic_difference,
    pooled_ic_difference,
    stationary_bootstrap_indices,
    stationary_bootstrap_se,
)


def bars_for_days(days: int) -> pd.DatetimeIndex:
    return bar_index(days * BARS_PER_DAY)


def trailing_and_forward_returns(
    rng: np.random.Generator, paths: int, bars: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    log_price = np.cumsum(rng.standard_normal((paths, bars + 2 * horizon)), axis=1)
    moves = log_price[:, horizon:] - log_price[:, :-horizon]
    return moves[:, :bars], moves[:, horizon:]


def standardised(values: np.ndarray) -> np.ndarray:
    return (values - values.mean(axis=-1, keepdims=True)) / values.std(axis=-1, keepdims=True)


def within_day_spearman(pred: np.ndarray, realised: np.ndarray) -> np.ndarray:
    by_day = (-1, BARS_PER_DAY)
    pred_ranks = standardised(rankdata(pred.reshape(by_day), axis=1))
    realised_ranks = standardised(rankdata(realised.reshape(by_day), axis=1))
    return (pred_ranks * realised_ranks).mean(axis=1)


def test_perfect_predictor_has_ic_of_one():
    rng = np.random.default_rng(SEED)
    realised = pd.Series(rng.standard_normal(10 * BARS_PER_DAY), index=bars_for_days(10))
    assert daily_rank_ic(np.exp(realised), realised).mean() == pytest.approx(1.0)


def test_daily_mean_equals_period_spearman():
    rng = np.random.default_rng(SEED)
    realised = pd.Series(rng.standard_normal(10 * BARS_PER_DAY), index=bars_for_days(10))
    pred = realised + 3 * rng.standard_normal(len(realised))
    expected = spearmanr(pred, realised).statistic
    assert daily_rank_ic(pred, realised).mean() == pytest.approx(expected)


def test_random_walk_null_and_per_day_bias():
    paths, days, horizon = 100, 182, 72
    rng = np.random.default_rng(SEED)
    trailing, forward = trailing_and_forward_returns(rng, paths, days * BARS_PER_DAY, horizon)
    index = bars_for_days(days)
    pooled_ic, per_day_spearman = [], []
    for past, future in zip(trailing, forward, strict=True):
        pooled_ic.append(daily_rank_ic(pd.Series(past, index), pd.Series(future, index)).mean())
        per_day_spearman.append(within_day_spearman(past, future).mean())
    assert abs(np.mean(pooled_ic)) < 0.01
    assert np.mean(per_day_spearman) < -0.2


def test_planted_signal_is_detected():
    days, horizon = 182, 12
    _, forward = trailing_and_forward_returns(
        np.random.default_rng(SEED), 2, days * BARS_PER_DAY, horizon
    )
    realised, noise = forward
    # the noise overlaps across bars like the label, as real predictions do
    pred = 0.1 * realised + np.sqrt(1 - 0.1**2) * noise
    index = bars_for_days(days)
    summary = ic_summary(daily_rank_ic(pd.Series(pred, index), pd.Series(realised, index)))
    assert summary["ic"] == pytest.approx(0.1, abs=0.03)
    assert summary["t"] > 3


def test_bootstrap_blocks_have_the_requested_mean_length():
    indices = stationary_bootstrap_indices(n=200, mean_block=5, draws=2000, seed=SEED)
    continues = indices[:, 1:] == (indices[:, :-1] + 1) % 200
    assert 1 / (1 - continues.mean()) == pytest.approx(5, rel=0.05)


def test_bootstrap_se_of_iid_days():
    daily = pd.Series(np.random.default_rng(SEED).standard_normal(1000))
    expected = daily.std() / np.sqrt(len(daily))
    assert stationary_bootstrap_se(daily, draws=2000) == pytest.approx(expected, rel=0.15)


def test_bootstrap_t():
    daily = pd.Series(np.random.default_rng(SEED).normal(0.1, 1.0, 300))
    assert bootstrap_t(daily) == pytest.approx(ic_summary(daily)["t"])
    assert np.isnan(bootstrap_t(pd.Series(0.0, index=range(30))))


def test_pooling_two_coins_shrinks_se_by_root_two():
    rng = np.random.default_rng(SEED)
    days = day_index(1000)
    coins = {coin: pd.Series(rng.standard_normal(len(days)), index=days) for coin in ("a", "b")}
    single = np.mean([stationary_bootstrap_se(daily, draws=2000) for daily in coins.values()])
    pooled = stationary_bootstrap_se(equal_weight(coins), draws=2000)
    assert single / pooled == pytest.approx(np.sqrt(2), rel=0.2)


def test_paired_difference_uses_shared_days():
    rng = np.random.default_rng(SEED)
    daily_a = pd.Series(rng.standard_normal(60), index=day_index(60))
    daily_b = daily_a.iloc[10:] - 0.1 + 0.01 * rng.standard_normal(50)
    result = paired_ic_difference(daily_a, daily_b)
    assert result["days"] == 50
    assert result["ic"] == pytest.approx(0.1, abs=0.01)


def test_pooled_difference_averages_each_coins_paired_days():
    rng = np.random.default_rng(SEED)
    days = day_index(60)
    model_a = {coin: pd.Series(rng.standard_normal(60), index=days) for coin in ("doge", "aave")}
    model_b = {"doge": model_a["doge"] - 0.1, "aave": model_a["aave"].iloc[30:] - 0.3}
    result = pooled_ic_difference(model_a, model_b)
    assert result["days"] == 60
    assert result["ic"] == pytest.approx((30 * 0.1 + 30 * 0.2) / 60)


def test_equal_weight_of_identical_coins_is_one_coin():
    rng = np.random.default_rng(SEED)
    daily = pd.DataFrame(rng.normal(0, 0.01, (30, 2)), index=day_index(30), columns=["net", "fees"])
    combined = equal_weight({"doge": daily, "aave": daily})
    pd.testing.assert_frame_equal(combined, daily, check_freq=False)
    pd.testing.assert_series_equal(
        equal_weight({"doge": daily["net"], "aave": daily["net"]}), daily["net"], check_freq=False
    )


def test_hit_rate_ignores_flat_sides_and_zero_moves():
    side = np.array([1, -1, 1, 0, -1])
    realised = np.array([0.02, 0.01, 0.0, 0.03, -0.01])
    assert hit_rate(side, realised) == pytest.approx(2 / 3)


def test_deciles_are_formed_within_each_month():
    month = pd.Series(np.repeat(["2025-12", "2026-01"], 20))
    pred = pd.Series(np.r_[np.arange(20.0), np.arange(20.0) + 100])
    table = decile_returns(pred, pred, month)
    expected = [np.arange(10) * 2 + 0.5, np.arange(10) * 2 + 100.5]
    np.testing.assert_allclose(table.to_numpy(), expected)
