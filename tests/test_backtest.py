import warnings

import numpy as np
import pandas as pd
import pytest
from conftest import bar_index

from src.backtest import (
    bar_pnl,
    breakeven_fee_bps,
    buy_and_hold,
    long_share,
    max_drawdown,
    sharpe,
    sharpe_interval,
    signals,
    tranche_position,
)
from src.config import SEED
from src.data.bars import BARS_PER_DAY

NO_FUNDING = pd.Series(dtype=float)


def flat_close(bars: int) -> pd.Series:
    return pd.Series(100.0, index=bar_index(bars))


def test_tranche_position_and_turnover():
    position = tranche_position([1, 1, -1, 0, 1, 1, 1], horizon=3)
    np.testing.assert_allclose(position, np.array([1, 2, 1, 0, 0, 2, 3]) / 3)
    turnover = bar_pnl(position, flat_close(7), 5.0, NO_FUNDING)["turnover"]
    np.testing.assert_allclose(turnover, np.array([1, 1, 1, 1, 0, 2, 1 + 3]) / 3)


def test_delay_shifts_the_position_by_one_bar():
    signal = [1, 1, -1, 0, 1, 1, 1]
    expected = np.r_[0.0, tranche_position(signal, horizon=3)[:-1]]
    np.testing.assert_allclose(tranche_position(signal, horizon=3, delay=1), expected)


def test_undefined_prediction_gives_a_flat_signal():
    side = signals([0.3, np.nan, -0.3, 0.05], q_lo=-0.1, q_hi=0.1)
    np.testing.assert_array_equal(side, [1, 0, -1, 0])


def test_median_rule_goes_flat_on_a_tie():
    median = np.array([0.0, 0.0, 0.2, 0.2])
    side = signals([-0.2, 0.0, 0.1, 0.3], q_lo=median, q_hi=median)
    np.testing.assert_array_equal(side, [-1, 0, -1, 1])


def test_signal_gap_does_not_spread_nan():
    position = tranche_position([1, np.nan, 1, 1], horizon=2)
    np.testing.assert_allclose(position, [0.5, 0.5, 0.5, 1.0])


def test_long_position_pays_a_positive_funding_rate():
    funding = pd.Series([3e-4], index=bar_index(4)[[2]])
    pnl = bar_pnl(np.ones(4), flat_close(4), 0.0, funding)
    np.testing.assert_allclose(pnl["net"], [0.0, 0.0, -3e-4, 0.0])


def test_breakeven_fee_makes_net_profit_zero_with_funding():
    rng = np.random.default_rng(SEED)
    index = bar_index(30 * BARS_PER_DAY)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 3e-3, len(index)))), index=index)
    position = tranche_position(rng.integers(0, 2, len(index)), horizon=12)
    funding = pd.Series(1e-4, index=index[::96])
    pnl = bar_pnl(position, close, 0.0, funding)
    fee = breakeven_fee_bps(pnl)
    assert fee < pnl["gross"].sum() / pnl["turnover"].sum() * 1e4
    assert bar_pnl(position, close, fee, funding)["net"].sum() == pytest.approx(0.0, abs=1e-12)


def test_halted_bar_books_its_move_on_the_first_bar_after():
    close = pd.Series([100.0, np.nan, np.nan, 110.0, 110.0], index=bar_index(5))
    pnl = bar_pnl(np.ones(5), close, 0.0, NO_FUNDING)
    np.testing.assert_allclose(pnl["gross"], [0.0, 0.0, 0.0, 0.1, 0.0])


def test_final_position_is_closed_and_pays_its_exit_fee():
    pnl = bar_pnl([0.0, 1.0, 1.0, 1.0], flat_close(4), 5.0, NO_FUNDING)
    np.testing.assert_allclose(pnl["fees"], [0.0, 5e-4, 0.0, 5e-4])


def test_loss_on_the_first_day_counts_as_drawdown():
    assert max_drawdown(pd.Series([-0.02, 0.01, -0.03, 0.05])) == pytest.approx(0.04)


def test_buy_and_hold_compounds_bars_into_daily_returns():
    index = bar_index(4, start="2025-12-01 23:50")
    close = pd.Series([100.0, 102.0, np.nan, 99.0], index=index)
    np.testing.assert_allclose(buy_and_hold(close), [0.02, 99 / 102 - 1])


def test_sharpe_interval_width():
    daily = pd.Series(np.random.default_rng(SEED).normal(0, 0.01, 182))
    low, high = sharpe_interval(daily)
    assert (high - low) / 2 == pytest.approx(2.3, rel=0.2)


def test_sparse_trading_still_gets_a_sharpe_interval():
    daily = pd.Series(0.0, index=range(92))
    daily[[3, 20, 21, 40, 41, 70, 88]] = np.random.default_rng(SEED).normal(1e-3, 5e-4, 7)
    low, high = sharpe_interval(daily)
    assert np.isfinite([low, high]).all()
    assert low < high


def test_no_trades_gives_nan_without_warnings():
    pnl = bar_pnl(np.zeros(BARS_PER_DAY * 3), flat_close(BARS_PER_DAY * 3), 5.0, NO_FUNDING)
    daily = pnl.resample("D").sum()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(sharpe(daily["net"]))
        assert np.isnan(sharpe_interval(daily["net"])).all()
        assert np.isnan(breakeven_fee_bps(pnl))
        assert np.isnan(long_share(np.zeros(5)))
