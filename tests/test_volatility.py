import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import bars_between, cross_join, utc
from scipy.signal import lfilter

from src.config import FEES_BPS, GRID_START, HORIZONS, SEED, TARGETS, TRAIN_START
from src.data.bars import BAR
from src.data.features import MODEL_INPUTS
from src.experiments.common import PERIODS, daily_ic
from src.experiments.volatility import (
    HAR_INPUTS,
    HAR_WINDOWS,
    LIGHTGBM_INPUTS,
    MODELS,
    SIZE_CAP,
    VERSIONS,
    choose_models,
    edge_covers_fee,
    fit_har,
    forward_log_rv,
    inverse_volatility_size,
    least_squares_har,
    predict_volatility,
    r2_against_training_mean,
    trailing_log_rv,
    tranche_scale,
    version_pnl,
    volatility_frame,
)
from src.folds import monthly_folds, scored_rows, split_fold
from src.plots import render

MONTHS = ("2025-08", "2025-09")
END = utc("2025-10-01")
NO_FUNDING = pd.Series(dtype=float)
COINS = list(TARGETS)


def clustered_volatility_close(index: pd.DatetimeIndex, seed: int = SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_volatility = lfilter([0.03], [1.0, -0.999], rng.standard_normal(len(index)))
    returns = 0.003 * np.exp(log_volatility) * rng.standard_normal(len(index))
    return pd.Series(100 * np.exp(np.cumsum(returns)), index=index)


def synthetic_features(seed: int = SEED) -> pd.DataFrame:
    index = bars_between(GRID_START, "2025-10-02")
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((len(index), len(MODEL_INPUTS)))
    inputs = pd.DataFrame(noise, index=index, columns=MODEL_INPUTS)
    inputs[index < utc(TRAIN_START)] = np.nan
    return inputs.assign(x_close=clustered_volatility_close(index, seed))


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return synthetic_features()


def test_label_is_the_log_of_the_next_h_squared_returns():
    close = pd.Series(
        [100.0, 101.0, 99.0, 99.5, 102.0, 101.0],
        index=bars_between("2025-12-01", "2025-12-01 00:30"),
    )
    returns = np.log(close).diff()
    label = forward_log_rv(close, 3)
    assert label.iloc[0] == pytest.approx(np.log(np.sqrt((returns.iloc[1:4] ** 2).sum())))
    assert label.iloc[-3:].isna().all()
    assert trailing_log_rv(close, 3).iloc[3] == pytest.approx(label.iloc[0])


def test_no_label_across_gap_or_flat_window():
    rng = np.random.default_rng(SEED)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.003, 60))))
    close.iloc[20] = np.nan
    close.iloc[40:46] = close.iloc[39]
    label = forward_log_rv(close, 4)
    assert label.iloc[16:21].isna().all()
    assert label.iloc[[15, 21]].notna().all()
    assert label.iloc[39:42].isna().all()
    assert label.iloc[[38, 42]].notna().all()
    assert not np.isinf(label).any()


@pytest.mark.parametrize("window", HAR_WINDOWS)
def test_har_input_includes_bar_t_and_ignores_every_later_bar(window):
    close = clustered_volatility_close(bars_between("2025-06-01", "2025-06-04"))
    cut = 700
    later = close.copy()
    later.iloc[cut + 1 :] *= np.exp(
        np.random.default_rng(SEED).normal(0, 0.05, len(close) - cut - 1)
    )
    moved = close.copy()
    moved.iloc[cut] *= 1.01
    original = trailing_log_rv(close, window)
    pd.testing.assert_series_equal(
        trailing_log_rv(later, window).iloc[: cut + 1], original.iloc[: cut + 1]
    )
    assert trailing_log_rv(moved, window).iloc[cut] != original.iloc[cut]


def test_har_recovers_planted_coefficients():
    rng = np.random.default_rng(SEED)
    rows = 20_000
    level = lfilter([0.05], [1.0, -0.99], rng.standard_normal(rows))
    frame = pd.DataFrame(
        {name: -5 + level + 0.2 * rng.standard_normal(rows) for name in HAR_INPUTS}
    )
    slopes = np.array([0.45, 0.3, 0.15])
    frame["y_12"] = -0.6 + frame[HAR_INPUTS].to_numpy() @ slopes + 0.1 * rng.standard_normal(rows)
    train, valid = frame.iloc[:15_000], frame.iloc[15_000:]
    model, valid_pred, setting = fit_har(train, valid, LIGHTGBM_INPUTS, "y_12", 12)
    np.testing.assert_allclose(model.slopes, slopes, atol=0.02)
    assert model.intercept == pytest.approx(-0.6, abs=0.1)
    assert setting == "slopes=" + ",".join(f"{slope:.2f}" for slope in model.slopes)
    np.testing.assert_allclose(valid_pred, least_squares_har(train, "y_12").predict(valid))


def test_both_volatility_models_score_the_same_rows(features):
    damaged = features.copy()
    no_input, no_bar = utc("2025-08-10"), utc("2025-08-20")
    damaged.loc[no_input, "btc_logret_1"] = np.nan
    damaged.loc[no_bar, "x_close"] = np.nan
    frame = volatility_frame(damaged, 12)
    har, lightgbm = (
        scored_rows(frame, 12, inputs, END) for inputs in (HAR_INPUTS, LIGHTGBM_INPUTS)
    )
    pd.testing.assert_series_equal(har, lightgbm)
    assert not har[no_input]
    assert not har[no_bar - 12 * BAR]
    assert (frame["hour"] == frame.index.hour).all()


def test_walk_forward_forecasts_clustered_volatility(features):
    out = predict_volatility(features, 12, "har", MONTHS, END)
    assert out["test_month"].unique().tolist() == list(MONTHS)
    assert daily_ic(out).mean() > 0.3
    assert r2_against_training_mean(out) > 0.1
    [fold] = monthly_folds(["2025-09"])
    refit = split_fold(volatility_frame(features, 12), fold, 12, HAR_INPUTS, END).refit
    september = out[out["test_month"] == "2025-09"]
    assert september["train_mean"].unique() == pytest.approx([refit["y_12"].mean()])


def strategy_frames(
    valid_ic: dict[str, float], sigma: float | np.ndarray, median_sigma: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    index = bars_between("2025-12-01", "2026-04-01")
    month = pd.Series(index.strftime("%Y-%m"), index=index)
    rng = np.random.default_rng(SEED)
    levels = {"q10": -1.28, "q30": -0.52, "q50": 0.0, "q70": 0.52, "q90": 1.28}
    direction = pd.DataFrame(
        {"pred": rng.standard_normal(len(index)), "test_month": month, **levels}, index=index
    ).assign(valid_ic=month.map(valid_ic))
    volatility = pd.DataFrame({"pred": np.log(sigma), "q50": np.log(median_sigma)}, index=index)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.003, len(index)))), index=index)
    return direction, volatility, close


def test_gate_shut_without_positive_valid_ic():
    ics = {"2025-12": 0.05, "2026-01": 0.0, "2026-02": -0.03, "2026-03": np.nan}
    direction, volatility, _ = strategy_frames(ics, sigma=0.01, median_sigma=0.01)
    scale = tranche_scale(direction, volatility, "outer10", "maker", "gated")
    by_month = scale.groupby(direction["test_month"]).agg(["min", "max"])
    assert by_month.loc["2025-12"].tolist() == [1.0, 1.0]
    assert (by_month.drop("2025-12") == 0.0).all().all()


def test_gate_needs_edge_above_round_trip():
    valid_ic, sigma_hat = pd.Series([0.05]), pd.Series([0.01])
    # 1.755 * 0.05 * 100 bps is 8.8 bps: above a maker round trip, below a taker one
    assert edge_covers_fee(valid_ic, sigma_hat, "outer10", 2.0).item()
    assert not edge_covers_fee(valid_ic, sigma_hat, "outer10", 5.0).item()
    assert not edge_covers_fee(valid_ic, sigma_hat / 10, "outer10", 2.0).item()


def test_inverse_volatility_size():
    sigma_hat = np.random.default_rng(SEED).lognormal(np.log(0.005), 1.0, 10_000)
    size = inverse_volatility_size(pd.DataFrame({"pred": np.log(sigma_hat), "q50": np.log(0.005)}))
    assert size.max() == SIZE_CAP
    below = size < SIZE_CAP
    assert below.mean() > 0.5
    np.testing.assert_allclose(size[below], 0.005 / sigma_hat[below])


def test_sizing_scales_each_tranche_before_the_tranche_mean():
    ics = dict.fromkeys(("2025-12", "2026-01", "2026-02", "2026-03"), 0.5)
    direction, volatility, close = strategy_frames(ics, sigma=0.01, median_sigma=0.01)
    position = {
        version: version_pnl(
            direction, volatility, close, NO_FUNDING, 12, "median", "taker", version
        )["position"]
        for version in VERSIONS
    }
    np.testing.assert_allclose(position["gated"], position["base"])
    np.testing.assert_allclose(position["sized"], position["base"])
    calm = volatility.assign(pred=np.log(0.004))
    doubled = version_pnl(direction, calm, close, NO_FUNDING, 12, "median", "taker", "sized")
    np.testing.assert_allclose(doubled["position"], 2 * position["base"])


def test_sized_position_never_exceeds_twice_a_full_position():
    ics = dict.fromkeys(("2025-12", "2026-01", "2026-02", "2026-03"), 0.5)
    direction, volatility, close = strategy_frames(ics, sigma=0.01, median_sigma=0.01)
    sigma = np.random.default_rng(SEED).lognormal(np.log(0.004), 1.5, len(volatility))
    volatility["pred"] = np.log(sigma)
    direction["pred"] = np.where(direction.index.day < 15, 2.0, -2.0)
    pnl = version_pnl(direction, volatility, close, NO_FUNDING, 12, "outer10", "maker", "sized")
    assert pnl["scale"].max() == SIZE_CAP
    assert pnl["position"].abs().max() <= SIZE_CAP
    assert pnl["position"].abs().max() > 1.5


def test_lightgbm_replaces_har_only_with_a_pooled_t_of_two():
    gains = cross_join(horizon=(4, 12), period=PERIODS, coin=["hype", "pooled"])
    gains["gain"] = 0.01
    gains["t"] = [9.0, 2.0, 9.0, 9.0, 9.0, 1.99, 9.0, 9.0]
    choice = choose_models(gains)
    assert {horizon: c["model"] for horizon, c in choice.items()} == {"4": "lightgbm", "12": "har"}
    assert choice["12"]["t"] == 1.99


def write_volatility(folder: Path, rng: np.random.Generator) -> None:
    table = cross_join(coin=COINS, horizon=HORIZONS, model=MODELS, period=PERIODS)
    table["ic"] = rng.uniform(0.4, 0.8, len(table))
    table["se"] = rng.uniform(0.01, 0.02, len(table))
    table["t"] = table["ic"] / table["se"]
    table["days"] = 182
    table["oos_r2"] = rng.uniform(0.1, 0.5, len(table))
    table.to_csv(folder / "volatility.csv", index=False)
    gain = cross_join(horizon=HORIZONS, period=PERIODS, coin=[*COINS, "pooled"])
    gain["gain"] = rng.normal(0.01, 0.01, len(gain))
    gain["se"] = rng.uniform(0.003, 0.008, len(gain))
    gain["t"] = gain["gain"] / gain["se"]
    gain["sigma_bps"] = np.where(gain["coin"] == "pooled", np.nan, rng.uniform(20, 280, len(gain)))
    gain.to_csv(folder / "gain.csv", index=False)
    months = pd.period_range("2025-12", "2026-08", freq="M").strftime("%Y-%m")
    by_month = cross_join(coin=COINS, horizon=HORIZONS, model=MODELS, month=months)
    by_month["period"] = np.where(by_month["month"] < "2026-06", "research", "forward")
    by_month["ic"] = rng.uniform(0.3, 0.8, len(by_month))
    by_month.to_csv(folder / "ic_by_month.csv", index=False)


def write_volatility_strategy(folder: Path, rng: np.random.Generator) -> None:
    stats = ("gross_bps_day", "net_bps_day", "max_drawdown", "turnover_day", "breakeven_bps")
    strategy = cross_join(coin=COINS, version=VERSIONS, fee=FEES_BPS, period=PERIODS)
    portfolio = cross_join(version=VERSIONS, fee=FEES_BPS, period=PERIODS)
    for table in (strategy, portfolio):
        table["sharpe"] = rng.normal(-3, 2, len(table))
        table["sharpe_low"] = table["sharpe"] - 2.3
        table["sharpe_high"] = table["sharpe"] + 2.3
        for column in stats:
            table[column] = rng.normal(0, 1, len(table))
    strategy["gate_share"] = rng.uniform(0, 1, len(strategy))
    strategy["mean_size"] = rng.uniform(0.8, 1.6, len(strategy))
    strategy.to_csv(folder / "strategy.csv", index=False)
    portfolio.to_csv(folder / "portfolio.csv", index=False)
    months = ("2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06")
    gate = cross_join(coin=COINS, fee=FEES_BPS, month=months)
    gate["period"] = np.where(gate["month"] < "2026-06", "research", "forward")
    gate["valid_ic"] = rng.normal(0.03, 0.03, len(gate))
    gate.loc[0, "valid_ic"] = np.nan
    gate["gate_share"] = np.where(gate["valid_ic"] > 0, rng.uniform(0, 1, len(gate)), 0.0)
    gate["mean_size"] = rng.uniform(0.8, 1.6, len(gate))
    gate.to_csv(folder / "gate.csv", index=False)
    days = pd.date_range("2025-12-01", "2026-09-01", freq="D", inclusive="left", tz="UTC")
    curves = pd.DataFrame({"day": days})
    for version in VERSIONS:
        for name in [*COINS, "portfolio"]:
            curves[f"{name}_{version}"] = np.cumsum(rng.normal(-1e-3, 0.005, len(days)))
    curves.to_csv(folder / "equity.csv", index=False)
    choice = {"horizon": 4, "rule": "outer10", "volatility_model": "har", "version": "gated"}
    (folder / "volatility-strategy.json").write_text(json.dumps(choice))


@pytest.mark.parametrize(
    ("experiment", "write", "figures"),
    [
        (
            "volatility",
            write_volatility,
            ("volatility_ic", "lightgbm_gain", "gain_vs_volatility", "ic_by_month"),
        ),
        ("volatility-strategy", write_volatility_strategy, ("sharpe", "gate", "equity")),
    ],
)
def test_every_volatility_figure_is_drawn(tmp_path, experiment, write, figures):
    folder = tmp_path / experiment
    folder.mkdir()
    write(folder, np.random.default_rng(SEED))
    render(experiment, tmp_path)
    for name in figures:
        assert (folder / f"{name}.png").stat().st_size > 0
