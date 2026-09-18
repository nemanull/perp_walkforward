import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from conftest import cross_join

from walkforward.config import FEES_BPS, HORIZONS, PAIRS, RESEARCH_MONTHS, SEED, TARGETS
from walkforward.data.bars import BARS_PER_DAY
from walkforward.experiments.common import THRESHOLDS
from walkforward.experiments.phase1 import DELAYS, FAMILIES, POLICIES
from walkforward.plots import STYLE, render

COINS = list(TARGETS)
FIGURES = {
    "audit": (
        "return_correlation",
        "rolling_btc_correlation",
        "lead_lag",
        "autocorrelation",
        "daily_volatility",
    ),
    "horizon-sweep": (
        "ic_heatmap",
        "pooled_ic",
        "ic_by_month",
        "deciles",
        "model_agreement",
        "prediction_correlation",
        "daily_ic_correlation",
        "detectability",
    ),
    "feature-sources": ("sources", "gain_vs_btc", "importance"),
    "retraining": ("retraining", "decay"),
    "economics": ("breakeven", "pnl_correlation", "equity", "needed_ic"),
}


def correlation_matrix(rng: np.random.Generator, names: list[str], key: str) -> pd.DataFrame:
    loadings = rng.uniform(0.4, 0.95, len(names))
    matrix = np.outer(loadings, loadings)
    np.fill_diagonal(matrix, 1.0)
    return pd.DataFrame(matrix, columns=names).round(4).assign(**{key: names})[[key, *names]]


def daily(
    rng: np.random.Generator, days: int, start: str, level: float, step: float
) -> pd.DataFrame:
    walk = level + np.cumsum(rng.normal(0, step, (days, len(COINS))), axis=0)
    table = pd.DataFrame(walk, columns=COINS)
    table.insert(0, "day", pd.date_range(start, periods=days, freq="D", tz="UTC"))
    return table


def ic_columns(rng: np.random.Generator, frame: pd.DataFrame) -> pd.DataFrame:
    frame["ic"] = rng.normal(0.01, 0.02, len(frame))
    frame["se"] = rng.uniform(0.005, 0.015, len(frame))
    frame["t"] = frame["ic"] / frame["se"]
    return frame


def write_audit(folder: Path, rng: np.random.Generator) -> None:
    correlation_matrix(rng, list(PAIRS), "asset").to_csv(
        folder / "return_correlation.csv", index=False
    )
    rolling = daily(rng, 300, "2025-06-21", 0.6, 0.01)
    rolling[COINS] = rolling[COINS].clip(0, 1)
    rolling.to_csv(folder / "rolling_btc_correlation.csv", index=False)
    lead_lag = pd.DataFrame(rng.normal(0, 0.01, (13, len(COINS))), columns=COINS)
    lead_lag.insert(0, "btc_lead_bars", range(-6, 7))
    lead_lag.loc[6, COINS] = rng.uniform(0.4, 0.8, len(COINS))
    lead_lag.to_csv(folder / "lead_lag.csv", index=False)
    autocorrelation = pd.DataFrame({"asset": list(PAIRS)})
    for column in ("lag1", "lag1_without_crash", "lag1_rank_without_crash"):
        autocorrelation[column] = rng.normal(-0.03, 0.02, len(PAIRS))
    autocorrelation.to_csv(folder / "autocorrelation.csv", index=False)
    volatility = daily(rng, 300, "2025-06-01", np.log(0.04), 0.05)
    volatility[COINS] = np.exp(volatility[COINS])
    volatility.loc[131, COINS] *= 4
    volatility.to_csv(folder / "daily_volatility.csv", index=False)


def write_horizon_sweep(folder: Path, rng: np.random.Generator) -> None:
    configurations = ic_columns(rng, cross_join(coin=COINS, horizon=HORIZONS, family=FAMILIES))
    for column in ("days", "hit_rate", "oos_r2", "gross", "funding", "turnover"):
        configurations[column] = rng.uniform(0, 1, len(configurations))
    configurations["breakeven_bps"] = rng.normal(1, 2, len(configurations))
    configurations["detected"] = configurations["t"] >= 3.2
    configurations.to_csv(folder / "configurations.csv", index=False)
    pooled = ic_columns(rng, cross_join(horizon=HORIZONS, family=FAMILIES))
    pooled["days"] = 182
    pooled["breakeven_bps"] = rng.normal(1, 2, len(pooled))
    pooled["detected"] = pooled["t"] >= 3
    pooled.to_csv(folder / "pooled.csv", index=False)
    by_month = cross_join(coin=COINS, horizon=HORIZONS, family=FAMILIES, month=RESEARCH_MONTHS)
    by_month["ic"] = rng.normal(0.01, 0.03, len(by_month))
    by_month.to_csv(folder / "ic_by_month.csv", index=False)
    deciles = cross_join(
        coin=COINS, horizon=HORIZONS, family=FAMILIES, month=RESEARCH_MONTHS, decile=range(1, 11)
    )
    deciles["mean_return"] = 2e-5 * (deciles["decile"] - 5.5) + rng.normal(0, 1e-4, len(deciles))
    deciles.to_csv(folder / "deciles.csv", index=False)
    agreement = cross_join(coin=COINS, horizon=HORIZONS)
    agreement["ridge_lightgbm_spearman"] = rng.uniform(0.2, 0.8, len(agreement))
    agreement.to_csv(folder / "model_agreement.csv", index=False)
    for name in ("prediction_correlation", "daily_ic_correlation"):
        correlation_matrix(rng, COINS, "coin").to_csv(folder / f"{name}.csv", index=False)
    horizons = np.array(HORIZONS)
    detectability = pd.DataFrame(
        {"horizon": horizons, "detectable_ic": 3 * np.sqrt(horizons / BARS_PER_DAY) / np.sqrt(182)}
    )
    detectability["best_pooled_ic"] = rng.normal(0.02, 0.01, len(horizons))
    for coin, sigma in zip(COINS, (36, 12, 30, 40, 34), strict=True):
        detectability[f"needed_ic_{coin}"] = 10 / (1.755 * sigma * np.sqrt(horizons))
    detectability.to_csv(folder / "detectability.csv", index=False)
    selection = {"horizon": 12, "family": "ridge", "detected": False, "pooled_t": 2.4}
    selection["per_coin_best"] = {c: {"horizon": 36, "family": "lightgbm", "t": 2.0} for c in COINS}
    (folder / "selection.json").write_text(json.dumps(selection))


def write_feature_sources(folder: Path, rng: np.random.Generator) -> None:
    sources = pd.DataFrame({"coin": COINS, "ic_all": rng.normal(0.02, 0.01, len(COINS))})
    sources["ic_own"] = sources["ic_all"] + rng.normal(0.003, 0.004, len(COINS))
    sources["gain"] = sources["ic_all"] - sources["ic_own"]
    sources["gain_t"] = sources["gain"] / 0.003
    sources["btc_correlation"] = rng.uniform(0.4, 0.9, len(COINS))
    sources.to_csv(folder / "sources.csv", index=False)
    importance = cross_join(coin=COINS, group=("x", "btc", "eth", "sol"))
    importance["ic_drop"] = rng.normal(0.004, 0.004, len(importance))
    importance.to_csv(folder / "importance.csv", index=False)
    (folder / "sources.json").write_text(json.dumps({"family": "ridge", "inputs": "all"}))


def write_retraining(folder: Path, rng: np.random.Generator) -> None:
    retraining = ic_columns(rng, cross_join(coin=COINS, policy=POLICIES))
    retraining.to_csv(folder / "retraining.csv", index=False)
    by_month = cross_join(coin=COINS, policy=POLICIES, month=RESEARCH_MONTHS)
    by_month["research_month"] = by_month["month"].map(RESEARCH_MONTHS.index) + 1
    by_month["ic"] = rng.normal(0.02, 0.02, len(by_month)) - 0.003 * by_month["research_month"]
    by_month.to_csv(folder / "retraining_by_month.csv", index=False)


def write_economics(folder: Path, rng: np.random.Generator) -> None:
    economics = cross_join(coin=COINS, rule=THRESHOLDS, fee=FEES_BPS, delay=DELAYS)
    for column in (
        "gross_bps_day",
        "net_bps_day",
        "sharpe",
        "sharpe_low",
        "sharpe_high",
        "max_drawdown",
        "turnover_day",
        "beta",
        "buy_and_hold_bps_day",
        "time_in_market",
        "long_share",
    ):
        economics[column] = rng.normal(0, 1, len(economics))
    economics["breakeven_bps"] = rng.normal(3, 2, len(economics)) - 1.5 * economics["delay"]
    economics.to_csv(folder / "economics.csv", index=False)
    correlation_matrix(rng, COINS, "coin").to_csv(folder / "pnl_correlation.csv", index=False)
    curves = pd.DataFrame({"day": pd.date_range("2025-12-01", periods=182, freq="D", tz="UTC")})
    for coin in COINS:
        curves[f"{coin}_strategy"] = np.cumsum(rng.normal(2e-4, 0.006, 182))
        curves[f"{coin}_buy_and_hold"] = np.cumsum(rng.normal(0, 0.04, 182))
    curves["portfolio_strategy"] = curves.filter(like="_strategy").mean(axis=1)
    curves.to_csv(folder / "equity.csv", index=False)
    sigma = rng.uniform(30, 120, len(COINS))
    needed = pd.DataFrame({"coin": COINS, "sigma_bps": sigma})
    needed["needed_ic_taker"] = 10 / (1.755 * sigma)
    needed["needed_ic_maker"] = 4 / (1.755 * sigma)
    needed["research_ic"] = rng.normal(0.03, 0.02, len(COINS))
    needed.to_csv(folder / "needed_ic.csv", index=False)
    (folder / "economics.json").write_text(json.dumps({"rule": "outer30"}))


WRITERS = {
    "audit": write_audit,
    "horizon-sweep": write_horizon_sweep,
    "feature-sources": write_feature_sources,
    "retraining": write_retraining,
    "economics": write_economics,
}


@pytest.mark.parametrize("experiment", FIGURES)
def test_every_figure_is_drawn(tmp_path, experiment):
    folder = tmp_path / experiment
    folder.mkdir()
    WRITERS[experiment](folder, np.random.default_rng(SEED))
    style = {name: mpl.rcParams[name] for name in STYLE}
    render(experiment, tmp_path)
    for name in FIGURES[experiment]:
        assert (folder / f"{name}.png").stat().st_size > 0
    assert {name: mpl.rcParams[name] for name in STYLE} == style
