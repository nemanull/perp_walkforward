from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

from walkforward.config import PAIRS, SEED, TARGETS
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


WRITERS = {
    "audit": write_audit,
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
