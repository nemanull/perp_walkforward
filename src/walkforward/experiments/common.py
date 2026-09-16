import functools
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from walkforward.backtest import signals
from walkforward.config import (
    CRASH,
    DATA_DIR,
    EARLY_STOPPING_ROUNDS,
    FORWARD_END,
    FORWARD_MONTHS,
    LIGHTGBM_PARAMS,
    RESEARCH_END,
    RESEARCH_MONTHS,
    RESULTS_DIR,
    RIDGE_ALPHAS,
    ROWS_PER_LEAF_PER_HORIZON_BAR,
    SEED,
    TARGETS,
    TRAIN_START,
)
from walkforward.data.bars import BAR
from walkforward.data.features import MODEL_INPUTS, OWN_INPUTS
from walkforward.folds import predict_out_of_sample, within
from walkforward.metrics import daily_rank_ic, ic_by_month
from walkforward.models import FITTERS

log = logging.getLogger(__name__)

PERIODS = {"research": (RESEARCH_MONTHS, RESEARCH_END), "forward": (FORWARD_MONTHS, FORWARD_END)}
FIRST_WINDOW = (TRAIN_START, RESEARCH_MONTHS[0])
INPUT_SETS = {"all": MODEL_INPUTS, "own": OWN_INPUTS}
THRESHOLDS = {"median": ("q50", "q50"), "outer30": ("q30", "q70"), "outer10": ("q10", "q90")}
ONE_INPUT_FAMILIES = ("momentum", "best_feature")


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def between(rows: pd.Series | pd.DataFrame, start: str, end: str) -> pd.Series | pd.DataFrame:
    return rows.loc[utc(start) : utc(end) - BAR]


def in_crash(index: pd.DatetimeIndex) -> np.ndarray:
    return within(index, utc(CRASH[0]), utc(CRASH[1]))


@functools.cache
def load_features(coin: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "features" / f"{coin}.parquet")


@functools.cache
def load_bars() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "bars.parquet")


@functools.cache
def load_funding() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "funding.parquet")


def period_close(coin: str, period: str) -> pd.Series:
    months, end = PERIODS[period]
    return between(load_bars()[f"{coin}_close"], months[0], end)


def output_dir(experiment: str) -> Path:
    path = RESULTS_DIR / experiment
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_table(table: pd.DataFrame, path: Path, digits: int = 6) -> None:
    floats = table.select_dtypes("float").columns
    table.round(dict.fromkeys(floats, digits)).to_csv(path, index=False)


def rounded(value: object, digits: int = 6) -> object:
    if isinstance(value, dict):
        return {key: rounded(item, digits) for key, item in value.items()}
    return round(value, digits) if isinstance(value, float) else value


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(rounded(payload), indent=2) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


@functools.cache
def code_tag() -> str:
    src = Path(__file__).resolve().parents[1]
    names = ("folds.py", "models.py", "data/features.py")
    sources = b"".join((src / name).read_bytes() for name in names)
    settings = (
        LIGHTGBM_PARAMS,
        RIDGE_ALPHAS,
        ROWS_PER_LEAF_PER_HORIZON_BAR,
        EARLY_STOPPING_ROUNDS,
        TRAIN_START,
        SEED,
    )
    return hashlib.sha1(sources + repr(settings).encode()).hexdigest()[:8]


def predictions(
    coin: str,
    horizon: int,
    family: str,
    inputs: str = "all",
    policy: str = "expanding",
    period: str = "research",
) -> pd.DataFrame:
    name = f"{coin}_{horizon}_{family}_{inputs}_{policy}_{period}.parquet"
    # cached per code version, see code_tag
    path = DATA_DIR / "predictions" / code_tag() / name
    if path.exists():
        return pd.read_parquet(path)
    months = PERIODS[period][0]
    log.info("predicting %s H=%d %s %s %s %s", coin, horizon, family, inputs, policy, period)
    frame = predict_out_of_sample(
        load_features(coin), horizon, FITTERS[family], INPUT_SETS[inputs], policy, months
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return frame


def fitted_family(family: str) -> str:
    return "ridge" if family in ONE_INPUT_FAMILIES else family


def centred_signal(frame: pd.DataFrame, column: str = "pred") -> pd.Series:
    # every month has its own model, so a value is read on its validation-month scale
    scale = frame["q90"] - frame["q10"]
    return ((frame[column] - frame["q50"]) / scale).where(scale > 0)


def rule_signal(frame: pd.DataFrame, rule: str) -> np.ndarray:
    levels = ("pred", *THRESHOLDS[rule])
    pred, low, high = (centred_signal(frame, column).to_numpy() for column in levels)
    return signals(pred, low, high)


def daily_ic(frame: pd.DataFrame) -> pd.Series:
    days = frame.index.floor("D").unique()
    ranked = daily_rank_ic(centred_signal(frame), frame["fwd_logret"])
    return ranked.reindex(days, fill_value=0.0)


def by_coin(results: dict[tuple, pd.Series], *key) -> dict[str, pd.Series]:
    return {coin: results[(coin, *key)] for coin in TARGETS}


def ic_by_month_table(daily: dict[tuple, pd.Series], names: list[str]) -> pd.DataFrame:
    by_month = {key: ic_by_month(series) for key, series in daily.items()}
    return pd.concat(by_month, names=names).rename("ic").reset_index()


def realised_sigma(coin: str, horizon: int) -> float:
    log_close = np.log(between(load_bars()[f"{coin}_close"], *FIRST_WINDOW))
    forward = log_close.shift(-horizon) - log_close
    label_end = forward.index + horizon * BAR
    touches_crash = (label_end >= utc(CRASH[0])) & (forward.index < utc(CRASH[1]))
    return float(forward[~touches_crash].std())
