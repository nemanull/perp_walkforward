import functools
from pathlib import Path

import numpy as np
import pandas as pd

from walkforward.config import CRASH, DATA_DIR, RESEARCH_MONTHS, RESULTS_DIR, TRAIN_START
from walkforward.data.bars import BAR
from walkforward.folds import within

FIRST_WINDOW = (TRAIN_START, RESEARCH_MONTHS[0])


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def between(rows: pd.Series | pd.DataFrame, start: str, end: str) -> pd.Series | pd.DataFrame:
    return rows.loc[utc(start) : utc(end) - BAR]


def in_crash(index: pd.DatetimeIndex) -> np.ndarray:
    return within(index, utc(CRASH[0]), utc(CRASH[1]))


@functools.cache
def load_bars() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "bars.parquet")


def output_dir(experiment: str) -> Path:
    path = RESULTS_DIR / experiment
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_table(table: pd.DataFrame, path: Path, digits: int = 6) -> None:
    floats = table.select_dtypes("float").columns
    table.round(dict.fromkeys(floats, digits)).to_csv(path, index=False)
