import itertools

import pandas as pd

from walkforward.data.bars import BAR


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def bar_index(bars: int, start: str = "2025-12-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=bars, freq=BAR, tz="UTC", name="open_time")


def bars_between(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq=BAR, inclusive="left", tz="UTC", name="open_time")


def day_index(days: int, start: str = "2025-12-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=days, freq="D", tz="UTC", name="day")


def cross_join(**levels) -> pd.DataFrame:
    return pd.DataFrame(list(itertools.product(*levels.values())), columns=list(levels))
