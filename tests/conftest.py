import pandas as pd

from walkforward.data.bars import BAR


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def bar_index(bars: int, start: str = "2025-12-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=bars, freq=BAR, tz="UTC", name="open_time")
