import pandas as pd


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")
