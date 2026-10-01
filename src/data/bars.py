import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR, FIRST_BAR, GRID_START, PAIRS, TARGETS
from src.data.download import funding_path, kline_path

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]
BAR_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
)
BAR = pd.Timedelta(minutes=5)
BARS_PER_DAY = pd.Timedelta(days=1) // BAR
SEAM_RATIO = 1.3
EXTREME_SIGMAS = 10
EXTREME_WINDOW = 576

log = logging.getLogger(__name__)


def monthly_files(pattern: Path) -> list[Path]:
    return sorted(pattern.parent.glob(pattern.name))


def has_header(path: Path) -> bool:
    with path.open() as csv:
        return not csv.readline().split(",", 1)[0].isdigit()


def read_kline_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=0 if has_header(path) else None, names=KLINE_COLUMNS)


def read_klines(asset: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    paths = monthly_files(kline_path(PAIRS[asset], "*", data_dir))
    klines = pd.concat(map(read_kline_csv, paths), ignore_index=True).drop_duplicates("open_time")
    open_time = pd.to_datetime(klines["open_time"], unit="ms", utc=True)
    bars = klines[list(BAR_FIELDS)].astype("float64").set_index(open_time).sort_index()
    return bars.loc[FIRST_BAR.get(asset) :]


def read_funding(asset: str, data_dir: Path = DATA_DIR) -> pd.Series:
    paths = monthly_files(funding_path(PAIRS[asset], "*", data_dir))
    funding = pd.concat(map(pd.read_csv, paths), ignore_index=True)
    settled = pd.to_datetime(funding["calc_time"], unit="ms", utc=True).dt.floor(BAR)
    rates = funding["last_funding_rate"].set_axis(settled.rename("open_time"))
    return rates[~rates.index.duplicated()].sort_index()


def drop_halts(bars: pd.DataFrame) -> pd.DataFrame:
    return bars[bars["count"] > 0]


def align_on_grid(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    end = min(bars.index.max() for bars in frames.values())
    grid = pd.date_range(pd.Timestamp(GRID_START, tz="UTC"), end, freq=BAR, name="open_time")
    aligned = [
        drop_halts(bars).reindex(grid).add_prefix(f"{asset}_") for asset, bars in frames.items()
    ]
    return pd.concat(aligned, axis=1)


def asset_seams(bars: pd.DataFrame, asset: str) -> pd.DataFrame:
    close = drop_halts(bars)["close"]
    before, after = close.index[:-1], close.index[1:]
    bars_between = (after - before) // BAR - 1
    ratio = close.to_numpy()[1:] / close.to_numpy()[:-1]
    seam = (bars_between > 0) & ((ratio > SEAM_RATIO) | (ratio < 1 / SEAM_RATIO))
    return pd.DataFrame(
        {
            "asset": asset,
            "last_bar_before": before[seam],
            "first_bar_after": after[seam],
            "bars_between": bars_between[seam],
            "price_ratio": ratio[seam],
        }
    )


def find_seams(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(
        [asset_seams(bars, asset) for asset, bars in frames.items()], ignore_index=True
    )


def is_extreme_return(close: pd.Series) -> pd.Series:
    # the window counts observed returns, so a gap does not blind it for two days
    logret = np.log(close).diff().dropna()
    trailing_sd = logret.rolling(EXTREME_WINDOW).std().shift()
    return (logret.abs() > EXTREME_SIGMAS * trailing_sd).reindex(close.index, fill_value=False)


def audit_bars(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    bars = align_on_grid(frames)
    month = bars.index.strftime("%Y-%m").rename("month")
    counts = {}
    for asset, raw in frames.items():
        trades = raw["count"].reindex(bars.index)
        flags = pd.DataFrame(
            {
                "expected": True,
                "present": trades.notna(),
                "missing": trades.isna(),
                "zero_trade": trades.eq(0),
                "extreme": is_extreme_return(bars[f"{asset}_close"]),
            }
        )
        counts[asset] = flags.groupby(month).sum()
    return pd.concat(counts, names=["asset", "month"]).reset_index()


def build(data_dir: Path = DATA_DIR) -> None:
    frames = {asset: read_klines(asset, data_dir) for asset in PAIRS}
    bars = align_on_grid(frames)
    bars.to_parquet(data_dir / "bars.parquet")
    funding = pd.concat(
        {asset: read_funding(asset, data_dir) for asset in TARGETS}, axis=1, sort=True
    )
    funding.to_parquet(data_dir / "funding.parquet")
    log.info(
        "wrote %d bars from %s to %s and %d funding times",
        len(bars),
        bars.index[0],
        bars.index[-1],
        len(funding),
    )

    audit = audit_bars(frames)
    totals = audit.groupby("asset", sort=False)[["missing", "zero_trade", "extreme"]].sum()
    seams = find_seams(frames)["asset"].value_counts()
    for asset, total in totals.iterrows():
        log.info(
            "%s: missing=%d zero_trade=%d extreme=%d seams=%d",
            asset,
            total["missing"],
            total["zero_trade"],
            total["extreme"],
            seams.get(asset, 0),
        )
