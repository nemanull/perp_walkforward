from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import FIRST_BAR, GRID_START, PAIRS, SEED, TARGETS
from src.data.bars import (
    BAR,
    BAR_FIELDS,
    BARS_PER_DAY,
    KLINE_COLUMNS,
    align_on_grid,
    audit_bars,
    build,
    find_seams,
    read_funding,
    read_klines,
)
from src.data.download import funding_path, kline_path
from src.experiments.common import utc


def make_bars(start: str, periods: int, seed: int = SEED, price: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = price * np.exp(np.cumsum(rng.normal(0, 0.002, periods)))
    open_time = pd.date_range(utc(start), periods=periods, freq=BAR, unit="ms", name="open_time")
    # only close and count matter here
    return pd.DataFrame({field: close for field in BAR_FIELDS}, index=open_time).assign(count=50.0)


def write_klines(
    bars: pd.DataFrame, data_dir: Path, pair: str, month: str, header: bool = True
) -> None:
    path = kline_path(pair, month, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    millis = bars.index.as_unit("ms").asi8
    raw = bars.assign(open_time=millis, close_time=millis + 299_999, ignore=0)
    raw[KLINE_COLUMNS].to_csv(path, index=False, header=header)


def write_funding(
    settlements: pd.DatetimeIndex,
    data_dir: Path,
    pair: str,
    month: str,
    jitter_ms: int | list[int] = 0,
) -> None:
    path = funding_path(pair, month, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    funding = pd.DataFrame(
        {
            "calc_time": settlements.as_unit("ms").asi8 + np.asarray(jitter_ms),
            "funding_interval_hours": 4,
            "last_funding_rate": np.linspace(-1e-4, 1e-4, len(settlements)),
        }
    )
    funding.to_csv(path, index=False)


def test_a_file_with_and_a_file_without_header_both_load(tmp_path):
    bars = make_bars("2025-06-01", 20)
    write_klines(bars.iloc[:10], tmp_path, "BTCUSDT", "2025-06")
    write_klines(bars.iloc[10:], tmp_path, "BTCUSDT", "2025-07", header=False)
    pd.testing.assert_frame_equal(read_klines("btc", tmp_path), bars, check_freq=False)


def test_duplicate_open_times_are_dropped(tmp_path):
    bars = make_bars("2025-06-01", 20)
    write_klines(bars.iloc[:12], tmp_path, "ETHUSDT", "2025-06")
    write_klines(bars.iloc[8:], tmp_path, "ETHUSDT", "2025-07")
    pd.testing.assert_frame_equal(read_klines("eth", tmp_path), bars, check_freq=False)


def test_nothing_before_first_bar_survives(tmp_path):
    write_klines(make_bars("2025-05-30 09:30", 30), tmp_path, "HYPEUSDT", "2025-05")
    klines = read_klines("hype", tmp_path)
    assert klines.index[0] == utc(FIRST_BAR["hype"])
    assert len(klines) == 18


def test_a_missing_bar_stays_missing_on_the_grid():
    bars = make_bars(GRID_START, 10)
    aligned = align_on_grid({"btc": bars.drop(bars.index[4])})
    assert len(aligned) == 10
    assert aligned.iloc[4].isna().all()
    assert aligned.drop(aligned.index[4]).notna().all().all()


def test_a_zero_trade_bar_becomes_missing_in_every_field():
    bars = make_bars(GRID_START, 10)
    bars.loc[bars.index[4], "count"] = 0.0
    aligned = align_on_grid({"btc": bars})
    assert aligned.columns.tolist() == [f"btc_{field}" for field in BAR_FIELDS]
    assert aligned.iloc[4].isna().all()
    assert aligned.drop(aligned.index[4]).notna().all().all()


def test_grid_length_matches_the_expected_bar_count():
    frames = {
        "hype": make_bars(GRID_START, 150_000),
        "btc": make_bars("2025-05-01", 150_000).loc[:"2026-08-31 23:55"],
    }
    aligned = align_on_grid(frames)
    assert aligned.index[0] == utc(GRID_START)
    assert aligned.index[-1] == utc("2026-08-31 23:55")
    assert len(aligned) == 450 + 457 * BARS_PER_DAY


def test_jittered_funding_times_land_on_the_grid(tmp_path):
    settlements = pd.date_range(utc("2025-06-01"), periods=6, freq="4h")
    write_funding(settlements[:4], tmp_path, "HYPEUSDT", "2025-06", jitter_ms=[6, 0, 26, 3])
    write_funding(settlements[3:], tmp_path, "HYPEUSDT", "2025-07", jitter_ms=[9, 1, 2])
    assert read_funding("hype", tmp_path).index.tolist() == settlements.tolist()


def test_a_planted_splice_is_reported_as_a_seam():
    old = make_bars(GRID_START, 12)
    halted = make_bars("2025-05-30 11:30", 6).assign(count=0.0)
    new = make_bars("2025-05-30 12:15", 12, seed=SEED + 1, price=11.5)
    seams = find_seams({"pump": pd.concat([old, halted, new])})
    assert seams.to_dict("records") == [
        {
            "asset": "pump",
            "last_bar_before": utc("2025-05-30 11:25"),
            "first_bar_after": utc("2025-05-30 12:15"),
            "bars_between": 9,
            "price_ratio": pytest.approx(new["close"].iloc[0] / old["close"].iloc[-1]),
        }
    ]


def test_a_normal_series_with_a_gap_and_a_halt_has_no_seam():
    bars = make_bars(GRID_START, 600)
    bars.loc[bars.index[300:305], "count"] = 0.0
    assert find_seams({"btc": bars.drop(bars.index[100:110])}).empty


def test_audit_counts_per_month():
    bars = make_bars(GRID_START, 450 + 600)
    bars.loc[utc("2025-05-31 12:00"), "count"] = 0.0
    bars.loc[utc("2025-06-02 12:00") :, "close"] *= 1.2
    audit = audit_bars({"btc": bars.drop(utc("2025-06-02 00:00"))})
    assert audit.to_dict("records") == [
        {
            "asset": "btc",
            "month": "2025-05",
            "expected": 450,
            "present": 450,
            "missing": 0,
            "zero_trade": 1,
            "extreme": 0,
        },
        {
            "asset": "btc",
            "month": "2025-06",
            "expected": 600,
            "present": 599,
            "missing": 1,
            "zero_trade": 0,
            "extreme": 1,
        },
    ]


def test_build_writes_bars_and_funding(tmp_path):
    for offset, pair in enumerate(PAIRS.values()):
        write_klines(make_bars(GRID_START, 300, SEED + offset), tmp_path, pair, "2025-05")
    four_hourly = pd.date_range(utc("2025-05-30 12:00"), periods=6, freq="4h")
    for asset, pair in TARGETS.items():
        settlements = four_hourly.delete(1) if asset == "hype" else four_hourly[1::2]
        write_funding(settlements, tmp_path, pair, "2025-05")

    build(tmp_path)

    bars = pd.read_parquet(tmp_path / "bars.parquet")
    funding = pd.read_parquet(tmp_path / "funding.parquet")
    assert bars.shape == (300, len(PAIRS) * len(BAR_FIELDS))
    assert funding.columns.tolist() == list(TARGETS)
    assert funding.index.tolist() == four_hourly.tolist()
    assert funding["hype"].isna().tolist() == [False, True, False, False, False, False]
    assert funding["trx"].notna().tolist() == [False, True, False, True, False, True]
