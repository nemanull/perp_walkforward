import pytest

from walkforward.data.download import (
    expected_sha256,
    funding_path,
    funding_url,
    kline_path,
    kline_url,
    month_range,
    months_for,
)


def test_month_range_is_inclusive_and_crosses_years():
    assert month_range("2025-11", "2026-02") == ["2025-11", "2025-12", "2026-01", "2026-02"]


def test_month_range_rejects_reversed_bounds():
    with pytest.raises(ValueError):
        month_range("2026-02", "2025-11")


def test_no_month_before_first_bar_is_requested():
    assert months_for("hype", "2025-03", "2025-06") == ["2025-05", "2025-06"]
    assert months_for("btc", "2025-03", "2025-04") == ["2025-03", "2025-04"]


def test_archive_urls_follow_the_binance_layout():
    assert kline_url("HYPEUSDT", "2025-06") == (
        "https://data.binance.vision/data/futures/um/monthly/klines/"
        "HYPEUSDT/5m/HYPEUSDT-5m-2025-06.zip"
    )
    assert funding_url("TRXUSDT", "2026-01").endswith(
        "fundingRate/TRXUSDT/TRXUSDT-fundingRate-2026-01.zip"
    )


def test_checksum_is_first_token():
    digest = "b6ff04d1555eb6a52e943538126997442bcd96a61ea4a7fbd8151c168ed084f7"
    assert expected_sha256(f"{digest}  HYPEUSDT-5m-2025-05.zip\n") == digest


def test_local_paths_mirror_the_archive_names(tmp_path):
    assert kline_path("HYPEUSDT", "2025-06", tmp_path) == (
        tmp_path / "raw" / "klines" / "HYPEUSDT" / "HYPEUSDT-5m-2025-06.csv"
    )
    assert funding_path("TRXUSDT", "2026-01", tmp_path).name == "TRXUSDT-fundingRate-2026-01.csv"
