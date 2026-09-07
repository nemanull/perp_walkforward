import math

import numpy as np
import pandas as pd
import pytest
from conftest import bar_index

from walkforward.config import CONTEXT, HORIZONS, SEED, TARGETS
from walkforward.data.bars import BAR_FIELDS
from walkforward.data.features import (
    EPS,
    FEATURES,
    MODEL_INPUTS,
    build,
    build_features,
    labels,
    rolling_cov,
    rolling_mean,
    rolling_std,
    rolling_zscore,
)


def synthetic_bars(assets, n: int = 2000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    columns = {}
    for asset in assets:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
        open_ = np.concatenate([[close[0]], close[:-1]])
        wicks = close * rng.uniform(0, 0.002, (2, n))
        volume = rng.lognormal(8, 1, n)
        buy_share = rng.uniform(0.3, 0.7, n)
        fields = {
            "open": open_,
            "high": np.maximum(open_, close) + wicks[0],
            "low": np.minimum(open_, close) - wicks[1],
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "count": rng.poisson(400, n).astype(float),
            "taker_buy_volume": buy_share * volume,
            "taker_buy_quote_volume": buy_share * volume * close,
        }
        columns |= {f"{asset}_{field}": values for field, values in fields.items()}
    return pd.DataFrame(columns, index=bar_index(n, start="2025-06-01"))


def noise_after(bars: pd.DataFrame, cut: pd.Timestamp, assets) -> pd.DataFrame:
    noise = 3 * synthetic_bars(assets, n=len(bars), seed=SEED + 1)
    return pd.concat([bars.loc[:cut], noise.loc[noise.index > cut]])


def bars_missing_one_hype_bar(position: int) -> pd.DataFrame:
    bars = synthetic_bars(("hype", *CONTEXT))
    bars.loc[bars.index[position], [f"hype_{field}" for field in BAR_FIELDS]] = np.nan
    return bars


def random_series_with_gaps() -> tuple[pd.Series, pd.Series]:
    values = np.random.default_rng(SEED).normal(1.0, 2.0, (2, 300))
    values[0, [10, 11, 150, 299]] = np.nan
    values[1, [40, 220]] = np.nan
    return pd.Series(values[0]), pd.Series(values[1])


def loop_over_windows(series, window, statistic) -> list[float]:
    result = []
    for end in range(len(series[0])):
        chunks = [list(values[end - window + 1 : end + 1]) for values in series]
        complete = end >= window - 1 and not any(math.isnan(v) for c in chunks for v in c)
        result.append(statistic(*chunks) if complete else math.nan)
    return result


def mean(values):
    return sum(values) / len(values)


def population_std(values):
    centre = mean(values)
    return math.sqrt(sum((v - centre) ** 2 for v in values) / len(values))


def population_cov(a, b):
    centre_a, centre_b = mean(a), mean(b)
    return sum((u - centre_a) * (v - centre_b) for u, v in zip(a, b, strict=True)) / len(a)


def zscore_of_last(values):
    return (values[-1] - mean(values)) / (population_std(values) + EPS)


@pytest.mark.parametrize(
    ("rolling", "statistic"),
    [(rolling_mean, mean), (rolling_std, population_std), (rolling_zscore, zscore_of_last)],
)
def test_rolling_statistic_matches_a_plain_loop(rolling, statistic):
    values, _ = random_series_with_gaps()
    expected = loop_over_windows([values], 20, statistic)
    np.testing.assert_allclose(rolling(values, 20), expected, rtol=1e-12, atol=1e-12)


def test_rolling_cov_matches_a_plain_loop():
    a, b = random_series_with_gaps()
    expected = loop_over_windows([a, b], 20, population_cov)
    np.testing.assert_allclose(rolling_cov(a, b, 20), expected, rtol=1e-12, atol=1e-12)


def test_features_do_not_look_ahead():
    assets = ("hype", *CONTEXT)
    bars = synthetic_bars(assets)
    cut = bars.index[1500]
    original = build_features(bars, "hype").loc[:cut, FEATURES]
    rewritten = build_features(noise_after(bars, cut, assets), "hype").loc[:cut, FEATURES]
    assert original.loc[cut].notna().all()
    pd.testing.assert_frame_equal(rewritten, original, check_exact=True)


def test_forward_log_return():
    close = [100.0, 101.0, 99.5, 102.0, 103.0, 101.5, 104.0, 105.0]
    bars = pd.DataFrame({"x_close": close}, index=bar_index(8, start="2025-06-01"))
    forward = labels(bars, horizons=(4,))["fwd_logret_4"]
    assert forward.iloc[2] == pytest.approx(math.log(104.0 / 99.5), rel=1e-12)
    assert forward.iloc[:4].notna().all()
    assert forward.iloc[-4:].isna().all()


def test_sigma_uses_past_returns_only():
    bars = synthetic_bars(["x"], n=1200)
    row = 900
    cut = bars.index[row]
    sigmas = [f"sigma_{horizon}" for horizon in HORIZONS]
    scales = labels(bars).loc[:cut, sigmas]
    returns = np.diff(np.log(bars["x_close"].to_numpy()))
    expected = np.std(returns[row - 576 : row]) * math.sqrt(12)
    assert scales.loc[cut, "sigma_12"] == pytest.approx(expected, rel=1e-9)
    rewritten = labels(noise_after(bars, cut, ["x"])).loc[:cut, sigmas]
    pd.testing.assert_frame_equal(rewritten, scales, check_exact=True)


@pytest.mark.parametrize(
    ("column", "bars_voided"),
    [
        ("x_logret_1", 2),
        ("x_roll_mean_logret_12", 13),
        ("x_roll_vol_logret_576", 577),
        ("x_volume_z_576", 576),
        ("x_taker_buy_ratio_vol_ma36", 36),
        ("corr_x_btc_logret_72", 73),
        ("beta_x_sol_288", 289),
    ],
)
def test_a_missing_bar_voids_every_window_that_touches_it(column, bars_voided):
    missing = 1000
    feature = build_features(bars_missing_one_hype_bar(missing), "hype")[column]
    around = feature.iloc[missing - 1 : missing + bars_voided + 1]
    assert around.isna().tolist() == [False, *[True] * bars_voided, False]


def test_trailing_return_across_gap():
    missing = 1000
    bars = bars_missing_one_hype_bar(missing)
    trail = build_features(bars, "hype")["x_trail_ret_12"]
    close = bars["hype_close"]
    expected = math.log(close.iloc[missing + 5] / close.iloc[missing - 7])
    assert trail.iloc[missing + 5] == pytest.approx(expected, rel=1e-9)
    assert trail.iloc[missing + 1 : missing + 12].notna().all()
    assert trail.iloc[[missing, missing + 12]].isna().all()


def test_targets_share_columns():
    bars = synthetic_bars(("hype", "doge", *CONTEXT))
    hype, doge = build_features(bars, "hype"), build_features(bars, "doge")
    label_columns = [f"{name}_{h}" for h in HORIZONS for name in ("fwd_logret", "sigma", "y")]
    expected = ["x_close", *FEATURES, *label_columns]
    assert list(hype.columns) == list(doge.columns) == expected
    assert set(MODEL_INPUTS) <= set(hype.columns)
    pd.testing.assert_index_equal(hype.index, bars.index)
    pd.testing.assert_series_equal(hype["x_close"], bars["hype_close"], check_names=False)
    pd.testing.assert_series_equal(doge["x_close"], bars["doge_close"], check_names=False)
    pd.testing.assert_frame_equal(hype.filter(regex="^btc_"), doge.filter(regex="^btc_"))


def test_roll_mean_logret_is_trail_ret_over_w():
    features = build_features(synthetic_bars(("hype", *CONTEXT)), "hype")
    for window in (12, 36, 72):
        np.testing.assert_allclose(
            features[f"x_roll_mean_logret_{window}"],
            features[f"x_trail_ret_{window}"] / window,
            rtol=1e-9,
            atol=1e-13,
        )


def test_feature_and_input_counts():
    assert len(FEATURES) == 111
    assert len(MODEL_INPUTS) == 75


def test_build_writes_one_feature_file_per_target(tmp_path):
    bars = synthetic_bars((*TARGETS, *CONTEXT), n=700)
    bars.to_parquet(tmp_path / "bars.parquet")
    build(tmp_path)
    for coin in TARGETS:
        written = pd.read_parquet(tmp_path / "features" / f"{coin}.parquet")
        pd.testing.assert_frame_equal(written, build_features(bars, coin), check_freq=False)
