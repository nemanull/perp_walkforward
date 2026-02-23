import numpy as np
import pandas as pd

from features import FIELDS, asset_features, cross_features, rolling_std, targets, zscore


def fake_bars(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {}
    for p in ["x", "btc", "eth", "sol"]:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
        cols[f"{p}_open"] = np.r_[close[0], close[:-1]]
        cols[f"{p}_high"] = close * 1.001
        cols[f"{p}_low"] = close * 0.999
        cols[f"{p}_close"] = close
        cols[f"{p}_volume"] = rng.uniform(100, 200, n)
        cols[f"{p}_quote_volume"] = cols[f"{p}_volume"] * close
        cols[f"{p}_count"] = rng.integers(10, 100, n)
        cols[f"{p}_taker_buy_volume"] = cols[f"{p}_volume"] * 0.5
        cols[f"{p}_taker_buy_quote_volume"] = cols[f"{p}_quote_volume"] * 0.5
    return pd.DataFrame(cols)


def test_rolling_std_is_population():
    s = pd.Series([1.0, 2.0, 4.0, 8.0])
    assert np.isclose(rolling_std(s, 4).iloc[-1], np.std([1, 2, 4, 8]))


def test_zscore_of_constant_is_zero():
    s = pd.Series(np.full(600, 3.0))
    assert (zscore(s, 576).dropna() == 0).all()


def test_asset_features_columns():
    feats = asset_features(fake_bars(), "btc")
    assert feats.shape[1] == 25
    assert all(c.startswith("btc_") for c in feats.columns)


def test_trail_ret_is_sum_of_logrets():
    feats = asset_features(fake_bars(), "x")
    summed = feats["x_logret_1"].rolling(12).sum()
    assert np.allclose(feats["x_trail_ret_12"].iloc[20:], summed.iloc[20:])


def test_beta_and_corr_with_itself():
    df = fake_bars()
    for f in FIELDS:
        df[f"btc_{f}"] = df[f"x_{f}"]
    cross = cross_features(df)
    assert np.allclose(cross["beta_x_btc_288"].dropna(), 1.0)
    assert np.allclose(cross["corr_x_btc_logret_72"].dropna(), 1.0)
    assert np.allclose(cross["x_minus_btc_trail_ret_12"].dropna(), 0.0)


def test_forward_return():
    df = fake_bars()
    t = targets(df, horizons=(4,))
    c = df["x_close"]
    assert np.isclose(t["x_fwd_logret_4"].iloc[10], np.log(c.iloc[14] / c.iloc[10]))
    assert t["x_fwd_logret_4"].iloc[-4:].isna().all()
