from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12

DATA = Path("Datasets")
CONTEXT = {
    "btc": DATA / "BTC" / "5MIN",
    "eth": DATA / "ETH" / "5MIN",
    "sol": DATA / "SOL" / "5MIN",
}
FIELDS = [
    "open", "high", "low", "close", "volume", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume",
]


def load_folder(folder: Path) -> pd.DataFrame:
    files = sorted(folder.rglob("*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.sort_values("close_time").reset_index(drop=True)


def load_aligned(symbol: str = "HYPE") -> pd.DataFrame:
    # same as alignCandlesByCloseTime in the ts code: keep the x bars
    # where btc, eth and sol all have a bar with the same close_time
    x = load_folder(DATA / "TOKEN_X" / symbol / "5MIN")
    df = x[["open_time", "close_time"] + FIELDS].rename(columns={f: f"x_{f}" for f in FIELDS})
    for name, folder in CONTEXT.items():
        other = load_folder(folder)[["close_time"] + FIELDS]
        other = other.rename(columns={f: f"{name}_{f}" for f in FIELDS})
        df = df.merge(other, on="close_time", how="inner")
    return df


def rolling_std(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).std()


def zscore(s: pd.Series, n: int) -> pd.Series:
    return (s - s.rolling(n).mean()) / (rolling_std(s, n) + EPS)


def asset_features(df: pd.DataFrame, p: str) -> pd.DataFrame:
    o, h, l, c = (df[f"{p}_{f}"] for f in ["open", "high", "low", "close"])
    logret = np.log(c / (c.shift(1) + EPS))
    tbr_vol = df[f"{p}_taker_buy_volume"] / (df[f"{p}_volume"] + EPS)
    tbr_qv = df[f"{p}_taker_buy_quote_volume"] / (df[f"{p}_quote_volume"] + EPS)

    out = pd.DataFrame(index=df.index)
    out["logret_1"] = logret
    for k in [4, 12, 36, 72]:
        out[f"trail_ret_{k}"] = np.log(c / (c.shift(k) + EPS))
    for k in [12, 36, 72]:
        out[f"roll_mean_logret_{k}"] = logret.rolling(k).mean()
    out["roll_vol_logret_576"] = rolling_std(logret, 576)
    out["candle_range_pct"] = (h - l) / (c + EPS)
    out["candle_body_pct"] = (c - o) / (c + EPS)
    out["upper_wick_pct"] = (h - np.maximum(o, c)) / (c + EPS)
    out["lower_wick_pct"] = (np.minimum(o, c) - l) / (c + EPS)

    for f in ["volume", "quote_volume", "count"]:
        out[f"{f}_log"] = np.log1p(df[f"{p}_{f}"])
    for f in ["volume", "quote_volume", "count"]:
        out[f"{f}_z_576"] = zscore(df[f"{p}_{f}"], 576)

    out["taker_buy_ratio_vol"] = tbr_vol
    out["taker_buy_ratio_qv"] = tbr_qv
    for k in [12, 36]:
        out[f"taker_buy_ratio_vol_ma{k}"] = tbr_vol.rolling(k).mean()
        out[f"taker_buy_ratio_qv_ma{k}"] = tbr_qv.rolling(k).mean()
    return out.add_prefix(f"{p}_")


if __name__ == "__main__":
    df = load_aligned("HYPE")
    print(df.shape)
    feats = pd.concat([asset_features(df, p) for p in ["btc", "eth", "sol", "x"]], axis=1)
    print(feats.shape)
    print(feats.describe().T)
