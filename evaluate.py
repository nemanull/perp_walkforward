import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from features import FIELDS, build

COINS = ["HYPE", "DOGE"]
HORIZONS = [4, 12, 36, 72]
PARAMS = dict(
    n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=200,
    colsample_bytree=0.8, random_state=42, verbose=-1,
)


def purged_kfold(n: int, k: int, h: int):
    # contiguous folds, h rows before and after the test fold are dropped from train
    # so no training label overlaps the test labels
    edges = np.linspace(0, n, k + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        train = np.r_[0:max(a - h, 0), min(b + h, n):n]
        yield train, np.arange(a, b)


def daily_ic(pred: pd.Series, y: pd.Series, day: pd.Series) -> pd.Series:
    df = pd.DataFrame({"pred": pred, "y": y, "day": day}).dropna()
    return df.groupby("day").apply(lambda g: spearmanr(g["pred"], g["y"]).statistic)


def report(name: str, ic: pd.Series) -> None:
    t = ic.mean() / ic.std() * np.sqrt(len(ic))
    print(f"  {name:9s} ic {ic.mean():7.4f}  t {t:5.1f}  days {len(ic)}")


for coin in COINS:
    df = build(coin)
    df = df[df["x_roll_vol_logret_576"].notna()].reset_index(drop=True)
    day = pd.to_datetime(df["close_time"], unit="ms").dt.date
    targets = [c for c in df.columns if c.startswith(("x_fwd_logret_", "x_sigma_", "x_z_", "x_score_"))]
    raw = [f"{p}_{f}" for p in ["x", "btc", "eth", "sol"] for f in FIELDS]
    feats = [c for c in df.columns if c not in targets + raw + ["open_time", "close_time"]]
    print(coin, len(df), "rows", len(feats), "features")

    for h in HORIZONS:
        y = df[f"x_z_{h}"]  # forward log return / (576-bar vol * sqrt(h))
        pred = pd.Series(np.nan, index=df.index)
        for train, test in purged_kfold(len(df), 5, h):
            train = train[y.iloc[train].notna().to_numpy()]
            model = lgb.LGBMRegressor(**PARAMS)
            model.fit(df.loc[train, feats], y.iloc[train])
            pred.iloc[test] = model.predict(df.loc[test, feats])
        print(f" H={h}")
        report("lightgbm", daily_ic(pred, y, day))
        report("reversal", daily_ic(-df[f"x_trail_ret_{h}"], y, day))  # 6h??
