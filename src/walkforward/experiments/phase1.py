import logging
import itertools

import numpy as np
import pandas as pd

from walkforward import backtest, config, metrics, models
from walkforward.data import bars
from walkforward.experiments import common

log = logging.getLogger(__name__)

FAMILIES = tuple(models.FITTERS)
TOP_DECILE_EDGE = 1.755


def five_minute_log_returns(start: str, end: str) -> pd.DataFrame:
    close = common.load_bars()[[f"{asset}_close" for asset in config.PAIRS]]
    returns = np.log(close).diff().set_axis(list(config.PAIRS), axis=1)
    return common.between(returns, start, end)


def needed_ic(coin: str, horizon: int, fee: str = "taker") -> float:
    sigma = common.realised_sigma(coin, horizon)
    return 2 * config.FEES_BPS[fee] / (TOP_DECILE_EDGE * sigma * 1e4)


def strategy_pnl(
    frame: pd.DataFrame,
    coin: str,
    horizon: int,
    rule: str = "median",
    fee: str = "taker",
    delay: int = 0,
    period: str = "research",
) -> pd.DataFrame:
    close = common.period_close(coin, period)
    signal = common.rule_signal(frame.reindex(close.index), rule)
    position = backtest.tranche_position(signal, horizon, delay)
    pnl = backtest.bar_pnl(position, close, config.FEES_BPS[fee], common.load_funding()[coin])
    return pnl.assign(position=position)


def extreme_bars_table() -> pd.DataFrame:
    tables = []
    for asset in config.PAIRS:
        close = common.load_bars()[f"{asset}_close"]
        close = common.between(close, config.GRID_START, config.RESEARCH_END)
        moves = np.log(close).diff()[bars.is_extreme_return(close)]
        tables.append(pd.DataFrame({"asset": asset, "open_time": moves.index, "log_return": moves}))
    return pd.concat(tables)


def rolling_btc_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    window, filled = 30 * bars.BARS_PER_DAY, 20 * bars.BARS_PER_DAY
    rolling = returns[list(config.TARGETS)].rolling(window, min_periods=filled)
    daily = rolling.corr(returns["btc"]).resample("1D").last().dropna(how="all")
    return daily.rename_axis("day").reset_index()


def lead_lag_table(returns: pd.DataFrame) -> pd.DataFrame:
    calm = returns.copy()
    calm.loc[common.in_crash(calm.index)] = np.nan
    lags = range(-6, 7)
    table = pd.DataFrame(
        {
            coin: [calm[coin].corr(calm["btc"].shift(lag)) for lag in lags]
            for coin in config.TARGETS
        },
        index=pd.Index(lags, name="btc_lead_bars"),
    )
    return table.reset_index()


def autocorrelation_table(returns: pd.DataFrame) -> pd.DataFrame:
    calm = returns[~common.in_crash(returns.index)]
    assets = list(config.PAIRS)
    return pd.DataFrame(
        {
            "asset": assets,
            "lag1": [returns[asset].autocorr() for asset in assets],
            "lag1_without_crash": [calm[asset].autocorr() for asset in assets],
            "lag1_rank_without_crash": [calm[asset].rank().autocorr() for asset in assets],
        }
    )


def daily_volatility_table(returns: pd.DataFrame) -> pd.DataFrame:
    volatility = returns[list(config.TARGETS)].resample("1D").std() * np.sqrt(bars.BARS_PER_DAY)
    return volatility.dropna(how="all").rename_axis("day").reset_index()


def run_audit() -> None:
    out = common.output_dir("audit")
    # research rows only
    end = common.utc(config.RESEARCH_END) - bars.BAR
    frames = {asset: bars.read_klines(asset).loc[:end] for asset in config.PAIRS}
    seams = bars.find_seams(frames)
    common.write_table(bars.audit_bars(frames), out / "bars.csv")
    common.write_table(seams, out / "seams.csv")
    late = seams[seams["first_bar_after"] >= common.utc(config.TRAIN_START)]
    if len(late):
        raise RuntimeError(f"price seams inside the study window:\n{late}")

    returns = five_minute_log_returns(config.GRID_START, config.RESEARCH_END)
    first = common.between(returns, *common.FIRST_WINDOW)
    correlation = returns.corr().rename_axis("asset").reset_index()
    common.write_table(extreme_bars_table(), out / "extreme_bars.csv")
    common.write_table(correlation, out / "return_correlation.csv", digits=4)
    rolling = rolling_btc_correlation(returns)
    common.write_table(rolling, out / "rolling_btc_correlation.csv", digits=4)
    common.write_table(lead_lag_table(first), out / "lead_lag.csv", digits=4)
    common.write_table(autocorrelation_table(first), out / "autocorrelation.csv", digits=4)
    common.write_table(daily_volatility_table(returns), out / "daily_volatility.csv", digits=5)
    log.info("audit written, %d seams", len(seams))


def evaluate(frame: pd.DataFrame, coin: str, horizon: int) -> dict:
    daily = common.daily_ic(frame)
    pnl = strategy_pnl(frame, coin, horizon)
    return metrics.ic_summary(daily) | {
        "hit_rate": metrics.hit_rate(common.rule_signal(frame, "median"), frame["fwd_logret"]),
        "oos_r2": metrics.oos_r2(frame["pred"], frame["y"]),
        "gross": pnl["gross"].sum(),
        "funding": pnl["funding"].sum(),
        "turnover": pnl["turnover"].sum(),
        "breakeven_bps": backtest.breakeven_fee_bps(pnl),
        "daily_ic": daily,
    }


def pooled_table(configurations: pd.DataFrame, daily: dict) -> pd.DataFrame:
    rows = []
    for (horizon, family), recipe in configurations.groupby(["horizon", "family"], sort=False):
        summary = metrics.pooled_ic_summary(common.by_coin(daily, horizon, family))
        edge = (recipe["gross"] - recipe["funding"]).sum() / recipe["turnover"].sum() * 1e4
        rows.append(
            {"horizon": horizon, "family": family}
            | summary
            | {"breakeven_bps": edge, "detected": summary["t"] >= config.POOLED_T}
        )
    return pd.DataFrame(rows)


def choose_recipe(pooled: pd.DataFrame) -> pd.Series:
    detected = pooled[pooled["detected"]]
    if len(detected):
        return detected.loc[detected["breakeven_bps"].idxmax()]
    return pooled.loc[pooled["t"].idxmax()]


def best_per_coin(configurations: pd.DataFrame) -> dict:
    best = configurations.loc[configurations.groupby("coin")["t"].idxmax()]
    return {
        row.coin: {"horizon": int(row.horizon), "family": row.family, "t": float(row.t)}
        for row in best.itertuples()
    }


def detectability_table(pooled: pd.DataFrame) -> pd.DataFrame:
    days = (common.utc(config.RESEARCH_END) - common.utc(config.RESEARCH_MONTHS[0])).days
    table = pd.DataFrame({"horizon": list(config.HORIZONS)})
    noise = np.sqrt(table["horizon"] / bars.BARS_PER_DAY)
    table["detectable_ic"] = config.POOLED_T * noise / np.sqrt(days)
    table["best_pooled_ic"] = table["horizon"].map(pooled.groupby("horizon")["ic"].max())
    for coin in config.TARGETS:
        table[f"needed_ic_{coin}"] = [needed_ic(coin, horizon) for horizon in table["horizon"]]
    return table


def settings_table(frames: dict) -> pd.DataFrame:
    settings = {
        key: frame.groupby("test_month")["setting"].first() for key, frame in frames.items()
    }
    return pd.concat(settings, names=["coin", "horizon", "family", "month"]).reset_index()


def deciles_table(frames: dict) -> pd.DataFrame:
    deciles = {
        key: metrics.decile_returns(frame["pred"], frame["fwd_logret"], frame["test_month"]).stack()
        for key, frame in frames.items()
    }
    table = pd.concat(deciles, names=["coin", "horizon", "family"]).rename("mean_return")
    return table.reset_index()


def agreement_table(frames: dict) -> pd.DataFrame:
    rows = []
    for coin, horizon in itertools.product(config.TARGETS, config.HORIZONS):
        ridge, lightgbm = frames[(coin, horizon, "ridge")], frames[(coin, horizon, "lightgbm")]
        signals = common.centred_signal(ridge), common.centred_signal(lightgbm)
        spearman = metrics.within_month_spearman(*signals, ridge["test_month"])
        rows.append({"coin": coin, "horizon": horizon, "ridge_lightgbm_spearman": spearman})
    return pd.DataFrame(rows)


def prediction_correlation(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    month = pd.concat([frame["test_month"] for frame in frames.values()]).groupby(level=0).first()
    signal = {coin: common.centred_signal(frame) for coin, frame in frames.items()}
    matrix = pd.DataFrame(
        {
            a: {b: metrics.within_month_spearman(signal[a], signal[b], month) for b in frames}
            for a in frames
        }
    )
    return matrix.rename_axis("coin").reset_index()


def run_horizon_sweep() -> None:
    out = common.output_dir("horizon-sweep")
    keys = list(itertools.product(config.TARGETS, config.HORIZONS, FAMILIES))
    frames = {key: common.predictions(*key) for key in keys}
    results = {(c, h, f): evaluate(frame, c, h) for (c, h, f), frame in frames.items()}
    daily = {key: result.pop("daily_ic") for key, result in results.items()}
    configurations = pd.DataFrame(
        [{"coin": c, "horizon": h, "family": f} | result for (c, h, f), result in results.items()]
    )
    configurations["detected"] = configurations["t"] >= config.PER_COIN_T
    pooled = pooled_table(configurations, daily)
    print(pooled.round(3).to_string())
    chosen = choose_recipe(pooled)
    horizon, family = int(chosen["horizon"]), str(chosen["family"])
    recipe_daily = pd.DataFrame({coin: daily[(coin, horizon, family)] for coin in config.TARGETS})
    recipe_frames = {coin: frames[(coin, horizon, family)] for coin in config.TARGETS}

    common.write_table(configurations, out / "configurations.csv")
    common.write_table(pooled, out / "pooled.csv")
    common.write_table(detectability_table(pooled), out / "detectability.csv")
    by_month = common.ic_by_month_table(daily, ["coin", "horizon", "family"])
    common.write_table(by_month, out / "ic_by_month.csv")
    common.write_table(settings_table(frames), out / "settings.csv")
    common.write_table(deciles_table(frames), out / "deciles.csv", digits=8)
    common.write_table(agreement_table(frames), out / "model_agreement.csv", digits=4)
    ic_correlation = recipe_daily.corr().rename_axis("coin").reset_index()
    common.write_table(ic_correlation, out / "daily_ic_correlation.csv", digits=4)
    common.write_table(
        prediction_correlation(recipe_frames), out / "prediction_correlation.csv", digits=4
    )
    detected = bool(pooled["detected"].any())
    selection = {
        "horizon": horizon,
        "family": family,
        "detected": detected,
        "pooled_t": float(chosen["t"]),
        "per_coin_best": best_per_coin(configurations),
    }
    common.write_json(out / "selection.json", selection)
    log.info("frozen recipe H=%d %s, detected=%s", horizon, family, detected)


EXPERIMENTS = {
    "audit": run_audit,
    "horizon-sweep": run_horizon_sweep,
}
