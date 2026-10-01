import logging
import itertools
from collections.abc import Iterable

import numpy as np
import pandas as pd

from src import backtest, config, folds, metrics, models
from src.data import bars, features
from src.experiments import common

log = logging.getLogger(__name__)

FAMILIES = tuple(models.FITTERS)
POLICIES = ("expanding", "fixed", "rolling_3m")
DELAYS = (0, 1)
TOP_DECILE_EDGE = 1.755


def read_choice(experiment: str, name: str, key: str) -> str | int:
    return common.read_json(config.RESULTS_DIR / experiment / name)[key]


def frozen_recipe() -> tuple[int, str]:
    selection = common.read_json(config.RESULTS_DIR / "horizon-sweep" / "selection.json")
    return selection["horizon"], selection["family"]


def chosen_configuration() -> dict:
    horizon, family = frozen_recipe()
    return {
        "horizon": horizon,
        "family": family,
        "inputs": read_choice("feature-sources", "sources.json", "inputs"),
        "policy": read_choice("retraining", "retraining.json", "policy"),
        "rule": read_choice("economics", "economics.json", "rule"),
    }


def predictions_by_coin(
    horizon: int, family: str, inputs: str, policy: str, period: str = "research"
) -> dict[str, pd.DataFrame]:
    return {
        coin: common.predictions(coin, horizon, family, inputs, policy, period)
        for coin in config.TARGETS
    }


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


def coin_strategy(
    frame: pd.DataFrame, coin: str, horizon: int, rule: str, fee: str, delay: int, period: str
) -> tuple[dict, pd.DataFrame]:
    pnl = strategy_pnl(frame, coin, horizon, rule, fee, delay, period)
    return common.pnl_stats(pnl, common.period_close(coin, period))


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


def permutation_importance(
    coin: str, horizon: int, family: str, inputs: list[str]
) -> dict[str, float]:
    frame = common.load_features(coin)
    rng = np.random.default_rng(config.SEED)
    base, shuffled = [], {group: [] for group in features.IMPORTANCE_GROUPS}
    for fold in folds.monthly_folds(config.RESEARCH_MONTHS):
        rows = folds.split_fold(frame, fold, horizon, inputs)
        model, thresholds, _, _ = folds.fit_window(rows, horizon, models.FITTERS[family], inputs)
        test = rows.test
        scored = pd.DataFrame(
            {"fwd_logret": test[f"fwd_logret_{horizon}"], **thresholds}, index=test.index
        )
        base.append(scored.assign(pred=model.predict(test[inputs])))
        for group, columns in features.IMPORTANCE_GROUPS.items():
            permuted = test[inputs].copy()
            permuted[columns] = test[columns].to_numpy()[rng.permutation(len(test))]
            shuffled[group].append(scored.assign(pred=model.predict(permuted)))
    base_ic = common.daily_ic(pd.concat(base)).mean()
    return {
        group: base_ic - common.daily_ic(pd.concat(parts)).mean()
        for group, parts in shuffled.items()
    }


def btc_correlation(coin: str) -> float:
    close = common.between(common.load_bars()[[f"{coin}_close", "btc_close"]], *common.FIRST_WINDOW)
    daily = close.resample("1D").last().pct_change().dropna()
    return daily[f"{coin}_close"].corr(daily["btc_close"])


def sources_table(daily_all: dict, daily_own: dict) -> pd.DataFrame:
    table = pd.DataFrame({"coin": list(config.TARGETS)})
    table["ic_all"] = [daily_all[coin].mean() for coin in config.TARGETS]
    table["ic_own"] = [daily_own[coin].mean() for coin in config.TARGETS]
    gains = [metrics.paired_ic_difference(daily_all[c], daily_own[c]) for c in config.TARGETS]
    table["gain"] = [gain["ic"] for gain in gains]
    table["gain_t"] = [gain["t"] for gain in gains]
    table["btc_correlation"] = [btc_correlation(coin) for coin in config.TARGETS]
    return table


def importance_table(horizon: int, family: str) -> pd.DataFrame:
    drops = {
        coin: pd.Series(permutation_importance(coin, horizon, family, features.MODEL_INPUTS))
        for coin in config.TARGETS
    }
    return pd.concat(drops, names=["coin", "group"]).rename("ic_drop").reset_index()


def choose_inputs(own_minus_all: dict) -> str:
    return "own" if own_minus_all["t"] >= config.PAIRED_T else "all"


def run_feature_sources() -> None:
    out = common.output_dir("feature-sources")
    horizon, family = frozen_recipe()
    family = common.fitted_family(family)
    everything = predictions_by_coin(horizon, family, "all", "expanding")
    own = predictions_by_coin(horizon, family, "own", "expanding")
    daily_all, daily_own = {}, {}
    for coin in config.TARGETS:
        shared = everything[coin].index.intersection(own[coin].index)
        daily_all[coin] = common.daily_ic(everything[coin].loc[shared])
        daily_own[coin] = common.daily_ic(own[coin].loc[shared])
    own_minus_all = metrics.pooled_ic_difference(daily_own, daily_all)
    choice = {"family": family, "inputs": choose_inputs(own_minus_all)}
    choice |= {"own_minus_all": own_minus_all["ic"], "t": own_minus_all["t"]}

    common.write_table(sources_table(daily_all, daily_own), out / "sources.csv")
    common.write_table(importance_table(horizon, family), out / "importance.csv")
    common.write_json(out / "sources.json", choice)
    log.info("feature set %s, own minus all t=%.2f", choice["inputs"], choice["t"])


def retraining_table(daily: dict) -> pd.DataFrame:
    rows = []
    for (coin, policy), series in daily.items():
        summary = metrics.ic_summary(series)
        rows.append({"coin": coin, "policy": policy} | {k: summary[k] for k in ("ic", "se", "t")})
    return pd.DataFrame(rows)


def retraining_by_month(daily: dict) -> pd.DataFrame:
    table = common.ic_by_month_table(daily, ["coin", "policy"])
    position = {month: number for number, month in enumerate(config.RESEARCH_MONTHS, 1)}
    table.insert(3, "research_month", table["month"].map(position))
    return table


def choose_policy(challengers: dict[str, dict]) -> str:
    winners = {policy: s["t"] for policy, s in challengers.items() if s["t"] >= config.PAIRED_T}
    return max(winners, key=winners.get) if winners else "expanding"


def run_retraining() -> None:
    out = common.output_dir("retraining")
    horizon, family = frozen_recipe()
    inputs = read_choice("feature-sources", "sources.json", "inputs")
    daily = {
        (coin, policy): common.daily_ic(common.predictions(coin, horizon, family, inputs, policy))
        for coin, policy in itertools.product(config.TARGETS, POLICIES)
    }
    expanding = common.by_coin(daily, "expanding")
    challengers = {
        policy: metrics.pooled_ic_difference(common.by_coin(daily, policy), expanding)
        for policy in POLICIES[1:]
    }
    policy = choose_policy(challengers)

    common.write_table(retraining_table(daily), out / "retraining.csv")
    common.write_table(retraining_by_month(daily), out / "retraining_by_month.csv")
    common.write_json(
        out / "retraining.json",
        {"policy": policy} | {f"{p}_minus_expanding_t": s["t"] for p, s in challengers.items()},
    )
    log.info("retraining policy %s", policy)


def strategy_table(
    frames: dict[str, pd.DataFrame],
    horizon: int,
    rules: Iterable[str],
    delays: Iterable[int],
    period: str,
) -> tuple[pd.DataFrame, dict]:
    rows, daily = [], {}
    for coin, rule, fee, delay in itertools.product(frames, rules, config.FEES_BPS, delays):
        stats, daily[(coin, rule, fee, delay)] = coin_strategy(
            frames[coin], coin, horizon, rule, fee, delay, period
        )
        rows.append({"coin": coin, "rule": rule, "fee": fee, "delay": delay} | stats)
    return pd.DataFrame(rows), daily


def portfolio_table(daily: dict) -> pd.DataFrame:
    rows = []
    for rule, fee, delay in itertools.product(common.THRESHOLDS, config.FEES_BPS, DELAYS):
        combined = metrics.equal_weight(common.by_coin(daily, rule, fee, delay))
        rows.append({"rule": rule, "fee": fee, "delay": delay} | common.strategy_stats(combined))
    return pd.DataFrame(rows)


def choose_rule(portfolio: pd.DataFrame) -> str:
    taker = portfolio[(portfolio["fee"] == "taker") & (portfolio["delay"] == 0)]
    return str(taker.loc[taker["sharpe"].idxmax(), "rule"])


def equity_table(net: dict[str, pd.Series]) -> pd.DataFrame:
    curves = {}
    for coin, series in net.items():
        curves[f"{coin}_strategy"] = series.cumsum()
        hold = backtest.buy_and_hold(common.period_close(coin, "research"))
        curves[f"{coin}_buy_and_hold"] = hold.cumsum()
    curves["portfolio_strategy"] = metrics.equal_weight(net).cumsum()
    return pd.DataFrame(curves).rename_axis("day").reset_index()


def needed_ic_table(horizon: int, research_ic: dict[str, float]) -> pd.DataFrame:
    table = pd.DataFrame({"coin": list(config.TARGETS)})
    table["sigma_bps"] = [common.realised_sigma(coin, horizon) * 1e4 for coin in table["coin"]]
    for fee in config.FEES_BPS:
        table[f"needed_ic_{fee}"] = [needed_ic(coin, horizon, fee) for coin in table["coin"]]
    table["research_ic"] = table["coin"].map(research_ic)
    return table


def run_economics() -> None:
    out = common.output_dir("economics")
    horizon, family = frozen_recipe()
    inputs = read_choice("feature-sources", "sources.json", "inputs")
    policy = read_choice("retraining", "retraining.json", "policy")
    frames = predictions_by_coin(horizon, family, inputs, policy)
    economics, daily = strategy_table(frames, horizon, common.THRESHOLDS, DELAYS, "research")
    portfolio = portfolio_table(daily)
    rule = choose_rule(portfolio)
    net = {coin: daily[(coin, rule, "taker", 0)]["net"] for coin in config.TARGETS}
    research_ic = {coin: common.daily_ic(frame).mean() for coin, frame in frames.items()}

    common.write_table(economics, out / "economics.csv")
    common.write_table(portfolio, out / "portfolio.csv")
    correlation = pd.DataFrame(net).corr().rename_axis("coin").reset_index()
    common.write_table(correlation, out / "pnl_correlation.csv", digits=4)
    common.write_table(equity_table(net), out / "equity.csv")
    common.write_table(needed_ic_table(horizon, research_ic), out / "needed_ic.csv")
    common.write_json(out / "economics.json", {"rule": rule})
    log.info("threshold rule %s", rule)


def is_consistent(research: dict, forward: dict) -> bool:
    band = 1.96 * np.hypot(research["se"], forward["se"])
    same_sign = np.sign(research["ic"]) == np.sign(forward["ic"])
    return bool(same_sign and abs(forward["ic"] - research["ic"]) < band)


def forward_row(name: str, research: dict, forward: dict) -> dict:
    return {
        "coin": name,
        "research_ic": research["ic"],
        "research_se": research["se"],
        "forward_ic": forward["ic"],
        "forward_se": forward["se"],
        "forward_t": forward["t"],
        "consistent": is_consistent(research, forward),
    }


def forward_table(research_daily: dict, forward_daily: dict) -> pd.DataFrame:
    research = {coin: metrics.ic_summary(daily) for coin, daily in research_daily.items()}
    forward = {coin: metrics.ic_summary(daily) for coin, daily in forward_daily.items()}
    research["pooled"] = metrics.pooled_ic_summary(research_daily)
    forward["pooled"] = metrics.pooled_ic_summary(forward_daily)
    return pd.DataFrame([forward_row(name, research[name], forward[name]) for name in research])


def context_table(horizon: int, inputs: str, policy: str) -> pd.DataFrame:
    rows = []
    for coin, family in itertools.product(config.TARGETS, FAMILIES):
        frame = common.predictions(coin, horizon, family, inputs, policy, "forward")
        rows.append({"coin": coin, "family": family} | metrics.ic_summary(common.daily_ic(frame)))
    return pd.DataFrame(rows)


def shrinkage_table(best: dict) -> pd.DataFrame:
    rows = []
    for coin in config.TARGETS:
        horizon, family = best[coin]["horizon"], best[coin]["family"]
        row = {"coin": coin, "horizon": horizon, "family": family}
        for period in common.PERIODS:
            frame = common.predictions(coin, horizon, family, period=period)
            row[f"{period}_ic"] = common.daily_ic(frame).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def tercile_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for coin, frame in frames.items():
        products = metrics.rank_products(common.centred_signal(frame), frame["fwd_logret"])
        volatility = common.load_features(coin).loc[products.index, "x_roll_vol_logret_576"]
        tercile = pd.qcut(volatility, 3, labels=["low", "middle", "high"])
        for name, group in products.groupby(tercile, observed=True):
            rows.append({"coin": coin, "tercile": name, "ic": group.mean(), "rows": len(group)})
    return pd.DataFrame(rows)


def forward_audit() -> pd.DataFrame:
    audit = bars.audit_bars({asset: bars.read_klines(asset) for asset in config.PAIRS})
    return audit[audit["month"] >= config.FORWARD_MONTHS[0]]


def run_forward() -> None:
    out = common.output_dir("forward")
    chosen = chosen_configuration()
    if chosen != config.FROZEN:
        raise RuntimeError(f"config.FROZEN {config.FROZEN} differs from E1 to E4: {chosen}")
    horizon, family, inputs, policy, rule = (
        chosen[key] for key in ("horizon", "family", "inputs", "policy", "rule")
    )
    research = predictions_by_coin(horizon, family, inputs, policy)
    forward = predictions_by_coin(horizon, family, inputs, policy, "forward")
    research_daily = {coin: common.daily_ic(frame) for coin, frame in research.items()}
    forward_daily = {coin: common.daily_ic(frame) for coin, frame in forward.items()}
    economics, daily = strategy_table(forward, horizon, [rule], [0], "forward")
    portfolio = {
        fee: metrics.equal_weight(common.by_coin(daily, rule, fee, 0)) for fee in config.FEES_BPS
    }
    curves = {f"portfolio_{fee}": portfolio[fee]["net"].cumsum() for fee in config.FEES_BPS}
    curves |= {f"{coin}_taker": daily[(coin, rule, "taker", 0)]["net"].cumsum() for coin in forward}
    selection = common.read_json(config.RESULTS_DIR / "horizon-sweep" / "selection.json")

    common.write_table(forward_table(research_daily, forward_daily), out / "forward.csv")
    common.write_table(context_table(horizon, inputs, policy), out / "context.csv")
    common.write_table(shrinkage_table(selection["per_coin_best"]), out / "shrinkage.csv")
    common.write_table(economics.drop(columns=["rule", "delay"]), out / "economics.csv")
    common.write_table(tercile_table(forward), out / "volatility_terciles.csv")
    common.write_table(pd.DataFrame(curves).rename_axis("day").reset_index(), out / "equity.csv")
    common.write_table(forward_audit(), out / "audit.csv")
    verdict = {fee: common.strategy_stats(combined) for fee, combined in portfolio.items()}
    common.write_json(out / "verdict.json", verdict)
    log.info("forward run written")


EXPERIMENTS = {
    "audit": run_audit,
    "horizon-sweep": run_horizon_sweep,
    "feature-sources": run_feature_sources,
    "retraining": run_retraining,
    "economics": run_economics,
    "forward": run_forward,
}
