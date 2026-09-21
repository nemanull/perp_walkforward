import functools
import hashlib
import itertools
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from walkforward import backtest, config, folds, metrics, models
from walkforward.data import features
from walkforward.experiments import common

log = logging.getLogger(__name__)

HAR_WINDOWS = (12, 72, 288)
HAR_INPUTS = [f"log_rv_{window}" for window in HAR_WINDOWS]
LIGHTGBM_INPUTS = [*features.MODEL_INPUTS, "hour"]
EDGE_PER_UNIT_IC = {"median": 0.80, "outer30": 1.16, "outer10": 1.755}  # k_q of each rule
SIZE_CAP = 2.0
VERSIONS = ("base", "gated", "sized")
KEYS = ("coin", "horizon", "model", "period")


@functools.lru_cache(maxsize=1)
def load_coin(coin: str) -> pd.DataFrame:
    path = config.DATA_DIR / "features" / f"{coin}.parquet"
    return pd.read_parquet(path, columns=["x_close", *features.MODEL_INPUTS])


def squared_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff() ** 2


def log_realised_volatility(sum_of_squares: pd.Series) -> pd.Series:
    # no price change in the window: ln 0, so no value, like a window with a gap
    return 0.5 * np.log(sum_of_squares.where(sum_of_squares > 0))


def trailing_log_rv(close: pd.Series, window: int) -> pd.Series:
    return log_realised_volatility(squared_returns(close).rolling(window).sum())


def forward_log_rv(close: pd.Series, horizon: int) -> pd.Series:
    return trailing_log_rv(close, horizon).shift(-horizon)


def volatility_frame(coin_features: pd.DataFrame, horizon: int) -> pd.DataFrame:
    close = coin_features["x_close"]
    har = {f"log_rv_{window}": trailing_log_rv(close, window) for window in HAR_WINDOWS}
    inputs = coin_features[features.MODEL_INPUTS].assign(hour=coin_features.index.hour, **har)
    # one row set for both models: no label where either model lacks an input
    label = forward_log_rv(close, horizon).where(inputs.notna().all(axis=1))
    # the walk-forward fits y_H and ranks fwd_logret_H, so both hold ln RV
    return inputs.assign(**{f"y_{horizon}": label, f"fwd_logret_{horizon}": label})


@dataclass(frozen=True)
class HarModel:
    intercept: float
    slopes: tuple[float, ...]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.intercept + X[HAR_INPUTS].to_numpy() @ np.array(self.slopes)


def least_squares_har(rows: pd.DataFrame, label: str) -> HarModel:
    design = np.column_stack([np.ones(len(rows)), rows[HAR_INPUTS].to_numpy()])
    intercept, *slopes = np.linalg.lstsq(design, rows[label].to_numpy(), rcond=None)[0]
    return HarModel(float(intercept), tuple(float(slope) for slope in slopes))


def fit_har(
    train: pd.DataFrame, valid: pd.DataFrame, inputs: list[str], label: str, horizon: int
) -> tuple[HarModel, np.ndarray, str]:
    model = least_squares_har(pd.concat([train, valid]), label)
    setting = "slopes=" + ",".join(f"{slope:.2f}" for slope in model.slopes)
    return model, least_squares_har(train, label).predict(valid), setting


MODELS = {"har": (fit_har, HAR_INPUTS), "lightgbm": (models.fit_lightgbm, LIGHTGBM_INPUTS)}


def training_means(
    frame: pd.DataFrame, horizon: int, months: Sequence[str], end: str | pd.Timestamp
) -> dict[str, float]:
    label = f"y_{horizon}"
    rows = frame[[*HAR_INPUTS, label]]
    return {
        fold.test_month: folds.split_fold(rows, fold, horizon, HAR_INPUTS, end).refit[label].mean()
        for fold in folds.monthly_folds(months)
    }


def predict_volatility(
    coin_features: pd.DataFrame,
    horizon: int,
    model: str,
    months: Sequence[str],
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    frame = volatility_frame(coin_features, horizon)
    fitter, inputs = MODELS[model]
    predictions = folds.predict_out_of_sample(
        frame, horizon, fitter, inputs, "expanding", months, end
    )
    means = training_means(frame, horizon, months, end)
    return predictions.assign(train_mean=predictions["test_month"].map(means))


@functools.cache
def code_tag() -> str:
    source = common.code_tag().encode() + Path(__file__).read_bytes()
    return hashlib.sha1(source).hexdigest()[:8]


def volatility_predictions(
    coin: str, horizon: int, model: str, period: str = "research"
) -> pd.DataFrame:
    name = f"volatility_{coin}_{horizon}_{model}_{period}.parquet"
    path = config.DATA_DIR / "predictions" / code_tag() / name
    if path.exists():
        return pd.read_parquet(path)
    log.info("predicting volatility %s H=%d %s %s", coin, horizon, model, period)
    frame = predict_volatility(load_coin(coin), horizon, model, *common.PERIODS[period])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return frame


def r2_against_training_mean(frame: pd.DataFrame) -> float:
    # oos_r2 is against a zero forecast, so both sides are measured from the training mean
    benchmark = frame["train_mean"]
    return metrics.oos_r2(frame["pred"] - benchmark, frame["y"] - benchmark)


def evaluate_volatility(frame: pd.DataFrame) -> dict:
    daily = common.daily_ic(frame)
    r2 = r2_against_training_mean(frame)
    return metrics.ic_summary(daily) | {"oos_r2": r2, "daily_ic": daily}


def pooled_table(daily: dict) -> pd.DataFrame:
    rows = [
        {"horizon": horizon, "model": model, "period": period}
        | metrics.pooled_ic_summary(common.by_coin(daily, horizon, model, period))
        for horizon, model, period in itertools.product(config.HORIZONS, MODELS, common.PERIODS)
    ]
    return pd.DataFrame(rows)


def gain_table(daily: dict) -> pd.DataFrame:
    rows = []
    for horizon, period in itertools.product(config.HORIZONS, common.PERIODS):
        lightgbm = common.by_coin(daily, horizon, "lightgbm", period)
        har = common.by_coin(daily, horizon, "har", period)
        gains = {coin: metrics.paired_ic_difference(lightgbm[coin], har[coin]) for coin in har}
        gains["pooled"] = metrics.pooled_ic_difference(lightgbm, har)
        for coin, gain in gains.items():
            row = {"horizon": horizon, "period": period, "coin": coin}
            row |= {"gain": gain["ic"], "se": gain["se"], "t": gain["t"]}
            if coin != "pooled":
                row["sigma_bps"] = common.realised_sigma(coin, horizon) * 1e4
            rows.append(row)
    return pd.DataFrame(rows)


def choose_models(gains: pd.DataFrame) -> dict[str, dict]:
    research = gains[(gains["coin"] == "pooled") & (gains["period"] == "research")]
    return {
        str(row.horizon): {
            "model": "lightgbm" if row.t >= config.PAIRED_T else "har",
            "lightgbm_minus_har": row.gain,
            "t": row.t,
        }
        for row in research.itertuples()
    }


def run_volatility() -> None:
    out = common.output_dir("volatility")
    keys = itertools.product(config.TARGETS, config.HORIZONS, MODELS, common.PERIODS)
    results = {key: evaluate_volatility(volatility_predictions(*key)) for key in keys}
    daily = {key: result.pop("daily_ic") for key, result in results.items()}
    table = pd.DataFrame([dict(zip(KEYS, key, strict=True)) | r for key, r in results.items()])
    gains = gain_table(daily)
    choice = choose_models(gains)

    common.write_table(table, out / "volatility.csv")
    common.write_table(common.ic_by_month_table(daily, list(KEYS)), out / "ic_by_month.csv")
    common.write_table(pooled_table(daily), out / "pooled.csv")
    common.write_table(gains, out / "gain.csv")
    common.write_json(out / "volatility.json", choice)
    log.info("volatility model by horizon %s", {h: c["model"] for h, c in choice.items()})


def expected_edge(valid_ic: pd.Series, sigma_hat: pd.Series, rule: str) -> pd.Series:
    return EDGE_PER_UNIT_IC[rule] * valid_ic * sigma_hat


def edge_covers_fee(
    valid_ic: pd.Series, sigma_hat: pd.Series, rule: str, fee_bps: float
) -> pd.Series:
    # NaN compares False, so no validation IC or no forecast keeps the gate shut
    return expected_edge(valid_ic, sigma_hat, rule) >= 2 * fee_bps / 1e4


def inverse_volatility_size(volatility: pd.DataFrame) -> pd.Series:
    # exp(q50) is the validation-month median of sigma_hat
    return np.exp(volatility["q50"] - volatility["pred"]).clip(upper=SIZE_CAP)


def tranche_scale(
    direction: pd.DataFrame, volatility: pd.DataFrame, rule: str, fee: str, version: str
) -> pd.Series:
    if version == "base":
        return pd.Series(1.0, index=direction.index)
    sigma_hat = np.exp(volatility["pred"])
    is_open = edge_covers_fee(direction["valid_ic"], sigma_hat, rule, config.FEES_BPS[fee])
    if version == "gated":
        return is_open.astype(float)
    return inverse_volatility_size(volatility).where(is_open, 0.0)


def version_pnl(
    direction: pd.DataFrame,
    volatility: pd.DataFrame,
    close: pd.Series,
    funding: pd.Series,
    horizon: int,
    rule: str,
    fee: str,
    version: str,
) -> pd.DataFrame:
    direction, volatility = direction.reindex(close.index), volatility.reindex(close.index)
    signal = common.rule_signal(direction, rule)
    scale = tranche_scale(direction, volatility, rule, fee, version).to_numpy()
    # each tranche opens at its scaled signal, then the position averages the last H
    position = backtest.tranche_position(signal * scale, horizon)
    pnl = backtest.bar_pnl(position, close, config.FEES_BPS[fee], funding)
    return pnl.assign(position=position, signal=signal, scale=scale)


def opened(pnl: pd.DataFrame) -> pd.Series:
    return (pnl["signal"] != 0) & (pnl["scale"] > 0)


def version_stats(pnl: pd.DataFrame, close: pd.Series) -> tuple[dict, pd.DataFrame]:
    stats, daily = common.pnl_stats(pnl, close)
    traded = opened(pnl)
    return stats | {
        "gate_share": traded.sum() / (pnl["signal"] != 0).sum(),
        "mean_size": pnl.loc[traded, "scale"].mean(),
    }, daily


def monthly_gate(pnl: pd.DataFrame, direction: pd.DataFrame) -> pd.DataFrame:
    month = pnl.index.strftime("%Y-%m").rename("month")
    traded = opened(pnl)
    return pd.DataFrame(
        {
            "valid_ic": direction.groupby("test_month")["valid_ic"].first(),
            "gate_share": traded.groupby(month).sum() / (pnl["signal"] != 0).groupby(month).sum(),
            "mean_size": pnl["scale"].where(traded).groupby(month).mean(),
        }
    ).rename_axis("month")


def strategy_tables(horizon: int, rule: str, model: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    frozen = config.FROZEN
    rows, gates, daily = [], [], {}
    for coin, period in itertools.product(config.TARGETS, common.PERIODS):
        direction = common.predictions(
            coin, horizon, frozen["family"], frozen["inputs"], frozen["policy"], period
        )
        volatility = volatility_predictions(coin, horizon, model, period)
        close, funding = common.period_close(coin, period), common.load_funding()[coin]
        for fee in config.FEES_BPS:
            pnls = {
                version: version_pnl(
                    direction, volatility, close, funding, horizon, rule, fee, version
                )
                for version in VERSIONS
            }
            for version, pnl in pnls.items():
                stats, daily[(coin, version, fee, period)] = version_stats(pnl, close)
                rows.append(
                    {"coin": coin, "version": version, "fee": fee, "period": period} | stats
                )
            gate = monthly_gate(pnls["sized"], direction).reset_index()
            gates.append(gate.assign(coin=coin, fee=fee, period=period))
    columns = ["coin", "fee", "period", "month", "valid_ic", "gate_share", "mean_size"]
    return pd.DataFrame(rows), pd.concat(gates)[columns], daily


def portfolio_daily(daily: dict, version: str, fee: str, period: str) -> pd.DataFrame:
    return metrics.equal_weight(common.by_coin(daily, version, fee, period))


def portfolio_table(daily: dict) -> pd.DataFrame:
    rows = [
        {"version": version, "fee": fee, "period": period}
        | common.strategy_stats(portfolio_daily(daily, version, fee, period))
        for version, fee, period in itertools.product(VERSIONS, config.FEES_BPS, common.PERIODS)
    ]
    return pd.DataFrame(rows)


def taker_equity(daily: dict) -> pd.DataFrame:
    curves = {}
    for version in VERSIONS:
        for coin in config.TARGETS:
            net = [daily[(coin, version, "taker", period)]["net"] for period in common.PERIODS]
            curves[f"{coin}_{version}"] = pd.concat(net).cumsum()
        net = [portfolio_daily(daily, version, "taker", period)["net"] for period in common.PERIODS]
        curves[f"portfolio_{version}"] = pd.concat(net).cumsum()
    return pd.DataFrame(curves).rename_axis("day").reset_index()


def research_taker_sharpe(portfolio: pd.DataFrame) -> pd.Series:
    research = portfolio[(portfolio["period"] == "research") & (portfolio["fee"] == "taker")]
    return research.set_index("version")["sharpe"]


def run_volatility_strategy() -> None:
    out = common.output_dir("volatility-strategy")
    horizon, rule = config.FROZEN["horizon"], config.FROZEN["rule"]
    choice = common.read_json(config.RESULTS_DIR / "volatility" / "volatility.json")
    model = choice[str(horizon)]["model"]
    strategy, gate, daily = strategy_tables(horizon, rule, model)
    portfolio = portfolio_table(daily)
    sharpe = research_taker_sharpe(portfolio)
    version = str(sharpe.idxmax())
    net = pd.DataFrame(
        {coin: daily[(coin, version, "taker", "research")]["net"] for coin in config.TARGETS}
    )

    common.write_table(strategy, out / "strategy.csv")
    common.write_table(gate, out / "gate.csv")
    common.write_table(portfolio, out / "portfolio.csv")
    correlation = net.corr().rename_axis("coin").reset_index()
    common.write_table(correlation, out / "pnl_correlation.csv", digits=4)
    common.write_table(taker_equity(daily), out / "equity.csv")
    common.write_json(
        out / "volatility-strategy.json",
        {
            "horizon": horizon,
            "rule": rule,
            "volatility_model": model,
            "version": version,
            "research_taker_sharpe": sharpe.to_dict(),
        },
    )
    log.info("phase 2 strategy version %s", version)


EXPERIMENTS = {"volatility": run_volatility, "volatility-strategy": run_volatility_strategy}
