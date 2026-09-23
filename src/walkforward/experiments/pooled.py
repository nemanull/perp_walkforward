import functools
import hashlib
import itertools
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone

from walkforward import config, folds, metrics, models
from walkforward.experiments import common

log = logging.getLogger(__name__)

MODELS = ("pooled", "per_coin")


def standardise(rows: pd.DataFrame, fitting: pd.DataFrame, inputs: list[str]) -> pd.DataFrame:
    scaled = (rows[inputs] - fitting[inputs].mean()) / fitting[inputs].std(ddof=0)
    return rows.assign(**scaled)


def stack(
    rows: dict[str, pd.DataFrame], fitting: dict[str, pd.DataFrame], inputs: list[str]
) -> pd.DataFrame:
    # each coin is scaled by its own rows, so TRX's small moves do not read as quiet spells
    scaled = {coin: standardise(rows[coin], fitting[coin], inputs) for coin in rows}
    return pd.concat(scaled, names=["coin"])


def part(splits: dict[str, folds.FoldRows], name: str) -> dict[str, pd.DataFrame]:
    return {coin: getattr(rows, name) for coin, rows in splits.items()}


def refit_model(model: Any, rows: pd.DataFrame, inputs: list[str], label: str) -> Any:
    # a LightGBM month that never split is a constant, which no scaling changes
    if isinstance(model, models.LinearRule):
        return model
    return clone(model).fit(rows[inputs], rows[label])


def thresholds(model: Any, rows: pd.DataFrame, inputs: list[str]) -> dict[str, float]:
    levels = np.quantile(model.predict(rows[inputs]), list(folds.QUANTILES.values()))
    return dict(zip(folds.QUANTILES, levels))


def pooled_fold(
    frames: dict[str, pd.DataFrame],
    fold: folds.Fold,
    horizon: int,
    fitter: models.Fitter,
    inputs: list[str],
    end: str | pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    label, realised = f"y_{horizon}", f"fwd_logret_{horizon}"
    splits = {
        coin: folds.split_fold(frame, fold, horizon, inputs, end) for coin, frame in frames.items()
    }
    train, valid, refit = (part(splits, name) for name in ("train", "valid", "refit"))
    inner_valid = stack(valid, train, inputs)
    model, valid_pred, setting = fitter(
        stack(train, train, inputs), inner_valid, inputs, label, horizon
    )
    # the chosen setting is fit again on rows scaled by the refit rows
    model = refit_model(model, stack(refit, refit, inputs), inputs, label)
    valid_pred = pd.Series(valid_pred, index=inner_valid.index)
    log.debug("%s H=%d pooled %s", fold.test_month, horizon, setting)
    predictions = {}
    for coin, rows in splits.items():
        test = standardise(rows.test, rows.refit, inputs)
        predictions[coin] = pd.DataFrame(
            {
                "pred": model.predict(test[inputs]),
                "y": test[label],
                "fwd_logret": test[realised],
                "test_month": fold.test_month,
                **thresholds(model, standardise(rows.threshold, rows.refit, inputs), inputs),
                "valid_ic": spearmanr(valid_pred.loc[coin], rows.valid[realised]).statistic,
                "setting": setting,
            },
            index=test.index,
        )
    return predictions


def pooled_out_of_sample(
    frames: dict[str, pd.DataFrame],
    horizon: int,
    fitter: models.Fitter,
    inputs: list[str],
    months: Sequence[str],
    end: str | pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    by_fold = [
        pooled_fold(frames, fold, horizon, fitter, inputs, end)
        for fold in folds.monthly_folds(months)
    ]
    return {coin: pd.concat([fold[coin] for fold in by_fold]) for coin in frames}


@functools.cache
def code_tag() -> str:
    source = common.code_tag().encode() + Path(__file__).read_bytes()
    return hashlib.sha1(source).hexdigest()[:8]


def load_frames(horizon: int, inputs: list[str]) -> dict[str, pd.DataFrame]:
    columns = [*inputs, f"y_{horizon}", f"fwd_logret_{horizon}"]
    return {
        coin: pd.read_parquet(config.DATA_DIR / "features" / f"{coin}.parquet", columns=columns)
        for coin in config.TARGETS
    }


def pooled_predictions(
    horizon: int, family: str, inputs: str = "all", period: str = "research"
) -> dict[str, pd.DataFrame]:
    folder = config.DATA_DIR / "predictions" / code_tag()
    paths = {
        coin: folder / f"pooled_{coin}_{horizon}_{family}_{inputs}_{period}.parquet"
        for coin in config.TARGETS
    }
    if all(path.exists() for path in paths.values()):
        return {coin: pd.read_parquet(path) for coin, path in paths.items()}
    log.info("predicting pooled H=%d %s %s %s", horizon, family, inputs, period)
    columns = common.INPUT_SETS[inputs]
    predictions = pooled_out_of_sample(
        load_frames(horizon, columns),
        horizon,
        models.FITTERS[family],
        columns,
        *common.PERIODS[period],
    )
    folder.mkdir(parents=True, exist_ok=True)
    for coin, frame in predictions.items():
        frame.to_parquet(paths[coin])
    return predictions


def comparison_row(pooled_daily: pd.Series, per_coin_daily: pd.Series, difference: dict) -> dict:
    return {
        "pooled_model_ic": pooled_daily.mean(),
        "per_coin_ic": per_coin_daily.mean(),
        "diff": difference["ic"],
        "se": difference["se"],
        "t": difference["t"],
    }


def comparison_table(daily: dict, horizon: int) -> pd.DataFrame:
    rows = []
    for period in common.PERIODS:
        pooled, per_coin = (common.by_coin(daily, model, period) for model in MODELS)
        for coin in config.TARGETS:
            difference = metrics.paired_ic_difference(pooled[coin], per_coin[coin])
            row = comparison_row(pooled[coin], per_coin[coin], difference)
            sigma = common.realised_sigma(coin, horizon) * 1e4
            rows.append({"coin": coin, "period": period} | row | {"sigma_bps": sigma})
        difference = metrics.pooled_ic_difference(pooled, per_coin)
        overall = comparison_row(
            metrics.equal_weight(pooled), metrics.equal_weight(per_coin), difference
        )
        rows.append({"coin": "pooled", "period": period} | overall)
    return pd.DataFrame(rows)


def settings_table(pooled: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    # one model serves every coin, so any coin's rows carry its setting
    settings = {
        period: next(iter(frames.values())).groupby("test_month")["setting"].first()
        for period, frames in pooled.items()
    }
    return pd.concat(settings, names=["period", "month"]).reset_index()


def run_pooled() -> None:
    out = common.output_dir("pooled")
    horizon, inputs = config.FROZEN["horizon"], config.FROZEN["inputs"]
    family = common.fitted_family(config.FROZEN["family"])
    pooled = {
        period: pooled_predictions(horizon, family, inputs, period) for period in common.PERIODS
    }
    daily = {}
    for period, model, coin in itertools.product(common.PERIODS, MODELS, config.TARGETS):
        if model == "pooled":
            frame = pooled[period][coin]
        else:
            frame = common.predictions(coin, horizon, family, inputs, "expanding", period)
        daily[(coin, model, period)] = common.daily_ic(frame)
    table = comparison_table(daily, horizon)
    research = table[(table["coin"] == "pooled") & (table["period"] == "research")].iloc[0]
    keep = "pooled" if research["t"] >= config.PAIRED_T else "per_coin"
    by_month = common.ic_by_month_table(daily, ["coin", "model", "period"])

    common.write_table(table, out / "pooled.csv")
    common.write_table(by_month, out / "ic_by_month.csv")
    common.write_table(settings_table(pooled), out / "settings.csv")
    common.write_json(
        out / "pooled.json",
        {
            "horizon": horizon,
            "family": family,
            "inputs": inputs,
            "pooled_minus_per_coin": research["diff"],
            "t": research["t"],
            "model": keep,
        },
    )
    log.info("pooled minus per-coin t=%.2f, keep %s", research["t"], keep)


EXPERIMENTS = {"pooled": run_pooled}
