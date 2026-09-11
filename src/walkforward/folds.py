import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from walkforward.config import RESEARCH_END, RESEARCH_MONTHS, TRAIN_START
from walkforward.data.bars import BAR
from walkforward.models import Fitter

MONTH = pd.DateOffset(months=1)
QUANTILES = {"q10": 0.1, "q30": 0.3, "q50": 0.5, "q70": 0.7, "q90": 0.9}

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    test_month: str
    train_start: pd.Timestamp
    valid_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class FoldRows:
    train: pd.DataFrame
    valid: pd.DataFrame
    refit: pd.DataFrame
    threshold: pd.DataFrame
    test: pd.DataFrame


def training_start(train_end: pd.Timestamp, policy: str) -> pd.Timestamp:
    match policy:
        case "expanding" | "fixed":
            return pd.Timestamp(TRAIN_START, tz="UTC")
        case "rolling_3m":
            return train_end - 3 * MONTH
    raise ValueError(f"unknown retraining policy {policy!r}")


def monthly_folds(test_months: Sequence[str], policy: str = "expanding") -> list[Fold]:
    folds = []
    for month in test_months:
        test_start = pd.Timestamp(month, tz="UTC")
        train_end = test_start
        if policy == "fixed":
            # never refit: the window before the research months, in forward months too
            first = pd.Timestamp(test_months[0], tz="UTC")
            train_end = min(first, pd.Timestamp(RESEARCH_MONTHS[0], tz="UTC"))
        folds.append(
            Fold(
                test_month=month,
                train_start=training_start(train_end, policy),
                valid_start=train_end - MONTH,
                train_end=train_end,
                test_start=test_start,
                test_end=test_start + MONTH,
            )
        )
    return folds


def label_known(index: pd.DatetimeIndex, horizon: int, end: pd.Timestamp) -> np.ndarray:
    # row t's label is known at the close of bar t + H
    return index + (horizon + 1) * BAR <= end


def within(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    return (index >= start) & (index < end)


def scored_rows(
    frame: pd.DataFrame, horizon: int, inputs: list[str], end: str | pd.Timestamp
) -> pd.Series:
    defined = frame.notna()[[*inputs, f"y_{horizon}"]].all(axis=1)
    return defined & label_known(frame.index, horizon, pd.to_datetime(end, utc=True))


def split_fold(
    frame: pd.DataFrame,
    fold: Fold,
    horizon: int,
    inputs: list[str],
    end: str | pd.Timestamp = RESEARCH_END,
) -> FoldRows:
    in_train = within(frame.index, fold.train_start, fold.valid_start)
    in_valid = within(frame.index, fold.valid_start, fold.train_end)
    in_test = within(frame.index, fold.test_start, fold.test_end)
    train = in_train & scored_rows(frame, horizon, inputs, fold.valid_start)
    valid = in_valid & scored_rows(frame, horizon, inputs, fold.train_end)
    return FoldRows(
        train=frame[train],
        valid=frame[valid],
        refit=frame[train | valid],
        threshold=frame[in_valid & frame.notna()[inputs].all(axis=1)],
        test=frame[in_test & scored_rows(frame, horizon, inputs, end)],
    )


def fit_window(
    rows: FoldRows, horizon: int, fitter: Fitter, inputs: list[str]
) -> tuple[Any, dict[str, float], float, str]:
    model, valid_pred, setting = fitter(rows.train, rows.valid, inputs, f"y_{horizon}", horizon)
    levels = np.quantile(model.predict(rows.threshold[inputs]), list(QUANTILES.values()))
    valid_ic = spearmanr(valid_pred, rows.valid[f"fwd_logret_{horizon}"]).statistic
    return model, dict(zip(QUANTILES, levels)), float(valid_ic), setting


def predict_out_of_sample(
    frame: pd.DataFrame,
    horizon: int,
    fitter: Fitter,
    inputs: list[str],
    policy: str = "expanding",
    months: Sequence[str] = RESEARCH_MONTHS,
    end: str | pd.Timestamp = RESEARCH_END,
) -> pd.DataFrame:
    frame = frame[[*inputs, f"y_{horizon}", f"fwd_logret_{horizon}"]]
    fitted = {}
    predictions = []
    for fold in monthly_folds(months, policy):
        rows = split_fold(frame, fold, horizon, inputs, end)
        window = (fold.train_start, fold.valid_start, fold.train_end)
        if window not in fitted:
            fitted[window] = fit_window(rows, horizon, fitter, inputs)
        model, thresholds, valid_ic, setting = fitted[window]
        test = rows.test
        predictions.append(
            pd.DataFrame(
                {
                    "pred": model.predict(test[inputs]),
                    "y": test[f"y_{horizon}"],
                    "fwd_logret": test[f"fwd_logret_{horizon}"],
                    "test_month": fold.test_month,
                    **thresholds,
                    "valid_ic": valid_ic,
                    "setting": setting,
                },
                index=test.index,
            )
        )
        log.debug("%s H=%d %s, valid IC %.3f", fold.test_month, horizon, setting, valid_ic)
    return pd.concat(predictions)
