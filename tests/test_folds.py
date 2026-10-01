import numpy as np
import pandas as pd
import pytest
from conftest import bars_between
from scipy.stats import spearmanr

from src.config import GRID_START, HORIZONS, RESEARCH_MONTHS, SEED, TRAIN_START
from src.data.bars import BAR
from src.data.features import MODEL_INPUTS
from src.experiments.common import utc
from src.folds import (
    QUANTILES,
    monthly_folds,
    predict_out_of_sample,
    scored_rows,
    split_fold,
)
from src.models import LinearRule, fit_momentum, fit_ridge

MONTHS = ("2025-08", "2025-09")
END = utc("2025-10-01")


def synthetic_features(seed: int = SEED) -> pd.DataFrame:
    index = bars_between(GRID_START, "2025-10-02")
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((len(index), len(MODEL_INPUTS)))
    inputs = pd.DataFrame(noise, index=index, columns=MODEL_INPUTS)
    inputs[index < utc(TRAIN_START)] = np.nan
    labels = {}
    for horizon in HORIZONS:
        sigma = 0.004 * np.sqrt(horizon)
        signal = 0.3 * inputs[f"x_trail_ret_{horizon}"]
        fwd_logret = sigma * (signal + rng.standard_normal(len(index)))
        fwd_logret.iloc[-horizon:] = np.nan
        labels[f"fwd_logret_{horizon}"] = fwd_logret
        labels[f"sigma_{horizon}"] = sigma
        labels[f"y_{horizon}"] = np.tanh(fwd_logret / sigma)
    return inputs.assign(**labels)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return synthetic_features()


def test_expanding_fold_validation_month():
    [fold] = monthly_folds(["2026-01"])
    assert fold.train_start == utc(TRAIN_START)
    assert fold.valid_start == utc("2025-12-01")
    assert fold.train_end == fold.test_start == utc("2026-01-01")
    assert fold.test_end == utc("2026-02-01")


def test_rolling_window_is_three_months_long():
    for fold in monthly_folds(RESEARCH_MONTHS, "rolling_3m"):
        assert fold.train_start + pd.DateOffset(months=3) == fold.train_end == fold.test_start
        assert fold.valid_start + pd.DateOffset(months=1) == fold.train_end
    assert monthly_folds(["2026-03"], "rolling_3m")[0].train_start == utc("2025-12-01")


def test_fixed_policy_keeps_the_first_window():
    folds = monthly_folds(RESEARCH_MONTHS, "fixed")
    windows = {(fold.train_start, fold.valid_start, fold.train_end) for fold in folds}
    assert windows == {(utc(TRAIN_START), utc("2025-11-01"), utc("2025-12-01"))}
    assert [fold.test_start for fold in folds] == [utc(month) for month in RESEARCH_MONTHS]


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        monthly_folds(RESEARCH_MONTHS, "rolling")


def test_a_label_is_known_when_its_last_bar_closes(frame):
    opened = utc("2025-09-10 12:00")
    for horizon in HORIZONS:
        known_at = opened + (horizon + 1) * BAR
        assert not scored_rows(frame, horizon, MODEL_INPUTS, known_at - BAR)[opened]
        assert scored_rows(frame, horizon, MODEL_INPUTS, known_at)[opened]


@pytest.mark.parametrize("horizon", HORIZONS)
def test_split_ends_purge_unknown_labels(frame, horizon):
    [fold] = monthly_folds(["2025-09"])
    rows = split_fold(frame, fold, horizon, MODEL_INPUTS, END)
    split_ends = [
        (rows.train, fold.valid_start),
        (rows.valid, fold.train_end),
        (rows.refit, fold.train_end),
        (rows.test, END),
    ]
    for step, end in split_ends:
        assert end - (horizon + 1) * BAR in step.index
        assert end - horizon * BAR not in step.index
        assert frame.loc[end - horizon * BAR, MODEL_INPUTS + [f"y_{horizon}"]].notna().all()


def test_only_the_period_end_purges_test_rows(frame):
    august, september = (
        split_fold(frame, fold, 12, MODEL_INPUTS, END).test for fold in monthly_folds(MONTHS)
    )
    assert august.index[0] == utc("2025-08-01")
    assert august.index[-1] == utc("2025-09-01") - BAR
    assert september.index[-1] == END - 13 * BAR


@pytest.mark.parametrize("policy", ["expanding", "rolling_3m", "fixed"])
def test_train_and_test_never_share_a_row(frame, policy):
    for fold in monthly_folds(MONTHS, policy):
        rows = split_fold(frame, fold, 72, MODEL_INPUTS, END)
        for before in (rows.train, rows.valid, rows.refit, rows.threshold):
            assert before.index.intersection(rows.test.index).empty
            assert before.index.max() < fold.test_start
        assert rows.train.index.min() >= max(fold.train_start, utc(TRAIN_START))


def test_threshold_rows_need_inputs_but_not_labels(frame):
    [fold] = monthly_folds(["2025-09"])
    unlabelled, without_input = utc("2025-08-10 00:00"), utc("2025-08-11 00:00")
    damaged = frame.copy()
    damaged.loc[unlabelled, "y_12"] = np.nan
    damaged.loc[without_input, "btc_logret_1"] = np.nan
    rows = split_fold(damaged, fold, 12, MODEL_INPUTS, END)
    unknown_label = fold.train_end - BAR
    for moment in (unlabelled, unknown_label):
        assert moment in rows.threshold.index
        assert moment not in rows.valid.index
    assert without_input not in rows.threshold.index
    assert without_input not in rows.valid.index


def test_fixed_policy_fits_once(frame):
    calls = []

    def counted_momentum(train, valid, inputs, label, horizon):
        calls.append(train.index[-1])
        return fit_momentum(train, valid, inputs, label, horizon)

    fixed = predict_out_of_sample(frame, 12, counted_momentum, MODEL_INPUTS, "fixed", MONTHS, END)
    assert len(calls) == 1
    assert fixed["setting"].nunique() == 1
    predict_out_of_sample(frame, 12, counted_momentum, MODEL_INPUTS, "expanding", MONTHS, END)
    assert len(calls) == 3


def test_thresholds_come_from_refit_model(frame):
    refit = LinearRule("x_logret_1", intercept=1.0, slope=1.0)

    def inner_differs_from_refit(train, valid, inputs, label, horizon):
        return refit, valid["x_logret_1"].to_numpy(), "stub"

    out = predict_out_of_sample(
        frame, 12, inner_differs_from_refit, MODEL_INPUTS, "expanding", ["2025-09"], END
    )
    expected = 1.0 + np.quantile(frame.loc["2025-08", "x_logret_1"], list(QUANTILES.values()))
    actual = out[list(QUANTILES)].drop_duplicates().to_numpy()
    np.testing.assert_allclose(actual, [expected])


def test_walk_forward_finds_a_planted_signal(frame):
    out = predict_out_of_sample(frame, 12, fit_ridge, MODEL_INPUTS, "expanding", MONTHS, END)
    thresholds = list(QUANTILES)
    columns = ["pred", "y", "fwd_logret", "test_month", *thresholds, "valid_ic", "setting"]
    assert list(out.columns) == columns
    assert out["test_month"].unique().tolist() == list(MONTHS)
    assert out.index.is_unique and out.index.max() == END - 13 * BAR
    assert spearmanr(out["pred"], out["y"]).statistic > 0.2
    assert (out["valid_ic"] > 0.2).all()
    assert (np.diff(out[thresholds].to_numpy(), axis=1) > 0).all()
