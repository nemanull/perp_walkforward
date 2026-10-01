import numpy as np
import pandas as pd
import pytest
from conftest import bars_between
from scipy.stats import spearmanr

from src.config import GRID_START, SEED, TARGETS, TRAIN_START
from src.experiments.common import PERIODS, utc
from src.experiments.pooled import pooled_out_of_sample, stack, standardise
from src.folds import QUANTILES, monthly_folds, predict_out_of_sample, split_fold
from src.models import LinearRule, fit_ridge
from src.plots import render

INPUTS = ["x_trail_ret_12", "x_logret_1", "btc_trail_ret_12", "eth_volume_z_576"]
MONTHS = ("2025-08", "2025-09")
END = utc("2025-10-01")
PHASE_1_COLUMNS = ["pred", "y", "fwd_logret", "test_month", *QUANTILES]


def coin_frame(scale: float, centre: float, seed: int) -> pd.DataFrame:
    index = bars_between(GRID_START, "2025-10-02")
    rng = np.random.default_rng(seed)
    standard = rng.standard_normal((len(index), len(INPUTS)))
    frame = pd.DataFrame(centre + scale * standard, index=index, columns=INPUTS)
    frame[index < utc(TRAIN_START)] = np.nan
    fwd_logret = 0.004 * (0.3 * standard[:, 0] + rng.standard_normal(len(index)))
    fwd_logret[-12:] = np.nan
    return frame.assign(fwd_logret_12=fwd_logret, y_12=np.tanh(fwd_logret / 0.004))


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    return {"calm": coin_frame(0.001, 0.02, SEED), "wild": coin_frame(1.0, -3.0, SEED + 1)}


def test_standardisation_reads_only_the_fitting_rows(frames):
    rows = frames["calm"].dropna()
    fitting, later = rows.iloc[:2000], rows.iloc[2000:4000]
    scaled = standardise(pd.concat([fitting, later]), fitting, INPUTS)
    moved = later.assign(**later[INPUTS] * 1000)
    rescaled = standardise(pd.concat([fitting, moved]), fitting, INPUTS)
    pd.testing.assert_frame_equal(rescaled.iloc[:2000], scaled.iloc[:2000])
    np.testing.assert_allclose(scaled[INPUTS].iloc[:2000].mean(), 0.0, atol=1e-9)
    np.testing.assert_allclose(scaled[INPUTS].iloc[:2000].std(ddof=0), 1.0)
    expected = (later[INPUTS] - fitting[INPUTS].mean()) / fitting[INPUTS].std(ddof=0)
    pd.testing.assert_frame_equal(scaled[INPUTS].iloc[2000:], expected)


def test_stacking_keeps_every_row_with_its_coin_and_timestamp(frames):
    stacked = stack(frames, frames, INPUTS)
    assert stacked.index.names == ["coin", "open_time"]
    assert list(stacked.columns) == list(frames["calm"].columns)
    for coin, frame in frames.items():
        assert stacked.loc[coin].index.equals(frame.index)
        pd.testing.assert_series_equal(stacked.loc[coin, "y_12"], frame["y_12"])


def test_each_coin_scaled_by_own_rows(frames):
    seen = {}

    def first_input(train, valid, inputs, label, horizon):
        seen["train"], seen["valid"] = train, valid
        return LinearRule(inputs[0], 0.0, 1.0), valid[inputs[0]].to_numpy(), "stub"

    out = pooled_out_of_sample(frames, 12, first_input, INPUTS, ["2025-09"], END)
    [fold] = monthly_folds(["2025-09"])
    column = INPUTS[0]
    for coin, frame in frames.items():
        rows = split_fold(frame, fold, 12, INPUTS, END)
        inner = seen["train"].loc[coin, INPUTS]
        np.testing.assert_allclose(inner.mean(), 0.0, atol=1e-9)
        np.testing.assert_allclose(inner.std(ddof=0), 1.0)
        valid = (rows.valid[column] - rows.train[column].mean()) / rows.train[column].std(ddof=0)
        np.testing.assert_allclose(seen["valid"].loc[coin, column], valid)
        refit = rows.refit[column].mean(), rows.refit[column].std(ddof=0)
        np.testing.assert_allclose(out[coin]["pred"], (rows.test[column] - refit[0]) / refit[1])
        own_month = (rows.threshold[column] - refit[0]) / refit[1]
        expected = np.quantile(own_month, [0.1, 0.5, 0.9])
        np.testing.assert_allclose(out[coin][["q10", "q50", "q90"]].iloc[0], expected)
        assert out[coin].index.equals(rows.test.index)


def test_pooled_model_across_scales(frames):
    out = pooled_out_of_sample(frames, 12, fit_ridge, INPUTS, MONTHS, END)
    for predictions in out.values():
        assert list(predictions.columns) == [*PHASE_1_COLUMNS, "valid_ic", "setting"]
        assert predictions["test_month"].unique().tolist() == list(MONTHS)
        assert spearmanr(predictions["pred"], predictions["y"]).statistic > 0.2
        assert (predictions["valid_ic"] > 0.2).all()
    assert out["calm"]["setting"].equals(out["wild"]["setting"])


def test_pooled_with_one_coin_matches_per_coin(frames):
    pooled = pooled_out_of_sample({"wild": frames["wild"]}, 12, fit_ridge, INPUTS, MONTHS, END)
    alone = predict_out_of_sample(frames["wild"], 12, fit_ridge, INPUTS, "expanding", MONTHS, END)
    pd.testing.assert_frame_equal(pooled["wild"], alone, rtol=1e-9)


def test_every_pooled_figure_is_drawn(tmp_path):
    folder = tmp_path / "pooled"
    folder.mkdir()
    rng = np.random.default_rng(SEED)
    names = [*TARGETS, "pooled"]
    table = pd.DataFrame({"coin": names * 2, "period": np.repeat(list(PERIODS), len(names))})
    table["pooled_model_ic"] = rng.normal(0.03, 0.01, len(table))
    table["per_coin_ic"] = table["pooled_model_ic"] + rng.normal(0, 0.005, len(table))
    table["diff"] = table["pooled_model_ic"] - table["per_coin_ic"]
    table["se"] = rng.uniform(0.002, 0.006, len(table))
    table["t"] = table["diff"] / table["se"]
    table["sigma_bps"] = np.where(
        table["coin"] == "pooled", np.nan, rng.uniform(20, 70, len(table))
    )
    table.to_csv(folder / "pooled.csv", index=False)
    render("pooled", tmp_path)
    for name in ("ic_difference", "difference_vs_volatility"):
        assert (folder / f"{name}.png").stat().st_size > 0
