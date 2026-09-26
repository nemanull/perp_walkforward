import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter
from scipy.stats import spearmanr

from src.config import RIDGE_ALPHAS, SEED
from src.data.features import MODEL_INPUTS
from src.models import (
    FITTERS,
    LinearRule,
    fit_best_feature,
    fit_lightgbm,
    fit_momentum,
    fit_ridge,
)

INPUTS = MODEL_INPUTS[:12]


def planted_frame(rows: int, planted: str, strength: float) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    frame = pd.DataFrame(rng.standard_normal((rows, len(INPUTS))), columns=INPUTS)
    frame["y_4"] = np.tanh(strength * frame[planted] + rng.standard_normal(rows))
    return frame


def split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = len(frame) * 3 // 4
    return frame.iloc[:cut], frame.iloc[cut:]


def test_linear_rule_predicts_from_its_column():
    rule = LinearRule("x_trail_ret_12", intercept=0.1, slope=-2.0)
    X = pd.DataFrame({"x_trail_ret_12": [0.0, 0.05], "btc_logret_1": [9.0, 9.0]})
    np.testing.assert_allclose(rule.predict(X), [0.1, 0.0])


def test_momentum_fades_a_mean_reverting_series():
    rng = np.random.default_rng(SEED)
    log_price = pd.Series(lfilter([1.0], [1.0, -0.9], 0.002 * rng.standard_normal(20_000)))
    fwd_logret = log_price.shift(-12) - log_price
    frame = pd.DataFrame(
        {"x_trail_ret_12": log_price.diff(12), "y_12": np.tanh(fwd_logret / 0.005)}
    ).dropna()
    train, valid = split(frame)
    model, valid_pred, setting = fit_momentum(train, valid, ["x_trail_ret_12"], "y_12", 12)
    assert model.column == "x_trail_ret_12"
    assert model.slope < 0
    assert setting.startswith("slope=-")
    assert spearmanr(valid_pred, valid["y_12"]).statistic > 0.2


def test_best_feature_finds_a_planted_input():
    train, valid = split(planted_frame(8_000, planted=INPUTS[7], strength=0.3))
    model, _, setting = fit_best_feature(train, valid, INPUTS, "y_4", 4)
    assert model.column == setting == INPUTS[7]
    assert model.slope > 0


def test_best_feature_breaks_ties_by_input_order():
    frame = planted_frame(4_000, planted=INPUTS[1], strength=0.3)
    frame[INPUTS[3]] = frame[INPUTS[1]]
    train, valid = split(frame)
    assert fit_best_feature(train, valid, INPUTS[:4], "y_4", 4)[0].column == INPUTS[1]
    assert fit_best_feature(train, valid, INPUTS[3::-1], "y_4", 4)[0].column == INPUTS[3]


def test_ridge_recovers_a_planted_linear_signal():
    rng = np.random.default_rng(SEED)
    beta = np.array([0.3, -0.2, 0.1, 0.0, 0.0, 0.05, -0.1, 0.0])
    standard = rng.standard_normal((20_000, len(beta)))
    raw_scales = 10.0 ** np.arange(-4, 4)
    frame = pd.DataFrame(standard * raw_scales, columns=INPUTS[:8])
    frame["y_12"] = standard @ beta + 0.2 * rng.standard_normal(len(frame))
    train, valid = split(frame)
    model, valid_pred, setting = fit_ridge(train, valid, INPUTS[:8], "y_12", 12)
    np.testing.assert_allclose(model[-1].coef_, beta, atol=0.02)
    assert setting in {f"alpha={alpha:.0e}" for alpha in RIDGE_ALPHAS}
    assert spearmanr(valid_pred, valid["y_12"]).statistic > 0.8


def test_ridge_clips_standardised_inputs_at_five():
    train, valid = split(planted_frame(4_000, planted=INPUTS[0], strength=0.5))
    model = fit_ridge(train, valid, INPUTS, "y_4", 4)[0]
    assert model[-1].coef_[0] > 0.1
    far, farther = valid.iloc[[0]].copy(), valid.iloc[[0]].copy()
    far[INPUTS[0]] = 50.0
    farther[INPUTS[0]] = 5_000.0
    np.testing.assert_array_equal(model.predict(far[INPUTS]), model.predict(farther[INPUTS]))


def test_lightgbm_is_deterministic_for_a_fixed_seed():
    train, valid = split(planted_frame(8_000, planted=INPUTS[2], strength=0.5))
    first = fit_lightgbm(train, valid, INPUTS, "y_4", 4)
    second = fit_lightgbm(train, valid, INPUTS, "y_4", 4)
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[0].predict(valid[INPUTS]), second[0].predict(valid[INPUTS]))
    assert first[2] == second[2] == f"trees={first[0].booster_.num_trees()}"


@pytest.mark.parametrize("name", FITTERS)
def test_fitters_predict_on_target_scale(name):
    train, valid = split(planted_frame(8_000, planted="x_trail_ret_4", strength=0.5))
    model, valid_pred, setting = FITTERS[name](train, valid, INPUTS, "y_4", 4)
    refit_pred = model.predict(valid[INPUTS])
    assert valid_pred.shape == refit_pred.shape == (len(valid),)
    assert setting
    assert spearmanr(valid_pred, valid["y_4"]).statistic > 0.2
    assert 0 < valid_pred.std() < valid["y_4"].std()
