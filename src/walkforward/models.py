from dataclasses import dataclass
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from walkforward.config import (
    EARLY_STOPPING_ROUNDS,
    LIGHTGBM_PARAMS,
    RIDGE_ALPHAS,
    ROWS_PER_LEAF_PER_HORIZON_BAR,
)

Fitter = Callable[[pd.DataFrame, pd.DataFrame, list[str], str, int], tuple[Any, np.ndarray, str]]


@dataclass(frozen=True)
class LinearRule:
    column: str
    intercept: float
    slope: float

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.intercept + self.slope * X[self.column].to_numpy()


def fit_line(rows: pd.DataFrame, column: str, label: str) -> LinearRule:
    slope, intercept = np.polyfit(rows[column], rows[label], deg=1)
    return LinearRule(column, float(intercept), float(slope))


def most_correlated_input(rows: pd.DataFrame, inputs: list[str], label: str) -> str:
    return rows[inputs].corrwith(rows[label], method="spearman").abs().idxmax()


def fit_momentum(
    train: pd.DataFrame, valid: pd.DataFrame, inputs: list[str], label: str, horizon: int
) -> tuple[LinearRule, np.ndarray, str]:
    column = f"x_trail_ret_{horizon}"
    rule = fit_line(pd.concat([train, valid]), column, label)
    return rule, fit_line(train, column, label).predict(valid), f"slope={rule.slope:+.3g}"


def fit_best_feature(
    train: pd.DataFrame, valid: pd.DataFrame, inputs: list[str], label: str, horizon: int
) -> tuple[LinearRule, np.ndarray, str]:
    inner = fit_line(train, most_correlated_input(train, inputs, label), label)
    both = pd.concat([train, valid])
    rule = fit_line(both, most_correlated_input(both, inputs, label), label)
    return rule, inner.predict(valid), rule.column


def ridge_pipeline(alpha: float) -> Pipeline:
    clip = FunctionTransformer(np.clip, kw_args={"a_min": -5.0, "a_max": 5.0})
    return make_pipeline(StandardScaler(), clip, Ridge(alpha=alpha))


def fit_ridge(
    train: pd.DataFrame, valid: pd.DataFrame, inputs: list[str], label: str, horizon: int
) -> tuple[Pipeline, np.ndarray, str]:
    X_train, X_valid = train[inputs], valid[inputs]
    valid_preds = {
        alpha: ridge_pipeline(alpha).fit(X_train, train[label]).predict(X_valid)
        for alpha in RIDGE_ALPHAS
    }
    alpha = min(RIDGE_ALPHAS, key=lambda a: mean_squared_error(valid[label], valid_preds[a]))
    both = pd.concat([train, valid])
    model = ridge_pipeline(alpha).fit(both[inputs], both[label])
    return model, valid_preds[alpha], f"alpha={alpha:.0e}"


def fit_lightgbm(
    train: pd.DataFrame, valid: pd.DataFrame, inputs: list[str], label: str, horizon: int
) -> tuple[lgb.LGBMRegressor, np.ndarray, str]:
    params = LIGHTGBM_PARAMS | {"min_child_samples": ROWS_PER_LEAF_PER_HORIZON_BAR * horizon}
    inner = lgb.LGBMRegressor(**params).fit(
        train[inputs],
        train[label],
        eval_X=valid[inputs],
        eval_y=valid[label],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    trees = inner.best_iteration_
    valid_pred = inner.predict(valid[inputs])
    both = pd.concat([train, valid])
    if np.ptp(valid_pred) == 0:
        # inner model never split: no model this month
        return LinearRule(inputs[0], float(both[label].mean()), 0.0), valid_pred, "trees=0"
    model = lgb.LGBMRegressor(**params | {"n_estimators": trees}).fit(both[inputs], both[label])
    return model, valid_pred, f"trees={trees}"


FITTERS: dict[str, Fitter] = {
    "momentum": fit_momentum,
    "best_feature": fit_best_feature,
    "ridge": fit_ridge,
    "lightgbm": fit_lightgbm,
}
