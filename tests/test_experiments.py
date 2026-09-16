import numpy as np
import pandas as pd
from conftest import cross_join, day_index

from walkforward.config import HORIZONS, PAIRED_T, POOLED_T, RESEARCH_MONTHS
from walkforward.experiments.phase1 import (
    FAMILIES,
    POLICIES,
    choose_inputs,
    choose_policy,
    choose_recipe,
    retraining_by_month,
)


def pooled_recipes(t: list[float], breakeven: list[float]) -> pd.DataFrame:
    pooled = cross_join(horizon=HORIZONS[:2], family=FAMILIES[:2])
    pooled["t"] = t
    pooled["breakeven_bps"] = breakeven
    pooled["detected"] = pooled["t"] >= POOLED_T
    return pooled


def test_recipe_has_highest_breakeven_among_detected():
    pooled = pooled_recipes(t=[8.0, 3.0, 2.9, 1.0], breakeven=[0.5, 0.7, 2.0, 0.1])
    chosen = choose_recipe(pooled)
    assert (chosen["horizon"], chosen["family"]) == (HORIZONS[0], FAMILIES[1])


def test_without_a_detection_the_recipe_has_the_highest_t():
    pooled = pooled_recipes(t=[1.0, 2.9, 2.5, -4.0], breakeven=[3.0, 0.1, 0.2, 5.0])
    chosen = choose_recipe(pooled)
    assert (chosen["horizon"], chosen["family"]) == (HORIZONS[0], FAMILIES[1])


def test_own_inputs_need_pooled_t_of_two():
    assert choose_inputs({"t": PAIRED_T}) == "own"
    assert choose_inputs({"t": PAIRED_T - 0.01}) == "all"
    assert choose_inputs({"t": float("nan")}) == "all"


def test_expanding_stays_unless_a_challenger_reaches_t_two():
    assert choose_policy({"fixed": {"t": 1.9}, "rolling_3m": {"t": -3.0}}) == "expanding"
    assert choose_policy({"fixed": {"t": 2.1}, "rolling_3m": {"t": 2.6}}) == "rolling_3m"


def test_every_policy_gets_the_position_of_its_research_month():
    days = day_index(182)
    daily = {
        (coin, policy): pd.Series(np.arange(182.0), index=days)
        for coin in ("hype", "trx")
        for policy in POLICIES
    }
    table = retraining_by_month(daily)
    assert list(table.columns) == ["coin", "policy", "month", "research_month", "ic"]
    for _, rows in table.groupby(["coin", "policy"]):
        assert rows["month"].tolist() == list(RESEARCH_MONTHS)
        assert rows["research_month"].tolist() == [1, 2, 3, 4, 5, 6]
