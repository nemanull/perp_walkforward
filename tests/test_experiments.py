import pandas as pd
from conftest import cross_join

from walkforward.config import HORIZONS, POOLED_T
from walkforward.experiments.phase1 import FAMILIES, choose_recipe


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
