from __future__ import annotations

from datetime import datetime, time

import pytest

from match import (
    NoGramServingError,
    grams_per_unit,
    is_outlier,
    lookup,
    meal_for,
    normalize,
    pick_serving,
    units_for,
)
from match import (
    plausible_range as match_plausible,
)
from models import Alias, Food, Serving

# --- normalize -----------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["Eggs", "Eggs ", "  eggs", "EGGS", " EggS ", "eggs\n"],
)
def test_normalize_equivalences(raw: str) -> None:
    assert normalize(raw) == "eggs"


def test_normalize_collapses_internal_whitespace() -> None:
    assert normalize("egg   whites\tboiled") == "egg whites boiled"


def test_normalize_is_idempotent() -> None:
    once = normalize("  Chicken Breast  ")
    assert normalize(once) == once


# --- lookup ----------------------------------------------------------------


def _alias(food_id: int = 1) -> Alias:
    return Alias(
        food_id=food_id,
        food_name="Eggs, whole",
        serving_id=100,
        serving_grams=50.0,
    )


def test_lookup_hit_exact() -> None:
    aliases = {"eggs": _alias()}
    assert lookup("eggs", aliases) is aliases["eggs"]


def test_lookup_hit_after_normalization() -> None:
    aliases = {"eggs": _alias()}
    assert lookup("  Eggs ", aliases) is aliases["eggs"]


def test_lookup_miss() -> None:
    aliases = {"eggs": _alias()}
    assert lookup("bacon", aliases) is None


def test_lookup_empty_aliases() -> None:
    assert lookup("anything", {}) is None


# --- units_for ---------------------------------------------------------


def test_units_for_100g_serving() -> None:
    assert units_for(100.0, 123.0) == 1.23


def test_units_for_non_100g_serving() -> None:
    assert units_for(50.0, 125.0) == 2.5


def test_units_for_rounds_to_three_decimals() -> None:
    # 1 / 3 g servings -> repeating decimal, must round rather than explode.
    assert units_for(3.0, 1.0) == 0.333


def test_units_for_zero_serving_grams_raises() -> None:
    with pytest.raises(ValueError, match="serving_grams"):
        units_for(0.0, 100.0)


def test_units_for_negative_serving_grams_raises() -> None:
    with pytest.raises(ValueError, match="serving_grams"):
        units_for(-10.0, 100.0)


def test_units_for_zero_grams_raises() -> None:
    with pytest.raises(ValueError, match="grams"):
        units_for(100.0, 0.0)


def test_units_for_negative_grams_raises() -> None:
    with pytest.raises(ValueError, match="grams"):
        units_for(100.0, -5.0)


# --- pick_serving --------------------------------------------------------


def test_pick_serving_prefers_gram_serving() -> None:
    food = Food(
        food_id=1,
        name="Milk",
        servings=[
            Serving(serving_id=1, description="1 cup", metric_amount=240.0, metric_unit="ml"),
            Serving(serving_id=2, description="100 g", metric_amount=100.0, metric_unit="g"),
        ],
    )
    serving, units = pick_serving(food, 250.0)
    assert serving.serving_id == 2
    assert units == 250.0


def test_pick_serving_scales_non_100g_serving() -> None:
    food = Food(
        food_id=1,
        name="Cheese",
        servings=[
            Serving(serving_id=1, description="1 slice", metric_amount=20.0, metric_unit="g"),
        ],
    )
    serving, units = pick_serving(food, 90.0)
    assert serving.serving_id == 1
    assert units == 4.5


def test_pick_serving_rejects_ml_only() -> None:
    food = Food(
        food_id=1,
        name="Milk",
        servings=[
            Serving(serving_id=1, description="1 cup", metric_amount=240.0, metric_unit="ml"),
        ],
    )
    with pytest.raises(NoGramServingError):
        pick_serving(food, 250.0)


def test_pick_serving_no_metric_servings_raises() -> None:
    food = Food(
        food_id=1,
        name="Protein Bar",
        servings=[
            Serving(serving_id=1, description="1 bar", metric_amount=None, metric_unit=None),
        ],
    )
    with pytest.raises(NoGramServingError):
        pick_serving(food, 60.0)


def test_pick_serving_no_servings_at_all_raises() -> None:
    food = Food(food_id=1, name="Mystery Food", servings=[])
    with pytest.raises(NoGramServingError):
        pick_serving(food, 60.0)


def test_pick_serving_skips_non_positive_metric_amount() -> None:
    food = Food(
        food_id=1,
        name="Weird Food",
        servings=[
            Serving(serving_id=1, description="broken", metric_amount=0.0, metric_unit="g"),
            Serving(serving_id=2, description="100 g", metric_amount=100.0, metric_unit="g"),
        ],
    )
    serving, units = pick_serving(food, 100.0)
    assert serving.serving_id == 2
    assert units == 100.0


def test_pick_serving_error_message_names_food() -> None:
    food = Food(food_id=42, name="Protein Bar", servings=[])
    with pytest.raises(NoGramServingError, match="42"):
        pick_serving(food, 60.0)


def test_pick_serving_grams_unit_case_insensitive() -> None:
    food = Food(
        food_id=1,
        name="Rice",
        servings=[
            Serving(serving_id=1, description="100 g", metric_amount=100.0, metric_unit="G"),
        ],
    )
    serving, units = pick_serving(food, 200.0)
    assert serving.serving_id == 1
    assert units == 200.0


# --- alias round trip, as end-to-end sanity ------------------------------


def test_alias_serving_grams_feeds_units_for() -> None:
    alias = Alias(
        food_id=7,
        food_name="Oats",
        serving_id=200,
        serving_grams=40.0,
        hits=3,
        last_used=datetime(2026, 1, 1),
    )
    assert units_for(alias.serving_grams, 100.0) == 2.5


def test_pick_serving_prefers_plain_100g_over_natural_units() -> None:
    # A weighed portion reads better as "1.23 x 100 g" than "2.46 x 1 large".
    food = Food(
        food_id=3092,
        name="Egg",
        servings=[
            Serving(serving_id=11206, description="1 large", metric_amount=50.0, metric_unit="g"),
            Serving(serving_id=51772, description="100 g", metric_amount=100.0, metric_unit="g"),
        ],
    )
    serving, units = pick_serving(food, 123.0)
    assert serving.serving_id == 51772
    assert units == 123.0


def test_pick_serving_falls_back_to_any_gram_serving() -> None:
    food = Food(
        food_id=1,
        name="Bar",
        servings=[Serving(serving_id=9, description="1 bar", metric_amount=40.0, metric_unit="g")],
    )
    serving, units = pick_serving(food, 80.0)
    assert serving.serving_id == 9
    assert units == 2.0


def test_grams_per_unit_metric_serving_is_one_gram() -> None:
    # Live-verified: 2.89 units of Mango's "100 g" serving logged 2.89 g, not 289 g.
    metric = Serving(serving_id=54116, description="100 g", metric_amount=100.0, metric_unit="g")
    assert grams_per_unit(metric) == 1.0
    natural = Serving(serving_id=1, description="1 slice", metric_amount=39.0, metric_unit="g")
    assert grams_per_unit(natural) == 39.0


def test_pick_serving_metric_units_equal_grams() -> None:
    food = Food(
        food_id=5436,
        name="Mango",
        servings=[
            Serving(serving_id=20463, description="1 mango", metric_amount=207.0, metric_unit="g"),
            Serving(serving_id=54116, description="100 g", metric_amount=100.0, metric_unit="g"),
        ],
    )
    serving, units = pick_serving(food, 289.0)
    assert serving.serving_id == 54116
    assert units == 289.0


def test_meal_for_time_of_day() -> None:
    assert meal_for(time(6, 0)) == "breakfast"
    assert meal_for(time(11, 59)) == "breakfast"
    assert meal_for(time(12, 0)) == "lunch"
    assert meal_for(time(17, 59)) == "lunch"
    assert meal_for(time(18, 0)) == "dinner"
    assert meal_for(time(23, 59)) == "dinner"
    assert meal_for(time(0, 0)) == "other"
    assert meal_for(time(5, 59)) == "other"


def test_is_outlier_catches_a_trailing_zero_misread() -> None:
    # 56 g of bread read as 560 g: the failure this exists for.
    assert is_outlier(560.0, [56.0])
    assert not is_outlier(56.0, [56.0])


def test_is_outlier_allows_a_wide_but_real_history() -> None:
    # Tomatoes genuinely span 25-188 g; the band widens with the spread.
    tomatoes = [25.0, 40.0, 60.0, 90.0, 188.0]
    assert not is_outlier(30.0, tomatoes)
    assert not is_outlier(180.0, tomatoes)
    assert is_outlier(900.0, tomatoes)


def test_plausible_range_is_the_observed_range_plus_a_quarter() -> None:
    assert match_plausible([100.0, 200.0]) == (75.0, 250.0)
    # Always logged at 190 g still leaves room for ordinary variation.
    assert not is_outlier(200.0, [190.0] * 5)
    assert is_outlier(1900.0, [190.0] * 5)


def test_is_outlier_allows_ordinary_variation() -> None:
    eggs = [120.0, 124.0, 124.0, 135.0]
    assert not is_outlier(117.0, eggs)
    assert is_outlier(1170.0, eggs)


def test_is_outlier_passes_everything_with_no_history() -> None:
    assert not is_outlier(999.0, [])
