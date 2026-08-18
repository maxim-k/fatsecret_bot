from __future__ import annotations

from datetime import datetime, time

import pytest

from match import (
    DAILY_VALUES,
    NoGramServingError,
    grams_per_unit,
    label_nutrients,
    lookup,
    meal_for,
    normalize,
    pick_serving,
    serving_grams,
    units_for,
)
from match import (
    plausible_range as match_plausible,
)
from models import Alias, Food, NutrientReading, NutritionLabel, Serving

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


def test_plausible_range_widens_when_evidence_is_thin() -> None:
    # Under five samples the margin doubles: mango logged twice at 145 and 188 g
    # must not make a real 289 g portion look like a misread.
    assert match_plausible([100.0, 200.0]) == (50.0, 400.0)
    assert match_plausible([100.0] * 5 + [200.0]) == (75.0, 250.0)
    # Always logged at 190 g still leaves room for ordinary variation, but not
    # for a trailing zero. This is the band bot._hold_reason questions against.
    low, high = match_plausible([190.0] * 5) or (0.0, 0.0)
    assert low <= 200.0 <= high
    assert not low <= 1900.0 <= high


def test_lookup_falls_back_to_token_containment() -> None:
    # The model wrote "bread" where it had previously written the full name.
    aliases = {
        "bread alone san francisco sourdough": _alias(),
        "wasa whole grain crispbread": _alias(),
    }
    assert lookup("bread", aliases) is not None
    assert lookup("Bread", aliases) is not None


def test_lookup_returns_none_when_two_aliases_match() -> None:
    aliases = {"greek yogurt plain": _alias(), "greek yogurt honey": _alias()}
    assert lookup("greek yogurt", aliases) is None


def test_lookup_prefers_an_exact_match() -> None:
    exact = Alias(food_id=7, food_name="Bread", serving_id=1, serving_grams=1.0)
    aliases = {"bread": exact, "bread alone san francisco sourdough": _alias()}
    assert lookup("bread", aliases) is exact


# --- nutrition labels -----------------------------------------------------------


def _label(**kwargs: object) -> NutritionLabel:
    base: dict[str, object] = {
        "name": "Super protein bar 23 g",
        "brand": "SuperCompany",
        "serving_description": "1 bar",
        "serving_amount": 60.0,
        "serving_unit": "g",
        "calories": 240.0,
        "fat": 11.0,
        "carbohydrate": 24.0,
        "protein": 23.0,
    }
    return NutritionLabel(**{**base, **kwargs})  # type: ignore[arg-type]


def _readings(**kwargs: float) -> list[NutrientReading]:
    """Nutrients the label gave only as a percentage."""
    return [NutrientReading(nutrient=n, percent_daily_value=p) for n, p in kwargs.items()]


def test_percent_daily_value_becomes_an_absolute_amount() -> None:
    label = _label(nutrients=_readings(calcium=10, iron=15, vitamin_d=20))
    nutrients = label_nutrients(label)
    assert nutrients["calcium"] == pytest.approx(130.0)  # 10% of 1300 mg
    assert nutrients["iron"] == pytest.approx(2.7)  # 15% of 18 mg
    assert nutrients["vitamin_d"] == pytest.approx(4.0)  # 20% of 20 mcg


def test_a_printed_amount_beats_a_percentage() -> None:
    # Label percentages are rounded to whole numbers; the milligrams beside
    # them are not, so the exact figure has to win.
    label = _label(
        nutrients=[NutrientReading(nutrient="calcium", amount=137.0, percent_daily_value=10)]
    )
    assert label_nutrients(label)["calcium"] == 137.0


def test_unknown_percent_keys_are_dropped_not_guessed() -> None:
    label = _label(nutrients=_readings(magnesium=20, riboflavin=15))
    nutrients = label_nutrients(label)
    assert "magnesium" not in nutrients
    assert "riboflavin" not in nutrients


def test_total_sugar_is_never_derived_from_a_percentage() -> None:
    # The 50 g Daily Value is for *added* sugars; FatSecret's field is total.
    assert "sugar" not in DAILY_VALUES
    assert "sugar" not in label_nutrients(_label(nutrients=_readings(sugar=40)))


def test_unstated_nutrients_are_omitted_rather_than_zeroed() -> None:
    # An absent field means "the label didn't say", which is not the same claim
    # as zero and must not be sent as one.
    nutrients = label_nutrients(_label())
    assert set(nutrients) == {"calories", "fat", "carbohydrate", "protein"}
    zeroed = _label(nutrients=[NutrientReading(nutrient="trans_fat", amount=0.0)])
    assert label_nutrients(zeroed)["trans_fat"] == 0.0


def test_serving_in_ounces_converts_to_grams() -> None:
    assert serving_grams(_label(serving_amount=1.0, serving_unit="oz")) == pytest.approx(28.3495)


def test_serving_in_millilitres_is_not_a_weight() -> None:
    # Treating 1 ml as 1 g misreports everything that is not water.
    assert serving_grams(_label(serving_amount=250.0, serving_unit="ml")) is None
    assert serving_grams(_label(serving_amount=None, serving_unit=None)) is None


def test_daily_values_match_the_fda_table() -> None:
    # fda.gov/food/nutrition-facts-label, adults and children 4+.
    assert DAILY_VALUES["fat"] == 78.0
    assert DAILY_VALUES["saturated_fat"] == 20.0
    assert DAILY_VALUES["carbohydrate"] == 275.0
    assert DAILY_VALUES["fiber"] == 28.0
    assert DAILY_VALUES["protein"] == 50.0
    assert DAILY_VALUES["cholesterol"] == 300.0
    assert DAILY_VALUES["sodium"] == 2300.0
    assert DAILY_VALUES["potassium"] == 4700.0
    assert DAILY_VALUES["calcium"] == 1300.0
    assert DAILY_VALUES["iron"] == 18.0
    assert DAILY_VALUES["vitamin_c"] == 90.0
    assert DAILY_VALUES["vitamin_a"] == 900.0
    assert DAILY_VALUES["vitamin_d"] == 20.0


def test_every_required_create_parameter_is_present() -> None:
    # food.create.v2 requires calories, fat, carbohydrate and protein. Leaving
    # calories out of the panel made the call fail with "Missing required
    # parameter" against the live API, which mocks did not catch.
    nutrients = label_nutrients(_label())
    for required in ("calories", "fat", "carbohydrate", "protein"):
        assert required in nutrients, required


def test_serving_unit_it_does_not_recognise_is_not_assumed_to_be_grams() -> None:
    # serving_unit is a plain string now, so the model can put anything there.
    # Guessing grams would misreport the food; "not a weight" is recoverable.
    assert serving_grams(_label(serving_amount=250.0, serving_unit="fl oz")) is None
    assert serving_grams(_label(serving_amount=100.0, serving_unit="")) is None
    # Spelling and case still resolve.
    assert serving_grams(_label(serving_amount=50.0, serving_unit="  Grams ")) == 50.0
    assert serving_grams(_label(serving_amount=1.0, serving_unit="OZ")) == pytest.approx(28.3495)
