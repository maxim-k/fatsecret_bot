"""Product text -> FatSecret food, serving, and unit count.

Pure logic only: alias lookup and the gram-to-``number_of_units`` maths that
FatSecret's diary API demands in place of a raw weight. No I/O lives here so
this is the module most worth unit-testing hard.
"""

from __future__ import annotations

import re
from datetime import time
from typing import Final

from models import Alias, Food, NutritionLabel, Serving

# Collapse any run of whitespace (spaces, tabs, newlines) to a single space so
# "eggs", " eggs", "eggs " and "eggs  " all normalize identically.
_WHITESPACE_RE = re.compile(r"\s+")

# FatSecret's metric_unit values are lowercase and unadorned in practice, but
# be liberal about surrounding whitespace/case since the field comes from a
# third-party API response we don't control.
_GRAMS_UNIT = "g"

# FatSecret exposes a plain "100 g" serving for most generic foods alongside
# natural-unit ones ("1 large"). Preferring it keeps diary entries readable.
_METRIC_SERVING_GRAMS = 100.0

# A metric serving's description is just the amount and unit, e.g. "100 g".
_METRIC_DESCRIPTION_RE = re.compile(r"^\s*\d+(?:\.\d+)?\s*g\s*$", re.IGNORECASE)


def grams_per_unit(serving: Serving) -> float:
    """How many grams one ``number_of_units`` step represents.

    FatSecret is not consistent here. Getting it wrong scales an entry by 100.
    For a natural-unit serving ("1 slice", 39 g) one unit is one slice. For a
    plain metric serving ("100 g") one unit is a single **gram**: 2.89 units of
    mango's "100 g" serving records 2.89 g, not 289 g. Verified against the
    live API.
    """
    if _METRIC_DESCRIPTION_RE.match(serving.description):
        return 1.0
    return serving.metric_amount or 0.0


class NoGramServingError(Exception):
    """Raised when a food has no serving that can be scaled by weight.

    Branded foods often expose only natural-unit servings ("1 bar", "1 can"),
    which have no metric weight to divide a gram amount by. The caller is
    expected to catch this and ask the user to pick a serving manually.
    """


def normalize(product: str) -> str:
    """Build a canonical alias key from free-form product text."""
    # Lowercase, trim, collapse whitespace, nothing else. Stemming or fuzzy
    # matching would make lookups unpredictable, which defeats an alias cache.
    return _WHITESPACE_RE.sub(" ", product.strip()).lower()


def lookup(product: str, aliases: dict[str, Alias]) -> Alias | None:
    """Resolve product text to a learned alias.

    Exact match first, then token containment. The model names the same food
    differently between runs — "Bread Alone San Francisco Sourdough" one time,
    "bread" the next — so keying on the exact string it happens to produce turns
    a food logged four times into a new one. Tokens are compared whole, so
    "bread" does not match "crispbread".

    Two or more candidates means the text is genuinely ambiguous; that is a
    question for the user, not a coin flip, so it resolves to ``None``.

    :param aliases: Keyed by :func:`normalize`-d product text.
    :returns: The matching alias, or ``None`` if unknown or ambiguous.
    """
    key = normalize(product)
    exact = aliases.get(key)
    if exact is not None:
        return exact
    matches = [aliases[k] for k in candidates(key, aliases)]
    return matches[0] if len(matches) == 1 else None


def candidates(key: str, aliases: dict[str, Alias]) -> list[str]:
    """Alias keys whose tokens subsume, or are subsumed by, a normalized ``key``."""
    wanted = set(key.split())
    if not wanted:
        return []
    return [
        alias_key
        for alias_key in aliases
        if wanted <= set(alias_key.split()) or set(alias_key.split()) <= wanted
    ]


def units_for(serving_grams: float, grams: float) -> float:
    """Compute how many servings ``grams`` amounts to.

    :returns: The ratio, rounded to 3 decimals. That is far finer than a
        kitchen scale reads. It keeps FatSecret's stored calorie total from
        drifting on rounding error.
    :raises ValueError: If either argument is not positive.
    """
    if serving_grams <= 0:
        raise ValueError(f"serving_grams must be positive, got {serving_grams!r}")
    if grams <= 0:
        raise ValueError(f"grams must be positive, got {grams!r}")
    return round(grams / serving_grams, 3)


def pick_serving(food: Food, grams: float) -> tuple[Serving, float]:
    """Choose the serving to log ``grams`` of ``food`` in.

    Only a gram-denominated serving can be scaled by weight. A serving in
    ``ml`` is a volume, not a mass — treating 1 ml as 1 g would silently
    misreport calories for anything that isn't water, so such servings are
    skipped rather than guessed at. Natural-unit servings ("1 bar") have no
    metric weight at all and are skipped for the same reason.

    :returns: The chosen serving, with the ``number_of_units`` to log it at.
    :raises NoGramServingError: If no serving has a gram ``metric_unit`` and a
        positive ``metric_amount``.
    """
    usable = [
        serving
        for serving in food.servings
        if serving.metric_unit is not None
        and serving.metric_amount is not None
        and serving.metric_amount > 0
        and serving.metric_unit.strip().lower() == _GRAMS_UNIT
    ]
    if not usable:
        raise NoGramServingError(f"food {food.food_id!r} ({food.name!r}) has no gram-based serving")

    # Any gram serving scales correctly, but the choice shows up in the diary the
    # user reads back: a weighed 123 g of egg reads far better as "1.23 x 100 g"
    # than as "2.46 x 1 large". Prefer a plain metric serving when one exists.
    chosen = next(
        (serving for serving in usable if serving.metric_amount == _METRIC_SERVING_GRAMS),
        usable[0],
    )
    return chosen, units_for(grams_per_unit(chosen), grams)


# The only four values FatSecret accepts. Anything else comes back as an
# opaque API error, so a meal named by hand is checked against this first.
MEALS: Final = frozenset({"breakfast", "lunch", "dinner", "other"})

# FDA Daily Values for adults and children 4+, as published at
# fda.gov/food/nutrition-facts-label (content current 2024-03-05) and set by
# 21 CFR 101.9. Units deliberately match both the label and FatSecret's
# food.create fields, so a percentage resolves with one multiplication and no
# unit conversion anywhere in between.
#
# Total sugar is absent on purpose: the 50 g Daily Value is for *added* sugars,
# while FatSecret's `sugar` field means total. Deriving one from the other would
# understate any product whose sugar is mostly intrinsic, and labels print total
# sugars in grams regardless, so there is nothing to derive.
DAILY_VALUES: Final[dict[str, float]] = {
    "fat": 78.0,  # g
    "saturated_fat": 20.0,  # g
    "carbohydrate": 275.0,  # g
    "fiber": 28.0,  # g
    "protein": 50.0,  # g
    "cholesterol": 300.0,  # mg
    "sodium": 2300.0,  # mg
    "potassium": 4700.0,  # mg
    "calcium": 1300.0,  # mg
    "iron": 18.0,  # mg
    "vitamin_c": 90.0,  # mg
    "vitamin_a": 900.0,  # mcg RAE
    "vitamin_d": 20.0,  # mcg
}

# The unit each nutrient is measured in, matching both the label and
# FatSecret's food.create fields. Kept beside DAILY_VALUES so the two cannot
# disagree about whether calcium is milligrams.
NUTRIENT_UNITS: Final[dict[str, str]] = {
    "fat": "g",
    "saturated_fat": "g",
    "polyunsaturated_fat": "g",
    "monounsaturated_fat": "g",
    "trans_fat": "g",
    "carbohydrate": "g",
    "fiber": "g",
    "sugar": "g",
    "protein": "g",
    "cholesterol": "mg",
    "sodium": "mg",
    "potassium": "mg",
    "calcium": "mg",
    "iron": "mg",
    "vitamin_c": "mg",
    "vitamin_a": "mcg",
    "vitamin_d": "mcg",
}

# One ounce in grams. Labels on imported goods sometimes give only ounces, and
# FatSecret would take "oz" directly, but everything downstream — the alias's
# serving_grams, the plausible-weight band, the scale itself — is in grams.
GRAMS_PER_OUNCE: Final = 28.349523125


def label_nutrients(label: NutritionLabel) -> dict[str, float]:
    """Flatten a label's panel into FatSecret's nutrient parameters.

    A printed amount always beats a percentage: label percentages are rounded to
    the nearest whole number, so 2% of the iron DV could be anything from 0.27
    to 0.45 mg, while the milligram figure beside it is exact.

    :returns: Nutrient name to per-serving value. Anything the label gave in
        neither form is left out.
    """
    values: dict[str, float] = {
        # Required by food.create.v2, and absent from DAILY_VALUES because the
        # 2,000 kcal reference is a diet, not a per-serving figure to derive.
        "calories": label.calories,
        "fat": label.fat,
        "carbohydrate": label.carbohydrate,
        "protein": label.protein,
    }
    for reading in label.nutrients:
        if reading.nutrient not in NUTRIENT_UNITS or reading.nutrient in values:
            continue  # not a field FatSecret takes, or already read off the macros
        if reading.amount is not None:
            values[reading.nutrient] = reading.amount
        elif reading.percent_daily_value is not None:
            daily = DAILY_VALUES.get(reading.nutrient)
            if daily is not None:
                values[reading.nutrient] = daily * reading.percent_daily_value / 100.0
    return values


def serving_grams(label: NutritionLabel) -> float | None:
    """Convert a label's serving quantity to grams.

    :returns: The serving in grams. ``None`` when the label gave no quantity,
        used millilitres, or used a unit not recognised here. A volume is not a
        weight. Treating one as the other misreports anything that is not
        water, so an unfamiliar unit is not assumed to be grams.
    """
    # A zero amount means the same as none: the schema cannot forbid it, so it
    # is rejected here.
    if not label.serving_amount or label.serving_unit is None:
        return None
    unit = label.serving_unit.strip().lower()
    if unit in ("g", "gram", "grams"):
        return label.serving_amount
    if unit in ("oz", "ounce", "ounces"):
        return label.serving_amount * GRAMS_PER_OUNCE
    return None


def meal_for(when: time) -> str:
    """Map a local time of day onto one of :data:`MEALS`.

    The small hours land in "other", which FatSecret's apps show as Snack.
    """
    if 6 <= when.hour < 12:
        return "breakfast"
    if 12 <= when.hour < 18:
        return "lunch"
    if 18 <= when.hour < 24:
        return "dinner"
    return "other"


def plausible_range(history: list[float]) -> tuple[float, float] | None:
    """Weights this food can be expected to weigh, from what it has weighed.

    The observed range with a quarter's slack at each end. That is wide enough
    to log ordinary variation without a prompt. It is tight enough to catch a
    trailing zero misread off the scale.

    :returns: Inclusive ``(low, high)``, or ``None`` with no history to judge.
    """
    if not history:
        return None
    # A handful of samples is not the range of portions this food can take, so
    # the margin starts loose and tightens as evidence accumulates. Mango logged
    # twice at 145 and 188 g must not make a real 289 g portion an anomaly.
    low, high = (0.5, 2.0) if len(history) < 5 else (0.75, 1.25)
    return min(history) * low, max(history) * high
