"""Product text -> FatSecret food, serving, and unit count.

Pure logic only: alias lookup and the gram-to-``number_of_units`` maths that
FatSecret's diary API demands in place of a raw weight. No I/O lives here so
this is the module most worth unit-testing hard.
"""

from __future__ import annotations

import re
from datetime import time

from models import Alias, Food, Serving

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

    FatSecret is not consistent here, and getting it wrong scales an entry by
    a factor of 100. For a natural-unit serving ("1 slice", 39 g) one unit is
    one slice, so units are multiples. For a plain metric serving ("100 g")
    one unit is a single **gram** — logging 2.89 units of the "100 g" mango
    serving records 2.89 g, not 289 g. Verified against the live API.

    :param serving: The serving an entry will be logged against.
    :returns: Grams represented by one unit.
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
    """Build a canonical alias key from free-form product text.

    Deliberately dumb: lowercasing, trimming, and whitespace collapsing only.
    No stemming, fuzzy matching, or translation — those would make alias
    lookups unpredictable, which defeats the point of an alias cache.

    :param product: Raw product text, e.g. as read from a photo or typed by
        the user.
    :returns: Canonical key suitable for use in an alias dict.
    """
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

    :param product: Raw product text to resolve.
    :param aliases: Known aliases, keyed by :func:`normalize`-d product text.
    :returns: The matching alias, or ``None`` if unknown or ambiguous.
    """
    key = normalize(product)
    exact = aliases.get(key)
    if exact is not None:
        return exact
    matches = [aliases[k] for k in candidates(key, aliases)]
    return matches[0] if len(matches) == 1 else None


def candidates(key: str, aliases: dict[str, Alias]) -> list[str]:
    """Alias keys whose tokens subsume, or are subsumed by, ``key``.

    :param key: A :func:`normalize`-d product string.
    :param aliases: Known aliases.
    :returns: Matching alias keys, empty if none.
    """
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

    :param serving_grams: Weight in grams of a single serving. Must be
        positive.
    :param grams: Weight actually measured. Must be positive.
    :returns: ``grams / serving_grams``, rounded to 3 decimal places — far
        finer than a kitchen scale is accurate to, but enough to keep
        FatSecret's stored calorie total from drifting off by rounding error.
    :raises ValueError: If either argument is not a positive number.
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

    :param food: Candidate food, as returned by a FatSecret food lookup.
    :param grams: Weight actually measured on the scale. Must be positive.
    :returns: The chosen serving together with the ``number_of_units`` to
        log it at.
    :raises NoGramServingError: If ``food`` exposes no serving whose
        ``metric_unit`` is grams with a positive ``metric_amount``.
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


def meal_for(when: time) -> str:
    """Map a local time of day onto a FatSecret meal.

    FatSecret only accepts breakfast, lunch, dinner and other, so the small
    hours land in "other", which its apps show as Snack.

    :param when: Local time the photo was taken.
    :returns: One of ``breakfast``, ``lunch``, ``dinner``, ``other``.
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

    The observed range with a quarter's slack at each end. Wide enough that
    ordinary variation logs without a prompt, tight enough that a trailing zero
    misread off the scale never does.

    :param history: Weights previously logged for this food.
    :returns: Inclusive ``(low, high)`` bounds, or ``None`` if there is no
        history to judge against.
    """
    if not history:
        return None
    # A handful of samples is not the range of portions this food can take, so
    # the margin starts loose and tightens as evidence accumulates. Mango logged
    # twice at 145 and 188 g must not make a real 289 g portion an anomaly.
    low, high = (0.5, 2.0) if len(history) < 5 else (0.75, 1.25)
    return min(history) * low, max(history) * high


def is_outlier(grams: float, history: list[float]) -> bool:
    """Judge whether a weight falls outside what this food normally weighs.

    Being asked about a good reading costs one tap; logging a bad one costs
    silent wrong data, so this errs towards asking.

    :param grams: The newly read weight.
    :param history: Weights previously logged for this food.
    :returns: ``True`` if the weight should be questioned rather than logged.
    """
    bounds = plausible_range(history)
    return bounds is not None and not bounds[0] <= grams <= bounds[1]
