"""Product text -> FatSecret food, serving, and unit count.

Pure logic only: alias lookup and the gram-to-``number_of_units`` maths that
FatSecret's diary API demands in place of a raw weight. No I/O lives here so
this is the module most worth unit-testing hard.
"""

from __future__ import annotations

import re

from models import Alias, Food, Serving

# Collapse any run of whitespace (spaces, tabs, newlines) to a single space so
# "eggs", " eggs", "eggs " and "eggs  " all normalize identically.
_WHITESPACE_RE = re.compile(r"\s+")

# FatSecret's metric_unit values are lowercase and unadorned in practice, but
# be liberal about surrounding whitespace/case since the field comes from a
# third-party API response we don't control.
_GRAMS_UNIT = "g"


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
    """Resolve product text to a learned alias by exact normalized match.

    :param product: Raw product text to resolve.
    :param aliases: Known aliases, keyed by :func:`normalize`-d product text.
    :returns: The matching alias, or ``None`` if this product was never
        taught.
    """
    return aliases.get(normalize(product))


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
    for serving in food.servings:
        if serving.metric_unit is None or serving.metric_amount is None:
            continue
        if serving.metric_unit.strip().lower() != _GRAMS_UNIT:
            continue
        if serving.metric_amount <= 0:
            continue
        return serving, units_for(serving.metric_amount, grams)

    raise NoGramServingError(f"food {food.food_id!r} ({food.name!r}) has no gram-based serving")
