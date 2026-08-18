"""Shared data contract.

Every module exchanges these types and nothing else. They are defined in one
place because ``vision``, ``match``, ``diary`` and ``store`` are written and
changed independently; a shared contract is what keeps them composable.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

# FatSecret counts diary days as whole days since the Unix epoch, not as a
# timestamp. Converting at the boundary keeps the rest of the code on real dates.
EPOCH = date(1970, 1, 1)


def to_epoch_days(day: date) -> int:
    """Convert a calendar date to FatSecret's day number.

    :param day: Calendar date of the diary entry.
    :returns: Whole days elapsed since 1970-01-01.
    """
    return (day - EPOCH).days


class ItemState(StrEnum):
    """Lifecycle of a single recognised food item."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    DISCARDED = "discarded"


class RecognizedItem(BaseModel):
    """One food item read out of a photo.

    ``raw_digits`` is what the model literally saw on the scale display, kept
    separate from the parsed weight so a misread digit stays auditable rather
    than silently becoming a wrong calorie count.
    """

    product: str
    weight_g: float = Field(gt=0, le=5000)
    raw_digits: str
    confidence: float = Field(ge=0, le=1)


class Recognition(BaseModel):
    """Everything one photo yielded. A plate may hold several items."""

    items: list[RecognizedItem]


class NutritionLabel(BaseModel):
    """A packaged product's identity and panel, read off its photos.

    Field names match ``food.create.v2``'s parameters exactly, and so do the
    units — grams for the macros, milligrams for minerals, micrograms for
    vitamins A and D. Keeping one set of names means the panel travels from the
    model to the API without a translation table in between, which is one fewer
    place to map ``carbs`` onto ``carbohydrate`` and be silently wrong.

    Every value is **per serving**, which is how ``food.create.v2`` reads them
    and how the label prints them.
    """

    name: str
    brand: str | None = None
    # The serving as the label words it: "1 bar", "2 cookies", "1 cup".
    serving_description: str
    # The same serving as a quantity. Reported in whatever unit is printed;
    # ounces are converted to grams downstream rather than by the model, since
    # arithmetic is the part of reading a label it is least reliable at.
    serving_amount: float | None = Field(default=None, gt=0)
    serving_unit: Literal["g", "ml", "oz"] | None = None

    calories: float = Field(ge=0)
    fat: float = Field(ge=0)
    carbohydrate: float = Field(ge=0)
    protein: float = Field(ge=0)

    saturated_fat: float | None = Field(default=None, ge=0)
    polyunsaturated_fat: float | None = Field(default=None, ge=0)
    monounsaturated_fat: float | None = Field(default=None, ge=0)
    trans_fat: float | None = Field(default=None, ge=0)
    cholesterol: float | None = Field(default=None, ge=0)
    sodium: float | None = Field(default=None, ge=0)
    potassium: float | None = Field(default=None, ge=0)
    fiber: float | None = Field(default=None, ge=0)
    sugar: float | None = Field(default=None, ge=0)
    vitamin_d: float | None = Field(default=None, ge=0)
    vitamin_a: float | None = Field(default=None, ge=0)
    vitamin_c: float | None = Field(default=None, ge=0)
    calcium: float | None = Field(default=None, ge=0)
    iron: float | None = Field(default=None, ge=0)

    # Nutrients the label gave only as a percentage of Daily Value, keyed by the
    # field names above. Older labels print "10% calcium" and no milligrams at
    # all. Resolved against the FDA table by ``match.label_nutrients`` rather
    # than by the model, because a lookup cannot misremember a constant.
    percent_daily_value: dict[str, float] = Field(default_factory=dict)


class Serving(BaseModel):
    """A FatSecret serving option for a food.

    ``metric_amount``/``metric_unit`` are optional because branded foods often
    describe servings only in natural units ("1 bar"), which cannot be scaled
    by weight.
    """

    serving_id: int
    description: str
    metric_amount: float | None = None
    metric_unit: str | None = None


class Food(BaseModel):
    """A FatSecret food and the servings it can be logged in."""

    food_id: int
    name: str
    servings: list[Serving]
    # Both come free from foods.search and are what makes one candidate
    # distinguishable from another when choosing: "Per 100g - Calories: 160kcal
    # | Fat: 14.66g | Carbs: 8.53g | Protein: 2.00g".
    description: str | None = None
    brand: str | None = None


class Alias(BaseModel):
    """A learned mapping from spoken product text to a FatSecret food.

    This is the core of the bot: once taught, a repeat food resolves with no
    interaction. ``serving_grams`` is cached so the gram-to-units conversion
    never needs another API round trip.
    """

    food_id: int
    food_name: str
    serving_id: int
    serving_grams: float = Field(gt=0)
    hits: int = 0
    last_used: datetime | None = None
    # Past logged weights for this food. A trailing zero misread off the scale
    # produces a number nothing like any portion the user has ever eaten.
    weights: list[float] = Field(default_factory=list)


class QueuedItem(BaseModel):
    """A recognised item awaiting review or submission.

    ``photo_sha256`` doubles as the deduplication key, so re-sending the same
    photo cannot double-log calories.
    """

    id: str
    photo_sha256: str
    day: date
    meal: str = "other"
    item: RecognizedItem
    alias: Alias | None = None
    state: ItemState = ItemState.PENDING
    # Telegram id of the message describing this item, so a reply to it can be
    # traced back here without printing a queue id at the user.
    message_id: int | None = None
    # The FatSecret entry this item became, needed to delete and replace it when
    # a logged weight turns out to be wrong.
    food_entry_id: int | None = None


class DiaryEntry(BaseModel):
    """An entry as it exists in the FatSecret diary after submission."""

    food_entry_id: int
    food_id: int
    name: str
    day: date
    # The diary read already returns each entry's macros scaled to the portion
    # that was logged, so showing them costs no extra API calls. Defaulted
    # rather than required because a food from an odd dataset can omit one, and
    # a missing carb count is no reason to fail the whole day's listing.
    calories: float = 0.0
    fat: float = 0.0
    carbs: float = 0.0
    protein: float = 0.0
