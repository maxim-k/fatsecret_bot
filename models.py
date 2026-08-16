"""Shared data contract.

Every module exchanges these types and nothing else. They are defined in one
place because ``vision``, ``match``, ``diary`` and ``store`` are written and
changed independently; a shared contract is what keeps them composable.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

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


class DiaryEntry(BaseModel):
    """An entry as it exists in the FatSecret diary after submission."""

    food_entry_id: int
    food_id: int
    name: str
    day: date
