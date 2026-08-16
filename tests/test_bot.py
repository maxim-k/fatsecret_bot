from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

import bot
import store
from models import Alias, ItemState, QueuedItem, RecognizedItem


@pytest.fixture(autouse=True)
def _data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))


def _item(product: str = "eggs", confidence: float = 0.95) -> RecognizedItem:
    return RecognizedItem(product=product, weight_g=123.0, raw_digits="123", confidence=confidence)


def _alias() -> Alias:
    return Alias(food_id=1, food_name="Eggs", serving_id=2, serving_grams=100.0)


def test_queue_item_resolves_a_known_alias() -> None:
    store.save_aliases({"eggs": _alias()})
    queued = bot._queue_item(_item(), "abc", date(2026, 8, 15), "lunch")
    assert queued.alias is not None
    assert queued.alias.food_name == "Eggs"
    assert queued.state == ItemState.PENDING


def test_queue_item_leaves_unknown_product_unresolved() -> None:
    queued = bot._queue_item(_item(product="dragonfruit"), "abc", date(2026, 8, 15), "lunch")
    assert queued.alias is None


def test_queue_item_normalises_before_lookup() -> None:
    store.save_aliases({"eggs": _alias()})
    queued = bot._queue_item(_item(product="  EGGS  "), "abc", date(2026, 8, 15), "lunch")
    assert queued.alias is not None


def test_describe_surfaces_raw_digits_so_a_misread_is_visible() -> None:
    queued = QueuedItem(
        id="0123456789abcdef",
        photo_sha256="abc",
        day=date(2026, 8, 15),
        item=_item(),
        alias=_alias(),
    )
    text = bot._describe(queued)
    assert "01234567" in text
    assert "123" in text
    assert "95%" in text


def test_describe_falls_back_to_recognised_product_when_unmatched() -> None:
    queued = QueuedItem(
        id="0123456789abcdef",
        photo_sha256="abc",
        day=date(2026, 8, 15),
        item=_item(product="dragonfruit"),
    )
    assert "dragonfruit" in bot._describe(queued)


def test_owner_id_rejects_missing_or_non_numeric(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TELEGRAM_OWNER_ID", raising=False)
    with pytest.raises(RuntimeError):
        bot._owner_id()
    monkeypatch.setenv("TELEGRAM_OWNER_ID", "not-a-number")
    with pytest.raises(RuntimeError):
        bot._owner_id()


def test_auto_submit_threshold_is_below_one() -> None:
    # A threshold of exactly 1.0 would never fire; 0.0 would submit unread weights.
    assert 0.0 < bot.AUTO_SUBMIT_CONFIDENCE < 1.0
