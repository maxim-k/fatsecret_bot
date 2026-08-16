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


def _item(
    product: str = "eggs", confidence: float = 0.95, weight_g: float = 123.0
) -> RecognizedItem:
    return RecognizedItem(
        product=product, weight_g=weight_g, raw_digits=str(int(weight_g)), confidence=confidence
    )


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
    assert "123" in text
    # The queue id is an implementation detail and must not leak into chat.
    assert "01234567" not in text


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


def test_concurrent_updates_stays_at_one() -> None:
    # on_photo does load_queue -> await vision -> save_queue. That read-modify-write
    # only holds while python-telegram-bot serialises updates; several photos sent
    # at once would otherwise overwrite each other's queue entries.
    from telegram.ext import ApplicationBuilder

    assert ApplicationBuilder().token("1:x").build().concurrent_updates == 1


def test_reread_hint_only_for_a_weight_outside_the_band() -> None:
    alias = _alias()
    alias.weights = [56.0]
    inband = QueuedItem(
        id="a" * 32,
        photo_sha256="x",
        day=date(2026, 8, 15),
        item=_item(weight_g=56.0),
        alias=alias,
    )
    assert bot._reread_hint(inband) is None

    outlier = QueuedItem(
        id="b" * 32,
        photo_sha256="x",
        day=date(2026, 8, 15),
        item=_item(weight_g=560.0),
        alias=alias,
    )
    hint = bot._reread_hint(outlier)
    assert hint is not None
    assert "560" in hint
    assert "trailing zero" in hint


def test_reread_hint_absent_without_alias_or_history() -> None:
    no_alias = QueuedItem(id="c" * 32, photo_sha256="x", day=date(2026, 8, 15), item=_item())
    assert bot._reread_hint(no_alias) is None

    no_history = QueuedItem(
        id="d" * 32,
        photo_sha256="x",
        day=date(2026, 8, 15),
        item=_item(weight_g=999.0),
        alias=_alias(),
    )
    assert bot._reread_hint(no_history) is None


def test_hold_reason_clears_once_the_weight_is_back_in_band() -> None:
    alias = _alias()
    alias.weights = [56.0]
    q = QueuedItem(
        id="e" * 32,
        photo_sha256="x",
        day=date(2026, 8, 15),
        item=_item(weight_g=560.0),
        alias=alias,
    )
    assert bot._hold_reason(q) is not None
    q.item.weight_g = 56.0
    assert bot._hold_reason(q) is None


@pytest.mark.parametrize(
    ("caption", "name", "weight"),
    [
        ("Avocado", "Avocado", None),
        ("56", None, 56.0),
        ("Avocado 56", "Avocado", 56.0),
        ("Avocado 56 g", "Avocado", 56.0),
        ("56,4", None, 56.4),
        ("  Greek Yogurt  ", "Greek Yogurt", None),
        ("Fiber One Fulfill Bars", "Fiber One Fulfill Bars", None),
    ],
)
def test_parse_correction(caption: str, name: str | None, weight: float | None) -> None:
    assert bot.parse_correction(caption) == (name, weight)


def test_caption_overrides_the_model_and_relooks_the_alias() -> None:
    store.save_aliases({"eggs": _alias()})
    queued = bot._queue_item(_item(product="bread"), "abc", date(2026, 8, 15), "lunch")
    assert queued.alias is None

    bot._apply_correction(queued, "eggs 200")
    assert queued.item.product == "eggs"
    assert queued.item.weight_g == 200.0
    assert queued.alias is not None
    assert queued.alias.food_name == "Eggs"


def test_correction_without_a_number_keeps_the_read_weight() -> None:
    queued = bot._queue_item(_item(weight_g=123.0), "abc", date(2026, 8, 15), "lunch")
    bot._apply_correction(queued, "Avocado")
    assert queued.item.product == "Avocado"
    assert queued.item.weight_g == 123.0


def _route(callback_data: str) -> tuple[str, str, list[str]]:
    """Mirror on_button's parsing so routing is tested, not just its pieces."""
    action, item_id, *extra = callback_data.split(":")
    return action, item_id, extra


def test_callback_routing_resolves_every_action() -> None:
    # "pick:ID:FOOD" used to be split on the first colon only, so the item id
    # came out as "ID:FOOD", never matched, and every pick answered
    # "That item is gone." The pieces were tested; the routing was not.
    item_id = "a" * 32
    assert _route(f"ok:{item_id}") == ("ok", item_id, [])
    assert _route(f"no:{item_id}") == ("no", item_id, [])
    assert _route(f"pick:{item_id}:66325") == ("pick", item_id, ["66325"])


def test_pick_callback_fits_telegram_limit() -> None:
    # Telegram rejects callback_data over 64 bytes; a 32-char id plus an 8-digit
    # food id has to fit or the picker silently breaks in production.
    assert len(f"pick:{'a' * 32}:{99999999}".encode()) <= 64


def test_submit_replaces_a_corrected_weight_instead_of_stacking() -> None:
    alias = _alias()
    alias.weights = [120.0, 108.0]
    store.save_aliases({"eggs": alias})
    queued = QueuedItem(
        id="f" * 32,
        photo_sha256="x",
        day=date(2026, 8, 15),
        item=_item(weight_g=118.0),
        alias=alias,
    )
    # The wrong 108 must leave the history, not sit beside the correction.
    weights = store.load_aliases()["eggs"].weights
    corrected = [w for w in weights if w != 108.0] + [queued.item.weight_g]
    assert corrected == [120.0, 118.0]


def test_escalation_fires_on_a_shaky_reading() -> None:
    alias = _alias()
    alias.weights = [120.0, 124.0]
    shaky = QueuedItem(
        id="g" * 32,
        photo_sha256="x",
        day=date(2026, 8, 15),
        item=_item(weight_g=122.0, confidence=0.3),
        alias=alias,
    )
    assert bot._reread_hint(shaky) is not None

    confident = QueuedItem(
        id="h" * 32,
        photo_sha256="x",
        day=date(2026, 8, 15),
        item=_item(weight_g=122.0, confidence=0.9),
        alias=alias,
    )
    assert bot._reread_hint(confident) is None
