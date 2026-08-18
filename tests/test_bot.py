from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

import bot
import match
import store
from models import Alias, DiaryEntry, Food, ItemState, QueuedItem, RecognizedItem


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


# --- /backlog directive parsing -------------------------------------------------


def test_backlog_parses_a_full_date_and_meal() -> None:
    assert bot.parse_backlog("/backlog 15/08/2026 lunch") == (date(2026, 8, 15), "lunch")


def test_backlog_reads_day_before_month_not_the_us_order() -> None:
    # 13/08 can only be the 13th of August; 08/13 is not a valid DD/MM and must
    # be refused rather than silently read as August 13th.
    assert bot.parse_backlog("/backlog 13/08/2026")[0] == date(2026, 8, 13)
    with pytest.raises(ValueError):
        bot.parse_backlog("/backlog 08/13/2026")


def test_backlog_fills_in_the_current_year() -> None:
    day, _meal = bot.parse_backlog("/backlog 15/08")
    assert (day.day, day.month) == (15, 8)
    assert day.year == datetime.now().astimezone().year


def test_backlog_accepts_today_and_yesterday() -> None:
    today = datetime.now().astimezone().date()
    assert bot.parse_backlog("/backlog today")[0] == today
    assert bot.parse_backlog("/backlog yesterday")[0] == today - timedelta(days=1)


def test_backlog_defaults_to_other_rather_than_the_time_of_sending() -> None:
    # The photo is of a past meal, so when it was sent says nothing about which
    # meal it was.
    assert bot.parse_backlog("/backlog 15/08/2026")[1] == "other"


@pytest.mark.parametrize("meal", sorted(match.MEALS))
def test_backlog_accepts_every_meal_fatsecret_takes(meal: str) -> None:
    assert bot.parse_backlog(f"/backlog 15/08/2026 {meal}")[1] == meal


@pytest.mark.parametrize(
    "caption",
    [
        "/backlog",
        "/backlog notadate lunch",
        "/backlog 32/13/2026",
        "/backlog 15/08/2026 brunch",
        "/backlog 15/08/2026 lunch and also eggs",
    ],
)
def test_backlog_refuses_what_it_cannot_read(caption: str) -> None:
    # Refusing beats defaulting to today: a whole batch would land on the wrong
    # day without saying so.
    with pytest.raises(ValueError):
        bot.parse_backlog(caption)


@pytest.mark.parametrize("caption", ["Avocado", "56", "Avocado 56", "Avocado 56 g", ""])
def test_backlog_ignores_ordinary_captions(caption: str) -> None:
    # These belong to parse_correction and must reach it untouched.
    assert bot.parse_backlog(caption) is None


# --- album batching -------------------------------------------------------------


async def test_album_applies_one_directive_to_every_photo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, date, str, str]] = []

    async def _fake_process(
        _message: object, file_id: str, _media_type: str, day: date, meal: str, caption: str
    ) -> None:
        calls.append((file_id, day, meal, caption))

    monkeypatch.setattr(bot, "_process_photo", _fake_process)
    monkeypatch.setattr(bot, "_ALBUM_SETTLE", 0)

    # The caption lands on the second photo, as it does whenever it is typed on
    # anything but the first frame of the album.
    bot._albums["grp"] = bot._Album(
        day=date(2026, 8, 16),
        meal="dinner",
        photos=[
            bot._AlbumPhoto(object(), "a", "image/jpeg", ""),  # type: ignore[arg-type]
            bot._AlbumPhoto(object(), "b", "image/jpeg", ""),  # type: ignore[arg-type]
            bot._AlbumPhoto(object(), "c", "image/jpeg", ""),  # type: ignore[arg-type]
        ],
    )
    bot._albums["grp"].day, bot._albums["grp"].meal = date(2026, 8, 15), "lunch"

    await bot._flush_album("grp")

    assert [c[0] for c in calls] == ["a", "b", "c"]
    assert {(c[1], c[2]) for c in calls} == {(date(2026, 8, 15), "lunch")}
    assert "grp" not in bot._albums


async def test_album_with_a_bad_directive_logs_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    processed: list[str] = []

    async def _fake_process(*args: object) -> None:
        processed.append("ran")

    replies: list[str] = []

    class _Msg:
        async def reply_text(self, text: str, **_kwargs: object) -> object:
            replies.append(text)
            return object()

    monkeypatch.setattr(bot, "_process_photo", _fake_process)
    monkeypatch.setattr(bot, "_ALBUM_SETTLE", 0)
    bot._albums["bad"] = bot._Album(
        day=date(2026, 8, 16),
        meal="other",
        photos=[bot._AlbumPhoto(_Msg(), "a", "image/jpeg", "")] * 3,  # type: ignore[arg-type]
        error="Couldn't read 'nope' as a date.",
    )

    await bot._flush_album("bad")

    assert processed == []  # nothing filed against the wrong day
    assert len(replies) == 1  # and the batch is told once, not three times


# --- /today macros --------------------------------------------------------------


def test_macro_line_matches_the_pickers_field_order_and_separators() -> None:
    line = bot._macro_line(158, 0.0, 30.16, 5.74)
    assert line == "Calories: 158kcal · Fat: 0.00g · Carbs: 30.16g · Protein: 5.74g"
    # The picker builds its line by reformatting FatSecret's own description;
    # the two must stay visually identical.
    picker = bot._food_line(
        1,
        Food(
            food_id=1,
            name="Sourdough",
            servings=[],
            description=(
                "Per 100g - Calories: 158kcal | Fat: 0.00g | Carbs: 30.16g | Protein: 5.74g"
            ),
        ),
        weighable=True,
    )
    assert picker.endswith("Per 100g — " + line)


def test_today_totals_every_entry() -> None:
    entries = [
        DiaryEntry(
            food_entry_id=1,
            food_id=1,
            name="Bread",
            day=date(2026, 8, 16),
            calories=158,
            fat=0.0,
            carbs=30.16,
            protein=5.74,
        ),
        DiaryEntry(
            food_entry_id=2,
            food_id=2,
            name="Egg",
            day=date(2026, 8, 16),
            calories=143,
            fat=9.51,
            carbs=0.72,
            protein=12.56,
        ),
    ]
    total = bot._macro_line(
        sum(e.calories for e in entries),
        sum(e.fat for e in entries),
        sum(e.carbs for e in entries),
        sum(e.protein for e in entries),
    )
    assert total == "Calories: 301kcal · Fat: 9.51g · Carbs: 30.88g · Protein: 18.30g"


def test_diary_entry_without_macros_reads_as_zero() -> None:
    # A food from an odd dataset can omit a field; the day's listing must still
    # render rather than blowing up on a missing carb count.
    entry = DiaryEntry(food_entry_id=1, food_id=1, name="Mystery", day=date(2026, 8, 16))
    assert bot._macro_line(entry.calories, entry.fat, entry.carbs, entry.protein).startswith(
        "Calories: 0kcal"
    )


# --- /new directive parsing -----------------------------------------------------


@pytest.mark.parametrize(
    ("caption", "name", "brand"),
    [
        ("/new", None, None),
        ("/new Super protein bar 23 g, SuperCompany", "Super protein bar 23 g", "SuperCompany"),
        ("/new Fiber One 10 g Fiber", "Fiber One 10 g Fiber", None),
        ("/new  Oat Milk ,  Oatly  ", "Oat Milk", "Oatly"),
        ("/NEW Bar, Brand", "Bar", "Brand"),
    ],
)
def test_parse_new(caption: str, name: str | None, brand: str | None) -> None:
    assert bot.parse_new(caption) == (name, brand)


def test_new_keeps_a_trailing_weight_in_the_product_name() -> None:
    # parse_correction would strip "23 g" off as a weight. Product names sell
    # their macros in the title now, and that number is part of the name.
    assert bot.parse_correction("Super protein bar 23 g")[1] == 23.0
    assert bot.parse_new("/new Super protein bar 23 g") == ("Super protein bar 23 g", None)


@pytest.mark.parametrize("caption", ["Avocado", "56", "/backlog 15/08/2026 lunch", "", "news"])
def test_parse_new_ignores_everything_else(caption: str) -> None:
    assert bot.parse_new(caption) is None
