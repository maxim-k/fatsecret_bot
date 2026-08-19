from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import BaseModel

import bot
import match
import store
from models import (
    Alias,
    CommandItem,
    DiaryEntry,
    Food,
    ItemState,
    LogCommand,
    NutritionLabel,
    QueuedItem,
    Recognition,
    RecognizedItem,
    WeightChange,
)


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
    ("caption", "name", "action", "grams"),
    [
        ("Avocado", "Avocado", None, None),
        ("56", None, "set", 56.0),
        ("Avocado 56", "Avocado", "set", 56.0),
        ("Avocado 56 g", "Avocado", "set", 56.0),
        ("56,4", None, "set", 56.4),
        ("  Greek Yogurt  ", "Greek Yogurt", None, None),
        ("Fiber One Fulfill Bars", "Fiber One Fulfill Bars", None, None),
        # A sign touching its digits is relative.
        ("-80", None, "subtract", 80.0),
        ("-80 g", None, "subtract", 80.0),
        ("Melon -80", "Melon", "subtract", 80.0),
        ("-80.5", None, "subtract", 80.5),
        ("+50", None, "add", 50.0),
        # A dash used as a separator is not, and gets trimmed off the name.
        ("Melon - 80", "Melon", "set", 80.0),
        ("- 80", None, "set", 80.0),
    ],
)
def test_parse_correction(
    caption: str, name: str | None, action: str | None, grams: float | None
) -> None:
    item = bot.parse_correction(caption)
    assert item is not None
    assert item.food_name == name
    if action is None:
        assert item.weight is None
    else:
        assert item.weight is not None
        assert (item.weight.action, item.weight.grams) == (action, grams)


def test_parse_correction_on_an_empty_line() -> None:
    assert bot.parse_correction("   ") is None


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
    simple = bot.parse_correction("Super protein bar 23 g")
    assert simple is not None and simple.weight is not None
    assert simple.weight.grams == 23.0
    assert bot.parse_new("/new Super protein bar 23 g") == ("Super protein bar 23 g", None)


@pytest.mark.parametrize("caption", ["Avocado", "56", "/backlog 15/08/2026 lunch", "", "news"])
def test_parse_new_ignores_everything_else(caption: str) -> None:
    assert bot.parse_new(caption) is None


# --- schemas the providers have to accept ---------------------------------------


@pytest.mark.parametrize("schema_model", [Recognition, NutritionLabel])
def test_response_schemas_convert_for_the_provider(schema_model: type[BaseModel]) -> None:
    """Guard the bug class that killed /new for a whole release.

    Every test mocked the client, so no request was ever built and a schema the
    API refuses looked fine all the way to the user. The SDK converts a Pydantic
    model to the provider's schema *client-side*, so the conversion that
    actually failed is checkable offline, with the provider's own rules rather
    than a hand-written imitation of them.

    Uses a private helper on purpose: it is the only way to reach the conversion
    without a network call, and a wrong schema is worth catching here rather
    than in production. If a SDK upgrade moves it, follow it.
    """
    from google.genai import _transformers

    assert _transformers.t_schema(None, schema_model) is not None


@pytest.mark.parametrize("schema_model", [Recognition, NutritionLabel])
def test_response_schemas_stay_portable(schema_model: type[BaseModel]) -> None:
    """Keep the schemas within the tightest limit any provider imposes.

    Gemini accepts more nullable fields than Anthropic does, so the conversion
    test above would not catch a re-flattened panel. Anthropic caps a schema at
    16 union-typed parameters; NutritionLabel had 17 and every /new returned a
    400 before the photo was sent.
    """
    schema = schema_model.model_json_schema()
    blocks = [schema, *schema.get("$defs", {}).values()]
    unions = [
        f"{block.get('title', '?')}.{name}"
        for block in blocks
        for name, prop in block.get("properties", {}).items()
        if "anyOf" in prop or isinstance(prop.get("type"), list)
    ]
    assert len(unions) <= 16, f"too many nullable fields to send: {unions}"


def test_a_zero_weight_is_held_rather_than_rejected() -> None:
    # A nutrition-label photo sent as food. weight_g used to be gt=0, so this
    # raised inside recognize() and the item vanished with no way to correct it.
    unread = QueuedItem(
        id="i" * 32,
        photo_sha256="x",
        day=date(2026, 8, 17),
        item=RecognizedItem(product="protein bar", weight_g=0, raw_digits="", confidence=0.0),
    )
    assert bot._hold_reason(unread) == "Couldn't read a weight."


def test_no_weight_outranks_new_food() -> None:
    # Both are true for a label photo, and answering "New food." would open the
    # picker offering to log the product at 0 g.
    store.save_aliases({})
    unread = QueuedItem(
        id="j" * 32,
        photo_sha256="x",
        day=date(2026, 8, 17),
        item=RecognizedItem(product="dragonfruit", weight_g=0, raw_digits="", confidence=0.9),
    )
    assert unread.alias is None
    assert bot._hold_reason(unread) == "Couldn't read a weight."


# --- subtracting scraps ---------------------------------------------------------


def _logged(weight: float = 350.0, product: str = "melon") -> QueuedItem:
    """An item already written to the diary, as a reply would find it."""
    alias = Alias(food_id=7, food_name="Melon", serving_id=2, serving_grams=100.0)
    alias.weights = [340.0, 350.0]
    return QueuedItem(
        id="m" * 32,
        photo_sha256="x",
        day=date(2026, 8, 17),
        item=RecognizedItem(
            product=product, weight_g=weight, raw_digits=str(int(weight)), confidence=0.9
        ),
        alias=alias,
        state=ItemState.SUBMITTED,
        message_id=99,
        food_entry_id=4242,
    )


def test_subtracting_scraps_takes_them_off_the_reading() -> None:
    queued = _logged()
    scraps = bot._apply_correction(queued, "-80")
    assert scraps == 80.0
    assert queued.item.weight_g == 270.0
    # raw_digits records where the number came from, not just the result.
    assert queued.item.raw_digits == "350-80"


def test_subtracting_more_than_there_is_refuses() -> None:
    queued = _logged()
    with pytest.raises(ValueError, match="leaves nothing"):
        bot._apply_correction(queued, "-400")
    # The item is untouched, so the caller can still fall back to it.
    assert queued.item.weight_g == 350.0


def test_subtracting_everything_refuses() -> None:
    # 0 g is the "no weight was read" signal, so it must not be reachable by
    # arithmetic.
    with pytest.raises(ValueError):
        bot._apply_correction(_logged(), "-350")


def test_a_plain_number_still_replaces_rather_than_adjusts() -> None:
    queued = _logged()
    assert bot._apply_correction(queued, "118") is None
    assert queued.item.weight_g == 118.0


# --- correcting a logged item end to end ----------------------------------------


class _Recorder:
    """Stands in for FatSecret, recording what the reply path actually did."""

    def __init__(self) -> None:
        self.deleted: list[int] = []
        self.created: list[float] = []
        self.replies: list[str] = []
        # _apply_reply repoints the item at its newest message, so a further
        # correction replies to the latest confirmation.
        self.message_id = 100

    async def delete_entry(self, food_entry_id: int) -> None:
        self.deleted.append(food_entry_id)

    async def create_entry(self, **kwargs: object) -> DiaryEntry:
        units = kwargs["units"]
        assert isinstance(units, float)
        self.created.append(units)
        return DiaryEntry(food_entry_id=5150, food_id=7, name="Melon", day=date(2026, 8, 17))

    async def reply_text(self, text: str, **_kwargs: object) -> _Recorder:
        self.replies.append(text)
        self.message_id += 1
        return self


@pytest.fixture
def _reply_env(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()
    monkeypatch.setattr(bot.fatsecret, "delete_entry", rec.delete_entry)
    monkeypatch.setattr(bot.fatsecret, "create_entry", rec.create_entry)
    store.save_aliases(
        {"melon": Alias(food_id=7, food_name="Melon", serving_id=2, serving_grams=100.0)}
    )
    return rec


async def test_reply_subtracting_scraps_replaces_the_entry(_reply_env: _Recorder) -> None:
    queued = _logged()
    store.save_queue([queued])

    await bot._apply_reply(_reply_env, "-80", 99)  # type: ignore[arg-type]

    assert _reply_env.deleted == [4242]  # the old entry went
    assert _reply_env.created == [2.7]  # 270 g against a 100 g serving
    assert "270 g" in _reply_env.replies[-1]
    assert "took off 80 g" in _reply_env.replies[-1]


async def test_reply_subtracting_too_much_leaves_the_diary_alone(
    _reply_env: _Recorder,
) -> None:
    # The guard this whole ordering exists for: the entry used to be deleted
    # before the reply was parsed, so an impossible correction lost the food.
    store.save_queue([_logged()])

    await bot._apply_reply(_reply_env, "-400", 99)  # type: ignore[arg-type]

    assert _reply_env.deleted == []
    assert _reply_env.created == []
    assert "leaves nothing" in _reply_env.replies[-1]
    still_there = store.load_queue()[0]
    assert still_there.food_entry_id == 4242
    assert still_there.item.weight_g == 350.0
    assert still_there.state == ItemState.SUBMITTED


async def test_reply_subtraction_trains_the_band_on_the_eaten_weight(
    _reply_env: _Recorder,
) -> None:
    # 270 g is what was actually eaten, so 270 is what the plausible band should
    # learn — and the 350 it replaces must not stay beside it.
    store.save_aliases(
        {
            "melon": Alias(
                food_id=7,
                food_name="Melon",
                serving_id=2,
                serving_grams=100.0,
                weights=[340.0, 350.0],
            )
        }
    )
    store.save_queue([_logged()])

    await bot._apply_reply(_reply_env, "-80", 99)  # type: ignore[arg-type]

    assert store.load_aliases()["melon"].weights == [340.0, 270.0]


async def test_reply_to_an_unknown_message_does_nothing(_reply_env: _Recorder) -> None:
    store.save_queue([_logged()])
    await bot._apply_reply(_reply_env, "-80", 12345)  # type: ignore[arg-type]
    assert _reply_env.deleted == []
    assert _reply_env.replies == []


# --- routing: local grammar vs the model ----------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "-150",
        "118",
        "56,4",
        "Avocado",
        "Avocado 56",
        "Avocado 56 g",
        "Melon -80",
        # The long names actually logged. Routing on token count would send
        # these to the model; they are plain name-and-weight and must not go.
        "trader joe's northwest super sweet frozen cut white corn 56",
        "optimum nutrition gold standard 100% whey - vanilla 56",
        "fiber one fulfill bars - caramel peanut 56",
    ],
)
def test_plain_corrections_never_reach_the_model(text: str) -> None:
    assert bot.is_command(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "y s banana +50",
        "t d salmon 200",
        "15/08 l bread 56",
        "15/08/2026 l",
        "y",
        "b",
        "y l chicken 200 rice 150",
        "/backlog 15/08/2026 lunch",
    ],
)
def test_commands_route_to_the_model(text: str) -> None:
    assert bot.is_command(text) is True


def test_a_percentage_inside_a_name_is_not_a_second_weight() -> None:
    # "100%" is not a standalone number, so this stays a one-weight correction.
    assert bot.is_command("optimum nutrition 100% whey 56") is False


# --- resolving the day ----------------------------------------------------------


def test_resolve_day() -> None:
    today = date(2026, 8, 18)
    assert bot.resolve_day(None, today) is None  # not stated stays not stated
    assert bot.resolve_day("today", today) == today
    assert bot.resolve_day("yesterday", today) == date(2026, 8, 17)
    assert bot.resolve_day("tomorrow", today) == date(2026, 8, 19)
    assert bot.resolve_day("2026-08-15", today) == date(2026, 8, 15)


def test_resolve_day_refuses_what_it_cannot_read() -> None:
    # Refusing beats defaulting: food filed quietly against the wrong day is
    # the corruption this grammar exists to avoid.
    with pytest.raises(ValueError, match="as a date"):
        bot.resolve_day("someday", date(2026, 8, 18))


# --- cross-checking the model ---------------------------------------------------


def _command(action: str, grams: float, name: str | None = None) -> LogCommand:
    return LogCommand(
        items=[CommandItem(food_name=name, weight=WeightChange(action=action, grams=grams))]
    )


def test_a_sign_in_the_text_must_match_the_action_chosen() -> None:
    # The model now stands between the user and their diary, so the checkable
    # part gets checked — same as vision._digits_agree does for the scale.
    with pytest.raises(ValueError, match="written with"):
        bot.check_command("-150", _command("set", 150))
    with pytest.raises(ValueError):
        bot.check_command("+50", _command("subtract", 50))
    # Agreement passes through untouched.
    assert bot.check_command("-150", _command("subtract", 150)).items[0].weight is not None


def test_a_weight_invented_for_a_line_with_no_numbers_is_dropped() -> None:
    checked = bot.check_command("Avocado", _command("set", 100, "Avocado"))
    assert checked.items[0].weight is None
    assert checked.items[0].food_name == "Avocado"


# --- applying a parsed item -----------------------------------------------------


def test_apply_item_handles_all_three_actions() -> None:
    for action, grams, expected, scraps in [
        ("set", 118.0, 118.0, None),
        ("add", 50.0, 400.0, None),
        ("subtract", 80.0, 270.0, 80.0),
    ]:
        queued = _logged()
        got = bot._apply_item(queued, CommandItem(weight=WeightChange(action=action, grams=grams)))
        assert queued.item.weight_g == expected, action
        assert got == scraps, action


def test_apply_item_refuses_a_subtraction_that_leaves_nothing() -> None:
    with pytest.raises(ValueError, match="leaves nothing"):
        bot._apply_item(_logged(), CommandItem(weight=WeightChange(action="subtract", grams=400)))


def test_command_items_match_recognised_foods_by_name() -> None:
    # "chicken 200 rice 150" has to land on the right halves of one plate.
    chicken = _logged(product="chicken")
    rice = _logged(product="rice")
    items = [
        CommandItem(food_name="rice", weight=WeightChange(action="set", grams=150)),
        CommandItem(food_name="chicken", weight=WeightChange(action="set", grams=200)),
    ]
    assert bot._command_item_for(chicken, items) is items[1]
    assert bot._command_item_for(rice, items) is items[0]
    # A lone unnamed item belongs to whatever the photo found.
    solo = [CommandItem(weight=WeightChange(action="subtract", grams=80))]
    assert bot._command_item_for(chicken, solo) is solo[0]


# --- a reply must not move an item to another day -------------------------------


async def test_a_reply_that_names_no_day_keeps_the_items_own_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bug in the original draft, pinned.

    A schema defaulting target_date to "today" cannot tell "the user said
    today" from "the user said nothing", so a bare -80 on a backlogged item
    silently dragged it to today.
    """
    backlogged = _logged()
    backlogged.day = date(2026, 8, 15)
    backlogged.meal = "lunch"

    async def _parse(_text: str, _today: date) -> LogCommand:
        # What the model returns for a bare "-80": nothing stated.
        return LogCommand(items=[CommandItem(weight=WeightChange(action="subtract", grams=80))])

    monkeypatch.setattr(bot.vision, "parse_command", _parse)
    # "-80" routes locally, so force the model path to prove the inheritance.
    monkeypatch.setattr(bot, "is_command", lambda _text: True)

    await bot._correct(backlogged, "-80")

    assert backlogged.day == date(2026, 8, 15)
    assert backlogged.meal == "lunch"
    assert backlogged.item.weight_g == 270.0


async def test_a_reply_that_names_a_day_moves_the_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = _logged()
    item.day = date(2026, 8, 15)

    async def _parse(_text: str, _today: date) -> LogCommand:
        return LogCommand(day="yesterday", meal="breakfast", items=[])

    monkeypatch.setattr(bot.vision, "parse_command", _parse)
    monkeypatch.setattr(bot, "is_command", lambda _text: True)

    await bot._correct(item, "y b")

    assert item.day == datetime.now().astimezone().date() - timedelta(days=1)
    assert item.meal == "breakfast"


async def test_a_reply_naming_several_foods_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _parse(_text: str, _today: date) -> LogCommand:
        return LogCommand(items=[CommandItem(food_name="a"), CommandItem(food_name="b")])

    monkeypatch.setattr(bot.vision, "parse_command", _parse)
    monkeypatch.setattr(bot, "is_command", lambda _text: True)

    with pytest.raises(ValueError, match="one at a time"):
        await bot._correct(_logged(), "chicken 200 rice 150")


def test_every_meal_the_schema_can_emit_is_one_fatsecret_takes() -> None:
    import typing

    emitted = set(typing.get_args(LogCommand.model_fields["meal"].annotation)[0].__args__)
    assert emitted <= match.MEALS
