"""Telegram intake and review.

Ingestion never blocks on FatSecret: a photo is acknowledged, recognised, and
queued regardless of whether the diary API is reachable. That separation is
deliberate — the whole point of the bot is that dropping a photo into a chat
is frictionless, so nothing downstream may make the user wait or fail.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Final

from telegram import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import fatsecret
import match
import store
import vision
from models import Alias, Food, ItemState, NutritionLabel, QueuedItem, RecognizedItem

log = logging.getLogger(__name__)

# The model reports 0.0 when it could not actually read the display (see
# vision._SYSTEM_PROMPT), and vision._digits_agree forces the same value when a
# reading contradicts its own transcription. Nothing between 0 and 1 is worth
# gating on: a self-graded 0.75 held two correct readings. A weight_g of 0 says
# the same thing in the field that matters, and _hold_reason checks both.
UNREADABLE: Final = 0.0

# Below this, the reading is worth a second look. The score is too noisy to
# interrupt the user over, but good enough to decide where to spend one more
# request — and the retry carries a hint the first pass never had.
SHAKY: Final = 0.5

# Telegram sends an album as one update per photo and attaches the caption to
# the single message it was typed on, so a /backlog directive naming a date
# reaches one photo out of ten. The group is collected instead and processed
# once its updates stop arriving. Only the file_id is stashed: downloading in
# the handler would let one slow full-resolution document outlast this window
# and flush a half-finished batch.
_ALBUM_SETTLE: Final = 2.0


@dataclass
class _AlbumPhoto:
    """One photo of an album, held until the batch is known to be complete."""

    message: Message
    file_id: str
    media_type: str
    caption: str


@dataclass
class _Album:
    """Photos of one Telegram media group, and the day they belong to."""

    day: date
    meal: str
    photos: list[_AlbumPhoto] = field(default_factory=list)
    # Set by a /new caption: the batch is a product to add, not food to log.
    product: tuple[str | None, str | None] | None = None
    error: str | None = None
    # Extended by every frame that arrives. The flush waits for it to pass
    # rather than being cancelled and re-created, which dropped the coroutine
    # unrun ("never awaited") and could split one batch across two flushes.
    deadline: float = 0.0
    task: asyncio.Task[None] | None = None


_albums: dict[str, _Album] = {}

# Display names for the nutrients shown back before a food is created. The units
# and the values themselves live in match; this is only how they are spelled.
_NUTRIENT_LABELS: Final[dict[str, str]] = {
    "saturated_fat": "Sat fat",
    "polyunsaturated_fat": "Poly fat",
    "monounsaturated_fat": "Mono fat",
    "trans_fat": "Trans fat",
    "cholesterol": "Cholesterol",
    "sodium": "Sodium",
    "potassium": "Potassium",
    "fiber": "Fiber",
    "sugar": "Sugar",
    "vitamin_d": "Vitamin D",
    "vitamin_a": "Vitamin A",
    "vitamin_c": "Vitamin C",
    "calcium": "Calcium",
    "iron": "Iron",
}

# A panel read but not yet created, waiting on a tap. food.create has no delete
# and no edit, so this handoff is deliberately seconds long and nothing lingers.
# ponytail: in memory, so a restart between reading and tapping drops the panel
# and the photos have to be re-sent. Persist it only if that actually happens.
_pending_labels: dict[str, NutritionLabel] = {}

# The queue is read, modified and written back around awaits. Updates alone are
# serialised (max_concurrent_updates == 1), but an album flush runs as its own
# task outside that, so a flush overlapping a button tap would lose one of the
# two writes.
_queue_lock = asyncio.Lock()


def _owner_id() -> int:
    """Read the single permitted Telegram user id.

    :returns: The owner's numeric Telegram id.
    :raises RuntimeError: If ``TELEGRAM_OWNER_ID`` is unset or not numeric.
    """
    raw = os.environ.get("TELEGRAM_OWNER_ID", "")
    if not raw.isdigit():
        raise RuntimeError("TELEGRAM_OWNER_ID must be set to your numeric Telegram user id")
    return int(raw)


def _describe(queued: QueuedItem) -> str:
    """Render a queued item as a single chat line.

    :param queued: Item to describe.
    :returns: Human-readable summary including the raw display transcription,
        so a misread is visible before the user approves it.
    """
    item = queued.item
    name = queued.alias.food_name if queued.alias else item.product
    return f"{name} — {item.weight_g:g} g (read “{item.raw_digits}”)"


async def _submit(queued: QueuedItem, *, record: bool = True, replaces: float | None = None) -> str:
    """Write one queued item to the FatSecret diary.

    :param queued: Item carrying a resolved alias.
    :param record: Whether to fold this weight into the alias's history. False
        for a reading that only got through because nothing could check it — a
        wrong value must not widen the band meant to catch wrong values.
    :param replaces: A previously recorded weight to drop from the history,
        set when this submission corrects an earlier one. Without it a misread
        keeps widening the band that exists to catch misreads.
    :returns: A chat-ready confirmation line.
    :raises RuntimeError: If the item has no alias resolved yet.
    """
    if queued.alias is None:
        raise RuntimeError(f"item {queued.id} has no alias to submit with")
    alias = queued.alias
    units = match.units_for(alias.serving_grams, queued.item.weight_g)
    entry = await fatsecret.create_entry(
        food_id=alias.food_id,
        name=alias.food_name,
        serving_id=alias.serving_id,
        units=units,
        day=queued.day,
        meal=queued.meal,
    )
    aliases = store.load_aliases()
    key = match.normalize(queued.item.product)
    if record and key in aliases:
        weights = aliases[key].weights
        if replaces is not None and replaces in weights:
            weights = [w for w in weights if w != replaces]
        # Keep the recent tail only; portion sizes drift over months.
        aliases[key].weights = [*weights, queued.item.weight_g][-20:]
        aliases[key].hits += 1
        store.save_aliases(aliases)
    queued.food_entry_id = entry.food_entry_id
    store.append_log(
        "submitted",
        {
            "item_id": queued.id,
            "food_entry_id": entry.food_entry_id,
            "grams": queued.item.weight_g,
            "raw_digits": queued.item.raw_digits,
        },
    )
    return f"Logged {alias.food_name} — {queued.item.weight_g:g} g"


def _reread_hint(queued: QueuedItem) -> str | None:
    """Build an instruction for a second look, or ``None`` if none would help.

    Fires when the fast model's answer is doubtful: the weight contradicts the
    food's own history, the reading contradicts its own transcription, or the
    model graded itself low. A new food is skipped — there is nothing to compare
    against, and the picker is about to ask anyway.

    :param queued: The item the first pass produced.
    :returns: Text to append to the retry prompt, or ``None``.
    """
    if queued.alias is None:
        return None
    if queued.item.confidence < SHAKY:
        return (
            "A previous read of this photo could not settle the weight. Read the "
            "display again, reporting only the large seven-segment digits and "
            "ignoring unit labels and status marks beside them."
        )
    bounds = match.plausible_range(queued.alias.weights)
    if bounds is None or bounds[0] <= queued.item.weight_g <= bounds[1]:
        return None
    return (
        f"A previous read of this photo reported {queued.item.weight_g:g} g for "
        f"{queued.alias.food_name!r}. That food normally weighs between "
        f"{bounds[0]:.0f} and {bounds[1]:.0f} g, so the earlier reading is very "
        "likely wrong. The usual cause is a status indicator on the display "
        "being counted as a digit, most often as a trailing zero. Read only the "
        "large seven-segment characters and report what they actually show. If "
        "the display really does read the earlier value, say so and keep it."
    )


def _hold_reason(queued: QueuedItem) -> str | None:
    """Decide whether this item needs a human before it is logged.

    Every reason here is a doubt about the model's output or the API's data,
    never about the user's input. Sending the same photo twice is the user's
    business; a reading that contradicts itself or is nothing like any portion
    they have ever eaten is not.

    :param queued: The freshly recognised item.
    :returns: A short reason to show the user, or ``None`` to log it now.
    """
    # No weight comes first, ahead of the unknown-food check: a photo with no
    # scale in it is usually also a food the bot has never seen, and answering
    # "New food." would open the picker offering to log it at 0 g.
    if queued.item.weight_g == 0 or queued.item.confidence <= UNREADABLE:
        return "Couldn't read a weight."
    if queued.alias is None:
        return "New food."
    bounds = match.plausible_range(queued.alias.weights)
    if bounds is not None and not bounds[0] <= queued.item.weight_g <= bounds[1]:
        return f"Unusual weight — expected {bounds[0]:.0f}-{bounds[1]:.0f} g."
    return None


def _queue_item(item: RecognizedItem, photo_sha256: str, day: date, meal: str) -> QueuedItem:
    """Wrap a recognised item for storage.

    :param item: Recognised food and weight.
    :param photo_sha256: Digest of the source photo, kept for the audit trail.
    :param day: Diary day the item belongs to.
    :param meal: FatSecret meal section, derived from the local time.
    :returns: A pending queue item with a fresh id.
    """
    return QueuedItem(
        id=uuid.uuid4().hex,
        photo_sha256=photo_sha256,
        day=day,
        meal=meal,
        item=item,
        alias=match.lookup(item.product, store.load_aliases()),
    )


def _macro_line(calories: float, fat: float, carbs: float, protein: float) -> str:
    """Render macros the way FatSecret's own food descriptions read.

    Deliberately mirrors the summary string `_food_line` shows for a search hit,
    down to the field order and separators, so the diary and the picker do not
    look like two different apps. The picker reformats FatSecret's string; this
    builds the same shape from numbers, because a diary entry has no such string.

    :param calories: Energy in kcal.
    :param fat: Fat in grams.
    :param carbs: Carbohydrate in grams.
    :param protein: Protein in grams.
    :returns: One line of text.
    """
    return (
        f"Calories: {calories:.0f}kcal · Fat: {fat:.2f}g · "
        f"Carbs: {carbs:.2f}g · Protein: {protein:.2f}g"
    )


def _food_line(index: int, food: Food, weighable: bool) -> str:
    """Render one search hit as a choosable line.

    :param index: 1-based position, matching the button label.
    :param food: The candidate, carrying its search-result macro summary.
    :param weighable: Whether it exposes a gram serving.
    :returns: Two lines of text.
    """
    label = f"{food.name}"
    if food.brand:
        label += f" · {food.brand}"
    if not weighable:
        label += " — can't log by weight"
    macros = (food.description or "").replace(" - ", " — ").replace(" | ", " · ")
    return f"{index}. {label}\n   {macros}"


async def _offer_foods(message: Message, queued: QueuedItem, queue: list[QueuedItem]) -> None:
    """Search FatSecret and offer the top three hits with their macros.

    Replaces silently committing the first weighable hit as a permanent alias.
    Servings are fetched up front so a candidate that cannot be logged by weight
    is marked before it is tapped rather than after.

    :param message: The message to reply to.
    :param queued: The item awaiting a food.
    :param queue: The queue to persist the sent message id into.
    """
    query = queued.item.product
    try:
        hits = (await fatsecret.search_foods(query))[:3]
    except fatsecret.FatSecretError as exc:
        await message.reply_text(f"Search failed: {exc}")
        return
    if not hits:
        await message.reply_text(f"Nothing found for {query!r}. Reply with another name.")
        return

    lines, buttons = [], []
    for position, hit in enumerate(hits, start=1):
        try:
            food = await fatsecret.get_food(hit.food_id)
            match.pick_serving(food, queued.item.weight_g)
            weighable = True
        except match.NoGramServingError:
            weighable = False
        except fatsecret.FatSecretError:
            weighable = False
        lines.append(_food_line(position, hit, weighable))
        buttons.append(
            InlineKeyboardButton(str(position), callback_data=f"pick:{queued.id}:{hit.food_id}")
        )
    buttons.append(InlineKeyboardButton("Discard", callback_data=f"no:{queued.id}"))

    sent = await message.reply_text(
        f"{queued.item.weight_g:g} g — which one?\n\n"
        + "\n".join(lines)
        + "\n\nReply with a different name to search again.",
        reply_markup=InlineKeyboardMarkup([buttons]),
    )
    for stored in queue:
        if stored.id == queued.id:
            stored.message_id = sent.message_id
    store.save_queue(queue)


def parse_correction(text: str) -> tuple[str | None, float | None]:
    """Split a caption or reply into an optional food name and weight.

    ``"Avocado"`` names the food, ``"56"`` sets the weight, ``"Avocado 56"``
    does both. Shared by captions and replies so the two cannot drift apart.

    :param text: Raw caption or reply text.
    :returns: ``(name, weight)``, either of which may be ``None``.
    """
    cleaned = text.strip().replace(",", ".")
    trailing = re.search(r"(?:^|\s)(\d+(?:\.\d+)?)\s*(?:g|grams?)?$", cleaned, re.IGNORECASE)
    weight = float(trailing.group(1)) if trailing else None
    name = cleaned[: trailing.start()].strip() if trailing else cleaned
    return (name or None), weight


BACKLOG_USAGE: Final = (
    "/backlog DD/MM/YYYY [meal] — files the photos against that day.\n"
    "Meal is breakfast, lunch, dinner or other (default other).\n"
    "DD/MM/YYYY, DD/MM, today or yesterday. Example: /backlog 15/08/2026 lunch"
)

# Day before month, matching how the dates are written when asking for them.
# Accepting both orders would make 08/09 mean either September 8th or August
# 9th, and a silently wrong day is the one failure this whole feature exists
# to prevent, so an unparseable date is refused rather than guessed at. A
# yearless date has the current year appended rather than its own format
# string: strptime without a year is deprecated and changes meaning in 3.15.
_BACKLOG_DATE_FORMAT: Final = "%d/%m/%Y"


def parse_backlog(caption: str) -> tuple[date, str] | None:
    """Read a ``/backlog`` directive out of a photo caption.

    :param caption: Raw caption text.
    :returns: ``(day, meal)`` for a backlog directive, or ``None`` when the
        caption is an ordinary correction and belongs to
        :func:`parse_correction` instead.
    :raises ValueError: If the directive is present but its date or meal cannot
        be understood. Refusing beats defaulting to today, which would file a
        week of food against the wrong day without saying so.
    """
    words = caption.strip().split()
    if not words or words[0].lower() != "/backlog":
        return None
    if len(words) < 2:
        raise ValueError(BACKLOG_USAGE)

    today = datetime.now().astimezone().date()
    spelled = {"today": today, "yesterday": today - timedelta(days=1)}
    day = spelled.get(words[1].lower())
    if day is None:
        written = words[1] if words[1].count("/") == 2 else f"{words[1]}/{today.year}"
        try:
            day = datetime.strptime(written, _BACKLOG_DATE_FORMAT).date()
        except ValueError:
            raise ValueError(f"Couldn't read {words[1]!r} as a date.\n\n{BACKLOG_USAGE}") from None

    meal = words[2].lower() if len(words) > 2 else "other"
    if meal not in match.MEALS:
        raise ValueError(f"{words[2]!r} is not a meal.\n\n{BACKLOG_USAGE}")
    if len(words) > 3:
        raise ValueError(f"Don't know what to do with {' '.join(words[3:])!r}.\n\n{BACKLOG_USAGE}")
    return day, meal


NEW_USAGE: Final = (
    "/new — photograph the product and its nutrition label, and I'll add it.\n"
    "Name it yourself with /new Name, Brand — for example:\n"
    "/new Super protein bar 23 g, SuperCompany"
)


def parse_new(caption: str) -> tuple[str | None, str | None] | None:
    """Read a ``/new`` directive out of a photo caption.

    Deliberately does not use :func:`parse_correction`: that reads a trailing
    number as a weight, and product names now routinely end in one — "Fiber One
    10 g Fiber" would become "Fiber One" weighing 10 grams. Everything before
    the last comma is the name, digits included.

    :param caption: Raw caption text.
    :returns: ``(name, brand)``, either of which may be ``None`` when the
        caption was a bare ``/new`` and the packaging should be read instead;
        or ``None`` when this is not a ``/new`` caption at all.
    """
    stripped = caption.strip()
    head, _, rest = stripped.partition(" ")
    if head.lower() != "/new":
        return None
    name, _, brand = rest.strip().rpartition(",")
    if not name:  # no comma: the whole thing is a name, or nothing at all
        return (brand.strip() or None), None
    return name.strip() or None, brand.strip() or None


def _apply_correction(queued: QueuedItem, text: str) -> None:
    """Overwrite a recognised item from what the user actually said.

    The user naming the food outranks the model guessing at it, so the alias is
    re-resolved against their text rather than the model's.

    :param queued: The item to correct, modified in place.
    :param text: Caption or reply text.
    """
    name, weight = parse_correction(text)
    if name is not None:
        queued.item.product = name
        queued.alias = match.lookup(name, store.load_aliases())
    if weight is not None:
        queued.item.weight_g = weight
        queued.item.raw_digits = f"{weight:g}"


def _review_keyboard(queued: QueuedItem) -> InlineKeyboardMarkup:
    """Approve/discard buttons for one held item.

    :param queued: The item the buttons act on.
    :returns: A one-row keyboard.
    """
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Approve", callback_data=f"ok:{queued.id}"),
                InlineKeyboardButton("Discard", callback_data=f"no:{queued.id}"),
            ]
        ]
    )


async def _reread(
    image_bytes: bytes,
    media_type: str,
    recent: list[str],
    hint: str,
    queued: QueuedItem,
) -> RecognizedItem | None:
    """Look at the same photo once more, with a hint the first pass lacked.

    :param image_bytes: The original photo, unmodified.
    :param media_type: Its MIME type.
    :param recent: The same reuse candidates the first pass got.
    :param hint: Why the first reading looks wrong.
    :param queued: The item being rechecked, used to match the retry's answer.
    :returns: The re-read item, or ``None`` if the retry failed or came back
        describing something else.
    """
    try:
        second = await vision.recognize(image_bytes, media_type, recent, hint=hint)
    except vision.RecognitionError:
        return None  # the first reading still stands and will be held
    # A multi-item plate can come back in a different order, so match on the name
    # rather than assuming the retry answers in the same sequence.
    key = match.normalize(queued.item.product)
    return next((i for i in second.items if match.normalize(i.product) == key), None)


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Accept a food photo, alone or as one frame of an album.

    A lone photo is processed straight away. An album is collected first: see
    :data:`_ALBUM_SETTLE` for why it cannot be handled a photo at a time.

    :param update: Incoming Telegram update carrying a photo or image document.
    :param context: Handler context, used to run the album flush as a task.
    """
    message = update.effective_message
    if message is None:
        return

    # Documents keep full resolution and EXIF; Telegram's compressed "photo"
    # can smear the scale's digits, so prefer the document when both exist.
    if message.document is not None:
        file_id, media_type = message.document.file_id, message.document.mime_type or "image/jpeg"
    elif message.photo:
        file_id, media_type = message.photo[-1].file_id, "image/jpeg"
    else:
        return

    # Telegram stamps message.date in UTC; an evening photo would otherwise
    # be filed against tomorrow's diary.
    local = message.date.astimezone()
    day, meal = local.date(), match.meal_for(local.time())
    caption, error = message.caption or "", None
    try:
        directive = parse_backlog(caption)
    except ValueError as exc:
        directive, error = None, str(exc)
    product = parse_new(caption)
    if directive is not None or product is not None:
        # A directive is an instruction about the batch, not a name for the food.
        caption = ""
    if directive is not None:
        day, meal = directive

    group = message.media_group_id
    if group is None:
        if error is not None:
            await message.reply_text(error)
            return
        if product is not None:
            await _offer_label([_AlbumPhoto(message, file_id, media_type, "")], *product)
            return
        async with _queue_lock:
            await _process_photo(message, file_id, media_type, day, meal, caption)
        return

    # setdefault, so the first photo of the group supplies the fallback day and
    # meal and later ones do not reset it.
    album = _albums.setdefault(group, _Album(day=day, meal=meal))
    if directive is not None:
        album.day, album.meal = directive
    if product is not None:
        album.product = product
    if error is not None:
        album.error = error
    album.photos.append(_AlbumPhoto(message, file_id, media_type, caption))
    album.deadline = time.monotonic() + _ALBUM_SETTLE
    if album.task is None:  # one task per group, never cancelled
        album.task = context.application.create_task(_flush_album(group), update=update)


async def _flush_album(group: str) -> None:
    """Process one media group once its updates have stopped arriving.

    :param group: Telegram ``media_group_id`` of the batch.
    """
    while True:
        album = _albums.get(group)
        if album is None:
            return
        remaining = album.deadline - time.monotonic()
        if remaining <= 0:
            break
        # Every new frame pushes the deadline out, so this re-sleeps rather
        # than firing on a batch that is still arriving.
        await asyncio.sleep(remaining)
    album = _albums.pop(group, None)
    if album is None or not album.photos:
        return
    if album.error is not None:
        # One reply for the batch, and nothing logged. A directive that cannot be
        # read must not quietly fall back to today: the whole point of naming a
        # date is that the photos do not belong to today.
        await album.photos[0].message.reply_text(album.error)
        return
    if album.product is not None:
        # One call with every frame: the name is on the pack and the numbers are
        # on the panel, so the model needs to see them together.
        await _offer_label(album.photos, *album.product)
        return
    for photo in album.photos:
        async with _queue_lock:
            await _process_photo(
                photo.message, photo.file_id, photo.media_type, album.day, album.meal, photo.caption
            )


async def _offer_label(photos: list[_AlbumPhoto], name: str | None, brand: str | None) -> None:
    """Read a product's label off its photos and offer to add it.

    Nothing is written yet. ``food.create`` has no delete and no edit, so a
    misread milligram would be permanent — the panel is shown first for the same
    reason `_offer_foods` shows three candidates instead of committing the first
    hit. This questions the model, not the user.

    :param photos: Every frame of the batch, sent to the model together.
    :param name: Product name from the caption, overriding the packaging.
    :param brand: Brand from the caption, same precedence.
    """
    message = photos[0].message
    images = []
    for photo in photos:
        data = await _download(photo.message, photo.file_id)
        store.save_photo(hashlib.sha256(data).hexdigest(), data)
        images.append((data, photo.media_type))

    try:
        label = await vision.read_label(images, name, brand)
    except vision.RecognitionError as exc:
        await message.reply_text(f"Couldn't read that label: {exc}")
        return

    label_id = uuid.uuid4().hex[:16]
    _pending_labels[label_id] = label
    store.append_log("label_read", {"label_id": label_id, **label.model_dump()})
    await message.reply_text(
        f"Read this label:\n\n{_label_summary(label)}\n\nAdd it? This can't be undone.",
        reply_markup=InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("Add", callback_data=f"add:{label_id}"),
                    InlineKeyboardButton("Discard", callback_data=f"drop:{label_id}"),
                ]
            ]
        ),
    )


async def _add_food(query: CallbackQuery, label_id: str) -> None:
    """Create the confirmed food and teach the alias for it.

    Logs nothing: the photos were of a package and its panel, not of a portion
    on a scale, so there is no weight to record. The next scale photo of it logs
    with no interaction, which is the whole point.

    :param query: The callback to answer with the outcome.
    :param label_id: Key of the panel awaiting confirmation.
    """
    label = _pending_labels.pop(label_id, None)
    if label is None:
        await query.edit_message_text("That label is gone — send the photos again.")
        return

    grams = match.serving_grams(label)
    try:
        food_id = await fatsecret.create_food(
            name=label.name,
            brand=label.brand or "Generic",
            serving_description=label.serving_description,
            nutrients=match.label_nutrients(label),
            serving_amount=grams if grams is not None else label.serving_amount,
            serving_unit="g" if grams is not None else (label.serving_unit or "g"),
        )
        food = await fatsecret.get_food(food_id)
    except fatsecret.FatSecretError as exc:
        await query.edit_message_text(f"Couldn't create that food: {exc}")
        return
    store.append_log("created_food", {"food_id": food_id, "name": label.name})

    try:
        serving, _units = match.pick_serving(food, grams or 100.0)
    except match.NoGramServingError:
        # A drink measured in millilitres, most often. The food is real and
        # usable in FatSecret; it just cannot be logged off a gram reading.
        await query.edit_message_text(
            f"Added {food.name}.\nIts serving is a volume, so I can't log it from a scale weight."
        )
        return

    aliases = store.load_aliases()
    aliases[match.normalize(label.name)] = Alias(
        food_id=food.food_id,
        food_name=food.name,
        serving_id=serving.serving_id,
        serving_grams=match.grams_per_unit(serving),
    )
    store.save_aliases(aliases)
    await query.edit_message_text(
        f"Added {food.name}.\nPhotograph it on the scale and it'll log itself."
    )


def _label_summary(label: NutritionLabel) -> str:
    """Render everything read off a label, so a misread is visible before it sticks.

    Anything derived from a percentage is marked as such: that is the figure most
    worth a second look, being both rounded on the label and converted here.

    :param label: The transcribed panel.
    :returns: Several lines — identity, serving, macros, then the rest.
    """
    head = label.name + (f" · {label.brand}" if label.brand else "")
    grams = match.serving_grams(label)
    serving = label.serving_description
    if grams is not None:
        serving += f" ({grams:g} g)"
    elif label.serving_amount is not None:
        serving += f" ({label.serving_amount:g} {label.serving_unit})"

    # Which readings arrived only as a percentage, so the summary can say so —
    # those are the figures most worth a second look, being rounded on the label
    # and converted here.
    derived = {
        r.nutrient: r.percent_daily_value
        for r in label.nutrients
        if r.amount is None and r.percent_daily_value is not None
    }
    rest = []
    for nutrient, value in match.label_nutrients(label).items():
        if nutrient in ("calories", "fat", "carbohydrate", "protein"):
            continue  # already in the macro line
        shown = (
            f"{_NUTRIENT_LABELS.get(nutrient, nutrient)} {value:g}{match.NUTRIENT_UNITS[nutrient]}"
        )
        percent = derived.get(nutrient)
        if percent is not None:
            shown += f" (from {percent:g}% DV)"
        rest.append(shown)

    lines = [
        head,
        serving,
        _macro_line(label.calories, label.fat, label.carbohydrate, label.protein),
    ]
    lines += [" · ".join(rest[i : i + 3]) for i in range(0, len(rest), 3)]
    return "\n".join(lines)


async def _download(message: Message, file_id: str) -> bytes:
    """Fetch one photo's bytes from Telegram.

    :param message: Any message from the same chat, used for its bot handle.
    :param file_id: Telegram file id to download.
    :returns: The image bytes, never resized — the scale's LCD and a label's
        small print are both destroyed by downsampling.
    """
    telegram_file = await message.get_bot().get_file(file_id)
    return bytes(await telegram_file.download_as_bytearray())


async def _process_photo(
    message: Message,
    file_id: str,
    media_type: str,
    day: date,
    meal: str,
    caption: str,
) -> None:
    """Recognise one photo and submit or queue each item it holds.

    :param message: Message the photo arrived on, replied to with the results.
    :param file_id: Telegram file id to download.
    :param media_type: MIME type of that file.
    :param day: Diary date to file the items against.
    :param meal: FatSecret meal to file them under.
    :param caption: Correction text to apply, empty when there is none.
    """
    image_bytes = await _download(message, file_id)
    digest = hashlib.sha256(image_bytes).hexdigest()
    store.save_photo(digest, image_bytes)

    aliases = store.load_aliases()
    # Most-eaten first: the candidate list is capped, and the model can only
    # reuse a name it was shown. Alias keys come from FatSecret's naming
    # ("Egg"), while the model would otherwise say "eggs" and miss the lookup.
    recent = [
        alias.food_name for alias in sorted(aliases.values(), key=lambda a: a.hits, reverse=True)
    ][:40]  # candidate list has to stay promptable
    try:
        recognition = await vision.recognize(image_bytes, media_type, recent)
    except vision.RecognitionError as exc:
        store.append_log("recognition_failed", {"sha256": digest, "error": str(exc)})
        await message.reply_text(f"Couldn't read that photo: {exc}")
        return

    queue = store.load_queue()
    logged: list[tuple[QueuedItem, str]] = []
    held: list[tuple[QueuedItem, str]] = []

    for item in recognition.items:
        queued = _queue_item(item, digest, day, meal)
        store.append_log("recognized", {"item_id": queued.id, **item.model_dump()})
        if caption:
            _apply_correction(queued, caption)

        # One re-read, never a loop: a hint that fails to settle the reading is
        # not going to settle it on the third try either.
        note = ""
        hint = _reread_hint(queued)
        if hint is not None:
            first = queued.item.weight_g
            retried = await _reread(image_bytes, media_type, recent, hint, queued)
            if retried is not None:
                store.append_log(
                    "reread",
                    {
                        "item_id": queued.id,
                        "first": first,
                        "second": retried.weight_g,
                        "model": vision.model_id(),
                    },
                )
                queued.item = retried
                if _hold_reason(queued) is None:
                    note = f" (re-read; first pass said {first:g} g)"
                else:
                    note = f" First pass said {first:g} g."

        hold = _hold_reason(queued)
        if hold is None:
            try:
                logged.append((queued, await _submit(queued) + note))
                queued.state = ItemState.SUBMITTED
            except fatsecret.FatSecretError as exc:
                held.append((queued, f"FatSecret error: {exc}"))
        else:
            held.append((queued, f"{hold}{note}"))
        queue.append(queued)

    # One message per logged item: a reply has to resolve to exactly one thing,
    # and a plate with several foods would otherwise be ambiguous.
    for queued, text in logged:
        sent = await message.reply_text(text)
        queued.message_id = sent.message_id
    store.save_queue(queue)
    # Each held item gets its own message so a reply lands on exactly one item.
    for queued, reason in held:
        if queued.alias is None:
            await _offer_foods(message, queued, queue)
            continue
        sent = await message.reply_text(
            f"{_describe(queued)}\n{reason}\n"
            "Reply with a number to fix the weight, or a name to fix the food.",
            reply_markup=_review_keyboard(queued),
        )
        queued.message_id = sent.message_id
    if not logged and not held:
        await message.reply_text("Nothing recognised in that photo.")
    store.save_queue(queue)


async def on_reply(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Correct an item by replying to the message describing it.

    Works on a held item and on one already logged: a number fixes the weight,
    anything else names the right food. An already-logged item has its FatSecret
    entry deleted and rewritten, so a correction never leaves two entries for one
    portion. The item is found by the id of the quoted message, so no
    conversation state has to be tracked between messages.

    :param update: Incoming reply.
    :param _context: Unused handler context.
    """
    message = update.effective_message
    if message is None or message.text is None or message.reply_to_message is None:
        return
    async with _queue_lock:
        await _apply_reply(message, message.text, message.reply_to_message.message_id)


async def _apply_reply(message: Message, text: str, replied_to: int) -> None:
    """Resolve a reply to its queued item and resubmit it corrected.

    :param message: The reply, replied to with the outcome.
    :param text: The reply's text, a weight or a food name.
    :param replied_to: Id of the quoted message, which identifies the item.
    """
    queue = store.load_queue()
    queued = next((i for i in queue if i.message_id == replied_to), None)
    if queued is None:
        return  # a reply to something that is not one of our items

    # Correcting something already logged means replacing it, not adding to it.
    superseded: float | None = None
    if queued.state == ItemState.SUBMITTED and queued.food_entry_id is not None:
        superseded = queued.item.weight_g
        try:
            await fatsecret.delete_entry(queued.food_entry_id)
        except fatsecret.FatSecretError as exc:
            # Stop rather than risk two entries for one portion.
            await message.reply_text(f"Couldn't remove the old entry: {exc}")
            return
        store.append_log("corrected", {"item_id": queued.id, "removed_entry": queued.food_entry_id})
        queued.food_entry_id = None
        queued.state = ItemState.PENDING

    _apply_correction(queued, text)
    if queued.alias is None:
        store.save_queue(queue)
        await _offer_foods(message, queued, queue)
        return
    try:
        text = await _submit(queued, record=True, replaces=superseded)
        queued.state = ItemState.SUBMITTED
    except (fatsecret.FatSecretError, match.NoGramServingError, ValueError) as exc:
        await message.reply_text(f"Couldn't submit: {exc}")
        return

    store.save_queue(queue)
    sent = await message.reply_text(text)
    queued.message_id = sent.message_id
    store.save_queue(queue)


async def on_backlog(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Answer a bare ``/backlog`` typed as a message rather than a caption.

    The directive only does anything attached to photos, and Telegram routes
    captions to the photo handler, never to a command handler. Without this the
    command would match nothing and look broken.

    :param update: Incoming ``/backlog`` command.
    :param _context: Unused handler context.
    """
    message = update.effective_message
    if message is not None:
        await message.reply_text(BACKLOG_USAGE)


async def on_new(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Answer a bare ``/new`` typed as a message rather than a photo caption.

    :param update: Incoming ``/new`` command.
    :param _context: Unused handler context.
    """
    message = update.effective_message
    if message is not None:
        await message.reply_text(NEW_USAGE)


async def on_pending(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """List items awaiting review, each with approve and discard buttons.

    :param update: Incoming ``/pending`` command.
    :param _context: Unused handler context.
    """
    message = update.effective_message
    if message is None:
        return
    pending = [i for i in store.load_queue() if i.state == ItemState.PENDING]
    if not pending:
        await message.reply_text("Nothing pending.")
        return
    queue = store.load_queue()
    for queued in pending:
        sent = await message.reply_text(_describe(queued), reply_markup=_review_keyboard(queued))
        for stored in queue:
            if stored.id == queued.id:
                stored.message_id = sent.message_id
    store.save_queue(queue)


async def on_button(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle an approve or discard tap from ``/pending``.

    :param update: Callback query update.
    :param _context: Unused handler context.
    """
    query = update.callback_query
    if query is None or query.data is None:
        return
    await query.answer()
    # Telegram exposes the message as possibly-inaccessible (too old to act on),
    # so narrow it once here rather than at each use.
    origin = query.message if isinstance(query.message, Message) else None
    action, item_id, *extra = query.data.split(":")
    # A pending label is not a queued item, so these have to be routed before the
    # queue lookup below, which would answer "That item is gone".
    if action == "add":
        await _add_food(query, item_id)
        return
    if action == "drop":
        _pending_labels.pop(item_id, None)
        await query.edit_message_text("Discarded.")
        return
    async with _queue_lock:
        await _apply_button(query, origin, action, item_id, extra)


async def _apply_button(
    query: CallbackQuery,
    origin: Message | None,
    action: str,
    item_id: str,
    extra: list[str],
) -> None:
    """Act on one already-parsed callback.

    :param query: The callback query, edited in place with the outcome.
    :param origin: The message the buttons are attached to, if still reachable.
    :param action: ``pick``, ``no``, or anything else meaning approve.
    :param item_id: Queue id the button belongs to.
    :param extra: Remaining colon-separated fields; ``pick`` puts a food id here.
    """
    queue = store.load_queue()
    queued = next((i for i in queue if i.id == item_id), None)
    if queued is None:
        await query.edit_message_text("That item is gone.")
        return

    if action == "pick":
        food_id = extra[0]
        try:
            food = await fatsecret.get_food(int(food_id))
            serving, _units = match.pick_serving(food, queued.item.weight_g)
            queued.alias = Alias(
                food_id=food.food_id,
                food_name=food.name,
                serving_id=serving.serving_id,
                serving_grams=match.grams_per_unit(serving),
            )
            aliases = store.load_aliases()
            aliases[match.normalize(queued.item.product)] = queued.alias
            store.save_aliases(aliases)
            text = await _submit(queued, record=True)
            queued.state = ItemState.SUBMITTED
        except (fatsecret.FatSecretError, match.NoGramServingError) as exc:
            await query.edit_message_text(
                f"{origin.text if origin else ''}\n\nCan't use that one: {exc}"
            )
            return
        store.save_queue(queue)
        await query.edit_message_text(text)
        return

    if action == "no":
        queued.state = ItemState.DISCARDED
        store.save_queue(queue)
        await query.edit_message_text(f"Discarded {queued.id[:8]}.")
        return

    if queued.alias is None:
        if origin is not None:
            await _offer_foods(origin, queued, queue)
        return
    try:
        text = await _submit(queued, record=True)
        queued.state = ItemState.SUBMITTED
    except (fatsecret.FatSecretError, match.NoGramServingError) as exc:
        # Paste-able fallback: the user can still enter it by hand.
        await query.edit_message_text(
            f"{_describe(queued)}\nCouldn't submit: {exc}\n"
            f"{queued.item.product} — {queued.item.weight_g:g} g"
        )
        return

    store.save_queue(queue)
    await query.edit_message_text(text)


async def on_undo(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Delete the most recently submitted diary entry.

    :param update: Incoming ``/undo`` command.
    :param _context: Unused handler context.
    """
    message = update.effective_message
    if message is None:
        return
    queue = store.load_queue()
    last = next((i for i in reversed(queue) if i.state == ItemState.SUBMITTED), None)
    if last is None:
        await message.reply_text("Nothing to undo.")
        return
    entries = await fatsecret.list_diary(last.day)
    for entry in entries:
        if last.alias is not None and entry.food_id == last.alias.food_id:
            await fatsecret.delete_entry(entry.food_entry_id)
            break
    last.state = ItemState.DISCARDED
    store.save_queue(queue)
    store.append_log("undone", {"item_id": last.id})
    await message.reply_text(f"Removed {_describe(last)}")


async def on_today(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show what the FatSecret diary holds for today.

    :param update: Incoming ``/today`` command.
    :param _context: Unused handler context.
    """
    message = update.effective_message
    if message is None:
        return
    entries = await fatsecret.list_diary(date.today())
    if not entries:
        await message.reply_text("Diary is empty today.")
        return
    lines = [
        f"• {e.name}\n   {_macro_line(e.calories, e.fat, e.carbs, e.protein)}" for e in entries
    ]
    total = _macro_line(
        sum(e.calories for e in entries),
        sum(e.fat for e in entries),
        sum(e.carbs for e in entries),
        sum(e.protein for e in entries),
    )
    await message.reply_text("\n".join(lines) + f"\n\nTotal — {total}")


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log any unhandled handler exception instead of failing silently.

    ``fs_bot`` had no error handler, so a malformed response simply produced a
    traceback in the console and silence in the chat.

    :param update: The update being processed, if any.
    :param context: Handler context carrying the exception.
    """
    log.exception("handler failed", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message is not None:
        await update.effective_message.reply_text(f"Something broke: {context.error}")


def main() -> int:
    """Configure logging and poll until interrupted.

    :returns: Process exit code.
    """
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # Telegram's own HTTP chatter buries our logs at INFO and is rarely useful.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logging.error("TELEGRAM_BOT_TOKEN is not set - see .env.example")
        return 1
    build_application(token).run_polling()
    return 0


def build_application(token: str) -> Application:  # type: ignore[type-arg]
    """Assemble the Telegram application with an owner-only filter.

    :param token: Telegram bot token.
    :returns: Configured application, ready for polling.
    """
    owner = filters.User(user_id=_owner_id())
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("backlog", on_backlog, filters=owner))
    app.add_handler(CommandHandler("new", on_new, filters=owner))
    app.add_handler(CommandHandler("pending", on_pending, filters=owner))
    app.add_handler(CommandHandler("today", on_today, filters=owner))
    app.add_handler(CommandHandler("undo", on_undo, filters=owner))
    # Before the photo handler, and gated on REPLY so ordinary text falls through.
    app.add_handler(MessageHandler(owner & filters.REPLY & filters.TEXT, on_reply))
    app.add_handler(MessageHandler(owner & (filters.PHOTO | filters.Document.IMAGE), on_photo))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_error_handler(on_error)
    return app


if __name__ == "__main__":
    raise SystemExit(main())
