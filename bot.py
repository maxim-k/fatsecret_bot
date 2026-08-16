"""Telegram intake and review.

Ingestion never blocks on FatSecret: a photo is acknowledged, recognised, and
queued regardless of whether the diary API is reachable. That separation is
deliberate — the whole point of the bot is that dropping a photo into a chat
is frictionless, so nothing downstream may make the user wait or fail.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import uuid
from datetime import date
from typing import Final

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
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
from models import Alias, Food, ItemState, QueuedItem, RecognizedItem

log = logging.getLogger(__name__)

# The model reports 0.0 when it could not actually read the display (see
# vision._SYSTEM_PROMPT), and vision._digits_agree forces the same value when a
# reading contradicts its own transcription. Nothing between 0 and 1 is worth
# gating on: a self-graded 0.75 held two correct readings.
UNREADABLE: Final = 0.0

# Below this, Haiku's reading is worth a second opinion from the slower model.
# The score is too noisy to interrupt the user over, but good enough to decide
# where to spend eight more seconds.
SHAKY: Final = 0.5


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
    if queued.alias is None:
        return "New food."
    if queued.item.confidence <= UNREADABLE:
        return "Couldn't read the display."
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
    """Ask the stronger model to look at the same photo once more.

    :param image_bytes: The original photo, unmodified.
    :param media_type: Its MIME type.
    :param recent: The same reuse candidates the first pass got.
    :param hint: Why the first reading looks wrong.
    :param queued: The item being rechecked, used to match the retry's answer.
    :returns: The re-read item, or ``None`` if the retry failed or came back
        describing something else.
    """
    try:
        second = await vision.recognize(
            image_bytes, media_type, recent, hint=hint, model=vision.escalation_model()
        )
    except vision.RecognitionError:
        return None  # the first reading still stands and will be held
    # A multi-item plate can come back in a different order, so match on the name
    # rather than assuming the retry answers in the same sequence.
    key = match.normalize(queued.item.product)
    return next((i for i in second.items if match.normalize(i.product) == key), None)


async def on_photo(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Accept a food photo, recognise it, and submit or queue each item.

    :param update: Incoming Telegram update carrying a photo or image document.
    :param _context: Unused handler context.
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

    telegram_file = await message.get_bot().get_file(file_id)
    image_bytes = bytes(await telegram_file.download_as_bytearray())
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

    # Telegram stamps message.date in UTC; an evening photo would otherwise
    # be filed against tomorrow's diary.
    local = message.date.astimezone()
    day = local.date()
    meal = match.meal_for(local.time())
    # ponytail: load/modify/save is only safe because python-telegram-bot runs
    # updates one at a time (max_concurrent_updates == 1), and there is an await
    # on the vision call in between. Raising concurrency loses queue entries when
    # several photos arrive together — switch to append-only or a lock if so.
    queue = store.load_queue()
    logged: list[tuple[QueuedItem, str]] = []
    held: list[tuple[QueuedItem, str]] = []

    for item in recognition.items:
        queued = _queue_item(item, digest, day, meal)
        store.append_log("recognized", {"item_id": queued.id, **item.model_dump()})
        if message.caption:
            _apply_correction(queued, message.caption)

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
                        "model": vision.escalation_model(),
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
    queue = store.load_queue()
    replied_to = message.reply_to_message.message_id
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

    _apply_correction(queued, message.text)
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
    await message.reply_text("\n".join(f"• {e.name}" for e in entries))


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
