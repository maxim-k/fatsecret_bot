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
import uuid
from datetime import date
from typing import Final

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
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
from models import Alias, ItemState, QueuedItem, RecognizedItem

log = logging.getLogger(__name__)

# A confident reading against an already-taught food is submitted without asking.
# Anything less is queued: the backlog this bot exists to kill comes from friction,
# but silently logging a misread weight would cost the user's trust entirely.
AUTO_SUBMIT_CONFIDENCE: Final = 0.8


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
    return (
        f"`{queued.id[:8]}` {name} — {item.weight_g:g} g "
        f"(read “{item.raw_digits}”, confidence {item.confidence:.0%})"
    )


async def _submit(queued: QueuedItem) -> str:
    """Write one queued item to the FatSecret diary.

    :param queued: Item carrying a resolved alias.
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


async def _teach(queued: QueuedItem) -> Alias:
    """Resolve an unknown product to a FatSecret food and learn it.

    Searches by the recognised product text and takes the first result with a
    weight-based serving. Teaching once is what makes every future photo of
    the same food submit with no interaction.

    :param queued: Item whose product text needs resolving.
    :returns: The learned alias, already saved.
    :raises FatSecretError: If the search returns nothing usable.
    """
    # search returns ~20 hits; one get_food each is the burst that trips
    # FatSecret's error 12. If the top few have no gram serving, the 20th won't.
    candidates = await fatsecret.search_foods(queued.item.product)
    for candidate in candidates[:5]:
        food = await fatsecret.get_food(candidate.food_id)
        try:
            serving, _ = match.pick_serving(food, queued.item.weight_g)
        except match.NoGramServingError:
            # Branded foods often only sell natural units; skip to one we can weigh.
            continue
        alias = Alias(
            food_id=food.food_id,
            food_name=food.name,
            serving_id=serving.serving_id,
            serving_grams=match.grams_per_unit(serving),
        )
        aliases = store.load_aliases()
        aliases[match.normalize(queued.item.product)] = alias
        store.save_aliases(aliases)
        return alias
    raise fatsecret.FatSecretError(
        "no_match", f"no weighable food found for {queued.item.product!r}"
    )


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
    lines: list[str] = []

    for item in recognition.items:
        queued = _queue_item(item, digest, day, meal)
        store.append_log("recognized", {"item_id": queued.id, **item.model_dump()})

        if queued.alias is not None and item.confidence >= AUTO_SUBMIT_CONFIDENCE:
            try:
                lines.append(await _submit(queued))
                queued.state = ItemState.SUBMITTED
            except fatsecret.FatSecretError as exc:
                # The sink failing must not lose the item — keep it reviewable.
                lines.append(f"{_describe(queued)}\nFatSecret error: {exc}")
        else:
            reason = "new food" if queued.alias is None else "low confidence"
            lines.append(f"{_describe(queued)}\nHeld — {reason}.")
        queue.append(queued)

    store.save_queue(queue)
    await message.reply_text("\n\n".join(lines) or "Nothing recognised in that photo.")


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
    for queued in pending:
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("Approve", callback_data=f"ok:{queued.id}"),
                    InlineKeyboardButton("Discard", callback_data=f"no:{queued.id}"),
                ]
            ]
        )
        await message.reply_text(_describe(queued), reply_markup=keyboard)


async def on_button(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle an approve or discard tap from ``/pending``.

    :param update: Callback query update.
    :param _context: Unused handler context.
    """
    query = update.callback_query
    if query is None or query.data is None:
        return
    await query.answer()
    action, _, item_id = query.data.partition(":")

    queue = store.load_queue()
    queued = next((i for i in queue if i.id == item_id), None)
    if queued is None:
        await query.edit_message_text("That item is gone.")
        return

    if action == "no":
        queued.state = ItemState.DISCARDED
        store.save_queue(queue)
        await query.edit_message_text(f"Discarded {queued.id[:8]}.")
        return

    try:
        if queued.alias is None:
            queued.alias = await _teach(queued)
        text = await _submit(queued)
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
    app.add_handler(MessageHandler(owner & (filters.PHOTO | filters.Document.IMAGE), on_photo))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_error_handler(on_error)
    return app


if __name__ == "__main__":
    raise SystemExit(main())
