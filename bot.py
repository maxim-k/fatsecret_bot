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

# Auto-approve a confident reading of a known food; anything shakier gets asked
# about. This gates the model's output, never the user's input.
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
    aliases = store.load_aliases()
    key = match.normalize(queued.item.product)
    if key in aliases:
        # Keep the recent tail only; portion sizes drift over months.
        aliases[key].weights = [*aliases[key].weights, queued.item.weight_g][-20:]
        aliases[key].hits += 1
        store.save_aliases(aliases)
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


def _reread_hint(queued: QueuedItem) -> str | None:
    """Build an instruction for a second look, or ``None`` if none would help.

    Only a weight the food's own history contradicts is worth re-reading: a new
    food has nothing to compare against, and a display the model already called
    illegible will not become legible on a second pass.

    :param queued: The item the first pass produced.
    :returns: Text to append to the retry prompt, or ``None``.
    """
    if queued.alias is None:
        return None
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
    if queued.item.confidence < AUTO_SUBMIT_CONFIDENCE:
        return "Low confidence."
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


async def _reread(
    image_bytes: bytes,
    media_type: str,
    recent: list[str],
    hint: str,
    queued: QueuedItem,
) -> RecognizedItem | None:
    """Ask the model to look at the same photo once more.

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
                    {"item_id": queued.id, "first": first, "second": retried.weight_g},
                )
                queued.item = retried
                if _hold_reason(queued) is None:
                    note = f" (re-read; first pass said {first:g} g)"
                else:
                    note = f" First pass said {first:g} g."

        hold = _hold_reason(queued)
        if hold is None:
            try:
                lines.append(await _submit(queued) + note)
                queued.state = ItemState.SUBMITTED
            except fatsecret.FatSecretError as exc:
                # The sink failing must not lose the item — keep it reviewable.
                lines.append(f"{_describe(queued)}\nFatSecret error: {exc}")
        else:
            lines.append(
                f"{_describe(queued)}\n{hold}{note}\n"
                "Reply with a number to fix the weight, or a name to fix the food."
            )
        queue.append(queued)

    store.save_queue(queue)
    await message.reply_text("\n\n".join(lines) or "Nothing recognised in that photo.")


async def on_reply(update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
    """Correct a held item by replying to the message describing it.

    A numeric reply fixes the weight, anything else names the right food. The
    item is recovered from the short id printed in the quoted message, so no
    conversation state has to be tracked between messages.

    :param update: Incoming reply.
    :param _context: Unused handler context.
    """
    message = update.effective_message
    if message is None or message.text is None or message.reply_to_message is None:
        return
    quoted = message.reply_to_message.text or ""
    short_id = re.match(r"\s*`?([0-9a-f]{8})`?", quoted)
    if short_id is None:
        return  # a reply to something else entirely

    queue = store.load_queue()
    queued = next((i for i in queue if i.id.startswith(short_id.group(1))), None)
    if queued is None:
        await message.reply_text("That item is gone.")
        return

    reply = message.text.strip().replace(",", ".")
    try:
        if re.fullmatch(r"\d+(?:\.\d+)?", reply):
            queued.item.weight_g = float(reply)
        else:
            queued.item.product = reply
            queued.alias = await _teach(queued)
        text = await _submit(queued)
        queued.state = ItemState.SUBMITTED
    except (fatsecret.FatSecretError, match.NoGramServingError, ValueError) as exc:
        await message.reply_text(f"Couldn't submit: {exc}")
        return

    store.save_queue(queue)
    await message.reply_text(text)


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
    # Before the photo handler, and gated on REPLY so ordinary text falls through.
    app.add_handler(MessageHandler(owner & filters.REPLY & filters.TEXT, on_reply))
    app.add_handler(MessageHandler(owner & (filters.PHOTO | filters.Document.IMAGE), on_photo))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_error_handler(on_error)
    return app


if __name__ == "__main__":
    raise SystemExit(main())
