"""Photo -> structured food items via Claude vision.

The user photographs food sitting on kitchen scales, with the scale's LCD in
frame, and forwards the photo to the bot. This module turns that image into
:class:`~models.Recognition`. The weight is always *read* off the display,
never estimated from the food's apparent size — a misread digit silently
corrupts calorie data, which is the failure mode most likely to make the user
abandon the bot, so the prompt and the response schema both treat the display
reading as auditable rather than trusted at face value.
"""

from __future__ import annotations

import base64
import os
import re
from typing import Final, Literal, cast

import pydantic
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam

from models import Recognition, RecognizedItem

# Deliberately the cheap model — see .env.example. The user upgrades to
# claude-sonnet-5 (then claude-opus-5) only if raw_digits proves unreliable on
# real photos, so this default must never silently drift to a pricier model.
_DEFAULT_MODEL: Final = "claude-haiku-4-5"

# Matches anthropic.types.base64_image_source_param.Base64ImageSourceParam's
# media_type field. Kept as a separate alias (rather than importing the SDK's
# internal type) so the public recognize() signature can stay a plain `str` —
# Telegram always reports one of these four for a photo, so the cast below is
# trusted input, not a weakening of the contract.
_ImageMediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]

_SYSTEM_PROMPT: Final = """\
You are reading a photograph of food sitting on a kitchen scale, taken by a \
home cook logging what they ate. The scale's LCD display is visible in the \
frame.

Weight — read the display, never estimate:
The weight you report MUST come from the digits actually shown on the \
scale's display, not from a visual estimate of the food's size or a guess at \
a typical serving. Misreading a single digit (for example "7" as "1") \
silently corrupts the calorie log, so treat the display as the single \
source of truth for weight, and never fill in a weight the display doesn't \
show.

For each item, report two things about the weight, kept separate so a \
misread digit stays auditable instead of silently becoming a wrong calorie \
count:
- raw_digits: the literal characters you see on the display (for example \
  "247.3" or "0247"), including any unit shown. This is a transcription, \
  not a calculation — if the display is ambiguous, transcribe what you see, \
  not what seems plausible.
- weight_g: the same reading, parsed to grams as a number. If the display \
  shows a unit other than grams (oz, lb, kg), convert it to grams here; \
  raw_digits still records what was literally displayed, before conversion. \
  Unless you are converting units, weight_g MUST be exactly the number in \
  raw_digits — do not add, drop, or reorder digits, and never invent a \
  decimal point that raw_digits does not show.

Status indicators are not digits:
Kitchen scales print small status marks in the same LCD area as the number: \
a zero or tare mark, a stability/"hold" mark, unit labels (g, oz, lb), a \
battery symbol, a net flag. Several of these are round or oval and look \
almost exactly like the digit 0, and reading one as a digit turns 117 into \
170 or 1170. Count only the large seven-segment characters that make up the \
weight itself. A mark that is smaller than the digits, offset above or \
below the digit row, sitting next to a unit label, or separated from the \
number by a gap is an indicator, not a digit. If you cannot tell whether a \
round mark is a trailing zero or an indicator, treat it as an indicator and \
lower confidence to reflect the doubt.

Confidence — calibrate it, don't default it:
Report confidence (0 to 1) for how sure you are that weight_g is correct. \
Lower it when the display is blurred, glared, cropped out of frame, at a \
steep angle, partially obscured by a hand or the food itself, or when \
digits are ambiguous (for example a "3" that could be an "8"). A confident, \
sharp, straight-on reading of an unambiguous display warrants a high \
confidence; anything that required guessing does not.

When you cannot read the display at all:
weight_g must always be a positive number, so there is no way to report \
"unknown" in that field. When no display is visible or its digits are \
illegible, put your best estimate in weight_g, put whatever you can make \
out (or an empty string) in raw_digits, and set confidence to 0.0. A \
confidence of 0.0 is the agreed signal that the weight was not actually \
read and must not be trusted — the entry will be held back for the user to \
correct by hand rather than logged. Never quietly present an estimate as a \
reading by giving it a normal confidence.

Multiple items:
A single photo may show several distinct foods sharing the scale, or one \
food weighed in stages. Return one item per distinct food, each with the \
weight reading that corresponds to it. Do not merge different foods into \
one item, and do not split one food into several.

Naming:
Give each item a short, plain product name suitable for a food diary (for \
example "chicken breast, raw" or "greek yogurt"). The user's previously \
logged product names are given with each request as reuse candidates — when \
a food on the scale matches one of them, use that exact string verbatim for \
product; do not paraphrase, reformat, or change its capitalization. Only \
propose a new name when none of the candidates match.
"""


class RecognitionError(Exception):
    """Raised when the model's response cannot be trusted as a Recognition.

    Covers both "the model produced no parseable text" (empty or refused
    response) and "the model's JSON didn't validate against the schema" —
    either way, the caller gets a clear failure instead of a corrupted or
    partially-populated Recognition.
    """


_client: AsyncAnthropic | None = None


def _get_client() -> AsyncAnthropic:
    """Return the module-level Anthropic client, creating it on first use.

    Lazy construction means importing this module never requires
    ``ANTHROPIC_API_KEY`` to be set (useful in tests that never call
    :func:`recognize`), and reusing one client across calls keeps the
    underlying HTTP connection pool warm instead of rebuilding it per photo.
    Tests monkeypatch this function to inject a fake client, so
    :func:`recognize` never has to accept a client parameter of its own.

    :returns: The shared :class:`AsyncAnthropic` client.
    """
    global _client
    if _client is None:
        # AsyncAnthropic, not the sync client: recognize() runs inside
        # python-telegram-bot's async handlers, and a sync HTTP call there
        # would block the whole bot's event loop for every other user's
        # update while one photo is being read.
        _client = AsyncAnthropic()
    return _client


async def recognize(
    image_bytes: bytes,
    media_type: str,
    recent_foods: list[str],
    hint: str | None = None,
) -> Recognition:
    """Turn one scale photo into structured, weight-verified food items.

    :param image_bytes: Raw bytes of the photo exactly as received. Never
        resized or recompressed before sending — the scale's LCD is often a
        small region of the frame, and downsampling is exactly what would
        make its digits unreadable, defeating the point of reading the
        display instead of estimating.
    :param media_type: MIME type of ``image_bytes`` (for example
        ``"image/jpeg"``), as reported by Telegram.
    :param recent_foods: Product names the user has logged before, offered
        to the model as reuse candidates so downstream alias lookup in
        ``match.py`` hits an existing alias instead of creating a near-
        duplicate.
    :param hint: Extra instruction for a second look at the same photo, telling
        the model what the first pass returned and why it looks wrong. Only ever
        set on a retry.
    :returns: The items recognized on the scale, each with its own weight
        reading and confidence.
    :raises RecognitionError: If the model returned no parseable text
        content, or its content failed schema validation.
    """
    client = _get_client()
    candidates = "\n".join(f"- {food}" for food in recent_foods)
    user_prompt = (
        "Identify every distinct food item on the scale in this photo and "
        "read the weight from the scale's display.\n\n"
        + (
            f"The user has previously logged these exact product names:\n{candidates}\n\n"
            "Reuse one of them verbatim for `product` whenever it matches a food on the scale."
            if recent_foods
            else "The user has no previously logged foods yet. Propose a clear, "
            "concise product name for each item."
        )
    )
    if hint:
        user_prompt = f"{user_prompt}\n\n{hint}"
    image_b64 = base64.standard_b64encode(image_bytes).decode("ascii")

    messages: list[MessageParam] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": cast("_ImageMediaType", media_type),
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": user_prompt},
            ],
        }
    ]

    try:
        response = await client.messages.parse(
            model=os.environ.get("ANTHROPIC_MODEL", _DEFAULT_MODEL),
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
            messages=messages,
            output_format=Recognition,
        )
    except pydantic.ValidationError as exc:
        raise RecognitionError(f"model response failed schema validation: {exc}") from exc

    if response.parsed_output is None:
        raise RecognitionError(
            f"model returned no parseable text content (stop_reason={response.stop_reason!r})"
        )
    for item in response.parsed_output.items:
        if not _digits_agree(item):
            # The model transcribed one number and reported another; that is not
            # a reading, and the user gets asked rather than told.
            item.confidence = 0.0
    return response.parsed_output


def _digits_agree(item: RecognizedItem) -> bool:
    """Check the parsed weight against the digits the model says it saw.

    Observed live: ``raw_digits="170"`` alongside ``weight_g=170.1``.

    :param item: One recognised item straight from the model.
    :returns: ``True`` if the two agree, or a unit conversion explains it.
    """
    if re.search(r"oz|lb|kg", item.raw_digits, re.IGNORECASE):
        return True  # converting to grams legitimately changes the number
    number = re.search(r"\d+(?:[.,]\d+)?", item.raw_digits)
    if number is None:
        return False
    return abs(float(number.group().replace(",", ".")) - item.weight_g) < 0.05
