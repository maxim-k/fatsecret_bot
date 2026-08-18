"""Photo -> structured food items via Gemini vision.

The user photographs food sitting on kitchen scales, with the scale's LCD in
frame, and forwards the photo to the bot. This module turns that image into
:class:`~models.Recognition`. The weight is always *read* off the display,
never estimated from the food's apparent size — a misread digit silently
corrupts calorie data, which is the failure mode most likely to make the user
abandon the bot, so the prompt and the response schema both treat the display
reading as auditable rather than trusted at face value.
"""

from __future__ import annotations

import os
import re
from typing import Final

import pydantic
from google import genai
from google.genai import types

from models import NutritionLabel, Recognition, RecognizedItem

# One model for every photo. The previous pairing ran Haiku on everything and
# escalated to Sonnet on doubt, which bought both of its failure modes at once:
# fast and wrong, or accurate, slow and expensive.
#
# Chosen by measurement, not by tier: against the 17-photo corpus this read 9 of
# 9 known weights correctly at a 4.7 s median, including every trailing-zero
# case that Haiku got wrong (64 not 640, 108 not 1080, 379 not 3790).
# gemini-3.5-flash-lite managed 4 of 11 and repeated exactly those mistakes, and
# 3.6/3.7-flash answered 503 on every attempt. Override with GEMINI_MODEL and
# re-run benchmark.py before believing a different one is better.
_DEFAULT_MODEL: Final = "gemini-3.5-flash"

# Enough for a multi-item plate or a full nutrition panel, and low enough that a
# runaway generation fails fast instead of billing for it.
_MAX_OUTPUT_TOKENS: Final = 2048

# Transcribing four digits off an LCD is not a reasoning problem. Leaving the
# budget automatic tripled the median latency (4.7 s to 13 s) without reading
# anything better, and thinking tokens count against _MAX_OUTPUT_TOKENS, so one
# photo came back truncated with finish_reason=MAX_TOKENS and no answer at all.
# Note gemini-3.5-flash-lite rejects a zero budget outright with a bare 400.
_THINKING: Final = types.ThinkingConfig(thinking_budget=0)


def model_id() -> str:
    """Return the vision model every request uses.

    :returns: Model id from ``GEMINI_MODEL``, or the default.
    """
    return os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL)


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
Report weight_g as 0 and confidence as 0.0, and put whatever you can make \
out (or an empty string) in raw_digits. A weight of 0 is the agreed way to \
say "no weight was read": the entry is held back for the user to fill in by \
hand rather than logged. Use it whenever there is no scale in the frame, the \
display is out of shot, or the digits are illegible. Never guess a plausible \
weight to fill the field — an estimate presented as a reading is the one \
outcome worse than no reading.

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


_LABEL_PROMPT: Final = """\
You are reading photographs of a packaged food product: its front-of-pack \
artwork and its Nutrition Facts panel, in either order and sometimes in one \
frame. Transcribe the panel into structured data so it can be added to a food \
database.

Name and brand:
The product name is what the item is called, without the manufacturer. The \
brand is the manufacturer, and goes in `brand`.

A quantity inside the product's own name stays part of the name. Modern \
packaging routinely sells the nutrition in the title — "Protein Bar 20g \
Protein", "Fiber One 10g Fiber", "Zero Sugar 15g Protein Shake". Those \
numbers are marketing, printed on the front of the pack in the same typeface \
as the rest of the name. They are NOT the serving weight and NOT a nutrient \
reading: take nutrient values only from the Nutrition Facts panel, and keep \
the name exactly as the package writes it, digits and all.

Serving size:
Read the serving from the panel's "Serving size" line, not from the net \
weight of the package — a 60 g bar and a 250 g bag of chips both have a \
serving size, and only one of them is the whole package.
- serving_description: the serving as the label words it, e.g. "1 bar", \
  "2 cookies", "1 cup", "3 pieces".
- serving_amount and serving_unit: the measured quantity in parentheses, \
  e.g. "1 bar (60g)" gives 60 and "g". Report the unit as printed — "g", \
  "ml" or "oz" — and do not convert between them; that conversion is done \
  afterwards from your reported unit. If the panel gives no measured \
  quantity at all, leave both unset.

Nutrition values:
Every value is PER SERVING, exactly as the panel lists it, never per 100 g \
unless the panel's serving is 100 g, and never per package. Transcribe the \
number printed; do not compute, round, or reconcile values that look \
inconsistent.

Calories, fat, carbohydrate and protein have their own fields. Every other \
nutrient goes in the `nutrients` list, one entry each, where `nutrient` must \
be exactly one of:
  saturated_fat, polyunsaturated_fat, monounsaturated_fat, trans_fat,
  cholesterol, sodium, potassium, fiber, sugar,
  vitamin_d, vitamin_a, vitamin_c, calcium, iron
Use the panel's own units: grams for the fats, fiber and sugar; milligrams \
for cholesterol, sodium, potassium, calcium, iron and vitamin C; micrograms \
for vitamin D and vitamin A. A nutrient the panel does not mention is simply \
left out of the list — do not add an entry with a zero, and do not invent a \
plausible value. Zero means the label printed a zero.

Percentages are not amounts:
Panels print "%DV" beside many nutrients, and older ones print a percentage \
INSTEAD of an amount. Never report a percentage as an amount — 10% calcium is \
not 10 mg. Each entry has both `amount` and `percent_daily_value`: fill in \
whichever the panel gives, or both when it gives both. When only a percentage \
is printed, leave `amount` unset — it is converted afterwards from a fixed \
reference table, so report the percentage as the percentage it is.

If the panel is unreadable in the photos, say so rather than reconstructing \
a plausible one from the product's name.
"""


class RecognitionError(Exception):
    """Raised when the model's response cannot be trusted as a Recognition.

    Covers both "the model produced no parseable text" (empty or refused
    response) and "the model's JSON didn't validate against the schema" —
    either way, the caller gets a clear failure instead of a corrupted or
    partially-populated Recognition.
    """


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Return the module-level Gemini client, creating it on first use.

    Lazy construction means importing this module never requires
    ``GEMINI_API_KEY`` to be set (useful in tests that never call
    :func:`recognize`), and reusing one client across calls keeps the
    underlying HTTP connection pool warm instead of rebuilding it per photo.
    Tests monkeypatch this function to inject a fake client, so
    :func:`recognize` never has to accept a client parameter of its own.

    :returns: The shared :class:`genai.Client`. Requests go through its
        ``.aio`` namespace: a sync HTTP call inside python-telegram-bot's async
        handlers would block the whole bot's event loop while one photo is read.
    """
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


async def _generate[T: pydantic.BaseModel](
    parts: list[types.Part | str],
    system_prompt: str,
    schema: type[T],
    thinking: types.ThinkingConfig | None = None,
) -> T:
    """Send one request and return its validated structured output.

    :param parts: Request content — image parts and prompt text, in order.
    :param system_prompt: Instructions that do not vary per request.
    :param schema: Pydantic model the response must satisfy.
    :param thinking: Override the thinking budget. Only benchmark.py sets it;
        production uses the module default.
    :returns: The parsed response.
    :raises RecognitionError: If the model returned nothing parseable, or its
        output failed schema validation.
    """
    try:
        response = await _get_client().aio.models.generate_content(
            model=model_id(),
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=schema,
                max_output_tokens=_MAX_OUTPUT_TOKENS,
                # No tools are passed, so the SDK's automatic function calling
                # has nothing to do but warn on every single call.
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                thinking_config=thinking or _THINKING,
            ),
        )
        # Unlike the Anthropic SDK, validation happens when .parsed is read, so
        # the access belongs inside the try rather than after it.
        parsed = response.parsed
    except pydantic.ValidationError as exc:
        raise RecognitionError(f"model response failed schema validation: {exc}") from exc
    except genai.errors.APIError as exc:
        # Transient 503s under load are routine. The caller already treats a
        # RecognitionError as "tell the user and keep the photo", which is the
        # right handling here too.
        raise RecognitionError(f"vision API error: {exc}") from exc

    if not isinstance(parsed, schema):
        # A refusal or a truncated generation leaves .parsed as None rather than
        # raising; finish_reason says which.
        finish = response.candidates[0].finish_reason if response.candidates else None
        raise RecognitionError(f"model returned no parseable output (finish_reason={finish!r})")
    return parsed


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
        set on a retry — and the retry's value is the hint, not a bigger model.
    :returns: The items recognized on the scale, each with its own weight
        reading and confidence. A ``weight_g`` of zero means no weight could be
        read, which is a held item rather than an error.
    :raises RecognitionError: If the model returned no parseable text
        content, or its content failed schema validation.
    """
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

    recognition = await _generate(
        [types.Part.from_bytes(data=image_bytes, mime_type=media_type), user_prompt],
        _SYSTEM_PROMPT,
        Recognition,
    )
    for item in recognition.items:
        if item.weight_g and not _digits_agree(item):
            # The model transcribed one number and reported another; that is not
            # a reading, and the user gets asked rather than told. A zero weight
            # is already an explicit "no reading" and needs no second opinion.
            item.confidence = 0.0
    return recognition


async def read_label(
    images: list[tuple[bytes, str]],
    name: str | None = None,
    brand: str | None = None,
) -> NutritionLabel:
    """Read a product's nutrition panel from one or more photos.

    :param images: ``(bytes, media_type)`` for each photo, typically the pack
        front and the panel. All are sent in one request so the model can take
        the name from one frame and the numbers from another.
    :param name: Product name supplied by the user, which overrides whatever
        the packaging says.
    :param brand: Brand supplied by the user, same precedence.
    :returns: The transcribed panel.
    :raises RecognitionError: If no photos were given, or the model returned
        nothing parseable or something that failed schema validation.
    """
    if not images:
        raise RecognitionError("no photos to read a label from")

    told = ""
    if name:
        told += f"\nThe user says the product name is exactly: {name}\n"
    if brand:
        told += f"The user says the brand is exactly: {brand}\n"
    if told:
        told += (
            "Use those strings verbatim, including any digits and units they "
            "contain, rather than what the packaging shows."
        )

    parts: list[types.Part | str] = [
        types.Part.from_bytes(data=data, mime_type=media_type) for data, media_type in images
    ]
    parts.append("Read this product's name, brand and Nutrition Facts panel." + told)
    return await _generate(parts, _LABEL_PROMPT, NutritionLabel)


def _digits_agree(item: RecognizedItem) -> bool:
    """Check the parsed weight against the digits the model says it saw.

    Containment, not equality: the scale prints unit and mode glyphs beside the
    number, so a display reading 379 comes back as ``"8379h"``. Demanding an
    exact match rejected correct readings for noise. What must hold is that the
    reported weight actually appears in the transcription — ``"170"`` alongside
    ``weight_g=170.1`` still fails, which is the case this was written for.

    :param item: One recognised item straight from the model.
    :returns: ``True`` if the reported weight appears in the transcription, or a
        unit conversion explains the difference.
    """
    if re.search(r"oz|lb|kg", item.raw_digits, re.IGNORECASE):
        return True  # converting to grams legitimately changes the number
    digits = re.sub(r"[^0-9.]", "", item.raw_digits)
    if not digits:
        return False
    reported = f"{item.weight_g:g}"
    return reported in digits or reported.rstrip("0").rstrip(".") in digits
