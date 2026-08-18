"""Score vision models against the saved photo corpus.

The bot keeps every photo it has read in ``DATA_DIR/photos`` and every reading
in ``log.jsonl``, which together make a regression set for the one question that
decides the model: does it read this particular kitchen scale correctly, and how
long does it take? Run it before changing ``GEMINI_MODEL``.

    uv run --env-file .env python benchmark.py
    uv run --env-file .env python benchmark.py gemini-3.5-flash gemini-3.7-flash

Ground truth comes from the diary: a weight the user corrected by hand is taken
as correct, and so is a weight that every submission of that photo agreed on.
Photos whose history disagrees are still read and printed, but not scored --
guessing at their true value would make the score say whatever we wanted.
"""

from __future__ import annotations

import asyncio
import collections
import json
import statistics
import sys
import time
from dataclasses import dataclass

from google.genai import types

import store
import vision
from models import Recognition

# Model, then thinking budget in tokens: 0 disables it, -1 lets the model
# decide. Reading four digits off an LCD is not a reasoning problem, and the
# default budget is most of the wall-clock time.
CANDIDATES: list[tuple[str, int]] = [
    ("gemini-3.5-flash", 0),
    ("gemini-3.5-flash", -1),
    ("gemini-3.6-flash", 0),
    ("gemini-3.7-flash", 0),
]

# gemini-3.5-flash-lite is deliberately absent: it rejects thinking_budget=0
# with a bare 400, and read 4 of 11 known weights correctly with thinking on --
# 640 for 64 g, 1080 for 108 g, 3790 for 379 g. The same trailing-zero failure
# that made Haiku unusable on this scale.

# The free tier rate-limits hard enough to fail a run, and a 429 measures the
# quota rather than the model.
_SPACING_S = 4.0

# A reading this close to the truth counts as correct: the scale itself drifts a
# gram, and the diary does not care.
TOLERANCE_G = 1.0


@dataclass
class Case:
    """One photo and what it is known to weigh."""

    digest: str
    image: bytes
    product: str
    expected: float | None  # None when the history disagrees with itself


def load_cases() -> list[Case]:
    """Join saved photos to their known weights, one case per photo on disk."""
    queue = json.loads((store._data_dir() / "queue.json").read_text())
    photo_of = {q["id"]: q["photo_sha256"] for q in queue}

    submitted: dict[str, list[float]] = collections.defaultdict(list)
    corrected: set[str] = set()
    product: dict[str, str] = {}
    for line in (store._data_dir() / "log.jsonl").read_text().splitlines():
        event = json.loads(line)
        payload, kind = event.get("payload", {}), event.get("event")
        item = payload.get("item_id")
        digest = photo_of.get(item)
        if digest is None:
            continue
        if kind == "recognized":
            product.setdefault(digest, payload["product"])
        elif kind == "submitted":
            submitted[digest].append(payload["grams"])
        elif kind == "corrected":
            # Whatever is submitted for this photo next is the weight the user
            # typed, which is the closest thing to ground truth there is.
            corrected.add(digest)

    cases = []
    for path in sorted((store._data_dir() / "photos").iterdir()):
        digest = path.stem
        weights = submitted.get(digest, [])
        if digest in corrected and weights:
            expected: float | None = weights[-1]
        elif weights and len(set(weights)) == 1:
            expected = weights[0]
        else:
            expected = None
        cases.append(Case(digest, path.read_bytes(), product.get(digest, "?"), expected))
    return cases


async def read_once(image: bytes, budget: int) -> tuple[Recognition, float]:
    """Read one photo, returning the recognition and its wall-clock seconds.

    :param budget: Thinking budget in tokens. 0 disables it.
    """
    started = time.monotonic()
    recognition = await vision._generate(
        [types.Part.from_bytes(data=image, mime_type="image/jpeg"), "Read the scale."],
        vision._SYSTEM_PROMPT,
        Recognition,
        thinking=types.ThinkingConfig(thinking_budget=budget),
    )
    return recognition, time.monotonic() - started


async def score(model: str, budget: int, cases: list[Case]) -> None:
    """Run every case through one model and print a row per photo."""
    import os

    os.environ["GEMINI_MODEL"] = model
    right = wrong = unscored = failed = 0
    latencies: list[float] = []

    print(f"\n=== {model}  thinking={'off' if budget == 0 else 'auto'}")
    for index, case in enumerate(cases):
        if index:
            await asyncio.sleep(_SPACING_S)
        try:
            recognition, seconds = await read_once(case.image, budget)
        except vision.RecognitionError as exc:
            print(f"  {case.digest[:10]}  FAILED  {str(exc)[:70]}")
            failed += 1
            continue
        latencies.append(seconds)
        got = recognition.items[0].weight_g if recognition.items else 0.0
        raw = recognition.items[0].raw_digits if recognition.items else ""
        if case.expected is None:
            mark, note = "?", "no agreed truth"
            unscored += 1
        elif abs(got - case.expected) <= TOLERANCE_G:
            mark, note = "ok", ""
            right += 1
        else:
            mark, note = "WRONG", f"expected {case.expected:g}"
            wrong += 1
        print(
            f"  {case.digest[:10]}  {mark:5}  {got:8.1f} g  raw={raw!r:<10} "
            f"{seconds:5.1f}s  {case.product[:22]:<22} {note}"
        )

    scored = right + wrong
    accuracy = f"{right}/{scored}" if scored else "n/a"
    median = statistics.median(latencies) if latencies else 0.0
    slowest = max(latencies) if latencies else 0.0
    print(
        f"  --> correct {accuracy}   unscored {unscored}   failed {failed}   "
        f"median {median:.1f}s   slowest {slowest:.1f}s"
    )


async def main() -> int:
    """Run the corpus through every candidate, returning an exit code."""
    cases = load_cases()
    known = sum(1 for c in cases if c.expected is not None)
    print(f"{len(cases)} photos, {known} with a known weight, tolerance ±{TOLERANCE_G:g} g")

    wanted = sys.argv[1:]
    candidates = [(m, b) for m, b in CANDIDATES if not wanted or m in wanted]
    for model, budget in candidates:
        # Sequential on purpose: these are rate limited, and a burst measures
        # the queue rather than the model.
        await score(model, budget, cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
