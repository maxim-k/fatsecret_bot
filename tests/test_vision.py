from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pydantic
import pytest

import vision
from models import NutritionLabel, Recognition, RecognizedItem


@dataclass
class _FakeCandidate:
    finish_reason: str = "STOP"


@dataclass
class _FakeResponse:
    parsed: Any
    candidates: list[_FakeCandidate] = field(default_factory=lambda: [_FakeCandidate()])


@dataclass
class _FakeModels:
    response: _FakeResponse
    last_kwargs: dict[str, Any] = field(default_factory=dict)

    async def generate_content(self, **kwargs: Any) -> _FakeResponse:
        self.last_kwargs = kwargs
        if isinstance(self.response.parsed, Exception):
            raise self.response.parsed
        return self.response


@dataclass
class _FakeAio:
    models: _FakeModels


@dataclass
class _FakeClient:
    response: _FakeResponse

    def __post_init__(self) -> None:
        self.aio = _FakeAio(_FakeModels(self.response))


def _install_fake_client(monkeypatch: pytest.MonkeyPatch, response: _FakeResponse) -> _FakeClient:
    fake_client = _FakeClient(response)
    monkeypatch.setattr(vision, "_get_client", lambda: fake_client)
    return fake_client


def _one_item(product: str = "chicken breast, raw") -> Recognition:
    return Recognition(
        items=[
            RecognizedItem(
                product=product,
                weight_g=150.0,
                raw_digits="150",
                confidence=0.95,
            )
        ]
    )


def _prompt_text(fake_client: _FakeClient) -> str:
    """Join the text parts of the last request; image parts are not strings."""
    contents = fake_client.aio.models.last_kwargs["contents"]
    return "\n".join(part for part in contents if isinstance(part, str))


async def test_recognize_happy_path_one_item(monkeypatch: pytest.MonkeyPatch) -> None:
    recognition = _one_item()
    _install_fake_client(monkeypatch, _FakeResponse(parsed=recognition))

    result = await vision.recognize(b"fake-bytes", "image/jpeg", [])

    assert result == recognition
    assert len(result.items) == 1
    assert result.items[0].weight_g == 150.0


async def test_recognize_multi_item_plate(monkeypatch: pytest.MonkeyPatch) -> None:
    recognition = Recognition(
        items=[
            RecognizedItem(product="rice", weight_g=200.0, raw_digits="200", confidence=0.9),
            RecognizedItem(product="salmon", weight_g=120.0, raw_digits="120", confidence=0.9),
        ]
    )
    _install_fake_client(monkeypatch, _FakeResponse(parsed=recognition))

    result = await vision.recognize(b"fake-bytes", "image/jpeg", [])

    assert [i.product for i in result.items] == ["rice", "salmon"]


async def test_recognize_raises_when_nothing_was_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Gemini leaves .parsed as None on a refusal or a truncated generation
    # rather than raising, so the caller must not read it as an empty result.
    _install_fake_client(
        monkeypatch, _FakeResponse(parsed=None, candidates=[_FakeCandidate("SAFETY")])
    )

    with pytest.raises(vision.RecognitionError) as caught:
        await vision.recognize(b"fake-bytes", "image/jpeg", [])
    assert "SAFETY" in str(caught.value)


async def test_recognize_reports_a_schema_violation_as_recognition_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Validation fires when .parsed is read, not when the request is sent.
    invalid = pydantic.ValidationError.from_exception_data("Recognition", [])
    _install_fake_client(monkeypatch, _FakeResponse(parsed=invalid))

    with pytest.raises(vision.RecognitionError):
        await vision.recognize(b"fake-bytes", "image/jpeg", [])


async def test_recognize_accepts_a_zero_weight_as_no_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A photo with no scale in it. This used to fail validation outright and
    # destroy the item; it must now come back for the caller to hold.
    unread = Recognition(
        items=[RecognizedItem(product="protein bar", weight_g=0, raw_digits="", confidence=0.0)]
    )
    _install_fake_client(monkeypatch, _FakeResponse(parsed=unread))

    result = await vision.recognize(b"fake-bytes", "image/jpeg", [])

    assert result.items[0].weight_g == 0


async def test_zero_weight_is_not_second_guessed_by_the_digit_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An empty raw_digits would fail _digits_agree, but a zero weight is already
    # an explicit "no reading" and needs no further downgrade.
    unread = Recognition(
        items=[RecognizedItem(product="bar", weight_g=0, raw_digits="", confidence=0.4)]
    )
    _install_fake_client(monkeypatch, _FakeResponse(parsed=unread))

    result = await vision.recognize(b"fake-bytes", "image/jpeg", [])

    assert result.items[0].confidence == 0.4


async def test_contradicting_digits_are_forced_to_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Containment, not equality — see _digits_agree. "170" against "117" still
    # agrees on the "17" prefix, so this uses a reading that genuinely conflicts.
    mismatch = Recognition(
        items=[RecognizedItem(product="egg", weight_g=170.0, raw_digits="56", confidence=0.95)]
    )
    _install_fake_client(monkeypatch, _FakeResponse(parsed=mismatch))

    result = await vision.recognize(b"fake-bytes", "image/jpeg", [])

    assert result.items[0].confidence == 0.0


async def test_recognize_injects_recent_foods_into_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _install_fake_client(monkeypatch, _FakeResponse(parsed=_one_item()))

    await vision.recognize(b"fake-bytes", "image/jpeg", ["chicken breast, raw", "greek yogurt"])

    combined = _prompt_text(fake_client)
    assert "chicken breast, raw" in combined
    assert "greek yogurt" in combined


async def test_recognize_sends_the_image_undownsampled(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _install_fake_client(monkeypatch, _FakeResponse(parsed=_one_item()))

    await vision.recognize(b"fake-bytes", "image/png", [])

    contents = fake_client.aio.models.last_kwargs["contents"]
    image_parts = [p for p in contents if not isinstance(p, str)]
    assert len(image_parts) == 1
    assert image_parts[0].inline_data is not None
    assert image_parts[0].inline_data.data == b"fake-bytes"
    assert image_parts[0].inline_data.mime_type == "image/png"


def test_model_defaults_to_gemini_flash_and_reads_the_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert vision.model_id() == "gemini-3.5-flash"
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    assert vision.model_id() == "gemini-3.5-flash-lite"


async def test_every_request_uses_the_one_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # There is no escalation tier any more: the re-read is worth doing for the
    # hint it carries, not for a bigger model behind it.
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash")
    fake_client = _install_fake_client(monkeypatch, _FakeResponse(parsed=_one_item()))

    await vision.recognize(b"fake-bytes", "image/jpeg", [])
    assert fake_client.aio.models.last_kwargs["model"] == "gemini-3.5-flash"

    await vision.recognize(b"fake-bytes", "image/jpeg", [], hint="Expect 42-70 g.")
    assert fake_client.aio.models.last_kwargs["model"] == "gemini-3.5-flash"


async def test_hint_reaches_the_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _install_fake_client(monkeypatch, _FakeResponse(parsed=_one_item()))

    await vision.recognize(b"x", "image/jpeg", [], hint="First read 560 g; expect 42-70 g.")

    assert "First read 560 g" in _prompt_text(fake_client)


async def test_no_hint_leaves_the_prompt_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _install_fake_client(monkeypatch, _FakeResponse(parsed=_one_item()))

    await vision.recognize(b"x", "image/jpeg", [])

    assert "First read" not in _prompt_text(fake_client)


# --- read_label -----------------------------------------------------------------


def _label() -> NutritionLabel:
    return NutritionLabel(
        name="Super protein bar 23 g",
        brand="SuperCompany",
        serving_description="1 bar",
        serving_amount=60.0,
        serving_unit="g",
        calories=240,
        fat=11,
        carbohydrate=24,
        protein=23,
    )


async def test_read_label_sends_every_photo_in_one_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The name is on the pack and the numbers are on the panel, so the model
    # has to see both frames at once.
    fake_client = _install_fake_client(monkeypatch, _FakeResponse(parsed=_label()))

    result = await vision.read_label([(b"front", "image/jpeg"), (b"panel", "image/jpeg")])

    assert result.name == "Super protein bar 23 g"
    contents = fake_client.aio.models.last_kwargs["contents"]
    assert len([p for p in contents if not isinstance(p, str)]) == 2


async def test_read_label_passes_the_users_name_and_brand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _install_fake_client(monkeypatch, _FakeResponse(parsed=_label()))

    await vision.read_label([(b"panel", "image/jpeg")], "Fiber One 10 g Fiber", "General Mills")

    prompt = _prompt_text(fake_client)
    assert "Fiber One 10 g Fiber" in prompt
    assert "General Mills" in prompt


async def test_read_label_refuses_an_empty_batch() -> None:
    with pytest.raises(vision.RecognitionError):
        await vision.read_label([])


async def test_read_label_raises_when_nothing_was_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_client(
        monkeypatch, _FakeResponse(parsed=None, candidates=[_FakeCandidate("MAX_TOKENS")])
    )

    with pytest.raises(vision.RecognitionError):
        await vision.read_label([(b"panel", "image/jpeg")])
