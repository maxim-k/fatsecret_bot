from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

import vision
from models import Recognition, RecognizedItem


@dataclass
class _FakeResponse:
    parsed_output: Recognition | None
    stop_reason: str = "end_turn"


@dataclass
class _FakeMessages:
    response: _FakeResponse
    last_kwargs: dict[str, Any] = field(default_factory=dict)

    async def parse(self, **kwargs: Any) -> _FakeResponse:
        self.last_kwargs = kwargs
        return self.response


@dataclass
class _FakeClient:
    response: _FakeResponse

    def __post_init__(self) -> None:
        self.messages = _FakeMessages(self.response)


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


async def test_recognize_happy_path_one_item(monkeypatch: pytest.MonkeyPatch) -> None:
    recognition = _one_item()
    _install_fake_client(monkeypatch, _FakeResponse(parsed_output=recognition))

    result = await vision.recognize(b"fake-bytes", "image/jpeg", [])

    assert result == recognition
    assert len(result.items) == 1


async def test_recognize_multi_item_plate(monkeypatch: pytest.MonkeyPatch) -> None:
    recognition = Recognition(
        items=[
            RecognizedItem(
                product="chicken breast, raw",
                weight_g=150.0,
                raw_digits="150",
                confidence=0.9,
            ),
            RecognizedItem(
                product="broccoli",
                weight_g=80.0,
                raw_digits="080",
                confidence=0.85,
            ),
        ]
    )
    _install_fake_client(monkeypatch, _FakeResponse(parsed_output=recognition))

    result = await vision.recognize(b"fake-bytes", "image/jpeg", [])

    assert len(result.items) == 2
    assert result.items[0].product == "chicken breast, raw"
    assert result.items[1].product == "broccoli"


async def test_recognize_raises_when_parsed_output_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_client(monkeypatch, _FakeResponse(parsed_output=None, stop_reason="refusal"))

    with pytest.raises(vision.RecognitionError):
        await vision.recognize(b"fake-bytes", "image/jpeg", [])


async def test_recognize_injects_recent_foods_into_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recognition = _one_item()
    fake_client = _install_fake_client(monkeypatch, _FakeResponse(parsed_output=recognition))

    await vision.recognize(b"fake-bytes", "image/jpeg", ["chicken breast, raw", "greek yogurt"])

    sent_messages = fake_client.messages.last_kwargs["messages"]
    text_blocks = [
        block["text"] for block in sent_messages[0]["content"] if block["type"] == "text"
    ]
    combined = "\n".join(text_blocks)
    assert "chicken breast, raw" in combined
    assert "greek yogurt" in combined


async def test_recognize_uses_env_model_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    recognition = _one_item()
    fake_client = _install_fake_client(monkeypatch, _FakeResponse(parsed_output=recognition))

    await vision.recognize(b"fake-bytes", "image/jpeg", [])

    assert fake_client.messages.last_kwargs["model"] == "claude-sonnet-5"


def test_model_id_defaults_to_haiku(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    assert vision._DEFAULT_MODEL == "claude-haiku-4-5"


def test_digits_agree_rejects_parsed_weight_that_differs_from_transcription() -> None:
    # Observed live: model transcribed "170" but reported 170.1 and auto-submitted.
    bad = RecognizedItem(product="Egg", weight_g=170.1, raw_digits="170", confidence=0.85)
    assert not vision._digits_agree(bad)
    good = RecognizedItem(product="Egg", weight_g=117.0, raw_digits="117", confidence=0.9)
    assert vision._digits_agree(good)


def test_digits_agree_allows_unit_conversion() -> None:
    oz = RecognizedItem(product="Egg", weight_g=113.4, raw_digits="4.0 oz", confidence=0.9)
    assert vision._digits_agree(oz)
