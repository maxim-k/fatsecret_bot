from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

import store
from models import Alias, ItemState, Meal, QueuedItem, RecognizedItem


@pytest.fixture(autouse=True)
def _data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    return data_dir


def _make_alias() -> Alias:
    return Alias(food_id=1, food_name="Chicken breast", serving_id=10, serving_grams=100.0, hits=3)


def _make_queued_item(
    photo_sha256: str = "abc123", state: ItemState = ItemState.PENDING
) -> QueuedItem:
    return QueuedItem(
        id="q1",
        photo_sha256=photo_sha256,
        day=date(2026, 8, 13),
        meal=Meal.LUNCH,
        item=RecognizedItem(
            product="chicken breast", weight_g=150.0, raw_digits="150", confidence=0.9
        ),
        state=state,
    )


def test_load_aliases_missing_file_is_empty() -> None:
    assert store.load_aliases() == {}


def test_aliases_roundtrip() -> None:
    aliases = {"chicken breast": _make_alias()}
    store.save_aliases(aliases)
    assert store.load_aliases() == aliases


def test_load_queue_missing_file_is_empty() -> None:
    assert store.load_queue() == []


def test_queue_roundtrip() -> None:
    items = [_make_queued_item()]
    store.save_queue(items)
    assert store.load_queue() == items


def test_empty_aliases_file_loads_empty(_data_dir: Path) -> None:
    _data_dir.mkdir(parents=True)
    (_data_dir / "aliases.json").write_text("")
    assert store.load_aliases() == {}


def test_empty_queue_file_loads_empty(_data_dir: Path) -> None:
    _data_dir.mkdir(parents=True)
    (_data_dir / "queue.json").write_text("   \n")
    assert store.load_queue() == []


def test_corrupt_aliases_file_raises(_data_dir: Path) -> None:
    _data_dir.mkdir(parents=True)
    (_data_dir / "aliases.json").write_text("not json at all")
    with pytest.raises(ValueError):
        store.load_aliases()


def test_aliases_file_with_wrong_schema_raises(_data_dir: Path) -> None:
    _data_dir.mkdir(parents=True)
    (_data_dir / "aliases.json").write_text(json.dumps({"x": {"food_id": "not an int"}}))
    with pytest.raises(ValueError):
        store.load_aliases()


def test_corrupt_queue_file_raises(_data_dir: Path) -> None:
    _data_dir.mkdir(parents=True)
    (_data_dir / "queue.json").write_text("{not valid json")
    with pytest.raises(ValueError):
        store.load_queue()


def test_atomic_write_leaves_no_temp_files(_data_dir: Path) -> None:
    store.save_aliases({"a": _make_alias()})
    leftovers = [p.name for p in _data_dir.iterdir() if p.name != "aliases.json"]
    assert leftovers == []


def test_data_dir_created_if_missing(_data_dir: Path) -> None:
    assert not _data_dir.exists()
    store.save_queue([])
    assert _data_dir.is_dir()


def test_is_duplicate_true_for_submitted_item() -> None:
    store.save_queue([_make_queued_item(photo_sha256="hash1", state=ItemState.SUBMITTED)])
    assert store.is_duplicate("hash1") is True


def test_is_duplicate_false_for_pending_only() -> None:
    store.save_queue([_make_queued_item(photo_sha256="hash2", state=ItemState.PENDING)])
    assert store.is_duplicate("hash2") is False


def test_is_duplicate_false_when_hash_absent() -> None:
    store.save_queue([_make_queued_item(photo_sha256="hash3", state=ItemState.SUBMITTED)])
    assert store.is_duplicate("does-not-exist") is False


def test_append_log_writes_one_line_per_call(_data_dir: Path) -> None:
    store.append_log("event_one", {"key": "value"})
    store.append_log("event_two", {"key": 2})

    lines = (_data_dir / "log.jsonl").read_text().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["event"] == "event_one"
    assert first["payload"] == {"key": "value"}
    assert "timestamp" in first

    second = json.loads(lines[1])
    assert second["event"] == "event_two"
    assert second["payload"] == {"key": 2}
