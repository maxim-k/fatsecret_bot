"""Pure persistence for aliases, the pending-item queue, and the audit log.

No business logic, normalisation, or API calls live here. Callers pass
already-normalised keys and validated :mod:`models` objects; this module's
only job is getting them on and off disk in ``DATA_DIR`` without corrupting
what's already there.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from pydantic import TypeAdapter

from models import Alias, QueuedItem

_ALIASES_FILENAME = "aliases.json"
_QUEUE_FILENAME = "queue.json"
_LOG_FILENAME = "log.jsonl"

_aliases_adapter: TypeAdapter[dict[str, Alias]] = TypeAdapter(dict[str, Alias])
_queue_adapter: TypeAdapter[list[QueuedItem]] = TypeAdapter(list[QueuedItem])


def _data_dir() -> Path:
    """Resolve the data directory, creating it on first use.

    Read on every call, not once at import, so a test can repoint the module
    without reloading it.
    """
    data_dir = Path(os.environ.get("DATA_DIR", "./data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _read_or_none(path: Path) -> bytes | None:
    """Read a file's bytes, or ``None`` if it is absent or holds only whitespace.

    A never-created file and a zero-byte one both mean "nothing saved yet".
    Neither is corruption. Neither may raise.
    """
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    return raw if raw.strip() else None


def _atomic_write(path: Path, data: bytes) -> None:
    """Replace ``path`` with ``data`` without ever leaving it half-written.

    A direct write can be interrupted by a crash or kill signal partway
    through, leaving truncated or mixed-content JSON on disk — for
    ``aliases.json`` that means losing the bot's learned state, the worst
    non-safety failure in this project. Writing the new content to a sibling
    temp file first and swapping it in with :meth:`Path.replace` (an atomic
    rename on POSIX) means a reader only ever sees the fully-old or
    fully-new file, never something in between.

    :param path: Destination file.
    :param data: Bytes to write.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as tmp_file:
            tmp_file.write(data)
        tmp_path.replace(path)
    except BaseException:
        # Best-effort cleanup: an incomplete temp file left behind is
        # harmless clutter, but leaving it silently masks the real error.
        tmp_path.unlink(missing_ok=True)
        raise


def load_aliases() -> dict[str, Alias]:
    """Read the alias table, or an empty one if nothing is saved yet."""
    raw = _read_or_none(_data_dir() / _ALIASES_FILENAME)
    if raw is None:
        return {}
    return _aliases_adapter.validate_json(raw)


def save_aliases(aliases: dict[str, Alias]) -> None:
    """Write the alias table, replacing whatever was there."""
    _atomic_write(_data_dir() / _ALIASES_FILENAME, _aliases_adapter.dump_json(aliases, indent=2))


def load_queue() -> list[QueuedItem]:
    """Read the review queue, or an empty one if nothing is saved yet."""
    raw = _read_or_none(_data_dir() / _QUEUE_FILENAME)
    if raw is None:
        return []
    return _queue_adapter.validate_json(raw)


def save_queue(items: list[QueuedItem]) -> None:
    """Write the review queue, replacing whatever was there."""
    _atomic_write(_data_dir() / _QUEUE_FILENAME, _queue_adapter.dump_json(items, indent=2))


def append_log(event: str, payload: dict[str, object]) -> None:
    """Append one record to the audit log.

    The log is append-only and never rewritten, so a plain append (unlike
    the alias and queue files) needs no temp-file dance: a crash mid-write
    can at worst truncate the last line, never corrupt an earlier one.

    :param event: Short machine-readable event name, e.g. ``"submitted"``.
    :param payload: Event-specific details; must be JSON-serialisable.
    """
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": event,
        "payload": payload,
    }
    with (_data_dir() / _LOG_FILENAME).open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_photo(photo_sha256: str, data: bytes) -> None:
    """Keep a copy of an incoming photo.

    Kept so model accuracy can be measured offline over real images rather than
    argued about, and so a re-read is possible after the request that carried
    the bytes has finished.

    :param photo_sha256: SHA-256 hex digest of the photo, used as the filename.
    :param data: The image bytes exactly as received.
    """
    photos = _data_dir() / "photos"
    photos.mkdir(exist_ok=True)
    (photos / f"{photo_sha256}.jpg").write_bytes(data)
