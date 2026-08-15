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

from models import Alias, ItemState, QueuedItem

_ALIASES_FILENAME = "aliases.json"
_QUEUE_FILENAME = "queue.json"
_LOG_FILENAME = "log.jsonl"

_aliases_adapter: TypeAdapter[dict[str, Alias]] = TypeAdapter(dict[str, Alias])
_queue_adapter: TypeAdapter[list[QueuedItem]] = TypeAdapter(list[QueuedItem])


def _data_dir() -> Path:
    """Resolve the data directory, creating it on first use.

    Read from the environment on every call, rather than once at import
    time, so tests (and anything else) can point the module at a different
    directory without reloading it.

    :returns: Directory that ``aliases.json``, ``queue.json`` and
        ``log.jsonl`` live in.
    """
    data_dir = Path(os.environ.get("DATA_DIR", "./data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _read_or_none(path: Path) -> bytes | None:
    """Read a file's raw bytes, treating "absent" and "empty" alike.

    Both a never-created file and a zero-byte one mean "nothing has been
    saved yet" — neither is corruption and neither should raise.

    :param path: File to read.
    :returns: File contents, or ``None`` if the file doesn't exist or holds
        only whitespace.
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
    """Load the learned product-to-food alias table.

    :returns: Mapping of already-normalised product string to :class:`Alias`.
        Empty if ``aliases.json`` is missing or empty.
    :raises ValueError: If the file exists but its contents aren't valid
        JSON matching the :class:`Alias` schema — surfaced rather than
        swallowed, since silently discarding learned state is worse than a
        crash.
    """
    raw = _read_or_none(_data_dir() / _ALIASES_FILENAME)
    if raw is None:
        return {}
    return _aliases_adapter.validate_json(raw)


def save_aliases(aliases: dict[str, Alias]) -> None:
    """Persist the alias table, atomically replacing the previous file.

    :param aliases: Mapping of already-normalised product string to
        :class:`Alias`, to write in full.
    """
    _atomic_write(_data_dir() / _ALIASES_FILENAME, _aliases_adapter.dump_json(aliases, indent=2))


def load_queue() -> list[QueuedItem]:
    """Load the queue of recognised items awaiting review or submission.

    :returns: Items in file order. Empty if ``queue.json`` is missing or
        empty.
    :raises ValueError: If the file exists but its contents aren't valid
        JSON matching the :class:`QueuedItem` schema.
    """
    raw = _read_or_none(_data_dir() / _QUEUE_FILENAME)
    if raw is None:
        return []
    return _queue_adapter.validate_json(raw)


def save_queue(items: list[QueuedItem]) -> None:
    """Persist the queue, atomically replacing the previous file.

    :param items: Full queue contents to write.
    """
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


def is_duplicate(photo_sha256: str) -> bool:
    """Check whether a photo has already produced a submitted queue item.

    Derived from the queue instead of tracked in a fourth file, so there is
    exactly one place recording what's been logged before.

    :param photo_sha256: SHA-256 hex digest of the photo bytes.
    :returns: ``True`` if some queue item with this hash reached
        :attr:`~models.ItemState.SUBMITTED`.
    """
    return any(
        item.photo_sha256 == photo_sha256 and item.state == ItemState.SUBMITTED
        for item in load_queue()
    )
