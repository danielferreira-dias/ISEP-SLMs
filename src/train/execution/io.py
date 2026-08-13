"""Atomic local file primitives for reproducible run artefacts."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace a UTF-8 text file on the same filesystem.

    Args:
        path: Destination file.
        text: Complete new file contents.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: JsonValue) -> None:
    """Atomically replace a canonical, human-readable JSON document."""
    atomic_write_text(
        path,
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
    )


def read_json_object(path: Path) -> dict[str, JsonValue]:
    """Read a JSON object and reject other root value types."""
    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        raise ValueError(f"Expected a JSON object in {path}")
    return _validate_json_object(loaded, path=path)


def read_json_array(path: Path) -> list[JsonValue]:
    """Read a JSON array and validate every nested value.

    Args:
        path: UTF-8 JSON document whose root must be an array.

    Returns:
        A recursively validated JSON array.

    Raises:
        ValueError: If the document root is not an array or contains a value
            that cannot be represented by :class:`JsonValue`.
    """

    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return [_validate_json_value(item, path=path) for item in loaded]


def _validate_json_object(
    raw: dict[object, object],
    *,
    path: Path,
) -> dict[str, JsonValue]:
    output: dict[str, JsonValue] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError(f"Non-string JSON key in {path}")
        output[key] = _validate_json_value(value, path=path)
    return output


def _validate_json_value(value: object, *, path: Path) -> JsonValue:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite JSON number in {path}")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_validate_json_value(item, path=path) for item in value]
    if isinstance(value, dict):
        return _validate_json_object(value, path=path)
    raise ValueError(f"Unsupported JSON value in {path}: {type(value).__name__}")
