"""Read and append UTF-8 JSONL used by Stage A and Stage B."""

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from project.teacher.schemas import (
    ManifestRow,
    RecordStatus,
    StageAFileRow,
    StageBFileRow,
)


def load_manifest(path: Path) -> list[ManifestRow]:
    """Load the generation manifest.

    Args:
        path: JSONL file with sample_id, image_path, gold_diagnosis.

    Returns:
        Rows in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a line is not valid JSON or fails ManifestRow.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    rows: list[ManifestRow] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        rows.append(_parse_manifest_line(line, line_number=line_number, path=path))
    return rows


def _parse_manifest_line(line: str, *, line_number: int, path: Path) -> ManifestRow:
    """Parse one manifest JSONL line."""
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{line_number}: invalid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{path}:{line_number}: JSONL value must be an object")

    try:
        return ManifestRow.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"{path}:{line_number}: invalid manifest row") from exc


def load_stage_a_rows(path: Path) -> list[StageAFileRow]:
    """Load Stage A JSONL.

    Args:
        path: File written by ``run_stage_a``.

    Returns:
        Rows in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a line cannot be parsed as StageAFileRow.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Stage A JSONL not found: {path}")

    rows: list[StageAFileRow] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        rows.append(_parse_stage_a_line(line, line_number=line_number, path=path))
    return rows


def _parse_stage_a_line(line: str, *, line_number: int, path: Path) -> StageAFileRow:
    """Parse one Stage A JSONL line."""
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{line_number}: invalid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{path}:{line_number}: JSONL value must be an object")

    try:
        return StageAFileRow.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"{path}:{line_number}: invalid Stage A row") from exc


def completed_ids(path: Path) -> set[str]:
    """Return sample ids whose first successful row is ``status=ok``.

    Invalid lines raise. A later error row does not remove an earlier ok.

    Args:
        path: Existing JSONL, or a path that does not exist yet.

    Returns:
        Ids that resume should skip.

    Raises:
        ValueError: If an existing line is invalid JSON or schema.
    """
    if not path.is_file():
        return set()

    done: set[str] = set()
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        payload = _load_json_object(line, line_number=line_number, path=path)
        sample_id = payload.get("sample_id")
        status = payload.get("status")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{path}:{line_number}: missing sample_id")
        if sample_id in done:
            continue
        if status == RecordStatus.OK:
            done.add(sample_id)
    return done


def _load_json_object(line: str, *, line_number: int, path: Path) -> dict[str, object]:
    """Load one JSON object from a JSONL line."""
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{line_number}: invalid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{path}:{line_number}: JSONL value must be an object")
    return payload


def append_jsonl(path: Path, record: StageAFileRow | StageBFileRow) -> None:
    """Append one validated record as a JSON line.

    Creates parent directories. Flushes after write.

    Args:
        path: JSONL destination.
        record: Stage A or Stage B file row.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = record.model_dump_json() + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()


def index_ok_stage_a(rows: Iterable[StageAFileRow]) -> dict[str, StageAFileRow]:
    """Index the first ok Stage A row per sample_id.

    Args:
        rows: Loaded Stage A file rows.

    Returns:
        Mapping sample_id → first ok row.
    """
    indexed: dict[str, StageAFileRow] = {}
    for row in rows:
        if row.status is not RecordStatus.OK:
            continue
        if row.sample_id in indexed:
            continue
        indexed[row.sample_id] = row
    return indexed
