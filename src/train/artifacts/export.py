"""Non-overwriting export of generated artefacts into the dissertation tree."""

from __future__ import annotations

import os
from pathlib import Path

from .types import ExportResult

_ALLOWED = {".csv", ".tex", ".png", ".svg"}


def _exclusive_copy(source: Path, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError:
        return False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(source.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return True


def export_thesis_artifacts(
    run_directory: Path,
    destination: Path,
) -> ExportResult:
    """Copy figures and tables without replacing any existing thesis file."""

    copied: list[Path] = []
    skipped: list[Path] = []
    for section in ("figures", "tables"):
        source_directory = run_directory / section
        if not source_directory.is_dir():
            continue
        for source in sorted(source_directory.iterdir()):
            if not source.is_file() or source.suffix.lower() not in _ALLOWED:
                continue
            target = destination / section / source.name
            if _exclusive_copy(source, target):
                copied.append(target)
            else:
                skipped.append(target)
    return ExportResult(tuple(copied), tuple(skipped))
