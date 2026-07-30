"""Durable benchmark run artifacts and hash-checked resume support."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import yaml


TERMINAL_RECORD_STATUSES = {
    "ok",
    "invalid_output",
    "truncated_output",
    "backend_error",
    "image_error",
}


def canonical_hash(value: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible value."""

    encoded = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Filesystem paths belonging to one benchmark run."""

    directory: Path
    manifest: Path
    config_snapshot: Path
    selection: Path
    predictions: Path
    metrics: Path
    report: Path
    environment: Path
    rendered_prompts: Path
    server_log: Path

    @classmethod
    def from_directory(cls, directory: Path) -> "RunPaths":
        return cls(
            directory=directory,
            manifest=directory / "run_manifest.yaml",
            config_snapshot=directory / "config_snapshot.yaml",
            selection=directory / "selection.json",
            predictions=directory / "predictions.jsonl",
            metrics=directory / "metrics.json",
            report=directory / "report.html",
            environment=directory / "environment.json",
            rendered_prompts=directory / "rendered_prompts.jsonl",
            server_log=directory / "vllm_server.log",
        )


def create_run_directory(
    *,
    output_root: Path,
    benchmark_id: str,
    model_id: str,
    identity_hash: str,
    now: datetime | None = None,
) -> RunPaths:
    """Create a collision-safe run directory without overwriting a prior run."""

    timestamp = (now or datetime.now(timezone.utc)).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    directory = (
        output_root
        / benchmark_id
        / model_id
        / f"{timestamp}_{identity_hash[:8]}"
    )
    if directory.exists():
        suffix = 2
        while directory.with_name(f"{directory.name}_{suffix}").exists():
            suffix += 1
        directory = directory.with_name(f"{directory.name}_{suffix}")
    directory.mkdir(parents=True, exist_ok=False)
    return RunPaths.from_directory(directory)


class RunWriter:
    """Write one append-safe run and validate its resume identity."""

    def __init__(
        self,
        paths: RunPaths,
        *,
        identity: dict[str, str],
        resume: bool,
    ) -> None:
        self.paths = paths
        self.identity = dict(identity)
        self.resume = resume
        self.paths.directory.mkdir(parents=True, exist_ok=True)
        if resume:
            self._validate_resume_identity()
        elif self.paths.manifest.exists() or self.paths.predictions.exists():
            raise FileExistsError(
                f"Run directory is not empty: {self.paths.directory}"
            )

    def initialize(
        self,
        *,
        manifest: dict[str, Any],
        config_snapshot: dict[str, Any],
        selection: dict[str, Any],
        environment: dict[str, Any],
    ) -> None:
        """Create immutable snapshots and the initial running manifest."""

        if self.paths.manifest.exists():
            if self.resume:
                document = _load_yaml(self.paths.manifest)
                previous_status = str(document.get("status", "unknown"))
                history = document.get("resume_history", [])
                if not isinstance(history, list):
                    history = []
                history.append(
                    {
                        "resumed_at": _utc_now(),
                        "previous_status": previous_status,
                    }
                )
                document["resume_history"] = history
                document["status"] = "running"
                document.pop("finished_at", None)
                document.pop("counts", None)
                document.pop("error", None)
                _atomic_yaml(self.paths.manifest, document)
            return
        document = dict(manifest)
        document["status"] = "running"
        document["identity"] = dict(self.identity)
        document["started_at"] = _utc_now()
        _atomic_yaml(self.paths.manifest, document)
        _atomic_yaml(self.paths.config_snapshot, config_snapshot)
        _atomic_json(self.paths.selection, selection)
        _atomic_json(self.paths.environment, environment)

    def append_prediction(self, record: dict[str, Any]) -> None:
        """Append and fsync one terminal prediction record."""

        status = str(record.get("status", ""))
        if status not in TERMINAL_RECORD_STATUSES:
            raise ValueError(f"Prediction status is not terminal: {status!r}")
        _append_jsonl(self.paths.predictions, record)

    def append_rendered_prompt(self, record: dict[str, Any]) -> None:
        """Append a rendered prompt without embedding image data."""

        _append_jsonl(self.paths.rendered_prompts, record)

    def completed_task_ids(self) -> set[str]:
        """Return terminal task IDs already written by this run."""

        return {
            str(record["task_id"])
            for record in read_jsonl(self.paths.predictions)
            if str(record.get("status")) in TERMINAL_RECORD_STATUSES
            and record.get("task_id") is not None
        }

    def write_metrics(self, metrics: dict[str, Any]) -> None:
        _atomic_json(self.paths.metrics, metrics)

    def finalize(
        self,
        *,
        status: str,
        counts: dict[str, int],
        error: str | None = None,
    ) -> None:
        """Set the final run status without altering immutable snapshots."""

        if status not in {"completed", "interrupted", "failed"}:
            raise ValueError(f"Unsupported final run status: {status!r}")
        document = _load_yaml(self.paths.manifest)
        document["status"] = status
        document["finished_at"] = _utc_now()
        document["counts"] = dict(counts)
        if error:
            document["error"] = error
        else:
            document.pop("error", None)
        _atomic_yaml(self.paths.manifest, document)

    def _validate_resume_identity(self) -> None:
        if not self.paths.manifest.exists():
            raise FileNotFoundError(
                f"Resume manifest is missing: {self.paths.manifest}"
            )
        document = _load_yaml(self.paths.manifest)
        existing = document.get("identity")
        if existing != self.identity:
            raise ValueError(
                "Resume identity mismatch; model, benchmark, dataset, "
                "selection, or configuration changed"
            )
        if document.get("status") == "completed":
            raise ValueError("Completed benchmark runs cannot be resumed")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL while tolerating only a truncated final line."""

    if not path.exists():
        return []
    raw = path.read_bytes()
    raw_lines = raw.splitlines()
    has_complete_final_line = raw.endswith((b"\n", b"\r"))
    records: list[dict[str, Any]] = []
    for index, raw_line in enumerate(raw_lines):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError:
            if (
                index == len(raw_lines) - 1
                and not has_complete_final_line
            ):
                break
            raise ValueError(
                f"Invalid JSONL record at {path}:{index + 1}"
            ) from None
        if not isinstance(value, dict):
            raise ValueError(
                f"JSONL record at {path}:{index + 1} must be an object"
            )
        records.append(value)
    return records


def count_statuses(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Count terminal prediction statuses."""

    result = {status: 0 for status in sorted(TERMINAL_RECORD_STATUSES)}
    total = 0
    for record in records:
        total += 1
        status = str(record.get("status", ""))
        result[status] = result.get(status, 0) + 1
    result["total"] = total
    return result


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            _json_safe(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    _atomic_write(path, payload + "\n")


def _atomic_yaml(path: Path, value: Any) -> None:
    payload = yaml.safe_dump(
        _json_safe(value),
        sort_keys=False,
        allow_unicode=True,
    )
    _atomic_write(path, payload)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML object in {path}")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)
