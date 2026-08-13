"""Atomic, path-safe storage for reproducible training runs."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from .types import JsonValue, RunStatus

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SECTIONS = (
    "manifests",
    "logs",
    "tensorboard",
    "checkpoints",
    "predictions",
    "metrics",
    "figures",
    "tables",
    "report",
)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Atomically replace a file with bytes written in the same directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace a UTF-8 text file."""

    atomic_write_bytes(path, content.encode("utf-8"))


@dataclass(frozen=True, slots=True)
class RunLayout:
    """Canonical directory layout for a single experiment execution."""

    run_directory: Path

    @property
    def experiment_id(self) -> str:
        """Return the experiment directory name."""

        return self.run_directory.parent.name

    @property
    def run_id(self) -> str:
        """Return the run directory name."""

        return self.run_directory.name

    def section(self, name: str) -> Path:
        """Return a known artefact section directory."""

        if name not in _SECTIONS:
            raise ValueError(f"Unknown artefact section: {name!r}")
        return self.run_directory / name

    @property
    def manifests(self) -> Path:
        """Return the manifests directory."""

        return self.section("manifests")

    @property
    def metrics(self) -> Path:
        """Return the metrics directory."""

        return self.section("metrics")

    @property
    def predictions(self) -> Path:
        """Return the predictions directory."""

        return self.section("predictions")

    @property
    def figures(self) -> Path:
        """Return the figures directory."""

        return self.section("figures")

    @property
    def tables(self) -> Path:
        """Return the tables directory."""

        return self.section("tables")

    @property
    def report(self) -> Path:
        """Return the report directory."""

        return self.section("report")


class ArtifactStore:
    """Atomic writer constrained to one canonical training run tree."""

    def __init__(self, layout: RunLayout) -> None:
        """Initialize a store for an already resolved run layout."""

        self.layout = layout

    @classmethod
    def create(
        cls,
        root: Path,
        experiment_id: str,
        run_id: str,
        *,
        resume: bool = False,
    ) -> ArtifactStore:
        """Create a run tree, rejecting accidental reuse unless resuming."""

        for name, value in (
            ("experiment_id", experiment_id),
            ("run_id", run_id),
        ):
            if not _IDENTIFIER.fullmatch(value):
                raise ValueError(f"Unsafe {name}: {value!r}")
        run_directory = (root / experiment_id / run_id).resolve()
        if run_directory.exists() and not resume:
            raise FileExistsError(f"Run already exists: {run_directory}")
        return cls.at(run_directory, create=True)

    @classmethod
    def at(cls, run_directory: Path, *, create: bool = False) -> ArtifactStore:
        """Open a run directory directly, optionally creating its tree."""

        resolved = run_directory.resolve()
        if create:
            resolved.mkdir(parents=True, exist_ok=True)
            for section in _SECTIONS:
                (resolved / section).mkdir(exist_ok=True)
        elif not resolved.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {resolved}")
        return cls(RunLayout(resolved))

    def path(self, section: str, filename: str) -> Path:
        """Resolve a simple filename inside a known run section."""

        if Path(filename).name != filename or filename in {"", ".", ".."}:
            raise ValueError(f"Unsafe artefact filename: {filename!r}")
        return self.layout.section(section) / filename

    def write_json(self, section: str, filename: str, payload: JsonValue) -> Path:
        """Atomically write indented, finite JSON."""

        destination = self.path(section, filename)
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        atomic_write_text(destination, encoded + "\n")
        return destination

    def write_yaml(self, section: str, filename: str, payload: JsonValue) -> Path:
        """Atomically write a human-readable YAML snapshot."""

        destination = self.path(section, filename)
        encoded = yaml.safe_dump(payload, allow_unicode=True, sort_keys=True)
        atomic_write_text(destination, encoded)
        return destination

    def write_text(self, section: str, filename: str, content: str) -> Path:
        """Atomically write UTF-8 text inside a run section."""

        destination = self.path(section, filename)
        atomic_write_text(destination, content)
        return destination

    def append_jsonl(self, section: str, filename: str, payload: JsonValue) -> Path:
        """Append one JSON object via an atomic whole-file replacement."""

        destination = self.path(section, filename)
        previous = (
            destination.read_text(encoding="utf-8") if destination.is_file() else ""
        )
        line = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        atomic_write_text(destination, previous + line + "\n")
        return destination

    def write_status(
        self,
        status: RunStatus,
        *,
        detail: str | None = None,
    ) -> Path:
        """Write the explicit terminal or in-progress run state."""

        payload: dict[str, JsonValue] = {
            "status": status,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        if detail is not None:
            payload["detail"] = detail
        return self.write_json("manifests", "status.json", payload)
