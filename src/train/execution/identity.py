"""Run identity, status transitions, and checkpoint resume validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import ClassVar

from src.train.backends.contracts import CheckpointEvent, CheckpointObserver
from src.train.execution.io import (
    JsonValue,
    atomic_write_json,
    read_json_object,
)


class RunStatus(StrEnum):
    """Durable lifecycle states for a training execution."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """Immutable inputs that must match when training is resumed."""

    experiment_id: str
    run_id: str
    config_hash: str
    dataset_hash: str
    model_id: str
    model_revision: str
    execution_profile: str

    def __post_init__(self) -> None:
        """Reject incomplete identities before creating output files."""
        values = asdict(self)
        empty = [name for name, value in values.items() if not value.strip()]
        if empty:
            raise ValueError(f"Run identity fields must not be empty: {empty}")


@dataclass(frozen=True, slots=True)
class RunLayout:
    """Expose the canonical directory layout for one run."""

    root: Path

    @property
    def manifests(self) -> Path:
        """Return the manifest directory."""
        return self.root / "manifests"

    @property
    def checkpoints(self) -> Path:
        """Return the checkpoint directory used as Trainer output."""
        return self.root / "checkpoints"

    @property
    def logs(self) -> Path:
        """Return the canonical log directory."""
        return self.root / "logs"

    def create(self) -> None:
        """Create every canonical run subdirectory."""
        for name in (
            "manifests",
            "logs",
            "tensorboard",
            "checkpoints",
            "predictions",
            "metrics",
            "figures",
            "tables",
            "report",
        ):
            (self.root / name).mkdir(parents=True, exist_ok=True)


class RunIdentityStore:
    """Persist an identity once and prevent incompatible reuse."""

    def __init__(self, path: Path) -> None:
        """Set the immutable identity manifest path."""
        self._path = path

    def ensure(self, identity: RunIdentity) -> None:
        """Create or verify the immutable identity document."""
        expected = _identity_json(identity)
        if self._path.exists():
            actual = read_json_object(self._path)
            if actual != expected:
                raise RuntimeError(f"Run directory identity mismatch: {self._path}")
            return
        atomic_write_json(self._path, expected)


class RunStatusStore:
    """Apply explicit and auditable lifecycle transitions."""

    _ALLOWED: ClassVar[dict[RunStatus, frozenset[RunStatus]]] = {
        RunStatus.CREATED: frozenset({RunStatus.RUNNING}),
        RunStatus.RUNNING: frozenset(
            {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.INTERRUPTED}
        ),
        RunStatus.FAILED: frozenset({RunStatus.RUNNING}),
        RunStatus.INTERRUPTED: frozenset({RunStatus.RUNNING}),
        RunStatus.COMPLETED: frozenset(),
    }

    def __init__(self, path: Path) -> None:
        """Set the mutable status manifest path."""
        self._path = path

    def initialize(self) -> None:
        """Create a new ``created`` state without overwriting existing state."""
        if self._path.exists():
            return
        now = datetime.now(UTC).isoformat()
        atomic_write_json(
            self._path,
            {
                "status": RunStatus.CREATED.value,
                "created_at_utc": now,
                "updated_at_utc": now,
                "error_type": None,
                "error_message": None,
            },
        )

    def current(self) -> RunStatus:
        """Read the current durable lifecycle state."""
        value = read_json_object(self._path).get("status")
        if not isinstance(value, str):
            raise ValueError(f"Missing status in {self._path}")
        return RunStatus(value)

    def transition(
        self,
        status: RunStatus,
        *,
        error: BaseException | None = None,
    ) -> None:
        """Atomically apply one valid state transition."""
        payload = read_json_object(self._path)
        previous_value = payload.get("status")
        if not isinstance(previous_value, str):
            raise ValueError(f"Missing status in {self._path}")
        previous = RunStatus(previous_value)
        if status not in self._ALLOWED[previous]:
            raise RuntimeError(
                f"Invalid run transition: {previous.value} -> {status.value}"
            )
        payload["status"] = status.value
        payload["updated_at_utc"] = datetime.now(UTC).isoformat()
        payload["error_type"] = type(error).__name__ if error else None
        payload["error_message"] = str(error) if error else None
        atomic_write_json(self._path, payload)


class CheckpointRecorder(CheckpointObserver):
    """Attach the immutable run identity to every Trainer checkpoint."""

    def __init__(
        self,
        identity: RunIdentity,
        *,
        downstream: CheckpointObserver | None = None,
    ) -> None:
        """Retain identity, events, and an optional post-manifest observer.

        The downstream observer is invoked only after the resumability
        manifest has been written. This lets remote storage reject an
        incomplete checkpoint without weakening the local resume contract.
        """
        self._identity = identity
        self._downstream = downstream
        self._events: list[CheckpointEvent] = []

    @property
    def events(self) -> tuple[CheckpointEvent, ...]:
        """Return checkpoint events in callback order."""
        return tuple(self._events)

    def on_checkpoint(self, event: CheckpointEvent) -> None:
        """Persist a resume manifest inside a completed checkpoint directory."""
        if not event.path.is_dir():
            raise RuntimeError(
                f"Trainer callback referenced a missing checkpoint: {event.path}"
            )
        artifacts = _checkpoint_artifacts(event.path)
        atomic_write_json(
            event.path / "isep_checkpoint.json",
            {
                "identity": _identity_json(self._identity),
                "global_step": event.global_step,
                "epoch": event.epoch,
                "artifacts": artifacts,
            },
        )
        self._events.append(event)
        if self._downstream is not None:
            self._downstream.on_checkpoint(event)


def read_checkpoint_event(checkpoint_path: Path) -> CheckpointEvent:
    """Reconstruct a typed event from an existing ISEP checkpoint manifest.

    Args:
        checkpoint_path: Locally persisted, already validated checkpoint.

    Returns:
        The checkpoint coordinates required by downstream observers.

    Raises:
        ValueError: If the manifest coordinates have invalid types.
    """

    payload = read_json_object(checkpoint_path / "isep_checkpoint.json")
    global_step = payload.get("global_step")
    epoch = payload.get("epoch")
    if isinstance(global_step, bool) or not isinstance(global_step, int):
        raise ValueError("Checkpoint manifest global_step must be an integer")
    if epoch is not None and (
        isinstance(epoch, bool) or not isinstance(epoch, int | float)
    ):
        raise ValueError("Checkpoint manifest epoch must be numeric or null")
    return CheckpointEvent(
        path=checkpoint_path,
        global_step=global_step,
        epoch=float(epoch) if epoch is not None else None,
    )


def validate_resume_checkpoint(
    checkpoint_path: Path,
    expected_identity: RunIdentity,
) -> None:
    """Reject checkpoint resume when any scientific input hash differs."""
    manifest_path = checkpoint_path / "isep_checkpoint.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Checkpoint is missing its ISEP resume manifest: {manifest_path}"
        )
    payload = read_json_object(manifest_path)
    identity = payload.get("identity")
    if identity != _identity_json(expected_identity):
        raise RuntimeError(
            "Checkpoint identity does not match config, dataset, or model"
        )
    actual_artifacts = _checkpoint_artifacts(checkpoint_path)
    declared_artifacts = payload.get("artifacts")
    if declared_artifacts != actual_artifacts:
        raise RuntimeError("Checkpoint state files are missing, replaced, or corrupted")


def _checkpoint_artifacts(checkpoint_path: Path) -> dict[str, JsonValue]:
    required = (
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "adapter_config.json",
    )
    missing = [name for name in required if not (checkpoint_path / name).is_file()]
    adapter_candidates = tuple(
        name
        for name in ("adapter_model.safetensors", "adapter_model.bin")
        if (checkpoint_path / name).is_file()
    )
    if missing or len(adapter_candidates) != 1:
        details = [*missing]
        if not adapter_candidates:
            details.append("adapter_model.safetensors|adapter_model.bin")
        elif len(adapter_candidates) > 1:
            details.append("multiple adapter weight files")
        raise RuntimeError(
            "Checkpoint is not resumable; invalid state files: " + ", ".join(details)
        )
    names = (*required, adapter_candidates[0])
    return {name: _sha256(checkpoint_path / name) for name in names}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: JsonValue) -> str:
    """Return a SHA-256 digest of a canonical JSON value."""
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _identity_json(identity: RunIdentity) -> dict[str, JsonValue]:
    return {
        "experiment_id": identity.experiment_id,
        "run_id": identity.run_id,
        "config_hash": identity.config_hash,
        "dataset_hash": identity.dataset_hash,
        "model_id": identity.model_id,
        "model_revision": identity.model_revision,
        "execution_profile": identity.execution_profile,
    }
