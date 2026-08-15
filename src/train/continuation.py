"""Audited staging of an E1 epoch-three checkpoint for continued training."""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Mapping
from pathlib import Path

from src.train.config import TrainingConfig
from src.train.execution import RunIdentity, read_checkpoint_event
from src.train.execution.identity import (
    CheckpointRecorder,
    validate_resume_checkpoint,
)
from src.train.execution.io import atomic_write_json, read_json_object


def stage_continuation_checkpoint(
    config: TrainingConfig,
    *,
    run_directory: Path,
    identity: RunIdentity,
) -> Path | None:
    """Copy and rebind a verified parent checkpoint into a new run.

    The parent run remains byte-for-byte unchanged. The copied checkpoint keeps
    the Trainer, optimizer, scheduler, RNG, and adapter states, while its ISEP
    resume manifest is rebound to the new run identity. A separate provenance
    manifest records the parent identity and adapter digest.

    Args:
        config: Validated training configuration.
        run_directory: Newly created continuation run directory.
        identity: Immutable identity of the continuation run.

    Returns:
        The staged checkpoint path, or ``None`` for a fresh three-epoch run.
    """

    continuation = config.continuation
    if continuation is None:
        return None
    parent_run = config.resolve_path(continuation.parent_run_directory)
    parent_checkpoint = parent_run / "checkpoints" / continuation.parent_checkpoint_id
    parent_identity = _parent_identity(parent_run)
    validate_resume_checkpoint(parent_checkpoint, parent_identity)
    _require_completed_parent(parent_run, continuation.parent_checkpoint_id)
    adapter = parent_checkpoint / "adapter_model.safetensors"
    adapter_sha256 = _sha256(adapter)
    if adapter_sha256 != continuation.parent_adapter_sha256:
        raise RuntimeError(
            "Parent adapter SHA-256 differs from the continuation config"
        )

    target = run_directory / "checkpoints" / continuation.parent_checkpoint_id
    if target.exists():
        validate_resume_checkpoint(target, identity)
        return target
    staging = target.with_name(target.name + ".staging")
    if staging.exists():
        raise RuntimeError(f"Stale continuation staging directory exists: {staging}")
    shutil.copytree(parent_checkpoint, staging, copy_function=shutil.copy2)
    os.replace(staging, target)
    CheckpointRecorder(identity).on_checkpoint(read_checkpoint_event(target))
    validate_resume_checkpoint(target, identity)
    atomic_write_json(
        run_directory / "manifests" / "continuation.json",
        {
            "additional_epochs": continuation.additional_epochs,
            "parent_adapter_sha256": adapter_sha256,
            "parent_checkpoint": str(parent_checkpoint),
            "parent_epoch": continuation.parent_epoch,
            "parent_identity": read_json_object(
                parent_run / "manifests" / "run_identity.json"
            ),
            "staged_checkpoint": str(target),
            "target_epoch": config.trainer.epochs,
        },
    )
    return target


def _parent_identity(parent_run: Path) -> RunIdentity:
    payload = read_json_object(parent_run / "manifests" / "run_identity.json")
    return RunIdentity(
        experiment_id=_required_str(payload, "experiment_id"),
        run_id=_required_str(payload, "run_id"),
        config_hash=_required_str(payload, "config_hash"),
        dataset_hash=_required_str(payload, "dataset_hash"),
        model_id=_required_str(payload, "model_id"),
        model_revision=_required_str(payload, "model_revision"),
        execution_profile=_required_str(payload, "execution_profile"),
    )


def _require_completed_parent(parent_run: Path, checkpoint_id: str) -> None:
    status = read_json_object(parent_run / "manifests" / "run_status.json")
    if status.get("status") != "completed":
        raise RuntimeError("Continuation parent run is not completed")
    best = read_json_object(parent_run / "manifests" / "best_checkpoint.json")
    if best.get("checkpoint_id") != checkpoint_id:
        raise RuntimeError("Continuation must start from the selected best checkpoint")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Parent run identity field {key!r} is missing")
    return value
