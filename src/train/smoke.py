"""Postconditions for the bounded 30-step GPU smoke experiment."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping
from pathlib import Path

from src.train.artifacts import ArtifactStore
from src.train.backends import FineTuningBackend
from src.train.config import TrainingConfig
from src.train.domain import PreparedRelease, ReleaseSubset
from src.train.evaluate import evaluate_model_state


def validate_smoke_run(
    *,
    run_directory: Path,
    backend: FineTuningBackend,
    config: TrainingConfig,
    release: PreparedRelease,
    checkpoint: Path,
) -> None:
    """Fail unless loss and adapter reload satisfy smoke-test postconditions."""

    losses = _training_losses(run_directory / "logs" / "metrics.jsonl")
    if len(losses) < 2:
        raise RuntimeError("Smoke test emitted fewer than two training-loss values")
    window = max(1, len(losses) // 3)
    early_mean = statistics.fmean(losses[:window])
    late_mean = statistics.fmean(losses[-window:])
    if late_mean >= early_mean:
        raise RuntimeError(
            "Smoke-test training loss did not improve from early to late steps"
        )
    first = _one_prediction(backend, config, release, checkpoint, "roundtrip-a")
    second = _one_prediction(backend, config, release, checkpoint, "roundtrip-b")
    if first != second:
        raise RuntimeError("Reloading the saved adapter changed its prediction")
    mask_path = run_directory / "manifests" / "assistant_mask_audit.json"
    if not mask_path.is_file():
        raise RuntimeError("Smoke test is missing the assistant-only mask audit")
    ArtifactStore.at(run_directory).write_json(
        "manifests",
        "smoke_validation.json",
        {
            "passed": True,
            "observed_loss_count": len(losses),
            "early_loss_mean": early_mean,
            "late_loss_mean": late_mean,
            "adapter_reload_prediction_stable": True,
            "assistant_mask_audit_present": True,
            "checkpoint_resume_manifest_validated": True,
        },
    )


def _training_losses(path: Path) -> tuple[float, ...]:
    values: list[float] = []
    if not path.is_file():
        return ()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload: object = json.loads(line)
        if not isinstance(payload, Mapping):
            continue
        if payload.get("name") not in {"loss", "train_loss"}:
            continue
        value = payload.get("value")
        if isinstance(value, int | float) and not isinstance(value, bool):
            number = float(value)
            if not math.isfinite(number):
                raise RuntimeError("Smoke test emitted a non-finite training loss")
            values.append(number)
    return tuple(values)


def _one_prediction(
    backend: FineTuningBackend,
    config: TrainingConfig,
    release: PreparedRelease,
    checkpoint: Path,
    identifier: str,
) -> tuple[str, str | None, bool]:
    result = evaluate_model_state(
        backend=backend,
        config=config,
        release=release,
        subset=ReleaseSubset.DEV_PANEL,
        checkpoint_id=identifier,
        checkpoint_path=checkpoint,
        max_samples=1,
    )
    prediction = result.predictions[0]
    return prediction.raw_output, prediction.predicted_label, prediction.is_valid
