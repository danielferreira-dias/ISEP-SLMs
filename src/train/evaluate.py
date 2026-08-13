"""Bounded-memory deterministic evaluation of base and LoRA checkpoints."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src.train.backends import (
    FineTuningBackend,
    GenerationSpec,
    ModelLoadSpec,
    PredictionSample,
)
from src.train.config import TrainingConfig
from src.train.data import iter_release_samples, load_taxonomy
from src.train.domain import PreparedRelease, ReleaseSubset
from src.train.evaluation import (
    ClassificationMetrics,
    LabelVocabulary,
    PredictionInput,
    PredictionRecord,
    canonicalize_predictions,
    evaluate_predictions,
)
from src.train.phases.label_only import LabelOnlyPhase


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Predictions and metrics for one model state and release subset."""

    checkpoint_id: str
    subset: ReleaseSubset
    epoch: float | None
    eval_loss: float | None
    predictions: tuple[PredictionRecord, ...]
    metrics: ClassificationMetrics


def model_spec(config: TrainingConfig) -> ModelLoadSpec:
    """Translate the validated external model config to a backend contract."""

    return ModelLoadSpec(
        model_id=config.model.repo_id,
        revision=config.model.revision,
        processor_id=config.model.processor_repo_id,
        processor_revision=config.model.processor_revision,
        dtype=config.model.dtype,
        load_in_4bit=config.model.load_in_4bit,
    )


def evaluate_model_state(
    *,
    backend: FineTuningBackend,
    config: TrainingConfig,
    release: PreparedRelease,
    subset: ReleaseSubset,
    checkpoint_id: str,
    checkpoint_path: Path | None,
    batch_size: int = 8,
    seed: int | None = None,
    max_samples: int | None = None,
) -> EvaluationResult:
    """Evaluate a base model or adapter without retaining all images in RAM.

    Args:
        backend: Fine-tuning backend that also owns deterministic inference.
        config: Frozen run configuration.
        release: Audited train/development assignments.
        subset: Development panel or full development view.
        checkpoint_id: Stable identifier written to predictions.
        checkpoint_path: Adapter directory, or ``None`` for the base model.
        batch_size: Number of decoded images retained at once.
        seed: Prediction provenance seed; defaults to the trainer seed.
        max_samples: Optional deterministic prefix used only by smoke tests.

    Returns:
        Canonical records and deterministic closed-set metrics.

    Raises:
        RuntimeError: If a backend changes sample identifiers or cardinality.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")
    taxonomy = load_taxonomy(config)
    phase = LabelOnlyPhase(taxonomy)
    vocabulary = LabelVocabulary(labels=taxonomy.labels)
    spec = model_spec(config)
    loaded = (
        backend.load_base(spec)
        if checkpoint_path is None
        else backend.load_checkpoint(model=spec, checkpoint_path=checkpoint_path)
    )
    raw_inputs: list[PredictionInput] = []
    batch: list[object] = []
    provenance_seed = config.trainer.seed if seed is None else seed
    try:
        for sample in iter_release_samples(config, release, subset):
            if max_samples is not None and len(raw_inputs) + len(batch) >= max_samples:
                break
            batch.append(sample)
            if len(batch) == batch_size:
                raw_inputs.extend(
                    _predict_batch(
                        backend=backend,
                        loaded=loaded,
                        samples=batch,
                        prompt=phase.prompt,
                        checkpoint_id=checkpoint_id,
                        seed=provenance_seed,
                    )
                )
                batch.clear()
        if batch:
            raw_inputs.extend(
                _predict_batch(
                    backend=backend,
                    loaded=loaded,
                    samples=batch,
                    prompt=phase.prompt,
                    checkpoint_id=checkpoint_id,
                    seed=provenance_seed,
                )
            )
    finally:
        backend.release(loaded)

    predictions = canonicalize_predictions(tuple(raw_inputs), vocabulary)
    epoch, eval_loss = checkpoint_training_state(checkpoint_path)
    return EvaluationResult(
        checkpoint_id=checkpoint_id,
        subset=subset,
        epoch=epoch,
        eval_loss=eval_loss,
        predictions=predictions,
        metrics=evaluate_predictions(predictions, taxonomy.labels),
    )


def checkpoint_training_state(
    checkpoint_path: Path | None,
) -> tuple[float | None, float | None]:
    """Read checkpoint epoch and most recent evaluation loss from Trainer."""

    if checkpoint_path is None:
        return 0.0, None
    state_path = checkpoint_path / "trainer_state.json"
    try:
        document: object = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read Trainer state {state_path}: {exc}") from exc
    state = _mapping(document, "trainer state")
    epoch = _optional_number(state.get("epoch"), "trainer epoch")
    global_step = _optional_integer(state.get("global_step"), "global step")
    history_value = state.get("log_history")
    if not isinstance(history_value, list):
        raise ValueError("Trainer state log_history must be a list")
    candidates: list[tuple[int, float]] = []
    for raw_event in history_value:
        if not isinstance(raw_event, Mapping):
            continue
        raw_loss = raw_event.get("eval_loss")
        raw_step = raw_event.get("step")
        if (
            isinstance(raw_loss, int | float)
            and not isinstance(raw_loss, bool)
            and isinstance(raw_step, int)
            and not isinstance(raw_step, bool)
            and (global_step is None or raw_step <= global_step)
        ):
            candidates.append((raw_step, float(raw_loss)))
    eval_loss = max(candidates, key=lambda item: item[0])[1] if candidates else None
    return epoch, eval_loss


def _predict_batch(
    *,
    backend: FineTuningBackend,
    loaded: object,
    samples: list[object],
    prompt: str,
    checkpoint_id: str,
    seed: int,
) -> list[PredictionInput]:
    # Imports stay local to keep the public signature independent of Pillow.
    from src.train.backends import LoadedCheckpoint
    from src.train.domain import LabeledImageSample

    if not isinstance(loaded, LoadedCheckpoint):
        raise TypeError("Backend returned an invalid loaded-checkpoint object")
    typed_samples: list[LabeledImageSample] = []
    for sample in samples:
        if not isinstance(sample, LabeledImageSample):
            raise TypeError("Release iterator returned an invalid sample")
        typed_samples.append(sample)
    requests = tuple(
        PredictionSample(
            sample_id=sample.sample_id,
            image=sample.image,
            prompt=prompt,
        )
        for sample in typed_samples
    )
    outputs = backend.predict(
        loaded,
        requests,
        generation=GenerationSpec(max_new_tokens=32),
    )
    if len(outputs) != len(typed_samples):
        raise RuntimeError("Backend returned a different prediction count")
    records: list[PredictionInput] = []
    for sample, output in zip(typed_samples, outputs, strict=True):
        if output.sample_id != sample.sample_id:
            raise RuntimeError("Backend changed prediction sample ordering")
        records.append(
            PredictionInput(
                sample_id=sample.sample_id,
                leakage_group_id=sample.leakage_group_id,
                true_label=sample.label,
                raw_output=output.text,
                checkpoint_id=checkpoint_id,
                seed=seed,
            )
        )
    return records


def _mapping(value: object, context: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _optional_number(value: object, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{context} must be numeric or null")
    return float(value)


def _optional_integer(value: object, context: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer or null")
    return value
