"""Generative SkinCAP evaluation for base and saved E2 checkpoints."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.train.artifacts import ArtifactStore
from src.train.backends import FineTuningBackend, GenerationSpec, PredictionSample
from src.train.config import TrainingConfig
from src.train.data.taxonomy import load_taxonomy
from src.train.domain import ReleaseSubset
from src.train.e2.caption_metrics import (
    CaptionMetrics,
    CaptionPredictionInput,
    CaptionPredictionRecord,
    canonicalize_caption_predictions,
    evaluate_caption_predictions,
)
from src.train.e2.dataset import build_e2_task_dataset
from src.train.e2.domain import E2ReleaseAudit, E2TaskName
from src.train.e2.phase import E2HumanPhase
from src.train.evaluate import checkpoint_training_state, model_spec


@dataclass(frozen=True, slots=True)
class CaptionEvaluationResult:
    """Predictions and judge-free SkinCAP metrics for one model state."""

    checkpoint_id: str
    epoch: float | None
    eval_loss: float | None
    predictions: tuple[CaptionPredictionRecord, ...]
    metrics: CaptionMetrics


def evaluate_caption_development(
    *,
    backend: FineTuningBackend,
    config: TrainingConfig,
    audit: E2ReleaseAudit | None,
    checkpoints: tuple[Path, ...],
    max_samples: int | None,
    store: ArtifactStore,
) -> tuple[CaptionEvaluationResult, ...]:
    """Evaluate and persist base plus every checkpoint on SkinCAP dev."""

    if audit is None or audit.caption_dev == 0:
        return ()
    results: list[CaptionEvaluationResult] = []
    for path in (None, *checkpoints):
        checkpoint_id = "base" if path is None else path.name
        result = evaluate_caption_state(
            backend=backend,
            config=config,
            audit=audit,
            checkpoint_id=checkpoint_id,
            checkpoint_path=path,
            cache_directory=store.layout.section("logs") / "hf_cache",
            max_samples=max_samples,
        )
        persist_caption_evaluation(store, result)
        results.append(result)
    return tuple(results)


def evaluate_caption_state(
    *,
    backend: FineTuningBackend,
    config: TrainingConfig,
    audit: E2ReleaseAudit,
    checkpoint_id: str,
    checkpoint_path: Path | None,
    cache_directory: Path,
    batch_size: int = 8,
    max_samples: int | None = None,
) -> CaptionEvaluationResult:
    """Evaluate free visual observations with linked human SKINCON concepts."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    taxonomy = load_taxonomy(config)
    phase = E2HumanPhase(taxonomy=taxonomy, ontology=audit.ontology)
    captions = build_e2_task_dataset(
        config,
        audit,
        ReleaseSubset.SFT_DEV,
        E2TaskName.CAPTION,
        phase,
        cache_directory,
    )
    morphology = build_e2_task_dataset(
        config,
        audit,
        ReleaseSubset.SFT_DEV,
        E2TaskName.MORPHOLOGY,
        phase,
        cache_directory,
    )
    concepts_by_image = {
        sample.image_sha256: sample.morphology.positive_concepts
        for sample in (morphology.sample(index) for index in range(len(morphology)))
        if sample.morphology is not None
    }
    spec = model_spec(config)
    loaded = (
        backend.load_base(spec)
        if checkpoint_path is None
        else backend.load_checkpoint(model=spec, checkpoint_path=checkpoint_path)
    )
    inputs: list[CaptionPredictionInput] = []
    limit = (
        min(len(captions), max_samples) if max_samples is not None else len(captions)
    )
    try:
        for start in range(0, limit, batch_size):
            samples = tuple(
                captions.sample(index)
                for index in range(start, min(start + batch_size, limit))
            )
            outputs = backend.predict(
                loaded,
                tuple(
                    PredictionSample(sample.sample_id, sample.image, sample.prompt)
                    for sample in samples
                ),
                generation=GenerationSpec(max_new_tokens=160),
            )
            if len(outputs) != len(samples):
                raise RuntimeError("Backend returned a different caption count")
            for sample, output in zip(samples, outputs, strict=True):
                if output.sample_id != sample.sample_id:
                    raise RuntimeError("Backend changed caption sample ordering")
                concepts = concepts_by_image.get(sample.image_sha256)
                if concepts is None:
                    raise RuntimeError("SkinCAP dev row has no linked SKINCON target")
                inputs.append(
                    CaptionPredictionInput(
                        sample_id=sample.sample_id,
                        leakage_group_id=sample.leakage_group_id,
                        reference_text=sample.target_text,
                        true_concepts=concepts,
                        raw_output=output.text,
                        checkpoint_id=checkpoint_id,
                        seed=config.trainer.seed,
                    )
                )
    finally:
        backend.release(loaded)
    records = canonicalize_caption_predictions(
        tuple(inputs),
        audit.ontology,
        taxonomy.labels,
    )
    epoch, eval_loss = checkpoint_training_state(checkpoint_path)
    return CaptionEvaluationResult(
        checkpoint_id=checkpoint_id,
        epoch=epoch,
        eval_loss=eval_loss,
        predictions=records,
        metrics=evaluate_caption_predictions(records),
    )


def persist_caption_evaluation(
    store: ArtifactStore,
    result: CaptionEvaluationResult,
) -> None:
    """Write SkinCAP metrics and per-sample evidence without clinical images."""

    stem = f"caption_sft_dev__{_safe_id(result.checkpoint_id)}"
    store.write_json(
        "metrics",
        f"{stem}.json",
        {
            "checkpoint_id": result.checkpoint_id,
            "subset": "sft_dev",
            "task": "caption",
            "epoch": result.epoch,
            "eval_loss": result.eval_loss,
            **asdict(result.metrics),
        },
    )
    store.write_text(
        "predictions",
        f"{stem}.jsonl",
        "".join(
            json.dumps(asdict(item), ensure_ascii=False, allow_nan=False) + "\n"
            for item in result.predictions
        ),
    )
    store.write_text("predictions", f"{stem}.csv", _prediction_csv(result.predictions))


def _prediction_csv(records: tuple[CaptionPredictionRecord, ...]) -> str:
    buffer = io.StringIO()
    names = tuple(CaptionPredictionRecord.__dataclass_fields__)
    writer = csv.DictWriter(buffer, fieldnames=names)
    writer.writeheader()
    writer.writerows(
        {
            **asdict(item),
            "true_concepts": "|".join(item.true_concepts),
            "predicted_concepts": "|".join(item.predicted_concepts),
        }
        for item in records
    )
    return buffer.getvalue()


def _safe_id(value: str) -> str:
    if not value or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for char in value
    ):
        raise ValueError(f"Unsafe checkpoint identifier: {value!r}")
    return value
