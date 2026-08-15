"""Generative SKINCON checkpoint evaluation for E2."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.train.artifacts import ArtifactStore
from src.train.backends import (
    FineTuningBackend,
    GenerationSpec,
    LoadedCheckpoint,
    PredictionSample,
)
from src.train.config import TrainingConfig
from src.train.data.taxonomy import load_taxonomy
from src.train.domain import ReleaseSubset
from src.train.e2.dataset import E2HumanDataset, build_e2_task_dataset
from src.train.e2.domain import E2ReleaseAudit, E2TaskName
from src.train.e2.metrics import (
    MorphologyMetrics,
    MorphologyPredictionInput,
    MorphologyPredictionRecord,
    canonicalize_morphology_predictions,
    evaluate_morphology_predictions,
)
from src.train.e2.phase import E2HumanPhase
from src.train.evaluate import checkpoint_training_state, model_spec


@dataclass(frozen=True, slots=True)
class MorphologyEvaluationResult:
    """Predictions and multilabel metrics for one model state."""

    checkpoint_id: str
    epoch: float | None
    eval_loss: float | None
    predictions: tuple[MorphologyPredictionRecord, ...]
    metrics: MorphologyMetrics


def evaluate_morphology_development(
    *,
    backend: FineTuningBackend,
    config: TrainingConfig,
    audit: E2ReleaseAudit | None,
    checkpoints: tuple[Path, ...],
    max_samples: int | None,
    store: ArtifactStore,
) -> tuple[MorphologyEvaluationResult, ...]:
    """Evaluate and persist base plus every checkpoint for an E2 run."""

    if audit is None:
        return ()
    results: list[MorphologyEvaluationResult] = []
    for path in (None, *checkpoints):
        checkpoint_id = "base" if path is None else path.name
        result = evaluate_morphology_state(
            backend=backend,
            config=config,
            audit=audit,
            checkpoint_id=checkpoint_id,
            checkpoint_path=path,
            cache_directory=store.layout.section("logs") / "hf_cache",
            max_samples=max_samples,
        )
        persist_morphology_evaluation(store, result)
        results.append(result)
    return tuple(results)


def evaluate_morphology_state(
    *,
    backend: FineTuningBackend,
    config: TrainingConfig,
    audit: E2ReleaseAudit,
    checkpoint_id: str,
    checkpoint_path: Path | None,
    cache_directory: Path,
    batch_size: int = 8,
    max_samples: int | None = None,
) -> MorphologyEvaluationResult:
    """Evaluate strict SKINCON JSON generation on the human E2 dev split."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    dataset = build_e2_task_dataset(
        config,
        audit,
        ReleaseSubset.SFT_DEV,
        E2TaskName.MORPHOLOGY,
        E2HumanPhase(
            taxonomy=load_taxonomy(config),
            ontology=audit.ontology,
        ),
        cache_directory,
    )
    spec = model_spec(config)
    loaded = (
        backend.load_base(spec)
        if checkpoint_path is None
        else backend.load_checkpoint(model=spec, checkpoint_path=checkpoint_path)
    )
    inputs: list[MorphologyPredictionInput] = []
    try:
        for start in range(0, _limit(len(dataset), max_samples), batch_size):
            stop = min(start + batch_size, _limit(len(dataset), max_samples))
            inputs.extend(
                _predict_batch(
                    backend,
                    loaded,
                    dataset,
                    range(start, stop),
                    checkpoint_id=checkpoint_id,
                    seed=config.trainer.seed,
                )
            )
    finally:
        backend.release(loaded)
    records = canonicalize_morphology_predictions(tuple(inputs), audit.ontology)
    epoch, eval_loss = checkpoint_training_state(checkpoint_path)
    return MorphologyEvaluationResult(
        checkpoint_id=checkpoint_id,
        epoch=epoch,
        eval_loss=eval_loss,
        predictions=records,
        metrics=evaluate_morphology_predictions(records, audit.ontology),
    )


def persist_morphology_evaluation(
    store: ArtifactStore,
    result: MorphologyEvaluationResult,
) -> None:
    """Write metrics and prediction provenance without clinical images."""

    stem = f"morphology_sft_dev__{_safe_id(result.checkpoint_id)}"
    store.write_json(
        "metrics",
        f"{stem}.json",
        {
            "checkpoint_id": result.checkpoint_id,
            "subset": "sft_dev",
            "task": "morphology",
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
    store.write_text(
        "predictions",
        f"{stem}.csv",
        _prediction_csv(result.predictions),
    )
    store.write_text(
        "tables",
        f"{stem}__per_concept.csv",
        _per_concept_csv(result.metrics),
    )


def _predict_batch(
    backend: FineTuningBackend,
    loaded: LoadedCheckpoint,
    dataset: E2HumanDataset,
    indices: range,
    *,
    checkpoint_id: str,
    seed: int,
) -> list[MorphologyPredictionInput]:
    samples = tuple(dataset.sample(index) for index in indices)
    requests = tuple(
        PredictionSample(sample.sample_id, sample.image, sample.prompt)
        for sample in samples
    )
    outputs = backend.predict(
        loaded,
        requests,
        generation=GenerationSpec(max_new_tokens=256),
    )
    if len(outputs) != len(samples):
        raise RuntimeError("Backend returned a different morphology count")
    result: list[MorphologyPredictionInput] = []
    for sample, output in zip(samples, outputs, strict=True):
        if output.sample_id != sample.sample_id:
            raise RuntimeError("Backend changed morphology sample ordering")
        if sample.morphology is None:
            raise RuntimeError("Morphology dataset returned no human target")
        result.append(
            MorphologyPredictionInput(
                sample_id=sample.sample_id,
                leakage_group_id=sample.leakage_group_id,
                true_concepts=sample.morphology.positive_concepts,
                raw_output=output.text,
                checkpoint_id=checkpoint_id,
                seed=seed,
            )
        )
    return result


def _prediction_csv(records: tuple[MorphologyPredictionRecord, ...]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        (
            "sample_id",
            "leakage_group_id",
            "true_concepts_json",
            "predicted_concepts_json",
            "is_valid",
            "raw_output",
            "checkpoint_id",
            "seed",
        )
    )
    for item in records:
        writer.writerow(
            (
                item.sample_id,
                item.leakage_group_id,
                json.dumps(item.true_concepts, ensure_ascii=False),
                json.dumps(item.predicted_concepts, ensure_ascii=False),
                item.is_valid,
                item.raw_output,
                item.checkpoint_id,
                item.seed,
            )
        )
    return buffer.getvalue()


def _per_concept_csv(metrics: MorphologyMetrics) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=(
            "concept",
            "support",
            "true_positive",
            "false_positive",
            "false_negative",
            "precision",
            "recall",
            "f1",
        ),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(asdict(item) for item in metrics.per_concept)
    return buffer.getvalue()


def _limit(length: int, max_samples: int | None) -> int:
    if max_samples is None:
        return length
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    return min(length, max_samples)


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
