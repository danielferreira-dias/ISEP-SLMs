"""Transparent, task-disaggregated E2 checkpoint summaries."""

from __future__ import annotations

import csv
import io
from dataclasses import asdict, dataclass

from src.train.artifacts import ArtifactStore
from src.train.artifacts.tables import write_latex_table
from src.train.artifacts.types import TableCell
from src.train.e2.caption_evaluation import CaptionEvaluationResult
from src.train.e2.evaluation import MorphologyEvaluationResult
from src.train.evaluate import EvaluationResult


@dataclass(frozen=True, slots=True)
class MultitaskCheckpointMetrics:
    """Comparable task scores plus an explicitly non-accuracy composite."""

    checkpoint_id: str
    diagnosis_top1_accuracy: float
    diagnosis_macro_f1: float
    diagnosis_balanced_accuracy: float
    morphology_micro_f1: float
    morphology_macro_f1: float
    morphology_exact_match: float
    caption_task_score: float | None
    caption_concept_f1: float | None
    caption_reference_similarity: float | None
    caption_clinical_compliance: float | None
    global_multitask_score: float
    global_score_task_count: int


def persist_multitask_metrics(
    *,
    store: ArtifactStore,
    diagnosis: tuple[EvaluationResult, ...],
    morphology: tuple[MorphologyEvaluationResult, ...],
    captions: tuple[CaptionEvaluationResult, ...],
) -> tuple[MultitaskCheckpointMetrics, ...]:
    """Join task metrics by model state and persist JSON plus thesis CSV."""

    morphology_by_id = {item.checkpoint_id: item for item in morphology}
    caption_by_id = {item.checkpoint_id: item for item in captions}
    results: list[MultitaskCheckpointMetrics] = []
    for diagnosis_result in diagnosis:
        morphology_result = morphology_by_id.get(diagnosis_result.checkpoint_id)
        if morphology_result is None:
            raise RuntimeError("Diagnosis checkpoint has no morphology evaluation")
        caption_result = caption_by_id.get(diagnosis_result.checkpoint_id)
        component_scores = [
            diagnosis_result.metrics.macro_f1,
            morphology_result.metrics.macro_f1,
        ]
        if caption_result is not None:
            component_scores.append(caption_result.metrics.caption_task_score)
        item = MultitaskCheckpointMetrics(
            checkpoint_id=diagnosis_result.checkpoint_id,
            diagnosis_top1_accuracy=diagnosis_result.metrics.top1_accuracy,
            diagnosis_macro_f1=diagnosis_result.metrics.macro_f1,
            diagnosis_balanced_accuracy=(diagnosis_result.metrics.balanced_accuracy),
            morphology_micro_f1=morphology_result.metrics.micro_f1,
            morphology_macro_f1=morphology_result.metrics.macro_f1,
            morphology_exact_match=morphology_result.metrics.exact_match,
            caption_task_score=(
                caption_result.metrics.caption_task_score
                if caption_result is not None
                else None
            ),
            caption_concept_f1=(
                caption_result.metrics.concept_f1
                if caption_result is not None
                else None
            ),
            caption_reference_similarity=(
                caption_result.metrics.reference_similarity_mean
                if caption_result is not None
                else None
            ),
            caption_clinical_compliance=(
                caption_result.metrics.clinical_compliance_rate
                if caption_result is not None
                else None
            ),
            global_multitask_score=sum(component_scores) / len(component_scores),
            global_score_task_count=len(component_scores),
        )
        store.write_json(
            "metrics",
            f"multitask_sft_dev__{item.checkpoint_id}.json",
            {
                **asdict(item),
                "global_metric_name": "macro_task_score_not_accuracy",
                "global_formula": (
                    "mean(diagnosis_macro_f1,morphology_macro_f1"
                    + (",caption_task_score)" if caption_result else ")")
                ),
                "comparable_only_when_task_set_matches": True,
            },
        )
        results.append(item)
    _write_csv(store, tuple(results))
    return tuple(results)


def _write_csv(
    store: ArtifactStore,
    rows: tuple[MultitaskCheckpointMetrics, ...],
) -> None:
    buffer = io.StringIO()
    names = tuple(MultitaskCheckpointMetrics.__dataclass_fields__)
    writer = csv.DictWriter(buffer, fieldnames=names)
    writer.writeheader()
    writer.writerows(asdict(item) for item in rows)
    store.write_text("tables", "e2_multitask_checkpoint_metrics.csv", buffer.getvalue())
    latex_rows: tuple[tuple[TableCell, ...], ...] = tuple(
        tuple(asdict(item)[name] for name in names) for item in rows
    )
    write_latex_table(
        store.path("tables", "e2_multitask_checkpoint_metrics.tex"),
        names,
        latex_rows,
        caption="Disaggregated E2 development metrics by checkpoint",
        label="tab:e2_multitask_checkpoint_metrics",
    )
