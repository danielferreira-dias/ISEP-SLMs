"""Human-only E2 diagnosis and SKINCON training support."""

from src.train.e2.caption_metrics import (
    CaptionMetrics,
    CaptionPredictionInput,
    CaptionPredictionRecord,
    canonicalize_caption_predictions,
    evaluate_caption_predictions,
)
from src.train.e2.dataset import (
    E2HumanDataset,
    build_e2_task_dataset,
    build_e2_training_dataset,
)
from src.train.e2.domain import (
    E2FormattedExample,
    E2HumanSample,
    E2ReleaseAudit,
    E2TaskName,
    MorphologyTarget,
    SkinConOntology,
)
from src.train.e2.metrics import (
    ConceptMetrics,
    MorphologyMetrics,
    MorphologyPredictionInput,
    MorphologyPredictionRecord,
    canonicalize_morphology_predictions,
    evaluate_morphology_predictions,
)
from src.train.e2.mixing import DeterministicTaskInterleave
from src.train.e2.phase import E2HumanPhase, caption_prompt, morphology_prompt
from src.train.e2.release import inspect_e2_release

__all__ = [
    "CaptionMetrics",
    "CaptionPredictionInput",
    "CaptionPredictionRecord",
    "ConceptMetrics",
    "DeterministicTaskInterleave",
    "E2FormattedExample",
    "E2HumanDataset",
    "E2HumanPhase",
    "E2HumanSample",
    "E2ReleaseAudit",
    "E2TaskName",
    "MorphologyMetrics",
    "MorphologyPredictionInput",
    "MorphologyPredictionRecord",
    "MorphologyTarget",
    "SkinConOntology",
    "build_e2_task_dataset",
    "build_e2_training_dataset",
    "canonicalize_caption_predictions",
    "canonicalize_morphology_predictions",
    "caption_prompt",
    "evaluate_caption_predictions",
    "evaluate_morphology_predictions",
    "inspect_e2_release",
    "morphology_prompt",
]
