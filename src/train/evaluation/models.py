"""Typed records produced by label-only training evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PredictionInput:
    """Raw model output and immutable identifiers for one example."""

    sample_id: str
    leakage_group_id: str
    true_label: str
    raw_output: str
    checkpoint_id: str = "unknown"
    seed: int = 0


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """Canonicalized prediction used for metrics and persistence."""

    sample_id: str
    leakage_group_id: str
    true_label: str
    raw_output: str
    predicted_label: str | None
    is_valid: bool
    checkpoint_id: str = "unknown"
    seed: int = 0

    @property
    def is_correct(self) -> bool:
        """Return whether the valid prediction equals the gold label."""

        return self.is_valid and self.predicted_label == self.true_label


@dataclass(frozen=True, slots=True)
class PerClassMetrics:
    """One-vs-rest metrics for a canonical label."""

    label: str
    support: int
    predicted_count: int
    true_positives: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Complete deterministic metric result for a prediction collection."""

    labels: tuple[str, ...]
    sample_count: int
    valid_count: int
    invalid_count: int
    top1_accuracy: float
    macro_f1: float
    balanced_accuracy: float
    invalid_rate: float
    per_class: tuple[PerClassMetrics, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class CheckpointScore:
    """Metrics needed to rank one resumable training checkpoint."""

    checkpoint_id: str
    epoch: float
    eval_loss: float
    metrics: ClassificationMetrics


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """Two-sided percentile confidence interval."""

    low: float
    high: float
    confidence: float = 0.95


@dataclass(frozen=True, slots=True)
class RunContract:
    """Scientific inputs that must match before paired comparison."""

    dataset_revision: str
    split_hash: str
    prompt_hash: str
    model_revision: str
    label_contract_hash: str
    training_contract_hash: str


@dataclass(frozen=True, slots=True)
class ComparableRun:
    """One completed run with predictions and resource measurements."""

    experiment_id: str
    run_id: str
    seed: int
    contract: RunContract
    predictions: tuple[PredictionRecord, ...]
    metrics: ClassificationMetrics
    duration_seconds: float | None = None
    gpu_hours: float | None = None
    peak_vram_gib: float | None = None
    trainable_parameters: int | None = None


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """Paired quality comparison between a baseline and candidate run."""

    baseline_run_id: str
    candidate_run_id: str
    sample_count: int
    group_count: int
    top1_delta: float
    macro_f1_delta: float
    balanced_accuracy_delta: float
    top1_delta_ci: ConfidenceInterval
    macro_f1_delta_ci: ConfidenceInterval
    baseline_only_correct: int
    candidate_only_correct: int
    mcnemar_exact_p: float
    bootstrap_iterations: int
    bootstrap_seed: int


@dataclass(frozen=True, slots=True)
class MetricAggregate:
    """Mean and sample standard deviation across independent seeds."""

    mean: float
    standard_deviation: float


@dataclass(frozen=True, slots=True)
class SeedAggregate:
    """Aggregate quality and resource results for one experiment."""

    experiment_id: str
    seeds: tuple[int, ...]
    top1_accuracy: MetricAggregate
    macro_f1: MetricAggregate
    balanced_accuracy: MetricAggregate
    invalid_rate: MetricAggregate
    duration_seconds: MetricAggregate | None
    gpu_hours: MetricAggregate | None
    peak_vram_gib: MetricAggregate | None
    trainable_parameters: MetricAggregate | None
