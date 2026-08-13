"""Paired run comparison and deterministic group bootstrap inference."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict

from .metrics import evaluate_predictions
from .models import (
    ComparableRun,
    ConfidenceInterval,
    MetricAggregate,
    PairedComparison,
    PredictionRecord,
    SeedAggregate,
)


def _aligned_predictions(
    baseline: ComparableRun,
    candidate: ComparableRun,
) -> tuple[tuple[PredictionRecord, PredictionRecord], ...]:
    if baseline.contract != candidate.contract:
        raise ValueError("Runs do not share the same scientific contract")
    baseline_by_id = {item.sample_id: item for item in baseline.predictions}
    candidate_by_id = {item.sample_id: item for item in candidate.predictions}
    if len(baseline_by_id) != len(baseline.predictions):
        raise ValueError("Baseline contains duplicate sample identifiers")
    if len(candidate_by_id) != len(candidate.predictions):
        raise ValueError("Candidate contains duplicate sample identifiers")
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("Paired runs must contain exactly the same samples")
    pairs: list[tuple[PredictionRecord, PredictionRecord]] = []
    for sample_id in sorted(baseline_by_id):
        left = baseline_by_id[sample_id]
        right = candidate_by_id[sample_id]
        if (
            left.true_label != right.true_label
            or left.leakage_group_id != right.leakage_group_id
        ):
            raise ValueError(f"Paired sample metadata differs for {sample_id!r}")
        pairs.append((left, right))
    return tuple(pairs)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    smaller = min(left_only, right_only)
    lower_tail = sum(math.comb(discordant, index) for index in range(smaller + 1)) / (
        2**discordant
    )
    return float(min(1.0, 2.0 * lower_tail))


def compare_paired_runs(
    baseline: ComparableRun,
    candidate: ComparableRun,
    *,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 3407,
) -> PairedComparison:
    """Compare runs using paired deltas, group bootstrap, and McNemar."""

    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be positive")
    pairs = _aligned_predictions(baseline, candidate)
    if not pairs:
        raise ValueError("Paired comparison requires predictions")
    labels = baseline.metrics.labels
    if labels != candidate.metrics.labels:
        raise ValueError("Runs use different canonical label orders")
    recomputed_baseline = evaluate_predictions(baseline.predictions, labels)
    recomputed_candidate = evaluate_predictions(candidate.predictions, labels)
    if recomputed_baseline != baseline.metrics:
        raise ValueError("Baseline metrics do not match stored predictions")
    if recomputed_candidate != candidate.metrics:
        raise ValueError("Candidate metrics do not match stored predictions")

    groups: dict[str, list[tuple[PredictionRecord, PredictionRecord]]] = defaultdict(
        list
    )
    left_only = 0
    right_only = 0
    for left, right in pairs:
        groups[left.leakage_group_id].append((left, right))
        if left.is_correct and not right.is_correct:
            left_only += 1
        elif right.is_correct and not left.is_correct:
            right_only += 1

    rng = random.Random(bootstrap_seed)
    group_ids = tuple(sorted(groups))
    accuracy_deltas: list[float] = []
    macro_f1_deltas: list[float] = []
    for _ in range(bootstrap_iterations):
        sampled_left: list[PredictionRecord] = []
        sampled_right: list[PredictionRecord] = []
        for group_id in rng.choices(group_ids, k=len(group_ids)):
            for left, right in groups[group_id]:
                sampled_left.append(left)
                sampled_right.append(right)
        left_metrics = evaluate_predictions(tuple(sampled_left), labels)
        right_metrics = evaluate_predictions(tuple(sampled_right), labels)
        accuracy_deltas.append(right_metrics.top1_accuracy - left_metrics.top1_accuracy)
        macro_f1_deltas.append(right_metrics.macro_f1 - left_metrics.macro_f1)

    return PairedComparison(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        sample_count=len(pairs),
        group_count=len(groups),
        top1_delta=(candidate.metrics.top1_accuracy - baseline.metrics.top1_accuracy),
        macro_f1_delta=(candidate.metrics.macro_f1 - baseline.metrics.macro_f1),
        balanced_accuracy_delta=(
            candidate.metrics.balanced_accuracy - baseline.metrics.balanced_accuracy
        ),
        top1_delta_ci=ConfidenceInterval(
            low=_percentile(accuracy_deltas, 0.025),
            high=_percentile(accuracy_deltas, 0.975),
        ),
        macro_f1_delta_ci=ConfidenceInterval(
            low=_percentile(macro_f1_deltas, 0.025),
            high=_percentile(macro_f1_deltas, 0.975),
        ),
        baseline_only_correct=left_only,
        candidate_only_correct=right_only,
        mcnemar_exact_p=_exact_mcnemar_p(left_only, right_only),
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )


def _aggregate(values: tuple[float, ...]) -> MetricAggregate:
    return MetricAggregate(
        mean=statistics.fmean(values),
        standard_deviation=statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def _optional_aggregate(values: tuple[float | None, ...]) -> MetricAggregate | None:
    present = tuple(value for value in values if value is not None)
    if len(present) != len(values):
        return None
    return _aggregate(present)


def aggregate_seed_runs(
    runs: tuple[ComparableRun, ...],
) -> tuple[SeedAggregate, ...]:
    """Aggregate repeated seeds by experiment using sample standard deviation."""

    grouped: dict[str, list[ComparableRun]] = defaultdict(list)
    for run in runs:
        grouped[run.experiment_id].append(run)
    results: list[SeedAggregate] = []
    for experiment_id in sorted(grouped):
        group = sorted(grouped[experiment_id], key=lambda item: item.seed)
        seeds = tuple(run.seed for run in group)
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Experiment {experiment_id!r} has duplicate seeds")
        results.append(
            SeedAggregate(
                experiment_id=experiment_id,
                seeds=seeds,
                top1_accuracy=_aggregate(
                    tuple(run.metrics.top1_accuracy for run in group)
                ),
                macro_f1=_aggregate(tuple(run.metrics.macro_f1 for run in group)),
                balanced_accuracy=_aggregate(
                    tuple(run.metrics.balanced_accuracy for run in group)
                ),
                invalid_rate=_aggregate(
                    tuple(run.metrics.invalid_rate for run in group)
                ),
                duration_seconds=_optional_aggregate(
                    tuple(run.duration_seconds for run in group)
                ),
                gpu_hours=_optional_aggregate(tuple(run.gpu_hours for run in group)),
                peak_vram_gib=_optional_aggregate(
                    tuple(run.peak_vram_gib for run in group)
                ),
                trainable_parameters=_optional_aggregate(
                    tuple(
                        float(run.trainable_parameters)
                        if run.trainable_parameters is not None
                        else None
                        for run in group
                    )
                ),
            )
        )
    return tuple(results)
