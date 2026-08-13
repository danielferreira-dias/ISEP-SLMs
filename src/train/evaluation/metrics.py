"""Dependency-light classification metrics for label-only evaluation."""

from __future__ import annotations

from .models import ClassificationMetrics, PerClassMetrics, PredictionRecord


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_predictions(
    predictions: tuple[PredictionRecord, ...],
    labels: tuple[str, ...],
) -> ClassificationMetrics:
    """Compute closed-set metrics while counting invalid output as incorrect.

    Invalid responses remain in the accuracy denominator and invalid-rate
    calculation. They are not placed in a canonical confusion-matrix column,
    so a row sum may be smaller than its support; the missing count is fully
    represented by ``invalid_count``.
    """

    if not labels or len(labels) != len(set(labels)):
        raise ValueError("labels must be a non-empty unique tuple")
    label_index = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    support = [0 for _ in labels]
    predicted = [0 for _ in labels]
    correct = 0
    invalid = 0

    for record in predictions:
        if record.true_label not in label_index:
            raise ValueError(f"Unknown gold label: {record.true_label!r}")
        true_index = label_index[record.true_label]
        support[true_index] += 1
        if not record.is_valid or record.predicted_label is None:
            invalid += 1
            continue
        if record.predicted_label not in label_index:
            raise ValueError(
                f"Valid prediction is outside label contract: "
                f"{record.predicted_label!r}"
            )
        predicted_index = label_index[record.predicted_label]
        matrix[true_index][predicted_index] += 1
        predicted[predicted_index] += 1
        if true_index == predicted_index:
            correct += 1

    per_class: list[PerClassMetrics] = []
    for index, label in enumerate(labels):
        true_positives = matrix[index][index]
        precision = _safe_ratio(true_positives, predicted[index])
        recall = _safe_ratio(true_positives, support[index])
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class.append(
            PerClassMetrics(
                label=label,
                support=support[index],
                predicted_count=predicted[index],
                true_positives=true_positives,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

    supported = tuple(metric for metric in per_class if metric.support > 0)
    macro_f1 = (
        sum(metric.f1 for metric in supported) / len(supported) if supported else 0.0
    )
    balanced_accuracy = (
        sum(metric.recall for metric in supported) / len(supported)
        if supported
        else 0.0
    )
    total = len(predictions)
    return ClassificationMetrics(
        labels=labels,
        sample_count=total,
        valid_count=total - invalid,
        invalid_count=invalid,
        top1_accuracy=_safe_ratio(correct, total),
        macro_f1=macro_f1,
        balanced_accuracy=balanced_accuracy,
        invalid_rate=_safe_ratio(invalid, total),
        per_class=tuple(per_class),
        confusion_matrix=tuple(tuple(row) for row in matrix),
    )
