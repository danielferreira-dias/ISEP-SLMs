"""Explicit JSON conversion for typed evaluation records."""

from __future__ import annotations

from collections.abc import Mapping

from src.train.evaluation.models import (
    ClassificationMetrics,
    PerClassMetrics,
    RunContract,
)

from .types import JsonValue


def classification_metrics_to_json(
    metrics: ClassificationMetrics,
) -> dict[str, JsonValue]:
    """Convert classification metrics to stable JSON-compatible fields."""

    return {
        "labels": list(metrics.labels),
        "sample_count": metrics.sample_count,
        "valid_count": metrics.valid_count,
        "invalid_count": metrics.invalid_count,
        "top1_accuracy": metrics.top1_accuracy,
        "macro_f1": metrics.macro_f1,
        "balanced_accuracy": metrics.balanced_accuracy,
        "invalid_rate": metrics.invalid_rate,
        "per_class": [
            {
                "label": item.label,
                "support": item.support,
                "predicted_count": item.predicted_count,
                "true_positives": item.true_positives,
                "precision": item.precision,
                "recall": item.recall,
                "f1": item.f1,
            }
            for item in metrics.per_class
        ],
        "confusion_matrix": [list(row) for row in metrics.confusion_matrix],
    }


def run_contract_to_json(contract: RunContract) -> dict[str, JsonValue]:
    """Convert a scientific comparison contract to JSON fields."""

    return {
        "dataset_revision": contract.dataset_revision,
        "split_hash": contract.split_hash,
        "prompt_hash": contract.prompt_hash,
        "model_revision": contract.model_revision,
        "label_contract_hash": contract.label_contract_hash,
        "training_contract_hash": contract.training_contract_hash,
    }


def _mapping(value: object, context: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a JSON array")
    return value


def _str(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    return value


def _int(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context} must be an integer")
    return value


def _float(value: object, context: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{context} must be numeric")
    return float(value)


def classification_metrics_from_json(payload: object) -> ClassificationMetrics:
    """Validate and restore classification metrics from decoded JSON."""

    root = _mapping(payload, "classification metrics")
    labels = tuple(
        _str(value, "labels item") for value in _list(root.get("labels"), "labels")
    )
    per_class: list[PerClassMetrics] = []
    for value in _list(root.get("per_class"), "per_class"):
        item = _mapping(value, "per_class item")
        per_class.append(
            PerClassMetrics(
                label=_str(item.get("label"), "per_class.label"),
                support=_int(item.get("support"), "per_class.support"),
                predicted_count=_int(
                    item.get("predicted_count"),
                    "per_class.predicted_count",
                ),
                true_positives=_int(
                    item.get("true_positives"),
                    "per_class.true_positives",
                ),
                precision=_float(item.get("precision"), "per_class.precision"),
                recall=_float(item.get("recall"), "per_class.recall"),
                f1=_float(item.get("f1"), "per_class.f1"),
            )
        )
    matrix = tuple(
        tuple(
            _int(cell, "confusion_matrix cell")
            for cell in _list(row, "confusion_matrix row")
        )
        for row in _list(root.get("confusion_matrix"), "confusion_matrix")
    )
    return ClassificationMetrics(
        labels=labels,
        sample_count=_int(root.get("sample_count"), "sample_count"),
        valid_count=_int(root.get("valid_count"), "valid_count"),
        invalid_count=_int(root.get("invalid_count"), "invalid_count"),
        top1_accuracy=_float(root.get("top1_accuracy"), "top1_accuracy"),
        macro_f1=_float(root.get("macro_f1"), "macro_f1"),
        balanced_accuracy=_float(root.get("balanced_accuracy"), "balanced_accuracy"),
        invalid_rate=_float(root.get("invalid_rate"), "invalid_rate"),
        per_class=tuple(per_class),
        confusion_matrix=matrix,
    )


def run_contract_from_json(payload: object) -> RunContract:
    """Validate and restore a scientific comparison contract."""

    root = _mapping(payload, "run contract")
    return RunContract(
        dataset_revision=_str(root.get("dataset_revision"), "dataset_revision"),
        split_hash=_str(root.get("split_hash"), "split_hash"),
        prompt_hash=_str(root.get("prompt_hash"), "prompt_hash"),
        model_revision=_str(root.get("model_revision"), "model_revision"),
        label_contract_hash=_str(
            root.get("label_contract_hash"), "label_contract_hash"
        ),
        training_contract_hash=_str(
            root.get("training_contract_hash"), "training_contract_hash"
        ),
    )
