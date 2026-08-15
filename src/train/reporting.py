"""Rebuild thesis figures and tables from durable run artefacts."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Protocol, runtime_checkable

from src.train.artifacts import (
    ArtifactStore,
    CheckpointMetricPoint,
    DistributionPoint,
    ResourcePoint,
    ThesisPlotter,
    TrainableParameterPoint,
    TrainingHistoryPoint,
    generate_report,
)
from src.train.artifacts.serialization import classification_metrics_from_json
from src.train.artifacts.tables import write_csv_table, write_latex_table
from src.train.artifacts.training_history import (
    TrainingMetricEvent,
    materialize_training_history,
)
from src.train.artifacts.types import TableCell
from src.train.data import load_assignments
from src.train.e2.caption_plots import render_caption_and_multitask_plots
from src.train.e2.plots import render_morphology_plots
from src.train.resource_metrics import resource_summary


def build_run_report(run_directory: Path) -> None:
    """Regenerate all available plots, tables, Markdown, and HTML.

    Args:
        run_directory: Completed or partially completed canonical run tree.
    """

    store = ArtifactStore.at(run_directory)
    plotter = ThesisPlotter(store.layout.figures)
    metric_log = store.path("logs", "metrics.jsonl")
    events = (
        materialize_training_history(
            metric_log,
            store.path("metrics", "training_history.csv"),
            store.path("metrics", "training_history.parquet"),
        )
        if metric_log.is_file()
        else ()
    )
    history = _training_history(events)
    plotter.training_history(history)
    checkpoints = _checkpoint_points(run_directory)
    plotter.checkpoint_metrics(checkpoints)
    render_morphology_plots(store)
    render_caption_and_multitask_plots(store)

    final_path = store.path("metrics", "classification.json")
    if final_path.is_file():
        final_metrics = classification_metrics_from_json(_read_json(final_path))
        plotter.per_class_metrics(final_metrics)
        plotter.confusion_matrix(final_metrics)

    assignments_path = _release_assignments_path(run_directory)
    if assignments_path is not None:
        assignments = load_assignments(assignments_path.parent)
        plotter.class_distribution(_distribution_points(assignments, "label"))
        plotter.source_distribution(_distribution_points(assignments, "source"))

    parameters = _trainable_parameter_points(run_directory)
    plotter.trainable_parameters(parameters)
    resources = _resource_points(run_directory)
    plotter.resource_usage(resources)
    _write_checkpoint_tables(store, checkpoints)
    _write_resource_tables(store, run_directory)

    if final_path.is_file():
        generate_report(
            run_directory,
            title=f"ISEP training run {run_directory.name}",
            limitations=(
                "Checkpoint selection uses only frozen sft_dev data.",
                "ISEPDermaBench, DermoBench, DDI, and SkinDisNet are excluded "
                "from model selection.",
                "A real NVIDIA BF16 smoke test remains required for any run "
                "created on a CPU-only development host.",
            ),
        )


def _training_history(
    events: tuple[TrainingMetricEvent, ...],
) -> tuple[TrainingHistoryPoint, ...]:
    grouped: dict[tuple[int, float], dict[str, float]] = defaultdict(dict)
    for event in events:
        epoch = event.epoch or 0.0
        grouped[(event.step, epoch)][event.name] = event.value
    return tuple(
        TrainingHistoryPoint(
            step=step,
            epoch=epoch,
            train_loss=values.get("loss", values.get("train_loss")),
            eval_loss=values.get("eval_loss"),
            learning_rate=values.get("learning_rate"),
        )
        for (step, epoch), values in sorted(grouped.items())
        if any(
            key in values
            for key in ("loss", "train_loss", "eval_loss", "learning_rate")
        )
    )


def _checkpoint_points(
    run_directory: Path,
) -> tuple[CheckpointMetricPoint, ...]:
    points: list[CheckpointMetricPoint] = []
    metrics_directory = run_directory / "metrics"
    for path in sorted(metrics_directory.glob("sft_dev__checkpoint-*.json")):
        payload = _mapping(_read_json(path), "checkpoint metrics")
        epoch = _optional_float(payload.get("epoch"))
        eval_loss = _optional_float(payload.get("eval_loss"))
        checkpoint_id = payload.get("checkpoint_id")
        if epoch is None or eval_loss is None or not isinstance(checkpoint_id, str):
            continue
        metrics = classification_metrics_from_json(payload)
        points.append(
            CheckpointMetricPoint(
                checkpoint_id=checkpoint_id,
                epoch=epoch,
                top1_accuracy=metrics.top1_accuracy,
                macro_f1=metrics.macro_f1,
                balanced_accuracy=metrics.balanced_accuracy,
                eval_loss=eval_loss,
            )
        )
    return tuple(sorted(points, key=lambda item: item.epoch))


def _release_assignments_path(run_directory: Path) -> Path | None:
    path = run_directory / "manifests" / "dataset_release.json"
    if not path.is_file():
        return None
    context_path = run_directory / "manifests" / "execution_context.json"
    config_path = run_directory / "manifests" / "config.resolved.json"
    if not context_path.is_file() or not config_path.is_file():
        return None
    context = _mapping(_read_json(context_path), "execution context")
    config = _mapping(_read_json(config_path), "resolved config")
    dataset = config.get("dataset")
    project_root = context.get("project_root")
    if not isinstance(dataset, Mapping) or not isinstance(project_root, str):
        return None
    release_directory = dataset.get("release_directory")
    if not isinstance(release_directory, str):
        return None
    root = Path(release_directory)
    if not root.is_absolute():
        root = Path(project_root) / root
    candidate = root / "assignments.parquet"
    return candidate if candidate.is_file() else None


def _distribution_points(frame: object, column: str) -> tuple[DistributionPoint, ...]:
    records = _assignment_records(frame)
    split_counts: dict[tuple[str, str], int] = defaultdict(int)
    panel_counts: dict[str, int] = defaultdict(int)
    for record in records:
        split = record.get("split")
        category = record.get(column)
        is_dev_panel = record.get("is_dev_panel")
        if not isinstance(split, str) or not isinstance(category, str):
            raise ValueError(f"Assignment has invalid split/{column} fields")
        if not isinstance(is_dev_panel, bool):
            raise ValueError("Assignment has invalid is_dev_panel field")
        split_counts[(split, category)] += 1
        if is_dev_panel:
            panel_counts[category] += 1
    points = [
        DistributionPoint(split, category, count)
        for (split, category), count in sorted(split_counts.items())
    ]
    points.extend(
        DistributionPoint("dev_panel", category, count)
        for category, count in sorted(panel_counts.items())
    )
    return tuple(points)


@runtime_checkable
class _RecordsFrame(Protocol):
    """Minimal dataframe boundary needed by reporting."""

    def to_dict(self, orient: str = "dict") -> object:
        """Return a representation of the dataframe."""


def _assignment_records(frame: object) -> tuple[Mapping[object, object], ...]:
    """Convert an assignment frame without importing pandas in reporting."""

    if not isinstance(frame, _RecordsFrame):
        raise TypeError("Assignment view must support record conversion")
    value = frame.to_dict(orient="records")
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError("Assignment record conversion did not return a sequence")
    records: list[Mapping[object, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"Assignment record {index} is not an object")
        records.append(item)
    return tuple(records)


def _trainable_parameter_points(
    run_directory: Path,
) -> tuple[TrainableParameterPoint, ...]:
    path = run_directory / "manifests" / "backend_result.json"
    if not path.is_file():
        return ()
    root = _mapping(_read_json(path), "backend result")
    trainable = root.get("trainable_parameters")
    if not isinstance(trainable, Mapping):
        return ()
    components = trainable.get("by_component")
    if not isinstance(components, Mapping):
        return ()
    return tuple(
        TrainableParameterPoint(str(name), int(value))
        for name, value in sorted(components.items(), key=lambda item: str(item[0]))
        if isinstance(value, int) and not isinstance(value, bool)
    )


def _resource_points(run_directory: Path) -> tuple[ResourcePoint, ...]:
    path = run_directory / "logs" / "resources.jsonl"
    points: list[ResourcePoint] = []
    elapsed_offset = 0.0
    previous_raw_elapsed: float | None = None
    previous_normalized_elapsed = 0.0
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = _mapping(json.loads(line), "resource sample")
            memory = _optional_float(value.get("gpu_memory_used_bytes"))
            raw_elapsed = _optional_float(value.get("elapsed_seconds")) or 0.0
            if previous_raw_elapsed is not None and raw_elapsed < previous_raw_elapsed:
                elapsed_offset = previous_normalized_elapsed
            normalized_elapsed = elapsed_offset + raw_elapsed
            points.append(
                ResourcePoint(
                    elapsed_seconds=normalized_elapsed,
                    step=None,
                    throughput_samples_per_second=None,
                    allocated_vram_gib=(
                        memory / (1024.0**3) if memory is not None else None
                    ),
                    gpu_utilization_percent=_optional_float(
                        value.get("gpu_utilization_percent")
                    ),
                    power_watts=_optional_float(value.get("gpu_power_watts")),
                    temperature_celsius=_optional_float(
                        value.get("gpu_temperature_celsius")
                    ),
                )
            )
            previous_raw_elapsed = raw_elapsed
            previous_normalized_elapsed = normalized_elapsed
    summary = resource_summary(run_directory)
    if summary.train_samples_per_second is not None:
        points.append(
            ResourcePoint(
                elapsed_seconds=max(
                    summary.duration_seconds or 0.0,
                    previous_normalized_elapsed,
                ),
                step=None,
                throughput_samples_per_second=summary.train_samples_per_second,
                allocated_vram_gib=summary.peak_vram_gib,
                gpu_utilization_percent=None,
                power_watts=summary.maximum_power_watts,
                temperature_celsius=summary.maximum_temperature_celsius,
            )
        )
    return tuple(points)


def _write_checkpoint_tables(
    store: ArtifactStore, points: tuple[CheckpointMetricPoint, ...]
) -> None:
    if not points:
        return
    headers = ("checkpoint", "epoch", "top1", "macro_f1", "balanced", "eval_loss")
    rows: tuple[tuple[TableCell, ...], ...] = tuple(
        (
            item.checkpoint_id,
            item.epoch,
            item.top1_accuracy,
            item.macro_f1,
            item.balanced_accuracy,
            item.eval_loss,
        )
        for item in points
    )
    write_csv_table(store.path("tables", "checkpoint_metrics.csv"), headers, rows)
    write_latex_table(
        store.path("tables", "checkpoint_metrics.tex"),
        headers,
        rows,
        caption="Development metrics by training checkpoint",
        label="tab:e1_checkpoint_metrics",
    )


def _write_resource_tables(store: ArtifactStore, run_directory: Path) -> None:
    summary = resource_summary(run_directory)
    rows: tuple[tuple[TableCell, ...], ...] = tuple(
        (name, value) for name, value in asdict(summary).items()
    )
    write_csv_table(
        store.path("tables", "resource_summary.csv"),
        ("metric", "value"),
        rows,
    )
    write_latex_table(
        store.path("tables", "resource_summary.tex"),
        ("metric", "value"),
        rows,
        caption="Measured training resource usage",
        label="tab:e1_resource_summary",
    )


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _mapping(value: object, context: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)
