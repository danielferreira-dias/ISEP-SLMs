"""Dataset, parameter, resource, and quality-cost thesis figures."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

from .plot_models import (
    DistributionPoint,
    QualityCostPoint,
    ResourcePoint,
    TrainableParameterPoint,
)
from .plotting import (
    LineSeries,
    grouped_bar_figure,
    line_figure,
    scatter_figure,
)
from .tables import write_csv_table
from .types import FigureArtifact


def plot_data_distribution(
    figure_directory: Path,
    points: tuple[DistributionPoint, ...],
    *,
    name: str = "data_distribution",
    title: str = "Frozen dataset distribution",
) -> FigureArtifact | None:
    """Render counts by frozen split and category."""

    if not points:
        return None
    source = write_csv_table(
        figure_directory / f"{name}_source.csv",
        ("split", "category", "count"),
        tuple((point.split, point.category, point.count) for point in points),
    )
    categories = tuple(sorted({point.category for point in points}))
    splits = tuple(sorted({point.split for point in points}))
    lookup = {(point.split, point.category): point.count for point in points}
    return grouped_bar_figure(
        name=name,
        title=title,
        y_label="Examples",
        categories=categories,
        series=tuple(
            (
                split,
                tuple(
                    float(lookup.get((split, category), 0)) for category in categories
                ),
            )
            for split in splits
        ),
        figure_directory=figure_directory,
        source_csv_path=source,
    )


def plot_trainable_parameters(
    figure_directory: Path,
    points: tuple[TrainableParameterPoint, ...],
) -> FigureArtifact | None:
    """Render LoRA trainable parameter counts by component."""

    if not points:
        return None
    source = write_csv_table(
        figure_directory / "trainable_parameters_source.csv",
        ("component", "parameter_count"),
        tuple((point.component, point.parameter_count) for point in points),
    )
    return grouped_bar_figure(
        name="trainable_parameters",
        title="Trainable parameters by component",
        y_label="Parameters",
        categories=tuple(point.component for point in points),
        series=(
            (
                "trainable",
                tuple(float(point.parameter_count) for point in points),
            ),
        ),
        figure_directory=figure_directory,
        source_csv_path=source,
    )


def plot_resource_usage(
    figure_directory: Path,
    points: tuple[ResourcePoint, ...],
) -> tuple[FigureArtifact, ...]:
    """Render each available resource channel against wall time."""

    if not points:
        return ()
    source = write_csv_table(
        figure_directory / "resource_usage_source.csv",
        (
            "elapsed_seconds",
            "step",
            "throughput_samples_per_second",
            "allocated_vram_gib",
            "gpu_utilization_percent",
            "power_watts",
            "temperature_celsius",
        ),
        tuple(
            (
                point.elapsed_seconds,
                point.step,
                point.throughput_samples_per_second,
                point.allocated_vram_gib,
                point.gpu_utilization_percent,
                point.power_watts,
                point.temperature_celsius,
            )
            for point in points
        ),
    )
    channels: tuple[tuple[str, str, Callable[[ResourcePoint], float | None]], ...] = (
        ("throughput", "Samples/s", lambda item: item.throughput_samples_per_second),
        ("vram", "Allocated VRAM (GiB)", lambda item: item.allocated_vram_gib),
        (
            "gpu_utilization",
            "GPU utilization (%)",
            lambda item: item.gpu_utilization_percent,
        ),
        ("power", "Power (W)", lambda item: item.power_watts),
        ("temperature", "Temperature (°C)", lambda item: item.temperature_celsius),
    )
    figures: list[FigureArtifact] = []
    for name, y_label, selector in channels:
        available: list[tuple[float, float]] = []
        for point in points:
            value = selector(point)
            if value is not None:
                available.append((point.elapsed_seconds, value))
        if not available:
            continue
        values = tuple(float(value) for _, value in available)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Resource channel {name!r} is non-finite")
        figures.append(
            line_figure(
                name=f"resource_{name}",
                title=f"Training {name.replace('_', ' ')}",
                x_label="Elapsed seconds",
                y_label=y_label,
                series=(
                    LineSeries(
                        name,
                        tuple(time for time, _ in available),
                        values,
                    ),
                ),
                figure_directory=figure_directory,
                source_csv_path=source,
            )
        )
    return tuple(figures)


def plot_quality_cost(
    figure_directory: Path,
    points: tuple[QualityCostPoint, ...],
) -> FigureArtifact | None:
    """Render macro-F1 against measured GPU-hours."""

    if not points:
        return None
    source = write_csv_table(
        figure_directory / "quality_cost_source.csv",
        (
            "run_id",
            "experiment_id",
            "top1_accuracy",
            "macro_f1",
            "gpu_hours",
            "peak_vram_gib",
            "trainable_parameters",
        ),
        tuple(
            (
                point.run_id,
                point.experiment_id,
                point.top1_accuracy,
                point.macro_f1,
                point.gpu_hours,
                point.peak_vram_gib,
                point.trainable_parameters,
            )
            for point in points
        ),
    )
    return scatter_figure(
        name="quality_vs_cost",
        title="Validation quality versus training cost",
        x_label="GPU-hours",
        y_label="Macro-F1",
        labels=tuple(point.experiment_id for point in points),
        x=tuple(point.gpu_hours for point in points),
        y=tuple(point.macro_f1 for point in points),
        figure_directory=figure_directory,
        source_csv_path=source,
    )
