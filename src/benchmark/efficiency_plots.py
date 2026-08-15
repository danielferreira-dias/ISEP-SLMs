"""Render thesis figures for the controlled same-hardware efficiency cohort."""

from __future__ import annotations

import csv
import os
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.benchmark.efficiency_comparison import (
        EfficiencyComparisonRow,
        ParetoPoint,
    )

_PLOT_SPECS = {
    "latency_p50_seconds": ("Median end-to-end latency (s)", "quality_vs_latency"),
    "idle_adjusted_energy_wh_per_correct": (
        "Idle-adjusted GPU energy (Wh / correct)",
        "quality_vs_energy_per_correct",
    ),
    "peak_vram_gib": ("Peak GPU memory (GiB)", "quality_vs_vram"),
    "parameters_billions": ("Model parameters (billions)", "quality_vs_parameters"),
}


def render_efficiency_figures(
    rows: tuple[EfficiencyComparisonRow, ...],
    points: tuple[ParetoPoint, ...],
    output_directory: Path,
) -> None:
    """Write PNG, SVG, and source CSV for every requested Pareto view."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)
    for x_metric, (x_label, stem) in _PLOT_SPECS.items():
        selected = tuple(point for point in points if point.x_metric == x_metric)
        _write_source(output_directory / f"{stem}_source.csv", selected)
        benchmark_ids = tuple(dict.fromkeys(point.benchmark_id for point in selected))
        if not benchmark_ids:
            continue
        figure, axes = plt.subplots(
            1,
            len(benchmark_ids),
            figsize=(6.0 * len(benchmark_ids), 5.0),
            squeeze=False,
        )
        color_map = _model_colors(rows)
        for axis, benchmark_id in zip(axes[0], benchmark_ids, strict=True):
            task_points = [
                point for point in selected if point.benchmark_id == benchmark_id
            ]
            for point in task_points:
                axis.scatter(
                    point.x_value,
                    100.0 * point.quality_value,
                    color=color_map[point.model_id],
                    marker="o" if point.on_frontier else "x",
                    s=75,
                    linewidth=1.8,
                    zorder=3,
                )
                axis.annotate(
                    point.model,
                    (point.x_value, 100.0 * point.quality_value),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
            frontier = sorted(
                (point for point in task_points if point.on_frontier),
                key=lambda point: point.x_value,
            )
            if len(frontier) > 1:
                axis.plot(
                    [point.x_value for point in frontier],
                    [100.0 * point.quality_value for point in frontier],
                    color="#111827",
                    linestyle="--",
                    linewidth=1.1,
                    alpha=0.7,
                )
            axis.set_title(_title(benchmark_id))
            axis.set_xlabel(x_label)
            axis.set_ylabel("Primary quality (%)")
            axis.grid(alpha=0.25)
        figure.suptitle("Same-hardware quality-efficiency Pareto comparison")
        figure.tight_layout()
        figure.savefig(output_directory / f"{stem}.png", dpi=220, bbox_inches="tight")
        figure.savefig(output_directory / f"{stem}.svg", bbox_inches="tight")
        plt.close(figure)

    _render_percentile_figure(
        rows,
        output_directory,
        metric_names=(
            "latency_p50_seconds",
            "latency_p95_seconds",
            "latency_p99_seconds",
        ),
        labels=("p50", "p95", "p99"),
        stem="latency_percentiles",
        y_label="End-to-end latency (s)",
    )
    _render_percentile_figure(
        rows,
        output_directory,
        metric_names=(
            "ttft_p50_seconds",
            "ttft_p95_seconds",
            "ttft_p99_seconds",
        ),
        labels=("p50", "p95", "p99"),
        stem="ttft_percentiles",
        y_label="Time to first token (s)",
    )
    _render_percentile_figure(
        rows,
        output_directory,
        metric_names=(
            "tpot_p50_seconds",
            "tpot_p95_seconds",
            "tpot_p99_seconds",
        ),
        labels=("p50", "p95", "p99"),
        stem="tpot_percentiles",
        y_label="Mean time per output token (s)",
    )
    _render_metric_panels(
        rows,
        output_directory,
        metrics=(
            ("output_tokens_per_second_mean", "Output tokens / second"),
            ("requests_per_second", "Requests / second"),
        ),
        stem="throughput",
    )
    _render_metric_panels(
        rows,
        output_directory,
        metrics=(
            ("peak_vram_gib", "Peak GPU memory (GiB)"),
            ("peak_server_ram_gib", "Peak server RAM (GiB)"),
        ),
        stem="memory",
    )
    _render_metric_panels(
        rows,
        output_directory,
        metrics=(
            ("idle_adjusted_energy_wh_per_request", "Idle-adjusted Wh / request"),
            ("gpu_seconds_per_request", "GPU-seconds / request"),
        ),
        stem="resource_per_request",
    )
    _render_metric_panels(
        rows,
        output_directory,
        metrics=(
            ("mean_gpu_utilization_percent", "Mean GPU utilization (%)"),
            ("mean_gpu_power_watts", "Mean GPU board power (W)"),
        ),
        stem="gpu_utilization_power",
    )
    _render_metric_panels(
        rows,
        output_directory,
        metrics=(
            ("peak_gpu_power_watts", "Peak GPU board power (W)"),
            ("peak_gpu_temperature_celsius", "Peak GPU temperature (°C)"),
        ),
        stem="gpu_peak_telemetry",
    )


def _model_colors(rows: tuple[EfficiencyComparisonRow, ...]) -> dict[str, str]:
    palette = ("#2563EB", "#DC2626", "#059669", "#7C3AED", "#D97706")
    model_ids = tuple(dict.fromkeys(row.model_id for row in rows))
    return {
        model_id: palette[index % len(palette)]
        for index, model_id in enumerate(model_ids)
    }


def _write_source(path: Path, points: tuple[ParetoPoint, ...]) -> None:
    records = [asdict(point) for point in points]
    if not records:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def _render_percentile_figure(
    rows: tuple[EfficiencyComparisonRow, ...],
    output_directory: Path,
    *,
    metric_names: tuple[str, str, str],
    labels: tuple[str, str, str],
    stem: str,
    y_label: str,
) -> None:
    """Render one line chart per task for p50/p95/p99 measurements."""

    import matplotlib.pyplot as plt

    _write_row_source(output_directory / f"{stem}_source.csv", rows, metric_names)
    benchmark_ids = tuple(dict.fromkeys(row.benchmark_id for row in rows))
    figure, axes = plt.subplots(
        1,
        len(benchmark_ids),
        figsize=(5.2 * len(benchmark_ids), 4.8),
        squeeze=False,
    )
    colors = _model_colors(rows)
    for axis, benchmark_id in zip(axes[0], benchmark_ids, strict=True):
        for row in (item for item in rows if item.benchmark_id == benchmark_id):
            values = tuple(getattr(row, name) for name in metric_names)
            if any(value is None for value in values):
                continue
            axis.plot(
                labels,
                values,
                marker="o",
                linewidth=1.7,
                color=colors[row.model_id],
                label=row.model,
            )
        axis.set_title(_title(benchmark_id))
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.25)
    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(handles),
        )
    figure.suptitle(f"Same-hardware {stem.replace('_', ' ')}", y=1.09)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    figure.savefig(output_directory / f"{stem}.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_directory / f"{stem}.svg", bbox_inches="tight")
    plt.close(figure)


def _render_metric_panels(
    rows: tuple[EfficiencyComparisonRow, ...],
    output_directory: Path,
    *,
    metrics: tuple[tuple[str, str], tuple[str, str]],
    stem: str,
) -> None:
    """Render task-grouped bars for two related resource metrics."""

    import matplotlib.pyplot as plt

    metric_names = tuple(name for name, _ in metrics)
    _write_row_source(output_directory / f"{stem}_source.csv", rows, metric_names)
    benchmark_ids = tuple(dict.fromkeys(row.benchmark_id for row in rows))
    model_ids = tuple(dict.fromkeys(row.model_id for row in rows))
    model_labels = {
        row.model_id: row.model for row in rows if row.model_id in model_ids
    }
    colors = _model_colors(rows)
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), squeeze=False)
    width = 0.8 / len(model_ids)
    for axis, (metric_name, y_label) in zip(axes[0], metrics, strict=True):
        for model_index, model_id in enumerate(model_ids):
            values: list[float] = []
            for benchmark_id in benchmark_ids:
                matching = next(
                    row
                    for row in rows
                    if row.model_id == model_id and row.benchmark_id == benchmark_id
                )
                raw = getattr(matching, metric_name)
                values.append(float(raw) if raw is not None else 0.0)
            positions = [
                index - 0.4 + width / 2 + model_index * width
                for index in range(len(benchmark_ids))
            ]
            axis.bar(
                positions,
                values,
                width=width,
                color=colors[model_id],
                label=model_labels[model_id],
            )
        axis.set_xticks(
            range(len(benchmark_ids)),
            tuple(_short_title(value) for value in benchmark_ids),
            rotation=20,
            ha="right",
        )
        axis.set_ylabel(y_label)
        axis.grid(axis="y", alpha=0.25)
    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(handles),
    )
    figure.suptitle(f"Same-hardware {stem.replace('_', ' ')}", y=1.09)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    figure.savefig(output_directory / f"{stem}.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_directory / f"{stem}.svg", bbox_inches="tight")
    plt.close(figure)


def _write_row_source(
    path: Path,
    rows: tuple[EfficiencyComparisonRow, ...],
    metric_names: tuple[str, ...],
) -> None:
    """Persist the exact model-task values underlying a resource figure."""

    fieldnames = ("model_id", "model", "benchmark_id", *metric_names)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field_name: getattr(row, field_name) for field_name in fieldnames}
            )
    os.replace(temporary, path)


def _title(benchmark_id: str) -> str:
    return benchmark_id.replace("_", " ").title()


def _short_title(benchmark_id: str) -> str:
    names = {
        "visual_top_k_closed_set": "Top-K",
        "visual_disease_confusion_sets": "Confusion",
        "evidence_grounded_diagnosis": "Evidence",
        "open_ended_diagnosis": "Open-ended",
    }
    return names.get(benchmark_id, _title(benchmark_id))
