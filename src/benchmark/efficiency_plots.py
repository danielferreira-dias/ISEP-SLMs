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


def _title(benchmark_id: str) -> str:
    return benchmark_id.replace("_", " ").title()
