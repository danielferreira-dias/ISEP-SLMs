"""Filesystem orchestration for paired, multi-seed run comparison."""

from __future__ import annotations

import base64
from collections import defaultdict
from html import escape
from pathlib import Path

from src.train.evaluation.comparison import (
    aggregate_seed_runs,
    compare_paired_runs,
)
from src.train.evaluation.models import (
    ComparableRun,
    PairedComparison,
    SeedAggregate,
)

from .plot_models import QualityCostPoint
from .snapshots import load_comparable_run
from .store import ArtifactStore
from .tables import write_csv_table, write_latex_table
from .thesis_plots import ThesisPlotter
from .types import ComparisonArtifacts, JsonValue, TableCell


def _comparison_json(comparison: PairedComparison) -> dict[str, JsonValue]:
    return {
        "baseline_run_id": comparison.baseline_run_id,
        "candidate_run_id": comparison.candidate_run_id,
        "sample_count": comparison.sample_count,
        "group_count": comparison.group_count,
        "top1_delta": comparison.top1_delta,
        "macro_f1_delta": comparison.macro_f1_delta,
        "balanced_accuracy_delta": comparison.balanced_accuracy_delta,
        "top1_delta_ci": {
            "low": comparison.top1_delta_ci.low,
            "high": comparison.top1_delta_ci.high,
            "confidence": comparison.top1_delta_ci.confidence,
        },
        "macro_f1_delta_ci": {
            "low": comparison.macro_f1_delta_ci.low,
            "high": comparison.macro_f1_delta_ci.high,
            "confidence": comparison.macro_f1_delta_ci.confidence,
        },
        "baseline_only_correct": comparison.baseline_only_correct,
        "candidate_only_correct": comparison.candidate_only_correct,
        "mcnemar_exact_p": comparison.mcnemar_exact_p,
        "bootstrap_iterations": comparison.bootstrap_iterations,
        "bootstrap_seed": comparison.bootstrap_seed,
    }


def _aggregate_json(aggregate: SeedAggregate) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "experiment_id": aggregate.experiment_id,
        "seeds": list(aggregate.seeds),
    }
    for name, metric in (
        ("top1_accuracy", aggregate.top1_accuracy),
        ("macro_f1", aggregate.macro_f1),
        ("balanced_accuracy", aggregate.balanced_accuracy),
        ("invalid_rate", aggregate.invalid_rate),
        ("duration_seconds", aggregate.duration_seconds),
        ("gpu_hours", aggregate.gpu_hours),
        ("peak_vram_gib", aggregate.peak_vram_gib),
        ("trainable_parameters", aggregate.trainable_parameters),
    ):
        payload[name] = (
            None
            if metric is None
            else {
                "mean": metric.mean,
                "standard_deviation": metric.standard_deviation,
            }
        )
    return payload


def _pair_runs(
    runs: tuple[ComparableRun, ...],
) -> tuple[
    str,
    str,
    tuple[tuple[ComparableRun, ComparableRun], ...],
]:
    if len(runs) < 2:
        raise ValueError("Comparison requires at least two run directories")
    baseline_experiment = runs[0].experiment_id
    candidate_experiment = next(
        (run.experiment_id for run in runs if run.experiment_id != baseline_experiment),
        "",
    )
    experiments = {run.experiment_id for run in runs}
    if not candidate_experiment or len(experiments) != 2:
        raise ValueError("Runs must contain exactly two distinct experiments")
    grouped: dict[str, dict[int, ComparableRun]] = defaultdict(dict)
    for run in runs:
        if run.seed in grouped[run.experiment_id]:
            raise ValueError(f"Duplicate seed {run.seed} for {run.experiment_id!r}")
        grouped[run.experiment_id][run.seed] = run
    baseline_seeds = grouped[baseline_experiment].keys()
    candidate_seeds = grouped[candidate_experiment].keys()
    if baseline_seeds != candidate_seeds:
        raise ValueError("Experiments must have exactly the same seed set")
    pairs = tuple(
        (
            grouped[baseline_experiment][seed],
            grouped[candidate_experiment][seed],
        )
        for seed in sorted(baseline_seeds)
    )
    return baseline_experiment, candidate_experiment, pairs


def _comparison_rows(
    pairs: tuple[tuple[ComparableRun, ComparableRun], ...],
    comparisons: tuple[PairedComparison, ...],
) -> tuple[tuple[TableCell, ...], ...]:
    return tuple(
        (
            baseline.seed,
            baseline.run_id,
            candidate.run_id,
            f"{comparison.top1_delta:.6f}",
            f"{comparison.top1_delta_ci.low:.6f}",
            f"{comparison.top1_delta_ci.high:.6f}",
            f"{comparison.macro_f1_delta:.6f}",
            f"{comparison.macro_f1_delta_ci.low:.6f}",
            f"{comparison.macro_f1_delta_ci.high:.6f}",
            f"{comparison.mcnemar_exact_p:.8f}",
            baseline.duration_seconds,
            candidate.duration_seconds,
            baseline.gpu_hours,
            candidate.gpu_hours,
            baseline.peak_vram_gib,
            candidate.peak_vram_gib,
            baseline.trainable_parameters,
            candidate.trainable_parameters,
        )
        for (baseline, candidate), comparison in zip(pairs, comparisons, strict=True)
    )


def _comparison_markdown(
    baseline_experiment: str,
    candidate_experiment: str,
    rows: tuple[tuple[TableCell, ...], ...],
    aggregates: tuple[SeedAggregate, ...],
) -> str:
    lines = [
        "# Paired training comparison",
        "",
        f"Baseline: `{baseline_experiment}`  ",
        f"Candidate: `{candidate_experiment}`",
        "",
        "| Seed | Δ Top-1 | Δ Macro-F1 | McNemar exact p |",
        "|---:|---:|---:|---:|",
    ]
    lines.extend(f"| {row[0]} | {row[3]} | {row[6]} | {row[9]} |" for row in rows)
    lines.extend(("", "## Aggregate across seeds", ""))
    for aggregate in aggregates:
        lines.append(
            f"- **{aggregate.experiment_id}**: Top-1 "
            f"{aggregate.top1_accuracy.mean:.4f} ± "
            f"{aggregate.top1_accuracy.standard_deviation:.4f}; Macro-F1 "
            f"{aggregate.macro_f1.mean:.4f} ± "
            f"{aggregate.macro_f1.standard_deviation:.4f}."
        )
        if aggregate.duration_seconds is not None:
            lines.append(
                f"  Duration: {aggregate.duration_seconds.mean:.1f} ± "
                f"{aggregate.duration_seconds.standard_deviation:.1f} s."
            )
        if aggregate.gpu_hours is not None:
            lines.append(
                f"  GPU-hours: {aggregate.gpu_hours.mean:.3f} ± "
                f"{aggregate.gpu_hours.standard_deviation:.3f}."
            )
        if aggregate.peak_vram_gib is not None:
            lines.append(
                f"  Peak VRAM: {aggregate.peak_vram_gib.mean:.2f} ± "
                f"{aggregate.peak_vram_gib.standard_deviation:.2f} GiB."
            )
        if aggregate.trainable_parameters is not None:
            lines.append(
                f"  Trainable parameters: "
                f"{aggregate.trainable_parameters.mean:.0f} ± "
                f"{aggregate.trainable_parameters.standard_deviation:.0f}."
            )
    return "\n".join(lines) + "\n"


def _comparison_html(markdown_summary: str, figure_path: Path | None) -> str:
    figure_html = ""
    if figure_path is not None:
        encoded = base64.b64encode(figure_path.read_bytes()).decode("ascii")
        figure_html = (
            f'<img alt="Quality versus cost" src="data:image/png;base64,{encoded}">'
        )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Paired training comparison</title><style>
body{{font:16px/1.5 system-ui,sans-serif;max-width:1000px;
margin:2rem auto;padding:0 1rem}}
pre{{white-space:pre-wrap}}
img{{max-width:100%;height:auto}}
</style></head><body><h1>Paired training comparison</h1>
<pre>{escape(markdown_summary)}</pre>{figure_html}</body></html>"""


def _quality_cost_point(run: ComparableRun) -> QualityCostPoint:
    """Build a quality-cost point after explicitly narrowing optional costs."""

    gpu_hours = run.gpu_hours
    peak_vram_gib = run.peak_vram_gib
    trainable_parameters = run.trainable_parameters
    if gpu_hours is None or peak_vram_gib is None or trainable_parameters is None:
        raise ValueError(f"Run {run.run_id!r} has incomplete resource metadata")
    return QualityCostPoint(
        run_id=run.run_id,
        experiment_id=f"{run.experiment_id} (seed {run.seed})",
        top1_accuracy=run.metrics.top1_accuracy,
        macro_f1=run.metrics.macro_f1,
        gpu_hours=gpu_hours,
        peak_vram_gib=peak_vram_gib,
        trainable_parameters=trainable_parameters,
    )


def compare_runs(
    run_directories: tuple[Path, ...],
    output_directory: Path,
    *,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 3407,
    render_plots: bool = True,
) -> ComparisonArtifacts:
    """Load, pair by seed, compare, and persist thesis-ready run results."""

    runs = tuple(load_comparable_run(path) for path in run_directories)
    baseline_experiment, candidate_experiment, pairs = _pair_runs(runs)
    comparisons = tuple(
        compare_paired_runs(
            baseline,
            candidate,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
        )
        for baseline, candidate in pairs
    )
    aggregates = aggregate_seed_runs(runs)
    store = ArtifactStore.at(output_directory, create=True)
    json_path = store.write_json(
        "metrics",
        "comparison.json",
        {
            "baseline_experiment": baseline_experiment,
            "candidate_experiment": candidate_experiment,
            "paired_results": [
                _comparison_json(comparison) for comparison in comparisons
            ],
            "seed_aggregates": [_aggregate_json(aggregate) for aggregate in aggregates],
        },
    )
    headers = (
        "seed",
        "baseline_run_id",
        "candidate_run_id",
        "top1_delta",
        "top1_ci_low",
        "top1_ci_high",
        "macro_f1_delta",
        "macro_f1_ci_low",
        "macro_f1_ci_high",
        "mcnemar_exact_p",
        "baseline_duration_seconds",
        "candidate_duration_seconds",
        "baseline_gpu_hours",
        "candidate_gpu_hours",
        "baseline_peak_vram_gib",
        "candidate_peak_vram_gib",
        "baseline_trainable_parameters",
        "candidate_trainable_parameters",
    )
    rows = _comparison_rows(pairs, comparisons)
    csv_path = write_csv_table(
        store.path("tables", "paired_comparison.csv"), headers, rows
    )
    latex_path = write_latex_table(
        store.path("tables", "paired_comparison.tex"),
        headers,
        rows,
        caption="Paired comparison of controlled vision LoRA experiments",
        label="tab:paired_vision_lora_comparison",
    )
    markdown = _comparison_markdown(
        baseline_experiment, candidate_experiment, rows, aggregates
    )
    markdown_path = store.write_text("report", "comparison_summary.md", markdown)
    complete_cost_runs = tuple(
        run
        for run in runs
        if run.gpu_hours is not None
        and run.peak_vram_gib is not None
        and run.trainable_parameters is not None
    )
    figure = None
    if render_plots and complete_cost_runs:
        figure = ThesisPlotter(store.layout.figures).quality_cost(
            tuple(_quality_cost_point(run) for run in complete_cost_runs)
        )
    html_path = store.write_text(
        "report",
        "comparison_report.html",
        _comparison_html(markdown, figure.png_path if figure else None),
    )
    return ComparisonArtifacts(
        json_path=json_path,
        csv_path=csv_path,
        latex_path=latex_path,
        markdown_path=markdown_path,
        html_path=html_path,
        figure=figure,
    )
