"""Consolidate same-hardware efficiency runs into thesis-ready artifacts."""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from src.benchmark.efficiency_plots import render_efficiency_figures

BENCHMARKS = (
    "visual_top_k_closed_set",
    "visual_disease_confusion_sets",
    "evidence_grounded_diagnosis",
    "open_ended_diagnosis",
)

PARETO_X_METRICS = (
    "latency_p50_seconds",
    "idle_adjusted_energy_wh_per_correct",
    "peak_vram_gib",
    "parameters_billions",
)


@dataclass(frozen=True, slots=True)
class EfficiencyComparisonRow:
    """One model-task quality and resource observation."""

    model_id: str
    model: str
    parameters_billions: float
    dtype: str
    hardware: str
    benchmark_id: str
    request_count: int
    quality_metric: str | None
    quality_value: float | None
    ttft_p50_seconds: float | None
    ttft_p95_seconds: float | None
    ttft_p99_seconds: float | None
    latency_p50_seconds: float | None
    latency_p95_seconds: float | None
    latency_p99_seconds: float | None
    tpot_p50_seconds: float | None
    tpot_p95_seconds: float | None
    tpot_p99_seconds: float | None
    output_tokens_per_second_mean: float | None
    output_tokens_per_second_p50: float | None
    requests_per_second: float | None
    gpu_seconds_per_request: float | None
    peak_vram_gib: float | None
    peak_server_ram_gib: float | None
    gpu_energy_wh: float | None
    idle_adjusted_gpu_energy_wh: float | None
    energy_wh_per_request: float | None
    idle_adjusted_energy_wh_per_request: float | None
    energy_wh_per_correct: float | None
    idle_adjusted_energy_wh_per_correct: float | None
    estimated_gpu_cost_usd: float | None
    estimated_gpu_cost_usd_per_request: float | None
    estimated_gpu_cost_usd_per_correct: float | None


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    """One candidate in a benchmark-specific quality-efficiency frontier."""

    benchmark_id: str
    x_metric: str
    model_id: str
    model: str
    x_value: float
    quality_value: float
    on_frontier: bool


def build_efficiency_comparison(
    root: Path,
    *,
    model_ids: tuple[str, ...],
    output_directory: Path | None = None,
) -> dict[str, object]:
    """Validate and combine completed same-hardware model runs.

    Args:
        root: Directory containing one completed run directory per model ID.
        model_ids: Exact model IDs required for the controlled comparison.
        output_directory: Optional destination; defaults to ``root/comparison``.

    Returns:
        JSON-compatible comparison summary.
    """

    if len(model_ids) < 2 or len(set(model_ids)) != len(model_ids):
        raise ValueError("model_ids must contain at least two unique models")
    rows, cohort_sha, hardware, concurrency = _load_runs(root, model_ids)
    destination = output_directory or root / "comparison"
    tables = destination / "tables"
    metrics = destination / "metrics"
    figures = destination / "figures"
    reports = destination / "report"
    for directory in (tables, metrics, figures, reports):
        directory.mkdir(parents=True, exist_ok=True)

    points = tuple(
        point
        for metric in PARETO_X_METRICS
        for point in pareto_points(rows, x_metric=metric)
    )
    _write_dataclass_csv(tables / "same_hardware_comparison.csv", rows)
    _write_dataclass_csv(tables / "pareto_frontiers.csv", points)
    _write_latex(tables / "same_hardware_comparison.tex", rows)
    _write_parquet(metrics / "same_hardware_comparison.parquet", rows)
    render_efficiency_figures(rows, points, figures)

    result: dict[str, object] = {
        "schema_version": 1,
        "model_ids": list(model_ids),
        "model_count": len(model_ids),
        "benchmark_ids": list(BENCHMARKS),
        "row_count": len(rows),
        "cohort_manifest_sha256": cohort_sha,
        "hardware": hardware,
        "concurrency": concurrency,
        "same_hardware": True,
        "frontier_metrics": list(PARETO_X_METRICS),
    }
    _atomic_json(metrics / "comparison_summary.json", result)
    _write_markdown(reports / "thesis_summary.md", rows, result)
    return result


def pareto_points(
    rows: tuple[EfficiencyComparisonRow, ...], *, x_metric: str
) -> tuple[ParetoPoint, ...]:
    """Mark non-dominated models, maximizing quality and minimizing X."""

    if x_metric not in PARETO_X_METRICS:
        raise ValueError(f"unsupported Pareto metric: {x_metric}")
    result: list[ParetoPoint] = []
    for benchmark_id in BENCHMARKS:
        candidates: list[tuple[EfficiencyComparisonRow, float, float]] = []
        for row in rows:
            if row.benchmark_id != benchmark_id or row.quality_value is None:
                continue
            x_value = getattr(row, x_metric)
            if isinstance(x_value, (int, float)) and math.isfinite(x_value):
                candidates.append((row, float(x_value), row.quality_value))
        for row, x_value, quality in candidates:
            dominated = any(
                other_x <= x_value
                and other_quality >= quality
                and (other_x < x_value or other_quality > quality)
                for _, other_x, other_quality in candidates
            )
            result.append(
                ParetoPoint(
                    benchmark_id=benchmark_id,
                    x_metric=x_metric,
                    model_id=row.model_id,
                    model=row.model,
                    x_value=x_value,
                    quality_value=quality,
                    on_frontier=not dominated,
                )
            )
    return tuple(result)


def _load_runs(
    root: Path, model_ids: tuple[str, ...]
) -> tuple[tuple[EfficiencyComparisonRow, ...], str, str, int]:
    rows: list[EfficiencyComparisonRow] = []
    cohort_values: set[str] = set()
    hardware_values: set[str] = set()
    concurrency_values: set[int] = set()
    for model_id in model_ids:
        run = root / model_id
        summary = _read_object(run / "metrics" / "efficiency_summary.json")
        status = _read_object(run / "campaign_status.json")
        if status.get("status") != "completed":
            raise ValueError(f"run is not complete: {run}")
        if summary.get("model_id") != model_id or summary.get("request_count") != 400:
            raise ValueError(f"invalid model identity or request count: {run}")
        cohort_values.add(_required_string(summary, "cohort_manifest_sha256"))
        hardware_values.add(_required_string(summary, "hardware"))
        concurrency_values.add(_required_int(summary, "concurrency"))
        table_path = run / "tables" / "quality_efficiency.csv"
        with table_path.open(encoding="utf-8", newline="") as stream:
            records = list(csv.DictReader(stream))
        if {record.get("benchmark_id") for record in records} != set(BENCHMARKS):
            raise ValueError(f"run does not contain the four cohort tasks: {run}")
        if sum(_required_csv_int(record, "request_count") for record in records) != 400:
            raise ValueError(f"CSV request count is not 400: {table_path}")
        rows.extend(_parse_row(model_id, record) for record in records)
    if len(cohort_values) != 1 or len(hardware_values) != 1:
        raise ValueError("runs do not share one cohort and one hardware profile")
    if concurrency_values != {1}:
        raise ValueError("same-hardware thesis comparison requires concurrency=1")
    return (
        tuple(rows),
        next(iter(cohort_values)),
        next(iter(hardware_values)),
        1,
    )


def _parse_row(model_id: str, record: dict[str, str]) -> EfficiencyComparisonRow:
    values: dict[str, object] = {"model_id": model_id}
    integer_fields = {"request_count"}
    text_fields = {"model", "dtype", "hardware", "benchmark_id", "quality_metric"}
    for field_name in EfficiencyComparisonRow.__dataclass_fields__:
        if field_name == "model_id":
            continue
        raw = record.get(field_name, "")
        if field_name in integer_fields:
            values[field_name] = _csv_int(raw, field_name)
        elif field_name in text_fields:
            values[field_name] = raw or None
        else:
            values[field_name] = _csv_float(raw, field_name)
    for field_name in ("model", "dtype", "hardware", "benchmark_id"):
        if values.get(field_name) is None:
            raise ValueError(f"missing required field {field_name!r}")
    return EfficiencyComparisonRow(**values)  # type: ignore[arg-type]


def _write_dataclass_csv(path: Path, rows: tuple[object, ...]) -> None:
    if not rows:
        raise ValueError("cannot write an empty comparison")
    records = [asdict(row) for row in rows]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def _write_parquet(path: Path, rows: tuple[EfficiencyComparisonRow, ...]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist([asdict(row) for row in rows])
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary)
    os.replace(temporary, path)


def _write_latex(path: Path, rows: tuple[EfficiencyComparisonRow, ...]) -> None:
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Model & Task & Quality & TTFT & Latency & Tok/s & VRAM & Wh/query \\",
        r"\midrule",
    ]
    for row in rows:
        values = (
            _latex(row.model),
            _latex(row.benchmark_id),
            _number(row.quality_value),
            _number(row.ttft_p50_seconds),
            _number(row.latency_p50_seconds),
            _number(row.output_tokens_per_second_mean),
            _number(row.peak_vram_gib),
            _number(row.idle_adjusted_energy_wh_per_request),
        )
        lines.append(" & ".join(values) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_markdown(
    path: Path,
    rows: tuple[EfficiencyComparisonRow, ...],
    summary: dict[str, object],
) -> None:
    lines = [
        "# Same-hardware model-efficiency comparison",
        "",
        f"- Hardware: `{summary['hardware']}`",
        f"- Models: {summary['model_count']}",
        "- Concurrency: 1",
        "- Measured requests per model: 400",
        f"- Cohort SHA-256: `{summary['cohort_manifest_sha256']}`",
        "",
        "| Model | Task | Quality | Latency p50 (s) | Tok/s | VRAM (GiB) | Wh/query* |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    row.model,
                    row.benchmark_id,
                    _number(row.quality_value),
                    _number(row.latency_p50_seconds),
                    _number(row.output_tokens_per_second_mean),
                    _number(row.peak_vram_gib),
                    _number(row.idle_adjusted_energy_wh_per_request),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "`*` Idle-adjusted GPU board energy. Open-ended quality remains blank "
            "until an external blinded judge is applied; latency and energy "
            "remain valid.",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _required_string(value: dict[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"missing required string {key!r}")
    return item


def _required_int(value: dict[str, object], key: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        raise ValueError(f"missing required integer {key!r}")
    return item


def _required_csv_int(record: dict[str, str], key: str) -> int:
    return _csv_int(record.get(key, ""), key)


def _csv_int(value: str, key: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"invalid integer {key!r}: {value!r}") from exc


def _csv_float(value: str, key: str) -> float | None:
    if value == "":
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"invalid number {key!r}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"non-finite number {key!r}: {value!r}")
    return number


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _latex(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
