"""Materialize thesis-ready inference efficiency tables from benchmark runs."""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

_REQUEST_COLUMNS = (
    "benchmark_id",
    "task_id",
    "sample_id",
    "status",
    "request_started_utc",
    "request_completed_utc",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "end_to_end_latency_seconds",
    "time_to_first_token_seconds",
    "mean_time_per_output_token_seconds",
    "output_tokens_per_second",
)

_TABLE_COLUMNS = (
    "model",
    "parameters_billions",
    "dtype",
    "hardware",
    "benchmark_id",
    "request_count",
    "quality_metric",
    "quality_value",
    "ttft_p50_seconds",
    "ttft_p95_seconds",
    "ttft_p99_seconds",
    "latency_p50_seconds",
    "latency_p95_seconds",
    "latency_p99_seconds",
    "tpot_p50_seconds",
    "tpot_p95_seconds",
    "tpot_p99_seconds",
    "output_tokens_per_second_mean",
    "output_tokens_per_second_p50",
    "requests_per_second",
    "gpu_seconds_per_request",
    "peak_vram_gib",
    "peak_server_ram_gib",
    "peak_system_memory_used_gib",
    "mean_gpu_utilization_percent",
    "mean_gpu_power_watts",
    "peak_gpu_power_watts",
    "peak_gpu_temperature_celsius",
    "gpu_energy_wh",
    "idle_adjusted_gpu_energy_wh",
    "energy_wh_per_request",
    "idle_adjusted_energy_wh_per_request",
    "energy_wh_per_correct",
    "idle_adjusted_energy_wh_per_correct",
    "estimated_gpu_cost_usd",
    "estimated_gpu_cost_usd_per_request",
    "estimated_gpu_cost_usd_per_correct",
)


def build_efficiency_report(
    output_root: Path,
    *,
    model_id: str,
    model: str,
    parameters_billions: float,
    dtype: str,
    hardware: str,
    concurrency: int | None = None,
    gpu_hourly_cost_usd: float | None = None,
    cohort_manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Create per-request Parquet and aggregate CSV/LaTeX/JSON artifacts."""

    rows, metrics_by_benchmark = _load_request_rows(output_root, model_id)
    if not rows:
        raise RuntimeError("no prediction timing records were found")
    resource_path = output_root / "metrics" / "resource_efficiency_summary.json"
    resource = _read_json(resource_path)
    by_phase = _mapping(resource.get("by_phase"))
    idle_phase = _mapping(by_phase.get("idle_baseline"))
    idle_power_watts = _float_or_none(idle_phase.get("mean_gpu_power_watts"))

    metrics_directory = output_root / "metrics"
    tables_directory = output_root / "tables"
    metrics_directory.mkdir(parents=True, exist_ok=True)
    tables_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(metrics_directory / "per_request_efficiency.csv", rows, _REQUEST_COLUMNS)
    _write_parquet(
        metrics_directory / "per_request_efficiency.parquet",
        rows,
        _REQUEST_COLUMNS,
    )

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["benchmark_id"]), []).append(row)
    table_rows: list[dict[str, object]] = []
    summaries: dict[str, object] = {}
    for benchmark_id, benchmark_rows in sorted(grouped.items()):
        timing = _timing_summary(benchmark_rows)
        phase = _mapping(by_phase.get(benchmark_id))
        quality_name, quality_value = _primary_quality(
            metrics_by_benchmark.get(benchmark_id, {})
        )
        request_count = len(benchmark_rows)
        duration = _float_or_none(phase.get("duration_seconds"))
        energy = _float_or_none(phase.get("energy_wh"))
        idle_energy = (
            idle_power_watts * duration / 3600.0
            if idle_power_watts is not None and duration is not None
            else None
        )
        adjusted_energy = (
            max(0.0, energy - idle_energy)
            if energy is not None and idle_energy is not None
            else None
        )
        correct_count = (
            request_count * quality_value
            if quality_name
            in {"accuracy", "top_1_accuracy", "canonical_top_1_accuracy"}
            and quality_value is not None
            else None
        )
        row = {
            "model": model,
            "parameters_billions": parameters_billions,
            "dtype": dtype,
            "hardware": hardware,
            "benchmark_id": benchmark_id,
            "request_count": request_count,
            "quality_metric": quality_name,
            "quality_value": quality_value,
            **timing,
            "requests_per_second": (
                request_count / duration if duration and duration > 0 else None
            ),
            "gpu_seconds_per_request": (
                duration / request_count if duration is not None else None
            ),
            "peak_vram_gib": _gib(phase.get("peak_gpu_memory_bytes")),
            "peak_server_ram_gib": _gib(phase.get("peak_server_process_rss_bytes")),
            "peak_system_memory_used_gib": _gib(
                phase.get("peak_system_memory_used_bytes")
            ),
            "mean_gpu_utilization_percent": _float_or_none(
                phase.get("mean_gpu_utilization_percent")
            ),
            "mean_gpu_power_watts": _float_or_none(phase.get("mean_gpu_power_watts")),
            "peak_gpu_power_watts": _float_or_none(phase.get("peak_gpu_power_watts")),
            "peak_gpu_temperature_celsius": _float_or_none(
                phase.get("peak_gpu_temperature_celsius")
            ),
            "gpu_energy_wh": energy,
            "idle_adjusted_gpu_energy_wh": adjusted_energy,
            "energy_wh_per_request": (
                energy / request_count if energy is not None else None
            ),
            "idle_adjusted_energy_wh_per_request": (
                adjusted_energy / request_count if adjusted_energy is not None else None
            ),
            "energy_wh_per_correct": (
                energy / correct_count
                if energy is not None and correct_count and correct_count > 0
                else None
            ),
            "idle_adjusted_energy_wh_per_correct": (
                adjusted_energy / correct_count
                if adjusted_energy is not None and correct_count and correct_count > 0
                else None
            ),
            "estimated_gpu_cost_usd": (
                duration * gpu_hourly_cost_usd / 3600.0
                if duration is not None and gpu_hourly_cost_usd is not None
                else None
            ),
            "estimated_gpu_cost_usd_per_request": (
                duration * gpu_hourly_cost_usd / 3600.0 / request_count
                if duration is not None and gpu_hourly_cost_usd is not None
                else None
            ),
            "estimated_gpu_cost_usd_per_correct": (
                duration * gpu_hourly_cost_usd / 3600.0 / correct_count
                if duration is not None
                and gpu_hourly_cost_usd is not None
                and correct_count
                and correct_count > 0
                else None
            ),
        }
        table_rows.append(row)
        summaries[benchmark_id] = row

    _write_csv(tables_directory / "quality_efficiency.csv", table_rows, _TABLE_COLUMNS)
    _write_latex(tables_directory / "quality_efficiency.tex", table_rows)
    result = {
        "schema_version": 1,
        "model": model,
        "model_id": model_id,
        "parameters_billions": parameters_billions,
        "dtype": dtype,
        "hardware": hardware,
        "concurrency": concurrency,
        "gpu_hourly_cost_usd": gpu_hourly_cost_usd,
        "cohort_manifest_sha256": cohort_manifest_sha256,
        "request_count": len(rows),
        "timing_semantics": {
            "ttft": "client-observed time to first non-empty streamed delta",
            "tpot": (
                "decode window divided by provider-reported output-token intervals "
                "(max(output_tokens - 1, 1)); stream chunks are not assumed to be "
                "individual tokens"
            ),
            "energy": "raw GPU board energy integrated from NVML power samples",
            "idle_adjusted_energy": (
                "raw GPU board energy minus mean idle-baseline board power "
                "multiplied by task duration, floored at zero"
            ),
        },
        "by_benchmark": summaries,
    }
    _atomic_json(metrics_directory / "efficiency_summary.json", result)
    return result


def _load_request_rows(
    output_root: Path, model_id: str
) -> tuple[list[dict[str, object]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, object]] = []
    metrics: dict[str, dict[str, Any]] = {}
    for path in sorted(output_root.glob(f"*/{model_id}/*/predictions.jsonl")):
        benchmark_id = path.relative_to(output_root).parts[0]
        metrics_path = path.with_name("metrics.json")
        if metrics_path.is_file():
            metrics[benchmark_id] = _read_json(metrics_path)
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            response = _mapping(record.get("response"))
            provider = _mapping(response.get("provider_metadata"))
            timing = _mapping(provider.get("timing"))
            usage = _mapping(response.get("usage"))
            rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "task_id": str(record.get("task_id", "")),
                    "sample_id": str(record.get("sample_id", "")),
                    "status": str(record.get("status", "")),
                    "request_started_utc": timing.get("request_started_utc"),
                    "request_completed_utc": timing.get("request_completed_utc"),
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "end_to_end_latency_seconds": timing.get(
                        "end_to_end_latency_seconds"
                    ),
                    "time_to_first_token_seconds": timing.get(
                        "time_to_first_token_seconds"
                    ),
                    "mean_time_per_output_token_seconds": timing.get(
                        "mean_time_per_output_token_seconds"
                    ),
                    "output_tokens_per_second": timing.get("output_tokens_per_second"),
                }
            )
    return rows, metrics


def _timing_summary(rows: list[dict[str, object]]) -> dict[str, float | None]:
    ttft = _finite(rows, "time_to_first_token_seconds")
    latency = _finite(rows, "end_to_end_latency_seconds")
    tpot = _finite(rows, "mean_time_per_output_token_seconds")
    token_rate = _finite(rows, "output_tokens_per_second")
    return {
        "ttft_p50_seconds": _quantile(ttft, 0.50),
        "ttft_p95_seconds": _quantile(ttft, 0.95),
        "ttft_p99_seconds": _quantile(ttft, 0.99),
        "latency_p50_seconds": _quantile(latency, 0.50),
        "latency_p95_seconds": _quantile(latency, 0.95),
        "latency_p99_seconds": _quantile(latency, 0.99),
        "tpot_p50_seconds": _quantile(tpot, 0.50),
        "tpot_p95_seconds": _quantile(tpot, 0.95),
        "tpot_p99_seconds": _quantile(tpot, 0.99),
        "output_tokens_per_second_mean": (
            sum(token_rate) / len(token_rate) if token_rate else None
        ),
        "output_tokens_per_second_p50": _quantile(token_rate, 0.50),
    }


def _primary_quality(metrics: dict[str, Any]) -> tuple[str | None, float | None]:
    for key in ("canonical_top_1_accuracy", "top_1_accuracy", "accuracy"):
        value = _float_or_none(metrics.get(key))
        if value is not None:
            return key, value
    context = _mapping(metrics.get("context_minus_image_only"))
    value = _float_or_none(context.get("canonical_top_1_accuracy"))
    if value is not None:
        return "context_top_1_delta", value
    return None, None


def _finite(rows: list[dict[str, object]], key: str) -> list[float]:
    values = [_float_or_none(row.get(key)) for row in rows]
    return sorted(value for value in values if value is not None)


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    columns: tuple[str, ...],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_parquet(
    path: Path, rows: list[dict[str, object]], columns: tuple[str, ...]
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(
        [{column: row.get(column) for column in columns} for row in rows]
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_table = cast(Callable[[object, Path], None], pq.write_table)
    write_table(table, temporary)
    os.replace(temporary, path)


def _write_latex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (
            r"Benchmark & Quality & TTFT p50 & Latency p50 & Tok/s "
            r"& VRAM GiB & Wh/query \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        values = (
            str(row["benchmark_id"]).replace("_", r"\_"),
            _format(row.get("quality_value")),
            _format(row.get("ttft_p50_seconds")),
            _format(row.get("latency_p50_seconds")),
            _format(row.get("output_tokens_per_second_mean")),
            _format(row.get("peak_vram_gib")),
            _format(row.get("energy_wh_per_request")),
        )
        lines.append(" & ".join(values) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format(value: object) -> str:
    number = _float_or_none(value)
    return "--" if number is None else f"{number:.4f}"


def _gib(value: object) -> float | None:
    number = _float_or_none(value)
    return number / (1024.0**3) if number is not None else None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
