"""Tests for inference telemetry persistence and thesis efficiency reports."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from scripts.run_efficiency_cohort import _write_server_model_manifest
from src.benchmark.efficiency import (
    BenchmarkResourceSample,
    _load_existing_samples,
)
from src.benchmark.efficiency_comparison import (
    BENCHMARKS,
    EfficiencyComparisonRow,
    build_efficiency_comparison,
    pareto_points,
)
from src.benchmark.efficiency_report import build_efficiency_report
from src.inference.openai_compatible import _timing_metadata


class BenchmarkEfficiencyTest(unittest.TestCase):
    """Exercise multi-session telemetry and model-agnostic report discovery."""

    def test_existing_telemetry_is_loaded_for_resumed_campaign(self) -> None:
        """A second suite must retain samples written by the first suite."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nvml_samples.jsonl"
            sample = BenchmarkResourceSample(
                timestamp_utc="2026-08-14T22:00:00+00:00",
                monotonic_seconds=10.0,
                phase="visual_top_k_closed_set",
                gpu_utilization_percent=95.0,
                gpu_memory_used_bytes=1024,
                gpu_memory_total_bytes=2048,
                gpu_power_watts=300.0,
                gpu_temperature_celsius=55.0,
                system_memory_used_bytes=4096,
                system_memory_available_bytes=8192,
                server_process_rss_bytes=512,
                server_process_cpu_percent=25.0,
            )
            path.write_text(json.dumps(asdict(sample)) + "\n", encoding="utf-8")

            self.assertEqual(_load_existing_samples(path), [sample])

    def test_stream_timing_uses_decode_intervals_consistently(self) -> None:
        """Twenty generated tokens contain nineteen inter-token intervals."""

        timing = _timing_metadata(
            started_utc="2026-08-14T22:00:00+00:00",
            completed_utc="2026-08-14T22:00:02+00:00",
            total_seconds=2.0,
            ttft_seconds=0.5,
            output_tokens=20,
            streamed=True,
            stream_chunk_count=12,
        )

        self.assertAlmostEqual(timing["mean_time_per_output_token_seconds"], 1.5 / 19)
        self.assertAlmostEqual(timing["output_tokens_per_second"], 19 / 1.5)

    def test_report_discovers_arbitrary_model_id_and_tpot_percentiles(self) -> None:
        """The five-model cohort cannot depend on the Qwen 3.8 directory name."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "visual_top_k_closed_set" / "custom_model" / "run-1"
            run.mkdir(parents=True)
            prediction = {
                "task_id": "task-1",
                "sample_id": "sample-1",
                "status": "ok",
                "response": {
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "total_tokens": 120,
                    },
                    "provider_metadata": {
                        "timing": {
                            "request_started_utc": "2026-08-14T22:00:00+00:00",
                            "request_completed_utc": "2026-08-14T22:00:02+00:00",
                            "end_to_end_latency_seconds": 2.0,
                            "time_to_first_token_seconds": 0.5,
                            "mean_time_per_output_token_seconds": 0.075,
                            "output_tokens_per_second": 13.3333333333,
                        }
                    },
                },
            }
            (run / "predictions.jsonl").write_text(
                json.dumps(prediction) + "\n", encoding="utf-8"
            )
            (run / "metrics.json").write_text(
                json.dumps({"top_1_accuracy": 1.0}) + "\n", encoding="utf-8"
            )
            metrics = root / "metrics"
            metrics.mkdir()
            (metrics / "resource_efficiency_summary.json").write_text(
                json.dumps(
                    {
                        "by_phase": {
                            "visual_top_k_closed_set": {
                                "duration_seconds": 2.0,
                                "energy_wh": 0.2,
                                "peak_gpu_memory_bytes": 2 * 1024**3,
                                "peak_server_process_rss_bytes": 1024**3,
                            }
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = build_efficiency_report(
                root,
                model_id="custom_model",
                model="Custom model",
                parameters_billions=4.0,
                dtype="BF16",
                hardware="Test GPU",
            )

            self.assertEqual(summary["model_id"], "custom_model")
            benchmark = summary["by_benchmark"]["visual_top_k_closed_set"]
            self.assertEqual(benchmark["tpot_p99_seconds"], 0.075)
            self.assertEqual(benchmark["ttft_p99_seconds"], 0.5)
            with (root / "tables" / "quality_efficiency.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["tpot_p95_seconds"], "0.075")

    def test_comparison_builds_pareto_tables_and_figures(self) -> None:
        """Controlled model runs produce thesis tables and plot source data."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_ids = ("small", "large")
            for model_id, quality, latency, energy, vram, parameters in (
                ("small", 0.80, 0.2, 0.01, 12.0, 4.0),
                ("large", 0.84, 1.0, 0.08, 60.0, 27.0),
            ):
                run = root / model_id
                (run / "metrics").mkdir(parents=True)
                (run / "tables").mkdir()
                (run / "campaign_status.json").write_text(
                    json.dumps({"status": "completed"}) + "\n", encoding="utf-8"
                )
                (run / "metrics" / "efficiency_summary.json").write_text(
                    json.dumps(
                        {
                            "model_id": model_id,
                            "request_count": 400,
                            "cohort_manifest_sha256": "a" * 64,
                            "hardware": "Test GPU",
                            "concurrency": 1,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                rows = tuple(
                    _comparison_row(
                        model_id=model_id,
                        benchmark_id=benchmark,
                        quality=(
                            None if benchmark == "open_ended_diagnosis" else quality
                        ),
                        latency=latency,
                        energy=energy,
                        vram=vram,
                        parameters=parameters,
                    )
                    for benchmark in BENCHMARKS
                )
                records = [
                    {
                        key: value
                        for key, value in asdict(row).items()
                        if key != "model_id"
                    }
                    for row in rows
                ]
                with (run / "tables" / "quality_efficiency.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as stream:
                    writer = csv.DictWriter(stream, fieldnames=tuple(records[0]))
                    writer.writeheader()
                    writer.writerows(records)

            summary = build_efficiency_comparison(root, model_ids=model_ids)

            self.assertEqual(summary["model_count"], 2)
            comparison = root / "comparison"
            self.assertTrue(
                (comparison / "tables/same_hardware_comparison.csv").is_file()
            )
            self.assertTrue((comparison / "tables/pareto_frontiers.csv").is_file())
            self.assertTrue((comparison / "figures/quality_vs_latency.png").is_file())
            self.assertTrue((comparison / "figures/quality_vs_latency.svg").is_file())
            self.assertTrue((comparison / "report/thesis_summary.md").is_file())

    def test_pareto_marks_tradeoffs_and_dominated_points(self) -> None:
        """A faster-weaker tradeoff survives while a worse duplicate is dominated."""

        rows = (
            _comparison_row("fast", "visual_top_k_closed_set", 0.80, 0.2),
            _comparison_row("strong", "visual_top_k_closed_set", 0.90, 1.0),
            _comparison_row("dominated", "visual_top_k_closed_set", 0.70, 1.2),
        )

        points = pareto_points(rows, x_metric="latency_p50_seconds")
        frontier = {point.model_id for point in points if point.on_frontier}

        self.assertEqual(frontier, {"fast", "strong"})

    def test_local_server_model_manifest_hashes_only_model_files(self) -> None:
        """Private weights are pinned without recording Hub cache internals."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            hidden = model / ".cache"
            hidden.mkdir()
            (hidden / "download.lock").write_text("", encoding="utf-8")
            manifest_path = root / "server_model_files.json"

            _write_server_model_manifest(model, manifest_path)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["file_count"], 2)
            self.assertEqual(
                [item["path"] for item in manifest["files"]],
                ["config.json", "model.safetensors"],
            )
            self.assertTrue(
                all(len(item["sha256"]) == 64 for item in manifest["files"])
            )


def _comparison_row(
    model_id: str,
    benchmark_id: str,
    quality: float | None,
    latency: float,
    energy: float = 0.01,
    vram: float = 12.0,
    parameters: float = 4.0,
) -> EfficiencyComparisonRow:
    """Construct a fully typed toy efficiency row with neutral optional fields."""

    return EfficiencyComparisonRow(
        model_id=model_id,
        model=model_id.title(),
        parameters_billions=parameters,
        dtype="BF16",
        hardware="Test GPU",
        benchmark_id=benchmark_id,
        request_count=100,
        quality_metric=("canonical_top_1_accuracy" if quality is not None else None),
        quality_value=quality,
        ttft_p50_seconds=latency / 2,
        ttft_p95_seconds=latency / 2,
        ttft_p99_seconds=latency / 2,
        latency_p50_seconds=latency,
        latency_p95_seconds=latency,
        latency_p99_seconds=latency,
        tpot_p50_seconds=0.01,
        tpot_p95_seconds=0.01,
        tpot_p99_seconds=0.01,
        output_tokens_per_second_mean=100.0,
        output_tokens_per_second_p50=100.0,
        requests_per_second=1.0 / latency,
        gpu_seconds_per_request=latency,
        peak_vram_gib=vram,
        peak_server_ram_gib=10.0,
        gpu_energy_wh=energy * 100,
        idle_adjusted_gpu_energy_wh=energy * 90,
        energy_wh_per_request=energy,
        idle_adjusted_energy_wh_per_request=energy * 0.9,
        energy_wh_per_correct=(energy / quality if quality else None),
        idle_adjusted_energy_wh_per_correct=(
            energy * 0.9 / quality if quality else None
        ),
        estimated_gpu_cost_usd=None,
        estimated_gpu_cost_usd_per_request=None,
        estimated_gpu_cost_usd_per_correct=None,
    )


if __name__ == "__main__":
    unittest.main()
