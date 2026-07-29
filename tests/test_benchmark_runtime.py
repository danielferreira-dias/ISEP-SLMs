"""Tests for deterministic selection and durable benchmark results."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
import yaml

from src.benchmark.datasets import load_benchmark_dataset
from src.benchmark.executor import BenchmarkExecutor, ExecutionConfig
from src.benchmark.results import (
    RunPaths,
    RunWriter,
    canonical_hash,
    read_jsonl,
)
from src.benchmark.runner import (
    BenchmarkSample,
    parse_and_validate_response,
)
from src.benchmark.selection import select_units, task_seed
from src.inference.base import (
    InferenceBackend,
    InferenceRequest,
    InferenceResult,
    ReasoningTrace,
    TokenUsage,
)


ROOT = Path(__file__).resolve().parents[1]


class DeterministicSelectionTests(unittest.TestCase):
    def test_selection_is_model_independent_and_stable(self) -> None:
        frame = pd.DataFrame(
            {
                "sample_id": [f"S{index}" for index in range(10)],
                "task_id": [f"T{index}" for index in range(10)],
            }
        )
        first, first_manifest = select_units(
            frame,
            unit_column="sample_id",
            task_column="task_id",
            limit=4,
            seed=42,
            benchmark_release_hash="release",
        )
        second, second_manifest = select_units(
            frame.sample(frac=1.0, random_state=7),
            unit_column="sample_id",
            task_column="task_id",
            limit=4,
            seed=42,
            benchmark_release_hash="release",
        )

        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first["task_id"].tolist(), second["task_id"].tolist())

    def test_confusion_limit_preserves_both_tasks_in_each_pair(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "pair_id": f"P{pair}",
                    "task_id": f"P{pair}::{condition}",
                }
                for pair in range(5)
                for condition in ("low", "high")
            ]
        )
        selected, manifest = select_units(
            frame,
            unit_column="pair_id",
            task_column="task_id",
            limit=2,
            seed=42,
            benchmark_release_hash="release",
        )

        self.assertEqual(manifest["selected_unit_count"], 2)
        self.assertEqual(manifest["selected_task_count"], 4)
        self.assertTrue((selected.groupby("pair_id").size() == 2).all())

    def test_task_seed_does_not_depend_on_execution_order(self) -> None:
        self.assertEqual(task_seed(42, "TASK_A"), task_seed(42, "TASK_A"))
        self.assertNotEqual(task_seed(42, "TASK_A"), task_seed(42, "TASK_B"))

    def test_repository_manifests_use_the_expected_selection_units(self) -> None:
        cases = [
            (
                "visual_top_k.yaml",
                "internal_benchmark_1000",
                2,
            ),
            ("visual_confusion_sets.yaml", None, 4),
            ("evidence_grounded_diagnosis.yaml", None, 2),
        ]
        for filename, evaluation_set, expected_tasks in cases:
            with self.subTest(filename=filename):
                config = yaml.safe_load(
                    (
                        ROOT / "configs/benchmarks" / filename
                    ).read_text(encoding="utf-8")
                )
                loaded = load_benchmark_dataset(
                    root=ROOT,
                    config=config,
                    evaluation_set=evaluation_set,
                    limit=2,
                    seed=42,
                )
                self.assertEqual(len(loaded.samples), expected_tasks)


class RunWriterTests(unittest.TestCase):
    def test_resume_skips_terminal_ids_and_rejects_hash_mismatch(self) -> None:
        with TemporaryDirectory() as temporary:
            paths = RunPaths.from_directory(Path(temporary) / "run")
            identity = {"run_hash": canonical_hash({"model": "test"})}
            writer = RunWriter(paths, identity=identity, resume=False)
            writer.initialize(
                manifest={"model_id": "test"},
                config_snapshot={"model": {"id": "test"}},
                selection={"task_ids": ["T1"]},
                environment={"python": "test"},
            )
            writer.append_prediction(
                {
                    "task_id": "T1",
                    "status": "invalid_output",
                }
            )

            resumed = RunWriter(paths, identity=identity, resume=True)
            self.assertEqual(resumed.completed_task_ids(), {"T1"})
            resumed.initialize(
                manifest={},
                config_snapshot={},
                selection={},
                environment={},
            )
            manifest = yaml.safe_load(
                paths.manifest.read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "running")
            self.assertEqual(
                manifest["resume_history"][0]["previous_status"],
                "running",
            )
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                RunWriter(
                    paths,
                    identity={"run_hash": "different"},
                    resume=True,
                )

    def test_reader_tolerates_only_a_truncated_final_record(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.jsonl"
            path.write_bytes(b'{"task_id":"T1"}\n{"task_id":')
            self.assertEqual(read_jsonl(path), [{"task_id": "T1"}])

            path.write_bytes(
                b'{"task_id":\n{"task_id":"T2"}\n'
            )
            with self.assertRaisesRegex(ValueError, "Invalid JSONL"):
                read_jsonl(path)

            path.write_bytes(b'{"task_id":"T1"}\nnot-json\n')
            with self.assertRaisesRegex(ValueError, "Invalid JSONL"):
                read_jsonl(path)


class BenchmarkExecutorTests(unittest.TestCase):
    def test_reasoning_is_saved_but_only_final_text_is_parsed(self) -> None:
        class Prepared:
            system_prompt = "system"
            user_prompt = "user"
            schema = {}

        class Adapter:
            def prepare(self, sample):
                return Prepared()

            def parse_response(
                self,
                model_id,
                raw_text,
                prepared_task,
                reasoning_text=None,
            ):
                del prepared_task
                self.reasoning_text = reasoning_text
                return parse_and_validate_response(
                    model_id=model_id,
                    raw_text=raw_text,
                    allowed_disease_ids={"D001"},
                    top_k=1,
                )

            def compute_metrics(self, predictions):
                return {"sample_count": len(list(predictions))}

        class Backend(InferenceBackend):
            model_id = "test"

            def complete(self, request: InferenceRequest) -> InferenceResult:
                return InferenceResult(
                    model_id=self.model_id,
                    final_text=(
                        '{"predictions":[{"rank":1,'
                        '"disease_id":"D001"}]}'
                    ),
                    reasoning=ReasoningTrace(
                        capture_mode="full",
                        text="Private model-provided reasoning.",
                        token_count=5,
                        source_field="reasoning",
                    ),
                    usage=TokenUsage(output_tokens=20, reasoning_tokens=5),
                    request_id=request.request_id,
                    finish_reason="stop",
                )

        with TemporaryDirectory() as temporary:
            paths = RunPaths.from_directory(Path(temporary) / "run")
            writer = RunWriter(
                paths,
                identity={"run_hash": "test"},
                resume=False,
            )
            writer.initialize(
                manifest={},
                config_snapshot={},
                selection={},
                environment={},
            )
            adapter = Adapter()
            summary = BenchmarkExecutor(
                backend=Backend(),
                adapter=adapter,
                image_loader=lambda _: b"\xff\xd8\xffimage",
                writer=writer,
                execution=ExecutionConfig(
                    batch_size=1,
                    max_output_tokens=4096,
                ),
            ).run(
                [
                    BenchmarkSample(
                        sample_id="S1",
                        task_id="T1",
                        image_uri="image.jpg",
                        disease_id="D001",
                        metadata={},
                    )
                ]
            )
            record = read_jsonl(paths.predictions)[0]

            self.assertEqual(summary.counts["ok"], 1)
            self.assertEqual(
                adapter.reasoning_text,
                "Private model-provided reasoning.",
            )
            self.assertEqual(
                record["response"]["reasoning"]["text"],
                "Private model-provided reasoning.",
            )
            self.assertNotIn(
                "Private model-provided reasoning.",
                record["response"]["final_text"],
            )


if __name__ == "__main__":
    unittest.main()
