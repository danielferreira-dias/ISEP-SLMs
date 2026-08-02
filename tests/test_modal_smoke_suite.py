"""Tests for shared Modal smoke-suite selection semantics."""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from src.modal._shared import (
    run_benchmark,
    smoke_runs,
    structured_output_modes,
)


class ModalSmokeSuiteTests(unittest.TestCase):
    def test_validation_suite_covers_four_protocols_with_exact_counts(self) -> None:
        runs = smoke_runs(
            benchmark="ignored",
            evaluation_set="ignored",
            limit=10,
            all_benchmarks=False,
            validation_suite=True,
        )

        self.assertEqual(
            [run.benchmark for run in runs],
            [
                "visual_top_k_closed_set",
                "visual_disease_confusion_sets",
                "evidence_grounded_diagnosis",
                "open_ended_diagnosis",
            ],
        )
        self.assertEqual(
            [run.evaluation_set for run in runs],
            ["validation"] * 4,
        )
        self.assertEqual(
            [run.expected_task_count for run in runs],
            [10, 10, 10, 10],
        )
        self.assertEqual(runs[1].selection_limit, 5)

    def test_all_benchmarks_produce_exact_task_count(self) -> None:
        runs = smoke_runs(
            benchmark="ignored",
            evaluation_set="ignored",
            limit=10,
            all_benchmarks=True,
        )

        self.assertEqual(len(runs), 3)
        self.assertEqual(
            [run.expected_task_count for run in runs],
            [10, 10, 10],
        )
        confusion = runs[1]
        self.assertEqual(
            confusion.benchmark,
            "visual_disease_confusion_sets",
        )
        self.assertEqual(confusion.selection_limit, 5)

    def test_teacher_screening_uses_fixed_task_id_files(self) -> None:
        runs = smoke_runs(
            benchmark="ignored",
            evaluation_set="ignored",
            limit=10,
            all_benchmarks=False,
            teacher_screening=True,
        )

        self.assertEqual(len(runs), 4)
        self.assertEqual(
            [run.expected_task_count for run in runs],
            [100, 200, 100, 100],
        )
        self.assertTrue(
            all(run.task_ids_file is not None for run in runs)
        )
        self.assertIn("100_pairs", runs[1].task_ids_file or "")

    def test_all_benchmarks_rejects_odd_task_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be even"):
            smoke_runs(
                benchmark="ignored",
                evaluation_set="ignored",
                limit=9,
                all_benchmarks=True,
            )

    def test_single_confusion_limit_still_means_pair_count(self) -> None:
        run = smoke_runs(
            benchmark="visual_disease_confusion_sets",
            evaluation_set="paired_confusion_tasks",
            limit=7,
            all_benchmarks=False,
        )[0]

        self.assertEqual(run.selection_limit, 7)
        self.assertEqual(run.expected_task_count, 14)

    def test_evidence_and_top_k_selects_exactly_two_benchmarks(self) -> None:
        runs = smoke_runs(
            benchmark="ignored",
            evaluation_set="ignored",
            limit=10,
            all_benchmarks=False,
            evidence_and_top_k=True,
        )

        self.assertEqual(
            [run.benchmark for run in runs],
            [
                "visual_top_k_closed_set",
                "evidence_grounded_diagnosis",
            ],
        )
        self.assertEqual(
            [run.expected_task_count for run in runs],
            [10, 10],
        )

    def test_structured_output_both_expands_to_separate_modes(self) -> None:
        self.assertEqual(
            structured_output_modes("both"),
            ("prompt_only", "json_schema"),
        )

    def test_run_benchmark_passes_explicit_structured_output_mode(self) -> None:
        run = smoke_runs(
            benchmark="visual_top_k_closed_set",
            evaluation_set="internal_benchmark_1000",
            limit=10,
            all_benchmarks=False,
        )[0]

        with patch("src.modal._shared.subprocess.run") as subprocess_run:
            run_benchmark(
                project_root=Path("/project"),
                model_config_id="model",
                model_id="provider/model",
                run=run,
                seed=42,
                batch_size=4,
                reasoning_capture="available",
                structured_output="json_schema",
                output_root=None,
                dry_run=True,
                server_url=None,
                thinking_mode="disabled",
            )

        command = subprocess_run.call_args.args[0]
        index = command.index("--structured-output")
        self.assertEqual(command[index + 1], "json_schema")
        thinking_index = command.index("--thinking-mode")
        self.assertEqual(command[thinking_index + 1], "disabled")

    def test_run_benchmark_uses_task_ids_instead_of_limit(self) -> None:
        run = smoke_runs(
            benchmark="ignored",
            evaluation_set="ignored",
            limit=100,
            all_benchmarks=False,
            teacher_screening=True,
        )[0]

        with patch("src.modal._shared.subprocess.run") as subprocess_run:
            run_benchmark(
                project_root=Path("/project"),
                model_config_id="model",
                model_id="provider/model",
                run=run,
                seed=42,
                batch_size=4,
                reasoning_capture="available",
                structured_output="prompt_only",
                output_root=None,
                dry_run=True,
                server_url=None,
                thinking_mode="disabled",
            )

        command = subprocess_run.call_args.args[0]
        self.assertIn("--task-ids-file", command)
        self.assertNotIn("--limit", command)

    def test_run_benchmark_passes_max_output_token_override(self) -> None:
        run = smoke_runs(
            benchmark="visual_top_k_closed_set",
            evaluation_set="validation",
            limit=10,
            all_benchmarks=False,
        )[0]

        with patch("src.modal._shared.subprocess.run") as subprocess_run:
            run_benchmark(
                project_root=Path("/project"),
                model_config_id="model",
                model_id="provider/model",
                run=run,
                seed=42,
                batch_size=4,
                reasoning_capture="available",
                structured_output="prompt_only",
                output_root=None,
                dry_run=True,
                server_url=None,
                thinking_mode="enabled",
                max_output_tokens=10_240,
            )

        command = subprocess_run.call_args.args[0]
        index = command.index("--max-output-tokens")
        self.assertEqual(command[index + 1], "10240")


if __name__ == "__main__":
    unittest.main()
