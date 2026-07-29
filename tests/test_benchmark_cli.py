"""Tests for benchmark command-line orchestration and offline validation."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import unittest

from src.benchmark.cli import main


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkCliTests(unittest.TestCase):
    def test_list_commands_load_all_typed_configs(self) -> None:
        stdout = StringIO()
        with redirect_stdout(stdout):
            status = main(["list-models"], root=ROOT)
        self.assertEqual(status, 0)
        self.assertIn("qwen_3_5_4b", stdout.getvalue())
        self.assertIn("gpt_5_6_luna", stdout.getvalue())

        stdout = StringIO()
        with redirect_stdout(stdout):
            status = main(["list-benchmarks"], root=ROOT)
        self.assertEqual(status, 0)
        self.assertIn("visual_top_k_closed_set", stdout.getvalue())
        self.assertIn("evidence_grounded_diagnosis", stdout.getvalue())

    def test_dry_run_reads_an_image_without_calling_a_model(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(
                [
                    "run",
                    "--model",
                    "qwen_3_5_4b",
                    "--benchmark",
                    "visual_top_k_closed_set",
                    "--limit",
                    "1",
                    "--dry-run",
                ],
                root=ROOT,
            )
        self.assertEqual(status, 0, stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "dry_run_valid")
        self.assertEqual(payload["selected_units"], 1)
        self.assertEqual(payload["selected_tasks"], 1)
        self.assertFalse(payload["network_or_model_called"])

    def test_confusion_limit_counts_pairs_not_individual_tasks(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(
                [
                    "run",
                    "--model",
                    "qwen_3_5_4b",
                    "--benchmark",
                    "visual_disease_confusion_sets",
                    "--limit",
                    "1",
                    "--dry-run",
                ],
                root=ROOT,
            )
        self.assertEqual(status, 0, stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["selected_units"], 1)
        self.assertEqual(payload["selected_tasks"], 2)

    def test_dry_run_rejects_managed_mode_for_api_model(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(
                [
                    "run",
                    "--model",
                    "gpt_5_6_luna",
                    "--benchmark",
                    "evidence_grounded_diagnosis",
                    "--limit",
                    "1",
                    "--server-mode",
                    "managed",
                    "--dry-run",
                ],
                root=ROOT,
            )
        self.assertEqual(status, 2)
        self.assertIn(
            "managed is available only for local vLLM",
            stderr.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
