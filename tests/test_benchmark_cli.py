"""Tests for benchmark command-line orchestration and offline validation."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import asyncio
from io import StringIO
import json
from pathlib import Path
import unittest

from src.benchmark.cli import _execute_and_close_backend, main


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkCliTests(unittest.TestCase):
    def test_async_backend_closes_on_request_event_loop(self) -> None:
        class Executor:
            loop = None

            async def arun(self, samples):
                self.loop = asyncio.get_running_loop()
                return list(samples)

        class Backend:
            loop = None

            async def aclose(self):
                self.loop = asyncio.get_running_loop()

        executor = Executor()
        backend = Backend()
        result = asyncio.run(
            _execute_and_close_backend(
                executor=executor,
                samples=["sample"],
                backend=backend,
            )
        )

        self.assertEqual(result, ["sample"])
        self.assertIs(executor.loop, backend.loop)

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
        self.assertIn("open_ended_diagnosis", stdout.getvalue())

    def test_open_ended_dry_run_uses_free_text_contract(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(
                [
                    "run",
                    "--model",
                    "gpt_5_6_luna",
                    "--benchmark",
                    "open_ended_diagnosis",
                    "--evaluation-set",
                    "validation",
                    "--limit",
                    "1",
                    "--dry-run",
                ],
                root=ROOT,
            )
        self.assertEqual(status, 0, stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "dry_run_valid")
        self.assertEqual(payload["selected_tasks"], 1)
        self.assertFalse(payload["network_or_model_called"])

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
