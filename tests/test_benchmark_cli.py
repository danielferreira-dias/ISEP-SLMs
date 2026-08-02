"""Tests for benchmark command-line orchestration and offline validation."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import asyncio
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from src.benchmark.cli import (
    _execute_and_close_backend,
    _override_thinking_mode,
    main,
)
from src.config import load_model_config


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkCliTests(unittest.TestCase):
    def test_thinking_override_updates_effective_model_without_mutating_yaml(
        self,
    ) -> None:
        model = load_model_config("qwen_3_5_4b", root=ROOT)
        enabled = _override_thinking_mode(model, "enabled")
        disabled = _override_thinking_mode(enabled, "disabled")

        self.assertFalse(model.reasoning.enabled)
        self.assertFalse(
            model.reasoning.chat_template_kwargs.enable_thinking
        )
        self.assertTrue(enabled.reasoning.enabled)
        self.assertTrue(
            enabled.reasoning.chat_template_kwargs.enable_thinking
        )
        self.assertEqual(enabled.generation.thinking_mode, "enabled")
        self.assertFalse(disabled.reasoning.enabled)
        self.assertFalse(
            disabled.reasoning.chat_template_kwargs.enable_thinking
        )
        self.assertEqual(disabled.generation.thinking_mode, "disabled")

    def test_dry_run_records_max_output_token_override(self) -> None:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(
                [
                    "run",
                    "--model",
                    "qwen_3_6_27b",
                    "--benchmark",
                    "visual_top_k_closed_set",
                    "--evaluation-set",
                    "validation",
                    "--limit",
                    "1",
                    "--thinking-mode",
                    "enabled",
                    "--max-output-tokens",
                    "14336",
                    "--dry-run",
                ],
                root=ROOT,
            )

        self.assertEqual(status, 0, stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["max_output_tokens"], 14_336)
        self.assertTrue(payload["thinking"]["reasoning_enabled"])
        self.assertEqual(
            payload["thinking"]["reasoning_max_tokens"],
            10_240,
        )

    def test_disabling_thinking_clears_configured_reasoning_budget(self) -> None:
        model = load_model_config("qwen_3_6_27b", root=ROOT)

        enabled = _override_thinking_mode(model, "enabled")
        disabled = _override_thinking_mode(enabled, "disabled")

        self.assertEqual(enabled.generation.reasoning_max_tokens, 10_240)
        self.assertIsNone(disabled.generation.reasoning_max_tokens)

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

    def test_open_ended_dry_run_accepts_audited_prompt_override(self) -> None:
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
                    "--prompt-override",
                    "src/benchmark/resources/open_ended_diagnosis/"
                    "model_prompt_v1_1_0.yaml",
                    "--dry-run",
                ],
                root=ROOT,
            )
        self.assertEqual(status, 0, stderr.getvalue())
        self.assertEqual(json.loads(stdout.getvalue())["selected_tasks"], 1)

    def test_prompt_override_is_rejected_for_other_benchmarks(self) -> None:
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
                    "--prompt-override",
                    "src/benchmark/resources/open_ended_diagnosis/"
                    "model_prompt_v1_1_0.yaml",
                    "--dry-run",
                ],
                root=ROOT,
            )
        self.assertEqual(status, 2)
        self.assertIn("only for open_ended_diagnosis", stderr.getvalue())

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

    def test_dry_run_reports_effective_thinking_override(self) -> None:
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
                    "--thinking-mode",
                    "enabled",
                    "--dry-run",
                ],
                root=ROOT,
            )
        self.assertEqual(status, 0, stderr.getvalue())
        thinking = json.loads(stdout.getvalue())["thinking"]
        self.assertEqual(thinking["request"], "enabled")
        self.assertTrue(thinking["reasoning_enabled"])
        self.assertTrue(thinking["chat_template_enable_thinking"])
        self.assertEqual(thinking["generation_mode"], "enabled")

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

    def test_explicit_confusion_cohort_retains_pair_unit_semantics(self) -> None:
        cohort = (
            ROOT
            / "data/benchmarks/ISEPDermaBench/metadata/"
            "validation_screening_v1/"
            "visual_confusion_sets_100_pairs.task_ids.txt"
        )
        task_ids = [
            line
            for line in cohort.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        ][:2]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "one_pair.txt"
            path.write_text("\n".join(task_ids) + "\n", encoding="utf-8")
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
                        "--evaluation-set",
                        "validation",
                        "--task-ids-file",
                        str(path),
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
