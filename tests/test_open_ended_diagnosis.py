"""Tests for the open-ended ISEPDermaBench protocol and blinded judge."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pyarrow.parquet as pq
import yaml

from src.benchmark.open_ended_judge import (
    _is_content_policy_violation,
    _judge_one,
    _judge_request,
    _parse_judgment,
    _validate_judgment_semantics,
    compute_judge_metrics,
    judge_run,
)
from src.inference.base import (
    InferenceBackend,
    InferenceResult,
    InferenceSafetyRefusal,
    ReasoningTrace,
)
from src.benchmark.runner import BenchmarkSample
from src.benchmark.task_adapters import OpenEndedDiagnosisTaskAdapter
from src.data_pipeline.open_ended_benchmark import validate_open_ended_release


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "data/benchmarks/ISEPDermaBench"


def _first_row(directory: Path, split: str) -> dict:
    path = sorted(directory.glob(f"{split}-*.parquet"))[0]
    return pq.read_table(path).slice(0, 1).to_pylist()[0]


class _ScriptedBackend(InferenceBackend):
    def __init__(self, outcomes: list[InferenceResult | Exception]) -> None:
        self.outcomes = list(outcomes)
        self.requests = []

    @property
    def model_id(self) -> str:
        return "scripted"

    def complete(self, request):  # pragma: no cover - async path is tested
        raise NotImplementedError

    async def acomplete(self, request):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _result(payload: dict) -> InferenceResult:
    return InferenceResult(
        model_id="scripted",
        final_text=json.dumps(payload),
        reasoning=ReasoningTrace(capture_mode="none"),
    )


def _valid_judgment() -> dict:
    return {
        "reference_diagnosis_rank": 1,
        "diagnosis_correctness": 4,
        "visual_findings_correctness": 3,
        "evidence_grounding": 4,
        "clinical_rationale_quality": 3,
        "differential_quality": 3,
        "unsupported_claim_count": 0,
        "unsupported_claim_examples": [],
        "overall_verdict": "correct",
        "judge_summary": "The diagnosis is ranked first and grounded.",
    }


class OpenEndedDiagnosisTests(unittest.TestCase):
    def test_dermatologist_vision_candidate_is_separate_from_frozen_prompt(self) -> None:
        frozen = yaml.safe_load(
            (
                ROOT
                / "src/benchmark/resources/open_ended_diagnosis/model_prompt.yaml"
            ).read_text(encoding="utf-8")
        )
        candidate = yaml.safe_load(
            (
                ROOT
                / "src/benchmark/resources/open_ended_diagnosis/"
                "model_prompt_v1_3_0_candidate.yaml"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(frozen["version"], "1.1.0")
        self.assertEqual(candidate["version"], "1.3.0-candidate.1")
        self.assertNotEqual(candidate, frozen)
        normalized = " ".join(candidate["system_prompt"].split())
        self.assertIn("global impression", normalized)
        self.assertIn("strongest plausible visual mimic", normalized)

    def test_open_ended_model_prompt_is_frozen_and_embedded(self) -> None:
        resource = yaml.safe_load(
            (
                ROOT
                / "src/benchmark/resources/open_ended_diagnosis/model_prompt.yaml"
            ).read_text(encoding="utf-8")
        )
        artifact = yaml.safe_load(
            (
                RELEASE / "artifacts/prompts/open_ended_diagnosis.yaml"
            ).read_text(encoding="utf-8")
        )
        task = _first_row(
            RELEASE / "tasks/open_ended_diagnosis",
            "validation",
        )
        metadata = json.loads((RELEASE / "release.json").read_text())["release"]

        self.assertEqual(resource, artifact)
        self.assertEqual(resource["version"], "1.1.0")
        self.assertEqual(task["system_prompt"], resource["system_prompt"])
        self.assertEqual(task["user_prompt"], resource["user_template"])
        self.assertEqual(
            metadata["open_ended_protocol_update"]["status"],
            "frozen",
        )

    def test_release_is_balanced_leakage_safe_and_reference_isolated(self) -> None:
        summary = validate_open_ended_release(
            ROOT,
            output_path=Path("data/benchmarks/ISEPDermaBench"),
        )

        self.assertEqual(summary["version"], "1.5.0")
        self.assertEqual(summary["splits"]["validation"]["tasks"], 100)
        self.assertEqual(
            summary["splits"]["internal_benchmark"]["tasks"],
            300,
        )
        self.assertEqual(summary["splits"]["validation"]["classes"], 21)
        self.assertEqual(
            summary["splits"]["internal_benchmark"]["classes"],
            21,
        )

    def test_free_text_adapter_preserves_the_visible_answer(self) -> None:
        adapter = OpenEndedDiagnosisTaskAdapter(
            benchmark_id="open_ended_diagnosis",
            system_prompt_template="system",
            user_prompt_template="user",
        )
        prepared = adapter.prepare(
            BenchmarkSample(
                sample_id="sample",
                task_id="task",
                image_uri="image.jpg",
                disease_id="D001",
                metadata={},
            )
        )
        response = adapter.parse_response(
            "model",
            "  A concise ranked differential.  ",
            prepared,
            reasoning_text="private reasoning must be ignored",
        )

        self.assertEqual(response.raw_text, "  A concise ranked differential.  ")
        self.assertEqual(response.metadata["output_mode"], "free_text")
        self.assertTrue(response.is_valid)

    def test_judge_request_is_blinded_to_model_identity_and_reasoning(self) -> None:
        task = _first_row(
            RELEASE / "tasks/open_ended_diagnosis",
            "validation",
        )
        reference = _first_row(
            RELEASE / "references/open_ended_diagnosis",
            "validation",
        )
        prompt = yaml.safe_load(
            (
                RELEASE
                / "artifacts/judges/open_ended_diagnosis_judge.yaml"
            ).read_text(encoding="utf-8")
        )
        schema = json.loads(
            (
                RELEASE
                / "artifacts/schemas/open_ended_diagnosis_judge.schema.json"
            ).read_text(encoding="utf-8")
        )
        prediction = {
            "model_id": "SECRET_TARGET_MODEL",
            "response": {
                "final_text": "VISIBLE FINAL RESPONSE",
                "reasoning": {"text": "SECRET PRIVATE REASONING"},
            },
        }

        request = _judge_request(
            task=task,
            reference=reference,
            prediction=prediction,
            prompt=prompt,
            schema=schema,
            generation={},
        )

        self.assertIn("VISIBLE FINAL RESPONSE", request.user_prompt)
        self.assertNotIn("SECRET_TARGET_MODEL", request.user_prompt)
        self.assertNotIn("SECRET PRIVATE REASONING", request.user_prompt)
        self.assertEqual(request.schema, schema)

    def test_judge_schema_and_aggregate_metrics(self) -> None:
        schema = json.loads(
            (
                RELEASE
                / "artifacts/schemas/open_ended_diagnosis_judge.schema.json"
            ).read_text(encoding="utf-8")
        )
        judgment = _valid_judgment()
        parsed = _parse_judgment(json.dumps(judgment), schema)
        metrics = compute_judge_metrics(
            [{"task_id": "task", "status": "ok", "judgment": parsed}]
        )

        self.assertEqual(metrics["judge_top_1_accuracy"], 1.0)
        self.assertEqual(metrics["judge_top_3_accuracy"], 1.0)
        self.assertEqual(metrics["mean_clinical_rationale_quality"], 3.0)
        self.assertEqual(metrics["unsupported_claim_rate"], 0.0)
        self.assertEqual(metrics["judge_coverage"], 1.0)

        qwen_metrics = compute_judge_metrics(
            [{"task_id": "task", "status": "ok", "judgment": parsed}],
            judge_model_id="qwen_3_7_flash_openrouter",
        )
        self.assertEqual(
            qwen_metrics["judge_model_id"],
            "qwen_3_7_flash_openrouter",
        )

    def test_semantically_inconsistent_judgment_is_retried(self) -> None:
        schema = json.loads(
            (
                RELEASE
                / "artifacts/schemas/open_ended_diagnosis_judge.schema.json"
            ).read_text(encoding="utf-8")
        )
        inconsistent = _valid_judgment() | {
            "reference_diagnosis_rank": 0,
        }
        backend = _ScriptedBackend(
            [_result(inconsistent), _result(_valid_judgment())]
        )
        request = _judge_request(
            task=_first_row(
                RELEASE / "tasks/open_ended_diagnosis",
                "validation",
            ),
            reference=_first_row(
                RELEASE / "references/open_ended_diagnosis",
                "validation",
            ),
            prediction={"response": {"final_text": "Rosacea is first."}},
            prompt=yaml.safe_load(
                (
                    RELEASE
                    / "artifacts/judges/open_ended_diagnosis_judge.yaml"
                ).read_text(encoding="utf-8")
            ),
            schema=schema,
            generation={},
        )

        result = asyncio.run(
            _judge_one(
                backend=backend,
                request=request,
                schema=schema,
                semaphore=asyncio.Semaphore(1),
            )
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["attempts"], 2)
        self.assertIn(
            "previous JSON judgment was invalid",
            backend.requests[1].user_prompt,
        )

    def test_rank_one_with_material_limitations_can_be_partial(self) -> None:
        judgment = _valid_judgment() | {
            "diagnosis_correctness": 3,
            "overall_verdict": "partially_correct",
            "unsupported_claim_count": 1,
            "unsupported_claim_examples": ["Invented symptom."],
        }

        _validate_judgment_semantics(judgment)

    def test_rank_specific_diagnosis_scores_are_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank 2 requires"):
            _validate_judgment_semantics(
                _valid_judgment()
                | {
                    "reference_diagnosis_rank": 2,
                    "diagnosis_correctness": 2,
                    "overall_verdict": "partially_correct",
                }
            )

    def test_fallback_trigger_is_limited_to_content_policy_codes(self) -> None:
        self.assertTrue(
            _is_content_policy_violation(
                {"safety_code": "content_policy_violation"}
            )
        )
        self.assertTrue(
            _is_content_policy_violation(
                {"safety_code": "content_filter"}
            )
        )
        self.assertFalse(
            _is_content_policy_violation(
                {"safety_code": "generic_safety_refusal"}
            )
        )

    def test_judge_metrics_report_safety_refusal_as_unavailable(self) -> None:
        judgment = {
            "reference_diagnosis_rank": 1,
            "diagnosis_correctness": 4,
            "visual_findings_correctness": 4,
            "evidence_grounding": 4,
            "clinical_rationale_quality": 4,
            "differential_quality": 4,
            "unsupported_claim_count": 0,
            "unsupported_claim_examples": [],
            "overall_verdict": "correct",
            "judge_summary": "Correct and grounded.",
        }
        metrics = compute_judge_metrics(
            [
                {"task_id": "ok", "status": "ok", "judgment": judgment},
                {
                    "task_id": "blocked",
                    "status": "judge_safety_refusal",
                    "error": "Image blocked by the judge provider.",
                },
            ]
        )

        self.assertEqual(metrics["total"], 2)
        self.assertEqual(metrics["evaluated_total"], 1)
        self.assertEqual(metrics["judge_coverage"], 0.5)
        self.assertEqual(metrics["judge_safety_refusal_count"], 1)
        self.assertEqual(metrics["judge_top_1_accuracy"], 1.0)

    def test_judge_dry_run_validates_without_writing_or_calling_model(self) -> None:
        task = _first_row(
            RELEASE / "tasks/open_ended_diagnosis",
            "validation",
        )
        with TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "run_manifest.yaml").write_text(
                yaml.safe_dump(
                    {
                        "benchmark": {"id": "open_ended_diagnosis"},
                        "evaluation": {"evaluation_set": "validation"},
                    }
                ),
                encoding="utf-8",
            )
            (run / "predictions.jsonl").write_text(
                json.dumps(
                    {
                        "task_id": task["task_id"],
                        "sample_id": task["sample_id"],
                        "status": "ok",
                        "response": {"final_text": "A ranked differential."},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = asyncio.run(
                judge_run(
                    root=ROOT,
                    run_directory=run,
                    dry_run=True,
                )
            )

            self.assertEqual(result["status"], "dry_run_valid")
            self.assertFalse(result["network_or_model_called"])
            self.assertFalse((run / "judge_manifest.yaml").exists())

    def test_content_policy_refusal_uses_fallback_and_records_provenance(
        self,
    ) -> None:
        task = _first_row(
            RELEASE / "tasks/open_ended_diagnosis",
            "validation",
        )
        primary = _ScriptedBackend(
            [
                InferenceSafetyRefusal(
                    "Image blocked.",
                    details={"code": "content_policy_violation"},
                )
            ]
        )
        fallback = _ScriptedBackend([_result(_valid_judgment())])
        with TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "run_manifest.yaml").write_text(
                yaml.safe_dump(
                    {
                        "benchmark": {"id": "open_ended_diagnosis"},
                        "evaluation": {"evaluation_set": "validation"},
                    }
                ),
                encoding="utf-8",
            )
            (run / "predictions.jsonl").write_text(
                json.dumps(
                    {
                        "task_id": task["task_id"],
                        "sample_id": task["sample_id"],
                        "status": "ok",
                        "response": {
                            "final_text": "The reference diagnosis is first."
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = asyncio.run(
                judge_run(
                    root=ROOT,
                    run_directory=run,
                    backend=primary,
                    fallback_judge_model_id=(
                        "qwen_3_7_flash_openrouter"
                    ),
                    fallback_backend=fallback,
                )
            )

            records = [
                json.loads(line)
                for line in Path(result["judgments_path"])
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            judgment = records[0]
            metrics = result["metrics"]
            self.assertEqual(judgment["judge_used"], "qwen_3_7_flash_openrouter")
            self.assertTrue(judgment["fallback_used"])
            self.assertEqual(
                judgment["fallback_reason"],
                "content_policy_violation",
            )
            self.assertEqual(metrics["fallback_used_count"], 1)
            self.assertEqual(metrics["judge_coverage"], 1.0)
            self.assertEqual(
                metrics["judge_usage_distribution"],
                {"qwen_3_7_flash_openrouter": 1},
            )

    def test_retry_invalid_rejudges_only_the_invalid_record(self) -> None:
        task = _first_row(
            RELEASE / "tasks/open_ended_diagnosis",
            "validation",
        )
        invalid = InferenceResult(
            model_id="scripted",
            final_text="",
            reasoning=ReasoningTrace(capture_mode="none"),
        )
        with TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "run_manifest.yaml").write_text(
                yaml.safe_dump(
                    {
                        "benchmark": {"id": "open_ended_diagnosis"},
                        "evaluation": {"evaluation_set": "validation"},
                    }
                ),
                encoding="utf-8",
            )
            (run / "predictions.jsonl").write_text(
                json.dumps(
                    {
                        "task_id": task["task_id"],
                        "sample_id": task["sample_id"],
                        "status": "ok",
                        "response": {"final_text": "A ranked differential."},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            first = asyncio.run(
                judge_run(
                    root=ROOT,
                    run_directory=run,
                    backend=_ScriptedBackend([invalid, invalid, invalid]),
                )
            )
            self.assertEqual(first["metrics"]["judge_invalid_count"], 1)

            retry_backend = _ScriptedBackend([_result(_valid_judgment())])
            second = asyncio.run(
                judge_run(
                    root=ROOT,
                    run_directory=run,
                    backend=retry_backend,
                    retry_invalid=True,
                )
            )
            records = [
                json.loads(line)
                for line in Path(second["judgments_path"])
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[-1]["status"], "ok")
            self.assertEqual(len(retry_backend.requests), 1)
            self.assertEqual(second["metrics"]["judge_invalid_count"], 0)
            self.assertEqual(second["metrics"]["judge_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
