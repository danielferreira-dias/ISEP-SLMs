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
    _judge_request,
    _parse_judgment,
    compute_judge_metrics,
    judge_run,
)
from src.benchmark.runner import BenchmarkSample
from src.benchmark.task_adapters import OpenEndedDiagnosisTaskAdapter
from src.data_pipeline.open_ended_benchmark import validate_open_ended_release


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "data/benchmarks/ISEPDermaBench"


def _first_row(directory: Path, split: str) -> dict:
    path = sorted(directory.glob(f"{split}-*.parquet"))[0]
    return pq.read_table(path).slice(0, 1).to_pylist()[0]


class OpenEndedDiagnosisTests(unittest.TestCase):
    def test_release_is_balanced_leakage_safe_and_reference_isolated(self) -> None:
        summary = validate_open_ended_release(
            ROOT,
            output_path=Path("data/benchmarks/ISEPDermaBench"),
        )

        self.assertEqual(summary["version"], "1.1.0")
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
        judgment = {
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
        parsed = _parse_judgment(json.dumps(judgment), schema)
        metrics = compute_judge_metrics(
            [{"task_id": "task", "status": "ok", "judgment": parsed}]
        )

        self.assertEqual(metrics["judge_top_1_accuracy"], 1.0)
        self.assertEqual(metrics["judge_top_3_accuracy"], 1.0)
        self.assertEqual(metrics["mean_clinical_rationale_quality"], 3.0)
        self.assertEqual(metrics["unsupported_claim_rate"], 0.0)

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


if __name__ == "__main__":
    unittest.main()
