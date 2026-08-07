"""Tests for the dedicated DermoBench runtime and batch-judge adapter."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import yaml

from src.benchmark.dermobench import (
    DermoBenchTaskAdapter,
    SPECS,
    list_dermobench_configs,
    load_dermobench_config,
    load_dermobench_dataset,
    resolve_dermobench_spec,
)
from src.benchmark.dermobench_judge import collect_batch, prepare_batch
from src.benchmark.runner import BenchmarkPrediction


ROOT = Path(__file__).resolve().parents[1]


class DermoBenchAdapterTests(unittest.TestCase):
    def test_lists_every_public_filtered_task(self) -> None:
        configs = list_dermobench_configs(root=ROOT)
        self.assertEqual(len(configs), 13)
        self.assertEqual(
            {config.benchmark.id for config in configs},
            {spec.benchmark_id for spec in SPECS},
        )

    def test_mcq_loader_preserves_options_and_hides_reference(self) -> None:
        config = load_dermobench_config(
            "task_2_1_diagnosis_mcq_4_choices",
            root=ROOT,
        )
        loaded = load_dermobench_dataset(
            root=ROOT,
            benchmark=config,
            evaluation_set="filtered",
            limit=3,
            seed=42,
        )

        self.assertEqual(len(loaded.samples), 3)
        for sample in loaded.samples:
            self.assertTrue(sample.task_id.startswith("dermobench:"))
            self.assertEqual(len(sample.candidate_disease_ids or ()), 4)
            self.assertIn("Respond with ONLY the letter", sample.user_prompt)
            # The correct option is necessarily visible among all choices,
            # but the adapter must not append a second answer-key copy.
            self.assertEqual(
                sample.user_prompt.count(sample.metadata["reference_answer"]),
                1,
            )
            self.assertTrue((ROOT / sample.image_uri).is_file())

    def test_duplicate_upstream_ids_receive_unique_execution_ids(self) -> None:
        config = load_dermobench_config("task1.3", root=ROOT)
        loaded = load_dermobench_dataset(
            root=ROOT,
            benchmark=config,
            evaluation_set=None,
            limit=None,
            seed=42,
        )
        task_ids = [sample.task_id for sample in loaded.samples]
        self.assertEqual(len(task_ids), 5530)
        self.assertEqual(len(task_ids), len(set(task_ids)))

    def test_mcq_parser_reports_strict_and_recovered_contracts(self) -> None:
        spec = resolve_dermobench_spec("task2.1-4")
        adapter = DermoBenchTaskAdapter(spec)
        config = load_dermobench_config(spec.key, root=ROOT)
        loaded = load_dermobench_dataset(
            root=ROOT,
            benchmark=config,
            evaluation_set=None,
            limit=1,
            seed=42,
        )
        sample = loaded.samples[0]
        prepared = adapter.prepare(sample)
        exact = adapter.parse_response(
            "model",
            sample.disease_id,
            prepared,
        )
        prose = adapter.parse_response(
            "model",
            f"The answer is {sample.disease_id}.",
            prepared,
        )
        invalid = adapter.parse_response("model", "unsure", prepared)

        self.assertTrue(exact.metadata["strict_choice_only"])
        self.assertEqual(prose.canonical_output, {"choice": sample.disease_id})
        self.assertFalse(invalid.is_valid)

        metrics = adapter.compute_metrics(
            [
                BenchmarkPrediction(
                    sample_id=sample.sample_id,
                    task_id=sample.task_id,
                    model_id="model",
                    ground_truth_disease_id=sample.disease_id,
                    response=exact,
                    metadata=sample.metadata,
                )
            ]
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertFalse(metrics["json_validity_applicable"])

    def test_open_response_is_staged_without_fake_content_score(self) -> None:
        spec = resolve_dermobench_spec("task3.1")
        adapter = DermoBenchTaskAdapter(spec)
        config = load_dermobench_config(spec.key, root=ROOT)
        loaded = load_dermobench_dataset(
            root=ROOT,
            benchmark=config,
            evaluation_set=None,
            limit=1,
            seed=42,
        )
        sample = loaded.samples[0]
        response = adapter.parse_response(
            "model",
            "<reasoning>Visible plaque.</reasoning>"
            "<final_diagnosis>Psoriasis</final_diagnosis>",
            adapter.prepare(sample),
        )
        prediction = BenchmarkPrediction(
            sample_id=sample.sample_id,
            task_id=sample.task_id,
            model_id="model",
            ground_truth_disease_id="",
            response=response,
            metadata=sample.metadata,
        )
        metrics = adapter.compute_metrics([prediction])

        self.assertTrue(response.is_valid)
        self.assertTrue(response.metadata["format_compliant"])
        self.assertEqual(metrics["pending_judge_count"], 1)
        self.assertNotIn("accuracy", metrics)

    def test_batch_payload_is_text_only_and_uses_upstream_voter_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "run_manifest.yaml").write_text(
                yaml.safe_dump(
                    {
                        "benchmark": {
                            "id": (
                                "dermobench_task_1_1_description_without_"
                                "morphology"
                            )
                        }
                    }
                ),
                encoding="utf-8",
            )
            prediction = {
                "task_id": "dermobench:task_1_1:000001:case",
                "status": "ok",
                "metadata": {
                    "reference_answer": "A red scaly plaque.",
                    "user_prompt": "Describe the lesion.",
                },
                "response": {"final_text": "A red plaque with scale."},
            }
            (run / "predictions.jsonl").write_text(
                json.dumps(prediction) + "\n",
                encoding="utf-8",
            )
            (run / "rendered_prompts.jsonl").write_text("", encoding="utf-8")

            summary = prepare_batch(run_directory=run)
            payload = json.loads(
                (run / "dermobench_judge/batch_request.json").read_text(
                    encoding="utf-8"
                )
            )

            judge_json = json.dumps(
                {
                    "claims": [],
                    "counts": {},
                    "rubric": {
                        "accuracy": 0.8,
                        "completeness": 0.7,
                        "consistency": 0.9,
                    },
                    "overall": 80.0,
                    "short_feedback": "Good agreement.",
                }
            )
            completed = {
                "id": "batch_test",
                "status": "completed",
                "model": payload["model"],
                "usage": {"cost": 0.01},
                "results": [
                    {
                        "custom_id": request["custom_id"],
                        "response": {
                            "status_code": 200,
                            "body": {
                                "choices": [
                                    {"message": {"content": judge_json}}
                                ]
                            },
                        },
                        "error": None,
                    }
                    for request in payload["requests"]
                ],
            }
            collected = collect_batch(
                run_directory=run,
                response=completed,
            )

        self.assertEqual(summary["request_count"], 3)
        self.assertEqual(payload["endpoint"], "/v1/chat/completions")
        self.assertTrue(payload["model"].endswith(":batch"))
        self.assertEqual(
            payload["requests"][0]["body"]["reasoning"]["effort"],
            "minimal",
        )
        serialized = json.dumps(payload)
        self.assertNotIn("image_url", serialized)
        self.assertNotIn("input_image", serialized)
        self.assertEqual(
            collected["summary"]["mean_final_score"],
            80.0,
        )


if __name__ == "__main__":
    unittest.main()
