"""Unit tests for benchmark output validation and metrics."""

from __future__ import annotations

import json
import unittest

from src.benchmark.metrics import (
    compute_confusion_set_metrics,
    compute_metrics,
)
from src.benchmark.runner import (
    BenchmarkRunner,
    BenchmarkSample,
    BenchmarkPrediction,
    parse_and_validate_response,
)


class BenchmarkValidationTests(unittest.TestCase):
    def test_valid_ranked_output_is_accepted_and_scored(self) -> None:
        allowed = [f"D{index:03d}" for index in range(1, 8)]
        raw_text = json.dumps(
            {
                "predictions": [
                    {"rank": index, "disease_id": disease_id}
                    for index, disease_id in enumerate(
                        allowed[:6],
                        start=1,
                    )
                ]
            }
        )
        response = parse_and_validate_response(
            model_id="test",
            raw_text=raw_text,
            allowed_disease_ids=set(allowed),
            top_k=6,
        )
        prediction = BenchmarkPrediction(
            sample_id="SAMPLE_1",
            model_id="test",
            ground_truth_disease_id="D003",
            response=response,
        )
        metrics = compute_metrics(
            [prediction],
            allowed_disease_ids=allowed,
        )

        self.assertTrue(response.is_valid)
        self.assertEqual(metrics["top_1_accuracy"], 0.0)
        self.assertEqual(metrics["top_3_accuracy"], 1.0)
        self.assertEqual(metrics["top_6_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["mean_reciprocal_rank"], 1 / 3)

    def test_duplicate_disease_id_is_rejected(self) -> None:
        raw_text = json.dumps(
            {
                "predictions": [
                    {"rank": index, "disease_id": "D001"}
                    for index in range(1, 7)
                ]
            }
        )
        response = parse_and_validate_response(
            model_id="test",
            raw_text=raw_text,
            allowed_disease_ids={"D001"},
            top_k=6,
        )
        self.assertFalse(response.schema_valid)
        self.assertIn(
            "disease_ids_must_be_unique",
            response.validation_errors,
        )

    def test_runner_restricts_each_task_to_its_candidate_ids(self) -> None:
        class CapturingBackend:
            model_id = "candidate_test"

            def __init__(self) -> None:
                self.schema = None

            def generate(self, **kwargs):
                self.schema = kwargs["schema"]
                return json.dumps(
                    {
                        "predictions": [
                            {"rank": 1, "disease_id": "D001"},
                            {"rank": 2, "disease_id": "D002"},
                            {"rank": 3, "disease_id": "D004"},
                        ]
                    }
                )

        backend = CapturingBackend()
        runner = BenchmarkRunner(
            backend=backend,
            system_prompt="Rank {{ top_k }} diseases.",
            user_prompt_template="{{ disease_taxonomy }}",
            schema={
                "properties": {
                    "predictions": {
                        "minItems": 3,
                        "maxItems": 3,
                        "items": {
                            "properties": {
                                "rank": {"minimum": 1, "maximum": 3},
                                "disease_id": {
                                    "enum": [
                                        "D001",
                                        "D002",
                                        "D003",
                                        "D004",
                                    ]
                                },
                            }
                        },
                    }
                }
            },
            taxonomy_items=[
                {"id": f"D00{index}", "display_name": f"Disease {index}"}
                for index in range(1, 5)
            ],
            top_k=3,
            image_loader=lambda _: b"image",
        )
        prediction = runner.run_sample(
            BenchmarkSample(
                task_id="TASK_1",
                sample_id="SAMPLE_1",
                image_uri="image.jpg",
                disease_id="D001",
                candidate_disease_ids=("D001", "D002", "D003"),
                metadata={"pair_id": "PAIR_1"},
            )
        )

        runtime_enum = backend.schema["properties"]["predictions"][
            "items"
        ]["properties"]["disease_id"]["enum"]
        self.assertEqual(runtime_enum, ["D001", "D002", "D003"])
        self.assertFalse(prediction.response.schema_valid)
        self.assertIn(
            "prediction_2_disease_id_unknown",
            prediction.response.validation_errors,
        )
        self.assertEqual(prediction.task_id, "TASK_1")

    def test_confusion_metrics_compare_paired_conditions(self) -> None:
        def prediction(
            *,
            difficulty: str,
            ranked_ids: list[str],
        ) -> BenchmarkPrediction:
            response = parse_and_validate_response(
                model_id="test",
                raw_text=json.dumps(
                    {
                        "predictions": [
                            {"rank": rank, "disease_id": disease_id}
                            for rank, disease_id in enumerate(
                                ranked_ids,
                                start=1,
                            )
                        ]
                    }
                ),
                allowed_disease_ids=set(ranked_ids),
                top_k=3,
            )
            return BenchmarkPrediction(
                task_id=f"PAIR_1::{difficulty}",
                sample_id="SAMPLE_1",
                model_id="test",
                ground_truth_disease_id="D001",
                response=response,
                metadata={
                    "pair_id": "PAIR_1",
                    "difficulty": difficulty,
                    "confusion_set_id": "lesions",
                    "candidate_disease_ids": ranked_ids,
                },
            )

        metrics = compute_confusion_set_metrics(
            [
                prediction(
                    difficulty="low_confusability",
                    ranked_ids=["D001", "D003", "D011"],
                ),
                prediction(
                    difficulty="high_confusability",
                    ranked_ids=["D002", "D001", "D006"],
                ),
            ],
            allowed_disease_ids=[
                "D001",
                "D002",
                "D003",
                "D006",
                "D011",
            ],
            bootstrap_resamples=100,
        )

        self.assertEqual(metrics["pair_count"], 1)
        self.assertEqual(metrics["top_1_accuracy"], 0.5)
        self.assertEqual(metrics["top_2_accuracy"], 1.0)
        self.assertEqual(metrics["low_confusability_accuracy"], 1.0)
        self.assertEqual(metrics["high_confusability_accuracy"], 0.0)
        self.assertEqual(metrics["confusability_accuracy_gap"], 1.0)
        self.assertEqual(
            metrics["confusability_accuracy_gap_ci_lower"],
            1.0,
        )
        self.assertEqual(
            metrics["confusability_accuracy_gap_ci_upper"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
