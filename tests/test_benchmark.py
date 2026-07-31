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
    def test_single_json_fence_is_recoverable_but_not_raw_valid(self) -> None:
        response = parse_and_validate_response(
            model_id="test",
            raw_text=(
                "```json\n"
                '{"predictions":[{"rank":1,"disease_id":"D001"}]}'
                "\n```"
            ),
            allowed_disease_ids={"D001"},
            top_k=1,
        )

        self.assertFalse(response.json_valid)
        self.assertTrue(response.recoverable_json_valid)
        self.assertFalse(response.schema_valid)
        self.assertTrue(response.canonical_schema_valid)
        self.assertEqual(
            response.canonicalization_rules,
            ["single_json_fence"],
        )
        self.assertEqual(
            response.metadata["json_recovery"],
            "single_json_fence",
        )

    def test_ranked_id_list_has_auditable_canonical_projection(self) -> None:
        response = parse_and_validate_response(
            model_id="minicpm",
            raw_text='{"predictions":["D002","D001","D003"]}',
            allowed_disease_ids={"D001", "D002", "D003"},
            top_k=3,
        )
        prediction = BenchmarkPrediction(
            sample_id="SAMPLE_1",
            model_id="minicpm",
            ground_truth_disease_id="D001",
            response=response,
        )
        metrics = compute_metrics(
            [prediction],
            allowed_disease_ids=["D001", "D002", "D003"],
        )

        self.assertTrue(response.json_valid)
        self.assertFalse(response.schema_valid)
        self.assertTrue(response.canonical_schema_valid)
        self.assertEqual(
            response.canonicalization_rules,
            ["ranked_disease_id_list_to_objects"],
        )
        self.assertEqual(
            response.canonical_output,
            {
                "predictions": [
                    {"rank": 1, "disease_id": "D002"},
                    {"rank": 2, "disease_id": "D001"},
                    {"rank": 3, "disease_id": "D003"},
                ]
            },
        )
        self.assertEqual(metrics["top_1_accuracy"], 0.0)
        self.assertEqual(metrics["canonical_top_1_accuracy"], 0.0)
        self.assertEqual(metrics["canonical_top_3_accuracy"], 1.0)
        self.assertEqual(
            metrics["canonical_schema_compliance_rate"],
            1.0,
        )

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

    def test_visual_top_k_reports_skin_tone_performance(self) -> None:
        allowed = [f"D{index:03d}" for index in range(1, 7)]

        def prediction(
            *,
            sample_id: str,
            ranked_ids: list[str],
            skin_tone_system: str | None,
            skin_tone: str | None,
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
                allowed_disease_ids=set(allowed),
                top_k=6,
            )
            return BenchmarkPrediction(
                sample_id=sample_id,
                model_id="test",
                ground_truth_disease_id="D001",
                response=response,
                metadata={
                    "skin_tone_system": skin_tone_system,
                    "skin_tone": skin_tone,
                    "leakage_group_id": sample_id,
                },
            )

        metrics = compute_metrics(
            [
                prediction(
                    sample_id="S1",
                    ranked_ids=allowed,
                    skin_tone_system="fitzpatrick",
                    skin_tone="FST_1",
                ),
                prediction(
                    sample_id="S2",
                    ranked_ids=["D002", "D001", *allowed[2:]],
                    skin_tone_system="fitzpatrick",
                    skin_tone="FST_2",
                ),
                prediction(
                    sample_id="S3",
                    ranked_ids=["D002", "D003", "D001", *allowed[3:]],
                    skin_tone_system="monk",
                    skin_tone="MST_3",
                ),
                prediction(
                    sample_id="S4",
                    ranked_ids=allowed,
                    skin_tone_system=None,
                    skin_tone=None,
                ),
            ],
            allowed_disease_ids=allowed,
            minimum_subgroup_unique_groups=1,
            minimum_per_disease_unique_groups=1,
        )

        fst_one = metrics["by_skin_tone"]["fitzpatrick:FST_1"]
        self.assertEqual(fst_one["sample_count"], 1)
        self.assertEqual(fst_one["top_1_accuracy"], 1.0)
        self.assertLess(fst_one["top_1_ci_lower"], 1.0)
        self.assertEqual(fst_one["top_1_ci_upper"], 1.0)
        self.assertTrue(fst_one["statistically_supported"])
        self.assertEqual(
            metrics["by_skin_tone_aggregate"][
                "fitzpatrick:FST_1-2"
            ]["top_1_accuracy"],
            0.5,
        )
        self.assertEqual(metrics["skin_tone_missing_count"], 1)
        self.assertEqual(metrics["skin_tone_coverage_rate"], 0.75)
        self.assertFalse(
            metrics["by_skin_tone"]["unknown"][
                "statistically_supported"
            ]
        )
        self.assertEqual(
            metrics["skin_tone_worst_group_top_1_accuracy"],
            0.0,
        )
        self.assertEqual(metrics["skin_tone_top_1_accuracy_gap"], 1.0)

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

    def test_confusion_metrics_support_legacy_string_candidates(self) -> None:
        response = parse_and_validate_response(
            model_id="test",
            raw_text=json.dumps(
                {
                    "predictions": [
                        {"rank": 1, "disease_id": "D001"},
                        {"rank": 2, "disease_id": "D002"},
                        {"rank": 3, "disease_id": "D003"},
                    ]
                }
            ),
            allowed_disease_ids={"D001", "D002", "D003"},
            top_k=3,
        )
        predictions = [
            BenchmarkPrediction(
                task_id=f"PAIR_1::{difficulty}",
                sample_id="SAMPLE_1",
                model_id="test",
                ground_truth_disease_id="D001",
                response=response,
                metadata={
                    "pair_id": "PAIR_1",
                    "difficulty": difficulty,
                    "confusion_set_id": "lesions",
                    "candidate_disease_ids": (
                        "['D001' 'D002' 'D003']"
                    ),
                },
            )
            for difficulty in (
                "low_confusability",
                "high_confusability",
            )
        ]

        metrics = compute_confusion_set_metrics(
            predictions,
            allowed_disease_ids=["D001", "D002", "D003"],
            bootstrap_resamples=10,
        )

        self.assertEqual(metrics["invalid_candidate_id_rate"], 0.0)

    def test_confusion_metrics_report_canonical_ranked_lists(self) -> None:
        predictions = []
        for difficulty, ranked_ids in (
            ("low_confusability", ["D001", "D002", "D003"]),
            ("high_confusability", ["D002", "D001", "D003"]),
        ):
            response = parse_and_validate_response(
                model_id="minicpm",
                raw_text=json.dumps({"predictions": ranked_ids}),
                allowed_disease_ids={"D001", "D002", "D003"},
                top_k=3,
            )
            predictions.append(
                BenchmarkPrediction(
                    task_id=f"PAIR_1::{difficulty}",
                    sample_id="SAMPLE_1",
                    model_id="minicpm",
                    ground_truth_disease_id="D001",
                    response=response,
                    metadata={
                        "pair_id": "PAIR_1",
                        "difficulty": difficulty,
                        "confusion_set_id": "lesions",
                        "candidate_disease_ids": [
                            "D001",
                            "D002",
                            "D003",
                        ],
                    },
                )
            )

        metrics = compute_confusion_set_metrics(
            predictions,
            allowed_disease_ids=["D001", "D002", "D003"],
            bootstrap_resamples=10,
        )

        self.assertEqual(metrics["top_1_accuracy"], 0.0)
        self.assertEqual(metrics["canonical_top_1_accuracy"], 0.5)
        self.assertEqual(metrics["canonical_top_2_accuracy"], 1.0)
        self.assertEqual(
            metrics["canonical_schema_compliance_rate"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
