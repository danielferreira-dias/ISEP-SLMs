"""Tests for the two fixed visual hallucination audits."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from src.benchmark.hallucination import (
    compute_dermatology_counterfactual_metrics,
    compute_general_visual_hallucination_metrics,
    parse_dermatology_counterfactual_response,
    parse_general_visual_hallucination_response,
)
from src.benchmark.runner import BenchmarkPrediction
from src.data_pipeline.dermatology_counterfactual_hallucination import (
    validate_dermatology_counterfactual_hallucination,
)
from src.data_pipeline.general_visual_hallucination import (
    validate_general_visual_hallucination,
)


ROOT = Path(__file__).resolve().parents[1]
DISEASE_IDS = {
    "D001", "D002", "D003", "D004", "D005", "D006", "D007",
    "D009", "D011", "D012", "D013", "D014", "D015", "D016",
    "D017", "D018", "D019", "D022", "D023", "D024", "D025",
}


class HallucinationBenchmarkTests(unittest.TestCase):
    def test_materialized_cohorts_validate(self) -> None:
        general = validate_general_visual_hallucination(ROOT)
        dermatology = validate_dermatology_counterfactual_hallucination(ROOT)
        self.assertEqual(general["case_count"], 300)
        self.assertEqual(dermatology["case_count"], 200)

    def test_general_status_metrics_score_all_three_conditions(self) -> None:
        cases = [
            ("false_premises", "false_premise", None),
            (
                "insufficient_context",
                "insufficient_visual_evidence",
                None,
            ),
            ("visual_challenge", "answerable", "blue"),
        ]
        predictions = []
        for index, (condition, status, answer) in enumerate(cases):
            raw = json.dumps(
                {
                    "question_status": status,
                    "answer": answer,
                    "confidence": "moderate",
                }
            )
            response = parse_general_visual_hallucination_response(
                model_id="test",
                raw_text=raw,
                reasoning_text=None,
            )
            predictions.append(
                BenchmarkPrediction(
                    sample_id=str(index),
                    task_id=str(index),
                    model_id="test",
                    ground_truth_disease_id="",
                    response=response,
                    metadata={"condition": condition},
                )
            )
        metrics = compute_general_visual_hallucination_metrics(predictions)
        self.assertEqual(metrics["question_status_accuracy"], 1.0)
        self.assertEqual(metrics["unanswerable_hallucination_rate"], 0.0)
        self.assertFalse(metrics["free_text_answer_correctness_scored"])

    def test_counterfactual_metrics_separate_conditions(self) -> None:
        shuffled = parse_dermatology_counterfactual_response(
            model_id="test",
            raw_text=json.dumps(
                {
                    "image_status": "not_evaluable",
                    "visual_findings": [],
                    "predictions": [],
                    "confidence": "low",
                }
            ),
            reasoning_text=None,
            allowed_disease_ids=DISEASE_IDS,
        )
        swapped = parse_dermatology_counterfactual_response(
            model_id="test",
            raw_text=json.dumps(
                {
                    "image_status": "evaluable",
                    "visual_findings": ["pigmented lesion"],
                    "predictions": [{"rank": 1, "disease_id": "D001"}],
                    "confidence": "moderate",
                }
            ),
            reasoning_text=None,
            allowed_disease_ids=DISEASE_IDS,
        )
        predictions = [
            BenchmarkPrediction(
                sample_id="shuffle",
                task_id="shuffle",
                model_id="test",
                ground_truth_disease_id="D003",
                response=shuffled,
                metadata={
                    "condition": "pixel_shuffle",
                    "reference_diagnoses_json": json.dumps(
                        {"source_prompt_disease_id": "D003"}
                    ),
                },
            ),
            BenchmarkPrediction(
                sample_id="swap",
                task_id="swap",
                model_id="test",
                ground_truth_disease_id="D001",
                response=swapped,
                metadata={
                    "condition": "hard_negative_image_swap",
                    "reference_diagnoses_json": json.dumps(
                        {"source_prompt_disease_id": "D003"}
                    ),
                },
            ),
        ]
        metrics = compute_dermatology_counterfactual_metrics(predictions)
        self.assertEqual(metrics["full_counterfactual_success_rate"], 1.0)
        self.assertEqual(metrics["hard_negative_top_1_accuracy"], 1.0)
        self.assertEqual(
            metrics["hard_negative_source_label_persistence_rate"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
