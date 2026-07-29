"""Unit tests for benchmark output validation and metrics."""

from __future__ import annotations

import json
import unittest

from src.benchmark.metrics import compute_metrics
from src.benchmark.runner import (
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


if __name__ == "__main__":
    unittest.main()
