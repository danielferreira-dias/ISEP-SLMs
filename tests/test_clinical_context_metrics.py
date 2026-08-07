"""Tests for paired clinical-context ablation metrics."""

from __future__ import annotations

import unittest

from src.benchmark.metrics import compute_clinical_context_ablation_metrics
from src.benchmark.runner import (
    BenchmarkPrediction,
    ModelResponse,
)


class ClinicalContextMetricsTests(unittest.TestCase):
    def test_reports_paired_improvement_and_condition_metrics(self) -> None:
        rows = [
            _prediction("P1", "image_only", truth="D001", predicted="D002"),
            _prediction(
                "P1", "image_plus_context", truth="D001", predicted="D001"
            ),
            _prediction("P2", "image_only", truth="D002", predicted="D002"),
            _prediction(
                "P2", "image_plus_context", truth="D002", predicted="D002"
            ),
        ]

        metrics = compute_clinical_context_ablation_metrics(
            rows,
            allowed_disease_ids=["D001", "D002"],
            bootstrap_resamples=100,
            bootstrap_seed=7,
        )

        self.assertEqual(metrics["pair_count"], 2)
        self.assertEqual(
            metrics["by_condition"]["image_only"]["top_1_accuracy"],
            0.5,
        )
        self.assertEqual(
            metrics["by_condition"]["image_plus_context"]["top_1_accuracy"],
            1.0,
        )
        self.assertEqual(
            metrics["paired_top_1"]["improved_pair_count"],
            1,
        )
        self.assertEqual(metrics["paired_top_1"]["accuracy_delta"], 0.5)


def _prediction(
    pair_id: str,
    condition: str,
    *,
    truth: str,
    predicted: str,
) -> BenchmarkPrediction:
    output = {"predictions": [{"rank": 1, "disease_id": predicted}]}
    response = ModelResponse(
        model_id="test",
        raw_text="{}",
        parsed_output=output,
        json_valid=True,
        schema_valid=True,
        recoverable_json_valid=True,
        canonical_output=output,
        canonical_schema_valid=True,
    )
    return BenchmarkPrediction(
        sample_id=pair_id,
        model_id="test",
        ground_truth_disease_id=truth,
        response=response,
        task_id=f"{pair_id}:{condition}",
        metadata={
            "pair_id": pair_id,
            "condition": condition,
            "leakage_group_id": pair_id,
        },
    )


if __name__ == "__main__":
    unittest.main()
