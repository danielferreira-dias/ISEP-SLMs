"""Contracts for the no-image visual-grounding reasoning ablation."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from src.benchmark.runner import BenchmarkPrediction
from src.benchmark.visual_grounding import (
    compute_visual_grounding_metrics,
    parse_and_validate_visual_grounding_response,
)
from src.data_pipeline.visual_grounding_no_image import (
    validate_visual_grounding_no_image,
)


ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    "D001",
    "D002",
    "D003",
    "D004",
    "D005",
    "D006",
    "D007",
    "D009",
    "D011",
    "D012",
    "D013",
    "D014",
    "D015",
    "D016",
    "D017",
    "D018",
    "D019",
    "D022",
    "D023",
    "D024",
    "D025",
}


class VisualGroundingNoImageTests(unittest.TestCase):
    def test_materialized_control_is_valid(self) -> None:
        summary = validate_visual_grounding_no_image(ROOT)
        self.assertEqual(
            summary,
            {"case_count": 50, "group_count": 50, "class_count": 21},
        )

    def test_correct_abstention_is_semantically_valid(self) -> None:
        response = _response(
            {
                "image_status": "not_evaluable",
                "visual_findings": [],
                "predictions": [],
                "confidence": "low",
            }
        )
        self.assertTrue(response.json_valid)
        self.assertTrue(response.schema_valid)
        self.assertTrue(response.metadata["semantic_valid"])

    def test_plausible_diagnosis_on_gray_image_is_a_grounding_failure(
        self,
    ) -> None:
        response = _response(
            {
                "image_status": "evaluable",
                "visual_findings": ["erythematous plaque"],
                "predictions": [{"rank": 1, "disease_id": "D003"}],
                "confidence": "high",
            },
            reasoning_text="The disease list suggests psoriasis.",
        )
        self.assertTrue(response.schema_valid)
        self.assertFalse(response.metadata["semantic_valid"])
        self.assertTrue(response.metadata["hallucinated_visual_findings"])
        self.assertTrue(response.metadata["hallucinated_diagnosis"])
        self.assertTrue(response.metadata["overconfident"])

        metrics = compute_visual_grounding_metrics(
            [
                BenchmarkPrediction(
                    sample_id="S1",
                    task_id="T1",
                    model_id="test",
                    ground_truth_disease_id="D003",
                    response=response,
                )
            ]
        )
        self.assertEqual(metrics["correct_abstention_rate"], 0.0)
        self.assertEqual(metrics["hallucinated_diagnosis_rate"], 1.0)
        self.assertEqual(
            metrics["hallucinated_top_1_hidden_reference_match_rate"],
            1.0,
        )
        self.assertLess(metrics["hallucinated_diagnosis_rate_ci_lower"], 1.0)
        self.assertEqual(metrics["correct_abstention_rate_ci_lower"], 0.0)

    def test_fenced_json_is_recoverable_but_not_strict(self) -> None:
        payload = {
            "image_status": "not_evaluable",
            "visual_findings": [],
            "predictions": [],
            "confidence": "low",
        }
        response = parse_and_validate_visual_grounding_response(
            model_id="test",
            raw_text="```json\n" + json.dumps(payload) + "\n```",
            reasoning_text=None,
            allowed_disease_ids=ALLOWED,
        )
        self.assertFalse(response.json_valid)
        self.assertTrue(response.recoverable_json_valid)
        self.assertTrue(response.schema_valid)


def _response(
    payload: dict,
    *,
    reasoning_text: str | None = None,
):
    return parse_and_validate_visual_grounding_response(
        model_id="test",
        raw_text=json.dumps(payload),
        reasoning_text=reasoning_text,
        allowed_disease_ids=ALLOWED,
    )


if __name__ == "__main__":
    unittest.main()
