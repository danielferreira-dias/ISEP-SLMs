"""Focused tests for deterministic evidence-grounded benchmark metrics."""

from __future__ import annotations

import json
import unittest

from src.benchmark.evidence_metrics import (
    compute_evidence_grounded_metrics,
)
from src.benchmark.evidence_validation import (
    parse_and_validate_evidence_response,
)
from src.benchmark.runner import BenchmarkPrediction


class EvidenceGroundedMetricTests(unittest.TestCase):
    def test_all_metric_families_use_the_declared_cohorts(self) -> None:
        concepts = ["erythema", "plaque", "scale"]
        diseases = ["D001", "D002"]
        first = _prediction(
            sample_id="S1",
            truth="D001",
            reference_concepts=["erythema", "plaque"],
            findings=["erythema", "plaque"],
            description="An erythematous plaque.",
            ranked=[
                ("D001", 0.8, ["F1"]),
                ("D002", 0.4, ["F2"]),
            ],
            diseases=diseases,
            concepts=concepts,
        )
        second = _prediction(
            sample_id="S2",
            truth="D002",
            reference_concepts=["scale"],
            findings=["plaque"],
            description="A plaque.",
            ranked=[
                ("D001", 0.7, ["F1"]),
                ("D002", 0.5, ["F1"]),
            ],
            diseases=diseases,
            concepts=concepts,
        )

        metrics = compute_evidence_grounded_metrics(
            [first, second],
            allowed_disease_ids=diseases,
            allowed_concept_ids=concepts,
            minimum_positive_cases_per_concept=1,
            calibration_bins=10,
        )

        self.assertEqual(metrics["sample_count"], 2)
        self.assertEqual(metrics["morphology_sample_count"], 2)
        self.assertAlmostEqual(metrics["finding_precision"], 0.5)
        self.assertAlmostEqual(metrics["finding_recall"], 0.5)
        self.assertAlmostEqual(metrics["finding_f1"], 0.5)
        self.assertAlmostEqual(
            metrics["micro_f1_all_concepts"],
            2 / 3,
        )
        self.assertEqual(metrics["supported_macro_concept_count"], 3)
        self.assertAlmostEqual(
            metrics["macro_f1_supported_concepts"],
            (1 + 2 / 3 + 0) / 3,
        )
        self.assertAlmostEqual(
            metrics["unsupported_finding_rate"],
            1 / 3,
        )
        self.assertAlmostEqual(
            metrics["description_findings_consistency"],
            1.0,
        )
        self.assertAlmostEqual(metrics["top_1_accuracy"], 0.5)
        self.assertEqual(metrics["diagnosis_class_count"], 2)
        self.assertAlmostEqual(metrics["top_3_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["top_6_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["mean_reciprocal_rank"], 0.75)
        self.assertAlmostEqual(metrics["macro_f1_top_1"], 1 / 3)
        self.assertAlmostEqual(
            metrics["visible_evidence_precision"],
            0.5,
        )
        self.assertAlmostEqual(metrics["valid_evidence_link_rate"], 1.0)
        self.assertAlmostEqual(metrics["grounded_top_1_success"], 0.5)
        self.assertAlmostEqual(
            metrics["correct_diagnosis_unsupported_evidence_rate"],
            0.0,
        )
        self.assertAlmostEqual(
            metrics["top_1_expected_calibration_error"],
            0.45,
        )
        self.assertAlmostEqual(metrics["top_1_brier_score"], 0.265)
        self.assertAlmostEqual(metrics["json_validity_rate"], 1.0)
        self.assertAlmostEqual(metrics["schema_compliance_rate"], 1.0)
        self.assertAlmostEqual(metrics["semantic_compliance_rate"], 1.0)

    def test_clinical_scores_do_not_require_semantic_compliance(self) -> None:
        prediction = _prediction(
            sample_id="S1",
            truth="D001",
            reference_concepts=["erythema", "plaque"],
            findings=["erythema", "plaque"],
            description="An erythematous plaque.",
            ranked=[
                ("D001", 0.8, ["F1"]),
                ("D002", 0.4, ["F2"]),
            ],
            diseases=["D001", "D002"],
            concepts=["erythema", "plaque"],
        )
        prediction.response.metadata["semantic_valid"] = False
        prediction.response.validation_errors.append(
            "semantic:case_confidence_must_match_top_confidence"
        )

        metrics = compute_evidence_grounded_metrics(
            [prediction],
            allowed_disease_ids=["D001", "D002"],
            allowed_concept_ids=["erythema", "plaque"],
            minimum_positive_cases_per_concept=1,
        )

        self.assertEqual(metrics["semantic_compliance_rate"], 0.0)
        self.assertEqual(metrics["top_1_accuracy"], 1.0)
        self.assertEqual(metrics["finding_f1"], 1.0)
        self.assertEqual(metrics["description_concept_f1"], 1.0)
        self.assertEqual(metrics["grounded_top_1_success"], 1.0)


def _prediction(
    *,
    sample_id: str,
    truth: str,
    reference_concepts: list[str],
    findings: list[str],
    description: str,
    ranked: list[tuple[str, float, list[str]]],
    diseases: list[str],
    concepts: list[str],
) -> BenchmarkPrediction:
    output = {
        "findings": [
            {
                "finding_id": f"F{index}",
                "concept_id": concept_id,
                "confidence": 0.9,
            }
            for index, concept_id in enumerate(findings, start=1)
        ],
        "clinical_description": description,
        "differential": [
            {
                "rank": index,
                "disease_id": disease_id,
                "confidence": confidence,
                "supporting_finding_ids": supports,
            }
            for index, (disease_id, confidence, supports) in enumerate(
                ranked,
                start=1,
            )
        ],
        "case_confidence": (
            "high"
            if ranked[0][1] >= 0.75
            else "moderate"
            if ranked[0][1] >= 0.40
            else "low"
        ),
    }
    response = parse_and_validate_evidence_response(
        model_id="test",
        raw_text=json.dumps(output),
        allowed_disease_ids=set(diseases),
        allowed_concept_ids=set(concepts),
        top_k=2,
    )
    if not response.metadata["semantic_valid"]:
        raise AssertionError(response.validation_errors)
    return BenchmarkPrediction(
        sample_id=sample_id,
        model_id="test",
        ground_truth_disease_id=truth,
        response=response,
        metadata={
            "morphology_concept_ids": reference_concepts,
            "score_morphology": True,
            "score_description": True,
            "score_diagnosis": True,
        },
    )


if __name__ == "__main__":
    unittest.main()
