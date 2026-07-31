"""Focused tests for deterministic evidence-grounded response validation."""

from __future__ import annotations

import json
import unittest

from src.benchmark.evidence_validation import (
    extract_morphology_concepts,
    parse_and_validate_evidence_response,
)


class EvidenceResponseValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.allowed_diseases = {"D001", "D002"}
        self.allowed_concepts = {"erythema", "plaque", "scale"}

    def test_valid_response_and_separate_reasoning_are_handled(self) -> None:
        response = parse_and_validate_evidence_response(
            model_id="test-model",
            raw_text=json.dumps(_valid_output()),
            reasoning_text=(
                "Psoriasis is likely; a biopsy might be useful."
            ),
            allowed_disease_ids=self.allowed_diseases,
            allowed_concept_ids=self.allowed_concepts,
            disease_terms=("psoriasis",),
            top_k=2,
        )

        self.assertTrue(response.json_valid)
        self.assertTrue(response.schema_valid)
        self.assertTrue(response.metadata["semantic_valid"])
        self.assertEqual(response.validation_errors, [])
        self.assertEqual(
            set(response.metadata["description_concept_ids"]),
            {"erythema", "plaque"},
        )
        self.assertNotIn("biopsy", repr(response.metadata).lower())

    def test_inline_thinking_makes_the_final_answer_invalid(self) -> None:
        raw = (
            "<think>Potential psoriasis; consider biopsy.</think>\n"
            + json.dumps(_valid_output())
        )
        response = parse_and_validate_evidence_response(
            model_id="test-model",
            raw_text=raw,
            allowed_disease_ids=self.allowed_diseases,
            allowed_concept_ids=self.allowed_concepts,
            disease_terms=("psoriasis",),
            top_k=2,
        )

        self.assertFalse(response.json_valid)
        self.assertFalse(response.metadata["semantic_valid"])

    def test_json_fence_is_format_invalid_but_semantically_compliant(
        self,
    ) -> None:
        response = parse_and_validate_evidence_response(
            model_id="test-model",
            raw_text=f"```json\n{json.dumps(_valid_output())}\n```",
            allowed_disease_ids=self.allowed_diseases,
            allowed_concept_ids=self.allowed_concepts,
            top_k=2,
        )

        self.assertFalse(response.json_valid)
        self.assertTrue(response.recoverable_json_valid)
        self.assertTrue(response.schema_valid)
        self.assertTrue(response.metadata["semantic_valid"])

    def test_forbidden_description_is_a_semantic_not_schema_error(
        self,
    ) -> None:
        output = _valid_output()
        output["clinical_description"] = (
            "An erythematous plaque consistent with psoriasis; biopsy advised."
        )
        response = parse_and_validate_evidence_response(
            model_id="test-model",
            raw_text=json.dumps(output),
            allowed_disease_ids=self.allowed_diseases,
            allowed_concept_ids=self.allowed_concepts,
            disease_terms=("psoriasis",),
            top_k=2,
        )

        self.assertTrue(response.schema_valid)
        self.assertFalse(response.metadata["semantic_valid"])
        self.assertIn(
            "semantic:clinical_description_contains_forbidden_content",
            response.validation_errors,
        )
        self.assertTrue(
            response.metadata["audit"]["forbidden_description_content"]
        )

    def test_broken_evidence_reference_is_audited(self) -> None:
        output = _valid_output()
        output["differential"][0]["supporting_finding_ids"] = ["F3"]
        response = parse_and_validate_evidence_response(
            model_id="test-model",
            raw_text=json.dumps(output),
            allowed_disease_ids=self.allowed_diseases,
            allowed_concept_ids=self.allowed_concepts,
            top_k=2,
        )

        self.assertTrue(response.schema_valid)
        self.assertFalse(response.metadata["semantic_valid"])
        self.assertTrue(
            response.metadata["audit"]["broken_evidence_reference"]
        )
        self.assertIn(
            "semantic:supporting_finding_id_must_resolve",
            response.validation_errors,
        )

    def test_alias_extraction_honours_simple_negation(self) -> None:
        concepts = extract_morphology_concepts(
            "An erythematous plaque without scale.",
            allowed_concept_ids=self.allowed_concepts,
        )
        self.assertEqual(concepts, {"erythema", "plaque"})

    def test_color_on_measuring_device_is_not_a_lesion_finding(self) -> None:
        concepts = extract_morphology_concepts(
            "A salmon-pink erythematous macule is partially obscured by a "
            "white measuring device.",
            allowed_concept_ids={
                "salmon",
                "erythema",
                "macule",
                "white_hypopigmentation",
            },
        )
        self.assertEqual(concepts, {"salmon", "erythema", "macule"})

    def test_color_attached_to_crust_is_a_lesion_finding(self) -> None:
        concepts = extract_morphology_concepts(
            "An erythematous plaque with scale and yellow crust.",
            allowed_concept_ids={"erythema", "plaque", "scale", "yellow"},
        )
        self.assertEqual(
            concepts,
            {"erythema", "plaque", "scale", "yellow"},
        )

    def test_white_scale_is_not_misread_as_hypopigmentation(self) -> None:
        concepts = extract_morphology_concepts(
            "An erythematous plaque with adherent white scale.",
            allowed_concept_ids={
                "erythema",
                "plaque",
                "scale",
                "white_hypopigmentation",
            },
        )
        self.assertEqual(concepts, {"erythema", "plaque", "scale"})

    def test_nonstandard_json_number_is_rejected(self) -> None:
        raw = json.dumps(_valid_output()).replace("0.91", "NaN", 1)
        response = parse_and_validate_evidence_response(
            model_id="test-model",
            raw_text=raw,
            allowed_disease_ids=self.allowed_diseases,
            allowed_concept_ids=self.allowed_concepts,
            top_k=2,
        )
        self.assertFalse(response.json_valid)
        self.assertIn(
            "non_standard_json_constant:NaN",
            response.validation_errors[0],
        )

    def test_duplicate_json_object_key_is_rejected(self) -> None:
        raw = (
            '{"findings":[],"findings":[],"clinical_description":"x",'
            '"differential":[],"case_confidence":"low"}'
        )
        response = parse_and_validate_evidence_response(
            model_id="test-model",
            raw_text=raw,
            allowed_disease_ids=self.allowed_diseases,
            allowed_concept_ids=self.allowed_concepts,
            top_k=2,
        )
        self.assertFalse(response.json_valid)
        self.assertIn(
            "duplicate_json_key:findings",
            response.validation_errors[0],
        )


def _valid_output() -> dict:
    return {
        "findings": [
            {
                "finding_id": "F1",
                "concept_id": "erythema",
                "confidence": 0.91,
            },
            {
                "finding_id": "F2",
                "concept_id": "plaque",
                "confidence": 0.87,
            },
        ],
        "clinical_description": "An erythematous plaque.",
        "differential": [
            {
                "rank": 1,
                "disease_id": "D001",
                "confidence": 0.80,
                "supporting_finding_ids": ["F1", "F2"],
            },
            {
                "rank": 2,
                "disease_id": "D002",
                "confidence": 0.45,
                "supporting_finding_ids": ["F2"],
            },
        ],
        "case_confidence": "high",
    }


if __name__ == "__main__":
    unittest.main()
