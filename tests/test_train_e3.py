"""CPU-only tests for E3 structured and deterministic teacher targets."""

from __future__ import annotations

import hashlib
import json
import unittest

from PIL import Image
from pydantic import ValidationError

from src.train.domain import Taxonomy, TaxonomyClass
from src.train.e3 import (
    OPEN_RESPONSE_TEMPLATES,
    DeterministicOpenResponseRenderer,
    E3StructuredPhase,
    E3StructuredSample,
    E3TrainingVariant,
    StructuredClinicalTarget,
    canonical_structured_json,
)


class E3DomainTests(unittest.TestCase):
    def test_json_arrays_are_frozen_and_canonical_serialization_is_stable(self) -> None:
        target = _target()
        self.assertIsInstance(target.observations, tuple)
        self.assertIsInstance(target.differential, tuple)

        first = canonical_structured_json(target)
        second = canonical_structured_json(
            StructuredClinicalTarget.model_validate(json.loads(first))
        )
        self.assertEqual(first, second)
        self.assertNotIn("NaN", first)

    def test_unknown_evidence_link_is_rejected(self) -> None:
        payload = _target_payload()
        differential = payload["differential"]
        assert isinstance(differential, list)
        first = differential[0]
        assert isinstance(first, dict)
        first["supporting_observation_ids"] = ["obs_unknown"]
        with self.assertRaisesRegex(ValidationError, "unknown observation links"):
            StructuredClinicalTarget.model_validate(payload)

    def test_duplicate_observation_ids_are_rejected(self) -> None:
        payload = _target_payload()
        observations = payload["observations"]
        assert isinstance(observations, list)
        observations.append(dict(observations[0]))
        with self.assertRaisesRegex(ValidationError, "Observation IDs must be unique"):
            StructuredClinicalTarget.model_validate(payload)

    def test_noncontiguous_differential_ranks_are_rejected(self) -> None:
        payload = _target_payload()
        differential = payload["differential"]
        assert isinstance(differential, list)
        second = differential[1]
        assert isinstance(second, dict)
        second["rank"] = 3
        with self.assertRaisesRegex(ValidationError, "ranks must be contiguous"):
            StructuredClinicalTarget.model_validate(payload)


class E3RenderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.renderer = DeterministicOpenResponseRenderer(_taxonomy())
        self.target = _target()

    def test_same_sample_is_byte_deterministic(self) -> None:
        first = self.renderer.render("sample-001", self.target)
        second = self.renderer.render("sample-001", self.target)
        self.assertEqual(first, second)
        self.assertEqual(
            first.target_sha256,
            hashlib.sha256(first.text.encode()).hexdigest(),
        )

    def test_sample_hash_distributes_across_all_frozen_templates(self) -> None:
        selected = {
            self.renderer.template_for(f"sample-{index:04d}").template_id
            for index in range(1000)
        }
        self.assertEqual(len(OPEN_RESPONSE_TEMPLATES), 12)
        self.assertEqual(selected, set(self.renderer.template_ids))

    def test_every_template_preserves_the_same_clinical_facts(self) -> None:
        outputs = {
            template.template_id: template.render(self.renderer._context(self.target))
            for template in OPEN_RESPONSE_TEMPLATES
        }
        self.assertEqual(len(set(outputs.values())), 12)
        for text in outputs.values():
            self.assertIn("melanoma", text.lower())
            self.assertIn("irregular border", text)
            self.assertIn("color variation", text)
            self.assertIn("melanocytic nevus", text)
            self.assertIn("recent evolution", text)
            self.assertIn("clinical context", text.lower())

    def test_out_of_taxonomy_differential_is_rejected_at_render_time(self) -> None:
        payload = _target_payload()
        differential = payload["differential"]
        assert isinstance(differential, list)
        second = differential[1]
        assert isinstance(second, dict)
        second["disease_id"] = "D999"
        with self.assertRaisesRegex(ValueError, "outside the taxonomy"):
            self.renderer.render(
                "sample-001",
                StructuredClinicalTarget.model_validate(payload),
            )


class E3PhaseTests(unittest.TestCase):
    def test_structured_and_open_variants_are_separate_tasks(self) -> None:
        sample = _sample()
        structured = E3StructuredPhase(
            _taxonomy(), E3TrainingVariant.STRUCTURED
        ).format_example(sample)
        opened = E3StructuredPhase(
            _taxonomy(), E3TrainingVariant.OPEN_RESPONSE
        ).format_example(sample)

        structured_record = structured.as_record()
        open_record = opened.as_record()
        self.assertNotEqual(structured_record["task_id"], open_record["task_id"])
        self.assertIsNone(structured_record["template_id"])
        self.assertIsNotNone(open_record["template_id"])

        structured_messages = structured_record["messages"]
        open_messages = open_record["messages"]
        assert isinstance(structured_messages, list)
        assert isinstance(open_messages, list)
        self.assertEqual(structured_messages[0]["content"][0]["type"], "image")
        self.assertEqual(open_messages[0]["content"][0]["type"], "image")
        self.assertNotIn("melanoma", structured_messages[0]["content"][1]["text"])
        self.assertNotIn("melanoma", open_messages[0]["content"][1]["text"])

    def test_leading_diagnosis_must_match_accepted_gold(self) -> None:
        sample = _sample(disease_id="D002", label="melanocytic nevus")
        with self.assertRaisesRegex(ValueError, "does not match"):
            E3StructuredPhase(
                _taxonomy(), E3TrainingVariant.OPEN_RESPONSE
            ).format_example(sample)


def _taxonomy() -> Taxonomy:
    return Taxonomy(
        taxonomy_id="test-taxonomy",
        classes=(
            TaxonomyClass("D001", "melanoma"),
            TaxonomyClass("D002", "melanocytic nevus"),
            TaxonomyClass("D003", "seborrheic keratosis"),
        ),
    )


def _target_payload() -> dict[str, object]:
    return {
        "image_assessment": {
            "is_evaluable": True,
            "views_available": ["close_up"],
            "quality_defects": [],
            "has_anatomic_overview": False,
            "has_scale": False,
            "has_lateral_profile": False,
            "distribution_assessability": "within_frame_only",
            "color_reliability": "uncertain",
        },
        "dominant_visual_pattern": "pigmented asymmetric lesion",
        "observations": [
            {
                "id": "obs_1",
                "concept": "irregular_border",
                "status": "present",
                "provenance": "clinical_photo",
                "scope": "index_lesion",
                "confidence": "moderate",
                "evidence_region": "lesion_periphery",
            },
            {
                "id": "obs_2",
                "concept": "color_variation",
                "status": "present",
                "provenance": "clinical_photo",
                "scope": "index_lesion",
                "confidence": "moderate",
                "evidence_region": "whole_lesion",
            },
        ],
        "not_assessable_features": ["recent_evolution"],
        "differential": [
            {
                "rank": 1,
                "disease_id": "D001",
                "supporting_observation_ids": ["obs_1", "obs_2"],
                "contradicting_observation_ids": [],
                "missing_discriminators": [
                    {
                        "feature": "recent_evolution",
                        "required_source": "history",
                    }
                ],
                "diagnostic_confidence": "moderate",
                "clinical_risk_if_missed": "high",
            },
            {
                "rank": 2,
                "disease_id": "D002",
                "supporting_observation_ids": ["obs_2"],
                "contradicting_observation_ids": ["obs_1"],
                "missing_discriminators": [],
                "diagnostic_confidence": "low",
                "clinical_risk_if_missed": "low",
            },
        ],
        "action": "REQUEST_CLINICAL_CONTEXT",
        "action_urgency": "prompt",
        "requested_information": "duration_and_recent_change",
        "concise_clinical_rationale": (
            "The irregular border and color variation support an atypical "
            "pigmented lesion, but recent evolution is not visible."
        ),
    }


def _target() -> StructuredClinicalTarget:
    return StructuredClinicalTarget.model_validate(_target_payload())


def _sample(
    *,
    disease_id: str = "D001",
    label: str = "melanoma",
) -> E3StructuredSample:
    return E3StructuredSample(
        sample_id="sample-001",
        leakage_group_id="case-001",
        disease_id=disease_id,
        label=label,
        image=Image.new("RGB", (32, 16), "red"),
        target=_target(),
    )


if __name__ == "__main__":
    unittest.main()
