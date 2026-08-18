"""CPU-only tests for E3 Stage-A/B review and task-isolated targets."""

from __future__ import annotations

import hashlib
import json
import unittest
from copy import deepcopy

from PIL import Image
from pydantic import ValidationError

from src.train.domain import Taxonomy, TaxonomyClass
from src.train.e3 import (
    GROUNDED_DIFFERENTIAL_TEMPLATES,
    DeterministicContextPolicyRenderer,
    DeterministicGroundedDifferentialRenderer,
    E3HardKDPhase,
    E3TrainingSample,
    E3TrainingVariant,
    StageATarget,
    StageBTarget,
    TeacherTargetBundle,
    canonical_stage_a_json,
)


class E3DomainTests(unittest.TestCase):
    def test_stage_a_json_is_frozen_complete_and_stable(self) -> None:
        target = _stage_a()
        self.assertIsInstance(target.observations, tuple)
        first = canonical_stage_a_json(target)
        second_target = StageATarget.model_validate(
            {**json.loads(first), "clinical_caption": target.clinical_caption}
        )
        self.assertEqual(first, canonical_stage_a_json(second_target))
        self.assertNotIn("clinical_caption", first)
        self.assertNotIn("NaN", first)

    def test_incomplete_stage_a_caption_is_rejected(self) -> None:
        payload = _stage_a_payload()
        payload["clinical_caption"] = "Well-defined white patches. It is"
        with self.assertRaisesRegex(ValidationError, "sentence boundary"):
            StageATarget.model_validate(payload)

        payload["clinical_caption"] = "Well-defined white patches which is."
        with self.assertRaisesRegex(ValidationError, "incomplete clause"):
            StageATarget.model_validate(payload)

    def test_duplicate_observation_ids_are_rejected(self) -> None:
        payload = _stage_a_payload()
        observations = payload["observations"]
        assert isinstance(observations, list)
        observations.append(dict(observations[0]))
        with self.assertRaisesRegex(ValidationError, "Observation IDs must be unique"):
            StageATarget.model_validate(payload)

    def test_unknown_cross_stage_evidence_link_is_rejected(self) -> None:
        stage_b = _stage_b_payload()
        assessment = stage_b["diagnostic_assessment"]
        assert isinstance(assessment, dict)
        differential = assessment["differential"]
        assert isinstance(differential, list)
        first = differential[0]
        assert isinstance(first, dict)
        first["supporting_observation_ids"] = ["obs_unknown"]
        payload = _accepted_bundle_payload(stage_b=stage_b)
        with self.assertRaisesRegex(ValidationError, "unknown Stage-A"):
            TeacherTargetBundle.model_validate(payload)

    def test_noncontiguous_differential_ranks_are_rejected(self) -> None:
        payload = _stage_b_payload()
        assessment = payload["diagnostic_assessment"]
        assert isinstance(assessment, dict)
        differential = assessment["differential"]
        assert isinstance(differential, list)
        second = differential[1]
        assert isinstance(second, dict)
        second["rank"] = 3
        with self.assertRaisesRegex(ValidationError, "ranks must be contiguous"):
            StageBTarget.model_validate(payload)

    def test_insufficient_context_requires_an_explicit_question(self) -> None:
        payload = _stage_b_payload()
        decision = payload["context_decision"]
        assert isinstance(decision, dict)
        decision["requests"] = []
        with self.assertRaisesRegex(ValidationError, "explicit request"):
            StageBTarget.model_validate(payload)

    def test_sufficient_context_cannot_request_more_information(self) -> None:
        payload = _stage_b_payload()
        decision = payload["context_decision"]
        assert isinstance(decision, dict)
        decision.update(
            {
                "information_sufficiency": "sufficient",
                "response_policy": "ANSWER_DIFFERENTIAL",
            }
        )
        with self.assertRaisesRegex(ValidationError, "cannot request context"):
            StageBTarget.model_validate(payload)

    def test_context_request_must_reference_the_differential(self) -> None:
        payload = _stage_b_payload()
        decision = payload["context_decision"]
        assert isinstance(decision, dict)
        requests = decision["requests"]
        assert isinstance(requests, list)
        request = requests[0]
        assert isinstance(request, dict)
        request["discriminates_between"] = ["D001", "D003"]
        with self.assertRaisesRegex(ValidationError, "outside the differential"):
            StageBTarget.model_validate(payload)

    def test_context_question_must_be_a_complete_question(self) -> None:
        payload = _stage_b_payload()
        decision = payload["context_decision"]
        assert isinstance(decision, dict)
        requests = decision["requests"]
        assert isinstance(requests, list)
        request = requests[0]
        assert isinstance(request, dict)
        request["question"] = "Has the lesion changed in size or color recently"
        with self.assertRaisesRegex(ValidationError, "question mark"):
            StageBTarget.model_validate(payload)

    def test_accepted_stage_b_requires_accepted_stage_a(self) -> None:
        payload = _accepted_bundle_payload()
        payload["stage_a_status"] = "rejected"
        payload["stage_a_rejection_reasons"] = ["unsupported finding"]
        with self.assertRaisesRegex(ValidationError, "requires accepted Stage A"):
            TeacherTargetBundle.model_validate(payload)

    def test_rejected_stage_b_still_requires_accepted_stage_a(self) -> None:
        payload = _accepted_bundle_payload()
        payload.update(
            {
                "stage_a_status": "rejected",
                "stage_a_target": None,
                "stage_a_rejection_reasons": ["unsupported finding"],
                "stage_b_status": "rejected",
                "stage_b_target": None,
                "stage_b_rejection_reasons": ["invalid differential"],
            }
        )
        with self.assertRaisesRegex(ValidationError, "requires accepted Stage A"):
            TeacherTargetBundle.model_validate(payload)

    def test_stage_a_teacher_output_must_be_answer_blind(self) -> None:
        payload = _accepted_bundle_payload()
        provenance = payload["stage_a_provenance"]
        assert isinstance(provenance, dict)
        provenance["gold_visible_to_teacher"] = True
        with self.assertRaisesRegex(ValidationError, "must be answer-blind"):
            TeacherTargetBundle.model_validate(payload)

    def test_stage_b_teacher_output_may_be_gold_conditioned(self) -> None:
        payload = _accepted_bundle_payload()
        provenance = payload["stage_b_provenance"]
        assert isinstance(provenance, dict)
        provenance["gold_visible_to_teacher"] = True

        bundle = TeacherTargetBundle.model_validate(payload)

        assert bundle.stage_b_provenance is not None
        self.assertTrue(bundle.stage_b_provenance.gold_visible_to_teacher)

    def test_stage_a_safety_refusal_is_typed_and_blocks_stage_b(self) -> None:
        bundle = TeacherTargetBundle.model_validate(
            {
                "stage_a_status": "not_applicable",
                "stage_a_target": None,
                "stage_a_provenance": _safety_refusal_provenance("stage-a"),
                "stage_a_rejection_reasons": [],
                "stage_b_status": "not_generated",
                "stage_b_target": None,
                "stage_b_provenance": None,
                "stage_b_rejection_reasons": [],
            }
        )
        assert bundle.stage_a_provenance is not None
        self.assertEqual(bundle.stage_a_status.value, "not_applicable")
        self.assertEqual(
            bundle.stage_a_provenance.generation_status.value,
            "provider_safety_refusal",
        )
        self.assertEqual(bundle.stage_b_status.value, "not_generated")

    def test_stage_b_safety_refusal_preserves_stage_a_tasks(self) -> None:
        bundle = _stage_b_safety_refusal_bundle()
        sample = _sample(bundle=bundle)
        for variant in (E3TrainingVariant.MORPHOLOGY, E3TrainingVariant.CAPTION):
            E3HardKDPhase(_taxonomy(), variant).format_example(sample)
        for variant in (
            E3TrainingVariant.GROUNDED_DIFFERENTIAL,
            E3TrainingVariant.CONTEXT_POLICY,
        ):
            with self.assertRaisesRegex(ValueError, "requires accepted Stage B"):
                E3HardKDPhase(_taxonomy(), variant).format_example(sample)

        assert bundle.stage_b_provenance is not None
        self.assertEqual(bundle.stage_b_status.value, "not_applicable")
        self.assertEqual(
            bundle.stage_b_provenance.provider_error_code,
            "content_filter",
        )
        self.assertEqual(
            bundle.stage_b_provenance.safety_categories[0].category,
            "medical",
        )

    def test_safety_refusal_cannot_be_scientifically_rejected(self) -> None:
        payload = _accepted_bundle_payload()
        payload.update(
            {
                "stage_b_status": "rejected",
                "stage_b_target": None,
                "stage_b_provenance": _safety_refusal_provenance("stage-b"),
                "stage_b_rejection_reasons": ["provider blocked the request"],
            }
        )
        with self.assertRaisesRegex(ValidationError, "successful generation"):
            TeacherTargetBundle.model_validate(payload)

    def test_not_applicable_requires_failed_generation_without_target(self) -> None:
        payload = _accepted_bundle_payload()
        payload.update(
            {
                "stage_b_status": "not_applicable",
                "stage_b_target": None,
                "stage_b_rejection_reasons": [],
            }
        )
        with self.assertRaisesRegex(ValidationError, "failed generation"):
            TeacherTargetBundle.model_validate(payload)

        payload["stage_b_provenance"] = _safety_refusal_provenance("stage-b")
        payload["stage_b_target"] = _stage_b_payload()
        with self.assertRaisesRegex(ValidationError, "cannot carry target"):
            TeacherTargetBundle.model_validate(payload)

    def test_partial_acceptance_preserves_stage_a(self) -> None:
        bundle = _partial_bundle()
        self.assertEqual(bundle.stage_a_status.value, "accepted")
        self.assertEqual(bundle.stage_b_status.value, "rejected")
        self.assertIsNotNone(bundle.stage_a_target)
        self.assertIsNone(bundle.stage_b_target)


class E3RenderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.renderer = DeterministicGroundedDifferentialRenderer(_taxonomy())
        self.stage_a = _stage_a()
        self.stage_b = _stage_b()

    def test_same_sample_is_byte_deterministic(self) -> None:
        first = self.renderer.render("sample-001", self.stage_a, self.stage_b)
        second = self.renderer.render("sample-001", self.stage_a, self.stage_b)
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
        self.assertEqual(len(GROUNDED_DIFFERENTIAL_TEMPLATES), 12)
        self.assertEqual(selected, set(self.renderer.template_ids))

    def test_every_template_is_complete_and_excludes_d4_action(self) -> None:
        context = self.renderer._context(self.stage_a, self.stage_b)
        outputs = {
            template.template_id: template.render(context)
            for template in GROUNDED_DIFFERENTIAL_TEMPLATES
        }
        self.assertEqual(len(set(outputs.values())), 12)
        for text in outputs.values():
            self.assertTrue(text.endswith("."))
            self.assertIn(self.stage_a.clinical_caption, text)
            self.assertIn("melanoma", text.lower())
            self.assertIn("irregular border", text)
            self.assertIn("color variation", text)
            self.assertIn("melanocytic nevus", text)
            self.assertIn("recent evolution", text)
            self.assertNotIn("REQUEST_CLINICAL_CONTEXT", text)
            self.assertNotIn("clinical context should be obtained", text.lower())

    def test_grounded_target_is_independent_of_context_policy(self) -> None:
        requesting = self.renderer.render("sample-001", self.stage_a, self.stage_b)
        sufficient = self.renderer.render(
            "sample-001",
            self.stage_a,
            StageBTarget.model_validate(_sufficient_stage_b_payload()),
        )
        self.assertEqual(requesting, sufficient)

    def test_out_of_taxonomy_differential_is_rejected_at_render_time(self) -> None:
        payload = _sufficient_stage_b_payload()
        assessment = payload["diagnostic_assessment"]
        assert isinstance(assessment, dict)
        differential = assessment["differential"]
        assert isinstance(differential, list)
        second = differential[1]
        assert isinstance(second, dict)
        second["disease_id"] = "D999"
        with self.assertRaisesRegex(ValueError, "outside the taxonomy"):
            self.renderer.render(
                "sample-001",
                self.stage_a,
                StageBTarget.model_validate(payload),
            )

    def test_context_policy_is_deterministic_and_contains_explicit_question(
        self,
    ) -> None:
        renderer = DeterministicContextPolicyRenderer(_taxonomy())
        first = renderer.render(self.stage_b.context_decision)
        second = renderer.render(self.stage_b.context_decision)
        self.assertEqual(first, second)
        payload = json.loads(first.text)
        self.assertEqual(payload["information_sufficiency"], "insufficient")
        self.assertEqual(payload["response_policy"], "REQUEST_CONTEXT")
        self.assertEqual(
            payload["requests"][0]["question"],
            "Has the lesion changed in size, color, or shape recently?",
        )
        self.assertEqual(
            payload["requests"][0]["discriminates_between"],
            ["melanoma", "melanocytic nevus"],
        )

    def test_sufficient_context_policy_has_no_requests(self) -> None:
        target = StageBTarget.model_validate(_sufficient_stage_b_payload())
        rendered = DeterministicContextPolicyRenderer(_taxonomy()).render(
            target.context_decision
        )
        payload = json.loads(rendered.text)
        self.assertEqual(payload["information_sufficiency"], "sufficient")
        self.assertEqual(payload["response_policy"], "ANSWER_DIFFERENTIAL")
        self.assertEqual(payload["requests"], [])


class E3PhaseTests(unittest.TestCase):
    def test_five_variants_are_separate_tasks_with_explicit_targets(self) -> None:
        records: dict[E3TrainingVariant, dict[str, object]] = {}
        for variant in E3TrainingVariant:
            example = E3HardKDPhase(_taxonomy(), variant).format_example(_sample())
            records[variant] = example.as_record()

        self.assertEqual(len({item["task_id"] for item in records.values()}), 5)
        for variant, record in records.items():
            self.assertEqual(record["phase"], "e3_hard_kd")
            self.assertEqual(record["task"], variant.value)
            messages = record["messages"]
            assert isinstance(messages, list)
            self.assertEqual(messages[0]["content"][0]["type"], "image")
            self.assertEqual(
                messages[1]["content"][0]["text"],
                record["target_text"],
            )

        diagnosis = records[E3TrainingVariant.DIAGNOSIS]
        self.assertEqual(diagnosis["target_text"], "melanoma")
        self.assertIsNone(diagnosis["stage_a_generation_id"])
        self.assertIsNone(diagnosis["stage_b_generation_id"])

        morphology = records[E3TrainingVariant.MORPHOLOGY]
        morphology_sources = morphology["target_source_fields"]
        assert isinstance(morphology_sources, list)
        self.assertIn("stage_a.observations", morphology_sources)
        self.assertIsNotNone(morphology["stage_a_generation_id"])
        self.assertIsNone(morphology["stage_b_generation_id"])

        caption = records[E3TrainingVariant.CAPTION]
        self.assertEqual(caption["target_text"], _stage_a().clinical_caption)

        grounded = records[E3TrainingVariant.GROUNDED_DIFFERENTIAL]
        self.assertIsNotNone(grounded["template_id"])
        self.assertIsNotNone(grounded["stage_b_generation_id"])
        grounded_target = grounded["target_text"]
        assert isinstance(grounded_target, str)
        self.assertNotIn("REQUEST_CLINICAL_CONTEXT", grounded_target)
        self.assertNotIn("REQUEST_CONTEXT", grounded_target)

        context = records[E3TrainingVariant.CONTEXT_POLICY]
        self.assertIsNone(context["template_id"])
        self.assertIsNotNone(context["renderer_version"])
        self.assertIsNotNone(context["stage_a_generation_id"])
        self.assertIsNotNone(context["stage_b_generation_id"])
        context_target = context["target_text"]
        assert isinstance(context_target, str)
        context_payload = json.loads(context_target)
        self.assertEqual(context_payload["response_policy"], "REQUEST_CONTEXT")
        self.assertTrue(context_payload["requests"][0]["question"].endswith("?"))
        self.assertNotIn(_stage_a().clinical_caption, context_target)
        self.assertNotIn("supporting_observation_ids", context_target)

    def test_non_diagnosis_prompts_do_not_expose_gold(self) -> None:
        for variant in (
            E3TrainingVariant.MORPHOLOGY,
            E3TrainingVariant.CAPTION,
            E3TrainingVariant.GROUNDED_DIFFERENTIAL,
            E3TrainingVariant.CONTEXT_POLICY,
        ):
            record = E3HardKDPhase(_taxonomy(), variant).format_example(
                _sample()
            ).as_record()
            messages = record["messages"]
            assert isinstance(messages, list)
            self.assertNotIn("melanoma", messages[0]["content"][1]["text"])

    def test_partial_acceptance_allows_a_tasks_and_blocks_b_tasks(self) -> None:
        sample = _sample(bundle=_partial_bundle())
        for variant in (E3TrainingVariant.MORPHOLOGY, E3TrainingVariant.CAPTION):
            E3HardKDPhase(_taxonomy(), variant).format_example(sample)
        for variant in (
            E3TrainingVariant.GROUNDED_DIFFERENTIAL,
            E3TrainingVariant.CONTEXT_POLICY,
        ):
            with self.assertRaisesRegex(ValueError, "requires accepted Stage B"):
                E3HardKDPhase(_taxonomy(), variant).format_example(sample)

    def test_diagnosis_replay_survives_rejected_teacher_stages(self) -> None:
        example = E3HardKDPhase(
            _taxonomy(), E3TrainingVariant.DIAGNOSIS
        ).format_example(_sample(bundle=_rejected_bundle()))
        self.assertEqual(example.target_text, "melanoma")

    def test_leading_diagnosis_must_match_accepted_gold(self) -> None:
        sample = _sample(disease_id="D002", label="melanocytic nevus")
        with self.assertRaisesRegex(ValueError, "does not match"):
            E3HardKDPhase(
                _taxonomy(), E3TrainingVariant.GROUNDED_DIFFERENTIAL
            ).format_example(sample)

        policy = E3HardKDPhase(
            _taxonomy(), E3TrainingVariant.CONTEXT_POLICY
        ).format_example(sample)
        self.assertEqual(policy.target_variant, E3TrainingVariant.CONTEXT_POLICY)

    def test_stage_a_diagnosis_leak_is_rejected(self) -> None:
        stage_a = _stage_a_payload()
        stage_a["clinical_caption"] = (
            "The photograph visibly demonstrates a melanoma with irregular borders."
        )
        bundle = TeacherTargetBundle.model_validate(
            _accepted_bundle_payload(stage_a=stage_a)
        )
        with self.assertRaisesRegex(ValueError, "canonical diagnosis term"):
            E3HardKDPhase(
                _taxonomy(), E3TrainingVariant.CAPTION
            ).format_example(_sample(bundle=bundle))


def _taxonomy() -> Taxonomy:
    return Taxonomy(
        taxonomy_id="test-taxonomy",
        classes=(
            TaxonomyClass("D001", "melanoma"),
            TaxonomyClass("D002", "melanocytic nevus"),
            TaxonomyClass("D003", "seborrheic keratosis"),
        ),
    )


def _stage_a_payload() -> dict[str, object]:
    return {
        "image_assessment": {
            "is_evaluable": True,
            "image_modality": "clinical_photo",
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
                "concept_id": "border.irregular",
                "concept_label": "irregular border",
                "concept_detail": None,
                "status": "present",
                "provenance": "clinical_photo",
                "scope": "index_lesion",
                "confidence": "moderate",
                "evidence_region": "lesion_periphery",
            },
            {
                "id": "obs_2",
                "concept_id": "color.multicolored",
                "concept_label": "multicolored",
                "concept_detail": "visible color variation",
                "status": "present",
                "provenance": "clinical_photo",
                "scope": "index_lesion",
                "confidence": "moderate",
                "evidence_region": "whole_lesion",
            },
        ],
        "not_assessable_features": ["recent_evolution"],
        "clinical_caption": (
            "The image shows a pigmented asymmetric lesion with an irregular "
            "border and visible color variation."
        ),
    }


def _stage_b_payload() -> dict[str, object]:
    return {
        "stage_b_corrections": [],
        "diagnostic_assessment": {
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
            "concise_clinical_rationale": (
                "The irregular border and color variation support an atypical "
                "pigmented lesion, while recent evolution is not visible."
            ),
        },
        "context_decision": {
            "information_sufficiency": "insufficient",
            "response_policy": "REQUEST_CONTEXT",
            "decision_rationale": (
                "Image-only evidence leaves recent evolution unresolved between "
                "the leading diagnoses."
            ),
            "requests": [
                {
                    "request_id": "ctx_1",
                    "priority": 1,
                    "context_type": "lesion_evolution",
                    "required_source": "clinical_history",
                    "question": (
                        "Has the lesion changed in size, color, or shape recently?"
                    ),
                    "discriminates_between": ["D001", "D002"],
                    "rationale": (
                        "Recent evolution would help distinguish these competing "
                        "pigmented diagnoses."
                    ),
                }
            ],
        },
    }


def _sufficient_stage_b_payload() -> dict[str, object]:
    payload = deepcopy(_stage_b_payload())
    payload["context_decision"] = {
        "information_sufficiency": "sufficient",
        "response_policy": "ANSWER_DIFFERENTIAL",
        "decision_rationale": (
            "The visible findings support a ranked differential without requiring "
            "additional context for this image-only response."
        ),
        "requests": [],
    }
    return payload


def _provenance(stage: str, *, gold_visible: bool = False) -> dict[str, object]:
    return {
        "generation_id": f"generation-{stage}",
        "generation_status": "succeeded",
        "provider": "test_provider",
        "teacher_model": "teacher/model",
        "teacher_revision": "revision-001",
        "prompt_id": f"e3-{stage}-v1",
        "prompt_sha256": "a" * 64,
        "gold_visible_to_teacher": gold_visible,
    }


def _safety_refusal_provenance(stage: str) -> dict[str, object]:
    return {
        "generation_id": f"generation-{stage}",
        "generation_status": "provider_safety_refusal",
        "provider": "azure_openai",
        "teacher_model": "teacher/model",
        "teacher_revision": "revision-001",
        "prompt_id": f"e3-{stage}-v1",
        "prompt_sha256": "a" * 64,
        "provider_response_id": None,
        "provider_request_id": "request-001",
        "finish_reason": "content_filter",
        "provider_error_code": "content_filter",
        "safety_categories": [
            {
                "category": "medical",
                "severity": "blocked",
                "filtered": True,
            }
        ],
        "gold_visible_to_teacher": False,
    }


def _accepted_bundle_payload(
    *,
    stage_a: dict[str, object] | None = None,
    stage_b: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "stage_a_status": "accepted",
        "stage_a_target": stage_a or _stage_a_payload(),
        "stage_a_provenance": _provenance("stage-a"),
        "stage_a_rejection_reasons": [],
        "stage_b_status": "accepted",
        "stage_b_target": stage_b or _stage_b_payload(),
        "stage_b_provenance": _provenance("stage-b"),
        "stage_b_rejection_reasons": [],
    }


def _accepted_bundle() -> TeacherTargetBundle:
    return TeacherTargetBundle.model_validate(_accepted_bundle_payload())


def _partial_bundle() -> TeacherTargetBundle:
    return TeacherTargetBundle.model_validate(
        {
            "stage_a_status": "accepted",
            "stage_a_target": _stage_a_payload(),
            "stage_a_provenance": _provenance("stage-a"),
            "stage_a_rejection_reasons": [],
            "stage_b_status": "rejected",
            "stage_b_target": None,
            "stage_b_provenance": _provenance("stage-b"),
            "stage_b_rejection_reasons": ["leading diagnosis disagrees with gold"],
        }
    )


def _stage_b_safety_refusal_bundle() -> TeacherTargetBundle:
    return TeacherTargetBundle.model_validate(
        {
            "stage_a_status": "accepted",
            "stage_a_target": _stage_a_payload(),
            "stage_a_provenance": _provenance("stage-a"),
            "stage_a_rejection_reasons": [],
            "stage_b_status": "not_applicable",
            "stage_b_target": None,
            "stage_b_provenance": _safety_refusal_provenance("stage-b"),
            "stage_b_rejection_reasons": [],
        }
    )


def _rejected_bundle() -> TeacherTargetBundle:
    return TeacherTargetBundle.model_validate(
        {
            "stage_a_status": "rejected",
            "stage_a_target": None,
            "stage_a_provenance": _provenance("stage-a"),
            "stage_a_rejection_reasons": ["unsupported visible finding"],
            "stage_b_status": "not_generated",
            "stage_b_target": None,
            "stage_b_provenance": None,
            "stage_b_rejection_reasons": [],
        }
    )


def _stage_a() -> StageATarget:
    return StageATarget.model_validate(_stage_a_payload())


def _stage_b() -> StageBTarget:
    return StageBTarget.model_validate(_stage_b_payload())


def _sample(
    *,
    disease_id: str = "D001",
    label: str = "melanoma",
    bundle: TeacherTargetBundle | None = None,
) -> E3TrainingSample:
    return E3TrainingSample(
        sample_id="sample-001",
        leakage_group_id="case-001",
        disease_id=disease_id,
        label=label,
        image=Image.new("RGB", (32, 16), "red"),
        teacher_targets=bundle or _accepted_bundle(),
    )


if __name__ == "__main__":
    unittest.main()
