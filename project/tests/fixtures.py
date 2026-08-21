"""Valid Stage A/B payloads for unit tests."""

from project.teacher.client import TeacherResponse
from project.teacher.schemas import StageAMorphology, parse_stage_a

STAGE_A_PAYLOAD: dict[str, object] = {
    "image_assessment": {
        "is_evaluable": True,
        "image_modality": "clinical",
        "views_available": ["close clinical view"],
        "quality_defects": ["tight_crop"],
        "has_anatomic_overview": False,
        "has_scale": False,
        "has_lateral_profile": False,
        "distribution_assessability": "partial",
        "color_reliability": "reliable",
    },
    "dominant_visual_pattern": "Solitary asymmetric variably pigmented macule.",
    "observations": [
        {
            "id": "obs_001",
            "concept_id": "lesion.primary",
            "value": "macule",
            "status": "present",
            "scope": "central lesion",
            "confidence": "high",
            "evidence_region": "central pigmented lesion",
        },
        {
            "id": "obs_002",
            "concept_id": "lesion.color",
            "value": "brown and black",
            "status": "present",
            "scope": "central lesion",
            "confidence": "high",
            "evidence_region": "throughout the central lesion",
        },
        {
            "id": "obs_003",
            "concept_id": "lesion.symmetry",
            "value": "asymmetric",
            "status": "present",
            "scope": "central lesion",
            "confidence": "high",
            "evidence_region": "overall lesion silhouette",
        },
        {
            "id": "obs_004",
            "concept_id": "lesion.border_regularity",
            "value": "irregular",
            "status": "present",
            "scope": "central lesion",
            "confidence": "high",
            "evidence_region": "peripheral lesion margin",
        },
        {
            "id": "obs_005",
            "concept_id": "lesion.profile",
            "value": "flat",
            "status": "uncertain",
            "scope": "single frontal image",
            "confidence": "low",
            "evidence_region": None,
        },
        {
            "id": "obs_006",
            "concept_id": "lesion.configuration",
            "value": "solitary",
            "status": "present",
            "scope": "visible field only",
            "confidence": "moderate",
            "evidence_region": "full visible field",
        },
    ],
    "not_assessable_features": [
        "absolute lesion size",
        "full-body distribution",
        "lateral profile",
    ],
    "clinical_caption": (
        "A solitary asymmetric brown-black macule has an irregular peripheral margin."
    ),
}

STAGE_B_PAYLOAD: dict[str, object] = {
    "anchor_evidence_status": "supported",
    "annotation_conflict": False,
    "annotation_conflict_reason": None,
    "diagnostic_confidence": "moderate",
    "diagnosis": "melanoma",
    "differential_comparisons": [
        {
            "alternative": "atypical nevus",
            "features_favoring_diagnosis": ["obs_002", "obs_003", "obs_004"],
            "features_favoring_alternative": ["obs_001", "obs_006"],
            "comparison": (
                "Melanoma is favored over an atypical nevus by the marked "
                "asymmetry, irregular margin, and brown-black color variation."
            ),
        },
    ],
    "limitations": ["duration_and_evolution", "dermoscopy"],
    "response_policy": "ANSWER_DIFFERENTIAL",
    "non_evaluable_reason": None,
    "clinical_reasoning": (
        "The visible asymmetric brown-black macule with an irregular margin "
        "supports melanoma with moderate confidence. Melanoma is favored over "
        "an atypical nevus because the asymmetry, border irregularity, and color "
        "variation are more concerning, although the macular and solitary "
        "presentation remains compatible with that alternative. Evolution and "
        "dermoscopic structures cannot be assessed from this image."
    ),
}


def stage_a_morphology() -> StageAMorphology:
    """Return a valid frozen Stage A record."""
    return parse_stage_a(STAGE_A_PAYLOAD)


def fake_response(payload: dict[str, object]) -> TeacherResponse:
    """Build a TeacherResponse around a JSON object."""
    return TeacherResponse(
        content_json=payload,
        raw_content="{}",
        usage=None,
        finish_reason="stop",
        native_finish_reason="stop",
    )
