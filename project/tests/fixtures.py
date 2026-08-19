"""Valid Stage A/B payloads for unit tests."""

from project.teacher.client import TeacherResponse
from project.teacher.schemas import StageAMorphology, parse_stage_a

STAGE_A_PAYLOAD: dict[str, object] = {
    "image_quality": "evaluable",
    "modality": "clinical",
    "primary_lesion": "macule",
    "size": "few_cm",
    "color": ["brown", "black"],
    "shape": "asymmetric",
    "border": "irregular",
    "surface": "flat",
    "secondary_morphology": [],
    "configuration": "solitary",
    "distribution": "localized",
    "additional_features": ["heterogeneous pigmentation"],
}

STAGE_B_PAYLOAD: dict[str, object] = {
    "differential_diagnosis": [
        {
            "rank": 1,
            "disease": "melanoma",
            "supporting": [
                {"field": "shape", "value": "asymmetric"},
                {"field": "border", "value": "irregular"},
            ],
            "contradicting": [],
            "missing": ["duration_and_evolution"],
        },
        {
            "rank": 2,
            "disease": "atypical nevus",
            "supporting": [{"field": "configuration", "value": "solitary"}],
            "contradicting": [{"field": "shape", "value": "asymmetric"}],
            "missing": [],
        },
    ],
    "reasoning": (
        "Asymmetry and irregular border favor melanoma over an atypical nevus."
    ),
    "diagnosis": "melanoma",
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
