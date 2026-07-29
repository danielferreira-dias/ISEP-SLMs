"""Deterministic validation for evidence-grounded benchmark responses.

The benchmark deliberately separates three levels of validity:

* JSON validity: the final answer can be decoded as one JSON value.
* Schema validity: the decoded object has the exact required shape and types.
* Semantic validity: cross-field rules and description restrictions hold.

Reasoning is never inspected. Backends must expose reasoning and final answer
separately; the final-answer channel itself is parsed as strict JSON without
repair or text extraction.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import math
import re
from types import MappingProxyType
from typing import Any

from src.benchmark.runner import ModelResponse


# These aliases are intentionally conservative. They support deterministic
# concept extraction from English descriptions without introducing a model-
# based metric or a dependency on a mutable external terminology service.
MORPHOLOGY_ALIASES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "vesicle": ("vesicle", "vesicles", "vesicular"),
        "papule": ("papule", "papules", "papular"),
        "macule": ("macule", "macules", "macular"),
        "plaque": ("plaque", "plaques"),
        "abscess": ("abscess", "abscesses"),
        "pustule": ("pustule", "pustules", "pustular"),
        "bulla": ("bulla", "bullae", "bullous"),
        "patch": ("patch", "patches"),
        "nodule": ("nodule", "nodules", "nodular"),
        "ulcer": ("ulcer", "ulcers", "ulcerated", "ulceration"),
        "crust": ("crust", "crusts", "crusted", "crusting"),
        "erosion": ("erosion", "erosions", "eroded"),
        "excoriation": (
            "excoriation",
            "excoriations",
            "excoriated",
        ),
        "atrophy": ("atrophy", "atrophic"),
        "exudate": ("exudate", "exudates", "exudative", "oozing"),
        "purpura_petechiae": (
            "purpura",
            "purpuric",
            "petechia",
            "petechiae",
            "petechial",
        ),
        "fissure": ("fissure", "fissures", "fissured"),
        "induration": ("induration", "indurated"),
        "xerosis": ("xerosis", "xerotic"),
        "telangiectasia": (
            "telangiectasia",
            "telangiectasias",
            "telangiectatic",
        ),
        "scale": ("scale", "scales", "scaling", "scaly"),
        "scar": ("scar", "scars", "scarred", "scarring"),
        "friable": ("friable", "friability"),
        "sclerosis": ("sclerosis", "sclerotic"),
        "pedunculated": ("pedunculated", "peduncle"),
        "exophytic_fungating": (
            "exophytic",
            "fungating",
            "fungation",
        ),
        "warty_papillomatous": (
            "warty",
            "verrucous",
            "papillomatous",
        ),
        "dome_shaped": ("dome shaped", "domed"),
        "flat_topped": ("flat topped", "flat top"),
        "brown_hyperpigmentation": (
            "brown",
            "hyperpigmentation",
            "hyperpigmented",
        ),
        "translucent": ("translucent", "translucency"),
        "white_hypopigmentation": (
            "white",
            "hypopigmentation",
            "hypopigmented",
            "depigmented",
        ),
        "purple": ("purple", "violaceous"),
        "yellow": ("yellow", "yellowish"),
        "black": ("black", "blackish"),
        "erythema": ("erythema", "erythematous", "redness"),
        "comedo": ("comedo", "comedones", "comedonal"),
        "lichenification": ("lichenification", "lichenified"),
        "blue": ("blue", "bluish"),
        "umbilicated": ("umbilicated", "umbilication"),
        "poikiloderma": ("poikiloderma", "poikilodermatous"),
        "salmon": ("salmon", "salmon colored", "salmon coloured"),
        "wheal": ("wheal", "wheals", "urticarial"),
        "acuminate": ("acuminate", "pointed"),
        "burrow": ("burrow", "burrows"),
        "gray": ("gray", "grey", "grayish", "greyish"),
        "pigmented": ("pigmented", "pigmentation"),
        "cyst": ("cyst", "cysts", "cystic"),
    }
)


FORBIDDEN_DESCRIPTION_LEXICON: Mapping[str, tuple[str, ...]] = (
    MappingProxyType(
        {
            "diagnostic_language": (
                "diagnosis",
                "diagnoses",
                "diagnose",
                "diagnosed",
                "diagnostic",
                "differential",
                "suggestive of",
                "consistent with",
                "compatible with",
                "likely represents",
                "suspicious for",
            ),
            "test_or_nonvisual_evidence": (
                "biopsy",
                "histology",
                "histopathology",
                "pathology",
                "dermoscopy",
                "dermatoscopy",
                "laboratory",
                "lab test",
                "blood test",
                "wood lamp",
                "microscopy",
                "culture",
            ),
            "treatment_or_management": (
                "treat",
                "treatment",
                "therapy",
                "medication",
                "prescribe",
                "topical steroid",
                "antibiotic",
                "antifungal",
                "surgery",
                "excision",
                "follow up",
                "monitoring",
                "management",
                "recommend",
                "recommended",
                "recommendation",
                "advise",
                "advised",
                "consult",
                "referral",
                "refer to",
                "seek medical",
                "see a doctor",
                "dermatologist",
            ),
            "nonvisual_context": (
                "patient reports",
                "patient is",
                "history of",
                "duration",
                "year old",
                "male patient",
                "female patient",
                "for several days",
                "for several weeks",
                "for several months",
                "itch",
                "itching",
                "itchy",
                "pruritus",
                "asymptomatic",
                "pain",
                "painful",
                "tender",
                "burning",
            ),
        }
    )
)


_FINDING_ID_PATTERN = re.compile(r"^F[1-9][0-9]*$")
_NEGATION_PATTERN = re.compile(
    r"(?:\bno\b|\bnot\b|\bwithout\b|\babsence\s+of\b|"
    r"\babsent\b|\bfree\s+of\b|\blacks?\b)"
    r"(?:\s+\w+){0,3}\s*$",
    flags=re.IGNORECASE,
)


def extract_morphology_concepts(
    description: str,
    *,
    allowed_concept_ids: Iterable[str] | None = None,
) -> set[str]:
    """Extract controlled morphology IDs from an English description.

    Matching uses frozen lexical aliases, token boundaries, and a small
    deterministic negation window. It is intentionally transparent and
    reproducible; it is not a semantic-similarity model.
    """

    if not isinstance(description, str) or not description.strip():
        return set()
    allowed = (
        set(MORPHOLOGY_ALIASES)
        if allowed_concept_ids is None
        else {str(value) for value in allowed_concept_ids}
    )
    normalized = _normalize_text(description)
    extracted: set[str] = set()
    for concept_id, aliases in MORPHOLOGY_ALIASES.items():
        if concept_id not in allowed:
            continue
        for alias in aliases:
            match = _alias_pattern(alias).search(normalized)
            if match is None:
                continue
            prefix = normalized[max(0, match.start() - 60) : match.start()]
            if _NEGATION_PATTERN.search(prefix):
                continue
            extracted.add(concept_id)
            break
    return extracted


def find_forbidden_description_content(
    description: str,
    *,
    disease_terms: Iterable[str] = (),
) -> dict[str, tuple[str, ...]]:
    """Return matched forbidden terms grouped by deterministic rule category."""

    if not isinstance(description, str) or not description.strip():
        return {}
    normalized = _normalize_text(description)
    matches: dict[str, tuple[str, ...]] = {}
    for category, terms in FORBIDDEN_DESCRIPTION_LEXICON.items():
        found = tuple(
            term
            for term in terms
            if _alias_pattern(term).search(normalized)
        )
        if found:
            matches[category] = found
    disease_matches = tuple(
        sorted(
            {
                str(term)
                for term in disease_terms
                if str(term).strip()
                and _alias_pattern(str(term)).search(normalized)
            },
            key=str.casefold,
        )
    )
    if disease_matches:
        matches["disease_name_or_id"] = disease_matches
    return matches


def parse_and_validate_evidence_response(
    *,
    model_id: str,
    raw_text: str,
    allowed_disease_ids: set[str],
    allowed_concept_ids: set[str],
    top_k: int = 6,
    disease_terms: Iterable[str] = (),
    reasoning_text: str | None = None,
) -> ModelResponse:
    """Parse and strictly validate one evidence-grounded final answer.

    ``reasoning_text`` is accepted only so backends can pass their separated
    reasoning channel without special-casing this validator. Its content is
    deliberately ignored and is neither stored nor searched.
    """

    del reasoning_text
    try:
        decoded = json.loads(
            raw_text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except (TypeError, ValueError) as exc:
        return ModelResponse(
            model_id=model_id,
            raw_text=raw_text,
            parsed_output=None,
            json_valid=False,
            schema_valid=False,
            validation_errors=[f"invalid_json:{exc}"],
            metadata={
                "semantic_valid": False,
                "schema_errors": (),
                "semantic_errors": (),
                "description_concept_ids": (),
                "audit": _empty_audit(),
            },
        )

    if not isinstance(decoded, dict):
        return ModelResponse(
            model_id=model_id,
            raw_text=raw_text,
            parsed_output=None,
            json_valid=True,
            schema_valid=False,
            validation_errors=["schema:root_must_be_object"],
            metadata={
                "semantic_valid": False,
                "schema_errors": ("root_must_be_object",),
                "semantic_errors": (),
                "description_concept_ids": (),
                "audit": _empty_audit(),
            },
        )

    schema_errors, facts = _validate_schema(
        decoded,
        allowed_disease_ids=allowed_disease_ids,
        allowed_concept_ids=allowed_concept_ids,
        top_k=top_k,
    )
    forbidden = find_forbidden_description_content(
        facts["description"],
        disease_terms=(
            *allowed_disease_ids,
            *(str(value) for value in disease_terms),
        ),
    )
    description_concepts = extract_morphology_concepts(
        facts["description"],
        allowed_concept_ids=allowed_concept_ids,
    )
    semantic_errors = _validate_semantics(
        facts,
        top_k=top_k,
        description_concepts=description_concepts,
        forbidden=forbidden,
    )
    schema_valid = not schema_errors
    semantic_valid = schema_valid and not semantic_errors
    audit = {
        "invalid_disease_id": bool(facts["invalid_disease_ids"]),
        "invalid_concept_id": bool(facts["invalid_concept_ids"]),
        "broken_evidence_reference": bool(
            facts["broken_evidence_references"]
        ),
        "duplicate_prediction": len(facts["disease_ids"])
        != len(set(facts["disease_ids"])),
        "duplicate_finding": (
            len(facts["finding_ids"]) != len(set(facts["finding_ids"]))
            or len(facts["concept_ids"]) != len(set(facts["concept_ids"]))
        ),
        "forbidden_description_content": bool(forbidden),
        "evidence_reference_count": facts["evidence_reference_count"],
        "valid_evidence_reference_count": (
            facts["valid_evidence_reference_count"]
        ),
        "forbidden_description_categories": tuple(sorted(forbidden)),
    }
    errors = [
        *(f"schema:{error}" for error in schema_errors),
        *(f"semantic:{error}" for error in semantic_errors),
    ]
    return ModelResponse(
        model_id=model_id,
        raw_text=raw_text,
        parsed_output=decoded,
        json_valid=True,
        schema_valid=schema_valid,
        validation_errors=errors,
        metadata={
            "semantic_valid": semantic_valid,
            "schema_errors": tuple(schema_errors),
            "semantic_errors": tuple(semantic_errors),
            "description_concept_ids": tuple(sorted(description_concepts)),
            "audit": audit,
        },
    )


def _validate_schema(
    output: dict[str, Any],
    *,
    allowed_disease_ids: set[str],
    allowed_concept_ids: set[str],
    top_k: int,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    required_root_fields = {
        "findings",
        "clinical_description",
        "differential",
        "case_confidence",
    }
    if set(output) != required_root_fields:
        errors.append("root_fields_invalid")

    facts: dict[str, Any] = {
        "finding_ids": [],
        "concept_ids": [],
        "finding_confidences": [],
        "disease_ids": [],
        "ranks": [],
        "disease_confidences": [],
        "supporting_finding_ids": [],
        "description": "",
        "case_confidence": output.get("case_confidence"),
        "invalid_disease_ids": [],
        "invalid_concept_ids": [],
        "broken_evidence_references": [],
        "evidence_reference_count": 0,
        "valid_evidence_reference_count": 0,
    }

    findings = output.get("findings")
    if not isinstance(findings, list):
        errors.append("findings_must_be_array")
        findings = []
    elif not 1 <= len(findings) <= 48:
        errors.append("finding_count_must_be_between_1_and_48")
    for index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            errors.append(f"finding_{index}_must_be_object")
            continue
        if set(finding) != {"finding_id", "concept_id", "confidence"}:
            errors.append(f"finding_{index}_fields_invalid")
        finding_id = finding.get("finding_id")
        concept_id = finding.get("concept_id")
        confidence = finding.get("confidence")
        if not isinstance(finding_id, str) or not _FINDING_ID_PATTERN.fullmatch(
            finding_id
        ):
            errors.append(f"finding_{index}_finding_id_invalid")
        else:
            facts["finding_ids"].append(finding_id)
        if not isinstance(concept_id, str):
            errors.append(f"finding_{index}_concept_id_invalid")
        else:
            facts["concept_ids"].append(concept_id)
            if concept_id not in allowed_concept_ids:
                errors.append(f"finding_{index}_concept_id_unknown")
                facts["invalid_concept_ids"].append(concept_id)
        if not _is_probability(confidence):
            errors.append(f"finding_{index}_confidence_invalid")
        else:
            facts["finding_confidences"].append(float(confidence))

    description = output.get("clinical_description")
    if not isinstance(description, str):
        errors.append("clinical_description_must_be_string")
    elif not 1 <= len(description) <= 1000:
        errors.append("clinical_description_length_invalid")
        facts["description"] = description
    else:
        facts["description"] = description

    differential = output.get("differential")
    if not isinstance(differential, list):
        errors.append("differential_must_be_array")
        differential = []
    elif len(differential) != top_k:
        errors.append(f"differential_count_must_equal_{top_k}")
    for index, diagnosis in enumerate(differential):
        if not isinstance(diagnosis, dict):
            errors.append(f"differential_{index}_must_be_object")
            continue
        if set(diagnosis) != {
            "rank",
            "disease_id",
            "confidence",
            "supporting_finding_ids",
        }:
            errors.append(f"differential_{index}_fields_invalid")
        rank = diagnosis.get("rank")
        disease_id = diagnosis.get("disease_id")
        confidence = diagnosis.get("confidence")
        support_ids = diagnosis.get("supporting_finding_ids")
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or not 1 <= rank <= top_k
        ):
            errors.append(f"differential_{index}_rank_invalid")
        else:
            facts["ranks"].append(rank)
        if not isinstance(disease_id, str):
            errors.append(f"differential_{index}_disease_id_invalid")
        else:
            facts["disease_ids"].append(disease_id)
            if disease_id not in allowed_disease_ids:
                errors.append(f"differential_{index}_disease_id_unknown")
                facts["invalid_disease_ids"].append(disease_id)
        if not _is_probability(confidence):
            errors.append(f"differential_{index}_confidence_invalid")
        else:
            facts["disease_confidences"].append(float(confidence))
        valid_support_values: list[str] = []
        if not isinstance(support_ids, list):
            errors.append(
                f"differential_{index}_supporting_finding_ids_must_be_array"
            )
            support_ids = []
        elif not 1 <= len(support_ids) <= 48:
            errors.append(
                f"differential_{index}_supporting_finding_count_invalid"
            )
        for support_index, support_id in enumerate(support_ids):
            if (
                not isinstance(support_id, str)
                or not _FINDING_ID_PATTERN.fullmatch(support_id)
            ):
                errors.append(
                    f"differential_{index}_support_{support_index}_invalid"
                )
            else:
                valid_support_values.append(support_id)
        if len(valid_support_values) != len(set(valid_support_values)):
            errors.append(
                f"differential_{index}_supporting_finding_ids_not_unique"
            )
        facts["supporting_finding_ids"].append(valid_support_values)

    case_confidence = output.get("case_confidence")
    if case_confidence not in {"low", "moderate", "high"}:
        errors.append("case_confidence_invalid")

    declared_ids = set(facts["finding_ids"])
    all_support_ids = [
        support_id
        for support_ids in facts["supporting_finding_ids"]
        for support_id in support_ids
    ]
    facts["evidence_reference_count"] = len(all_support_ids)
    facts["valid_evidence_reference_count"] = sum(
        support_id in declared_ids for support_id in all_support_ids
    )
    facts["broken_evidence_references"] = [
        support_id
        for support_id in all_support_ids
        if support_id not in declared_ids
    ]
    return errors, facts


def _validate_semantics(
    facts: dict[str, Any],
    *,
    top_k: int,
    description_concepts: set[str],
    forbidden: Mapping[str, tuple[str, ...]],
) -> list[str]:
    errors: list[str] = []
    finding_ids = facts["finding_ids"]
    concept_ids = facts["concept_ids"]
    disease_ids = facts["disease_ids"]
    ranks = facts["ranks"]
    confidences = facts["disease_confidences"]

    if finding_ids != [
        f"F{index}" for index in range(1, len(finding_ids) + 1)
    ]:
        errors.append("finding_ids_must_be_consecutive")
    if len(concept_ids) != len(set(concept_ids)):
        errors.append("concept_ids_must_be_unique")
    if len(disease_ids) != len(set(disease_ids)):
        errors.append("disease_ids_must_be_unique")
    if ranks != list(range(1, top_k + 1)):
        errors.append("ranks_must_be_consecutive")
    if any(
        later > earlier
        for earlier, later in zip(confidences, confidences[1:])
    ):
        errors.append("disease_confidence_must_be_non_increasing")
    if any(
        not support_ids
        for support_ids in facts["supporting_finding_ids"]
    ):
        errors.append("every_diagnosis_must_cite_evidence")
    if facts["broken_evidence_references"]:
        errors.append("supporting_finding_id_must_resolve")
    if forbidden:
        errors.append("clinical_description_contains_forbidden_content")
    if not facts["description"].strip():
        errors.append("clinical_description_must_not_be_blank")
    if description_concepts - set(concept_ids):
        errors.append("description_concepts_must_be_declared_findings")

    if confidences:
        expected_band = _confidence_band(confidences[0])
        if facts["case_confidence"] != expected_band:
            errors.append("case_confidence_must_match_top_confidence")
    return errors


def _confidence_band(confidence: float) -> str:
    if confidence < 0.40:
        return "low"
    if confidence < 0.75:
        return "moderate"
    return "high"


def _is_probability(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non_standard_json_constant:{value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate_json_key:{key}")
        output[key] = value
    return output


def _normalize_text(text: str) -> str:
    normalized = text.casefold().replace("_", " ").replace("-", " ")
    normalized = re.sub(r"[/\\]", " ", normalized)
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _alias_pattern(alias: str) -> re.Pattern[str]:
    normalized = _normalize_text(alias)
    parts = [re.escape(part) for part in normalized.split()]
    return re.compile(r"\b" + r"\s+".join(parts) + r"\b")


def _empty_audit() -> dict[str, Any]:
    return {
        "invalid_disease_id": False,
        "invalid_concept_id": False,
        "broken_evidence_reference": False,
        "duplicate_prediction": False,
        "duplicate_finding": False,
        "forbidden_description_content": False,
        "evidence_reference_count": 0,
        "valid_evidence_reference_count": 0,
        "forbidden_description_categories": (),
    }
