"""Schemas and frozen prompts for the first ISEPDistillDataset release."""

from __future__ import annotations

import json

from datasets import Features, Image, List, Value

SCHEMA_VERSION = "0.3.0"
CAPTION_SCHEMA_VERSION = "0.4.1"
TAXONOMY_VERSION = "2.2.0"
SKINCON_ONTOLOGY_VERSION = "skincon_48_v1"


def diagnosis_features() -> Features:
    """Return the explicit Arrow/Hugging Face schema for diagnosis rows."""

    return Features(
        {
            "image": Image(decode=True),
            "sample_id": Value("string"),
            "case_id": Value("string"),
            "task_id": Value("string"),
            "image_asset_id": Value("string"),
            "view_type": Value("string"),
            "leakage_group_id": Value("string"),
            "source_dataset": Value("string"),
            "source_sample_id": Value("string"),
            "license_id": Value("string"),
            "split": Value("string"),
            "is_dev_panel": Value("bool"),
            "disease_id": Value("string"),
            "gold_diagnosis": Value("string"),
            "source_label": Value("string"),
            "gold_provenance": Value("string"),
            "taxonomy_version": Value("string"),
            "image_sha256": Value("string"),
            "target_variant": Value("string"),
            "target_source": Value("string"),
            "prompt": Value("string"),
            "prompt_sha256": Value("string"),
            "target_text": Value("string"),
            "schema_version": Value("string"),
            "quality_status": Value("string"),
            "messages": _message_feature(),
        }
    )


def morphology_features() -> Features:
    """Return the explicit Arrow/Hugging Face schema for SKINCON rows."""

    return Features(
        {
            "image": Image(decode=True),
            "sample_id": Value("string"),
            "case_id": Value("string"),
            "task_id": Value("string"),
            "image_asset_id": Value("string"),
            "view_type": Value("string"),
            "leakage_group_id": Value("string"),
            "source_dataset": Value("string"),
            "source_sample_id": Value("string"),
            "license_id": Value("string"),
            "split": Value("string"),
            "split_inherited_from_e1": Value("bool"),
            "disease_id": Value("string"),
            "gold_diagnosis": Value("string"),
            "gold_provenance": Value("string"),
            "taxonomy_version": Value("string"),
            "taxonomy_mapping_status": Value("string"),
            "image_sha256": Value("string"),
            "skincon": {
                "ontology_version": Value("string"),
                "annotation_source": Value("string"),
                "source_subset": Value("string"),
                "source_image_id": Value("string"),
                "positive_concepts": List(Value("string")),
                "all_concepts_annotated": Value("bool"),
            },
            "target_variant": Value("string"),
            "target_source": Value("string"),
            "prompt": Value("string"),
            "prompt_sha256": Value("string"),
            "target_text": Value("string"),
            "schema_version": Value("string"),
            "quality_status": Value("string"),
            "messages": _message_feature(),
        }
    )


def caption_features() -> Features:
    """Return the explicit schema for filtered SkinCAP observation rows."""

    return Features(
        {
            "image": Image(decode=True),
            "sample_id": Value("string"),
            "case_id": Value("string"),
            "task_id": Value("string"),
            "image_asset_id": Value("string"),
            "view_type": Value("string"),
            "leakage_group_id": Value("string"),
            "source_dataset": Value("string"),
            "source_sample_id": Value("string"),
            "license_id": Value("string"),
            "split": Value("string"),
            "split_inherited_from_e1": Value("bool"),
            "split_source": Value("string"),
            "image_sha256": Value("string"),
            "source_caption_sha256": Value("string"),
            "caption_source_revision": Value("string"),
            "caption_variant": Value("string"),
            "transform_version": Value("string"),
            "boundary_kind": Value("string"),
            "target_variant": Value("string"),
            "target_source": Value("string"),
            "prompt": Value("string"),
            "prompt_sha256": Value("string"),
            "target_text": Value("string"),
            "schema_version": Value("string"),
            "quality_status": Value("string"),
            "messages": _message_feature(),
        }
    )


def diagnosis_prompt(labels: tuple[str, ...]) -> str:
    """Render the exact label-only prompt used by the E1 training phase."""

    allowed = "\n".join(f"- {label}" for label in labels)
    return (
        "Classify the clinical dermatology image using the closed taxonomy "
        "below. Return exactly one canonical label and nothing else: no "
        "explanation, reasoning, punctuation, or additional text.\n\n"
        f"Allowed labels:\n{allowed}\n\n/no_think"
    )


def morphology_prompt(concepts: tuple[str, ...]) -> str:
    """Render the frozen answer-blind SKINCON concept prompt."""

    allowed = "\n".join(f"- {concept}" for concept in concepts)
    return (
        "Identify every morphology concept that is visibly present in the "
        "clinical dermatology image. Use only the SKINCON ontology below. "
        "Return one compact JSON object with keys positive_concepts and "
        "all_concepts_annotated; do not diagnose the disease or add prose.\n\n"
        f"Allowed concepts:\n{allowed}\n\n/no_think"
    )


def caption_prompt() -> str:
    """Return the frozen answer-blind short-observation prompt."""

    return (
        "Describe only the visible dermatological findings in the clinical "
        "image using one short clinical sentence. Do not provide a diagnosis, "
        "differential diagnosis, testing, management, prognosis, or advice."
        "\n\n/no_think"
    )


def morphology_target(concepts: tuple[str, ...]) -> str:
    """Serialize one deterministic morphology target."""

    return json.dumps(
        {
            "positive_concepts": list(concepts),
            "all_concepts_annotated": True,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def messages(prompt: str, target: str) -> list[dict[str, object]]:
    """Build a stable two-message multimodal conversation."""

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "text": ""},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": target}],
        },
    ]


def _message_feature() -> List:
    return List(
        {
            "role": Value("string"),
            "content": List(
                {
                    "type": Value("string"),
                    "text": Value("string"),
                }
            ),
        }
    )
