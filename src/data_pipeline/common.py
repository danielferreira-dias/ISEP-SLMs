"""Shared manifest types and disease-label normalization utilities."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


MANIFEST_SCHEMA_VERSION = "1.2.0"

REFERENCE_DIAGNOSIS_TYPE = pa.struct(
    [
        pa.field("disease_original", pa.string(), nullable=False),
        pa.field("disease_id", pa.string()),
        pa.field("canonical_source_label", pa.string(), nullable=False),
        pa.field("mapping_status", pa.string(), nullable=False),
        pa.field("rank", pa.int32(), nullable=False),
        pa.field("weight", pa.float64()),
        pa.field("source", pa.string(), nullable=False),
    ]
)

MANIFEST_ARROW_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("dataset_id", pa.string(), nullable=False),
        pa.field("dataset_version", pa.string(), nullable=False),
        pa.field("original_image_id", pa.string(), nullable=False),
        pa.field("original_case_id", pa.string()),
        pa.field("patient_id", pa.string()),
        pa.field("group_id", pa.string(), nullable=False),
        pa.field("image_uri", pa.string(), nullable=False),
        pa.field("disease_original", pa.string()),
        pa.field("canonical_source_label", pa.string()),
        pa.field("disease_id", pa.string()),
        pa.field(
            "reference_diagnoses",
            pa.list_(REFERENCE_DIAGNOSIS_TYPE),
            nullable=False,
        ),
        pa.field("mapping_status", pa.string(), nullable=False),
        pa.field("diagnosis_basis", pa.string(), nullable=False),
        pa.field("diagnosis_gradable", pa.bool_(), nullable=False),
        pa.field("taxonomy_id", pa.string(), nullable=False),
        pa.field("taxonomy_version", pa.string(), nullable=False),
        pa.field("age_years", pa.int32()),
        pa.field("age_group_source", pa.string()),
        pa.field("age_group_standardized", pa.string()),
        pa.field("age_source", pa.string()),
        pa.field("sex_or_gender", pa.string()),
        pa.field("sex_or_gender_system", pa.string()),
        pa.field("sex_or_gender_source", pa.string()),
        pa.field("race_ethnicity", pa.string()),
        pa.field("race_ethnicity_source", pa.string()),
        pa.field("skin_tone_system", pa.string()),
        pa.field("skin_tone", pa.string()),
        pa.field("skin_tone_source", pa.string()),
        pa.field("image_sha256", pa.string()),
        pa.field("perceptual_hash", pa.string()),
        pa.field("perceptual_hash_algorithm", pa.string()),
        pa.field("license_id", pa.string(), nullable=False),
        pa.field("split", pa.string()),
        pa.field("split_version", pa.string()),
        pa.field("include", pa.bool_(), nullable=False),
        pa.field("exclusion_reason", pa.string()),
        pa.field("source_metadata", pa.string(), nullable=False),
    ]
)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML mapping."""

    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def normalize_label(value: Any) -> str:
    """Normalize punctuation and spacing without performing a clinical mapping."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.casefold().replace("&", " and ")
    text = re.sub(r"[_/\\-]+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class TaxonomyDisease:
    disease_id: str
    canonical_name: str
    display_name: str


class DiseaseMapper:
    """Map source labels to the current candidate disease taxonomy."""

    def __init__(self, taxonomy_path: Path, mapping_path: Path) -> None:
        taxonomy_document = load_yaml(taxonomy_path)
        mapping_document = load_yaml(mapping_path)

        taxonomy = taxonomy_document["taxonomy"]
        self.taxonomy_id = str(taxonomy["id"])
        self.taxonomy_version = str(taxonomy["version"])
        self.diseases = [
            TaxonomyDisease(
                disease_id=str(item["id"]),
                canonical_name=str(item["canonical_name"]),
                display_name=str(item["display_name"]),
            )
            for item in taxonomy_document["diseases"]
        ]
        self.valid_ids = {disease.disease_id for disease in self.diseases}
        self._canonical_by_id = {
            disease.disease_id: disease.canonical_name
            for disease in self.diseases
        }

        self._base_mapping: dict[str, str] = {}
        for item in taxonomy_document["diseases"]:
            disease_id = str(item["id"])
            values = [
                item["canonical_name"],
                item["display_name"],
                *item.get("aliases", []),
            ]
            for value in values:
                self._base_mapping[normalize_label(value)] = disease_id

        self._global_mapping = self._validate_mapping(
            mapping_document.get("global", {}),
            "global",
        )
        self._dataset_mappings = {
            str(dataset_id): self._validate_mapping(values, str(dataset_id))
            for dataset_id, values in mapping_document.get(
                "dataset_specific",
                {},
            ).items()
        }

    def _validate_mapping(
        self,
        values: dict[str, Any],
        section: str,
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        for source_label, disease_id_value in values.items():
            disease_id = str(disease_id_value)
            if disease_id not in self.valid_ids:
                raise ValueError(
                    f"Unknown disease ID {disease_id!r} in mapping section {section!r}"
                )
            result[normalize_label(source_label)] = disease_id
        return result

    def map(self, dataset_id: str, source_label: Any) -> str | None:
        normalized = normalize_label(source_label)
        if not normalized:
            return None
        dataset_mapping = self._dataset_mappings.get(dataset_id, {})
        return (
            dataset_mapping.get(normalized)
            or self._global_mapping.get(normalized)
            or self._base_mapping.get(normalized)
        )

    def canonical_source_label(self, dataset_id: str, source_label: Any) -> str:
        """Return a countable canonical label for every non-empty source label."""

        normalized = normalize_label(source_label)
        if not normalized:
            return "missing_source_label"
        disease_id = self.map(dataset_id, source_label)
        if disease_id is not None:
            return self._canonical_by_id[disease_id]
        return normalized.replace(" ", "_")


def reference_diagnosis(
    *,
    mapper: DiseaseMapper,
    dataset_id: str,
    disease_original: Any,
    rank: int,
    weight: float | None,
    source: str,
) -> dict[str, Any]:
    """Build one normalized diagnosis candidate."""

    original = "" if disease_original is None else str(disease_original).strip()
    disease_id = mapper.map(dataset_id, original)
    return {
        "disease_original": original,
        "disease_id": disease_id,
        "canonical_source_label": mapper.canonical_source_label(
            dataset_id,
            original,
        ),
        "mapping_status": (
            "benchmark_mapped"
            if disease_id is not None
            else "out_of_benchmark_scope"
        ),
        "rank": int(rank),
        "weight": None if weight is None else float(weight),
        "source": source,
    }


def json_metadata(values: dict[str, Any]) -> str:
    """Serialize source-specific metadata deterministically."""

    cleaned = {
        str(key): _json_safe(value)
        for key, value in values.items()
    }
    return json.dumps(cleaned, ensure_ascii=False, sort_keys=True)


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def optional_string(value: Any) -> str | None:
    """Convert a scalar to a non-empty string or null."""

    safe_value = _json_safe(value)
    if safe_value is None:
        return None
    text = str(safe_value).strip()
    return text or None


def make_manifest_row(
    *,
    mapper: DiseaseMapper,
    dataset_id: str,
    dataset_version: str,
    sample_id: str,
    original_image_id: str,
    original_case_id: str | None,
    patient_id: str | None,
    group_id: str,
    image_uri: str,
    reference_diagnoses: list[dict[str, Any]],
    diagnosis_basis: str,
    diagnosis_gradable: bool,
    license_id: str,
    source_metadata: dict[str, Any],
    age_years: int | None = None,
    age_group_source: str | None = None,
    age_group_standardized: str | None = None,
    age_source: str | None = None,
    sex_or_gender: str | None = None,
    sex_or_gender_system: str | None = None,
    sex_or_gender_source: str | None = None,
    race_ethnicity: str | None = None,
    race_ethnicity_source: str | None = None,
    skin_tone_system: str | None = None,
    skin_tone: str | None = None,
    skin_tone_source: str | None = None,
    force_exclusion_reason: str | None = None,
) -> dict[str, Any]:
    """Create one row conforming to the shared Arrow manifest schema."""

    primary = reference_diagnoses[0] if reference_diagnoses else None
    disease_original = primary["disease_original"] if primary else None
    canonical_source_label = primary["canonical_source_label"] if primary else None
    disease_id = primary["disease_id"] if primary else None
    mapping_status = (
        primary["mapping_status"]
        if primary
        else "missing_reference_diagnosis"
    )

    if not diagnosis_gradable:
        include = False
        exclusion_reason = "diagnosis_not_gradable"
    elif not reference_diagnoses:
        include = False
        exclusion_reason = "missing_reference_diagnosis"
    elif disease_id is None:
        include = False
        exclusion_reason = "primary_disease_out_of_benchmark_scope"
    else:
        include = True
        exclusion_reason = None
    if force_exclusion_reason is not None:
        include = False
        exclusion_reason = force_exclusion_reason

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "sample_id": sample_id,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "original_image_id": original_image_id,
        "original_case_id": original_case_id,
        "patient_id": patient_id,
        "group_id": group_id,
        "image_uri": image_uri,
        "disease_original": disease_original,
        "canonical_source_label": canonical_source_label,
        "disease_id": disease_id,
        "reference_diagnoses": reference_diagnoses,
        "mapping_status": mapping_status,
        "diagnosis_basis": diagnosis_basis,
        "diagnosis_gradable": bool(diagnosis_gradable),
        "taxonomy_id": mapper.taxonomy_id,
        "taxonomy_version": mapper.taxonomy_version,
        "age_years": age_years,
        "age_group_source": age_group_source,
        "age_group_standardized": age_group_standardized,
        "age_source": age_source,
        "sex_or_gender": sex_or_gender,
        "sex_or_gender_system": sex_or_gender_system,
        "sex_or_gender_source": sex_or_gender_source,
        "race_ethnicity": race_ethnicity,
        "race_ethnicity_source": race_ethnicity_source,
        "skin_tone_system": skin_tone_system,
        "skin_tone": skin_tone,
        "skin_tone_source": skin_tone_source,
        "image_sha256": None,
        "perceptual_hash": None,
        "perceptual_hash_algorithm": None,
        "license_id": license_id,
        "split": None,
        "split_version": None,
        "include": include,
        "exclusion_reason": exclusion_reason,
        "source_metadata": json_metadata(source_metadata),
    }


def write_manifest(rows: Iterable[dict[str, Any]], output_path: Path) -> int:
    """Write normalized rows as a schema-stable compressed Parquet file."""

    row_list = list(rows)
    table = pa.Table.from_pylist(row_list, schema=MANIFEST_ARROW_SCHEMA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="zstd")
    return table.num_rows


def standardize_age_group(
    *,
    age_years: int | None = None,
    source_group: Any = None,
) -> str | None:
    """Map exact ages or known SCIN groups to project-wide age bands."""

    source_mapping = {
        "AGE_18_TO_29": "18_to_29",
        "AGE_30_TO_39": "30_to_39",
        "AGE_40_TO_49": "40_to_49",
        "AGE_50_TO_59": "50_to_59",
        "AGE_60_TO_69": "60_to_69",
        "AGE_70_TO_79": "70_and_over",
        "AGE_80_OR_ABOVE": "70_and_over",
        "AGE_UNKNOWN": "unknown",
    }
    source_text = optional_string(source_group)
    if source_text is not None:
        return source_mapping.get(source_text.upper(), "unknown")
    if age_years is None:
        return None
    if age_years < 0:
        return None
    if age_years < 18:
        return "under_18"
    if age_years < 30:
        return "18_to_29"
    if age_years < 40:
        return "30_to_39"
    if age_years < 50:
        return "40_to_49"
    if age_years < 60:
        return "50_to_59"
    if age_years < 70:
        return "60_to_69"
    return "70_and_over"
