"""Build and validate the evidence-grounded DDI benchmark release."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.data_pipeline.common import load_yaml


TASK_SCHEMA_VERSION = "1.0.0"
TASK_ARROW_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("image_uri", pa.string(), nullable=False),
        pa.field("dataset_id", pa.string(), nullable=False),
        pa.field("evaluation_origin", pa.string(), nullable=False),
        pa.field("leakage_group_id", pa.string(), nullable=False),
        pa.field("disease_id", pa.string(), nullable=True),
        pa.field("disease_original", pa.string(), nullable=False),
        pa.field("diagnosis_basis", pa.string(), nullable=False),
        pa.field("skin_tone", pa.string(), nullable=True),
        pa.field(
            "morphology_concept_ids",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field("morphology_positive_count", pa.int16(), nullable=False),
        pa.field(
            "reference_clinical_description",
            pa.string(),
            nullable=True,
        ),
        pa.field("score_morphology", pa.bool_(), nullable=False),
        pa.field("score_description", pa.bool_(), nullable=False),
        pa.field("score_diagnosis", pa.bool_(), nullable=False),
    ]
)


def build_evidence_grounded_release(root: Path) -> dict[str, Any]:
    """Materialize the joined DDI, SKINCON, and SkinCAP task manifest."""

    config_path = (
        root
        / "configs/benchmarks/derma_isep/evidence_grounded_diagnosis.yaml"
    )
    config = load_yaml(config_path)
    dataset = config["dataset"]
    source_path = root / dataset["source_manifest"]
    annotations_path = root / dataset["morphology_annotations"]
    captions_path = root / dataset["caption_metadata"]
    manifest_path = root / dataset["manifest"]
    release_path = root / dataset["release_manifest"]

    source = pq.read_table(source_path).to_pandas()
    annotations = pd.read_csv(annotations_path)
    captions = pd.read_csv(captions_path, low_memory=False)
    frame = build_evidence_grounded_frame(
        source=source,
        annotations=annotations,
        captions=captions,
        config=config,
    )
    integrity = validate_evidence_grounded_frame(frame, config=config)
    if not integrity["passed"]:
        raise ValueError(
            "Evidence-grounded benchmark integrity checks failed: "
            + ", ".join(integrity["errors"])
        )

    _write_manifest(frame, manifest_path)
    release = {
        "release": {
            "id": "evidence_grounded_diagnosis_dataset_v1",
            "version": "1.1.0",
            "status": "frozen",
            "release_date": "2026-07-29",
            "task_schema_version": TASK_SCHEMA_VERSION,
            "evaluation_origin": "external",
            "integrity_passed": True,
            "cohorts": integrity["cohorts"],
            "sources": {
                "ddi_manifest": _path_record(root, source_path),
                "skincon_annotations": _path_record(
                    root,
                    annotations_path,
                ),
                "skincap_captions": _path_record(root, captions_path),
            },
            "configuration": {
                "benchmark": _path_record(root, config_path),
                "prompt": _path_record(
                    root,
                    root / config["prompt"]["path"],
                ),
                "output_schema": _path_record(
                    root,
                    root / config["schema"]["path"],
                ),
                "disease_taxonomy": _path_record(
                    root,
                    root / config["taxonomy"]["disease"]["path"],
                ),
            },
            "artifacts": {
                "task_manifest": _path_record(root, manifest_path),
            },
        }
    }
    _write_yaml(release, release_path)
    development_validation = build_evidence_grounded_validation_release(root)
    internal_benchmark = build_evidence_grounded_internal_release(root)
    return {
        "manifest_path": manifest_path,
        "release_path": release_path,
        "frame": frame,
        "integrity": integrity,
        "release": release,
        "development_validation": development_validation,
        "internal_benchmark": internal_benchmark,
    }


def build_evidence_grounded_validation_release(
    root: Path,
) -> dict[str, Any]:
    """Build internal Validation evidence tasks from Fitzpatrick annotations."""

    config_path = (
        root
        / "configs/benchmarks/derma_isep/evidence_grounded_diagnosis.yaml"
    )
    config = load_yaml(config_path)
    development = config["dataset"]["development_validation"]
    source_path = root / development["source_manifest"]
    annotations_path = root / development["morphology_annotations"]
    captions_path = root / development["caption_metadata"]
    manifest_path = root / development["manifest"]
    release_path = root / development["release_manifest"]

    source = pq.read_table(source_path).to_pandas()
    annotations = pd.read_csv(annotations_path)
    captions = pd.read_csv(captions_path, low_memory=False)
    frame = build_evidence_grounded_validation_frame(
        source=source,
        annotations=annotations,
        captions=captions,
        config=config,
    )
    integrity = validate_evidence_grounded_frame(
        frame,
        config=config,
        expected_origin="development_validation",
        expected_counts=development["expected_counts"],
    )
    if not integrity["passed"]:
        raise ValueError(
            "Evidence-grounded Validation integrity checks failed: "
            + ", ".join(integrity["errors"])
        )
    if frame["disease_id"].nunique() != int(
        development["expected_counts"]["covered_disease_count"]
    ):
        raise ValueError(
            "Evidence-grounded Validation disease coverage mismatch"
        )

    _write_manifest(frame, manifest_path)
    release = {
        "release": {
            "id": "evidence_grounded_diagnosis_validation_v1",
            "version": "1.0.0",
            "status": "development",
            "release_date": "2026-07-31",
            "task_schema_version": TASK_SCHEMA_VERSION,
            "evaluation_origin": "development_validation",
            "integrity_passed": True,
            "cohorts": integrity["cohorts"],
            "covered_disease_count": int(frame["disease_id"].nunique()),
            "sources": {
                "validation_manifest": _path_record(root, source_path),
                "skincon_fitzpatrick_annotations": _path_record(
                    root,
                    annotations_path,
                ),
                "skincap_captions": _path_record(root, captions_path),
            },
            "configuration": {
                "benchmark": _path_record(root, config_path),
                "prompt": _path_record(
                    root,
                    root / config["prompt"]["path"],
                ),
                "output_schema": _path_record(
                    root,
                    root / config["schema"]["path"],
                ),
                "disease_taxonomy": _path_record(
                    root,
                    root / config["taxonomy"]["disease"]["path"],
                ),
            },
            "artifacts": {
                "task_manifest": _path_record(root, manifest_path),
            },
        }
    }
    _write_yaml(release, release_path)
    return {
        "manifest_path": manifest_path,
        "release_path": release_path,
        "frame": frame,
        "integrity": integrity,
        "release": release,
    }


def build_evidence_grounded_internal_release(
    root: Path,
) -> dict[str, Any]:
    """Build sealed Internal Benchmark evidence tasks from Fitzpatrick data."""

    config_path = (
        root
        / "configs/benchmarks/derma_isep/evidence_grounded_diagnosis.yaml"
    )
    config = load_yaml(config_path)
    internal = config["dataset"]["internal_benchmark"]
    source_path = root / internal["source_manifest"]
    annotations_path = root / internal["morphology_annotations"]
    captions_path = root / internal["caption_metadata"]
    manifest_path = root / internal["manifest"]
    release_path = root / internal["release_manifest"]

    source = pq.read_table(source_path).to_pandas()
    annotations = pd.read_csv(annotations_path)
    captions = pd.read_csv(captions_path, low_memory=False)
    frame = build_evidence_grounded_internal_frame(
        source=source,
        annotations=annotations,
        captions=captions,
        config=config,
    )
    integrity = validate_evidence_grounded_frame(
        frame,
        config=config,
        expected_origin="internal_benchmark",
        expected_counts=internal["expected_counts"],
    )
    if not integrity["passed"]:
        raise ValueError(
            "Evidence-grounded Internal Benchmark integrity checks failed: "
            + ", ".join(integrity["errors"])
        )
    expected_diseases = int(
        internal["expected_counts"]["covered_disease_count"]
    )
    if frame["disease_id"].nunique() != expected_diseases:
        raise ValueError(
            "Evidence-grounded Internal Benchmark disease coverage mismatch"
        )

    _write_manifest(frame, manifest_path)
    release = {
        "release": {
            "id": "evidence_grounded_diagnosis_internal_benchmark_v1",
            "version": "1.0.0",
            "status": "frozen",
            "release_date": "2026-08-01",
            "task_schema_version": TASK_SCHEMA_VERSION,
            "evaluation_origin": "internal_benchmark",
            "integrity_passed": True,
            "cohorts": integrity["cohorts"],
            "covered_disease_count": int(frame["disease_id"].nunique()),
            "sources": {
                "internal_benchmark_manifest": _path_record(
                    root,
                    source_path,
                ),
                "skincon_fitzpatrick_annotations": _path_record(
                    root,
                    annotations_path,
                ),
                "skincap_captions": _path_record(root, captions_path),
            },
            "configuration": {
                "benchmark": _path_record(root, config_path),
                "prompt": _path_record(
                    root,
                    root / config["prompt"]["path"],
                ),
                "output_schema": _path_record(
                    root,
                    root / config["schema"]["path"],
                ),
                "disease_taxonomy": _path_record(
                    root,
                    root / config["taxonomy"]["disease"]["path"],
                ),
            },
            "artifacts": {
                "task_manifest": _path_record(root, manifest_path),
            },
        }
    }
    _write_yaml(release, release_path)
    return {
        "manifest_path": manifest_path,
        "release_path": release_path,
        "frame": frame,
        "integrity": integrity,
        "release": release,
    }


def build_evidence_grounded_frame(
    *,
    source: pd.DataFrame,
    annotations: pd.DataFrame,
    captions: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Join source records to their image-level evidence references."""

    dataset = config["dataset"]
    required_source = {
        "sample_id",
        "image_uri",
        "dataset_id",
        "leakage_group_id",
        "disease_id",
        "disease_original",
        "diagnosis_basis",
        "skin_tone",
        "include",
    }
    _require_columns(source, required_source, "DDI source manifest")

    annotation_id = dataset["annotation_image_id_column"]
    exclusion_column = dataset["annotation_exclusion_column"]
    morphology_columns = _morphology_column_mapping(
        annotations=annotations,
        concepts=config["taxonomy"]["morphology"]["concepts"],
        annotation_id=annotation_id,
        exclusion_column=exclusion_column,
    )
    _require_columns(
        annotations,
        {annotation_id, exclusion_column, *morphology_columns},
        "SKINCON annotations",
    )

    caption_id = dataset["caption_image_id_column"]
    caption_source = dataset["caption_source_column"]
    caption_text = dataset["caption_text_column"]
    _require_columns(
        captions,
        {caption_id, caption_source, caption_text},
        "SkinCAP captions",
    )

    source = source.copy()
    source["_image_basename"] = source["image_uri"].map(
        lambda value: Path(str(value)).name
    )
    if source["_image_basename"].duplicated().any():
        raise ValueError("DDI source image basenames must be unique")
    if annotations[annotation_id].duplicated().any():
        raise ValueError("SKINCON ImageID values must be unique")

    ddi_captions = captions[
        captions[caption_source].astype(str).str.casefold().eq("ddi")
    ][[caption_id, caption_text]].copy()
    if ddi_captions[caption_id].duplicated().any():
        raise ValueError("SkinCAP DDI image paths must be unique")

    joined = source.merge(
        annotations,
        left_on="_image_basename",
        right_on=annotation_id,
        how="left",
        validate="one_to_one",
    ).merge(
        ddi_captions,
        left_on="_image_basename",
        right_on=caption_id,
        how="left",
        validate="one_to_one",
    )
    if joined[annotation_id].isna().any():
        raise ValueError("Every DDI image must have a SKINCON annotation")

    joined = joined[joined[exclusion_column].ne(1)].copy()
    concept_ids_by_column = {
        column: concept_id
        for column, concept_id in morphology_columns.items()
    }
    joined["morphology_concept_ids"] = joined.apply(
        lambda row: [
            concept_id
            for column, concept_id in concept_ids_by_column.items()
            if int(row[column]) == 1
        ],
        axis=1,
    )
    joined["morphology_positive_count"] = joined[
        "morphology_concept_ids"
    ].map(len)
    joined["score_morphology"] = True
    joined["score_description"] = joined[caption_text].notna()
    joined["score_diagnosis"] = (
        joined["include"].eq(True) & joined["disease_id"].notna()
    )

    frame = pd.DataFrame(
        {
            "schema_version": TASK_SCHEMA_VERSION,
            "sample_id": joined["sample_id"].astype(str),
            "image_uri": joined["image_uri"].astype(str),
            "dataset_id": joined["dataset_id"].astype(str),
            "evaluation_origin": "external",
            "leakage_group_id": joined["leakage_group_id"].astype(str),
            "disease_id": joined["disease_id"],
            "disease_original": joined["disease_original"].astype(str),
            "diagnosis_basis": joined["diagnosis_basis"].astype(str),
            "skin_tone": joined["skin_tone"],
            "morphology_concept_ids": joined[
                "morphology_concept_ids"
            ],
            "morphology_positive_count": joined[
                "morphology_positive_count"
            ].astype("int16"),
            "reference_clinical_description": joined[caption_text],
            "score_morphology": joined["score_morphology"],
            "score_description": joined["score_description"],
            "score_diagnosis": joined["score_diagnosis"],
        }
    )
    return frame.sort_values("sample_id", kind="mergesort").reset_index(
        drop=True
    )


def build_evidence_grounded_validation_frame(
    *,
    source: pd.DataFrame,
    annotations: pd.DataFrame,
    captions: pd.DataFrame,
    config: dict[str, Any],
    cohort: dict[str, Any] | None = None,
    evaluation_origin: str = "development_validation",
    source_name: str = "Validation",
) -> pd.DataFrame:
    """Join a Fitzpatrick evaluation cohort to SKINCON/SkinCAP references."""

    development = cohort or config["dataset"]["development_validation"]
    required_source = {
        "sample_id",
        "image_uri",
        "dataset_id",
        "leakage_group_id",
        "disease_id",
        "disease_original",
        "diagnosis_basis",
        "skin_tone",
        "include",
    }
    _require_columns(
        source,
        required_source,
        f"{source_name} source manifest",
    )
    source = source[
        source["dataset_id"].astype(str).eq(
            str(development["source_dataset_id"])
        )
    ].copy()
    source["_image_basename"] = source["image_uri"].map(
        lambda value: Path(str(value).split("::")[-1]).name
    )
    if source["_image_basename"].duplicated().any():
        raise ValueError("Fitzpatrick Validation image basenames must be unique")

    annotation_id = development["annotation_image_id_column"]
    exclusion_column = development["annotation_exclusion_column"]
    morphology_columns = _morphology_column_mapping(
        annotations=annotations,
        concepts=config["taxonomy"]["morphology"]["concepts"],
        annotation_id=annotation_id,
        exclusion_column=exclusion_column,
    )
    _require_columns(
        annotations,
        {annotation_id, exclusion_column, *morphology_columns},
        "SKINCON Fitzpatrick annotations",
    )
    if annotations[annotation_id].duplicated().any():
        raise ValueError("SKINCON Fitzpatrick ImageID values must be unique")

    caption_id = development["caption_image_id_column"]
    caption_source = development["caption_source_column"]
    caption_text = development["caption_text_column"]
    _require_columns(
        captions,
        {caption_id, caption_source, caption_text},
        "SkinCAP captions",
    )
    selected_captions = captions[
        captions[caption_source]
        .astype(str)
        .str.casefold()
        .eq(str(development["caption_source_value"]).casefold())
    ][[caption_id, caption_text]].copy()
    if selected_captions[caption_id].duplicated().any():
        raise ValueError("SkinCAP Fitzpatrick image paths must be unique")

    joined = source.merge(
        annotations,
        left_on="_image_basename",
        right_on=annotation_id,
        how="inner",
        validate="one_to_one",
    )
    joined = joined[joined[exclusion_column].ne(1)].copy()
    joined = joined.merge(
        selected_captions,
        left_on="_image_basename",
        right_on=caption_id,
        how="left",
        validate="one_to_one",
    )
    concept_ids_by_column = {
        column: concept_id
        for column, concept_id in morphology_columns.items()
    }
    joined["morphology_concept_ids"] = joined.apply(
        lambda row: [
            concept_id
            for column, concept_id in concept_ids_by_column.items()
            if int(row[column]) == 1
        ],
        axis=1,
    )
    joined["morphology_positive_count"] = joined[
        "morphology_concept_ids"
    ].map(len)
    joined["score_morphology"] = True
    joined["score_description"] = joined[caption_text].notna()
    joined["score_diagnosis"] = (
        joined["include"].eq(True) & joined["disease_id"].notna()
    )

    frame = pd.DataFrame(
        {
            "schema_version": TASK_SCHEMA_VERSION,
            "sample_id": joined["sample_id"].astype(str),
            "image_uri": joined["image_uri"].astype(str),
            "dataset_id": joined["dataset_id"].astype(str),
            "evaluation_origin": evaluation_origin,
            "leakage_group_id": joined["leakage_group_id"].astype(str),
            "disease_id": joined["disease_id"],
            "disease_original": joined["disease_original"].astype(str),
            "diagnosis_basis": joined["diagnosis_basis"].astype(str),
            "skin_tone": joined["skin_tone"],
            "morphology_concept_ids": joined["morphology_concept_ids"],
            "morphology_positive_count": joined[
                "morphology_positive_count"
            ].astype("int16"),
            "reference_clinical_description": joined[caption_text],
            "score_morphology": joined["score_morphology"],
            "score_description": joined["score_description"],
            "score_diagnosis": joined["score_diagnosis"],
        }
    )
    return frame.sort_values("sample_id", kind="mergesort").reset_index(
        drop=True
    )


def build_evidence_grounded_internal_frame(
    *,
    source: pd.DataFrame,
    annotations: pd.DataFrame,
    captions: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Join sealed Internal Benchmark rows to independent references."""

    return build_evidence_grounded_validation_frame(
        source=source,
        annotations=annotations,
        captions=captions,
        config=config,
        cohort=config["dataset"]["internal_benchmark"],
        evaluation_origin="internal_benchmark",
        source_name="Internal Benchmark",
    )


def validate_evidence_grounded_frame(
    frame: pd.DataFrame,
    *,
    config: dict[str, Any],
    expected_origin: str | None = None,
    expected_counts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate schema, cohort counts, references, and identifiers."""

    errors: list[str] = []
    expected_columns = TASK_ARROW_SCHEMA.names
    if list(frame.columns) != expected_columns:
        errors.append("task_columns_do_not_match_schema")
    if frame["sample_id"].duplicated().any():
        errors.append("sample_ids_must_be_unique")
    origin = expected_origin or str(
        config["dataset"].get("evaluation_origin", "external")
    )
    if not frame["evaluation_origin"].eq(origin).all():
        errors.append(f"evaluation_origin_must_be_{origin}")
    if not frame["score_morphology"].all():
        errors.append("all_rows_must_have_morphology_reference")
    if (
        frame["score_description"]
        & frame["reference_clinical_description"].isna()
    ).any():
        errors.append("description_flag_requires_reference")
    if (
        frame["score_diagnosis"] & frame["disease_id"].isna()
    ).any():
        errors.append("diagnosis_flag_requires_disease_id")

    allowed_concepts = {
        str(item["id"])
        for item in config["taxonomy"]["morphology"]["concepts"]
    }
    observed_concepts = {
        concept_id
        for values in frame["morphology_concept_ids"]
        for concept_id in values
    }
    if observed_concepts - allowed_concepts:
        errors.append("unknown_morphology_concept_id")
    if (
        frame["morphology_positive_count"]
        != frame["morphology_concept_ids"].map(len)
    ).any():
        errors.append("morphology_positive_count_mismatch")

    cohorts = {
        "morphology": int(frame["score_morphology"].sum()),
        "description": int(frame["score_description"].sum()),
        "diagnosis": int(frame["score_diagnosis"].sum()),
    }
    expected = expected_counts or config["dataset"]["expected_counts"]
    expected_cohorts = {
        "morphology": int(expected["morphology_cohort"]),
        "description": int(expected["description_cohort"]),
        "diagnosis": int(expected["diagnosis_cohort"]),
    }
    if cohorts != expected_cohorts:
        errors.append(
            f"cohort_counts_mismatch:{cohorts}!={expected_cohorts}"
        )
    return {
        "passed": not errors,
        "errors": errors,
        "sample_count": int(len(frame)),
        "unique_group_count": int(frame["leakage_group_id"].nunique()),
        "cohorts": cohorts,
    }


def validate_evidence_grounded_release(root: Path) -> dict[str, Any]:
    """Validate the materialized task manifest without rebuilding it."""

    config = load_yaml(
        root
        / "configs/benchmarks/derma_isep/evidence_grounded_diagnosis.yaml"
    )
    path = root / config["dataset"]["manifest"]
    release_path = root / config["dataset"]["release_manifest"]
    if not path.exists():
        raise FileNotFoundError(f"Missing task manifest: {path}")
    if not release_path.exists():
        raise FileNotFoundError(f"Missing release manifest: {release_path}")
    frame = pq.read_table(path).to_pandas()
    result = validate_evidence_grounded_frame(frame, config=config)
    release = load_yaml(release_path)["release"]
    checksum_errors = _validate_release_checksums(root, release)
    result["checksum_errors"] = checksum_errors
    if checksum_errors:
        result["passed"] = False
        result["errors"].extend(checksum_errors)
    if not result["passed"]:
        raise ValueError(
            "Evidence-grounded release is invalid: "
            + ", ".join(result["errors"])
        )
    validate_evidence_grounded_validation_release(root)
    validate_evidence_grounded_internal_release(root)
    return result


def validate_evidence_grounded_validation_release(
    root: Path,
) -> dict[str, Any]:
    """Validate the internal development Validation evidence release."""

    config = load_yaml(
        root
        / "configs/benchmarks/derma_isep/evidence_grounded_diagnosis.yaml"
    )
    development = config["dataset"]["development_validation"]
    path = root / development["manifest"]
    release_path = root / development["release_manifest"]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing evidence-grounded Validation manifest: {path}"
        )
    if not release_path.exists():
        raise FileNotFoundError(
            f"Missing evidence-grounded Validation release: {release_path}"
        )
    table = pq.read_table(path)
    if table.schema != TASK_ARROW_SCHEMA:
        raise ValueError(
            "Evidence-grounded Validation task manifest schema mismatch"
        )
    frame = table.to_pandas()
    result = validate_evidence_grounded_frame(
        frame,
        config=config,
        expected_origin="development_validation",
        expected_counts=development["expected_counts"],
    )
    expected_diseases = int(
        development["expected_counts"]["covered_disease_count"]
    )
    if frame["disease_id"].nunique() != expected_diseases:
        result["passed"] = False
        result["errors"].append("covered_disease_count_mismatch")
    release = load_yaml(release_path)["release"]
    checksum_errors = _validate_release_checksums(root, release)
    result["checksum_errors"] = checksum_errors
    if checksum_errors:
        result["passed"] = False
        result["errors"].extend(checksum_errors)
    if not result["passed"]:
        raise ValueError(
            "Evidence-grounded Validation release is invalid: "
            + ", ".join(result["errors"])
        )
    return result


def validate_evidence_grounded_internal_release(
    root: Path,
) -> dict[str, Any]:
    """Validate the sealed internal evidence-grounded release."""

    config = load_yaml(
        root
        / "configs/benchmarks/derma_isep/evidence_grounded_diagnosis.yaml"
    )
    internal = config["dataset"]["internal_benchmark"]
    path = root / internal["manifest"]
    release_path = root / internal["release_manifest"]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing evidence-grounded Internal Benchmark manifest: {path}"
        )
    if not release_path.exists():
        raise FileNotFoundError(
            f"Missing evidence-grounded Internal Benchmark release: "
            f"{release_path}"
        )
    table = pq.read_table(path)
    if table.schema != TASK_ARROW_SCHEMA:
        raise ValueError(
            "Evidence-grounded Internal Benchmark task manifest schema "
            "mismatch"
        )
    frame = table.to_pandas()
    result = validate_evidence_grounded_frame(
        frame,
        config=config,
        expected_origin="internal_benchmark",
        expected_counts=internal["expected_counts"],
    )
    expected_diseases = int(
        internal["expected_counts"]["covered_disease_count"]
    )
    if frame["disease_id"].nunique() != expected_diseases:
        result["passed"] = False
        result["errors"].append("covered_disease_count_mismatch")
    release = load_yaml(release_path)["release"]
    checksum_errors = _validate_release_checksums(root, release)
    result["checksum_errors"] = checksum_errors
    if checksum_errors:
        result["passed"] = False
        result["errors"].extend(checksum_errors)
    if not result["passed"]:
        raise ValueError(
            "Evidence-grounded Internal Benchmark release is invalid: "
            + ", ".join(result["errors"])
        )
    return result


def _morphology_column_mapping(
    *,
    annotations: pd.DataFrame,
    concepts: list[dict[str, str]],
    annotation_id: str,
    exclusion_column: str,
) -> dict[str, str]:
    available = {
        _normalized_label(column): column
        for column in annotations.columns
        if column not in {annotation_id, exclusion_column, "Unnamed: 0"}
    }
    mapping: dict[str, str] = {}
    for concept in concepts:
        key = _normalized_label(concept["display_name"])
        column = available.get(key)
        if column is None:
            raise ValueError(
                "No SKINCON annotation column for morphology concept "
                f"{concept['id']!r}"
            )
        mapping[column] = str(concept["id"])
    if len(mapping) != len(concepts):
        raise ValueError("Morphology annotation mapping must be one-to-one")
    return mapping


def _normalized_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    source_name: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{source_name} is missing columns: {', '.join(missing)}"
        )


def _write_manifest(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(
        frame,
        schema=TASK_ARROW_SCHEMA,
        preserve_index=False,
    )
    pq.write_table(table, path, compression="zstd")


def _write_yaml(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            value,
            handle,
            sort_keys=False,
            allow_unicode=False,
        )


def _path_record(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _validate_release_checksums(
    root: Path,
    release: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    for section in ["sources", "configuration", "artifacts"]:
        records = release.get(section)
        if not isinstance(records, dict):
            errors.append(f"release_{section}_missing")
            continue
        for name, record in records.items():
            if not isinstance(record, dict):
                errors.append(f"release_{section}_{name}_invalid")
                continue
            relative_path = record.get("path")
            expected_sha256 = record.get("sha256")
            if not isinstance(relative_path, str):
                errors.append(f"release_{section}_{name}_path_invalid")
                continue
            path = root / relative_path
            if not path.exists():
                errors.append(f"release_{section}_{name}_missing")
                continue
            actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_sha256 != expected_sha256:
                errors.append(f"release_{section}_{name}_checksum_mismatch")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the evidence-grounded DDI benchmark release."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the existing release without rebuilding it.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.validate_only:
        result = validate_evidence_grounded_release(root)
        print(
            "Evidence-grounded release valid: "
            f"{result['sample_count']} samples"
        )
        return
    result = build_evidence_grounded_release(root)
    print(
        "Evidence-grounded release built: "
        f"{result['integrity']['sample_count']} samples"
    )
    print(result["manifest_path"].relative_to(root))
    print(result["release_path"].relative_to(root))
    development = result["development_validation"]
    print(development["manifest_path"].relative_to(root))
    print(development["release_path"].relative_to(root))
    internal = result["internal_benchmark"]
    print(internal["manifest_path"].relative_to(root))
    print(internal["release_path"].relative_to(root))


if __name__ == "__main__":
    main()
