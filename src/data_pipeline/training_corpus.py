"""Build the leakage-safe multimodal dermatology training corpus.

This module deliberately does not rebuild or modify any benchmark split.  It
starts from the frozen visual-top-k training split and adds only clinical
photographs from explicitly configured training-only sources.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable
import zipfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data_pipeline.common import (
    MANIFEST_ARROW_SCHEMA,
    DiseaseMapper,
    make_manifest_row,
    optional_string,
    reference_diagnosis,
    standardize_age_group,
)
from src.data_pipeline.deduplication import (
    BKTree,
    ImageResolver,
    PERCEPTUAL_HASH_ALGORITHM,
    compute_fingerprint,
    hamming_distance_int,
)


TRAINING_SCHEMA_VERSION = "1.0.0"
TRAINING_FIELDS = [
    pa.field("training_schema_version", pa.string(), nullable=False),
    pa.field("image_modality", pa.string(), nullable=False),
    pa.field("training_role", pa.string(), nullable=False),
    pa.field("caption", pa.string()),
    pa.field("visual_concepts", pa.string()),
    pa.field("body_location", pa.string()),
    pa.field("symptoms", pa.string()),
    pa.field("source_split", pa.string(), nullable=False),
    pa.field("source_name", pa.string(), nullable=False),
    pa.field("clinical_filter_rule", pa.string(), nullable=False),
    pa.field("duplicate_match_scope", pa.string()),
]
TRAINING_ARROW_SCHEMA = pa.schema(
    [*MANIFEST_ARROW_SCHEMA, *TRAINING_FIELDS]
)

DERM1M_NON_CLINICAL_PATTERN = re.compile(
    r"""
    dermoscop|histolog|histopath|microscop|immunohisto|hematoxylin|
    \bh\s*&\s*e\b|\bh&e\b|\bhistochemical[ -]stain\b|
    \bimmunostain|\bstained[ -](?:section|slide)\b|
    \bdiagram\b|\bschematic\b|\bflowchart\b|\bchart\b|\btable\b|
    \balgorithm\b|\bradiograph\b|\bx[ -]?ray\b|\bmri\b|
    \bct[ -]scan\b|\bultrasound\b|\bpathology[ -]slide\b|
    \blecture[ -]slide\b|\bpresentation[ -]slide\b|\bscreenshot\b|
    \billustration\b|\bdrawing\b
    """,
    re.IGNORECASE | re.VERBOSE,
)
DERM1M_STRONG_CLINICAL_PATTERN = re.compile(
    r"""
    clinical[ -](?:photograph|photo|image)|
    photograph[ -]of[ -](?:a[ -]|the[ -])?patient|
    patient[ -](?:with|showing|presented|presents)|
    skin[ -]lesion|cutaneous[ -]lesion
    """,
    re.IGNORECASE | re.VERBOSE,
)
DERM1M_FORUM_SOURCES = {"IIYI_chinese", "reddit_english"}
DERM1M_PUBLIC_SOURCES = {"public"}
DERM1M_STRONG_EVIDENCE_SOURCES = {
    "pubmed_english",
    "textbook_english",
    "twitter_english",
    "youtube",
}
DERM1M_ARCHIVES = {
    "IIYI": "IIYI.zip",
    "edu": "edu.zip",
    "note": "note.zip",
    "public": "public.zip",
    "pubmed": "pubmed.zip",
    "reddit": "reddit.zip",
    "twitter": "twitter.zip",
    "validation_data": "validation_data.zip",
    "youtube": "youtube.zip",
}
NO_DIAGNOSIS_LABELS = {
    "",
    "no definitive diagnosis",
    "no diagnosis",
    "unknown",
}

DEFAULT_BASE_TRAIN = Path(
    "data/benchmarks/derma_isep/visual_top_k_v1/datasets/internal/train.parquet"
)
DEFAULT_PROTECTED = [
    Path("data/benchmarks/derma_isep/visual_top_k_v1/datasets/internal/validation.parquet"),
    Path("data/benchmarks/derma_isep/visual_top_k_v1/datasets/internal/internal_test.parquet"),
    Path("data/benchmarks/derma_isep/visual_top_k_v1/datasets/external/external_ddi.parquet"),
    Path(
        "data/benchmarks/derma_isep/visual_top_k_v1/datasets/external/"
        "external_skindisnet.parquet"
    ),
]
DEFAULT_OUTPUT_DIR = Path("data/training/dermatology_multimodal_v1")


def classify_derm1m_clinical_image(
    *,
    source: Any,
    filename: Any,
    caption: Any,
) -> tuple[bool, str]:
    """Apply a conservative, metadata-based clinical-photograph filter."""

    source_text = optional_string(source) or ""
    combined = " ".join(
        value
        for value in (
            optional_string(filename),
            optional_string(caption),
        )
        if value
    ).replace("_", " ")

    if DERM1M_NON_CLINICAL_PATTERN.search(combined):
        return False, "explicit_non_clinical_modality"
    if source_text in DERM1M_FORUM_SOURCES:
        return True, "forum_user_clinical_photo"
    if (
        source_text in DERM1M_PUBLIC_SOURCES
        and DERM1M_STRONG_CLINICAL_PATTERN.search(combined)
    ):
        return True, "public_explicit_clinical_image"
    if (
        source_text in DERM1M_STRONG_EVIDENCE_SOURCES
        and DERM1M_STRONG_CLINICAL_PATTERN.search(combined)
    ):
        return True, "explicit_clinical_language"
    return False, "insufficient_clinical_evidence"


def build_training_corpus(
    root: Path,
    *,
    base_train_path: Path = DEFAULT_BASE_TRAIN,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    protected_paths: Iterable[Path] = DEFAULT_PROTECTED,
    limit_derm1m: int | None = None,
) -> dict[str, Any]:
    """Materialize the augmented train manifest and its audit artifacts."""

    root = root.resolve()
    output = root / output_dir
    reports = output / "reports"
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    mapper = DiseaseMapper(
        root / "configs/taxonomies/diseases.yaml",
        root / "configs/taxonomies/source_disease_mappings.yaml",
    )
    base_table = pq.read_table(root / base_train_path)
    base_rows = _base_training_rows(base_table)
    protected_index = _build_protected_index(
        root,
        [root / path for path in protected_paths],
    )
    train_index = _FingerprintIndex.from_rows(base_rows)

    derm1m_rows, derm1m_decisions = _build_derm1m_rows(
        root,
        mapper,
        limit=limit_derm1m,
    )
    hiba_rows = _build_hiba_rows(root, mapper)
    new_rows = [*derm1m_rows, *hiba_rows]
    _fingerprint_and_screen(
        root=root,
        rows=new_rows,
        protected_index=protected_index,
        train_index=train_index,
    )

    all_rows = [*base_rows, *new_rows]
    included_rows = [row for row in all_rows if bool(row["include"])]
    _write_training_rows(all_rows, output / "all_candidates.parquet")
    _write_training_rows(included_rows, output / "train_images.parquet")
    _write_training_rows(
        included_rows,
        output / "teacher_annotation_queue.parquet",
    )

    decision_table = pa.Table.from_pylist(derm1m_decisions)
    pq.write_table(
        decision_table,
        reports / "derm1m_filter_decisions.parquet",
        compression="zstd",
    )
    _write_reports(
        rows=all_rows,
        derm1m_decisions=derm1m_decisions,
        reports_dir=reports,
    )
    release = _write_release(
        root=root,
        output_dir=output,
        all_rows=all_rows,
        included_rows=included_rows,
        protected_paths=list(protected_paths),
    )
    return {
        "base_train_count": len(base_rows),
        "new_candidate_count": len(new_rows),
        "included_count": len(included_rows),
        "new_included_count": sum(
            bool(row["include"]) for row in new_rows
        ),
        "release": release,
    }


def validate_training_corpus(
    root: Path,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, int]:
    """Validate materialized row counts, identities, hashes, and checksums."""

    output = root.resolve() / output_dir
    release_path = output / "release/training_release_v1.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))["release"]
    train = pq.read_table(output / "train_images.parquet").to_pandas()
    queue = pq.read_table(
        output / "teacher_annotation_queue.parquet",
        columns=["sample_id"],
    ).to_pandas()
    all_candidates = pq.read_table(
        output / "all_candidates.parquet",
        columns=["sample_id"],
    ).to_pandas()

    expected_total = int(release["total_train_images"])
    if len(train) != expected_total:
        raise ValueError(
            f"Expected {expected_total} train images, found {len(train)}"
        )
    if not train["sample_id"].is_unique:
        raise ValueError("train_images.parquet contains duplicate sample IDs")
    if set(queue["sample_id"]) != set(train["sample_id"]):
        raise ValueError("Teacher queue and train manifest have different samples")
    if not train["include"].eq(True).all():
        raise ValueError("Excluded rows are present in train_images.parquet")
    if not train["image_modality"].eq("clinical").all():
        raise ValueError("A non-clinical modality is present in the train manifest")
    if train[["image_sha256", "perceptual_hash"]].isna().any().any():
        raise ValueError("An included training image is missing a fingerprint")

    artifact_paths = {
        "train_images": output / "train_images.parquet",
        "teacher_annotation_queue": output / "teacher_annotation_queue.parquet",
        "all_candidates": output / "all_candidates.parquet",
        "derm1m_filter_decisions": output
        / "reports/derm1m_filter_decisions.parquet",
    }
    for name, path in artifact_paths.items():
        actual = _sha256_file(path)
        expected = str(release["artifact_sha256"][name])
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {path}")

    return {
        "train_images": len(train),
        "teacher_queue_images": len(queue),
        "all_candidate_rows": len(all_candidates),
    }


def _base_training_rows(table: pa.Table) -> list[dict[str, Any]]:
    rows = table.to_pylist()
    for row in rows:
        row.update(
            {
                "training_schema_version": TRAINING_SCHEMA_VERSION,
                "image_modality": "clinical",
                "training_role": "in_domain_diagnosis",
                "caption": None,
                "visual_concepts": None,
                "body_location": None,
                "symptoms": None,
                "source_split": "frozen_train",
                "source_name": str(row["dataset_id"]),
                "clinical_filter_rule": "existing_frozen_train",
                "duplicate_match_scope": None,
            }
        )
    return rows


def _build_derm1m_rows(
    root: Path,
    mapper: DiseaseMapper,
    *,
    limit: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data_dir = root / "configs/datasets/derm1m/data"
    metadata_paths = {
        "pretrain": data_dir / "Derm1M_v2_pretrain.csv",
    }
    for metadata_path in metadata_paths.values():
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Derm1M metadata not found: {metadata_path}")

    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    selected_records: list[tuple[dict[str, Any], str, str]] = []
    selected = 0
    for source_split, metadata_path in metadata_paths.items():
        frame = pd.read_csv(metadata_path)
        for record in frame.to_dict(orient="records"):
            filename = str(record["filename"])
            include, rule = classify_derm1m_clinical_image(
                source=record.get("source"),
                filename=filename,
                caption=record.get("caption"),
            )
            decisions.append(
                {
                    "filename": filename,
                    "source": optional_string(record.get("source")),
                    "source_split": source_split,
                    "include_as_clinical": include,
                    "clinical_filter_rule": rule,
                }
            )
            if not include:
                continue
            if limit is not None and selected >= limit:
                continue
            selected_records.append((record, rule, source_split))
            selected += 1

    required_archives = {
        _derm1m_archive_for(str(record["filename"]))
        for record, _, _ in selected_records
    }
    archive_members = _archive_member_indexes(data_dir, required_archives)
    for record, rule, source_split in selected_records:
        filename = str(record["filename"])
        archive_name = _derm1m_archive_for(filename)
        archive_member = _derm1m_archive_member(filename)
        if archive_member not in archive_members[archive_name]:
            raise ValueError(
                f"Derm1M image {archive_member!r} was not found in {archive_name}"
            )

        disease_original = optional_string(record.get("disease_label")) or ""
        has_diagnosis = disease_original.casefold() not in NO_DIAGNOSIS_LABELS
        references = (
            [
                reference_diagnosis(
                    mapper=mapper,
                    dataset_id="derm1m",
                    disease_original=disease_original,
                    rank=1,
                    weight=1.0,
                    source="disease_label",
                )
            ]
            if has_diagnosis
            else []
        )
        row = make_manifest_row(
            mapper=mapper,
            dataset_id="derm1m",
            dataset_version="2",
            sample_id=_stable_id("DERM1M", filename),
            original_image_id=filename,
            original_case_id=None,
            patient_id=None,
            group_id=_derm1m_group_id(filename),
            image_uri=(
                "zip://configs/datasets/derm1m/data/"
                f"{archive_name}::{archive_member}"
            ),
            reference_diagnoses=references,
            diagnosis_basis=(
                "source_derived" if has_diagnosis else "description_only"
            ),
            diagnosis_gradable=has_diagnosis,
            license_id="CC_BY_NC_4_0",
            sex_or_gender=_normalize_derm1m_gender(record.get("gender")),
            sex_or_gender_system=(
                "source_gender"
                if _normalize_derm1m_gender(record.get("gender"))
                else None
            ),
            sex_or_gender_source=(
                "source_metadata"
                if _normalize_derm1m_gender(record.get("gender"))
                else None
            ),
            source_metadata={
                "source": record.get("source"),
                "source_type": record.get("source_type"),
                "hierarchical_disease_label": record.get(
                    "hierarchical_disease_label"
                ),
                "source_age": record.get("age"),
                "archive_member": archive_member,
            },
        )
        row["include"] = True
        row["exclusion_reason"] = None
        row.update(
            {
                "training_schema_version": TRAINING_SCHEMA_VERSION,
                "image_modality": "clinical",
                "training_role": _training_role(row, has_diagnosis),
                "caption": optional_string(record.get("truncated_caption"))
                or optional_string(record.get("caption")),
                "visual_concepts": _clean_derm1m_value(
                    record.get("skin_concept"),
                    "No visual concepts",
                ),
                "body_location": _clean_derm1m_value(
                    record.get("body_location"),
                    "No body location information",
                ),
                "symptoms": _clean_derm1m_value(
                    record.get("symptoms"),
                    "No symptom information",
                ),
                "source_split": source_split,
                "source_name": optional_string(record.get("source"))
                or "derm1m",
                "clinical_filter_rule": rule,
                "duplicate_match_scope": None,
            }
        )
        rows.append(row)
    return rows, decisions


def _build_hiba_rows(
    root: Path,
    mapper: DiseaseMapper,
) -> list[dict[str, Any]]:
    data_dir = root / "configs/datasets/hiba/data"
    metadata_path = data_dir / "hiba-skin-lesions.csv"
    archive_path = data_dir / "hiba-skin-lesions.zip"
    if not metadata_path.is_file() or not archive_path.is_file():
        raise FileNotFoundError(
            "HIBA CSV and ZIP must exist under configs/datasets/hiba/data"
        )
    with zipfile.ZipFile(archive_path) as archive:
        members = set(archive.namelist())

    frame = pd.read_csv(metadata_path)
    frame = frame[
        frame["image_type"].astype(str).str.startswith("clinical")
    ]
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        image_id = str(record["isic_id"])
        archive_member = f"images/{image_id}.jpg"
        if archive_member not in members:
            raise ValueError(f"HIBA image not found: {archive_member}")
        diagnosis = reference_diagnosis(
            mapper=mapper,
            dataset_id="hiba",
            disease_original=record.get("diagnosis"),
            rank=1,
            weight=1.0,
            source="diagnosis",
        )
        age_years = _optional_int(record.get("age_approx"))
        patient_id = optional_string(record.get("patient_id"))
        lesion_id = optional_string(record.get("lesion_id"))
        confirmation = optional_string(record.get("diagnosis_confirm_type"))
        row = make_manifest_row(
            mapper=mapper,
            dataset_id="hiba",
            dataset_version="2025",
            sample_id=f"HIBA_{image_id}",
            original_image_id=image_id,
            original_case_id=lesion_id,
            patient_id=patient_id,
            group_id=(
                f"HIBA_PATIENT_{patient_id}"
                if patient_id
                else f"HIBA_LESION_{lesion_id or image_id}"
            ),
            image_uri=(
                "zip://configs/datasets/hiba/data/"
                f"hiba-skin-lesions.zip::{archive_member}"
            ),
            reference_diagnoses=[diagnosis],
            diagnosis_basis=(
                "pathology"
                if confirmation and confirmation.casefold() == "histopathology"
                else "source_diagnosis_unspecified"
            ),
            diagnosis_gradable=True,
            license_id="CC_BY_4_0",
            age_years=age_years,
            age_group_standardized=standardize_age_group(age_years=age_years),
            age_source="clinical_metadata" if age_years is not None else None,
            sex_or_gender=_normalized_text(record.get("sex")),
            sex_or_gender_system=(
                "source_sex" if optional_string(record.get("sex")) else None
            ),
            sex_or_gender_source=(
                "clinical_metadata"
                if optional_string(record.get("sex"))
                else None
            ),
            skin_tone_system=(
                "fitzpatrick"
                if optional_string(record.get("fitzpatrick_skin_type"))
                else None
            ),
            skin_tone=(
                f"FST_{optional_string(record.get('fitzpatrick_skin_type'))}"
                if optional_string(record.get("fitzpatrick_skin_type"))
                else None
            ),
            skin_tone_source=(
                "clinical_metadata"
                if optional_string(record.get("fitzpatrick_skin_type"))
                else None
            ),
            source_metadata={
                "image_type": record.get("image_type"),
                "diagnosis_confirm_type": confirmation,
                "benign_malignant": record.get("benign_malignant"),
                "anatom_site_general": record.get("anatom_site_general"),
                "anatom_site_special": record.get("anatom_site_special"),
                "archive_member": archive_member,
            },
        )
        row["include"] = True
        row["exclusion_reason"] = None
        row.update(
            {
                "training_schema_version": TRAINING_SCHEMA_VERSION,
                "image_modality": "clinical",
                "training_role": _training_role(row, True),
                "caption": None,
                "visual_concepts": None,
                "body_location": optional_string(
                    record.get("anatom_site_special")
                )
                or optional_string(record.get("anatom_site_general")),
                "symptoms": None,
                "source_split": "official_release",
                "source_name": "hiba",
                "clinical_filter_rule": str(record["image_type"]),
                "duplicate_match_scope": None,
            }
        )
        rows.append(row)
    return rows


class _FingerprintIndex:
    def __init__(self) -> None:
        self.sha256: set[str] = set()
        self.tree = BKTree()
        self.hash_members: dict[int, list[str]] = defaultdict(list)

    @classmethod
    def from_rows(cls, rows: Iterable[dict[str, Any]]) -> "_FingerprintIndex":
        index = cls()
        for row in rows:
            sha256 = optional_string(row.get("image_sha256"))
            perceptual = optional_string(row.get("perceptual_hash"))
            if sha256:
                index.sha256.add(sha256)
            if perceptual:
                index.add_perceptual(perceptual)
        return index

    def add_perceptual(self, value: str) -> None:
        integer = int(value, 16)
        if not self.hash_members[integer]:
            self.tree.add(integer)
        self.hash_members[integer].append(value)

    def add(self, sha256: str, perceptual: str) -> None:
        self.sha256.add(sha256)
        self.add_perceptual(perceptual)

    def near(self, perceptual: str, threshold: int = 4) -> bool:
        integer = int(perceptual, 16)
        return any(
            hamming_distance_int(integer, candidate) <= threshold
            for candidate in self.tree.query(integer, threshold)
        )


def _build_protected_index(
    root: Path,
    paths: list[Path],
) -> _FingerprintIndex:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            continue
        available = set(pq.read_schema(path).names)
        if not {"image_sha256", "perceptual_hash"}.issubset(available):
            continue
        rows.extend(
            pq.read_table(
                path,
                columns=["image_sha256", "perceptual_hash"],
            ).to_pylist()
        )
    return _FingerprintIndex.from_rows(rows)


def _fingerprint_and_screen(
    *,
    root: Path,
    rows: list[dict[str, Any]],
    protected_index: _FingerprintIndex,
    train_index: _FingerprintIndex,
) -> None:
    with ImageResolver(root) as resolver:
        for number, row in enumerate(rows, start=1):
            try:
                fingerprint = compute_fingerprint(
                    resolver.read_bytes(str(row["image_uri"]))
                )
            except Exception as exc:
                row["include"] = False
                row["exclusion_reason"] = "image_decode_failed"
                row["duplicate_match_scope"] = type(exc).__name__
                continue
            row["image_sha256"] = fingerprint.image_sha256
            row["perceptual_hash"] = fingerprint.perceptual_hash
            row["perceptual_hash_algorithm"] = PERCEPTUAL_HASH_ALGORITHM

            if fingerprint.image_sha256 in protected_index.sha256:
                row["include"] = False
                row["exclusion_reason"] = "exact_duplicate_of_protected_evaluation"
                row["duplicate_match_scope"] = "protected_evaluation_exact"
                row["deduplication_status"] = "excluded_protected_exact"
            elif protected_index.near(fingerprint.perceptual_hash):
                row["include"] = False
                row["exclusion_reason"] = (
                    "perceptual_duplicate_of_protected_evaluation"
                )
                row["duplicate_match_scope"] = "protected_evaluation_perceptual"
                row["deduplication_status"] = "excluded_protected_perceptual"
            elif fingerprint.image_sha256 in train_index.sha256:
                row["include"] = False
                row["exclusion_reason"] = "exact_duplicate_within_training"
                row["duplicate_match_scope"] = "training_exact"
                row["deduplication_status"] = "redundant_exact"
            else:
                if train_index.near(fingerprint.perceptual_hash):
                    row["duplicate_match_scope"] = "training_perceptual_candidate"
                    row["deduplication_status"] = "perceptual_candidate"
                train_index.add(
                    fingerprint.image_sha256,
                    fingerprint.perceptual_hash,
                )
            if number == 1 or number % 1000 == 0 or number == len(rows):
                print(f"[training-corpus] Fingerprinted {number}/{len(rows)}")


def _archive_member_indexes(
    data_dir: Path,
    archive_names: Iterable[str],
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for archive_name in sorted(set(archive_names)):
        path = data_dir / archive_name
        if not path.is_file():
            raise FileNotFoundError(f"Derm1M archive not found: {path}")
        with zipfile.ZipFile(path) as archive:
            result[archive_name] = set(archive.namelist())
    return result


def _derm1m_archive_for(filename: str) -> str:
    prefix, separator, _ = filename.partition("/")
    if not separator or prefix not in DERM1M_ARCHIVES:
        raise ValueError(f"Unsupported Derm1M filename: {filename!r}")
    return DERM1M_ARCHIVES[prefix]


def _derm1m_archive_member(filename: str) -> str:
    _, separator, member = filename.partition("/")
    if not separator or not member:
        raise ValueError(f"Unsupported Derm1M filename: {filename!r}")
    return member


def _derm1m_group_id(filename: str) -> str:
    prefix, _, remainder = filename.partition("/")
    stem = Path(remainder).stem
    if prefix == "youtube":
        lineage = stem.split("_frame_", 1)[0]
    elif prefix == "IIYI":
        lineage = re.sub(r"_\d+$", "", stem)
    elif prefix == "pubmed":
        match = re.search(r"PMC\d+", stem, re.IGNORECASE)
        lineage = match.group(0).upper() if match else stem
    else:
        lineage = stem
    return _stable_id("DERM1M_GROUP", f"{prefix}/{lineage}")


def _training_role(row: dict[str, Any], has_diagnosis: bool) -> str:
    if not has_diagnosis:
        return "description_only"
    if row.get("disease_id"):
        return "in_domain_diagnosis"
    return "out_of_domain"


def _clean_derm1m_value(value: Any, missing_marker: str) -> str | None:
    text = optional_string(value)
    if text is None or text.casefold() == missing_marker.casefold():
        return None
    return text


def _normalize_derm1m_gender(value: Any) -> str | None:
    text = _normalized_text(value)
    if text in {None, "no gender information", "unknown"}:
        return None
    return text


def _normalized_text(value: Any) -> str | None:
    text = optional_string(value)
    return text.casefold() if text else None


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(float(value))


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _write_training_rows(rows: list[dict[str, Any]], path: Path) -> None:
    table = pa.Table.from_pylist(rows, schema=TRAINING_ARROW_SCHEMA)
    pq.write_table(table, path, compression="zstd")


def _write_reports(
    *,
    rows: list[dict[str, Any]],
    derm1m_decisions: list[dict[str, Any]],
    reports_dir: Path,
) -> None:
    frame = pd.DataFrame(rows)
    (
        frame.groupby(
            ["dataset_id", "training_role", "include"],
            dropna=False,
        )
        .size()
        .rename("image_count")
        .reset_index()
        .to_csv(reports_dir / "source_summary.csv", index=False)
    )
    (
        frame[frame["include"] & frame["disease_id"].notna()]
        .groupby(["disease_id", "canonical_source_label", "dataset_id"])
        .size()
        .rename("image_count")
        .reset_index()
        .sort_values(["disease_id", "dataset_id"])
        .to_csv(reports_dir / "class_distribution.csv", index=False)
    )
    exclusion = frame[~frame["include"]][
        [
            "sample_id",
            "dataset_id",
            "image_uri",
            "exclusion_reason",
            "duplicate_match_scope",
        ]
    ]
    exclusion.to_csv(reports_dir / "excluded_candidates.csv", index=False)
    decisions = pd.DataFrame(derm1m_decisions)
    (
        decisions.groupby(
            ["source", "include_as_clinical", "clinical_filter_rule"],
            dropna=False,
        )
        .size()
        .rename("image_count")
        .reset_index()
        .to_csv(reports_dir / "derm1m_filter_summary.csv", index=False)
    )


def _write_release(
    *,
    root: Path,
    output_dir: Path,
    all_rows: list[dict[str, Any]],
    included_rows: list[dict[str, Any]],
    protected_paths: list[Path],
) -> dict[str, Any]:
    all_frame = pd.DataFrame(all_rows)
    included_frame = pd.DataFrame(included_rows)
    artifact_paths = {
        "train_images": output_dir / "train_images.parquet",
        "teacher_annotation_queue": output_dir
        / "teacher_annotation_queue.parquet",
        "all_candidates": output_dir / "all_candidates.parquet",
        "derm1m_filter_decisions": output_dir
        / "reports/derm1m_filter_decisions.parquet",
    }
    document = {
        "release": {
            "id": "dermatology_multimodal_training_v1",
            "version": "1.0.0",
            "purpose": "training_only",
            "base_train_images": int(
                (all_frame["clinical_filter_rule"] == "existing_frozen_train").sum()
            ),
            "total_train_images": len(included_frame),
            "images_by_dataset": {
                str(key): int(value)
                for key, value in included_frame["dataset_id"]
                .value_counts()
                .sort_index()
                .items()
            },
            "images_by_training_role": {
                str(key): int(value)
                for key, value in included_frame["training_role"]
                .value_counts()
                .sort_index()
                .items()
            },
            "excluded_new_candidates": int((~all_frame["include"]).sum()),
            "exclusions_by_reason": {
                str(key): int(value)
                for key, value in all_frame.loc[
                    ~all_frame["include"],
                    "exclusion_reason",
                ]
                .value_counts()
                .sort_index()
                .items()
            },
            "protected_evaluation_manifests": [
                path.as_posix() for path in protected_paths
            ],
            "rules": {
                "clinical_photographs_only": True,
                "benchmark_splits_modified": False,
                "protected_exact_and_perceptual_duplicates_excluded": True,
                "perceptual_hamming_threshold": 4,
                "derm1m_filter": "metadata_high_precision_v1",
            },
            "artifacts": {
                "train_images": "train_images.parquet",
                "teacher_annotation_queue": "teacher_annotation_queue.parquet",
                "all_candidates": "all_candidates.parquet",
                "reports": "reports/",
            },
            "artifact_sha256": {
                name: _sha256_file(path)
                for name, path in artifact_paths.items()
            },
        }
    }
    release_dir = output_dir / "release"
    release_dir.mkdir(parents=True, exist_ok=True)
    release_path = release_dir / "training_release_v1.json"
    release_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return document


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the augmented dermatology multimodal training corpus."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--limit-derm1m",
        type=int,
        default=None,
        help="Optional deterministic Derm1M limit for smoke tests.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the existing materialized release without rebuilding it.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.validate_only:
        result = validate_training_corpus(args.project_root)
    else:
        result = build_training_corpus(
            args.project_root,
            limit_derm1m=args.limit_derm1m,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
