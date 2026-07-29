"""Source-specific adapters for benchmark and audit-only classification data."""

from __future__ import annotations

import ast
import glob
import hashlib
from pathlib import Path
from typing import Any, Callable, Iterable
import zipfile

import pandas as pd
import pyarrow.parquet as pq

from src.data_pipeline.common import (
    DiseaseMapper,
    make_manifest_row,
    optional_string,
    reference_diagnosis,
    standardize_age_group,
)


ProgressCallback = Callable[[str], None]


def build_fitzpatrick17k_c(
    root: Path,
    config: dict[str, Any],
    mapper: DiseaseMapper,
    progress: ProgressCallback,
) -> list[dict[str, Any]]:
    dataset = config["dataset"]
    source = config["source"]
    metadata_path = root / source["metadata"]
    archive_path = root / source["image_archive"]
    with zipfile.ZipFile(archive_path) as archive:
        archive_members = set(archive.namelist())
    frame = pd.read_csv(metadata_path)
    progress(f"Reading {len(frame)} Fitzpatrick17k-C metadata rows")

    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        image_id = str(record["md5hash"])
        original_name = str(record["orig_img_name"])
        archive_member = f"data/finalfitz17k/{original_name}"
        if archive_member not in archive_members:
            raise ValueError(
                f"Fitzpatrick17k-C image {archive_member!r} was not found in the archive"
            )
        diagnosis = reference_diagnosis(
            mapper=mapper,
            dataset_id=dataset["id"],
            disease_original=record.get("label"),
            rank=1,
            weight=1.0,
            source="label",
        )
        fst = _fitzpatrick_value(record.get("fst"))
        rows.append(
            make_manifest_row(
                mapper=mapper,
                dataset_id=dataset["id"],
                dataset_version=str(dataset["version"]),
                sample_id=f"FITZPATRICK17K_C_{image_id}",
                original_image_id=image_id,
                original_case_id=None,
                patient_id=None,
                group_id=f"FITZPATRICK17K_C_IMAGE_{image_id}",
                image_uri=_zip_uri(root, archive_path, archive_member),
                reference_diagnoses=[diagnosis],
                diagnosis_basis="atlas_label",
                diagnosis_gradable=bool(optional_string(record.get("label"))),
                license_id=dataset["license_id"],
                skin_tone_system="fitzpatrick" if fst is not None else None,
                skin_tone=f"FST_{fst}" if fst is not None else None,
                skin_tone_source="expert_or_crowd_annotated" if fst is not None else None,
                source_metadata={
                    "diag": record.get("diag"),
                    "source_partition": record.get("partition"),
                    "nine_partition_label": record.get("nine_partition_label"),
                    "three_partition_label": record.get("three_partition_label"),
                    "qc": record.get("qc"),
                    "source_url": record.get("url"),
                    "archive_member": archive_member,
                },
            )
        )
    return rows


def build_pad_ufes_20(
    root: Path,
    config: dict[str, Any],
    mapper: DiseaseMapper,
    progress: ProgressCallback,
) -> list[dict[str, Any]]:
    dataset = config["dataset"]
    source = config["source"]
    metadata_path = root / source["metadata"]
    archive_paths = [root / item for item in source["image_archives"]]
    archive_index = _index_zip_members(root, archive_paths)
    frame = pd.read_csv(metadata_path)
    progress(f"Reading {len(frame)} PAD-UFES-20 metadata rows")

    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        image_name = str(record["img_id"])
        archive_entry = archive_index.get(image_name)
        if archive_entry is None:
            raise ValueError(f"PAD-UFES-20 image {image_name!r} was not found in the archives")
        archive_path, archive_member = archive_entry
        patient_id = str(record["patient_id"])
        lesion_id = str(record["lesion_id"])
        biopsied = bool(record.get("biopsed"))
        diagnosis = reference_diagnosis(
            mapper=mapper,
            dataset_id=dataset["id"],
            disease_original=record.get("diagnostic"),
            rank=1,
            weight=1.0,
            source="diagnostic",
        )
        fst = _fitzpatrick_value(record.get("fitspatrick"))
        age_years = _optional_age(record.get("age"))
        gender = optional_string(record.get("gender"))
        rows.append(
            make_manifest_row(
                mapper=mapper,
                dataset_id=dataset["id"],
                dataset_version=str(dataset["version"]),
                sample_id=f"PAD_UFES_20_{Path(image_name).stem}",
                original_image_id=image_name,
                original_case_id=lesion_id,
                patient_id=patient_id,
                group_id=f"PAD_UFES_20_PATIENT_{patient_id}",
                image_uri=_zip_uri(root, archive_path, archive_member),
                reference_diagnoses=[diagnosis],
                diagnosis_basis="pathology" if biopsied else "clinical_consensus",
                diagnosis_gradable=True,
                license_id=dataset["license_id"],
                age_years=age_years,
                age_group_standardized=standardize_age_group(
                    age_years=age_years
                ),
                age_source="clinical_metadata" if age_years is not None else None,
                sex_or_gender=gender.casefold() if gender is not None else None,
                sex_or_gender_system="source_gender" if gender is not None else None,
                sex_or_gender_source="clinical_metadata" if gender is not None else None,
                skin_tone_system="fitzpatrick" if fst is not None else None,
                skin_tone=f"FST_{fst}" if fst is not None else None,
                skin_tone_source="source_metadata" if fst is not None else None,
                source_metadata={
                    "lesion_id": lesion_id,
                    "biopsed": biopsied,
                    "region": record.get("region"),
                    "archive_member": archive_member,
                },
            )
        )
    return rows


def build_ddi(
    root: Path,
    config: dict[str, Any],
    mapper: DiseaseMapper,
    progress: ProgressCallback,
) -> list[dict[str, Any]]:
    dataset = config["dataset"]
    source = config["source"]
    metadata_path = root / source["metadata"]
    image_root = root / source["image_root"]
    frame = pd.read_csv(metadata_path)
    progress(f"Reading {len(frame)} DDI metadata rows")

    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        numeric_id = int(record["DDI_ID"])
        image_name = str(record["DDI_file"])
        image_path = image_root / image_name
        if not image_path.is_file():
            raise ValueError(f"DDI image {image_path} does not exist")
        diagnosis = reference_diagnosis(
            mapper=mapper,
            dataset_id=dataset["id"],
            disease_original=record.get("disease"),
            rank=1,
            weight=1.0,
            source="disease",
        )
        skin_tone = optional_string(record.get("skin_tone"))
        rows.append(
            make_manifest_row(
                mapper=mapper,
                dataset_id=dataset["id"],
                dataset_version=str(dataset["version"]),
                sample_id=f"DDI_{numeric_id:06d}",
                original_image_id=str(numeric_id),
                original_case_id=str(numeric_id),
                patient_id=None,
                group_id=f"DDI_IMAGE_{numeric_id:06d}",
                image_uri=image_path.relative_to(root).as_posix(),
                reference_diagnoses=[diagnosis],
                diagnosis_basis="pathology",
                diagnosis_gradable=True,
                license_id=dataset["license_id"],
                skin_tone_system="fitzpatrick_group" if skin_tone is not None else None,
                skin_tone=f"FST_{skin_tone}" if skin_tone is not None else None,
                skin_tone_source="expert_annotated" if skin_tone is not None else None,
                source_metadata={
                    "malignant": record.get("malignant"),
                    "patient_grouping_available": False,
                },
            )
        )
    return rows


def build_scin(
    root: Path,
    config: dict[str, Any],
    mapper: DiseaseMapper,
    progress: ProgressCallback,
) -> list[dict[str, Any]]:
    dataset = config["dataset"]
    source = config["source"]
    source_files = sorted(
        Path(path)
        for path in glob.glob(str(root / source["files"]))
    )
    if not source_files:
        raise ValueError(f"No SCIN shards matched {source['files']!r}")

    base_columns = [
        "case_id",
        "age_group",
        "sex_at_birth",
        "combined_race",
        "race_ethnicity_two_or_more_after_mitigation",
        "weighted_skin_condition_label",
        "dermatologist_skin_condition_on_label_name",
        "dermatologist_skin_condition_confidence",
        "dermatologist_gradable_for_skin_condition_1",
        "dermatologist_gradable_for_skin_condition_2",
        "dermatologist_gradable_for_skin_condition_3",
        "fitzpatrick_skin_type",
        "dermatologist_fitzpatrick_skin_type_label_1",
        "dermatologist_fitzpatrick_skin_type_label_2",
        "dermatologist_fitzpatrick_skin_type_label_3",
        "monk_skin_tone_label_india",
        "monk_skin_tone_label_us",
    ]

    rows: list[dict[str, Any]] = []
    for shard_number, source_file in enumerate(source_files, start=1):
        progress(f"Reading SCIN shard {shard_number}/{len(source_files)}: {source_file.name}")
        frame = pq.read_table(source_file, columns=base_columns).to_pandas()
        image_paths = {
            image_number: pq.read_table(
                source_file,
                columns=[f"image_{image_number}_path.path"],
            ).column(0).to_pylist()
            for image_number in range(1, 4)
        }
        for source_row, record in enumerate(frame.to_dict(orient="records")):
            reference_diagnoses = _parse_scin_differential(
                record.get("weighted_skin_condition_label"),
                mapper,
                dataset["id"],
            )
            gradable = _scin_is_gradable(record) and bool(reference_diagnoses)
            case_id = str(record["case_id"])
            age_group_source = optional_string(record.get("age_group"))
            sex_at_birth = optional_string(record.get("sex_at_birth"))
            race_ethnicity = _normalize_scin_race_ethnicity(
                record.get("combined_race")
            )
            for image_number in range(1, 4):
                image_name = optional_string(image_paths[image_number][source_row])
                if image_name is None:
                    continue
                relative_shard = source_file.relative_to(root).as_posix()
                image_column = f"image_{image_number}_path"
                image_uri = (
                    f"parquet://{relative_shard}"
                    f"::row={source_row}::column={image_column}"
                )
                monk_value, monk_source = _preferred_scin_monk_value(record)
                rows.append(
                    make_manifest_row(
                        mapper=mapper,
                        dataset_id=dataset["id"],
                        dataset_version=str(dataset["version"]),
                        sample_id=f"SCIN_{case_id}_IMAGE_{image_number}",
                        original_image_id=image_name,
                        original_case_id=case_id,
                        patient_id=None,
                        group_id=f"SCIN_CASE_{case_id}",
                        image_uri=image_uri,
                        reference_diagnoses=reference_diagnoses,
                        diagnosis_basis="dermatologist_differential",
                        diagnosis_gradable=gradable,
                        license_id=dataset["license_id"],
                        age_group_source=age_group_source,
                        age_group_standardized=standardize_age_group(
                            source_group=age_group_source
                        ),
                        age_source="self_reported" if age_group_source is not None else None,
                        sex_or_gender=(
                            sex_at_birth.casefold()
                            if sex_at_birth is not None
                            else None
                        ),
                        sex_or_gender_system=(
                            "sex_at_birth"
                            if sex_at_birth is not None
                            else None
                        ),
                        sex_or_gender_source=(
                            "self_reported"
                            if sex_at_birth is not None
                            else None
                        ),
                        race_ethnicity=race_ethnicity,
                        race_ethnicity_source=(
                            "self_reported_combined"
                            if race_ethnicity is not None
                            else None
                        ),
                        skin_tone_system="monk" if monk_value is not None else None,
                        skin_tone=f"MST_{monk_value}" if monk_value is not None else None,
                        skin_tone_source=monk_source,
                        source_metadata={
                            "source_shard": relative_shard,
                            "source_row": source_row,
                            "image_column": image_column,
                            "self_reported_fitzpatrick": record.get("fitzpatrick_skin_type"),
                            "dermatologist_fitzpatrick_1": record.get(
                                "dermatologist_fitzpatrick_skin_type_label_1"
                            ),
                            "dermatologist_fitzpatrick_2": record.get(
                                "dermatologist_fitzpatrick_skin_type_label_2"
                            ),
                            "dermatologist_fitzpatrick_3": record.get(
                                "dermatologist_fitzpatrick_skin_type_label_3"
                            ),
                            "reviewer_condition_labels": record.get(
                                "dermatologist_skin_condition_on_label_name"
                            ),
                            "reviewer_condition_confidence": record.get(
                                "dermatologist_skin_condition_confidence"
                            ),
                        },
                    )
                )
    return rows


def build_dermnet_kaggle(
    root: Path,
    config: dict[str, Any],
    mapper: DiseaseMapper,
    progress: ProgressCallback,
) -> list[dict[str, Any]]:
    """Index the coarse Kaggle directory labels without using them as benchmark truth."""

    dataset = config["dataset"]
    archive_path = root / config["source"]["image_archive"]
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            member
            for member in archive.namelist()
            if not member.endswith("/")
        ]
    progress(f"Reading {len(members)} Dermnet Kaggle archive members")

    for member in members:
        parts = Path(member).parts
        if len(parts) < 3:
            raise ValueError(f"Unexpected Dermnet archive path: {member!r}")
        source_split, category = parts[0], parts[1]
        digest = hashlib.sha256(member.encode("utf-8")).hexdigest()[:20]
        diagnosis = reference_diagnosis(
            mapper=mapper,
            dataset_id=dataset["id"],
            disease_original=category,
            rank=1,
            weight=1.0,
            source="parent_directory",
        )
        rows.append(
            make_manifest_row(
                mapper=mapper,
                dataset_id=dataset["id"],
                dataset_version=str(dataset["version"]),
                sample_id=f"DERMNET_KAGGLE_{digest}",
                original_image_id=member,
                original_case_id=None,
                patient_id=None,
                group_id=f"DERMNET_KAGGLE_IMAGE_{digest}",
                image_uri=_zip_uri(root, archive_path, member),
                reference_diagnoses=[diagnosis],
                diagnosis_basis="atlas_label",
                diagnosis_gradable=True,
                license_id=dataset["license_id"],
                force_exclusion_reason="dataset_excluded_from_benchmark",
                source_metadata={
                    "source_split": source_split,
                    "directory_category": category,
                    "archive_member": member,
                    "label_granularity": "coarse_mixed_category",
                    "patient_grouping_available": False,
                },
            )
        )
    return rows


def build_skindisnet(
    root: Path,
    config: dict[str, Any],
    mapper: DiseaseMapper,
    progress: ProgressCallback,
) -> list[dict[str, Any]]:
    """Index patient-grouped preprocessed images and exclude augmented derivatives."""

    dataset = config["dataset"]
    source = config["source"]
    archive_path = root / source["image_archive"]
    metadata_member = source["metadata_member"]
    preprocessed_root = source["preprocessed_root"]

    with zipfile.ZipFile(archive_path) as archive:
        archive_members = set(archive.namelist())
        with archive.open(metadata_member) as metadata:
            frame = pd.read_csv(metadata)

    progress(f"Reading {len(frame)} SkinDisNet metadata rows")
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        folder_name = str(record["Folder_name"])
        image_id = str(record["Image_id"])
        patient_id = str(record["Patient_id"])
        archive_member = f"{preprocessed_root}/{folder_name}/{image_id}.jpg"
        if archive_member not in archive_members:
            raise ValueError(
                f"SkinDisNet image {archive_member!r} was not found in the archive"
            )

        diagnosis = reference_diagnosis(
            mapper=mapper,
            dataset_id=dataset["id"],
            disease_original=record.get("Diagnosis"),
            rank=1,
            weight=1.0,
            source="Diagnosis",
        )
        age_source = optional_string(record.get("Age"))
        age_numeric = float(age_source) if age_source is not None else None
        age_years = (
            int(age_numeric)
            if age_numeric is not None and age_numeric.is_integer()
            else None
        )
        age_for_grouping = (
            int(age_numeric)
            if age_numeric is not None and age_numeric >= 0
            else None
        )
        sex = optional_string(record.get("Sex"))
        digest = hashlib.sha256(archive_member.encode("utf-8")).hexdigest()[:20]
        rows.append(
            make_manifest_row(
                mapper=mapper,
                dataset_id=dataset["id"],
                dataset_version=str(dataset["version"]),
                sample_id=f"SKINDISNET_{digest}",
                original_image_id=image_id,
                original_case_id=None,
                patient_id=patient_id,
                group_id=f"SKINDISNET_PATIENT_{patient_id}",
                image_uri=_zip_uri(root, archive_path, archive_member),
                reference_diagnoses=[diagnosis],
                diagnosis_basis="dermatologist_review",
                diagnosis_gradable=True,
                license_id=dataset["license_id"],
                age_years=age_years,
                age_group_standardized=standardize_age_group(
                    age_years=age_for_grouping
                ),
                age_source="clinical_metadata" if age_source is not None else None,
                sex_or_gender=sex.casefold() if sex is not None else None,
                sex_or_gender_system="source_gender" if sex is not None else None,
                sex_or_gender_source="clinical_metadata" if sex is not None else None,
                source_metadata={
                    "folder_name": folder_name,
                    "anatomical_site": record.get("Leision_location"),
                    "source_age_years": age_source,
                    "archive_member": archive_member,
                    "preprocessing": "cropped_background_removed_resized_512x512",
                    "augmented_derivatives_excluded": True,
                },
            )
        )
    return rows


def _parse_scin_differential(
    value: Any,
    mapper: DiseaseMapper,
    dataset_id: str,
) -> list[dict[str, Any]]:
    if value is None or (isinstance(value, float) and value != value):
        return []
    if isinstance(value, dict):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid SCIN differential: {text!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a SCIN differential mapping, received {type(parsed)}")

    ordered = sorted(
        enumerate(parsed.items()),
        key=lambda item: (-float(item[1][1]), item[0]),
    )
    return [
        reference_diagnosis(
            mapper=mapper,
            dataset_id=dataset_id,
            disease_original=label,
            rank=rank,
            weight=float(weight),
            source="weighted_skin_condition_label",
        )
        for rank, (_, (label, weight)) in enumerate(ordered, start=1)
    ]


def _scin_is_gradable(record: dict[str, Any]) -> bool:
    values = [
        record.get(f"dermatologist_gradable_for_skin_condition_{index}")
        for index in range(1, 4)
    ]
    return any(
        value is not None and "YES" in str(value).upper()
        for value in values
    )


def _normalize_scin_race_ethnicity(value: Any) -> str | None:
    """Normalize SCIN's combined category without exposing its boolean helper."""

    normalized = optional_string(value)
    if normalized is None:
        return None
    normalized = normalized.casefold()
    if normalized == "two_or_more_after_mitigation":
        return "two_or_more_races"
    return normalized


def _preferred_scin_monk_value(
    record: dict[str, Any],
) -> tuple[str | None, str | None]:
    for field, source in [
        ("monk_skin_tone_label_us", "trained_layperson_us"),
        ("monk_skin_tone_label_india", "trained_layperson_india"),
    ]:
        value = optional_string(record.get(field))
        if value is not None:
            if value.endswith(".0"):
                value = value[:-2]
            return value, source
    return None, None


def _fitzpatrick_value(value: Any) -> str | None:
    text = optional_string(value)
    if text is None:
        return None
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric < 1 or numeric > 6:
        return None
    return str(int(numeric)) if numeric.is_integer() else text


def _optional_age(value: Any) -> int | None:
    text = optional_string(value)
    if text is None:
        return None
    numeric = float(text)
    if numeric < 0:
        return None
    return int(numeric)


def _index_zip_members(
    root: Path,
    archive_paths: Iterable[Path],
) -> dict[str, tuple[Path, str]]:
    index: dict[str, tuple[Path, str]] = {}
    for archive_path in archive_paths:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.namelist():
                name = Path(member).name
                if not name:
                    continue
                if name in index:
                    raise ValueError(f"Duplicate image filename {name!r} across ZIP archives")
                index[name] = (archive_path, member)
    return index


def _zip_uri(root: Path, archive_path: Path, member: str) -> str:
    return f"zip://{archive_path.relative_to(root).as_posix()}::{member}"
