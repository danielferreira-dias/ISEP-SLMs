"""Cross-dataset inventory and preliminary disease coverage reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.data_pipeline.common import DiseaseMapper, normalize_label


def build_reports(
    *,
    manifest_paths: dict[str, Path],
    contributor_ids: list[str],
    mapper: DiseaseMapper,
    policy: dict[str, Any],
    root: Path,
) -> dict[str, Path]:
    """Generate source-label inventory, coverage, long-tail, and mapping reports."""

    inventory = _source_inventory(manifest_paths, mapper)
    all_source_coverage = _all_source_coverage(manifest_paths, mapper)
    demographic_availability = _demographic_availability(
        manifest_paths,
        policy,
    )
    subgroup_coverage = _subgroup_coverage(
        manifest_paths,
        policy,
    )
    coverage = _coverage_table(
        manifest_paths=manifest_paths,
        contributor_ids=contributor_ids,
        mapper=mapper,
        policy=policy,
    )

    outputs = policy["outputs"]
    inventory_path = root / outputs["source_label_inventory"]
    all_source_coverage_path = root / outputs["all_source_disease_coverage"]
    demographic_availability_path = root / outputs["demographic_availability"]
    subgroup_coverage_path = root / outputs["subgroup_coverage"]
    coverage_csv_path = root / outputs["coverage_csv"]
    coverage_parquet_path = root / outputs["coverage_parquet"]
    included_path = root / outputs["included_diseases"]
    long_tail_path = root / outputs["long_tail_diseases"]
    out_of_scope_path = root / outputs["out_of_scope_labels"]

    for path in [
        inventory_path,
        all_source_coverage_path,
        demographic_availability_path,
        subgroup_coverage_path,
        coverage_csv_path,
        coverage_parquet_path,
        included_path,
        long_tail_path,
        out_of_scope_path,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    inventory.to_csv(inventory_path, index=False)
    all_source_coverage.to_csv(all_source_coverage_path, index=False)
    demographic_availability.to_csv(demographic_availability_path, index=False)
    subgroup_coverage.to_csv(subgroup_coverage_path, index=False)
    coverage.to_csv(coverage_csv_path, index=False)
    pq.write_table(pa.Table.from_pandas(coverage, preserve_index=False), coverage_parquet_path)

    summary = coverage[coverage["dataset_id"] == "all_contributors"].copy()
    included = summary[summary["eligibility_status"] == "pending_split_validation"]
    long_tail = summary[summary["eligibility_status"] != "pending_split_validation"]
    long_tail.to_csv(long_tail_path, index=False)

    out_of_scope = inventory[
        inventory["mapping_status"] == "out_of_benchmark_scope"
    ].copy()
    out_of_scope.to_csv(out_of_scope_path, index=False)

    included_document = {
        "selection": {
            "id": included_path.stem,
            "status": "provisional_pending_split_validation",
            "description": (
                "Diseases that pass total-support and independent-dataset thresholds. "
                "Final inclusion still requires group-safe split validation and "
                "duplicate analysis."
            ),
            "diseases": [
                {
                    "disease_id": row["disease_id"],
                    "canonical_name": row["canonical_name"],
                    "unique_group_count": int(row["unique_group_count"]),
                    "image_count": int(row["image_count"]),
                    "contributing_dataset_count": int(
                        row["contributing_dataset_count"]
                    ),
                }
                for row in included.to_dict(orient="records")
            ],
        }
    }
    with included_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            included_document,
            handle,
            sort_keys=False,
            allow_unicode=False,
        )

    return {
        "source_label_inventory": inventory_path,
        "all_source_disease_coverage": all_source_coverage_path,
        "demographic_availability": demographic_availability_path,
        "subgroup_coverage": subgroup_coverage_path,
        "coverage_csv": coverage_csv_path,
        "coverage_parquet": coverage_parquet_path,
        "included_diseases": included_path,
        "long_tail_diseases": long_tail_path,
        "out_of_scope_labels": out_of_scope_path,
    }


def write_combined_pool(
    manifest_paths: dict[str, Path],
    contributor_ids: Iterable[str],
    output_path: Path,
) -> int:
    """Concatenate contributor manifests without applying final split selection."""

    tables = [pq.read_table(manifest_paths[dataset_id]) for dataset_id in contributor_ids]
    combined = pa.concat_tables(tables)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(combined, output_path, compression="zstd")
    return combined.num_rows


def _source_inventory(
    manifest_paths: dict[str, Path],
    mapper: DiseaseMapper,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for dataset_id, manifest_path in manifest_paths.items():
        frame = pq.read_table(
            manifest_path,
            columns=[
                "sample_id",
                "leakage_group_id",
                "disease_original",
                "disease_id",
                "reference_diagnoses",
            ],
        ).to_pandas()
        frame["group_id"] = frame["leakage_group_id"]
        for row in frame.to_dict(orient="records"):
            for reference in row["reference_diagnoses"]:
                records.append(
                    {
                        "dataset_id": dataset_id,
                        "sample_id": row["sample_id"],
                        "group_id": row["group_id"],
                        "disease_original": reference["disease_original"],
                        "normalized_source_label": normalize_label(
                            reference["disease_original"]
                        ),
                        "canonical_source_label": mapper.canonical_source_label(
                            dataset_id, reference["disease_original"]
                        ),
                        "disease_id": reference["disease_id"],
                        "mapping_status": reference["mapping_status"],
                        "is_primary": int(reference["rank"]) == 1,
                    }
                )

    columns = [
        "dataset_id",
        "disease_original",
        "normalized_source_label",
        "canonical_source_label",
        "disease_id",
        "mapping_status",
        "any_reference_image_count",
        "any_reference_group_count",
        "primary_image_count",
        "primary_group_count",
    ]
    if not records:
        return pd.DataFrame(columns=columns)

    exploded = pd.DataFrame.from_records(records)
    grouped_records: list[dict[str, Any]] = []
    for keys, group in exploded.groupby(
        [
            "dataset_id",
            "disease_original",
            "normalized_source_label",
            "canonical_source_label",
            "disease_id",
            "mapping_status",
        ],
        dropna=False,
        sort=True,
    ):
        (
            dataset_id,
            disease_original,
            normalized_source_label,
            canonical_source_label,
            disease_id,
            mapping_status,
        ) = keys
        primary = group[group["is_primary"]]
        grouped_records.append(
            {
                "dataset_id": dataset_id,
                "disease_original": disease_original,
                "normalized_source_label": normalized_source_label,
                "canonical_source_label": canonical_source_label,
                "disease_id": disease_id,
                "mapping_status": mapping_status,
                "any_reference_image_count": int(group["sample_id"].nunique()),
                "any_reference_group_count": int(group["group_id"].nunique()),
                "primary_image_count": int(primary["sample_id"].nunique()),
                "primary_group_count": int(primary["group_id"].nunique()),
            }
        )
    return pd.DataFrame(grouped_records, columns=columns).sort_values(
        ["dataset_id", "primary_group_count", "disease_original"],
        ascending=[True, False, True],
        ignore_index=True,
    )


def _all_source_coverage(
    manifest_paths: dict[str, Path],
    mapper: DiseaseMapper,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for dataset_id, manifest_path in manifest_paths.items():
        frame = pq.read_table(
            manifest_path,
            columns=[
                "sample_id",
                "leakage_group_id",
                "reference_diagnoses",
            ],
        ).to_pandas()
        frame["group_id"] = frame["leakage_group_id"]
        for row in frame.to_dict(orient="records"):
            for reference in row["reference_diagnoses"]:
                records.append(
                    {
                        "dataset_id": dataset_id,
                        "sample_id": row["sample_id"],
                        "group_id": row["group_id"],
                        "disease_original": reference["disease_original"],
                        "canonical_source_label": reference[
                            "canonical_source_label"
                        ],
                        "benchmark_disease_id": reference["disease_id"],
                        "is_primary": int(reference["rank"]) == 1,
                    }
                )

    columns = [
        "canonical_source_label",
        "benchmark_disease_id",
        "mapping_status",
        "dataset_id",
        "primary_group_count",
        "primary_image_count",
        "any_reference_group_count",
        "any_reference_image_count",
        "source_label_count",
    ]
    if not records:
        return pd.DataFrame(columns=columns)

    exploded = pd.DataFrame.from_records(records)
    report_records: list[dict[str, Any]] = []
    dataset_values = [*sorted(manifest_paths), "all_sources"]
    for canonical_label, canonical_rows in exploded.groupby(
        "canonical_source_label",
        sort=True,
    ):
        mapped_ids = sorted(
            {
                str(value)
                for value in canonical_rows["benchmark_disease_id"].dropna()
            }
        )
        if len(mapped_ids) > 1:
            raise ValueError(
                f"Canonical source label {canonical_label!r} maps to multiple "
                f"benchmark IDs: {mapped_ids}"
            )
        benchmark_id = mapped_ids[0] if mapped_ids else None
        for dataset_id in dataset_values:
            rows = (
                canonical_rows
                if dataset_id == "all_sources"
                else canonical_rows[canonical_rows["dataset_id"] == dataset_id]
            )
            if rows.empty and dataset_id != "all_sources":
                continue
            primary = rows[rows["is_primary"]]
            report_records.append(
                {
                    "canonical_source_label": canonical_label,
                    "benchmark_disease_id": benchmark_id,
                    "mapping_status": (
                        "benchmark_mapped"
                        if benchmark_id is not None
                        else "out_of_benchmark_scope"
                    ),
                    "dataset_id": dataset_id,
                    "primary_group_count": int(primary["group_id"].nunique()),
                    "primary_image_count": int(primary["sample_id"].nunique()),
                    "any_reference_group_count": int(rows["group_id"].nunique()),
                    "any_reference_image_count": int(rows["sample_id"].nunique()),
                    "source_label_count": int(
                        rows[["dataset_id", "disease_original"]]
                        .drop_duplicates()
                        .shape[0]
                    ),
                }
            )

    return pd.DataFrame(report_records, columns=columns).sort_values(
        ["dataset_id", "primary_group_count", "canonical_source_label"],
        ascending=[True, False, True],
        ignore_index=True,
    )


def _coverage_table(
    *,
    manifest_paths: dict[str, Path],
    contributor_ids: list[str],
    mapper: DiseaseMapper,
    policy: dict[str, Any],
) -> pd.DataFrame:
    frames: dict[str, pd.DataFrame] = {}
    for dataset_id in contributor_ids:
        frame = pq.read_table(
            manifest_paths[dataset_id],
            columns=[
                "sample_id",
                "leakage_group_id",
                "disease_id",
                "diagnosis_gradable",
                "include",
            ],
        ).to_pandas()
        frame["group_id"] = frame["leakage_group_id"]
        frames[dataset_id] = frame[
            frame["diagnosis_gradable"]
            & frame["disease_id"].notna()
            & frame["include"]
        ]

    minimum_groups = int(policy["eligibility"]["minimum_unique_groups_total"])
    minimum_datasets = int(policy["eligibility"]["minimum_contributing_datasets"])
    records: list[dict[str, Any]] = []

    for disease in mapper.diseases:
        dataset_counts: list[tuple[str, int, int]] = []
        for dataset_id, frame in frames.items():
            disease_rows = frame[frame["disease_id"] == disease.disease_id]
            dataset_counts.append(
                (
                    dataset_id,
                    int(disease_rows["group_id"].nunique()),
                    int(disease_rows["sample_id"].nunique()),
                )
            )

        total_groups = sum(group_count for _, group_count, _ in dataset_counts)
        total_images = sum(image_count for _, _, image_count in dataset_counts)
        contributing_count = sum(
            group_count > 0
            for _, group_count, _ in dataset_counts
        )
        status, reason = _support_status(
            total_groups=total_groups,
            contributing_count=contributing_count,
            minimum_groups=minimum_groups,
            minimum_datasets=minimum_datasets,
        )

        for dataset_id, group_count, image_count in [
            *dataset_counts,
            ("all_contributors", total_groups, total_images),
        ]:
            records.append(
                {
                    "disease_id": disease.disease_id,
                    "canonical_name": disease.canonical_name,
                    "dataset_id": dataset_id,
                    "unique_group_count": group_count,
                    "image_count": image_count,
                    "train_group_count": pd.NA,
                    "validation_group_count": pd.NA,
                    "test_group_count": pd.NA,
                    "contributing_dataset_count": contributing_count,
                    "eligibility_status": status,
                    "exclusion_reason": reason,
                }
            )

    return pd.DataFrame.from_records(records)


def _support_status(
    *,
    total_groups: int,
    contributing_count: int,
    minimum_groups: int,
    minimum_datasets: int,
) -> tuple[str, str | None]:
    reasons: list[str] = []
    if total_groups < minimum_groups:
        reasons.append("insufficient_unique_groups")
    if contributing_count < minimum_datasets:
        reasons.append("insufficient_independent_datasets")
    if reasons:
        return "long_tail", ";".join(reasons)
    return "pending_split_validation", None


def _demographic_availability(
    manifest_paths: dict[str, Path],
    policy: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "dataset_id",
        "dataset_role",
        "record_scope",
        "image_count",
        "unique_group_count",
        "exact_age_group_count",
        "standardized_age_group_count",
        "standardized_age_group_coverage",
        "skin_tone_group_count",
        "skin_tone_group_coverage",
        "sex_or_gender_group_count",
        "sex_or_gender_group_coverage",
        "race_ethnicity_group_count",
        "race_ethnicity_group_coverage",
    ]
    records: list[dict[str, Any]] = []
    for dataset_id, path in manifest_paths.items():
        frame = pq.read_table(
            path,
            columns=[
                "sample_id",
                "leakage_group_id",
                "include",
                "age_years",
                "age_group_standardized",
                "skin_tone",
                "sex_or_gender",
                "race_ethnicity",
            ],
        ).to_pandas()
        frame["group_id"] = frame["leakage_group_id"]
        for scope, scoped in [
            ("all_source_rows", frame),
            ("benchmark_mapped_rows", frame[frame["include"]]),
        ]:
            total_groups = int(scoped["group_id"].nunique())
            informative_age = scoped[
                _informative_demographic_mask(
                    scoped["age_group_standardized"],
                    dimension="age_group",
                )
            ]
            informative_sex_or_gender = scoped[
                _informative_demographic_mask(
                    scoped["sex_or_gender"],
                    dimension="sex_or_gender",
                )
            ]
            informative_race_ethnicity = scoped[
                _informative_demographic_mask(
                    scoped["race_ethnicity"],
                    dimension="race_ethnicity",
                )
            ]
            records.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_role": _dataset_role(dataset_id, policy),
                    "record_scope": scope,
                    "image_count": int(scoped["sample_id"].nunique()),
                    "unique_group_count": total_groups,
                    "exact_age_group_count": int(
                        scoped.loc[scoped["age_years"].notna(), "group_id"].nunique()
                    ),
                    "standardized_age_group_count": int(
                        informative_age["group_id"].nunique()
                    ),
                    "standardized_age_group_coverage": _coverage_fraction(
                        informative_age["group_id"].nunique(),
                        total_groups,
                    ),
                    "skin_tone_group_count": int(
                        scoped.loc[scoped["skin_tone"].notna(), "group_id"].nunique()
                    ),
                    "skin_tone_group_coverage": _coverage_fraction(
                        scoped.loc[
                            scoped["skin_tone"].notna(),
                            "group_id",
                        ].nunique(),
                        total_groups,
                    ),
                    "sex_or_gender_group_count": int(
                        informative_sex_or_gender["group_id"].nunique()
                    ),
                    "sex_or_gender_group_coverage": _coverage_fraction(
                        informative_sex_or_gender["group_id"].nunique(),
                        total_groups,
                    ),
                    "race_ethnicity_group_count": int(
                        informative_race_ethnicity["group_id"].nunique()
                    ),
                    "race_ethnicity_group_coverage": _coverage_fraction(
                        informative_race_ethnicity["group_id"].nunique(),
                        total_groups,
                    ),
                }
            )
    return pd.DataFrame(records, columns=columns)


def _subgroup_coverage(
    manifest_paths: dict[str, Path],
    policy: dict[str, Any],
) -> pd.DataFrame:
    manifest_frames: dict[str, pd.DataFrame] = {}
    for dataset_id, path in manifest_paths.items():
        frame = pq.read_table(
            path,
            columns=[
                "sample_id",
                "leakage_group_id",
                "include",
                "disease_id",
                "age_group_standardized",
                "skin_tone_system",
                "skin_tone",
                "sex_or_gender_system",
                "sex_or_gender",
                "race_ethnicity_source",
                "race_ethnicity",
            ],
        ).to_pandas()
        frame["group_id"] = frame["leakage_group_id"]
        manifest_frames[dataset_id] = frame[frame["include"]].copy()

    contributor_ids = policy["dataset_roles"]["taxonomy_contributors"]
    contributor_frames = [
        manifest_frames[dataset_id]
        for dataset_id in contributor_ids
        if dataset_id in manifest_frames
    ]
    scopes = dict(manifest_frames)
    if contributor_frames:
        scopes["all_contributors"] = pd.concat(
            contributor_frames,
            ignore_index=True,
        )

    minimum_overall = int(
        policy["subgroup_reporting"]["minimum_subgroup_unique_groups"]
    )
    minimum_per_disease = int(
        policy["subgroup_reporting"][
            "minimum_disease_subgroup_unique_groups"
        ]
    )
    records: list[dict[str, Any]] = []
    for dataset_id, frame in scopes.items():
        if frame.empty:
            continue
        dimension_frames = [
            (
                "age_group",
                pd.Series("standardized_age_v1", index=frame.index),
                frame["age_group_standardized"],
            ),
            (
                "skin_tone",
                frame["skin_tone_system"],
                frame["skin_tone"],
            ),
            (
                "sex_or_gender",
                frame["sex_or_gender_system"],
                frame["sex_or_gender"],
            ),
            (
                "race_ethnicity",
                frame["race_ethnicity_source"],
                frame["race_ethnicity"],
            ),
        ]
        for dimension, systems, values in dimension_frames:
            dimension_frame = frame.assign(
                _subgroup_system=systems,
                _subgroup_value=values,
            ).dropna(subset=["_subgroup_system", "_subgroup_value"])
            dimension_frame = dimension_frame[
                _informative_demographic_mask(
                    dimension_frame["_subgroup_value"],
                    dimension=dimension,
                )
            ]
            for (system, value), subgroup in dimension_frame.groupby(
                ["_subgroup_system", "_subgroup_value"],
                sort=True,
            ):
                _append_subgroup_record(
                    records,
                    dataset_id=dataset_id,
                    dimension=dimension,
                    system=str(system),
                    value=str(value),
                    disease_id="all_active_diseases",
                    frame=subgroup,
                    minimum_required=minimum_overall,
                )
                for disease_id, disease_rows in subgroup.groupby(
                    "disease_id",
                    sort=True,
                ):
                    _append_subgroup_record(
                        records,
                        dataset_id=dataset_id,
                        dimension=dimension,
                        system=str(system),
                        value=str(value),
                        disease_id=str(disease_id),
                        frame=disease_rows,
                        minimum_required=minimum_per_disease,
                    )

    columns = [
        "dataset_id",
        "subgroup_dimension",
        "subgroup_system",
        "subgroup_value",
        "disease_id",
        "unique_group_count",
        "image_count",
        "minimum_required",
        "eligible_for_metric",
    ]
    return pd.DataFrame(records, columns=columns).sort_values(
        [
            "dataset_id",
            "subgroup_dimension",
            "subgroup_system",
            "subgroup_value",
            "disease_id",
        ],
        ignore_index=True,
    )


def _append_subgroup_record(
    records: list[dict[str, Any]],
    *,
    dataset_id: str,
    dimension: str,
    system: str,
    value: str,
    disease_id: str,
    frame: pd.DataFrame,
    minimum_required: int,
) -> None:
    group_count = int(frame["group_id"].nunique())
    records.append(
        {
            "dataset_id": dataset_id,
            "subgroup_dimension": dimension,
            "subgroup_system": system,
            "subgroup_value": value,
            "disease_id": disease_id,
            "unique_group_count": group_count,
            "image_count": int(frame["sample_id"].nunique()),
            "minimum_required": minimum_required,
            "eligible_for_metric": group_count >= minimum_required,
        }
    )


def _coverage_fraction(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def _informative_demographic_mask(
    values: pd.Series,
    *,
    dimension: str,
) -> pd.Series:
    """Select values that define reportable subgroups rather than missingness."""

    excluded_values = {
        "age_group": {"unknown"},
        "sex_or_gender": {"other_or_unspecified", "unknown", "unspecified"},
        "race_ethnicity": {
            "prefer_not_to_answer",
            "unknown",
            "unspecified",
        },
    }.get(dimension, set())
    normalized = values.astype("string").str.casefold()
    return values.notna() & ~normalized.isin(excluded_values)


def _dataset_role(dataset_id: str, policy: dict[str, Any]) -> str:
    roles = policy["dataset_roles"]
    if dataset_id in roles["taxonomy_contributors"]:
        return "taxonomy_contributor"
    if dataset_id in roles["external_evaluation_only"]:
        return "external_evaluation_only"
    if dataset_id in roles["excluded_from_coverage"]:
        return "excluded_from_coverage"
    return "unregistered"
