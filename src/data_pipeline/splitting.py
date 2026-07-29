"""Leakage-safe grouped splitting and benchmark release generation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.data_pipeline.common import MANIFEST_ARROW_SCHEMA, load_yaml


SPLIT_ALGORITHM_VERSION = "greedy_multilabel_group_stratification_v1"


def build_benchmark_release(
    *,
    root: Path,
    manifest_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Create deterministic internal splits, external sets, and audit reports."""

    config_path = root / "configs/datasets/visual_top_k_split.yaml"
    document = load_yaml(config_path)
    config = document["split"]
    if config["algorithm"] != SPLIT_ALGORITHM_VERSION:
        raise ValueError(
            f"Unsupported split algorithm: {config['algorithm']!r}"
        )
    if manifest_paths is None:
        manifest_paths = _manifest_paths_from_catalog(root)

    internal_ids = list(config["internal"]["source_datasets"])
    external_ids = list(config["external"]["datasets"])
    required_ids = set(internal_ids) | set(external_ids)
    missing = sorted(required_ids - set(manifest_paths))
    if missing:
        raise ValueError(
            "Missing manifests required for benchmark release: "
            + ", ".join(missing)
        )

    internal = _eligible_frame(manifest_paths, internal_ids)
    ratios = {
        str(name): float(value)
        for name, value in config["internal"]["ratios"].items()
    }
    assignments = assign_groups(
        internal,
        ratios=ratios,
        seed=int(config["seed"]),
        secondary_feature_weight=float(
            config["stratification"]["secondary_feature_weight"]
        ),
    )
    internal = internal.copy()
    internal["split"] = internal["leakage_group_id"].map(assignments)
    internal["split_version"] = str(config["version"])

    external_frames: dict[str, pd.DataFrame] = {}
    internal_groups = set(internal["leakage_group_id"].astype(str))
    for dataset_id in external_ids:
        frame = _eligible_frame(manifest_paths, [dataset_id]).copy()
        overlap = internal_groups.intersection(
            frame["leakage_group_id"].astype(str)
        )
        if overlap and config["external"]["overlap_policy"] == "fail":
            raise ValueError(
                f"Eligible internal/external leakage overlap for {dataset_id}: "
                f"{len(overlap)} groups"
            )
        frame["split"] = str(config["external"]["split_name"])
        frame["split_version"] = str(config["version"])
        external_frames[dataset_id] = frame

    output_directory = root / config["outputs"]["directory"]
    output_directory.mkdir(parents=True, exist_ok=True)
    artifact_paths: dict[str, Path] = {}
    evaluation_frames: dict[str, pd.DataFrame] = {}
    for split_name in ratios:
        scoped = internal[internal["split"] == split_name].copy()
        path = output_directory / config["outputs"][split_name]
        _write_manifest_frame(scoped, path)
        artifact_paths[split_name] = path
        evaluation_frames[split_name] = scoped

    paired_config = config["internal"]["paired_benchmark"]
    source_split_name = str(paired_config["source_split"])
    paired_benchmark, reserve, paired_assignments = (
        select_fixed_case_benchmark(
            evaluation_frames[source_split_name],
            sample_count=int(paired_config["sample_count"]),
            seed=int(paired_config["seed"]),
            secondary_feature_weight=float(
                config["stratification"]["secondary_feature_weight"]
            ),
        )
    )
    paired_benchmark["split_version"] = str(config["version"])
    reserve["split_version"] = str(config["version"])
    paired_output_key = (
        f"internal_benchmark_{int(paired_config['sample_count'])}"
    )
    paired_path = output_directory / config["outputs"][paired_output_key]
    reserve_path = output_directory / config["outputs"][
        "internal_test_reserve"
    ]
    _write_manifest_frame(paired_benchmark, paired_path)
    _write_manifest_frame(reserve, reserve_path)
    artifact_paths[paired_output_key] = paired_path
    artifact_paths["internal_test_reserve"] = reserve_path
    evaluation_frames[paired_output_key] = paired_benchmark
    evaluation_frames["internal_test_reserve"] = reserve

    for dataset_id, frame in external_frames.items():
        output_key = f"external_{dataset_id}"
        path = output_directory / config["outputs"][output_key]
        _write_manifest_frame(frame, path)
        artifact_paths[output_key] = path
        evaluation_frames[output_key] = frame

    summary = _split_summary(evaluation_frames)
    subgroup_summary = _subgroup_summary(evaluation_frames)
    benchmark_balance = _selection_balance_report(
        full_frame=_representative_rows(
            evaluation_frames[source_split_name],
            seed=int(paired_config["seed"]),
        ),
        selected_frame=paired_benchmark,
    )
    summary_path = output_directory / config["outputs"]["split_summary"]
    subgroup_path = output_directory / config["outputs"]["subgroup_summary"]
    balance_path = output_directory / config["outputs"][
        "benchmark_1000_balance"
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    subgroup_path.parent.mkdir(parents=True, exist_ok=True)
    balance_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    subgroup_summary.to_csv(subgroup_path, index=False)
    benchmark_balance.to_csv(balance_path, index=False)
    artifact_paths["split_summary"] = summary_path
    artifact_paths["subgroup_summary"] = subgroup_path
    artifact_paths["benchmark_1000_balance"] = balance_path

    integrity = _integrity_report(
        evaluation_frames,
        internal_split_names=list(ratios),
        external_names=[f"external_{value}" for value in external_ids],
        config=config,
    )
    integrity_path = output_directory / config["outputs"]["integrity_report"]
    _write_yaml({"integrity": integrity}, integrity_path)
    artifact_paths["integrity_report"] = integrity_path

    release_path = output_directory / config["outputs"]["release_manifest"]
    release = _release_manifest(
        root=root,
        document=document,
        config_path=config_path,
        source_manifest_paths={
            dataset_id: manifest_paths[dataset_id]
            for dataset_id in sorted(required_ids)
        },
        artifact_paths=artifact_paths,
        integrity=integrity,
    )
    _write_yaml({"release": release}, release_path)
    artifact_paths["release_manifest"] = release_path

    return {
        "paths": artifact_paths,
        "assignments": assignments,
        "paired_assignments": paired_assignments,
        "integrity": integrity,
        "summary": summary,
        "subgroup_summary": subgroup_summary,
        "benchmark_balance": benchmark_balance,
    }


def assign_groups(
    frame: pd.DataFrame,
    *,
    ratios: dict[str, float],
    seed: int,
    secondary_feature_weight: float = 0.25,
) -> dict[str, str]:
    """Assign complete leakage groups using deterministic multilabel balancing."""

    _validate_ratios(ratios)
    if frame.empty:
        return {}
    group_column = "leakage_group_id"
    group_features: dict[str, dict[str, float]] = {}
    for group_id, scoped in frame.groupby(group_column, sort=True):
        group_features[str(group_id)] = _group_features(
            scoped,
            secondary_feature_weight=secondary_feature_weight,
        )

    feature_support: Counter[str] = Counter()
    for features in group_features.values():
        feature_support.update(features.keys())
    ordered_groups = sorted(
        group_features,
        key=lambda group_id: (
            min(
                feature_support[feature]
                for feature in group_features[group_id]
                if feature.startswith("primary:")
            ),
            -len(group_features[group_id]),
            _seeded_digest(seed, group_id),
        ),
    )

    split_names = list(ratios)
    split_capacities = _split_capacities(
        total=len(group_features),
        ratios=ratios,
    )
    feature_targets = {
        split_name: {
            feature: support * ratios[split_name]
            for feature, support in feature_support.items()
        }
        for split_name in split_names
    }
    feature_counts: dict[str, Counter[str]] = {
        split_name: Counter()
        for split_name in split_names
    }
    group_counts: Counter[str] = Counter()
    assignments: dict[str, str] = {}

    for group_id in ordered_groups:
        features = group_features[group_id]
        candidate_scores: list[tuple[float, str, str]] = []
        for split_name in split_names:
            if group_counts[split_name] >= split_capacities[split_name]:
                continue
            score = _incremental_assignment_cost(
                features=features,
                current=feature_counts[split_name],
                targets=feature_targets[split_name],
            )
            candidate_scores.append(
                (
                    score,
                    _seeded_digest(seed, f"{group_id}:{split_name}"),
                    split_name,
                )
            )
        selected = min(candidate_scores)[2]
        assignments[group_id] = selected
        group_counts[selected] += 1
        feature_counts[selected].update(features)
    if dict(group_counts) != split_capacities:
        raise ValueError(
            "Split-capacity mismatch after assignment: "
            f"{dict(group_counts)} != {split_capacities}"
        )
    return assignments


def _group_features(
    frame: pd.DataFrame,
    *,
    secondary_feature_weight: float,
) -> dict[str, float]:
    features: dict[str, float] = {}
    for disease_id in sorted(set(frame["disease_id"].astype(str))):
        features[f"primary:disease:{disease_id}"] = 1.0
    for dataset_id in sorted(set(frame["dataset_id"].astype(str))):
        features[f"primary:dataset:{dataset_id}"] = 1.0
    for disease_id, dataset_id in sorted(
        set(
            frame[["disease_id", "dataset_id"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )
    ):
        features[
            f"primary:disease_dataset:{disease_id}:{dataset_id}"
        ] = 1.0

    secondary_columns = [
        ("age", "age_group_standardized"),
        ("race_ethnicity", "race_ethnicity"),
    ]
    for prefix, column in secondary_columns:
        values = _informative_values(frame[column])
        for value in values:
            features[f"secondary:{prefix}:{value}"] = secondary_feature_weight
        if not values:
            features[
                f"secondary:{prefix}:missing_or_unknown"
            ] = secondary_feature_weight
    for prefix, system_column, value_column in [
        ("skin_tone", "skin_tone_system", "skin_tone"),
        (
            "sex_or_gender",
            "sex_or_gender_system",
            "sex_or_gender",
        ),
    ]:
        values = {
            f"{system}:{value}"
            for system, value in frame[
                [system_column, value_column]
            ].itertuples(index=False, name=None)
            if _is_informative(system) and _is_informative(value)
        }
        for value in sorted(values):
            features[f"secondary:{prefix}:{value}"] = secondary_feature_weight
        if not values:
            features[
                f"secondary:{prefix}:missing_or_unknown"
            ] = secondary_feature_weight
    return features


def select_fixed_case_benchmark(
    frame: pd.DataFrame,
    *,
    sample_count: int,
    seed: int,
    secondary_feature_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Select one representative image from each of a fixed number of groups."""

    group_count = int(frame["leakage_group_id"].nunique())
    if sample_count <= 0 or sample_count >= group_count:
        raise ValueError(
            "Fixed benchmark sample count must be positive and smaller than "
            f"the source group count ({group_count})"
        )
    benchmark_name = f"internal_benchmark_{sample_count}"
    reserve_name = "internal_test_reserve"
    assignments = assign_groups(
        frame,
        ratios={
            benchmark_name: sample_count / group_count,
            reserve_name: (group_count - sample_count) / group_count,
        },
        seed=seed,
        secondary_feature_weight=secondary_feature_weight,
    )
    representatives = _representative_rows(frame, seed=seed)
    representatives["_selection"] = representatives[
        "leakage_group_id"
    ].map(assignments)
    benchmark = representatives[
        representatives["_selection"] == benchmark_name
    ].drop(columns="_selection")
    reserve = representatives[
        representatives["_selection"] == reserve_name
    ].drop(columns="_selection")
    benchmark = benchmark.copy()
    reserve = reserve.copy()
    benchmark["split"] = benchmark_name
    reserve["split"] = reserve_name
    if len(benchmark) != sample_count:
        raise ValueError(
            f"Expected {sample_count} benchmark cases, found {len(benchmark)}"
        )
    if benchmark["leakage_group_id"].nunique() != sample_count:
        raise ValueError("Fixed benchmark must contain one image per group")
    return benchmark, reserve, assignments


def _representative_rows(
    frame: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    disease_support = (
        frame[["leakage_group_id", "disease_id"]]
        .drop_duplicates()
        .groupby("disease_id")["leakage_group_id"]
        .nunique()
        .to_dict()
    )
    evidence_priority = {
        "pathology": 0,
        "clinical_consensus": 1,
        "dermatologist_review": 2,
        "dermatologist_differential": 3,
        "atlas_label": 4,
        "self_reported": 5,
        "derived": 6,
        "unknown": 7,
    }
    selected_indexes: list[int] = []
    for _, scoped in frame.groupby("leakage_group_id", sort=True):
        selected_indexes.append(
            min(
                scoped.index,
                key=lambda index: (
                    disease_support.get(frame.at[index, "disease_id"], 999999),
                    evidence_priority.get(
                        str(frame.at[index, "diagnosis_basis"]),
                        999,
                    ),
                    _seeded_digest(
                        seed,
                        str(frame.at[index, "sample_id"]),
                    ),
                ),
            )
        )
    return frame.loc[selected_indexes].sort_values(
        "sample_id",
        ignore_index=True,
    )


def _incremental_assignment_cost(
    *,
    features: dict[str, float],
    current: Counter[str],
    targets: dict[str, float],
) -> float:
    score = 0.0
    for feature, weight in features.items():
        target = max(targets[feature], 1.0)
        before = (current[feature] - target) / target
        after = (current[feature] + 1 - target) / target
        score += weight * (after * after - before * before)
    return score


def _eligible_frame(
    manifest_paths: dict[str, Path],
    dataset_ids: Iterable[str],
) -> pd.DataFrame:
    frames = [
        pq.read_table(manifest_paths[dataset_id]).to_pandas()
        for dataset_id in dataset_ids
    ]
    frame = pd.concat(frames, ignore_index=True)
    return frame[
        frame["include"].astype(bool)
        & frame["diagnosis_gradable"].astype(bool)
        & frame["disease_id"].notna()
    ].reset_index(drop=True)


def _manifest_paths_from_catalog(root: Path) -> dict[str, Path]:
    catalog = load_yaml(root / "configs/datasets/catalog.yaml")
    paths: dict[str, Path] = {}
    for entry in catalog["datasets"]:
        config_path = root / entry["config"]
        if not config_path.exists():
            continue
        config = load_yaml(config_path)
        output = config.get("manifest", {}).get("output")
        if output:
            path = root / output
            if path.exists():
                paths[str(entry["id"])] = path
    return paths


def _write_manifest_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame[MANIFEST_ARROW_SCHEMA.names].copy()
    table = pa.Table.from_pandas(
        ordered,
        schema=MANIFEST_ARROW_SCHEMA,
        preserve_index=False,
    )
    pq.write_table(table, path, compression="zstd")


def _split_summary(
    evaluation_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for evaluation_set, frame in evaluation_frames.items():
        scopes = [
            *[
                (dataset_id, scoped)
                for dataset_id, scoped in frame.groupby(
                    "dataset_id",
                    sort=True,
                )
            ],
            ("all_datasets", frame),
        ]
        for dataset_id, scoped in scopes:
            for disease_id, disease_rows in scoped.groupby(
                "disease_id",
                sort=True,
            ):
                records.append(
                    {
                        "evaluation_set": evaluation_set,
                        "dataset_id": dataset_id,
                        "disease_id": disease_id,
                        "unique_group_count": int(
                            disease_rows["leakage_group_id"].nunique()
                        ),
                        "image_count": int(len(disease_rows)),
                    }
                )
    return pd.DataFrame.from_records(records).sort_values(
        ["evaluation_set", "dataset_id", "disease_id"],
        ignore_index=True,
    )


def _subgroup_summary(
    evaluation_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for evaluation_set, frame in evaluation_frames.items():
        dimensions = {
            "age_group_standardized": frame["age_group_standardized"],
            "race_ethnicity": frame["race_ethnicity"],
            "skin_tone_system_and_value": _combined_dimension(
                frame["skin_tone_system"],
                frame["skin_tone"],
            ),
            "sex_or_gender_system_and_value": _combined_dimension(
                frame["sex_or_gender_system"],
                frame["sex_or_gender"],
            ),
        }
        for dimension, values in dimensions.items():
            scoped = frame.assign(_subgroup=values)
            scoped = scoped[scoped["_subgroup"].map(_is_informative)]
            for subgroup, rows in scoped.groupby("_subgroup", sort=True):
                records.append(
                    {
                        "evaluation_set": evaluation_set,
                        "dimension": dimension,
                        "subgroup": subgroup,
                        "unique_group_count": int(
                            rows["leakage_group_id"].nunique()
                        ),
                        "image_count": int(len(rows)),
                    }
                )
    columns = [
        "evaluation_set",
        "dimension",
        "subgroup",
        "unique_group_count",
        "image_count",
    ]
    return pd.DataFrame.from_records(records, columns=columns).sort_values(
        ["evaluation_set", "dimension", "subgroup"],
        ignore_index=True,
    )


def _selection_balance_report(
    *,
    full_frame: pd.DataFrame,
    selected_frame: pd.DataFrame,
) -> pd.DataFrame:
    full_dimensions = _balance_dimensions(full_frame)
    selected_dimensions = _balance_dimensions(selected_frame)
    records: list[dict[str, Any]] = []
    for dimension in full_dimensions:
        full_values = full_dimensions[dimension]
        selected_values = selected_dimensions[dimension]
        categories = sorted(
            set(full_values.astype(str))
            | set(selected_values.astype(str))
        )
        full_counts = full_values.astype(str).value_counts()
        selected_counts = selected_values.astype(str).value_counts()
        for category in categories:
            full_count = int(full_counts.get(category, 0))
            selected_count = int(selected_counts.get(category, 0))
            full_fraction = full_count / len(full_frame)
            selected_fraction = selected_count / len(selected_frame)
            records.append(
                {
                    "dimension": dimension,
                    "category": category,
                    "internal_test_case_count": full_count,
                    "internal_test_fraction": full_fraction,
                    "benchmark_1000_case_count": selected_count,
                    "benchmark_1000_fraction": selected_fraction,
                    "absolute_fraction_difference": abs(
                        selected_fraction - full_fraction
                    ),
                }
            )
    return pd.DataFrame.from_records(records).sort_values(
        ["dimension", "category"],
        ignore_index=True,
    )


def _balance_dimensions(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "dataset_id": frame["dataset_id"].astype(str),
        "disease_id": frame["disease_id"].astype(str),
        "age_group_standardized": frame["age_group_standardized"].map(
            _reported_value
        ),
        "race_ethnicity": frame["race_ethnicity"].map(_reported_value),
        "skin_tone_system_and_value": _combined_dimension(
            frame["skin_tone_system"],
            frame["skin_tone"],
        ).map(_reported_value),
        "sex_or_gender_system_and_value": _combined_dimension(
            frame["sex_or_gender_system"],
            frame["sex_or_gender"],
        ).map(_reported_value),
    }


def _reported_value(value: Any) -> str:
    return str(value) if _is_informative(value) else "missing_or_unknown"


def _integrity_report(
    evaluation_frames: dict[str, pd.DataFrame],
    *,
    internal_split_names: list[str],
    external_names: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    internal_group_sets = {
        name: set(evaluation_frames[name]["leakage_group_id"].astype(str))
        for name in internal_split_names
    }
    internal_sample_sets = {
        name: set(evaluation_frames[name]["sample_id"].astype(str))
        for name in internal_split_names
    }
    group_overlaps: dict[str, int] = {}
    sample_overlaps: dict[str, int] = {}
    for index, left in enumerate(internal_split_names):
        for right in internal_split_names[index + 1 :]:
            key = f"{left}__{right}"
            group_overlaps[key] = len(
                internal_group_sets[left] & internal_group_sets[right]
            )
            sample_overlaps[key] = len(
                internal_sample_sets[left] & internal_sample_sets[right]
            )

    internal_groups = set().union(*internal_group_sets.values())
    external_group_overlaps = {
        name: len(
            internal_groups
            & set(evaluation_frames[name]["leakage_group_id"].astype(str))
        )
        for name in external_names
    }
    class_counts = {
        name: int(frame["disease_id"].nunique())
        for name, frame in evaluation_frames.items()
    }
    paired_config = config["internal"]["paired_benchmark"]
    paired_name = f"internal_benchmark_{int(paired_config['sample_count'])}"
    source_name = str(paired_config["source_split"])
    reserve_name = "internal_test_reserve"
    paired_groups = set(
        evaluation_frames[paired_name]["leakage_group_id"].astype(str)
    )
    source_groups = set(
        evaluation_frames[source_name]["leakage_group_id"].astype(str)
    )
    reserve_groups = set(
        evaluation_frames[reserve_name]["leakage_group_id"].astype(str)
    )
    paired_checks = {
        "expected_case_count": int(paired_config["sample_count"]),
        "image_count": int(len(evaluation_frames[paired_name])),
        "group_count": len(paired_groups),
        "one_image_per_group": (
            len(evaluation_frames[paired_name]) == len(paired_groups)
        ),
        "is_subset_of_internal_test": paired_groups.issubset(source_groups),
        "reserve_group_overlap_count": len(paired_groups & reserve_groups),
        "source_group_reconstruction_complete": (
            paired_groups | reserve_groups
        ) == source_groups,
        "disease_count": class_counts[paired_name],
        "contains_all_internal_test_diseases": (
            class_counts[paired_name] == class_counts[source_name]
        ),
    }
    passed = (
        not any(group_overlaps.values())
        and not any(sample_overlaps.values())
        and not any(external_group_overlaps.values())
        and paired_checks["image_count"]
        == paired_checks["expected_case_count"]
        and paired_checks["group_count"]
        == paired_checks["expected_case_count"]
        and paired_checks["one_image_per_group"]
        and paired_checks["is_subset_of_internal_test"]
        and paired_checks["reserve_group_overlap_count"] == 0
        and paired_checks["source_group_reconstruction_complete"]
        and paired_checks["contains_all_internal_test_diseases"]
        and all(
            class_counts[name] > 0
            for name in evaluation_frames
        )
    )
    return {
        "split_id": config["id"],
        "split_version": config["version"],
        "algorithm": config["algorithm"],
        "seed": int(config["seed"]),
        "passed": passed,
        "evaluation_set_image_counts": {
            name: int(len(frame))
            for name, frame in evaluation_frames.items()
        },
        "evaluation_set_group_counts": {
            name: int(frame["leakage_group_id"].nunique())
            for name, frame in evaluation_frames.items()
        },
        "evaluation_set_disease_counts": class_counts,
        "internal_group_overlap_counts": group_overlaps,
        "internal_sample_overlap_counts": sample_overlaps,
        "internal_external_group_overlap_counts": external_group_overlaps,
        "paired_benchmark_checks": paired_checks,
    }


def _release_manifest(
    *,
    root: Path,
    document: dict[str, Any],
    config_path: Path,
    source_manifest_paths: dict[str, Path],
    artifact_paths: dict[str, Path],
    integrity: dict[str, Any],
) -> dict[str, Any]:
    release = document["release"]
    referenced_paths = {
        "split_config": config_path,
        "taxonomy": root / release["taxonomy"],
        "benchmark": root / release["benchmark"],
        "prompt": root / release["prompt"],
        "output_schema": root / release["output_schema"],
        "disease_policy": root / release["disease_policy"],
        "duplicate_review": root / release["duplicate_review"],
    }
    return {
        "id": release["id"],
        "version": release["version"],
        "status": release["status"],
        "release_date": document["split"]["release_date"],
        "split_algorithm": SPLIT_ALGORITHM_VERSION,
        "seed": int(document["split"]["seed"]),
        "integrity_passed": bool(integrity["passed"]),
        "source_manifests": {
            dataset_id: {
                "path": str(path.relative_to(root)),
                "sha256": _file_sha256(path),
            }
            for dataset_id, path in source_manifest_paths.items()
        },
        "configuration": {
            name: {
                "path": str(path.relative_to(root)),
                "sha256": _file_sha256(path),
            }
            for name, path in referenced_paths.items()
        },
        "artifacts": {
            name: {
                "path": str(path.relative_to(root)),
                "sha256": _file_sha256(path),
            }
            for name, path in artifact_paths.items()
        },
    }


def validate_benchmark_release(root: Path) -> dict[str, Any]:
    """Validate the frozen release checksums and leakage assertions."""

    config = load_yaml(root / "configs/datasets/visual_top_k_split.yaml")
    output_directory = root / config["split"]["outputs"]["directory"]
    release_path = (
        output_directory
        / config["split"]["outputs"]["release_manifest"]
    )
    release = load_yaml(release_path)["release"]
    for section in ["source_manifests", "configuration", "artifacts"]:
        for name, entry in release[section].items():
            path = root / entry["path"]
            actual = _file_sha256(path)
            if actual != entry["sha256"]:
                raise ValueError(
                    f"Release checksum mismatch for {section}.{name}: {path}"
                )
    if not bool(release["integrity_passed"]):
        raise ValueError("Frozen benchmark release did not pass integrity checks")
    return release


def _validate_ratios(ratios: dict[str, float]) -> None:
    if not ratios or any(value <= 0 for value in ratios.values()):
        raise ValueError("Every split ratio must be positive")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")


def _split_capacities(
    *,
    total: int,
    ratios: dict[str, float],
) -> dict[str, int]:
    exact = {
        split_name: total * ratio
        for split_name, ratio in ratios.items()
    }
    capacities = {
        split_name: int(value)
        for split_name, value in exact.items()
    }
    remainder = total - sum(capacities.values())
    order = sorted(
        ratios,
        key=lambda split_name: (
            -(exact[split_name] - capacities[split_name]),
            list(ratios).index(split_name),
        ),
    )
    for split_name in order[:remainder]:
        capacities[split_name] += 1
    return capacities


def _combined_dimension(
    systems: pd.Series,
    values: pd.Series,
) -> pd.Series:
    return pd.Series(
        [
            f"{system}:{value}"
            if _is_informative(system) and _is_informative(value)
            else None
            for system, value in zip(systems, values, strict=True)
        ],
        index=systems.index,
    )


def _informative_values(values: pd.Series) -> list[str]:
    return sorted(
        {
            str(value)
            for value in values
            if _is_informative(value)
        }
    )


def _is_informative(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    normalized = str(value).strip().lower()
    return normalized not in {
        "",
        "unknown",
        "unk",
        "other_or_unspecified",
        "not_reported",
    }


def _seeded_digest(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_yaml(document: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            document,
            handle,
            sort_keys=False,
            allow_unicode=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or validate the visual top-k benchmark release."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the frozen release checksums without rebuilding it.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.validate_only:
        release = validate_benchmark_release(root)
        print(
            f"Validated benchmark release {release['id']} "
            f"version {release['version']}"
        )
        return
    result = build_benchmark_release(root=root)
    print(
        "Built benchmark release with "
        f"{len(result['assignments'])} internal leakage groups"
    )
    for name, path in result["paths"].items():
        print(f"{name}: {path.relative_to(root)}")


if __name__ == "__main__":
    main()
