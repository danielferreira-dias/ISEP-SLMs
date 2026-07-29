"""Build and validate the paired visual disease confusion-set benchmark."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.data_pipeline.common import load_yaml
from src.data_pipeline.splitting import validate_benchmark_release


CONFUSION_TASK_SCHEMA_VERSION = "1.0.0"
CONFUSION_TASK_ARROW_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("pair_id", pa.string(), nullable=False),
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("image_uri", pa.string(), nullable=False),
        pa.field("disease_id", pa.string(), nullable=False),
        pa.field("confusion_set_id", pa.string(), nullable=False),
        pa.field("difficulty", pa.string(), nullable=False),
        pa.field(
            "candidate_disease_ids",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field("candidate_count", pa.int16(), nullable=False),
        pa.field("leakage_group_id", pa.string(), nullable=False),
        pa.field("dataset_id", pa.string(), nullable=False),
    ]
)


def build_confusion_set_release(root: Path) -> dict[str, Any]:
    """Create the deterministic paired confusion-set task release."""

    benchmark_path = root / "configs/benchmarks/visual_confusion_sets.yaml"
    benchmark = load_yaml(benchmark_path)
    definition_path = (
        root / benchmark["taxonomy"]["confusion_sets"]["path"]
    )
    disease_taxonomy_path = (
        root / benchmark["taxonomy"]["disease"]["path"]
    )
    prompt_path = root / benchmark["prompt"]["path"]
    schema_path = root / benchmark["schema"]["path"]
    definition = load_yaml(definition_path)
    disease_taxonomy = load_yaml(disease_taxonomy_path)
    source_path = root / benchmark["dataset"]["source_manifest"]
    task_path = root / benchmark["dataset"]["task_manifest"]
    summary_path = root / benchmark["dataset"]["summary_report"]
    integrity_path = root / benchmark["dataset"]["integrity_report"]
    release_path = root / benchmark["dataset"]["release_manifest"]

    validate_benchmark_release(root)
    source = pq.read_table(source_path).to_pandas()
    active_ids = {
        str(item["id"])
        for item in disease_taxonomy["diseases"]
    }
    tasks, selection = build_confusion_tasks(
        source=source,
        definition=definition,
        active_disease_ids=active_ids,
    )
    integrity = validate_confusion_task_frame(
        tasks=tasks,
        source=source,
        definition=definition,
        active_disease_ids=active_ids,
    )
    if not integrity["passed"]:
        raise ValueError("Generated confusion-set task integrity checks failed")

    _write_task_manifest(tasks, task_path)
    summary = _build_summary(tasks)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    _write_yaml({"integrity": integrity}, integrity_path)

    configuration_paths = {
        "benchmark": benchmark_path,
        "confusion_sets": definition_path,
        "disease_taxonomy": disease_taxonomy_path,
        "prompt": prompt_path,
        "output_schema": schema_path,
    }
    artifact_paths = {
        "task_manifest": task_path,
        "summary_report": summary_path,
        "integrity_report": integrity_path,
    }
    definition_config = _definition_config(definition)
    source_release_path = (
        root / definition_config["selection"]["source_release"]
    )
    release_config = benchmark["release"]
    release = {
        "id": str(release_config["id"]),
        "version": str(release_config["version"]),
        "status": str(release_config["status"]),
        "release_date": str(release_config["release_date"]),
        "task_schema_version": CONFUSION_TASK_SCHEMA_VERSION,
        "selection_seed": int(
            definition_config["selection"]["seed"]
        ),
        "integrity_passed": True,
        "source": {
            "visual_top_k_manifest": _path_record(root, source_path),
            "visual_top_k_release": _path_record(
                root,
                source_release_path,
            ),
        },
        "configuration": {
            name: _path_record(root, path)
            for name, path in configuration_paths.items()
        },
        "artifacts": {
            name: _path_record(root, path)
            for name, path in artifact_paths.items()
        },
    }
    _write_yaml({"release": release}, release_path)
    return {
        "paths": artifact_paths | {"release_manifest": release_path},
        "tasks": tasks,
        "selection": selection,
        "summary": summary,
        "integrity": integrity,
        "release": release,
    }


def build_confusion_tasks(
    *,
    source: pd.DataFrame,
    definition: dict[str, Any],
    active_disease_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select balanced cases and create paired low/high candidate tasks."""

    taxonomy = _definition_config(definition)
    candidate_count = int(taxonomy["candidate_count"])
    if candidate_count != 3:
        raise ValueError("Confusion-set benchmark v1 requires three candidates")
    _validate_definition(
        taxonomy=taxonomy,
        active_disease_ids=active_disease_ids,
    )
    required_columns = {
        "sample_id",
        "image_uri",
        "disease_id",
        "leakage_group_id",
        "dataset_id",
    }
    missing_columns = sorted(required_columns - set(source.columns))
    if missing_columns:
        raise ValueError(
            "Source manifest is missing columns: "
            + ", ".join(missing_columns)
        )
    if source["sample_id"].duplicated().any():
        raise ValueError("Source benchmark must contain unique sample IDs")
    if source["leakage_group_id"].duplicated().any():
        raise ValueError(
            "Source benchmark must contain one image per leakage group"
        )

    seed = int(taxonomy["selection"]["seed"])
    high_sets = taxonomy["high_confusability_sets"]
    selected_frames: list[pd.DataFrame] = []
    cases_per_disease: dict[str, int] = {}
    disease_to_set: dict[str, str] = {}
    for set_config in high_sets:
        set_id = str(set_config["id"])
        disease_ids = [str(value) for value in set_config["disease_ids"]]
        scoped = source[source["disease_id"].isin(disease_ids)].copy()
        support = {
            disease_id: int((scoped["disease_id"] == disease_id).sum())
            for disease_id in disease_ids
        }
        if any(value == 0 for value in support.values()):
            raise ValueError(
                f"Confusion set {set_id} has an unsupported disease: {support}"
            )
        target = min(support.values())
        cases_per_disease[set_id] = target
        for disease_id in disease_ids:
            disease_to_set[disease_id] = set_id
            disease_rows = scoped[
                scoped["disease_id"] == disease_id
            ].copy()
            disease_rows["_selection_key"] = disease_rows[
                "sample_id"
            ].map(
                lambda sample_id: _seeded_digest(
                    seed,
                    f"sample:{set_id}:{sample_id}",
                )
            )
            selected = disease_rows.sort_values(
                ["_selection_key", "sample_id"],
                kind="mergesort",
            ).head(target)
            selected_frames.append(selected.drop(columns=["_selection_key"]))

    selected_cases = pd.concat(selected_frames, ignore_index=True)
    selected_cases["_set_id"] = selected_cases["disease_id"].map(
        disease_to_set
    )
    selected_cases["_order_key"] = selected_cases["sample_id"].map(
        lambda sample_id: _seeded_digest(
            seed,
            f"task-order:{sample_id}",
        )
    )
    selected_cases = selected_cases.sort_values(
        ["_order_key", "sample_id"],
        kind="mergesort",
    ).drop(columns=["_order_key"])

    partitions = {
        str(partition["id"]): [
            str(value)
            for value in partition["disease_ids"]
        ]
        for partition in taxonomy[
            "low_confusability_partitions"
        ]["partitions"]
    }
    disease_to_partition = {
        disease_id: partition_id
        for partition_id, disease_ids in partitions.items()
        for disease_id in disease_ids
    }
    distractor_usage: defaultdict[
        tuple[str, str],
        Counter[str],
    ] = defaultdict(Counter)
    task_rows: list[dict[str, Any]] = []
    set_lookup = {
        str(item["id"]): [
            str(value)
            for value in item["disease_ids"]
        ]
        for item in high_sets
    }
    for record in selected_cases.to_dict(orient="records"):
        sample_id = str(record["sample_id"])
        disease_id = str(record["disease_id"])
        set_id = str(record["_set_id"])
        pair_id = f"{set_id}::{sample_id}"
        truth_partition = disease_to_partition[disease_id]

        low_candidates = [disease_id]
        for partition_id, pool in partitions.items():
            if partition_id == truth_partition:
                continue
            usage = distractor_usage[(truth_partition, partition_id)]
            minimum_usage = min(
                (usage[candidate] for candidate in pool),
                default=0,
            )
            eligible = [
                candidate
                for candidate in pool
                if usage[candidate] == minimum_usage
            ]
            chosen = min(
                eligible,
                key=lambda candidate: _seeded_digest(
                    seed,
                    (
                        f"distractor:{pair_id}:{partition_id}:"
                        f"{candidate}"
                    ),
                ),
            )
            usage[chosen] += 1
            low_candidates.append(chosen)

        conditions = {
            "low_confusability": low_candidates,
            "high_confusability": set_lookup[set_id],
        }
        for difficulty, candidates in conditions.items():
            task_id = f"{pair_id}::{difficulty}"
            ordered_candidates = sorted(
                candidates,
                key=lambda candidate: _seeded_digest(
                    seed,
                    f"candidate-order:{task_id}:{candidate}",
                ),
            )
            task_rows.append(
                {
                    "schema_version": CONFUSION_TASK_SCHEMA_VERSION,
                    "task_id": task_id,
                    "pair_id": pair_id,
                    "sample_id": sample_id,
                    "image_uri": str(record["image_uri"]),
                    "disease_id": disease_id,
                    "confusion_set_id": set_id,
                    "difficulty": difficulty,
                    "candidate_disease_ids": ordered_candidates,
                    "candidate_count": candidate_count,
                    "leakage_group_id": str(
                        record["leakage_group_id"]
                    ),
                    "dataset_id": str(record["dataset_id"]),
                }
            )

    tasks = pd.DataFrame(task_rows)
    tasks["_difficulty_order"] = tasks["difficulty"].map(
        {"low_confusability": 0, "high_confusability": 1}
    )
    tasks = tasks.sort_values(
        ["pair_id", "_difficulty_order"],
        kind="mergesort",
    ).drop(columns=["_difficulty_order"]).reset_index(drop=True)
    return tasks, {
        "cases_per_disease_by_set": cases_per_disease,
        "unique_image_count": int(selected_cases["sample_id"].nunique()),
        "task_count": int(len(tasks)),
    }


def validate_confusion_task_frame(
    *,
    tasks: pd.DataFrame,
    source: pd.DataFrame,
    definition: dict[str, Any],
    active_disease_ids: set[str],
) -> dict[str, Any]:
    """Validate paired task, candidate, balance, and source invariants."""

    taxonomy = _definition_config(definition)
    expected = taxonomy["expected_release"]
    candidate_count = int(taxonomy["candidate_count"])
    set_lookup = {
        str(item["id"]): {
            str(value)
            for value in item["disease_ids"]
        }
        for item in taxonomy["high_confusability_sets"]
    }
    partitions = {
        str(item["id"]): {
            str(value)
            for value in item["disease_ids"]
        }
        for item in taxonomy[
            "low_confusability_partitions"
        ]["partitions"]
    }
    disease_to_partition = {
        disease_id: partition_id
        for partition_id, disease_ids in partitions.items()
        for disease_id in disease_ids
    }
    candidate_checks = {
        "candidate_count_is_three": True,
        "candidate_ids_are_unique": True,
        "candidate_ids_are_active": True,
        "reference_is_candidate": True,
        "high_condition_matches_set": True,
        "low_condition_crosses_partitions": True,
        "low_distractor_usage_is_balanced": True,
    }
    low_distractor_usage: defaultdict[
        tuple[str, str],
        Counter[str],
    ] = defaultdict(Counter)
    for record in tasks.to_dict(orient="records"):
        candidates = [
            str(value)
            for value in record["candidate_disease_ids"]
        ]
        candidate_checks["candidate_count_is_three"] &= (
            len(candidates) == candidate_count
        )
        candidate_checks["candidate_ids_are_unique"] &= (
            len(candidates) == len(set(candidates))
        )
        candidate_checks["candidate_ids_are_active"] &= set(
            candidates
        ).issubset(active_disease_ids)
        candidate_checks["reference_is_candidate"] &= (
            str(record["disease_id"]) in candidates
        )
        if record["difficulty"] == "high_confusability":
            candidate_checks["high_condition_matches_set"] &= (
                set(candidates)
                == set_lookup[str(record["confusion_set_id"])]
            )
        elif record["difficulty"] == "low_confusability":
            represented_partitions = {
                disease_to_partition[candidate]
                for candidate in candidates
            }
            candidate_checks["low_condition_crosses_partitions"] &= (
                represented_partitions == set(partitions)
            )
            truth_partition = disease_to_partition[
                str(record["disease_id"])
            ]
            for candidate in candidates:
                candidate_partition = disease_to_partition[candidate]
                if candidate_partition != truth_partition:
                    low_distractor_usage[
                        (truth_partition, candidate_partition)
                    ][candidate] += 1
        else:
            candidate_checks["low_condition_crosses_partitions"] = False

    low_distractor_balance: dict[str, bool] = {}
    for (
        truth_partition,
        candidate_partition,
    ), usage in low_distractor_usage.items():
        counts = [
            usage[disease_id]
            for disease_id in partitions[candidate_partition]
        ]
        key = f"{truth_partition}->{candidate_partition}"
        low_distractor_balance[key] = max(counts) - min(counts) <= 1
    candidate_checks["low_distractor_usage_is_balanced"] = (
        bool(low_distractor_balance)
        and all(low_distractor_balance.values())
    )

    pair_sizes = tasks.groupby("pair_id").size()
    pair_condition_counts = tasks.groupby("pair_id")[
        "difficulty"
    ].nunique()
    pair_sample_counts = tasks.groupby("pair_id")["sample_id"].nunique()
    pair_label_counts = tasks.groupby("pair_id")["disease_id"].nunique()
    pair_image_counts = tasks.groupby("pair_id")["image_uri"].nunique()
    balance = (
        tasks[tasks["difficulty"] == "high_confusability"]
        .groupby(["confusion_set_id", "disease_id"])
        .size()
    )
    set_balance = {
        set_id: int(scoped.nunique()) == 1
        for set_id, scoped in balance.groupby(level=0)
    }
    source_sample_ids = set(source["sample_id"].astype(str))
    task_sample_ids = set(tasks["sample_id"].astype(str))
    group_counts = (
        tasks[["sample_id", "leakage_group_id"]]
        .drop_duplicates()["leakage_group_id"]
        .nunique()
    )
    paired_checks = {
        "two_tasks_per_pair": bool((pair_sizes == 2).all()),
        "two_conditions_per_pair": bool(
            (pair_condition_counts == 2).all()
        ),
        "same_sample_per_pair": bool((pair_sample_counts == 1).all()),
        "same_reference_per_pair": bool((pair_label_counts == 1).all()),
        "same_image_per_pair": bool((pair_image_counts == 1).all()),
    }
    count_checks = {
        "expected_confusion_set_count": (
            tasks["confusion_set_id"].nunique()
            == int(expected["active_confusion_set_count"])
        ),
        "expected_covered_disease_count": (
            tasks["disease_id"].nunique()
            == int(expected["covered_disease_count"])
        ),
        "expected_unique_image_count": (
            tasks["sample_id"].nunique()
            == int(expected["unique_image_count"])
        ),
        "expected_pair_count": (
            tasks["pair_id"].nunique()
            == int(expected["unique_image_count"])
        ),
        "expected_task_count": (
            len(tasks) == int(expected["paired_task_count"])
        ),
        "one_group_per_unique_image": (
            group_counts == tasks["sample_id"].nunique()
        ),
        "source_subset": task_sample_ids.issubset(source_sample_ids),
        "unique_task_ids": not tasks["task_id"].duplicated().any(),
    }
    passed = (
        all(candidate_checks.values())
        and all(paired_checks.values())
        and all(count_checks.values())
        and all(set_balance.values())
    )
    return {
        "passed": passed,
        "task_schema_version": CONFUSION_TASK_SCHEMA_VERSION,
        "task_count": int(len(tasks)),
        "pair_count": int(tasks["pair_id"].nunique()),
        "unique_image_count": int(tasks["sample_id"].nunique()),
        "unique_group_count": int(group_counts),
        "covered_disease_count": int(tasks["disease_id"].nunique()),
        "confusion_set_count": int(tasks["confusion_set_id"].nunique()),
        "candidate_checks": candidate_checks,
        "low_distractor_balance": low_distractor_balance,
        "paired_checks": paired_checks,
        "count_checks": count_checks,
        "balanced_within_set": set_balance,
        "high_condition_cases_per_disease": {
            f"{set_id}:{disease_id}": int(count)
            for (set_id, disease_id), count in balance.items()
        },
    }


def validate_confusion_set_release(root: Path) -> dict[str, Any]:
    """Validate release checksums and recompute task-frame invariants."""

    benchmark = load_yaml(
        root / "configs/benchmarks/visual_confusion_sets.yaml"
    )
    release_path = root / benchmark["dataset"]["release_manifest"]
    release = load_yaml(release_path)["release"]
    for section in ["source", "configuration", "artifacts"]:
        for name, entry in release[section].items():
            path = root / entry["path"]
            if _file_sha256(path) != entry["sha256"]:
                raise ValueError(
                    "Confusion-set release checksum mismatch for "
                    f"{section}.{name}: {path}"
                )
    if not bool(release["integrity_passed"]):
        raise ValueError(
            "Confusion-set release records failed integrity checks"
        )

    definition = load_yaml(
        root / benchmark["taxonomy"]["confusion_sets"]["path"]
    )
    disease_taxonomy = load_yaml(
        root / benchmark["taxonomy"]["disease"]["path"]
    )
    active_ids = {
        str(item["id"])
        for item in disease_taxonomy["diseases"]
    }
    source = pq.read_table(
        root / benchmark["dataset"]["source_manifest"]
    ).to_pandas()
    task_table = pq.read_table(
        root / benchmark["dataset"]["task_manifest"]
    )
    if task_table.schema != CONFUSION_TASK_ARROW_SCHEMA:
        raise ValueError("Confusion task manifest schema mismatch")
    tasks = task_table.to_pandas()
    integrity = validate_confusion_task_frame(
        tasks=tasks,
        source=source,
        definition=definition,
        active_disease_ids=active_ids,
    )
    if not integrity["passed"]:
        raise ValueError(
            "Confusion-set release failed recomputed integrity checks"
        )
    return release


def _validate_definition(
    *,
    taxonomy: dict[str, Any],
    active_disease_ids: set[str],
) -> None:
    candidate_count = int(taxonomy["candidate_count"])
    high_sets = taxonomy["high_confusability_sets"]
    set_ids = [str(item["id"]) for item in high_sets]
    if len(set_ids) != len(set(set_ids)):
        raise ValueError("Confusion set IDs must be unique")
    covered: list[str] = []
    for item in high_sets:
        disease_ids = [
            str(value)
            for value in item["disease_ids"]
        ]
        if len(disease_ids) != candidate_count:
            raise ValueError(
                f"Confusion set {item['id']} must have {candidate_count} diseases"
            )
        if len(disease_ids) != len(set(disease_ids)):
            raise ValueError(
                f"Confusion set {item['id']} repeats a disease"
            )
        if not set(disease_ids).issubset(active_disease_ids):
            raise ValueError(
                f"Confusion set {item['id']} contains an inactive disease"
            )
        covered.extend(disease_ids)
    if taxonomy["selection"][
        "require_disjoint_high_confusability_sets"
    ] and len(covered) != len(set(covered)):
        raise ValueError(
            "High-confusability sets must not overlap in benchmark v1"
        )

    partitions = taxonomy[
        "low_confusability_partitions"
    ]["partitions"]
    partition_ids = [str(item["id"]) for item in partitions]
    if len(partition_ids) != candidate_count:
        raise ValueError(
            "Low-confusability construction requires three partitions"
        )
    partition_diseases = [
        str(value)
        for item in partitions
        for value in item["disease_ids"]
    ]
    if len(partition_diseases) != len(set(partition_diseases)):
        raise ValueError("Appearance partitions must not overlap")
    if set(partition_diseases) != set(covered):
        raise ValueError(
            "Appearance partitions must cover every confusion-set disease"
        )
    excluded = {
        str(value)
        for value in taxonomy["excluded_from_v1"]["disease_ids"]
    }
    if excluded != active_disease_ids - set(covered):
        raise ValueError(
            "Excluded v1 diseases must equal the uncovered active taxonomy"
        )


def _definition_config(definition: dict[str, Any]) -> dict[str, Any]:
    return definition["taxonomy"] | {
        key: value
        for key, value in definition.items()
        if key != "taxonomy"
    }


def _build_summary(tasks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (
        difficulty,
        set_id,
        disease_id,
    ), scoped in tasks.groupby(
        ["difficulty", "confusion_set_id", "disease_id"],
        sort=True,
    ):
        rows.append(
            {
                "difficulty": difficulty,
                "confusion_set_id": set_id,
                "disease_id": disease_id,
                "task_count": int(len(scoped)),
                "unique_sample_count": int(
                    scoped["sample_id"].nunique()
                ),
                "unique_group_count": int(
                    scoped["leakage_group_id"].nunique()
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_task_manifest(tasks: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        tasks.to_dict(orient="records"),
        schema=CONFUSION_TASK_ARROW_SCHEMA,
    )
    pq.write_table(table, path, compression="zstd")


def _path_record(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": _file_sha256(path),
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
        description="Build or validate the visual confusion-set benchmark."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the existing release without rebuilding it.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.validate_only:
        release = validate_confusion_set_release(root)
        print(
            f"Validated confusion-set release {release['id']} "
            f"version {release['version']}"
        )
        return
    result = build_confusion_set_release(root)
    print(
        "Built confusion-set release with "
        f"{result['integrity']['pair_count']} pairs and "
        f"{result['integrity']['task_count']} tasks"
    )
    for name, path in result["paths"].items():
        print(f"{name}: {path.relative_to(root)}")


if __name__ == "__main__":
    main()
