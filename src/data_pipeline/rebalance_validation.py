"""Rebalance ISEPDermaBench Validation and release rows to Train.

This migration reduces the visual Top-K Validation split from 1,683 to 1,000
tasks without splitting leakage groups. Every group used by another Validation
task is mandatory, so Evidence, Confusion Sets, and Open-ended Diagnosis remain
unchanged. The remaining capacity is filled by a deterministic group-level
selection. Exactly 683 released image rows are recorded for promotion into
ISEPDermData.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
from typing import Any

from datasets import Dataset
import pandas as pd

from src.data_pipeline.huggingface_benchmark_export import (
    REFERENCE_FEATURES,
    TASK_FEATURES,
)


DEFAULT_SOURCE = Path("data/benchmarks/ISEPDermaBench")
DEFAULT_OUTPUT = Path("data/benchmarks/ISEPDermaBench-v1.2.0")
TARGET_TASKS = 1_000
SELECTION_SEED = 42
SHARD_SIZE = 512


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_split(
    release: Path,
    *,
    kind: str,
    task: str,
    split: str,
) -> pd.DataFrame:
    paths = sorted((release / kind / task).glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"Missing {kind}/{task}/{split} shards")
    return pd.concat(
        [pd.read_parquet(path) for path in paths],
        ignore_index=True,
    )


def _stable_group_order(groups: list[str]) -> list[str]:
    return sorted(
        groups,
        key=lambda group: (
            sha256(f"{SELECTION_SEED}:{group}".encode()).hexdigest(),
            group,
        ),
    )


def _exact_group_subset(
    group_sizes: dict[str, int],
    target_rows: int,
) -> set[str]:
    """Return a deterministic whole-group subset with an exact row total."""

    reachable: dict[int, tuple[int, str] | None] = {0: None}
    for group in _stable_group_order(list(group_sizes)):
        size = group_sizes[group]
        for total in sorted(list(reachable), reverse=True):
            candidate = total + size
            if candidate > target_rows or candidate in reachable:
                continue
            reachable[candidate] = (total, group)
        if target_rows in reachable:
            break
    if target_rows not in reachable:
        raise ValueError(
            f"Cannot select whole groups totalling {target_rows} rows"
        )
    selected: set[str] = set()
    total = target_rows
    while total:
        previous = reachable[total]
        if previous is None:
            raise RuntimeError("Invalid subset reconstruction")
        total, group = previous
        selected.add(group)
    return selected


def plan_rebalance(source: Path) -> dict[str, Any]:
    visual_tasks = _load_split(
        source,
        kind="tasks",
        task="visual_top_k",
        split="validation",
    )
    visual_refs = _load_split(
        source,
        kind="references",
        task="visual_top_k",
        split="validation",
    )
    if len(visual_tasks) != 1_683:
        raise ValueError(
            f"Expected 1,683 visual Validation tasks, found {len(visual_tasks)}"
        )
    if set(visual_tasks["task_id"]) != set(visual_refs["task_id"]):
        raise ValueError("Visual Validation task/reference IDs differ")

    protected_groups: set[str] = set()
    for task in (
        "visual_confusion_sets",
        "evidence_grounded_diagnosis",
        "open_ended_diagnosis",
    ):
        frame = _load_split(
            source,
            kind="tasks",
            task=task,
            split="validation",
        )
        protected_groups.update(frame["leakage_group_id"].astype(str))

    group_values = visual_tasks["leakage_group_id"].astype(str)
    mandatory = visual_tasks[group_values.isin(protected_groups)]
    optional = visual_tasks[~group_values.isin(protected_groups)]
    capacity = TARGET_TASKS - len(mandatory)
    if capacity < 0:
        raise ValueError("Other Validation tasks require more than 1,000 rows")

    optional_sizes = (
        optional.assign(leakage_group_id=optional["leakage_group_id"].astype(str))
        .groupby("leakage_group_id")
        .size()
        .to_dict()
    )
    selected_optional = _exact_group_subset(optional_sizes, capacity)
    kept_groups = protected_groups | selected_optional
    keep_mask = group_values.isin(kept_groups)
    kept_tasks = visual_tasks[keep_mask].copy()
    released_tasks = visual_tasks[~keep_mask].copy()
    kept_ids = set(kept_tasks["task_id"].astype(str))
    kept_refs = visual_refs[
        visual_refs["task_id"].astype(str).isin(kept_ids)
    ].copy()
    released_refs = visual_refs[
        ~visual_refs["task_id"].astype(str).isin(kept_ids)
    ].copy()

    if len(kept_tasks) != TARGET_TASKS or len(released_tasks) != 683:
        raise ValueError("Rebalance did not produce 1,000/683 task counts")
    released_groups = set(released_tasks["leakage_group_id"].astype(str))
    if released_groups & protected_groups:
        raise ValueError("Released groups overlap another Validation task")
    if released_groups & set(kept_tasks["leakage_group_id"].astype(str)):
        raise ValueError("A leakage group was split across Validation and Train")

    audit = released_tasks[
        ["task_id", "sample_id", "source", "leakage_group_id", "license_id"]
    ].merge(
        released_refs[["task_id", "reference_disease_id"]],
        on="task_id",
        validate="one_to_one",
    )
    audit = audit.sort_values("sample_id", kind="stable").reset_index(drop=True)
    return {
        "kept_tasks": kept_tasks.reset_index(drop=True),
        "kept_refs": kept_refs.reset_index(drop=True),
        "released_tasks": released_tasks.reset_index(drop=True),
        "released_refs": released_refs.reset_index(drop=True),
        "audit": audit,
        "summary": {
            "original_tasks": len(visual_tasks),
            "original_groups": visual_tasks["leakage_group_id"].nunique(),
            "kept_tasks": len(kept_tasks),
            "kept_groups": kept_tasks["leakage_group_id"].nunique(),
            "released_images": len(released_tasks),
            "released_groups": released_tasks["leakage_group_id"].nunique(),
            "mandatory_tasks": len(mandatory),
            "mandatory_groups": mandatory["leakage_group_id"].nunique(),
            "selection_seed": SELECTION_SEED,
            "policy": "group_safe_validation_to_train_v1",
        },
    }


def _write_shards(
    frame: pd.DataFrame,
    *,
    directory: Path,
    split: str,
    features: Any,
) -> list[dict[str, Any]]:
    for old in directory.glob(f"{split}-*.parquet"):
        old.unlink()
    shard_count = math.ceil(len(frame) / SHARD_SIZE)
    shards: list[dict[str, Any]] = []
    for index in range(shard_count):
        rows = frame.iloc[index * SHARD_SIZE : (index + 1) * SHARD_SIZE]
        path = directory / (
            f"{split}-{index:05d}-of-{shard_count:05d}.parquet"
        )
        records = []
        for record in rows.to_dict(orient="records"):
            records.append(
                {
                    key: None
                    if pd.api.types.is_scalar(value) and pd.isna(value)
                    else value
                    for key, value in record.items()
                }
            )
        Dataset.from_list(records, features=features).to_parquet(path)
        shards.append(
            {
                "path": path.as_posix(),
                "rows": len(rows),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return shards


def build_rebalanced_release(source: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    plan = plan_rebalance(source)
    shutil.copytree(source, output)

    task_shards = _write_shards(
        plan["kept_tasks"],
        directory=output / "tasks/visual_top_k",
        split="validation",
        features=TASK_FEATURES,
    )
    reference_shards = _write_shards(
        plan["kept_refs"],
        directory=output / "references/visual_top_k",
        split="validation",
        features=REFERENCE_FEATURES,
    )
    for item in task_shards + reference_shards:
        item["path"] = Path(item["path"]).relative_to(output).as_posix()

    metadata = output / "metadata"
    audit_path = metadata / "visual_top_k_validation_to_train.csv"
    plan["audit"].to_csv(audit_path, index=False)

    release_path = output / "release.json"
    document = json.loads(release_path.read_text(encoding="utf-8"))
    release = document["release"]
    release["version"] = "1.2.0"
    release["created_at"] = "2026-08-01"
    for split in release["splits"]:
        if split["benchmark"] == "visual_top_k" and split["split"] == "validation":
            split["task_count"] = TARGET_TASKS
            split["sample_count"] = plan["kept_tasks"]["sample_id"].nunique()
            split["group_count"] = plan["kept_tasks"]["leakage_group_id"].nunique()
            split["task_shards"] = task_shards
            split["reference_shards"] = reference_shards
            break
    else:
        raise ValueError("Visual Validation release entry is missing")
    release["validation_rebalance"] = {
        **plan["summary"],
        "audit_file": "metadata/visual_top_k_validation_to_train.csv",
        "audit_sha256": _sha256(audit_path),
        "other_validation_tasks_preserved": [
            "visual_confusion_sets",
            "evidence_grounded_diagnosis",
            "open_ended_diagnosis",
        ],
    }
    release["embedded_image_bytes"] = sum(
        len(image["bytes"])
        for path in (output / "tasks").glob("*/*.parquet")
        for image in pd.read_parquet(path, columns=["image"])["image"]
    )
    release_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    readme_path = output / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    readme = readme.replace(
        "| visual_top_k | validation | 1,683 | 1,683 | 1,063 |",
        (
            "| visual_top_k | validation | 1,000 | 1,000 | "
            f"{plan['summary']['kept_groups']:,} |"
        ),
    )
    marker = "## Split policy\n"
    note = (
        "## Validation rebalance\n\n"
        "Release 1.2.0 reduces visual Top-K Validation from 1,683 to 1,000 "
        "image tasks using whole leakage groups. All groups required by the "
        "other Validation tasks remain protected. The 683 released images "
        "are promoted to ISEPDermData Train under the auditable "
        "`group_safe_validation_to_train_v1` policy.\n\n"
    )
    if marker not in readme:
        raise ValueError("Dataset-card split-policy marker is missing")
    readme = readme.replace(marker, note + marker)
    readme_path.write_text(readme, encoding="utf-8")
    return plan["summary"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    source = root / args.source
    plan = plan_rebalance(source)
    if not args.apply:
        print(json.dumps(plan["summary"], indent=2))
        return
    summary = build_rebalanced_release(source, root / args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
