"""Integrity validation for the frozen human-only E2 dataset release."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq

from src.train.config import E2DatasetConfig, TrainingConfig
from src.train.e2.domain import E2ReleaseAudit, E2Shard, E2TaskName, SkinConOntology

_read_parquet_table = cast(Callable[..., pa.Table], pq.read_table)


def inspect_e2_release(config: TrainingConfig) -> E2ReleaseAudit:
    """Validate the E2 manifest, ontology, every shard hash, and row count."""

    declared = require_e2_config(config)
    root = config.resolve_path(declared.source_directory)
    if not root.is_dir():
        raise FileNotFoundError(f"E2 dataset release is missing: {root}")
    manifest_path = _inside(root, declared.release_manifest_file)
    ontology_path = _inside(root, declared.ontology_file)
    manifest_sha = _sha256_file(manifest_path)
    ontology_sha = _sha256_file(ontology_path)
    if manifest_sha != declared.release_manifest_sha256:
        raise ValueError("E2 release manifest SHA-256 differs from the config")
    if ontology_sha != declared.ontology_sha256:
        raise ValueError("E2 SKINCON ontology SHA-256 differs from the config")

    manifest = _object(_read_json(manifest_path), "E2 release manifest")
    if manifest.get("release_id") != declared.release_id:
        raise ValueError("E2 release ID differs from the config")
    if manifest.get("schema_version") != declared.schema_version:
        raise ValueError("E2 schema version differs from the config")
    shards = _parse_shards(root, manifest.get("shards"))
    if declared.verify_all_shards:
        for shard in shards:
            _validate_shard(shard)
    _validate_cross_task_splits(shards)
    annotation_availability = _annotation_availability(shards)
    counts = _counts(shards)
    expected = declared.expected
    expected_counts = {
        (E2TaskName.DIAGNOSIS, "sft_train"): expected.diagnosis_train,
        (E2TaskName.DIAGNOSIS, "sft_dev"): expected.diagnosis_dev,
        (E2TaskName.MORPHOLOGY, "sft_train"): expected.morphology_train,
        (E2TaskName.MORPHOLOGY, "sft_dev"): expected.morphology_dev,
    }
    if expected.caption_train > 0 or expected.caption_dev > 0:
        expected_counts.update(
            {
                (E2TaskName.CAPTION, "sft_train"): expected.caption_train,
                (E2TaskName.CAPTION, "sft_dev"): expected.caption_dev,
            }
        )
    if counts != expected_counts:
        raise ValueError(f"E2 shard row counts differ: {counts}")
    schema_versions = _config_schema_versions(manifest, shards)
    ontology = _load_ontology(ontology_path)
    if len(ontology.concepts) != expected.morphology_concepts:
        raise ValueError("E2 morphology ontology cardinality differs")
    return E2ReleaseAudit(
        root=root,
        release_manifest_path=manifest_path,
        ontology_path=ontology_path,
        release_id=declared.release_id,
        schema_version=declared.schema_version,
        manifest_sha256=manifest_sha,
        ontology_sha256=ontology_sha,
        shards=shards,
        diagnosis_train=expected.diagnosis_train,
        diagnosis_dev=expected.diagnosis_dev,
        morphology_train=expected.morphology_train,
        morphology_dev=expected.morphology_dev,
        caption_train=expected.caption_train,
        caption_dev=expected.caption_dev,
        config_schema_versions=schema_versions,
        annotation_availability=annotation_availability,
        ontology=ontology,
    )


def require_e2_config(config: TrainingConfig) -> E2DatasetConfig:
    """Return the phase-specific section or reject an E1 configuration."""

    if config.e2 is None:
        raise ValueError("Training configuration does not declare an E2 release")
    return config.e2


def e2_shards(
    audit: E2ReleaseAudit,
    task: E2TaskName,
    subset: str,
) -> tuple[Path, ...]:
    """Return manifest-ordered shards for one task and split."""

    selected = tuple(
        shard.path
        for shard in audit.shards
        if shard.task is task and shard.subset == subset
    )
    if not selected:
        raise ValueError(f"E2 release has no {task.value}/{subset} shards")
    return selected


def _parse_shards(root: Path, value: object) -> tuple[E2Shard, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("E2 release manifest has no shards")
    shards: list[E2Shard] = []
    seen: set[Path] = set()
    for index, raw in enumerate(value):
        item = _object(raw, f"E2 shard {index}")
        relative = Path(_string(item, "path"))
        path = _inside(root, relative)
        if path in seen:
            raise ValueError(f"Duplicate E2 shard path: {relative}")
        seen.add(path)
        try:
            task = E2TaskName(_string(item, "config"))
        except ValueError as exc:
            raise ValueError(f"Unknown E2 shard config at index {index}") from exc
        subset = _string(item, "split")
        if subset not in {"sft_train", "sft_dev"}:
            raise ValueError(f"Unknown E2 shard split: {subset}")
        digest = _string(item, "sha256")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"Invalid E2 shard SHA-256: {relative}")
        shards.append(
            E2Shard(
                path=path,
                task=task,
                subset=subset,
                rows=_positive_integer(item, "rows"),
                bytes=_positive_integer(item, "bytes"),
                sha256=digest,
            )
        )
    return tuple(shards)


def _validate_shard(shard: E2Shard) -> None:
    if not shard.path.is_file():
        raise FileNotFoundError(f"E2 shard is missing: {shard.path}")
    if shard.path.stat().st_size != shard.bytes:
        raise ValueError(f"E2 shard byte size differs: {shard.path}")
    if _sha256_file(shard.path) != shard.sha256:
        raise ValueError(f"E2 shard SHA-256 differs: {shard.path}")


def _validate_cross_task_splits(shards: tuple[E2Shard, ...]) -> None:
    """Reject leakage groups assigned to train in one task and dev in another."""

    assigned: dict[str, str] = {}
    for shard in shards:
        table = _read_parquet_table(
            shard.path,
            columns=["leakage_group_id", "split"],
        )
        for group_id, split in zip(
            table["leakage_group_id"].to_pylist(),
            table["split"].to_pylist(),
            strict=True,
        ):
            key = str(group_id)
            value = str(split)
            previous = assigned.get(key)
            if previous is not None and previous != value:
                raise ValueError(
                    "E2 cross-task train/dev leakage for group "
                    f"{key}: {previous} versus {value}"
                )
            assigned[key] = value


def _counts(shards: tuple[E2Shard, ...]) -> dict[tuple[E2TaskName, str], int]:
    result: dict[tuple[E2TaskName, str], int] = {}
    for shard in shards:
        key = (shard.task, shard.subset)
        result[key] = result.get(key, 0) + shard.rows
    return result


def _annotation_availability(
    shards: tuple[E2Shard, ...],
) -> tuple[tuple[str, tuple[E2TaskName, ...]], ...]:
    """Build an immutable image-to-task availability index for cost audits."""

    tasks_by_image: dict[str, set[E2TaskName]] = {}
    for shard in shards:
        table = _read_parquet_table(shard.path, columns=["image_sha256"])
        for raw_digest in table.column("image_sha256").to_pylist():
            if (
                not isinstance(raw_digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", raw_digest) is None
            ):
                raise ValueError(f"Invalid image SHA-256 in E2 shard: {shard.path}")
            tasks_by_image.setdefault(raw_digest, set()).add(shard.task)
    order = {task: index for index, task in enumerate(E2TaskName)}
    return tuple(
        (
            digest,
            tuple(sorted(tasks, key=lambda item: order[item])),
        )
        for digest, tasks in sorted(tasks_by_image.items())
    )


def _load_ontology(path: Path) -> SkinConOntology:
    document = _object(_read_json(path), "SKINCON ontology")
    raw_concepts = document.get("concepts")
    if not isinstance(raw_concepts, list):
        raise ValueError("SKINCON ontology concepts must be an array")
    concepts: list[str] = []
    for value in raw_concepts:
        if not isinstance(value, str) or not value:
            raise ValueError("SKINCON ontology contains an invalid concept")
        concepts.append(value)
    return SkinConOntology(
        ontology_id=_string(document, "ontology_id"),
        concepts=tuple(concepts),
    )


def _config_schema_versions(
    manifest: Mapping[object, object],
    shards: tuple[E2Shard, ...],
) -> tuple[tuple[E2TaskName, str], ...]:
    raw = manifest.get("config_schema_versions")
    tasks = tuple(dict.fromkeys(shard.task for shard in shards))
    if raw is None:
        version = _string(manifest, "schema_version")
        return tuple((task, version) for task in tasks)
    document = _object(raw, "E2 config schema versions")
    parsed: list[tuple[E2TaskName, str]] = []
    for task in tasks:
        value = document.get(task.value)
        if not isinstance(value, str) or not value:
            raise ValueError(f"E2 release has no schema version for {task.value}")
        parsed.append((task, value))
    return tuple(parsed)


def _inside(root: Path, relative: Path) -> Path:
    if relative.is_absolute():
        raise ValueError("E2 release paths must be relative")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("E2 release path escapes the dataset root")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON {path}: {exc}") from exc


def _object(value: object, context: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _string(value: Mapping[object, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{key} must be a non-empty string")
    return item


def _positive_integer(value: Mapping[object, object], key: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return item
