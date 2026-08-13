"""Read and audit the pinned local ISEPDermData source pool."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pyarrow.parquet as pq

from src.train.config import TrainingConfig
from src.train.domain import Taxonomy

SOURCE_METADATA_COLUMNS = (
    "source",
    "label",
    "disease_id",
    "sample_id",
    "source_image_id",
    "source_label",
    "leakage_group_id",
    "diagnosis_basis",
    "image_sha256",
    "license_id",
)


def source_shards(config: TrainingConfig) -> tuple[Path, ...]:
    """Return the ordered Parquet shards of the configured source pool."""

    source_root = config.resolve_path(config.dataset.source_directory)
    shards = tuple(sorted((source_root / "data").glob("train-*.parquet")))
    if not shards:
        raise ValueError(f"No source shards found under {source_root / 'data'}")
    return shards


def load_source_frame(
    config: TrainingConfig,
    *,
    include_images: bool = False,
) -> pd.DataFrame:
    """Load source rows into memory, optionally including embedded images.

    Metadata-only loading is the default because embedded images occupy several
    gigabytes.  Training code should prefer the batch iterator for image data.
    """

    columns = [*SOURCE_METADATA_COLUMNS]
    if include_images:
        columns.insert(0, "image")
    return pq.read_table(  # type: ignore[no-untyped-call]
        list(source_shards(config)), columns=columns
    ).to_pandas()


def source_release_path(config: TrainingConfig) -> Path:
    """Return the local source release manifest path."""

    source_root = config.resolve_path(config.dataset.source_directory)
    child = config.dataset.source_release_file
    return child if child.is_absolute() else source_root / child


def source_release_sha256(config: TrainingConfig) -> str:
    """Hash the pinned source release manifest."""

    path = source_release_path(config)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_source_shards(config: TrainingConfig) -> None:
    """Verify every declared Parquet shard by path, size, rows, and SHA-256."""

    source_root = config.resolve_path(config.dataset.source_directory).resolve()
    release = _load_release_document(source_release_path(config))
    declared_value = release.get("shards")
    if not isinstance(declared_value, list) or not declared_value:
        raise ValueError("Source release must declare a non-empty shards list")
    declared_paths: set[Path] = set()
    for index, raw_item in enumerate(declared_value):
        if not isinstance(raw_item, Mapping):
            raise ValueError(f"Source shard {index} must be an object")
        relative = _required_string(raw_item, "path")
        expected_sha = _required_string(raw_item, "sha256")
        expected_bytes = _required_integer(raw_item, "bytes")
        expected_rows = _required_integer(raw_item, "rows")
        path = (source_root / relative).resolve()
        if not path.is_relative_to(source_root):
            raise ValueError(f"Source shard escapes dataset root: {relative}")
        if path in declared_paths:
            raise ValueError(f"Duplicate source shard declaration: {relative}")
        declared_paths.add(path)
        if not path.is_file():
            raise ValueError(f"Declared source shard is missing: {path}")
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"Source shard byte count mismatch: {relative}")
        if pq.ParquetFile(path).metadata.num_rows != expected_rows:  # type: ignore[no-untyped-call]
            raise ValueError(f"Source shard row count mismatch: {relative}")
        if _sha256_file(path) != expected_sha:
            raise ValueError(f"Source shard SHA-256 mismatch: {relative}")
    actual_paths = {path.resolve() for path in source_shards(config)}
    if declared_paths != actual_paths:
        raise ValueError("Declared and local source shard paths differ")


def validate_source_pool(
    config: TrainingConfig,
    frame: pd.DataFrame,
    taxonomy: Taxonomy,
) -> None:
    """Fail if source metadata differs from the pinned dataset contract."""

    missing = sorted(set(SOURCE_METADATA_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Source pool is missing columns: {', '.join(missing)}")
    if frame[list(SOURCE_METADATA_COLUMNS)].isna().any().any():
        raise ValueError("Source pool contains null required metadata")
    if not frame["sample_id"].astype(str).is_unique:
        raise ValueError("Source pool contains duplicate sample_id values")

    expected = config.dataset.expected
    observed = {
        "image_count": len(frame),
        "group_count": int(frame["leakage_group_id"].nunique()),
        "class_count": int(frame["label"].nunique()),
        "source_count": int(frame["source"].nunique()),
    }
    for name, value in observed.items():
        expected_value = int(getattr(expected, name))
        if value != expected_value:
            raise ValueError(
                f"Source {name} mismatch: expected {expected_value}, found {value}"
            )

    canonical = dict(zip(taxonomy.disease_ids, taxonomy.labels, strict=True))
    observed_pairs = set(
        frame[["disease_id", "label"]].astype(str).itertuples(index=False, name=None)
    )
    expected_pairs = set(canonical.items())
    if observed_pairs != expected_pairs:
        raise ValueError("Source disease_id/label pairs differ from the taxonomy")

    release = _load_release_document(source_release_path(config))
    if _required_string(release, "id") != "ISEPDermData":
        raise ValueError("Unexpected source dataset ID")
    if _required_string(release, "version") != config.dataset.source_version:
        raise ValueError("Source dataset version does not match configuration")
    declared_images = release.get("image_count")
    if declared_images != expected.image_count:
        raise ValueError("Source release image count does not match configuration")


def _load_release_document(path: Path) -> Mapping[object, object]:
    try:
        document: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load source release {path}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ValueError("Source release root must be an object")
    release = document.get("release")
    if not isinstance(release, Mapping):
        raise ValueError("Source release must contain a release object")
    return release


def _required_string(document: Mapping[object, object], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Source release {key} must be a non-empty string")
    return value


def _required_integer(document: Mapping[object, object], key: str) -> int:
    value = document.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Source release {key} must be a non-negative integer")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
