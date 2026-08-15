"""Materialize frozen release views from source shards and ID assignments."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pyarrow.parquet as pq

from src.train.config import TrainingConfig
from src.train.data.images import preprocess_image_with_metadata
from src.train.data.integrity import validate_assignment_columns
from src.train.data.source import (
    SOURCE_METADATA_COLUMNS,
    load_source_frame,
    source_shards,
)
from src.train.domain import LabeledImageSample, PreparedRelease, ReleaseSubset


def load_assignments(release: PreparedRelease | Path) -> pd.DataFrame:
    """Load and validate the canonical assignment manifest."""

    path = (
        release.assignments_path
        if isinstance(release, PreparedRelease)
        else release.resolve() / "assignments.parquet"
    )
    assignments = pq.read_table(path).to_pandas()  # type: ignore[no-untyped-call]
    validate_assignment_columns(assignments)
    return assignments


def apply_assignments(
    source_frame: pd.DataFrame,
    assignments: pd.DataFrame,
    subset: ReleaseSubset,
) -> pd.DataFrame:
    """Select one release view while retaining source image data when present.

    Args:
        source_frame: Source metadata or source rows including ``image``.
        assignments: Canonical ID-only assignments.
        subset: Train, full development, or fixed development panel.

    Returns:
        Source rows for the requested view with split audit fields appended.
    """

    validate_assignment_columns(assignments)
    if "sample_id" not in source_frame.columns:
        raise ValueError("Source frame must contain sample_id")
    if not source_frame["sample_id"].is_unique:
        raise ValueError("Source frame sample_id values must be unique")
    if subset is ReleaseSubset.DEV_PANEL:
        scoped = assignments[assignments["is_dev_panel"].astype(bool)]
    else:
        scoped = assignments[assignments["split"] == subset.value]
    selector = scoped[["sample_id", "split", "is_dev_panel"]]
    result = source_frame.merge(
        selector,
        on="sample_id",
        how="inner",
        validate="one_to_one",
    ).sort_values("sample_id", ignore_index=True)
    if len(result) != len(scoped):
        missing = set(scoped["sample_id"]) - set(result["sample_id"])
        raise ValueError(f"Source pool is missing {len(missing)} assigned samples")
    return result


def load_release_frame(
    config: TrainingConfig,
    release: PreparedRelease | Path,
    subset: ReleaseSubset,
    *,
    include_images: bool = False,
) -> pd.DataFrame:
    """Load one complete release view into a pandas DataFrame.

    Set ``include_images`` only when sufficient host memory is available.  For
    training, :func:`iter_release_samples` keeps image memory bounded.
    """

    source = load_source_frame(config, include_images=include_images)
    return apply_assignments(source, load_assignments(release), subset)


def iter_release_samples(
    config: TrainingConfig,
    release: PreparedRelease | Path,
    subset: ReleaseSubset,
    *,
    batch_size: int = 8,
) -> Iterator[LabeledImageSample]:
    """Yield decoded and preprocessed images from one frozen release view."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    assignments = load_assignments(release)
    if subset is ReleaseSubset.DEV_PANEL:
        selected = assignments[assignments["is_dev_panel"].astype(bool)]
    else:
        selected = assignments[assignments["split"] == subset.value]
    selected_ids = frozenset(selected["sample_id"].astype(str))
    yielded = 0
    columns = ["image", *SOURCE_METADATA_COLUMNS]
    source_root = config.resolve_path(config.dataset.source_directory)
    for shard in source_shards(config):
        parquet = pq.ParquetFile(shard)  # type: ignore[no-untyped-call]
        for batch in parquet.iter_batches(  # type: ignore[no-untyped-call]
            batch_size=batch_size, columns=columns
        ):
            raw_rows: object = batch.to_pylist()
            if not isinstance(raw_rows, list):
                raise ValueError("Unexpected Arrow batch representation")
            for raw_row in raw_rows:
                if not isinstance(raw_row, Mapping):
                    raise ValueError("Unexpected Arrow row representation")
                sample_id = _required_string(raw_row, "sample_id")
                if sample_id not in selected_ids:
                    continue
                yielded += 1
                image, geometry = preprocess_image_with_metadata(
                    _image_input(raw_row.get("image"), source_root),
                    max_edge_pixels=config.dataset.image.max_edge_pixels,
                )
                yield LabeledImageSample(
                    sample_id=sample_id,
                    leakage_group_id=_required_string(raw_row, "leakage_group_id"),
                    disease_id=_required_string(raw_row, "disease_id"),
                    label=_required_string(raw_row, "label"),
                    source=_required_string(raw_row, "source"),
                    image=image,
                    subset=subset.value,
                    image_width=geometry.image_width,
                    image_height=geometry.image_height,
                    pixel_count=geometry.pixel_count,
                    resized_width=geometry.resized_width,
                    resized_height=geometry.resized_height,
                )
    if yielded != len(selected_ids):
        raise ValueError(
            f"Expected {len(selected_ids)} release samples, yielded {yielded}"
        )


def _required_string(row: Mapping[object, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Source row {key} must be a non-empty string")
    return value


def _image_input(value: object, source_root: Path) -> bytes | Path:
    if not isinstance(value, Mapping):
        raise ValueError("Source image must use the Hugging Face image struct")
    encoded = value.get("bytes")
    if isinstance(encoded, bytes):
        return encoded
    relative = value.get("path")
    if isinstance(relative, str) and relative:
        path = Path(relative)
        return path if path.is_absolute() else source_root / path
    raise ValueError("Source image has neither embedded bytes nor a path")
