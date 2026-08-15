"""Lazy CUDA-free dataset view for TRL and Unsloth vision collators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast, overload

from PIL import Image

from src.train.config import TrainingConfig
from src.train.data.images import preprocess_image_with_metadata
from src.train.data.loading import load_assignments
from src.train.data.source import load_source_frame, source_shards
from src.train.data.taxonomy import load_taxonomy
from src.train.domain import LabeledImageSample, PreparedRelease, ReleaseSubset
from src.train.phases.base import TrainingPhase
from src.train.phases.registry import get_phase


class _ArrowDataset(Protocol):
    """Minimal Hugging Face Dataset surface used by the lazy adapter."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> object: ...

    def select(self, indices: Sequence[int]) -> _ArrowDataset: ...


@dataclass(frozen=True, slots=True)
class _AssignedSample:
    sample_id: str
    leakage_group_id: str
    disease_id: str
    label: str
    source: str


class LazyReleaseDataset(Sequence[dict[str, object]]):
    """Render one image-backed E1 record at a time from memory-mapped Arrow."""

    def __init__(
        self,
        *,
        backing: _ArrowDataset,
        samples: tuple[_AssignedSample, ...],
        phase: TrainingPhase,
        max_edge_pixels: int,
        source_root: Path,
        subset: ReleaseSubset = ReleaseSubset.SFT_TRAIN,
    ) -> None:
        """Initialize a view whose metadata order matches its Arrow indices."""

        if len(backing) != len(samples):
            raise ValueError("Backing dataset and assignments have different sizes")
        self._backing = backing
        self._samples = samples
        self._phase = phase
        self._max_edge_pixels = max_edge_pixels
        self._source_root = source_root
        self._subset = subset

    def __len__(self) -> int:
        """Return the number of assigned samples in this release view."""

        return len(self._samples)

    @overload
    def __getitem__(self, index: int) -> dict[str, object]: ...

    @overload
    def __getitem__(self, index: slice) -> list[dict[str, object]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> dict[str, object] | list[dict[str, object]]:
        """Decode, normalize, and phase-format requested records lazily."""

        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        normalized_index = index if index >= 0 else len(self) + index
        if normalized_index < 0 or normalized_index >= len(self):
            raise IndexError("LazyReleaseDataset index out of range")
        row = self._backing[normalized_index]
        if not isinstance(row, Mapping):
            raise ValueError("Hugging Face Dataset returned a non-mapping row")
        assigned = self._samples[normalized_index]
        _verify_backing_identity(row, assigned)
        image, geometry = preprocess_image_with_metadata(
            _image_input(row.get("image"), self._source_root),
            max_edge_pixels=self._max_edge_pixels,
        )
        sample = LabeledImageSample(
            sample_id=assigned.sample_id,
            leakage_group_id=assigned.leakage_group_id,
            disease_id=assigned.disease_id,
            label=assigned.label,
            source=assigned.source,
            image=image,
            subset=self._subset.value,
            image_width=geometry.image_width,
            image_height=geometry.image_height,
            pixel_count=geometry.pixel_count,
            resized_width=geometry.resized_width,
            resized_height=geometry.resized_height,
        )
        return self._phase.format_example(sample).as_record()


def build_lazy_phase_dataset(
    config: TrainingConfig,
    release: PreparedRelease | Path,
    subset: ReleaseSubset,
    *,
    phase: TrainingPhase | None = None,
    cache_directory: Path | None = None,
) -> LazyReleaseDataset:
    """Build a lazy Hugging Face-backed dataset for TRL/Unsloth.

    Only source metadata and assignment IDs are loaded eagerly.  The Hugging
    Face Dataset keeps Arrow data memory-mapped and decodes one image when its
    index is requested.
    """

    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "The training extra with the 'datasets' package is required"
        ) from exc

    assignments = load_assignments(release)
    if subset is ReleaseSubset.DEV_PANEL:
        selected = assignments[assignments["is_dev_panel"].astype(bool)]
    else:
        selected = assignments[assignments["split"] == subset.value]
    selected = selected.sort_values("sample_id", ignore_index=True)

    metadata = load_source_frame(config).reset_index(drop=True)
    source_indices = {
        str(sample_id): index
        for index, sample_id in enumerate(metadata["sample_id"].astype(str))
    }
    missing = set(selected["sample_id"].astype(str)) - set(source_indices)
    if missing:
        raise ValueError(f"Source pool is missing {len(missing)} assigned samples")
    indices = [source_indices[str(value)] for value in selected["sample_id"]]
    loaded: object = load_dataset(
        "parquet",
        data_files=[str(path) for path in source_shards(config)],
        split="train",
        cache_dir=(
            str(cache_directory.resolve()) if cache_directory is not None else None
        ),
    )
    backing = cast(_ArrowDataset, loaded).select(indices)
    samples = tuple(
        _AssignedSample(
            sample_id=str(row.sample_id),
            leakage_group_id=str(row.leakage_group_id),
            disease_id=str(row.disease_id),
            label=str(row.label),
            source=str(row.source),
        )
        for row in selected.itertuples(index=False)
    )
    selected_phase = phase or get_phase(
        config.experiment.phase,
        load_taxonomy(config),
    )
    return LazyReleaseDataset(
        backing=backing,
        samples=samples,
        phase=selected_phase,
        max_edge_pixels=config.dataset.image.max_edge_pixels,
        source_root=config.resolve_path(config.dataset.source_directory),
        subset=subset,
    )


def _image_input(value: object, source_root: Path) -> Image.Image | bytes | Path:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, bytes):
        return value
    if isinstance(value, Mapping):
        encoded = value.get("bytes")
        if isinstance(encoded, bytes):
            return encoded
        path_value = value.get("path")
        if isinstance(path_value, str) and path_value:
            path = Path(path_value)
            return path if path.is_absolute() else source_root / path
    if isinstance(value, str) and value:
        path = Path(value)
        return path if path.is_absolute() else source_root / path
    raise ValueError("Dataset row does not contain a decodable image")


def _verify_backing_identity(
    row: Mapping[object, object], assigned: _AssignedSample
) -> None:
    """Reject any Arrow-index drift before pairing an image with a label."""

    expected = {
        "sample_id": assigned.sample_id,
        "leakage_group_id": assigned.leakage_group_id,
        "disease_id": assigned.disease_id,
        "label": assigned.label,
        "source": assigned.source,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(
                f"Backing row identity mismatch for {key}: expected "
                f"{value!r}, found {row.get(key)!r}"
            )
