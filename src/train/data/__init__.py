"""Public data preparation API for reproducible E1 training."""

from src.train.data.dataset import LazyReleaseDataset, build_lazy_phase_dataset
from src.train.data.images import preprocess_image
from src.train.data.integrity import inspect_data_release
from src.train.data.loading import (
    apply_assignments,
    iter_release_samples,
    load_assignments,
    load_release_frame,
)
from src.train.data.release import prepare_data_release
from src.train.data.source import (
    load_source_frame,
    source_shards,
    validate_source_shards,
)
from src.train.data.taxonomy import load_taxonomy

__all__ = [
    "LazyReleaseDataset",
    "apply_assignments",
    "build_lazy_phase_dataset",
    "inspect_data_release",
    "iter_release_samples",
    "load_assignments",
    "load_release_frame",
    "load_source_frame",
    "load_taxonomy",
    "prepare_data_release",
    "preprocess_image",
    "source_shards",
    "validate_source_shards",
]
