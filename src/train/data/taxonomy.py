"""Closed-taxonomy loading and validation for E1 training."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from src.train.config import TrainingConfig
from src.train.domain import Taxonomy, TaxonomyClass


def load_taxonomy(config: TrainingConfig) -> Taxonomy:
    """Load the ordered taxonomy pinned by a training configuration.

    Args:
        config: Complete training configuration.

    Returns:
        Validated immutable taxonomy.

    Raises:
        ValueError: If the taxonomy is missing, malformed, or inconsistent.
    """

    source_root = config.resolve_path(config.dataset.source_directory)
    taxonomy_path = _child_path(source_root, config.dataset.taxonomy_file)
    try:
        document: object = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load taxonomy {taxonomy_path}: {exc}") from exc
    root = _mapping(document, "taxonomy root")
    taxonomy_id = _required_string(root, "taxonomy_id")
    raw_classes = root.get("classes")
    if not isinstance(raw_classes, list) or not raw_classes:
        raise ValueError("Taxonomy classes must be a non-empty list")

    classes: list[TaxonomyClass] = []
    for index, raw_class in enumerate(raw_classes):
        item = _mapping(raw_class, f"taxonomy class {index}")
        classes.append(
            TaxonomyClass(
                disease_id=_required_string(item, "disease_id"),
                label=_required_string(item, "label"),
            )
        )
    taxonomy = Taxonomy(taxonomy_id=taxonomy_id, classes=tuple(classes))
    if len(set(taxonomy.labels)) != len(taxonomy.labels):
        raise ValueError("Taxonomy contains duplicate labels")
    if len(set(taxonomy.disease_ids)) != len(taxonomy.disease_ids):
        raise ValueError("Taxonomy contains duplicate disease IDs")
    active_count = root.get("active_class_count")
    if not isinstance(active_count, int) or isinstance(active_count, bool):
        raise ValueError("active_class_count must be an integer")
    if active_count != len(taxonomy.classes):
        raise ValueError(
            "active_class_count does not match the number of taxonomy classes"
        )
    return taxonomy


def _child_path(root: Path, child: Path) -> Path:
    return child if child.is_absolute() else root / child


def _mapping(value: object, name: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _required_string(document: Mapping[object, object], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value
