"""Lazy image-backed loader for human diagnosis and SKINCON E2 tasks."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast, overload

from src.train.config import TrainingConfig
from src.train.data.images import preprocess_image_with_metadata
from src.train.data.taxonomy import load_taxonomy
from src.train.domain import ReleaseSubset
from src.train.e2.domain import (
    E2HumanSample,
    E2ReleaseAudit,
    E2TaskName,
    MorphologyTarget,
)
from src.train.e2.mixing import DeterministicTaskInterleave
from src.train.e2.phase import E2HumanPhase
from src.train.e2.release import e2_shards


class _ArrowDataset(Protocol):
    """Minimal Hugging Face Dataset surface required by the adapter."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> object: ...

    def cast_column(self, column: str, feature: object) -> _ArrowDataset: ...


class E2HumanDataset(Sequence[dict[str, object]]):
    """Decode, audit, and format one E2 row only when requested."""

    def __init__(
        self,
        *,
        backing: _ArrowDataset,
        task: E2TaskName,
        subset: ReleaseSubset,
        phase: E2HumanPhase,
        schema_version: str,
        max_edge_pixels: int,
        availability_by_image: dict[str, tuple[E2TaskName, ...]],
    ) -> None:
        """Bind one immutable task/split view to its formatting contract."""

        if subset is ReleaseSubset.DEV_PANEL:
            raise ValueError("E2 does not define a separate morphology panel")
        self._backing = backing
        self._task = task
        self._subset = subset
        self._phase = phase
        self._schema_version = schema_version
        self._max_edge_pixels = max_edge_pixels
        self._availability_by_image = availability_by_image

    def __len__(self) -> int:
        """Return the number of rows in this task view."""

        return len(self._backing)

    @overload
    def __getitem__(self, index: int) -> dict[str, object]: ...

    @overload
    def __getitem__(self, index: slice) -> list[dict[str, object]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> dict[str, object] | list[dict[str, object]]:
        """Validate and phase-format one row or a materialized slice."""

        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        normalized = index if index >= 0 else len(self) + index
        if normalized < 0 or normalized >= len(self):
            raise IndexError("E2 dataset index out of range")
        return self._phase.format_example(self.sample(normalized)).as_record()

    def sample(self, index: int) -> E2HumanSample:
        """Return one validated typed sample for deterministic evaluation."""

        normalized = index if index >= 0 else len(self) + index
        if normalized < 0 or normalized >= len(self):
            raise IndexError("E2 dataset index out of range")
        raw = self._backing[normalized]
        if not isinstance(raw, Mapping):
            raise ValueError("E2 backing dataset returned a non-mapping row")
        return _sample_from_row(
            raw,
            task=self._task,
            subset=self._subset,
            schema_version=self._schema_version,
            max_edge_pixels=self._max_edge_pixels,
            availability_by_image=self._availability_by_image,
        )


def build_e2_training_dataset(
    config: TrainingConfig,
    audit: E2ReleaseAudit,
    subset: ReleaseSubset,
    *,
    cache_directory: Path | None = None,
) -> DeterministicTaskInterleave:
    """Load admitted human E2 configs and interleave every row exactly once."""

    if subset is ReleaseSubset.DEV_PANEL:
        raise ValueError("E2 training dataset supports only train and dev")
    taxonomy = load_taxonomy(config)
    phase = E2HumanPhase(taxonomy=taxonomy, ontology=audit.ontology)
    diagnosis = build_e2_task_dataset(
        config,
        audit,
        subset,
        E2TaskName.DIAGNOSIS,
        phase,
        cache_directory,
    )
    morphology = build_e2_task_dataset(
        config,
        audit,
        subset,
        E2TaskName.MORPHOLOGY,
        phase,
        cache_directory,
    )
    caption = (
        build_e2_task_dataset(
            config,
            audit,
            subset,
            E2TaskName.CAPTION,
            phase,
            cache_directory,
        )
        if audit.caption_train > 0 or audit.caption_dev > 0
        else None
    )
    expected = (
        (audit.diagnosis_train, audit.morphology_train, audit.caption_train)
        if subset is ReleaseSubset.SFT_TRAIN and caption is not None
        else (
            (audit.diagnosis_dev, audit.morphology_dev, audit.caption_dev)
            if caption is not None
            else (
                (audit.diagnosis_train, audit.morphology_train)
                if subset is ReleaseSubset.SFT_TRAIN
                else (audit.diagnosis_dev, audit.morphology_dev)
            )
        )
    )
    loaded_counts = (
        (len(diagnosis), len(morphology), len(caption))
        if caption is not None
        else (len(diagnosis), len(morphology))
    )
    if loaded_counts != expected:
        raise ValueError("Loaded E2 task counts differ from the audited manifest")
    return DeterministicTaskInterleave(diagnosis, morphology, caption)


def build_e2_task_dataset(
    config: TrainingConfig,
    audit: E2ReleaseAudit,
    subset: ReleaseSubset,
    task: E2TaskName,
    phase: E2HumanPhase,
    cache_directory: Path | None,
) -> E2HumanDataset:
    """Build one lazy, audited E2 task/split view."""

    try:
        from datasets import Image as DatasetImage  # type: ignore[import-untyped]
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The training extra with datasets is required") from exc
    loaded: object = load_dataset(
        "parquet",
        data_files=[str(path) for path in e2_shards(audit, task, subset.value)],
        split="train",
        cache_dir=(str(cache_directory.resolve()) if cache_directory else None),
    )
    backing = cast(_ArrowDataset, loaded).cast_column(
        "image", DatasetImage(decode=False)
    )
    return E2HumanDataset(
        backing=backing,
        task=task,
        subset=subset,
        phase=phase,
        schema_version=audit.schema_version_for(task),
        max_edge_pixels=config.dataset.image.max_edge_pixels,
        availability_by_image=audit.annotation_availability_by_image(),
    )


def _sample_from_row(
    row: Mapping[object, object],
    *,
    task: E2TaskName,
    subset: ReleaseSubset,
    schema_version: str,
    max_edge_pixels: int,
    availability_by_image: dict[str, tuple[E2TaskName, ...]],
) -> E2HumanSample:
    _validate_common(row, task=task, subset=subset, schema_version=schema_version)
    encoded = _image_bytes(row.get("image"))
    if hashlib.sha256(encoded).hexdigest() != _string(row, "image_sha256"):
        raise ValueError("E2 row image SHA-256 differs from its provenance")
    prompt = _string(row, "prompt")
    if hashlib.sha256(prompt.encode()).hexdigest() != _string(row, "prompt_sha256"):
        raise ValueError("E2 row prompt SHA-256 differs")
    target = _string(row, "target_text")
    sample_id = _string(row, "sample_id")
    leakage_group_id = _string(row, "leakage_group_id")
    source = _string(row, "source_dataset")
    image_sha256 = _string(row, "image_sha256")
    image, geometry = preprocess_image_with_metadata(
        encoded,
        max_edge_pixels=max_edge_pixels,
    )
    availability = availability_by_image.get(image_sha256)
    if availability is None or task not in availability:
        raise ValueError("E2 row has inconsistent annotation availability")
    if task is E2TaskName.DIAGNOSIS:
        return E2HumanSample(
            sample_id=sample_id,
            leakage_group_id=leakage_group_id,
            task=task,
            source=source,
            image=image,
            prompt=prompt,
            target_text=target,
            disease_id=_string(row, "disease_id"),
            label=_string(row, "gold_diagnosis"),
            subset=subset.value,
            image_sha256=image_sha256,
            image_width=geometry.image_width,
            image_height=geometry.image_height,
            pixel_count=geometry.pixel_count,
            resized_width=geometry.resized_width,
            resized_height=geometry.resized_height,
            annotation_availability=availability,
        )
    if task is E2TaskName.MORPHOLOGY:
        return E2HumanSample(
            sample_id=sample_id,
            leakage_group_id=leakage_group_id,
            task=task,
            source=source,
            image=image,
            prompt=prompt,
            target_text=target,
            morphology=_morphology_target(row, target),
            subset=subset.value,
            image_sha256=image_sha256,
            image_width=geometry.image_width,
            image_height=geometry.image_height,
            pixel_count=geometry.pixel_count,
            resized_width=geometry.resized_width,
            resized_height=geometry.resized_height,
            annotation_availability=availability,
        )
    return E2HumanSample(
        sample_id=sample_id,
        leakage_group_id=leakage_group_id,
        task=task,
        source=source,
        image=image,
        prompt=prompt,
        target_text=target,
        source_caption_sha256=_digest(row, "source_caption_sha256"),
        caption_transform_version=_string(row, "transform_version"),
        subset=subset.value,
        image_sha256=image_sha256,
        image_width=geometry.image_width,
        image_height=geometry.image_height,
        pixel_count=geometry.pixel_count,
        resized_width=geometry.resized_width,
        resized_height=geometry.resized_height,
        annotation_availability=availability,
    )


def _validate_common(
    row: Mapping[object, object],
    *,
    task: E2TaskName,
    subset: ReleaseSubset,
    schema_version: str,
) -> None:
    task_contracts = {
        E2TaskName.DIAGNOSIS: (
            "diagnosis_label_only_v1",
            "human_source_join",
            "canonical_label_v1",
        ),
        E2TaskName.MORPHOLOGY: (
            "skincon_morphology_v1",
            "human_annotated",
            "skincon_positive_concepts_v1",
        ),
        E2TaskName.CAPTION: (
            "skincap_observation_caption_v1",
            "human_caption_gold_conditioned_filtered",
            "observation_only_single_sentence_v1",
        ),
    }
    task_id, target_source, target_variant = task_contracts[task]
    expected = {
        "split": subset.value,
        "schema_version": schema_version,
        "quality_status": "accepted",
        "task_id": task_id,
        "target_source": target_source,
        "target_variant": target_variant,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"E2 {task.value} row has invalid {key}: {row.get(key)!r}")


def _morphology_target(
    row: Mapping[object, object], target_text: str
) -> MorphologyTarget:
    skincon = row.get("skincon")
    if not isinstance(skincon, Mapping):
        raise ValueError("Morphology row has no SKINCON object")
    if skincon.get("ontology_version") != "skincon_48_v1":
        raise ValueError("Morphology row has an unknown ontology version")
    if skincon.get("annotation_source") != "SKINCON":
        raise ValueError("Morphology row is not human-annotated SKINCON")
    complete = skincon.get("all_concepts_annotated")
    if complete is not True:
        raise ValueError("Morphology row is not fully annotated")
    raw = skincon.get("positive_concepts")
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise ValueError("Morphology positives must be an array of strings")
    positives = tuple(cast(str, item) for item in raw)
    target = MorphologyTarget(positives, all_concepts_annotated=True)
    if target.canonical_json() != target_text:
        raise ValueError("Morphology row target differs from its SKINCON object")
    return target


def _image_bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, Mapping):
        encoded = value.get("bytes")
        if isinstance(encoded, bytes):
            return encoded
    raise ValueError("E2 row does not contain embedded image bytes")


def _string(row: Mapping[object, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"E2 row {key} must be a non-empty string")
    return value


def _digest(row: Mapping[object, object], key: str) -> str:
    value = _string(row, key)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"E2 row {key} must be a lowercase SHA-256")
    return value
