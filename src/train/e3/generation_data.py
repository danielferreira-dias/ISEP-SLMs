"""Integrity-checked, deterministic sample selection for E3 teacher generation."""

from __future__ import annotations

import hashlib
import io
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pyarrow.parquet as pq
from PIL import Image

from src.inference.base import detect_image_mime_type
from src.train.domain import Taxonomy, TaxonomyClass
from src.train.e3.generation_config import E3TeacherGenerationConfig


@dataclass(frozen=True, slots=True)
class E3Candidate:
    """Private diagnosis row identity without loading image bytes."""

    sample_id: str
    leakage_group_id: str
    disease_id: str
    gold_diagnosis: str
    split: Literal["sft_train", "sft_dev"]
    image_sha256: str
    shard_path: Path
    shard_manifest_sha256: str
    row_index: int


@dataclass(frozen=True, slots=True)
class E3TeacherSample:
    """One selected private row with verified embedded image bytes."""

    candidate: E3Candidate
    image_bytes: bytes
    image_mime_type: str
    image_width: int
    image_height: int


@dataclass(frozen=True, slots=True)
class E3Selection:
    taxonomy: Taxonomy
    candidates: tuple[E3Candidate, ...]
    selection_sha256: str
    release_id: str
    release_manifest_sha256: str


_METADATA_COLUMNS = (
    "sample_id",
    "leakage_group_id",
    "disease_id",
    "gold_diagnosis",
    "split",
    "image_sha256",
)


def select_e3_samples(
    config: E3TeacherGenerationConfig,
    *,
    limit: int | None = None,
) -> E3Selection:
    """Select a balanced stable sample set without reading image payloads."""

    taxonomy = load_taxonomy(config.path(config.dataset.taxonomy))
    label_by_id = {item.disease_id: item.label for item in taxonomy.classes}
    manifest_path = config.path(config.dataset.release_manifest)
    manifest = _load_json_object(manifest_path)
    release_root = config.path(config.dataset.release_root)
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError("E3 release manifest is missing shards")

    candidates: list[E3Candidate] = []
    expected_total = 0
    selected_splits = set(config.dataset.splits)
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError("E3 release manifest contains a non-object shard")
        if shard.get("config") != config.dataset.config:
            continue
        split = shard.get("split")
        if split not in selected_splits:
            continue
        relative = _required_text(shard, "path")
        shard_path = (release_root / relative).resolve()
        try:
            shard_path.relative_to(release_root)
        except ValueError:
            raise ValueError(f"Release shard escapes its root: {relative}") from None
        expected_rows = _required_int(shard, "rows")
        expected_total += expected_rows
        if shard_path.stat().st_size != _required_int(shard, "bytes"):
            raise ValueError(f"Release shard byte-size mismatch: {shard_path}")
        table = pq.read_table(  # type: ignore[no-untyped-call]
            shard_path,
            columns=list(_METADATA_COLUMNS),
        )
        if table.num_rows != expected_rows:
            raise ValueError(f"Release shard row-count mismatch: {shard_path}")
        shard_sha = _required_sha256(shard, "sha256")
        for row_index, row in enumerate(table.to_pylist()):
            row_split = _row_text(row, "split")
            if row_split not in {"sft_train", "sft_dev"}:
                raise ValueError(f"Unsupported E3 split in {shard_path}")
            candidate = E3Candidate(
                sample_id=_row_text(row, "sample_id"),
                leakage_group_id=_row_text(row, "leakage_group_id"),
                disease_id=_row_text(row, "disease_id"),
                gold_diagnosis=_row_text(row, "gold_diagnosis"),
                split=cast(Literal["sft_train", "sft_dev"], row_split),
                image_sha256=_row_sha256(row, "image_sha256"),
                shard_path=shard_path,
                shard_manifest_sha256=shard_sha,
                row_index=row_index,
            )
            if candidate.split != split:
                raise ValueError(f"Row split does not match shard: {shard_path}")
            if label_by_id.get(candidate.disease_id) != candidate.gold_diagnosis:
                raise ValueError(
                    f"Non-canonical private gold pair for {candidate.sample_id}"
                )
            candidates.append(candidate)
    if len(candidates) != expected_total:
        raise ValueError("E3 diagnosis release total does not match shard manifests")
    missing_classes = tuple(
        disease_id
        for disease_id in taxonomy.disease_ids
        if not any(item.disease_id == disease_id for item in candidates)
    )
    if missing_classes:
        raise ValueError(
            "E3 diagnosis release is missing taxonomy classes: "
            + ", ".join(missing_classes)
        )

    target = limit or config.dataset.selection.pilot_samples
    selected = _stratified_round_robin(
        candidates,
        taxonomy=taxonomy,
        seed=config.dataset.selection.seed,
        limit=target,
    )
    digest_payload = "\n".join(
        "\t".join(
            (
                item.sample_id,
                item.leakage_group_id,
                item.disease_id,
                item.split,
                item.image_sha256,
            )
        )
        for item in selected
    ).encode("utf-8")
    return E3Selection(
        taxonomy=taxonomy,
        candidates=selected,
        selection_sha256=hashlib.sha256(digest_payload).hexdigest(),
        release_id=_required_text(manifest, "release_id"),
        release_manifest_sha256=_sha256_file(manifest_path),
    )


def load_selected_images(
    selection: E3Selection,
    *,
    verify_shard_sha256: bool,
    verify_image_sha256: bool,
) -> tuple[E3TeacherSample, ...]:
    """Load each selected shard once and validate image integrity/decodability."""

    by_shard: dict[Path, list[E3Candidate]] = defaultdict(list)
    for candidate in selection.candidates:
        by_shard[candidate.shard_path].append(candidate)

    loaded: dict[str, E3TeacherSample] = {}
    for shard_path, candidates in by_shard.items():
        expected_shas = {item.shard_manifest_sha256 for item in candidates}
        if len(expected_shas) != 1:
            raise ValueError(f"Conflicting manifest hashes for shard {shard_path}")
        if verify_shard_sha256 and _sha256_file(shard_path) not in expected_shas:
            raise ValueError(f"Release shard SHA-256 mismatch: {shard_path}")
        table = pq.read_table(  # type: ignore[no-untyped-call]
            shard_path,
            columns=["image"],
        )
        for candidate in candidates:
            image_value = table.column("image")[candidate.row_index].as_py()
            if not isinstance(image_value, dict):
                raise ValueError(f"Invalid embedded image for {candidate.sample_id}")
            image_bytes = image_value.get("bytes")
            if not isinstance(image_bytes, bytes) or not image_bytes:
                raise ValueError(f"Missing embedded image for {candidate.sample_id}")
            actual_sha = hashlib.sha256(image_bytes).hexdigest()
            if verify_image_sha256 and actual_sha != candidate.image_sha256:
                raise ValueError(f"Image SHA-256 mismatch for {candidate.sample_id}")
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.verify()
            with Image.open(io.BytesIO(image_bytes)) as image:
                width, height = image.size
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image dimensions for {candidate.sample_id}")
            loaded[candidate.sample_id] = E3TeacherSample(
                candidate=candidate,
                image_bytes=image_bytes,
                image_mime_type=detect_image_mime_type(image_bytes),
                image_width=width,
                image_height=height,
            )
    return tuple(loaded[item.sample_id] for item in selection.candidates)


def selection_manifest(selection: E3Selection) -> dict[str, Any]:
    """Return the private, image-free manifest that freezes selected rows."""

    return {
        "schema_version": 1,
        "release_id": selection.release_id,
        "release_manifest_sha256": selection.release_manifest_sha256,
        "selection_sha256": selection.selection_sha256,
        "total_samples": len(selection.candidates),
        "samples": [
            {
                "sample_id": item.sample_id,
                "leakage_group_id": item.leakage_group_id,
                "disease_id": item.disease_id,
                "gold_diagnosis": item.gold_diagnosis,
                "split": item.split,
                "image_sha256": item.image_sha256,
                "shard": str(item.shard_path.name),
                "row_index": item.row_index,
            }
            for item in selection.candidates
        ],
    }


def load_taxonomy(path: Path) -> Taxonomy:
    document = _load_json_object(path)
    values = document.get("classes")
    if not isinstance(values, list) or not values:
        raise ValueError("E3 taxonomy contains no classes")
    classes = tuple(
        TaxonomyClass(
            disease_id=_required_text(item, "disease_id"),
            label=_required_text(item, "label"),
        )
        for item in values
        if isinstance(item, dict)
    )
    if len(classes) != len(values):
        raise ValueError("E3 taxonomy contains a non-object class")
    taxonomy = Taxonomy(
        taxonomy_id=_required_text(document, "taxonomy_id"),
        classes=classes,
    )
    if len(taxonomy.disease_ids) != len(set(taxonomy.disease_ids)):
        raise ValueError("E3 taxonomy disease IDs must be unique")
    if len(taxonomy.labels) != len(set(taxonomy.labels)):
        raise ValueError("E3 taxonomy labels must be unique")
    return taxonomy


def _stratified_round_robin(
    candidates: list[E3Candidate],
    *,
    taxonomy: Taxonomy,
    seed: int,
    limit: int,
) -> tuple[E3Candidate, ...]:
    if limit <= 0 or limit > len(candidates):
        raise ValueError("E3 selection limit exceeds the available diagnosis rows")
    grouped: dict[str, list[E3Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.disease_id].append(candidate)
    for disease_id in taxonomy.disease_ids:
        grouped[disease_id].sort(
            key=lambda item: (
                hashlib.sha256(f"{seed}:{item.sample_id}".encode()).hexdigest(),
                item.sample_id,
            )
        )

    positions = {disease_id: 0 for disease_id in taxonomy.disease_ids}
    used_groups: set[str] = set()
    result: list[E3Candidate] = []
    while len(result) < limit:
        before = len(result)
        for disease_id in taxonomy.disease_ids:
            values = grouped[disease_id]
            position = positions[disease_id]
            while position < len(values):
                candidate = values[position]
                position += 1
                if candidate.leakage_group_id in used_groups:
                    continue
                result.append(candidate)
                used_groups.add(candidate.leakage_group_id)
                break
            positions[disease_id] = position
            if len(result) == limit:
                break
        if len(result) == before:
            raise ValueError("E3 selection cannot satisfy leakage-group uniqueness")
    return tuple(result)


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _required_text(value: dict[str, Any], field: str) -> str:
    result = value.get(field)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{field} must be a non-empty string")
    return result


def _required_int(value: dict[str, Any], field: str) -> int:
    result = value.get(field)
    if not isinstance(result, int) or isinstance(result, bool) or result < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return result


def _required_sha256(value: dict[str, Any], field: str) -> str:
    result = _required_text(value, field)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return result


def _row_text(row: dict[str, Any], field: str) -> str:
    return _required_text(row, field)


def _row_sha256(row: dict[str, Any], field: str) -> str:
    return _required_sha256(row, field)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
