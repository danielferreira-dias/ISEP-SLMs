"""Image fingerprinting and leakage-safe duplicate grouping."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from io import BytesIO
import itertools
import json
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urldefrag
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.data_pipeline.common import MANIFEST_ARROW_SCHEMA


ProgressCallback = Callable[[str], None]
PERCEPTUAL_HASH_ALGORITHM = "phash_dct_32x32_low8x8_median_v1"


@dataclass(frozen=True)
class Fingerprint:
    """Computed identifiers and decode metadata for one image."""

    image_sha256: str
    perceptual_hash: str
    width: int
    height: int
    image_format: str | None


class ImageResolver(AbstractContextManager["ImageResolver"]):
    """Read normalized direct, ZIP, and embedded-Parquet image locators."""

    def __init__(self, root: Path, parquet_cache_size: int = 6) -> None:
        self.root = root
        self.parquet_cache_size = parquet_cache_size
        self._zip_archives: dict[Path, zipfile.ZipFile] = {}
        self._parquet_columns: OrderedDict[
            tuple[Path, str],
            pa.ChunkedArray,
        ] = OrderedDict()

    def read_bytes(self, image_uri: str) -> bytes:
        """Resolve one manifest image URI to its encoded byte representation."""

        if image_uri.startswith("zip://"):
            return self._read_zip(image_uri)
        if image_uri.startswith("parquet://"):
            return self._read_parquet(image_uri)
        return (self.root / image_uri).read_bytes()

    def _read_zip(self, image_uri: str) -> bytes:
        locator = image_uri.removeprefix("zip://")
        archive_value, separator, member = locator.partition("::")
        if not separator or not archive_value or not member:
            raise ValueError(f"Invalid ZIP image URI: {image_uri!r}")
        archive_path = self.root / archive_value
        archive = self._zip_archives.get(archive_path)
        if archive is None:
            archive = zipfile.ZipFile(archive_path)
            self._zip_archives[archive_path] = archive
        return archive.read(member)

    def _read_parquet(self, image_uri: str) -> bytes:
        locator = image_uri.removeprefix("parquet://")
        tokens = locator.split("::")
        if len(tokens) != 3:
            raise ValueError(f"Invalid Parquet image URI: {image_uri!r}")
        parquet_path = self.root / tokens[0]
        row_token = tokens[1].partition("=")
        column_token = tokens[2].partition("=")
        if row_token[:2] != ("row", "=") or column_token[:2] != ("column", "="):
            raise ValueError(f"Invalid Parquet image URI: {image_uri!r}")
        row_number = int(row_token[2])
        column_name = column_token[2]
        cache_key = (parquet_path, column_name)
        values = self._parquet_columns.get(cache_key)
        if values is None:
            values = pq.read_table(
                parquet_path,
                columns=[f"{column_name}.bytes"],
            ).column(0)
            self._parquet_columns[cache_key] = values
            if len(self._parquet_columns) > self.parquet_cache_size:
                self._parquet_columns.popitem(last=False)
        else:
            self._parquet_columns.move_to_end(cache_key)
        encoded = values[row_number].as_py()
        if encoded is None:
            raise ValueError(f"Missing embedded bytes for {image_uri!r}")
        return encoded

    def close(self) -> None:
        """Close open ZIP archives and release cached Parquet columns."""

        for archive in self._zip_archives.values():
            archive.close()
        self._zip_archives.clear()
        self._parquet_columns.clear()

    def __exit__(self, *args: object) -> None:
        self.close()


class UnionFind:
    """Small deterministic disjoint-set implementation."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1

    def components(self) -> dict[int, list[int]]:
        result: dict[int, list[int]] = defaultdict(list)
        for index in range(len(self.parent)):
            result[self.find(index)].append(index)
        return dict(result)


class BKTree:
    """Index integer hashes for bounded Hamming-distance searches."""

    def __init__(self) -> None:
        self.root: tuple[int, dict[int, Any]] | None = None

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = (value, {})
            return
        node = self.root
        while True:
            node_value, children = node
            distance = hamming_distance_int(value, node_value)
            if distance == 0:
                return
            child = children.get(distance)
            if child is None:
                children[distance] = (value, {})
                return
            node = child

    def query(self, value: int, threshold: int) -> list[int]:
        if self.root is None:
            return []
        matches: list[int] = []
        pending = [self.root]
        while pending:
            node_value, children = pending.pop()
            distance = hamming_distance_int(value, node_value)
            if distance <= threshold:
                matches.append(node_value)
            lower = distance - threshold
            upper = distance + threshold
            pending.extend(
                child
                for edge_distance, child in children.items()
                if lower <= edge_distance <= upper
            )
        return matches


def compute_fingerprint(encoded_image: bytes) -> Fingerprint:
    """Compute an encoded-byte SHA-256 and orientation-normalized 64-bit pHash."""

    image_sha256 = hashlib.sha256(encoded_image).hexdigest()
    with Image.open(BytesIO(encoded_image)) as source:
        image_format = source.format
        oriented = ImageOps.exif_transpose(source)
        width, height = oriented.size
        grayscale = oriented.convert("L").resize(
            (32, 32),
            Image.Resampling.LANCZOS,
        )
        pixels = np.asarray(grayscale, dtype=np.float64)

    basis = _dct_basis(32)
    coefficients = basis @ pixels @ basis.T
    low_frequency = coefficients[:8, :8].reshape(-1)
    median = float(np.median(low_frequency[1:]))
    bits = low_frequency > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return Fingerprint(
        image_sha256=image_sha256,
        perceptual_hash=f"{value:016x}",
        width=int(width),
        height=int(height),
        image_format=image_format,
    )


@lru_cache(maxsize=2)
def _dct_basis(size: int) -> np.ndarray:
    positions = np.arange(size, dtype=np.float64)
    frequencies = positions[:, None]
    basis = np.cos(np.pi * (2 * positions + 1) * frequencies / (2 * size))
    basis[0, :] *= np.sqrt(1.0 / size)
    basis[1:, :] *= np.sqrt(2.0 / size)
    return basis


def hamming_distance(left: str, right: str) -> int:
    """Return bit distance between two hexadecimal perceptual hashes."""

    return hamming_distance_int(int(left, 16), int(right, 16))


def hamming_distance_int(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def deduplicate_manifests(
    *,
    manifest_paths: dict[str, Path],
    root: Path,
    policy: dict[str, Any],
    progress: ProgressCallback,
) -> dict[str, Any]:
    """Fingerprint every manifest, group duplicates, update rows, and write reports."""

    frames: list[pd.DataFrame] = []
    for dataset_id, path in manifest_paths.items():
        frame = pq.read_table(path).to_pandas()
        frame["_manifest_path"] = str(path)
        frame["_include_before_deduplication"] = frame["include"].astype(bool)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    review_document = _load_review_document(root, policy)

    fingerprints: list[Fingerprint] = []
    total = len(combined)
    with ImageResolver(root) as resolver:
        for index, image_uri in enumerate(combined["image_uri"], start=1):
            encoded = resolver.read_bytes(str(image_uri))
            fingerprints.append(compute_fingerprint(encoded))
            if index == 1 or index % 500 == 0 or index == total:
                progress(f"Fingerprinted {index}/{total} images")

    combined["image_sha256"] = [
        value.image_sha256 for value in fingerprints
    ]
    combined["perceptual_hash"] = [
        value.perceptual_hash for value in fingerprints
    ]
    combined["perceptual_hash_algorithm"] = PERCEPTUAL_HASH_ALGORITHM
    combined["_image_width"] = [value.width for value in fingerprints]
    combined["_image_height"] = [value.height for value in fingerprints]
    combined["_image_format"] = [value.image_format for value in fingerprints]

    analysis = analyze_duplicate_frame(
        combined,
        policy=policy,
        review_document=review_document,
    )
    updated = analysis["frame"]
    _write_updated_manifests(
        updated,
        manifest_paths=manifest_paths,
    )

    output_config = policy["deduplication"]["outputs"]
    output_paths = {
        name: root / relative_path
        for name, relative_path in output_config.items()
    }
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    analysis["pairs"].to_csv(output_paths["duplicate_pairs"], index=False)
    analysis["groups"].to_csv(output_paths["duplicate_groups"], index=False)
    analysis["summary"].to_csv(output_paths["duplicate_summary"], index=False)
    analysis["review_queue"].to_csv(
        output_paths["perceptual_review_queue"],
        index=False,
    )

    return {
        "paths": output_paths,
        "pair_count": len(analysis["pairs"]),
        "group_count": len(analysis["groups"]),
        "review_queue_count": len(analysis["review_queue"]),
        "exact_exclusion_count": int(
            (
                (~updated["include"])
                & updated["_include_before_deduplication"]
            ).sum()
        ),
    }


def analyze_duplicate_frame(
    frame: pd.DataFrame,
    *,
    policy: dict[str, Any],
    review_document: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build exact, perceptual, and source-lineage relations for hashed rows."""

    frame = frame.copy().reset_index(drop=True)
    threshold = int(
        policy["deduplication"]["perceptual_hash"][
            "hamming_distance_threshold"
        ]
    )
    lineage_keys = list(
        policy["deduplication"].get("source_lineage_keys", [])
    )
    edge_evidence: dict[tuple[int, int], set[str]] = defaultdict(set)

    exact_groups: list[list[int]] = []
    for _, members in frame.groupby("image_sha256", sort=True).groups.items():
        indexes = sorted(int(value) for value in members)
        if len(indexes) < 2:
            continue
        exact_groups.append(indexes)
        for left, right in itertools.combinations(indexes, 2):
            edge_evidence[(left, right)].add("exact")

    tree = BKTree()
    hash_members: dict[int, list[int]] = defaultdict(list)
    for index, value in enumerate(frame["perceptual_hash"]):
        integer_hash = int(str(value), 16)
        for candidate_hash in tree.query(integer_hash, threshold):
            for candidate_index in hash_members[candidate_hash]:
                if (
                    frame.at[candidate_index, "image_sha256"]
                    == frame.at[index, "image_sha256"]
                ):
                    continue
                edge_evidence[(candidate_index, index)].add("perceptual")
        if not hash_members[integer_hash]:
            tree.add(integer_hash)
        hash_members[integer_hash].append(index)

    lineage_members: dict[str, list[int]] = defaultdict(list)
    for index, row in frame.iterrows():
        for lineage_value in _source_lineage_values(row, lineage_keys):
            lineage_members[lineage_value].append(int(index))
    for members in lineage_members.values():
        indexes = sorted(set(members))
        if len(indexes) < 2:
            continue
        for left, right in itertools.combinations(indexes, 2):
            edge_evidence[(left, right)].add("source_lineage")

    _apply_rejected_candidate_edges(
        frame,
        edge_evidence,
        review_document=review_document,
    )

    duplicate_union = UnionFind(len(frame))
    leakage_union = UnionFind(len(frame))
    group_members: dict[str, list[int]] = defaultdict(list)
    for index, group_id in enumerate(frame["group_id"]):
        group_members[str(group_id)].append(index)
    for members in group_members.values():
        first = members[0]
        for other in members[1:]:
            leakage_union.union(first, other)
    for left, right in edge_evidence:
        duplicate_union.union(left, right)
        leakage_union.union(left, right)

    frame["duplicate_group_id"] = None
    frame["duplicate_match_type"] = None
    frame["deduplication_status"] = "unique"
    frame["leakage_group_id"] = frame["group_id"].astype(str)

    duplicate_components = [
        members
        for members in duplicate_union.components().values()
        if len(members) > 1
    ]
    for members in duplicate_components:
        component_edges = {
            evidence
            for (left, right), values in edge_evidence.items()
            if left in members and right in members
            for evidence in values
        }
        group_id = _stable_group_id(
            "DUPLICATE",
            frame.loc[members, "sample_id"],
        )
        match_type = (
            next(iter(component_edges))
            if len(component_edges) == 1
            else "mixed"
        )
        frame.loc[members, "duplicate_group_id"] = group_id
        frame.loc[members, "duplicate_match_type"] = match_type
        candidate_status = (
            "source_lineage_candidate"
            if component_edges == {"source_lineage"}
            else "perceptual_candidate"
        )
        frame.loc[members, "deduplication_status"] = candidate_status

    dataset_priority = _dataset_priority(policy)
    exact_conflict_decisions = _exact_conflict_decisions(review_document)
    for members in exact_groups:
        mapped_ids = {
            str(value)
            for value in frame.loc[members, "disease_id"].dropna()
        }
        if len(mapped_ids) > 1:
            duplicate_group_id = str(
                frame.loc[members, "duplicate_group_id"].iloc[0]
            )
            decision = exact_conflict_decisions.get(duplicate_group_id)
            _apply_exact_conflict_decision(
                frame,
                members,
                duplicate_group_id=duplicate_group_id,
                decision=decision,
            )
            continue

        canonical = min(
            members,
            key=lambda index: _canonical_priority(
                frame.loc[index],
                dataset_priority,
            ),
        )
        frame.at[canonical, "deduplication_status"] = "canonical"
        for index in members:
            if index == canonical:
                continue
            frame.at[index, "deduplication_status"] = "redundant_exact"
            if bool(frame.at[index, "include"]):
                frame.at[index, "include"] = False
                frame.at[index, "exclusion_reason"] = "exact_duplicate_redundant"

    for members in leakage_union.components().values():
        original_groups = sorted(
            set(frame.loc[members, "group_id"].astype(str))
        )
        leakage_group_id = (
            original_groups[0]
            if len(original_groups) == 1
            else _stable_group_id("LEAKAGE", original_groups)
        )
        frame.loc[members, "leakage_group_id"] = leakage_group_id

    pairs = _pair_report(frame, edge_evidence)
    groups = _group_report(frame, duplicate_components, edge_evidence)
    summary = _summary_report(frame, groups)
    review_queue = _perceptual_review_queue(pairs, policy)
    return {
        "frame": frame,
        "pairs": pairs,
        "groups": groups,
        "summary": summary,
        "review_queue": review_queue,
    }


def _load_review_document(
    root: Path,
    policy: dict[str, Any],
) -> dict[str, Any] | None:
    review_config = policy.get("deduplication", {}).get("review", {})
    relative_path = review_config.get("decisions")
    if not relative_path:
        return None
    path = root / str(relative_path)
    if not path.exists():
        raise FileNotFoundError(f"Duplicate-review file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Duplicate-review file must contain a mapping: {path}")
    if document.get("schema_version") != 1:
        raise ValueError(f"Unsupported duplicate-review schema in {path}")
    return document


def _exact_conflict_decisions(
    review_document: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if review_document is None:
        return {}
    decisions: dict[str, dict[str, Any]] = {}
    for decision in review_document.get("exact_conflict_decisions", []):
        duplicate_group_id = str(decision["duplicate_group_id"])
        if duplicate_group_id in decisions:
            raise ValueError(
                "Duplicate exact-conflict review decision for "
                f"{duplicate_group_id}"
            )
        decisions[duplicate_group_id] = decision
    return decisions


def _perceptual_decisions(
    review_document: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if review_document is None:
        return {}
    decisions: dict[str, dict[str, Any]] = {}
    for decision in review_document.get("perceptual_decisions", []):
        duplicate_group_id = str(decision["duplicate_group_id"])
        if duplicate_group_id in decisions:
            raise ValueError(
                "Duplicate perceptual review decision for "
                f"{duplicate_group_id}"
            )
        decisions[duplicate_group_id] = decision
    return decisions


def _apply_rejected_candidate_edges(
    frame: pd.DataFrame,
    edge_evidence: dict[tuple[int, int], set[str]],
    *,
    review_document: dict[str, Any] | None,
) -> None:
    decisions = _perceptual_decisions(review_document)
    if not decisions or not edge_evidence:
        return

    preview_union = UnionFind(len(frame))
    for left, right in edge_evidence:
        preview_union.union(left, right)
    preview_components = [
        members
        for members in preview_union.components().values()
        if len(members) > 1
    ]
    component_by_id = {
        _stable_group_id("DUPLICATE", frame.loc[members, "sample_id"]): members
        for members in preview_components
    }
    for duplicate_group_id, decision in decisions.items():
        if str(decision.get("action")) != "reject_candidate":
            continue
        members = component_by_id.get(duplicate_group_id)
        if members is None:
            continue
        member_sample_ids = set(frame.loc[members, "sample_id"].astype(str))
        reviewed_sample_ids = {
            str(value)
            for value in decision.get("reviewed_sample_ids", [])
        }
        if reviewed_sample_ids != member_sample_ids:
            raise ValueError(
                f"Reviewed sample IDs for {duplicate_group_id} must equal "
                f"{sorted(member_sample_ids)}"
            )
        member_set = set(members)
        for edge in list(edge_evidence):
            if edge[0] not in member_set or edge[1] not in member_set:
                continue
            edge_evidence[edge].discard("perceptual")
            if not edge_evidence[edge]:
                del edge_evidence[edge]


def _apply_exact_conflict_decision(
    frame: pd.DataFrame,
    members: list[int],
    *,
    duplicate_group_id: str,
    decision: dict[str, Any] | None,
) -> None:
    member_by_sample_id = {
        str(frame.at[index, "sample_id"]): index
        for index in members
    }
    action = "exclude_all" if decision is None else str(decision["action"])
    if action == "exclude_all":
        frame.loc[members, "deduplication_status"] = "exact_label_conflict"
        for index in members:
            if bool(frame.at[index, "include"]):
                frame.at[index, "include"] = False
                frame.at[index, "exclusion_reason"] = (
                    "exact_duplicate_label_conflict"
                )
        return
    if action != "keep_reviewed_canonical":
        raise ValueError(
            f"Unsupported exact-conflict action {action!r} for "
            f"{duplicate_group_id}"
        )

    canonical_sample_id = str(decision.get("canonical_sample_id", ""))
    if canonical_sample_id not in member_by_sample_id:
        raise ValueError(
            f"Reviewed canonical sample {canonical_sample_id!r} is not in "
            f"{duplicate_group_id}"
        )
    rejected_sample_ids = {
        str(value)
        for value in decision.get("rejected_sample_ids", [])
    }
    expected_rejected = set(member_by_sample_id) - {canonical_sample_id}
    if rejected_sample_ids != expected_rejected:
        raise ValueError(
            f"Rejected sample IDs for {duplicate_group_id} must equal "
            f"{sorted(expected_rejected)}"
        )

    canonical_index = member_by_sample_id[canonical_sample_id]
    frame.at[canonical_index, "deduplication_status"] = "canonical"
    for sample_id in sorted(rejected_sample_ids):
        index = member_by_sample_id[sample_id]
        frame.at[index, "deduplication_status"] = "redundant_exact"
        if bool(frame.at[index, "include"]):
            frame.at[index, "include"] = False
        frame.at[index, "exclusion_reason"] = (
            "exact_duplicate_rejected_label_association"
        )


def _canonical_priority(
    row: pd.Series,
    dataset_priority: dict[str, int],
) -> tuple[Any, ...]:
    evidence_priority = {
        "pathology": 0,
        "clinical_consensus": 1,
        "dermatologist_review": 2,
        "dermatologist_differential": 3,
        "atlas_label": 4,
        "self_reported": 5,
        "derived": 6,
        "unknown": 7,
    }
    return (
        not bool(row["include"]),
        dataset_priority.get(str(row["dataset_id"]), 999),
        evidence_priority.get(str(row["diagnosis_basis"]), 999),
        str(row["sample_id"]),
    )


def _dataset_priority(policy: dict[str, Any]) -> dict[str, int]:
    roles = policy["dataset_roles"]
    ordered = [
        *roles["taxonomy_contributors"],
        *roles["external_evaluation_only"],
        *roles["excluded_from_coverage"].keys(),
    ]
    return {
        dataset_id: index
        for index, dataset_id in enumerate(ordered)
    }


def _source_lineage_values(
    row: pd.Series,
    keys: Iterable[str],
) -> set[str]:
    try:
        metadata = json.loads(str(row["source_metadata"]))
    except (TypeError, ValueError, json.JSONDecodeError):
        return set()
    values: set[str] = set()
    for key in keys:
        value = metadata.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        normalized, _ = urldefrag(value.strip())
        values.add(f"{key}:{normalized}")
    return values


def _stable_group_id(prefix: str, values: Iterable[Any]) -> str:
    serialized = "\n".join(sorted(str(value) for value in values))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def _pair_report(
    frame: pd.DataFrame,
    edge_evidence: dict[tuple[int, int], set[str]],
) -> pd.DataFrame:
    columns = [
        "duplicate_group_id",
        "match_evidence",
        "sample_id_a",
        "dataset_id_a",
        "group_id_a",
        "disease_id_a",
        "include_after_a",
        "sample_id_b",
        "dataset_id_b",
        "group_id_b",
        "disease_id_b",
        "include_after_b",
        "image_sha256_equal",
        "perceptual_hamming_distance",
        "cross_dataset",
        "cross_source_group",
        "label_agreement",
        "requires_review",
    ]
    records: list[dict[str, Any]] = []
    for (left, right), evidence in sorted(edge_evidence.items()):
        left_row = frame.loc[left]
        right_row = frame.loc[right]
        left_disease = _nullable_string(left_row["disease_id"])
        right_disease = _nullable_string(right_row["disease_id"])
        records.append(
            {
                "duplicate_group_id": left_row["duplicate_group_id"],
                "match_evidence": "+".join(sorted(evidence)),
                "sample_id_a": left_row["sample_id"],
                "dataset_id_a": left_row["dataset_id"],
                "group_id_a": left_row["group_id"],
                "disease_id_a": left_disease,
                "include_after_a": bool(left_row["include"]),
                "sample_id_b": right_row["sample_id"],
                "dataset_id_b": right_row["dataset_id"],
                "group_id_b": right_row["group_id"],
                "disease_id_b": right_disease,
                "include_after_b": bool(right_row["include"]),
                "image_sha256_equal": (
                    left_row["image_sha256"] == right_row["image_sha256"]
                ),
                "perceptual_hamming_distance": hamming_distance(
                    str(left_row["perceptual_hash"]),
                    str(right_row["perceptual_hash"]),
                ),
                "cross_dataset": (
                    left_row["dataset_id"] != right_row["dataset_id"]
                ),
                "cross_source_group": (
                    left_row["group_id"] != right_row["group_id"]
                ),
                "label_agreement": (
                    left_disease is not None
                    and left_disease == right_disease
                ),
                "requires_review": bool(
                    evidence.intersection({"perceptual", "source_lineage"})
                    or (
                        left_disease is not None
                        and right_disease is not None
                        and left_disease != right_disease
                    )
                ),
            }
        )
    return pd.DataFrame(records, columns=columns)


def _group_report(
    frame: pd.DataFrame,
    components: list[list[int]],
    edge_evidence: dict[tuple[int, int], set[str]],
) -> pd.DataFrame:
    columns = [
        "duplicate_group_id",
        "duplicate_match_type",
        "sample_count",
        "source_group_count",
        "leakage_group_count",
        "dataset_count",
        "dataset_ids",
        "disease_ids",
        "included_before_count",
        "included_after_count",
        "cross_dataset",
        "label_conflict",
        "requires_review",
    ]
    records: list[dict[str, Any]] = []
    for members in components:
        scoped = frame.loc[members]
        evidence = {
            item
            for (left, right), values in edge_evidence.items()
            if left in members and right in members
            for item in values
        }
        disease_ids = sorted(
            {
                str(value)
                for value in scoped["disease_id"].dropna()
            }
        )
        dataset_ids = sorted(set(scoped["dataset_id"].astype(str)))
        records.append(
            {
                "duplicate_group_id": scoped["duplicate_group_id"].iloc[0],
                "duplicate_match_type": scoped[
                    "duplicate_match_type"
                ].iloc[0],
                "sample_count": len(scoped),
                "source_group_count": int(scoped["group_id"].nunique()),
                "leakage_group_count": int(
                    scoped["leakage_group_id"].nunique()
                ),
                "dataset_count": len(dataset_ids),
                "dataset_ids": "|".join(dataset_ids),
                "disease_ids": "|".join(disease_ids),
                "included_before_count": int(
                    scoped["_include_before_deduplication"].sum()
                ),
                "included_after_count": int(scoped["include"].sum()),
                "cross_dataset": len(dataset_ids) > 1,
                "label_conflict": len(disease_ids) > 1,
                "requires_review": bool(
                    evidence.intersection({"perceptual", "source_lineage"})
                    or len(disease_ids) > 1
                ),
            }
        )
    return pd.DataFrame(records, columns=columns).sort_values(
        ["requires_review", "cross_dataset", "sample_count"],
        ascending=[False, False, False],
        ignore_index=True,
    )


def _summary_report(
    frame: pd.DataFrame,
    groups: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "dataset_id",
        "sample_count",
        "source_group_count",
        "leakage_group_count",
        "hashed_sample_count",
        "duplicate_sample_count",
        "exact_canonical_count",
        "exact_redundant_count",
        "exact_label_conflict_count",
        "perceptual_candidate_count",
        "source_lineage_candidate_count",
        "excluded_by_deduplication_count",
        "duplicate_group_count",
        "cross_dataset_duplicate_sample_count",
    ]
    records: list[dict[str, Any]] = []
    scopes = [
        *[
            (dataset_id, frame[frame["dataset_id"] == dataset_id])
            for dataset_id in sorted(frame["dataset_id"].unique())
        ],
        ("all_datasets", frame),
    ]
    cross_dataset_ids = set(
        groups.loc[groups["cross_dataset"], "duplicate_group_id"]
    )
    for dataset_id, scoped in scopes:
        statuses = scoped["deduplication_status"]
        records.append(
            {
                "dataset_id": dataset_id,
                "sample_count": len(scoped),
                "source_group_count": int(scoped["group_id"].nunique()),
                "leakage_group_count": int(
                    scoped["leakage_group_id"].nunique()
                ),
                "hashed_sample_count": int(
                    scoped["image_sha256"].notna().sum()
                ),
                "duplicate_sample_count": int(
                    scoped["duplicate_group_id"].notna().sum()
                ),
                "exact_canonical_count": int(statuses.eq("canonical").sum()),
                "exact_redundant_count": int(
                    statuses.eq("redundant_exact").sum()
                ),
                "exact_label_conflict_count": int(
                    statuses.eq("exact_label_conflict").sum()
                ),
                "perceptual_candidate_count": int(
                    statuses.eq("perceptual_candidate").sum()
                ),
                "source_lineage_candidate_count": int(
                    statuses.eq("source_lineage_candidate").sum()
                ),
                "excluded_by_deduplication_count": int(
                    (
                        scoped["_include_before_deduplication"]
                        & ~scoped["include"]
                    ).sum()
                ),
                "duplicate_group_count": int(
                    scoped["duplicate_group_id"].dropna().nunique()
                ),
                "cross_dataset_duplicate_sample_count": int(
                    scoped["duplicate_group_id"].isin(
                        cross_dataset_ids
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(records, columns=columns)


def _perceptual_review_queue(
    pairs: pd.DataFrame,
    policy: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "review_priority",
        "priority_reason",
        "review_status",
        "duplicate_group_id",
        "match_evidence",
        "sample_id_a",
        "dataset_id_a",
        "disease_id_a",
        "sample_id_b",
        "dataset_id_b",
        "disease_id_b",
        "perceptual_hamming_distance",
        "cross_dataset",
        "cross_source_group",
        "label_agreement",
    ]
    if pairs.empty:
        return pd.DataFrame(columns=columns)

    contributor_ids = set(
        policy["dataset_roles"]["taxonomy_contributors"]
    )
    external_ids = set(
        policy["dataset_roles"]["external_evaluation_only"]
    )
    records: list[dict[str, Any]] = []
    for row in pairs.to_dict(orient="records"):
        evidence = str(row["match_evidence"]).split("+")
        if not {"perceptual", "source_lineage"}.intersection(evidence):
            continue
        both_labels_present = (
            _nullable_string(row["disease_id_a"]) is not None
            and _nullable_string(row["disease_id_b"]) is not None
        )
        internal_external = {
            row["dataset_id_a"],
            row["dataset_id_b"],
        }.intersection(contributor_ids) and {
            row["dataset_id_a"],
            row["dataset_id_b"],
        }.intersection(external_ids)
        distance = int(row["perceptual_hamming_distance"])
        if both_labels_present and not bool(row["label_agreement"]):
            priority, reason = 1, "mapped_label_conflict"
        elif bool(row["cross_dataset"]) and internal_external:
            priority, reason = 2, "internal_external_overlap"
        elif bool(row["cross_dataset"]):
            priority, reason = 3, "cross_dataset_overlap"
        elif distance == 0:
            priority, reason = 4, "perceptual_distance_0"
        elif distance == 2:
            priority, reason = 5, "perceptual_distance_2"
        elif distance <= 4:
            priority, reason = 6, "perceptual_distance_4"
        else:
            priority, reason = 7, "source_lineage_candidate"
        records.append(
            {
                "review_priority": priority,
                "priority_reason": reason,
                "review_status": "pending",
                "duplicate_group_id": row["duplicate_group_id"],
                "match_evidence": row["match_evidence"],
                "sample_id_a": row["sample_id_a"],
                "dataset_id_a": row["dataset_id_a"],
                "disease_id_a": row["disease_id_a"],
                "sample_id_b": row["sample_id_b"],
                "dataset_id_b": row["dataset_id_b"],
                "disease_id_b": row["disease_id_b"],
                "perceptual_hamming_distance": distance,
                "cross_dataset": bool(row["cross_dataset"]),
                "cross_source_group": bool(row["cross_source_group"]),
                "label_agreement": bool(row["label_agreement"]),
            }
        )
    return pd.DataFrame(records, columns=columns).sort_values(
        [
            "review_priority",
            "perceptual_hamming_distance",
            "duplicate_group_id",
            "sample_id_a",
            "sample_id_b",
        ],
        ignore_index=True,
    )


def _write_updated_manifests(
    frame: pd.DataFrame,
    *,
    manifest_paths: dict[str, Path],
) -> None:
    update_columns = [
        "image_sha256",
        "perceptual_hash",
        "perceptual_hash_algorithm",
        "duplicate_group_id",
        "duplicate_match_type",
        "deduplication_status",
        "leakage_group_id",
        "include",
        "exclusion_reason",
    ]
    for dataset_id, path in manifest_paths.items():
        table = pq.read_table(path)
        scoped = frame[frame["dataset_id"] == dataset_id].reset_index(
            drop=True
        )
        if len(scoped) != table.num_rows:
            raise ValueError(
                f"Deduplication row-count mismatch for {dataset_id}"
            )
        table_sample_ids = table.column("sample_id").to_pylist()
        if table_sample_ids != scoped["sample_id"].tolist():
            raise ValueError(
                f"Deduplication row-order mismatch for {dataset_id}"
            )
        for column in update_columns:
            index = table.schema.get_field_index(column)
            field = MANIFEST_ARROW_SCHEMA.field(column)
            values = pa.array(
                scoped[column].where(scoped[column].notna(), None),
                type=field.type,
            )
            table = table.set_column(index, field, values)
        temporary_path = path.with_suffix(".deduplicated.parquet")
        pq.write_table(table, temporary_path, compression="zstd")
        temporary_path.replace(path)


def _nullable_string(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    return str(value)
