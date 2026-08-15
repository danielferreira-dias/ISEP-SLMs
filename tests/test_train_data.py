"""Tests for group-safe split releases and lazy multimodal loading."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from src.train.config import TrainingConfig
from src.train.data import (
    build_lazy_phase_dataset,
    inspect_data_release,
    iter_release_samples,
    load_assignments,
    load_release_frame,
    prepare_data_release,
    preprocess_image,
    validate_source_shards,
)
from src.train.data.dataset import LazyReleaseDataset, _AssignedSample
from src.train.data.taxonomy import load_taxonomy
from src.train.domain import ReleaseSubset
from src.train.phases.label_only import LabelOnlyPhase


class TrainingDataReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.release = self.root / "prepared"
        _write_toy_source(self.source)
        self.config = _toy_config(self.root, self.source, self.release)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_release_is_group_safe_image_free_and_idempotent(self) -> None:
        prepared = prepare_data_release(self.config)
        audit = inspect_data_release(prepared.root)
        assignments = load_assignments(prepared)

        self.assertEqual(audit.source_image_count, 40)
        self.assertEqual(audit.train_image_count, 20)
        self.assertEqual(audit.dev_image_count, 20)
        self.assertEqual(audit.dev_panel_image_count, 4)
        self.assertEqual(audit.group_overlap_count, 0)
        self.assertNotIn("image", assignments.columns)
        self.assertEqual(
            assignments[assignments["is_dev_panel"]]
            .groupby("label")["leakage_group_id"]
            .nunique()
            .to_dict(),
            {"class_a": 2, "class_b": 2},
        )
        repeated = prepare_data_release(self.config)
        self.assertEqual(repeated.audit.assignment_sha256, audit.assignment_sha256)

    def test_changed_split_cannot_reuse_release_identifier(self) -> None:
        prepare_data_release(self.config)
        changed_split = self.config.dataset.split.model_copy(update={"seed": 99})
        changed_data = self.config.dataset.model_copy(update={"split": changed_split})
        changed = self.config.model_copy(update={"dataset": changed_data})
        with self.assertRaises(ValueError):
            prepare_data_release(changed)

    def test_release_views_can_stream_or_load_metadata(self) -> None:
        prepared = prepare_data_release(self.config)
        dev = load_release_frame(
            self.config,
            prepared,
            ReleaseSubset.SFT_DEV,
        )
        first = next(
            iter_release_samples(
                self.config,
                prepared,
                ReleaseSubset.DEV_PANEL,
                batch_size=3,
            )
        )

        self.assertEqual(len(dev), 20)
        self.assertNotIn("image", dev.columns)
        self.assertEqual(first.image.mode, "RGB")
        self.assertEqual(first.image.size, (512, 256))

    def test_lazy_dataset_formats_one_item_without_loading_all_images(self) -> None:
        prepared = prepare_data_release(self.config)
        dataset = build_lazy_phase_dataset(
            self.config,
            prepared,
            ReleaseSubset.DEV_PANEL,
            cache_directory=self.root / "hf-cache",
        )
        record = dataset[0]

        self.assertEqual(len(dataset), 4)
        self.assertIn(record["label"], {"class_a", "class_b"})
        self.assertEqual(len(record["messages"]), 2)
        self.assertEqual(record["phase"], "e1_label")
        self.assertEqual(record["task"], "diagnosis")
        self.assertEqual(record["split"], "dev_panel")
        self.assertEqual(record["annotation_availability"], ["diagnosis"])
        self.assertEqual(record["pixel_count"], 800 * 400)
        self.assertEqual(
            (record["resized_width"], record["resized_height"]), (512, 256)
        )

    def test_image_preprocessing_preserves_aspect_and_does_not_upscale(self) -> None:
        large = preprocess_image(Image.new("L", (1000, 500), 127), max_edge_pixels=512)
        small = preprocess_image(
            Image.new("RGB", (100, 50), "blue"), max_edge_pixels=512
        )
        self.assertEqual(large.mode, "RGB")
        self.assertEqual(large.size, (512, 256))
        self.assertEqual(small.size, (100, 50))

    def test_source_shard_corruption_is_rejected(self) -> None:
        shard = self.source / "data" / "train-00000-of-00001.parquet"
        with shard.open("ab") as handle:
            handle.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "byte count mismatch"):
            validate_source_shards(self.config)

    def test_lazy_dataset_rejects_image_label_index_drift(self) -> None:
        assigned = _AssignedSample(
            sample_id="sample_000",
            leakage_group_id="group_000",
            disease_id="D001",
            label="class_a",
            source="source_0",
        )
        wrong_row = {
            "sample_id": "sample_001",
            "leakage_group_id": "group_001",
            "disease_id": "D002",
            "label": "class_b",
            "source": "source_1",
            "image": _png_bytes(),
        }
        dataset = LazyReleaseDataset(
            backing=_FakeArrowDataset((wrong_row,)),
            samples=(assigned,),
            phase=LabelOnlyPhase(load_taxonomy(self.config)),
            max_edge_pixels=512,
            source_root=self.source,
        )
        with self.assertRaisesRegex(ValueError, "Backing row identity mismatch"):
            _ = dataset[0]


def _toy_config(root: Path, source: Path, release: Path) -> TrainingConfig:
    document: dict[str, object] = {
        "schema_version": 1,
        "experiment": {
            "id": "e1_toy",
            "phase": "e1_label",
            "vision_profile": "frozen_vision",
        },
        "dataset": {
            "source_directory": source,
            "source_version": "1.3.0",
            "hub_repo_id": "test/ISEPDermData",
            "hub_revision": "a" * 40,
            "release_id": "toy_e1_v1",
            "release_directory": release,
            "split": {
                "train_ratio": 0.5,
                "dev_ratio": 0.5,
                "seed": 42,
                "secondary_feature_weight": 0.25,
                "panel_groups_per_class": 2,
                "panel_seed": 1042,
            },
            "expected": {
                "image_count": 40,
                "group_count": 40,
                "class_count": 2,
                "source_count": 2,
                "train_image_count": 20,
                "train_group_count": 20,
                "dev_image_count": 20,
                "dev_group_count": 20,
            },
        },
        "model": {},
        "lora": {"finetune_vision_layers": False},
        "trainer": {},
        "project_root": root,
    }
    return TrainingConfig.model_validate(document, strict=True)


def _write_toy_source(root: Path) -> None:
    data_directory = root / "data"
    metadata_directory = root / "metadata"
    data_directory.mkdir(parents=True)
    metadata_directory.mkdir(parents=True)
    image_bytes = _png_bytes()
    rows: list[dict[str, object]] = []
    for index in range(40):
        class_index = index % 2
        rows.append(
            {
                "image": {"bytes": image_bytes, "path": None},
                "source": f"source_{index % 2}",
                "label": f"class_{'a' if class_index == 0 else 'b'}",
                "disease_id": f"D{class_index + 1:03d}",
                "sample_id": f"sample_{index:03d}",
                "source_image_id": f"image_{index:03d}",
                "source_label": f"source label {class_index}",
                "leakage_group_id": f"group_{index:03d}",
                "diagnosis_basis": "pathology",
                "image_sha256": f"{index:064x}",
                "license_id": "research_only",
            }
        )
    image_type = pa.struct(
        [pa.field("bytes", pa.binary()), pa.field("path", pa.string())]
    )
    arrays = {
        key: pa.array(
            [row[key] for row in rows],
            type=image_type if key == "image" else pa.string(),
        )
        for key in rows[0]
    }
    shard = data_directory / "train-00000-of-00001.parquet"
    pq.write_table(pa.table(arrays), shard)
    (root / "release.json").write_text(
        json.dumps(
            {
                "release": {
                    "id": "ISEPDermData",
                    "version": "1.3.0",
                    "image_count": 40,
                    "shards": [
                        {
                            "path": "data/train-00000-of-00001.parquet",
                            "bytes": shard.stat().st_size,
                            "rows": 40,
                            "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    (metadata_directory / "taxonomy.json").write_text(
        json.dumps(
            {
                "taxonomy_id": "toy_dermatology",
                "active_class_count": 2,
                "classes": [
                    {"disease_id": "D001", "label": "class_a"},
                    {"disease_id": "D002", "label": "class_b"},
                ],
            }
        ),
        encoding="utf-8",
    )


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("L", (800, 400), 128).save(buffer, format="PNG")
    return buffer.getvalue()


class _FakeArrowDataset:
    def __init__(self, rows: tuple[dict[str, object], ...]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> object:
        return self._rows[index]

    def select(self, indices: list[int]) -> _FakeArrowDataset:
        return _FakeArrowDataset(tuple(self._rows[index] for index in indices))


if __name__ == "__main__":
    unittest.main()
