"""CPU-only tests for deterministic SkinCAP observation extraction."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from src.data_pipeline.isep_distill_caption import build_caption_release
from src.train.e2.skincap import (
    BoundaryKind,
    RejectionReason,
    SkinCapTransformPolicy,
    audit_skincap_observations,
    transform_caption,
)


class SkinCapTransformTests(unittest.TestCase):
    def test_materializer_requires_explicit_written_permission_attestation(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            self.assertRaises(PermissionError),
        ):
            build_caption_release(
                Path(temporary),
                authorization_attested=False,
            )

    def test_keeps_visual_prefix_and_removes_diagnosis(self) -> None:
        result = transform_caption(
            "A pale pink patch has a raised edge and central ulcer. "
            "The diagnosis was basal cell carcinoma.",
            "basal-cell-carcinoma",
        )

        self.assertTrue(result.accepted)
        self.assertEqual(
            result.observation_text,
            "A pale pink patch has a raised edge and central ulcer.",
        )
        self.assertEqual(result.boundary_kind, BoundaryKind.DIAGNOSTIC)
        self.assertIn("basal cell carcinoma", result.removed_suffix.lower())

    def test_removes_testing_recommendation_after_observation(self) -> None:
        result = transform_caption(
            "Well-defined white patches are visible on the lower leg. "
            "Dermoscopy is recommended to confirm the diagnosis.",
            "vitiligo",
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.boundary_kind, BoundaryKind.TESTING)
        self.assertNotIn("dermoscopy", result.observation_text.lower())

    def test_gold_diagnosis_at_start_is_rejected(self) -> None:
        result = transform_caption(
            "Psoriasis is characterized by sharply demarcated scaly plaques.",
            "psoriasis",
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.boundary_kind, BoundaryKind.GOLD_DIAGNOSIS)
        self.assertIn(RejectionReason.EMPTY_OBSERVATION, result.rejection_reasons)

    def test_observation_only_caption_is_accepted_unchanged(self) -> None:
        caption = "Several grouped erythematous papules are visible on the cheek."
        result = transform_caption(caption, "rosacea")

        self.assertTrue(result.accepted)
        self.assertEqual(result.observation_text, caption)
        self.assertEqual(result.removed_suffix, "")
        self.assertEqual(result.boundary_kind, BoundaryKind.NONE)

    def test_short_prefix_is_rejected(self) -> None:
        result = transform_caption(
            "Blue papule. Diagnosis of melanoma.",
            "melanoma",
        )

        self.assertFalse(result.accepted)
        self.assertIn(RejectionReason.TOO_FEW_WORDS, result.rejection_reasons)
        self.assertIn(RejectionReason.TOO_FEW_CHARACTERS, result.rejection_reasons)

    def test_reordered_gold_tokens_form_a_boundary(self) -> None:
        result = transform_caption(
            "An acral lesion is visible. Lentiginous melanoma is suspected.",
            "melanoma-acral-lentiginous",
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.observation_text, "An acral lesion is visible.")
        self.assertEqual(result.boundary_kind, BoundaryKind.GOLD_DIAGNOSIS)

    def test_transform_is_byte_stable(self) -> None:
        policy = SkinCapTransformPolicy()
        first = transform_caption(
            "  A red   plaque is visible. Diagnosis: eczema. ",
            "eczema",
            policy,
        )
        second = transform_caption(
            "A red plaque is visible. Diagnosis: eczema.",
            "eczema",
            policy,
        )

        self.assertEqual(first, second)


@unittest.skipUnless(
    Path("configs/datasets/skincap/data/skincap_v240715.xlsx").is_file(),
    "gated SkinCAP snapshot is not available",
)
class SkinCapLocalAuditTests(unittest.TestCase):
    def test_pinned_snapshot_produces_only_aggregate_expected_counts(self) -> None:
        report = audit_skincap_observations()
        record = report.as_record()

        self.assertEqual(report.downloaded_rows, 4_000)
        self.assertEqual(report.technical_candidate_rows, 3_318)
        self.assertEqual(report.accepted_observation_rows, 3_250)
        self.assertEqual(report.rejected_observation_rows, 68)
        self.assertFalse(report.derivatives_materialized)
        self.assertNotIn("caption", record)
        self.assertNotIn("diagnosis", record)


@unittest.skipUnless(
    Path(
        "data/training/ISEPDistillDataset/releases/"
        "isep_distill_dataset_v0.4.1/release.json"
    ).is_file(),
    "authorized SkinCAP caption release is not materialized",
)
class SkinCapMaterializedReleaseTests(unittest.TestCase):
    def test_v0_4_1_counts_and_cross_task_groups_are_frozen(self) -> None:
        root = Path("data/training/ISEPDistillDataset")
        manifest = json.loads(
            (root / "releases/isep_distill_dataset_v0.4.1/release.json").read_text()
        )
        assignments = pq.read_table(
            root / "releases/isep_distill_dataset_v0.4.1/caption_assignments.parquet"
        ).to_pandas()

        self.assertEqual(manifest["configs"]["caption"]["rows"], 3_250)
        self.assertEqual(manifest["configs"]["caption"]["sft_train"], 2_767)
        self.assertEqual(manifest["configs"]["caption"]["sft_dev"], 483)
        train = set(
            assignments.loc[assignments["split"].eq("sft_train"), "leakage_group_id"]
        )
        dev = set(
            assignments.loc[assignments["split"].eq("sft_dev"), "leakage_group_id"]
        )
        self.assertFalse(train & dev)

        frozen = pq.read_table(
            root / "metadata/morphology_assignments.parquet",
            columns=["leakage_group_id", "split"],
        ).to_pandas()
        caption_splits = dict(
            zip(assignments["leakage_group_id"], assignments["split"], strict=True)
        )
        for group_id, split in frozen.itertuples(index=False, name=None):
            if group_id in caption_splits:
                self.assertEqual(caption_splits[group_id], split)

    def test_caption_schema_does_not_expose_diagnosis_or_raw_caption(self) -> None:
        path = next(
            Path("data/training/ISEPDistillDataset/data/caption_v0_4_1").glob(
                "sft_train-*.parquet"
            )
        )
        names = set(pq.ParquetFile(path).schema_arrow.names)

        self.assertNotIn("gold_diagnosis", names)
        self.assertNotIn("disease_id", names)
        self.assertNotIn("raw_caption", names)
        self.assertNotIn("removed_suffix", names)
        self.assertIn("source_caption_sha256", names)


if __name__ == "__main__":
    unittest.main()
