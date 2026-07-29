"""Unit tests for image fingerprinting and duplicate decisions."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from PIL import Image
import yaml

from src.data_pipeline.deduplication import (
    analyze_duplicate_frame,
    compute_fingerprint,
    hamming_distance,
)


ROOT = Path(__file__).resolve().parents[1]


class FingerprintTests(unittest.TestCase):
    def test_perceptual_hash_is_stable_across_image_encoding(self) -> None:
        pixels = np.zeros((64, 64, 3), dtype=np.uint8)
        pixels[:32, :32] = [220, 30, 30]
        pixels[:32, 32:] = [40, 40, 200]
        pixels[32:, 32:] = [20, 180, 80]
        image = Image.fromarray(pixels)

        png = BytesIO()
        jpeg = BytesIO()
        image.save(png, format="PNG")
        image.save(jpeg, format="JPEG", quality=88)

        png_fingerprint = compute_fingerprint(png.getvalue())
        jpeg_fingerprint = compute_fingerprint(jpeg.getvalue())

        self.assertNotEqual(
            png_fingerprint.image_sha256,
            jpeg_fingerprint.image_sha256,
        )
        self.assertLessEqual(
            hamming_distance(
                png_fingerprint.perceptual_hash,
                jpeg_fingerprint.perceptual_hash,
            ),
            4,
        )


class DuplicateDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        document = yaml.safe_load(
            (ROOT / "configs/datasets/disease_inclusion.yaml").read_text()
        )
        cls.policy = document["disease_inclusion"] | {
            key: value
            for key, value in document.items()
            if key != "disease_inclusion"
        }

    def test_exact_duplicate_keeps_one_canonical_row(self) -> None:
        frame = _analysis_frame(
            disease_ids=["D014", "D014", "D018"],
            hashes=["same", "same", "different"],
            perceptual_hashes=[
                "0000000000000000",
                "0000000000000000",
                "ffffffffffffffff",
            ],
        )
        result = analyze_duplicate_frame(frame, policy=self.policy)["frame"]

        self.assertEqual(result.at[0, "deduplication_status"], "canonical")
        self.assertEqual(
            result.at[1, "deduplication_status"],
            "redundant_exact",
        )
        self.assertTrue(result.at[0, "include"])
        self.assertFalse(result.at[1, "include"])
        self.assertEqual(
            result.at[1, "exclusion_reason"],
            "exact_duplicate_redundant",
        )
        self.assertEqual(
            result.at[0, "leakage_group_id"],
            result.at[1, "leakage_group_id"],
        )

    def test_exact_label_conflict_excludes_both_rows(self) -> None:
        frame = _analysis_frame(
            disease_ids=["D014", "D018"],
            hashes=["same", "same"],
            perceptual_hashes=[
                "0000000000000000",
                "0000000000000000",
            ],
        )
        result = analyze_duplicate_frame(frame, policy=self.policy)["frame"]

        self.assertTrue(
            result["deduplication_status"].eq(
                "exact_label_conflict"
            ).all()
        )
        self.assertFalse(result["include"].any())
        self.assertTrue(
            result["exclusion_reason"].eq(
                "exact_duplicate_label_conflict"
            ).all()
        )

    def test_reviewed_exact_conflict_keeps_supported_canonical(self) -> None:
        frame = _analysis_frame(
            disease_ids=["D014", "D018"],
            hashes=["same", "same"],
            perceptual_hashes=[
                "0000000000000000",
                "0000000000000000",
            ],
        )
        preview = analyze_duplicate_frame(
            frame,
            policy=self.policy,
        )["frame"]
        duplicate_group_id = preview.at[0, "duplicate_group_id"]
        review = {
            "exact_conflict_decisions": [
                {
                    "duplicate_group_id": duplicate_group_id,
                    "action": "keep_reviewed_canonical",
                    "canonical_sample_id": "SAMPLE_0",
                    "rejected_sample_ids": ["SAMPLE_1"],
                }
            ]
        }

        result = analyze_duplicate_frame(
            frame,
            policy=self.policy,
            review_document=review,
        )["frame"]

        self.assertTrue(result.at[0, "include"])
        self.assertEqual(result.at[0, "deduplication_status"], "canonical")
        self.assertFalse(result.at[1, "include"])
        self.assertEqual(
            result.at[1, "exclusion_reason"],
            "exact_duplicate_rejected_label_association",
        )

    def test_rejected_perceptual_candidate_is_not_grouped(self) -> None:
        frame = _analysis_frame(
            disease_ids=["D014", "D018"],
            hashes=["left", "right"],
            perceptual_hashes=[
                "0000000000000000",
                "0000000000000000",
            ],
        )
        preview = analyze_duplicate_frame(
            frame,
            policy=self.policy,
        )["frame"]
        duplicate_group_id = preview.at[0, "duplicate_group_id"]
        review = {
            "perceptual_decisions": [
                {
                    "duplicate_group_id": duplicate_group_id,
                    "action": "reject_candidate",
                    "reviewed_sample_ids": ["SAMPLE_0", "SAMPLE_1"],
                }
            ]
        }

        result = analyze_duplicate_frame(
            frame,
            policy=self.policy,
            review_document=review,
        )

        self.assertTrue(result["frame"]["duplicate_group_id"].isna().all())
        self.assertNotEqual(
            result["frame"].at[0, "leakage_group_id"],
            result["frame"].at[1, "leakage_group_id"],
        )
        self.assertTrue(result["pairs"].empty)


def _analysis_frame(
    *,
    disease_ids: list[str],
    hashes: list[str],
    perceptual_hashes: list[str],
) -> pd.DataFrame:
    size = len(disease_ids)
    return pd.DataFrame(
        {
            "sample_id": [f"SAMPLE_{index}" for index in range(size)],
            "dataset_id": ["fitzpatrick17k_c"] * size,
            "group_id": [f"GROUP_{index}" for index in range(size)],
            "disease_id": disease_ids,
            "diagnosis_basis": ["atlas_label"] * size,
            "include": [True] * size,
            "exclusion_reason": [None] * size,
            "image_sha256": hashes,
            "perceptual_hash": perceptual_hashes,
            "source_metadata": ["{}"] * size,
            "_include_before_deduplication": [True] * size,
        }
    )


if __name__ == "__main__":
    unittest.main()
