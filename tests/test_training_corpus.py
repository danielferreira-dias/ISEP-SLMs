"""Tests for training-only clinical-image selection."""

from __future__ import annotations

import unittest

from src.data_pipeline.training_corpus import (
    _derm1m_archive_member,
    _derm1m_archive_for,
    _derm1m_group_id,
    classify_derm1m_clinical_image,
)


class Derm1MClinicalFilterTests(unittest.TestCase):
    def test_forum_image_is_included(self) -> None:
        self.assertEqual(
            classify_derm1m_clinical_image(
                source="IIYI_chinese",
                filename="IIYI/123_1.png",
                caption="Patient asks about an itchy plaque.",
            ),
            (True, "forum_user_clinical_photo"),
        )

    def test_explicit_non_clinical_modality_wins(self) -> None:
        include, rule = classify_derm1m_clinical_image(
            source="IIYI_chinese",
            filename="IIYI/123_1.png",
            caption="Histopathology slide with H&E staining.",
        )
        self.assertFalse(include)
        self.assertEqual(rule, "explicit_non_clinical_modality")

    def test_public_dermoscopy_is_excluded(self) -> None:
        include, _ = classify_derm1m_clinical_image(
            source="public",
            filename="public/ISIC_1.jpg",
            caption="This is a dermoscopic image of melanoma.",
        )
        self.assertFalse(include)

    def test_explicit_clinical_image_is_included(self) -> None:
        include, rule = classify_derm1m_clinical_image(
            source="pubmed_english",
            filename="pubmed/case.png",
            caption="Clinical photograph of a patient with a scaly plaque.",
        )
        self.assertTrue(include)
        self.assertEqual(rule, "explicit_clinical_language")

    def test_archive_and_group_are_stable(self) -> None:
        self.assertEqual(_derm1m_archive_for("youtube/a_frame_1.jpg"), "youtube.zip")
        self.assertEqual(
            _derm1m_archive_member("youtube/a_frame_1.jpg"),
            "a_frame_1.jpg",
        )
        self.assertEqual(
            _derm1m_group_id("youtube/a_frame_1.jpg"),
            _derm1m_group_id("youtube/a_frame_2.jpg"),
        )


if __name__ == "__main__":
    unittest.main()
