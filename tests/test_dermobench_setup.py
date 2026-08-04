"""Tests for preparing the external DermoBench release."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from src.data_pipeline.dermobench import prepare_dermobench


class DermoBenchSetupTests(unittest.TestCase):
    def test_extracts_images_and_indexes_filename_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory)
            annotation_path = release / "task1" / "items.json"
            annotation_path.parent.mkdir(parents=True)
            annotation_path.write_text(
                json.dumps(
                    [
                        {"id": "exact", "image": "ddi/DDI_image/000001.png"},
                        {
                            "id": "normalized",
                            "image": "derm1m/edu/Book_Name_2020_00001_00002.png",
                        },
                        {
                            "id": "suffix",
                            "image": "derm1m/edu/E_itim_2020_00003_00004.png",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            with zipfile.ZipFile(release / "dermobench_release_imgs.zip", "w") as archive:
                archive.writestr("imgs/ddi/DDI_image/000001.png", b"exact")
                archive.writestr(
                    "imgs/derm1m/edu/Book Name (2020)_00001_00002.png",
                    b"normalized",
                )
                archive.writestr(
                    "imgs/derm1m/edu/Egitim (2020)_00003_00004.png",
                    b"suffix",
                )

            manifest = prepare_dermobench(release_root=release, extract=True)

            self.assertEqual(manifest["resolution"], {
                "exact": 1,
                "normalized_filename": 1,
                "suffix_and_similarity": 1,
            })
            index = manifest["image_paths"]
            assert isinstance(index, dict)
            self.assertEqual(
                index["derm1m/edu/E_itim_2020_00003_00004.png"],
                "derm1m/edu/Egitim (2020)_00003_00004.png",
            )
            self.assertEqual(
                (
                    release
                    / "images"
                    / "derm1m/edu/Egitim (2020)_00003_00004.png"
                ).read_bytes(),
                b"suffix",
            )


if __name__ == "__main__":
    unittest.main()
