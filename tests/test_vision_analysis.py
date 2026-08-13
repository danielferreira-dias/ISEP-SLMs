"""Pure tests for the qualitative visual-attribution utilities."""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from src.vision_analysis.occlusion import (
    build_occluded_images,
    compute_score_drop_map,
)
from src.vision_analysis.render import render_signed_overlay


class OcclusionTests(unittest.TestCase):
    def test_grid_covers_image_without_changing_dimensions(self) -> None:
        image = Image.new("RGB", (11, 7), "white")
        tiles = build_occluded_images(image, grid_size=3, blur_radius=2)
        self.assertEqual(len(tiles), 9)
        self.assertEqual(tiles[0].box[:2], (0, 0))
        self.assertEqual(tiles[-1].box[2:], image.size)
        self.assertTrue(all(tile.image.size == image.size for tile in tiles))

    def test_score_drops_and_normalization(self) -> None:
        values = iter([10.0, 8.0, 11.0, 10.0, 6.0])
        result = compute_score_drop_map(
            Image.new("RGB", (8, 8), "white"),
            lambda _image: next(values),
            grid_size=2,
            blur_radius=1,
        )
        np.testing.assert_allclose(
            result.score_drops,
            np.array([[2.0, -1.0], [0.0, 4.0]], dtype=np.float32),
        )
        self.assertEqual(float(result.positive_importance.max()), 1.0)
        self.assertEqual(float(result.signed_importance.min()), -0.25)

    def test_signed_overlay_preserves_image_size(self) -> None:
        image = Image.new("RGB", (40, 20), "gray")
        overlay = render_signed_overlay(
            image,
            np.array([[1.0, -1.0], [0.0, 0.5]], dtype=np.float32),
        )
        self.assertEqual(overlay.size, image.size)
        self.assertEqual(overlay.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
