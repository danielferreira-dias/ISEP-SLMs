"""Tests for deterministic benchmark image preprocessing."""

from __future__ import annotations

from io import BytesIO
import unittest

import numpy as np
from PIL import Image

from src.benchmark.images import prepare_benchmark_image
from src.config.benchmarks import ImagePreprocessingConfig


class BenchmarkImagePreprocessingTests(unittest.TestCase):
    def test_noisy_image_is_rgb_jpeg_within_configured_limits(self) -> None:
        pixels = np.random.default_rng(42).integers(
            0,
            256,
            size=(900, 1200, 3),
            dtype=np.uint8,
        )
        source = BytesIO()
        Image.fromarray(pixels, mode="RGB").save(source, format="PNG")
        config = ImagePreprocessingConfig(
            profile="test",
            max_edge_pixels=768,
            max_encoded_bytes=45_000,
            jpeg_quality=85,
            minimum_jpeg_quality=40,
            minimum_edge_pixels=224,
        )

        first = prepare_benchmark_image(source.getvalue(), config)
        second = prepare_benchmark_image(source.getvalue(), config)

        self.assertEqual(first, second)
        self.assertLessEqual(len(first), config.max_encoded_bytes)
        with Image.open(BytesIO(first)) as result:
            self.assertEqual(result.format, "JPEG")
            self.assertEqual(result.mode, "RGB")
            self.assertLessEqual(max(result.size), config.max_edge_pixels)

