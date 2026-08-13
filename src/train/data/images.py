"""Deterministic image decoding and normalization for visual SFT."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps


def preprocess_image(
    image: Image.Image | bytes | Path,
    *,
    max_edge_pixels: int = 512,
) -> Image.Image:
    """Normalize one image while preserving clinical geometry.

    EXIF orientation is applied before conversion to RGB.  Images larger than
    ``max_edge_pixels`` are downsampled with Lanczos while preserving aspect
    ratio; smaller images are never upscaled.

    Args:
        image: Decoded PIL image, encoded bytes, or local image path.
        max_edge_pixels: Maximum width or height in the returned image.

    Returns:
        Fully loaded RGB image detached from its input stream.

    Raises:
        ValueError: If ``max_edge_pixels`` is not positive.
        PIL.UnidentifiedImageError: If encoded input is not a supported image.
    """

    if max_edge_pixels <= 0:
        raise ValueError("max_edge_pixels must be positive")
    if isinstance(image, Image.Image):
        decoded = image.copy()
    elif isinstance(image, bytes):
        with Image.open(BytesIO(image)) as opened:
            decoded = opened.copy()
    else:
        with Image.open(image) as opened:
            decoded = opened.copy()

    normalized = ImageOps.exif_transpose(decoded).convert("RGB")
    if max(normalized.size) > max_edge_pixels:
        normalized.thumbnail(
            (max_edge_pixels, max_edge_pixels),
            Image.Resampling.LANCZOS,
        )
    normalized.load()
    return normalized
