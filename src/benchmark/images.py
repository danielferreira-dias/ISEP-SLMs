"""Deterministic image preparation for comparable multimodal benchmarks."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageOps

from src.config.benchmarks import ImagePreprocessingConfig


class ImagePreprocessingError(ValueError):
    """Raised when an image cannot satisfy the benchmark transport profile."""


def prepare_benchmark_image(
    image_bytes: bytes,
    config: ImagePreprocessingConfig,
) -> bytes:
    """Normalize one image to the benchmark's RGB JPEG byte budget.

    The byte budget keeps base64 data URLs below strict API-gateway limits.
    All models receive this same deterministic representation, including
    local models, so transport constraints cannot create model-specific
    image inputs.
    """

    try:
        with Image.open(BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
    except Exception as exc:
        raise ImagePreprocessingError(
            f"Could not decode image: {type(exc).__name__}: {exc}"
        ) from exc

    image.thumbnail(
        (config.max_edge_pixels, config.max_edge_pixels),
        Image.Resampling.LANCZOS,
    )
    while True:
        encoded = _encode_with_quality_budget(image, config)
        if encoded is not None:
            return encoded

        longest_edge = max(image.size)
        if longest_edge <= config.minimum_edge_pixels:
            raise ImagePreprocessingError(
                "Image cannot satisfy max_encoded_bytes without reducing "
                "below minimum_edge_pixels"
            )
        scale = max(
            config.minimum_edge_pixels / longest_edge,
            0.85,
        )
        resized = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        if resized == image.size:
            raise ImagePreprocessingError(
                "Image preprocessing made no progress toward the byte budget"
            )
        image = image.resize(resized, Image.Resampling.LANCZOS)


def _encode_with_quality_budget(
    image: Image.Image,
    config: ImagePreprocessingConfig,
) -> bytes | None:
    for quality in range(
        config.jpeg_quality,
        config.minimum_jpeg_quality - 1,
        -5,
    ):
        buffer = BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=quality,
            optimize=True,
            subsampling=2,
        )
        encoded = buffer.getvalue()
        if len(encoded) <= config.max_encoded_bytes:
            return encoded
    return None
