"""Encode a local clinical image as an OpenRouter data URL."""

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

MAX_IMAGE_SIDE = 1600
JPEG_QUALITY = 85

_HEIC_SUFFIXES = {".heic", ".heif"}


def encode_image_data_url(path: Path, *, max_side: int = MAX_IMAGE_SIDE) -> str:
    """Load an image, cap the long edge, and return a JPEG data URL.

    PNG/JPEG/WebP are decoded with Pillow. HEIC is rejected until a decoder
    dependency is added. The data URL is never logged by this function.

    Args:
        path: Local image file.
        max_side: Longest edge in pixels after resize.

    Returns:
        A ``data:image/jpeg;base64,...`` string.

    Raises:
        FileNotFoundError: If ``path`` is missing.
        ValueError: If the file cannot be decoded or is HEIC.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = path.suffix.lower()
    if suffix in _HEIC_SUFFIXES:
        raise ValueError(f"HEIC is not supported without an extra decoder: {path}")

    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            resized = _fit_long_edge(rgb, max_side=max_side)
            jpeg_bytes = _to_jpeg_bytes(resized)
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unsupported or corrupt image: {path}") from exc

    encoded = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _fit_long_edge(image: Image.Image, *, max_side: int) -> Image.Image:
    """Shrink the image so neither side exceeds ``max_side``."""
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image

    scale = max_side / longest
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _to_jpeg_bytes(image: Image.Image) -> bytes:
    """Encode an RGB image as JPEG bytes."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buffer.getvalue()
