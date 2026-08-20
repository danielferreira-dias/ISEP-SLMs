"""Encode a local clinical image as an OpenRouter data URL."""

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

MAX_IMAGE_SIDE = 1600
JPEG_QUALITY = 85

_HEIC_SUFFIXES = {".heic", ".heif"}


def encode_image_data_url(path: Path, *, max_side: int = MAX_IMAGE_SIDE) -> str:
    """Load a file, cap the long edge, and return a JPEG data URL.

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
            return encode_pil_image_data_url(image, max_side=max_side)
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unsupported or corrupt image: {path}") from exc


def encode_pil_image_data_url(
    image: Image.Image,
    *,
    max_side: int = MAX_IMAGE_SIDE,
) -> str:
    """Cap a PIL image's long edge and return a JPEG data URL.

    Args:
        image: In-memory image, including Hub ``datasets`` decoded columns.
        max_side: Longest edge in pixels after resize.

    Returns:
        A ``data:image/jpeg;base64,...`` string.
    """
    rgb = image.convert("RGB")
    resized = _fit_long_edge(rgb, max_side=max_side)
    encoded = base64.b64encode(_to_jpeg_bytes(resized)).decode("ascii")
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
