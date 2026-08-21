"""Deterministically prepare clinical images for teacher requests."""

import base64
import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError

from project.teacher.schemas import ImagePreprocessingInfo

MAX_IMAGE_SIDE = 1600
JPEG_QUALITY = 95
JPEG_SUBSAMPLING = 0

_HEIC_SUFFIXES = {".heic", ".heif"}
_EXIF_ORIENTATION = 274


@dataclass(slots=True, kw_only=True, frozen=True)
class PreparedImage:
    """Exact image payload plus its non-sensitive preprocessing manifest."""

    data_url: str
    info: ImagePreprocessingInfo


def prepare_image(path: Path, *, max_side: int = MAX_IMAGE_SIDE) -> PreparedImage:
    """Load and prepare a local image, rejecting unsupported or corrupt files."""
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() in _HEIC_SUFFIXES:
        raise ValueError(f"HEIC is not supported without an extra decoder: {path}")

    try:
        with Image.open(path) as image:
            return prepare_pil_image(image, max_side=max_side)
    except UnidentifiedImageError as exc:
        raise ValueError(f"Unsupported or corrupt image: {path}") from exc


def prepare_pil_image(
    image: Image.Image,
    *,
    max_side: int = MAX_IMAGE_SIDE,
) -> PreparedImage:
    """Orient, colour-normalize, resize, encode, and hash an in-memory image."""
    if max_side <= 0:
        raise ValueError("max_side must be positive")

    orientation = image.getexif().get(_EXIF_ORIENTATION, 1)
    exif_transposed = orientation not in (None, 1)
    oriented = ImageOps.exif_transpose(image)
    source_width, source_height = oriented.size
    source_mode = oriented.mode
    source_hash = _pixel_sha256(oriented)
    icc_profile = oriented.info.get("icc_profile")
    icc_profile_present = isinstance(icc_profile, bytes) and bool(icc_profile)

    rgb = _convert_to_srgb(oriented, icc_profile=icc_profile)
    resized = _fit_long_edge(rgb, max_side=max_side)
    jpeg_bytes = _to_jpeg_bytes(resized)
    output_hash = hashlib.sha256(jpeg_bytes).hexdigest()
    encoded = base64.b64encode(jpeg_bytes).decode("ascii")

    return PreparedImage(
        data_url=f"data:image/jpeg;base64,{encoded}",
        info=ImagePreprocessingInfo(
            source_pixel_sha256=source_hash,
            source_width=source_width,
            source_height=source_height,
            source_mode=source_mode,
            exif_transposed=exif_transposed,
            icc_profile_present=icc_profile_present,
            output_sha256=output_hash,
            output_width=resized.width,
            output_height=resized.height,
            output_media_type="image/jpeg",
            max_side=max_side,
            jpeg_quality=JPEG_QUALITY,
        ),
    )


def encode_image_data_url(path: Path, *, max_side: int = MAX_IMAGE_SIDE) -> str:
    """Compatibility wrapper returning only the local image data URL."""
    return prepare_image(path, max_side=max_side).data_url


def encode_pil_image_data_url(
    image: Image.Image,
    *,
    max_side: int = MAX_IMAGE_SIDE,
) -> str:
    """Compatibility wrapper returning only the in-memory image data URL."""
    return prepare_pil_image(image, max_side=max_side).data_url


def _pixel_sha256(image: Image.Image) -> str:
    """Hash oriented source pixels together with mode and dimensions."""
    digest = hashlib.sha256()
    digest.update(image.mode.encode("utf-8"))
    digest.update(f"{image.width}x{image.height}".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _convert_to_srgb(image: Image.Image, *, icc_profile: object) -> Image.Image:
    """Convert an embedded ICC profile to sRGB or use explicit RGB conversion."""
    if not isinstance(icc_profile, bytes) or not icc_profile:
        return image.convert("RGB")

    try:
        source_profile = ImageCms.ImageCmsProfile(BytesIO(icc_profile))
        target_profile = ImageCms.createProfile("sRGB")
        converted = ImageCms.profileToProfile(
            image,
            source_profile,
            target_profile,
            outputMode="RGB",
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(
            "Image contains an invalid or unsupported ICC profile"
        ) from exc
    if converted is None:
        raise ValueError("ICC conversion returned no image")
    return converted


def _fit_long_edge(image: Image.Image, *, max_side: int) -> Image.Image:
    """Shrink the image so neither side exceeds ``max_side``."""
    longest = max(image.size)
    if longest <= max_side:
        return image.copy()

    scale = max_side / longest
    new_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _to_jpeg_bytes(image: Image.Image) -> bytes:
    """Encode an RGB image using the frozen high-fidelity JPEG profile."""
    buffer = BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=JPEG_QUALITY,
        subsampling=JPEG_SUBSAMPLING,
        optimize=True,
    )
    return buffer.getvalue()
