"""Image encoding tests."""

from pathlib import Path

import pytest
from PIL import Image

from project.teacher.utils.images import (
    encode_image_data_url,
    encode_pil_image_data_url,
    prepare_image,
    prepare_pil_image,
)


def test_encode_png_returns_jpeg_data_url(tmp_path: Path) -> None:
    path = tmp_path / "lesion.png"
    Image.new("RGB", (2000, 100), "red").save(path)
    url = encode_image_data_url(path, max_side=160)
    assert url.startswith("data:image/jpeg;base64,")
    assert "heic" not in url


def test_encode_heic_raises(tmp_path: Path) -> None:
    path = tmp_path / "shot.heic"
    path.write_bytes(b"not-an-image")
    with pytest.raises(ValueError, match="HEIC"):
        encode_image_data_url(path)


def test_missing_image_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        encode_image_data_url(tmp_path / "missing.jpg")


def test_encode_pil_image_data_url() -> None:
    image = Image.new("RGB", (64, 32), "blue")
    url = encode_pil_image_data_url(image, max_side=32)
    assert url.startswith("data:image/jpeg;base64,")


def test_prepared_image_records_deterministic_hash_and_dimensions() -> None:
    image = Image.new("RGB", (64, 32), "blue")
    first = prepare_pil_image(image, max_side=32)
    second = prepare_pil_image(image, max_side=32)
    assert first.info == second.info
    assert first.info.output_width == 32
    assert first.info.output_height == 16
    assert first.info.jpeg_quality == 95


def test_prepare_image_applies_exif_orientation(tmp_path: Path) -> None:
    path = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (20, 10), "red")
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, exif=exif)

    prepared = prepare_image(path)

    assert prepared.info.exif_transposed is True
    assert prepared.info.source_width == 10
    assert prepared.info.source_height == 20
