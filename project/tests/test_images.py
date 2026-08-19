"""Image encoding tests."""

from pathlib import Path

import pytest
from PIL import Image

from project.teacher.utils.images import encode_image_data_url


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
