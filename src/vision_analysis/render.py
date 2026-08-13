"""Render signed attribution overlays without a plotting dependency."""

from __future__ import annotations

import numpy as np
from PIL import Image


def render_signed_overlay(
    image: Image.Image,
    signed_importance: np.ndarray,
    *,
    alpha: float = 0.52,
) -> Image.Image:
    """Overlay supporting regions in red and suppressing regions in blue."""

    if signed_importance.ndim != 2:
        raise ValueError("signed_importance must be a two-dimensional grid")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie between zero and one")
    values = np.clip(signed_importance.astype(np.float32), -1.0, 1.0)
    positive = np.maximum(values, 0.0)
    negative = np.maximum(-values, 0.0)
    heat = np.stack(
        [
            255.0 * positive,
            80.0 * positive + 100.0 * negative,
            255.0 * negative,
        ],
        axis=-1,
    ).astype(np.uint8)
    mask = (255.0 * np.abs(values)).astype(np.uint8)
    heat_image = Image.fromarray(heat).resize(
        image.size,
        Image.Resampling.BICUBIC,
    )
    mask_image = Image.fromarray(mask).resize(
        image.size,
        Image.Resampling.BICUBIC,
    )
    blended = Image.blend(image.convert("RGB"), heat_image, alpha=alpha)
    return Image.composite(blended, image.convert("RGB"), mask_image)
