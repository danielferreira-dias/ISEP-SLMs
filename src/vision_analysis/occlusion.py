"""Model-agnostic patch-occlusion attribution utilities.

Occlusion sensitivity is deliberately the first implemented method. Unlike a
raw attention map, it measures whether replacing a region changes the score of
a pre-specified diagnosis. It is still an attribution heuristic rather than a
clinical explanation, but it provides a useful causal sensitivity check for
the later gradient-based analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True, slots=True)
class OcclusionTile:
    """One rectangular region and its blurred-image perturbation."""

    row: int
    column: int
    box: tuple[int, int, int, int]
    image: Image.Image


@dataclass(frozen=True, slots=True)
class OcclusionResult:
    """Raw and normalized score changes for one target diagnosis."""

    baseline_score: float
    occluded_scores: np.ndarray
    score_drops: np.ndarray
    positive_importance: np.ndarray
    signed_importance: np.ndarray


def build_occluded_images(
    image: Image.Image,
    *,
    grid_size: int,
    blur_radius: float | None = None,
) -> list[OcclusionTile]:
    """Create a deterministic grid of images with one blurred tile each."""

    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    source = image.convert("RGB")
    radius = (
        max(source.size) * 0.08 if blur_radius is None else float(blur_radius)
    )
    if radius <= 0:
        raise ValueError("blur_radius must be positive")
    blurred = source.filter(ImageFilter.GaussianBlur(radius=radius))
    x_edges = np.linspace(0, source.width, grid_size + 1, dtype=int)
    y_edges = np.linspace(0, source.height, grid_size + 1, dtype=int)
    tiles: list[OcclusionTile] = []
    for row in range(grid_size):
        for column in range(grid_size):
            box = (
                int(x_edges[column]),
                int(y_edges[row]),
                int(x_edges[column + 1]),
                int(y_edges[row + 1]),
            )
            perturbed = source.copy()
            perturbed.paste(blurred.crop(box), box)
            tiles.append(
                OcclusionTile(
                    row=row,
                    column=column,
                    box=box,
                    image=perturbed,
                )
            )
    return tiles


def compute_score_drop_map(
    image: Image.Image,
    score: Callable[[Image.Image], float],
    *,
    grid_size: int,
    blur_radius: float | None = None,
) -> OcclusionResult:
    """Measure the target-score change caused by each occluded tile."""

    baseline_score = float(score(image.convert("RGB")))
    scores = np.empty((grid_size, grid_size), dtype=np.float32)
    for tile in build_occluded_images(
        image,
        grid_size=grid_size,
        blur_radius=blur_radius,
    ):
        scores[tile.row, tile.column] = float(score(tile.image))
    drops = baseline_score - scores
    positive = np.maximum(drops, 0.0)
    positive_max = float(positive.max(initial=0.0))
    positive_normalized = (
        positive / positive_max if positive_max > 0 else np.zeros_like(positive)
    )
    signed_max = float(np.abs(drops).max(initial=0.0))
    signed_normalized = (
        drops / signed_max if signed_max > 0 else np.zeros_like(drops)
    )
    return OcclusionResult(
        baseline_score=baseline_score,
        occluded_scores=scores,
        score_drops=drops,
        positive_importance=positive_normalized.astype(np.float32),
        signed_importance=signed_normalized.astype(np.float32),
    )


def iter_tile_records(result: OcclusionResult) -> Iterable[dict[str, float | int]]:
    """Yield JSON-ready tile measurements in row-major order."""

    rows, columns = result.score_drops.shape
    for row in range(rows):
        for column in range(columns):
            yield {
                "row": row,
                "column": column,
                "occluded_score": float(result.occluded_scores[row, column]),
                "score_drop": float(result.score_drops[row, column]),
                "positive_importance": float(
                    result.positive_importance[row, column]
                ),
                "signed_importance": float(
                    result.signed_importance[row, column]
                ),
            }
