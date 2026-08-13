"""Qualitative visual-attribution tools for student checkpoint comparison."""

from src.vision_analysis.occlusion import (
    OcclusionResult,
    build_occluded_images,
    compute_score_drop_map,
)

__all__ = [
    "OcclusionResult",
    "build_occluded_images",
    "compute_score_drop_map",
]
