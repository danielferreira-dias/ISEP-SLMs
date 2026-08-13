"""Deterministic checkpoint selection for scientific runs."""

from __future__ import annotations

import math

from .models import CheckpointScore


def select_best_checkpoint(
    checkpoints: tuple[CheckpointScore, ...],
) -> CheckpointScore:
    """Select by macro-F1, balanced accuracy, loss, then earlier epoch."""

    if not checkpoints:
        raise ValueError("At least one checkpoint score is required")
    for checkpoint in checkpoints:
        if not checkpoint.checkpoint_id:
            raise ValueError("checkpoint_id cannot be blank")
        values = (
            checkpoint.metrics.macro_f1,
            checkpoint.metrics.balanced_accuracy,
            checkpoint.eval_loss,
            checkpoint.epoch,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                f"Checkpoint {checkpoint.checkpoint_id!r} has non-finite "
                "selection values"
            )
    return max(
        checkpoints,
        key=lambda item: (
            item.metrics.macro_f1,
            item.metrics.balanced_accuracy,
            -item.eval_loss,
            -item.epoch,
        ),
    )
