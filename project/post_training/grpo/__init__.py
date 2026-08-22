"""E5 GRPO availability contract.

E5 is intentionally a plan, not a trainer implementation.  It requires a
frozen, selected E3 SFT checkpoint; a selected E4 OPD checkpoint may be used as
an optional parent for the corresponding ablation.  Execution stays disabled
until the multimodal reward protocol has been audited and frozen.
"""

from __future__ import annotations

from typing import NoReturn

from project.post_training._availability import (
    StageAvailability,
    StageNotImplementedError,
)

GRPO_STAGE = StageAvailability(
    stage_id="e5_grpo",
    experiment_id="E5",
    method="GRPO",
    implemented=False,
    required_parent_stages=("e3_sft_selected",),
    optional_parent_stages=("e4_opd_selected",),
    planned_framework="trl.GRPOTrainer",
    description=(
        "Multimodal policy optimization with audited diagnosis, morphology, "
        "differential-diagnosis, hierarchy, and output-validity rewards."
    ),
)

# A descriptive alias is convenient for status and manifest code.
GRPO_AVAILABILITY = GRPO_STAGE


def run(*_args: object, **_kwargs: object) -> NoReturn:
    """Refuse execution because the E5 GRPO trainer is not implemented."""

    GRPO_STAGE.require_implemented()
    raise AssertionError("unreachable")


__all__ = [
    "GRPO_AVAILABILITY",
    "GRPO_STAGE",
    "StageNotImplementedError",
    "run",
]
