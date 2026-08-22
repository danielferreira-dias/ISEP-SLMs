"""E4 on-policy distillation (OPD) availability contract.

E4 is intentionally a plan, not a trainer implementation.  The intended
starting point is a frozen, selected E3 SFT checkpoint.  Execution must remain
disabled until the privileged teacher-prompt construction and multimodal
distillation protocol have been implemented and validated.
"""

from __future__ import annotations

from typing import NoReturn

from project.post_training._availability import (
    StageAvailability,
    StageNotImplementedError,
)

OPD_STAGE = StageAvailability(
    stage_id="e4_opd",
    experiment_id="E4",
    method="OPD",
    implemented=False,
    required_parent_stages=("e3_sft_selected",),
    optional_parent_stages=(),
    planned_framework="trl.experimental.gold.GOLDTrainer",
    description=(
        "On-policy multimodal distillation from student-generated trajectories, "
        "using a ground-truth-conditioned teacher prompt that remains hidden "
        "from the student."
    ),
)

# A descriptive alias is convenient for status and manifest code.
OPD_AVAILABILITY = OPD_STAGE


def run(*_args: object, **_kwargs: object) -> NoReturn:
    """Refuse execution because the E4 OPD trainer is not implemented."""

    OPD_STAGE.require_implemented()
    raise AssertionError("unreachable")


__all__ = [
    "OPD_AVAILABILITY",
    "OPD_STAGE",
    "StageNotImplementedError",
    "run",
]
