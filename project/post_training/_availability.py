"""Shared project contracts for post-training stages not implemented yet.

The descriptors in this module are deliberately inert.  They let callers
inspect the intended experiment lineage without accidentally treating a
placeholder as a runnable trainer.
"""

from __future__ import annotations

from dataclasses import dataclass


class StageNotImplementedError(RuntimeError):
    """Raised when code tries to execute a planned post-training stage."""

    def __init__(self, stage: StageAvailability) -> None:
        required = ", ".join(stage.required_parent_stages) or "none"
        optional = ", ".join(stage.optional_parent_stages) or "none"
        super().__init__(
            f"{stage.experiment_id}/{stage.method} ({stage.stage_id}) is planned "
            "but not implemented; no training was started. "
            f"Required parent stages: {required}. "
            f"Optional parent stages: {optional}. "
            f"Planned framework: {stage.planned_framework}."
        )
        self.stage = stage


@dataclass(frozen=True, slots=True)
class StageAvailability:
    """Immutable description of the implementation state of a future stage."""

    stage_id: str
    experiment_id: str
    method: str
    implemented: bool
    required_parent_stages: tuple[str, ...]
    optional_parent_stages: tuple[str, ...]
    planned_framework: str
    description: str

    def require_implemented(self) -> None:
        """Fail closed unless the stage has a real implementation."""

        if not self.implemented:
            raise StageNotImplementedError(self)
