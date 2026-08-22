"""Dataset-generation and materialization entry points for E3.

Student post-training lives under :mod:`project.post_training`; the
``project.pipeline`` namespace owns only the offline teacher-data pipeline.
"""

from project.pipeline.materialize_sft import (
    SCHEMA_VERSION,
    MaterializationResult,
    MaterializationSource,
    MaterializedSFTRow,
    SFTTask,
    materialize_multitask_rows,
)

__all__ = [
    "SCHEMA_VERSION",
    "MaterializationResult",
    "MaterializationSource",
    "MaterializedSFTRow",
    "SFTTask",
    "materialize_multitask_rows",
]
