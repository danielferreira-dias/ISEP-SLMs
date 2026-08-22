"""Compatibility facade for the E3 SFT dataset materializer.

The implementation lives in :mod:`project.pipeline.materialize_sft` so this
module cannot be mistaken for the student-training runtime. Existing imports
and the historical ``isep-materialize-e3`` entry point remain supported.
"""

from project.pipeline.materialize_sft import (
    CAPTION_PROMPT,
    CLINICAL_ASSESSMENT_PROMPT,
    MORPHOLOGY_PROMPT,
    SCHEMA_VERSION,
    MaterializationResult,
    MaterializationSource,
    MaterializedSFTRow,
    SFTTask,
    StageBErrorAttempt,
    StageBRejectedAttempt,
    load_materialization_sources,
    main,
    materialize_multitask_rows,
    parse_args,
    source_from_hub_row,
    write_multitask_release,
)

__all__ = [
    "CAPTION_PROMPT",
    "CLINICAL_ASSESSMENT_PROMPT",
    "MORPHOLOGY_PROMPT",
    "SCHEMA_VERSION",
    "MaterializationResult",
    "MaterializationSource",
    "MaterializedSFTRow",
    "SFTTask",
    "StageBErrorAttempt",
    "StageBRejectedAttempt",
    "load_materialization_sources",
    "main",
    "materialize_multitask_rows",
    "parse_args",
    "source_from_hub_row",
    "write_multitask_release",
]


if __name__ == "__main__":
    main()
