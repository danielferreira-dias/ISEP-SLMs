"""Build non-sensitive, hash-addressed provenance for teacher generations."""

import hashlib
import json
from datetime import UTC, datetime
from uuid import uuid4

from project.teacher.client import TeacherResponse
from project.teacher.schemas import GenerationProvenance
from project.teacher.teacher import TeacherModel


def generation_provenance(
    teacher: TeacherModel,
    stage_key: str,
    *,
    response: TeacherResponse | None = None,
) -> GenerationProvenance:
    """Identify the exact model, prompt template, schema, seed, and attempt."""
    stage = teacher.stage(stage_key)
    prompt_bytes = stage.prompt.source_path.read_bytes()
    schema_bytes = json.dumps(
        stage.json_schema.schema,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return GenerationProvenance(
        attempt_id=str(uuid4()),
        created_at=datetime.now(UTC),
        provider=teacher.provider,
        teacher_name=teacher.name,
        teacher_model=teacher.model.id,
        seed=teacher.generation.seed,
        max_output_tokens=teacher.generation.max_tokens,
        reasoning_effort=teacher.reasoning.effort,
        reasoning_excluded=teacher.reasoning.exclude,
        transport_retry_max_attempts=(
            teacher.retry.max_attempts if teacher.retry is not None else None
        ),
        transport_retry_status_codes=(
            teacher.retry.retryable_status_codes
            if teacher.retry is not None
            else None
        ),
        prompt_sha256=hashlib.sha256(prompt_bytes).hexdigest(),
        schema_sha256=hashlib.sha256(schema_bytes).hexdigest(),
        finish_reason=response.finish_reason if response is not None else None,
        native_finish_reason=(
            response.native_finish_reason if response is not None else None
        ),
    )
