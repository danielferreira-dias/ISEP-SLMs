"""Materialize validated Stage A/B outputs into multitask VLM SFT rows.

The teacher-generation JSONL files remain the private audit trail. This module
creates the narrower trainer-visible release: one conversation per task, with
the image stored separately and only an image marker plus prompt in messages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast

from project.dataset.dataset import DistillDataset
from project.teacher.schemas import (
    RecordStatus,
    ResponsePolicy,
    StageAFileRow,
    StageBFileRow,
)
from project.teacher.teacher import PROJECT_ROOT
from project.teacher.utils.jsonl import load_stage_a_rows, load_stage_b_rows
from project.teacher.validate import validate_stage_b

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

SCHEMA_VERSION = "e3_multitask_sft_v1"

MORPHOLOGY_PROMPT = (
    "Assess only what is visible in the dermatology image. Return one compact "
    "JSON object containing image_assessment, dominant_visual_pattern, "
    "observations, and not_assessable_features. Do not diagnose a disease or "
    "invent history, palpation, tests, or metadata.\n\n/no_think"
)

CAPTION_PROMPT = (
    "Describe only the visible dermatological findings in one complete short "
    "clinical sentence. Do not provide a diagnosis, differential diagnosis, "
    "testing, management, prognosis, or advice.\n\n/no_think"
)

CLINICAL_ASSESSMENT_PROMPT = (
    "Assess the dermatology image using only visible evidence. Describe the "
    "visible morphology, provide the most likely diagnosis, and briefly "
    "distinguish it from plausible alternatives. If the image is not evaluable, "
    "explain why and request a better image without guessing a diagnosis. Do not "
    "invent history, tests, or non-visible findings.\n\n/no_think"
)

_WORD_BOUNDARY = r"(?<![a-z0-9]){name}(?![a-z0-9])"


class SFTTask(StrEnum):
    """Trainer-visible behaviors in the E3 multitask release."""

    DIAGNOSIS = "diagnosis"
    MORPHOLOGY = "morphology"
    CAPTION = "caption"
    GROUNDED_DIFFERENTIAL = "grounded_differential"
    REQUEST_NEW_IMAGE = "request_new_image"


_TASK_IDS = {
    SFTTask.DIAGNOSIS: "e3_diagnosis_replay_v1",
    SFTTask.MORPHOLOGY: "e3_stage_a_morphology_v1",
    SFTTask.CAPTION: "e3_stage_a_caption_v1",
    SFTTask.GROUNDED_DIFFERENTIAL: "e3_grounded_differential_v1",
    SFTTask.REQUEST_NEW_IMAGE: "e3_request_new_image_v1",
}

_TARGET_SOURCES = {
    SFTTask.DIAGNOSIS: "human_gold_diagnosis",
    SFTTask.MORPHOLOGY: "teacher_stage_a_structured",
    SFTTask.CAPTION: "teacher_stage_a_clinical_caption",
    SFTTask.GROUNDED_DIFFERENTIAL: "teacher_stage_b_clinical_reasoning",
    SFTTask.REQUEST_NEW_IMAGE: "teacher_stage_b_clinical_reasoning",
}


class _ArrowDataset(Protocol):
    """Minimal Hugging Face Dataset surface used by the source adapter."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> object: ...

    def cast_column(self, column: str, feature: object) -> _ArrowDataset: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class MaterializationSource:
    """One immutable diagnosis row supplying image, split, and human gold."""

    sample_id: str
    image: object
    source_ref: str
    split: str
    leakage_group_id: str
    disease_id: str
    gold_diagnosis: str
    source_dataset: str
    source_sample_id: str
    image_sha256: str
    diagnosis_prompt: str
    diagnosis_prompt_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "sample_id",
            "source_ref",
            "split",
            "leakage_group_id",
            "disease_id",
            "gold_diagnosis",
            "source_dataset",
            "source_sample_id",
            "diagnosis_prompt",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Materialization source {name} must be non-empty")
        if self.split not in {"sft_train", "sft_dev"}:
            raise ValueError(f"Unsupported materialization split: {self.split}")
        _require_digest(self.image_sha256, "image_sha256")
        _require_digest(self.diagnosis_prompt_sha256, "diagnosis_prompt_sha256")
        expected = _sha256_text(self.diagnosis_prompt)
        if self.diagnosis_prompt_sha256 != expected:
            raise ValueError("Diagnosis prompt SHA-256 differs from its text")


@dataclass(frozen=True, slots=True, kw_only=True)
class MaterializedSFTRow:
    """One task-isolated conversation ready for HF/TRL serialization."""

    image: object
    row_id: str
    sample_id: str
    source_ref: str
    split: str
    leakage_group_id: str
    disease_id: str
    gold_diagnosis: str
    source_dataset: str
    source_sample_id: str
    image_sha256: str
    task: SFTTask
    task_id: str
    prompt: str
    prompt_sha256: str
    target_text: str
    target_sha256: str
    target_source: str
    stage_a_attempt_id: str
    stage_b_attempt_id: str
    messages: tuple[dict[str, object], ...]

    def as_record(self) -> dict[str, object]:
        """Return a Hugging Face Dataset-compatible mapping."""
        return {
            "image": self.image,
            "row_id": self.row_id,
            "sample_id": self.sample_id,
            "source_ref": self.source_ref,
            "split": self.split,
            "leakage_group_id": self.leakage_group_id,
            "disease_id": self.disease_id,
            "gold_diagnosis": self.gold_diagnosis,
            "source_dataset": self.source_dataset,
            "source_sample_id": self.source_sample_id,
            "image_sha256": self.image_sha256,
            "task": self.task.value,
            "task_id": self.task_id,
            "prompt": self.prompt,
            "prompt_sha256": self.prompt_sha256,
            "target_text": self.target_text,
            "target_sha256": self.target_sha256,
            "target_source": self.target_source,
            "stage_a_attempt_id": self.stage_a_attempt_id,
            "stage_b_attempt_id": self.stage_b_attempt_id,
            "schema_version": SCHEMA_VERSION,
            "quality_status": "accepted",
            "messages": list(self.messages),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class MaterializationResult:
    """Rows plus coverage statistics for one split materialization."""

    rows: tuple[MaterializedSFTRow, ...]
    source_samples: int
    stage_a_ok: int
    stage_b_ok: int
    task_counts: dict[str, int]
    source_ids_without_stage_a: tuple[str, ...]
    source_ids_without_stage_b: tuple[str, ...]
    stage_b_status_counts: dict[str, int]
    stage_b_rejected_attempts: tuple[StageBRejectedAttempt, ...]
    stage_b_error_attempts: tuple[StageBErrorAttempt, ...]
    stage_b_missing_attempt_ids: tuple[str, ...]
    stage_b_duplicate_attempt_counts: dict[str, int]


@dataclass(frozen=True, slots=True, kw_only=True)
class StageBRejectedAttempt:
    """Audit summary for one Stage B quality-gate rejection."""

    sample_id: str
    attempt_id: str
    reasons: tuple[str, ...]
    anchor_evidence_status: str
    diagnostic_confidence: str
    annotation_conflict: bool

    def as_record(self) -> dict[str, object]:
        """Return the JSON-serializable manifest representation."""
        return {
            "sample_id": self.sample_id,
            "attempt_id": self.attempt_id,
            "reasons": list(self.reasons),
            "anchor_evidence_status": self.anchor_evidence_status,
            "diagnostic_confidence": self.diagnostic_confidence,
            "annotation_conflict": self.annotation_conflict,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class StageBErrorAttempt:
    """Audit summary for one retryable Stage B provider/schema error."""

    sample_id: str
    attempt_id: str
    error: str

    def as_record(self) -> dict[str, object]:
        """Return the JSON-serializable manifest representation."""
        return {
            "sample_id": self.sample_id,
            "attempt_id": self.attempt_id,
            "error": self.error,
        }


def source_from_hub_row(
    raw: Mapping[str, object],
    *,
    repo_id: str,
    revision: str,
    config: str,
    split: str,
) -> MaterializationSource:
    """Validate and adapt one frozen diagnosis config row."""
    sample_id = _required_string(raw, "sample_id")
    row_split = _required_string(raw, "split")
    if row_split != split:
        raise ValueError(
            f"Source row {sample_id} split {row_split!r} differs from {split!r}"
        )
    image_sha256 = _required_string(raw, "image_sha256")
    image = _normalized_image(raw.get("image"), sample_id=sample_id)
    encoded = image.get("bytes")
    if (
        isinstance(encoded, bytes)
        and hashlib.sha256(encoded).hexdigest() != image_sha256
    ):
        raise ValueError(f"Source row {sample_id} image SHA-256 differs")
    return MaterializationSource(
        sample_id=sample_id,
        image=image,
        source_ref=(f"hf://datasets/{repo_id}@{revision}/{config}/{split}/{sample_id}"),
        split=split,
        leakage_group_id=_required_string(raw, "leakage_group_id"),
        disease_id=_required_string(raw, "disease_id"),
        gold_diagnosis=_required_string(raw, "gold_diagnosis"),
        source_dataset=_required_string(raw, "source_dataset"),
        source_sample_id=_required_string(raw, "source_sample_id"),
        image_sha256=image_sha256,
        diagnosis_prompt=_required_string(raw, "prompt"),
        diagnosis_prompt_sha256=_required_string(raw, "prompt_sha256"),
    )


def load_materialization_sources(
    *,
    config: str = "diagnosis",
    split: str = "sft_train",
) -> tuple[MaterializationSource, ...]:
    """Load image bytes and metadata from the pinned private Hub release."""
    try:
        from datasets import Image as DatasetImage  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("The training extra with datasets is required") from exc

    loaded = DistillDataset.load(config=config, split=split)
    table = cast(_ArrowDataset, loaded.get(config, split)).cast_column(
        "image", DatasetImage(decode=False)
    )
    repo = loaded.spec.huggingface
    return tuple(
        source_from_hub_row(
            _mapping(table[index], context=f"source row {index}"),
            repo_id=repo.repo_id,
            revision=repo.revision,
            config=config,
            split=split,
        )
        for index in range(len(table))
    )


def materialize_multitask_rows(
    sources: Iterable[MaterializationSource],
    stage_a_rows: Iterable[StageAFileRow],
    stage_b_rows: Iterable[StageBFileRow],
    *,
    require_complete: bool = True,
) -> MaterializationResult:
    """Expand one source image into independently prompted task rows.

    Diagnosis replay is human-supervised and independent of teacher acceptance.
    Accepted Stage A adds morphology and caption. Accepted Stage B adds exactly
    one clinical-assessment behavior selected by its validated response policy.
    """
    ordered_sources = tuple(sorted(sources, key=lambda item: item.sample_id))
    _require_unique_sources(ordered_sources)
    a_rows = tuple(stage_a_rows)
    b_rows = tuple(stage_b_rows)
    a_ok = _index_first_ok_stage_a(a_rows)
    b_ok = _index_first_ok_stage_b(b_rows)
    a_seen = {row.sample_id for row in a_rows}
    b_seen = {row.sample_id for row in b_rows}
    source_ids = {source.sample_id for source in ordered_sources}
    b_by_id: dict[str, list[StageBFileRow]] = {}
    for row in b_rows:
        if row.sample_id in source_ids:
            b_by_id.setdefault(row.sample_id, []).append(row)

    incomplete_a = tuple(
        source.sample_id for source in ordered_sources if source.sample_id not in a_seen
    )
    incomplete_b = tuple(
        source.sample_id
        for source in ordered_sources
        if source.sample_id in a_ok and source.sample_id not in b_seen
    )
    if require_complete and (incomplete_a or incomplete_b):
        raise ValueError(
            "Generation coverage is incomplete: "
            f"missing Stage A attempts={len(incomplete_a)}, "
            f"missing Stage B attempts after accepted A={len(incomplete_b)}"
        )

    rows: list[MaterializedSFTRow] = []
    missing_a: list[str] = []
    missing_b: list[str] = []
    for source in ordered_sources:
        rows.append(
            _make_row(
                source,
                task=SFTTask.DIAGNOSIS,
                prompt=source.diagnosis_prompt,
                target=source.gold_diagnosis,
            )
        )
        stage_a = a_ok.get(source.sample_id)
        if stage_a is None:
            missing_a.append(source.sample_id)
            continue
        _validate_stage_a_source(stage_a, source)
        assert stage_a.morphology is not None
        morphology_target = _morphology_target(stage_a)
        if _phrase_in_text(source.gold_diagnosis, morphology_target):
            raise ValueError(f"Stage A morphology leaks gold for {source.sample_id}")
        if _phrase_in_text(source.gold_diagnosis, stage_a.morphology.clinical_caption):
            raise ValueError(f"Stage A caption leaks gold for {source.sample_id}")
        rows.extend(
            (
                _make_row(
                    source,
                    task=SFTTask.MORPHOLOGY,
                    prompt=MORPHOLOGY_PROMPT,
                    target=morphology_target,
                    stage_a=stage_a,
                ),
                _make_row(
                    source,
                    task=SFTTask.CAPTION,
                    prompt=CAPTION_PROMPT,
                    target=stage_a.morphology.clinical_caption,
                    stage_a=stage_a,
                ),
            )
        )

        stage_b = b_ok.get(source.sample_id)
        if stage_b is None:
            missing_b.append(source.sample_id)
            continue
        _validate_stage_b_source(stage_a, stage_b, source)
        assert stage_b.reasoning is not None
        task = (
            SFTTask.GROUNDED_DIFFERENTIAL
            if stage_b.reasoning.response_policy is ResponsePolicy.ANSWER_DIFFERENTIAL
            else SFTTask.REQUEST_NEW_IMAGE
        )
        rows.append(
            _make_row(
                source,
                task=task,
                prompt=CLINICAL_ASSESSMENT_PROMPT,
                target=stage_b.reasoning.clinical_reasoning,
                stage_a=stage_a,
                stage_b=stage_b,
            )
        )

    row_ids = tuple(row.row_id for row in rows)
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("Materialized row_id values are not unique")
    task_counts = Counter(row.task.value for row in rows)
    stage_b_status_counts = Counter(
        _stage_b_source_status(
            source.sample_id,
            stage_a_ok=source.sample_id in a_ok,
            attempts=b_by_id.get(source.sample_id, ()),
        )
        for source in ordered_sources
    )
    coverage_statuses = (
        "ok",
        "rejected",
        "error",
        "missing_attempt",
        "not_eligible_stage_a",
    )
    for status in coverage_statuses:
        stage_b_status_counts.setdefault(status, 0)
    rejected_attempts = tuple(
        _rejected_attempt(row)
        for row in b_rows
        if row.sample_id in source_ids and row.status is RecordStatus.REJECTED
    )
    error_attempts = tuple(
        _error_attempt(row)
        for row in b_rows
        if row.sample_id in source_ids and row.status is RecordStatus.ERROR
    )
    duplicate_attempt_counts = {
        sample_id: len(attempts)
        for sample_id, attempts in sorted(b_by_id.items())
        if len(attempts) > 1
    }
    return MaterializationResult(
        rows=tuple(rows),
        source_samples=len(ordered_sources),
        stage_a_ok=sum(source.sample_id in a_ok for source in ordered_sources),
        stage_b_ok=sum(source.sample_id in b_ok for source in ordered_sources),
        task_counts=dict(sorted(task_counts.items())),
        source_ids_without_stage_a=tuple(missing_a),
        source_ids_without_stage_b=tuple(missing_b),
        stage_b_status_counts=dict(sorted(stage_b_status_counts.items())),
        stage_b_rejected_attempts=rejected_attempts,
        stage_b_error_attempts=error_attempts,
        stage_b_missing_attempt_ids=incomplete_b,
        stage_b_duplicate_attempt_counts=duplicate_attempt_counts,
    )


def _stage_b_source_status(
    sample_id: str,
    *,
    stage_a_ok: bool,
    attempts: Iterable[StageBFileRow],
) -> str:
    """Return final coverage using ok > rejected > error precedence."""
    if not stage_a_ok:
        return "not_eligible_stage_a"
    statuses = {row.status for row in attempts if row.sample_id == sample_id}
    if RecordStatus.OK in statuses:
        return "ok"
    if RecordStatus.REJECTED in statuses:
        return "rejected"
    if RecordStatus.ERROR in statuses:
        return "error"
    return "missing_attempt"


def _rejected_attempt(row: StageBFileRow) -> StageBRejectedAttempt:
    """Project a validated rejected row into compact release audit metadata."""
    assert row.status is RecordStatus.REJECTED
    assert row.reasoning is not None
    return StageBRejectedAttempt(
        sample_id=row.sample_id,
        attempt_id=_attempt_id(row),
        reasons=row.reasons,
        anchor_evidence_status=row.reasoning.anchor_evidence_status.value,
        diagnostic_confidence=row.reasoning.diagnostic_confidence.value,
        annotation_conflict=row.reasoning.annotation_conflict,
    )


def _error_attempt(row: StageBFileRow) -> StageBErrorAttempt:
    """Project a validated error row into compact release audit metadata."""
    assert row.status is RecordStatus.ERROR
    assert row.error is not None
    return StageBErrorAttempt(
        sample_id=row.sample_id,
        attempt_id=_attempt_id(row),
        error=row.error,
    )


def write_multitask_release(
    result: MaterializationResult,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Write one image-aware Parquet and an adjacent integrity manifest."""
    if not result.rows:
        raise ValueError("Cannot write an empty multitask release")
    manifest_path = output_path.with_suffix(".manifest.json")
    if not overwrite and (output_path.exists() or manifest_path.exists()):
        raise FileExistsError(f"Materialized output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=".e3-materialize-", dir=output_path.parent)
    )
    temporary_parquet = temporary_root / output_path.name
    temporary_manifest = temporary_root / manifest_path.name
    try:
        _write_parquet(result.rows, temporary_parquet)
        manifest = _release_manifest(result, temporary_parquet, output_path)
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_parquet, output_path)
        os.replace(temporary_manifest, manifest_path)
        return manifest
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def _make_row(
    source: MaterializationSource,
    *,
    task: SFTTask,
    prompt: str,
    target: str,
    stage_a: StageAFileRow | None = None,
    stage_b: StageBFileRow | None = None,
) -> MaterializedSFTRow:
    if not target.strip():
        raise ValueError(f"Empty target for {source.sample_id}::{task.value}")
    prompt_hash = _sha256_text(prompt)
    target_hash = _sha256_text(target)
    if task is SFTTask.DIAGNOSIS and prompt_hash != source.diagnosis_prompt_sha256:
        raise ValueError(f"Diagnosis prompt drift for {source.sample_id}")
    messages: tuple[dict[str, object], ...] = (
        {
            "role": "user",
            "content": [
                {"type": "image", "text": ""},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": target}],
        },
    )
    return MaterializedSFTRow(
        image=source.image,
        row_id=f"{source.sample_id}::{task.value}",
        sample_id=source.sample_id,
        source_ref=source.source_ref,
        split=source.split,
        leakage_group_id=source.leakage_group_id,
        disease_id=source.disease_id,
        gold_diagnosis=source.gold_diagnosis,
        source_dataset=source.source_dataset,
        source_sample_id=source.source_sample_id,
        image_sha256=source.image_sha256,
        task=task,
        task_id=_TASK_IDS[task],
        prompt=prompt,
        prompt_sha256=prompt_hash,
        target_text=target,
        target_sha256=target_hash,
        target_source=_TARGET_SOURCES[task],
        stage_a_attempt_id=_attempt_id(stage_a),
        stage_b_attempt_id=_attempt_id(stage_b),
        messages=messages,
    )


def _morphology_target(stage_a: StageAFileRow) -> str:
    assert stage_a.morphology is not None
    payload = stage_a.morphology.model_dump(
        mode="json",
        exclude={"clinical_caption"},
    )
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _validate_stage_a_source(
    stage_a: StageAFileRow,
    source: MaterializationSource,
) -> None:
    if stage_a.sample_id != source.sample_id or stage_a.morphology is None:
        raise ValueError(f"Stage A identity mismatch for {source.sample_id}")
    if stage_a.image_path != source.source_ref:
        raise ValueError(f"Stage A image reference mismatch for {source.sample_id}")


def _validate_stage_b_source(
    stage_a: StageAFileRow,
    stage_b: StageBFileRow,
    source: MaterializationSource,
) -> None:
    if stage_b.sample_id != source.sample_id:
        raise ValueError(f"Stage B identity mismatch for {source.sample_id}")
    if stage_b.stage_a_sample_id != stage_a.sample_id:
        raise ValueError(f"Stage B Stage-A reference mismatch for {source.sample_id}")
    if stage_b.gold_diagnosis != source.gold_diagnosis:
        raise ValueError(f"Stage B gold mismatch for {source.sample_id}")
    if stage_b.image_path != source.source_ref:
        raise ValueError(f"Stage B image reference mismatch for {source.sample_id}")
    if stage_b.reasoning is None or stage_a.morphology is None:
        raise ValueError(
            f"Stage B accepted payload is incomplete for {source.sample_id}"
        )
    check = validate_stage_b(
        stage_a.morphology,
        stage_b.reasoning,
        source.gold_diagnosis,
    )
    if not check.ok:
        raise ValueError(
            f"Stage B no longer passes validation for {source.sample_id}: "
            f"{', '.join(check.reasons)}"
        )
    if (
        stage_a.image_preprocessing
        and stage_b.image_preprocessing
        and stage_a.image_preprocessing.output_sha256
        != stage_b.image_preprocessing.output_sha256
    ):
        raise ValueError(f"Stage A/B image mismatch for {source.sample_id}")


def _index_first_ok_stage_a(
    rows: Iterable[StageAFileRow],
) -> dict[str, StageAFileRow]:
    indexed: dict[str, StageAFileRow] = {}
    for row in rows:
        if row.status is RecordStatus.OK and row.sample_id not in indexed:
            indexed[row.sample_id] = row
    return indexed


def _index_first_ok_stage_b(
    rows: Iterable[StageBFileRow],
) -> dict[str, StageBFileRow]:
    indexed: dict[str, StageBFileRow] = {}
    for row in rows:
        if row.status is RecordStatus.OK and row.sample_id not in indexed:
            indexed[row.sample_id] = row
    return indexed


def _attempt_id(row: StageAFileRow | StageBFileRow | None) -> str:
    return row.provenance.attempt_id if row and row.provenance else ""


def _require_unique_sources(sources: tuple[MaterializationSource, ...]) -> None:
    ids = tuple(source.sample_id for source in sources)
    if len(ids) != len(set(ids)):
        raise ValueError("Materialization sources contain duplicate sample_id values")
    group_splits: dict[str, str] = {}
    for source in sources:
        previous = group_splits.setdefault(source.leakage_group_id, source.split)
        if previous != source.split:
            raise ValueError(
                f"Leakage group {source.leakage_group_id} crosses release splits"
            )


def _normalized_image(value: object, *, sample_id: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Source row {sample_id} image is not an image mapping")
    encoded = value.get("bytes")
    path = value.get("path")
    if isinstance(encoded, (bytes, bytearray)):
        return {
            "bytes": bytes(encoded),
            "path": str(path) if isinstance(path, str) else None,
        }
    if isinstance(path, str) and path:
        return {"bytes": None, "path": path}
    raise ValueError(f"Source row {sample_id} image has neither bytes nor a path")


def _required_string(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Source row {key} must be a non-empty string")
    return value.strip()


def _mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} is not a mapping")
    return cast(Mapping[str, object], value)


def _require_digest(value: str, name: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _phrase_in_text(phrase: str, text: str) -> bool:
    pattern = _WORD_BOUNDARY.format(name=re.escape(phrase.strip().casefold()))
    return re.search(pattern, text.casefold()) is not None


def _write_parquet(rows: tuple[MaterializedSFTRow, ...], path: Path) -> None:
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("The training extra with datasets is required") from exc

    def generate_records() -> Iterable[dict[str, object]]:
        for row in rows:
            yield row.as_record()

    # ``from_list`` first expands every image-bearing record in RAM. E3 has up
    # to four task rows per source image, which can exhaust memory before Arrow
    # starts writing. The generator path builds a memory-mapped Arrow cache in
    # bounded batches inside the same temporary release directory.
    dataset = Dataset.from_generator(
        generate_records,
        features=_materialized_features(),
        cache_dir=str(path.parent / ".hf-cache"),
        keep_in_memory=False,
    )
    dataset.to_parquet(path, batch_size=128, compression="zstd")


def _materialized_features() -> object:
    try:
        from datasets import Features, Image, List, Value
    except ImportError as exc:
        raise RuntimeError("The training extra with datasets is required") from exc
    return Features(
        {
            "image": Image(decode=True),
            "row_id": Value("string"),
            "sample_id": Value("string"),
            "source_ref": Value("string"),
            "split": Value("string"),
            "leakage_group_id": Value("string"),
            "disease_id": Value("string"),
            "gold_diagnosis": Value("string"),
            "source_dataset": Value("string"),
            "source_sample_id": Value("string"),
            "image_sha256": Value("string"),
            "task": Value("string"),
            "task_id": Value("string"),
            "prompt": Value("string"),
            "prompt_sha256": Value("string"),
            "target_text": Value("string"),
            "target_sha256": Value("string"),
            "target_source": Value("string"),
            "stage_a_attempt_id": Value("string"),
            "stage_b_attempt_id": Value("string"),
            "schema_version": Value("string"),
            "quality_status": Value("string"),
            "messages": List(
                {
                    "role": Value("string"),
                    "content": List(
                        {
                            "type": Value("string"),
                            "text": Value("string"),
                        }
                    ),
                }
            ),
        }
    )


def _release_manifest(
    result: MaterializationResult,
    parquet_path: Path,
    installed_path: Path,
) -> dict[str, object]:
    splits = {row.split for row in result.rows}
    if len(splits) != 1:
        raise ValueError("One materialized file cannot mix splits")
    return {
        "schema_version": SCHEMA_VERSION,
        "split": next(iter(splits)),
        "output": str(installed_path),
        "rows": len(result.rows),
        "source_samples": result.source_samples,
        "stage_a_ok": result.stage_a_ok,
        "stage_b_ok": result.stage_b_ok,
        "task_counts": result.task_counts,
        "source_ids_without_stage_a": list(result.source_ids_without_stage_a),
        "source_ids_without_stage_b": list(result.source_ids_without_stage_b),
        "stage_b_coverage": result.stage_b_status_counts,
        "stage_b_rejected_attempts": [
            attempt.as_record() for attempt in result.stage_b_rejected_attempts
        ],
        "stage_b_error_attempts": [
            attempt.as_record() for attempt in result.stage_b_error_attempts
        ],
        "stage_b_missing_attempt_ids": list(result.stage_b_missing_attempt_ids),
        "stage_b_duplicate_attempt_counts": result.stage_b_duplicate_attempt_counts,
        "bytes": parquet_path.stat().st_size,
        "sha256": _sha256_file(parquet_path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the E3 multitask materializer CLI."""
    parser = argparse.ArgumentParser(
        description="Materialize validated Stage A/B into multitask SFT Parquet."
    )
    parser.add_argument(
        "--stage-a",
        type=Path,
        default=PROJECT_ROOT / "data" / "morphology" / "stage_a.jsonl",
    )
    parser.add_argument(
        "--stage-b",
        type=Path,
        default=PROJECT_ROOT / "data" / "reasoning" / "stage_b.jsonl",
    )
    parser.add_argument("--hub-config", default="diagnosis")
    parser.add_argument("--hub-split", default="sft_train")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow missing generation attempts and report reduced task coverage.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing Parquet and manifest explicitly.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load the pinned source release, materialize, and write atomically."""
    args = parse_args(argv)
    output = args.output or (
        PROJECT_ROOT / "data" / "sft" / "e3_multitask" / f"{args.hub_split}.parquet"
    )
    result = materialize_multitask_rows(
        load_materialization_sources(
            config=args.hub_config,
            split=args.hub_split,
        ),
        load_stage_a_rows(args.stage_a),
        load_stage_b_rows(args.stage_b),
        require_complete=not args.allow_partial,
    )
    manifest = write_multitask_release(result, output, overwrite=args.overwrite)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
