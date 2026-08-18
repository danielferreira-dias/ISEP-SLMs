"""Private append-only E3 targets and finalized teacher bundles."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from src.train.e3.domain import (
    ResponsePolicy,
    StageATarget,
    StageBTarget,
    StageReviewStatus,
    TeacherGenerationProvenance,
    TeacherGenerationStatus,
    TeacherTargetBundle,
)
from src.train.e3.progress import (
    E3GenerationStage,
    E3ProgressEvent,
)


def _stage_from_json(value: object) -> object:
    return E3GenerationStage(value) if isinstance(value, str) else value


def _review_from_json(value: object) -> object:
    return StageReviewStatus(value) if isinstance(value, str) else value


def _policy_from_json(value: object) -> object:
    return ResponsePolicy(value) if isinstance(value, str) else value


def _tuple_from_json(value: object) -> object:
    return tuple(value) if isinstance(value, list) else value


type _StageValue = Annotated[E3GenerationStage, BeforeValidator(_stage_from_json)]
type _ReviewValue = Annotated[StageReviewStatus, BeforeValidator(_review_from_json)]
type _PolicyValue = Annotated[ResponsePolicy, BeforeValidator(_policy_from_json)]
type _StringTuple = Annotated[tuple[str, ...], BeforeValidator(_tuple_from_json)]


class _PrivateRecord(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")


class E3StageArtifact(_PrivateRecord):
    """Parsed target and sanitized provenance for one terminal stage call."""

    schema_version: Literal[1] = 1
    event_id: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    stage: _StageValue
    review_status: _ReviewValue
    target: StageATarget | StageBTarget | None = None
    provenance: TeacherGenerationProvenance
    rejection_reasons: _StringTuple = ()
    response_policy: _PolicyValue | None = None
    leading_label_match: bool | None = None
    gold_in_top3: bool | None = None
    diagnostic_review_status: _ReviewValue | None = None
    diagnostic_rejection_reasons: _StringTuple = ()
    context_policy_review_status: _ReviewValue | None = None
    context_policy_rejection_reasons: _StringTuple = ()
    recorded_at: str = Field(min_length=1)
    latency_seconds: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _backfill_legacy_stage_b_reviews(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        document = dict(value)
        stage = document.get("stage")
        if stage is E3GenerationStage.STAGE_B or stage == "stage_b":
            status = document.get("review_status")
            reasons = document.get("rejection_reasons", ())
            document.setdefault("diagnostic_review_status", status)
            document.setdefault("diagnostic_rejection_reasons", reasons)
            document.setdefault("context_policy_review_status", status)
            document.setdefault("context_policy_rejection_reasons", reasons)
        return document

    @model_validator(mode="after")
    def _validate_stage_artifact(self) -> E3StageArtifact:
        generation_status = self.provenance.generation_status
        if generation_status is TeacherGenerationStatus.SUCCEEDED:
            if self.review_status not in {
                StageReviewStatus.ACCEPTED,
                StageReviewStatus.REJECTED,
            }:
                raise ValueError("Successful stage artifact requires review")
            if self.target is None:
                raise ValueError("Successful stage artifact requires a parsed target")
        else:
            if self.review_status is not StageReviewStatus.NOT_APPLICABLE:
                raise ValueError("Failed stage artifact must be not_applicable")
            if self.target is not None or self.rejection_reasons:
                raise ValueError("Failed stage artifact cannot carry review data")

        if self.review_status is StageReviewStatus.ACCEPTED and self.rejection_reasons:
            raise ValueError("Accepted stage artifact cannot carry rejection reasons")
        if (
            self.review_status is StageReviewStatus.REJECTED
            and not self.rejection_reasons
        ):
            raise ValueError("Rejected stage artifact requires rejection reasons")

        if self.stage is E3GenerationStage.STAGE_A:
            if self.target is not None and not isinstance(self.target, StageATarget):
                raise ValueError("Stage-A artifact requires StageATarget")
            if any(
                item is not None
                for item in (
                    self.response_policy,
                    self.leading_label_match,
                    self.gold_in_top3,
                    self.diagnostic_review_status,
                    self.context_policy_review_status,
                )
            ):
                raise ValueError("Stage-A artifact cannot carry Stage-B audit fields")
            if (
                self.diagnostic_rejection_reasons
                or self.context_policy_rejection_reasons
            ):
                raise ValueError("Stage-A artifact cannot carry Stage-B review reasons")
        else:
            if self.target is not None and not isinstance(self.target, StageBTarget):
                raise ValueError("Stage-B artifact requires StageBTarget")
            if generation_status is TeacherGenerationStatus.SUCCEEDED:
                if self.response_policy is None:
                    raise ValueError("Successful Stage B requires a response policy")
                if self.leading_label_match is None or self.gold_in_top3 is None:
                    raise ValueError(
                        "Successful Stage B requires private audit booleans"
                    )
                self._validate_subtarget_review(
                    "diagnostic",
                    self.diagnostic_review_status,
                    self.diagnostic_rejection_reasons,
                )
                self._validate_subtarget_review(
                    "context policy",
                    self.context_policy_review_status,
                    self.context_policy_rejection_reasons,
                )
                accepted_subtarget = StageReviewStatus.ACCEPTED in {
                    self.diagnostic_review_status,
                    self.context_policy_review_status,
                }
                if (
                    self.review_status is StageReviewStatus.ACCEPTED
                    and not accepted_subtarget
                ):
                    raise ValueError(
                        "Accepted Stage B requires at least one accepted subtarget"
                    )
                if (
                    self.review_status is StageReviewStatus.REJECTED
                    and accepted_subtarget
                ):
                    raise ValueError(
                        "Rejected Stage B cannot carry an accepted subtarget"
                    )
            elif any(
                item is not None
                for item in (
                    self.response_policy,
                    self.leading_label_match,
                    self.gold_in_top3,
                )
            ):
                raise ValueError("Failed Stage B cannot carry private audit fields")
            elif (
                self.diagnostic_review_status is not StageReviewStatus.NOT_APPLICABLE
                or self.context_policy_review_status
                is not StageReviewStatus.NOT_APPLICABLE
                or self.diagnostic_rejection_reasons
                or self.context_policy_rejection_reasons
            ):
                raise ValueError("Failed Stage B subtargets must be not_applicable")
        return self

    @staticmethod
    def _validate_subtarget_review(
        name: str,
        status: StageReviewStatus | None,
        reasons: tuple[str, ...],
    ) -> None:
        if status not in {
            StageReviewStatus.ACCEPTED,
            StageReviewStatus.REJECTED,
        }:
            raise ValueError(f"Successful Stage-B {name} requires review")
        if status is StageReviewStatus.ACCEPTED and reasons:
            raise ValueError(f"Accepted Stage-B {name} cannot have rejection reasons")
        if status is StageReviewStatus.REJECTED and not reasons:
            raise ValueError(f"Rejected Stage-B {name} requires rejection reasons")

    def progress_event(self) -> E3ProgressEvent:
        """Derive the sanitized public progress event without clinical targets."""

        return E3ProgressEvent.model_validate(
            {
                "event_id": self.event_id,
                "recorded_at": self.recorded_at,
                "sample_id": self.sample_id,
                "stage": self.stage.value,
                "generation_status": self.provenance.generation_status.value,
                "review_status": self.review_status.value,
                "response_policy": (
                    self.response_policy.value if self.response_policy else None
                ),
                "leading_label_match": self.leading_label_match,
                "gold_in_top3": self.gold_in_top3,
                "diagnostic_review_status": (
                    self.diagnostic_review_status.value
                    if self.diagnostic_review_status
                    else None
                ),
                "context_policy_review_status": (
                    self.context_policy_review_status.value
                    if self.context_policy_review_status
                    else None
                ),
                "latency_seconds": self.latency_seconds,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "provider_error_code": self.provenance.provider_error_code,
            }
        )


class E3TeacherBundleRecord(_PrivateRecord):
    """Final private row from which task-isolated student records are built."""

    schema_version: Literal[1] = 1
    sample_id: str = Field(min_length=1)
    leakage_group_id: str = Field(min_length=1)
    disease_id: str = Field(pattern=r"^D[0-9]{3}$")
    gold_diagnosis: str = Field(min_length=1)
    split: Literal["sft_train", "sft_dev"]
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    teacher_targets: TeacherTargetBundle


@dataclass(frozen=True, slots=True)
class E3ArtifactPaths:
    stage_results: Path
    bundles: Path


class E3ArtifactStore:
    """Append-only private store; it never retains prompts or raw responses."""

    def __init__(self, root: Path) -> None:
        self.paths = E3ArtifactPaths(
            stage_results=root / "stage_results.jsonl",
            bundles=root / "teacher_bundles.jsonl",
        )

    @classmethod
    def start(cls, root: Path, *, resume: bool) -> E3ArtifactStore:
        store = cls(root.resolve())
        if resume:
            missing = tuple(
                path
                for path in (store.paths.stage_results, store.paths.bundles)
                if not path.exists()
            )
            if missing:
                raise FileNotFoundError(
                    "Cannot resume E3 campaign with missing private artifacts"
                )
            return store
        occupied = tuple(
            path
            for path in (store.paths.stage_results, store.paths.bundles)
            if path.exists()
        )
        if occupied:
            raise FileExistsError("E3 private generation artifacts already exist")
        store.paths.stage_results.touch(exist_ok=False)
        store.paths.bundles.touch(exist_ok=False)
        return store

    def read_stage_results(self) -> tuple[E3StageArtifact, ...]:
        return tuple(
            E3StageArtifact.model_validate(value)
            for value in _read_jsonl(self.paths.stage_results)
        )

    def read_bundles(self) -> tuple[E3TeacherBundleRecord, ...]:
        return tuple(
            E3TeacherBundleRecord.model_validate(value)
            for value in _read_jsonl(self.paths.bundles)
        )

    def append_stage(self, artifact: E3StageArtifact) -> None:
        existing = self.read_stage_results()
        if any(
            item.sample_id == artifact.sample_id and item.stage is artifact.stage
            for item in existing
        ):
            raise ValueError("E3 stage result already exists; retries are forbidden")
        if any(item.event_id == artifact.event_id for item in existing):
            raise ValueError("Duplicate E3 stage event ID")
        _append_jsonl(self.paths.stage_results, artifact.model_dump(mode="json"))

    def append_bundle(self, record: E3TeacherBundleRecord) -> None:
        if any(item.sample_id == record.sample_id for item in self.read_bundles()):
            raise ValueError("E3 teacher bundle already exists")
        _append_jsonl(self.paths.bundles, record.model_dump(mode="json"))


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    raw = path.read_bytes()
    lines = raw.splitlines()
    complete_final_line = raw.endswith((b"\n", b"\r"))
    values: list[dict[str, object]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not complete_final_line:
                break
            raise ValueError(
                f"Invalid E3 artifact JSONL at {path}:{index + 1}"
            ) from None
        if not isinstance(value, dict):
            raise ValueError(f"E3 artifact JSONL row must be an object: {path}")
        values.append(value)
    return tuple(values)


def _append_jsonl(path: Path, value: dict[str, object]) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
