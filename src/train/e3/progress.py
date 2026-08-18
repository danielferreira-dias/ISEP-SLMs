"""Durable, backend-neutral progress reporting for E3 teacher generation.

The future E3 generation runner records one terminal event per attempted stage.
This module deliberately stores no prompts, images, gold labels, raw responses,
provider messages, headers, or secrets.  Private gold comparison is represented
only by aggregate-safe booleans computed after a successful Stage-B generation.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from src.train.e3.domain import (
    ResponsePolicy,
    StageReviewStatus,
    TeacherGenerationStatus,
)


class E3GenerationStage(StrEnum):
    """Teacher-generation stage represented by one progress event."""

    STAGE_A = "stage_a"
    STAGE_B = "stage_b"


class E3CampaignState(StrEnum):
    """Lifecycle state for a generation campaign."""

    RUNNING = "running"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


def _datetime_from_json(value: object) -> object:
    return datetime.fromisoformat(value) if isinstance(value, str) else value


def _stage_from_json(value: object) -> object:
    return E3GenerationStage(value) if isinstance(value, str) else value


def _generation_status_from_json(value: object) -> object:
    return TeacherGenerationStatus(value) if isinstance(value, str) else value


def _review_status_from_json(value: object) -> object:
    return StageReviewStatus(value) if isinstance(value, str) else value


def _response_policy_from_json(value: object) -> object:
    return ResponsePolicy(value) if isinstance(value, str) else value


type _DatetimeValue = Annotated[datetime, BeforeValidator(_datetime_from_json)]
type _StageValue = Annotated[E3GenerationStage, BeforeValidator(_stage_from_json)]
type _GenerationStatusValue = Annotated[
    TeacherGenerationStatus,
    BeforeValidator(_generation_status_from_json),
]
type _ReviewStatusValue = Annotated[
    StageReviewStatus,
    BeforeValidator(_review_status_from_json),
]
type _ResponsePolicyValue = Annotated[
    ResponsePolicy,
    BeforeValidator(_response_policy_from_json),
]


class _ImmutableModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")


class E3CampaignSpec(_ImmutableModel):
    """Immutable identity of one versioned teacher-generation campaign."""

    schema_version: Literal[1] = 1
    campaign_id: str = Field(min_length=1)
    total_samples: int = Field(gt=0)
    provider: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    teacher_model: str = Field(min_length=1)
    teacher_revision: str = Field(min_length=1)
    stage_a_prompt_id: str = Field(min_length=1)
    stage_b_prompt_id: str = Field(min_length=1)
    stage_a_gold_visible_to_teacher: bool = False
    stage_b_gold_visible_to_teacher: bool = False
    model_config_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    reasoning_effort: str | None = None
    structured_output_mode: str | None = None
    selection_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    stage_a_prompt_resource_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    stage_a_rendered_prompt_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    stage_b_prompt_resource_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    terminology_lexicon_id: str | None = None
    terminology_resource_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @property
    def identity_sha256(self) -> str:
        """Return a stable digest used to detect accidental campaign mixing."""

        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class E3ProgressEvent(_ImmutableModel):
    """One sanitized terminal Stage-A or Stage-B outcome."""

    schema_version: Literal[1] = 1
    event_id: str = Field(min_length=1)
    recorded_at: _DatetimeValue
    sample_id: str = Field(min_length=1)
    stage: _StageValue
    generation_status: _GenerationStatusValue | None
    review_status: _ReviewStatusValue
    response_policy: _ResponsePolicyValue | None = None
    leading_label_match: bool | None = None
    gold_in_top3: bool | None = None
    diagnostic_review_status: _ReviewStatusValue | None = None
    context_policy_review_status: _ReviewStatusValue | None = None
    latency_seconds: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    provider_error_code: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _backfill_legacy_stage_b_reviews(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        document = dict(value)
        stage = document.get("stage")
        if stage is E3GenerationStage.STAGE_B or stage == "stage_b":
            status = document.get("review_status")
            document.setdefault("diagnostic_review_status", status)
            document.setdefault("context_policy_review_status", status)
        return document

    @model_validator(mode="after")
    def _validate_terminal_outcome(self) -> E3ProgressEvent:
        if self.recorded_at.tzinfo is None:
            raise ValueError("recorded_at must include a timezone")

        if self.review_status is StageReviewStatus.NOT_GENERATED:
            if self.generation_status is not None:
                raise ValueError("not_generated cannot carry a generation status")
        elif self.generation_status is None:
            raise ValueError("an attempted stage requires a generation status")

        if self.generation_status is TeacherGenerationStatus.SUCCEEDED:
            if self.review_status not in {
                StageReviewStatus.ACCEPTED,
                StageReviewStatus.REJECTED,
            }:
                raise ValueError("successful generation requires scientific review")
            if self.provider_error_code is not None:
                raise ValueError("successful generation cannot carry an error code")
        elif self.generation_status is not None:
            if self.review_status is not StageReviewStatus.NOT_APPLICABLE:
                raise ValueError("failed generation must be not_applicable")

        if self.stage is E3GenerationStage.STAGE_A:
            if any(
                value is not None
                for value in (
                    self.response_policy,
                    self.leading_label_match,
                    self.gold_in_top3,
                    self.diagnostic_review_status,
                    self.context_policy_review_status,
                )
            ):
                raise ValueError("Stage A cannot carry diagnostic audit fields")
        elif self.generation_status is TeacherGenerationStatus.SUCCEEDED:
            if self.response_policy is None:
                raise ValueError("successful Stage B requires a response policy")
            if self.leading_label_match is None or self.gold_in_top3 is None:
                raise ValueError(
                    "successful Stage B requires post-generation gold audit booleans"
                )
            subreviews = {
                self.diagnostic_review_status,
                self.context_policy_review_status,
            }
            if not subreviews.issubset(
                {StageReviewStatus.ACCEPTED, StageReviewStatus.REJECTED}
            ):
                raise ValueError("successful Stage B requires both subtarget reviews")
            accepted_subtarget = StageReviewStatus.ACCEPTED in subreviews
            if (
                self.review_status is StageReviewStatus.ACCEPTED
                and not accepted_subtarget
            ):
                raise ValueError(
                    "accepted Stage B requires at least one accepted subtarget"
                )
            if (
                self.review_status is StageReviewStatus.REJECTED
                and accepted_subtarget
            ):
                raise ValueError(
                    "rejected Stage B cannot carry an accepted subtarget"
                )
        elif any(
            value is not None
            for value in (
                self.response_policy,
                self.leading_label_match,
                self.gold_in_top3,
            )
        ):
            raise ValueError(
                "unsuccessful Stage B cannot carry diagnostic audit fields"
            )
        elif self.generation_status is not None and (
            self.diagnostic_review_status is not StageReviewStatus.NOT_APPLICABLE
            or self.context_policy_review_status
            is not StageReviewStatus.NOT_APPLICABLE
        ):
            raise ValueError("failed Stage B subtargets must be not_applicable")
        return self


@dataclass(frozen=True)
class E3ProgressPaths:
    """Stable output paths owned by one E3 generation campaign."""

    root: Path
    manifest: Path
    events: Path
    status: Path
    report: Path

    @classmethod
    def below(cls, root: Path) -> E3ProgressPaths:
        resolved = root.resolve()
        return cls(
            root=resolved,
            manifest=resolved / "campaign_manifest.json",
            events=resolved / "generations.jsonl",
            status=resolved / "campaign_status.json",
            report=resolved / "report.html",
        )


class E3ProgressStore:
    """Append-only event store with atomic snapshots and a live HTML report."""

    def __init__(self, paths: E3ProgressPaths, spec: E3CampaignSpec) -> None:
        self.paths = paths
        self.spec = spec

    @classmethod
    def start(
        cls,
        output_directory: Path,
        spec: E3CampaignSpec,
        *,
        resume: bool = False,
    ) -> E3ProgressStore:
        """Create a campaign or reopen the exact same non-completed campaign."""

        paths = E3ProgressPaths.below(output_directory)
        paths.root.mkdir(parents=True, exist_ok=True)
        if paths.manifest.exists():
            if not resume:
                raise FileExistsError(
                    f"E3 progress campaign already exists: {paths.root}"
                )
            store = cls.open(paths.root)
            if store.spec != spec:
                raise ValueError("E3 campaign identity mismatch on resume")
            current = store.read_snapshot()
            if current["status"] == E3CampaignState.COMPLETED.value:
                raise ValueError("completed E3 campaigns cannot be resumed")
            return store._refresh(state=E3CampaignState.RUNNING)

        occupied = tuple(
            path
            for path in (paths.events, paths.status, paths.report)
            if path.exists()
        )
        if occupied:
            raise FileExistsError(
                "E3 output directory contains progress artifacts without a manifest"
            )

        started_at = _utc_now()
        manifest = {
            "schema_version": 1,
            "identity_sha256": spec.identity_sha256,
            "campaign": spec.model_dump(mode="json"),
            "started_at": started_at,
        }
        _atomic_json(paths.manifest, manifest)
        paths.events.touch(exist_ok=False)
        return cls(paths, spec)._refresh(state=E3CampaignState.RUNNING)

    @classmethod
    def open(cls, output_directory: Path) -> E3ProgressStore:
        """Open and integrity-check an existing progress campaign."""

        paths = E3ProgressPaths.below(output_directory)
        manifest = _read_object(paths.manifest)
        campaign = manifest.get("campaign")
        if not isinstance(campaign, dict):
            raise ValueError("E3 campaign manifest is missing its identity")
        spec = E3CampaignSpec.model_validate(campaign)
        if manifest.get("identity_sha256") != spec.identity_sha256:
            raise ValueError("E3 campaign manifest identity hash mismatch")
        return cls(paths, spec)

    def record(self, event: E3ProgressEvent) -> dict[str, Any]:
        """Append one new terminal stage event and refresh both interfaces."""

        snapshot = self.read_snapshot()
        if snapshot["status"] != E3CampaignState.RUNNING.value:
            raise ValueError("events can only be recorded while a campaign is running")

        events = self.read_events()
        keys = {(item.sample_id, item.stage) for item in events}
        event_key = (event.sample_id, event.stage)
        if event_key in keys:
            raise ValueError(
                "a terminal event already exists for "
                f"{event.sample_id}/{event.stage.value}; silent retries are forbidden"
            )
        if any(item.event_id == event.event_id for item in events):
            raise ValueError(f"duplicate E3 event_id: {event.event_id}")

        stage_a = {
            item.sample_id: item
            for item in events
            if item.stage is E3GenerationStage.STAGE_A
        }
        if event.stage is E3GenerationStage.STAGE_A:
            if len(stage_a) >= self.spec.total_samples:
                raise ValueError("Stage-A event count exceeds campaign total_samples")
        else:
            source = stage_a.get(event.sample_id)
            if source is None or source.review_status is not StageReviewStatus.ACCEPTED:
                raise ValueError("Stage B requires an accepted Stage-A event")

        _append_jsonl(self.paths.events, event.model_dump(mode="json"))
        return self._refresh(state=E3CampaignState.RUNNING).read_snapshot()

    def finalize(
        self,
        state: E3CampaignState,
        *,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Finalize a campaign without modifying its immutable event history."""

        if state is E3CampaignState.RUNNING:
            raise ValueError("final campaign state cannot be running")
        snapshot = self._snapshot(state=state, error=error)
        progress = _object(snapshot, "progress")
        if (
            state is E3CampaignState.COMPLETED
            and progress["samples_finished"] != self.spec.total_samples
        ):
            raise ValueError("completed campaign requires every sample to finish")
        _atomic_json(self.paths.status, snapshot)
        _atomic_text(self.paths.report, render_html(snapshot, auto_refresh=False))
        return snapshot

    def read_events(self) -> tuple[E3ProgressEvent, ...]:
        """Read the append-only event log, tolerating only a partial final line."""

        if not self.paths.events.exists():
            return ()
        raw = self.paths.events.read_bytes()
        lines = raw.splitlines()
        complete_final_line = raw.endswith((b"\n", b"\r"))
        events: list[E3ProgressEvent] = []
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1 and not complete_final_line:
                    break
                raise ValueError(
                    f"invalid progress JSONL at {self.paths.events}:{index + 1}"
                ) from None
            events.append(E3ProgressEvent.model_validate(payload))
        return tuple(events)

    def read_snapshot(self) -> dict[str, Any]:
        """Read the latest atomically published campaign snapshot."""

        return _read_object(self.paths.status)

    def _refresh(self, *, state: E3CampaignState) -> E3ProgressStore:
        snapshot = self._snapshot(state=state)
        _atomic_json(self.paths.status, snapshot)
        _atomic_text(
            self.paths.report,
            render_html(
                snapshot,
                auto_refresh=state is E3CampaignState.RUNNING,
            ),
        )
        return self

    def _snapshot(
        self,
        *,
        state: E3CampaignState,
        error: str | None = None,
    ) -> dict[str, Any]:
        manifest = _read_object(self.paths.manifest)
        started_at = _parse_datetime(manifest.get("started_at"), "started_at")
        now = datetime.now(UTC)
        events = self.read_events()
        stage_a = {
            event.sample_id: event
            for event in events
            if event.stage is E3GenerationStage.STAGE_A
        }
        stage_b = {
            event.sample_id: event
            for event in events
            if event.stage is E3GenerationStage.STAGE_B
        }

        finished = {
            sample_id
            for sample_id, event in stage_a.items()
            if event.review_status is not StageReviewStatus.ACCEPTED
            or sample_id in stage_b
        }
        elapsed = max(0.0, (now - started_at).total_seconds())
        rate = len(finished) / elapsed if elapsed else 0.0
        remaining = self.spec.total_samples - len(finished)
        eta = remaining / rate if rate > 0 and remaining > 0 else None

        a_generation = _generation_counts(stage_a.values())
        a_review = _review_counts(stage_a.values())
        b_generation = _generation_counts(stage_b.values())
        b_review = _review_counts(stage_b.values())
        b_diagnostic_review = _subreview_counts(
            stage_b.values(),
            field="diagnostic_review_status",
        )
        b_context_policy_review = _subreview_counts(
            stage_b.values(),
            field="context_policy_review_status",
        )
        implicitly_blocked_b = sum(
            event.review_status is not StageReviewStatus.ACCEPTED
            for event in stage_a.values()
        )
        b_review[StageReviewStatus.NOT_GENERATED.value] += implicitly_blocked_b
        b_diagnostic_review[
            StageReviewStatus.NOT_GENERATED.value
        ] += implicitly_blocked_b
        b_context_policy_review[
            StageReviewStatus.NOT_GENERATED.value
        ] += implicitly_blocked_b

        evaluable_b = tuple(
            event
            for event in stage_b.values()
            if event.generation_status is TeacherGenerationStatus.SUCCEEDED
            and event.leading_label_match is not None
            and event.gold_in_top3 is not None
        )
        leading_matches = sum(
            event.leading_label_match is True for event in evaluable_b
        )
        top3_matches = sum(event.gold_in_top3 is True for event in evaluable_b)
        policies = Counter(
            event.response_policy.value
            for event in evaluable_b
            if event.response_policy is not None
        )
        accepted_a = a_review[StageReviewStatus.ACCEPTED.value]
        accepted_b_diagnostic = b_diagnostic_review[
            StageReviewStatus.ACCEPTED.value
        ]
        accepted_b_context_policy = b_context_policy_review[
            StageReviewStatus.ACCEPTED.value
        ]

        total_input_tokens = sum(event.input_tokens or 0 for event in events)
        total_output_tokens = sum(event.output_tokens or 0 for event in events)
        observed_latencies = tuple(
            event.latency_seconds
            for event in events
            if event.latency_seconds is not None
        )
        failed_statuses = tuple(
            status
            for event in events
            if (status := event.generation_status) is not None
            and status is not TeacherGenerationStatus.SUCCEEDED
        )
        failure_counts = Counter(status.value for status in failed_statuses)

        result: dict[str, Any] = {
            "schema_version": 1,
            "status": state.value,
            "updated_at": now.isoformat(),
            "finished_at": (
                now.isoformat() if state is not E3CampaignState.RUNNING else None
            ),
            "identity_sha256": self.spec.identity_sha256,
            "campaign": self.spec.model_dump(mode="json"),
            "timing": {
                "started_at": started_at.isoformat(),
                "elapsed_seconds": elapsed,
                "samples_per_minute": rate * 60,
                "eta_seconds": eta,
            },
            "progress": {
                "total_samples": self.spec.total_samples,
                "samples_finished": len(finished),
                "samples_remaining": remaining,
                "completion_rate": len(finished) / self.spec.total_samples,
            },
            "stage_a": {
                "terminal": len(stage_a),
                "generation": a_generation,
                "review": a_review,
            },
            "stage_b": {
                "eligible": accepted_a,
                "terminal": len(stage_b),
                "pending": max(0, accepted_a - len(stage_b)),
                "generation": b_generation,
                "review": b_review,
                "diagnostic_review": b_diagnostic_review,
                "context_policy_review": b_context_policy_review,
            },
            "private_validation": {
                "evaluable_stage_b": len(evaluable_b),
                "leading_label_matches": leading_matches,
                "leading_label_match_rate": _ratio(
                    leading_matches,
                    len(evaluable_b),
                ),
                "gold_in_top3_matches": top3_matches,
                "gold_in_top3_rate": _ratio(top3_matches, len(evaluable_b)),
                "gold_was_visible_to_teacher": (
                    self.spec.stage_b_gold_visible_to_teacher
                ),
                "stage_a_gold_was_visible_to_teacher": (
                    self.spec.stage_a_gold_visible_to_teacher
                ),
                "stage_b_gold_was_visible_to_teacher": (
                    self.spec.stage_b_gold_visible_to_teacher
                ),
                "leading_label_metric_interpretation": (
                    "anchor_compliance"
                    if self.spec.stage_b_gold_visible_to_teacher
                    else "answer_blind_accuracy"
                ),
            },
            "response_policy": {
                ResponsePolicy.ANSWER_DIFFERENTIAL.value: policies[
                    ResponsePolicy.ANSWER_DIFFERENTIAL.value
                ],
                ResponsePolicy.REQUEST_CONTEXT.value: policies[
                    ResponsePolicy.REQUEST_CONTEXT.value
                ],
            },
            "materializable_rows": {
                "diagnosis": len(finished),
                "morphology": accepted_a,
                "caption": accepted_a,
                "grounded_differential": accepted_b_diagnostic,
                "context_policy": accepted_b_context_policy,
            },
            "failures": dict(sorted(failure_counts.items())),
            "usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "mean_latency_seconds": (
                    sum(observed_latencies) / len(observed_latencies)
                    if observed_latencies
                    else None
                ),
            },
            "artifacts": {
                "manifest": self.paths.manifest.name,
                "events": self.paths.events.name,
                "status": self.paths.status.name,
                "report": self.paths.report.name,
            },
        }
        if error is not None:
            result["error"] = error
        return result


def render_terminal(snapshot: dict[str, Any], *, width: int = 34) -> str:
    """Render a compact, dependency-free terminal dashboard."""

    campaign = _object(snapshot, "campaign")
    progress = _object(snapshot, "progress")
    timing = _object(snapshot, "timing")
    stage_a = _object(snapshot, "stage_a")
    stage_b = _object(snapshot, "stage_b")
    validation = _object(snapshot, "private_validation")
    policies = _object(snapshot, "response_policy")
    rows = _object(snapshot, "materializable_rows")
    usage = _object(snapshot, "usage")

    total = int(progress["total_samples"])
    finished = int(progress["samples_finished"])
    fraction = float(progress["completion_rate"])
    filled = min(width, round(width * fraction))
    bar = "#" * filled + "-" * (width - filled)
    eta = _duration(timing.get("eta_seconds"))
    rate = float(timing["samples_per_minute"])

    a_generation = _object(stage_a, "generation")
    a_review = _object(stage_a, "review")
    b_generation = _object(stage_b, "generation")
    b_review = _object(stage_b, "review")
    b_diagnostic_review = _object(stage_b, "diagnostic_review")
    b_context_policy_review = _object(stage_b, "context_policy_review")

    lines = [
        f"E3 TEACHER GENERATION  [{snapshot['status']}]",
        (
            f"{campaign['teacher_model']} @ {campaign['teacher_revision']}  "
            f"{campaign['provider']}/{campaign['backend']}"
        ),
        f"Progress  [{bar}] {finished}/{total} ({fraction:.1%})",
        f"Rate {rate:.2f} samples/min  ETA {eta}",
        "",
        "Stage A",
        _metric_line(
            ("terminal", stage_a["terminal"]),
            ("succeeded", a_generation["succeeded"]),
            ("accepted", a_review["accepted"]),
            ("rejected", a_review["rejected"]),
        ),
        "Stage B",
        _metric_line(
            ("eligible", stage_b["eligible"]),
            ("terminal", stage_b["terminal"]),
            ("succeeded", b_generation["succeeded"]),
            ("accepted", b_review["accepted"]),
        ),
        _metric_line(
            ("diagnostic accepted", b_diagnostic_review["accepted"]),
            ("context-policy accepted", b_context_policy_review["accepted"]),
        ),
        "",
        (
            "Private validation (Stage-B anchor compliance)"
            if validation.get("stage_b_gold_was_visible_to_teacher")
            else "Private validation (gold never sent to teacher)"
        ),
        _rate_line(
            "leading-label match",
            validation["leading_label_matches"],
            validation["evaluable_stage_b"],
            validation["leading_label_match_rate"],
        ),
        _rate_line(
            "gold in top-3",
            validation["gold_in_top3_matches"],
            validation["evaluable_stage_b"],
            validation["gold_in_top3_rate"],
        ),
        _metric_line(
            ("ANSWER_DIFFERENTIAL", policies["ANSWER_DIFFERENTIAL"]),
            ("REQUEST_CONTEXT", policies["REQUEST_CONTEXT"]),
        ),
        "",
        "Materializable student rows",
        _metric_line(
            ("diagnosis", rows["diagnosis"]),
            ("morphology", rows["morphology"]),
            ("caption", rows["caption"]),
        ),
        _metric_line(
            ("grounded_differential", rows["grounded_differential"]),
            ("context_policy", rows["context_policy"]),
        ),
        "",
        _metric_line(
            ("input tokens", usage["input_tokens"]),
            ("output tokens", usage["output_tokens"]),
        ),
    ]
    failures = _object(snapshot, "failures")
    if failures:
        lines.extend(
            (
                "Failures",
                "  ".join(f"{key}={value}" for key, value in failures.items()),
            )
        )
    if snapshot.get("error"):
        lines.extend(("", f"ERROR: {snapshot['error']}"))
    return "\n".join(lines)


def render_html(snapshot: dict[str, Any], *, auto_refresh: bool) -> str:
    """Render a self-contained local dashboard without clinical row data."""

    campaign = _object(snapshot, "campaign")
    progress = _object(snapshot, "progress")
    timing = _object(snapshot, "timing")
    stage_a = _object(snapshot, "stage_a")
    stage_b = _object(snapshot, "stage_b")
    validation = _object(snapshot, "private_validation")
    policies = _object(snapshot, "response_policy")
    rows = _object(snapshot, "materializable_rows")
    usage = _object(snapshot, "usage")
    failures = _object(snapshot, "failures")
    refresh = '<meta http-equiv="refresh" content="2">' if auto_refresh else ""
    fraction = float(progress["completion_rate"])
    leading_card = _html_rate_card(
        "leading-label match",
        validation["leading_label_matches"],
        validation["evaluable_stage_b"],
        validation["leading_label_match_rate"],
    )
    top3_card = _html_rate_card(
        "gold in top-3",
        validation["gold_in_top3_matches"],
        validation["evaluable_stage_b"],
        validation["gold_in_top3_rate"],
    )
    failure_cards = "".join(
        _html_card(key.replace("_", " "), value)
        for key, value in failures.items()
    ) or _html_card("generation failures", 0)
    diagnostic_review_card = _html_card(
        "diagnostic accepted",
        _object(stage_b, "diagnostic_review")["accepted"],
    )
    context_policy_review_card = _html_card(
        "context-policy accepted",
        _object(stage_b, "context_policy_review")["accepted"],
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  {refresh}
  <title>E3 teacher-generation progress</title>
  <style>
    :root {{ color-scheme: dark; --bg:#07111f; --panel:#101d2d;
      --line:#27394d; --text:#ecf4ff; --muted:#91a7bd; --accent:#47d7ac;
      --blue:#72a7ff; --warn:#ffca6e; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:radial-gradient(circle at top,#112942,var(--bg) 42%);
      color:var(--text); font:15px/1.45 ui-sans-serif,system-ui,sans-serif; }}
    main {{ width:min(1120px,calc(100% - 32px)); margin:32px auto 64px; }}
    header,section {{ background:color-mix(in srgb,var(--panel) 92%,transparent);
      border:1px solid var(--line); border-radius:18px; padding:22px;
      box-shadow:0 16px 45px #0005; margin-bottom:16px; }}
    h1,h2 {{ margin:0 0 8px; }} h1 {{ font-size:clamp(24px,4vw,40px); }}
    h2 {{ font-size:18px; }} .muted {{ color:var(--muted); }}
    .status {{ color:var(--accent); font-weight:800; letter-spacing:.08em; }}
    .bar {{ height:16px; border-radius:999px; background:#07101c; overflow:hidden;
      border:1px solid var(--line); margin:18px 0 8px; }}
    .bar span {{ display:block; height:100%; width:{fraction:.4%};
      background:linear-gradient(90deg,var(--blue),var(--accent)); }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
      gap:12px; }}
    .card {{ border:1px solid var(--line); border-radius:13px; padding:15px;
      background:#091524; min-height:92px; }}
    .card strong {{ display:block; font-size:27px; margin-top:5px; }}
    .card small {{ color:var(--muted); text-transform:uppercase;
      letter-spacing:.06em; }}
    .split {{ display:grid;
      grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; }}
    .note {{ border-left:3px solid var(--warn); padding-left:12px;
      color:var(--muted); }}
    code {{ color:var(--accent); }}
  </style>
</head>
<body><main>
  <header>
    <div class="status">{_esc(snapshot['status'])}</div>
    <h1>E3 teacher generation</h1>
    <div class="muted">{_esc(campaign['teacher_model'])} @
      <code>{_esc(campaign['teacher_revision'])}</code> ·
      {_esc(campaign['provider'])}/{_esc(campaign['backend'])}</div>
    <div class="bar"><span></span></div>
    <strong>{progress['samples_finished']} / {progress['total_samples']}</strong>
    <span class="muted"> · {fraction:.1%} ·
      {_esc(timing['samples_per_minute'])} samples/min
      · ETA {_esc(_duration(timing.get('eta_seconds')))}</span>
  </header>
  <div class="split">
    <section><h2>Stage A</h2><div class="grid">
      {_html_card('terminal', stage_a['terminal'])}
      {_html_card('succeeded', _object(stage_a, 'generation')['succeeded'])}
      {_html_card('accepted', _object(stage_a, 'review')['accepted'])}
      {_html_card('rejected', _object(stage_a, 'review')['rejected'])}
    </div></section>
    <section><h2>Stage B</h2><div class="grid">
      {_html_card('eligible', stage_b['eligible'])}
      {_html_card('terminal', stage_b['terminal'])}
      {_html_card('succeeded', _object(stage_b, 'generation')['succeeded'])}
      {_html_card('accepted', _object(stage_b, 'review')['accepted'])}
      {diagnostic_review_card}
      {context_policy_review_card}
    </div></section>
  </div>
  <section><h2>Private validation</h2>
    <p class="note">Gold is compared only after generation and is never sent
      to the teacher.
      This dashboard stores aggregate-safe booleans, not gold labels.</p>
    <div class="grid">
      {leading_card}
      {top3_card}
      {_html_card('answer differential', policies['ANSWER_DIFFERENTIAL'])}
      {_html_card('request context', policies['REQUEST_CONTEXT'])}
    </div>
  </section>
  <section><h2>Materializable student rows</h2><div class="grid">
    {''.join(_html_card(key.replace('_', ' '), value) for key, value in rows.items())}
  </div></section>
  <section><h2>Failures and usage</h2><div class="grid">
    {failure_cards}
    {_html_card('input tokens', usage['input_tokens'])}
    {_html_card('output tokens', usage['output_tokens'])}
    {_html_card('mean latency (s)', usage['mean_latency_seconds'] or '—')}
  </div></section>
  <p class="muted">Updated {_esc(snapshot['updated_at'])} · campaign
    <code>{_esc(campaign['campaign_id'])}</code></p>
</main></body></html>"""


def _generation_counts(events: Any) -> dict[str, int]:
    counts = {status.value: 0 for status in TeacherGenerationStatus}
    for event in events:
        if event.generation_status is not None:
            counts[event.generation_status.value] += 1
    return counts


def _review_counts(events: Any) -> dict[str, int]:
    counts = {status.value: 0 for status in StageReviewStatus}
    for event in events:
        counts[event.review_status.value] += 1
    return counts


def _subreview_counts(events: Any, *, field: str) -> dict[str, int]:
    counts = {status.value: 0 for status in StageReviewStatus}
    for event in events:
        status = getattr(event, field)
        if status is not None:
            counts[status.value] += 1
    return counts


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_datetime(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO-8601 string")
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return result


def _object(document: dict[str, Any], field: str) -> dict[str, Any]:
    value = document.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"progress snapshot field {field!r} must be an object")
    return value


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
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


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _duration(value: object) -> str:
    if value is None:
        return "—"
    seconds = max(0, round(_number(value)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _metric_line(*values: tuple[str, object]) -> str:
    return "  ".join(f"{name}={value}" for name, value in values)


def _rate_line(
    name: str,
    hits: object,
    total: object,
    rate: object,
) -> str:
    percentage = "—" if rate is None else f"{_number(rate):.1%}"
    return f"{name}: {hits}/{total} ({percentage})"


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _html_card(name: str, value: object) -> str:
    return (
        f'<div class="card"><small>{_esc(name)}</small>'
        f"<strong>{_esc(value)}</strong></div>"
    )


def _html_rate_card(
    name: str,
    hits: object,
    total: object,
    rate: object,
) -> str:
    percentage = "—" if rate is None else f"{_number(rate):.1%}"
    return _html_card(name, f"{hits}/{total} · {percentage}")


def _number(value: object) -> float:
    if not isinstance(value, int | float):
        raise ValueError("progress metric must be numeric")
    return float(value)
