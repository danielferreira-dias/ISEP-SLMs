"""Run E3 teacher Stage A and Stage B sequentially with live progress."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO, cast, overload

from PIL import Image

from project.dataset.dataset import DistillDataset
from project.dataset.examples import DistillExample, example_from_hub_row
from project.stages.stage_a import run_stage_a
from project.stages.stage_b import run_stage_b
from project.teacher.client import StageCompleter, create_teacher_client
from project.teacher.provenance import generation_provenance
from project.teacher.schemas import (
    GenerationProvenance,
    RecordStatus,
    StageAFileRow,
    StageBFileRow,
    UsageInfo,
)
from project.teacher.teacher import DEFAULT_CONFIG, PROJECT_ROOT, TeacherModel
from project.teacher.utils.jsonl import (
    index_ok_stage_a,
    load_manifest,
    load_stage_a_rows,
    load_stage_b_rows,
)

LOGGER = logging.getLogger("project.pipeline.generate")
_BAR_WIDTH = 28


class CampaignFailure(RuntimeError):
    """A stage finished with missing attempts or retryable errors."""


class CampaignBudgetExceeded(CampaignFailure):
    """The local estimated-cost guard reached its configured USD ceiling."""


class _HubTable(Protocol):
    """Dataset indexing needed without decoding every image up front."""

    def __len__(self) -> int: ...

    @overload
    def __getitem__(self, key: int) -> object: ...

    @overload
    def __getitem__(self, key: str) -> object: ...


type ExampleFactory = Callable[[tuple[str, ...]], Iterator[DistillExample]]


@dataclass(frozen=True, slots=True, kw_only=True)
class ExampleCohort:
    """Ordered source identities plus a lazy image-decoding factory."""

    sample_ids: tuple[str, ...]
    factory: ExampleFactory

    def __post_init__(self) -> None:
        if not self.sample_ids:
            raise ValueError("Teacher campaign source is empty")
        if any(not sample_id.strip() for sample_id in self.sample_ids):
            raise ValueError("Teacher campaign contains a blank sample_id")
        if len(self.sample_ids) != len(set(self.sample_ids)):
            raise ValueError("Teacher campaign source contains duplicate sample_id")

    def select(self, limit: int | None) -> tuple[str, ...]:
        """Return the deterministic campaign prefix selected by ``limit``."""
        if limit is not None and limit <= 0:
            raise ValueError("Campaign limit must be greater than zero")
        selected = self.sample_ids if limit is None else self.sample_ids[:limit]
        if not selected:
            raise ValueError("Teacher campaign selection is empty")
        return selected

    def iter_selected(self, sample_ids: Sequence[str]) -> Iterator[DistillExample]:
        """Decode exactly the requested source examples in source order."""
        selected = tuple(sample_ids)
        if len(selected) != len(set(selected)):
            raise ValueError("Requested campaign sample IDs are not unique")
        unknown = set(selected).difference(self.sample_ids)
        if unknown:
            raise KeyError(f"Unknown campaign sample IDs: {len(unknown)}")
        yield from self.factory(selected)


@dataclass(frozen=True, slots=True, kw_only=True)
class CampaignResult:
    """Terminal coverage of one sequential Stage A/B generation campaign."""

    selected_samples: int
    stage_a_completed: int
    stage_b_completed: int
    stage_b_ok: int
    stage_b_rejected: int
    stage_a_output: Path
    stage_b_output: Path
    elapsed_seconds: float
    estimated_cost_usd: float | None


@dataclass(slots=True, kw_only=True)
class CampaignCostTracker:
    """Accumulate provider-reported or pinned list-price USD request costs."""

    teacher: TeacherModel
    budget_usd: float | None = None
    estimated_usd: float = 0.0
    enabled: bool = False

    def __post_init__(self) -> None:
        if self.budget_usd is not None and self.budget_usd <= 0:
            raise ValueError("Estimated-cost budget must be greater than zero")
        if self.budget_usd is not None and self.teacher.pricing is None:
            raise ValueError(
                "--max-estimated-cost-usd requires pinned teacher pricing"
            )
        if self.teacher.pricing is not None:
            self.enabled = True

    def add_usage(self, usage: UsageInfo | None) -> None:
        """Add one charged response when a USD cost can be established."""
        cost = _usage_cost_usd(self.teacher, usage)
        if cost is None:
            return
        self.enabled = True
        self.estimated_usd += cost

    def assert_below_budget(self) -> None:
        """Stop before another request once the local soft ceiling is reached."""
        if self.budget_usd is None or self.estimated_usd < self.budget_usd:
            return
        raise CampaignBudgetExceeded(
            "Local estimated Vertex cost reached the configured ceiling: "
            f"${self.estimated_usd:.4f} >= ${self.budget_usd:.2f}. "
            "No further request was started. This is not the finalized Cloud "
            "Billing balance and does not account for promotional credits."
        )


class TerminalProgress:
    """Dependency-free single-line progress bar with ETA and outcome counts."""

    def __init__(
        self,
        *,
        stage: str,
        total: int,
        completed: int = 0,
        successes: int | None = None,
        rejections: int = 0,
        stream: TextIO = sys.stderr,
        width: int = _BAR_WIDTH,
        cost_tracker: CampaignCostTracker | None = None,
    ) -> None:
        if total <= 0:
            raise ValueError("Progress total must be greater than zero")
        if not 0 <= completed <= total:
            raise ValueError("Initial progress lies outside its total")
        initial_successes = completed if successes is None else successes
        if min(initial_successes, rejections) < 0:
            raise ValueError("Initial progress counts cannot be negative")
        if initial_successes + rejections != completed:
            raise ValueError(
                "Initial ok and rejected counts must equal completed progress"
            )
        if width < 10:
            raise ValueError("Progress width must be at least 10")
        self.stage = stage
        self.total = total
        self.completed = completed
        self.successes = initial_successes
        self.rejections = rejections
        self.failures = 0
        self._initial_completed = completed
        self._stream = stream
        self._width = width
        self._started = time.monotonic()
        self._last_length = 0
        self._interactive = bool(getattr(stream, "isatty", lambda: False)())
        self._cost_tracker = cost_tracker
        self._render(current="resume", terminal=False)

    def record(self, sample_id: str, status: RecordStatus) -> None:
        """Advance once after a record has been durably appended."""
        if self.completed >= self.total:
            raise ValueError(f"{self.stage} progress exceeded its total")
        self.completed += 1
        if status is RecordStatus.OK:
            self.successes += 1
        elif status is RecordStatus.REJECTED:
            self.rejections += 1
        else:
            self.failures += 1
        self._render(current=sample_id, terminal=False)

    def finish(self, state: str) -> None:
        """Render a final newline-terminated state."""
        self._render(current=state, terminal=True)

    def _render(self, *, current: str, terminal: bool) -> None:
        fraction = self.completed / self.total
        filled = min(self._width, round(self._width * fraction))
        bar = "#" * filled + "-" * (self._width - filled)
        remaining = self.total - self.completed
        attempted = self.completed - self._initial_completed
        elapsed = max(0.0, time.monotonic() - self._started)
        rate = attempted / elapsed if attempted and elapsed > 0 else 0.0
        eta = _format_duration(remaining / rate) if rate > 0 else "--:--"
        sample = _short_identifier(current)
        line = (
            f"{self.stage:<7} [{bar}] {self.completed}/{self.total} "
            f"({fraction:6.2%}) left={remaining} ok={self.successes} "
            f"rejected={self.rejections} failed={self.failures} "
            f"eta={eta} current={sample}"
        )
        if self._cost_tracker is not None and self._cost_tracker.enabled:
            line += f" est_cost=${self._cost_tracker.estimated_usd:.4f}"
            if self._cost_tracker.budget_usd is not None:
                line += f"/${self._cost_tracker.budget_usd:.2f}"
        padding = " " * max(0, self._last_length - len(line))
        end = "\n" if terminal or not self._interactive else "\r"
        self._stream.write("\r" + line + padding + end)
        self._stream.flush()
        self._last_length = len(line)


def load_hub_cohort(
    *,
    config: str = "diagnosis",
    split: str = "sft_train",
) -> ExampleCohort:
    """Load one pinned Hub table while keeping image decoding lazy."""
    loaded = DistillDataset.load(config=config, split=split)
    table = cast(_HubTable, loaded.get(config, split))
    raw_ids = table["sample_id"]
    if not isinstance(raw_ids, Sequence):
        raise TypeError("Hub sample_id column is not a sequence")
    sample_ids = tuple(_required_sample_id(value) for value in raw_ids)
    index_by_id = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    repo = loaded.spec.huggingface

    def factory(selected: tuple[str, ...]) -> Iterator[DistillExample]:
        for sample_id in selected:
            raw = table[index_by_id[sample_id]]
            if not isinstance(raw, Mapping):
                raise TypeError(f"Hub row {sample_id!r} is not a mapping")
            yield example_from_hub_row(
                cast(Mapping[str, object], raw),
                repo_id=repo.repo_id,
                revision=repo.revision,
                config=config,
                split=split,
            )

    return ExampleCohort(sample_ids=sample_ids, factory=factory)


def load_manifest_cohort(path: Path, *, project_root: Path) -> ExampleCohort:
    """Build a lazy local-file cohort from the optional test manifest."""
    rows = load_manifest(path)
    sample_ids = tuple(row.sample_id for row in rows)
    row_by_id = {row.sample_id: row for row in rows}

    def factory(selected: tuple[str, ...]) -> Iterator[DistillExample]:
        for sample_id in selected:
            row = row_by_id[sample_id]
            image_path = Path(row.image_path)
            if not image_path.is_absolute():
                image_path = project_root / image_path
            with Image.open(image_path) as image:
                copied = image.copy()
            yield DistillExample(
                sample_id=row.sample_id,
                gold_diagnosis=row.gold_diagnosis,
                image=copied,
                source_ref=str(image_path),
            )

    return ExampleCohort(sample_ids=sample_ids, factory=factory)


def run_teacher_campaign(
    *,
    teacher: TeacherModel,
    completer: StageCompleter,
    cohort: ExampleCohort,
    stage_a_output: Path,
    stage_b_output: Path,
    limit: int | None = None,
    resume: bool = True,
    progress_stream: TextIO = sys.stderr,
    max_estimated_cost_usd: float | None = None,
) -> CampaignResult:
    """Run Stage A to completion, then Stage B, never retrying inside a run."""
    if stage_a_output.resolve() == stage_b_output.resolve():
        raise ValueError("Stage A and Stage B outputs must be different files")
    started = time.monotonic()
    selected = cohort.select(limit)
    selected_set = set(selected)
    costs = CampaignCostTracker(
        teacher=teacher,
        budget_usd=max_estimated_cost_usd,
    )

    stage_a_before = (
        _compatible_stage_a_ids(
            stage_a_output,
            teacher=teacher,
            selected=selected_set,
        )
        if resume
        else set()
    )
    pending_a = tuple(
        sample_id for sample_id in selected if sample_id not in stage_a_before
    )
    _add_stage_a_costs(costs, stage_a_output, selected_set)
    costs.assert_below_budget()
    progress_a = TerminalProgress(
        stage="Stage A",
        total=len(selected),
        completed=len(selected_set.intersection(stage_a_before)),
        stream=progress_stream,
        cost_tracker=costs,
    )
    try:
        failures_a = run_stage_a(
            teacher=teacher,
            completer=completer,
            examples=cohort.iter_selected(pending_a),
            output_path=stage_a_output,
            resume=False,
            on_record=lambda row: _record_with_cost(progress_a, costs, row),
        )
    except BaseException:
        progress_a.finish("aborted")
        raise
    stage_a_after = _stage_a_ok_ids(stage_a_output)
    missing_a = selected_set.difference(stage_a_after)
    progress_a.finish("complete" if not failures_a and not missing_a else "failed")
    if failures_a or missing_a:
        raise CampaignFailure(
            "Stage A did not complete cleanly; Stage B was not started. "
            f"failures={failures_a}, missing_ok={len(missing_a)}"
        )

    stage_b_before = (
        _compatible_stage_b_ids(
            stage_b_output,
            teacher=teacher,
            selected=selected_set,
        )
        if resume
        else set()
    )
    pending_b = tuple(
        sample_id for sample_id in selected if sample_id not in stage_b_before
    )
    _add_stage_b_costs(costs, stage_b_output, selected_set)
    costs.assert_below_budget()
    stage_b_before_coverage = (
        _stage_b_coverage(stage_b_output)
        if resume
        else _StageBCoverage(frozenset(), frozenset(), frozenset())
    )
    resumed_ok = len(selected_set.intersection(stage_b_before_coverage.ok_ids))
    resumed_rejected = len(
        selected_set.intersection(stage_b_before_coverage.rejected_ids)
    )
    progress_b = TerminalProgress(
        stage="Stage B",
        total=len(selected),
        completed=resumed_ok + resumed_rejected,
        successes=resumed_ok,
        rejections=resumed_rejected,
        stream=progress_stream,
        cost_tracker=costs,
    )
    try:
        failures_b = run_stage_b(
            teacher=teacher,
            completer=completer,
            examples=cohort.iter_selected(pending_b),
            stage_a_path=stage_a_output,
            output_path=stage_b_output,
            resume=False,
            on_record=lambda row: _record_with_cost(progress_b, costs, row),
        )
    except BaseException:
        progress_b.finish("aborted")
        raise
    stage_b_after = _stage_b_coverage(stage_b_output)
    terminal_b = stage_b_after.terminal_ids
    missing_b = selected_set.difference(terminal_b)
    progress_b.finish("complete" if not failures_b and not missing_b else "failed")
    if failures_b or missing_b:
        raise CampaignFailure(
            "Stage B did not complete cleanly. "
            f"retryable_errors={failures_b}, missing_terminal={len(missing_b)}"
        )

    return CampaignResult(
        selected_samples=len(selected),
        stage_a_completed=len(selected_set.intersection(stage_a_after)),
        stage_b_completed=len(selected_set.intersection(terminal_b)),
        stage_b_ok=len(selected_set.intersection(stage_b_after.ok_ids)),
        stage_b_rejected=len(
            selected_set.intersection(stage_b_after.rejected_ids)
        ),
        stage_a_output=stage_a_output,
        stage_b_output=stage_b_output,
        elapsed_seconds=time.monotonic() - started,
        estimated_cost_usd=costs.estimated_usd if costs.enabled else None,
    )


def _record_with_cost(
    progress: TerminalProgress,
    costs: CampaignCostTracker,
    row: StageAFileRow | StageBFileRow,
) -> None:
    """Update durable progress and enforce the soft cap before the next call."""
    costs.add_usage(row.usage)
    progress.record(row.sample_id, row.status)
    costs.assert_below_budget()


def _usage_cost_usd(
    teacher: TeacherModel,
    usage: UsageInfo | None,
) -> float | None:
    if usage is None:
        return None
    if usage.cost is not None and usage.cost_currency in {None, "USD"}:
        return usage.cost
    if teacher.pricing is None:
        return None
    return teacher.pricing.estimate_usage(usage).cost


def _add_stage_a_costs(
    costs: CampaignCostTracker,
    path: Path,
    selected: set[str],
) -> None:
    if not path.is_file():
        return
    for row in load_stage_a_rows(path):
        if row.sample_id in selected:
            costs.add_usage(row.usage)


def _add_stage_b_costs(
    costs: CampaignCostTracker,
    path: Path,
    selected: set[str],
) -> None:
    if not path.is_file():
        return
    for row in load_stage_b_rows(path):
        if row.sample_id in selected:
            costs.add_usage(row.usage)


def _stage_a_ok_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return set(index_ok_stage_a(load_stage_a_rows(path)))


@dataclass(frozen=True, slots=True)
class _StageBCoverage:
    """Final per-sample Stage B outcome with terminal outcomes taking priority."""

    ok_ids: frozenset[str]
    rejected_ids: frozenset[str]
    error_ids: frozenset[str]

    @property
    def terminal_ids(self) -> set[str]:
        """Return accepted and quality-gate-rejected sample IDs."""
        return set(self.ok_ids | self.rejected_ids)


def _stage_b_coverage(path: Path) -> _StageBCoverage:
    """Summarize Stage B rows using ok > rejected > error precedence."""
    if not path.is_file():
        return _StageBCoverage(frozenset(), frozenset(), frozenset())
    ok_ids: set[str] = set()
    rejected_ids: set[str] = set()
    error_ids: set[str] = set()
    for row in load_stage_b_rows(path):
        sample_id = row.sample_id
        if row.status is RecordStatus.OK:
            ok_ids.add(sample_id)
            rejected_ids.discard(sample_id)
            error_ids.discard(sample_id)
        elif row.status is RecordStatus.REJECTED and sample_id not in ok_ids:
            rejected_ids.add(sample_id)
            error_ids.discard(sample_id)
        elif sample_id not in ok_ids and sample_id not in rejected_ids:
            error_ids.add(sample_id)
    return _StageBCoverage(
        ok_ids=frozenset(ok_ids),
        rejected_ids=frozenset(rejected_ids),
        error_ids=frozenset(error_ids),
    )


def _compatible_stage_a_ids(
    path: Path,
    *,
    teacher: TeacherModel,
    selected: set[str],
) -> set[str]:
    """Return resumable Stage A IDs after checking generation provenance."""
    if not path.is_file():
        return set()
    indexed = index_ok_stage_a(load_stage_a_rows(path))
    expected = generation_provenance(teacher, "A")
    _assert_compatible_provenance(
        stage="Stage A",
        path=path,
        rows=(row for sample_id, row in indexed.items() if sample_id in selected),
        expected=expected,
    )
    return set(indexed)


def _compatible_stage_b_ids(
    path: Path,
    *,
    teacher: TeacherModel,
    selected: set[str],
) -> set[str]:
    """Return resumable Stage B IDs after checking generation provenance."""
    if not path.is_file():
        return set()
    terminal_rows = tuple(
        row
        for row in load_stage_b_rows(path)
        if row.status in {RecordStatus.OK, RecordStatus.REJECTED}
        and row.sample_id in selected
    )
    expected = generation_provenance(teacher, "B")
    _assert_compatible_provenance(
        stage="Stage B",
        path=path,
        rows=iter(terminal_rows),
        expected=expected,
    )
    return _stage_b_coverage(path).terminal_ids


def _assert_compatible_provenance(
    *,
    stage: str,
    path: Path,
    rows: Iterator[object],
    expected: GenerationProvenance,
) -> None:
    """Reject resume across any immutable teacher-request protocol boundary."""
    expected_signature = _provenance_signature(expected)
    for row in rows:
        provenance = getattr(row, "provenance", None)
        if not isinstance(provenance, GenerationProvenance):
            raise CampaignFailure(
                f"{stage} resume provenance is missing in {path}; "
                "use new output paths for the current protocol"
            )
        if _provenance_signature(provenance) != expected_signature:
            raise CampaignFailure(
                f"{stage} resume provenance mismatch in {path}; "
                "use new output paths instead of mixing protocols"
            )


def _provenance_signature(provenance: GenerationProvenance) -> tuple[object, ...]:
    """Return the immutable request protocol fields relevant to resume."""
    return (
        provenance.provider,
        provenance.teacher_name,
        provenance.teacher_model,
        provenance.seed,
        provenance.max_output_tokens,
        provenance.reasoning_effort,
        provenance.reasoning_excluded,
        provenance.prompt_sha256,
        provenance.schema_sha256,
    )


def _required_sample_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("Hub sample_id column contains a blank or non-string value")
    return value.strip()


def _short_identifier(value: str, width: int = 28) -> str:
    if len(value) <= width:
        return value
    return "..." + value[-(width - 3) :]


def _format_duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed < float("inf"):
        raise argparse.ArgumentTypeError("must be positive and finite")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the sequential E3 teacher campaign CLI."""
    parser = argparse.ArgumentParser(
        description="Run E3 teacher Stage A then Stage B with live progress."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--hub-config", default="diagnosis")
    parser.add_argument("--hub-split", default="sft_train")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional local-file JSONL. Default is the pinned Hub dataset.",
    )
    parser.add_argument(
        "--stage-a-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "morphology" / "stage_a.jsonl",
    )
    parser.add_argument(
        "--stage-b-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "reasoning" / "stage_b.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="Select the first N source samples as one stable A+B cohort.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Attempt the selected cohort even when accepted rows already exist.",
    )
    parser.add_argument(
        "--max-estimated-cost-usd",
        type=_positive_float,
        default=None,
        help=(
            "Local soft stop based on pinned token list prices. It is not the "
            "finalized invoice or remaining promotional-credit balance."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load source/provider, run the sequential campaign, and exit fail-closed."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # SDK request logs share the terminal with the live progress line. Provider
    # failures still propagate through the campaign's own error handling.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)
    args = parse_args(argv)
    try:
        teacher = TeacherModel.from_yaml(args.config)
        cohort = (
            load_manifest_cohort(args.manifest, project_root=teacher.project_root)
            if args.manifest is not None
            else load_hub_cohort(config=args.hub_config, split=args.hub_split)
        )
        result = run_teacher_campaign(
            teacher=teacher,
            completer=create_teacher_client(teacher),
            cohort=cohort,
            stage_a_output=args.stage_a_output,
            stage_b_output=args.stage_b_output,
            limit=args.limit,
            resume=not args.no_resume,
            max_estimated_cost_usd=args.max_estimated_cost_usd,
        )
    except (
        CampaignFailure,
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        LOGGER.error("E3 teacher campaign failed: %s: %s", type(exc).__name__, exc)
        raise SystemExit(1) from exc

    print(
        "E3 teacher campaign completed: "
        f"samples={result.selected_samples}, "
        f"stage_a={result.stage_a_completed}, "
        f"stage_b_terminal={result.stage_b_completed}, "
        f"stage_b_ok={result.stage_b_ok}, "
        f"stage_b_rejected={result.stage_b_rejected}, "
        f"elapsed={_format_duration(result.elapsed_seconds)}"
    )
    print(f"Stage A: {result.stage_a_output}")
    print(f"Stage B: {result.stage_b_output}")
    if result.estimated_cost_usd is not None:
        print(
            "Estimated request cost (USD list price, before credits): "
            f"${result.estimated_cost_usd:.4f}"
        )


if __name__ == "__main__":
    main()
