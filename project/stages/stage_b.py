"""Stage B generation: image + frozen A + gold to differential JSON."""

import argparse
import logging
from collections.abc import Callable, Iterable
from pathlib import Path

from pydantic import ValidationError

from project.dataset.examples import (
    DistillExample,
    examples_from_manifest,
    iter_distill_examples,
)
from project.teacher.client import (
    StageCompleter,
    TeacherCompletionError,
    create_teacher_client,
)
from project.teacher.provenance import generation_provenance
from project.teacher.schemas import (
    GenerationProvenance,
    ImagePreprocessingInfo,
    RecordStatus,
    StageAFileRow,
    StageAMorphology,
    StageBFileRow,
    UsageInfo,
    parse_stage_b,
)
from project.teacher.teacher import DEFAULT_CONFIG, PROJECT_ROOT, TeacherModel
from project.teacher.utils.images import prepare_pil_image
from project.teacher.utils.jsonl import (
    append_jsonl,
    completed_stage_b_ids,
    index_ok_stage_a,
    load_stage_a_rows,
)
from project.teacher.validate import validate_stage_b

LOGGER = logging.getLogger("project.stages")


def build_stage_b_messages(
    teacher: TeacherModel,
    image_data_url: str,
    morphology: StageAMorphology,
    gold_diagnosis: str,
) -> list[dict[str, object]]:
    """Build OpenRouter messages for Stage B.

    Args:
        teacher: Loaded teacher config.
        image_data_url: JPEG data URL.
        morphology: Frozen Stage A record, dumped into the user prompt.
        gold_diagnosis: Private anchor copied by the teacher.

    Returns:
        System and user messages with image, gold, and Stage A JSON.
    """
    stage = teacher.stage("B")
    user_text = stage.prompt.render_user(
        gold_diagnosis=gold_diagnosis,
        stage_a_json=morphology.model_dump_json(),
    )
    return [
        {"role": "system", "content": stage.prompt.system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url, "detail": "auto"},
                },
            ],
        },
    ]


def generate_reasoning(
    completer: StageCompleter,
    teacher: TeacherModel,
    example: DistillExample,
    stage_a: StageAFileRow,
    image_data_url: str,
    image_preprocessing: ImagePreprocessingInfo | None = None,
) -> StageBFileRow:
    """Call the teacher, parse Stage B, and apply validation gates.

    Args:
        completer: HTTP client or test fake.
        teacher: Loaded config.
        example: Hub example that supplies gold_diagnosis and source_ref.
        stage_a: Frozen ok Stage A row.
        image_data_url: Encoded image.
        image_preprocessing: Hash-addressed manifest of the encoded image.

    Returns:
        ok, rejected, or error row. Does not rewrite Stage A.
    """
    if stage_a.morphology is None:
        return _error_row(teacher, example, "missing_stage_a_morphology")

    messages = build_stage_b_messages(
        teacher,
        image_data_url,
        stage_a.morphology,
        example.gold_diagnosis,
    )

    usage = None
    try:
        response = completer.complete_stage("B", messages)
        usage = response.usage
        parsed = parse_stage_b(response.content_json)
    except (TeacherCompletionError, ValidationError, TypeError, ValueError) as exc:
        LOGGER.exception("Stage B failed for %s", example.sample_id)
        if isinstance(exc, TeacherCompletionError) and exc.usage is not None:
            usage = exc.usage
        return _error_row(
            teacher,
            example,
            _short_error(exc),
            usage=usage,
            image_preprocessing=image_preprocessing,
            provenance=generation_provenance(teacher, "B"),
        )

    check = validate_stage_b(stage_a.morphology, parsed, example.gold_diagnosis)
    status = RecordStatus.OK if check.ok else RecordStatus.REJECTED
    return StageBFileRow(
        sample_id=example.sample_id,
        status=status,
        reasoning=parsed,
        reasons=check.reasons,
        error=None,
        usage=response.usage,
        teacher=teacher.name,
        gold_diagnosis=example.gold_diagnosis,
        stage_a_sample_id=stage_a.sample_id,
        image_path=example.source_ref,
        image_preprocessing=image_preprocessing,
        provenance=generation_provenance(teacher, "B", response=response),
    )


def run_stage_b(
    *,
    teacher: TeacherModel,
    completer: StageCompleter,
    examples: Iterable[DistillExample],
    stage_a_path: Path,
    output_path: Path,
    limit: int | None = None,
    resume: bool = True,
    on_record: Callable[[StageBFileRow], None] | None = None,
) -> int:
    """Generate Stage B for examples that have an ok Stage A row.

    Args:
        teacher: Loaded YAML config.
        completer: Stage B completer.
        examples: Same Hub/manifest stream as Stage A.
        stage_a_path: Stage A JSONL.
        output_path: Stage B JSONL.
        limit: Cap on new attempts after resume.
        resume: Skip ids already terminal (ok or rejected) in ``output_path``.
        on_record: Optional observer called after each durable JSONL append.

    Returns:
        Count of error rows written. Rejected output is a terminal, auditable
        quality-gate outcome and is not retried automatically.
    """
    stage_a_ok = index_ok_stage_a(load_stage_a_rows(stage_a_path))
    skip = completed_stage_b_ids(output_path) if resume else set()
    failures = 0
    attempted = 0

    for example in examples:
        if example.sample_id in skip:
            continue
        if limit is not None and attempted >= limit:
            break

        attempted += 1
        stage_a = stage_a_ok.get(example.sample_id)
        if stage_a is None:
            record = _error_row(
                teacher,
                example,
                "missing_ok_stage_a_record",
            )
        else:
            record = _generate_one(teacher, completer, example, stage_a)
        append_jsonl(output_path, record)
        if on_record is not None:
            on_record(record)
        if record.status is RecordStatus.ERROR:
            failures += 1

    return failures


def _generate_one(
    teacher: TeacherModel,
    completer: StageCompleter,
    example: DistillExample,
    stage_a: StageAFileRow,
) -> StageBFileRow:
    """Encode the PIL image and run Stage B for one joined sample."""
    try:
        prepared = prepare_pil_image(example.image)
    except (TypeError, ValueError, OSError) as exc:
        LOGGER.exception("Stage B image failed for %s", example.sample_id)
        return _error_row(teacher, example, _short_error(exc))

    return generate_reasoning(
        completer,
        teacher,
        example,
        stage_a,
        prepared.data_url,
        prepared.info,
    )


def _error_row(
    teacher: TeacherModel,
    example: DistillExample,
    error: str,
    usage: UsageInfo | None = None,
    image_preprocessing: ImagePreprocessingInfo | None = None,
    provenance: GenerationProvenance | None = None,
) -> StageBFileRow:
    """Build a Stage B error record."""
    return StageBFileRow(
        sample_id=example.sample_id,
        status=RecordStatus.ERROR,
        reasoning=None,
        reasons=(),
        error=error,
        usage=usage,
        teacher=teacher.name,
        gold_diagnosis=example.gold_diagnosis,
        stage_a_sample_id=example.sample_id,
        image_path=example.source_ref,
        image_preprocessing=image_preprocessing,
        provenance=provenance,
    )


def _short_error(exc: BaseException) -> str:
    """Format an error without secrets or data URLs."""
    return f"{type(exc).__name__}: {exc}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Stage B CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate Stage B reasoning JSONL.")
    parser.add_argument(
        "--stage-a",
        type=Path,
        default=PROJECT_ROOT / "data" / "morphology" / "stage_a.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "reasoning" / "stage_b.jsonl",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--hub-config", default="diagnosis")
    parser.add_argument("--hub-split", default="sft_train")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSONL of local files. Default is ISEPDistillDataset.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip sample ids already terminal (ok or rejected).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load the Hub dataset (or a manifest), run Stage B, exit 1 on errors."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    teacher = TeacherModel.from_yaml(args.config)
    completer = create_teacher_client(teacher)
    if args.manifest is not None:
        examples: Iterable[DistillExample] = examples_from_manifest(
            args.manifest,
            project_root=teacher.project_root,
        )
    else:
        examples = iter_distill_examples(
            config=args.hub_config,
            split=args.hub_split,
        )
    failures = run_stage_b(
        teacher=teacher,
        completer=completer,
        examples=examples,
        stage_a_path=args.stage_a,
        output_path=args.output,
        limit=args.limit,
        resume=not args.no_resume,
    )
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
