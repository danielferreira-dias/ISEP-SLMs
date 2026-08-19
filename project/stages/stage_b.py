"""Stage B generation: image + frozen A + gold to differential JSON."""

import argparse
import logging
from pathlib import Path

from pydantic import ValidationError

from project.teacher.client import StageCompleter, TeacherClient, TeacherCompletionError
from project.teacher.utils.images import encode_image_data_url
from project.teacher.utils.jsonl import (
    append_jsonl,
    completed_ids,
    index_ok_stage_a,
    load_manifest,
    load_stage_a_rows,
)
from project.teacher.schemas import (
    ManifestRow,
    RecordStatus,
    StageAFileRow,
    StageAMorphology,
    StageBFileRow,
    UsageInfo,
    parse_stage_b,
)
from project.teacher.teacher import DEFAULT_CONFIG, TeacherModel
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
    row: ManifestRow,
    stage_a: StageAFileRow,
    image_data_url: str,
) -> StageBFileRow:
    """Call the teacher, parse Stage B, and apply validation gates.

    Args:
        completer: HTTP client or test fake.
        teacher: Loaded config.
        row: Manifest row that supplies gold_diagnosis.
        stage_a: Frozen ok Stage A row.
        image_data_url: Encoded image.

    Returns:
        ok, rejected, or error row. Does not rewrite Stage A.
    """
    if stage_a.morphology is None:
        return _error_row(teacher, row, "missing_stage_a_morphology")

    messages = build_stage_b_messages(
        teacher,
        image_data_url,
        stage_a.morphology,
        row.gold_diagnosis,
    )

    try:
        response = completer.complete_stage("B", messages)
        parsed = parse_stage_b(response.content_json)
    except (TeacherCompletionError, ValidationError, TypeError, ValueError) as exc:
        LOGGER.exception("Stage B failed for %s", row.sample_id)
        return _error_row(teacher, row, _short_error(exc), usage=None)

    check = validate_stage_b(stage_a.morphology, parsed, row.gold_diagnosis)
    status = RecordStatus.OK if check.ok else RecordStatus.REJECTED
    return StageBFileRow(
        sample_id=row.sample_id,
        status=status,
        reasoning=parsed,
        reasons=check.reasons,
        error=None,
        usage=response.usage,
        teacher=teacher.name,
        gold_diagnosis=row.gold_diagnosis,
        stage_a_sample_id=stage_a.sample_id,
        image_path=str(row.image_path),
    )


def run_stage_b(
    *,
    teacher: TeacherModel,
    completer: StageCompleter,
    manifest_path: Path,
    stage_a_path: Path,
    output_path: Path,
    limit: int | None = None,
    resume: bool = True,
) -> int:
    """Generate Stage B for samples that have an ok Stage A row.

    Args:
        teacher: Loaded YAML config.
        completer: Stage B completer.
        manifest_path: Same manifest as Stage A.
        stage_a_path: Stage A JSONL.
        output_path: Stage B JSONL.
        limit: Cap on new attempts after resume.
        resume: Skip ids already ok in ``output_path``.

    Returns:
        Count of error rows written (rejected does not count as process failure).
    """
    manifest = {row.sample_id: row for row in load_manifest(manifest_path)}
    stage_a_ok = index_ok_stage_a(load_stage_a_rows(stage_a_path))
    skip = completed_ids(output_path) if resume else set()
    failures = 0
    attempted = 0

    for sample_id, stage_a in stage_a_ok.items():
        if sample_id in skip:
            continue
        row = manifest.get(sample_id)
        if row is None:
            LOGGER.error("Stage A sample %s is missing from the manifest", sample_id)
            continue
        if limit is not None and attempted >= limit:
            break

        attempted += 1
        record = _generate_one(teacher, completer, row, stage_a)
        append_jsonl(output_path, record)
        if record.status is RecordStatus.ERROR:
            failures += 1

    return failures


def _generate_one(
    teacher: TeacherModel,
    completer: StageCompleter,
    row: ManifestRow,
    stage_a: StageAFileRow,
) -> StageBFileRow:
    """Encode the image and run Stage B for one joined sample."""
    image_path = Path(row.image_path)
    if not image_path.is_absolute():
        image_path = teacher.project_root / image_path

    try:
        image_data_url = encode_image_data_url(image_path)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.exception("Stage B image failed for %s", row.sample_id)
        return _error_row(teacher, row, _short_error(exc))

    return generate_reasoning(completer, teacher, row, stage_a, image_data_url)


def _error_row(
    teacher: TeacherModel,
    row: ManifestRow,
    error: str,
    usage: UsageInfo | None = None,
) -> StageBFileRow:
    """Build a Stage B error record."""
    return StageBFileRow(
        sample_id=row.sample_id,
        status=RecordStatus.ERROR,
        reasoning=None,
        reasons=(),
        error=error,
        usage=usage,
        teacher=teacher.name,
        gold_diagnosis=row.gold_diagnosis,
        stage_a_sample_id=row.sample_id,
        image_path=row.image_path,
    )


def _short_error(exc: BaseException) -> str:
    """Format an error without secrets or data URLs."""
    return f"{type(exc).__name__}: {exc}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Stage B CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate Stage B reasoning JSONL.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage-a", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip sample ids already marked ok.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load config, run Stage B, exit 1 if any sample errored."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    teacher = TeacherModel.from_yaml(args.config)
    completer = TeacherClient(teacher)
    failures = run_stage_b(
        teacher=teacher,
        completer=completer,
        manifest_path=args.manifest,
        stage_a_path=args.stage_a,
        output_path=args.output,
        limit=args.limit,
        resume=not args.no_resume,
    )
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
