"""Stage A generation: image only to morphology JSON."""

import argparse
import logging
from pathlib import Path

from pydantic import ValidationError

from project.teacher.client import StageCompleter, TeacherClient, TeacherCompletionError
from project.teacher.utils.images import encode_image_data_url
from project.teacher.utils.jsonl import append_jsonl, completed_ids, load_manifest
from project.teacher.schemas import (
    ImageSample,
    ManifestRow,
    RecordStatus,
    StageAFileRow,
    image_sample_from_manifest,
    parse_stage_a,
)
from project.teacher.teacher import DEFAULT_CONFIG, TeacherModel

LOGGER = logging.getLogger("project.stages")


def build_stage_a_messages(
    teacher: TeacherModel,
    image_data_url: str,
) -> list[dict[str, object]]:
    """Build OpenRouter messages for Stage A.

    The gold diagnosis is not a parameter and must not appear in the payload.

    Args:
        teacher: Loaded teacher config.
        image_data_url: JPEG data URL for the sample image.

    Returns:
        System and user messages, with the image on the user turn.
    """
    stage = teacher.stage("A")
    return [
        {"role": "system", "content": stage.prompt.system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": stage.prompt.user},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url, "detail": "auto"},
                },
            ],
        },
    ]


def generate_morphology(
    completer: StageCompleter,
    teacher: TeacherModel,
    sample: ImageSample,
    image_data_url: str,
) -> StageAFileRow:
    """Call the teacher and parse Stage A JSON.

    Args:
        completer: HTTP client or test fake.
        teacher: Loaded config (name stored on the row).
        sample: Image identity without gold.
        image_data_url: Encoded image.

    Returns:
        An ok row or an error row. Never includes gold_diagnosis.
    """
    messages = build_stage_a_messages(teacher, image_data_url)
    image_path = str(sample.image_path)

    try:
        response = completer.complete_stage("A", messages)
        morphology = parse_stage_a(response.content_json)
    except (TeacherCompletionError, ValidationError, TypeError, ValueError) as exc:
        LOGGER.exception("Stage A failed for %s", sample.sample_id)
        return StageAFileRow(
            sample_id=sample.sample_id,
            status=RecordStatus.ERROR,
            morphology=None,
            error=_short_error(exc),
            usage=None,
            teacher=teacher.name,
            image_path=image_path,
        )

    return StageAFileRow(
        sample_id=sample.sample_id,
        status=RecordStatus.OK,
        morphology=morphology,
        error=None,
        usage=response.usage,
        teacher=teacher.name,
        image_path=image_path,
    )


def run_stage_a(
    *,
    teacher: TeacherModel,
    completer: StageCompleter,
    manifest_path: Path,
    output_path: Path,
    limit: int | None = None,
    resume: bool = True,
) -> int:
    """Generate Stage A for each remaining manifest row.

    Resume skips sample ids that already have ``status=ok``. ``--limit``
    counts remaining work, not the first N manifest lines.

    Args:
        teacher: Loaded YAML config.
        completer: Stage A completer.
        manifest_path: Input JSONL.
        output_path: Destination JSONL.
        limit: Optional cap on new attempts.
        resume: Skip ids already ok in ``output_path``.

    Returns:
        Count of rows written with ``status=error``.
    """
    rows = load_manifest(manifest_path)
    skip = completed_ids(output_path) if resume else set()
    failures = 0
    attempted = 0

    for row in rows:
        if row.sample_id in skip:
            continue
        if limit is not None and attempted >= limit:
            break

        attempted += 1
        record = _generate_one(teacher, completer, row)
        append_jsonl(output_path, record)
        if record.status is RecordStatus.ERROR:
            failures += 1

    return failures


def _generate_one(
    teacher: TeacherModel,
    completer: StageCompleter,
    row: ManifestRow,
) -> StageAFileRow:
    """Encode one image and run Stage A. Gold on ``row`` is not passed through."""
    sample = image_sample_from_manifest(row, project_root=teacher.project_root)

    try:
        image_data_url = encode_image_data_url(sample.image_path)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.exception("Stage A image failed for %s", sample.sample_id)
        return StageAFileRow(
            sample_id=sample.sample_id,
            status=RecordStatus.ERROR,
            morphology=None,
            error=_short_error(exc),
            usage=None,
            teacher=teacher.name,
            image_path=str(sample.image_path),
        )

    return generate_morphology(completer, teacher, sample, image_data_url)


def _short_error(exc: BaseException) -> str:
    """Format an error without secrets or data URLs."""
    return f"{type(exc).__name__}: {exc}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Stage A CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate Stage A morphology JSONL.")
    parser.add_argument("--manifest", type=Path, required=True)
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
    """Load config, run Stage A, exit 1 if any sample failed."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    teacher = TeacherModel.from_yaml(args.config)
    completer = TeacherClient(teacher)
    failures = run_stage_a(
        teacher=teacher,
        completer=completer,
        manifest_path=args.manifest,
        output_path=args.output,
        limit=args.limit,
        resume=not args.no_resume,
    )
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
