"""Stage A generation: image only to morphology JSON."""

import argparse
import logging
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from project.dataset.examples import (
    DistillExample,
    examples_from_manifest,
    iter_distill_examples,
)
from project.teacher.client import StageCompleter, TeacherClient, TeacherCompletionError
from project.teacher.schemas import (
    ImageSample,
    RecordStatus,
    StageAFileRow,
    parse_stage_a,
)
from project.teacher.teacher import DEFAULT_CONFIG, PROJECT_ROOT, TeacherModel
from project.teacher.utils.images import encode_pil_image_data_url
from project.teacher.utils.jsonl import append_jsonl, completed_ids

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
    examples: Iterable[DistillExample],
    output_path: Path,
    limit: int | None = None,
    resume: bool = True,
) -> int:
    """Generate Stage A for each remaining example.

    Resume skips sample ids that already have ``status=ok``. ``--limit``
    counts remaining work, not the first N examples. Gold on the example
    is not sent to the teacher.

    Args:
        teacher: Loaded YAML config.
        completer: Stage A completer.
        examples: Hub or manifest examples.
        output_path: Destination JSONL.
        limit: Optional cap on new attempts.
        resume: Skip ids already ok in ``output_path``.

    Returns:
        Count of rows written with ``status=error``.
    """
    skip = completed_ids(output_path) if resume else set()
    failures = 0
    attempted = 0

    for example in examples:
        if example.sample_id in skip:
            continue
        if limit is not None and attempted >= limit:
            break

        attempted += 1
        record = _generate_one(teacher, completer, example)
        append_jsonl(output_path, record)
        if record.status is RecordStatus.ERROR:
            failures += 1

    return failures


def _generate_one(
    teacher: TeacherModel,
    completer: StageCompleter,
    example: DistillExample,
) -> StageAFileRow:
    """Encode one PIL image and run Stage A. Gold is not passed through."""
    sample = ImageSample(
        sample_id=example.sample_id,
        image_path=Path(example.source_ref),
    )

    try:
        image_data_url = encode_pil_image_data_url(example.image)
    except (TypeError, ValueError, OSError) as exc:
        LOGGER.exception("Stage A image failed for %s", example.sample_id)
        return StageAFileRow(
            sample_id=example.sample_id,
            status=RecordStatus.ERROR,
            morphology=None,
            error=_short_error(exc),
            usage=None,
            teacher=teacher.name,
            image_path=example.source_ref,
        )

    return generate_morphology(completer, teacher, sample, image_data_url)


def _short_error(exc: BaseException) -> str:
    """Format an error without secrets or data URLs."""
    return f"{type(exc).__name__}: {exc}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Stage A CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate Stage A morphology JSONL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "morphology" / "stage_a.jsonl",
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
        help="Do not skip sample ids already marked ok.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load the Hub dataset (or a manifest), run Stage A, exit 1 on errors."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    teacher = TeacherModel.from_yaml(args.config)
    completer = TeacherClient(teacher)
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
    failures = run_stage_a(
        teacher=teacher,
        completer=completer,
        examples=examples,
        output_path=args.output,
        limit=args.limit,
        resume=not args.no_resume,
    )
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
