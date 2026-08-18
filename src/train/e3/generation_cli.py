"""CLI gates for offline validation, smoke, quality slices, and the E3 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.inference.factory import create_backend
from src.train.e3.generation import E3TeacherGenerationRunner
from src.train.e3.generation_config import (
    E3TeacherGenerationConfig,
    load_e3_teacher_generation_config,
)
from src.train.e3.generation_data import (
    E3Selection,
    E3TeacherSample,
    load_selected_images,
    select_e3_samples,
    selection_manifest,
)
from src.train.e3.prompts import (
    load_stage_a_prompt,
    load_stage_b_prompt,
    prompt_resource_sha256,
    render_stage_a_prompt,
    stage_a_output_schema,
    stage_b_output_schema,
)
from src.train.e3.terminology import terminology_resource_sha256

DEFAULT_CONFIG = Path(
    "configs/training/e3_teacher_generation_gpt_5_6_luna_high_gold_stage_b.yaml"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fail-closed, versioned E3 teacher pipeline.",
    )
    parser.add_argument(
        "mode",
        choices=("dry-run", "smoke", "quality", "pilot"),
        help=(
            "Offline validation, exactly one external case, a bounded quality "
            "slice, or the frozen pilot."
        ),
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--confirm-external-image-upload",
        action="store_true",
        help=(
            "Required for smoke/quality/pilot: confirms private images leave "
            "this machine."
        ),
    )
    parser.add_argument(
        "--approved-smoke",
        type=Path,
        help="Completed compatible one-case campaign required before pilot.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the exact same non-completed campaign without repeating calls.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help=(
            "Dry-run only: validate the first N cases of the frozen selection "
            "without changing the smoke or pilot size."
        ),
    )
    return parser


def _resolve_selection_limit(
    *,
    mode: str,
    requested_limit: int | None,
    pilot_samples: int,
) -> int:
    if requested_limit is not None and mode != "dry-run":
        raise ValueError("--limit is supported only for dry-run")
    if requested_limit is not None:
        if not 1 <= requested_limit <= pilot_samples:
            raise ValueError(
                f"--limit must be between 1 and {pilot_samples} for dry-run"
            )
        return requested_limit
    return 1 if mode == "smoke" else pilot_samples


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_e3_teacher_generation_config(args.config)
    if args.mode == "dry-run" and args.resume:
        raise ValueError("dry-run does not support resume")
    if args.mode != "dry-run" and not args.confirm_external_image_upload:
        raise PermissionError(
            "External E3 image upload requires --confirm-external-image-upload"
        )

    limit = _resolve_selection_limit(
        mode=args.mode,
        requested_limit=args.limit,
        pilot_samples=config.dataset.selection.pilot_samples,
    )
    selection = select_e3_samples(config, limit=limit)
    samples = load_selected_images(
        selection,
        verify_shard_sha256=config.integrity.verify_selected_shard_sha256,
        verify_image_sha256=config.integrity.verify_image_sha256,
    )
    stage_a = load_stage_a_prompt(config.path(config.prompts.stage_a))
    stage_b = load_stage_b_prompt(config.path(config.prompts.stage_b))
    terminology = config.load_terminology()
    render_stage_a_prompt(stage_a, terminology=terminology)
    preflight = _preflight_manifest(
        config,
        selection=selection,
        samples=samples,
        mode=args.mode,
        stage_a_prompt_id=stage_a.prompt_id,
        stage_b_prompt_id=stage_b.prompt_id,
        stage_b_gold_visible_to_teacher=stage_b.gold_visible_to_teacher,
    )
    model = config.load_teacher_model()
    if args.mode == "pilot":
        if args.approved_smoke is None:
            raise PermissionError("pilot requires --approved-smoke")
        _validate_smoke_gate(
            args.approved_smoke,
            config=config,
            stage_a_prompt_id=stage_a.prompt_id,
            stage_b_prompt_id=stage_b.prompt_id,
        )
    _write_or_validate_preflight(
        args.output,
        selection=selection,
        preflight=preflight,
        resume=args.resume,
    )
    if args.mode == "dry-run":
        print(json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    backend = create_backend(
        model,
        reasoning_capture="none",
        use_json_schema=True,
    )
    campaign_id = (
        f"{config.campaign.id}-smoke" if args.mode == "smoke" else config.campaign.id
    )
    snapshot = E3TeacherGenerationRunner(
        config=config,
        selection=selection,
        samples=samples,
        backend=backend,
        stage_a_prompt=stage_a,
        stage_b_prompt=stage_b,
        output_directory=args.output,
        resume=args.resume,
        campaign_id=campaign_id,
        gate_first_case=args.mode == "quality",
    ).run()
    print(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if snapshot["status"] == "completed" else 1


def _preflight_manifest(
    config: E3TeacherGenerationConfig,
    *,
    selection: E3Selection,
    samples: tuple[E3TeacherSample, ...],
    mode: str,
    stage_a_prompt_id: str,
    stage_b_prompt_id: str,
    stage_b_gold_visible_to_teacher: bool,
) -> dict[str, Any]:
    model = config.load_teacher_model()
    terminology = config.load_terminology()
    stage_a = load_stage_a_prompt(config.path(config.prompts.stage_a))
    rendered_stage_a = render_stage_a_prompt(stage_a, terminology=terminology)
    class_counts = Counter(item.candidate.disease_id for item in samples)
    split_counts = Counter(item.candidate.split for item in samples)
    mime_counts = Counter(item.image_mime_type for item in samples)
    return {
        "schema_version": 1,
        "status": "validated_offline",
        "mode": mode,
        "external_calls_made": 0,
        "credentials_checked": False,
        "model": {
            "id": model.model.id,
            "provider_model": model.source.model_name,
            "config_sha256": _sha256_file(model.config_path),
            "api_style": model.backend.active_profile.api_style,
            "reasoning_effort": model.generation.reasoning_effort,
            "structured_output_mode": config.model.structured_output_mode,
        },
        "prompts": {
            "stage_a_id": stage_a_prompt_id,
            "stage_a_resource_sha256": prompt_resource_sha256(
                config.path(config.prompts.stage_a)
            ),
            "stage_a_rendered_sha256": rendered_stage_a.prompt_sha256,
            "stage_b_id": stage_b_prompt_id,
            "stage_b_resource_sha256": prompt_resource_sha256(
                config.path(config.prompts.stage_b)
            ),
        },
        "terminology": {
            "lexicon_id": terminology.lexicon_id,
            "resource_sha256": terminology_resource_sha256(
                config.path(config.terminology.resource)
            ),
            "concepts": len(terminology.concepts),
            "sources": len(terminology.sources),
            "diagnosis_examples_included": False,
            "runtime_web_search": False,
        },
        "schemas": {
            "stage_a_sha256": _canonical_sha256(
                stage_a_output_schema(terminology)
            ),
            "stage_b_sha256": _canonical_sha256(stage_b_output_schema()),
            "strict_required_fields": True,
            "local_fail_closed_validation": True,
        },
        "data": {
            "release_id": selection.release_id,
            "release_manifest_sha256": selection.release_manifest_sha256,
            "selection_sha256": selection.selection_sha256,
            "samples": len(samples),
            "unique_sample_ids": len({item.candidate.sample_id for item in samples}),
            "unique_leakage_groups": len(
                {item.candidate.leakage_group_id for item in samples}
            ),
            "class_counts": dict(sorted(class_counts.items())),
            "split_counts": dict(sorted(split_counts.items())),
            "mime_counts": dict(sorted(mime_counts.items())),
            "embedded_image_bytes": sum(len(item.image_bytes) for item in samples),
            "image_sha256_verified": config.integrity.verify_image_sha256,
            "selected_shard_sha256_verified": (
                config.integrity.verify_selected_shard_sha256
            ),
        },
        "execution": {
            "sequential": config.generation.sequential,
            "retries": config.generation.retries,
            "stop_on_transport_error": config.generation.stop_on_transport_error,
            "raw_prompts_persisted": False,
            "raw_responses_persisted": False,
            "stage_a_gold_visible_to_teacher": False,
            "stage_b_gold_visible_to_teacher": stage_b_gold_visible_to_teacher,
        },
    }


def _write_or_validate_preflight(
    output: Path,
    *,
    selection: E3Selection,
    preflight: dict[str, Any],
    resume: bool,
) -> None:
    root = output.resolve()
    preflight_path = root / "preflight_manifest.json"
    selection_path = root / "selection_manifest.json"
    if resume:
        if not preflight_path.exists() or not selection_path.exists():
            raise FileNotFoundError("E3 resume is missing frozen preflight artifacts")
        existing = _read_object(preflight_path)
        if existing != preflight:
            raise ValueError("E3 preflight identity mismatch on resume")
        existing_selection = _read_object(selection_path)
        if existing_selection.get("selection_sha256") != selection.selection_sha256:
            raise ValueError("E3 frozen selection mismatch on resume")
        return
    if root.exists():
        raise FileExistsError(f"E3 output already exists: {root}")
    root.mkdir(parents=True)
    _atomic_json(preflight_path, preflight)
    _atomic_json(selection_path, selection_manifest(selection))


def _validate_smoke_gate(
    smoke_root: Path,
    *,
    config: E3TeacherGenerationConfig,
    stage_a_prompt_id: str,
    stage_b_prompt_id: str,
) -> None:
    status = _read_object(smoke_root.resolve() / "campaign_status.json")
    manifest = _read_object(smoke_root.resolve() / "campaign_manifest.json")
    if status.get("status") != "completed":
        raise ValueError("E3 smoke gate is not completed")
    campaign = manifest.get("campaign")
    if not isinstance(campaign, dict) or campaign.get("total_samples") != 1:
        raise ValueError("E3 smoke gate must contain exactly one case")
    model = config.load_teacher_model()
    terminology = config.load_terminology()
    stage_a = load_stage_a_prompt(config.path(config.prompts.stage_a))
    stage_b = load_stage_b_prompt(config.path(config.prompts.stage_b))
    expected = {
        "teacher_model": model.source.model_name or model.model.id,
        "model_config_sha256": _sha256_file(model.config_path),
        "reasoning_effort": config.model.reasoning_effort,
        "structured_output_mode": config.model.structured_output_mode,
        "stage_a_prompt_id": stage_a_prompt_id,
        "stage_b_prompt_id": stage_b_prompt_id,
        "stage_a_gold_visible_to_teacher": False,
        "stage_b_gold_visible_to_teacher": stage_b.gold_visible_to_teacher,
        "stage_a_prompt_resource_sha256": prompt_resource_sha256(
            config.path(config.prompts.stage_a)
        ),
        "stage_a_rendered_prompt_sha256": render_stage_a_prompt(
            stage_a,
            terminology=terminology,
        ).prompt_sha256,
        "stage_b_prompt_resource_sha256": prompt_resource_sha256(
            config.path(config.prompts.stage_b)
        ),
        "terminology_lexicon_id": terminology.lexicon_id,
        "terminology_resource_sha256": terminology_resource_sha256(
            config.path(config.terminology.resource)
        ),
    }
    mismatches = tuple(
        key for key, value in expected.items() if campaign.get(key) != value
    )
    if mismatches:
        raise ValueError("E3 smoke gate identity mismatch: " + ", ".join(mismatches))


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
