"""Command-line entry point for the initial student visual-attribution pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np

from src.vision_analysis.cases import load_pilot_config, load_visual_cases
from src.vision_analysis.occlusion import (
    compute_score_drop_map,
    iter_tile_records,
)
from src.vision_analysis.qwen import QwenVisualScorer
from src.vision_analysis.render import render_signed_overlay
from src.vision_analysis.report import write_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a frozen Qwen visual occlusion-attribution pilot."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs/vision_analysis/student_visual_attribution_pilot_v1.yaml",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        default="mps" if platform.system() == "Darwin" else "cuda",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default=None,
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate frozen cases without loading the model.",
    )
    args = parser.parse_args()
    config = load_pilot_config(args.config, project_root=PROJECT_ROOT)
    cases = load_visual_cases(config)
    if args.max_cases is not None:
        if args.max_cases < 1:
            parser.error("--max-cases must be positive")
        cases = cases[: args.max_cases]
    if args.validate_only:
        print(
            json.dumps(
                {
                    "analysis_id": config.analysis_id,
                    "status": "validated",
                    "cases": [case.task_id for case in cases],
                },
                indent=2,
            )
        )
        return
    dtype = args.dtype or ("float16" if args.device == "mps" else "bfloat16")
    output = args.output or PROJECT_ROOT / "outputs/vision_analysis" / config.analysis_id
    output.mkdir(parents=True, exist_ok=True)
    scorer = QwenVisualScorer(
        config.model_config_path,
        device=args.device,
        dtype=dtype,
        revision=config.model_revision,
    )
    scorer.load()
    report_cases: list[dict[str, Any]] = []
    manifest_cases: list[dict[str, Any]] = []
    for case in cases:
        case_directory = output / "cases" / case.task_id / "E0_base"
        case_directory.mkdir(parents=True, exist_ok=True)
        case.image.save(case_directory / "original.jpg", quality=95, optimize=True)
        target_results: dict[str, dict[str, Any]] = {}
        for target_kind, disease_id, disease_name in (
            ("gold", case.gold_disease_id, case.gold_disease_name),
            (
                "benchmark_predicted",
                case.predicted_disease_id,
                case.predicted_disease_name,
            ),
        ):
            target_text = disease_id

            def score(image):
                return scorer.score_label(
                    image,
                    system_prompt=config.attribution_system_prompt,
                    user_prompt=config.attribution_user_prompt,
                    target=target_text,
                ).mean_log_probability

            result = compute_score_drop_map(
                case.image,
                score,
                grid_size=config.grid_size,
                blur_radius=max(case.image.size)
                * config.blur_radius_fraction,
            )
            target_directory = case_directory / target_kind
            target_directory.mkdir(parents=True, exist_ok=True)
            np.save(target_directory / "attribution.npy", result.score_drops)
            overlay = render_signed_overlay(case.image, result.signed_importance)
            overlay.save(target_directory / "overlay.png")
            metadata = {
                "method": "blurred_patch_occlusion",
                "target_kind": target_kind,
                "target_disease_id": disease_id,
                "target_disease_name": disease_name,
                "baseline_mean_log_probability": result.baseline_score,
                "grid_size": config.grid_size,
                "blur_radius_fraction": config.blur_radius_fraction,
                "tiles": list(iter_tile_records(result)),
            }
            (target_directory / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            target_results[target_kind] = {
                "overlay": overlay,
                "score": result.baseline_score,
                "metadata": metadata,
            }
        report_cases.append(
            {
                "task_id": case.task_id,
                "cohort": case.cohort,
                "source": case.source,
                "skin_tone": case.skin_tone,
                "age_group": case.age_group,
                "gold_label": f"{case.gold_disease_id} · {case.gold_disease_name}",
                "predicted_label": (
                    f"{case.predicted_disease_id} · {case.predicted_disease_name}"
                ),
                "gold_score": target_results["gold"]["score"],
                "predicted_score": target_results["benchmark_predicted"]["score"],
                "original_image": case.image,
                "gold_overlay": target_results["gold"]["overlay"],
                "predicted_overlay": target_results["benchmark_predicted"][
                    "overlay"
                ],
            }
        )
        manifest_cases.append(
            {
                "task_id": case.task_id,
                "cohort": case.cohort,
                "source": case.source,
                "skin_tone": case.skin_tone,
                "age_group": case.age_group,
                "gold_disease_id": case.gold_disease_id,
                "predicted_disease_id": case.predicted_disease_id,
                "source_image_sha256": case.source_image_sha256,
                "benchmark_image_sha256": case.benchmark_image_sha256,
                "gold_baseline_mean_log_probability": target_results["gold"][
                    "score"
                ],
                "predicted_baseline_mean_log_probability": target_results[
                    "benchmark_predicted"
                ]["score"],
            }
        )
    write_report(
        output / "report.html",
        analysis_id=config.analysis_id,
        model_name=scorer.repo_id,
        cases=report_cases,
    )
    manifest = {
        "schema_version": 1,
        "analysis_id": config.analysis_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "model": {
            "repo_id": scorer.repo_id,
            "revision": scorer.revision,
            "processor_repo_id": scorer.processor_repo_id,
            "processor_revision": scorer.processor_revision,
            "device": args.device,
            "dtype": dtype,
        },
        "method": {
            "id": "blurred_patch_occlusion",
            "grid_size": config.grid_size,
            "blur_radius_fraction": config.blur_radius_fraction,
            "score": "mean teacher-forced target-token log_probability",
            "prompt_sha256": sha256(
                (
                    config.attribution_system_prompt
                    + "\n"
                    + config.attribution_user_prompt
                ).encode("utf-8")
            ).hexdigest(),
        },
        "cases": manifest_cases,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(output / "report.html")


if __name__ == "__main__":
    main()
