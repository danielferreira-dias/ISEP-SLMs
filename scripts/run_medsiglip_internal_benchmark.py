#!/usr/bin/env python3
"""Run zero-shot MedSigLIP on the disease-ranking Internal Benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from src.benchmark.executor import _record_to_prediction
from src.benchmark.isep_dermabench import (
    load_isep_dermabench_config,
    load_isep_dermabench_dataset,
)
from src.benchmark.medsiglip import (
    DEFAULT_PROMPT_TEMPLATE,
    MODEL_ID,
    MODEL_OUTPUT_ID,
    MODEL_REVISION,
    MedSigLIPEmbedder,
    build_prediction,
    load_disease_labels,
    source_overlap_summary,
    stratified_metrics,
    validate_sample_images,
)
from src.benchmark.task_adapters import build_task_adapter

BENCHMARKS = (
    ("visual_top_k_closed_set", "visual_top_k", 6),
    ("visual_disease_confusion_sets", "visual_confusion_sets", 3),
)
EXPECTED_FULL_COUNTS = {
    "visual_top_k_closed_set": 1000,
    "visual_disease_confusion_sets": 828,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--stage",
        choices=("smoke", "gate", "full"),
        help="smoke=1 unit, gate=10 units, full=all frozen tasks",
    )
    action.add_argument(
        "--rescore-run",
        type=Path,
        help="Recompute reports from a completed run without model inference",
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "mps", "cuda")
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/medsiglip_internal_benchmark"),
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Permit Transformers to access the Hub; default is cache-only",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.rescore_run is not None:
        run_directory = args.rescore_run
        if not run_directory.is_absolute():
            run_directory = root / run_directory
        return _rescore_existing_run(run_directory.resolve(), root=root)
    if args.stage is None:
        raise AssertionError("stage is required for inference")
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = root / output_root
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_directory = output_root / args.stage / run_id
    suffix = 2
    while run_directory.exists():
        run_directory = output_root / args.stage / f"{run_id}_{suffix}"
        suffix += 1
    (run_directory / "predictions").mkdir(parents=True, exist_ok=False)
    (run_directory / "metrics").mkdir()
    manifest_path = run_directory / "campaign_manifest.json"
    started = time.monotonic()
    manifest: dict[str, Any] = {
        "status": "running",
        "stage": args.stage,
        "started_at": _utc_now(),
        "model": {
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "output_id": MODEL_OUTPUT_ID,
            "architecture": "SigLIP dual encoder",
            "dtype": "float32",
        },
        "protocol": {
            "seed": args.seed,
            "prompt_template": args.prompt_template,
            "prompt_selection": "a_priori_not_tuned_on_internal_benchmark",
            "image_size": 448,
            "text_max_tokens": 64,
            "scoring": "L2-normalized image/text cosine similarity",
            "sampling": "not_applicable_deterministic_ranking",
            "output_contract_metrics_applicable": False,
        },
        "data_policy": {
            "benchmark_source": "local_frozen_release",
            "external_data_upload": False,
            "judge_api": False,
        },
        "command": " ".join(sys.argv),
    }
    _write_json(manifest_path, manifest)
    try:
        labels = load_disease_labels(
            root / "data/benchmarks/ISEPDermaBench/artifacts/taxonomies/diseases.yaml",
            prompt_template=args.prompt_template,
        )
        label_ids = [label.disease_id for label in labels]
        label_prompts = {label.disease_id: label.prompt for label in labels}
        datasets: dict[str, Any] = {}
        adapters: dict[str, Any] = {}
        limits = {"smoke": 1, "gate": 10, "full": None}
        for benchmark_id, release_key, _ in BENCHMARKS:
            config = load_isep_dermabench_config(benchmark_id, root=root)
            dataset = load_isep_dermabench_dataset(
                root=root,
                benchmark=config,
                evaluation_set="internal_benchmark",
                limit=limits[args.stage],
                seed=args.seed,
                source="local",
            )
            validate_sample_images(dataset.samples)
            _validate_candidates(dataset.samples, benchmark_id, label_ids)
            if (
                args.stage == "full"
                and len(dataset.samples) != EXPECTED_FULL_COUNTS[benchmark_id]
            ):
                raise ValueError(
                    f"{benchmark_id} expected "
                    f"{EXPECTED_FULL_COUNTS[benchmark_id]} tasks, "
                    f"found {len(dataset.samples)}"
                )
            datasets[benchmark_id] = dataset
            raw_benchmark = _read_yaml(config.config_path)
            prompt = _read_yaml(config.prompt_path)
            schema = json.loads(config.schema_path.read_text(encoding="utf-8"))
            taxonomy = _read_yaml(config.taxonomy.disease_path)
            adapters[benchmark_id] = build_task_adapter(
                benchmark_config=raw_benchmark,
                prompt_config=prompt,
                schema=schema,
                disease_taxonomy_items=taxonomy["diseases"],
            )
            _write_json(
                run_directory / f"selection_{release_key}.json",
                dataset.selection,
            )

        embedder = MedSigLIPEmbedder(
            device=args.device,
            local_files_only=not args.allow_model_download,
        )
        manifest["runtime"] = {
            "device": str(embedder.device),
            "batch_size": args.batch_size,
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
        _write_json(manifest_path, manifest)
        text_features = embedder.encode_texts([label.prompt for label in labels])

        unique_images: dict[str, bytes] = {}
        for dataset in datasets.values():
            for sample in dataset.samples:
                image_hash = str(sample.metadata["benchmark_image_sha256"])
                value = sample.image_bytes
                if value is None:
                    raise ValueError(f"Task {sample.task_id} has no image")
                unique_images.setdefault(image_hash, value)
        image_features: dict[str, Any] = {}
        image_items = list(unique_images.items())
        for index in range(0, len(image_items), args.batch_size):
            batch = image_items[index : index + args.batch_size]
            features = embedder.encode_images([value for _, value in batch])
            for (image_hash, _), feature in zip(batch, features, strict=True):
                image_features[image_hash] = feature
            completed = min(index + len(batch), len(image_items))
            print(
                f"images {completed}/{len(image_items)} "
                f"({100.0 * completed / len(image_items):.1f}%)",
                flush=True,
            )

        campaign_metrics: dict[str, Any] = {}
        contamination: dict[str, Any] = {}
        counts: dict[str, int] = {}
        for benchmark_id, release_key, ranking_count in BENCHMARKS:
            dataset = datasets[benchmark_id]
            predictions = []
            records = []
            for sample in dataset.samples:
                image_hash = str(sample.metadata["benchmark_image_sha256"])
                similarities = image_features[image_hash] @ text_features.T
                scores = {
                    disease_id: float(similarities[position].item())
                    for position, disease_id in enumerate(label_ids)
                }
                prediction, record = build_prediction(
                    sample=sample,
                    model_id=MODEL_OUTPUT_ID,
                    scores_by_id=scores,
                    ranking_count=ranking_count,
                    label_prompts=label_prompts,
                )
                predictions.append(prediction)
                records.append(record)
            adapter = adapters[benchmark_id]
            native_metrics = adapter.compute_metrics(predictions)
            stratified = stratified_metrics(
                predictions,
                metric_fn=adapter.compute_metrics,
            )
            metrics = {
                "benchmark_id": benchmark_id,
                "native_metrics": native_metrics,
                "stratified_metrics": stratified,
                "output_contract_metrics_note": (
                    "JSON/schema fields are internal serialization and are not "
                    "comparable to generative-model output reliability."
                ),
            }
            _write_jsonl(
                run_directory / "predictions" / f"{release_key}.jsonl",
                records,
            )
            _write_json(
                run_directory / "metrics" / f"{release_key}.json",
                metrics,
            )
            campaign_metrics[benchmark_id] = metrics
            contamination[benchmark_id] = source_overlap_summary(predictions)
            counts[benchmark_id] = len(predictions)

        _write_json(run_directory / "metrics.json", campaign_metrics)
        _write_json(run_directory / "source_overlap_audit.json", contamination)
        manifest.update(
            {
                "status": "completed",
                "finished_at": _utc_now(),
                "duration_seconds": time.monotonic() - started,
                "counts": counts,
                "unique_image_count": len(unique_images),
                "artifacts": {
                    "metrics": "metrics.json",
                    "source_overlap_audit": "source_overlap_audit.json",
                    "predictions": "predictions/",
                },
            }
        )
        _write_json(manifest_path, manifest)
        _write_checksums(run_directory)
        print(json.dumps({"run_directory": str(run_directory), **counts}, indent=2))
        return 0
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "duration_seconds": time.monotonic() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json(manifest_path, manifest)
        raise


def _rescore_existing_run(run_directory: Path, *, root: Path) -> int:
    """Rebuild deterministic reports while preserving the original versions."""

    manifest_path = run_directory / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Only a completed MedSigLIP run can be rescored")
    if manifest.get("model", {}).get("output_id") != MODEL_OUTPUT_ID:
        raise ValueError("Run model ID does not match the MedSigLIP protocol")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    history = run_directory / "reporting_history" / timestamp
    history.mkdir(parents=True, exist_ok=False)
    preserved = (
        manifest_path,
        run_directory / "checksums.sha256",
        run_directory / "metrics.json",
        run_directory / "metrics" / "visual_top_k.json",
        run_directory / "metrics" / "visual_confusion_sets.json",
        run_directory / "source_overlap_audit.json",
    )
    for path in preserved:
        if not path.is_file():
            raise FileNotFoundError(path)
        shutil.copy2(path, history / path.name)

    campaign_metrics: dict[str, Any] = {}
    overlap_audit: dict[str, Any] = {}
    for benchmark_id, release_key, _ in BENCHMARKS:
        records = _read_jsonl(
            run_directory / "predictions" / f"{release_key}.jsonl"
        )
        expected = int(manifest["counts"][benchmark_id])
        if len(records) != expected:
            raise ValueError(
                f"{benchmark_id} has {len(records)} predictions; expected {expected}"
            )
        predictions = [_record_to_prediction(record) for record in records]
        config = load_isep_dermabench_config(benchmark_id, root=root)
        adapter = build_task_adapter(
            benchmark_config=_read_yaml(config.config_path),
            prompt_config=_read_yaml(config.prompt_path),
            schema=json.loads(config.schema_path.read_text(encoding="utf-8")),
            disease_taxonomy_items=_read_yaml(config.taxonomy.disease_path)[
                "diseases"
            ],
        )
        metrics = {
            "benchmark_id": benchmark_id,
            "native_metrics": adapter.compute_metrics(predictions),
            "stratified_metrics": stratified_metrics(
                predictions,
                metric_fn=adapter.compute_metrics,
            ),
            "output_contract_metrics_note": (
                "JSON/schema fields are internal serialization and are not "
                "comparable to generative-model output reliability."
            ),
        }
        campaign_metrics[benchmark_id] = metrics
        overlap_audit[benchmark_id] = source_overlap_summary(predictions)
        _write_json(run_directory / "metrics" / f"{release_key}.json", metrics)

    _write_json(run_directory / "metrics.json", campaign_metrics)
    _write_json(run_directory / "source_overlap_audit.json", overlap_audit)
    corrections = manifest.setdefault("postprocessing_corrections", [])
    corrections.append(
        {
            "applied_at": _utc_now(),
            "command": " ".join(sys.argv),
            "inference_repeated": False,
            "reason": (
                "Normalize benchmark source identifiers case-insensitively for "
                "source-overlap and lower-known-overlap reports."
            ),
            "preserved_original_reports": str(history.relative_to(run_directory)),
        }
    )
    _write_json(manifest_path, manifest)
    _write_checksums(run_directory)
    print(
        json.dumps(
            {
                "run_directory": str(run_directory),
                "inference_repeated": False,
                "reporting_history": str(history),
            },
            indent=2,
        )
    )
    return 0


def _validate_candidates(samples: Any, benchmark_id: str, label_ids: list[str]) -> None:
    expected = 21 if benchmark_id == "visual_top_k_closed_set" else 3
    allowed = set(label_ids)
    for sample in samples:
        candidates = tuple(sample.candidate_disease_ids or ())
        if len(candidates) != expected or len(candidates) != len(set(candidates)):
            raise ValueError(
                f"Task {sample.task_id} expected {expected} unique candidates"
            )
        if not set(candidates) <= allowed:
            raise ValueError(f"Task {sample.task_id} contains unknown candidates")
        if sample.disease_id not in candidates:
            raise ValueError(f"Task {sample.task_id} reference is not a candidate")


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        records.append(value)
    return records


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_checksums(run_directory: Path) -> None:
    checksum_path = run_directory / "checksums.sha256"
    rows = []
    for path in sorted(run_directory.rglob("*")):
        if path.is_file() and path != checksum_path:
            rows.append(
                f"{sha256(path.read_bytes()).hexdigest()}  "
                f"{path.relative_to(run_directory)}"
            )
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
