"""Deterministic end-to-end smoke test for the frozen benchmark release."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from src.benchmark.metrics import compute_metrics
from src.benchmark.runner import BenchmarkRunner, BenchmarkSample
from src.data_pipeline.common import load_yaml
from src.data_pipeline.deduplication import ImageResolver
from src.data_pipeline.splitting import validate_benchmark_release


class DeterministicSmokeBackend:
    """Backend that emits one known-valid response without model inference."""

    def __init__(self, ranked_ids: list[str]) -> None:
        self.ranked_ids = ranked_ids

    @property
    def model_id(self) -> str:
        return "deterministic_smoke_backend"

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        schema: dict[str, Any],
    ) -> str:
        if not system_prompt or not user_prompt or not image_bytes or not schema:
            raise ValueError("Smoke-test inputs must be non-empty")
        return json.dumps(
            {
                "predictions": [
                    {"rank": rank, "disease_id": disease_id}
                    for rank, disease_id in enumerate(
                        self.ranked_ids,
                        start=1,
                    )
                ]
            }
        )


def run_smoke_test(root: Path) -> dict[str, Any]:
    """Load one frozen sample, render prompts, validate output, and score it."""

    release = validate_benchmark_release(root)
    benchmark = load_yaml(root / "configs/benchmarks/visual_top_k.yaml")
    prompt = load_yaml(root / benchmark["prompt"]["path"])
    taxonomy = load_yaml(root / benchmark["taxonomy"]["path"])
    schema = json.loads(
        (root / benchmark["schema"]["path"]).read_text(encoding="utf-8")
    )
    manifest_path = root / benchmark["dataset"]["manifest"]
    frame = pq.read_table(manifest_path).slice(0, 1).to_pandas()
    if frame.empty:
        raise ValueError("Internal test manifest is empty")
    row = frame.iloc[0]
    taxonomy_items = [
        {"id": item["id"], "display_name": item["display_name"]}
        for item in taxonomy["diseases"]
    ]
    allowed_ids = [item["id"] for item in taxonomy_items]
    ranked_ids = [
        str(row["disease_id"]),
        *[
            disease_id
            for disease_id in allowed_ids
            if disease_id != row["disease_id"]
        ][:5],
    ]
    with ImageResolver(root) as resolver:
        runner = BenchmarkRunner(
            backend=DeterministicSmokeBackend(ranked_ids),
            system_prompt=prompt["system_prompt"],
            user_prompt_template=prompt["user_template"],
            schema=schema,
            taxonomy_items=taxonomy_items,
            top_k=int(benchmark["benchmark"]["top_k"]),
            image_loader=resolver.read_bytes,
        )
        predictions = runner.run(
            [
                BenchmarkSample(
                    sample_id=str(row["sample_id"]),
                    image_uri=str(row["image_uri"]),
                    disease_id=str(row["disease_id"]),
                    metadata={
                        "dataset_id": str(row["dataset_id"]),
                        "split": str(row["split"]),
                    },
                )
            ]
        )
    metrics = compute_metrics(
        predictions,
        allowed_disease_ids=allowed_ids,
    )
    output_directory = (
        root / "outputs/smoke/visual_top_k_v1"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    prediction_path = output_directory / "smoke_predictions.jsonl"
    metrics_path = output_directory / "smoke_metrics.json"
    report_path = output_directory / "smoke_report.yaml"
    prediction = predictions[0]
    with prediction_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "sample_id": prediction.sample_id,
                    "model_id": prediction.model_id,
                    "ground_truth_disease_id": (
                        prediction.ground_truth_disease_id
                    ),
                    "raw_text": prediction.response.raw_text,
                    "parsed_output": prediction.response.parsed_output,
                    "json_valid": prediction.response.json_valid,
                    "schema_valid": prediction.response.schema_valid,
                    "validation_errors": (
                        prediction.response.validation_errors
                    ),
                },
                sort_keys=True,
            )
            + "\n"
        )
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with report_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "smoke_test": {
                    "status": (
                        "passed"
                        if prediction.response.schema_valid
                        else "failed"
                    ),
                    "benchmark_release": {
                        "id": release["id"],
                        "version": release["version"],
                    },
                    "sample_id": prediction.sample_id,
                    "model_id": prediction.model_id,
                    "metrics": metrics,
                }
            },
            handle,
            sort_keys=False,
            allow_unicode=False,
        )
    return {
        "prediction_path": prediction_path,
        "metrics_path": metrics_path,
        "report_path": report_path,
        "metrics": metrics,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    result = run_smoke_test(root)
    print("Smoke test passed")
    for name in ["prediction_path", "metrics_path", "report_path"]:
        print(f"{name}: {result[name].relative_to(root)}")


if __name__ == "__main__":
    main()
