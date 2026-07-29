"""Deterministic end-to-end smoke test for the confusion-set benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from src.benchmark.metrics import compute_confusion_set_metrics
from src.benchmark.runner import BenchmarkRunner, BenchmarkSample
from src.data_pipeline.common import load_yaml
from src.data_pipeline.confusion_sets import (
    validate_confusion_set_release,
)
from src.data_pipeline.deduplication import ImageResolver


class CandidateSchemaSmokeBackend:
    """Rank the runtime schema candidates in their supplied order."""

    @property
    def model_id(self) -> str:
        return "candidate_schema_smoke_backend"

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        schema: dict[str, Any],
    ) -> str:
        if not system_prompt or not user_prompt or not image_bytes:
            raise ValueError("Smoke-test inputs must be non-empty")
        candidate_ids = schema["properties"]["predictions"]["items"][
            "properties"
        ]["disease_id"]["enum"]
        return json.dumps(
            {
                "predictions": [
                    {"rank": rank, "disease_id": disease_id}
                    for rank, disease_id in enumerate(
                        candidate_ids,
                        start=1,
                    )
                ]
            }
        )


def run_confusion_smoke_test(root: Path) -> dict[str, Any]:
    """Run both difficulty conditions for one paired benchmark image."""

    release = validate_confusion_set_release(root)
    benchmark = load_yaml(
        root / "configs/benchmarks/visual_confusion_sets.yaml"
    )
    prompt = load_yaml(root / benchmark["prompt"]["path"])
    taxonomy = load_yaml(
        root / benchmark["taxonomy"]["disease"]["path"]
    )
    schema = json.loads(
        (root / benchmark["schema"]["path"]).read_text(encoding="utf-8")
    )
    manifest = pq.read_table(
        root / benchmark["dataset"]["task_manifest"]
    ).to_pandas()
    first_pair_id = str(manifest.iloc[0]["pair_id"])
    pair = manifest[manifest["pair_id"] == first_pair_id]
    if len(pair) != 2:
        raise ValueError("Smoke-test pair must contain two tasks")
    taxonomy_items = [
        {"id": item["id"], "display_name": item["display_name"]}
        for item in taxonomy["diseases"]
    ]
    samples = [
        BenchmarkSample(
            task_id=str(row.task_id),
            sample_id=str(row.sample_id),
            image_uri=str(row.image_uri),
            disease_id=str(row.disease_id),
            candidate_disease_ids=tuple(
                str(value)
                for value in row.candidate_disease_ids
            ),
            metadata={
                "pair_id": str(row.pair_id),
                "difficulty": str(row.difficulty),
                "confusion_set_id": str(row.confusion_set_id),
                "candidate_disease_ids": [
                    str(value)
                    for value in row.candidate_disease_ids
                ],
            },
        )
        for row in pair.itertuples(index=False)
    ]
    with ImageResolver(root) as resolver:
        runner = BenchmarkRunner(
            backend=CandidateSchemaSmokeBackend(),
            system_prompt=prompt["system_prompt"],
            user_prompt_template=prompt["user_template"],
            schema=schema,
            taxonomy_items=taxonomy_items,
            top_k=int(benchmark["benchmark"]["ranking_count"]),
            image_loader=resolver.read_bytes,
        )
        predictions = runner.run(samples)
    if not all(item.response.schema_valid for item in predictions):
        raise ValueError("Confusion-set smoke output failed validation")
    metrics = compute_confusion_set_metrics(
        predictions,
        allowed_disease_ids=[
            item["id"]
            for item in taxonomy_items
        ],
        bootstrap_resamples=100,
    )

    output_directory = root / "outputs/smoke/visual_confusion_sets_v1"
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / "smoke_report.yaml"
    with report_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "smoke_test": {
                    "status": "passed",
                    "benchmark_release": {
                        "id": release["id"],
                        "version": release["version"],
                    },
                    "pair_id": first_pair_id,
                    "task_count": len(predictions),
                    "model_id": predictions[0].model_id,
                    "metrics": metrics,
                }
            },
            handle,
            sort_keys=False,
            allow_unicode=False,
        )
    return {
        "report_path": report_path,
        "metrics": metrics,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    result = run_confusion_smoke_test(root)
    print("Confusion-set smoke test passed")
    print(f"report_path: {result['report_path'].relative_to(root)}")


if __name__ == "__main__":
    main()
