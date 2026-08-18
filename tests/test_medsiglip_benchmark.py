from __future__ import annotations

from pathlib import Path

import pytest

from src.benchmark.medsiglip import (
    DEFAULT_PROMPT_TEMPLATE,
    build_prediction,
    load_disease_labels,
    pooled_feature_tensor,
    rank_candidates,
    resolve_cached_snapshot,
    source_overlap_summary,
    stratified_metrics,
)
from src.benchmark.runner import BenchmarkSample

ROOT = Path(__file__).resolve().parents[1]


def test_loads_frozen_21_label_taxonomy() -> None:
    labels = load_disease_labels(
        ROOT / "data/benchmarks/ISEPDermaBench/artifacts/taxonomies/diseases.yaml"
    )

    assert len(labels) == 21
    assert labels[0].disease_id == "D001"
    assert labels[0].prompt == "a clinical photograph of Melanoma"
    assert DEFAULT_PROMPT_TEMPLATE == "a clinical photograph of {display_name}"


def test_rank_candidates_is_descending_and_stable_on_ties() -> None:
    predictions, scored = rank_candidates(
        candidate_ids=("D002", "D001", "D003"),
        scores_by_id={"D001": 0.5, "D002": 0.5, "D003": 0.4},
        ranking_count=3,
    )

    assert [row["disease_id"] for row in predictions] == [
        "D002",
        "D001",
        "D003",
    ]
    assert [row["rank"] for row in scored] == [1, 2, 3]


def test_rank_candidates_rejects_missing_score() -> None:
    with pytest.raises(ValueError, match="Missing scores"):
        rank_candidates(
            candidate_ids=("D001", "D002"),
            scores_by_id={"D001": 0.5},
            ranking_count=2,
        )


def test_resolves_complete_local_snapshot_without_hub_lookup(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "models--google--medsiglip-448" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    for filename in (
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        (snapshot / filename).write_text("value", encoding="utf-8")

    resolved = resolve_cached_snapshot(
        model_id="google/medsiglip-448",
        revision="revision",
        cache_root=tmp_path,
    )

    assert resolved == snapshot.resolve()


def test_extracts_pooler_output_across_transformers_versions() -> None:
    class Output:
        pooler_output = "pooled"

    assert pooled_feature_tensor(Output()) == "pooled"


def test_prediction_is_native_scorer_compatible_and_auditable() -> None:
    sample = BenchmarkSample(
        sample_id="sample-1",
        task_id="task-1",
        image_uri="embedded://task-1",
        image_bytes=b"image",
        disease_id="D002",
        candidate_disease_ids=("D001", "D002", "D003"),
        metadata={"source": "SCIN"},
    )
    prediction, record = build_prediction(
        sample=sample,
        model_id="medsiglip-test",
        scores_by_id={"D001": 0.2, "D002": 0.9, "D003": 0.1},
        ranking_count=3,
        label_prompts={
            "D001": "prompt 1",
            "D002": "prompt 2",
            "D003": "prompt 3",
        },
    )

    assert prediction.response.is_valid
    assert prediction.response.parsed_output == {
        "predictions": [
            {"rank": 1, "disease_id": "D002"},
            {"rank": 2, "disease_id": "D001"},
            {"rank": 3, "disease_id": "D003"},
        ]
    }
    assert record["response"]["metadata"]["scores"][0]["cosine_similarity"] == 0.9
    assert not record["response"]["metadata"]["output_contract_metrics_applicable"]


def test_source_overlap_summary_does_not_claim_exact_overlap() -> None:
    predictions = []
    for index, source in enumerate(("SCIN", "PAD_UFES_20", "Fitzpatrick17k_C")):
        sample = BenchmarkSample(
            sample_id=f"sample-{index}",
            task_id=f"task-{index}",
            image_uri=f"embedded://{index}",
            disease_id="D001",
            candidate_disease_ids=("D001",),
            metadata={"source": source},
        )
        prediction, _ = build_prediction(
            sample=sample,
            model_id="medsiglip-test",
            scores_by_id={"D001": 1.0},
            ranking_count=1,
            label_prompts={"D001": "prompt"},
        )
        predictions.append(prediction)

    summary = source_overlap_summary(predictions)

    assert summary["known_source_overlap_task_count"] == 2
    assert summary["lower_known_overlap_task_count"] == 1
    assert "does not prove exact-image memorization" in summary["interpretation"]


def test_source_strata_are_case_insensitive() -> None:
    predictions = []
    for index, source in enumerate(("scin", "PAD_UFES_20", "fitzpatrick17k_c")):
        sample = BenchmarkSample(
            sample_id=f"sample-{index}",
            task_id=f"task-{index}",
            image_uri=f"embedded://{index}",
            disease_id="D001",
            candidate_disease_ids=("D001",),
            metadata={"source": source},
        )
        prediction, _ = build_prediction(
            sample=sample,
            model_id="medsiglip-test",
            scores_by_id={"D001": 1.0},
            ranking_count=1,
            label_prompts={"D001": "prompt"},
        )
        predictions.append(prediction)

    summary = source_overlap_summary(predictions)
    stratified = stratified_metrics(predictions, metric_fn=lambda rows: len(rows))

    assert summary["known_source_overlap_task_count"] == 2
    assert summary["lower_known_overlap_task_count"] == 1
    assert stratified["lower_known_overlap"] == 1
