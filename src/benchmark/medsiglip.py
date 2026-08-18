"""Deterministic MedSigLIP evaluation helpers for disease ranking.

MedSigLIP is a dual-encoder rather than a generative model.  This module
therefore ranks the frozen disease-label prompts by image/text cosine
similarity and converts that ranking into the canonical prediction objects
used by the existing ISEPDermaBench scorers.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from src.benchmark.runner import (
    BenchmarkPrediction,
    BenchmarkSample,
    ModelResponse,
)

MODEL_ID = "google/medsiglip-448"
MODEL_REVISION = "9cea28a1a1195f665105faa6e8544c112fd960a4"
MODEL_OUTPUT_ID = "google_medsiglip_448_zero_shot"
DEFAULT_PROMPT_TEMPLATE = "a clinical photograph of {display_name}"
KNOWN_PRETRAINING_SOURCE_OVERLAP = frozenset({"scin", "pad_ufes_20"})
LOWER_KNOWN_OVERLAP_SOURCES = frozenset({"fitzpatrick17k_c"})
MODEL_REQUIRED_FILES = frozenset(
    {
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)


@dataclass(frozen=True, slots=True)
class DiseaseLabel:
    """One frozen disease label and its text prompt."""

    disease_id: str
    display_name: str
    prompt: str


def load_disease_labels(
    taxonomy_path: Path,
    *,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> tuple[DiseaseLabel, ...]:
    """Load and validate the active disease taxonomy in declared order."""

    document = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("diseases"), list):
        raise ValueError("Disease taxonomy must contain a diseases list")
    if "{display_name}" not in prompt_template:
        raise ValueError("Prompt template must contain {display_name}")
    labels: list[DiseaseLabel] = []
    seen: set[str] = set()
    for index, item in enumerate(document["diseases"]):
        if not isinstance(item, dict):
            raise ValueError(f"Disease taxonomy item {index} is not a mapping")
        disease_id = item.get("id")
        display_name = item.get("display_name")
        if not isinstance(disease_id, str) or not disease_id:
            raise ValueError(f"Disease taxonomy item {index} has no valid id")
        if not isinstance(display_name, str) or not display_name:
            raise ValueError(f"Disease taxonomy item {index} has no display_name")
        if disease_id in seen:
            raise ValueError(f"Duplicate disease ID: {disease_id}")
        seen.add(disease_id)
        labels.append(
            DiseaseLabel(
                disease_id=disease_id,
                display_name=display_name,
                prompt=prompt_template.format(display_name=display_name),
            )
        )
    if not labels:
        raise ValueError("Disease taxonomy is empty")
    return tuple(labels)


def rank_candidates(
    *,
    candidate_ids: Sequence[str],
    scores_by_id: Mapping[str, float],
    ranking_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return canonical ranks plus auditable scores with stable tie-breaking."""

    candidates = list(candidate_ids)
    if not candidates or len(candidates) != len(set(candidates)):
        raise ValueError("Candidate disease IDs must be non-empty and unique")
    if ranking_count <= 0 or ranking_count > len(candidates):
        raise ValueError("ranking_count is outside the candidate range")
    missing = [value for value in candidates if value not in scores_by_id]
    if missing:
        raise ValueError("Missing scores for: " + ", ".join(missing))
    candidate_order = {value: index for index, value in enumerate(candidates)}
    ranked_ids = sorted(
        candidates,
        key=lambda value: (-float(scores_by_id[value]), candidate_order[value]),
    )[:ranking_count]
    predictions = [
        {"rank": rank, "disease_id": disease_id}
        for rank, disease_id in enumerate(ranked_ids, start=1)
    ]
    scored = [
        {
            "rank": rank,
            "disease_id": disease_id,
            "cosine_similarity": float(scores_by_id[disease_id]),
        }
        for rank, disease_id in enumerate(ranked_ids, start=1)
    ]
    return predictions, scored


def build_prediction(
    *,
    sample: BenchmarkSample,
    model_id: str,
    scores_by_id: Mapping[str, float],
    ranking_count: int,
    label_prompts: Mapping[str, str],
) -> tuple[BenchmarkPrediction, dict[str, Any]]:
    """Create one scorer-compatible prediction from deterministic scores."""

    candidates = tuple(sample.candidate_disease_ids or scores_by_id)
    output, scored = rank_candidates(
        candidate_ids=candidates,
        scores_by_id=scores_by_id,
        ranking_count=ranking_count,
    )
    parsed = {"predictions": output}
    raw_text = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    response = ModelResponse(
        model_id=model_id,
        raw_text=raw_text,
        parsed_output=parsed,
        json_valid=True,
        schema_valid=True,
        recoverable_json_valid=True,
        canonical_output=parsed,
        canonical_schema_valid=True,
        metadata={
            "inference_type": "image_text_similarity_ranking",
            "output_contract_metrics_applicable": False,
            "scores": scored,
            "candidate_prompts": {
                disease_id: label_prompts[disease_id] for disease_id in candidates
            },
        },
    )
    prediction = BenchmarkPrediction(
        sample_id=sample.sample_id,
        task_id=sample.task_id or sample.sample_id,
        model_id=model_id,
        ground_truth_disease_id=sample.disease_id,
        response=response,
        metadata=dict(sample.metadata),
    )
    record = {
        "task_id": prediction.task_id,
        "sample_id": prediction.sample_id,
        "model_id": prediction.model_id,
        "status": "ok",
        "image_uri": sample.image_uri,
        "ground_truth_disease_id": prediction.ground_truth_disease_id,
        "metadata": prediction.metadata,
        "response": {
            "final_text": raw_text,
            "parsed_output": parsed,
            "canonical_output": parsed,
            "json_valid": True,
            "recoverable_json_valid": True,
            "schema_valid": True,
            "canonical_schema_valid": True,
            "canonicalization_rules": [],
            "validation_errors": [],
            "metadata": response.metadata,
            "reasoning": {
                "capture_mode": "none",
                "availability": "not_applicable",
                "text": None,
                "token_count": None,
                "source": None,
            },
            "usage": {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "reasoning_tokens": None,
            },
            "finish_reason": "deterministic_ranking",
            "provider_response_id": None,
            "provider_metadata": {},
        },
    }
    return prediction, record


def validate_sample_images(samples: Iterable[BenchmarkSample]) -> None:
    """Verify embedded image availability and frozen benchmark hashes."""

    for sample in samples:
        if not sample.image_bytes:
            raise ValueError(f"Task {sample.task_id} has no embedded image")
        expected = sample.metadata.get("benchmark_image_sha256")
        actual = sha256(sample.image_bytes).hexdigest()
        if isinstance(expected, str) and expected and actual != expected:
            raise ValueError(
                f"Task {sample.task_id} image hash mismatch: {actual} != {expected}"
            )


def source_overlap_summary(
    predictions: Iterable[BenchmarkPrediction],
) -> dict[str, Any]:
    """Count disclosed source-level pretraining overlap, without overclaiming."""

    rows = list(predictions)
    by_source: defaultdict[str, list[BenchmarkPrediction]] = defaultdict(list)
    for row in rows:
        by_source[_normalize_source(row.metadata.get("source"))].append(row)
    overlap = [
        row
        for row in rows
        if _normalize_source(row.metadata.get("source"))
        in KNOWN_PRETRAINING_SOURCE_OVERLAP
    ]
    lower = [
        row
        for row in rows
        if _normalize_source(row.metadata.get("source"))
        in LOWER_KNOWN_OVERLAP_SOURCES
    ]
    return {
        "task_count": len(rows),
        "unique_image_count": len({row.sample_id for row in rows}),
        "known_source_overlap_task_count": len(overlap),
        "known_source_overlap_unique_image_count": len(
            {row.sample_id for row in overlap}
        ),
        "known_source_overlap_task_rate": (len(overlap) / len(rows) if rows else 0.0),
        "lower_known_overlap_task_count": len(lower),
        "lower_known_overlap_unique_image_count": len({row.sample_id for row in lower}),
        "by_source": {
            source: {
                "task_count": len(values),
                "unique_image_count": len({row.sample_id for row in values}),
                "disclosed_direct_dataset_overlap": (
                    source in KNOWN_PRETRAINING_SOURCE_OVERLAP
                ),
            }
            for source, values in sorted(by_source.items())
        },
        "interpretation": (
            "Source-level overlap does not prove exact-image memorization. "
            "The lower-known-overlap stratum is not contamination-free."
        ),
    }


def stratified_metrics(
    predictions: Iterable[BenchmarkPrediction],
    *,
    metric_fn: Any,
) -> dict[str, Any]:
    """Compute the native metric bundle per source and lower-overlap stratum."""

    rows = list(predictions)
    by_source: defaultdict[str, list[BenchmarkPrediction]] = defaultdict(list)
    for row in rows:
        by_source[_normalize_source(row.metadata.get("source"))].append(row)
    lower = [
        row
        for row in rows
        if _normalize_source(row.metadata.get("source"))
        in LOWER_KNOWN_OVERLAP_SOURCES
    ]
    return {
        "by_source": {
            source: metric_fn(values) for source, values in sorted(by_source.items())
        },
        "lower_known_overlap": metric_fn(lower) if lower else None,
        "lower_known_overlap_sources": sorted(LOWER_KNOWN_OVERLAP_SOURCES),
    }


def _normalize_source(value: Any) -> str:
    """Canonicalize dataset source names used by overlap strata."""

    normalized = str(value or "unknown").strip().casefold()
    return normalized or "unknown"


def resolve_cached_snapshot(
    *,
    model_id: str,
    revision: str,
    cache_root: Path | None = None,
) -> Path:
    """Resolve a complete immutable Hub snapshot without network access."""

    if cache_root is None:
        configured = os.environ.get("HF_HUB_CACHE")
        cache_root = (
            Path(configured) if configured else Path.home() / ".cache/huggingface/hub"
        )
    repository = "models--" + model_id.replace("/", "--")
    snapshot = cache_root / repository / "snapshots" / revision
    missing = sorted(
        filename
        for filename in MODEL_REQUIRED_FILES
        if not (snapshot / filename).is_file()
    )
    if missing:
        raise FileNotFoundError(
            f"Incomplete cached snapshot {snapshot}: missing " + ", ".join(missing)
        )
    return snapshot.resolve()


def pooled_feature_tensor(output: Any) -> Any:
    """Extract projected pooled features across Transformers API versions."""

    pooler_output = getattr(output, "pooler_output", None)
    if pooler_output is not None:
        return pooler_output
    if hasattr(output, "float"):
        return output
    if isinstance(output, (list, tuple)) and output:
        candidate = output[1] if len(output) > 1 else output[0]
        if hasattr(candidate, "float"):
            return candidate
    raise TypeError(
        "Model feature output has neither a tensor nor pooler_output: "
        f"{type(output).__name__}"
    )


class MedSigLIPEmbedder:
    """Lazy Transformers wrapper with deterministic, normalized embeddings."""

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        revision: str = MODEL_REVISION,
        device: str = "auto",
        local_files_only: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("MedSigLIP requires torch and transformers") from exc
        self.torch = torch
        self.device = self._resolve_device(device)
        model_source: str | Path = model_id
        effective_revision: str | None = revision
        if local_files_only:
            model_source = resolve_cached_snapshot(
                model_id=model_id,
                revision=revision,
            )
            effective_revision = None
        self.processor = AutoProcessor.from_pretrained(
            model_source,
            revision=effective_revision,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model_source,
            revision=effective_revision,
            local_files_only=local_files_only,
            dtype=torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()

    def _resolve_device(self, requested: str) -> Any:
        torch = self.torch
        if requested == "auto":
            if torch.cuda.is_available():
                requested = "cuda"
            elif torch.backends.mps.is_available():
                requested = "mps"
            else:
                requested = "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device(requested)

    def encode_texts(self, texts: Sequence[str]) -> Any:
        """Encode and L2-normalize the frozen label prompts."""

        torch = self.torch
        values = self.processor(
            text=list(texts),
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        values = {key: value.to(self.device) for key, value in values.items()}
        with torch.inference_mode():
            output = self.model.get_text_features(**values)
            features = pooled_feature_tensor(output)
        return torch.nn.functional.normalize(features.float(), dim=-1).cpu()

    def encode_images(self, image_bytes: Sequence[bytes]) -> Any:
        """Decode, encode, and L2-normalize a batch of benchmark images."""

        torch = self.torch
        images = []
        for value in image_bytes:
            with Image.open(BytesIO(value)) as image:
                images.append(image.convert("RGB"))
        values = self.processor(images=images, return_tensors="pt")
        pixel_values = values["pixel_values"].to(self.device)
        with torch.inference_mode():
            output = self.model.get_image_features(pixel_values=pixel_values)
            features = pooled_feature_tensor(output)
        return torch.nn.functional.normalize(features.float(), dim=-1).cpu()
