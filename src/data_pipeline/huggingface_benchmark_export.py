"""Export the executable ISEP dermatology benchmarks to Hugging Face.

The release deliberately separates model inputs from scoring references.
Every task row embeds the exact benchmark-preprocessed image, rendered prompts,
and response schema. Matching ``*_references`` configurations contain the
gold labels and evaluation-only metadata keyed by ``task_id``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from datasets import Dataset, Features, Image as HFImage, Sequence, Value
from PIL import Image
import pyarrow.parquet as pq
import yaml

from src.benchmark.datasets import load_benchmark_dataset
from src.benchmark.images import prepare_benchmark_image
from src.benchmark.task_adapters import build_task_adapter
from src.config.benchmarks import load_benchmark_config
from src.data_pipeline.deduplication import ImageResolver


EXPORT_SCHEMA_VERSION = "1.0.0"
EXPORT_RELEASE_VERSION = "1.0.0"
DEFAULT_OUTPUT = Path("data/benchmarks/ISEPDermaBench")
DEFAULT_SHARD_SIZE = 512


@dataclass(frozen=True, slots=True)
class EvaluationSpec:
    """One source evaluation set mapped to a public Hub split name."""

    split: str
    evaluation_set: str


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """One benchmark task and the evaluation sets included in the release."""

    key: str
    config_path: Path
    evaluations: tuple[EvaluationSpec, ...]


BENCHMARK_SPECS = (
    BenchmarkSpec(
        key="visual_top_k",
        config_path=Path(
            "configs/benchmarks/derma_isep/visual_top_k.yaml"
        ),
        evaluations=(
            EvaluationSpec("validation", "validation"),
            EvaluationSpec(
                "internal_benchmark",
                "internal_benchmark_1000",
            ),
            EvaluationSpec("external_ddi", "external_ddi"),
            EvaluationSpec(
                "external_skindisnet",
                "external_skindisnet",
            ),
        ),
    ),
    BenchmarkSpec(
        key="visual_confusion_sets",
        config_path=Path(
            "configs/benchmarks/derma_isep/visual_confusion_sets.yaml"
        ),
        evaluations=(
            EvaluationSpec(
                "validation",
                "validation_paired_confusion_tasks",
            ),
            EvaluationSpec(
                "internal_benchmark",
                "paired_confusion_tasks",
            ),
        ),
    ),
    BenchmarkSpec(
        key="evidence_grounded_diagnosis",
        config_path=Path(
            "configs/benchmarks/derma_isep/"
            "evidence_grounded_diagnosis.yaml"
        ),
        evaluations=(
            EvaluationSpec(
                "validation",
                "validation_fitzpatrick_evidence",
            ),
            EvaluationSpec(
                "internal_benchmark",
                "internal_benchmark_evidence",
            ),
            EvaluationSpec(
                "external_ddi",
                "external_ddi_evidence",
            ),
        ),
    ),
)


TASK_FEATURES = Features(
    {
        "image": HFImage(decode=True),
        "task_id": Value("string"),
        "sample_id": Value("string"),
        "benchmark_id": Value("string"),
        "benchmark_version": Value("string"),
        "evaluation_set": Value("string"),
        "source": Value("string"),
        "leakage_group_id": Value("string"),
        "system_prompt": Value("string"),
        "user_prompt": Value("string"),
        "response_schema_json": Value("string"),
        "prompt_id": Value("string"),
        "prompt_version": Value("string"),
        "top_k": Value("int16"),
        "candidate_disease_ids": Sequence(Value("string")),
        "pair_id": Value("string"),
        "condition": Value("string"),
        "confusion_set_id": Value("string"),
        "prompt_sha256": Value("string"),
        "response_schema_sha256": Value("string"),
        "benchmark_config_sha256": Value("string"),
        "taxonomy_sha256": Value("string"),
        "source_image_sha256": Value("string"),
        "benchmark_image_sha256": Value("string"),
        "image_preprocessing_profile": Value("string"),
        "license_id": Value("string"),
    }
)


REFERENCE_FEATURES = Features(
    {
        "task_id": Value("string"),
        "sample_id": Value("string"),
        "benchmark_id": Value("string"),
        "evaluation_set": Value("string"),
        "source": Value("string"),
        "leakage_group_id": Value("string"),
        "reference_disease_id": Value("string"),
        "reference_diagnoses_json": Value("string"),
        "diagnosis_basis": Value("string"),
        "morphology_concept_ids": Sequence(Value("string")),
        "reference_clinical_description": Value("string"),
        "score_morphology": Value("bool"),
        "score_description": Value("bool"),
        "score_diagnosis": Value("bool"),
        "pair_id": Value("string"),
        "condition": Value("string"),
        "confusion_set_id": Value("string"),
        "age_years": Value("int32"),
        "age_group_standardized": Value("string"),
        "skin_tone_system": Value("string"),
        "skin_tone": Value("string"),
        "sex_or_gender_system": Value("string"),
        "sex_or_gender": Value("string"),
        "race_ethnicity": Value("string"),
        "license_id": Value("string"),
    }
)


METADATA_MANIFESTS = (
    Path(
        "data/benchmarks/derma_isep/visual_top_k_v1/datasets/"
        "internal/validation.parquet"
    ),
    Path(
        "data/benchmarks/derma_isep/visual_top_k_v1/datasets/"
        "internal/internal_benchmark_1000.parquet"
    ),
    Path(
        "data/benchmarks/derma_isep/visual_top_k_v1/datasets/"
        "external/external_ddi.parquet"
    ),
    Path(
        "data/benchmarks/derma_isep/visual_top_k_v1/datasets/"
        "external/external_skindisnet.parquet"
    ),
    Path("data/manifests/ddi_v3.parquet"),
)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _metadata_index(root: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for relative in METADATA_MANIFESTS:
        path = root / relative
        if not path.is_file():
            continue
        for row in pq.read_table(path).to_pylist():
            sample_id = str(row["sample_id"])
            existing = index.setdefault(sample_id, {})
            for key, value in row.items():
                if value is not None and key not in existing:
                    existing[key] = value
    return index


def _merged_metadata(
    row: dict[str, Any],
    metadata_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    merged = dict(metadata_index.get(sample_id, {}))
    merged.update(
        {
            key: value
            for key, value in row.items()
            if value is not None
        }
    )
    return merged


def _nullable_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value)
    return text if text and text.casefold() != "nan" else None


def _nullable_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return int(value)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [str(item) for item in value]


def _json_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def _safe_filename(task_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)
    return f"{safe[:180]}.jpg"


def _validate_image(encoded: bytes, *, task_id: str) -> None:
    try:
        with Image.open(BytesIO(encoded)) as image:
            image.verify()
    except Exception as error:
        raise ValueError(
            f"Benchmark image decode failed for {task_id}: {error}"
        ) from error


def _task_record(
    *,
    source_row: dict[str, Any],
    metadata: dict[str, Any],
    prepared: Any,
    benchmark_id: str,
    benchmark_version: str,
    evaluation_set: str,
    prompt: dict[str, Any],
    top_k: int,
    benchmark_image: bytes,
    benchmark_config_sha256: str,
    taxonomy_sha256: str,
    image_preprocessing_profile: str,
) -> dict[str, Any]:
    task_id = str(prepared.task_id)
    benchmark_image_sha256 = sha256(benchmark_image).hexdigest()
    rendered_schema = json.dumps(
        prepared.schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if not rendered_schema:
        raise ValueError(f"Empty rendered schema for {task_id}")
    prompt_payload = prepared.system_prompt + "\0" + prepared.user_prompt
    return {
        "image": {
            "bytes": benchmark_image,
            "path": _safe_filename(task_id),
        },
        "task_id": task_id,
        "sample_id": str(prepared.sample_id),
        "benchmark_id": benchmark_id,
        "benchmark_version": benchmark_version,
        "evaluation_set": evaluation_set,
        "source": str(metadata.get("dataset_id", "unknown")),
        "leakage_group_id": str(
            metadata.get("leakage_group_id", prepared.sample_id)
        ),
        "system_prompt": str(prepared.system_prompt),
        "user_prompt": str(prepared.user_prompt),
        "response_schema_json": rendered_schema,
        "prompt_id": str(prompt.get("id", "unknown")),
        "prompt_version": str(prompt.get("version", "unknown")),
        "top_k": top_k,
        "candidate_disease_ids": list(prepared.allowed_disease_ids),
        "pair_id": _nullable_text(source_row.get("pair_id")),
        "condition": _nullable_text(source_row.get("difficulty")),
        "confusion_set_id": _nullable_text(
            source_row.get("confusion_set_id")
        ),
        "prompt_sha256": _text_sha256(prompt_payload),
        "response_schema_sha256": _text_sha256(rendered_schema),
        "benchmark_config_sha256": benchmark_config_sha256,
        "taxonomy_sha256": taxonomy_sha256,
        "source_image_sha256": _nullable_text(
            metadata.get("image_sha256")
        ),
        "benchmark_image_sha256": benchmark_image_sha256,
        "image_preprocessing_profile": image_preprocessing_profile,
        "license_id": str(metadata.get("license_id", "unknown")),
    }


def _reference_record(
    *,
    source_row: dict[str, Any],
    metadata: dict[str, Any],
    benchmark_id: str,
    evaluation_set: str,
    task_id: str,
) -> dict[str, Any]:
    is_evidence = "morphology_concept_ids" in source_row
    reference_disease = source_row.get(
        "disease_id",
        metadata.get("disease_id"),
    )
    morphology_ids = (
        _string_list(source_row.get("morphology_concept_ids"))
        if is_evidence
        else []
    )
    return {
        "task_id": task_id,
        "sample_id": str(source_row["sample_id"]),
        "benchmark_id": benchmark_id,
        "evaluation_set": evaluation_set,
        "source": str(metadata.get("dataset_id", "unknown")),
        "leakage_group_id": str(
            metadata.get("leakage_group_id", source_row["sample_id"])
        ),
        "reference_disease_id": _nullable_text(reference_disease),
        "reference_diagnoses_json": _json_value(
            metadata.get("reference_diagnoses")
        ),
        "diagnosis_basis": _nullable_text(
            source_row.get(
                "diagnosis_basis",
                metadata.get("diagnosis_basis"),
            )
        ),
        "morphology_concept_ids": morphology_ids,
        "reference_clinical_description": _nullable_text(
            source_row.get("reference_clinical_description")
        ),
        "score_morphology": bool(
            source_row.get("score_morphology", False)
        ),
        "score_description": bool(
            source_row.get("score_description", False)
        ),
        "score_diagnosis": bool(
            source_row.get("score_diagnosis", reference_disease is not None)
        ),
        "pair_id": _nullable_text(source_row.get("pair_id")),
        "condition": _nullable_text(source_row.get("difficulty")),
        "confusion_set_id": _nullable_text(
            source_row.get("confusion_set_id")
        ),
        "age_years": _nullable_int(metadata.get("age_years")),
        "age_group_standardized": _nullable_text(
            metadata.get("age_group_standardized")
        ),
        "skin_tone_system": _nullable_text(
            metadata.get("skin_tone_system")
        ),
        "skin_tone": _nullable_text(
            source_row.get("skin_tone", metadata.get("skin_tone"))
        ),
        "sex_or_gender_system": _nullable_text(
            metadata.get("sex_or_gender_system")
        ),
        "sex_or_gender": _nullable_text(
            metadata.get("sex_or_gender")
        ),
        "race_ethnicity": _nullable_text(
            metadata.get("race_ethnicity")
        ),
        "license_id": str(metadata.get("license_id", "unknown")),
    }


def _write_shards(
    *,
    records: list[dict[str, Any]],
    features: Features,
    directory: Path,
    split: str,
    shard_size: int,
) -> list[dict[str, Any]]:
    directory.mkdir(parents=True, exist_ok=True)
    shard_count = max(1, math.ceil(len(records) / shard_size))
    shards: list[dict[str, Any]] = []
    for shard_index in range(shard_count):
        start = shard_index * shard_size
        shard_records = records[start : start + shard_size]
        path = directory / (
            f"{split}-{shard_index:05d}-of-{shard_count:05d}.parquet"
        )
        Dataset.from_list(shard_records, features=features).to_parquet(path)
        shards.append(
            {
                "path": path.as_posix(),
                "rows": len(shard_records),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    return shards


def _copy_artifacts(root: Path, temporary: Path) -> list[dict[str, Any]]:
    source_paths = {
        "configs/visual_top_k.yaml": BENCHMARK_SPECS[0].config_path,
        "configs/visual_confusion_sets.yaml": BENCHMARK_SPECS[1].config_path,
        "configs/evidence_grounded_diagnosis.yaml": BENCHMARK_SPECS[2].config_path,
        "prompts/top_k.yaml": Path("prompts/benchmarks/top_k.yaml"),
        "prompts/confusion_sets.yaml": Path(
            "prompts/benchmarks/confusion_sets.yaml"
        ),
        "prompts/evidence_grounded_diagnosis.yaml": Path(
            "prompts/benchmarks/evidence_grounded_diagnosis.yaml"
        ),
        "schemas/visual_top_k.schema.json": Path(
            "schemas/visual_top_k.schema.json"
        ),
        "schemas/visual_confusion_sets.schema.json": Path(
            "schemas/visual_confusion_sets.schema.json"
        ),
        "schemas/evidence_grounded_diagnosis.schema.json": Path(
            "schemas/evidence_grounded_diagnosis.schema.json"
        ),
        "taxonomies/diseases.yaml": Path(
            "configs/taxonomies/diseases.yaml"
        ),
        "taxonomies/disease_confusion_sets.yaml": Path(
            "configs/taxonomies/disease_confusion_sets.yaml"
        ),
    }
    records: list[dict[str, Any]] = []
    for destination_value, source_relative in source_paths.items():
        source = root / source_relative
        destination = temporary / "artifacts" / destination_value
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "path": destination.relative_to(temporary).as_posix(),
                "source": source_relative.as_posix(),
                "sha256": _file_sha256(destination),
            }
        )
    return records


def _source_licenses() -> dict[str, Any]:
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "sources": {
            "fitzpatrick17k_c": {
                "license_id": "CC_BY_NC_SA_3_0",
            },
            "pad_ufes_20": {"license_id": "CC_BY_4_0"},
            "scin": {"license_id": "SCIN_DATA_USE_LICENSE"},
            "ddi": {
                "license_id": "DDI_RESEARCH_USE_AGREEMENT",
                "notes": (
                    "Keep this repository private and comply with the DDI "
                    "Research Use Agreement. Do not infer redistribution "
                    "rights from this combined dataset card."
                ),
            },
            "skindisnet": {
                "license_id": "SKINDISNET_RESEARCH_USE",
            },
        },
    }


def _card_metadata() -> dict[str, Any]:
    configs = []
    for spec in BENCHMARK_SPECS:
        task_files = []
        reference_files = []
        for evaluation in spec.evaluations:
            task_files.append(
                {
                    "split": evaluation.split,
                    "path": (
                        f"tasks/{spec.key}/{evaluation.split}-*.parquet"
                    ),
                }
            )
            reference_files.append(
                {
                    "split": evaluation.split,
                    "path": (
                        f"references/{spec.key}/"
                        f"{evaluation.split}-*.parquet"
                    ),
                }
            )
        configs.append(
            {"config_name": spec.key, "data_files": task_files}
        )
        configs.append(
            {
                "config_name": f"{spec.key}_references",
                "data_files": reference_files,
            }
        )
    return {
        "pretty_name": "ISEPDermaBench",
        "language": ["en"],
        "license": "other",
        "task_categories": [
            "visual-question-answering",
            "image-classification",
        ],
        "tags": [
            "dermatology",
            "medical",
            "multimodal",
            "benchmark",
            "private-research-dataset",
        ],
        "configs": configs,
    }


def _dataset_card(summary_rows: list[dict[str, Any]]) -> str:
    metadata = yaml.safe_dump(
        _card_metadata(),
        sort_keys=False,
        allow_unicode=True,
    ).strip()
    rows = "\n".join(
        "| {task} | {split} | {tasks:,} | {samples:,} | {groups:,} |".format(
            task=row["benchmark"],
            split=row["split"],
            tasks=row["task_count"],
            samples=row["sample_count"],
            groups=row["group_count"],
        )
        for row in summary_rows
    )
    return f"""---
{metadata}
---

# ISEPDermaBench

ISEPDermaBench is the private, versioned evaluation dataset for the ISEP
small multimodal language model thesis. It packages the exact image and
rendered request seen by a model while keeping scoring references in separate
Hugging Face configurations.

## Configurations

Load task inputs and references independently:

```python
from datasets import load_dataset

tasks = load_dataset(
    "danielfdias98/ISEPDermaBench",
    "visual_top_k",
    split="validation",
)
references = load_dataset(
    "danielfdias98/ISEPDermaBench",
    "visual_top_k_references",
    split="validation",
)
```

Join the two views only inside the scorer by `task_id`. Never include fields
from a `_references` configuration in a model request.

| Benchmark | Split | Tasks | Unique images | Leakage groups |
| --- | --- | ---: | ---: | ---: |
{rows}

### `visual_top_k`

Closed-set ranking of exactly six diseases from the frozen 21-class taxonomy.
It includes Validation, the 1,000-case internal paired benchmark, DDI external
evaluation, and SkinDisNet external evaluation.

### `visual_confusion_sets`

Paired three-way ranking under low- and high-confusability candidate sets. The
same image is evaluated once per condition.

### `evidence_grounded_diagnosis`

Morphology grounding, observation-only clinical description, six-disease
differential diagnosis, and explicit evidence links. It includes Validation,
the newly materialized sealed internal evidence cohort, and external DDI.

## Input schema

Task configurations begin with the multimodal input columns:

```text
image
task_id
sample_id
system_prompt
user_prompt
response_schema_json
candidate_disease_ids
```

Images are deterministic RGB JPEG representations produced with the frozen
`dermatology_api_safe_rgb_jpeg_v1` preprocessing profile. They are the exact
bytes intended for every API and local-model backend, not an additional
training representation.

## Reference isolation

Reference configurations contain the correct disease, morphology concepts,
reference description, scoring flags, and evaluation-only subgroup metadata.
The task Parquets contain no `reference_disease_id`, morphology gold labels,
or reference clinical descriptions.

## Split policy

- Validation may be used for dry runs, prompt/parser development, teacher
  selection, checkpoint selection, and threshold calibration.
- Internal Benchmark is sealed and supports the paired before/after result.
- External sets measure generalization and must not select the teacher or tune
  the system.
- No split from this repository may be used for student fine-tuning while it
  retains an evaluation role.

## DDI restrictions

DDI rows and images are included because this repository is private and used
for the approved research workflow. They remain governed by the upstream DDI
Research Use Agreement. This repository does not grant permission to make DDI
public, redistribute it, or use it outside the upstream terms.

## Reproducibility

Canonical benchmark YAMLs, prompt templates, JSON schemas, and taxonomies are
copied under `artifacts/`. Every row stores hashes for its rendered prompt,
response schema, benchmark config, taxonomy, source image, and final benchmark
image. Exact shard checksums and counts are recorded in `release.json`.

Build and validate locally with:

```bash
python -m src.data_pipeline.huggingface_benchmark_export
python -m src.data_pipeline.huggingface_benchmark_export --validate-only
```
"""


def build_huggingface_benchmark_export(
    root: Path,
    *,
    output_path: Path = DEFAULT_OUTPUT,
    shard_size: int = DEFAULT_SHARD_SIZE,
    verify_images: bool = True,
) -> dict[str, Any]:
    """Materialize the complete private ISEPDermaBench release."""

    root = root.resolve()
    output = root / output_path
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. Move it aside before rebuilding."
        )
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent)
    )
    metadata_index = _metadata_index(root)
    summary_rows: list[dict[str, Any]] = []
    split_records: list[dict[str, Any]] = []
    total_embedded_image_bytes = 0

    try:
        with ImageResolver(root) as resolver:

            @lru_cache(maxsize=1024)
            def benchmark_image(image_uri: str, config_key: str) -> bytes:
                config = next(
                    item
                    for item in BENCHMARK_SPECS
                    if item.key == config_key
                )
                parsed = load_benchmark_config(config.config_path, root=root)
                encoded = prepare_benchmark_image(
                    resolver.read_bytes(image_uri),
                    parsed.image_preprocessing,
                )
                if verify_images:
                    _validate_image(encoded, task_id=image_uri)
                if len(encoded) > parsed.image_preprocessing.max_encoded_bytes:
                    raise ValueError(
                        f"Benchmark image exceeds byte budget: {image_uri}"
                    )
                return encoded

            for spec in BENCHMARK_SPECS:
                parsed_config = load_benchmark_config(
                    spec.config_path,
                    root=root,
                )
                raw_config = _load_yaml(root / spec.config_path)
                prompt = _load_yaml(parsed_config.prompt_path)
                schema = _load_json(parsed_config.schema_path)
                taxonomy = _load_yaml(parsed_config.taxonomy.disease_path)
                disease_items = taxonomy.get("diseases")
                if not isinstance(disease_items, list):
                    raise ValueError("Disease taxonomy must contain diseases")
                adapter = build_task_adapter(
                    benchmark_config=raw_config,
                    prompt_config=prompt,
                    schema=schema,
                    disease_taxonomy_items=disease_items,
                )
                config_sha = _file_sha256(root / spec.config_path)
                taxonomy_sha = _file_sha256(
                    parsed_config.taxonomy.disease_path
                )
                requested_top_k = int(
                    raw_config["benchmark"].get(
                        "top_k",
                        raw_config["benchmark"].get("ranking_count"),
                    )
                )

                for evaluation in spec.evaluations:
                    loaded = load_benchmark_dataset(
                        root=root,
                        config=raw_config,
                        evaluation_set=evaluation.evaluation_set,
                        limit=None,
                        seed=42,
                    )
                    source_rows = loaded.frame.to_dict(orient="records")
                    if len(source_rows) != len(loaded.samples):
                        raise ValueError("Loaded rows and samples differ")
                    task_rows: list[dict[str, Any]] = []
                    reference_rows: list[dict[str, Any]] = []
                    groups: set[str] = set()
                    samples: set[str] = set()
                    for source_row, sample in zip(
                        source_rows,
                        loaded.samples,
                        strict=True,
                    ):
                        prepared = adapter.prepare(sample)
                        metadata = _merged_metadata(
                            source_row,
                            metadata_index,
                        )
                        encoded = benchmark_image(
                            sample.image_uri,
                            spec.key,
                        )
                        source_sha = _nullable_text(
                            metadata.get("image_sha256")
                        )
                        if source_sha is not None:
                            actual_source_sha = sha256(
                                resolver.read_bytes(sample.image_uri)
                            ).hexdigest()
                            if actual_source_sha != source_sha:
                                raise ValueError(
                                    "Source image checksum mismatch for "
                                    f"{sample.sample_id}"
                                )
                        task_row = _task_record(
                            source_row=source_row,
                            metadata=metadata,
                            prepared=prepared,
                            benchmark_id=parsed_config.benchmark.id,
                            benchmark_version=(
                                parsed_config.benchmark.version
                            ),
                            evaluation_set=evaluation.split,
                            prompt=prompt,
                            top_k=requested_top_k,
                            benchmark_image=encoded,
                            benchmark_config_sha256=config_sha,
                            taxonomy_sha256=taxonomy_sha,
                            image_preprocessing_profile=(
                                parsed_config.image_preprocessing.profile
                            ),
                        )
                        reference_row = _reference_record(
                            source_row=source_row,
                            metadata=metadata,
                            benchmark_id=parsed_config.benchmark.id,
                            evaluation_set=evaluation.split,
                            task_id=str(prepared.task_id),
                        )
                        task_rows.append(task_row)
                        reference_rows.append(reference_row)
                        groups.add(task_row["leakage_group_id"])
                        samples.add(task_row["sample_id"])
                        total_embedded_image_bytes += len(encoded)

                    task_ids = [row["task_id"] for row in task_rows]
                    reference_ids = [
                        row["task_id"] for row in reference_rows
                    ]
                    if len(task_ids) != len(set(task_ids)):
                        raise ValueError(
                            f"Duplicate task IDs in {spec.key}/{evaluation.split}"
                        )
                    if task_ids != reference_ids:
                        raise ValueError(
                            f"Input/reference IDs differ for {spec.key}/"
                            f"{evaluation.split}"
                        )

                    task_directory = temporary / "tasks" / spec.key
                    reference_directory = (
                        temporary / "references" / spec.key
                    )
                    task_shards = _write_shards(
                        records=task_rows,
                        features=TASK_FEATURES,
                        directory=task_directory,
                        split=evaluation.split,
                        shard_size=shard_size,
                    )
                    reference_shards = _write_shards(
                        records=reference_rows,
                        features=REFERENCE_FEATURES,
                        directory=reference_directory,
                        split=evaluation.split,
                        shard_size=shard_size,
                    )
                    for item in task_shards + reference_shards:
                        item["path"] = Path(item["path"]).relative_to(
                            temporary
                        ).as_posix()
                    split_record = {
                        "benchmark": spec.key,
                        "benchmark_id": parsed_config.benchmark.id,
                        "split": evaluation.split,
                        "source_evaluation_set": evaluation.evaluation_set,
                        "task_count": len(task_rows),
                        "sample_count": len(samples),
                        "group_count": len(groups),
                        "task_shards": task_shards,
                        "reference_shards": reference_shards,
                    }
                    split_records.append(split_record)
                    summary_rows.append(
                        {
                            key: split_record[key]
                            for key in (
                                "benchmark",
                                "split",
                                "task_count",
                                "sample_count",
                                "group_count",
                            )
                        }
                    )
                    print(
                        f"Wrote {spec.key}/{evaluation.split}: "
                        f"{len(task_rows)} tasks"
                    )

        artifacts = _copy_artifacts(root, temporary)
        metadata_directory = temporary / "metadata"
        metadata_directory.mkdir(parents=True, exist_ok=True)
        license_path = metadata_directory / "source_licenses.json"
        license_path.write_text(
            json.dumps(_source_licenses(), indent=2) + "\n",
            encoding="utf-8",
        )
        release = {
            "release": {
                "id": "ISEPDermaBench",
                "version": EXPORT_RELEASE_VERSION,
                "schema_version": EXPORT_SCHEMA_VERSION,
                "visibility": "private",
                "created_at": "2026-08-01",
                "reference_isolation": True,
                "image_preprocessing_profile": (
                    "dermatology_api_safe_rgb_jpeg_v1"
                ),
                "embedded_image_bytes": total_embedded_image_bytes,
                "splits": split_records,
                "artifacts": artifacts,
                "source_licenses": {
                    "path": "metadata/source_licenses.json",
                    "sha256": _file_sha256(license_path),
                },
            }
        }
        (temporary / "release.json").write_text(
            json.dumps(release, indent=2) + "\n",
            encoding="utf-8",
        )
        (temporary / "README.md").write_text(
            _dataset_card(summary_rows),
            encoding="utf-8",
        )
        temporary.rename(output)
    except Exception:
        print(f"Incomplete export retained for inspection at: {temporary}")
        raise

    return release["release"]


def validate_huggingface_benchmark_export(
    root: Path,
    *,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Validate shards, checksums, input/reference joins, and split isolation."""

    output = root.resolve() / output_path
    release = json.loads(
        (output / "release.json").read_text(encoding="utf-8")
    )["release"]
    total_tasks = 0
    task_groups: dict[tuple[str, str], set[str]] = {}
    for split in release["splits"]:
        task_ids: list[str] = []
        reference_ids: list[str] = []
        groups: set[str] = set()
        samples: set[str] = set()
        for shard_info in split["task_shards"]:
            path = output / shard_info["path"]
            if _file_sha256(path) != shard_info["sha256"]:
                raise ValueError(f"Task shard checksum mismatch: {path}")
            parquet = pq.ParquetFile(path)
            if b"huggingface" not in (parquet.schema_arrow.metadata or {}):
                raise ValueError(f"Missing Hugging Face metadata: {path}")
            table = pq.read_table(
                path,
                columns=[
                    "image",
                    "task_id",
                    "sample_id",
                    "leakage_group_id",
                    "system_prompt",
                    "user_prompt",
                    "response_schema_json",
                    "benchmark_image_sha256",
                ],
            )
            for row in table.to_pylist():
                image = row["image"]
                encoded = image["bytes"]
                if sha256(encoded).hexdigest() != row[
                    "benchmark_image_sha256"
                ]:
                    raise ValueError(
                        f"Image checksum mismatch: {row['task_id']}"
                    )
                _validate_image(encoded, task_id=row["task_id"])
                if not row["system_prompt"] or not row["user_prompt"]:
                    raise ValueError(f"Empty prompt: {row['task_id']}")
                json.loads(row["response_schema_json"])
                task_ids.append(row["task_id"])
                samples.add(row["sample_id"])
                groups.add(row["leakage_group_id"])
        for shard_info in split["reference_shards"]:
            path = output / shard_info["path"]
            if _file_sha256(path) != shard_info["sha256"]:
                raise ValueError(f"Reference shard checksum mismatch: {path}")
            reference_ids.extend(
                pq.read_table(path, columns=["task_id"])["task_id"].to_pylist()
            )
        if task_ids != reference_ids:
            raise ValueError(
                f"Task/reference join mismatch: {split['benchmark']}/"
                f"{split['split']}"
            )
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Duplicate task IDs in materialized release")
        actual = {
            "task_count": len(task_ids),
            "sample_count": len(samples),
            "group_count": len(groups),
        }
        expected = {
            key: int(split[key])
            for key in actual
        }
        if actual != expected:
            raise ValueError(
                f"Split counts differ: {actual} != {expected}"
            )
        total_tasks += len(task_ids)
        task_groups[(split["benchmark"], split["split"])] = groups

    for benchmark in {spec.key for spec in BENCHMARK_SPECS}:
        validation = task_groups.get((benchmark, "validation"), set())
        internal = task_groups.get(
            (benchmark, "internal_benchmark"),
            set(),
        )
        overlap = validation & internal
        if overlap:
            raise ValueError(
                f"Validation/Internal group overlap in {benchmark}: "
                f"{sorted(overlap)[:3]}"
            )

    for artifact in release["artifacts"]:
        path = output / artifact["path"]
        if _file_sha256(path) != artifact["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {path}")
    return {
        "split_count": len(release["splits"]),
        "task_count": total_tasks,
        "embedded_image_bytes": int(release["embedded_image_bytes"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or validate the private ISEPDermaBench release."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
    )
    args = parser.parse_args()
    if args.validate_only:
        result = validate_huggingface_benchmark_export(
            args.project_root,
            output_path=args.output,
        )
    else:
        result = build_huggingface_benchmark_export(
            args.project_root,
            output_path=args.output,
            shard_size=args.shard_size,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
