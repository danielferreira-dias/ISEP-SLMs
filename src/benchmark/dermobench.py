"""Dedicated adapter for the official, leakage-filtered DermoBench release.

The upstream suite stores one image-grounded conversation per row.  This
module preserves the evaluated-model prompts, hides the reference answer,
normalizes globally duplicated upstream IDs, and keeps deterministic MCQ
scoring separate from the four text-only LLM-as-a-judge protocols.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd

from src.benchmark.datasets import LoadedBenchmarkDataset
from src.benchmark.results import file_sha256
from src.benchmark.runner import (
    BenchmarkPrediction,
    BenchmarkSample,
    ModelResponse,
)
from src.benchmark.selection import select_units
from src.benchmark.task_adapters import PreparedTask
from src.config.benchmarks import (
    BenchmarkConfig,
    BenchmarkDatasetConfig,
    BenchmarkExecutionConfig,
    BenchmarkMetadataConfig,
    EvaluationSetConfig,
    ImagePreprocessingConfig,
    StructuredOutputConfig,
    TaxonomyConfig,
)


DEFAULT_LOCAL_RELEASE = Path("data/benchmarks/DermoBench")
MCQ_SUFFIX = "NOTE: Respond with ONLY the letter of your choice."
_OPTION_RE = re.compile(r"(?m)^\s*([A-Z])\)\s*(.+?)\s*$")
_INITIAL_CHOICE_RE = re.compile(r"^\s*([A-Z])(?:\s*[\)\].:\-]|\s|$)")
_ANSWER_CHOICE_RE = re.compile(
    r"(?i)\b(?:answer|choice|option)\s*(?:is|:)?\s*([A-Z])\b"
)


@dataclass(frozen=True, slots=True)
class DermoBenchSpec:
    """One executable task in the public DermoBench release."""

    key: str
    path: str
    scoring: str
    max_output_tokens: int
    aliases: tuple[str, ...] = ()

    @property
    def judge_required(self) -> bool:
        return self.scoring == "llm_as_a_judge"

    @property
    def benchmark_id(self) -> str:
        return f"dermobench_{self.key}"


SPECS = (
    DermoBenchSpec(
        "task_1_1_description_without_morphology",
        "task1/1_1_description_wo_morph/task1_1_final.jsonl",
        "llm_as_a_judge",
        4096,
        ("task1.1", "task_1_1"),
    ),
    DermoBenchSpec(
        "task_1_2_description_with_morphology",
        "task1/1_2_description_w_morph/task1_2_final.jsonl",
        "llm_as_a_judge",
        4096,
        ("task1.2", "task_1_2"),
    ),
    DermoBenchSpec(
        "task_1_3_derm7pt_morphology_mcq",
        "task1/1_3_mcq_derm7pt/derm7pt_test_mcq.json",
        "exact_choice_match",
        128,
        ("task1.3", "task_1_3"),
    ),
    DermoBenchSpec(
        "task_1_4_skincon_morphology_mcq",
        "task1/1_4_mcq_skincon/skincon_all_mcq.json",
        "exact_choice_match",
        128,
        ("task1.4", "task_1_4"),
    ),
    DermoBenchSpec(
        "task_2_1_diagnosis_mcq_25_choices",
        "task2/2_1_mcq/25_choices/"
        "task2.1_25choices_test_2k_non_uniform_sample_final.json",
        "exact_choice_match",
        128,
        ("task2.1-25", "task_2_1_25"),
    ),
    DermoBenchSpec(
        "task_2_1_diagnosis_mcq_4_choices",
        "task2/2_1_mcq/4_choices/"
        "task2.1_test_2k_non_uniform_sample_final.json",
        "exact_choice_match",
        128,
        ("task2.1-4", "task_2_1_4"),
    ),
    DermoBenchSpec(
        "task_2_1_ddi_diagnosis_mcq",
        "task2/2_1_mcq/ddi/task2.1_ddi_4choices.json",
        "exact_choice_match",
        128,
        ("task2.1-ddi",),
    ),
    DermoBenchSpec(
        "task_2_1_derm1m_edu_diagnosis_mcq",
        "task2/2_1_mcq/derm1m_edu/task2.1_derm1m_edu_final.json",
        "exact_choice_match",
        128,
        ("task2.1-derm1m",),
    ),
    DermoBenchSpec(
        "task_2_1_derm7pt_diagnosis_mcq",
        "task2/2_1_mcq/derm7pt/task2.1_derm7pt_4choices.json",
        "exact_choice_match",
        128,
        ("task2.1-derm7pt",),
    ),
    DermoBenchSpec(
        "task_2_1_snu134_diagnosis_mcq",
        "task2/2_1_mcq/snu134/task2.1_snu134_4choices.json",
        "exact_choice_match",
        128,
        ("task2.1-snu134",),
    ),
    DermoBenchSpec(
        "task_3_1_diagnostic_reasoning_without_morphology",
        "task3/3_1/task3_1_final.jsonl",
        "llm_as_a_judge",
        10240,
        ("task3.1", "task_3_1"),
    ),
    DermoBenchSpec(
        "task_3_2_diagnostic_reasoning_with_morphology",
        "task3/3_2/task3_2_final.jsonl",
        "llm_as_a_judge",
        10240,
        ("task3.2", "task_3_2"),
    ),
    DermoBenchSpec(
        "task_4_ddi_fairness_mcq",
        "task4/ddi_4choices_final.jsonl",
        "exact_choice_match",
        128,
        ("task4", "task_4"),
    ),
)


SKINCON_FEATURES = (
    "Abscess", "Acuminate", "Atrophy", "Black", "Blue",
    "Brown(Hyperpigmentation)", "Bulla", "Burrow", "Comedo", "Crust",
    "Cyst", "Dome-shaped", "Erosion", "Erythema", "Excoriation",
    "Exophytic/Fungating", "Exudate", "Fissure", "Flat topped", "Friable",
    "Gray", "Induration", "Lichenification", "Macule", "Nodule", "Papule",
    "Patch", "Pedunculated", "Pigmented", "Plaque", "Poikiloderma",
    "Purple", "Purpura/Petechiae", "Pustule", "Salmon", "Scale", "Scar",
    "Sclerosis", "Telangiectasia", "Translucent", "Ulcer", "Umbilicated",
    "Vesicle", "Warty/Papillomatous", "Wheal", "White(Hypopigmentation)",
    "Xerosis", "Yellow",
)

TASK_1_2_SKINCON_SYSTEM = """You are a dermatology VQA classifier.
Output exactly one <morph> block containing valid JSON with the single key
"morphological_features_skincon", followed by one detailed clinical
morphological paragraph. Choose only visibly present values from this closed
set and sort them alphabetically:
{features}
Do not add code fences, diagnoses, management, probabilities, lists, or any
text outside the required morph block and paragraph.""".format(
    features=json.dumps(SKINCON_FEATURES)
)

TASK_1_2_DERM7PT_SYSTEM = """You are a dermoscopy VQA classifier.
Output exactly one <morph> block containing valid JSON with the single key
"morphological_features_Derm7pt". Its object must contain all seven official
Derm7pt keys with one allowed value per key. Follow it with exactly one
detailed dermoscopic paragraph. Do not add code fences, diagnoses, management,
probabilities, lists, or text outside the required block and paragraph."""

TASK_3_1_SYSTEM = """You are a dermatology VQA assistant.
Output exactly two blocks in this order and nothing else:
<reasoning>Concise, step-by-step, image-grounded reasoning.</reasoning>
<final_diagnosis>ONE most likely free-text clinical diagnosis.</final_diagnosis>
Do not echo the question or add markdown, probabilities, disclaimers,
management, tests, or treatment."""

TASK_3_2_SKINCON_SYSTEM = """You are a dermatology VQA assistant.
Output exactly three blocks in this order and nothing else:
<reasoning>Concise, step-by-step, image-grounded reasoning.</reasoning>
<morph>{"morphological_features_skincon": [...]}</morph>
<final_diagnosis>ONE label from the provided taxonomy.</final_diagnosis>
The morph array may contain only visibly present values from the official
SkinCon closed set and must be sorted alphabetically. Do not use code fences
or add extra text."""

TASK_3_2_DERM7PT_SYSTEM = """You are a dermoscopy VQA assistant.
Output exactly three blocks in this order and nothing else:
<reasoning>Concise, step-by-step, image-grounded reasoning.</reasoning>
<morph>A valid Derm7pt JSON object with all seven official keys.</morph>
<final_diagnosis>ONE label from the provided taxonomy.</final_diagnosis>
Use exactly one allowed value per Derm7pt key; use "absent" when a structure is
not present. Do not use code fences or add extra text."""


def is_dermobench_config(value: str | Path) -> bool:
    """Return whether a CLI benchmark selector belongs to DermoBench."""

    token = str(value).strip().casefold()
    return token.startswith("dermobench/") or any(
        token in _spec_tokens(spec) for spec in SPECS
    )


def resolve_dermobench_spec(value: str | Path) -> DermoBenchSpec:
    """Resolve a public benchmark selector to one unambiguous task."""

    token = str(value).strip().casefold()
    if token.startswith("dermobench/"):
        token = token.split("/", 1)[1]
    matches = [spec for spec in SPECS if token in _spec_tokens(spec)]
    if len(matches) != 1:
        choices = ", ".join(f"dermobench/{spec.key}" for spec in SPECS)
        raise ValueError(
            f"Unknown DermoBench task {value!r}; expected one of {choices}"
        )
    return matches[0]


def list_dermobench_configs(*, root: Path) -> tuple[BenchmarkConfig, ...]:
    """Return typed configs for all public DermoBench tasks."""

    return tuple(load_dermobench_config(spec.key, root=root) for spec in SPECS)


def load_dermobench_config(
    value: str | Path,
    *,
    root: Path,
) -> BenchmarkConfig:
    """Build a typed runtime config without mutating the upstream release."""

    spec = resolve_dermobench_spec(value)
    release_root = (root / DEFAULT_LOCAL_RELEASE).resolve()
    artifacts = release_root / "evaluation/artifacts"
    evaluation_path = release_root / "evaluation/tasks" / spec.path
    preprocessing = ImagePreprocessingConfig(
        profile="isep_multimodal_jpeg_v1",
        max_edge_pixels=1536,
        max_encoded_bytes=4_000_000,
        jpeg_quality=90,
        minimum_jpeg_quality=65,
        minimum_edge_pixels=512,
    )
    return BenchmarkConfig(
        benchmark=BenchmarkMetadataConfig(
            id=spec.benchmark_id,
            version="filtered_2026_08_v1",
            task=(
                "dermobench_open_ended"
                if spec.judge_required
                else "dermobench_mcq"
            ),  # type: ignore[arg-type]
            description=(
                "Official DermoBench task after ISEPDermData train-overlap "
                "exclusion."
            ),
            status="external_evaluation",
        ),
        prompt_path=artifacts / "model_prompts.yaml",
        schema_path=artifacts / "free_text.schema.json",
        taxonomy=TaxonomyConfig(disease_path=artifacts / "taxonomy.yaml"),
        dataset=BenchmarkDatasetConfig(
            default_evaluation_set="filtered",
            evaluation_sets=(
                EvaluationSetConfig(
                    id="filtered",
                    manifest=evaluation_path,
                    role="external_evaluation",
                    description="Training-overlap-filtered DermoBench view.",
                ),
            ),
            image_column="image_uri",
            sample_id_column="sample_id",
            label_column="reference_choice",
            task_id_column="task_id",
        ),
        image_preprocessing=preprocessing,
        execution=BenchmarkExecutionConfig(
            max_output_tokens=spec.max_output_tokens,
            batch_size=8,
            resume=True,
            save_raw_responses=True,
            save_rendered_prompts=True,
            fail_fast_on_invalid_output=False,
            implementation_status="implemented",
        ),
        structured_output=StructuredOutputConfig(mode="prompt_only"),
        output_directory=(root / "outputs/dermobench").resolve(),
        config_path=(root / "configs/datasets/dermobench/config.yaml").resolve(),
    )


def load_dermobench_dataset(
    *,
    root: Path,
    benchmark: BenchmarkConfig,
    evaluation_set: str | None,
    limit: int | None,
    seed: int,
) -> LoadedBenchmarkDataset:
    """Load one official task and resolve images through the verified index."""

    if evaluation_set not in {None, "filtered"}:
        raise ValueError("DermoBench exposes only evaluation set 'filtered'")
    spec = resolve_dermobench_spec(benchmark.benchmark.id)
    release_root = (root / DEFAULT_LOCAL_RELEASE / "release").resolve()
    task_path = (
        root / DEFAULT_LOCAL_RELEASE / "evaluation/tasks" / spec.path
    ).resolve()
    if not task_path.is_file():
        raise FileNotFoundError(f"DermoBench task file is missing: {task_path}")
    index_path = release_root / "image_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            "DermoBench image index is missing; run "
            "`python -m src.data_pipeline.dermobench --extract`"
        )
    index = _load_object(index_path)
    image_paths = index.get("image_paths")
    archive = index.get("archive")
    if not isinstance(image_paths, dict) or not isinstance(archive, dict):
        raise ValueError("DermoBench image index has an invalid schema")
    image_root = str(archive.get("image_root", "images"))
    rows = _load_rows(task_path)
    ddi_skin_tones = _load_ddi_skin_tones(root)
    normalized = [
        _normalize_row(
            row,
            row_index=row_index,
            spec=spec,
            root=root,
            release_root=release_root,
            image_root=image_root,
            image_paths=image_paths,
            ddi_skin_tones=ddi_skin_tones,
        )
        for row_index, row in enumerate(rows)
    ]
    frame = pd.DataFrame(normalized)
    release_manifest = root / DEFAULT_LOCAL_RELEASE / "evaluation/release.json"
    release_sha256 = file_sha256(release_manifest)
    selected, selection = select_units(
        frame,
        unit_column="task_id",
        task_column="task_id",
        limit=limit,
        seed=seed,
        benchmark_release_hash=release_sha256,
    )
    samples = tuple(
        BenchmarkSample(
            sample_id=str(row["sample_id"]),
            task_id=str(row["task_id"]),
            image_uri=str(row["image_uri"]),
            disease_id=str(row.get("reference_choice", "")),
            candidate_disease_ids=tuple(row.get("option_labels", [])),
            system_prompt=str(row["system_prompt"]),
            user_prompt=str(row["user_prompt"]),
            response_schema={},
            metadata={
                key: _python_value(value)
                for key, value in row.items()
                if key not in {"system_prompt", "user_prompt"}
            },
        )
        for row in selected.to_dict(orient="records")
    )
    return LoadedBenchmarkDataset(
        manifest_path=task_path,
        manifest_sha256=file_sha256(task_path),
        release_sha256=release_sha256,
        evaluation_set="filtered",
        frame=selected,
        samples=samples,
        selection=selection
        | {
            "benchmark_source": "local",
            "task_configuration": spec.key,
            "upstream_path": spec.path,
            "scoring": spec.scoring,
            "judge_required": spec.judge_required,
        },
    )


class DermoBenchTaskAdapter:
    """Parse exact-choice outputs or stage open responses for a text judge."""

    def __init__(self, spec: DermoBenchSpec) -> None:
        self.spec = spec

    @property
    def benchmark_id(self) -> str:
        return self.spec.benchmark_id

    def prepare(self, sample: BenchmarkSample) -> PreparedTask:
        if sample.system_prompt is None or sample.user_prompt is None:
            raise ValueError("DermoBench sample has no frozen prompt")
        return PreparedTask(
            benchmark_id=self.benchmark_id,
            task_id=sample.task_id or sample.sample_id,
            sample_id=sample.sample_id,
            system_prompt=sample.system_prompt,
            user_prompt=sample.user_prompt,
            schema={},
            allowed_disease_ids=tuple(sample.candidate_disease_ids or ()),
        )

    def parse_response(
        self,
        model_id: str,
        raw_text: str,
        prepared_task: PreparedTask,
        reasoning_text: str | None = None,
    ) -> ModelResponse:
        del reasoning_text
        if prepared_task.benchmark_id != self.benchmark_id:
            raise ValueError("Prepared task belongs to a different benchmark")
        if self.spec.judge_required:
            return _parse_open_response(model_id, raw_text, self.spec)
        return _parse_mcq_response(
            model_id,
            raw_text,
            allowed=set(prepared_task.allowed_disease_ids),
        )

    def compute_metrics(
        self,
        predictions: Iterable[BenchmarkPrediction],
    ) -> dict[str, Any]:
        values = list(predictions)
        if self.spec.judge_required:
            return _open_metrics(values, self.spec)
        return _mcq_metrics(values, self.spec)


def _normalize_row(
    row: dict[str, Any],
    *,
    row_index: int,
    spec: DermoBenchSpec,
    root: Path,
    release_root: Path,
    image_root: str,
    image_paths: dict[str, Any],
    ddi_skin_tones: dict[str, str],
) -> dict[str, Any]:
    upstream_id = str(row.get("id", ""))
    upstream_image = str(row.get("image", ""))
    actual = image_paths.get(upstream_image)
    if not isinstance(actual, str) or not actual:
        raise ValueError(
            f"Image index has no mapping for {upstream_image!r}"
        )
    image_path = _resolve_release_image_path(
        release_root / image_root / actual
    )
    if not image_path.is_file():
        raise FileNotFoundError(f"DermoBench image is missing: {image_path}")
    human, reference = _conversation_pair(row)
    user_prompt = human.replace("<image>", "", 1).lstrip("\n")
    options = dict(_OPTION_RE.findall(user_prompt))
    option_labels = list(options)
    reference_choice = _reference_choice(reference, set(option_labels))
    if spec.scoring == "exact_choice_match":
        if not option_labels or reference_choice is None:
            raise ValueError(
                f"Invalid MCQ row {spec.key}:{row_index}: no options/answer"
            )
        user_prompt = f"{user_prompt.rstrip()}\n\n{MCQ_SUFFIX}"
    task_id = f"dermobench:{spec.key}:{row_index:06d}:{upstream_id}"
    image_uri = str(image_path.relative_to(root.resolve()))
    modality = _image_modality(upstream_image)
    skin_tone = ddi_skin_tones.get(Path(upstream_image).name)
    return {
        "task_id": task_id,
        "sample_id": upstream_id,
        "upstream_id": upstream_id,
        "upstream_row_index": row_index,
        "upstream_task_path": spec.path,
        "upstream_image": upstream_image,
        "image_uri": image_uri,
        "image_modality": modality,
        "system_prompt": _system_prompt(spec, modality),
        "user_prompt": user_prompt,
        "reference_answer": reference,
        "reference_choice": reference_choice or "",
        "option_labels": option_labels,
        "options": options,
        "scoring": spec.scoring,
        "judge_required": spec.judge_required,
        "judge_status": "pending" if spec.judge_required else "not_applicable",
        "skin_tone_system": "Fitzpatrick grouped" if skin_tone else None,
        "skin_tone": skin_tone,
    }


def _resolve_release_image_path(image_path: Path) -> Path:
    """Resolve archive paths whose case differs on case-sensitive hosts.

    The official Derm7pt archive contains directory and filename casing that
    differs from some entries in DermoBench's image index. macOS resolves
    those entries transparently, whereas Linux inference hosts do not. Exact
    paths remain the fast path; only missing paths use a component-wise,
    case-insensitive lookup against the extracted archive.
    """

    if image_path.is_file():
        return image_path

    anchor = Path(image_path.anchor)
    current = anchor
    for part in image_path.parts[1:]:
        exact = current / part
        if exact.exists():
            current = exact
            continue
        match = _casefold_children(current).get(part.casefold())
        if match is None:
            return image_path
        current = current / match
    return current


@lru_cache(maxsize=None)
def _casefold_children(directory: Path) -> dict[str, str]:
    """Cache a case-insensitive name index for one archive directory."""

    if not directory.is_dir():
        return {}
    children: dict[str, str] = {}
    for child in directory.iterdir():
        key = child.name.casefold()
        if key in children and children[key] != child.name:
            raise ValueError(
                f"Ambiguous case-insensitive DermoBench path in {directory}: "
                f"{children[key]!r} and {child.name!r}"
            )
        children[key] = child.name
    return children


def _conversation_pair(row: dict[str, Any]) -> tuple[str, str]:
    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("DermoBench row has no conversations list")
    human = next(
        (
            str(turn.get("value", ""))
            for turn in conversations
            if isinstance(turn, dict) and turn.get("from") == "human"
        ),
        "",
    )
    reference = next(
        (
            str(turn.get("value", ""))
            for turn in conversations
            if isinstance(turn, dict) and turn.get("from") == "gpt"
        ),
        "",
    )
    if not human or not reference:
        raise ValueError("DermoBench row lacks a human prompt or GPT reference")
    return human, reference


def _system_prompt(spec: DermoBenchSpec, modality: str) -> str:
    if spec.key == "task_1_2_description_with_morphology":
        return (
            TASK_1_2_DERM7PT_SYSTEM
            if modality == "dermoscopy"
            else TASK_1_2_SKINCON_SYSTEM
        )
    if spec.key == "task_3_1_diagnostic_reasoning_without_morphology":
        return TASK_3_1_SYSTEM
    if spec.key == "task_3_2_diagnostic_reasoning_with_morphology":
        return (
            TASK_3_2_DERM7PT_SYSTEM
            if modality == "dermoscopy"
            else TASK_3_2_SKINCON_SYSTEM
        )
    return ""


def _image_modality(path: str) -> str:
    normalized = f"/{path.casefold().strip('/')}"
    if "/derm7pt/" in normalized or "/dermoscopy" in normalized:
        return "dermoscopy"
    return "clinical"


def _parse_mcq_response(
    model_id: str,
    raw_text: str,
    *,
    allowed: set[str],
) -> ModelResponse:
    text = raw_text.strip()
    choice: str | None = None
    rule: str | None = None
    initial = _INITIAL_CHOICE_RE.search(text)
    if initial and initial.group(1).upper() in allowed:
        choice = initial.group(1).upper()
        rule = "leading_choice_letter"
    else:
        answer = _ANSWER_CHOICE_RE.search(text)
        if answer and answer.group(1).upper() in allowed:
            choice = answer.group(1).upper()
            rule = "explicit_answer_phrase"
    if choice is None:
        return ModelResponse(
            model_id=model_id,
            raw_text=raw_text,
            parsed_output=None,
            json_valid=False,
            schema_valid=False,
            validation_errors=["invalid_mcq_choice"],
            metadata={
                "output_contract": "single_choice_text",
                "json_applicable": False,
            },
        )
    canonical = {"choice": choice}
    strict = text == choice
    return ModelResponse(
        model_id=model_id,
        raw_text=raw_text,
        parsed_output=canonical,
        json_valid=True,
        schema_valid=True,
        recoverable_json_valid=True,
        canonical_output=canonical,
        canonical_schema_valid=True,
        canonicalization_rules=[] if strict else [str(rule)],
        metadata={
            "output_contract": "single_choice_text",
            "json_applicable": False,
            "strict_choice_only": strict,
            "choice_extraction_rule": rule,
        },
    )


def _parse_open_response(
    model_id: str,
    raw_text: str,
    spec: DermoBenchSpec,
) -> ModelResponse:
    text = raw_text.strip()
    if not text:
        return ModelResponse(
            model_id=model_id,
            raw_text=raw_text,
            parsed_output=None,
            json_valid=False,
            schema_valid=False,
            validation_errors=["empty_open_response"],
            metadata={
                "output_contract": "free_text",
                "json_applicable": False,
                "judge_status": "not_judgeable",
            },
        )
    diagnostics = _format_diagnostics(text, spec)
    canonical = {"text": text}
    return ModelResponse(
        model_id=model_id,
        raw_text=raw_text,
        parsed_output=canonical,
        json_valid=True,
        schema_valid=True,
        recoverable_json_valid=True,
        canonical_output=canonical,
        canonical_schema_valid=True,
        metadata={
            "output_contract": "free_text",
            "json_applicable": False,
            "judge_status": "pending",
            **diagnostics,
        },
    )


def _format_diagnostics(text: str, spec: DermoBenchSpec) -> dict[str, Any]:
    if spec.key == "task_1_1_description_without_morphology":
        return {"format_compliant": True, "format_errors": []}
    errors: list[str] = []
    if spec.key == "task_1_2_description_with_morphology":
        if not _tag_block(text, "morph"):
            errors.append("missing_morph_block")
        elif _morph_json(text) is None:
            errors.append("invalid_morph_json")
        morph_end = text.casefold().find("</morph>")
        if morph_end < 0 or not text[morph_end + len("</morph>") :].strip():
            errors.append("missing_description_after_morph")
    elif spec.key == "task_3_1_diagnostic_reasoning_without_morphology":
        if not _ordered_tags(text, ("reasoning", "final_diagnosis")):
            errors.append("required_blocks_missing_or_out_of_order")
    elif spec.key == "task_3_2_diagnostic_reasoning_with_morphology":
        if not _ordered_tags(text, ("reasoning", "morph", "final_diagnosis")):
            errors.append("required_blocks_missing_or_out_of_order")
        if _morph_json(text) is None:
            errors.append("invalid_morph_json")
    return {"format_compliant": not errors, "format_errors": errors}


def _tag_block(text: str, tag: str) -> str | None:
    match = re.search(
        rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1) if match else None


def _ordered_tags(text: str, tags: tuple[str, ...]) -> bool:
    position = 0
    for tag in tags:
        match = re.search(
            rf"<{re.escape(tag)}>.*?</{re.escape(tag)}>",
            text[position:],
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match is None:
            return False
        position += match.end()
    return True


def _morph_json(text: str) -> dict[str, Any] | None:
    block = _tag_block(text, "morph")
    if block is None:
        return None
    start = block.find("{")
    end = block.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(block[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _mcq_metrics(
    predictions: list[BenchmarkPrediction],
    spec: DermoBenchSpec,
) -> dict[str, Any]:
    total = len(predictions)
    valid = [p for p in predictions if p.response.canonical_output]
    correct = [
        p
        for p in valid
        if str(p.response.canonical_output.get("choice"))
        == p.ground_truth_disease_id
    ]
    strict_count = sum(
        bool(p.response.metadata.get("strict_choice_only"))
        for p in predictions
    )
    result: dict[str, Any] = {
        "benchmark": spec.benchmark_id,
        "scoring": "deterministic_exact_choice",
        "sample_count": total,
        "valid_choice_count": len(valid),
        "valid_choice_rate": _ratio(len(valid), total),
        "strict_choice_only_rate": _ratio(strict_count, total),
        "correct_count": len(correct),
        "accuracy": _ratio(len(correct), total),
        "invalid_output_count": total - len(valid),
        "json_validity_applicable": False,
        "judge_required": False,
    }
    if spec.key == "task_1_3_derm7pt_morphology_mcq":
        result["accuracy_by_derm7pt_attribute"] = _accuracy_by(
            predictions,
            lambda p: _derm7pt_attribute(str(p.metadata.get("upstream_id", ""))),
        )
    if spec.key == "task_1_4_skincon_morphology_mcq":
        result["accuracy_by_reference_feature"] = _accuracy_by(
            predictions,
            lambda p: str(
                p.metadata.get("options", {}).get(
                    p.ground_truth_disease_id,
                    "unknown",
                )
            ),
        )
    if spec.key == "task_4_ddi_fairness_mcq":
        by_skin = _accuracy_by(
            predictions,
            lambda p: str(p.metadata.get("skin_tone") or "unknown"),
        )
        known = [
            float(value["accuracy"])
            for key, value in by_skin.items()
            if key != "unknown" and int(value["sample_count"]) > 0
        ]
        if len(known) <= 1:
            fairness = 1.0
        elif max(known) <= 0:
            fairness = 0.0
        else:
            fairness = 1.0 - (max(known) - min(known)) / max(known)
        result["accuracy_by_skin_tone"] = by_skin
        result["fairness_score"] = fairness
        result["fairness_score_definition"] = (
            "1 - (max subgroup accuracy - min subgroup accuracy) / "
            "max subgroup accuracy"
        )
    return result


def _open_metrics(
    predictions: list[BenchmarkPrediction],
    spec: DermoBenchSpec,
) -> dict[str, Any]:
    total = len(predictions)
    judgeable = [p for p in predictions if p.response.canonical_output]
    compliant = sum(
        bool(p.response.metadata.get("format_compliant"))
        for p in judgeable
    )
    return {
        "benchmark": spec.benchmark_id,
        "scoring": "llm_as_a_judge_pending",
        "sample_count": total,
        "judgeable_count": len(judgeable),
        "judgeable_rate": _ratio(len(judgeable), total),
        "format_compliant_count": compliant,
        "format_compliance_rate": _ratio(compliant, total),
        "pending_judge_count": len(judgeable),
        "invalid_output_count": total - len(judgeable),
        "json_validity_applicable": False,
        "judge_required": True,
        "judge_protocol": "gemini_3_5_flash_lite_text_only_v1",
        "note": (
            "Clinical content scores are intentionally absent until the "
            "task-specific judge is run."
        ),
    }


def _accuracy_by(
    predictions: list[BenchmarkPrediction],
    key_fn: Any,
) -> dict[str, dict[str, int | float]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for prediction in predictions:
        key = key_fn(prediction) or "unknown"
        counts[str(key)][1] += 1
        output = prediction.response.canonical_output or {}
        if str(output.get("choice", "")) == prediction.ground_truth_disease_id:
            counts[str(key)][0] += 1
    return {
        key: {
            "correct_count": values[0],
            "sample_count": values[1],
            "accuracy": _ratio(values[0], values[1]),
        }
        for key, values in sorted(counts.items())
    }


def _derm7pt_attribute(upstream_id: str) -> str:
    aliases = {
        "pigment_network": "pigment_network",
        "blue_whitish_veil": "blue_whitish_veil",
        "vascular_structures": "vascular_structures",
        "pigmentation": "pigmentation",
        "streaks": "streaks",
        "dots_and_globules": "dots_and_globules",
        "regression_structures": "regression_structures",
    }
    lowered = upstream_id.casefold()
    return next((value for key, value in aliases.items() if key in lowered), "unknown")


def _reference_choice(reference: str, allowed: set[str]) -> str | None:
    match = _INITIAL_CHOICE_RE.search(reference.strip())
    if match and match.group(1).upper() in allowed:
        return match.group(1).upper()
    return None


def _load_ddi_skin_tones(root: Path) -> dict[str, str]:
    path = root / "configs/datasets/ddi/data/ddi_metadata.csv"
    if not path.is_file():
        return {}
    frame = pd.read_csv(path, dtype=str)
    if not {"DDI_file", "skin_tone"}.issubset(frame.columns):
        return {}
    return {
        str(row["DDI_file"]): str(row["skin_tone"])
        for row in frame.to_dict(orient="records")
        if row.get("DDI_file") and row.get("skin_tone")
    }


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not all(
        isinstance(value, dict) for value in values
    ):
        raise ValueError(f"DermoBench task must contain a list of rows: {path}")
    return values


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _spec_tokens(spec: DermoBenchSpec) -> set[str]:
    return {
        spec.key.casefold(),
        spec.benchmark_id.casefold(),
        *(alias.casefold() for alias in spec.aliases),
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _python_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): _python_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_python_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _python_value(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)
