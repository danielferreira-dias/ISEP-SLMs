"""Load and validate frozen qualitative cases from ISEPDermaBench."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from PIL import Image
import yaml


@dataclass(frozen=True, slots=True)
class VisualCase:
    """One image with frozen gold and benchmark-predicted targets."""

    task_id: str
    cohort: str
    image: Image.Image
    image_bytes: bytes
    source: str
    skin_tone: str | None
    age_group: str | None
    gold_disease_id: str
    predicted_disease_id: str
    gold_disease_name: str
    predicted_disease_name: str
    source_image_sha256: str
    benchmark_image_sha256: str


@dataclass(frozen=True, slots=True)
class PilotConfig:
    """Normalized visual-attribution pilot configuration."""

    analysis_id: str
    model_config_path: Path
    model_revision: str
    dataset_root: Path
    split: str
    taxonomy_path: Path
    prediction_path: Path
    case_specs: tuple[dict[str, Any], ...]
    grid_size: int
    blur_radius_fraction: float
    attribution_system_prompt: str
    attribution_user_prompt: str


def load_pilot_config(path: Path, *, project_root: Path) -> PilotConfig:
    """Read a project-relative pilot YAML and validate its core fields."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("visual pilot schema_version must be 1")
    method = raw["method"]
    grid_size = int(method["grid_size"])
    if grid_size < 2:
        raise ValueError("method.grid_size must be at least 2")
    fraction = float(method["blur_radius_fraction"])
    if fraction <= 0:
        raise ValueError("method.blur_radius_fraction must be positive")

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else project_root / candidate

    dataset = raw["dataset"]
    prompts = raw["prompts"]
    return PilotConfig(
        analysis_id=str(raw["analysis_id"]),
        model_config_path=resolve(raw["model_config"]),
        model_revision=str(raw["model_revision"]),
        dataset_root=resolve(dataset["root"]),
        split=str(dataset["split"]),
        taxonomy_path=resolve(dataset["taxonomy"]),
        prediction_path=resolve(dataset["prediction_file"]),
        case_specs=tuple(raw["cases"]),
        grid_size=grid_size,
        blur_radius_fraction=fraction,
        attribution_system_prompt=str(prompts["system_prompt"]),
        attribution_user_prompt=str(prompts["user_prompt"]),
    )


def load_visual_cases(config: PilotConfig) -> list[VisualCase]:
    """Join embedded images, references, taxonomy, and prior predictions."""

    task_ids = [str(item["task_id"]) for item in config.case_specs]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("pilot cases must contain unique task_id values")
    tasks = _read_split(
        config.dataset_root / "tasks/visual_top_k",
        config.split,
    )
    references = _read_split(
        config.dataset_root / "references/visual_top_k",
        config.split,
    )
    tasks = tasks[tasks["task_id"].astype(str).isin(task_ids)].copy()
    references = references[
        references["task_id"].astype(str).isin(task_ids)
    ].copy()
    if len(tasks) != len(task_ids) or len(references) != len(task_ids):
        raise ValueError("one or more frozen task_ids are missing from the split")
    merged = tasks.merge(
        references,
        on="task_id",
        suffixes=("_task", "_reference"),
        validate="one_to_one",
    ).set_index("task_id")
    taxonomy = yaml.safe_load(config.taxonomy_path.read_text(encoding="utf-8"))
    names = {
        str(item["id"]): str(item["display_name"])
        for item in taxonomy["diseases"]
    }
    predictions = _load_predictions(config.prediction_path)
    cases: list[VisualCase] = []
    for spec in config.case_specs:
        task_id = str(spec["task_id"])
        row = merged.loc[task_id]
        image_bytes = _extract_image_bytes(row["image"])
        benchmark_hash = sha256(image_bytes).hexdigest()
        expected_hash = str(spec["benchmark_image_sha256"])
        if benchmark_hash != expected_hash:
            raise ValueError(
                f"benchmark image hash mismatch for {task_id}: "
                f"{benchmark_hash} != {expected_hash}"
            )
        gold_id = str(row["reference_disease_id"])
        predicted_id = predictions[task_id]
        if gold_id != str(spec["gold_disease_id"]):
            raise ValueError(f"gold target drift detected for {task_id}")
        if predicted_id != str(spec["predicted_disease_id"]):
            raise ValueError(f"predicted target drift detected for {task_id}")
        cases.append(
            VisualCase(
                task_id=task_id,
                cohort=str(spec["cohort"]),
                image=Image.open(BytesIO(image_bytes)).convert("RGB"),
                image_bytes=image_bytes,
                source=str(row["source_task"]),
                skin_tone=_optional_text(row.get("skin_tone")),
                age_group=_optional_text(row.get("age_group_standardized")),
                gold_disease_id=gold_id,
                predicted_disease_id=predicted_id,
                gold_disease_name=names[gold_id],
                predicted_disease_name=names[predicted_id],
                source_image_sha256=str(row["source_image_sha256"]),
                benchmark_image_sha256=benchmark_hash,
            )
        )
    return cases


def _read_split(directory: Path, split: str) -> pd.DataFrame:
    shards = sorted(directory.glob(f"{split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no {split!r} shards found under {directory}")
    return pd.concat((pd.read_parquet(path) for path in shards), ignore_index=True)


def _load_predictions(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            output = record.get("response", {}).get("canonical_output", {})
            predictions = output.get("predictions") or []
            if predictions:
                values[str(record["task_id"])] = str(
                    predictions[0]["disease_id"]
                )
    return values


def _extract_image_bytes(value: Any) -> bytes:
    if isinstance(value, dict) and isinstance(value.get("bytes"), bytes):
        return value["bytes"]
    raise ValueError("expected an embedded Hugging Face image struct")


def _optional_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None
