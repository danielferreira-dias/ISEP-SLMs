"""Hashes and invariants for controlled training comparisons."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from src.train.config import TrainingConfig
from src.train.domain import JsonValue, Taxonomy


def canonical_json_hash(value: JsonValue) -> str:
    """Return a SHA-256 digest of one canonical JSON value.

    Args:
        value: JSON-compatible value whose mapping keys may be reordered.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolved_config_document(config: TrainingConfig) -> dict[str, JsonValue]:
    """Convert a validated config to a path-independent JSON document.

    The source YAML path and auto-discovered project root are execution
    details, not scientific hyperparameters, and are deliberately excluded.
    """

    decoded: object = json.loads(
        config.model_dump_json(
            exclude={"project_root", "source_config_path"},
            exclude_none=False,
        )
    )
    return _json_object(decoded, "resolved training config")


def config_hash(config: TrainingConfig) -> str:
    """Hash every declared run setting except local path-discovery fields."""

    return canonical_json_hash(resolved_config_document(config))


def controlled_training_document(
    config: TrainingConfig,
) -> dict[str, JsonValue]:
    """Return settings that must match across the two E1 conditions.

    Experiment naming and the single intended intervention—whether visual
    LoRA is enabled—are removed. Artifact destinations are also excluded.
    Every data, model, optimization, seed, preprocessing, and evaluation
    setting remains in the comparison contract.
    """

    document = resolved_config_document(config)
    experiment = _child_object(document, "experiment")
    experiment.pop("id", None)
    experiment.pop("vision_profile", None)
    lora = _child_object(document, "lora")
    lora.pop("finetune_vision_layers", None)
    document.pop("artifacts", None)
    return document


def controlled_training_hash(config: TrainingConfig) -> str:
    """Hash all scientific settings shared by the paired E1 recipes."""

    return canonical_json_hash(controlled_training_document(config))


def replicate_training_document(config: TrainingConfig) -> dict[str, JsonValue]:
    """Return the cross-seed contract, excluding only the random seed."""

    document = controlled_training_document(config)
    trainer = _child_object(document, "trainer")
    trainer.pop("seed", None)
    return document


def prompt_hash(prompt: str) -> str:
    """Hash the exact UTF-8 prompt shown to the model."""

    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def label_contract_hash(taxonomy: Taxonomy) -> str:
    """Hash the ordered disease-ID and canonical-label contract."""

    pairs: list[JsonValue] = [
        {"disease_id": item.disease_id, "label": item.label}
        for item in taxonomy.classes
    ]
    return canonical_json_hash(pairs)


def validate_controlled_pair(
    frozen_vision: TrainingConfig,
    unsloth_all: TrainingConfig,
) -> None:
    """Verify that two configs differ only in the intended vision condition.

    Args:
        frozen_vision: Configuration expected to disable visual LoRA.
        unsloth_all: Configuration expected to enable visual LoRA.

    Raises:
        ValueError: If the profiles are reversed or another scientific
            setting differs.
    """

    if frozen_vision.lora.finetune_vision_layers:
        raise ValueError("The frozen-vision config enables visual LoRA")
    if not unsloth_all.lora.finetune_vision_layers:
        raise ValueError("The Unsloth-all config disables visual LoRA")
    if controlled_training_document(frozen_vision) != controlled_training_document(
        unsloth_all
    ):
        raise ValueError("Controlled E1 recipes differ outside finetune_vision_layers")


def _json_object(value: object, context: str) -> dict[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a JSON object")
    result: dict[str, JsonValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{context} contains a non-string key")
        result[key] = _json_value(item, context)
    return result


def _json_value(value: object, context: str) -> JsonValue:
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, list):
        return [_json_value(item, context) for item in value]
    if isinstance(value, Mapping):
        return _json_object(value, context)
    raise TypeError(f"{context} contains a non-JSON value: {type(value).__name__}")


def _child_object(document: dict[str, JsonValue], key: str) -> dict[str, JsonValue]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Resolved config field {key!r} must be an object")
    return value
