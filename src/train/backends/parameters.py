"""Auditable trainable-parameter inspection for LoRA experiments."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, cast

from src.train.backends.contracts import (
    LoraSpec,
    ParameterComponent,
    TrainableParameter,
    TrainableParameterManifest,
)


class _Parameter(Protocol):
    """Structural subset of a torch parameter used by this module."""

    requires_grad: bool

    def numel(self) -> int:
        """Return the number of scalar values in the parameter."""


class _NamedParameterModel(Protocol):
    """Structural subset of a torch module used by this module."""

    def named_parameters(self) -> Iterable[tuple[str, _Parameter]]:
        """Iterate over parameter names and tensors."""


def classify_parameter(name: str) -> ParameterComponent:
    """Classify a parameter name into a thesis reporting component.

    Args:
        name: Fully qualified parameter name returned by PyTorch.

    Returns:
        Stable component used in manifests and figures.
    """
    lowered = name.casefold()
    if any(token in lowered for token in ("visual", "vision", "image_tower")):
        return "vision"
    if any(
        token in lowered
        for token in (
            "self_attn",
            "attention",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        )
    ):
        return "attention"
    if any(
        token in lowered
        for token in ("mlp", "gate_proj", "up_proj", "down_proj", "ffn")
    ):
        return "mlp"
    if any(token in lowered for token in ("embed", "lm_head")):
        return "embedding"
    return "language_other"


def build_trainable_parameter_manifest(
    model: object,
) -> TrainableParameterManifest:
    """Build a deterministic manifest from a model's trainable tensors.

    Args:
        model: Object implementing PyTorch's ``named_parameters`` contract.

    Returns:
        Immutable manifest sorted by fully qualified tensor name.

    Raises:
        TypeError: If the model does not expose ``named_parameters``.
        ValueError: If no trainable parameter exists.
    """
    if not hasattr(model, "named_parameters"):
        raise TypeError("model does not expose named_parameters()")
    typed_model = cast(_NamedParameterModel, model)
    parameters = tuple(
        sorted(
            (
                TrainableParameter(
                    name=name,
                    component=classify_parameter(name),
                    count=int(parameter.numel()),
                )
                for name, parameter in typed_model.named_parameters()
                if parameter.requires_grad
            ),
            key=lambda item: item.name,
        )
    )
    if not parameters:
        raise ValueError("LoRA setup produced zero trainable parameters")

    components: tuple[ParameterComponent, ...] = (
        "vision",
        "attention",
        "mlp",
        "embedding",
        "language_other",
    )
    by_component: dict[ParameterComponent, int] = {
        component: sum(
            parameter.count
            for parameter in parameters
            if parameter.component == component
        )
        for component in components
    }
    return TrainableParameterManifest(
        parameters=parameters,
        total_trainable=sum(parameter.count for parameter in parameters),
        by_component=by_component,
    )


def validate_trainable_parameter_manifest(
    manifest: TrainableParameterManifest,
    lora: LoraSpec,
) -> None:
    """Verify that trainable tensors match the requested LoRA intervention.

    Args:
        manifest: Manifest produced after applying the PEFT adapter.
        lora: Requested intervention flags.

    Raises:
        RuntimeError: If a requested component is absent or an excluded vision
            component is trainable.
    """
    vision_count = manifest.by_component["vision"]
    non_lora = tuple(
        parameter.name
        for parameter in manifest.parameters
        if "lora_" not in parameter.name.casefold()
    )
    if non_lora:
        preview = ", ".join(non_lora[:5])
        raise RuntimeError(
            "The controlled recipe permits LoRA tensors only; unexpected "
            "trainable parameters: " + preview
        )
    if lora.finetune_vision_layers and vision_count == 0:
        raise RuntimeError(
            "finetune_vision_layers=True but no vision parameter is trainable"
        )
    if not lora.finetune_vision_layers and vision_count > 0:
        raise RuntimeError(
            "finetune_vision_layers=False but vision parameters are trainable"
        )

    language_count = manifest.total_trainable - vision_count
    if lora.finetune_language_layers and language_count == 0:
        raise RuntimeError(
            "finetune_language_layers=True but no language parameter is trainable"
        )
    if lora.finetune_attention_modules and manifest.by_component["attention"] == 0:
        raise RuntimeError(
            "finetune_attention_modules=True but no attention parameter is trainable"
        )
    if lora.finetune_mlp_modules and manifest.by_component["mlp"] == 0:
        raise RuntimeError(
            "finetune_mlp_modules=True but no MLP parameter is trainable"
        )
