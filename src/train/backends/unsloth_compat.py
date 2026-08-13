"""Private dynamic compatibility boundary for Unsloth, TRL, and PEFT."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType


@dataclass(frozen=True, slots=True)
class UnslothApi:
    """Hold dynamically imported framework factories."""

    fast_vision_model: object
    vision_collator: object
    sft_trainer: object
    sft_config: object
    trainer_callback: type[object]
    peft_model: object
    torch: ModuleType


def load_unsloth_api() -> UnslothApi:
    """Import Unsloth before libraries whose modules it patches.

    Returns:
        Dynamic API handles used only by ``UnslothBackend``.
    """
    unsloth = importlib.import_module("unsloth")
    unsloth_trainer = importlib.import_module("unsloth.trainer")
    trl = importlib.import_module("trl")
    transformers = importlib.import_module("transformers")
    peft = importlib.import_module("peft")
    torch = importlib.import_module("torch")
    callback = required_attribute(transformers, "TrainerCallback")
    if not isinstance(callback, type):
        raise TypeError("transformers.TrainerCallback is not a class")
    return UnslothApi(
        fast_vision_model=required_attribute(unsloth, "FastVisionModel"),
        vision_collator=required_attribute(
            unsloth_trainer,
            "UnslothVisionDataCollator",
        ),
        sft_trainer=required_attribute(trl, "SFTTrainer"),
        sft_config=required_attribute(trl, "SFTConfig"),
        trainer_callback=callback,
        peft_model=required_attribute(peft, "PeftModel"),
        torch=torch,
    )


def required_attribute(instance: object, name: str) -> object:
    """Return a named attribute or fail with an actionable error."""
    value = getattr(instance, name, None)
    if value is None:
        raise AttributeError(f"{type(instance).__name__} is missing {name}")
    return value


def invoke(
    callable_object: object,
    /,
    *args: object,
    required_keywords: frozenset[str] = frozenset(),
    **kwargs: object,
) -> object:
    """Invoke a dynamic API while refusing to drop scientific arguments.

    Unknown non-critical keywords are omitted only for older signatures.
    Critical keywords must be accepted explicitly or through ``**kwargs``.
    """
    if not callable(callable_object):
        raise TypeError(f"Object is not callable: {callable_object!r}")
    parameters, accepts_extra = signature_parameters(callable_object)
    if parameters is None or accepts_extra:
        return callable_object(*args, **kwargs)
    missing = required_keywords - parameters
    if missing:
        raise RuntimeError(
            "Installed training package does not support required arguments: "
            + ", ".join(sorted(missing))
        )
    compatible = {key: value for key, value in kwargs.items() if key in parameters}
    return callable_object(*args, **compatible)


def invoke_method(
    instance: object,
    name: str,
    /,
    *args: object,
    required_keywords: frozenset[str] = frozenset(),
    **kwargs: object,
) -> object:
    """Invoke a named method through the same strict compatibility boundary."""
    return invoke(
        required_attribute(instance, name),
        *args,
        required_keywords=required_keywords,
        **kwargs,
    )


def select_keyword(callable_object: object, candidates: tuple[str, ...]) -> str:
    """Select the first installed alias for a semantically required argument."""
    parameters, accepts_extra = signature_parameters(callable_object)
    if parameters is None or accepts_extra:
        return candidates[0]
    for candidate in candidates:
        if candidate in parameters:
            return candidate
    raise RuntimeError(
        "Installed training package supports none of the required aliases: "
        + ", ".join(candidates)
    )


def signature_parameters(
    callable_object: object,
) -> tuple[frozenset[str] | None, bool]:
    """Inspect a callable signature, returning ``None`` when unavailable."""
    if not callable(callable_object):
        return None, True
    callable_value: Callable[..., object] = callable_object
    try:
        signature = inspect.signature(callable_value)
    except (TypeError, ValueError):
        return None, True
    parameters = signature.parameters
    accepts_extra = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    return frozenset(parameters), accepts_extra
