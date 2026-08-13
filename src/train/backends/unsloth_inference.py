"""Deterministic multimodal inference for the Unsloth adapter."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import Protocol, cast

from src.train.backends.contracts import (
    BackendPrediction,
    GenerationSpec,
    LoadedCheckpoint,
    PredictionSample,
)
from src.train.backends.unsloth_compat import (
    invoke,
    invoke_method,
    load_unsloth_api,
    required_attribute,
)


def predict_samples(
    *,
    loaded: LoadedCheckpoint,
    samples: Sequence[PredictionSample],
    generation: GenerationSpec,
) -> tuple[BackendPrediction, ...]:
    """Generate deterministic, thinking-disabled responses in input order."""
    api = load_unsloth_api()
    invoke_method(loaded.model, "eval")
    predictions: list[BackendPrediction] = []
    for sample in samples:
        rendered = _render_prompt(loaded.processor, sample)
        inputs = invoke(
            loaded.processor,
            required_keywords=frozenset({"text", "images", "return_tensors"}),
            text=[rendered],
            images=[sample.image],
            return_tensors="pt",
            padding=True,
        )
        inputs = invoke_method(
            inputs,
            "to",
            getattr(loaded.model, "device", "cuda:0"),
        )
        tensor_inputs = _string_mapping(inputs)
        input_length = _last_dimension(tensor_inputs.get("input_ids"))
        context = invoke(required_attribute(api.torch, "inference_mode"))
        with cast(AbstractContextManager[object], context):
            output = invoke_method(
                loaded.model,
                "generate",
                required_keywords=frozenset(
                    {"max_new_tokens", "do_sample", "num_beams"}
                ),
                **tensor_inputs,
                max_new_tokens=generation.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
        generated = _slice_first_row(output, input_length)
        decoded = invoke_method(
            loaded.processor,
            "batch_decode",
            [generated],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        predictions.append(
            BackendPrediction(
                sample_id=sample.sample_id,
                text=_first_string(decoded).strip(),
            )
        )
    return tuple(predictions)


def _render_prompt(processor: object, sample: PredictionSample) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample.image},
                {"type": "text", "text": sample.prompt},
            ],
        }
    ]
    rendered = invoke_method(
        processor,
        "apply_chat_template",
        messages,
        required_keywords=frozenset(
            {"tokenize", "add_generation_prompt", "enable_thinking"}
        ),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str):
        raise TypeError("apply_chat_template did not return text")
    return rendered


def _string_mapping(value: object) -> dict[str, object]:
    items = getattr(value, "items", None)
    if not callable(items):
        raise TypeError("Processor output is not a mapping")
    output: dict[str, object] = {}
    for key, item in items():
        if not isinstance(key, str):
            raise TypeError("Processor output contains a non-string key")
        output[key] = item
    return output


def _last_dimension(value: object | None) -> int:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 1:
        raise TypeError("Processor input_ids has no shape")
    return int(shape[-1])


def _slice_first_row(value: object, start: int) -> object:
    first = cast(_Indexable, value)[0]
    return cast(_Indexable, first)[slice(start, None)]


class _Indexable(Protocol):
    """Minimal tensor-like indexing contract used by dynamic inference."""

    def __getitem__(self, key: int | slice) -> object:
        """Return an indexed or sliced tensor-like value."""


def _first_string(value: object) -> str:
    if not isinstance(value, (list, tuple)) or not value:
        raise TypeError("batch_decode did not return a non-empty sequence")
    first = value[0]
    if not isinstance(first, str):
        raise TypeError("batch_decode returned a non-string item")
    return first
