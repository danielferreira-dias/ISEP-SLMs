"""Direct Transformers backend for local multimodal smoke inference.

This backend is intentionally secondary to vLLM. It makes small, sequential
Apple MPS or CPU smoke tests possible while preserving the same normalized
request and result contracts used by the production benchmark executor.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from io import BytesIO
from typing import Any, Mapping

from src.config.models import LocalModelConfig
from src.inference.base import (
    InferenceBackend,
    InferenceConfigurationError,
    InferenceRequest,
    InferenceResult,
    InferenceTransportError,
    TokenUsage,
    build_reasoning_trace,
    merge_generation,
    validate_reasoning_capture,
)


class TransformersBackend(InferenceBackend):
    """Load one Hugging Face multimodal model in-process."""

    def __init__(
        self,
        config: LocalModelConfig,
        *,
        reasoning_capture: str = "available",
    ) -> None:
        self.config = config
        profile = config.backend.active_profile
        if profile.engine != "transformers":
            raise InferenceConfigurationError(
                "TransformersBackend requires engine='transformers'"
            )
        self.device = profile.device or "cpu"
        self.dtype = profile.dtype or "auto"
        reasoning_enabled = bool(config.reasoning.enabled)
        configured_capture = (
            "full"
            if reasoning_capture == "available" and reasoning_enabled
            else "none"
            if reasoning_capture == "available"
            else reasoning_capture
        )
        self.reasoning_capture = validate_reasoning_capture(
            configured_capture
        )
        self.chat_template_kwargs = _mapping_values(
            config.reasoning.chat_template_kwargs
        )
        self.default_generation = config.generation
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    @property
    def model_id(self) -> str:
        return self.config.model.id

    def complete(self, request: InferenceRequest) -> InferenceResult:
        self._ensure_loaded()
        try:
            return self._generate(request)
        except Exception as exc:
            raise InferenceTransportError(
                f"Direct Transformers inference failed for "
                f"{self.model_id!r}: {type(exc).__name__}: {exc}"
            ) from None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import (
                AutoModelForMultimodalLM,
                AutoProcessor,
            )
        except ImportError:
            raise InferenceConfigurationError(
                "Direct Transformers inference requires torch, transformers, "
                "and Pillow"
            ) from None

        repo_id = self.config.source.repo_id
        if not repo_id:
            raise InferenceConfigurationError(
                "A Hugging Face source.repo_id is required"
            )
        processor_repo = (
            self.config.processor.repo_id
            if self.config.processor is not None
            else repo_id
        )
        revision = self.config.source.revision or "main"
        processor_revision = (
            self.config.processor.revision
            if self.config.processor is not None
            else revision
        )
        dtype = (
            "auto"
            if self.dtype == "auto"
            else getattr(torch, self.dtype, None)
        )
        if dtype is None:
            raise InferenceConfigurationError(
                f"Unsupported torch dtype: {self.dtype!r}"
            )
        self._processor = AutoProcessor.from_pretrained(
            processor_repo,
            revision=processor_revision,
            trust_remote_code=self.config.security.trust_remote_code,
        )
        self._model = AutoModelForMultimodalLM.from_pretrained(
            repo_id,
            revision=revision,
            dtype=dtype,
            trust_remote_code=self.config.security.trust_remote_code,
        )
        self._model.to(self.device)
        self._model.eval()
        self._torch = torch

    def _generate(self, request: InferenceRequest) -> InferenceResult:
        from PIL import Image

        generation = merge_generation(
            self.default_generation,
            request.generation,
        )
        presence_penalty = generation.get("presence_penalty")
        if presence_penalty not in {None, 0, 0.0}:
            raise InferenceConfigurationError(
                "Direct Transformers generation does not support a "
                "non-zero presence_penalty"
            )
        image = Image.open(BytesIO(request.image_bytes)).convert("RGB")
        messages: list[dict[str, Any]] = []
        if request.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": request.system_prompt,
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": request.user_prompt},
                ],
            }
        )
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        ).to(self.device)
        input_tokens = int(inputs["input_ids"].shape[-1])
        max_new_tokens = _max_output_tokens(generation)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": bool(generation.get("do_sample", False)),
        }
        if generate_kwargs["do_sample"]:
            for name in ("temperature", "top_p", "top_k", "min_p"):
                if generation.get(name) is not None:
                    generate_kwargs[name] = generation[name]
        repetition_penalty = generation.get("repetition_penalty")
        if repetition_penalty is not None:
            generate_kwargs["repetition_penalty"] = repetition_penalty
        seed = generation.get("seed")
        if seed is not None:
            self._torch.manual_seed(int(seed))

        with self._torch.inference_mode():
            outputs = self._model.generate(**inputs, **generate_kwargs)
        generated_ids = outputs[0, input_tokens:]
        output_tokens = int(generated_ids.shape[-1])
        raw_text = self._processor.decode(
            generated_ids,
            skip_special_tokens=True,
        )
        reasoning_text, final_text = _split_qwen_reasoning(raw_text)
        reasoning_token_count = (
            len(
                self._processor.tokenizer.encode(
                    reasoning_text,
                    add_special_tokens=False,
                )
            )
            if reasoning_text
            else None
        )
        reasoning = build_reasoning_trace(
            mode=self.reasoning_capture,
            full_text=reasoning_text,
            summary_text=None,
            token_count=reasoning_token_count,
            full_source="generated_think_block" if reasoning_text else None,
        )
        truncated = output_tokens >= max_new_tokens
        return InferenceResult(
            model_id=self.model_id,
            final_text=final_text,
            reasoning=reasoning,
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                reasoning_tokens=reasoning_token_count,
            ),
            request_id=request.request_id,
            finish_reason="length" if truncated else "stop",
            metadata={
                "runtime": "transformers_direct",
                "device": self.device,
                "truncated": truncated,
            },
        )


def _max_output_tokens(generation: Mapping[str, Any]) -> int:
    for name in ("max_output_tokens", "max_new_tokens", "max_tokens"):
        value = generation.get(name)
        if isinstance(value, int) and value > 0:
            return value
    return 512


def _split_qwen_reasoning(text: str) -> tuple[str | None, str]:
    value = text.strip()
    start = value.find("<think>")
    end = value.find("</think>")
    if end >= 0:
        reasoning_start = start + len("<think>") if start >= 0 else 0
        reasoning = value[reasoning_start:end].strip() or None
        return reasoning, value[end + len("</think>"):].strip()
    if start >= 0:
        return value[start + len("<think>"):].strip() or None, ""
    return None, value


def _mapping_values(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return {}
