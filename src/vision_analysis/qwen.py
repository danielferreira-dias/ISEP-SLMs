"""Direct Transformers scoring adapter for Qwen 3.5 multimodal models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
import torch
import yaml


@dataclass(frozen=True, slots=True)
class LabelScore:
    """Teacher-forced log-probability summary for a diagnosis string."""

    target: str
    target_token_ids: tuple[int, ...]
    sum_log_probability: float
    mean_log_probability: float


class QwenVisualScorer:
    """Score fixed diagnosis continuations without sampling or JSON parsing."""

    def __init__(
        self,
        model_config_path: Path,
        *,
        device: str,
        dtype: str,
        revision: str | None = None,
    ) -> None:
        raw = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
        self.repo_id = str(raw["source"]["repo_id"])
        self.revision = str(
            revision or raw["source"].get("revision") or "main"
        )
        self.processor_repo_id = str(raw["processor"]["repo_id"])
        self.processor_revision = self.revision
        self.trust_remote_code = bool(
            raw.get("security", {}).get("trust_remote_code", False)
        )
        self.device = device
        self.dtype_name = dtype
        self.processor: Any | None = None
        self.model: Any | None = None

    def load(self) -> None:
        """Load the processor and checkpoint on the requested device."""

        from transformers import AutoModelForMultimodalLM, AutoProcessor

        dtype = _resolve_dtype(self.dtype_name)
        self.processor = AutoProcessor.from_pretrained(
            self.processor_repo_id,
            revision=self.processor_revision,
            trust_remote_code=self.trust_remote_code,
        )
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.repo_id,
            revision=self.revision,
            dtype=dtype,
            trust_remote_code=self.trust_remote_code,
            low_cpu_mem_usage=True,
            device_map={"": self.device},
        )
        self.model.eval()
        self.model.requires_grad_(False)

    def score_label(
        self,
        image: Image.Image,
        *,
        system_prompt: str,
        user_prompt: str,
        target: str,
    ) -> LabelScore:
        """Return the log-probability of a fixed assistant continuation."""

        if self.model is None or self.processor is None:
            raise RuntimeError("load() must be called before score_label()")
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = inputs.to(self.device)
        target_ids = self.processor.tokenizer(
            target,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)
        if target_ids.shape[1] == 0:
            raise ValueError("target must contain at least one token")
        full_inputs = dict(inputs)
        full_inputs["input_ids"] = torch.cat(
            [inputs["input_ids"], target_ids],
            dim=1,
        )
        full_inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones_like(target_ids)],
            dim=1,
        )
        if "mm_token_type_ids" in inputs:
            full_inputs["mm_token_type_ids"] = torch.cat(
                [inputs["mm_token_type_ids"], torch.zeros_like(target_ids)],
                dim=1,
            )
        token_count = int(target_ids.shape[1])
        with torch.inference_mode():
            outputs = self.model(
                **full_inputs,
                use_cache=False,
                logits_to_keep=token_count + 1,
            )
        logits = outputs.logits[:, :token_count, :].float()
        log_probabilities = torch.log_softmax(logits, dim=-1)
        selected = log_probabilities.gather(
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)
        return LabelScore(
            target=target,
            target_token_ids=tuple(int(value) for value in target_ids[0]),
            sum_log_probability=float(selected.sum().item()),
            mean_log_probability=float(selected.mean().item()),
        )


def _resolve_dtype(name: str) -> torch.dtype:
    normalized = name.lower().replace("torch.", "")
    by_name = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if normalized not in by_name:
        raise ValueError(f"unsupported dtype: {name!r}")
    return by_name[normalized]
