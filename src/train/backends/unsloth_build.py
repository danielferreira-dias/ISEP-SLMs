"""Private Unsloth builders for the fixed E1 scientific recipe."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from src.train.backends.contracts import (
    CheckpointEvent,
    CheckpointObserver,
    FineTuneRequest,
    ModelLoadSpec,
)
from src.train.backends.unsloth_compat import (
    UnslothApi,
    invoke,
    invoke_method,
    required_attribute,
    select_keyword,
)
from src.train.execution.callbacks import TrainerEventBridge


class CollectingCheckpointObserver:
    """Collect checkpoint events while forwarding them durably."""

    def __init__(self, downstream: CheckpointObserver) -> None:
        """Retain a downstream observer and an ordered local event history."""
        self._downstream = downstream
        self._events: list[CheckpointEvent] = []

    @property
    def events(self) -> tuple[CheckpointEvent, ...]:
        """Return checkpoint events in callback order."""
        return tuple(self._events)

    def on_checkpoint(self, event: CheckpointEvent) -> None:
        """Forward and retain one completed checkpoint event."""
        self._downstream.on_checkpoint(event)
        self._events.append(event)


def load_base_pair(api: UnslothApi, spec: ModelLoadSpec) -> tuple[object, object]:
    """Load a pinned BF16 base model and processor without quantization."""
    loaded = invoke_method(
        api.fast_vision_model,
        "from_pretrained",
        required_keywords=frozenset(
            {"model_name", "revision", "dtype", "load_in_4bit"}
        ),
        model_name=spec.model_id,
        revision=spec.revision,
        dtype=required_attribute(api.torch, "bfloat16"),
        load_in_4bit=False,
    )
    if not isinstance(loaded, (tuple, list)) or len(loaded) != 2:
        raise TypeError("FastVisionModel.from_pretrained must return model, processor")
    processor = loaded[1]
    enforce_non_thinking_processor(processor)
    return loaded[0], processor


def enforce_non_thinking_processor(processor: object) -> None:
    """Force every training/inference template render into non-thinking mode."""

    original = required_attribute(processor, "apply_chat_template")
    if getattr(processor, "_isep_non_thinking", False) is True:
        return

    def apply_chat_template(
        *args: object,
        **kwargs: object,
    ) -> object:
        requested = kwargs.get("enable_thinking")
        if requested is not None and requested is not False:
            raise RuntimeError("ISEP E1 forbids enable_thinking=True")
        kwargs["enable_thinking"] = False
        return invoke(
            original,
            *args,
            required_keywords=frozenset({"enable_thinking"}),
            **kwargs,
        )

    try:
        setattr(  # noqa: B010 - dynamic third-party processor boundary
            processor,
            "apply_chat_template",
            apply_chat_template,
        )
        setattr(  # noqa: B010 - marker on dynamic third-party processor
            processor,
            "_isep_non_thinking",
            True,
        )
    except (AttributeError, TypeError) as exc:
        raise RuntimeError("Processor cannot enforce enable_thinking=False") from exc


def apply_lora(api: UnslothApi, model: object, request: FineTuneRequest) -> object:
    """Apply the exact controlled LoRA intervention to a base model.

    ``target_modules="all-linear"`` has special semantics in Unsloth: it
    forcibly enables vision tuning before the component filters are built.
    Passing ``None`` asks Unsloth to discover every eligible linear module
    while still respecting ``finetune_vision_layers``.  The public ISEP
    configuration keeps ``all-linear`` as the scientific intent, and the
    trainable-parameter manifest records the concrete resolved tensors.
    """
    lora = request.lora
    return invoke_method(
        api.fast_vision_model,
        "get_peft_model",
        model,
        required_keywords=frozenset(
            {
                "finetune_vision_layers",
                "finetune_language_layers",
                "finetune_attention_modules",
                "finetune_mlp_modules",
                "r",
                "lora_alpha",
                "lora_dropout",
                "bias",
                "target_modules",
                "use_rslora",
                "use_gradient_checkpointing",
            }
        ),
        finetune_vision_layers=lora.finetune_vision_layers,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        bias=lora.bias,
        target_modules=None,
        use_rslora=False,
        loftq_config=None,
        use_gradient_checkpointing="unsloth",
        random_state=request.trainer.seed,
    )


def build_vision_collator(
    api: UnslothApi,
    model: object,
    processor: object,
) -> object:
    """Create completion-only masking for the canonical assistant label.

    Qwen3.5 renders an empty ``<think>...</think>`` block even when
    ``enable_thinking=False``.  Starting supervision after the closing marker
    keeps those template tokens outside the loss, so E1 learns exactly the
    canonical diagnosis label.
    """
    return invoke(
        api.vision_collator,
        model,
        processor,
        required_keywords=frozenset(
            {
                "train_on_responses_only",
                "instruction_part",
                "response_part",
                "force_match",
            }
        ),
        train_on_responses_only=True,
        instruction_part="<|im_start|>user\n",
        response_part="</think>\n\n",
        force_match=True,
    )


def build_sft_config(api: UnslothApi, request: FineTuneRequest) -> object:
    """Build an SFTConfig with no silent scientific argument downgrade."""
    spec = request.trainer
    eval_key = select_keyword(
        api.sft_config,
        ("eval_strategy", "evaluation_strategy"),
    )
    length_key = select_keyword(
        api.sft_config,
        ("max_length", "max_seq_length"),
    )
    values: dict[str, object] = {
        "output_dir": str(spec.output_dir),
        "per_device_train_batch_size": spec.per_device_train_batch_size,
        "per_device_eval_batch_size": spec.per_device_eval_batch_size,
        "gradient_accumulation_steps": spec.gradient_accumulation_steps,
        "num_train_epochs": spec.num_train_epochs,
        "max_steps": spec.max_steps,
        "learning_rate": spec.learning_rate,
        "weight_decay": spec.weight_decay,
        "warmup_ratio": spec.warmup_ratio,
        "max_grad_norm": spec.max_grad_norm,
        "logging_steps": spec.logging_steps,
        "eval_steps": spec.eval_steps,
        "seed": spec.seed,
        "data_seed": spec.seed,
        "bf16": True,
        "fp16": False,
        "optim": "adamw_8bit",
        "lr_scheduler_type": "linear",
        "save_strategy": "epoch",
        eval_key: "steps",
        length_key: spec.max_length,
        "packing": False,
        "gradient_checkpointing": True,
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "dataset_num_proc": spec.dataset_num_proc,
        "report_to": "none",
        "save_only_model": False,
        "completion_only_loss": True,
    }
    scientific = frozenset(
        {
            "output_dir",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "num_train_epochs",
            "max_steps",
            "learning_rate",
            "weight_decay",
            "warmup_ratio",
            "max_grad_norm",
            "logging_steps",
            "eval_steps",
            "seed",
            "data_seed",
            "bf16",
            "fp16",
            "optim",
            "lr_scheduler_type",
            "save_strategy",
            eval_key,
            length_key,
            "packing",
            "gradient_checkpointing",
            "remove_unused_columns",
            "completion_only_loss",
        }
    )
    return invoke(api.sft_config, required_keywords=scientific, **values)


def build_sft_trainer(
    api: UnslothApi,
    *,
    request: FineTuneRequest,
    model: object,
    processor: object,
    collator: object,
    trainer_config: object,
    callback: object,
) -> object:
    """Create the TRL trainer with the multimodal collator unchanged."""
    processor_key = select_keyword(
        api.sft_trainer,
        ("processing_class", "tokenizer"),
    )
    values: dict[str, object] = {
        "model": model,
        processor_key: processor,
        "data_collator": collator,
        "train_dataset": request.train_dataset,
        "eval_dataset": request.eval_dataset,
        "args": trainer_config,
        "callbacks": [callback],
    }
    return invoke(api.sft_trainer, required_keywords=frozenset(values), **values)


def framework_callback(base: type[object], bridge: TrainerEventBridge) -> object:
    """Create a patched-framework callback only after Unsloth imports."""

    def on_log(
        callback_self: object,
        args: object,
        state: object,
        control: object,
        logs: object | None = None,
        **kwargs: object,
    ) -> object:
        del callback_self, args, kwargs
        bridge.on_log(state=state, logs=logs)
        return control

    def on_save(
        callback_self: object,
        args: object,
        state: object,
        control: object,
        **kwargs: object,
    ) -> object:
        del callback_self, args, kwargs
        bridge.on_save(state=state)
        return control

    callback_type = type(
        "ISEPTrainerCallback",
        (base,),
        {"on_log": on_log, "on_save": on_save},
    )
    return callback_type()


def record_unobserved_checkpoints(
    output_dir: Path,
    collector: CollectingCheckpointObserver,
) -> None:
    """Record checkpoint directories missed by a framework callback."""
    observed = {event.path.resolve() for event in collector.events}
    for path in sorted(output_dir.glob("checkpoint-*"), key=checkpoint_step):
        if path.is_dir() and path.resolve() not in observed:
            collector.on_checkpoint(
                CheckpointEvent(
                    path=path,
                    global_step=checkpoint_step(path),
                    epoch=None,
                )
            )


def checkpoint_step(path: Path) -> int:
    """Extract a global step from a Trainer checkpoint directory."""
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def numeric_mapping(value: object) -> dict[str, float | int]:
    """Keep only scalar numeric values from a dynamic metrics mapping."""
    if not isinstance(value, Mapping):
        return {}
    return {
        key: item
        for key, item in value.items()
        if isinstance(key, str)
        and not isinstance(item, bool)
        and isinstance(item, (int, float))
    }


def numeric_attribute(instance: object, name: str, *, integer: bool) -> int | float:
    """Read a required numeric dynamic-framework attribute."""
    value = getattr(instance, name, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Trainer state {name} is not numeric")
    return int(value) if integer else float(value)
