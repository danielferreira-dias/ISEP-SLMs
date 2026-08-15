"""Unsloth implementation of the typed fine-tuning backend contract."""

from __future__ import annotations

import gc
from collections.abc import Sequence, Sized
from pathlib import Path

from src.train.backends.contracts import (
    BackendFitResult,
    BackendPrediction,
    CheckpointObserver,
    FineTuneRequest,
    GenerationSpec,
    LoadedCheckpoint,
    MetricEvent,
    MetricSink,
    ModelLoadSpec,
    PredictionSample,
    RuntimeInfo,
)
from src.train.backends.masking import audit_response_only_mask
from src.train.backends.parameters import (
    build_trainable_parameter_manifest,
    validate_trainable_parameter_manifest,
)
from src.train.backends.runtime import validate_nvidia_bf16_runtime
from src.train.backends.sample_costs import (
    SampleCostAuditingCollator,
    materialize_sample_costs,
)
from src.train.backends.unsloth_build import (
    CollectingCheckpointObserver,
    apply_lora,
    build_sft_config,
    build_sft_trainer,
    build_vision_collator,
    framework_callback,
    load_base_pair,
    numeric_attribute,
    numeric_mapping,
    record_unobserved_checkpoints,
)
from src.train.backends.unsloth_compat import (
    invoke_method,
    load_unsloth_api,
    required_attribute,
)
from src.train.backends.unsloth_inference import predict_samples
from src.train.execution.callbacks import TrainerEventBridge


class UnslothBackend:
    """Train and evaluate BF16 vision-language LoRA adapters with Unsloth."""

    @property
    def name(self) -> str:
        """Return the stable backend identifier written to manifests."""
        return "unsloth"

    def validate_runtime(self) -> RuntimeInfo:
        """Reject CPU, non-NVIDIA CUDA, and devices without BF16."""
        return validate_nvidia_bf16_runtime()

    def fit(
        self,
        request: FineTuneRequest,
        *,
        metric_sink: MetricSink,
        checkpoint_observer: CheckpointObserver,
        resume_from_checkpoint: Path | None = None,
    ) -> BackendFitResult:
        """Fit one adapter without fallback, retry, or parameter mutation."""
        runtime = self.validate_runtime()
        api = load_unsloth_api()
        model, processor = load_base_pair(api, request.model)
        model = apply_lora(api, model, request)
        trainable = build_trainable_parameter_manifest(model)
        validate_trainable_parameter_manifest(trainable, request.lora)

        collector = CollectingCheckpointObserver(checkpoint_observer)
        bridge = TrainerEventBridge(
            metric_sink=metric_sink,
            checkpoint_observer=collector,
            output_dir=request.trainer.output_dir,
            examples_per_step=(
                request.trainer.per_device_train_batch_size
                * request.trainer.gradient_accumulation_steps
            ),
        )
        callback = framework_callback(api.trainer_callback, bridge)
        raw_collator = build_vision_collator(api, model, processor)
        sample_cost_path = (
            request.trainer.output_dir.parent / "logs" / "sample_costs.jsonl"
        )
        collator = SampleCostAuditingCollator(
            collator=raw_collator,
            processor=processor,
            model=model,
            jsonl_path=sample_cost_path,
        )
        mask_audit = audit_response_only_mask(
            collator=collator,
            processor=processor,
            train_dataset=request.train_dataset,
            output_path=(
                request.trainer.output_dir.parent
                / "manifests"
                / "assistant_mask_audit.json"
            ),
        )
        metric_sink.write(
            MetricEvent(
                name="audit/supervised_answer_tokens",
                value=mask_audit.supervised_token_count,
                step=0,
                epoch=0.0,
            )
        )
        metric_sink.write(
            MetricEvent(
                name="audit/ignored_prompt_visual_padding_tokens",
                value=mask_audit.ignored_token_count,
                step=0,
                epoch=0.0,
            )
        )
        trainer_config = build_sft_config(api, request)
        trainer = build_sft_trainer(
            api,
            request=request,
            model=model,
            processor=processor,
            collator=collator,
            trainer_config=trainer_config,
            callback=callback,
        )

        resume_value = (
            str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
        )
        training_completed = False
        try:
            train_output = invoke_method(
                trainer,
                "train",
                required_keywords=(
                    frozenset({"resume_from_checkpoint"})
                    if resume_from_checkpoint is not None
                    else frozenset()
                ),
                resume_from_checkpoint=resume_value,
            )
            training_completed = True
        finally:
            materialize_sample_costs(
                sample_cost_path,
                request.trainer.output_dir.parent / "metrics",
                expected_record_count=(
                    _dataset_length(request.train_dataset, "train_dataset")
                    + _dataset_length(request.eval_dataset, "eval_dataset")
                    if training_completed
                    else None
                ),
            )
        invoke_method(trainer, "save_state")
        final_adapter = request.trainer.output_dir / "final_adapter"
        invoke_method(trainer, "save_model", str(final_adapter))

        state = required_attribute(trainer, "state")
        global_step = numeric_attribute(state, "global_step", integer=True)
        record_unobserved_checkpoints(request.trainer.output_dir, collector)
        metrics = numeric_mapping(getattr(train_output, "metrics", None))
        raw_loss = metrics.get("train_loss")
        return BackendFitResult(
            global_step=int(global_step),
            training_loss=float(raw_loss) if raw_loss is not None else None,
            metrics=metrics,
            checkpoints=collector.events,
            final_adapter_dir=final_adapter,
            trainable_parameters=trainable,
            runtime=runtime,
        )

    def load_base(self, model: ModelLoadSpec) -> LoadedCheckpoint:
        """Load the pinned base model for deterministic pre-update inference."""
        runtime = self.validate_runtime()
        api = load_unsloth_api()
        loaded_model, processor = load_base_pair(api, model)
        invoke_method(api.fast_vision_model, "for_inference", loaded_model)
        return LoadedCheckpoint(
            model=loaded_model,
            processor=processor,
            runtime=runtime,
            checkpoint_path=None,
        )

    def load_checkpoint(
        self,
        *,
        model: ModelLoadSpec,
        checkpoint_path: Path,
    ) -> LoadedCheckpoint:
        """Load a PEFT adapter over the independently pinned base revision."""
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(checkpoint_path)
        runtime = self.validate_runtime()
        api = load_unsloth_api()
        base_model, processor = load_base_pair(api, model)
        loaded_model = invoke_method(
            api.peft_model,
            "from_pretrained",
            base_model,
            str(checkpoint_path),
            required_keywords=frozenset({"is_trainable"}),
            is_trainable=False,
        )
        invoke_method(api.fast_vision_model, "for_inference", loaded_model)
        return LoadedCheckpoint(
            model=loaded_model,
            processor=processor,
            runtime=runtime,
            checkpoint_path=checkpoint_path,
        )

    def predict(
        self,
        loaded: LoadedCheckpoint,
        samples: Sequence[PredictionSample],
        *,
        generation: GenerationSpec | None = None,
    ) -> tuple[BackendPrediction, ...]:
        """Generate ordered greedy responses with thinking disabled."""
        return predict_samples(
            loaded=loaded,
            samples=samples,
            generation=generation or GenerationSpec(),
        )

    def release(self, loaded: LoadedCheckpoint) -> None:
        """Move loaded weights off GPU and release cached CUDA allocations."""
        api = load_unsloth_api()
        invoke_method(loaded.model, "to", "cpu")
        gc.collect()
        cuda = required_attribute(api.torch, "cuda")
        invoke_method(cuda, "empty_cache")


def _dataset_length(value: object, context: str) -> int:
    """Return a declared dataset cardinality or fail before accepting coverage."""

    if not isinstance(value, Sized):
        raise TypeError(f"{context} must expose a stable length")
    return len(value)
