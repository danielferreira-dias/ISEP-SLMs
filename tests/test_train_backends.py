"""CPU-only tests for typed training backend boundaries."""

from __future__ import annotations

import importlib
import json
import math
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from src.train.backends.contracts import (
    BackendFitResult,
    CheckpointEvent,
    FineTuneRequest,
    LoraSpec,
    MetricEvent,
    ModelLoadSpec,
    RuntimeInfo,
    TrainableParameterManifest,
    TrainerSpec,
)
from src.train.backends.masking import (
    audit_response_only_mask,
    audit_response_only_masks,
)
from src.train.backends.parameters import (
    build_trainable_parameter_manifest,
    validate_trainable_parameter_manifest,
)
from src.train.backends.runtime import (
    RuntimeValidationError,
    validate_nvidia_bf16_runtime,
)
from src.train.backends.sample_costs import (
    SampleCostAuditingCollator,
    materialize_sample_costs,
)
from src.train.backends.unsloth_build import (
    CollectingCheckpointObserver,
    apply_lora,
    build_sft_config,
    build_vision_collator,
    load_base_pair,
    record_unobserved_checkpoints,
)
from src.train.backends.unsloth_compat import UnslothApi


@dataclass(slots=True)
class _Parameter:
    requires_grad: bool
    size: int

    def numel(self) -> int:
        return self.size


class _ParameterModel:
    def __init__(self, parameters: list[tuple[str, _Parameter]]) -> None:
        self._parameters = parameters

    def named_parameters(self) -> list[tuple[str, _Parameter]]:
        return self._parameters


class _CaptureFactory:
    def __init__(self, result: object) -> None:
        self.result = result
        self.args: tuple[object, ...] = ()
        self.kwargs: dict[str, object] = {}

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.args = args
        self.kwargs = kwargs
        return self.result


class _Processor:
    def apply_chat_template(
        self,
        messages: object,
        *,
        enable_thinking: bool = False,
        **kwargs: object,
    ) -> str:
        del messages, enable_thinking, kwargs
        return "rendered"


class _FastVision:
    from_pretrained_capture = _CaptureFactory((object(), _Processor()))
    get_peft_capture = _CaptureFactory(object())

    @classmethod
    def from_pretrained(cls, **kwargs: object) -> object:
        return cls.from_pretrained_capture(**kwargs)

    @classmethod
    def get_peft_model(cls, model: object, **kwargs: object) -> object:
        return cls.get_peft_capture(model, **kwargs)


class _MaskProcessor:
    pad_token_id = 0
    image_token_id = 1

    def batch_decode(self, rows: object, **kwargs: object) -> list[str]:
        del rows, kwargs
        return ["class_a"]


class _MaskCollator:
    def __call__(self, records: object) -> dict[str, list[list[int]]]:
        del records
        return {"labels": [[-100, -100, 101]]}


class _CostCollator:
    def __call__(self, records: object) -> dict[str, list[list[int]]]:
        assert isinstance(records, list | tuple)
        count = len(records)
        return {
            "input_ids": [[10, 99, 99, 20, 0] for _ in range(count)],
            "labels": [[-100, -100, -100, 20, -100] for _ in range(count)],
            "attention_mask": [[1, 1, 1, 1, 0] for _ in range(count)],
        }


class _CheckpointObserver:
    def __init__(self) -> None:
        self.events: list[CheckpointEvent] = []

    def on_checkpoint(self, event: CheckpointEvent) -> None:
        self.events.append(event)


class TrainBackendContractTests(unittest.TestCase):
    def test_checkpoint_recovery_is_chronological_and_resume_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            checkpoint_100 = output / "checkpoint-100"
            checkpoint_200 = output / "checkpoint-200"
            checkpoint_300 = output / "checkpoint-300"
            for checkpoint in (checkpoint_300, checkpoint_100, checkpoint_200):
                checkpoint.mkdir()
            (checkpoint_100 / "isep_checkpoint.json").write_text(
                json.dumps({"global_step": 100, "epoch": 1.0}),
                encoding="utf-8",
            )
            (checkpoint_200 / "trainer_state.json").write_text(
                json.dumps({"global_step": 200, "epoch": 2.0}),
                encoding="utf-8",
            )
            downstream = _CheckpointObserver()
            collector = CollectingCheckpointObserver(downstream)
            callback_event = CheckpointEvent(
                checkpoint_300,
                global_step=300,
                epoch=3.0,
            )
            collector.on_checkpoint(callback_event)
            collector.on_checkpoint(callback_event)

            record_unobserved_checkpoints(output, collector)
            record_unobserved_checkpoints(output, collector)

        self.assertEqual(
            [event.global_step for event in collector.events],
            [100, 200, 300],
        )
        self.assertEqual(
            [event.epoch for event in collector.events],
            [1.0, 2.0, 3.0],
        )
        # The old resume checkpoint already has an ISEP manifest; only the
        # callback checkpoint and the previously missed framework checkpoint
        # are forwarded, exactly once each.
        self.assertEqual(len(downstream.events), 2)

    def test_checkpoint_recovery_enriches_callback_epoch_from_trainer_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint-40"
            checkpoint.mkdir()
            (checkpoint / "trainer_state.json").write_text(
                json.dumps({"global_step": 40, "epoch": 0.5}),
                encoding="utf-8",
            )
            downstream = _CheckpointObserver()
            collector = CollectingCheckpointObserver(downstream)
            collector.on_checkpoint(
                CheckpointEvent(checkpoint, global_step=40, epoch=None)
            )

            record_unobserved_checkpoints(Path(temporary), collector)

        self.assertEqual(len(collector.events), 1)
        self.assertEqual(collector.events[0].global_step, 40)
        self.assertEqual(collector.events[0].epoch, 0.5)
        self.assertEqual(len(downstream.events), 1)

    def test_checkpoint_recovery_never_rewrites_an_existing_bad_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint-40"
            checkpoint.mkdir()
            manifest = checkpoint / "isep_checkpoint.json"
            original = b'{"global_step": 999, "epoch": 99.0}'
            manifest.write_bytes(original)
            (checkpoint / "trainer_state.json").write_text(
                json.dumps({"global_step": 40, "epoch": 0.5}),
                encoding="utf-8",
            )
            downstream = _CheckpointObserver()
            collector = CollectingCheckpointObserver(downstream)

            record_unobserved_checkpoints(Path(temporary), collector)

            self.assertEqual(manifest.read_bytes(), original)
            self.assertEqual(collector.events[0].epoch, 0.5)
            self.assertEqual(downstream.events, [])

    def test_cost_collator_records_exact_geometry_and_token_decomposition(self) -> None:
        record: dict[object, object] = {
            "sample_id": "sample-1",
            "split": "sft_train",
            "leakage_group_id": "group-1",
            "image_width": 100,
            "image_height": 50,
            "pixel_count": 5000,
            "resized_width": 100,
            "resized_height": 50,
            "annotation_availability": ["diagnosis", "caption"],
            "phase": "e2_skincon",
            "task": "caption",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collator = SampleCostAuditingCollator(
                collator=_CostCollator(),
                processor=object(),
                model=SimpleNamespace(config=SimpleNamespace(image_token_id=99)),
                jsonl_path=root / "logs" / "sample_costs.jsonl",
            )

            collator([record])
            collator([record])
            count = materialize_sample_costs(
                root / "logs" / "sample_costs.jsonl",
                root / "metrics",
                expected_record_count=1,
            )

            self.assertEqual(count, 1)
            self.assertTrue((root / "metrics" / "sample_costs.csv").is_file())
            self.assertTrue((root / "metrics" / "sample_costs.parquet").is_file())
            manifest = json.loads(
                (root / "metrics" / "sample_costs_manifest.json").read_text()
            )
            self.assertTrue(manifest["coverage_complete"])
            self.assertEqual(manifest["record_count"], 1)

    def test_importing_backend_does_not_import_gpu_frameworks(self) -> None:
        initially_absent = {
            name
            for name in ("unsloth", "trl", "transformers")
            if name not in sys.modules
        }

        importlib.import_module("src.train.backends")

        for name in initially_absent:
            self.assertNotIn(name, sys.modules)

    def test_qwen_revision_is_immutable_and_exact(self) -> None:
        with self.assertRaisesRegex(ValueError, "thesis-pinned revision"):
            ModelLoadSpec(revision="deadbeef")
        with self.assertRaisesRegex(ValueError, "immutable commit"):
            ModelLoadSpec(model_id="other/model", revision="main")

    def test_trainable_manifest_enforces_vision_ablation(self) -> None:
        language_model = _ParameterModel(
            [
                ("model.layers.0.self_attn.q_proj.lora_A", _Parameter(True, 16)),
                ("model.layers.0.mlp.up_proj.lora_A", _Parameter(True, 32)),
                ("model.layers.0.norm.weight", _Parameter(False, 64)),
            ]
        )
        manifest = build_trainable_parameter_manifest(language_model)

        validate_trainable_parameter_manifest(
            manifest,
            LoraSpec(finetune_vision_layers=False),
        )
        self.assertEqual(manifest.total_trainable, 48)
        self.assertEqual(manifest.by_component["vision"], 0)
        with self.assertRaisesRegex(RuntimeError, "no vision parameter"):
            validate_trainable_parameter_manifest(
                manifest,
                LoraSpec(finetune_vision_layers=True),
            )

    def test_runtime_rejects_cpu_without_fallback(self) -> None:
        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)

        with (
            patch(
                "src.train.backends.runtime.importlib.import_module",
                return_value=fake_torch,
            ),
            self.assertRaisesRegex(
                RuntimeValidationError,
                "CPU fallback is intentionally disabled",
            ),
        ):
            validate_nvidia_bf16_runtime()

    def test_runtime_records_nvidia_bf16_device(self) -> None:
        fake_torch = ModuleType("torch")
        fake_torch.__version__ = "test"
        fake_torch.version = SimpleNamespace(cuda="12.8")
        fake_torch.cuda = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda index: "NVIDIA H200",
            is_bf16_supported=lambda: True,
            get_device_properties=lambda index: SimpleNamespace(
                total_memory=143_000_000_000
            ),
        )

        with patch(
            "src.train.backends.runtime.importlib.import_module",
            return_value=fake_torch,
        ):
            runtime = validate_nvidia_bf16_runtime()

        self.assertEqual(runtime.device_name, "NVIDIA H200")
        self.assertTrue(runtime.bf16_supported)

    def test_unsloth_builders_preserve_scientific_recipe(self) -> None:
        torch = ModuleType("torch")
        torch.bfloat16 = object()
        collator = _CaptureFactory(object())
        sft_config = _CaptureFactory(object())
        api = UnslothApi(
            fast_vision_model=_FastVision,
            vision_collator=collator,
            sft_trainer=_CaptureFactory(object()),
            sft_config=sft_config,
            trainer_callback=object,
            peft_model=object(),
            torch=torch,
        )
        with tempfile.TemporaryDirectory() as temporary:
            request = FineTuneRequest(
                model=ModelLoadSpec(),
                lora=LoraSpec(finetune_vision_layers=True),
                trainer=TrainerSpec(output_dir=Path(temporary)),
                train_dataset=object(),
                eval_dataset=object(),
            )
            base_model, processor = load_base_pair(api, request.model)
            apply_lora(api, base_model, request)
            build_vision_collator(api, base_model, processor)
            build_sft_config(api, request)

        load_kwargs = _FastVision.from_pretrained_capture.kwargs
        self.assertFalse(load_kwargs["load_in_4bit"])
        self.assertEqual(load_kwargs["revision"], request.model.revision)
        lora_kwargs = _FastVision.get_peft_capture.kwargs
        self.assertTrue(lora_kwargs["finetune_vision_layers"])
        self.assertEqual(lora_kwargs["r"], 16)
        # Unsloth treats the literal ``all-linear`` as an instruction to
        # enable vision LoRA unconditionally. ``None`` retains all eligible
        # linears while allowing the controlled vision flag to scope them.
        self.assertIsNone(lora_kwargs["target_modules"])
        self.assertEqual(lora_kwargs["use_gradient_checkpointing"], "unsloth")
        self.assertTrue(collator.kwargs["train_on_responses_only"])
        self.assertEqual(collator.kwargs["response_part"], "</think>\n\n")
        self.assertEqual(sft_config.kwargs["eval_strategy"], "steps")
        self.assertEqual(sft_config.kwargs["save_strategy"], "epoch")
        self.assertEqual(sft_config.kwargs["data_seed"], request.trainer.seed)
        self.assertFalse(sft_config.kwargs["save_only_model"])
        self.assertIsNone(sft_config.kwargs["max_length"])
        self.assertEqual(
            sft_config.kwargs["include_num_input_tokens_seen"],
            "all",
        )
        self.assertNotIn("include_tokens_per_second", sft_config.kwargs)

    def test_runtime_mask_audit_proves_only_label_is_supervised(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "mask.json"
            audit = audit_response_only_mask(
                collator=_MaskCollator(),
                processor=_MaskProcessor(),
                train_dataset=[{"sample_id": "sample-1", "label": "class_a"}],
                output_path=output,
            )

        self.assertEqual(audit.decoded_supervision, "class_a")
        self.assertEqual(audit.supervised_token_count, 1)
        self.assertEqual(audit.ignored_token_count, 2)
        self.assertEqual(audit.forbidden_visual_token_count, 0)

    def test_runtime_mask_audit_covers_one_row_per_task(self) -> None:
        class _TaskDataset:
            def mask_audit_records(self) -> list[dict[str, str]]:
                return [
                    {
                        "sample_id": "diagnosis-1",
                        "task": "diagnosis",
                        "label": "class_a",
                    },
                    {
                        "sample_id": "caption-1",
                        "task": "caption",
                        "label": "class_a",
                    },
                ]

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "mask.json"
            audits = audit_response_only_masks(
                collator=_MaskCollator(),
                processor=_MaskProcessor(),
                train_dataset=_TaskDataset(),
                output_path=output,
            )
            manifest = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual([audit.task for audit in audits], ["diagnosis", "caption"])
        self.assertEqual(manifest["audit_count"], 2)
        self.assertEqual(manifest["tasks"], ["diagnosis", "caption"])

    def test_nonfinite_metrics_are_rejected_before_serialization(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            MetricEvent("loss", math.nan, step=1)
        with self.assertRaisesRegex(ValueError, "finite"):
            BackendFitResult(
                global_step=1,
                training_loss=math.inf,
                metrics={},
                checkpoints=(),
                final_adapter_dir=Path("adapter"),
                trainable_parameters=TrainableParameterManifest((), 0, {}),
                runtime=RuntimeInfo("test", "test", "test", 1, True),
            )


if __name__ == "__main__":
    unittest.main()
