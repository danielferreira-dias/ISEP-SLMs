#!/usr/bin/env python3
"""Merge one E1 PEFT adapter into the pinned Qwen base in BF16.

The resulting standalone checkpoint is intended for inference engines whose
runtime LoRA implementation cannot wrap every multimodal module (notably the
Qwen visual patch embedding).  The script refuses to overwrite an existing
directory so a partial or stale merge cannot silently become an evaluation
checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_args() -> argparse.Namespace:
    """Parse the base, adapter and output paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Load on CPU, merge in BF16 and save a standalone checkpoint."""

    args = parse_args()
    adapter = args.adapter.resolve()
    output = args.output.resolve()
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Adapter weights do not exist: {adapter}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite merge output: {output}")

    base = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        revision=args.revision,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    merged = peft_model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(
        output,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    processor = AutoProcessor.from_pretrained(
        args.base_model,
        revision=args.revision,
    )
    processor.save_pretrained(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
