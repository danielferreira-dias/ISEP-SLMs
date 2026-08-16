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
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import snapshot_download, split_torch_state_dict_into_shards
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText, AutoProcessor


QWEN35_PREFIX_REWRITES = (
    (
        "model.language_model.language_model.language_model.",
        "model.language_model.",
    ),
    ("model.language_model.visual.", "model.visual."),
)


def canonical_qwen35_key(key: str) -> str:
    """Remove PEFT's erroneous wrapper namespaces from a Qwen 3.5 key."""

    for source, destination in QWEN35_PREFIX_REWRITES:
        if key.startswith(source):
            return destination + key.removeprefix(source)
    return key


def keyset_sha256(keys: set[str]) -> str:
    """Hash a tensor-key set in stable order."""

    payload = "".join(f"{key}\n" for key in sorted(keys)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def base_weight_map(snapshot: Path) -> dict[str, str]:
    """Return the official checkpoint key-to-shard mapping."""

    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        mapping = payload.get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise RuntimeError(f"Invalid weight index: {index}")
        return {str(key): str(value) for key, value in mapping.items()}
    weights = snapshot / "model.safetensors"
    if weights.is_file():
        with safe_open(str(weights), framework="pt", device="cpu") as handle:
            return {key: weights.name for key in handle.keys()}
    raise FileNotFoundError(f"No safetensors checkpoint found below {snapshot}")


def adapter_target_keys(adapter: Path) -> set[str]:
    """Map LoRA tensor names back to the canonical base tensors they adapt."""

    weights = adapter / "adapter_model.safetensors"
    targets: set[str] = set()
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if ".lora_A.weight" not in key and ".lora_B.weight" not in key:
                continue
            if not key.startswith("base_model.model."):
                raise RuntimeError(f"Unexpected adapter tensor namespace: {key}")
            target = key.removeprefix("base_model.model.")
            target = target.replace(".lora_A.weight", ".weight")
            target = target.replace(".lora_B.weight", ".weight")
            targets.add(target)
    if not targets:
        raise RuntimeError(f"No LoRA target tensors found in {weights}")
    return targets


def load_base_tensors(
    snapshot: Path,
    weight_map: dict[str, str],
    keys: set[str],
) -> dict[str, torch.Tensor]:
    """Load selected official tensors, grouped by shard."""

    grouped: dict[str, list[str]] = defaultdict(list)
    for key in sorted(keys):
        grouped[weight_map[key]].append(key)
    tensors: dict[str, torch.Tensor] = {}
    for shard, shard_keys in sorted(grouped.items()):
        with safe_open(
            str(snapshot / shard), framework="pt", device="cpu"
        ) as handle:
            for key in shard_keys:
                tensors[key] = handle.get_tensor(key)
    return tensors


def saved_weight_keys(output: Path) -> set[str]:
    """Read the tensor keys physically stored in a standalone checkpoint."""

    keys: set[str] = set()
    for path in sorted(output.glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            overlap = keys.intersection(handle.keys())
            if overlap:
                raise RuntimeError(f"Duplicate saved tensor keys: {sorted(overlap)}")
            keys.update(handle.keys())
    return keys


def save_canonical_checkpoint(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    output: Path,
) -> None:
    """Write canonical safetensors directly, bypassing model save hooks."""

    output.mkdir(parents=True, exist_ok=False)
    state_split = split_torch_state_dict_into_shards(
        state_dict,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size="5GB",
    )
    for filename, tensor_names in sorted(
        state_split.filename_to_tensors.items()
    ):
        shard = {
            key: state_dict[key].detach().cpu().contiguous()
            for key in tensor_names
        }
        save_file(shard, output / filename, metadata={"format": "pt"})
    if state_split.is_sharded:
        index = {
            "metadata": state_split.metadata,
            "weight_map": state_split.tensor_to_filename,
        }
        (output / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    model.config.save_pretrained(output)
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.save_pretrained(output)


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

    snapshot = Path(
        snapshot_download(
            args.base_model,
            revision=args.revision,
            local_files_only=True,
        )
    )
    official_map = base_weight_map(snapshot)
    official_keys = set(official_map)
    canonical_state: dict[str, torch.Tensor] = {}
    ignored_tied_aliases: list[str] = []
    source_keys = set(merged.state_dict())
    for source_key, tensor in merged.state_dict().items():
        key = canonical_qwen35_key(source_key)
        if key not in official_keys:
            if merged.config.tie_word_embeddings and key.endswith("lm_head.weight"):
                ignored_tied_aliases.append(source_key)
                continue
            raise RuntimeError(
                f"Merged tensor does not map to the pinned base: {source_key} -> {key}"
            )
        if key in canonical_state:
            raise RuntimeError(f"Multiple merged tensors map to {key}")
        canonical_state[key] = tensor

    missing = official_keys.difference(canonical_state)
    if len(missing) != 15 or any(not key.startswith("mtp.") for key in missing):
        raise RuntimeError(
            "Unexpected base-only tensors after canonicalization: "
            f"count={len(missing)}, keys={sorted(missing)}"
        )
    canonical_state.update(load_base_tensors(snapshot, official_map, missing))
    if set(canonical_state) != official_keys:
        raise RuntimeError("Canonical state does not exactly match the pinned base keys")

    targets = adapter_target_keys(adapter)
    if not targets.issubset(canonical_state):
        raise RuntimeError(
            f"Adapter targets are absent from canonical state: {sorted(targets - set(canonical_state))}"
        )
    base_targets = load_base_tensors(snapshot, official_map, targets)
    changed_targets = sorted(
        key
        for key in targets
        if not torch.equal(canonical_state[key].detach().cpu(), base_targets[key])
    )
    if not changed_targets:
        raise RuntimeError("The merged LoRA did not change any targeted base tensor")

    save_canonical_checkpoint(merged, canonical_state, output)
    processor = AutoProcessor.from_pretrained(
        args.base_model,
        revision=args.revision,
    )
    processor.save_pretrained(output)
    stored_keys = saved_weight_keys(output)
    if stored_keys != official_keys:
        raise RuntimeError(
            "Saved checkpoint key mismatch: "
            f"missing={sorted(official_keys - stored_keys)}, "
            f"extra={sorted(stored_keys - official_keys)}"
        )
    manifest = {
        "status": "validated",
        "base_model": args.base_model,
        "base_revision": args.revision,
        "source_tensor_count": len(source_keys),
        "official_tensor_count": len(official_keys),
        "stored_tensor_count": len(stored_keys),
        "base_only_mtp_tensors": sorted(missing),
        "ignored_tied_aliases": sorted(ignored_tied_aliases),
        "adapter_target_count": len(targets),
        "changed_adapter_target_count": len(changed_targets),
        "official_keyset_sha256": keyset_sha256(official_keys),
        "stored_keyset_sha256": keyset_sha256(stored_keys),
        "prefix_rewrites": [
            {"source": source, "destination": destination}
            for source, destination in QWEN35_PREFIX_REWRITES
        ],
    }
    (output / "canonical_weight_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
