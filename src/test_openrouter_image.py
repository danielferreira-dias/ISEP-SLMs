"""Send one frozen ISEPDermaBench image through an OpenRouter model profile.

This diagnostic uses the same typed model configuration, inference backend,
prompts, and image bytes as the benchmark runner. It is useful for reproducing
provider-specific image refusals without changing benchmark selection logic.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from src.config import load_model_config
from src.inference.base import InferenceRequest
from src.inference.factory import create_backend


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_ID = (
    "open_ended_diagnosis:validation:"
    "FITZPATRICK17K_C_d8194545dff1a3a3950e607c29c64894"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce one benchmark image request through OpenRouter."
    )
    parser.add_argument("--model", default="gpt_5_6_luna")
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID)
    parser.add_argument("--reasoning-capture", default="available")
    parser.add_argument(
        "--reasoning-effort",
        default="none",
        choices=("none", "low", "medium", "high"),
    )
    return parser.parse_args()


def _task(task_id: str) -> dict[str, Any]:
    split = task_id.split(":", maxsplit=2)[1]
    directory = (
        ROOT
        / "data/benchmarks/ISEPDermaBench/tasks/open_ended_diagnosis"
    )
    for path in sorted(directory.glob(f"{split}-*.parquet")):
        table = pq.read_table(
            path,
            filters=[("task_id", "=", task_id)],
        )
        if table.num_rows:
            return table.to_pylist()[0]
    raise ValueError(f"Task not found in frozen release: {task_id}")


def _image_bytes(value: Any) -> bytes:
    if not isinstance(value, dict):
        raise ValueError("Frozen task image must be an Arrow image mapping")
    raw = value.get("bytes")
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    if not isinstance(raw, bytes) or not raw:
        raise ValueError("Frozen task image bytes are unavailable")
    return raw


async def _run() -> None:
    args = _arguments()
    task = _task(args.task_id)
    model = load_model_config(
        args.model,
        root=ROOT,
        backend_profile="openrouter",
    )
    backend = create_backend(
        model,
        reasoning_capture=args.reasoning_capture,
        use_json_schema=False,
    )
    request = InferenceRequest(
        request_id=args.task_id,
        system_prompt=str(task["system_prompt"]),
        user_prompt=str(task["user_prompt"]),
        image_bytes=_image_bytes(task["image"]),
        image_mime_type="image/jpeg",
        schema={},
        generation={
            "max_output_tokens": 2048,
            "reasoning_effort": args.reasoning_effort,
        },
    )
    try:
        result = await backend.acomplete(request)
    finally:
        await backend.aclose()

    print(f"model: {model.backend.active_profile.request_model}")
    print(f"task_id: {args.task_id}")
    print(f"finish_reason: {result.finish_reason}")
    print(f"reasoning_capture_mode: {result.reasoning.capture_mode}")
    print(f"reasoning_source: {result.reasoning.source_field}")
    print(f"reasoning_tokens: {result.reasoning.token_count}")
    print("answer:")
    print(result.final_text)


if __name__ == "__main__":
    asyncio.run(_run())
