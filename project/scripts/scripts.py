"""OpenRouter-compatible tools.

The model does not execute these functions. It returns a tool_call; this
module runs the function locally and returns a string the caller can send
back as a tool result.

The smoke probe disables native thinking and offers ``deep_thinking`` as the
only scratchpad, so any internal CoT has to land in the tool arguments.

Run from the repo root:

    set -a && source .env && set +a && uv run python project/scripts/scripts.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEEP_THINKING_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "deep_thinking",
        "description": (
            "Private thinking scratchpad. Call this before answering. "
            "Put the full internal chain of thought in rational; do not "
            "summarize and do not answer the user here."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "rational": {
                    "type": "string",
                    "description": "Internal chain of thought.",
                }
            },
            "required": ["rational"],
            "additionalProperties": False,
        },
    },
}

TOOLS: list[dict[str, Any]] = [DEEP_THINKING_TOOL]


def deep_thinking(rational: str) -> str:
    """Acknowledge a recorded reasoning step.

    Args:
        rational: The model's reasoning text.

    Returns:
        A short acknowledgement string for the next OpenRouter turn.
    """
    if not isinstance(rational, str) or not rational.strip():
        raise ValueError("rational must be a non-empty string")
    return "acknowledged"


TOOL_MAPPING = {
    "deep_thinking": deep_thinking,
}


def extract_rational(arguments: str | dict[str, Any]) -> str | None:
    """Pull ``rational`` even if the model truncated or broke the JSON."""
    if isinstance(arguments, dict):
        value = arguments.get("rational")
        return value if isinstance(value, str) else None
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get("rational"), str):
        return parsed["rational"]
    marker = '"rational":"'
    start = arguments.find(marker)
    if start == -1:
        return None
    return arguments[start + len(marker) :]


def execute_tool_call(name: str, arguments: str | dict[str, Any]) -> str:
    """Run a tool the model requested.

    `arguments` may be the JSON string from OpenRouter (`function.arguments`)
    or an already-parsed dict. Truncated hidden-CoT dumps are acknowledged
    without requiring valid JSON.
    """
    if name not in TOOL_MAPPING:
        raise KeyError(f"Unknown tool {name!r}. Known tools: {', '.join(sorted(TOOL_MAPPING))}")

    parsed: dict[str, Any]
    if isinstance(arguments, dict):
        parsed = arguments
    else:
        try:
            loaded = json.loads(arguments)
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            parsed = loaded
        else:
            rational = extract_rational(arguments)
            if rational is None:
                raise TypeError("tool arguments must be a JSON object")
            parsed = {"rational": rational}
    return TOOL_MAPPING[name](**parsed)


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-5.6-luna"
DEFAULT_PROMPT = (
    "Find all primitive Pythagorean triples (a, b, c) with c < 80. "
    "Use the coprime m>n generation with opposite parity. "
    "For each triple prove it is primitive and that the generation is unique. "
    "Then count how many such triples exist."
)
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
LEAK_DIR = SCRIPTS_DIR / "leak"
VERTEX_PROVIDER = {
    "only": ["google-vertex"],
    "allow_fallbacks": False,
    "require_parameters": True,
}


def _load_repo_env() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Disable native thinking and capture CoT via the deep_thinking tool."
        )
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--image",
        help="Optional local image path. HEIC is converted to JPEG first.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        default="none",
        help='OpenRouter reasoning.effort. Use "none" to disable native thinking.',
    )
    parser.add_argument(
        "--tool-choice",
        choices=("auto", "required"),
        default="required",
        help="required forces the first turn to call deep_thinking.",
    )
    parser.add_argument("--max-tokens", type=int, default=65536)
    parser.add_argument(
        "--skip-final",
        action="store_true",
        help="Stop after the tool call; do not request the final answer.",
    )
    return parser.parse_args()


def _tool_choice(mode: str) -> str | dict[str, Any]:
    if mode == "auto":
        return "auto"
    return {"type": "function", "function": {"name": "deep_thinking"}}


def _extra_body(model: str, reasoning_effort: str) -> dict[str, Any]:
    effort = reasoning_effort
    if model.startswith("google/") and effort == "none":
        # Gemini 3.7 Flash rejects effort=none: reasoning is mandatory.
        effort = "low"
        print(
            "warning: native thinking cannot be disabled on "
            f"{model}; using reasoning.effort=low"
        )
    body: dict[str, Any] = {
        "reasoning": {
            "effort": effort,
            "exclude": True,
        }
    }
    if model.startswith("google/"):
        body["provider"] = VERTEX_PROVIDER
    return body


def _native_reasoning(message: Any) -> Any:
    for attr in ("reasoning", "reasoning_content", "reasoning_details"):
        value = getattr(message, attr, None)
        if value:
            return {attr: value}
    return None


def _dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _as_jpeg(path: Path) -> Path:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return path
    jpeg = Path("/tmp") / f"{path.stem}_api.jpg"
    subprocess.run(
        ["sips", "-s", "format", "jpeg", "-Z", "1600", str(path), "--out", str(jpeg)],
        check=True,
        capture_output=True,
    )
    return jpeg


def _user_content(prompt: str, image: str | None) -> str | list[dict[str, Any]]:
    if not image:
        return prompt
    import base64

    jpeg = _as_jpeg(Path(image).expanduser().resolve())
    b64 = base64.b64encode(jpeg.read_bytes()).decode("ascii")
    return [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        },
    ]


def _write_leak(payload: dict[str, Any]) -> Path:
    LEAK_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = str(payload.get("model") or "unknown").replace("/", "_")
    path = LEAK_DIR / f"{stamp}__{slug}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def main() -> None:
    args = _parse_args()
    _load_repo_env()
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing environment variable OPENROUTER_API_KEY")

    from openai import OpenAI

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    extra = _extra_body(args.model, args.reasoning_effort)
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": _user_content(args.prompt, args.image)}
    ]

    print("model:", args.model)
    print("requested reasoning effort:", args.reasoning_effort)
    print("sent reasoning:", extra["reasoning"])
    print("tool_choice:", args.tool_choice)
    print("assisted off reasoning: deep_thinking")

    first = client.chat.completions.create(
        model=args.model,
        messages=messages,
        tools=TOOLS,
        tool_choice=_tool_choice(args.tool_choice),
        extra_body=extra,
        max_tokens=args.max_tokens,
    )
    assistant = first.choices[0].message
    native = _native_reasoning(assistant)
    tool_records: list[dict[str, Any]] = []
    tool_messages: list[dict[str, Any]] = []
    for tool_call in assistant.tool_calls or []:
        arguments = tool_call.function.arguments
        rational = extract_rational(arguments)
        result = execute_tool_call(tool_call.function.name, arguments)
        tool_records.append(
            {
                "id": tool_call.id,
                "name": tool_call.function.name,
                "arguments_raw": arguments,
                "arguments_valid_json": True,
                "rational": rational,
                "local_result": result,
            }
        )
        try:
            json.loads(arguments)
        except (TypeError, json.JSONDecodeError):
            tool_records[-1]["arguments_valid_json"] = False
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
        )

    payload: dict[str, Any] = {
        "model": args.model,
        "prompt": args.prompt,
        "image": args.image,
        "tool_choice": args.tool_choice,
        "max_tokens": args.max_tokens,
        "requested_reasoning_effort": args.reasoning_effort,
        "sent_reasoning": extra["reasoning"],
        "provider": extra.get("provider"),
        "first_turn": {
            "id": getattr(first, "id", None),
            "provider": getattr(first, "provider", None),
            "finish_reason": first.choices[0].finish_reason,
            "native_reasoning": _dump(native),
            "content": assistant.content,
            "tool_calls": tool_records,
            "usage": _dump(getattr(first, "usage", None)),
        },
        "final": None,
    }

    if assistant.tool_calls and not args.skip_final:
        messages.append(
            {
                "role": "assistant",
                "content": assistant.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in assistant.tool_calls
                ],
            }
        )
        messages.extend(tool_messages)
        second = client.chat.completions.create(
            model=args.model,
            messages=messages,
            tools=TOOLS,
            extra_body=extra,
            max_tokens=args.max_tokens,
        )
        payload["final"] = {
            "id": getattr(second, "id", None),
            "provider": getattr(second, "provider", None),
            "finish_reason": second.choices[0].finish_reason,
            "native_reasoning": _dump(_native_reasoning(second.choices[0].message)),
            "content": second.choices[0].message.content,
            "usage": _dump(getattr(second, "usage", None)),
        }

    leak_path = _write_leak(payload)
    first_tools = payload["first_turn"]["tool_calls"]
    print("finish_reason:", payload["first_turn"]["finish_reason"])
    print("native_reasoning:", payload["first_turn"]["native_reasoning"] is not None)
    print("tool_calls:", len(first_tools))
    if first_tools:
        print("rational_chars:", len(first_tools[0].get("rational") or ""))
        print("arguments_valid_json:", first_tools[0]["arguments_valid_json"])
    if payload["final"] is not None:
        print("final_chars:", len(payload["final"].get("content") or ""))
    print("leak_file:", leak_path)


if __name__ == "__main__":
    main()
