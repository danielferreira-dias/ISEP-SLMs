# Inference backends

This package provides one normalized interface for local vLLM servers,
OpenAI-compatible chat endpoints, and the Azure/OpenAI Responses API. Model
weights and provider SDKs are loaded lazily, so configuration and unit tests
also work on macOS machines that do not run vLLM.

## Public interface

All backends implement `InferenceBackend`:

```python
result = backend.generate_result(
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    image_bytes=image_bytes,
    schema=output_schema,
    generation={
        "max_output_tokens": 512,
        "seed": 42,
    },
    request_id=sample_id,
)

print(result.final_text)
print(result.reasoning)
print(result.usage)
```

The benchmark executor uses the asynchronous contract:

```python
result = await backend.acomplete(request)
```

OpenAI-compatible chat and vLLM backends implement this natively with
`AsyncOpenAI`. Synchronous-only transports use `asyncio.to_thread` as a
compatibility fallback. `generate_batch_async(...)` is also available for
direct callers.

`generate(...)` remains available for older benchmark code and returns only
`final_text`, while `generate_batch(...)` retains its stable sequential
compatibility behavior.

Local image bytes are sent as base64 `data:` URLs. JPEG, PNG, GIF, WebP, and
BMP signatures are recognized automatically.

## Transports

- `VllmBackend` uses vLLM's OpenAI-compatible chat-completions server and adds
  `/health` plus `/v1/models` preflight checks. Responses are streamed so long
  reasoning generations do not leave cloud proxy connections idle, and SDK
  retries are disabled to avoid duplicating expensive generations. Benchmark
  requests use `AsyncOpenAI`; the async HTTP client is closed before a
  managed server is stopped.
- `OpenAICompatibleChatBackend` also supports provider-hosted, OpenAI-compatible
  chat endpoints.
- `AzureResponsesBackend` uses the Responses API. It supports the Azure
  `/openai/v1` endpoint through the standard `OpenAI` client and legacy Azure
  API-version deployments through `AzureOpenAI`.
- `LocalBackend` and `AzureBackend` adapt normalized model configuration to
  these transports.

The checked-in benchmark mode is `prompt_only`. JSON Schema constraints stay
disabled by default so cross-model comparisons remain prompt-only. For a
separate production-reliability run on a compatible model, select
`--structured-output json_schema`.

In `json_schema` mode, the OpenAI-compatible backend sends the benchmark's
task-specific schema through `response_format`:

```python
{
    "type": "json_schema",
    "json_schema": {
        "name": "benchmark_response",
        "strict": True,
        "schema": task_schema,
    },
}
```

This is equivalent to supplying `BaseModel.model_json_schema()` in an
application with a fixed Pydantic model. The benchmark schemas are loaded
dynamically from the benchmark configuration, so validation remains in the
benchmark validators instead of requiring one hard-coded Pydantic class per
task. Keep `prompt_only` and `json_schema` as distinct experiment conditions:
the former measures instruction following, while the latter measures
diagnostic performance with constrained serialization.

## Reasoning capture

Reasoning is never mixed into `final_text`. Four retention modes are
supported:

- `full`: retain the complete reasoning field exposed by a chat provider.
- `summary`: retain only a provider-supplied summary.
- `tokens_only`: retain only the provider's reasoning-token count.
- `none`: retain no reasoning data.

The facade and factory also accept the run-level policy `available`. It maps
to `full` for vLLM and chat-completions backends, and to `summary` for the
Responses API, whose public contract does not expose raw chain of thought.

The chat backend recognizes both the current `reasoning` field and the legacy
`reasoning_content` field. Summary mode never relabels raw reasoning as a
summary. The Responses API requests `reasoning={"summary": "auto"}` for
`full` and `summary` capture; it stores only that official provider summary
and never attempts to extract raw chain of thought.

## Reasoning budgets

`generation.reasoning_max_tokens` is the normalized numeric budget used by
thinking-capable OpenAI-compatible models. It is independent from
`max_output_tokens`, which remains the total completion cap and therefore
includes both exposed reasoning and the final answer.

| Transport | Provider request field |
| --- | --- |
| Local vLLM 0.23 | `thinking_token_budget` |
| OpenRouter | `reasoning.max_tokens` |

The budget must be a positive integer and cannot be combined with
`reasoning_effort`. It is dormant when `thinking_mode` is disabled. For
controllable teacher-screening profiles, the total cap is 14,336 and the
reasoning budget is 10,240, reserving up to 4,096 tokens for the scored answer.

A reasoning budget is request-time generation control, not a prompt injected
into an active completion. Chat-completion streaming is one-way: the client can
observe deltas or cancel the request, but cannot add a new user message to that
same generation. A forced intervention would require cancelling and issuing a
second request while preserving the provider's complete, unmodified
`reasoning_details`. That is a different two-turn protocol and is intentionally
not used for the paired teacher-screening benchmark.

See the official [vLLM sampling-parameter documentation](https://docs.vllm.ai/en/v0.23.0/api/vllm/sampling_params/)
and [OpenRouter reasoning-token documentation](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).

## Managed vLLM process

`ManagedVllmServer` starts one server without using a shell, waits for its
health endpoint, and terminates it on context-manager exit:

```python
from pathlib import Path

from src.inference.vllm import (
    ManagedVllmServer,
    server_config_from_model,
)

server_config = server_config_from_model(model_config, port=8000)

with ManagedVllmServer(
    server_config,
    log_path=Path("runs/vllm.log"),
):
    # Execute the benchmark against http://127.0.0.1:8000/v1.
    ...
```

`server_config_from_model(...)` transfers the active local profile's dtype,
tensor parallelism, context length, GPU-memory target, image limit, and
reasoning parser into the vLLM command. `ManagedVllmServer` can
discard subprocess output (the default) or write both stdout and stderr to
`log_path`; it closes the file during every shutdown path.

Credentials must be passed through the process environment. Secret-bearing
CLI flags such as `--api-key` and `--hf-token` are rejected because command
lines can appear in system diagnostics. Provider exceptions are converted to
sanitized transport errors and API-key values are never included in results
or error messages.
