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

`generate(...)` remains available for older benchmark code and returns only
`final_text`. `generate_batch(...)` currently runs a stable sequential loop;
the method is intentionally part of the interface so a concurrent
implementation can be added without changing the benchmark executor.

Local image bytes are sent as base64 `data:` URLs. JPEG, PNG, GIF, WebP, and
BMP signatures are recognized automatically.

## Transports

- `VllmBackend` uses vLLM's OpenAI-compatible chat-completions server and adds
  `/health` plus `/v1/models` preflight checks.
- `OpenAICompatibleChatBackend` also supports provider-hosted chat endpoints,
  including Kimi-compatible deployments.
- `AzureResponsesBackend` uses the Responses API. It supports the Azure
  `/openai/v1` endpoint through the standard `OpenAI` client and legacy Azure
  API-version deployments through `AzureOpenAI`.
- `LocalBackend` and `AzureBackend` adapt normalized model configuration to
  these transports.

The checked-in benchmark mode is `prompt_only`. JSON Schema response
constraints are therefore disabled by default and must be explicitly enabled
by a future constrained-decoding experiment.

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

MedGemma requests do not use a separate system role. The system instructions
are prepended to the first user text block while preserving image-first
ordering, as required by that model's chat-template contract.

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
reasoning parser into the vLLM command. MiniCPM image processor settings are
also forwarded through `--mm-processor-kwargs`. `ManagedVllmServer` can
discard subprocess output (the default) or write both stdout and stderr to
`log_path`; it closes the file during every shutdown path.

Credentials must be passed through the process environment. Secret-bearing
CLI flags such as `--api-key` and `--hf-token` are rejected because command
lines can appear in system diagnostics. Provider exceptions are converted to
sanitized transport errors and API-key values are never included in results
or error messages.
