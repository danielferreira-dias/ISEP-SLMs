#!/usr/bin/env bash
set -euo pipefail

INSTALL=0
PROJECT_DIR="$(pwd)"
MIN_FREE_GB=80
PYTHON_VERSION="${RUNPOD_PYTHON_VERSION:-3.12}"

usage() {
  echo "Usage: $0 [--install] [--project-dir PATH] [--min-free-gb N]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install) INSTALL=1; shift ;;
    --project-dir) PROJECT_DIR="$2"; shift 2 ;;
    --min-free-gb) MIN_FREE_GB="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! "$MIN_FREE_GB" =~ ^[0-9]+$ ]]; then
  echo "--min-free-gb must be a non-negative integer" >&2
  exit 2
fi

echo "== GPU =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is unavailable; use a GPU RunPod image." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free \
  --format=csv,noheader

echo "== Storage =="
if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "ERROR: project directory does not exist: $PROJECT_DIR" >&2
  exit 1
fi
df -h "$PROJECT_DIR"
FREE_KB=$(df -Pk "$PROJECT_DIR" | awk 'NR==2 {print $4}')
FREE_GB=$((FREE_KB / 1024 / 1024))
if (( FREE_GB < MIN_FREE_GB )); then
  echo "WARNING: ${FREE_GB} GiB free; requested minimum is ${MIN_FREE_GB} GiB." >&2
fi

echo "== Existing servers and ports =="
ps -ef | grep '[v]llm serve' || true
if command -v ss >/dev/null 2>&1; then
  ss -ltnp || true
fi

echo "== Credentials =="
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN: present (value intentionally hidden)"
else
  echo "HF_TOKEN: missing; public downloads work but are rate limited"
fi

if (( INSTALL == 0 )); then
  echo "Preflight complete. Re-run with --install to install the pinned project runtime."
  exit 0
fi

cd "$PROJECT_DIR"
if [[ ! -f pyproject.toml || ! -f uv.lock ]]; then
  echo "ERROR: pyproject.toml and uv.lock are required in $PROJECT_DIR" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl is required to install uv." >&2
    exit 1
  fi
  INSTALLER=$(mktemp)
  trap 'rm -f "$INSTALLER"' EXIT
  curl -LsSf https://astral.sh/uv/install.sh -o "$INSTALLER"
  sh "$INSTALLER"
  export PATH="${UV_INSTALL_DIR:-$HOME/.local/bin}:$HOME/.cargo/bin:$PATH"
fi

echo "== Installing pinned dependencies =="
echo "Python runtime: ${PYTHON_VERSION}"
uv python install "$PYTHON_VERSION"
uv sync --frozen --extra gpu --python "$PYTHON_VERSION"

echo "== Runtime verification =="
uv run python - <<'PY'
import importlib
import os

for package in ("torch", "vllm", "openai", "yaml"):
    module = importlib.import_module(package)
    print(f"{package}: {getattr(module, '__version__', 'installed')}")

import torch
print(f"torch CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is unavailable. Inspect the driver/runtime compatibility before starting vLLM."
    )
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"HF_TOKEN inherited: {bool(os.environ.get('HF_TOKEN'))}")
PY

if [[ -n "${HF_TOKEN:-}" ]] && uv run hf auth whoami >/dev/null 2>&1; then
  echo "Hugging Face authentication: verified"
elif [[ -n "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is present but Hugging Face authentication failed." >&2
fi

echo "RunPod runtime is ready for a one-case vLLM smoke test."
