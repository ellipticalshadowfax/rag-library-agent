#!/usr/bin/env bash
# RAG Library Agent - one-command launcher for a new machine.
#
#   ./run.sh                        -> CPU install + start web app
#   RAG_DEVICE=gpu ./run.sh         -> GPU install (NVIDIA driver + VRAM required)
#   RAG_PIP_EXTRAS=rapidocr ./run.sh -> also install the rapidocr OCR backend
#   RAG_PORT=8080 ./run.sh          -> custom port
#
# Install scheme (pyproject.toml):
#   - default (CPU) installs the CORE deps only, with the lean CPU-only torch
#     wheel pulled from the PyTorch CPU index. (PyPI's default `torch` now
#     bundles CUDA on Linux, which would bloat a CPU-only install.)
#   - optional feature extras add themselves via RAG_PIP_EXTRAS (comma list):
#       rapidocr   - heavy factored OCR backend (onnxruntime/opencv)
#       anki       - export quizzes as Anki .apkg decks
#   - GPU: RAG_DEVICE=gpu pre-installs the CUDA torch wheel off the PyTorch index
#     (which brings the matching nvidia-* runtime libs), then installs the [gpu]
#     extra. An extra alone can't express the PyTorch --extra-index-url, so run.sh
#     handles that step.
#   - requirements.txt / requirements-gpu.txt are kept as reproducibility lock
#     files; they are NOT the install path.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Data root: default to this directory. A symlinked deployment (code lives in
# the canonical repo, data lives in its own dir, e.g. /path/to/Data/RAG) sets
# this to the deployment dir so config.json/index/.venv are found there.
export RAG_ROOT="$HERE"

PORT="${RAG_PORT:-5000}"
HOST="${RAG_HOST:-127.0.0.1}"
PY=python3

command -v "$PY" >/dev/null || { echo "ERROR: python3 not found. Install it first (python3-venv + python3-pip)."; exit 1; }

# ─── uv bootstrap ─────────────────────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
  echo "==> uv not found — installing..."
  if curl -LsSf https://astral.sh/uv/install.sh 2>/dev/null | sh 2>/dev/null; then
    export PATH="$HOME/.local/bin:$PATH"
    echo "    uv installed to ~/.local/bin."
  elif "$PY" -m pip install uv >/dev/null 2>&1; then
    echo "    uv installed via pip."
  else
    echo "ERROR: failed to install uv. Install manually: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
  fi
fi

# ─── Device selection (CPU by default) ────────────────────────────────────────
DEVICE="${RAG_DEVICE:-cpu}"
if [ "${RAG_GPU:-}" = "1" ]; then
  DEVICE=gpu
fi

case "$DEVICE" in
  cpu)
    echo "==> Install mode: CPU (torch CPU wheel, no CUDA libraries)"
    ;;
  gpu)
    echo "==> Install mode: GPU (CUDA torch + nvidia libraries)"
    ;;
  *)
    echo "ERROR: unknown RAG_DEVICE='$DEVICE'. Use 'cpu' or 'gpu'."
    exit 1
    ;;
esac

# Optional feature extras (comma list, e.g. "rapidocr,anki")
PIP_EXTRAS="${RAG_PIP_EXTRAS:-}"

# CUDA wheel line used for GPU installs. Overridable. cu126 is verified with this
# project's torch 2.14.0 pin and CUDA 12.x system toolkits (Pascal+ cards).
# Set RAG_CUDA_VERSION=cu128 etc. for a different driver/toolkit.
CUDA_LINE="${RAG_CUDA_VERSION:-cu126}"
CUDA_INDEX="https://download.pytorch.org/whl/${CUDA_LINE}"
# PyTorch CPU-only wheel index. Used so the default install stays lean: PyPI's
# default `torch` wheel on Linux now bundles CUDA (~870 MB+), whereas the CPU-only
# wheel is ~200 MB and needs no CUDA/nvidia libs.
CPU_INDEX="https://download.pytorch.org/whl/cpu"

# 1. Create venv if needed
if [ ! -x "$HERE/.venv/bin/python" ]; then
  echo "==> Creating virtual environment (.venv)..."
  uv venv .venv --python "$PY"
fi

# 2. Install deps (fast no-op once up to date)
echo "==> Ensuring dependencies are installed..."
"$HERE/.venv/bin/python" -m pip install -q --upgrade pip 2>/dev/null || true

EXTRA_FLAGS=""
if [ -n "$PIP_EXTRAS" ]; then
  EXTRA_FLAGS="[${PIP_EXTRAS}]"
fi

if [ "$DEVICE" = "gpu" ]; then
  # GPU: install CUDA torch first from the PyTorch index (brings nvidia libs).
  # This prevents uv/pip from resolving torch's CUDA metadata from PyPI when the
  # extra-index-url is used below.
  uv pip install "torch" \
    --index-url "${CUDA_INDEX}" \
    --python "$HERE/.venv/bin/python"
  # Then install the app + [gpu] extra + any user extras.
  GPU_EXTRAS="gpu"
  if [ -n "$PIP_EXTRAS" ]; then
    GPU_EXTRAS="${PIP_EXTRAS},gpu"
  fi
  uv pip install -e ".[${GPU_EXTRAS}]" \
    --extra-index-url "${CUDA_INDEX}" \
    --extra-index-url "https://pypi.org/simple" \
    --python "$HERE/.venv/bin/python"
else
  # CPU: install the lean CPU-only torch wheel first (PyPI's default torch now
  # bundles CUDA on Linux ~870MB+; the +cpu wheel is ~200MB and needs no CUDA),
  # then the rest of the app from PyPI.
  echo "  (installing CPU-only torch wheel to keep the default install lean)"
  uv pip install "torch" \
    --index-url "${CPU_INDEX}" \
    --python "$HERE/.venv/bin/python"
  uv pip install -e ".${EXTRA_FLAGS}" \
    --extra-index-url "https://pypi.org/simple" \
    --python "$HERE/.venv/bin/python"
fi

# 3. First launch: download embedding model (pulled on demand, but fail fast here).
#    Device is auto-resolved at runtime (CPU unless a usable CUDA is present).
echo "==> Downloading embedding model (one-time)..."
"$HERE/.venv/bin/python" - <<'PY'
from sentence_transformers import SentenceTransformer
import json, pathlib
cfg = json.loads(pathlib.Path("config.json").read_text())
SentenceTransformer(cfg.get("embed_model", "intfloat/multilingual-e5-small"), device="cpu")
print("Embedding model ready.")
PY

# 4. Check the LLM API is reachable (warn only)
echo
if curl -s -m 3 "$(python3 -c 'import json;print(json.load(open("config.json"))["llm_base_url"])')/models" >/dev/null 2>&1; then
  echo "==> LLM API: online (good)."
else
  echo "==> NOTE: LLM API not detected on the configured URL."
  echo "    (Setup tab configures it: local LM Studio/llama.cpp, or a remote/cloud provider.)"
fi
echo

# 5. Launch web app
echo "==> Starting RAG Library Agent"
echo "    Device: ${DEVICE}"
echo "    Open:  http://${HOST}:${PORT}"
echo "    Press Ctrl+C to stop."
echo
exec "$HERE/.venv/bin/python" scripts/server.py
