#!/usr/bin/env bash
# RAG Library Agent - one-command launcher for a new machine.
#
#   ./run.sh              -> CPU install + start web app
#   RAG_DEVICE=gpu ./run.sh   -> GPU install (NVIDIA driver + VRAM required)
#   RAG_PORT=8080 ./run.sh    -> custom port
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

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
    REQ_FILE="requirements.txt"
    echo "==> Install mode: CPU (torch CPU wheel, no CUDA libraries)"
    ;;
  gpu)
    REQ_FILE="requirements-gpu.txt"
    echo "==> Install mode: GPU (CUDA torch + nvidia libraries)"
    ;;
  *)
    echo "ERROR: unknown RAG_DEVICE='$DEVICE'. Use 'cpu' or 'gpu'."
    exit 1
    ;;
esac

# 1. Create venv if needed
if [ ! -x "$HERE/.venv/bin/python" ]; then
  echo "==> Creating virtual environment (.venv)..."
  uv venv .venv --python "$PY"
fi

# 2. Install deps (fast no-op once up to date)
echo "==> Ensuring dependencies are installed..."
"$HERE/.venv/bin/python" -m pip install -q --upgrade pip 2>/dev/null || true

if [ "$DEVICE" = "cpu" ]; then
  # Pre-install torch CPU wheel first — this prevents uv from resolving torch's
  # CUDA metadata from PyPI when the extra-index-url is used below.
  uv pip install "torch==2.14.0+cpu" \
    --index-url "https://download.pytorch.org/whl/cpu" \
    --python "$HERE/.venv/bin/python"
  uv pip install -r "$REQ_FILE" \
    --extra-index-url "https://pypi.org/simple" \
    --python "$HERE/.venv/bin/python"
else
  uv pip install -r "$REQ_FILE" --python "$HERE/.venv/bin/python"
fi

# 3. First launch: install embedding model (pulled on demand, but fail fast here)
echo "==> Downloading embedding model (one-time)..."
"$HERE/.venv/bin/python" - <<'PY'
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from sentence_transformers import SentenceTransformer
import json, pathlib
cfg = json.loads(pathlib.Path("config.json").read_text())
SentenceTransformer(cfg.get("embed_model", "intfloat/multilingual-e5-small"), device="cpu")
print("Embedding model ready.")
PY

# 4. Check LM Studio is reachable (warn only)
echo
if curl -s -m 3 "$(python3 -c 'import json;print(json.load(open("config.json"))["llm_base_url"])')/models" >/dev/null 2>&1; then
  echo "==> LM Studio: online (good)."
else
  echo "==> NOTE: LM Studio not detected. Chat tab needs it on the configured URL."
  echo "    (Setup tab shows the exact URL; start LM Studio and enable its local server.)"
fi
echo

# 5. Launch web app
echo "==> Starting RAG Library Agent"
echo "    Device: ${DEVICE}"
echo "    Open:  http://${HOST}:${PORT}"
echo "    Press Ctrl+C to stop."
echo
exec "$HERE/.venv/bin/python" scripts/server.py
