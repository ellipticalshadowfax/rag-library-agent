#!/usr/bin/env bash
# RAG Library Agent - one-command launcher for a new machine.
#
#   ./run.sh          -> checks Python, creates .venv, installs deps, starts web app
#   ./run.sh --port 5000
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PORT="${RAG_PORT:-5000}"
HOST="${RAG_HOST:-127.0.0.1}"
PY=python3

command -v "$PY" >/dev/null || { echo "ERROR: python3 not found. Install it first (python3-venv + python3-pip)."; exit 1; }

# 1. Create venv if needed
if [ ! -x "$HERE/.venv/bin/python" ]; then
  echo "==> Creating virtual environment (.venv)..."
  "$PY" -m venv .venv
fi

# 2. Install deps if missing (fast no-op once up to date)
echo "==> Ensuring dependencies are installed..."
"$HERE/.venv/bin/pip" install -q --upgrade pip
"$HERE/.venv/bin/pip" install -q -r requirements.txt

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
echo "    Open:  http://${HOST}:${PORT}"
echo "    Press Ctrl+C to stop."
echo
exec "$HERE/.venv/bin/python" scripts/server.py