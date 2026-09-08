#!/usr/bin/env bash
# start_lmstudio.sh - Launch LMStudio headless with a model for the RAG agent
#
# Usage:
#   ./start_lmstudio.sh              # Start with default model (auto-detect)
#   ./start_lmstudio.sh <model_path> # Start with a specific GGUF file
#
# The server will be available at http://localhost:1234/v1
# Use Ctrl+C to stop.

set -euo pipefail

LMSTUDIO_DIR="$HOME/.lmstudio"
PORT="${LMSTUDIO_PORT:-1234}"

# Detect LMStudio binary
LMSTUDIO_BIN=""
for candidate in \
    "$LMSTUDIO_DIR/bin/lmstudio" \
    "/usr/local/bin/lmstudio" \
    "$HOME/.local/bin/lmstudio" \
    "$(which lmstudio 2>/dev/null || true)"; do
    if [ -x "${candidate:-}" ]; then
        LMSTUDIO_BIN="$candidate"
        break
    fi
done

if [ -z "$LMSTUDIO_BIN" ]; then
    echo "Error: LMStudio binary not found."
    echo "Expected locations:"
    echo "  $LMSTUDIO_DIR/bin/lmstudio"
    echo "  /usr/local/bin/lmstudio"
    echo "  ~/.local/bin/lmstudio"
    echo ""
    echo "Please install LMStudio or set LMSTUDIO_BIN environment variable."
    exit 1
fi

echo "Using LMStudio: $LMSTUDIO_BIN"

if [ $# -gt 0 ]; then
    MODEL_PATH="$1"
    echo "Loading model: $MODEL_PATH"
    "$LMSTUDIO_BIN" serve --port "$PORT" "$MODEL_PATH"
else
    echo "Loading default model (last used)..."
    echo "Port: $PORT"
    echo ""
    echo "Note: Load a model via the LMStudio GUI first, then:"
    echo "  Server > Start Server (port $PORT)"
    echo ""
    echo "Or start directly with:"
    echo "  $0 /path/to/model.gguf"
    echo ""
    "$LMSTUDIO_BIN" serve --port "$PORT" 2>/dev/null || {
        echo "Could not start headless. Open LMStudio GUI:"
        echo "  $LMSTUDIO_BIN"
        echo "Then: Load a model > Developer tab > Start Server on port $PORT"
    }
fi
