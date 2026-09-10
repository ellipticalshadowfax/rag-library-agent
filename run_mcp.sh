#!/usr/bin/env bash
# run_mcp.sh - Launch the RAG library as an MCP server for chat clients
# (LM Studio, Claude Desktop, etc.).
#
# Usage:
#   ./run_mcp.sh               # stdio transport (for LM Studio "Local" servers)
#   ./run_mcp.sh --http        # SSE/HTTP on 127.0.0.1:8765 (remote servers)
#   ./run_mcp.sh --http --port 9999
#
# For stdio, point the client at THIS script as the command (it knows how to
# find the project venv). Requires the index to already exist: run ./run.sh /
# scripts/ingest.py first.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# Data root override (see run.sh) so a symlinked deployment finds its data.
export RAG_ROOT="$HERE"

if [ ! -x .venv/bin/python ]; then
    echo "No .venv found. Run ./run.sh first (or create the venv manually)." >&2
    exit 1
fi

exec .venv/bin/python scripts/mcp_server.py "$@"