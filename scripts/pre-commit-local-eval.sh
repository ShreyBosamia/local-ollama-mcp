#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

.venv/bin/python -m py_compile server.py eval_local_mcp.py
.venv/bin/python eval_local_mcp.py --routing-check --no-warm
