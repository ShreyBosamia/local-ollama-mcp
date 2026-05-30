#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="$(date -u +"%Y-%m-%dT%H-%M-%SZ")"
RUN_DIR=".local_ollama_mcp/eval_runs/$STAMP"
mkdir -p "$RUN_DIR"

echo "== py_compile =="
.venv/bin/python -m py_compile server.py eval_local_mcp.py

echo "== ollama ps =="
ollama ps || true

echo "== nvidia-smi =="
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,clocks.gr --format=csv || true

echo "== routing check =="
.venv/bin/python eval_local_mcp.py \
  --routing-check \
  --no-warm \
  --output-dir "$RUN_DIR/routing" \
  --json-name results.json \
  --markdown-name report.md

echo "== synthetic suite =="
.venv/bin/python eval_local_mcp.py \
  --suite synthetic \
  --output-dir "$RUN_DIR/synthetic" \
  --json-name results.json \
  --markdown-name report.md

echo "== reasoning suite =="
.venv/bin/python eval_local_mcp.py \
  --suite reasoning \
  --output-dir "$RUN_DIR/reasoning" \
  --json-name results.json \
  --markdown-name report.md

echo "== pipeline suite =="
.venv/bin/python eval_local_mcp.py \
  --suite pipeline \
  --output-dir "$RUN_DIR/pipeline" \
  --json-name results.json \
  --markdown-name report.md

echo "wrote offline eval run to $RUN_DIR"

