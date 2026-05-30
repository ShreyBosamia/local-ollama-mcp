#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="$(date -u +"%Y-%m-%dT%H-%M-%SZ")"
RUN_DIR=".local_ollama_mcp/eval_runs/hybrid_$STAMP"
mkdir -p "$RUN_DIR"

export LOCAL_CODE_MODEL="${LOCAL_CODE_MODEL:-qwen3.5:9b}"
export LOCAL_PLAN_MODEL="${LOCAL_PLAN_MODEL:-qwen3.5:9b}"
export LOCAL_REASON_MODEL="${LOCAL_REASON_MODEL:-qwen3.5:9b}"
export LOCAL_WARM_MODEL="${LOCAL_WARM_MODEL:-qwen3.5:9b}"

echo "== py_compile =="
.venv/bin/python -m py_compile server.py eval_local_mcp.py scripts/live_cloud_eval.py

echo "== hardware before =="
ollama ps > "$RUN_DIR/ollama_ps_before.txt" || true
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,clocks.gr --format=csv > "$RUN_DIR/nvidia_smi_before.csv" || true

echo "== deterministic routing check =="
.venv/bin/python eval_local_mcp.py \
  --routing-check \
  --no-warm \
  --output-dir "$RUN_DIR/routing" \
  --json-name results.json \
  --markdown-name report.md

echo "== qwen-only synthetic suite =="
.venv/bin/python eval_local_mcp.py \
  --suite synthetic \
  --model qwen3.5:9b \
  --plan-model qwen3.5:9b \
  --reason-model qwen3.5:9b \
  --warm-model qwen3.5:9b \
  --output-dir "$RUN_DIR/qwen_synthetic" \
  --json-name results.json \
  --markdown-name report.md

echo "== qwen-only pipeline suite =="
.venv/bin/python eval_local_mcp.py \
  --suite pipeline \
  --model qwen3.5:9b \
  --plan-model qwen3.5:9b \
  --reason-model qwen3.5:9b \
  --warm-model qwen3.5:9b \
  --output-dir "$RUN_DIR/qwen_pipeline" \
  --json-name results.json \
  --markdown-name report.md

echo "== live hybrid local/cloud suite =="
.venv/bin/python scripts/live_cloud_eval.py \
  --output-dir "$RUN_DIR/live_hybrid"

echo "== hardware after =="
ollama ps > "$RUN_DIR/ollama_ps_after.txt" || true
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,clocks.gr --format=csv > "$RUN_DIR/nvidia_smi_after.csv" || true

echo "== dashboard =="
.venv/bin/python eval_local_mcp.py --write-dashboard --dashboard-path EVAL_DASHBOARD.md

echo "wrote hybrid eval run to $RUN_DIR"
