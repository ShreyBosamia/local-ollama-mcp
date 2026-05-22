# Agent Notes

This repo contains a local Ollama MCP server for Codex-style workflows.

## Working Rules

- Do not commit `.venv/`, `__pycache__/`, or local evaluation output files.
- Keep the default code model as `qwen2.5-coder:7b-instruct-q5_K_M` unless the user explicitly asks to change it.
- Preserve the 8 GB VRAM guardrails: `num_ctx=4096` by default, `6144` only after `ollama ps` shows `100% GPU`.
- Keep GPU tuning outside `server.py`; server tools may read telemetry but should not mutate GPU clocks, voltage, fan curves, or power limits.
- Treat local model output as compression or second-opinion context, not as final authority.

## Verification

Use:

```bash
.venv/bin/python -m py_compile server.py eval_local_mcp.py
```

For runtime checks, use:

```bash
ollama ps
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,clocks.gr --format=csv
.venv/bin/python eval_local_mcp.py
```

The evaluation report is intentionally ignored by Git.
