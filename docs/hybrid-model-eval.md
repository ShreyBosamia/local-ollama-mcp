# Hybrid Model Evaluation

Use this workflow to validate that MCP tools match the strengths of the local
Qwen model and the cloud models used through Antigravity.

## Full Run

```bash
scripts/run_hybrid_model_eval.sh
```

The runner writes all outputs under `.local_ollama_mcp/eval_runs/`, which is
ignored by Git. It forces the local code, plan, reason, and warm models to
`qwen3.5:9b` so local-only metrics are not mixed with another local model.

## What It Checks

- `py_compile` for `server.py`, `eval_local_mcp.py`, and the live cloud harness.
- Deterministic routing decisions without model calls.
- Qwen-only `synthetic` and `pipeline` suites.
- Live tool calls for:
  - local Qwen summarization, code review, test ideas, planning, and reasoning.
  - Gemini Flash routes for large summarize/review/diff compression payloads
    and generated walkthroughs.
  - Claude Thinking routes for large plan/reason payloads.
- `ollama ps` and `nvidia-smi` snapshots before and after model-backed runs.

## Interpreting Results

Local Qwen is a good default for a tool when it preserves required facts,
returns no `<think>` markers, keeps latency acceptable, and reduces cloud
context. Cloud routes are considered healthy when large payloads select
`antigravity`, avoid fallback prefixes such as `[agy_timeout]`, and preserve the
exact file names, errors, and risky lines that GPT-5.5 needs to verify.
The live harness also checks required terms and minimum compression for Gemini
compression routes. Walkthrough runs are scored against the actual `git diff
HEAD~1` payload because the tool reads the diff internally.

`local_generate_walkthrough` can receive an Antigravity response that points to
a generated local `file://.../walkthrough.md` artifact instead of returning the
Markdown inline. The server hydrates those Antigravity artifact links back into
the generated Markdown before returning or writing `walkthrough.md`.

The current Antigravity CLI may not support per-call `--model` selection. When
that is true, the live report records `agy per-call model selection: false` and
uses the model configured in `~/.gemini/antigravity-cli/settings.json`. Gemini
routes are valid only when that settings model is Gemini; Claude/Opus thinking
routes should be treated as `model_selection_unavailable` until the CLI exposes
per-call model selection or the settings model is changed deliberately before
the run.

If `agy /usage` fails with a TTY error, run the live cloud portion from an
interactive terminal or use the report as a local/fallback check only:

```bash
.venv/bin/python scripts/live_cloud_eval.py --output-dir .local_ollama_mcp/eval_runs/live_manual
```

For local-only validation:

```bash
.venv/bin/python scripts/live_cloud_eval.py --skip-cloud
```
