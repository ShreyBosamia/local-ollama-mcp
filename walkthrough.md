# Walkthrough - Gemini Token Savings and Active Telemetry Command

## Executive Summary

This session added first-class saved-token accounting for successful Gemini Flash MCP routes and exposed the savings both in tool output and passive Codex telemetry. It also added a `codex-telemetry-active` command that enables capture, echo, and color from any working directory while keeping ledger paths anchored to this repo.

## Component & File Changes

| File Path | Description of Changes & Purpose |
| --- | --- |
| `server.py` | Added Gemini route detection and saved-token calculation helpers. Successful `antigravity/*gemini*` routes with `route_outcome: agy-default-gemini` now append a compact `token_savings` line to tool output. Ledger rows persist top-level Gemini token fields and nested `token_estimates.gemini_*` fields. |
| `server.py` | Updated capture flow so savings are calculated from the raw Gemini output before metadata is appended. Fallback outputs such as `[agy_timeout]`, `[agy_circuit_open]`, and `[agy_error]` do not claim Gemini savings. |
| `server.py` | Updated Gemini paths for `local_summarize`, `local_code_review`, and `agy_compress_diff` so the original text or diff is included in the captured input payload. This makes `gemini_input_tokens_est` reflect the actual payload being compressed, not only a small metadata object. |
| `codex_telemetry.py` | Added `saved_tokens_for_record()`. For Gemini Antigravity records it prefers `gemini_saved_tokens_est`, then `token_estimates.gemini_saved`, then falls back to older `token_estimates.context_reduction` rows. |
| `codex_telemetry.py` | Updated local ledger matching so `mcp_agy_saved_tokens_est` uses the new Gemini savings fields when available while preserving backward compatibility. |
| `test_eval_local_mcp.py` | Added and updated tests for successful Gemini `token_savings` output, fallback suppression, and captured ledger fields. |
| `test_codex_telemetry.py` | Added tests proving Gemini saved-token fields take precedence and older ledger rows still work through `context_reduction`. |
| `codex-telemetry-active` | Added an executable Bash wrapper that resolves the repo root from its own path, uses `.venv/bin/python` when available, enables telemetry capture/echo/color defaults, enables local MCP capture, anchors output paths to this repo, and forwards normal watcher arguments. |
| `/home/shrey/.local/bin/codex-telemetry-active` | Added a symlink to the repo wrapper so `codex-telemetry-active` works from any working directory because `/home/shrey/.local/bin` is already on `PATH`. |
| `README.md` | Documented `codex-telemetry-active`, including the default repo-anchored paths and an example `--once --session-file` invocation. |

## Behavior Details

Successful Gemini routes now report savings using this definition:

```text
original Gemini input tokens - Gemini output tokens sent back to GPT-5.5
```

The emitted tool-output line is:

```text
token_savings: gemini_input_est=<n> gpt_payload_est=<n> gpt_saved_est=<n> saved_pct=<n>%
```

The route must be a confirmed Gemini default route:

```text
model starts with antigravity/
model contains gemini
route_outcome == agy-default-gemini
```

Fallbacks and non-Gemini routes are intentionally excluded so telemetry does not overstate savings.

## Using the New Command

Run active telemetry from any directory:

```bash
codex-telemetry-active
```

It defaults these environment variables:

```bash
CODEX_TELEMETRY_CAPTURE=1
CODEX_TELEMETRY_ECHO=1
CODEX_TELEMETRY_COLOR=1
LOCAL_MCP_CAPTURE=1
```

It anchors paths to this repo by default:

```text
CODEX_TELEMETRY_DIR=/home/shrey/local-ollama-mcp/.local_ollama_mcp/codex_telemetry
LOCAL_MCP_LEDGER_PATH=/home/shrey/local-ollama-mcp/.local_ollama_mcp/ledger.jsonl
```

Normal watcher arguments still work:

```bash
codex-telemetry-active --once --session-file ~/.codex/sessions/YYYY/MM/DD/rollout-....jsonl
```

## Verification Guide

The following checks passed during this session:

```bash
.venv/bin/python -m py_compile server.py eval_local_mcp.py scripts/live_cloud_eval.py test_eval_local_mcp.py codex_telemetry.py test_codex_telemetry.py
.venv/bin/python -m unittest test_eval_local_mcp test_codex_telemetry
.venv/bin/python -m unittest discover
.venv/bin/python eval_local_mcp.py --routing-check --no-warm
bash -n codex-telemetry-active
```

The focused unit suite and discovery both ran `49` tests successfully.

A live Gemini route smoke test also succeeded and returned:

```text
route_outcome: agy-default-gemini
token_savings: ... gpt_saved_est=10743 ...
```

The active telemetry command was smoke-tested from `/tmp` with a temporary session JSONL and temporary telemetry directory. It emitted colored echo output and wrote one turn row.

## Risks & Edge Cases

- Token counts are estimates from the repo's existing tokenizer, so the savings line is an operational estimate rather than provider-billed token accounting.
- Gemini savings are only claimed for confirmed `agy-default-gemini` routes. If Antigravity falls back locally or times out, savings are intentionally omitted.
- The worktree already had unrelated modified and untracked files before these changes. This walkthrough focuses on the changes made in this session, not every dirty file currently present.
