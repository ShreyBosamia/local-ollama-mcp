# Local Ollama MCP

Local Ollama MCP is a small Model Context Protocol server that gives Codex/GPT a local helper model for summarization, code review, test ideas, model warmup, and local GPU/Ollama status checks.

The default target model is:

```text
qwen3.5:9b
```

The server is tuned for an 8 GB NVIDIA GPU workflow:

- default `num_ctx` is `4096`
- default warm model keep-alive is `2h`
- `6144` context is guarded behind a `100% GPU` residency check
- raw `<think>...</think>` output is stripped before returning content
- tool responses use bounded `num_predict` values

## Setup

Install Ollama and pull the default model:

```bash
ollama pull qwen3.5:9b
```

Create a Python environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run the MCP server:

```bash
.venv/bin/python server.py
```

## MCP Tools

- `local_summarize`: compress code, logs, docs, or command output for Codex.
- `local_code_review`: review a diff for likely bugs, regressions, and missing tests.
- `local_test_ideas`: generate concise test ideas.
- `local_reason_check`: ask the configured reasoning model for a short second opinion.
- `local_plan_check`: ask the configured planning model for a compact plan.
- `local_warm_model`: keep the default local model resident. Run this before planned heavy local work so cold model loads do not consume your analysis budget.
- `local_unload_model`: request model unload.
- `local_ollama_status`: report Ollama residency and NVIDIA telemetry without changing GPU settings.
- `local_capture_status`: report whether local tool-call capture is enabled, where the ledger is written, and recent-window capture counts (note: scans bounded recent ledger rows, not lifetime totals).
- `local_record_outcome`: append an accepted-answer quality label for a captured local tool call.

## Agy Gemini MCP Tools

`agy_gemini_server.py` exposes Gemini 3.5 Flash High context reducers for large inputs that should be compressed before GPT/Codex sees them. Successful Gemini routes include `route_outcome` plus estimated saved-token telemetry where savings are raw input tokens minus Gemini output tokens.

- `gemini_route_context`: advisory router for raw context. It classifies `diff`, `logs`, `repo_map`, `config`, `pr_thread`, `mixed_context`, or `auto`, selects a Gemini reducer, and records `context_route_decision`, `selected_reducer`, `route_outcome`, and `gemini_saved_tokens_est` in the shared local MCP ledger when it runs a reducer.
- `gemini_compress_diff`: compress large git diffs through the Antigravity Gemini route.
- `gemini_summarize_context`: compress large logs, docs, command output, or pasted code.
- `gemini_debug_digest`: extract likely causes, exact errors, transitions, and next checks.
- `gemini_plan_task`: produce a compact implementation plan for GPT/Codex to verify.
- `gemini_review_diff`: pre-filter diffs for high-confidence findings and missing tests.
- `gemini_test_plan`: produce concise test scenarios without executable test code.
- `gemini_repo_map_digest`: compress repo trees or file lists into project orientation.
- `gemini_symbol_contract_digest`: compress signatures, types, schemas, or exported symbols.
- `gemini_config_surface_digest`: compress manifests, env docs, CI snippets, and tool settings.
- `gemini_pr_thread_digest`: compress PR comments, issue discussion, or CI summaries.
- `gemini_context_pack`: consolidate mixed task context into a minimal GPT-ready fact pack.

## Optional Local Capture

Capture is disabled by default. Enable it only when you want this MCP server to append local tool-call records for later evaluation or training-data export:

```bash
LOCAL_MCP_CAPTURE=1 .venv/bin/python server.py
```

Records are written to `.local_ollama_mcp/ledger.jsonl`, which is ignored by Git. Default capture redacts common secrets, secret-looking environment values, and home-directory paths. Raw capture is opt-in:

```bash
LOCAL_MCP_CAPTURE=1 LOCAL_MCP_CAPTURE_RAW=1 .venv/bin/python server.py
```

Useful environment variables:

- `LOCAL_MCP_CAPTURE=1`: enable MCP-boundary capture.
- `LOCAL_MCP_CAPTURE_RAW=1`: store raw input/output instead of redacted input/output.
- `LOCAL_MCP_LEDGER_PATH`: override `.local_ollama_mcp/ledger.jsonl`.

Outcome labels can be added after Codex finishes the real task:

```text
local_record_outcome(task_id, outcome, accepted_solution, notes="")
```

Allowed outcomes are `useful`, `needs_raw_verification`, `misleading`, `too_verbose`, and `skip_for_small_context`.

## Passive Codex Token Telemetry

`codex-telemetry-watcher` records Codex cloud token usage from Codex session JSONL files without asking the model to emit anything. It watches `event_msg` records where `payload.type` is `token_count`, stores exact cumulative `total_token_usage` numbers plus turn-local `last_token_usage` snapshots, and joins local MCP outputs to later token turns when a matching local ledger record is available.

Capture is disabled by default:

```bash
CODEX_TELEMETRY_CAPTURE=1 ./codex-telemetry-watcher
```

For a one-time fixture or real-session import:

```bash
CODEX_TELEMETRY_CAPTURE=1 ./codex-telemetry-watcher --once --session-file ~/.codex/sessions/YYYY/MM/DD/rollout-....jsonl
```

Optional compact terminal echo:

```bash
CODEX_TELEMETRY_CAPTURE=1 CODEX_TELEMETRY_ECHO=1 ./codex-telemetry-watcher
```

Convenience active watcher with capture, echo, and color enabled:

```bash
codex-telemetry-active
```

`codex-telemetry-active` resolves this repo from the command location, so it can
be run from any working directory. It defaults telemetry output to
`/home/shrey/local-ollama-mcp/.local_ollama_mcp/codex_telemetry` and local MCP
source attribution to `/home/shrey/local-ollama-mcp/.local_ollama_mcp/ledger.jsonl`.
Any normal watcher arguments still work, for example:

```bash
codex-telemetry-active --once --session-file ~/.codex/sessions/YYYY/MM/DD/rollout-....jsonl
```

Telemetry is written under `.local_ollama_mcp/codex_telemetry/`, which is ignored by Git:

- `events.jsonl`: normalized observations with payload hashes.
- `turns.jsonl`: one row per unique token-count payload with exact cumulative token fields, turn-local `turn_*` fields, quota usage provenance, cache pressure diagnostics, and payload hashes.
- `sources.jsonl`: estimated source attribution for local MCP output, shell output, or unknown function output.
- `state.sqlite3`: local dedupe/rotation state; it stores turn keys and rotation metadata, not raw session text.

Active telemetry ledgers rotate automatically when any active JSONL file reaches the configured byte threshold. Rotation is coordinated across `events.jsonl`, `turns.jsonl`, and `sources.jsonl`: compressed archives are written under `.local_ollama_mcp/codex_telemetry/archive/YYYY-MM-DD/`, and the active files stay in place for new writes. Archives older than the retention window are deleted during watcher writes or an explicit compaction pass:

```bash
CODEX_TELEMETRY_CAPTURE=1 ./codex-telemetry-watcher --compact-now
```

The terminal echo uses turn-local input/cache counts and active quota usage. It does not invert `used_percent`; it only converts fields explicitly named remaining or balance into usage. Multi-million cached-token observations are treated as provider-side cache accounting, not active context size, and high cache pressure is reported as passive diagnostics only.

The telemetry ledger stores hashes and token estimates by default, not raw prompt/session text. Repeated identical token-count snapshots are deduplicated by session, thread, turn, and cumulative usage signature so repeated imports do not create duplicate turn rows or repeated local-savings attribution.

Large function/tool outputs detected after a turn get advisory fields on the existing `sources.jsonl` row: `post_turn_advisory`, `advisory_category`, `advisory_raw_tokens_est`, and `advisory_reducer`. This only recommends using `gemini_route_context` or the selected reducer next time; it does not prevent current-turn token usage.

Native Codex pre-tool or post-tool hooks are a v2 path. Verify the hook/plugin contract with tiny fixtures before wiring automatic calls to `gemini_route_context`.
If no session JSONL files are available, `--tui-log` can recover thread/turn/model lifecycle metadata from `~/.codex/log/codex-tui.log`; it is diagnostic only and never provides token usage.

## Evaluation

Run the local utility benchmark:

```bash
.venv/bin/python eval_local_mcp.py
```

It writes local-only reports:

- `local_mcp_eval_results.json`
- `local_mcp_eval_report.md`

These files are ignored by Git because they contain machine-specific telemetry and run-specific model output.

Run named offline suites without any cloud model judge:

```bash
.venv/bin/python eval_local_mcp.py --suite synthetic
.venv/bin/python eval_local_mcp.py --suite reasoning
.venv/bin/python eval_local_mcp.py --suite artifacts
```

The synthetic suite evaluates fixed regression fixtures for summarization, code review, and test ideas. The reasoning suite targets `deepseek-r1:8b` through `local_reason_check` and fails structural scoring if `<think>` markers leak. The artifacts suite replays local JSONL cases from `.local_ollama_mcp/eval_cases/` and saved diffs/logs from `.local_ollama_mcp/eval_artifacts/`.

For recurring local regression runs:

```bash
scripts/run_offline_eval.sh
```

It runs syntax checks, local GPU/Ollama telemetry, deterministic routing checks, and model-backed synthetic/reasoning suites. Timestamped reports and the append-only run index are written under `.local_ollama_mcp/eval_runs/`.

Use `eval_cases.example.jsonl` as the tracked sanitized example format. Private fixtures and raw outputs should stay in `.local_ollama_mcp/`.

Run the Qwen-only local plus live cloud routing validation:

```bash
scripts/run_hybrid_model_eval.sh
```

This forces the local code, plan, reason, and warm models to `qwen3.5:9b`, runs deterministic local scoring, then exercises the Antigravity-backed cloud routes for large summarize/review/compress and plan/reason payloads. See `docs/hybrid-model-eval.md` for interpretation notes and the local-only fallback command.

Analyze captured real-session records without invoking Ollama:

```bash
.venv/bin/python eval_local_mcp.py --from-ledger
```

Run deterministic routing checks without invoking Ollama:

```bash
.venv/bin/python eval_local_mcp.py --routing-check --no-warm
```

The router marks artifacts under 120 estimated tokens as `skip_local`, diffs/logs from 120 to 4000 tokens as `use_local` only when the local output preserves required facts, avoids hard risk flags, and reduces cloud payload by at least 40%, incomplete but useful output as `verify_raw`, and tool errors, contradictions, think leakage, or oversized artifacts as `raw_cloud`.

Before large branch pushes or review summaries, route local diffs/logs through `local_code_review` or `local_summarize` and send only the accepted local summary to cloud.

Export labeled records for future training or regression evals:

```bash
.venv/bin/python export_training_data.py --format sft
.venv/bin/python export_training_data.py --format preference
.venv/bin/python export_training_data.py --format eval
```

## Environment Variables

- `OLLAMA_BASE_URL`: defaults to `http://localhost:11434`
- `LOCAL_CODE_MODEL`: defaults to `qwen3.5:9b`
- `LOCAL_WARM_MODEL`: defaults to `LOCAL_CODE_MODEL`
- `LOCAL_KEEP_ALIVE`: defaults to `2h`
- `LOCAL_NUM_CTX`: defaults to `4096`
- `LOCAL_EXTENDED_NUM_CTX`: defaults to `6144`
- `LOCAL_OLLAMA_TIMEOUT`: defaults to `180`
- `LOCAL_MCP_CAPTURE`: set to `1` to enable local JSONL capture
- `LOCAL_MCP_CAPTURE_RAW`: set to `1` to disable redaction for deliberate raw capture
- `LOCAL_MCP_LEDGER_PATH`: override the capture ledger path
- `CODEX_TELEMETRY_CAPTURE`: set to `1` to enable passive Codex session telemetry
- `CODEX_TELEMETRY_DIR`: defaults to `.local_ollama_mcp/codex_telemetry`
- `CODEX_TELEMETRY_ECHO`: set to `1` for compact watcher status lines
- `CODEX_TELEMETRY_SESSIONS_GLOB`: defaults to `~/.codex/sessions/**/*.jsonl`
- `CODEX_TELEMETRY_TUI_LOG`: defaults to `~/.codex/log/codex-tui.log`
- `CODEX_TELEMETRY_REDACT`: defaults to `1`; raw session text is not recorded in V1
- `CODEX_TELEMETRY_ROTATE_BYTES`: defaults to `25000000`
- `CODEX_TELEMETRY_RETENTION_DAYS`: defaults to `90`; set to `0` to keep archives
- `CODEX_TELEMETRY_ROTATION`: defaults to `1`; set to `0` to disable rotation

## License

No open-source license has been selected yet.
