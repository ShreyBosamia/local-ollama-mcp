# Local Ollama MCP

Local Ollama MCP is a small Model Context Protocol server that gives Codex/GPT a local helper model for summarization, code review, test ideas, model warmup, and local GPU/Ollama status checks.

The default target model is:

```text
qwen2.5-coder:7b-instruct-q5_K_M
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
ollama pull qwen2.5-coder:7b-instruct-q5_K_M
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
- `local_warm_model`: keep the default local model resident.
- `local_unload_model`: request model unload.
- `local_ollama_status`: report Ollama residency and NVIDIA telemetry without changing GPU settings.
- `local_capture_status`: report whether local tool-call capture is enabled and where the ledger is written.
- `local_record_outcome`: append an accepted-answer quality label for a captured local tool call.

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

`codex-telemetry-watcher` records Codex cloud token usage from Codex session JSONL files without asking the model to emit anything. It watches `event_msg` records where `payload.type` is `token_count`, stores exact `total_token_usage` and `last_token_usage` numbers, and joins local MCP outputs to later token turns when a matching local ledger record is available.

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

Telemetry is written under `.local_ollama_mcp/codex_telemetry/`, which is ignored by Git:

- `events.jsonl`: normalized observations with payload hashes.
- `turns.jsonl`: one row per token-count payload with exact cumulative and delta token fields.
- `sources.jsonl`: estimated source attribution for local MCP output, shell output, or unknown function output.

The telemetry ledger stores hashes and token estimates by default, not raw prompt/session text.
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

Analyze captured real-session records without invoking Ollama:

```bash
.venv/bin/python eval_local_mcp.py --from-ledger
```

Export labeled records for future training or regression evals:

```bash
.venv/bin/python export_training_data.py --format sft
.venv/bin/python export_training_data.py --format preference
.venv/bin/python export_training_data.py --format eval
```

## Environment Variables

- `OLLAMA_BASE_URL`: defaults to `http://localhost:11434`
- `LOCAL_CODE_MODEL`: defaults to `qwen2.5-coder:7b-instruct-q5_K_M`
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

## License

No open-source license has been selected yet.
