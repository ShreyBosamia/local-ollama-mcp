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

## Evaluation

Run the local utility benchmark:

```bash
.venv/bin/python eval_local_mcp.py
```

It writes local-only reports:

- `local_mcp_eval_results.json`
- `local_mcp_eval_report.md`

These files are ignored by Git because they contain machine-specific telemetry and run-specific model output.

## Environment Variables

- `OLLAMA_BASE_URL`: defaults to `http://localhost:11434`
- `LOCAL_CODE_MODEL`: defaults to `qwen2.5-coder:7b-instruct-q5_K_M`
- `LOCAL_WARM_MODEL`: defaults to `LOCAL_CODE_MODEL`
- `LOCAL_KEEP_ALIVE`: defaults to `2h`
- `LOCAL_NUM_CTX`: defaults to `4096`
- `LOCAL_EXTENDED_NUM_CTX`: defaults to `6144`
- `LOCAL_OLLAMA_TIMEOUT`: defaults to `180`

## License

No open-source license has been selected yet.
