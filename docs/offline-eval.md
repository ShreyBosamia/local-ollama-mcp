# Offline Evaluation

This repo evaluates local MCP model outputs without using a cloud model as a judge. The harness calls the local FastMCP tool functions in `server.py` directly, sends prompts to Ollama only when a model-backed suite is requested, and scores outputs with deterministic local rules.

## Suites

- `synthetic`: built-in fixed fixtures for `local_summarize`, `local_code_review`, and `local_test_ideas`.
- `reasoning`: built-in DeepSeek-R1 checks through `local_reason_check`, including a hard failure for leaked `<think>` markers.
- `artifacts`: local JSONL cases plus saved files under `.local_ollama_mcp/eval_artifacts/`.
- `all`: combines all suites.
- `--routing-check`: deterministic routing checks that do not call Ollama.
- `--from-ledger`: analyzes captured MCP ledger records without calling Ollama.

## Commands

```bash
.venv/bin/python eval_local_mcp.py --routing-check --no-warm
.venv/bin/python eval_local_mcp.py --suite synthetic
.venv/bin/python eval_local_mcp.py --suite reasoning
.venv/bin/python eval_local_mcp.py --suite artifacts
scripts/run_offline_eval.sh
```

The recurring runner writes timestamped reports to `.local_ollama_mcp/eval_runs/` and appends a compact row to `.local_ollama_mcp/eval_runs/index.jsonl`.

## Local Case Format

Place private cases in `.local_ollama_mcp/eval_cases/*.jsonl`. Each line is one JSON object:

```json
{"name":"saved_log_place_required","category":"log","tool":"local_summarize","task":"Summarize actionable errors.","artifact_path":"../eval_artifacts/logs/provider-import.log","focus":"exact errors and files","expected_recommendation":"use_local","max_bullets":6,"max_output_tokens":220,"expected_facts":[{"label":"Place required","pattern":"Place is required"},{"label":"DialogTitle warning","pattern":"DialogTitle|accessibility","required":false}],"forbidden_facts":[{"label":"think tag leaked","pattern":"</?think\\b"}]}
```

For artifact discovery without explicit facts:

- `.local_ollama_mcp/eval_artifacts/diffs/*.diff` is routed to `local_code_review`.
- `.local_ollama_mcp/eval_artifacts/logs/*.txt` and `*.log` are routed to `local_summarize`.

Discovered artifact cases are useful for latency and compression trend tracking. JSONL cases are better for accuracy regression testing because they include required and forbidden facts.

## Scoring

Pass/fail authority is deterministic:

- required fact regex coverage
- forbidden fact regex hits
- `<think>` marker leakage
- output token, character, and bullet limits
- JSON validity when requested
- estimated context reduction
- latency and route recommendation

`usefulness_score` combines fact accuracy, structure, and compression. `--local-judge` can add a local Ollama judge score, but that score is metadata only and should not be used as the release gate.

## Retention

Generated reports, raw fixtures, and run ledgers stay under `.local_ollama_mcp/`, which is ignored by Git. Only sanitized examples and harness code should be committed.
