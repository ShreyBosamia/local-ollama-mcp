#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any
from unittest.mock import AsyncMock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import codex_telemetry  # noqa: E402
import agy_gemini_server  # noqa: E402
import server  # noqa: E402


VERIFY_TOOL_NAME = "gemini_summarize_context"


class VerificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class VerificationResult:
    ledger_row: dict[str, Any]
    source_row: dict[str, Any]
    turn_row: dict[str, Any]
    echo_line: str
    prompt_tokens_est: int
    prompt_chars: int
    elapsed_sec: float


@dataclass(frozen=True)
class ToolInvocationResult:
    output: str
    prompt_tokens_est: int
    prompt_chars: int
    elapsed_sec: float


@contextmanager
def patched_env(updates: dict[str, str]):
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.exists(), f"missing JSONL file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def synthetic_diff() -> str:
    lines = [
        (
            f"+ changed line {idx}: update payment reconciliation branch "
            f"with repeated customer invoice refund ledger retry context {idx % 17}"
        )
        for idx in range(1700)
    ]
    diff = "\n".join(lines)
    token_count = server.estimate_tokens(diff)
    require(
        token_count >= server.AGY_ROUTING_MIN_TOKENS,
        f"synthetic diff only estimated {token_count} tokens; need >= {server.AGY_ROUTING_MIN_TOKENS}",
    )
    return diff


def synthetic_context() -> str:
    lines = [
        (
            f"log line {idx}: worker={idx % 9} retry invoice refund state "
            f"transition from pending to settled with correlation id verify-{idx % 31}"
        )
        for idx in range(1700)
    ]
    text = "\n".join(lines)
    token_count = server.estimate_tokens(text)
    require(
        token_count >= server.AGY_ROUTING_MIN_TOKENS,
        f"synthetic context only estimated {token_count} tokens; need >= {server.AGY_ROUTING_MIN_TOKENS}",
    )
    return text


def token_count_event(timestamp: str) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "total_token_usage": {
                    "input_tokens": 18_000,
                    "cached_input_tokens": 2_000,
                    "output_tokens": 500,
                    "reasoning_output_tokens": 25,
                    "total_tokens": 18_500,
                },
                "last_token_usage": {
                    "input_tokens": 1_200,
                    "cached_input_tokens": 200,
                    "output_tokens": 120,
                    "reasoning_output_tokens": 10,
                    "total_tokens": 1_320,
                },
                "model_context_window": 128_000,
            },
            "rate_limits": {
                "primary": {"used_percent": 12.0, "window_minutes": 300, "resets_in_seconds": 60},
                "secondary": {"used_percent": 44.0, "window_minutes": 10_080, "resets_in_seconds": 120},
            },
        },
    }


def write_synthetic_session(session_path: Path, tool_output: str, *, timestamp: str) -> None:
    write_jsonl(
        session_path,
        [
            {
                "timestamp": timestamp,
                "type": "turn_context",
                "payload": {
                    "thread_id": "verify-gemini-thread",
                    "turn_id": "verify-gemini-turn",
                    "cwd": str(REPO_ROOT),
                    "model": "gpt-5-codex",
                    "effort": "high",
                },
            },
            {
                "timestamp": timestamp,
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": VERIFY_TOOL_NAME,
                    "arguments": "{}",
                    "call_id": "call-gemini-telemetry",
                },
            },
            {
                "timestamp": timestamp,
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-gemini-telemetry",
                    "output": json.dumps({"output": tool_output}),
                },
            },
            token_count_event(timestamp),
        ],
    )


async def invoke_gemini_context_reducer(*, mock_agy: bool) -> ToolInvocationResult:
    text = synthetic_context()
    started = time.perf_counter()
    if not mock_agy:
        output = await agy_gemini_server.gemini_summarize_context(text)
        return ToolInvocationResult(
            output=output,
            prompt_tokens_est=server.estimate_tokens(text),
            prompt_chars=len(text),
            elapsed_sec=round(time.perf_counter() - started, 3),
        )

    mock_output = (
        "SUMMARY\n"
        "- Invoice refund retry logs preserve worker ids, state transitions, and correlation ids.\n"
        "RISKS_OR_GAPS\n"
        "- None visible."
    )
    mock = AsyncMock(return_value=mock_output)
    with patch("server.ask_antigravity_with_fallback", mock):
        result = await agy_gemini_server.gemini_summarize_context(text)
    require(mock.await_count >= 1, "mock agy route was not invoked")
    for call in mock.await_args_list:
        require(
            call.kwargs.get("model") == server.AGY_FLASH_MODEL,
            f"mock agy route used unexpected model: {call.kwargs.get('model')}",
        )
    return ToolInvocationResult(
        output=result,
        prompt_tokens_est=server.estimate_tokens(text),
        prompt_chars=len(text),
        elapsed_sec=round(time.perf_counter() - started, 3),
    )


def validate_ledger(ledger_path: Path) -> dict[str, Any]:
    rows = read_jsonl(ledger_path)
    require(len(rows) == 1, f"expected exactly one MCP ledger row, found {len(rows)}")
    row = rows[0]
    require(row.get("tool_name") == VERIFY_TOOL_NAME, f"unexpected tool_name: {row.get('tool_name')}")
    require(row.get("model") == f"antigravity/{server.AGY_FLASH_MODEL}", f"unexpected model: {row.get('model')}")
    require(row.get("route_outcome") == "agy-default-gemini", f"unexpected route_outcome: {row.get('route_outcome')}")
    for field in ("gemini_input_tokens_est", "gemini_output_tokens_est", "gemini_saved_tokens_est"):
        require(isinstance(row.get(field), int), f"missing integer ledger field: {field}")
    require(row["gemini_input_tokens_est"] > row["gemini_output_tokens_est"], "Gemini output was not smaller than input")
    require(row["gemini_saved_tokens_est"] > 0, "gemini_saved_tokens_est was not positive")
    require(
        row["gemini_saved_tokens_est"] == row["gemini_input_tokens_est"] - row["gemini_output_tokens_est"],
        "gemini_saved_tokens_est does not equal input minus output",
    )
    estimates = row.get("token_estimates") or {}
    require(estimates.get("gemini_saved") == row["gemini_saved_tokens_est"], "token_estimates.gemini_saved mismatch")
    return row


def route_diagnostic(tool_output: str) -> str:
    lines = [line.strip() for line in tool_output.splitlines() if line.strip()]
    interesting = [
        line
        for line in lines
        if line.startswith(("route_outcome:", "token_savings:", "[agy_", "[local_timeout]", "fallback_also_failed"))
    ]
    if interesting:
        return " | ".join(interesting[:4])
    return tool_output[:240].replace("\n", " ")


def failure_stage(tool_output: str) -> str:
    if "[local_timeout]" in tool_output:
        return "outer-timeout"
    if "[agy_timeout]" in tool_output:
        return "agy"
    if "fallback_also_failed" in tool_output:
        return "local-fallback"
    if "[agy_circuit_open]" in tool_output:
        return "agy-circuit"
    if "[agy_missing_binary]" in tool_output:
        return "agy-binary"
    if "[agy_rate_limited]" in tool_output:
        return "agy-quota"
    if "[agy_error]" in tool_output:
        return "agy"
    return "validation"


def validate_echo_line(echo_line: str, saved_tokens: int) -> None:
    require(echo_line, "echo output was empty")
    formatted_saved = codex_telemetry.format_count(saved_tokens)
    expected = rf"saved local 0\s+agy {re.escape(formatted_saved)}\s+total {re.escape(formatted_saved)}"
    require(re.search(expected, echo_line), f"echo line missing saved-token breakdown: {echo_line}")


def validate_telemetry(telemetry_dir: Path, saved_tokens: int) -> tuple[dict[str, Any], dict[str, Any]]:
    sources = read_jsonl(telemetry_dir / "sources.jsonl")
    turns = read_jsonl(telemetry_dir / "turns.jsonl")
    require(len(sources) == 1, f"expected exactly one telemetry source row, found {len(sources)}")
    require(len(turns) == 1, f"expected exactly one telemetry turn row, found {len(turns)}")

    source = sources[0]
    turn = turns[0]
    require(source.get("tool_name") == VERIFY_TOOL_NAME, f"unexpected source tool_name: {source.get('tool_name')}")
    require(source.get("mcp_provider") == "agy", f"unexpected source mcp_provider: {source.get('mcp_provider')}")
    require(source.get("requested_provider") == "agy", f"unexpected requested_provider: {source.get('requested_provider')}")
    require(source.get("raw_context_avoided") == saved_tokens, "source raw_context_avoided did not match Gemini savings")
    require(turn.get("mcp_local_saved_tokens_est") == 0, "turn local saved tokens should be zero")
    require(turn.get("mcp_agy_saved_tokens_est") == saved_tokens, "turn agy saved tokens mismatch")
    require(turn.get("mcp_total_saved_tokens_est") == saved_tokens, "turn total saved tokens mismatch")
    return source, turn


async def run_verification(*, mock_agy: bool) -> VerificationResult:
    with tempfile.TemporaryDirectory(prefix="verify-gemini-telemetry-") as temp_name:
        root = Path(temp_name)
        ledger_path = root / "mcp-ledger.jsonl"
        telemetry_dir = root / "codex-telemetry"
        session_path = root / "rollout-2026-05-29T12-00-00-verify-gemini-telemetry.jsonl"

        with patched_env({"LOCAL_MCP_CAPTURE": "1", "LOCAL_MCP_LEDGER_PATH": str(ledger_path)}):
            agy_enabled = patch.object(server, "AGY_ENABLED", True) if mock_agy else nullcontext()
            with agy_enabled, patch("server.ensure_quota_monitor_started"):
                invocation = await invoke_gemini_context_reducer(mock_agy=mock_agy)

        tool_output = invocation.output
        size_detail = (
            f"prompt_tokens={invocation.prompt_tokens_est} "
            f"prompt_chars={invocation.prompt_chars} elapsed={invocation.elapsed_sec}s"
        )

        require(
            "route_outcome: agy-default-gemini" in tool_output,
            "tool output did not report Gemini route success "
            f"stage={failure_stage(tool_output)} {size_detail}: {route_diagnostic(tool_output)}",
        )
        require(
            "token_savings:" in tool_output,
            f"tool output did not include token_savings line {size_detail}: {route_diagnostic(tool_output)}",
        )
        ledger_row = validate_ledger(ledger_path)

        write_synthetic_session(session_path, tool_output, timestamp=str(ledger_row["timestamp"]))
        stream = io.StringIO()
        codex_telemetry.run_once(
            session_files=[session_path],
            telemetry_dir=telemetry_dir,
            local_mcp_ledger_path=ledger_path,
            env={"CODEX_TELEMETRY_CAPTURE": "1"},
            stdout=stream,
            echo=True,
            rotation_enabled=False,
        )
        source_row, turn_row = validate_telemetry(
            telemetry_dir,
            int(ledger_row["gemini_saved_tokens_est"]),
        )
        echo_line = stream.getvalue().strip()
        validate_echo_line(echo_line, int(ledger_row["gemini_saved_tokens_est"]))
        return VerificationResult(
            ledger_row=ledger_row,
            source_row=source_row,
            turn_row=turn_row,
            echo_line=echo_line,
            prompt_tokens_est=invocation.prompt_tokens_est,
            prompt_chars=invocation.prompt_chars,
            elapsed_sec=invocation.elapsed_sec,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Gemini Flash saved-token telemetry with isolated MCP and Codex fixtures."
    )
    parser.add_argument(
        "--mock-agy",
        action="store_true",
        help="Patch the Antigravity call for deterministic CI-style verification.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = datetime.now(timezone.utc).isoformat()
    try:
        result = asyncio.run(run_verification(mock_agy=args.mock_agy))
    except VerificationError as exc:
        print(f"FAIL verify_gemini_telemetry: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        mode = "mock" if args.mock_agy else "real"
        print(f"FAIL verify_gemini_telemetry ({mode} agy): {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    saved = result.ledger_row["gemini_saved_tokens_est"]
    mode = "mock" if args.mock_agy else "real"
    print(
        f"PASS verify_gemini_telemetry mode={mode} started={started} "
        f"route={result.ledger_row['route_outcome']} saved={saved} "
        f"prompt_tokens={result.prompt_tokens_est} prompt_chars={result.prompt_chars} "
        f"elapsed={result.elapsed_sec}s echo={result.echo_line}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
