#!/usr/bin/env python3
"""
Passive Codex token telemetry capture from persisted session JSONL files.

This module intentionally does not ask the model to print or calculate usage.
It tails Codex session records, normalizes token_count payloads, and writes
local-only append-only JSONL ledgers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, TextIO


DEFAULT_TELEMETRY_DIR = ".local_ollama_mcp/codex_telemetry"
DEFAULT_SESSIONS_GLOB = "~/.codex/sessions/**/*.jsonl"
DEFAULT_LOCAL_MCP_LEDGER = ".local_ollama_mcp/ledger.jsonl"
DEFAULT_TUI_LOG = "~/.codex/log/codex-tui.log"
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)
TUI_THREAD_RE = re.compile(r"session_loop\{thread_id=([^}]+)\}")
TUI_TURN_RE = re.compile(r"turn\.id=([^\s}]+)")
TUI_MODEL_RE = re.compile(r"\bmodel=([^\s}]+)")
TUI_EFFORT_RE = re.compile(r"codex\.turn\.reasoning_effort=([^\s}]+)")


def capture_enabled(env: dict[str, str] | None = None) -> bool:
    env = env or os.environ
    return env.get("CODEX_TELEMETRY_CAPTURE") == "1"


def echo_enabled(env: dict[str, str] | None = None) -> bool:
    env = env or os.environ
    return env.get("CODEX_TELEMETRY_ECHO") == "1"


def redact_enabled(env: dict[str, str] | None = None) -> bool:
    env = env or os.environ
    return env.get("CODEX_TELEMETRY_REDACT", "1") != "0"


def default_telemetry_dir(env: dict[str, str] | None = None) -> Path:
    env = env or os.environ
    return Path(env.get("CODEX_TELEMETRY_DIR", DEFAULT_TELEMETRY_DIR))


def default_sessions_glob(env: dict[str, str] | None = None) -> str:
    env = env or os.environ
    return env.get("CODEX_TELEMETRY_SESSIONS_GLOB", DEFAULT_SESSIONS_GLOB)


def default_tui_log(env: dict[str, str] | None = None) -> Path:
    env = env or os.environ
    return Path(os.path.expanduser(env.get("CODEX_TELEMETRY_TUI_LOG", DEFAULT_TUI_LOG)))


def estimate_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def value_hash(value: Any) -> str:
    return hashlib.sha256(json_text(value).encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json_text(value)


def session_thread_id(session_file: Path, context: dict[str, Any] | None = None) -> str:
    context = context or {}
    for key in ("thread_id", "conversation_id", "session_id"):
        if context.get(key):
            return str(context[key])
    match = UUID_RE.search(session_file.name)
    if match:
        return match.group(0)
    return session_file.stem


def event_turn_id(context: dict[str, Any], thread_id: str, event_seq: int) -> str:
    for key in ("turn_id", "turn_index", "step_id"):
        if context.get(key) is not None:
            return str(context[key])
    return f"{thread_id}:event-{event_seq:06d}"


def parse_arguments(raw_arguments: Any) -> Any:
    if not isinstance(raw_arguments, str):
        return raw_arguments
    try:
        return json.loads(raw_arguments)
    except json.JSONDecodeError:
        return raw_arguments


def extract_output_text(raw_output: Any) -> str:
    if not isinstance(raw_output, str):
        return stable_text(raw_output)
    try:
        decoded = json.loads(raw_output)
    except json.JSONDecodeError:
        return raw_output
    if isinstance(decoded, dict) and "output" in decoded:
        return stable_text(decoded.get("output", ""))
    return raw_output


def tool_name_from_payload(payload: dict[str, Any]) -> str:
    name = payload.get("name") or payload.get("tool_name") or "unknown"
    return str(name).split(".")[-1]


def source_kind_for_tool(tool_name: str) -> str:
    if tool_name in {"shell", "exec_command", "apply_patch"}:
        return "shell_output"
    if tool_name.startswith("local_"):
        return "local_mcp_output"
    return "unknown"


def source_ref(tool_name: str, call_id: str, arguments: Any) -> str:
    _ = arguments
    return f"{tool_name}:{call_id}"


def rate_reset_at(observed: str, seconds: Any) -> str | None:
    observed_dt = parse_timestamp(observed)
    if observed_dt is None or not isinstance(seconds, (int, float)):
        return None
    return (observed_dt + timedelta(seconds=float(seconds))).isoformat()


def usage_signature(info: dict[str, Any]) -> str:
    return value_hash(info.get("total_token_usage") or {})


@dataclass
class FunctionCall:
    call_id: str
    tool_name: str
    arguments: Any
    timestamp: str
    event_seq: int


@dataclass
class PendingSource:
    source_kind: str
    source_ref: str
    token_estimate: int
    output_hash: str
    call_id: str
    tool_name: str
    timestamp: str
    event_seq: int
    raw_context_avoided: int = 0
    confidence: str = "unknown"
    task_id: str | None = None


@dataclass
class ProcessResult:
    events: list[dict[str, Any]] = field(default_factory=list)
    turns: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)

    def extend(self, other: "ProcessResult") -> None:
        self.events.extend(other.events)
        self.turns.extend(other.turns)
        self.sources.extend(other.sources)


class JsonlLedger:
    def __init__(self, telemetry_dir: Path):
        self.telemetry_dir = telemetry_dir

    def append(self, name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        path = self.telemetry_dir / name
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json_text(row) + "\n")

    def append_result(self, result: ProcessResult) -> None:
        self.append("events.jsonl", result.events)
        self.append("turns.jsonl", result.turns)
        self.append("sources.jsonl", result.sources)


class LocalLedgerIndex:
    def __init__(self, path: Path):
        self.records = self._read(path)

    @staticmethod
    def _read(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    records.append(value)
        return records

    def match(self, source: PendingSource) -> PendingSource:
        if source.source_kind != "local_mcp_output":
            return source

        source_dt = parse_timestamp(source.timestamp)
        best_time_match: tuple[float, dict[str, Any]] | None = None
        for record in self.records:
            if record.get("record_type") != "tool_call":
                continue
            if str(record.get("tool_name", "")) != source.tool_name:
                continue

            local_output = stable_text(record.get("local_output", ""))
            local_hashes = {text_hash(local_output), text_hash(extract_output_text(local_output))}
            if source.output_hash in local_hashes:
                return self._apply_record(source, record, "matched_hash")

            record_dt = parse_timestamp(record.get("timestamp"))
            if source_dt is None or record_dt is None:
                continue
            distance = abs((source_dt - record_dt).total_seconds())
            if distance <= 120 and (best_time_match is None or distance < best_time_match[0]):
                best_time_match = (distance, record)

        if best_time_match is not None:
            return self._apply_record(source, best_time_match[1], "timestamp_match")
        return source

    @staticmethod
    def _apply_record(source: PendingSource, record: dict[str, Any], confidence: str) -> PendingSource:
        token_estimates = record.get("token_estimates") or {}
        raw_context_avoided = int(token_estimates.get("context_reduction") or 0)
        source.raw_context_avoided = raw_context_avoided
        source.confidence = confidence
        if record.get("task_id"):
            source.task_id = str(record["task_id"])
        return source


class SessionProcessor:
    def __init__(self, session_file: Path, local_ledger: LocalLedgerIndex | None = None):
        self.session_file = session_file
        self.local_ledger = local_ledger or LocalLedgerIndex(Path("__missing_local_ledger__"))
        self.context: dict[str, Any] = {}
        self.calls: dict[str, FunctionCall] = {}
        self.pending_sources: list[PendingSource] = []
        self.last_usage_signature: str | None = None

    def process_line(self, line: str, event_seq: int) -> ProcessResult:
        result = ProcessResult()
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return result
        if not isinstance(record, dict):
            return result

        record_type = str(record.get("type", ""))
        payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
        timestamp = str(record.get("timestamp") or utc_now())
        thread_id = session_thread_id(self.session_file, self.context)
        turn_id = event_turn_id(self.context, thread_id, event_seq)

        if record_type == "turn_context":
            self.context = payload
            thread_id = session_thread_id(self.session_file, self.context)
            turn_id = event_turn_id(self.context, thread_id, event_seq)
            result.events.append(self._event_row(record_type, timestamp, event_seq, thread_id, turn_id, payload))
            return result

        if record_type == "response_item":
            result.events.append(self._event_row(record_type, timestamp, event_seq, thread_id, turn_id, payload))
            self._handle_response_item(payload, timestamp, event_seq)
            return result

        if record_type == "event_msg" and payload.get("type") == "token_count":
            result.events.append(self._event_row("token_count", timestamp, event_seq, thread_id, turn_id, payload))
            turn = self._turn_row(payload, timestamp, event_seq, thread_id, turn_id)
            if turn is None:
                return result

            current_signature = usage_signature(payload.get("info") or {})
            usage_advanced = current_signature != self.last_usage_signature
            self.last_usage_signature = current_signature
            result.turns.append(turn)
            if usage_advanced:
                result.sources.extend(self._source_rows_for_turn(turn))
                self.pending_sources.clear()
            return result

        return result

    def _event_row(
        self,
        record_type: str,
        timestamp: str,
        event_seq: int,
        thread_id: str,
        turn_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "record_type": record_type,
            "observed_at": timestamp,
            "session_file": str(self.session_file),
            "thread_id": thread_id,
            "turn_id": turn_id,
            "event_seq": event_seq,
            "payload_hash": value_hash(payload),
        }

    def _handle_response_item(self, payload: dict[str, Any], timestamp: str, event_seq: int) -> None:
        payload_type = payload.get("type")
        if payload_type in {"function_call", "custom_tool_call"}:
            call_id = str(payload.get("call_id") or f"missing-call-{event_seq}")
            self.calls[call_id] = FunctionCall(
                call_id=call_id,
                tool_name=tool_name_from_payload(payload),
                arguments=parse_arguments(payload.get("arguments") or payload.get("input")),
                timestamp=timestamp,
                event_seq=event_seq,
            )
            return

        if payload_type not in {"function_call_output", "custom_tool_call_output"}:
            return
        call_id = str(payload.get("call_id") or f"missing-output-{event_seq}")
        call = self.calls.get(call_id)
        tool_name = call.tool_name if call else "unknown"
        arguments = call.arguments if call else {}
        output_text = extract_output_text(payload.get("output", ""))
        pending = PendingSource(
            source_kind=source_kind_for_tool(tool_name),
            source_ref=source_ref(tool_name, call_id, arguments),
            token_estimate=estimate_tokens(output_text),
            output_hash=text_hash(output_text),
            call_id=call_id,
            tool_name=tool_name,
            timestamp=timestamp,
            event_seq=event_seq,
        )
        self.pending_sources.append(self.local_ledger.match(pending))

    def _turn_row(
        self,
        payload: dict[str, Any],
        timestamp: str,
        event_seq: int,
        thread_id: str,
        turn_id: str,
    ) -> dict[str, Any] | None:
        info = payload.get("info")
        if not isinstance(info, dict):
            return None
        total = info.get("total_token_usage") or {}
        delta = info.get("last_token_usage") or {}
        context_window = info.get("model_context_window")
        if not isinstance(total, dict) or not isinstance(delta, dict):
            return None

        input_tokens = int(total.get("input_tokens") or 0)
        context_used_pct = (
            round((input_tokens / int(context_window)) * 100, 3)
            if isinstance(context_window, int) and context_window > 0
            else None
        )
        primary = (payload.get("rate_limits") or {}).get("primary") or {}
        secondary = (payload.get("rate_limits") or {}).get("secondary") or {}

        return {
            "record_type": "turn",
            "observed_at": timestamp,
            "session_file": str(self.session_file),
            "thread_id": thread_id,
            "turn_id": turn_id,
            "event_seq": event_seq,
            "model": self.context.get("model"),
            "reasoning_effort": self.context.get("effort") or self.context.get("reasoning_effort"),
            "mode": self.context.get("mode") or self.context.get("summary"),
            "cwd": self.context.get("cwd"),
            "input_tokens": input_tokens,
            "cached_input_tokens": int(total.get("cached_input_tokens") or 0),
            "output_tokens": int(total.get("output_tokens") or 0),
            "reasoning_output_tokens": int(total.get("reasoning_output_tokens") or 0),
            "total_tokens": int(total.get("total_tokens") or 0),
            "delta_input_tokens": int(delta.get("input_tokens") or 0),
            "delta_cached_input_tokens": int(delta.get("cached_input_tokens") or 0),
            "delta_output_tokens": int(delta.get("output_tokens") or 0),
            "delta_reasoning_output_tokens": int(delta.get("reasoning_output_tokens") or 0),
            "delta_total_tokens": int(delta.get("total_tokens") or 0),
            "context_window": context_window,
            "context_used_pct": context_used_pct,
            "primary_rate_used_pct": primary.get("used_percent"),
            "primary_rate_window_minutes": primary.get("window_minutes"),
            "primary_rate_resets_in_seconds": primary.get("resets_in_seconds"),
            "primary_rate_reset_at": rate_reset_at(timestamp, primary.get("resets_in_seconds")),
            "secondary_rate_used_pct": secondary.get("used_percent"),
            "secondary_rate_window_minutes": secondary.get("window_minutes"),
            "secondary_rate_resets_in_seconds": secondary.get("resets_in_seconds"),
            "secondary_rate_reset_at": rate_reset_at(timestamp, secondary.get("resets_in_seconds")),
            "plan_type": payload.get("plan_type") or info.get("plan_type"),
            "snapshot_kind": "current_final_at_observation",
        }

    def _source_rows_for_turn(self, turn: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for source in self.pending_sources:
            rows.append(
                {
                    "record_type": "source",
                    "observed_at": turn["observed_at"],
                    "session_file": turn["session_file"],
                    "thread_id": turn["thread_id"],
                    "turn_id": turn["turn_id"],
                    "turn_event_seq": turn["event_seq"],
                    "source_event_seq": source.event_seq,
                    "source_kind": source.source_kind,
                    "source_ref": source.source_ref,
                    "call_id": source.call_id,
                    "task_id": source.task_id,
                    "token_estimate": source.token_estimate,
                    "raw_context_avoided": source.raw_context_avoided,
                    "confidence": source.confidence,
                    "source_hash": source.output_hash,
                }
            )
        return rows


def process_session_file(session_file: Path, local_ledger: LocalLedgerIndex | None = None) -> ProcessResult:
    processor = SessionProcessor(session_file, local_ledger)
    result = ProcessResult()
    if not session_file.exists():
        return result
    with session_file.open("r", encoding="utf-8") as handle:
        for event_seq, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            result.extend(processor.process_line(line, event_seq))
    return result


def process_session_chunk(
    session_file: Path,
    lines: list[str],
    first_event_seq: int,
    processor: SessionProcessor,
) -> ProcessResult:
    result = ProcessResult()
    for offset, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        result.extend(processor.process_line(line, first_event_seq + offset))
    return result


def session_files_from_glob(pattern: str) -> list[Path]:
    expanded = os.path.expanduser(pattern)
    return [Path(path) for path in sorted(glob.glob(expanded, recursive=True))]


def parse_tui_log_lifecycle(path: Path, max_lines: int = 2000) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]
    events: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str | None, str | None]] = set()
    for event_seq, line in enumerate(lines, start=1):
        thread_match = TUI_THREAD_RE.search(line)
        turn_match = TUI_TURN_RE.search(line)
        if not thread_match or not turn_match:
            continue
        timestamp = line.split(" ", 1)[0]
        thread_id = thread_match.group(1)
        turn_id = turn_match.group(1)
        model_match = TUI_MODEL_RE.search(line)
        effort_match = TUI_EFFORT_RE.search(line)
        model = model_match.group(1).strip('"') if model_match else None
        effort = effort_match.group(1).strip('"') if effort_match else None
        key = (thread_id, turn_id, model, effort)
        if key in seen:
            continue
        seen.add(key)
        payload = {
            "source": "codex-tui.log",
            "thread_id": thread_id,
            "turn_id": turn_id,
            "model": model,
            "reasoning_effort": effort,
        }
        events.append(
            {
                "record_type": "tui_lifecycle",
                "observed_at": timestamp,
                "session_file": str(path),
                "thread_id": thread_id,
                "turn_id": turn_id,
                "event_seq": event_seq,
                "payload_hash": value_hash(payload),
                "model": model,
                "reasoning_effort": effort,
            }
        )
    return events


def format_count(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 10_000:
        return f"{value / 1_000:.0f}K"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(int(value))


def format_echo_line(turn: dict[str, Any], sources: list[dict[str, Any]]) -> str:
    local_saved = sum(int(source.get("raw_context_avoided") or 0) for source in sources)
    ctx = turn.get("context_used_pct")
    ctx_text = f"{ctx:.0f}%" if isinstance(ctx, (int, float)) else "n/a"
    primary = turn.get("primary_rate_used_pct")
    secondary = turn.get("secondary_rate_used_pct")
    primary_text = f"{primary:.0f}%" if isinstance(primary, (int, float)) else "n/a"
    secondary_text = f"{secondary:.0f}%" if isinstance(secondary, (int, float)) else "n/a"
    return (
        "codex-telemetry: tokens: "
        f"in {format_count(turn.get('input_tokens'))} "
        f"(+{format_count(turn.get('cached_input_tokens'))} cached), "
        f"out {format_count(turn.get('delta_output_tokens'))}, "
        f"reason {format_count(turn.get('delta_reasoning_output_tokens'))} | "
        f"ctx {ctx_text} of {format_count(turn.get('context_window'))} | "
        f"5h {primary_text}, weekly {secondary_text} | "
        f"local saved est {format_count(local_saved)}"
    )


def emit_echo(result: ProcessResult, stdout: TextIO) -> None:
    if not result.turns:
        return
    for turn in result.turns:
        turn_sources = [
            source
            for source in result.sources
            if source.get("thread_id") == turn.get("thread_id")
            and source.get("turn_id") == turn.get("turn_id")
            and source.get("turn_event_seq") == turn.get("event_seq")
        ]
        print(format_echo_line(turn, turn_sources), file=stdout, flush=True)


def run_once(
    *,
    session_files: list[Path] | None = None,
    sessions_glob: str | None = None,
    telemetry_dir: Path | None = None,
    local_mcp_ledger_path: Path | None = None,
    tui_log_path: Path | None = None,
    env: dict[str, str] | None = None,
    stdout: TextIO | None = None,
    echo: bool | None = None,
) -> ProcessResult:
    env = env or os.environ
    stdout = stdout or sys.stdout
    if not capture_enabled(env):
        return ProcessResult()

    telemetry_dir = telemetry_dir or default_telemetry_dir(env)
    local_mcp_ledger_path = local_mcp_ledger_path or Path(DEFAULT_LOCAL_MCP_LEDGER)
    local_ledger = LocalLedgerIndex(local_mcp_ledger_path)
    files = session_files if session_files is not None else session_files_from_glob(sessions_glob or default_sessions_glob(env))
    ledger = JsonlLedger(telemetry_dir)
    combined = ProcessResult()
    for session_file in files:
        result = process_session_file(session_file, local_ledger)
        ledger.append_result(result)
        combined.extend(result)
        should_echo = echo if echo is not None else echo_enabled(env)
        if should_echo:
            emit_echo(result, stdout)
    if not files or not combined.events:
        fallback_events = parse_tui_log_lifecycle(tui_log_path or default_tui_log(env))
        fallback_result = ProcessResult(events=fallback_events)
        ledger.append_result(fallback_result)
        combined.extend(fallback_result)
    return combined


def watch(
    *,
    sessions_glob: str,
    telemetry_dir: Path,
    local_mcp_ledger_path: Path,
    poll_interval: float,
    from_end: bool,
    stdout: TextIO,
    env: dict[str, str] | None = None,
    echo: bool | None = None,
) -> None:
    env = env or os.environ
    if not capture_enabled(env):
        return

    ledger = JsonlLedger(telemetry_dir)
    local_ledger = LocalLedgerIndex(local_mcp_ledger_path)
    processors: dict[Path, SessionProcessor] = {}
    offsets: dict[Path, int] = {}
    seqs: dict[Path, int] = {}
    should_echo = echo if echo is not None else echo_enabled(env)

    while True:
        for session_file in session_files_from_glob(sessions_glob):
            if session_file not in offsets:
                offsets[session_file] = session_file.stat().st_size if from_end else 0
                seqs[session_file] = 1
                processors[session_file] = SessionProcessor(session_file, local_ledger)

            size = session_file.stat().st_size
            if size < offsets[session_file]:
                offsets[session_file] = 0
                seqs[session_file] = 1
                processors[session_file] = SessionProcessor(session_file, local_ledger)
            if size == offsets[session_file]:
                continue

            with session_file.open("r", encoding="utf-8") as handle:
                handle.seek(offsets[session_file])
                chunk = handle.read()
                offsets[session_file] = handle.tell()
            lines = chunk.splitlines()
            result = process_session_chunk(
                session_file,
                lines,
                seqs[session_file],
                processors[session_file],
            )
            seqs[session_file] += len(lines)
            ledger.append_result(result)
            if should_echo:
                emit_echo(result, stdout)

        time.sleep(poll_interval)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Passive Codex token telemetry watcher.")
    parser.add_argument("--once", action="store_true", help="Process matching session files once and exit.")
    parser.add_argument("--session-file", action="append", default=[], help="Specific Codex session JSONL file to process.")
    parser.add_argument("--sessions-glob", default="", help="Session glob; defaults to CODEX_TELEMETRY_SESSIONS_GLOB or ~/.codex/sessions/**/*.jsonl.")
    parser.add_argument("--telemetry-dir", default="", help="Output directory; defaults to CODEX_TELEMETRY_DIR or .local_ollama_mcp/codex_telemetry.")
    parser.add_argument("--local-mcp-ledger", default=DEFAULT_LOCAL_MCP_LEDGER, help="Local MCP capture ledger used for source attribution.")
    parser.add_argument("--tui-log", default="", help="Diagnostic codex-tui.log fallback for lifecycle metadata.")
    parser.add_argument("--echo", action="store_true", help="Print compact post-turn telemetry lines.")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Watch poll interval in seconds.")
    parser.add_argument("--from-start", action="store_true", help="When watching, replay existing file contents before following appends.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, stdout: TextIO | None = None) -> int:
    args = parse_args(argv)
    stdout = stdout or sys.stdout
    env = os.environ
    telemetry_dir = Path(args.telemetry_dir) if args.telemetry_dir else default_telemetry_dir(env)
    sessions_glob = args.sessions_glob or default_sessions_glob(env)
    session_files = [Path(path) for path in args.session_file] if args.session_file else None
    echo = args.echo or echo_enabled(env)

    if args.once:
        run_once(
            session_files=session_files,
            sessions_glob=sessions_glob,
            telemetry_dir=telemetry_dir,
            local_mcp_ledger_path=Path(args.local_mcp_ledger),
            tui_log_path=Path(args.tui_log) if args.tui_log else default_tui_log(env),
            env=env,
            stdout=stdout,
            echo=echo,
        )
        return 0

    watch(
        sessions_glob=sessions_glob,
        telemetry_dir=telemetry_dir,
        local_mcp_ledger_path=Path(args.local_mcp_ledger),
        poll_interval=args.poll_interval,
        from_end=not args.from_start,
        stdout=stdout,
        env=env,
        echo=echo,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
