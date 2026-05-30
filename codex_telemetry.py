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
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import time
from typing import Any, TextIO


DEFAULT_TELEMETRY_DIR = ".local_ollama_mcp/codex_telemetry"
DEFAULT_SESSIONS_GLOB = "~/.codex/sessions/**/*.jsonl"
DEFAULT_LOCAL_MCP_LEDGER = ".local_ollama_mcp/ledger.jsonl"
DEFAULT_TUI_LOG = "~/.codex/log/codex-tui.log"
DEFAULT_TELEMETRY_ROTATE_BYTES = 25_000_000
DEFAULT_TELEMETRY_RETENTION_DAYS = 90
TELEMETRY_LEDGER_FILES = ("events.jsonl", "turns.jsonl", "sources.jsonl")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)
TUI_THREAD_RE = re.compile(r"session_loop\{thread_id=([^}]+)\}")
TUI_TURN_RE = re.compile(r"turn\.id=([^\s}]+)")
TUI_MODEL_RE = re.compile(r"\bmodel=([^\s}]+)")
TUI_EFFORT_RE = re.compile(r"codex\.turn\.reasoning_effort=([^\s}]+)")
AGY_FALLBACK_PREFIXES = (
    "[agy_rate_limited]",
    "[agy_timeout]",
    "[agy_missing_binary]",
    "[agy_error]",
    "[agy_circuit_open]",
)
POST_TURN_ADVISORY_MIN_TOKENS = int(os.getenv("CODEX_TELEMETRY_ADVISORY_MIN_TOKENS", "4000"))
ADVISORY_DIFF_RE = re.compile(r"(?m)^(diff --git|@@ |\+\+\+ |--- |\+[^+]|-[^-])")
ADVISORY_LOG_RE = re.compile(r"(?i)(traceback|exception|error|failed|failure|timeout|segfault|panic|stack trace)")
ADVISORY_REPO_MAP_RE = re.compile(
    r"(?m)^([./\w-]+/|[./\w-]+\.(py|js|ts|tsx|go|rs|md|toml|json|ya?ml|txt))$"
)
ADVISORY_CONFIG_RE = re.compile(r"(?im)^([A-Z][A-Z0-9_]{2,}=|[\w.-]+\.(toml|json|ya?ml|ini|env)|\s*[-\w]+\s*:)")
ADVISORY_PR_RE = re.compile(r"(?i)(pull request|reviewer|requested changes|unresolved|approve|merge blocker|ci failed)")
ANSI_RESET = "\033[0m"
ANSI_MUTED = "\033[2m"
ANSI_BOLD = "\033[1m"
ANSI_CYAN = "\033[36m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_RED = "\033[31m"
ANSI_BLUE = "\033[34m"
ANSI_MAGENTA = "\033[35m"


def capture_enabled(env: dict[str, str] | None = None) -> bool:
    env = env or os.environ
    return env.get("CODEX_TELEMETRY_CAPTURE") == "1"


def echo_enabled(env: dict[str, str] | None = None) -> bool:
    env = env or os.environ
    return env.get("CODEX_TELEMETRY_ECHO") == "1"


def color_echo_enabled(stdout: TextIO, env: dict[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    override = env.get("CODEX_TELEMETRY_COLOR")
    if override == "1":
        return True
    if override == "0" or "NO_COLOR" in env:
        return False
    return bool(getattr(stdout, "isatty", lambda: False)())


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


def env_int(env: dict[str, str], name: str, default: int) -> int:
    try:
        return int(env.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def default_rotate_bytes(env: dict[str, str] | None = None) -> int:
    env = env or os.environ
    return max(0, env_int(env, "CODEX_TELEMETRY_ROTATE_BYTES", DEFAULT_TELEMETRY_ROTATE_BYTES))


def default_retention_days(env: dict[str, str] | None = None) -> int:
    env = env or os.environ
    return max(0, env_int(env, "CODEX_TELEMETRY_RETENTION_DAYS", DEFAULT_TELEMETRY_RETENTION_DAYS))


def default_rotation_enabled(env: dict[str, str] | None = None) -> bool:
    env = env or os.environ
    return env.get("CODEX_TELEMETRY_ROTATION", "1") != "0"


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
    if tool_name.startswith(("local_", "agy_", "gemini_")):
        return "local_mcp_output"
    return "unknown"


def source_ref(tool_name: str, call_id: str, arguments: Any) -> str:
    _ = arguments
    return f"{tool_name}:{call_id}"


def infer_advisory_category(tool_name: str, output_text: str) -> str:
    if tool_name in {"shell", "exec_command"} and ADVISORY_LOG_RE.search(output_text):
        return "logs"
    if ADVISORY_DIFF_RE.search(output_text):
        return "diff"
    if ADVISORY_LOG_RE.search(output_text):
        return "logs"
    if ADVISORY_PR_RE.search(output_text):
        return "pr_thread"
    if ADVISORY_CONFIG_RE.search(output_text):
        return "config"
    lines = [line.strip() for line in output_text.splitlines() if line.strip()]
    path_like = sum(1 for line in lines if ADVISORY_REPO_MAP_RE.search(line))
    if lines and path_like / max(1, len(lines)) >= 0.50:
        return "repo_map"
    return "mixed_context"


def suggested_gemini_reducer(category: str) -> str:
    return {
        "diff": "gemini_compress_diff",
        "logs": "gemini_debug_digest",
        "repo_map": "gemini_repo_map_digest",
        "config": "gemini_config_surface_digest",
        "pr_thread": "gemini_pr_thread_digest",
        "mixed_context": "gemini_context_pack",
    }.get(category, "gemini_context_pack")


def post_turn_advisory(tool_name: str, output_text: str, token_estimate: int) -> dict[str, Any] | None:
    if token_estimate < POST_TURN_ADVISORY_MIN_TOKENS:
        return None
    category = infer_advisory_category(tool_name, output_text)
    return {
        "post_turn_advisory": "should_compress_next_time",
        "advisory_category": category,
        "advisory_raw_tokens_est": token_estimate,
        "advisory_route_decision": "gemini_recommended",
        "advisory_reducer": suggested_gemini_reducer(category),
    }


def rate_reset_at(observed: str, seconds: Any) -> str | None:
    observed_dt = parse_timestamp(observed)
    if observed_dt is None or not isinstance(seconds, (int, float)):
        return None
    return (observed_dt + timedelta(seconds=float(seconds))).isoformat()


def usage_signature(info: dict[str, Any]) -> str:
    return value_hash(info.get("total_token_usage") or {})


def clamp_pct(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return round(max(0.0, min(100.0, float(value))), 3)


def first_number(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, (int, float)):
            return float(value)
    return None


def mapping_get(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else None


def normalize_quota(payload: Any) -> tuple[float | None, str]:
    """
    Normalize quota payloads without inverting active usage.

    `used_percent` and equivalent names are already terminal-style usage.
    Only fields explicitly named remaining/balance are converted to used pct.
    """
    if isinstance(payload, (int, float)):
        return clamp_pct(payload), "bare_percent"
    if isinstance(payload, list):
        values = [normalize_quota(item)[0] for item in payload]
        values = [value for value in values if value is not None]
        return (max(values), "quota_array") if values else (None, "missing")
    if not isinstance(payload, dict):
        return None, "missing"

    used = first_number(
        payload.get("used_percent"),
        payload.get("usage_percent"),
        payload.get("used_pct"),
        payload.get("usage_pct"),
    )
    if used is not None:
        return clamp_pct(used), "used_percent"

    remaining = first_number(
        payload.get("remaining_percent"),
        payload.get("remaining_pct"),
    )
    if remaining is not None:
        return clamp_pct(100.0 - remaining), "remaining_percent"

    balance_pct = first_number(payload.get("balance_percent"), payload.get("balance_pct"))
    if balance_pct is not None:
        return clamp_pct(100.0 - balance_pct), "balance_percent"

    balance = payload.get("balance")
    if isinstance(balance, (int, float)):
        return clamp_pct(100.0 - float(balance)), "balance_percent"
    if isinstance(balance, (dict, list)):
        value, source = normalize_quota(balance)
        if value is None:
            return None, "missing"
        if source in {"used_percent", "bare_percent"}:
            return clamp_pct(100.0 - value), "balance_object"
        return value, "balance_object"

    return None, "missing"


def requested_provider_for_record(record: dict[str, Any]) -> str:
    provider = record.get("requested_provider")
    if isinstance(provider, str) and provider:
        return provider
    model = record.get("model")
    if not isinstance(model, str) or not model:
        return "unknown"
    if model.startswith("antigravity/"):
        return "agy"
    return "local"


def actual_provider_for_record(record: dict[str, Any]) -> str:
    requested_provider = requested_provider_for_record(record)
    output = stable_text(record.get("local_output", "")).lstrip()
    if requested_provider == "agy" and output.startswith(AGY_FALLBACK_PREFIXES):
        return "local"
    return requested_provider


def int_field(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def saved_tokens_for_record(record: dict[str, Any]) -> int:
    token_estimates = record.get("token_estimates") or {}
    model = record.get("model")
    model_text = model.lower() if isinstance(model, str) else ""
    if model_text.startswith("antigravity/") and "gemini" in model_text:
        for value in (
            record.get("gemini_saved_tokens_est"),
            token_estimates.get("gemini_saved"),
        ):
            if value is not None:
                return max(0, int_field(value))
    return int_field(token_estimates.get("context_reduction"))


def mcp_saved_breakdown(sources: list[dict[str, Any]]) -> dict[str, int]:
    local_saved = 0
    agy_saved = 0
    for source in sources:
        saved = int(source.get("raw_context_avoided") or 0)
        provider = source.get("mcp_provider")
        if provider == "agy":
            agy_saved += saved
        elif provider in {"local", None, ""}:
            local_saved += saved
    return {
        "local": local_saved,
        "agy": agy_saved,
        "total": local_saved + agy_saved,
    }


def mcp_source_count_breakdown(sources: list[dict[str, Any]]) -> dict[str, int]:
    local_count = 0
    agy_count = 0
    for source in sources:
        provider = source.get("mcp_provider")
        if provider == "agy":
            agy_count += 1
        elif provider in {"local", None, ""}:
            local_count += 1
    return {
        "local": local_count,
        "agy": agy_count,
        "total": local_count + agy_count,
    }


def cache_pressure_level(turn_input_tokens: int, turn_cached_input_tokens: int) -> str:
    if turn_input_tokens <= 0 or turn_cached_input_tokens <= 0:
        return "none"
    ratio = turn_cached_input_tokens / turn_input_tokens
    if turn_cached_input_tokens >= 250_000:
        return "high"
    if ratio >= 0.80 and turn_input_tokens >= 10_000:
        return "high"
    if turn_cached_input_tokens >= 100_000 or (ratio >= 0.60 and turn_input_tokens >= 10_000):
        return "medium"
    return "low"


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
    mcp_provider: str = "unknown"
    requested_provider: str = "unknown"
    model: str | None = None
    confidence: str = "unknown"
    task_id: str | None = None
    post_turn_advisory: str | None = None
    advisory_category: str | None = None
    advisory_raw_tokens_est: int = 0
    advisory_route_decision: str | None = None
    advisory_reducer: str | None = None


@dataclass
class ProcessResult:
    events: list[dict[str, Any]] = field(default_factory=list)
    turns: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)

    def extend(self, other: "ProcessResult") -> None:
        self.events.extend(other.events)
        self.turns.extend(other.turns)
        self.sources.extend(other.sources)


class TelemetryStorage:
    def __init__(
        self,
        telemetry_dir: Path,
        *,
        rotate_bytes: int = DEFAULT_TELEMETRY_ROTATE_BYTES,
        retention_days: int = DEFAULT_TELEMETRY_RETENTION_DAYS,
        rotation_enabled: bool = True,
    ):
        self.telemetry_dir = telemetry_dir
        self.rotate_bytes = max(0, rotate_bytes)
        self.retention_days = max(0, retention_days)
        self.rotation_enabled = rotation_enabled
        self.state_path = self.telemetry_dir / "state.sqlite3"
        self._connection: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            self.telemetry_dir.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self.state_path)
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS turn_keys (
                    session_file TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    usage_signature TEXT NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    PRIMARY KEY (session_file, thread_id, turn_id, usage_signature)
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rotations (
                    rotation_id TEXT PRIMARY KEY,
                    rotated_at TEXT NOT NULL,
                    trigger_name TEXT NOT NULL
                )
                """
            )
            self._connection.commit()
        return self._connection

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
        new_turns, accepted_turn_keys = self._dedupe_turns(result.turns)
        self.append("turns.jsonl", new_turns)
        if accepted_turn_keys:
            self.append(
                "sources.jsonl",
                [
                    source
                    for source in result.sources
                    if self._source_turn_key(source) in accepted_turn_keys
                ],
            )
        self.rotate_if_needed()
        self.cleanup_archives()

    def compact_now(self) -> None:
        self.rotate_if_needed()
        self.cleanup_archives()

    def rotate_if_needed(self) -> bool:
        if not self.rotation_enabled or self.rotate_bytes <= 0:
            return False
        for name in TELEMETRY_LEDGER_FILES:
            path = self.telemetry_dir / name
            if path.exists() and path.stat().st_size >= self.rotate_bytes:
                self._rotate_active_ledgers(name)
                return True
        return False

    def cleanup_archives(self) -> None:
        if self.retention_days <= 0:
            return
        archive_root = self.telemetry_dir / "archive"
        if not archive_root.exists():
            return
        cutoff = time.time() - (self.retention_days * 24 * 60 * 60)
        for path in archive_root.glob("*/*.jsonl.gz"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except FileNotFoundError:
                continue

    def _rotate_active_ledgers(self, trigger_name: str) -> None:
        rotation_dt = datetime.now(timezone.utc)
        rotation_id = rotation_dt.strftime("%Y-%m-%dT%H-%M-%S.%fZ")
        archive_dir = self.telemetry_dir / "archive" / rotation_dt.strftime("%Y-%m-%d")
        archive_dir.mkdir(parents=True, exist_ok=True)

        rotated_any = False
        for name in TELEMETRY_LEDGER_FILES:
            active_path = self.telemetry_dir / name
            if not active_path.exists() or active_path.stat().st_size == 0:
                active_path.parent.mkdir(parents=True, exist_ok=True)
                active_path.touch(exist_ok=True)
                continue
            archive_path = self._unique_archive_path(archive_dir / f"{rotation_id}.{name}.gz")
            with active_path.open("rb") as source, gzip.open(archive_path, "wb") as target:
                shutil.copyfileobj(source, target)
            active_path.write_text("", encoding="utf-8")
            rotated_any = True

        if rotated_any:
            conn = self._connect()
            conn.execute(
                "INSERT OR IGNORE INTO rotations (rotation_id, rotated_at, trigger_name) VALUES (?, ?, ?)",
                (rotation_id, rotation_dt.isoformat(), trigger_name),
            )
            conn.commit()

    @staticmethod
    def _unique_archive_path(path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.name
        counter = 1
        while True:
            candidate = path.with_name(f"{stem}.{counter}")
            if not candidate.exists():
                return candidate
            counter += 1

    @staticmethod
    def _turn_key(row: dict[str, Any]) -> tuple[str, str, str, str] | None:
        signature = row.get("usage_signature")
        if not signature:
            return None
        return (
            str(row.get("session_file")),
            str(row.get("thread_id")),
            str(row.get("turn_id")),
            str(signature),
        )

    @staticmethod
    def _source_turn_key(row: dict[str, Any]) -> tuple[str, str, str, str] | None:
        signature = row.get("turn_usage_signature")
        if not signature:
            return None
        return (
            str(row.get("session_file")),
            str(row.get("thread_id")),
            str(row.get("turn_id")),
            str(signature),
        )

    def _dedupe_turns(
        self, rows: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], set[tuple[str, str, str, str]]]:
        accepted: set[tuple[str, str, str, str]] = set()
        output: list[dict[str, Any]] = []
        conn = self._connect()
        for row in rows:
            key = self._turn_key(row)
            if key is None:
                output.append(row)
                continue
            if key in accepted:
                continue
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO turn_keys (
                    session_file,
                    thread_id,
                    turn_id,
                    usage_signature,
                    first_seen_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (*key, utc_now()),
            )
            if cursor.rowcount:
                accepted.add(key)
                output.append(row)
        conn.commit()
        return output, accepted


JsonlLedger = TelemetryStorage


class LocalLedgerIndex:
    def __init__(self, path: Path):
        self.path = path
        self.records: list[dict[str, Any]] = []
        self._last_fingerprint: tuple[int, int] | None = None
        self.refresh_if_changed()

    def _fingerprint(self) -> tuple[int, int] | None:
        try:
            stat = self.path.stat()
        except OSError:
            return None
        return (stat.st_mtime_ns, stat.st_size)

    def refresh_if_changed(self) -> bool:
        fingerprint = self._fingerprint()
        if fingerprint == self._last_fingerprint:
            return False
        self._last_fingerprint = fingerprint
        self.records = self._read(self.path) if fingerprint is not None else []
        return True

    @staticmethod
    def _read(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        try:
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
        except OSError:
            return []
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
        source.raw_context_avoided = saved_tokens_for_record(record)
        source.model = str(record["model"]) if record.get("model") is not None else None
        source.requested_provider = requested_provider_for_record(record)
        source.mcp_provider = actual_provider_for_record(record)
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
        self.seen_usage_keys: set[tuple[str, str, str, str]] = set()

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
            usage_key = (str(self.session_file), thread_id, turn_id, current_signature)
            if usage_key in self.seen_usage_keys:
                self.last_usage_signature = current_signature
                return result
            self.seen_usage_keys.add(usage_key)
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
        token_estimate = estimate_tokens(output_text)
        advisory = post_turn_advisory(tool_name, output_text, token_estimate) or {}
        pending = PendingSource(
            source_kind=source_kind_for_tool(tool_name),
            source_ref=source_ref(tool_name, call_id, arguments),
            token_estimate=token_estimate,
            output_hash=text_hash(output_text),
            call_id=call_id,
            tool_name=tool_name,
            timestamp=timestamp,
            event_seq=event_seq,
            post_turn_advisory=advisory.get("post_turn_advisory"),
            advisory_category=advisory.get("advisory_category"),
            advisory_raw_tokens_est=int(advisory.get("advisory_raw_tokens_est") or 0),
            advisory_route_decision=advisory.get("advisory_route_decision"),
            advisory_reducer=advisory.get("advisory_reducer"),
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
        turn_input_tokens = int(delta.get("input_tokens") or 0)
        cached_input_tokens = int(total.get("cached_input_tokens") or 0)
        turn_cached_input_tokens = int(delta.get("cached_input_tokens") or 0)
        context_used_pct = (
            round((turn_input_tokens / int(context_window)) * 100, 3)
            if isinstance(context_window, int) and context_window > 0
            else None
        )
        rate_limits = payload.get("rate_limits") if isinstance(payload.get("rate_limits"), dict) else {}
        primary = rate_limits.get("primary") or {}
        secondary = rate_limits.get("secondary") or {}
        primary_used, primary_source = normalize_quota(primary)
        secondary_used, secondary_source = normalize_quota(secondary)
        cache_ratio = round(turn_cached_input_tokens / turn_input_tokens, 3) if turn_input_tokens else 0.0
        pressure_level = cache_pressure_level(turn_input_tokens, turn_cached_input_tokens)
        cache_warnings = []
        if pressure_level == "high":
            cache_warnings.append("high_cached_input_turn")
        mcp_local_saved = sum(
            source.raw_context_avoided for source in self.pending_sources if source.mcp_provider == "local"
        )
        mcp_agy_saved = sum(source.raw_context_avoided for source in self.pending_sources if source.mcp_provider == "agy")
        mcp_local_count = sum(1 for source in self.pending_sources if source.mcp_provider == "local")
        mcp_agy_count = sum(1 for source in self.pending_sources if source.mcp_provider == "agy")

        return {
            "record_type": "turn",
            "observed_at": timestamp,
            "session_file": str(self.session_file),
            "thread_id": thread_id,
            "turn_id": turn_id,
            "event_seq": event_seq,
            "usage_signature": usage_signature(info),
            "model": self.context.get("model"),
            "reasoning_effort": self.context.get("effort") or self.context.get("reasoning_effort"),
            "mode": self.context.get("mode") or self.context.get("summary"),
            "cwd": self.context.get("cwd"),
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_input_tokens,
            "output_tokens": int(total.get("output_tokens") or 0),
            "reasoning_output_tokens": int(total.get("reasoning_output_tokens") or 0),
            "total_tokens": int(total.get("total_tokens") or 0),
            "turn_input_tokens": turn_input_tokens,
            "turn_cached_input_tokens": turn_cached_input_tokens,
            "turn_output_tokens": int(delta.get("output_tokens") or 0),
            "turn_reasoning_output_tokens": int(delta.get("reasoning_output_tokens") or 0),
            "turn_total_tokens": int(delta.get("total_tokens") or 0),
            "delta_input_tokens": turn_input_tokens,
            "delta_cached_input_tokens": turn_cached_input_tokens,
            "delta_output_tokens": int(delta.get("output_tokens") or 0),
            "delta_reasoning_output_tokens": int(delta.get("reasoning_output_tokens") or 0),
            "delta_total_tokens": int(delta.get("total_tokens") or 0),
            "context_window": context_window,
            "context_used_pct": context_used_pct,
            "turn_context_used_pct": context_used_pct,
            "cached_input_tokens_total": cached_input_tokens,
            "cached_input_tokens_turn": turn_cached_input_tokens,
            "cache_ratio_turn": cache_ratio,
            "cache_pressure_level": pressure_level,
            "cache_warnings": cache_warnings,
            "mcp_local_saved_tokens_est": mcp_local_saved,
            "mcp_agy_saved_tokens_est": mcp_agy_saved,
            "mcp_total_saved_tokens_est": mcp_local_saved + mcp_agy_saved,
            "mcp_local_source_count": mcp_local_count,
            "mcp_agy_source_count": mcp_agy_count,
            "mcp_total_source_count": mcp_local_count + mcp_agy_count,
            "primary_rate_used_pct": primary_used,
            "primary_rate_source": primary_source,
            "primary_rate_payload_hash": value_hash(primary),
            "primary_rate_window_minutes": mapping_get(primary, "window_minutes"),
            "primary_rate_resets_in_seconds": mapping_get(primary, "resets_in_seconds"),
            "primary_rate_reset_at": rate_reset_at(timestamp, mapping_get(primary, "resets_in_seconds")),
            "secondary_rate_used_pct": secondary_used,
            "secondary_rate_source": secondary_source,
            "secondary_rate_payload_hash": value_hash(secondary),
            "secondary_rate_window_minutes": mapping_get(secondary, "window_minutes"),
            "secondary_rate_resets_in_seconds": mapping_get(secondary, "resets_in_seconds"),
            "secondary_rate_reset_at": rate_reset_at(timestamp, mapping_get(secondary, "resets_in_seconds")),
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
                    "turn_usage_signature": turn.get("usage_signature"),
                    "source_event_seq": source.event_seq,
                    "source_kind": source.source_kind,
                    "source_ref": source.source_ref,
                    "call_id": source.call_id,
                    "task_id": source.task_id,
                    "tool_name": source.tool_name,
                    "mcp_provider": source.mcp_provider,
                    "requested_provider": source.requested_provider,
                    "model": source.model,
                    "token_estimate": source.token_estimate,
                    "raw_context_avoided": source.raw_context_avoided,
                    "confidence": source.confidence,
                    "source_hash": source.output_hash,
                    "post_turn_advisory": source.post_turn_advisory,
                    "advisory_category": source.advisory_category,
                    "advisory_raw_tokens_est": source.advisory_raw_tokens_est,
                    "advisory_route_decision": source.advisory_route_decision,
                    "advisory_reducer": source.advisory_reducer,
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


def format_pct(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:.0f}%"


def ansi(text: str, color: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{color}{text}{ANSI_RESET}"


def quota_color(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return ANSI_MUTED
    if value >= 90:
        return ANSI_RED
    if value >= 70:
        return ANSI_YELLOW
    return ANSI_GREEN


def echo_saved_breakdown(turn: dict[str, Any], sources: list[dict[str, Any]]) -> dict[str, int]:
    saved = mcp_saved_breakdown(sources)
    if sources:
        return saved
    return {
        "local": int(turn.get("mcp_local_saved_tokens_est") or 0),
        "agy": int(turn.get("mcp_agy_saved_tokens_est") or 0),
        "total": int(turn.get("mcp_total_saved_tokens_est") or 0),
    }


def styled_value(text: str, style: str, color: bool) -> str:
    return ansi(text, ANSI_MUTED if text == "n/a" else style, color)


def format_echo_line(turn: dict[str, Any], sources: list[dict[str, Any]], *, color: bool = False) -> str:
    saved = echo_saved_breakdown(turn, sources)
    ctx = turn.get("context_used_pct")
    primary = turn.get("primary_rate_used_pct")
    secondary = turn.get("secondary_rate_used_pct")
    cache_pressure = turn.get("cache_pressure_level")
    if not isinstance(cache_pressure, str):
        cache_pressure = cache_pressure_level(
            int(turn.get("turn_input_tokens") or 0),
            int(turn.get("turn_cached_input_tokens") or 0),
        )
    cache_style = ANSI_YELLOW if cache_pressure == "high" else ANSI_CYAN
    cached_count = format_count(turn.get("turn_cached_input_tokens"))
    return (
        f"{ansi('codex', ANSI_MUTED, color)}  "
        f"in {styled_value(format_count(turn.get('turn_input_tokens')), '', color)} "
        f"(+{styled_value(cached_count, cache_style, color)} cached)  "
        f"out {styled_value(format_count(turn.get('turn_output_tokens')), '', color)}  "
        f"reason {styled_value(format_count(turn.get('turn_reasoning_output_tokens')), '', color)}  "
        f"ctx {styled_value(format_pct(ctx), '', color)}  "
        f"5h {ansi(format_pct(primary), quota_color(primary), color)}  "
        f"weekly {ansi(format_pct(secondary), quota_color(secondary), color)}  "
        f"saved local {styled_value(format_count(saved['local']), ANSI_BLUE, color)}  "
        f"agy {styled_value(format_count(saved['agy']), ANSI_MAGENTA, color)}  "
        f"total {styled_value(format_count(saved['total']), ANSI_BOLD, color)}"
    )


def emit_echo(
    result: ProcessResult,
    stdout: TextIO,
    *,
    env: dict[str, str] | None = None,
    color: bool | None = None,
) -> None:
    if not result.turns:
        return
    use_color = color_echo_enabled(stdout, env) if color is None else color
    for turn in result.turns:
        turn_sources = [
            source
            for source in result.sources
            if source.get("thread_id") == turn.get("thread_id")
            and source.get("turn_id") == turn.get("turn_id")
            and source.get("turn_event_seq") == turn.get("event_seq")
        ]
        print(format_echo_line(turn, turn_sources, color=use_color), file=stdout, flush=True)


def build_telemetry_storage(
    telemetry_dir: Path,
    *,
    env: dict[str, str],
    rotate_bytes: int | None = None,
    retention_days: int | None = None,
    rotation_enabled: bool | None = None,
) -> TelemetryStorage:
    return TelemetryStorage(
        telemetry_dir,
        rotate_bytes=default_rotate_bytes(env) if rotate_bytes is None else rotate_bytes,
        retention_days=default_retention_days(env) if retention_days is None else retention_days,
        rotation_enabled=default_rotation_enabled(env) if rotation_enabled is None else rotation_enabled,
    )


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
    rotate_bytes: int | None = None,
    retention_days: int | None = None,
    rotation_enabled: bool | None = None,
) -> ProcessResult:
    env = env or os.environ
    stdout = stdout or sys.stdout
    if not capture_enabled(env):
        return ProcessResult()

    telemetry_dir = telemetry_dir or default_telemetry_dir(env)
    local_mcp_ledger_path = local_mcp_ledger_path or Path(DEFAULT_LOCAL_MCP_LEDGER)
    local_ledger = LocalLedgerIndex(local_mcp_ledger_path)
    files = session_files if session_files is not None else session_files_from_glob(sessions_glob or default_sessions_glob(env))
    ledger = build_telemetry_storage(
        telemetry_dir,
        env=env,
        rotate_bytes=rotate_bytes,
        retention_days=retention_days,
        rotation_enabled=rotation_enabled,
    )
    combined = ProcessResult()
    for session_file in files:
        result = process_session_file(session_file, local_ledger)
        ledger.append_result(result)
        combined.extend(result)
        should_echo = echo if echo is not None else echo_enabled(env)
        if should_echo:
            emit_echo(result, stdout, env=env)
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
    rotate_bytes: int | None = None,
    retention_days: int | None = None,
    rotation_enabled: bool | None = None,
) -> None:
    env = env or os.environ
    if not capture_enabled(env):
        return

    ledger = build_telemetry_storage(
        telemetry_dir,
        env=env,
        rotate_bytes=rotate_bytes,
        retention_days=retention_days,
        rotation_enabled=rotation_enabled,
    )
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

            local_ledger.refresh_if_changed()
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
                emit_echo(result, stdout, env=env)

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
    parser.add_argument("--rotate-bytes", type=int, default=None, help="Rotate active telemetry ledgers at this byte size; defaults to CODEX_TELEMETRY_ROTATE_BYTES or 25000000.")
    parser.add_argument("--retention-days", type=int, default=None, help="Delete compressed telemetry archives older than this many days; defaults to CODEX_TELEMETRY_RETENTION_DAYS or 90. Use 0 to keep archives.")
    parser.add_argument("--no-rotation", action="store_true", help="Disable active telemetry ledger rotation.")
    parser.add_argument("--compact-now", action="store_true", help="Rotate oversized active telemetry ledgers, run archive retention cleanup, and exit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, stdout: TextIO | None = None) -> int:
    args = parse_args(argv)
    stdout = stdout or sys.stdout
    env = os.environ
    telemetry_dir = Path(args.telemetry_dir) if args.telemetry_dir else default_telemetry_dir(env)
    sessions_glob = args.sessions_glob or default_sessions_glob(env)
    session_files = [Path(path) for path in args.session_file] if args.session_file else None
    echo = args.echo or echo_enabled(env)
    rotate_bytes = args.rotate_bytes
    retention_days = args.retention_days
    rotation_enabled = False if args.no_rotation else None

    if args.compact_now:
        build_telemetry_storage(
            telemetry_dir,
            env=env,
            rotate_bytes=rotate_bytes,
            retention_days=retention_days,
            rotation_enabled=rotation_enabled,
        ).compact_now()
        return 0

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
            rotate_bytes=rotate_bytes,
            retention_days=retention_days,
            rotation_enabled=rotation_enabled,
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
        rotate_bytes=rotate_bytes,
        retention_days=retention_days,
        rotation_enabled=rotation_enabled,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
