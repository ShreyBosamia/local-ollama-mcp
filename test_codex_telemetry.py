from __future__ import annotations

import gzip
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

import codex_telemetry


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_gzip_jsonl(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_all_telemetry_rows(root: Path, name: str) -> list[dict]:
    rows: list[dict] = []
    active = root / "telemetry" / name
    if active.exists():
        rows.extend(read_jsonl(active))
    for archive in sorted((root / "telemetry" / "archive").glob(f"*/*.{name}.gz")):
        rows.extend(read_gzip_jsonl(archive))
    return rows


class TtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


def token_count(
    *,
    timestamp: str = "2026-05-22T17:00:02.000Z",
    input_tokens: int = 1000,
    cached_input_tokens: int = 512,
    output_tokens: int = 70,
    reasoning_output_tokens: int = 10,
    total_tokens: int = 1070,
    delta_input_tokens: int = 250,
    delta_cached_input_tokens: int = 128,
    delta_output_tokens: int = 12,
    delta_reasoning_output_tokens: int = 4,
    delta_total_tokens: int = 262,
) -> dict:
    return {
        "timestamp": timestamp,
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "total_token_usage": {
                    "input_tokens": input_tokens,
                    "cached_input_tokens": cached_input_tokens,
                    "output_tokens": output_tokens,
                    "reasoning_output_tokens": reasoning_output_tokens,
                    "total_tokens": total_tokens,
                },
                "last_token_usage": {
                    "input_tokens": delta_input_tokens,
                    "cached_input_tokens": delta_cached_input_tokens,
                    "output_tokens": delta_output_tokens,
                    "reasoning_output_tokens": delta_reasoning_output_tokens,
                    "total_tokens": delta_total_tokens,
                },
                "model_context_window": 2000,
            },
            "rate_limits": {
                "primary": {
                    "used_percent": 34.0,
                    "window_minutes": 299,
                    "resets_in_seconds": 60,
                },
                "secondary": {
                    "used_percent": 69.0,
                    "window_minutes": 10079,
                    "resets_in_seconds": 120,
                },
            },
        },
    }


class CodexTelemetryTests(unittest.TestCase):
    def test_no_telemetry_when_capture_is_unset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "rollout-2026-05-22T10-00-00-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
            write_jsonl(session, [token_count()])

            result = codex_telemetry.run_once(
                session_files=[session],
                telemetry_dir=root / "telemetry",
                env={},
            )

            self.assertEqual(result.turns, [])
            self.assertFalse((root / "telemetry").exists())

    def test_turn_summary_uses_exact_total_and_last_usage_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "rollout-2026-05-22T10-00-00-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
            write_jsonl(
                session,
                [
                    {
                        "timestamp": "2026-05-22T17:00:01.000Z",
                        "type": "turn_context",
                        "payload": {
                            "thread_id": "thread-1",
                            "turn_id": "turn-1",
                            "cwd": "/repo",
                            "model": "gpt-5-codex",
                            "effort": "high",
                            "mode": "default",
                        },
                    },
                    token_count(
                        input_tokens=109000,
                        cached_input_tokens=100000,
                        output_tokens=221,
                        reasoning_output_tokens=18,
                        total_tokens=109221,
                        delta_input_tokens=9000,
                        delta_cached_input_tokens=8000,
                        delta_output_tokens=23,
                        delta_reasoning_output_tokens=5,
                        delta_total_tokens=9023,
                    ),
                ],
            )

            codex_telemetry.run_once(
                session_files=[session],
                telemetry_dir=root / "telemetry",
                env={"CODEX_TELEMETRY_CAPTURE": "1"},
            )
            turns = read_jsonl(root / "telemetry" / "turns.jsonl")

            self.assertEqual(len(turns), 1)
            turn = turns[0]
            self.assertEqual(turn["thread_id"], "thread-1")
            self.assertEqual(turn["turn_id"], "turn-1")
            self.assertEqual(turn["input_tokens"], 109000)
            self.assertEqual(turn["cached_input_tokens"], 100000)
            self.assertEqual(turn["total_tokens"], 109221)
            self.assertEqual(turn["turn_input_tokens"], 9000)
            self.assertEqual(turn["turn_cached_input_tokens"], 8000)
            self.assertEqual(turn["delta_cached_input_tokens"], 8000)
            self.assertEqual(turn["delta_reasoning_output_tokens"], 5)
            self.assertEqual(turn["context_window"], 2000)
            self.assertEqual(turn["context_used_pct"], 450.0)
            self.assertEqual(turn["turn_context_used_pct"], 450.0)
            self.assertNotEqual(turn["context_used_pct"], 5450.0)
            self.assertEqual(turn["cached_input_tokens_total"], 100000)
            self.assertEqual(turn["cached_input_tokens_turn"], 8000)
            self.assertEqual(turn["cache_ratio_turn"], 0.889)
            self.assertEqual(turn["cache_pressure_level"], "low")
            self.assertEqual(turn["primary_rate_used_pct"], 34.0)
            self.assertEqual(turn["primary_rate_source"], "used_percent")
            self.assertEqual(turn["secondary_rate_used_pct"], 69.0)
            self.assertEqual(turn["secondary_rate_source"], "used_percent")

    def test_quota_normalization_does_not_invert_used_percent(self) -> None:
        self.assertEqual(codex_telemetry.normalize_quota({"used_percent": 14}), (14.0, "used_percent"))
        self.assertEqual(codex_telemetry.normalize_quota({"remaining_percent": 86}), (14.0, "remaining_percent"))
        self.assertEqual(codex_telemetry.normalize_quota({"balance_percent": 1}), (99.0, "balance_percent"))
        self.assertEqual(codex_telemetry.normalize_quota({"balance": {"remaining_percent": 77}}), (23.0, "balance_object"))
        self.assertEqual(codex_telemetry.normalize_quota([{"remaining_percent": 77}, {"used_percent": 14}]), (23.0, "quota_array"))
        self.assertEqual(codex_telemetry.normalize_quota([{"remaining_percent": 86}, {"used_percent": 14}]), (14.0, "quota_array"))
        self.assertEqual(codex_telemetry.normalize_quota([{"remaining_percent": 10}, {"used_percent": 14}]), (90.0, "quota_array"))
        self.assertEqual(codex_telemetry.normalize_quota({}), (None, "missing"))

    def test_turn_deltas_do_not_compound_from_growing_totals(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "rollout-2026-05-22T10-00-00-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
            write_jsonl(
                session,
                [
                    {
                        "timestamp": "2026-05-22T17:00:01.000Z",
                        "type": "turn_context",
                        "payload": {"thread_id": "thread-1", "turn_id": "turn-1"},
                    },
                    token_count(
                        input_tokens=1000,
                        cached_input_tokens=100,
                        output_tokens=50,
                        total_tokens=1050,
                        delta_input_tokens=120,
                        delta_cached_input_tokens=20,
                        delta_output_tokens=7,
                        delta_total_tokens=127,
                    ),
                    token_count(
                        timestamp="2026-05-22T17:00:03.000Z",
                        input_tokens=1500,
                        cached_input_tokens=200,
                        output_tokens=70,
                        total_tokens=1570,
                        delta_input_tokens=120,
                        delta_cached_input_tokens=20,
                        delta_output_tokens=7,
                        delta_total_tokens=127,
                    ),
                ],
            )

            codex_telemetry.run_once(
                session_files=[session],
                telemetry_dir=root / "telemetry",
                env={"CODEX_TELEMETRY_CAPTURE": "1"},
            )
            turns = read_jsonl(root / "telemetry" / "turns.jsonl")

            self.assertEqual(len(turns), 2)
            self.assertEqual([turn["input_tokens"] for turn in turns], [1000, 1500])
            self.assertEqual([turn["turn_input_tokens"] for turn in turns], [120, 120])
            self.assertEqual([turn["delta_input_tokens"] for turn in turns], [120, 120])
            self.assertEqual([turn["turn_total_tokens"] for turn in turns], [127, 127])
            self.assertNotEqual(turns[1]["turn_input_tokens"], 500)

    def test_quota_echo_handles_active_usage_and_reset_state(self) -> None:
        base_turn = {
            "turn_input_tokens": 1000,
            "turn_cached_input_tokens": 200,
            "turn_output_tokens": 20,
            "turn_reasoning_output_tokens": 5,
            "context_used_pct": 10.0,
            "context_window": 10000,
            "primary_rate_used_pct": 14.0,
            "secondary_rate_used_pct": 23.0,
        }
        self.assertIn("5h 14%  weekly 23%", codex_telemetry.format_echo_line(base_turn, []))
        reset_turn = dict(base_turn)
        reset_turn["primary_rate_used_pct"] = 99.0
        self.assertIn("5h 99%  weekly 23%", codex_telemetry.format_echo_line(reset_turn, []))

    def test_local_mcp_attribution_skips_duplicate_token_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "rollout-2026-05-22T10-00-00-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
            local_ledger = root / "ledger.jsonl"
            write_jsonl(
                local_ledger,
                [
                    {
                        "record_type": "tool_call",
                        "task_id": "task-123",
                        "timestamp": "2026-05-22T17:00:03.000Z",
                        "tool_name": "local_summarize",
                        "local_output": "summary text",
                        "token_estimates": {
                            "input": 200,
                            "local_output": 20,
                            "context_reduction": 180,
                        },
                    }
                ],
            )
            first_usage = token_count(
                timestamp="2026-05-22T17:00:02.000Z",
                input_tokens=1000,
                cached_input_tokens=512,
                output_tokens=70,
                total_tokens=1070,
            )
            duplicate_usage = token_count(
                timestamp="2026-05-22T17:00:04.000Z",
                input_tokens=1000,
                cached_input_tokens=512,
                output_tokens=70,
                total_tokens=1070,
            )
            next_usage = token_count(
                timestamp="2026-05-22T17:00:05.000Z",
                input_tokens=1300,
                cached_input_tokens=700,
                output_tokens=90,
                total_tokens=1390,
            )
            write_jsonl(
                session,
                [
                    {
                        "timestamp": "2026-05-22T17:00:01.000Z",
                        "type": "turn_context",
                        "payload": {"thread_id": "thread-1", "turn_id": "turn-1"},
                    },
                    first_usage,
                    {
                        "timestamp": "2026-05-22T17:00:03.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": "local_summarize",
                            "arguments": "{\"text\":\"large raw context\"}",
                            "call_id": "call-local",
                        },
                    },
                    {
                        "timestamp": "2026-05-22T17:00:03.100Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call_output",
                            "call_id": "call-local",
                            "output": json.dumps({"output": "summary text", "metadata": {"exit_code": 0}}),
                        },
                    },
                    duplicate_usage,
                    next_usage,
                ],
            )

            codex_telemetry.run_once(
                session_files=[session],
                telemetry_dir=root / "telemetry",
                local_mcp_ledger_path=local_ledger,
                env={"CODEX_TELEMETRY_CAPTURE": "1"},
            )
            sources = read_jsonl(root / "telemetry" / "sources.jsonl")
            turns = read_jsonl(root / "telemetry" / "turns.jsonl")

            self.assertEqual(len(turns), 2)
            self.assertEqual(len(sources), 1)
            source = sources[0]
            self.assertEqual(source["turn_event_seq"], 6)
            self.assertTrue(source["turn_usage_signature"])
            self.assertEqual(source["source_kind"], "local_mcp_output")
            self.assertEqual(source["call_id"], "call-local")
            self.assertEqual(source["task_id"], "task-123")
            self.assertEqual(source["tool_name"], "local_summarize")
            self.assertEqual(source["mcp_provider"], "unknown")
            self.assertEqual(source["requested_provider"], "unknown")
            self.assertIsNone(source["model"])
            self.assertEqual(source["token_estimate"], 2)
            self.assertEqual(source["raw_context_avoided"], 180)
            self.assertEqual(source["confidence"], "matched_hash")
            self.assertEqual(turns[1]["mcp_local_saved_tokens_est"], 0)
            self.assertEqual(turns[1]["mcp_agy_saved_tokens_est"], 0)
            self.assertEqual(turns[1]["mcp_total_saved_tokens_est"], 0)

    def test_mcp_saved_breakdown_splits_local_and_agy_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "rollout-2026-05-22T10-00-00-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
            local_ledger = root / "ledger.jsonl"
            write_jsonl(
                local_ledger,
                [
                    {
                        "record_type": "tool_call",
                        "task_id": "task-local",
                        "timestamp": "2026-05-22T17:00:02.000Z",
                        "tool_name": "local_summarize",
                        "model": "qwen3.5:9b",
                        "local_output": "local summary",
                        "token_estimates": {"input": 9000, "local_output": 800, "context_reduction": 8200},
                    },
                    {
                        "record_type": "tool_call",
                        "task_id": "task-agy",
                        "timestamp": "2026-05-22T17:00:03.000Z",
                        "tool_name": "agy_compress_diff",
                        "model": "antigravity/gemini-3.5-flash-high",
                        "local_output": "agy summary",
                        "token_estimates": {"input": 32000, "local_output": 1000, "context_reduction": 31000},
                    },
                ],
            )
            write_jsonl(
                session,
                [
                    {
                        "timestamp": "2026-05-22T17:00:01.000Z",
                        "type": "turn_context",
                        "payload": {"thread_id": "thread-1", "turn_id": "turn-1"},
                    },
                    {
                        "timestamp": "2026-05-22T17:00:02.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": "local_summarize",
                            "arguments": "{}",
                            "call_id": "call-local",
                        },
                    },
                    {
                        "timestamp": "2026-05-22T17:00:02.100Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call_output",
                            "call_id": "call-local",
                            "output": json.dumps({"output": "local summary"}),
                        },
                    },
                    {
                        "timestamp": "2026-05-22T17:00:03.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": "agy_compress_diff",
                            "arguments": "{}",
                            "call_id": "call-agy",
                        },
                    },
                    {
                        "timestamp": "2026-05-22T17:00:03.100Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call_output",
                            "call_id": "call-agy",
                            "output": json.dumps({"output": "agy summary"}),
                        },
                    },
                    token_count(timestamp="2026-05-22T17:00:04.000Z"),
                ],
            )

            codex_telemetry.run_once(
                session_files=[session],
                telemetry_dir=root / "telemetry",
                local_mcp_ledger_path=local_ledger,
                env={"CODEX_TELEMETRY_CAPTURE": "1"},
            )
            sources = read_jsonl(root / "telemetry" / "sources.jsonl")
            turns = read_jsonl(root / "telemetry" / "turns.jsonl")

            self.assertEqual([source["mcp_provider"] for source in sources], ["local", "agy"])
            self.assertEqual([source["requested_provider"] for source in sources], ["local", "agy"])
            self.assertEqual([source["model"] for source in sources], ["qwen3.5:9b", "antigravity/gemini-3.5-flash-high"])
            self.assertEqual([source["tool_name"] for source in sources], ["local_summarize", "agy_compress_diff"])
            self.assertEqual(turns[0]["mcp_local_saved_tokens_est"], 8200)
            self.assertEqual(turns[0]["mcp_agy_saved_tokens_est"], 31000)
            self.assertEqual(turns[0]["mcp_total_saved_tokens_est"], 39200)
            self.assertEqual(turns[0]["mcp_local_source_count"], 1)
            self.assertEqual(turns[0]["mcp_agy_source_count"], 1)
            self.assertEqual(turns[0]["mcp_total_source_count"], 2)
            self.assertIn("saved local 8.2K  agy 31K  total 39K", codex_telemetry.format_echo_line(turns[0], sources))

    def test_large_tool_output_records_post_turn_compression_advisory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "rollout-2026-05-22T10-00-00-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
            large_output = "\n".join(
                f"ERROR timeout in worker {index}: Traceback failed while draining queue"
                for index in range(900)
            )
            write_jsonl(
                session,
                [
                    {
                        "timestamp": "2026-05-22T17:00:01.000Z",
                        "type": "turn_context",
                        "payload": {"thread_id": "thread-1", "turn_id": "turn-1"},
                    },
                    {
                        "timestamp": "2026-05-22T17:00:02.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": "exec_command",
                            "arguments": "{}",
                            "call_id": "call-shell",
                        },
                    },
                    {
                        "timestamp": "2026-05-22T17:00:02.100Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call_output",
                            "call_id": "call-shell",
                            "output": json.dumps({"output": large_output}),
                        },
                    },
                    token_count(timestamp="2026-05-22T17:00:03.000Z"),
                ],
            )

            codex_telemetry.run_once(
                session_files=[session],
                telemetry_dir=root / "telemetry",
                env={"CODEX_TELEMETRY_CAPTURE": "1"},
            )
            turns = read_jsonl(root / "telemetry" / "turns.jsonl")
            sources = read_jsonl(root / "telemetry" / "sources.jsonl")

            self.assertEqual(len(turns), 1)
            self.assertEqual(len(sources), 1)
            source = sources[0]
            self.assertEqual(source["source_kind"], "shell_output")
            self.assertEqual(source["post_turn_advisory"], "should_compress_next_time")
            self.assertEqual(source["advisory_category"], "logs")
            self.assertEqual(source["advisory_route_decision"], "gemini_recommended")
            self.assertEqual(source["advisory_reducer"], "gemini_debug_digest")
            self.assertGreaterEqual(source["advisory_raw_tokens_est"], 4000)

    def test_gemini_saved_tokens_est_takes_precedence_over_context_reduction(self) -> None:
        record = {
            "model": "antigravity/gemini-3.5-flash-high",
            "gemini_saved_tokens_est": 24000,
            "token_estimates": {
                "context_reduction": 31000,
                "gemini_saved": 25000,
            },
        }

        self.assertEqual(codex_telemetry.saved_tokens_for_record(record), 24000)

    def test_older_gemini_ledger_rows_fall_back_to_context_reduction(self) -> None:
        record = {
            "model": "antigravity/gemini-3.5-flash-high",
            "token_estimates": {
                "input": 32000,
                "local_output": 1000,
                "context_reduction": 31000,
            },
        }

        self.assertEqual(codex_telemetry.saved_tokens_for_record(record), 31000)

    def test_local_ledger_index_refreshes_after_append(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            local_ledger = root / "ledger.jsonl"
            write_jsonl(local_ledger, [])
            index = codex_telemetry.LocalLedgerIndex(local_ledger)
            append_jsonl(
                local_ledger,
                {
                    "record_type": "tool_call",
                    "task_id": "task-gemini",
                    "timestamp": "2026-05-22T17:00:02.000Z",
                    "tool_name": "gemini_compress_diff",
                    "model": "antigravity/gemini-3.5-flash-high",
                    "local_output": "compressed diff",
                    "gemini_saved_tokens_est": 2622,
                    "token_estimates": {"context_reduction": 3000},
                },
            )

            self.assertTrue(index.refresh_if_changed())
            source = codex_telemetry.PendingSource(
                source_kind="local_mcp_output",
                source_ref="gemini_compress_diff:call-gemini",
                token_estimate=2,
                output_hash=codex_telemetry.text_hash("compressed diff"),
                call_id="call-gemini",
                tool_name="gemini_compress_diff",
                timestamp="2026-05-22T17:00:02.100Z",
                event_seq=3,
            )

            matched = index.match(source)

            self.assertEqual(matched.mcp_provider, "agy")
            self.assertEqual(matched.requested_provider, "agy")
            self.assertEqual(matched.model, "antigravity/gemini-3.5-flash-high")
            self.assertEqual(matched.raw_context_avoided, 2622)
            self.assertEqual(matched.task_id, "task-gemini")
            self.assertEqual(matched.confidence, "matched_hash")

    def test_live_watch_refreshes_ledger_before_tool_output_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "rollout-2026-05-22T10-00-00-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
            local_ledger = root / "ledger.jsonl"
            write_jsonl(local_ledger, [])
            ledger_index = codex_telemetry.LocalLedgerIndex(local_ledger)
            processor = codex_telemetry.SessionProcessor(session, ledger_index)
            result = codex_telemetry.ProcessResult()
            for event_seq, record in enumerate(
                [
                    {
                        "timestamp": "2026-05-22T17:00:01.000Z",
                        "type": "turn_context",
                        "payload": {"thread_id": "thread-1", "turn_id": "turn-1"},
                    },
                    {
                        "timestamp": "2026-05-22T17:00:02.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": "gemini_compress_diff",
                            "arguments": "{}",
                            "call_id": "call-gemini",
                        },
                    },
                ],
                start=1,
            ):
                result.extend(processor.process_line(json.dumps(record), event_seq))

            append_jsonl(
                local_ledger,
                {
                    "record_type": "tool_call",
                    "task_id": "task-gemini",
                    "timestamp": "2026-05-22T17:00:02.000Z",
                    "tool_name": "gemini_compress_diff",
                    "model": "antigravity/gemini-3.5-flash-high",
                    "local_output": "compressed diff",
                    "gemini_saved_tokens_est": 2622,
                    "token_estimates": {"context_reduction": 3000},
                },
            )
            ledger_index.refresh_if_changed()

            for event_seq, record in enumerate(
                [
                    {
                        "timestamp": "2026-05-22T17:00:02.100Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call_output",
                            "call_id": "call-gemini",
                            "output": json.dumps({"output": "compressed diff"}),
                        },
                    },
                    token_count(timestamp="2026-05-22T17:00:03.000Z"),
                ],
                start=3,
            ):
                result.extend(processor.process_line(json.dumps(record), event_seq))

            self.assertEqual(len(result.turns), 1)
            self.assertEqual(len(result.sources), 1)
            self.assertEqual(result.sources[0]["mcp_provider"], "agy")
            self.assertEqual(result.sources[0]["raw_context_avoided"], 2622)
            self.assertEqual(result.turns[0]["mcp_local_saved_tokens_est"], 0)
            self.assertEqual(result.turns[0]["mcp_agy_saved_tokens_est"], 2622)
            self.assertEqual(result.turns[0]["mcp_total_saved_tokens_est"], 2622)
            self.assertIn(
                "saved local 0  agy 2.6K  total 2.6K",
                codex_telemetry.format_echo_line(result.turns[0], result.sources),
            )

    def test_agy_fallback_counts_as_local_actual_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "rollout-2026-05-22T10-00-00-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
            local_ledger = root / "ledger.jsonl"
            output = "[agy_timeout]\nlocal fallback summary"
            write_jsonl(
                local_ledger,
                [
                    {
                        "record_type": "tool_call",
                        "task_id": "task-fallback",
                        "timestamp": "2026-05-22T17:00:02.000Z",
                        "tool_name": "agy_compress_diff",
                        "model": "antigravity/gemini-3.5-flash-high",
                        "local_output": output,
                        "token_estimates": {"input": 5200, "local_output": 200, "context_reduction": 5000},
                    }
                ],
            )
            write_jsonl(
                session,
                [
                    {
                        "timestamp": "2026-05-22T17:00:01.000Z",
                        "type": "turn_context",
                        "payload": {"thread_id": "thread-1", "turn_id": "turn-1"},
                    },
                    {
                        "timestamp": "2026-05-22T17:00:02.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": "agy_compress_diff",
                            "arguments": "{}",
                            "call_id": "call-agy",
                        },
                    },
                    {
                        "timestamp": "2026-05-22T17:00:02.100Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call_output",
                            "call_id": "call-agy",
                            "output": json.dumps({"output": output}),
                        },
                    },
                    token_count(timestamp="2026-05-22T17:00:03.000Z"),
                ],
            )

            codex_telemetry.run_once(
                session_files=[session],
                telemetry_dir=root / "telemetry",
                local_mcp_ledger_path=local_ledger,
                env={"CODEX_TELEMETRY_CAPTURE": "1"},
            )
            source = read_jsonl(root / "telemetry" / "sources.jsonl")[0]
            turn = read_jsonl(root / "telemetry" / "turns.jsonl")[0]

            self.assertEqual(source["mcp_provider"], "local")
            self.assertEqual(source["requested_provider"], "agy")
            self.assertEqual(turn["mcp_local_saved_tokens_est"], 5000)
            self.assertEqual(turn["mcp_agy_saved_tokens_est"], 0)
            self.assertEqual(turn["mcp_total_saved_tokens_est"], 5000)

    def test_echo_line_is_watcher_output(self) -> None:
        turn = {
            "turn_input_tokens": 9000,
            "turn_cached_input_tokens": 8000,
            "turn_output_tokens": 221,
            "turn_reasoning_output_tokens": 18,
            "context_used_pct": 42.0,
            "context_window": 258000,
            "primary_rate_used_pct": 34.0,
            "secondary_rate_used_pct": 69.0,
        }
        line = codex_telemetry.format_echo_line(
            turn,
            [{"raw_context_avoided": 8200}],
        )

        self.assertEqual(
            line,
            "codex  in 9.0K (+8.0K cached)  out 221  reason 18  "
            "ctx 42%  5h 34%  weekly 69%  saved local 8.2K  agy 0  total 8.2K",
        )

    def test_echo_line_color_enabled_adds_ansi_escapes(self) -> None:
        line = codex_telemetry.format_echo_line(
            {
                "turn_input_tokens": 102000,
                "turn_cached_input_tokens": 101000,
                "turn_output_tokens": 567,
                "turn_reasoning_output_tokens": 110,
                "context_used_pct": 40.0,
                "primary_rate_used_pct": 4.0,
                "secondary_rate_used_pct": 9.0,
                "cache_pressure_level": "high",
                "mcp_local_saved_tokens_est": 8200,
                "mcp_agy_saved_tokens_est": 31000,
                "mcp_total_saved_tokens_est": 39200,
            },
            [],
            color=True,
        )

        self.assertIn("\033[", line)
        self.assertIn("\033[33m101K\033[0m", line)
        self.assertIn("\033[32m4%\033[0m", line)
        self.assertIn("\033[34m8.2K\033[0m", line)
        self.assertIn("\033[35m31K\033[0m", line)
        self.assertIn("\033[1m39K\033[0m", line)

    def test_echo_color_env_suppression(self) -> None:
        stream = TtyStringIO()

        self.assertTrue(codex_telemetry.color_echo_enabled(stream, {}))
        self.assertFalse(codex_telemetry.color_echo_enabled(stream, {"NO_COLOR": "1"}))
        self.assertFalse(codex_telemetry.color_echo_enabled(stream, {"CODEX_TELEMETRY_COLOR": "0"}))
        self.assertTrue(codex_telemetry.color_echo_enabled(io.StringIO(), {"CODEX_TELEMETRY_COLOR": "1"}))

    def test_quota_color_thresholds(self) -> None:
        self.assertEqual(codex_telemetry.quota_color(69.9), codex_telemetry.ANSI_GREEN)
        self.assertEqual(codex_telemetry.quota_color(70), codex_telemetry.ANSI_YELLOW)
        self.assertEqual(codex_telemetry.quota_color(89.9), codex_telemetry.ANSI_YELLOW)
        self.assertEqual(codex_telemetry.quota_color(90), codex_telemetry.ANSI_RED)

    def test_repeated_import_does_not_append_duplicate_turns_or_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "rollout-2026-05-22T10-00-00-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
            write_jsonl(
                session,
                [
                    {
                        "timestamp": "2026-05-22T17:00:01.000Z",
                        "type": "turn_context",
                        "payload": {"thread_id": "thread-1", "turn_id": "turn-1"},
                    },
                    token_count(),
                ],
            )

            kwargs = {
                "session_files": [session],
                "telemetry_dir": root / "telemetry",
                "env": {"CODEX_TELEMETRY_CAPTURE": "1"},
            }
            codex_telemetry.run_once(**kwargs)
            codex_telemetry.run_once(**kwargs)

            turns = read_jsonl(root / "telemetry" / "turns.jsonl")
            self.assertEqual(len(turns), 1)

    def test_rotation_does_not_run_below_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "rollout-2026-05-22T10-00-00-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
            write_jsonl(session, [token_count()])

            codex_telemetry.run_once(
                session_files=[session],
                telemetry_dir=root / "telemetry",
                env={"CODEX_TELEMETRY_CAPTURE": "1"},
                rotate_bytes=1_000_000,
            )

            self.assertTrue((root / "telemetry" / "turns.jsonl").exists())
            self.assertFalse((root / "telemetry" / "archive").exists())

    def test_rotation_archives_active_ledgers_as_gzip_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "rollout-2026-05-22T10-00-00-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
            local_ledger = root / "ledger.jsonl"
            write_jsonl(
                local_ledger,
                [
                    {
                        "record_type": "tool_call",
                        "timestamp": "2026-05-22T17:00:02.000Z",
                        "tool_name": "local_summarize",
                        "local_output": "summary text",
                        "token_estimates": {"context_reduction": 180},
                    }
                ],
            )
            write_jsonl(
                session,
                [
                    {"timestamp": "2026-05-22T17:00:01.000Z", "type": "turn_context", "payload": {"thread_id": "thread-1", "turn_id": "turn-1"}},
                    {
                        "timestamp": "2026-05-22T17:00:02.000Z",
                        "type": "response_item",
                        "payload": {"type": "function_call", "name": "local_summarize", "arguments": "{}", "call_id": "call-local"},
                    },
                    {
                        "timestamp": "2026-05-22T17:00:02.100Z",
                        "type": "response_item",
                        "payload": {"type": "function_call_output", "call_id": "call-local", "output": json.dumps({"output": "summary text"})},
                    },
                    token_count(timestamp="2026-05-22T17:00:03.000Z"),
                ],
            )

            codex_telemetry.run_once(
                session_files=[session],
                telemetry_dir=root / "telemetry",
                local_mcp_ledger_path=local_ledger,
                env={"CODEX_TELEMETRY_CAPTURE": "1"},
                rotate_bytes=1,
            )

            archives = sorted((root / "telemetry" / "archive").glob("*/*.jsonl.gz"))
            self.assertTrue(any(path.name.endswith(".events.jsonl.gz") for path in archives))
            self.assertTrue(any(path.name.endswith(".turns.jsonl.gz") for path in archives))
            self.assertTrue(any(path.name.endswith(".sources.jsonl.gz") for path in archives))
            self.assertEqual(len(read_all_telemetry_rows(root, "turns.jsonl")), 1)
            self.assertEqual(len(read_all_telemetry_rows(root, "sources.jsonl")), 1)
            self.assertEqual(read_jsonl(root / "telemetry" / "turns.jsonl"), [])

    def test_rotation_state_prevents_duplicate_turns_and_sources_after_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "rollout-2026-05-22T10-00-00-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
            local_ledger = root / "ledger.jsonl"
            write_jsonl(
                local_ledger,
                [
                    {
                        "record_type": "tool_call",
                        "timestamp": "2026-05-22T17:00:02.000Z",
                        "tool_name": "local_summarize",
                        "local_output": "summary text",
                        "token_estimates": {"context_reduction": 180},
                    }
                ],
            )
            write_jsonl(
                session,
                [
                    {"timestamp": "2026-05-22T17:00:01.000Z", "type": "turn_context", "payload": {"thread_id": "thread-1", "turn_id": "turn-1"}},
                    {
                        "timestamp": "2026-05-22T17:00:02.000Z",
                        "type": "response_item",
                        "payload": {"type": "function_call", "name": "local_summarize", "arguments": "{}", "call_id": "call-local"},
                    },
                    {
                        "timestamp": "2026-05-22T17:00:02.100Z",
                        "type": "response_item",
                        "payload": {"type": "function_call_output", "call_id": "call-local", "output": json.dumps({"output": "summary text"})},
                    },
                    token_count(timestamp="2026-05-22T17:00:03.000Z"),
                ],
            )

            kwargs = {
                "session_files": [session],
                "telemetry_dir": root / "telemetry",
                "local_mcp_ledger_path": local_ledger,
                "env": {"CODEX_TELEMETRY_CAPTURE": "1"},
                "rotate_bytes": 1,
            }
            codex_telemetry.run_once(**kwargs)
            codex_telemetry.run_once(**kwargs)

            self.assertEqual(len(read_all_telemetry_rows(root, "turns.jsonl")), 1)
            self.assertEqual(len(read_all_telemetry_rows(root, "sources.jsonl")), 1)

    def test_archive_retention_deletes_old_gzip_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_dir = root / "telemetry" / "archive" / "2026-01-01"
            archive_dir.mkdir(parents=True)
            old_archive = archive_dir / "old.events.jsonl.gz"
            new_archive = archive_dir / "new.events.jsonl.gz"
            for path in (old_archive, new_archive):
                with gzip.open(path, "wt", encoding="utf-8") as handle:
                    handle.write(json.dumps({"ok": True}) + "\n")
            old_time = time.time() - (91 * 24 * 60 * 60)
            os.utime(old_archive, (old_time, old_time))

            codex_telemetry.TelemetryStorage(root / "telemetry", retention_days=90).compact_now()

            self.assertFalse(old_archive.exists())
            self.assertTrue(new_archive.exists())

    def test_rotation_can_be_disabled_by_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "rollout-2026-05-22T10-00-00-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
            write_jsonl(session, [token_count()])

            codex_telemetry.run_once(
                session_files=[session],
                telemetry_dir=root / "telemetry",
                env={"CODEX_TELEMETRY_CAPTURE": "1", "CODEX_TELEMETRY_ROTATION": "0"},
                rotate_bytes=1,
            )

            self.assertFalse((root / "telemetry" / "archive").exists())
            self.assertEqual(len(read_jsonl(root / "telemetry" / "turns.jsonl")), 1)

    def test_run_once_echo_prints_to_supplied_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = root / "rollout-2026-05-22T10-00-00-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
            write_jsonl(session, [token_count()])
            stream = io.StringIO()

            codex_telemetry.run_once(
                session_files=[session],
                telemetry_dir=root / "telemetry",
                env={"CODEX_TELEMETRY_CAPTURE": "1"},
                stdout=stream,
                echo=True,
            )

            self.assertTrue(stream.getvalue().startswith("codex  in"))

    def test_tui_log_fallback_records_lifecycle_metadata_without_turns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tui_log = root / "codex-tui.log"
            tui_log.write_text(
                "2026-05-22T17:00:00.000000Z  INFO "
                "session_loop{thread_id=thread-abc}:"
                "turn{thread.id=thread-abc turn.id=turn-def model=gpt-5.5 "
                "codex.turn.reasoning_effort=high}: codex_core::client: new\n",
                encoding="utf-8",
            )

            result = codex_telemetry.run_once(
                session_files=[],
                telemetry_dir=root / "telemetry",
                tui_log_path=tui_log,
                env={"CODEX_TELEMETRY_CAPTURE": "1"},
            )
            events = read_jsonl(root / "telemetry" / "events.jsonl")

            self.assertEqual(result.turns, [])
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["record_type"], "tui_lifecycle")
            self.assertEqual(events[0]["thread_id"], "thread-abc")
            self.assertEqual(events[0]["turn_id"], "turn-def")
            self.assertEqual(events[0]["model"], "gpt-5.5")
            self.assertEqual(events[0]["reasoning_effort"], "high")


if __name__ == "__main__":
    unittest.main()
