from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest

import codex_telemetry


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
        self.assertEqual(codex_telemetry.normalize_quota([{"remaining_percent": 77}, {"used_percent": 14}]), (23.0, "balance_array"))
        self.assertEqual(codex_telemetry.normalize_quota({}), (None, "missing"))

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
        self.assertIn("5h 14%, weekly 23%", codex_telemetry.format_echo_line(base_turn, []))
        reset_turn = dict(base_turn)
        reset_turn["primary_rate_used_pct"] = 99.0
        self.assertIn("5h 99%, weekly 23%", codex_telemetry.format_echo_line(reset_turn, []))

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
            self.assertEqual(source["token_estimate"], 2)
            self.assertEqual(source["raw_context_avoided"], 180)
            self.assertEqual(source["confidence"], "matched_hash")

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
            "codex-telemetry: tokens: in 9.0K (+8.0K cached), out 221, "
            "reason 18 | ctx 42% of 258K | 5h 34%, weekly 69% | local saved est 8.2K",
        )

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

            self.assertTrue(stream.getvalue().startswith("codex-telemetry: tokens:"))

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
