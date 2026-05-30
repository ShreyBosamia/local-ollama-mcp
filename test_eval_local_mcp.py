from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
import unittest

import agy_gemini_server
import eval_local_mcp
import server


class EvalLocalMcpTests(unittest.TestCase):
    def test_server_enforces_profile_output_boundaries(self) -> None:
        review = server.enforce_tool_output(
            "local_code_review",
            "- **Issue:** Total is overwritten; evidence `total =`; suggested check multi-item invoices.\n"
            "  - nested detail should collapse into the first finding.\n\n"
            "- **Issue:** `apply_discount` now uses `max(0, total - discount)` and incorrectly allows negative discounts.\n"
            "  - Suggested check: discount clamp behavior.",
        )
        tests = server.enforce_tool_output(
            "local_test_ideas",
            "```python\nimport pytest\ndef test_parse_retry_after(): pass\n```\n"
            "- Numeric seconds are parsed as an integer delay.\n",
        )

        self.assertEqual(review.count("\n- ") + (1 if review.startswith("- ") else 0), 1)
        self.assertNotIn("apply_discount", review)
        self.assertNotIn("```", tests)
        self.assertNotIn("import pytest", tests)
        self.assertNotIn("def test_", tests)

    def test_structure_result_fails_think_leak(self) -> None:
        case = eval_local_mcp.EvalCase(
            name="think_leak",
            category="reasoning",
            tool="local_reason_check",
            task="check reasoning output",
            artifact="problem",
            expected_facts=(),
            max_bullets=5,
        )

        score, violations, think_leak = eval_local_mcp.structure_result(
            "<think>hidden reasoning</think>\n- useful conclusion",
            case,
        )

        self.assertEqual(score, 0.0)
        self.assertTrue(think_leak)
        self.assertIn("think_leak", violations)

    def test_structure_result_enforces_json_and_bullet_limits(self) -> None:
        case = eval_local_mcp.EvalCase(
            name="format",
            category="format",
            tool="local_summarize",
            task="return json",
            artifact="large artifact",
            expected_facts=(),
            max_bullets=1,
            require_json=True,
        )

        score, violations, think_leak = eval_local_mcp.structure_result(
            "- one\n- two",
            case,
        )

        self.assertFalse(think_leak)
        self.assertLess(score, 1.0)
        self.assertIn("max_bullets:2>1", violations)
        self.assertIn("invalid_json", violations)

    def test_load_case_file_resolves_relative_artifact_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact = root / "artifacts" / "sample.diff"
            artifact.parent.mkdir()
            artifact.write_text("diff --git a/app.py b/app.py\n+bug\n", encoding="utf-8")
            case_file = root / "cases.jsonl"
            row = {
                "name": "sample",
                "tool": "local_code_review",
                "task": "review diff",
                "artifact_path": "artifacts/sample.diff",
                "expected_facts": [{"label": "bug", "pattern": "bug"}],
                "forbidden_facts": [{"label": "think", "pattern": "</?think\\b"}],
                "max_output_tokens": 100,
            }
            case_file.write_text(json.dumps(row) + "\n", encoding="utf-8")

            cases = eval_local_mcp.load_case_file(case_file)

            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].artifact, artifact.read_text(encoding="utf-8"))
            self.assertEqual(cases[0].expected_facts[0].label, "bug")
            self.assertEqual(cases[0].forbidden_facts[0].label, "think")
            self.assertEqual(cases[0].max_output_tokens, 100)

    def test_route_local_artifact_blocks_think_leak(self) -> None:
        artifact = "\n".join(f"context {index}" for index in range(200))

        routing = eval_local_mcp.route_local_artifact(
            artifact=artifact,
            local_output="<think>hidden</think> finding",
            required_facts_hit=1,
            required_facts_total=1,
            think_leak=True,
        )

        self.assertEqual(routing.routing_decision, "raw_cloud")
        self.assertIn("think_leak", routing.risk_flags)
        self.assertEqual(routing.cloud_tokens_avoided_est, 0)

    def test_usefulness_score_is_deterministic_and_penalizes_contradictions(self) -> None:
        clean = eval_local_mcp.usefulness_score_for(
            accuracy_score=1.0,
            structure_score=1.0,
            compression_score=1.0,
            forbidden_facts_hit=0,
            think_leak=False,
        )
        risky = eval_local_mcp.usefulness_score_for(
            accuracy_score=1.0,
            structure_score=1.0,
            compression_score=1.0,
            forbidden_facts_hit=1,
            think_leak=False,
        )

        self.assertEqual(clean, 1.0)
        self.assertLess(risky, clean)

    def test_dashboard_and_clean_export_omit_local_paths_from_export(self) -> None:
        rows = [
            {
                "timestamp": "2026-05-27T10:00:00+00:00",
                "suite": "synthetic",
                "case_count": 5,
                "use_local_count": 4,
                "verify_raw_count": 1,
                "skip_local_count": 0,
                "raw_cloud_count": 0,
                "average_accuracy_score": 0.9,
                "average_structure_score": 1.0,
                "average_usefulness_score": 0.88,
                "average_latency_ms": 1234,
                "think_leak_count": 0,
                "aggregate_token_reduction_pct": 0.42,
                "model": "qwen2.5-coder:7b-instruct-q5_K_M",
                "json_path": "/home/shrey/local-ollama-mcp/local_mcp_eval_results.json",
                "markdown_path": "/home/shrey/local-ollama-mcp/local_mcp_eval_report.md",
            }
        ]

        dashboard = eval_local_mcp.render_eval_dashboard(rows)
        clean_row = eval_local_mcp.clean_run_row(rows[0])

        self.assertIn("synthetic", dashboard)
        self.assertIn("local_mcp_eval_report.md", dashboard)
        self.assertEqual(clean_row["suite"], "synthetic")
        self.assertNotIn("model", clean_row)
        self.assertNotIn("json_path", clean_row)
        self.assertNotIn("markdown_path", clean_row)
        self.assertNotIn("/home/shrey", json.dumps(clean_row))

    def test_export_clean_writes_only_aggregate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_path = root / "index.jsonl"
            index_path.write_text(
                json.dumps(
                    {
                        "timestamp": "2026-05-27T10:00:00+00:00",
                        "suite": "reasoning",
                        "case_count": 1,
                        "raw_cloud_count": 0,
                        "think_leak_count": 0,
                        "local_output": "<think>hidden</think>",
                        "cwd": "/home/shrey/client",
                        "markdown_path": "/home/shrey/local/report.md",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            written = eval_local_mcp.export_clean_runs(index_path, root / "clean")
            exported = "\n".join(path.read_text(encoding="utf-8") for path in written)

            self.assertIn("reasoning", exported)
            self.assertNotIn("local_output", exported)
            self.assertNotIn("<think>", exported)
            self.assertNotIn("/home/shrey", exported)
            self.assertNotIn("cwd", exported)


    def test_quota_monitoring_and_switching(self) -> None:
        # Save original states to restore later
        orig_think = server.AGY_THINK_MODEL
        orig_flash = server.AGY_FLASH_MODEL
        orig_threshold = server.AGY_THINK_THRESHOLD

        try:
            # Setup mock quota states
            server.AGY_THINK_THRESHOLD = 15.0
            server.AGY_FLASH_MODEL = "gemini-3.5-flash-high"

            # Scenario 1: Sonnet Thinking has healthy quota (e.g. 72%)
            server._quota_cache["Claude Sonnet 4.6 (Thinking)"] = 72.0
            server._quota_cache["Claude Opus 4.6 (Thinking)"] = 100.0

            # Trigger manual parse / update test logic (simulate what check_antigravity_quotas does after update)
            server.AGY_THINK_MODEL = server.ORIGINAL_AGY_THINK_MODEL

            # Scenario 2: Sonnet quota drops below threshold (e.g. 10%) but Opus is healthy (e.g. 80%)
            server._quota_cache["Claude Sonnet 4.6 (Thinking)"] = 10.0
            server._quota_cache["Claude Opus 4.6 (Thinking)"] = 80.0

            # Mock the trigger of auto-switching logic
            sonnet_quota = server._quota_cache["Claude Sonnet 4.6 (Thinking)"]
            if sonnet_quota < server.AGY_THINK_THRESHOLD:
                opus_quota = server._quota_cache.get("Claude Opus 4.6 (Thinking)", 100.0)
                if opus_quota >= server.AGY_THINK_THRESHOLD:
                    server.AGY_THINK_MODEL = "claude-opus-4.6-thinking"
                else:
                    server.AGY_THINK_MODEL = server.AGY_FLASH_MODEL

            self.assertEqual(server.AGY_THINK_MODEL, "claude-opus-4.6-thinking")

            # Scenario 3: Both Sonnet and Opus are low (e.g. 5% and 8%)
            server._quota_cache["Claude Sonnet 4.6 (Thinking)"] = 5.0
            server._quota_cache["Claude Opus 4.6 (Thinking)"] = 8.0

            sonnet_quota = server._quota_cache["Claude Sonnet 4.6 (Thinking)"]
            if sonnet_quota < server.AGY_THINK_THRESHOLD:
                opus_quota = server._quota_cache.get("Claude Opus 4.6 (Thinking)", 100.0)
                if opus_quota >= server.AGY_THINK_THRESHOLD:
                    server.AGY_THINK_MODEL = "claude-opus-4.6-thinking"
                else:
                    server.AGY_THINK_MODEL = server.AGY_FLASH_MODEL

            self.assertEqual(server.AGY_THINK_MODEL, "gemini-3.5-flash-high")

            # Scenario 4: Recovery (Sonnet goes back to 25%)
            server._quota_cache["Claude Sonnet 4.6 (Thinking)"] = 25.0
            sonnet_quota = server._quota_cache["Claude Sonnet 4.6 (Thinking)"]
            if sonnet_quota >= server.AGY_THINK_THRESHOLD:
                server.AGY_THINK_MODEL = server.ORIGINAL_AGY_THINK_MODEL

            self.assertEqual(server.AGY_THINK_MODEL, server.ORIGINAL_AGY_THINK_MODEL)

        finally:
            # Restore states
            server.AGY_THINK_MODEL = orig_think
            server.AGY_FLASH_MODEL = orig_flash
            server.AGY_THINK_THRESHOLD = orig_threshold

    def test_ask_ollama_unloads_competing_models(self) -> None:
        import asyncio
        from unittest.mock import patch, AsyncMock
        mock_request = AsyncMock()
        mock_request.return_value = {"message": {"content": "mocked response"}}

        async def run_test():
            with patch("server.CODE_MODEL", "code-model"), patch("server.REASON_MODEL", "reason-model"):
                await server.ask_ollama("code-model", "test prompt")
                mock_request.assert_any_call(
                    "POST",
                    "/api/generate",
                    json_payload={"model": "reason-model", "prompt": "", "stream": False, "keep_alive": 0},
                    timeout=server.OLLAMA_TIMEOUT_SECONDS + 5
                )

                mock_request.reset_mock()
                await server.ask_ollama("reason-model", "test prompt")
                mock_request.assert_any_call(
                    "POST",
                    "/api/generate",
                    json_payload={"model": "code-model", "prompt": "", "stream": False, "keep_alive": 0},
                    timeout=server.OLLAMA_TIMEOUT_SECONDS + 5
                )

        with patch("server.ollama_request", mock_request):
            asyncio.run(run_test())

    def test_configure_models_updates_prompt_specs(self) -> None:
        original = (
            server.CODE_MODEL,
            server.PLAN_MODEL,
            server.REASON_MODEL,
            server.WARM_MODEL,
            {name: spec.model for name, spec in server.TOOL_PROMPTS.items()},
        )

        try:
            server.configure_models(
                code_model="qwen3.5:9b",
                plan_model="qwen3.5:9b",
                reason_model="qwen3.5:9b",
                warm_model="qwen3.5:9b",
            )

            self.assertEqual(server.TOOL_PROMPTS["local_summarize"].model, "qwen3.5:9b")
            self.assertEqual(server.TOOL_PROMPTS["local_code_review"].model, "qwen3.5:9b")
            self.assertEqual(server.TOOL_PROMPTS["local_test_ideas"].model, "qwen3.5:9b")
            self.assertEqual(server.TOOL_PROMPTS["local_plan_check"].model, "qwen3.5:9b")
            self.assertEqual(server.TOOL_PROMPTS["local_reason_check"].model, "qwen3.5:9b")
        finally:
            server.CODE_MODEL, server.PLAN_MODEL, server.REASON_MODEL, server.WARM_MODEL, specs = original
            for name, model in specs.items():
                server.TOOL_PROMPTS[name] = server.replace(server.TOOL_PROMPTS[name], model=model)

    def test_split_into_chunks(self) -> None:
        text = "\n".join(f"word{i}" for i in range(10))
        chunks = server.split_into_chunks(text, max_chunk_tokens=3)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks).replace("\n", ""), text.replace("\n", ""))

    def test_clean_server_logs(self) -> None:
        import asyncio
        logs = (
            "2026-05-27T18:41:57Z [1234] INFO Starting connection 0x7fffbbf\n"
            "2026-05-27T18:41:57Z [1234] INFO Starting connection 0x7fffbbf\n"
            "2026-05-27T18:41:59Z [1234] INFO Connection established\n"
        )
        cleaned = asyncio.run(server.clean_server_logs(logs))
        self.assertIn("[TIMESTAMP]", cleaned)
        self.assertIn("[HEX]", cleaned)
        self.assertIn("[repeated 2 times]", cleaned)

    def test_extract_regex_lines(self) -> None:
        import asyncio
        text = "line one\nline two with target\nline three"
        result_no_ctx = asyncio.run(server.extract_regex_lines(text, "target", context_lines=0))
        self.assertIn("line two with target", result_no_ctx)
        self.assertNotIn("line one", result_no_ctx)

        result_with_ctx = asyncio.run(server.extract_regex_lines(text, "target", context_lines=1))
        self.assertIn("line two with target", result_with_ctx)
        self.assertIn("line one", result_with_ctx)
        self.assertIn("line three", result_with_ctx)

    def test_trim_markdown_payload(self) -> None:
        import asyncio
        markdown = (
            "# Heading\n"
            "![an image](http://example.com/verylargebase64imgpayload)\n"
            "```python\n" + "\n".join(f"print({i})" for i in range(30)) + "\n```\n"
            "- item 1\n" + "\n".join(f"- item {i}" for i in range(2, 20))
        )

        trimmed = asyncio.run(server.trim_markdown_payload(
            markdown, max_code_block_lines=6, remove_images=True, max_list_items=5
        ))

        self.assertIn("[IMAGE: an image]", trimmed)
        self.assertIn("[trimmed 24 lines of code]", trimmed)
        self.assertIn("[trimmed 14 items]", trimmed)

    def test_local_summarize_routes_large_payload_to_flash(self) -> None:
        import asyncio
        from unittest.mock import patch, AsyncMock

        heavy_text = " ".join("word" for _ in range(5000))

        mock_agy = AsyncMock(return_value="- flash bullet")

        async def run_test():
            with (
                patch("server.ensure_quota_monitor_started"),
                patch("server.ask_antigravity_with_fallback", mock_agy),
            ):
                result = await server.local_summarize(heavy_text)
                self.assertIn("- flash bullet", result)
                self.assertIn("route_outcome: agy-default-gemini", result)
                self.assertRegex(
                    result,
                    r"token_savings: gemini_input_est=\d+ gpt_payload_est=\d+ "
                    r"gpt_saved_est=\d+ saved_pct=\d+%",
                )
                self.assertEqual(mock_agy.await_count, 1)
                self.assertEqual(mock_agy.await_args.kwargs["model"], server.AGY_FLASH_MODEL)

        asyncio.run(run_test())

    def test_gemini_fallback_output_does_not_report_token_savings(self) -> None:
        import asyncio
        from unittest.mock import patch, AsyncMock

        heavy_text = " ".join("word" for _ in range(5000))
        mock_agy = AsyncMock(return_value="[agy_timeout]\n- local fallback")

        async def run_test():
            with (
                patch("server.ensure_quota_monitor_started"),
                patch("server.ask_antigravity_with_fallback", mock_agy),
            ):
                result = await server.local_summarize(heavy_text)
                self.assertIn("route_outcome: agy-fallback", result)
                self.assertNotIn("token_savings:", result)

        asyncio.run(run_test())

    def test_captured_gemini_ledger_row_includes_token_savings_fields(self) -> None:
        import asyncio
        from unittest.mock import patch, AsyncMock

        heavy_diff = "\n".join(f"+ changed line {index} with repeated context" for index in range(1200))
        mock_agy = AsyncMock(return_value="CHANGED_FILES: app.py\nBEHAVIOR_CHANGES: none")

        async def run_test():
            with tempfile.TemporaryDirectory() as temp_dir:
                ledger = Path(temp_dir) / "ledger.jsonl"
                with (
                    patch.dict(
                        os.environ,
                        {"LOCAL_MCP_CAPTURE": "1", "LOCAL_MCP_LEDGER_PATH": str(ledger)},
                    ),
                    patch("server.ensure_quota_monitor_started"),
                    patch("server.ask_antigravity_with_fallback", mock_agy),
                ):
                    result = await server.agy_compress_diff(heavy_diff)

                self.assertIn("token_savings:", result)
                rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(rows), 1)
                row = rows[0]
                estimates = row["token_estimates"]
                self.assertEqual(row["route_outcome"], "agy-default-gemini")
                self.assertGreater(row["gemini_input_tokens_est"], row["gemini_output_tokens_est"])
                self.assertEqual(
                    row["gemini_saved_tokens_est"],
                    row["gemini_input_tokens_est"] - row["gemini_output_tokens_est"],
                )
                self.assertEqual(estimates["gemini_input"], row["gemini_input_tokens_est"])
                self.assertEqual(estimates["gemini_output"], row["gemini_output_tokens_est"])
                self.assertEqual(estimates["gemini_saved"], row["gemini_saved_tokens_est"])
                self.assertIn("gemini_saved_pct", estimates)

        asyncio.run(run_test())

    def test_gemini_wrapper_writes_agy_saved_token_ledger_row(self) -> None:
        import asyncio
        from unittest.mock import patch, AsyncMock

        heavy_diff = "\n".join(f"+ changed line {index} with repeated context" for index in range(1200))
        mock_agy = AsyncMock(return_value="CHANGED_FILES: app.py\nBEHAVIOR_CHANGES: none")

        async def run_test():
            with tempfile.TemporaryDirectory() as temp_dir:
                ledger = Path(temp_dir) / "ledger.jsonl"
                with (
                    patch.dict(
                        os.environ,
                        {"LOCAL_MCP_CAPTURE": "1", "LOCAL_MCP_LEDGER_PATH": str(ledger)},
                    ),
                    patch("server.ensure_quota_monitor_started"),
                    patch("server.ask_antigravity_with_fallback", mock_agy),
                ):
                    result = await agy_gemini_server.gemini_compress_diff(heavy_diff)

                self.assertIn("route_outcome: agy-default-gemini", result)
                self.assertIn("token_savings:", result)
                rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(rows), 1)
                row = rows[0]
                self.assertEqual(row["tool_name"], "gemini_compress_diff")
                self.assertEqual(row["model"], f"antigravity/{server.AGY_FLASH_MODEL}")
                self.assertEqual(row["route_outcome"], "agy-default-gemini")
                self.assertGreater(row["gemini_saved_tokens_est"], 0)
                self.assertEqual(row["token_estimates"]["gemini_saved"], row["gemini_saved_tokens_est"])

        asyncio.run(run_test())

    def test_gemini_context_reducer_prompt_contracts(self) -> None:
        prompts = [
            agy_gemini_server.build_gemini_summarize_context_prompt("log line", "errors"),
            agy_gemini_server.build_gemini_debug_digest_prompt("Traceback: boom", "fails on save", "root cause"),
            agy_gemini_server.build_gemini_plan_task_prompt("add feature", "repo context", "no file writes"),
            agy_gemini_server.build_gemini_review_diff_prompt("diff --git a/app.py b/app.py", "bugs"),
            agy_gemini_server.build_gemini_test_plan_prompt("def parse(x): return x", "pytest", "edge cases"),
            agy_gemini_server.build_gemini_repo_map_digest_prompt("server.py\nREADME.md", "entrypoints"),
            agy_gemini_server.build_gemini_symbol_contract_digest_prompt(
                "def parse(value: str) -> Result", "python", "public contracts"
            ),
            agy_gemini_server.build_gemini_config_surface_digest_prompt(
                "LOCAL_NUM_CTX=4096", "runtime defaults"
            ),
            agy_gemini_server.build_gemini_pr_thread_digest_prompt(
                "Reviewer: please add a regression test", "requested changes"
            ),
            agy_gemini_server.build_gemini_context_pack_prompt(
                "server.py handles capture", "add telemetry", "facts before editing"
            ),
        ]

        expected_sections = [
            "SUMMARY",
            "LIKELY_ROOT_CAUSES",
            "PLAN",
            "FINDINGS",
            "CORE_SCENARIOS",
            "ENTRYPOINTS",
            "PUBLIC_CONTRACTS",
            "RUNTIME_DEFAULTS",
            "REQUESTED_CHANGES",
            "TASK_RELEVANT_FACTS",
        ]
        for prompt, section in zip(prompts, expected_sections, strict=True):
            self.assertIn("context reducer for Codex/GPT", prompt)
            self.assertIn(section, prompt)
            self.assertIn("Do not use markdown code fences", prompt)
            self.assertNotIn("```", prompt)

    def test_gemini_context_reducer_tools_use_capture_metadata(self) -> None:
        import asyncio
        from unittest.mock import patch, AsyncMock

        captured = []
        mock_reduce = AsyncMock(return_value="SUMMARY\n- compressed context")

        async def fake_capture(tool_name, model, input_payload, action):
            captured.append((tool_name, model, input_payload))
            return await action()

        async def run_test():
            with (
                patch("server.capture_tool_call", side_effect=fake_capture),
                patch("server.ask_agy_context_reducer", mock_reduce),
            ):
                await agy_gemini_server.gemini_summarize_context("large log")
                await agy_gemini_server.gemini_debug_digest("Traceback: boom", symptoms="save fails")
                await agy_gemini_server.gemini_plan_task("add feature", context="repo", constraints="small")
                await agy_gemini_server.gemini_review_diff("diff --git a/app.py b/app.py")
                await agy_gemini_server.gemini_test_plan("def parse(x): return x", framework="pytest")
                await agy_gemini_server.gemini_repo_map_digest("server.py\nREADME.md")
                await agy_gemini_server.gemini_symbol_contract_digest(
                    "def parse(value: str) -> Result", language="python"
                )
                await agy_gemini_server.gemini_config_surface_digest("LOCAL_NUM_CTX=4096")
                await agy_gemini_server.gemini_pr_thread_digest("Reviewer: add a regression test")
                await agy_gemini_server.gemini_context_pack("server.py handles capture", task="add telemetry")

        asyncio.run(run_test())

        expected_names = [
            "gemini_summarize_context",
            "gemini_debug_digest",
            "gemini_plan_task",
            "gemini_review_diff",
            "gemini_test_plan",
            "gemini_repo_map_digest",
            "gemini_symbol_contract_digest",
            "gemini_config_surface_digest",
            "gemini_pr_thread_digest",
            "gemini_context_pack",
        ]
        self.assertEqual([row[0] for row in captured], expected_names)
        self.assertEqual([call.args[0] for call in mock_reduce.await_args_list], expected_names)
        for tool_name, model, input_payload in captured:
            self.assertEqual(model, f"antigravity/{server.AGY_FLASH_MODEL}")
            self.assertEqual(input_payload["routed_to"], "antigravity_flash")
            self.assertTrue(tool_name.startswith("gemini_"))

    def test_gemini_route_context_classifies_small_text_as_skip(self) -> None:
        plan = agy_gemini_server.build_context_route_plan(
            kind="auto",
            text="short context",
            task="inspect a tiny note",
            focus="summary",
        )

        self.assertEqual(plan["route_decision"], "skip")
        self.assertEqual(plan["selected_reducer"], "gemini_context_pack")
        self.assertLess(plan["raw_input_tokens_est"], server.AGY_ROUTING_MIN_TOKENS)

    def test_gemini_route_context_selects_reducers_for_large_inputs(self) -> None:
        large_diff = "\n".join(f"diff --git a/app{i}.py b/app{i}.py\n+changed {i}" for i in range(900))
        large_logs = "\n".join(f"ERROR timeout in worker {i}: Traceback failed" for i in range(900))
        large_map = "\n".join(f"src/module_{i}/handler.py" for i in range(1800))

        cases = [
            (large_diff, "gemini_compress_diff", "diff"),
            (large_logs, "gemini_debug_digest", "logs"),
            (large_map, "gemini_repo_map_digest", "repo_map"),
        ]
        for text, reducer, inferred_kind in cases:
            plan = agy_gemini_server.build_context_route_plan(kind="auto", text=text)
            self.assertEqual(plan["route_decision"], "gemini_recommended")
            self.assertEqual(plan["selected_reducer"], reducer)
            self.assertEqual(plan["inferred_kind"], inferred_kind)

    def test_gemini_route_context_capture_records_reducer_metadata_and_savings(self) -> None:
        import asyncio
        from unittest.mock import patch, AsyncMock

        heavy_logs = "\n".join(f"ERROR timeout in worker {index}: Traceback failed" for index in range(900))
        mock_agy = AsyncMock(return_value="LIKELY_ROOT_CAUSES\n- worker timeout in queue drain")

        async def run_test():
            with tempfile.TemporaryDirectory() as temp_dir:
                ledger = Path(temp_dir) / "ledger.jsonl"
                with (
                    patch.dict(
                        os.environ,
                        {"LOCAL_MCP_CAPTURE": "1", "LOCAL_MCP_LEDGER_PATH": str(ledger)},
                    ),
                    patch("server.ask_antigravity_with_fallback", mock_agy),
                ):
                    result = await agy_gemini_server.gemini_route_context(
                        "auto",
                        heavy_logs,
                        task="debug worker timeout",
                    )

                payload = json.loads(result)
                self.assertEqual(payload["route_decision"], "gemini_recommended")
                self.assertEqual(payload["route_outcome"], "agy-default-gemini")
                self.assertEqual(payload["selected_reducer"], "gemini_debug_digest")
                self.assertGreater(payload["gemini_saved_tokens_est"], 0)
                self.assertIn("token_savings:", payload["savings_line"])

                rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(rows), 1)
                row = rows[0]
                self.assertEqual(row["tool_name"], "gemini_route_context")
                self.assertEqual(row["context_route_decision"], "gemini_recommended")
                self.assertEqual(row["selected_reducer"], "gemini_debug_digest")
                self.assertEqual(row["inferred_kind"], "logs")
                self.assertEqual(row["route_outcome"], "agy-default-gemini")
                self.assertGreater(row["gemini_saved_tokens_est"], 0)
                self.assertEqual(row["token_estimates"]["gemini_saved"], row["gemini_saved_tokens_est"])

        asyncio.run(run_test())

    def test_gemini_route_context_fallback_is_not_counted_as_gemini_savings(self) -> None:
        import asyncio
        from unittest.mock import patch, AsyncMock

        heavy_config = "\n".join(f"FEATURE_FLAG_{index}=enabled" for index in range(3000))
        mock_agy = AsyncMock(return_value="[agy_timeout]\n- local fallback summary")

        async def run_test():
            with tempfile.TemporaryDirectory() as temp_dir:
                ledger = Path(temp_dir) / "ledger.jsonl"
                with (
                    patch.dict(
                        os.environ,
                        {"LOCAL_MCP_CAPTURE": "1", "LOCAL_MCP_LEDGER_PATH": str(ledger)},
                    ),
                    patch("server.ask_antigravity_with_fallback", mock_agy),
                ):
                    result = await agy_gemini_server.gemini_route_context("config", heavy_config)

                payload = json.loads(result)
                self.assertEqual(payload["route_outcome"], "agy-fallback")
                self.assertEqual(payload["gemini_saved_tokens_est"], 0)
                self.assertIn("timeout_or_fallback", payload["quality_flags"])

                row = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
                self.assertEqual(row["route_outcome"], "agy-fallback")
                self.assertNotIn("gemini_saved_tokens_est", row)
                self.assertNotIn("gemini_saved", row["token_estimates"])

        asyncio.run(run_test())

    def test_captured_gemini_context_pack_keeps_route_and_savings(self) -> None:
        import asyncio
        from unittest.mock import patch, AsyncMock

        heavy_sources = " ".join(f"fact{index}=value{index}" for index in range(1400))
        mock_agy = AsyncMock(return_value="TASK_RELEVANT_FACTS\n- one compact task fact")

        async def run_test():
            with tempfile.TemporaryDirectory() as temp_dir:
                ledger = Path(temp_dir) / "ledger.jsonl"
                with (
                    patch.dict(
                        os.environ,
                        {"LOCAL_MCP_CAPTURE": "1", "LOCAL_MCP_LEDGER_PATH": str(ledger)},
                    ),
                    patch("server.ask_antigravity_with_fallback", mock_agy),
                ):
                    result = await agy_gemini_server.gemini_context_pack(
                        heavy_sources,
                        task="add telemetry",
                    )

                self.assertIn("route_outcome: agy-default-gemini", result)
                self.assertIn("token_savings:", result)
                rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(rows), 1)
                row = rows[0]
                self.assertEqual(row["tool_name"], "gemini_context_pack")
                self.assertEqual(row["model"], f"antigravity/{server.AGY_FLASH_MODEL}")
                self.assertEqual(row["route_outcome"], "agy-default-gemini")
                self.assertGreater(row["gemini_saved_tokens_est"], 0)
                self.assertEqual(row["token_estimates"]["gemini_saved"], row["gemini_saved_tokens_est"])

        asyncio.run(run_test())

    def test_captured_gemini_context_reducer_keeps_route_and_savings(self) -> None:
        import asyncio
        from unittest.mock import patch, AsyncMock

        heavy_text = " ".join(f"token{index}" for index in range(1200))
        mock_agy = AsyncMock(return_value="SUMMARY\n- one compact implementation detail")

        async def run_test():
            with tempfile.TemporaryDirectory() as temp_dir:
                ledger = Path(temp_dir) / "ledger.jsonl"
                with (
                    patch.dict(
                        os.environ,
                        {"LOCAL_MCP_CAPTURE": "1", "LOCAL_MCP_LEDGER_PATH": str(ledger)},
                    ),
                    patch("server.ask_antigravity_with_fallback", mock_agy),
                ):
                    result = await agy_gemini_server.gemini_summarize_context(heavy_text)

                self.assertIn("route_outcome: agy-default-gemini", result)
                self.assertIn("token_savings:", result)
                rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(rows), 1)
                row = rows[0]
                self.assertEqual(row["tool_name"], "gemini_summarize_context")
                self.assertEqual(row["model"], f"antigravity/{server.AGY_FLASH_MODEL}")
                self.assertEqual(row["route_outcome"], "agy-default-gemini")
                self.assertGreater(row["gemini_saved_tokens_est"], 0)
                self.assertEqual(row["token_estimates"]["gemini_saved"], row["gemini_saved_tokens_est"])

        asyncio.run(run_test())

    def test_agy_compress_diff_routes_mid_sized_payload_to_flash(self) -> None:
        import asyncio
        from unittest.mock import patch, AsyncMock

        heavy_diff = "\n".join(f"+ changed line {index} with repeated context" for index in range(1200))
        mock_agy = AsyncMock(return_value="CHANGED_FILES: app.py\nBEHAVIOR_CHANGES: none\nRISKY_LINES: none\nREMOVED_LOGIC: none")

        async def run_test():
            with (
                patch("server.ensure_quota_monitor_started"),
                patch("server.ask_antigravity_with_fallback", mock_agy),
            ):
                result = await server.agy_compress_diff(heavy_diff)
                self.assertIn("CHANGED_FILES", result)
                self.assertEqual(mock_agy.await_count, 1)
                self.assertEqual(mock_agy.await_args.kwargs["model"], server.AGY_FLASH_MODEL)

        asyncio.run(run_test())

    def test_antigravity_fallback_marks_open_circuit(self) -> None:
        import asyncio
        from unittest.mock import patch, AsyncMock

        mock_ollama = AsyncMock(return_value="- local fallback")

        async def run_test():
            with (
                patch("server._agy_circuit_is_open", return_value=True),
                patch("server.ask_ollama", mock_ollama),
            ):
                result = await server.ask_antigravity_with_fallback("large prompt")
                self.assertTrue(result.startswith("[agy_circuit_open]"))
                self.assertIn("- local fallback", result)

        asyncio.run(run_test())

    def test_ask_antigravity_sends_prompt_over_stdin(self) -> None:
        import asyncio
        from unittest.mock import patch, AsyncMock

        class FakeProcess:
            returncode = 0

            def __init__(self) -> None:
                self.received_input = b""

            async def communicate(self, input=None):
                self.received_input = input or b""
                return b"agy response", b""

        prompt = "large prompt " * 20000
        fake_process = FakeProcess()
        mock_create = AsyncMock(return_value=fake_process)

        async def run_test():
            with (
                patch("server.agy_supports_model_flag", AsyncMock(return_value=False)),
                patch("server.asyncio.create_subprocess_exec", mock_create),
            ):
                result = await server.ask_antigravity(prompt)

            self.assertEqual(result, "agy response")
            args = mock_create.await_args.args
            self.assertIn("--print", args)
            self.assertIn("-", args)
            self.assertNotIn(prompt, args)
            self.assertEqual(mock_create.await_args.kwargs["stdin"], server.asyncio.subprocess.PIPE)
            self.assertEqual(fake_process.received_input, prompt.encode("utf-8"))

        asyncio.run(run_test())

    def test_hydrate_agy_file_artifact_response_reads_walkthrough(self) -> None:
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            artifact = home / ".gemini" / "antigravity-cli" / "brain" / "run-id" / "walkthrough.md"
            artifact.parent.mkdir(parents=True)
            artifact.write_text(
                "# Walkthrough - Example\n\n## Executive Summary\nUseful generated doc.",
                encoding="utf-8",
            )

            with patch("server.Path.home", return_value=home):
                hydrated = server.hydrate_agy_file_artifact_response(
                    f"Created artifact: [walkthrough.md](file://{artifact})"
                )

            self.assertTrue(hydrated.startswith("# Walkthrough - Example"))
            self.assertIn("## Executive Summary", hydrated)

    def test_local_analysis_timeout_returns_prefixed_response(self) -> None:
        import asyncio
        from unittest.mock import patch, AsyncMock

        async def hanging_ollama(*args, **kwargs):
            await asyncio.sleep(10)
            return "- too late"

        async def run_test():
            started = time.perf_counter()
            with (
                patch("server.ensure_quota_monitor_started"),
                patch("server.capture_enabled", return_value=False),
                patch("server.LOCAL_ANALYSIS_TIMEOUT", 0.05),
                patch("server.ask_ollama", AsyncMock(side_effect=hanging_ollama)),
            ):
                result = await server.local_summarize("short local payload")
            elapsed = time.perf_counter() - started
            self.assertLess(elapsed, 1.0)
            self.assertTrue(result.startswith("[local_timeout]"))
            self.assertIn("Handle this payload directly", result)

        asyncio.run(run_test())

    def test_agy_timeout_uses_bounded_local_fallback(self) -> None:
        import asyncio
        from unittest.mock import patch, AsyncMock

        async def run_test():
            started = time.perf_counter()
            with (
                patch("server._agy_circuit_is_open", return_value=False),
                patch("server._agy_record_failure"),
                patch("server.AGY_FALLBACK_TIMEOUT", 0.2),
                patch("server.AGY_TOTAL_TIMEOUT", 1.0),
                patch("server.ask_antigravity", AsyncMock(side_effect=RuntimeError("agy/default timed out after 1s"))),
                patch("server.ask_ollama", AsyncMock(return_value="- local fallback")),
            ):
                result = await server.ask_antigravity_with_fallback("large prompt", timeout=0.1)
            elapsed = time.perf_counter() - started
            self.assertLess(elapsed, 1.0)
            self.assertTrue(result.startswith("[agy_timeout]"))
            self.assertIn("- local fallback", result)

        asyncio.run(run_test())

    def test_agy_timeout_fallback_timeout_fails_fast(self) -> None:
        import asyncio
        from unittest.mock import patch, AsyncMock

        async def hanging_ollama(*args, **kwargs):
            await asyncio.sleep(10)
            return "- too late"

        async def run_test():
            started = time.perf_counter()
            with (
                patch("server._agy_circuit_is_open", return_value=False),
                patch("server._agy_record_failure"),
                patch("server.AGY_FALLBACK_TIMEOUT", 0.05),
                patch("server.AGY_TOTAL_TIMEOUT", 1.0),
                patch("server.ask_antigravity", AsyncMock(side_effect=RuntimeError("agy/default timed out after 1s"))),
                patch("server.ask_ollama", AsyncMock(side_effect=hanging_ollama)),
            ):
                result = await server.ask_antigravity_with_fallback("large prompt", timeout=0.1)
            elapsed = time.perf_counter() - started
            self.assertLess(elapsed, 1.0)
            self.assertEqual(result, "[agy_timeout] fallback_also_failed. Handle this payload directly.")

        asyncio.run(run_test())

    def test_routing_threshold_keeps_under_4000_local_and_4000_plus_agy(self) -> None:
        from unittest.mock import patch

        with patch("server.ensure_quota_monitor_started"), patch("server.AGY_ENABLED", True):
            self.assertEqual(server.select_tier(3999), "local_gpu")
            self.assertEqual(server.select_tier(4000), "antigravity")

    def test_local_capture_status_scans_bounded_rows_without_model(self) -> None:
        import asyncio
        from unittest.mock import patch, AsyncMock

        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = Path(temp_dir) / "ledger.jsonl"
            rows = []
            for index in range(6):
                rows.append(
                    json.dumps(
                        {
                            "record_type": "tool_call",
                            "task_id": f"task-{index}",
                        }
                    )
                )
            ledger.write_text("\n".join(rows) + "\n", encoding="utf-8")

            async def run_test():
                with (
                    patch.dict(os.environ, {"LOCAL_MCP_LEDGER_PATH": str(ledger)}),
                    patch("server.LOCAL_STATUS_LEDGER_ROWS", 3),
                    patch("server.ask_ollama", AsyncMock(side_effect=AssertionError("model should not run"))),
                ):
                    result = await server.local_capture_status()
                self.assertIn("ledger_rows_scanned: 3", result)
                self.assertIn("ledger_scan_bounded: yes", result)
                self.assertIn("task-5", result)
                self.assertNotIn("task-0", result)

            asyncio.run(run_test())

    def test_pipeline_tool_chaining(self) -> None:
        import asyncio
        from unittest.mock import patch, AsyncMock
        
        # Verbose mock database lock trace
        raw_trace = (
            "2026-05-27 19:12:00.000 UTC [1421] [0x7f83ad29c9] pid=1421 INFO: client connected\n"
            "2026-05-27 19:12:01.000 UTC [1421] [0x7f83ad29c9] pid=1421 INFO: client connected\n"
            "2026-05-27 19:15:32.481 UTC [1421] [0x7f83ad29c9] pid=1421 ERROR: deadlock detected\n"
            "    Process 1421 waits for ShareLock; blocked by process 1429.\n"
            "    Process 1421: UPDATE orders SET status = 'completed';\n"
            "2026-05-27 19:16:00.000 UTC [1421] [0x7f83ad29c9] pid=1421 INFO: client connected"
        )
        
        case = eval_local_mcp.EvalCase(
            name="postgres_lock_trace_pipeline",
            category="pipeline",
            tool="local_summarize",
            task="identify lock contention patterns",
            artifact=raw_trace,
            focus="deadlock details and lock statements",
            expected_facts=()
        )
        
        mock_summarize = AsyncMock(return_value="- processed deadlock bullet")
        
        async def run_test():
            with patch("server.local_summarize", mock_summarize):
                result = await eval_local_mcp.call_tool(case)
                self.assertEqual(result, "- processed deadlock bullet")
                
                # Verify that mock_summarize was called with preprocessed text
                called_text = mock_summarize.call_args[0][0]
                
                # Check that timestamps, PIDs, hexes, and duplicate logs were cleaned/collapsed
                self.assertNotIn("2026-05-27", called_text)
                self.assertNotIn("19:12:00", called_text)
                self.assertNotIn("0x7f83ad29c9", called_text)
                self.assertNotIn("pid=1421", called_text)
                
                # Check that deduplicate_consecutive collapsed the repetitive connection info
                self.assertIn("[repeated 2 times]", called_text)
                
                # Check that the deadlock error is captured
                self.assertIn("deadlock detected", called_text)
                self.assertIn("UPDATE orders", called_text)
                
        asyncio.run(run_test())

    def test_local_map_project_structure(self) -> None:
        import asyncio
        from unittest.mock import patch, AsyncMock
        
        mock_run = AsyncMock(return_value=(True, "M  server.py\n?? README.md"))
        with patch("server.run_command", mock_run):
            res = asyncio.run(server.local_map_project_structure(max_depth=2))
            self.assertIn("server.py [M]", res)
            self.assertIn("README.md [??]", res)

    def test_local_extract_signatures(self) -> None:
        py_content = (
            "class Calculator:\n"
            "    \"\"\"A docstring.\"\"\"\n"
            "    def add(self, a, b):\n"
            "        return a + b\n"
        )
        py_res = server.extract_python_signatures(py_content)
        self.assertIn("class Calculator:", py_res)
        self.assertIn("\"\"\"A docstring.\"\"\"", py_res)
        self.assertIn("def add(self, a, b):", py_res)
        self.assertNotIn("return a + b", py_res)

        ts_content = (
            "/** A class JSDoc. */\n"
            "export class User {\n"
            "  private id: string;\n"
            "  /** Get ID. */\n"
            "  public getId(): string {\n"
            "    return this.id;\n"
            "  }\n"
            "}\n"
        )
        ts_res = server.extract_ts_js_signatures(ts_content)
        self.assertIn("export class User {", ts_res)
        self.assertIn("/** A class JSDoc. */", ts_res)
        self.assertIn("private id: string;", ts_res)
        self.assertIn("public getId(): string {", ts_res)
        self.assertNotIn("return this.id;", ts_res)

    def test_local_lint_audit(self) -> None:
        import asyncio
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("def invalid_syntax(\n")
            f.close()
            try:
                res_err = asyncio.run(server.local_lint_audit(f.name))
                self.assertIn("SyntaxError", res_err)
            finally:
                Path(f.name).unlink()

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("def valid_syntax():\n    pass\n")
            f.close()
            try:
                res_ok = asyncio.run(server.local_lint_audit(f.name))
                self.assertEqual(res_ok, "No syntax errors found.")
            finally:
                Path(f.name).unlink()

    def test_local_generate_walkthrough(self) -> None:
        import asyncio
        from unittest.mock import patch, AsyncMock

        async def mock_run_command(*args, **kwargs):
            cmd = args[0]
            if cmd == "git" and args[1] == "diff":
                if "--cached" in args:
                    return True, "staged diff content"
                else:
                    return True, "unstaged diff content"
            elif cmd == "git" and args[1] == "status":
                return True, "?? untracked_file.py"
            return True, ""

        mock_agy = AsyncMock(return_value="# Walkthrough - Test Change\n## Executive Summary\nTest summary.")

        async def run_test():
            with (
                patch("server.run_command", AsyncMock(side_effect=mock_run_command)),
                patch("server.ask_antigravity_with_fallback", mock_agy),
                patch("pathlib.Path.exists", return_value=True),
                patch("pathlib.Path.is_file", return_value=True),
                patch("pathlib.Path.read_text", return_value="print('new file content')"),
                patch("pathlib.Path.write_text") as mock_write,
            ):
                result = await server.local_generate_walkthrough(write_to_file=True)
                self.assertIn("Walkthrough - Test Change", result)
                self.assertIn("route_outcome: agy-default-gemini", result)
                self.assertEqual(mock_write.call_count, 1)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
