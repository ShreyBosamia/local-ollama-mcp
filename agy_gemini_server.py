#!/usr/bin/env python3
import json
import re
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP

import server


mcp = FastMCP("agy-gemini")


AGY_PROVIDER = "antigravity_flash"
GEMINI_ROUTE_THRESHOLD = server.AGY_ROUTING_MIN_TOKENS
LOCAL_CANDIDATE_MIN_TOKENS = 1000
AGY_FALLBACK_PREFIXES = (
    "[agy_rate_limited]",
    "[agy_timeout]",
    "[agy_missing_binary]",
    "[agy_error]",
    "[agy_circuit_open]",
    "[local_timeout]",
    "fallback_also_failed",
)
SUPPORTED_ROUTE_KINDS = {
    "diff",
    "logs",
    "repo_map",
    "config",
    "pr_thread",
    "mixed_context",
    "auto",
}
RAW_REQUIRED_RE = re.compile(
    r"(?is)(\bverbatim\b|\bexact raw\b|\bdo not summarize\b|BEGIN [A-Z ]*PRIVATE KEY|"
    r"\x00|[A-Za-z0-9+/]{500,}={0,2})"
)
DIFF_RE = re.compile(r"(?m)^(diff --git|@@ |\+\+\+ |--- |\+[^+]|-[^-])")
LOG_RE = re.compile(r"(?i)(traceback|exception|error|failed|failure|timeout|segfault|panic|stack trace)")
REPO_MAP_RE = re.compile(r"(?m)^([./\w-]+/|[./\w-]+\.(py|js|ts|tsx|go|rs|md|toml|json|ya?ml|txt))$")
CONFIG_RE = re.compile(r"(?im)^([A-Z][A-Z0-9_]{2,}=|[\w.-]+\.(toml|json|ya?ml|ini|env)|\s*[-\w]+\s*:)")
PR_THREAD_RE = re.compile(r"(?i)(pull request|reviewer|requested changes|unresolved|approve|merge blocker|ci failed)")
DETAIL_MARKER_RE = re.compile(
    r"(?i)([A-Za-z0-9_./-]+\.(?:py|js|ts|tsx|go|rs|md|toml|json|ya?ml|txt|log)|"
    r"\b[A-Z][A-Z0-9_]{2,}\b|"
    r"\b(?:Traceback|Exception|Error|Timeout|failed|panic|segfault)\b|"
    r"\b\d{3}\b)"
)


def _json_response(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def normalize_route_kind(kind: str) -> str:
    normalized = (kind or "auto").strip().lower()
    normalized = normalized.replace("-", "_")
    return normalized if normalized in SUPPORTED_ROUTE_KINDS else "auto"


def infer_route_kind(text: str, kind: str) -> str:
    normalized = normalize_route_kind(kind)
    if normalized != "auto":
        return normalized
    if DIFF_RE.search(text):
        return "diff"
    if LOG_RE.search(text):
        return "logs"
    if PR_THREAD_RE.search(text):
        return "pr_thread"
    if CONFIG_RE.search(text):
        return "config"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    path_like = sum(1 for line in lines if REPO_MAP_RE.search(line))
    if lines and path_like / max(1, len(lines)) >= 0.50:
        return "repo_map"
    return "mixed_context"


def reducer_for_kind(kind: str) -> str:
    return {
        "diff": "gemini_compress_diff",
        "logs": "gemini_debug_digest",
        "repo_map": "gemini_repo_map_digest",
        "config": "gemini_config_surface_digest",
        "pr_thread": "gemini_pr_thread_digest",
        "mixed_context": "gemini_context_pack",
    }.get(kind, "gemini_context_pack")


def route_decision_for_context(text: str, task: str = "", focus: str = "") -> str:
    token_count = server.estimate_tokens(text)
    raw_required_context = "\n".join([task or "", focus or "", text[:2000]])
    if RAW_REQUIRED_RE.search(raw_required_context):
        return "raw_required"
    if token_count >= GEMINI_ROUTE_THRESHOLD:
        return "gemini_recommended"
    if token_count >= LOCAL_CANDIDATE_MIN_TOKENS:
        return "local_candidate"
    return "skip"


def build_context_route_plan(
    *,
    kind: str,
    text: str,
    task: str = "",
    focus: str = "",
) -> dict[str, Any]:
    inferred_kind = infer_route_kind(text, kind)
    raw_tokens = server.estimate_tokens(text)
    decision = route_decision_for_context(text, task, focus)
    reducer = reducer_for_kind(inferred_kind)
    return {
        "kind": normalize_route_kind(kind),
        "inferred_kind": inferred_kind,
        "route_decision": decision,
        "selected_reducer": reducer,
        "raw_input_tokens_est": raw_tokens,
        "threshold_tokens": GEMINI_ROUTE_THRESHOLD,
        "route_outcome": "advisory",
        "gemini_saved_tokens_est": 0,
    }


def quality_flags_for_reduction(raw_text: str, reduced_output: str, route_outcome: str) -> list[str]:
    flags: list[str] = []
    stripped = reduced_output.strip()
    raw_tokens = server.estimate_tokens(raw_text)
    output_tokens = server.estimate_tokens(reduced_output)
    if not stripped:
        flags.append("empty_output")
    if route_outcome == "agy-fallback" or any(prefix in reduced_output for prefix in AGY_FALLBACK_PREFIXES):
        flags.append("timeout_or_fallback")
    if raw_tokens and output_tokens >= max(1000, int(raw_tokens * 0.75)):
        flags.append("overlong_output")

    raw_markers = DETAIL_MARKER_RE.findall(raw_text)
    if raw_markers:
        flattened = [item[0] if isinstance(item, tuple) else item for item in raw_markers]
        required = [marker for marker in flattened if marker][:5]
        output_lower = reduced_output.lower()
        if required and not any(marker.lower() in output_lower for marker in required):
            flags.append("missing_required_looking_details")
    return sorted(set(flags))


def _route_prompt_builder(
    reducer: str,
    *,
    text: str,
    task: str,
    focus: str,
) -> tuple[str, Callable[[], str]]:
    if reducer == "gemini_compress_diff":
        return reducer, lambda: build_gemini_review_diff_prompt(text, focus or "bugs, regressions, API surface changes")
    if reducer == "gemini_debug_digest":
        return reducer, lambda: build_gemini_debug_digest_prompt(text, task, focus or "root cause and next checks")
    if reducer == "gemini_repo_map_digest":
        return reducer, lambda: build_gemini_repo_map_digest_prompt(text, focus or "implementation-relevant structure")
    if reducer == "gemini_config_surface_digest":
        return reducer, lambda: build_gemini_config_surface_digest_prompt(text, focus or "runtime and build behavior")
    if reducer == "gemini_pr_thread_digest":
        return reducer, lambda: build_gemini_pr_thread_digest_prompt(text, focus or "actionable unresolved work")
    return reducer, lambda: build_gemini_context_pack_prompt(
        text,
        task or "Compress mixed context for Codex to verify.",
        focus or "facts GPT needs before editing",
    )


def _agy_model_name() -> str:
    return f"antigravity/{server.AGY_FLASH_MODEL}"


def _wrap_prompt(
    *,
    task: str,
    focus: str,
    format_rules: str,
    payload_label: str,
    payload: str,
    extra_context: str = "",
) -> str:
    context_block = f"<context>\n{extra_context}\n</context>\n" if extra_context else ""
    return (
        "You are Gemini 3.5 Flash High running as an explicit agy_gemini MCP "
        "context reducer for Codex/GPT.\n"
        "Reduce raw context before GPT sees it. Do not act as final authority, "
        "do not request file writes, and do not suggest running shell commands as if you ran them.\n"
        "Use fixed section labels exactly. Do not use markdown code fences unless briefly quoting input text.\n\n"
        f"<task>\n{task}\n</task>\n"
        f"<focus>\n{focus}\n</focus>\n"
        f"{context_block}"
        f"<format>\n{format_rules}\n</format>\n"
        f"<{payload_label}>\n{payload}\n</{payload_label}>"
    )


async def _call_context_reducer(
    *,
    tool_name: str,
    input_payload: dict,
    prompt: str,
) -> str:
    return await server.capture_tool_call(
        tool_name,
        _agy_model_name(),
        {**input_payload, "routed_to": AGY_PROVIDER},
        lambda: server.ask_agy_context_reducer(tool_name, prompt),
    )


def _reducer_output_for_estimate(router_output: str) -> str:
    try:
        payload = json.loads(router_output)
    except json.JSONDecodeError:
        return router_output
    if isinstance(payload, dict):
        return str(payload.get("reducer_output") or "")
    return router_output


@mcp.tool()
async def gemini_route_context(
    kind: str,
    text: str,
    task: str = "",
    focus: str = "",
    run_reducer: bool = True,
) -> str:
    """
    Deterministically decide whether context should be reduced through Gemini
    before Codex spends tokens on it. v1 is advisory: raw context remains
    available and the decision never blocks callers from using it directly.
    """
    plan = build_context_route_plan(kind=kind, text=text, task=task, focus=focus)
    reducer = str(plan["selected_reducer"])
    raw_tokens = int(plan["raw_input_tokens_est"])

    if plan["route_decision"] != "gemini_recommended" or not run_reducer:
        plan.update(
            {
                "gemini_output_tokens_est": 0,
                "quality_flags": ["reducer_not_run"],
                "reducer_output": "",
                "savings_line": "token_savings: gemini_input_est=0 gpt_payload_est=0 gpt_saved_est=0 saved_pct=0%",
            }
        )
        return _json_response(plan)

    _, prompt_builder = _route_prompt_builder(reducer, text=text, task=task, focus=focus)
    input_payload = {
        "kind": plan["kind"],
        "inferred_kind": plan["inferred_kind"],
        "route_decision": plan["route_decision"],
        "selected_reducer": reducer,
        "text": text,
        "text_tokens": raw_tokens,
        "task": task,
        "focus": focus,
        "routed_to": AGY_PROVIDER,
    }
    record_metadata = {
        "context_route_decision": plan["route_decision"],
        "selected_reducer": reducer,
        "inferred_kind": plan["inferred_kind"],
        "raw_input_tokens_est": raw_tokens,
        "route_threshold_tokens": GEMINI_ROUTE_THRESHOLD,
    }

    async def run_router_reducer() -> str:
        if reducer == "gemini_compress_diff":
            reducer_output = await server.run_local_analysis(
                lambda: server.ask_agy_diff_compression(text, focus or "bugs, regressions, API surface changes"),
                timeout=server.AGY_DIFF_TOTAL_TIMEOUT,
            )
        else:
            reducer_output = await server.ask_agy_context_reducer("gemini_route_context", prompt_builder())

        outcome = await server.determine_route_outcome("gemini_route_context", _agy_model_name(), reducer_output)
        output_tokens = server.estimate_tokens(reducer_output)
        saved_tokens = max(0, raw_tokens - output_tokens) if outcome == "agy-default-gemini" else 0
        saved_pct = round(saved_tokens / raw_tokens, 3) if raw_tokens else 0.0
        quality_flags = quality_flags_for_reduction(text, reducer_output, outcome)
        return _json_response(
            {
                **plan,
                "route_outcome": outcome,
                "gemini_output_tokens_est": output_tokens,
                "gemini_saved_tokens_est": saved_tokens,
                "gemini_saved_pct": saved_pct,
                "quality_flags": quality_flags,
                "reducer_output": reducer_output,
                "savings_line": server.gemini_token_savings_line(
                    {
                        "gemini_input_tokens_est": raw_tokens,
                        "gemini_output_tokens_est": output_tokens,
                        "gemini_saved_tokens_est": saved_tokens,
                        "gemini_saved_pct": saved_pct,
                    }
                ),
            }
        )

    return await server.capture_tool_call(
        "gemini_route_context",
        _agy_model_name(),
        input_payload,
        run_router_reducer,
        record_metadata=record_metadata,
        record_output_for_token_estimate=_reducer_output_for_estimate,
    )


def build_gemini_summarize_context_prompt(
    text: str,
    focus: str = "important implementation details",
) -> str:
    return _wrap_prompt(
        task="Compress large logs, docs, command output, or pasted code for Codex.",
        focus=focus,
        format_rules=(
            "SUMMARY\n"
            "- Return 6 to 10 dense bullets total.\n"
            "- Each bullet must preserve concrete names, numbers, paths, commands, options, and error strings.\n"
            "- Prefer implementation details and decision-relevant facts over broad explanation.\n"
            "RISKS_OR_GAPS\n"
            "- Include only concrete ambiguities or missing facts visible from the input.\n"
            "- If none are visible, write: - None visible."
        ),
        payload_label="text",
        payload=text,
    )


def build_gemini_debug_digest_prompt(
    logs_or_error: str,
    symptoms: str = "",
    focus: str = "root cause and next checks",
) -> str:
    return _wrap_prompt(
        task="Digest debugging logs or errors for Codex to verify.",
        focus=focus,
        extra_context=symptoms,
        format_rules=(
            "LIKELY_ROOT_CAUSES\n"
            "- 1 to 4 bullets with the most plausible causes and confidence cues.\n"
            "EXACT_ERROR_STRINGS\n"
            "- Preserve exact error messages, exception names, command names, status codes, and paths.\n"
            "SUSPICIOUS_TRANSITIONS\n"
            "- Note ordering changes, before/after state, retries, timeouts, or first bad line.\n"
            "NEXT_CHECKS\n"
            "- 2 to 5 concrete checks for GPT/Codex to run or inspect next."
        ),
        payload_label="logs_or_error",
        payload=logs_or_error,
    )


def build_gemini_plan_task_prompt(
    task: str,
    context: str = "",
    constraints: str = "",
) -> str:
    extra_context = (
        f"Context:\n{context or 'None provided.'}\n\n"
        f"Constraints:\n{constraints or 'None provided.'}"
    )
    return _wrap_prompt(
        task="Produce a compact implementation plan for GPT/Codex to verify before editing.",
        focus="minimal safe implementation, risks, and verification",
        extra_context=extra_context,
        format_rules=(
            "PLAN\n"
            "- 3 to 7 ordered bullets, each one concrete implementation action.\n"
            "RISKS\n"
            "- 1 to 4 bullets with likely regressions, ownership boundaries, or assumptions to verify.\n"
            "VERIFICATION\n"
            "- 1 to 4 bullets with compile, test, smoke, or telemetry checks.\n"
            "NON_GOALS\n"
            "- Mention file writes, shell execution, or autonomous changes only as non-goals for this reducer."
        ),
        payload_label="task_request",
        payload=task,
    )


def build_gemini_review_diff_prompt(
    diff: str,
    focus: str = "bugs, regressions, missing tests",
) -> str:
    return _wrap_prompt(
        task="Review a git diff as a high-confidence pre-filter for Codex.",
        focus=focus,
        format_rules=(
            "FINDINGS\n"
            "- 0 to 5 bullets only. Each bullet format: severity; file or symbol; issue; evidence; suggested check.\n"
            "- Report only likely bugs, regressions, or missing tests that GPT should verify.\n"
            "- If there are no high-confidence findings, write exactly: - No high-confidence findings.\n"
            "MISSING_TESTS\n"
            "- 0 to 4 bullets with test gaps tied to changed behavior.\n"
            "VERIFY_NEXT\n"
            "- 1 to 4 bullets with targeted inspections or commands for GPT/Codex to consider."
        ),
        payload_label="diff",
        payload=diff,
    )


def build_gemini_test_plan_prompt(
    code_or_diff: str,
    framework: str = "unknown",
    focus: str = "edge cases and regressions",
) -> str:
    return _wrap_prompt(
        task="Create concise test scenarios for Codex without writing executable tests.",
        focus=focus,
        extra_context=f"Framework: {framework}",
        format_rules=(
            "CORE_SCENARIOS\n"
            "- 2 to 5 bullets covering expected behavior and changed paths.\n"
            "EDGE_CASES\n"
            "- 2 to 5 bullets covering boundary values, malformed inputs, races, or empty states.\n"
            "REGRESSION_CHECKS\n"
            "- 1 to 4 bullets tied to prior behavior that could break.\n"
            "TEST_DATA\n"
            "- 0 to 4 bullets naming fixtures, inputs, or states. Do not write executable test code."
        ),
        payload_label="code_or_diff",
        payload=code_or_diff,
    )


def build_gemini_repo_map_digest_prompt(
    project_tree: str,
    focus: str = "implementation-relevant structure",
) -> str:
    return _wrap_prompt(
        task="Compress a repository tree, file list, or module map for GPT/Codex orientation.",
        focus=focus,
        format_rules=(
            "ENTRYPOINTS\n"
            "- 1 to 6 bullets naming likely runtime, CLI, server, app, or test entrypoints.\n"
            "KEY_MODULES\n"
            "- 2 to 8 bullets naming important files or directories and their visible role.\n"
            "DATA_FLOW\n"
            "- 1 to 5 bullets describing visible flow between modules, configs, or artifacts.\n"
            "LIKELY_TOUCH_POINTS\n"
            "- 1 to 6 bullets naming files GPT should inspect for the requested focus.\n"
            "OMIT_FROM_GPT\n"
            "- 1 to 6 bullets naming generated, vendored, cache, or low-signal paths to skip."
        ),
        payload_label="project_tree",
        payload=project_tree,
    )


def build_gemini_symbol_contract_digest_prompt(
    signatures_or_types: str,
    language: str = "unknown",
    focus: str = "public interfaces",
) -> str:
    return _wrap_prompt(
        task="Compress exported symbols, function signatures, classes, schemas, or type definitions for GPT/Codex.",
        focus=focus,
        extra_context=f"Language: {language}",
        format_rules=(
            "PUBLIC_CONTRACTS\n"
            "- 2 to 8 bullets naming public functions, classes, types, fields, routes, or schemas.\n"
            "CALL_PATTERNS\n"
            "- 1 to 6 bullets describing required call order, ownership, lifecycle, or async behavior.\n"
            "INPUT_OUTPUT_SHAPES\n"
            "- 1 to 8 bullets preserving argument names, return fields, status values, or error shapes.\n"
            "COMPATIBILITY_RISKS\n"
            "- 0 to 5 bullets naming contract changes GPT should verify before editing."
        ),
        payload_label="signatures_or_types",
        payload=signatures_or_types,
    )


def build_gemini_config_surface_digest_prompt(
    configs: str,
    focus: str = "runtime and build behavior",
) -> str:
    return _wrap_prompt(
        task="Compress manifests, environment docs, CI snippets, and tool settings for GPT/Codex.",
        focus=focus,
        format_rules=(
            "RUNTIME_DEFAULTS\n"
            "- 1 to 8 bullets preserving default values, model names, ports, paths, and env vars.\n"
            "BUILD_OR_TEST_COMMANDS\n"
            "- 0 to 8 bullets naming commands, scripts, suites, and required arguments.\n"
            "FEATURE_FLAGS\n"
            "- 0 to 8 bullets naming toggles, thresholds, timeouts, and behavior switches.\n"
            "ROUTING_OR_TIMEOUTS\n"
            "- 0 to 6 bullets preserving routing thresholds, retry limits, and timeout budgets.\n"
            "CONFIG_RISKS\n"
            "- 0 to 5 bullets naming conflicting, missing, or dangerous config assumptions."
        ),
        payload_label="configs",
        payload=configs,
    )


def build_gemini_pr_thread_digest_prompt(
    thread_or_review: str,
    focus: str = "actionable unresolved work",
) -> str:
    return _wrap_prompt(
        task="Compress PR comments, review threads, issue discussion, or CI summaries for GPT/Codex.",
        focus=focus,
        format_rules=(
            "REQUESTED_CHANGES\n"
            "- 0 to 8 bullets with concrete requested edits, reviewer names if visible, and exact file or symbol mentions.\n"
            "BLOCKERS\n"
            "- 0 to 6 bullets naming failing checks, unresolved questions, merge blockers, or missing evidence.\n"
            "DECISIONS_ALREADY_MADE\n"
            "- 0 to 6 bullets preserving accepted decisions so GPT does not reopen them.\n"
            "FILES_OR_SYMBOLS_MENTIONED\n"
            "- 0 to 10 bullets naming referenced files, functions, commands, jobs, or errors.\n"
            "NEXT_ACTIONS\n"
            "- 1 to 6 bullets with the smallest concrete actions GPT/Codex should verify."
        ),
        payload_label="thread_or_review",
        payload=thread_or_review,
    )


def build_gemini_context_pack_prompt(
    sources: str,
    task: str,
    focus: str = "facts GPT needs before editing",
) -> str:
    return _wrap_prompt(
        task="Consolidate mixed task context into the minimal fact pack GPT/Codex needs before editing.",
        focus=focus,
        extra_context=f"Task: {task}",
        format_rules=(
            "TASK_RELEVANT_FACTS\n"
            "- 3 to 10 bullets preserving concrete names, values, files, behaviors, and constraints.\n"
            "CONSTRAINTS\n"
            "- 1 to 8 bullets naming explicit user, repo, runtime, privacy, or compatibility constraints.\n"
            "KNOWN_DECISIONS\n"
            "- 0 to 8 bullets preserving decisions already made or defaults already chosen.\n"
            "OPEN_QUESTIONS\n"
            "- 0 to 5 bullets for missing facts that are truly not visible in the supplied sources.\n"
            "MINIMAL_CONTEXT_FOR_GPT\n"
            "- 3 to 8 bullets written as compact context GPT can rely on while still verifying."
        ),
        payload_label="sources",
        payload=sources,
    )


@mcp.tool()
async def gemini_compress_diff(
    diff: str,
    focus: str = "bugs, regressions, API surface changes",
) -> str:
    """
    Compress a git diff through the Antigravity Gemini Flash route and write
    saved-token telemetry to the shared local MCP ledger when capture is enabled.
    """
    token_count = server.estimate_tokens(diff)
    return await server.capture_tool_call(
        "gemini_compress_diff",
        _agy_model_name(),
        {"diff": diff, "diff_tokens": token_count, "focus": focus, "routed_to": "antigravity_flash"},
        lambda: server.run_local_analysis(
            lambda: server.ask_agy_diff_compression(diff, focus),
            timeout=server.AGY_DIFF_TOTAL_TIMEOUT,
        ),
    )


@mcp.tool()
async def gemini_summarize_context(
    text: str,
    focus: str = "important implementation details",
) -> str:
    """Compress large context into dense bullets for GPT/Codex to inspect."""
    return await _call_context_reducer(
        tool_name="gemini_summarize_context",
        input_payload={"text": text, "text_tokens": server.estimate_tokens(text), "focus": focus},
        prompt=build_gemini_summarize_context_prompt(text, focus),
    )


@mcp.tool()
async def gemini_debug_digest(
    logs_or_error: str,
    symptoms: str = "",
    focus: str = "root cause and next checks",
) -> str:
    """Extract likely causes, exact errors, transitions, and next checks."""
    return await _call_context_reducer(
        tool_name="gemini_debug_digest",
        input_payload={
            "logs_or_error": logs_or_error,
            "logs_or_error_tokens": server.estimate_tokens(logs_or_error),
            "symptoms": symptoms,
            "focus": focus,
        },
        prompt=build_gemini_debug_digest_prompt(logs_or_error, symptoms, focus),
    )


@mcp.tool()
async def gemini_plan_task(
    task: str,
    context: str = "",
    constraints: str = "",
) -> str:
    """Produce a compact implementation plan for GPT/Codex to verify."""
    return await _call_context_reducer(
        tool_name="gemini_plan_task",
        input_payload={
            "task": task,
            "context": context,
            "constraints": constraints,
            "task_tokens": server.estimate_tokens(task),
            "context_tokens": server.estimate_tokens(context),
        },
        prompt=build_gemini_plan_task_prompt(task, context, constraints),
    )


@mcp.tool()
async def gemini_review_diff(
    diff: str,
    focus: str = "bugs, regressions, missing tests",
) -> str:
    """Return high-confidence review findings for GPT/Codex to verify."""
    return await _call_context_reducer(
        tool_name="gemini_review_diff",
        input_payload={"diff": diff, "diff_tokens": server.estimate_tokens(diff), "focus": focus},
        prompt=build_gemini_review_diff_prompt(diff, focus),
    )


@mcp.tool()
async def gemini_test_plan(
    code_or_diff: str,
    framework: str = "unknown",
    focus: str = "edge cases and regressions",
) -> str:
    """Produce concise test scenarios, not executable test code."""
    return await _call_context_reducer(
        tool_name="gemini_test_plan",
        input_payload={
            "code_or_diff": code_or_diff,
            "code_or_diff_tokens": server.estimate_tokens(code_or_diff),
            "framework": framework,
            "focus": focus,
        },
        prompt=build_gemini_test_plan_prompt(code_or_diff, framework, focus),
    )


@mcp.tool()
async def gemini_repo_map_digest(
    project_tree: str,
    focus: str = "implementation-relevant structure",
) -> str:
    """Compress repo trees or file lists into GPT-ready project orientation."""
    return await _call_context_reducer(
        tool_name="gemini_repo_map_digest",
        input_payload={
            "project_tree": project_tree,
            "project_tree_tokens": server.estimate_tokens(project_tree),
            "focus": focus,
        },
        prompt=build_gemini_repo_map_digest_prompt(project_tree, focus),
    )


@mcp.tool()
async def gemini_symbol_contract_digest(
    signatures_or_types: str,
    language: str = "unknown",
    focus: str = "public interfaces",
) -> str:
    """Compress signatures, types, schemas, or exported symbols for GPT/Codex."""
    return await _call_context_reducer(
        tool_name="gemini_symbol_contract_digest",
        input_payload={
            "signatures_or_types": signatures_or_types,
            "signatures_or_types_tokens": server.estimate_tokens(signatures_or_types),
            "language": language,
            "focus": focus,
        },
        prompt=build_gemini_symbol_contract_digest_prompt(signatures_or_types, language, focus),
    )


@mcp.tool()
async def gemini_config_surface_digest(
    configs: str,
    focus: str = "runtime and build behavior",
) -> str:
    """Compress manifests, env docs, CI snippets, and tool settings."""
    return await _call_context_reducer(
        tool_name="gemini_config_surface_digest",
        input_payload={
            "configs": configs,
            "configs_tokens": server.estimate_tokens(configs),
            "focus": focus,
        },
        prompt=build_gemini_config_surface_digest_prompt(configs, focus),
    )


@mcp.tool()
async def gemini_pr_thread_digest(
    thread_or_review: str,
    focus: str = "actionable unresolved work",
) -> str:
    """Compress PR comments, issue discussion, or CI summaries into actions."""
    return await _call_context_reducer(
        tool_name="gemini_pr_thread_digest",
        input_payload={
            "thread_or_review": thread_or_review,
            "thread_or_review_tokens": server.estimate_tokens(thread_or_review),
            "focus": focus,
        },
        prompt=build_gemini_pr_thread_digest_prompt(thread_or_review, focus),
    )


@mcp.tool()
async def gemini_context_pack(
    sources: str,
    task: str,
    focus: str = "facts GPT needs before editing",
) -> str:
    """Consolidate mixed task context into the smallest GPT-ready fact pack."""
    return await _call_context_reducer(
        tool_name="gemini_context_pack",
        input_payload={
            "sources": sources,
            "sources_tokens": server.estimate_tokens(sources),
            "task": task,
            "task_tokens": server.estimate_tokens(task),
            "focus": focus,
        },
        prompt=build_gemini_context_pack_prompt(sources, task, focus),
    )


if __name__ == "__main__":
    mcp.run()
