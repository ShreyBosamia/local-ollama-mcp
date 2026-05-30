#!/usr/bin/env python3
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import time
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("local-ollama")
ROUTE_FOOTER_TOOLS = {
    "local_summarize",
    "local_code_review",
    "local_test_ideas",
    "local_reason_check",
    "local_plan_check",
    "agy_compress_diff",
    "gemini_compress_diff",
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
    "local_generate_walkthrough",
}

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CODE_MODEL = os.getenv("LOCAL_CODE_MODEL", "qwen3.5:9b")
PLAN_MODEL = os.getenv("LOCAL_PLAN_MODEL", "qwen3.5:9b")
REASON_MODEL = os.getenv("LOCAL_REASON_MODEL", "deepseek-r1:8b")
WARM_MODEL = os.getenv("LOCAL_WARM_MODEL", CODE_MODEL)
DEFAULT_KEEP_ALIVE = os.getenv("LOCAL_KEEP_ALIVE", "2h")
DEFAULT_NUM_CTX = int(os.getenv("LOCAL_NUM_CTX", "4096"))
EXTENDED_NUM_CTX = int(os.getenv("LOCAL_EXTENDED_NUM_CTX", "6144"))
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("LOCAL_OLLAMA_TIMEOUT", "180"))
OLLAMA_HTTP_TIMEOUT = httpx.Timeout(
    OLLAMA_TIMEOUT_SECONDS,
    connect=float(os.getenv("LOCAL_OLLAMA_CONNECT_TIMEOUT", "5")),
)
LOCAL_ANALYSIS_TIMEOUT = float(os.getenv("LOCAL_ANALYSIS_TIMEOUT", "60"))
AGY_FALLBACK_TIMEOUT = float(os.getenv("AGY_FALLBACK_TIMEOUT", "35"))
AGY_TOTAL_TIMEOUT = float(os.getenv("AGY_TOTAL_TIMEOUT", "85"))
LOCAL_STATUS_TIMEOUT = float(os.getenv("LOCAL_STATUS_TIMEOUT", "20"))
LOCAL_STATUS_LEDGER_ROWS = int(os.getenv("LOCAL_STATUS_LEDGER_ROWS", "1000"))

# ---------------------------------------------------------------------------
# Antigravity CLI (agy) — per-task routing configuration
# ---------------------------------------------------------------------------
AGY_BIN            = os.getenv("AGY_BIN", str(Path.home() / ".local" / "bin" / "agy"))
# Flash: fast structural extraction — summarize, code_review
AGY_FLASH_MODEL    = os.getenv("AGY_FLASH_MODEL", "gemini-3.5-flash-high")
AGY_FLASH_TIMEOUT  = float(os.getenv("AGY_FLASH_TIMEOUT", "30"))
# Large diff compression prompts routinely exceed 200k characters. Keep the
# small Flash helpers fast, but give this path enough room to finish for real
# Gemini telemetry verification.
AGY_DIFF_COMPRESS_TIMEOUT = float(os.getenv("AGY_DIFF_COMPRESS_TIMEOUT", "180"))
AGY_DIFF_TOTAL_TIMEOUT = float(os.getenv("AGY_DIFF_TOTAL_TIMEOUT", "420"))
AGY_DIFF_CHUNK_TIMEOUT = float(os.getenv("AGY_DIFF_CHUNK_TIMEOUT", "90"))
AGY_DIFF_CHUNK_ATTEMPTS = int(os.getenv("AGY_DIFF_CHUNK_ATTEMPTS", "2"))
AGY_DIFF_DIRECT_PROMPT_TOKENS = int(os.getenv("AGY_DIFF_DIRECT_PROMPT_TOKENS", "5500"))
AGY_DIFF_SKETCH_TOKENS = int(os.getenv("AGY_DIFF_SKETCH_TOKENS", "3500"))
# Thinking: deep semantic analysis — reason_check, plan_check
AGY_THINK_MODEL    = os.getenv("AGY_THINK_MODEL", "claude-sonnet-4.6-thinking")
AGY_THINK_TIMEOUT  = float(os.getenv("AGY_THINK_TIMEOUT", "60"))
# Routing threshold: payloads >= this token count bypass local GPU and go to agy
AGY_ROUTING_MIN_TOKENS = int(os.getenv("AGY_ROUTING_MIN", "4000"))
AGY_ENABLED        = os.getenv("AGY_ENABLED", "1") == "1"

# Quota monitoring & auto-switching configs
AGY_THINK_THRESHOLD = float(os.getenv("AGY_THINK_THRESHOLD", "15"))  # % remaining limit
AGY_QUOTA_CACHE_TTL = float(os.getenv("AGY_QUOTA_CACHE_TTL", "600"))  # 10 minutes (600s)
AGY_USAGE_TIMEOUT   = float(os.getenv("AGY_USAGE_TIMEOUT", "20"))     # timeout for agy /usage
PROCESS_KILL_WAIT_TIMEOUT = float(os.getenv("PROCESS_KILL_WAIT_TIMEOUT", "2"))

# Preserved original thinking model for recovery path
ORIGINAL_AGY_THINK_MODEL = AGY_THINK_MODEL

# In-memory quota cache with pre-populated healthy defaults (ensures zero cold-start delay)
_quota_cache = {
    "Claude Sonnet 4.6 (Thinking)": 100.0,
    "Claude Opus 4.6 (Thinking)": 100.0,
    "Gemini 3.5 Flash (High)": 100.0,
}
_quota_cache_ts: float = 0.0
_quota_task: asyncio.Task | None = None


@dataclass(frozen=True)
class ToolPromptSpec:
    tool_name: str
    model: str
    max_items: int
    num_predict: int
    temperature: float
    keep_alive: str | None
    system: str
    template: str
    unload_warm_model_first: bool = False


STRICT_HELPER_SYSTEM = """
You are a local MCP compression tool for Codex.
CRITICAL OUTPUT RULES:
- Do not include chain-of-thought or hidden reasoning.
- Do not include <think> tags.
- Do not write code blocks unless the user explicitly asks for code.
- Obey the exact item limit. Never exceed it.
- Stop immediately after the final allowed item.
- Return only the requested output.
""".strip()


TOOL_PROMPTS = {
    "local_summarize": ToolPromptSpec(
        tool_name="local_summarize",
        model=CODE_MODEL,
        max_items=6,
        num_predict=220,
        temperature=0.1,
        keep_alive=None,
        system=STRICT_HELPER_SYSTEM,
        template="""
<task>
Summarize this content for Codex.
</task>
<focus>
{focus}
</focus>
<format>
Return 1 to 6 bullets total.
Each bullet must be one short sentence.
Never write more than 6 bullets.
Do not add headings, prefaces, or conclusions.
Preserve exact error strings, option names, and numeric values when visible.
</format>
<negative_constraints>
Do not quote large blocks.
Do not invent files, functions, or risks.
Do not mention generic advice.
</negative_constraints>
<content>
{text}
</content>
""".strip(),
    ),
    "local_code_review": ToolPromptSpec(
        tool_name="local_code_review",
        model=CODE_MODEL,
        max_items=5,
        num_predict=260,
        temperature=0.1,
        keep_alive=None,
        system=STRICT_HELPER_SYSTEM,
        template="""
<task>
Review this git diff for likely bugs, regressions, or missing tests.
</task>
<focus>
{focus}
</focus>
<format>
Return 0 to 5 findings total.
Never write more than 5 findings.
Each finding must be exactly one bullet with: issue; evidence; suggested check.
If no meaningful issue exists, return exactly: No obvious issue found.
Report only high-confidence behavior regressions.
</format>
<negative_constraints>
Do not fixate on harmless refactors or obvious intentional clamps.
Do not report a newly added lower-bound or upper-bound clamp unless the diff shows it violates documented behavior.
Do not discuss style-only changes.
Do not produce patches.
Do not create nested bullets.
</negative_constraints>
<diff>
{diff}
</diff>
""".strip(),
    ),
    "local_test_ideas": ToolPromptSpec(
        tool_name="local_test_ideas",
        model=CODE_MODEL,
        max_items=7,
        num_predict=220,
        temperature=0.1,
        keep_alive=None,
        system=STRICT_HELPER_SYSTEM,
        template="""
<task>
Generate concise test ideas for Codex.
</task>
<framework>
{test_framework}
</framework>
<format>
Return 1 to 7 bullets total.
Each bullet must be one sentence describing a test scenario.
Never write executable test code.
Never write imports, fixtures, or code fences.
</format>
<negative_constraints>
Do not output pytest scripts.
Do not output function definitions.
Do not include setup boilerplate.
</negative_constraints>
<code_or_diff>
{code_or_diff}
</code_or_diff>
""".strip(),
    ),
    "local_reason_check": ToolPromptSpec(
        tool_name="local_reason_check",
        model=REASON_MODEL,
        max_items=5,
        num_predict=220,
        temperature=0.1,
        keep_alive="0",
        system=STRICT_HELPER_SYSTEM,
        template="""
<task>
Give a concise second opinion for this debugging or problem-solving task.
</task>
<format>
Return 1 to 5 bullets total.
Each bullet must be one short conclusion or next check.
Never write more than 5 bullets.
</format>
<negative_constraints>
Do not include hidden reasoning.
Do not include <think> tags.
Do not write long explanations.
</negative_constraints>
<problem>
{problem}
</problem>
""".strip(),
        unload_warm_model_first=True,
    ),
    "local_plan_check": ToolPromptSpec(
        tool_name="local_plan_check",
        model=PLAN_MODEL,
        max_items=6,
        num_predict=260,
        temperature=0.2,
        keep_alive="0",
        system=STRICT_HELPER_SYSTEM,
        template="""
<task>
Create a compact implementation plan for Codex.
</task>
<format>
Return 1 to 6 bullets total.
Each bullet must be one concrete action, risk check, or verification step.
Never write more than 6 bullets.
</format>
<negative_constraints>
Do not include hidden reasoning.
Do not include <think> tags.
Do not write a preface or conclusion.
</negative_constraints>
<problem>
{problem}
</problem>
""".strip(),
        unload_warm_model_first=True,
    ),
}


def configure_models(
    *,
    code_model: str | None = None,
    plan_model: str | None = None,
    reason_model: str | None = None,
    warm_model: str | None = None,
) -> None:
    """Update runtime model constants and prompt specs together."""
    global CODE_MODEL, PLAN_MODEL, REASON_MODEL, WARM_MODEL

    if code_model:
        CODE_MODEL = code_model
    if plan_model:
        PLAN_MODEL = plan_model
    if reason_model:
        REASON_MODEL = reason_model
    if warm_model:
        WARM_MODEL = warm_model
    elif code_model and WARM_MODEL != code_model:
        WARM_MODEL = CODE_MODEL

    TOOL_PROMPTS["local_summarize"] = replace(
        TOOL_PROMPTS["local_summarize"], model=CODE_MODEL
    )
    TOOL_PROMPTS["local_code_review"] = replace(
        TOOL_PROMPTS["local_code_review"], model=CODE_MODEL
    )
    TOOL_PROMPTS["local_test_ideas"] = replace(
        TOOL_PROMPTS["local_test_ideas"], model=CODE_MODEL
    )
    TOOL_PROMPTS["local_plan_check"] = replace(
        TOOL_PROMPTS["local_plan_check"], model=PLAN_MODEL
    )
    TOOL_PROMPTS["local_reason_check"] = replace(
        TOOL_PROMPTS["local_reason_check"], model=REASON_MODEL
    )

THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
UNCLOSED_THINK_RE = re.compile(r"^\s*<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
THINK_MARKER_RE = re.compile(r"</?think\b", re.IGNORECASE)
TOP_LEVEL_BULLET_RE = re.compile(r"^(?:[-*+]|\d+[.)])\s+(.*)$")
ANY_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")
CODE_FENCE_RE = re.compile(r"^\s*```")
TEST_CODE_LINE_RE = re.compile(r"^\s*(?:from\s+\S+\s+import\s+|import\s+\S+|def\s+test_)", re.IGNORECASE)
CLAMP_FALSE_POSITIVE_RE = re.compile(
    r"(?:apply_discount|discount).*(?:clamp|max\s*\(\s*0|negative discount)|"
    r"(?:clamp|max\s*\(\s*0|negative discount).*(?:apply_discount|discount)",
    re.IGNORECASE | re.DOTALL,
)
SECRET_PATTERNS = (
    re.compile(r"\bsk-proj-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)
ASSIGNMENT_SECRET_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|PASS|AUTH|CREDENTIAL)[A-Z0-9_]*)"
    r"(\s*[:=]\s*)([^\s,;\"']+)"
)
JSON_SECRET_RE = re.compile(
    r"(?i)(\"[^\"]*(?:api[_-]?key|token|secret|password|credential)[^\"]*\"\s*:\s*\")([^\"]+)(\")"
)
OUTCOME_LABELS = {
    "useful",
    "needs_raw_verification",
    "misleading",
    "too_verbose",
    "skip_for_small_context",
}
LOCAL_ROUTING_MIN_TOKENS = 120
LOCAL_ROUTING_MAX_TOKENS = 4000


def strip_thinking(content: str) -> str:
    """Remove hidden-reasoning blocks that some local models leak in content."""
    content = THINK_BLOCK_RE.sub("", content)
    content = UNCLOSED_THINK_RE.sub("", content)
    return content.strip()


def collapse_bullets(content: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    saw_top_level = False
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        top_match = TOP_LEVEL_BULLET_RE.match(line)
        if top_match:
            saw_top_level = True
            if current:
                items.append(" ".join(current).strip())
            current = [top_match.group(1).strip()]
            continue
        nested_match = ANY_BULLET_RE.match(line)
        if nested_match and current:
            current.append(nested_match.group(1).strip())
            continue
        if current:
            current.append(stripped)
    if current:
        items.append(" ".join(current).strip())
    return items if saw_top_level else []


def enforce_tool_output(tool_name: str | None, content: str) -> str:
    content = strip_thinking(content)
    if not tool_name or tool_name not in TOOL_PROMPTS:
        return content

    spec = TOOL_PROMPTS[tool_name]
    if tool_name == "local_test_ideas":
        content = "\n".join(
            line
            for line in content.splitlines()
            if not CODE_FENCE_RE.match(line) and not TEST_CODE_LINE_RE.match(line)
        ).strip()
    if content == "No obvious issue found.":
        return content

    items = collapse_bullets(content)
    if not items:
        return content
    if tool_name == "local_code_review" and len(items) > 1:
        filtered = [item for item in items if not CLAMP_FALSE_POSITIVE_RE.search(item)]
        if filtered:
            items = filtered
    items = items[: spec.max_items]
    return "\n".join(f"- {item}" for item in items).strip()


def ns_to_ms(duration_ns: Any) -> str:
    if not isinstance(duration_ns, int):
        return "n/a"
    return f"{duration_ns / 1_000_000:.0f} ms"


def capture_enabled() -> bool:
    return os.getenv("LOCAL_MCP_CAPTURE") == "1"


def raw_capture_enabled() -> bool:
    return os.getenv("LOCAL_MCP_CAPTURE_RAW") == "1"


def ledger_path() -> Path:
    return Path(os.getenv("LOCAL_MCP_LEDGER_PATH", ".local_ollama_mcp/ledger.jsonl"))


def estimate_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def split_into_chunks(text: str, max_chunk_tokens: int) -> list[str]:
    if max_chunk_tokens <= 0:
        raise ValueError("max_chunk_tokens must be positive")
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for line in text.splitlines(keepends=True):
        line_tokens = estimate_tokens(line)
        if current and current_tokens + line_tokens > max_chunk_tokens:
            chunks.append("".join(current))
            current = []
            current_tokens = 0
        current.append(line)
        current_tokens += line_tokens
    if current:
        chunks.append("".join(current))
    return chunks


def redact_text(text: str) -> tuple[str, list[str]]:
    redacted = text
    flags: list[str] = []

    home = str(Path.home())
    if home and home in redacted:
        redacted = redacted.replace(home, "[HOME]")
        flags.append("home_path_redacted")

    for pattern in SECRET_PATTERNS:
        redacted, count = pattern.subn("[REDACTED_SECRET]", redacted)
        if count:
            flags.append("secret_pattern_redacted")

    redacted, count = ASSIGNMENT_SECRET_RE.subn(r"\1\2[REDACTED_SECRET]", redacted)
    if count:
        flags.append("secret_assignment_redacted")

    redacted, count = JSON_SECRET_RE.subn(r"\1[REDACTED_SECRET]\3", redacted)
    if count:
        flags.append("json_secret_redacted")

    for key, value in os.environ.items():
        if (
            value
            and len(value) >= 8
            and re.search(r"(?i)(TOKEN|SECRET|PASSWORD|PASS|KEY|CREDENTIAL|AUTH)", key)
            and value in redacted
        ):
            redacted = redacted.replace(value, "[REDACTED_ENV_VALUE]")
            flags.append("env_value_redacted")

    return redacted, sorted(set(flags))


def redact_value(value: Any) -> tuple[Any, list[str]]:
    if raw_capture_enabled():
        return value, []
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        redacted_items = []
        flags: list[str] = []
        for item in value:
            redacted_item, item_flags = redact_value(item)
            redacted_items.append(redacted_item)
            flags.extend(item_flags)
        return redacted_items, sorted(set(flags))
    if isinstance(value, tuple):
        redacted_items, flags = redact_value(list(value))
        return redacted_items, flags
    if isinstance(value, dict):
        redacted_dict: dict[str, Any] = {}
        flags: list[str] = []
        for key, item in value.items():
            redacted_item, item_flags = redact_value(item)
            redacted_dict[str(key)] = redacted_item
            flags.extend(item_flags)
        return redacted_dict, sorted(set(flags))
    return value, []


def compact_json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)


def recommendation_for_capture(
    *,
    input_tokens: int,
    output_tokens: int,
    risk_flags: list[str],
) -> str:
    if "tool_error" in risk_flags or output_tokens == 0:
        return "raw_cloud"
    if "think_leak" in risk_flags:
        return "verify_raw"
    if input_tokens < LOCAL_ROUTING_MIN_TOKENS:
        return "skip_local"
    if input_tokens > LOCAL_ROUTING_MAX_TOKENS:
        return "raw_cloud"

    reduction_pct = (input_tokens - output_tokens) / input_tokens if input_tokens else 0.0
    if reduction_pct >= 0.40:
        return "use_local"
    if reduction_pct > 0:
        return "verify_raw"
    return "raw_cloud"


def confidence_score_for_capture(
    *,
    input_tokens: int,
    output_tokens: int,
    risk_flags: list[str],
) -> float:
    if input_tokens <= 0 or "tool_error" in risk_flags or output_tokens == 0:
        return 0.0
    reduction_pct = max(0.0, (input_tokens - output_tokens) / input_tokens)
    score = min(1.0, 0.45 + reduction_pct)
    if "think_leak" in risk_flags:
        score -= 0.45
    if "empty_output" in risk_flags:
        score -= 0.45
    return round(max(0.0, score), 3)


def append_ledger_record(record: dict[str, Any]) -> None:
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str, ensure_ascii=False) + "\n")


def is_successful_gemini_route(model: str, route_outcome: str | None) -> bool:
    model_lower = model.lower()
    return (
        model_lower.startswith("antigravity/")
        and "gemini" in model_lower
        and route_outcome == "agy-default-gemini"
    )


def gemini_token_savings_fields(
    *,
    model: str,
    route_outcome: str | None,
    input_tokens: int,
    output_tokens: int,
) -> dict[str, int | float] | None:
    if not is_successful_gemini_route(model, route_outcome):
        return None
    saved_tokens = max(0, input_tokens - output_tokens)
    saved_ratio = round(saved_tokens / input_tokens, 3) if input_tokens else 0.0
    return {
        "gemini_input_tokens_est": input_tokens,
        "gemini_output_tokens_est": output_tokens,
        "gemini_saved_tokens_est": saved_tokens,
        "gemini_saved_pct": saved_ratio,
    }


def gemini_token_savings_line(fields: dict[str, int | float]) -> str:
    saved_pct = round(float(fields["gemini_saved_pct"]) * 100)
    return (
        "token_savings: "
        f"gemini_input_est={int(fields['gemini_input_tokens_est'])} "
        f"gpt_payload_est={int(fields['gemini_output_tokens_est'])} "
        f"gpt_saved_est={int(fields['gemini_saved_tokens_est'])} "
        f"saved_pct={saved_pct}%"
    )


def build_tool_record(
    *,
    task_id: str,
    tool_name: str,
    model: str,
    input_payload: dict[str, Any],
    local_output: Any,
    latency_ms: int,
    error: str | None = None,
    route_outcome: str | None = None,
    output_for_token_estimate: Any | None = None,
    record_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stored_input, input_flags = redact_value(input_payload)
    stored_output, output_flags = redact_value(local_output)
    stored_error, error_flags = redact_value(error) if error else (None, [])
    input_text = compact_json_text(stored_input)
    output_text = str(output_for_token_estimate if output_for_token_estimate is not None else stored_output)
    input_tokens = estimate_tokens(input_text)
    output_tokens = estimate_tokens(output_text)
    risk_flags = sorted(set(input_flags + output_flags + error_flags))
    if THINK_MARKER_RE.search(output_text):
        risk_flags.append("think_leak")
    if not output_text.strip():
        risk_flags.append("empty_output")
    if error:
        risk_flags.append("tool_error")

    risk_flags = sorted(set(risk_flags))
    routing_decision = recommendation_for_capture(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        risk_flags=risk_flags,
    )
    cloud_tokens_avoided = max(0, input_tokens - output_tokens) if routing_decision == "use_local" else 0
    token_estimates: dict[str, Any] = {
        "input": input_tokens,
        "local_output": output_tokens,
        "context_reduction": input_tokens - output_tokens,
        "context_reduction_pct": round((input_tokens - output_tokens) / input_tokens, 3)
        if input_tokens
        else 0.0,
    }
    gemini_savings = gemini_token_savings_fields(
        model=model,
        route_outcome=route_outcome,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    if gemini_savings:
        token_estimates.update(
            {
                "gemini_input": gemini_savings["gemini_input_tokens_est"],
                "gemini_output": gemini_savings["gemini_output_tokens_est"],
                "gemini_saved": gemini_savings["gemini_saved_tokens_est"],
                "gemini_saved_pct": gemini_savings["gemini_saved_pct"],
            }
        )

    record = {
        "record_type": "tool_call",
        "task_id": task_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "model": model,
        "privacy_mode": "raw" if raw_capture_enabled() else "redacted",
        "input": stored_input,
        "local_output": stored_output,
        "latency_ms": latency_ms,
        "route_outcome": route_outcome,
        "token_estimates": token_estimates,
        "recommendation": routing_decision,
        "routing_decision": routing_decision,
        "confidence_score": confidence_score_for_capture(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            risk_flags=risk_flags,
        ),
        "artifact_tokens_est": input_tokens,
        "local_output_tokens_est": output_tokens,
        "cloud_tokens_avoided_est": cloud_tokens_avoided,
        "risk_flags": risk_flags,
        "error": stored_error,
    }
    if gemini_savings:
        record.update(
            {
                "gemini_input_tokens_est": gemini_savings["gemini_input_tokens_est"],
                "gemini_output_tokens_est": gemini_savings["gemini_output_tokens_est"],
                "gemini_saved_tokens_est": gemini_savings["gemini_saved_tokens_est"],
            }
        )
    if record_metadata:
        safe_metadata, metadata_flags = redact_value(record_metadata)
        if metadata_flags:
            record["risk_flags"] = sorted(set(record["risk_flags"] + metadata_flags))
        record.update(safe_metadata)
    return record


async def capture_tool_call(
    tool_name: str,
    model: str,
    input_payload: dict[str, Any],
    action: Callable[[], Awaitable[str]],
    record_metadata: dict[str, Any] | None = None,
    record_output_for_token_estimate: Callable[[str], Any] | None = None,
) -> str:
    if not capture_enabled():
        output = await action()
        outcome = await determine_route_outcome(tool_name, model, output)
        if tool_name in ROUTE_FOOTER_TOOLS:
            savings = gemini_token_savings_fields(
                model=model,
                route_outcome=outcome,
                input_tokens=estimate_tokens(compact_json_text(input_payload)),
                output_tokens=estimate_tokens(output),
            )
            output = f"{output}\nroute_outcome: {outcome}"
            if savings:
                output = f"{output}\n{gemini_token_savings_line(savings)}"
        return output

    task_id = uuid4().hex
    started = time.perf_counter()
    try:
        raw_output = await action()
    except Exception as exc:
        latency_ms = round((time.perf_counter() - started) * 1000)
        error_text = f"{type(exc).__name__}: {exc}"
        outcome = await determine_route_outcome(tool_name, model, error_text)
        append_ledger_record(
            build_tool_record(
                task_id=task_id,
                tool_name=tool_name,
                model=model,
                input_payload=input_payload,
                local_output="",
                latency_ms=latency_ms,
                error=error_text,
                route_outcome=outcome,
                record_metadata=record_metadata,
            )
        )
        raise

    latency_ms = round((time.perf_counter() - started) * 1000)
    outcome = await determine_route_outcome(tool_name, model, raw_output)

    output = raw_output
    if tool_name in ROUTE_FOOTER_TOOLS:
        savings = gemini_token_savings_fields(
            model=model,
            route_outcome=outcome,
            input_tokens=estimate_tokens(compact_json_text(input_payload)),
            output_tokens=estimate_tokens(raw_output),
        )
        output = f"{output}\nroute_outcome: {outcome}"
        if savings:
            output = f"{output}\n{gemini_token_savings_line(savings)}"

    append_ledger_record(
        build_tool_record(
            task_id=task_id,
            tool_name=tool_name,
            model=model,
            input_payload=input_payload,
            local_output=output,
            latency_ms=latency_ms,
            route_outcome=outcome,
            output_for_token_estimate=record_output_for_token_estimate(raw_output)
            if record_output_for_token_estimate
            else raw_output,
            record_metadata=record_metadata,
        )
    )
    return output


def read_ledger_records(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or ledger_path()
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def read_recent_ledger_records(
    path: Path | None = None,
    *,
    max_rows: int | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    path = path or ledger_path()
    if not path.exists():
        return [], False
    max_rows = max_rows or LOCAL_STATUS_LEDGER_ROWS

    raw_lines: list[str] = []
    truncated = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(raw_lines) >= max_rows:
                raw_lines.pop(0)
                truncated = True
            raw_lines.append(line)

    records: list[dict[str, Any]] = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records, truncated


async def run_with_tool_timeout(
    action: Callable[[], Awaitable[str]],
    *,
    timeout: float,
    timeout_prefix: str,
    timeout_message: str,
) -> str:
    try:
        return await asyncio.wait_for(action(), timeout=timeout)
    except asyncio.TimeoutError:
        return f"{timeout_prefix} timed out after {timeout:.0f}s. {timeout_message}"


async def run_local_analysis(
    action: Callable[[], Awaitable[str]],
    *,
    timeout: float | None = None,
) -> str:
    timeout = timeout or LOCAL_ANALYSIS_TIMEOUT
    return await run_with_tool_timeout(
        action,
        timeout=timeout,
        timeout_prefix="[local_timeout]",
        timeout_message="Handle this payload directly.",
    )


async def run_status_tool(action: Callable[[], Awaitable[str]]) -> str:
    return await run_with_tool_timeout(
        action,
        timeout=LOCAL_STATUS_TIMEOUT,
        timeout_prefix="[status_timeout]",
        timeout_message="Retry later or inspect the underlying command directly.",
    )


async def kill_process(process: asyncio.subprocess.Process | None) -> None:
    if process is None:
        return
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.communicate(), timeout=PROCESS_KILL_WAIT_TIMEOUT)
        except Exception:
            try:
                await asyncio.wait_for(process.wait(), timeout=PROCESS_KILL_WAIT_TIMEOUT)
            except Exception:
                pass
    transport = getattr(process, "_transport", None)
    if transport is not None:
        try:
            transport.close()
        except Exception:
            pass


async def terminate_process(process: asyncio.subprocess.Process | None) -> None:
    if process is None:
        return
    if process.returncode is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.communicate(), timeout=PROCESS_KILL_WAIT_TIMEOUT)
        except Exception:
            await kill_process(process)
    else:
        await kill_process(process)


async def run_command(*args: str, timeout: float = 10) -> tuple[bool, str]:
    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except FileNotFoundError:
        return False, f"{args[0]} not found"
    except asyncio.TimeoutError:
        await kill_process(process)
        return False, f"{args[0]} timed out after {timeout:.0f}s"

    output = (stdout.decode(errors="replace") + stderr.decode(errors="replace")).strip()
    return process.returncode == 0, output


# ---------------------------------------------------------------------------
# Antigravity CLI (agy) helpers — subprocess handler + circuit breaker
# ---------------------------------------------------------------------------

class _Tier:
    """Routing tier tokens for the three-tier hybrid mesh."""
    LOCAL = "local_gpu"
    AGY   = "antigravity"


def select_tier(token_count: int) -> str:
    """
    Deterministic routing decision based on estimated token count.

    < AGY_ROUTING_MIN_TOKENS  → Local GPU  (VRAM-safe, zero network cost)
    >= AGY_ROUTING_MIN_TOKENS → Antigravity CLI  (cloud, zero VRAM cost)

    If AGY_ENABLED is false the function always returns LOCAL so the
    agy layer can be disabled at runtime via env var without code changes.
    """
    ensure_quota_monitor_started()
    if not AGY_ENABLED or token_count < AGY_ROUTING_MIN_TOKENS:
        return _Tier.LOCAL
    return _Tier.AGY


async def check_antigravity_quotas() -> None:
    """
    Executes 'agy /usage' via asyncio.create_subprocess_exec.
    Parses remaining quota percentages and dynamically auto-switches
    the Thinking model if Sonnet Thinking drops below the threshold.
    Logs errors gracefully to the telemetry ledger.
    """
    global AGY_THINK_MODEL, _quota_cache_ts

    cmd = [AGY_BIN, "/usage"]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ},  # pass HOME so agy can find OAuth creds
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=AGY_USAGE_TIMEOUT)

        out = stdout.decode(errors="replace").strip()
        err = stderr.decode(errors="replace").strip()

        if proc.returncode != 0:
            raise RuntimeError(f"agy exited {proc.returncode}: {err or out}")

        combined_output = out + "\n" + err

        # Robust regexes matching model name and remaining percentage
        sonnet_match = re.search(r"Claude Sonnet 4\.6 \(Thinking\)[^\d]*(\d+(?:\.\d+)?)\s*%", combined_output, re.IGNORECASE)
        opus_match = re.search(r"Claude Opus 4\.6 \(Thinking\)[^\d]*(\d+(?:\.\d+)?)\s*%", combined_output, re.IGNORECASE)
        gemini_match = re.search(r"Gemini 3\.5 Flash \(High\)[^\d]*(\d+(?:\.\d+)?)\s*%", combined_output, re.IGNORECASE)

        updated = False
        if sonnet_match:
            _quota_cache["Claude Sonnet 4.6 (Thinking)"] = float(sonnet_match.group(1))
            updated = True
        if opus_match:
            _quota_cache["Claude Opus 4.6 (Thinking)"] = float(opus_match.group(1))
            updated = True
        if gemini_match:
            _quota_cache["Gemini 3.5 Flash (High)"] = float(gemini_match.group(1))
            updated = True

        if updated:
            _quota_cache_ts = time.time()

            # Failover logic
            sonnet_quota = _quota_cache["Claude Sonnet 4.6 (Thinking)"]
            if sonnet_quota < AGY_THINK_THRESHOLD:
                # Proactive auto-switching
                opus_quota = _quota_cache.get("Claude Opus 4.6 (Thinking)", 100.0)
                if opus_quota >= AGY_THINK_THRESHOLD:
                    new_model = "claude-opus-4.6-thinking"
                else:
                    new_model = AGY_FLASH_MODEL  # Fall back to Gemini 3.5 Flash High

                if AGY_THINK_MODEL != new_model:
                    append_ledger_record({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "event_type": "quota_auto_switch",
                        "reason": f"Claude Sonnet 4.6 (Thinking) quota ({sonnet_quota}%) dropped below threshold ({AGY_THINK_THRESHOLD}%)",
                        "previous_model": AGY_THINK_MODEL,
                        "new_model": new_model,
                        "quotas": dict(_quota_cache)
                    })
                    AGY_THINK_MODEL = new_model
            else:
                # Restore to default/configured original model
                if AGY_THINK_MODEL != ORIGINAL_AGY_THINK_MODEL:
                    append_ledger_record({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "event_type": "quota_restore",
                        "reason": f"Claude Sonnet 4.6 (Thinking) quota recovered to {sonnet_quota}% (>= threshold {AGY_THINK_THRESHOLD}%)",
                        "previous_model": AGY_THINK_MODEL,
                        "new_model": ORIGINAL_AGY_THINK_MODEL,
                        "quotas": dict(_quota_cache)
                    })
                    AGY_THINK_MODEL = ORIGINAL_AGY_THINK_MODEL

    except Exception as exc:
        # Never crash the main chat: log to existing telemetry ledger, keep current models
        error_text = f"Quota monitoring check failed: {type(exc).__name__}: {exc}"
        append_ledger_record({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "quota_check_error",
            "error": error_text,
            "current_think_model": AGY_THINK_MODEL,
            "quotas": dict(_quota_cache)
        })


async def _quota_monitor_loop() -> None:
    """Decoupled periodic loop running in the background."""
    while True:
        await check_antigravity_quotas()
        await asyncio.sleep(AGY_QUOTA_CACHE_TTL)


def ensure_quota_monitor_started() -> None:
    """Starts the background monitoring task safely on the running event loop."""
    global _quota_task
    if _quota_task is None or _quota_task.done():
        try:
            loop = asyncio.get_running_loop()
            _quota_task = loop.create_task(_quota_monitor_loop())
        except RuntimeError:
            pass  # Loop not active yet



# Sliding-window circuit breaker — prevents runaway subprocess spawning
# during network outages.  Resets on server restart (in-process state only).
_agy_error_window: list[float] = []
_AGY_CIRCUIT_WINDOW_SECS = 120.0  # 2-minute rolling window
_AGY_CIRCUIT_THRESHOLD   = 3      # open circuit after this many failures
_agy_model_flag_supported: bool | None = None


def _agy_circuit_is_open() -> bool:
    now = time.monotonic()
    _agy_error_window[:] = [
        t for t in _agy_error_window if now - t < _AGY_CIRCUIT_WINDOW_SECS
    ]
    return len(_agy_error_window) >= _AGY_CIRCUIT_THRESHOLD


def _agy_record_failure() -> None:
    _agy_error_window.append(time.monotonic())


async def agy_supports_model_flag() -> bool:
    """Return whether the installed agy binary accepts per-call --model."""
    global _agy_model_flag_supported
    if _agy_model_flag_supported is not None:
        return _agy_model_flag_supported
    proc: asyncio.subprocess.Process | None = None
    try:
        proc = await asyncio.create_subprocess_exec(
            AGY_BIN,
            "help",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ},
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
        combined = stdout.decode(errors="replace") + "\n" + stderr.decode(errors="replace")
        _agy_model_flag_supported = "--model" in combined
    except asyncio.TimeoutError:
        await kill_process(proc)
        _agy_model_flag_supported = False
    except Exception:
        _agy_model_flag_supported = False
    return _agy_model_flag_supported


async def determine_route_outcome(
    tool_name: str,
    model: str,
    output: str,
) -> str:
    """
    Determine the compact route outcome for tool execution.
    Outcomes: local, agy-default-gemini, agy-fallback, unavailable.
    """
    if not model.startswith("antigravity/"):
        return "local"

    output_str = str(output)

    # Check for fallback indicators
    if any(prefix in output_str for prefix in ["[agy_circuit_open]", "[agy_rate_limited]", "[agy_timeout]", "[agy_missing_binary]", "[agy_error]", "[local_timeout]", "fallback_also_failed"]):
        return "agy-fallback"

    # If it's a plan or reason check
    if tool_name in ["local_plan_check", "local_reason_check"]:
        if not await agy_supports_model_flag():
            return "unavailable"

    return "agy-default-gemini"


def get_model_display_name(model_id: str) -> str:
    """Map standard lowercase/hyphenated model IDs to display names in settings.json."""
    lower_id = model_id.lower()
    if "sonnet" in lower_id:
        return "Claude Sonnet 4.6 (Thinking)"
    if "opus" in lower_id:
        return "Claude Opus 4.6 (Thinking)"
    if "gemini" in lower_id:
        return "Gemini 3.5 Flash (High)"
    return model_id


async def prepare_agy_temp_home(model: str) -> str:
    """
    Creates a temporary home directory with symlinks to the real Antigravity
    configuration and a custom settings.json targeting the specified model.
    """
    root_dir = Path(__file__).resolve().parent
    base_tmp_dir = root_dir / ".local_ollama_mcp" / "agy_tmp"
    base_tmp_dir.mkdir(parents=True, exist_ok=True)
    
    unique_id = str(uuid4())
    temp_home = base_tmp_dir / f"home_{unique_id}"
    temp_home.mkdir(parents=True, exist_ok=True)
    
    agy_config_dir = temp_home / ".gemini" / "antigravity-cli"
    agy_config_dir.mkdir(parents=True, exist_ok=True)
    
    real_home_cli = Path.home() / ".gemini" / "antigravity-cli"
    
    items_to_link = [
        "antigravity-oauth-token",
        "bin",
        "conversations",
        "installation_id",
        "implicit",
        "updater",
        "keybindings.json",
        "knowledge",
        "scratch"
    ]
    
    for item in items_to_link:
        real_item = real_home_cli / item
        if real_item.exists():
            link_target = agy_config_dir / item
            try:
                os.symlink(str(real_item), str(link_target))
            except Exception:
                pass
                
    display_model = get_model_display_name(model)
    settings = {
        "model": display_model,
        "notifications": True,
        "statusLine": {
            "type": "",
            "command": "",
            "enabled": True
        },
        "toolPermission": "always-proceed",
        "trustedWorkspaces": [
            "/home/shrey",
            str(root_dir)
        ],
        "verbosity": "low"
    }
    
    settings_file = agy_config_dir / "settings.json"
    with open(settings_file, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
        
    return str(temp_home)


async def cleanup_agy_temp_home(temp_home: str) -> None:
    """Removes the temporary home directory asynchronously."""
    import shutil
    try:
        if os.path.exists(temp_home):
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: shutil.rmtree(temp_home, ignore_errors=True))
    except Exception:
        pass


def agy_print_timeout_arg(timeout: float) -> str:
    """Convert a Python timeout budget to the duration syntax agy expects."""
    return f"{max(1, round(timeout))}s"


def hydrate_agy_file_artifact_response(content: str) -> str:
    """Replace agy file-link responses with the generated local artifact text."""
    if "file://" not in content:
        return content

    allowed_root = (Path.home() / ".gemini" / "antigravity-cli").resolve()
    for raw_uri in re.findall(r"file://[^\s)>\]]+", content):
        parsed = urlparse(raw_uri.rstrip(".,;"))
        if parsed.scheme != "file":
            continue
        try:
            artifact_path = Path(unquote(parsed.path)).resolve()
        except Exception:
            continue
        if allowed_root != artifact_path and allowed_root not in artifact_path.parents:
            continue
        if not artifact_path.is_file() or artifact_path.stat().st_size > 1_000_000:
            continue
        try:
            artifact_text = artifact_path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue
        if artifact_text.startswith("#") and "## Executive Summary" in artifact_text:
            return artifact_text

    return content


async def ask_antigravity(
    prompt: str,
    *,
    model: str = "",
    timeout: float | None = None,
) -> str:
    """
    Run a single prompt through Antigravity CLI in non-interactive print mode.
    Passes --model <model> only when the installed CLI supports that flag.
    Uses the configured default model when --model is not supported.
    Raises RuntimeError on failure, timeout, or missing binary.

    NOTE: This function makes a network call via agy to a cloud endpoint.
    It does not mutate GPU clocks, voltage, fans, or power limits.
    """
    timeout = timeout or AGY_FLASH_TIMEOUT
    cmd: list[str] = [AGY_BIN]
    temp_home = None
    proc: asyncio.subprocess.Process | None = None
    custom_env = {**os.environ}

    if model:
        if await agy_supports_model_flag():
            cmd += ["--model", model]
        # Without a real --model flag, agy can only be confirmed against its
        # configured default. Keep reports honest instead of simulating named
        # per-call routes with a temporary HOME.

    # Use stdin so large walkthrough/diff prompts do not exceed OS argv limits.
    # Pass agy's own print timeout so the child process and Python wrapper share
    # the same budget instead of waiting on the CLI default.
    cmd += ["--print-timeout", agy_print_timeout_arg(timeout), "--print", "-"]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=custom_env,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=prompt.encode("utf-8")),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        await kill_process(proc)
        raise RuntimeError(
            f"agy/{model or 'default'} timed out after {timeout:.0f}s"
        )
    except asyncio.CancelledError:
        await kill_process(proc)
        raise
    except FileNotFoundError:
        raise RuntimeError(f"agy binary not found at {AGY_BIN}")
    finally:
        if temp_home:
            await cleanup_agy_temp_home(temp_home)

    out = stdout.decode(errors="replace").strip()
    err = stderr.decode(errors="replace").strip()

    if proc.returncode != 0:
        snippet = (err or out)[:300]
        raise RuntimeError(
            f"agy/{model or 'default'} exited {proc.returncode}: {snippet}"
        )

    return out or err


async def ask_antigravity_with_fallback(
    prompt: str,
    *,
    model: str = "",
    timeout: float | None = None,
    total_timeout: float | None = None,
    agy_attempts: int = 1,
    fallback_model: str = CODE_MODEL,
    fallback_num_predict: int = 300,
) -> str:
    """
    Try Antigravity CLI; on any failure fall back to a local Ollama model.
    Respects the sliding-window circuit breaker: if agy has failed
    >= _AGY_CIRCUIT_THRESHOLD times in the last two minutes, the circuit
    opens and agy is skipped entirely until the window expires.

    Failure-mode prefixes returned to Codex:
      [agy_rate_limited]     — quota / rate-limit response from agy
      [agy_timeout]          — agy exceeded its timeout budget
      [agy_missing_binary]   — AGY_BIN not found on PATH
      [agy_error]            — any other non-zero exit
    """
    timeout = timeout or AGY_FLASH_TIMEOUT
    total_timeout = total_timeout or max(AGY_TOTAL_TIMEOUT, timeout)

    async def run_fallback(
        prefix: str,
        *,
        fallback_timeout: float = AGY_FALLBACK_TIMEOUT,
    ) -> str:
        try:
            fallback_result = await asyncio.wait_for(
                ask_ollama(
                    fallback_model,
                    prompt,
                    num_predict=fallback_num_predict,
                    num_ctx=DEFAULT_NUM_CTX,
                ),
                timeout=fallback_timeout,
            )
            return f"{prefix}\n{fallback_result}"
        except Exception:
            return f"{prefix} fallback_also_failed. Handle this payload directly."

    if _agy_circuit_is_open():
        return await run_fallback("[agy_circuit_open]")

    last_exc: RuntimeError | None = None
    for _ in range(max(1, agy_attempts)):
        try:
            return await ask_antigravity(prompt, model=model, timeout=timeout)
        except RuntimeError as exc:
            last_exc = exc

    err_str = str(last_exc).lower() if last_exc else "unknown agy failure"
    _agy_record_failure()

    if "rate limit" in err_str or "quota" in err_str:
        prefix = "[agy_rate_limited]"
    elif "timed out" in err_str:
        prefix = "[agy_timeout]"
    elif "not found" in err_str:
        prefix = "[agy_missing_binary]"
    else:
        prefix = "[agy_error]"

    remaining_timeout = max(1.0, total_timeout - timeout)
    fallback_timeout = min(AGY_FALLBACK_TIMEOUT, remaining_timeout)
    return await run_fallback(prefix, fallback_timeout=fallback_timeout)


async def ollama_request(
    method: str,
    path: str,
    *,
    json_payload: dict[str, Any] | None = None,
    timeout: float = OLLAMA_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.close()
    except PermissionError as exc:
        raise RuntimeError(
            "Python socket access is blocked in this runtime; cannot reach local Ollama HTTP API."
        ) from exc

    client = httpx.AsyncClient(timeout=OLLAMA_HTTP_TIMEOUT, trust_env=False)
    try:
        response = await asyncio.wait_for(
            client.request(method, f"{OLLAMA_BASE_URL}{path}", json=json_payload),
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    finally:
        try:
            await asyncio.wait_for(client.aclose(), timeout=2)
        except Exception:
            pass


async def ollama_ps_text() -> str:
    ok, output = await run_command("ollama", "ps", timeout=10)
    return output if ok else ""


async def model_is_fully_on_gpu(model: str) -> bool:
    ps = await ollama_ps_text()
    for line in ps.splitlines():
        if model in line and "100% GPU" in line:
            return True
    return False


async def validate_num_ctx(model: str, num_ctx: int) -> None:
    if num_ctx <= DEFAULT_NUM_CTX:
        return
    if num_ctx > EXTENDED_NUM_CTX:
        raise ValueError(
            f"num_ctx={num_ctx} exceeds the local RTX 2070 SUPER guardrail. "
            f"Use {DEFAULT_NUM_CTX} by default; {EXTENDED_NUM_CTX} is the maximum opt-in value."
        )
    if not await model_is_fully_on_gpu(model):
        raise ValueError(
            f"num_ctx={num_ctx} requires a prior ollama ps check showing {model} at 100% GPU. "
            f"Warm or run the model at num_ctx={DEFAULT_NUM_CTX}, verify local_ollama_status, then retry."
        )


def default_keep_alive_for(model: str) -> str:
    return DEFAULT_KEEP_ALIVE if model == WARM_MODEL else "0"


def num_predict_for(tool_name: str | None, requested: int) -> int:
    if tool_name and tool_name in TOOL_PROMPTS:
        return min(requested, TOOL_PROMPTS[tool_name].num_predict)
    return requested


def render_tool_prompt(tool_name: str, **values: Any) -> str:
    return TOOL_PROMPTS[tool_name].template.format(**values)


async def unload_warm_model_for_large_model(target_model: str) -> str:
    if target_model == WARM_MODEL:
        return "skip_unload_same_model"
    payload = {"model": WARM_MODEL, "prompt": "", "stream": False, "keep_alive": 0}
    try:
        await ollama_request(
            "POST",
            "/api/generate",
            json_payload=payload,
            timeout=OLLAMA_TIMEOUT_SECONDS + 5,
        )
    except Exception as exc:
        return f"warm_model_unload_failed:{type(exc).__name__}"
    return "warm_model_unload_requested"


async def unload_model_from_ollama(model: str) -> str:
    """Unload a specific model from Ollama memory to free up VRAM."""
    payload = {"model": model, "prompt": "", "stream": False, "keep_alive": 0}
    try:
        await ollama_request(
            "POST",
            "/api/generate",
            json_payload=payload,
            timeout=OLLAMA_TIMEOUT_SECONDS + 5,
        )
        return "unloaded"
    except Exception as exc:
        return f"unload_failed:{type(exc).__name__}"


async def ollama_chat(
    model: str,
    prompt: str,
    *,
    temperature: float = 0.2,
    num_predict: int = 512,
    num_ctx: int = DEFAULT_NUM_CTX,
    keep_alive: str | None = None,
    system: str | None = None,
) -> dict[str, Any]:
    await validate_num_ctx(model, num_ctx)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system
                or (
                    "You are a local helper model for Codex. "
                    "Be concise. Do not include chain-of-thought. "
                    "Return only the requested output."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,
        "keep_alive": keep_alive if keep_alive is not None else default_keep_alive_for(model),
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }

    return await ollama_request(
        "POST",
        "/api/chat",
        json_payload=payload,
        timeout=OLLAMA_TIMEOUT_SECONDS + 5,
    )


async def ask_ollama(
    model: str,
    prompt: str,
    *,
    temperature: float = 0.2,
    num_predict: int = 512,
    num_ctx: int = DEFAULT_NUM_CTX,
    keep_alive: str | None = None,
    system: str | None = None,
    tool_name: str | None = None,
) -> str:
    # Prevent VRAM thrashing between CODE_MODEL and REASON_MODEL
    if CODE_MODEL != REASON_MODEL:
        if model == CODE_MODEL:
            await unload_model_from_ollama(REASON_MODEL)
        elif model == REASON_MODEL:
            await unload_model_from_ollama(CODE_MODEL)

    capped_num_predict = num_predict_for(tool_name, num_predict)
    data = await ollama_chat(
        model,
        prompt,
        temperature=temperature,
        num_predict=capped_num_predict,
        num_ctx=num_ctx,
        keep_alive=keep_alive,
        system=system,
    )
    message = data.get("message", {})
    return enforce_tool_output(tool_name, str(message.get("content", "")))


@mcp.tool()
async def local_summarize(
    text: str,
    focus: str = "important implementation details",
    num_ctx: int = DEFAULT_NUM_CTX,
) -> str:
    """
    Summarize code, logs, docs, or command output.

    Routing:
      < AGY_ROUTING_MIN_TOKENS  → local qwen3.5:9b (GPU, fast)
      >= AGY_ROUTING_MIN_TOKENS → Antigravity CLI / Gemini Flash (30s timeout)

    Gemini Flash handles large payloads (terminal logs, raw output) with
    lightning-fast keyword extraction and structural pattern matching at
    zero local VRAM cost.  Falls back to local Ollama if agy is unavailable.
    Returns a compact bullet summary for Codex.
    """
    spec = TOOL_PROMPTS["local_summarize"]
    token_count = estimate_tokens(text)

    tier = select_tier(token_count)

    if tier == _Tier.AGY:
        agy_model = AGY_FLASH_MODEL
        agy_prompt = (
            f"You are a high-speed pre-chewer for Codex (GPT-5.5). "
            f"Your task is to structurally compress the following raw content into a high-density, "
            f"lossless structural summary that cuts token bloat by 90% while preserving every actionable detail.\n"
            f"Focus specifically on: {focus}.\n"
            f"Rules:\n"
            f"- Output 1 to 6 extremely concise, fact-packed bullet points.\n"
            f"- Absolute ban on preambles, chatty openings, headings, or summaries.\n"
            f"- Retain exact error codes, function signatures, SQL locks, and hardware telemetry metrics.\n\n"
            f"<raw_content>\n{text}\n</raw_content>"
        )
        return await capture_tool_call(
            spec.tool_name,
            f"antigravity/{agy_model}",
            {"text": text, "text_tokens": token_count, "focus": focus, "routed_to": "antigravity_flash"},
            lambda: run_local_analysis(
                lambda: ask_antigravity_with_fallback(
                    agy_prompt,
                    model=agy_model,
                    timeout=AGY_FLASH_TIMEOUT,
                    fallback_model=spec.model,
                    fallback_num_predict=spec.num_predict,
                ),
                timeout=AGY_TOTAL_TIMEOUT,
            ),
        )

    prompt = render_tool_prompt(spec.tool_name, text=text, focus=focus)
    return await capture_tool_call(
        spec.tool_name,
        spec.model,
        {"text": text, "focus": focus, "num_ctx": num_ctx},
        lambda: run_local_analysis(
            lambda: ask_ollama(
                spec.model,
                prompt,
                temperature=spec.temperature,
                num_predict=spec.num_predict,
                num_ctx=num_ctx,
                keep_alive=spec.keep_alive,
                system=spec.system,
                tool_name=spec.tool_name,
            )
        ),
    )


@mcp.tool()
async def local_code_review(
    diff: str,
    focus: str = "bugs, regressions, missing tests",
    num_ctx: int = DEFAULT_NUM_CTX,
) -> str:
    """
    Review a git diff for likely bugs, regressions, or missing tests.

    Routing:
      < AGY_ROUTING_MIN_TOKENS  → local qwen3.5:9b (GPU, fast)
      >= AGY_ROUTING_MIN_TOKENS → Antigravity CLI / Gemini Flash (30s timeout)

    Gemini Flash shreds large diffs (10k-50k+ tokens) for structural pattern
    matching at near-zero latency and zero local VRAM cost.  Falls back to
    local Ollama if agy is unavailable.  For very large diffs, prefer
    agy_compress_diff which returns a richer structured compression.
    Returns only high-confidence issues for Codex to verify.
    """
    spec = TOOL_PROMPTS["local_code_review"]
    token_count = estimate_tokens(diff)

    tier = select_tier(token_count)

    if tier == _Tier.AGY:
        agy_model = AGY_FLASH_MODEL
        agy_prompt = (
            f"You are a pre-filtering code review helper for Codex (GPT-5.5). "
            f"Analyze this raw git diff and extract only critical regressions, logical bugs, or missing tests.\n"
            f"Focus specifically on: {focus}.\n"
            f"Rules:\n"
            f"- Output strict bullet points mapping files directly to bugs (issue, code evidence, and suggested checks).\n"
            f"- If no meaningful issue exists, return exactly: No obvious issue found.\n"
            f"- Absolute ban on explaining simple styling changes, indentation corrections, or conversational fluff.\n"
            f"- No chatty intro or conclusion.\n\n"
            f"<git_diff>\n{diff}\n</git_diff>"
        )
        return await capture_tool_call(
            spec.tool_name,
            f"antigravity/{agy_model}",
            {"diff": diff, "diff_tokens": token_count, "focus": focus, "routed_to": "antigravity_flash"},
            lambda: run_local_analysis(
                lambda: ask_antigravity_with_fallback(
                    agy_prompt,
                    model=agy_model,
                    timeout=AGY_FLASH_TIMEOUT,
                    fallback_model=spec.model,
                    fallback_num_predict=spec.num_predict,
                ),
                timeout=AGY_TOTAL_TIMEOUT,
            ),
        )

    prompt = render_tool_prompt(spec.tool_name, diff=diff, focus=focus)
    return await capture_tool_call(
        spec.tool_name,
        spec.model,
        {"diff": diff, "focus": focus, "num_ctx": num_ctx},
        lambda: run_local_analysis(
            lambda: ask_ollama(
                spec.model,
                prompt,
                temperature=spec.temperature,
                num_predict=spec.num_predict,
                num_ctx=num_ctx,
                keep_alive=spec.keep_alive,
                system=spec.system,
                tool_name=spec.tool_name,
            )
        ),
    )


@mcp.tool()
async def local_test_ideas(
    code_or_diff: str,
    test_framework: str = "unknown",
    num_ctx: int = DEFAULT_NUM_CTX,
) -> str:
    """
    Generate concise test ideas from code or a diff using a local model.
    """
    spec = TOOL_PROMPTS["local_test_ideas"]
    prompt = render_tool_prompt(
        spec.tool_name,
        code_or_diff=code_or_diff,
        test_framework=test_framework,
    )
    return await capture_tool_call(
        spec.tool_name,
        spec.model,
        {"code_or_diff": code_or_diff, "test_framework": test_framework, "num_ctx": num_ctx},
        lambda: run_local_analysis(
            lambda: ask_ollama(
                spec.model,
                prompt,
                temperature=spec.temperature,
                num_predict=spec.num_predict,
                num_ctx=num_ctx,
                keep_alive=spec.keep_alive,
                system=spec.system,
                tool_name=spec.tool_name,
            )
        ),
    )


@mcp.tool()
async def local_reason_check(problem: str, num_ctx: int = DEFAULT_NUM_CTX) -> str:
    """
    Ask a reasoning model for a concise second opinion on a debugging problem.

    Routing:
      < AGY_ROUTING_MIN_TOKENS  → local deepseek-r1:8b (GPU, unloads warm model)
      >= AGY_ROUTING_MIN_TOKENS → Antigravity CLI / Claude Sonnet Thinking (60s timeout)

    Claude Sonnet Thinking's adaptive internal scratchpad is the gold standard
    for navigating complex debugging deadlocks and concurrency issues without
    hallucinating.  Falls back to local deepseek-r1 if agy is unavailable.
    Returns 1-5 conclusions or next checks for Codex.
    """
    spec = TOOL_PROMPTS["local_reason_check"]
    token_count = estimate_tokens(problem)
    tier = select_tier(token_count)

    if tier == _Tier.AGY:
        agy_model = AGY_THINK_MODEL
        agy_prompt = (
            f"Give a concise second opinion for this debugging or problem-solving task.\n"
            f"Return 1 to 5 bullets total. Each bullet is one short conclusion or next check.\n"
            f"Do not include hidden reasoning tags. Do not write long explanations.\n\n"
            f"<problem>\n{problem}\n</problem>"
        )
        return await capture_tool_call(
            spec.tool_name,
            f"antigravity/{agy_model}",
            {"problem_tokens": token_count, "routed_to": "antigravity_thinking"},
            lambda: run_local_analysis(
                lambda: ask_antigravity_with_fallback(
                    agy_prompt,
                    model=agy_model,
                    timeout=AGY_THINK_TIMEOUT,
                    fallback_model=spec.model,
                    fallback_num_predict=spec.num_predict,
                ),
                timeout=AGY_TOTAL_TIMEOUT,
            ),
        )

    prompt = render_tool_prompt(spec.tool_name, problem=problem)

    async def run_reason_check() -> str:
        unload_status = await unload_warm_model_for_large_model(spec.model)
        output = await ask_ollama(
            spec.model,
            prompt,
            temperature=spec.temperature,
            num_predict=spec.num_predict,
            num_ctx=num_ctx,
            keep_alive=spec.keep_alive,
            system=spec.system,
            tool_name=spec.tool_name,
        )
        if unload_status.startswith("warm_model_unload_failed:"):
            return f"[preflight: {unload_status}]\n{output}".strip()
        return output

    return await capture_tool_call(
        spec.tool_name,
        spec.model,
        {"problem": problem, "num_ctx": num_ctx},
        lambda: run_local_analysis(run_reason_check),
    )


@mcp.tool()
async def local_plan_check(problem: str, num_ctx: int = DEFAULT_NUM_CTX) -> str:
    """
    Ask a planning model for a compact implementation plan.

    Routing:
      < AGY_ROUTING_MIN_TOKENS  → local opt-in 9B model (GPU; not kept warm
                                   due to 8GB VRAM risk; unloads warm model first)
      >= AGY_ROUTING_MIN_TOKENS → Antigravity CLI / Claude Sonnet Thinking (60s timeout)

    Claude Sonnet Thinking provides elite-tier architectural analysis for complex
    multi-file refactors and system design decisions without GPU pressure.
    Falls back to the local qwen3.5:9b model if agy is unavailable.
    Returns 1-6 concrete action steps, risk checks, and verification steps.
    """
    spec = TOOL_PROMPTS["local_plan_check"]
    token_count = estimate_tokens(problem)
    tier = select_tier(token_count)

    if tier == _Tier.AGY:
        agy_model = AGY_THINK_MODEL
        agy_prompt = (
            f"Create a compact implementation plan.\n"
            f"Return 1 to 6 bullets total. Each bullet is one concrete action, "
            f"risk check, or verification step.\n"
            f"Do not include hidden reasoning tags. Do not write a preface or conclusion.\n\n"
            f"<problem>\n{problem}\n</problem>"
        )
        return await capture_tool_call(
            spec.tool_name,
            f"antigravity/{agy_model}",
            {"problem_tokens": token_count, "routed_to": "antigravity_thinking"},
            lambda: run_local_analysis(
                lambda: ask_antigravity_with_fallback(
                    agy_prompt,
                    model=agy_model,
                    timeout=AGY_THINK_TIMEOUT,
                    fallback_model=spec.model,
                    fallback_num_predict=spec.num_predict,
                ),
                timeout=AGY_TOTAL_TIMEOUT,
            ),
        )

    prompt = render_tool_prompt(spec.tool_name, problem=problem)

    async def run_plan_check() -> str:
        unload_status = await unload_warm_model_for_large_model(spec.model)
        output = await ask_ollama(
            spec.model,
            prompt,
            temperature=spec.temperature,
            num_predict=spec.num_predict,
            num_ctx=num_ctx,
            keep_alive=spec.keep_alive,
            system=spec.system,
            tool_name=spec.tool_name,
        )
        if unload_status.startswith("warm_model_unload_failed:"):
            return f"[preflight: {unload_status}]\n{output}".strip()
        return output

    return await capture_tool_call(
        spec.tool_name,
        spec.model,
        {"problem": problem, "num_ctx": num_ctx},
        lambda: run_local_analysis(run_plan_check),
    )


@mcp.tool()
async def local_warm_model(
    model: str = WARM_MODEL,
    keep_alive: str = DEFAULT_KEEP_ALIVE,
    num_ctx: int = DEFAULT_NUM_CTX,
) -> str:
    """
    Warm one Ollama model and keep it resident for the requested keep_alive window.
    Defaults to the primary 7B coder model for 2h.
    """
    async def run_warm() -> str:
        start = time.perf_counter()
        data = await ollama_chat(
            model,
            "Reply with OK only.",
            temperature=0,
            num_predict=8,
            num_ctx=num_ctx,
            keep_alive=keep_alive,
            system="Reply with OK only. Do not include explanations or chain-of-thought.",
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        content = strip_thinking(str(data.get("message", {}).get("content", "")))
        fully_gpu = await model_is_fully_on_gpu(model)
        return "\n".join(
            [
                f"model: {model}",
                f"keep_alive: {keep_alive}",
                f"num_ctx: {num_ctx}",
                f"response: {content or 'n/a'}",
                f"elapsed: {elapsed_ms:.0f} ms",
                f"load_duration: {ns_to_ms(data.get('load_duration'))}",
                f"eval_duration: {ns_to_ms(data.get('eval_duration'))}",
                f"ollama ps 100% GPU: {'yes' if fully_gpu else 'no or unavailable'}",
            ]
        )

    return await capture_tool_call(
        "local_warm_model",
        model,
        {"model": model, "keep_alive": keep_alive, "num_ctx": num_ctx},
        run_warm,
    )


@mcp.tool()
async def local_unload_model(model: str = WARM_MODEL) -> str:
    """
    Ask Ollama to unload a model from memory.
    """
    async def run_unload() -> str:
        payload = {"model": model, "prompt": "", "stream": False, "keep_alive": 0}
        await ollama_request(
            "POST",
            "/api/generate",
            json_payload=payload,
            timeout=OLLAMA_TIMEOUT_SECONDS + 5,
        )
        return f"Requested unload for {model}."

    return await capture_tool_call(
        "local_unload_model",
        model,
        {"model": model},
        run_unload,
    )


@mcp.tool()
async def local_ollama_status() -> str:
    """
    Report Ollama residency plus NVIDIA telemetry. This tool observes health only;
    it does not change GPU clocks, fans, voltage, or power limits.
    """
    async def run_status() -> str:
        status: list[str] = [
            "Local Ollama defaults:",
            f"- warm_model: {WARM_MODEL}",
            f"- keep_alive: {DEFAULT_KEEP_ALIVE}",
            f"- default_num_ctx: {DEFAULT_NUM_CTX}",
            f"- extended_num_ctx_guardrail: {EXTENDED_NUM_CTX} requires prior 100% GPU verification",
            "- non-primary planning/reasoning models use keep_alive=0 unless explicitly warmed",
        ]

        try:
            ps_data = await ollama_request("GET", "/api/ps", timeout=6)
            models = ps_data.get("models", [])
            status.append("\nOllama /api/ps:")
            status.append(json.dumps(models, indent=2, default=str) if models else "[]")
        except Exception as exc:
            status.append(f"\nOllama /api/ps unavailable: {type(exc).__name__}: {exc}")

        ok, ps_output = await run_command("ollama", "ps", timeout=10)
        status.append("\nollama ps:")
        status.append(ps_output if ok and ps_output else "unavailable or no loaded models")

        ok, nvidia_output = await run_command(
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,clocks.gr",
            "--format=csv",
            timeout=10,
        )
        status.append("\nnvidia-smi telemetry:")
        status.append(nvidia_output if ok and nvidia_output else "unavailable")

        ok, process_output = await run_command(
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv",
            timeout=10,
        )
        status.append("\nGPU compute processes:")
        status.append(process_output if ok and process_output else "unavailable or no compute processes")

        return "\n".join(status)

    return await capture_tool_call(
        "local_ollama_status",
        "n/a",
        {},
        lambda: run_status_tool(run_status),
    )


@mcp.tool()
async def local_capture_status() -> str:
    """
    Report whether local MCP capture is enabled and where redacted records go.
    Note: tool and outcome counts are bounded recent counts from the recent-window
    of the ledger, not lifetime counts.
    """
    async def run_capture_status() -> str:
        path = ledger_path()
        records, truncated = read_recent_ledger_records(path)
        tool_records = [record for record in records if record.get("record_type") == "tool_call"]
        outcome_records = [record for record in records if record.get("record_type") == "outcome"]
        recent_task_ids = [
            str(record.get("task_id"))
            for record in reversed(tool_records)
            if record.get("task_id")
        ][:5]

        lines = [
            f"capture_enabled: {'yes' if capture_enabled() else 'no'}",
            f"privacy_mode: {'raw' if raw_capture_enabled() else 'redacted'}",
            f"ledger_path: {path}",
            f"ledger_exists: {'yes' if path.exists() else 'no'}",
            f"ledger_rows_scanned: {len(records)}",
            f"ledger_scan_bounded: {'yes' if truncated else 'no'}",
            f"recent_tool_record_count: {len(tool_records)}",
            f"recent_outcome_record_count: {len(outcome_records)}",
            "recent_task_ids: " + (", ".join(recent_task_ids) if recent_task_ids else "none"),
        ]
        return "\n".join(lines)

    return await run_status_tool(run_capture_status)


@mcp.tool()
async def local_record_outcome(
    task_id: str,
    outcome: str,
    accepted_solution: str,
    notes: str = "",
) -> str:
    """
    Append a quality label and accepted answer for a captured local tool record.
    """
    if outcome not in OUTCOME_LABELS:
        allowed = ", ".join(sorted(OUTCOME_LABELS))
        raise ValueError(f"outcome must be one of: {allowed}")

    existing_task_ids = {
        str(record.get("task_id"))
        for record in read_ledger_records()
        if record.get("record_type") == "tool_call" and record.get("task_id")
    }
    accepted_stored, accepted_flags = redact_value(accepted_solution)
    notes_stored, notes_flags = redact_value(notes)
    record = {
        "record_type": "outcome",
        "task_id": task_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "outcome": outcome,
        "accepted_solution": accepted_stored,
        "notes": notes_stored,
        "privacy_mode": "raw" if raw_capture_enabled() else "redacted",
        "risk_flags": sorted(set(accepted_flags + notes_flags)),
    }
    append_ledger_record(record)

    linked = task_id in existing_task_ids
    return "\n".join(
        [
            f"recorded_outcome: {outcome}",
            f"task_id: {task_id}",
            f"linked_to_tool_record: {'yes' if linked else 'no'}",
            f"ledger_path: {ledger_path()}",
        ]
    )


# ---------------------------------------------------------------------------
# Dedicated large-diff compression tool
# ---------------------------------------------------------------------------

_AGY_DIFF_COMPRESS_INSTRUCTIONS = """
Analyze this git diff and return a structured compression in this exact format:

CHANGED_FILES: <comma-separated list of modified files>
BEHAVIOR_CHANGES: <up to 8 bullets — one sentence each describing what behaviour changed>
RISKY_LINES: <up to 5 entries as "file:approx_line — one-line description of the risk">
REMOVED_LOGIC: <up to 3 bullets of deleted functionality that callers may depend on>

Return ONLY the four sections above. No preamble, no code blocks, no extra prose.
""".strip()


def build_agy_diff_compress_prompt(
    diff: str,
    focus: str,
    *,
    chunk_label: str | None = None,
) -> str:
    chunk_line = f"\nCHUNK: {chunk_label}" if chunk_label else ""
    return (
        f"{_AGY_DIFF_COMPRESS_INSTRUCTIONS}\n\n"
        f"FOCUS: {focus}{chunk_line}\n\n"
        f"<diff>\n{diff}\n</diff>"
    )


def compact_diff_for_agy(diff: str, max_tokens: int = AGY_DIFF_SKETCH_TOKENS) -> str:
    if estimate_tokens(diff) <= max_tokens:
        return diff

    metadata: list[str] = []
    changes: list[tuple[int, str]] = []
    for line_no, line in enumerate(diff.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("diff --git ", "index ", "--- ", "+++ ", "@@ ")):
            metadata.append(f"{line_no}: {line}")
        elif stripped.startswith(("+", "-")) and not stripped.startswith(("+++", "---")):
            changes.append((line_no, line))

    selected: list[str] = []
    current_tokens = 0

    def add_line(line: str) -> bool:
        nonlocal current_tokens
        line_tokens = estimate_tokens(line)
        if selected and current_tokens + line_tokens > max_tokens:
            return False
        selected.append(line)
        current_tokens += line_tokens
        return True

    for line in metadata:
        if not add_line(line):
            break

    if changes:
        remaining_budget = max(1, max_tokens - current_tokens)
        average_change_tokens = max(1, estimate_tokens(changes[0][1]) + 3)
        target_change_count = max(1, remaining_budget // average_change_tokens)
        step = max(1, len(changes) // target_change_count)
        sampled = changes[::step]
        if changes[-1] not in sampled:
            sampled.append(changes[-1])
        for line_no, line in sampled:
            if not add_line(f"{line_no}: {line}"):
                break

    omitted = max(0, len(diff.splitlines()) - len(selected))
    header = (
        "[deterministic_diff_sketch]\n"
        f"original_tokens_est={estimate_tokens(diff)} "
        f"original_lines={len(diff.splitlines())} "
        f"sketched_lines={len(selected)} omitted_lines={omitted}\n"
    )
    return header + "\n".join(selected)


async def ask_agy_diff_compression(diff: str, focus: str) -> str:
    full_prompt = build_agy_diff_compress_prompt(diff, focus)
    if estimate_tokens(full_prompt) <= AGY_DIFF_DIRECT_PROMPT_TOKENS:
        return await ask_antigravity_with_fallback(
            full_prompt,
            model=AGY_FLASH_MODEL,
            timeout=AGY_DIFF_COMPRESS_TIMEOUT,
            total_timeout=AGY_DIFF_TOTAL_TIMEOUT,
            fallback_model=CODE_MODEL,
            fallback_num_predict=300,
        )

    sketch = compact_diff_for_agy(diff, max_tokens=AGY_DIFF_SKETCH_TOKENS)
    sketch_prompt = build_agy_diff_compress_prompt(
        sketch,
        (
            f"{focus}. The diff body is a deterministic line-numbered sketch "
            "created to avoid agy large-stdin stalls; preserve the sketch metadata."
        ),
    )
    return await ask_antigravity_with_fallback(
        sketch_prompt,
        model=AGY_FLASH_MODEL,
        timeout=AGY_DIFF_CHUNK_TIMEOUT,
        total_timeout=AGY_DIFF_CHUNK_TIMEOUT + AGY_FALLBACK_TIMEOUT + 5,
        agy_attempts=AGY_DIFF_CHUNK_ATTEMPTS,
        fallback_model=CODE_MODEL,
        fallback_num_predict=300,
    )


async def ask_agy_context_reducer(
    tool_name: str,
    prompt: str,
    *,
    timeout: float = AGY_FLASH_TIMEOUT,
) -> str:
    """Run explicit Gemini reducer prompts through the shared agy wrapper."""
    _ = tool_name
    return await run_local_analysis(
        lambda: ask_antigravity_with_fallback(
            prompt,
            model=AGY_FLASH_MODEL,
            timeout=timeout,
            total_timeout=max(AGY_TOTAL_TIMEOUT, timeout),
            fallback_model=CODE_MODEL,
            fallback_num_predict=300,
        ),
        timeout=max(AGY_TOTAL_TIMEOUT, timeout),
    )


@mcp.tool()
async def agy_compress_diff(
    diff: str,
    focus: str = "bugs, regressions, API surface changes",
) -> str:
    """
    Pre-chew a large git diff (ideally 4k–50k+ tokens) using Antigravity CLI
    (Gemini Flash) before passing the structured result to Codex.

    A 10k-token raw diff compresses to ~300 tokens of structured output
    (CHANGED_FILES / BEHAVIOR_CHANGES / RISKY_LINES / REMOVED_LOGIC), cutting
    Codex cloud token consumption by up to 95%.

    Payloads under the routing threshold fall back to local_code_review logic
    so this tool is safe to call regardless of diff size.
    """
    token_count = estimate_tokens(diff)

    tier = select_tier(token_count)

    if tier == _Tier.LOCAL:
        # Small diff — existing local review is cheaper
        spec = TOOL_PROMPTS["local_code_review"]
        prompt = render_tool_prompt("local_code_review", diff=diff, focus=focus)
        return await capture_tool_call(
            "agy_compress_diff",
            spec.model,
            {"diff_tokens": token_count, "focus": focus, "routed_to": "local_gpu"},
            lambda: run_local_analysis(
                lambda: ask_ollama(
                    spec.model,
                    prompt,
                    temperature=0.1,
                    num_predict=300,
                    num_ctx=DEFAULT_NUM_CTX,
                    tool_name="local_code_review",
                )
            ),
        )

    agy_model = AGY_FLASH_MODEL
    return await capture_tool_call(
        "agy_compress_diff",
        f"antigravity/{agy_model}",
        {"diff": diff, "diff_tokens": token_count, "focus": focus, "routed_to": "antigravity_flash"},
        lambda: run_local_analysis(
            lambda: ask_agy_diff_compression(diff, focus),
            timeout=AGY_DIFF_TOTAL_TIMEOUT,
        ),
    )


@mcp.tool()
async def agy_quota_status() -> str:
    """
    Check the current Antigravity quota levels and routing assignments.
    Provides real-time caching metrics and dynamic failover targets.
    """
    ensure_quota_monitor_started()

    last_check = "Never"
    if _quota_cache_ts > 0:
        last_check = datetime.fromtimestamp(_quota_cache_ts, tz=timezone.utc).isoformat()

    status = {
        "active_thinking_model": AGY_THINK_MODEL,
        "original_thinking_model": ORIGINAL_AGY_THINK_MODEL,
        "active_flash_model": AGY_FLASH_MODEL,
        "per_call_model_selection_supported": await agy_supports_model_flag(),
        "think_threshold_pct": AGY_THINK_THRESHOLD,
        "last_successful_check": last_check,
        "cached_quotas": _quota_cache,
    }
    return json.dumps(status, indent=2)


@mcp.tool()
async def clean_server_logs(
    raw_logs: str,
    remove_timestamps: bool = True,
    remove_hex_hashes: bool = True,
    deduplicate_consecutive: bool = True,
) -> str:
    """
    Clean and reduce server logs by removing timestamps, hex codes/UUIDs/PIDs,
    and collapsing consecutive identical logs to dramatically reduce context size.
    """
    lines = raw_logs.splitlines()
    cleaned_lines = []

    # Pre-compile regex patterns
    # ISO-8601 or common log timestamps
    timestamp_re = re.compile(
        r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b|"
        r"\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b"
    )
    # Hex addresses, UUIDs, and general hex values >= 6 chars
    hex_uuid_re = re.compile(
        r"\b0x[0-9a-fA-F]+\b|"
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b|"
        r"\b[0-9a-fA-F]{12,}\b"
    )
    # Bracketed numbers like [12345] (often PIDs or thread IDs)
    pid_re = re.compile(r"\[\d+\]|\bpid=\d+\b")

    for line in lines:
        cleaned = line
        if remove_timestamps:
            cleaned = timestamp_re.sub("[TIMESTAMP]", cleaned)
        if remove_hex_hashes:
            cleaned = hex_uuid_re.sub("[HEX]", cleaned)
            cleaned = pid_re.sub("[ID]", cleaned)
        cleaned_lines.append(cleaned)

    if not deduplicate_consecutive:
        return "\n".join(cleaned_lines)

    collapsed_lines = []
    if cleaned_lines:
        prev_line = cleaned_lines[0]
        count = 1
        for current_line in cleaned_lines[1:]:
            if current_line == prev_line:
                count += 1
            else:
                if count > 1:
                    collapsed_lines.append(f"{prev_line} [repeated {count} times]")
                else:
                    collapsed_lines.append(prev_line)
                prev_line = current_line
                count = 1
        if count > 1:
            collapsed_lines.append(f"{prev_line} [repeated {count} times]")
        else:
            collapsed_lines.append(prev_line)

    return "\n".join(collapsed_lines)


@mcp.tool()
async def extract_regex_lines(
    text: str,
    pattern: str,
    case_insensitive: bool = True,
    context_lines: int = 0,
) -> str:
    """
    Scan a massive text payload locally and extract only the lines matching a target regex,
    with optional surrounding context lines, to isolate relevant details.
    """
    flags = re.IGNORECASE if case_insensitive else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        return f"Error: Invalid regular expression pattern: {exc}"

    lines = text.splitlines()
    matching_indices = []

    for i, line in enumerate(lines):
        if regex.search(line):
            matching_indices.append(i)

    if not matching_indices:
        return "No matching lines found."

    # Determine which line indices should be included based on context_lines
    included_indices = set()
    for index in matching_indices:
        start = max(0, index - context_lines)
        end = min(len(lines), index + context_lines + 1)
        for j in range(start, end):
            included_indices.add(j)

    sorted_indices = sorted(included_indices)

    output_parts = []
    last_idx = -2
    for idx in sorted_indices:
        # If there's a gap between lines, add a separator
        if last_idx != -2 and idx > last_idx + 1:
            output_parts.append("...")
        line_num = idx + 1
        indicator = ">" if idx in matching_indices else " "
        output_parts.append(f"{line_num:4d}{indicator} {lines[idx]}")
        last_idx = idx

    return "\n".join(output_parts)


@mcp.tool()
async def trim_markdown_payload(
    markdown: str,
    max_code_block_lines: int = 15,
    remove_images: bool = True,
    max_list_items: int = 10,
) -> str:
    """
    Trim huge markdown payloads locally. Condenses large code blocks, collapses long lists,
    and strips heavy image payloads or binary assets to retain semantic structure with minimal tokens.
    """
    # 1. Strip images first if requested
    if remove_images:
        # Match ![alt](url)
        markdown = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", r"[IMAGE: \1]", markdown)
        # Match raw HTML img tags
        markdown = re.sub(r"<img\s+[^>]*src=\"[^\"]*\"[^>]*>", r"[IMAGE]", markdown)

    lines = markdown.splitlines()
    output_lines = []

    in_code_block = False
    code_block_info = ""
    current_code_lines = []

    in_list = False
    current_list_items = []

    def flush_list():
        nonlocal in_list, current_list_items
        if not in_list:
            return
        if len(current_list_items) > max_list_items:
            trimmed = len(current_list_items) - max_list_items
            output_lines.extend(current_list_items[:max_list_items])
            # Keep bullet style
            bullet_char = "-"
            for item in current_list_items:
                match = bullet_re.match(item) or ordered_re.match(item)
                if match:
                    bullet_char = match.group(1).strip()
                    break
            output_lines.append(f"{bullet_char} ... [trimmed {trimmed} items] ...")
        else:
            output_lines.extend(current_list_items)
        current_list_items = []
        in_list = False

    def flush_code_block():
        nonlocal in_code_block, current_code_lines, code_block_info
        if not in_code_block:
            return
        output_lines.append(code_block_info)  # e.g. ```python
        n = len(current_code_lines)
        if n > max_code_block_lines:
            half = max_code_block_lines // 2
            first_half = current_code_lines[:half]
            second_half = current_code_lines[-half:]
            trimmed = n - (half * 2)
            output_lines.extend(first_half)
            output_lines.append(f"... [trimmed {trimmed} lines of code] ...")
            output_lines.extend(second_half)
        else:
            output_lines.extend(current_code_lines)
        output_lines.append("```")
        current_code_lines = []
        in_code_block = False

    bullet_re = re.compile(r"^(\s*[-*+]\s+)(.*)$")
    ordered_re = re.compile(r"^(\s*\d+\.\s+)(.*)$")

    for line in lines:
        stripped = line.strip()

        # Code block toggle
        if stripped.startswith("```"):
            if in_code_block:
                # End of code block
                flush_code_block()
            else:
                # Start of code block
                flush_list()
                in_code_block = True
                code_block_info = line
            continue

        if in_code_block:
            current_code_lines.append(line)
            continue

        # List detection
        bullet_match = bullet_re.match(line)
        ordered_match = ordered_re.match(line)

        if bullet_match or ordered_match:
            if not in_list:
                in_list = True
            current_list_items.append(line)
        else:
            flush_list()
            output_lines.append(line)

    # Flush any remaining state at the end of the file
    flush_list()
    flush_code_block()

    return "\n".join(output_lines)


# ---------------------------------------------------------------------------
# High-Signal MCP Tools for GPT-5.5 (Codex)
# ---------------------------------------------------------------------------

def extract_python_signatures(content: str) -> str:
    """Extract Python class and method signatures with docstrings, stripping body logic."""
    lines = content.splitlines()
    out = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        
        # Match start of class or def/async def
        match = re.match(r"^(\s*)(class|def|async\s+def)\s+", line)
        if match:
            sig_lines = [line]
            # Read until the signature definition ends with ':'
            temp_i = i
            while temp_i < n:
                current_stripped = lines[temp_i].split("#")[0].strip()
                if current_stripped.endswith(":"):
                    break
                temp_i += 1
                if temp_i < n and temp_i != i:
                    sig_lines.append(lines[temp_i])
            
            if temp_i < n:
                i = temp_i
            
            out.extend(sig_lines)
            
            # Look for docstring immediately following
            i += 1
            while i < n and not lines[i].strip():
                i += 1
            if i < n:
                next_stripped = lines[i].strip()
                if next_stripped.startswith('"""') or next_stripped.startswith("'''"):
                    quote_char = '"""' if next_stripped.startswith('"""') else "'''"
                    doc_lines = []
                    if next_stripped.endswith(quote_char) and len(next_stripped) >= 6:
                        doc_lines.append(lines[i])
                        i += 1
                    else:
                        doc_lines.append(lines[i])
                        i += 1
                        while i < n:
                            doc_lines.append(lines[i])
                            if quote_char in lines[i]:
                                i += 1
                                break
                            i += 1
                    out.extend(doc_lines)
            continue
        i += 1
    return "\n".join(out)


def extract_ts_js_signatures(content: str) -> str:
    """Extract TS/JS classes, interfaces, types, methods, and JSDocs, stripping body logic."""
    lines = content.splitlines()
    out = []
    i = 0
    n = len(lines)
    
    jsdoc_buffer = []
    in_jsdoc = False
    
    while i < n:
        line = lines[i]
        stripped = line.strip()
        
        # Check for JSDoc start
        if stripped.startswith("/**"):
            in_jsdoc = True
            jsdoc_buffer = [line]
            if stripped.endswith("*/") and len(stripped) > 3:
                in_jsdoc = False
            i += 1
            continue
        elif in_jsdoc:
            jsdoc_buffer.append(line)
            if "*/" in line:
                in_jsdoc = False
            i += 1
            continue
            
        if not stripped:
            i += 1
            continue
            
        # Match class, interface, type, function, export, constructor, methods
        is_signature_start = False
        
        decl_match = re.match(
            r"^\s*(export\s+)?(class|interface|type|enum|function|async\s+function|const)\s+(\w+)",
            line
        )
        if decl_match:
            is_signature_start = True
        elif re.match(
            r"^\s*(public|private|protected|static|async|get|set|readonly)*\s*(\w+)\s*\([^)]*",
            line
        ):
            # Avoid matching control structures
            word = re.match(r"^\s*(\w+)", line)
            if word and word.group(1) not in {"if", "for", "while", "switch", "catch", "return", "throw", "import", "export"}:
                is_signature_start = True
        elif re.match(r"^\s*(public|private|protected|readonly)\s+\w+", line):
            if ";" in stripped:
                is_signature_start = True
                
        if is_signature_start:
            sig_lines = [line]
            temp_i = i
            # Read until we hit '{', ';', '}', or '=>'
            while temp_i < n:
                curr_stripped = lines[temp_i].strip()
                if "{" in curr_stripped or ";" in curr_stripped or "}" in curr_stripped or "=>" in curr_stripped:
                    break
                temp_i += 1
                if temp_i < n and temp_i != i:
                    sig_lines.append(lines[temp_i])
            
            if temp_i < n:
                i = temp_i
                
            if jsdoc_buffer:
                out.extend(jsdoc_buffer)
                jsdoc_buffer = []
            out.extend(sig_lines)
        else:
            if stripped and not stripped.startswith("//"):
                jsdoc_buffer = []
                
        i += 1
    return "\n".join(out)


@mcp.tool()
async def local_map_project_structure(max_depth: int = 3) -> str:
    """
    Generate a high-signal, lightweight tree map of the workspace directory.
    Highlights files modified or untracked in git. Ignores virtual environments,
    caches, and node modules to keep the token footprint extremely small (<150 tokens).
    """
    # 1. Run git status to get modified files
    ok, git_out = await run_command("git", "status", "--porcelain", timeout=5)
    modified = {}
    if ok:
        for line in git_out.splitlines():
            if len(line) > 3:
                status = line[:2].strip()
                path_str = line[3:].strip()
                if " -> " in path_str:
                    path_str = path_str.split(" -> ")[-1].strip()
                modified[path_str] = status

    # 2. Define ignores
    ignored_names = {
        ".git", ".venv", "node_modules", "__pycache__",
        ".local_ollama_mcp", ".codex", ".pytest_cache",
        ".idea", ".vscode", ".gemini", ".github"
    }

    workspace_root = Path.cwd()

    def has_modified_subfiles(dir_path: Path) -> bool:
        try:
            rel_dir = dir_path.relative_to(workspace_root)
        except ValueError:
            return False
        rel_dir_str = str(rel_dir)
        if rel_dir_str == ".":
            return len(modified) > 0
        
        prefix = rel_dir_str + os.sep
        return any(k == rel_dir_str or k.startswith(prefix) for k in modified)

    def format_node(path: Path, is_dir: bool) -> str:
        try:
            rel_path = path.relative_to(workspace_root)
        except ValueError:
            rel_path = path
        rel_str = str(rel_path)
        
        name = path.name
        if is_dir:
            name = name + "/"
        
        if rel_str in modified:
            return f"{name} [{modified[rel_str]}]"
        return name

    lines = []
    
    def recurse(current_dir: Path, depth: int, prefix: str) -> None:
        if not current_dir.exists() or not current_dir.is_dir():
            return
        
        try:
            children = list(current_dir.iterdir())
        except PermissionError:
            return
        
        def sort_key(p: Path) -> tuple[bool, str]:
            return (not p.is_dir(), p.name.lower())
        
        filtered_children = []
        for child in children:
            if child.name in ignored_names:
                continue
            filtered_children.append(child)
            
        filtered_children.sort(key=sort_key)
        
        count = len(filtered_children)
        for i, child in enumerate(filtered_children):
            is_last = (i == count - 1)
            connector = "└── " if is_last else "├── "
            
            is_dir = child.is_dir()
            node_text = format_node(child, is_dir)
            
            should_show = False
            if not is_dir:
                try:
                    rel_child = child.relative_to(workspace_root)
                except ValueError:
                    rel_child = child
                should_show = (depth <= max_depth) or (str(rel_child) in modified)
            else:
                should_show = (depth <= max_depth) or has_modified_subfiles(child)
                
            if should_show:
                lines.append(f"{prefix}{connector}{node_text}")
                if is_dir:
                    next_prefix = prefix + ("    " if is_last else "│   ")
                    if depth < max_depth or has_modified_subfiles(child):
                        recurse(child, depth + 1, next_prefix)

    lines.append(f". (workspace: {workspace_root.name})")
    recurse(workspace_root, 1, "")
    return "\n".join(lines)


@mcp.tool()
async def local_extract_signatures(file_path: str) -> str:
    """
    Locate a source file (Python, TypeScript, JavaScript) and extract only class,
    interface, method, and function signatures with their docstrings.
    Strips all loop logic, variable assignments, and implementation details to
    expose clean API contracts at minimal token footprint (up to 95% savings).
    """
    path = Path(file_path)
    if not path.is_absolute():
        path = Path.cwd() / path
        
    if not path.exists():
        return f"Error: File '{file_path}' not found."
    if not path.is_file():
        return f"Error: '{file_path}' is not a file."
        
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Error reading file '{file_path}': {exc}"
        
    ext = path.suffix.lower()
    if ext == ".py":
        return extract_python_signatures(content)
    elif ext in {".ts", ".tsx", ".js", ".jsx"}:
        return extract_ts_js_signatures(content)
    else:
        lines = content.splitlines()
        if len(lines) > 50:
            return "\n".join(lines[:50]) + f"\n... [trimmed {len(lines) - 50} lines; signature extraction not supported for {ext}]"
        return content


@mcp.tool()
async def local_lint_audit(file_path: str) -> str:
    """
    Perform a rapid syntax and lint audit on a local source file.
    Runs in-memory Python syntax compilation or invokes lightweight CLI compilers
    for JS/TS. Returns line-by-line syntax diagnostics for GPT-5.5 self-correction.
    """
    path = Path(file_path)
    if not path.is_absolute():
        path = Path.cwd() / path
        
    if not path.exists():
        return f"Error: File '{file_path}' not found."
    if not path.is_file():
        return f"Error: '{file_path}' is not a file."
        
    ext = path.suffix.lower()
    
    if ext == ".py":
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            compile(content, str(path), "exec")
            return "No syntax errors found."
        except SyntaxError as err:
            return (
                f"SyntaxError in {err.filename} at line {err.lineno}, col {err.offset}:\n"
                f"Code: {err.text.strip() if err.text else 'n/a'}\n"
                f"Error: {err.msg}"
            )
        except Exception as exc:
            return f"Error auditing Python file: {exc}"
            
    elif ext in {".js", ".jsx"}:
        ok, output = await run_command("node", "--check", str(path), timeout=5)
        if ok:
            return "No syntax errors found."
        return f"JavaScript Syntax Error:\n{output}"
        
    elif ext in {".ts", ".tsx"}:
        ok, output = await run_command("npx", "tsc", "--noEmit", "--skipLibCheck", str(path), timeout=8)
        if ok:
            return "No TS syntax/type errors found."
        if "not found" in output or "npx" in output:
            ok_node, node_output = await run_command("node", "--check", str(path), timeout=5)
            if ok_node:
                return "No syntax errors found (checked with node)."
            return f"TypeScript Syntax Error (checked with node):\n{node_output}"
        return f"TypeScript Compilation Errors:\n{output}"
        
    else:
        return f"Lint audit not supported for file type: {ext}"


@mcp.tool()
async def local_generate_walkthrough(
    commit_or_branch: str = "",
    task_description: str = "",
    write_to_file: bool = True,
) -> str:
    """
    Generate a detailed markdown walkthrough of local uncommitted changes or a specific commit/branch.
    Saves a premium walkthrough.md to the workspace root and returns the markdown text.
    """
    # 1. Resolve Diff
    diff_content = ""
    notice = ""

    if commit_or_branch:
        # Check diff against a commit/branch
        ok, git_diff = await run_command("git", "diff", commit_or_branch, timeout=10)
        if ok and git_diff.strip():
            diff_content = git_diff
        else:
            return f"Error: No diff found for commit/branch '{commit_or_branch}'."
    else:
        # Check local uncommitted changes (both staged and unstaged)
        ok_staged, staged_diff = await run_command("git", "diff", "--cached", timeout=10)
        ok_unstaged, unstaged_diff = await run_command("git", "diff", timeout=10)

        combined_parts = []
        if ok_staged and staged_diff.strip():
            combined_parts.append(f"--- STAGED CHANGES ---\n{staged_diff}")
        if ok_unstaged and unstaged_diff.strip():
            combined_parts.append(f"--- UNSTAGED CHANGES ---\n{unstaged_diff}")

        if combined_parts:
            diff_content = "\n\n".join(combined_parts)
        else:
            # Fallback to last commit if no uncommitted changes
            ok_last, last_diff = await run_command("git", "diff", "HEAD~1", timeout=10)
            if ok_last and last_diff.strip():
                diff_content = last_diff
                notice = "> [!NOTE]\n> No local uncommitted changes were found. Generated walkthrough using the most recent commit (`HEAD~1`) as a fallback.\n\n"
            else:
                return "Error: No uncommitted changes and no previous commits found in this git repository to generate a walkthrough."

    # 2. Gather untracked files if any (only when commit_or_branch is empty)
    untracked_content = []
    if not commit_or_branch:
        ok_status, status_out = await run_command("git", "status", "--porcelain", timeout=5)
        if ok_status:
            for line in status_out.splitlines():
                if line.startswith("?? "):
                    file_path = line[3:].strip()
                    path = Path(file_path)
                    if not path.is_absolute():
                        path = Path.cwd() / path
                    if path.exists() and path.is_file():
                        # Exclude ignored dirs/sizes
                        if not any(part in file_path.split("/") for part in [".git", ".venv", "node_modules", "__pycache__", ".local_ollama_mcp"]):
                            try:
                                # Limit size to prevent token blowup (e.g. 4000 chars)
                                if path.stat().st_size < 4000:
                                    content = path.read_text(encoding="utf-8", errors="replace")
                                    untracked_content.append(f"--- UNTRACKED NEW FILE: {file_path} ---\n{content}")
                            except Exception:
                                pass

    untracked_section = "\n\n".join(untracked_content) if untracked_content else ""

    # 3. Assemble Prompt
    task_context = f"\nORIGINAL TASK DESCRIPTION:\n{task_description}\n" if task_description else ""

    prompt = (
        f"You are a premium documentation and onboarding assistant for Codex. "
        f"Your task is to analyze the following git diff and new file contents, "
        f"and output a premium-tier, clear, and comprehensive Markdown walkthrough (`walkthrough.md`) "
        f"summarizing the work done and explaining it so a developer can easily familiarize themselves with the changes.\n\n"
        f"CRITICAL DOCUMENT FORMATTING RULES:\n"
        f"- Do NOT include conversational opening, intro, or concluding remarks. Start immediately with the first header.\n"
        f"- Use clean and robust markdown with proper spacing.\n"
        f"- Retain absolute technical accuracy regarding class names, file names, variables, and function signatures.\n"
        f"- Use a standard professional tone.\n\n"
        f"SECTIONS REQUIRED IN THE WALKTHROUGH:\n"
        f"1. `# Walkthrough - [Descriptive Project/Change Title]`\n"
        f"2. `## Executive Summary`: 2 to 3 sentences explaining the core high-level architectural goal of the change.\n"
        f"3. `## Component & File Changes`: A beautifully formatted Markdown table mapping changed/new files to concise descriptions of exactly what changed and why:\n"
        f"   `| File Path | Description of Changes & Purpose |`\n"
        f"   `| --- | --- |`\n"
        f"4. `## Familiarization & Verification Guide`: Simple, actionable, step-by-step instructions (e.g. commands to run, tests to execute, files to check) for the developer to run, test, and verify the changes.\n"
        f"5. `## Risks & Edge Cases`: Up to 3 points describing any edge cases, concurrency concerns, hardware profile limitations, or potential regression areas introduced by this change.\n\n"
        f"{task_context}"
        f"GIT CHANGESET DIFF:\n"
        f"```diff\n"
        f"{diff_content}\n"
        f"```\n\n"
    )
    if untracked_section:
        prompt += (
            f"NEW UNTRACKED FILE CONTENTS:\n"
            f"```\n"
            f"{untracked_section}\n"
            f"```\n\n"
        )

    # 4. Invoke agy cloud Flash routing
    agy_model = AGY_FLASH_MODEL

    walkthrough_text = await capture_tool_call(
        "local_generate_walkthrough",
        f"antigravity/{agy_model}",
        {"commit_or_branch": commit_or_branch, "has_task_description": bool(task_description)},
        lambda: run_local_analysis(
            lambda: ask_antigravity_with_fallback(
                prompt,
                model=agy_model,
                timeout=AGY_FLASH_TIMEOUT,
                fallback_model=CODE_MODEL,
                fallback_num_predict=800,
            ),
            timeout=AGY_TOTAL_TIMEOUT,
        ),
    )

    # 5. Prepend notice if HEAD~1 fallback occurred
    result_text = hydrate_agy_file_artifact_response(walkthrough_text)
    if notice:
        result_text = notice + result_text

    # 6. Save to walkthrough.md if requested
    if write_to_file:
        try:
            clean_file_text = result_text
            if "\nroute_outcome: " in clean_file_text:
                clean_file_text = clean_file_text.split("\nroute_outcome: ")[0]

            Path("walkthrough.md").write_text(clean_file_text, encoding="utf-8")
        except Exception as err:
            result_text = f"> [!WARNING]\n> Failed to write walkthrough.md: {err}\n\n" + result_text

    return result_text


if __name__ == "__main__":
    mcp.run()
