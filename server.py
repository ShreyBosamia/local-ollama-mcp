#!/usr/bin/env python3
import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import time
from typing import Any
from uuid import uuid4

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("local-ollama")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CODE_MODEL = os.getenv("LOCAL_CODE_MODEL", "qwen2.5-coder:7b-instruct-q5_K_M")
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

THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
UNCLOSED_THINK_RE = re.compile(r"^\s*<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
THINK_MARKER_RE = re.compile(r"</?think\b", re.IGNORECASE)
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


def strip_thinking(content: str) -> str:
    """Remove hidden-reasoning blocks that some local models leak in content."""
    content = THINK_BLOCK_RE.sub("", content)
    content = UNCLOSED_THINK_RE.sub("", content)
    return content.strip()


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
        return "needs_raw_verification"
    if input_tokens < 120:
        return "skip_for_small_context"

    reduction_pct = (input_tokens - output_tokens) / input_tokens if input_tokens else 0.0
    if reduction_pct >= 0.40:
        return "use_local"
    if reduction_pct > 0:
        return "needs_raw_verification"
    return "skip_for_small_context"


def append_ledger_record(record: dict[str, Any]) -> None:
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str, ensure_ascii=False) + "\n")


def build_tool_record(
    *,
    task_id: str,
    tool_name: str,
    model: str,
    input_payload: dict[str, Any],
    local_output: Any,
    latency_ms: int,
    error: str | None = None,
) -> dict[str, Any]:
    stored_input, input_flags = redact_value(input_payload)
    stored_output, output_flags = redact_value(local_output)
    stored_error, error_flags = redact_value(error) if error else (None, [])
    input_text = compact_json_text(stored_input)
    output_text = str(stored_output)
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
    return {
        "record_type": "tool_call",
        "task_id": task_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "model": model,
        "privacy_mode": "raw" if raw_capture_enabled() else "redacted",
        "input": stored_input,
        "local_output": stored_output,
        "latency_ms": latency_ms,
        "token_estimates": {
            "input": input_tokens,
            "local_output": output_tokens,
            "context_reduction": input_tokens - output_tokens,
            "context_reduction_pct": round((input_tokens - output_tokens) / input_tokens, 3)
            if input_tokens
            else 0.0,
        },
        "recommendation": recommendation_for_capture(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            risk_flags=risk_flags,
        ),
        "risk_flags": risk_flags,
        "error": stored_error,
    }


async def capture_tool_call(
    tool_name: str,
    model: str,
    input_payload: dict[str, Any],
    action: Callable[[], Awaitable[str]],
) -> str:
    if not capture_enabled():
        return await action()

    task_id = uuid4().hex
    started = time.perf_counter()
    try:
        output = await action()
    except Exception as exc:
        latency_ms = round((time.perf_counter() - started) * 1000)
        error_text = f"{type(exc).__name__}: {exc}"
        append_ledger_record(
            build_tool_record(
                task_id=task_id,
                tool_name=tool_name,
                model=model,
                input_payload=input_payload,
                local_output="",
                latency_ms=latency_ms,
                error=error_text,
            )
        )
        raise

    latency_ms = round((time.perf_counter() - started) * 1000)
    append_ledger_record(
        build_tool_record(
            task_id=task_id,
            tool_name=tool_name,
            model=model,
            input_payload=input_payload,
            local_output=output,
            latency_ms=latency_ms,
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


async def run_command(*args: str, timeout: float = 10) -> tuple[bool, str]:
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
        return False, f"{args[0]} timed out after {timeout:.0f}s"

    output = (stdout.decode(errors="replace") + stderr.decode(errors="replace")).strip()
    return process.returncode == 0, output


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
) -> str:
    data = await ollama_chat(
        model,
        prompt,
        temperature=temperature,
        num_predict=num_predict,
        num_ctx=num_ctx,
        keep_alive=keep_alive,
        system=system,
    )
    message = data.get("message", {})
    return strip_thinking(str(message.get("content", "")))


@mcp.tool()
async def local_summarize(
    text: str,
    focus: str = "important implementation details",
    num_ctx: int = DEFAULT_NUM_CTX,
) -> str:
    """
    Summarize code, logs, docs, or command output using a local Ollama model.
    Returns a compact summary for Codex.
    """
    prompt = f"""
Summarize the following content for Codex.

Focus: {focus}

Rules:
- Max 6 bullets.
- Mention only details useful for the current coding task.
- Include file/function evidence when visible.
- Do not quote large blocks.
- Do not include chain-of-thought.
- Do not include raw model scratchpad text.

Content:
{text}
"""
    return await capture_tool_call(
        "local_summarize",
        CODE_MODEL,
        {"text": text, "focus": focus, "num_ctx": num_ctx},
        lambda: ask_ollama(CODE_MODEL, prompt, num_predict=450, num_ctx=num_ctx),
    )


@mcp.tool()
async def local_code_review(
    diff: str,
    focus: str = "bugs, regressions, missing tests",
    num_ctx: int = DEFAULT_NUM_CTX,
) -> str:
    """
    Review a git diff using a local coding model.
    Returns only likely issues that Codex should verify.
    """
    prompt = f"""
Review this git diff for Codex.

Focus: {focus}

Rules:
- Max 5 findings.
- Each finding must include: issue, file/function evidence, suggested check.
- Include "No obvious issue found" if nothing stands out.
- Do not suggest applying patches directly.
- Do not include chain-of-thought.
- Keep each finding to 3 short lines or less.

Diff:
{diff}
"""
    return await capture_tool_call(
        "local_code_review",
        CODE_MODEL,
        {"diff": diff, "focus": focus, "num_ctx": num_ctx},
        lambda: ask_ollama(CODE_MODEL, prompt, num_predict=700, num_ctx=num_ctx),
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
    prompt = f"""
Generate test ideas for Codex.

Test framework: {test_framework}

Rules:
- Max 7 tests.
- Group by unit/integration/regression if useful.
- Include edge cases.
- Do not write full test files unless asked.
- Do not include chain-of-thought.
- Keep each test idea to one sentence.

Code or diff:
{code_or_diff}
"""
    return await capture_tool_call(
        "local_test_ideas",
        CODE_MODEL,
        {"code_or_diff": code_or_diff, "test_framework": test_framework, "num_ctx": num_ctx},
        lambda: ask_ollama(CODE_MODEL, prompt, num_predict=600, num_ctx=num_ctx),
    )


@mcp.tool()
async def local_reason_check(problem: str, num_ctx: int = DEFAULT_NUM_CTX) -> str:
    """
    Ask a local reasoning model for a concise second opinion on a debugging problem.
    """
    prompt = f"""
Give a concise second opinion for this debugging/problem-solving task.

Rules:
- Max 5 bullets.
- Include likely causes and next checks.
- No chain-of-thought.
- No long explanations.
- No raw <think> blocks.

Problem:
{problem}
"""
    return await capture_tool_call(
        "local_reason_check",
        REASON_MODEL,
        {"problem": problem, "num_ctx": num_ctx},
        lambda: ask_ollama(
            REASON_MODEL,
            prompt,
            temperature=0.1,
            num_predict=380,
            num_ctx=num_ctx,
            keep_alive="0",
            system=(
                "You are a diagnostic second-opinion model for Codex. "
                "Return short conclusions only. Do not include chain-of-thought."
            ),
        ),
    )


@mcp.tool()
async def local_plan_check(problem: str, num_ctx: int = DEFAULT_NUM_CTX) -> str:
    """
    Ask the opt-in planning model for a compact implementation plan.
    The 9B model is not kept warm by default because it is high-risk on 8GB VRAM.
    """
    prompt = f"""
Create a compact implementation plan for Codex.

Rules:
- Max 6 bullets.
- Include risk checks and verification steps.
- Prefer concrete files/functions when provided.
- Do not include chain-of-thought.
- Do not include raw <think> blocks.

Problem:
{problem}
"""
    return await capture_tool_call(
        "local_plan_check",
        PLAN_MODEL,
        {"problem": problem, "num_ctx": num_ctx},
        lambda: ask_ollama(
            PLAN_MODEL,
            prompt,
            temperature=0.2,
            num_predict=520,
            num_ctx=num_ctx,
            keep_alive="0",
            system=(
                "You are an opt-in planning model for Codex. "
                "Be direct and concise. Do not include chain-of-thought."
            ),
        ),
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
        run_status,
    )


@mcp.tool()
async def local_capture_status() -> str:
    """
    Report whether local MCP capture is enabled and where redacted records go.
    """
    path = ledger_path()
    records = read_ledger_records(path)
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
        f"tool_record_count: {len(tool_records)}",
        f"outcome_record_count: {len(outcome_records)}",
        "recent_task_ids: " + (", ".join(recent_task_ids) if recent_task_ids else "none"),
    ]
    return "\n".join(lines)


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


if __name__ == "__main__":
    mcp.run()
