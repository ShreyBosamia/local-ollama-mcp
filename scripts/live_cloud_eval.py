#!/usr/bin/env python3
"""
Run live hybrid MCP checks across local Qwen and Antigravity-backed cloud tools.

This script is intentionally report-oriented: cloud failures are recorded as
case outcomes instead of raising, so one run can still validate local behavior,
routing decisions, fallback prefixes, and hardware snapshots.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import server  # noqa: E402


@dataclass(frozen=True)
class LiveCase:
    name: str
    tool: str
    payload: str
    focus: str
    expected_tier: str
    expected_model_role: str
    required_terms: tuple[str, ...] = ()
    min_reduction_pct: float = 0.0


@dataclass
class LiveResult:
    name: str
    tool: str
    expected_tier: str
    selected_tier: str
    expected_model_role: str
    artifact_tokens_est: int
    output_tokens_est: int
    reduction_pct: float
    latency_ms: int
    status: str
    fallback_prefix: str
    think_leak: bool
    missing_required_terms: list[str]
    usefulness_notes: list[str]
    output_preview: str
    error: str


def run_capture(cmd: list[str], timeout: float = 20.0) -> dict[str, object]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=ROOT_DIR,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "duration_ms": round((time.perf_counter() - started) * 1000),
            "stdout": proc.stdout.strip()[-4000:],
            "stderr": proc.stderr.strip()[-4000:],
        }
    except Exception as exc:
        return {
            "cmd": cmd,
            "returncode": None,
            "duration_ms": round((time.perf_counter() - started) * 1000),
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def repeat_until_tokens(seed: str, target_tokens: int) -> str:
    lines = [seed]
    while server.estimate_tokens("\n".join(lines)) < target_tokens:
        lines.append(seed)
    return "\n".join(lines)


def build_cases(skip_cloud: bool = False) -> list[LiveCase]:
    small_log = repeat_until_tokens(
        "ERROR provider import failed: Place is required in sanityProviderWrite.ts:114.",
        450,
    )
    small_diff = repeat_until_tokens(
        "diff --git a/billing.py b/billing.py\n+ total = item.price\n- total += item.price",
        650,
    )
    small_plan = repeat_until_tokens(
        "Plan a contained refactor for a Python MCP eval harness with stable reports.",
        550,
    )
    small_reason = repeat_until_tokens(
        "A local MCP server sometimes returns an empty summary after cleaning logs. Identify next checks.",
        550,
    )
    large_log = repeat_until_tokens(
        "2026-05-28T12:00:00Z ERROR pipeline/findhelpDirectory.ts:91 directory_expansion capped at 50 provider candidates; canonical_url fallback used.",
        4300,
    )
    large_diff = repeat_until_tokens(
        "diff --git a/app/provider.py b/app/provider.py\n+ provider.canonical_url = directory_url\n- provider.canonical_url = provider_website\n+ skipped_urls.append(url)",
        4600,
    )
    huge_diff = repeat_until_tokens(
        "diff --git a/app/review.tsx b/app/review.tsx\n+ setStaffView(rawJson)\n- setStaffView(renderStructuredDiff(rawJson))\n+ console.log(providerPayload)",
        12500,
    )
    large_plan = repeat_until_tokens(
        "Design a multi-repo rollout for a staff review UI, provider ingestion diagnostics, and MCP eval dashboard without exposing raw JSON to staff.",
        4300,
    )
    large_reason = repeat_until_tokens(
        "Debug a race where local model capture, Codex telemetry import, and Antigravity fallback write conflicting ledger rows.",
        4300,
    )

    cases = [
        LiveCase("qwen_summarize_small", "local_summarize", small_log, "exact errors and files", "local_gpu", "qwen3.5:9b", ("Place is required", "sanityProviderWrite.ts")),
        LiveCase("qwen_code_review_small", "local_code_review", small_diff, "logic regressions", "local_gpu", "qwen3.5:9b", ("billing.py",)),
        LiveCase("qwen_test_ideas_small", "local_test_ideas", small_diff, "pytest", "local_gpu", "qwen3.5:9b", ("pytest",)),
        LiveCase("qwen_plan_small", "local_plan_check", small_plan, "", "local_gpu", "qwen3.5:9b", ("refactor", "eval harness")),
        LiveCase("qwen_reason_small", "local_reason_check", small_reason, "", "local_gpu", "qwen3.5:9b", ("empty summary",)),
    ]

    if not skip_cloud:
        cases.extend(
            [
                LiveCase("gemini_summarize_large", "local_summarize", large_log, "directory expansion diagnostics", "antigravity", "gemini_flash", ("directory_expansion", "canonical_url"), 0.80),
                LiveCase("gemini_code_review_large", "local_code_review", large_diff, "canonical URL regressions", "antigravity", "gemini_flash", ("canonical_url", "provider_website"), 0.70),
                LiveCase("gemini_compress_diff_huge", "agy_compress_diff", huge_diff, "staff JSON exposure and secret logging", "antigravity", "gemini_flash", ("CHANGED_FILES", "BEHAVIOR_CHANGES", "RISKY_LINES", "REMOVED_LOGIC"), 0.80),
                LiveCase(
                    "gemini_generate_walkthrough",
                    "local_generate_walkthrough",
                    "Generate a walkthrough for the current local MCP Gemini route validation changes.",
                    "",
                    "antigravity",
                    "gemini_flash",
                    ("# Walkthrough", "Executive Summary", "Component & File Changes", "Familiarization & Verification Guide", "Risks & Edge Cases"),
                ),
                LiveCase("claude_plan_large", "local_plan_check", large_plan, "", "antigravity", "claude_thinking"),
                LiveCase("claude_reason_large", "local_reason_check", large_reason, "", "antigravity", "claude_thinking"),
            ]
        )
    return cases


def selected_tier_for(case: LiveCase) -> str:
    if case.tool == "local_test_ideas":
        return "local_gpu"
    if case.tool == "local_generate_walkthrough":
        return "antigravity" if server.AGY_ENABLED else "local_gpu"
    token_count = server.estimate_tokens(case.payload)
    if server.AGY_ENABLED and token_count >= server.AGY_ROUTING_MIN_TOKENS:
        return "antigravity"
    return "local_gpu"


def artifact_tokens_for(case: LiveCase) -> int:
    if case.tool != "local_generate_walkthrough":
        return server.estimate_tokens(case.payload)
    try:
        proc = subprocess.run(
            ["git", "diff", "HEAD~1"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=ROOT_DIR,
        )
    except Exception:
        return server.estimate_tokens(case.payload)
    if proc.returncode != 0 or not proc.stdout.strip():
        return server.estimate_tokens(case.payload)
    return server.estimate_tokens(f"{case.payload}\n{proc.stdout}")


def antigravity_settings_model() -> str:
    settings_path = Path.home() / ".gemini" / "antigravity-cli" / "settings.json"
    try:
        return str(json.loads(settings_path.read_text(encoding="utf-8")).get("model", ""))
    except Exception:
        return ""


def model_selection_available(case: LiveCase, *, supports_model_flag: bool, settings_model: str) -> bool:
    if case.expected_tier != "antigravity":
        return True
    if supports_model_flag:
        return True
    if case.expected_model_role == "gemini_flash":
        return "gemini" in settings_model.lower()
    return False


async def call_case(case: LiveCase) -> str:
    if case.tool == "local_summarize":
        return await server.local_summarize(case.payload, focus=case.focus)
    if case.tool == "local_code_review":
        return await server.local_code_review(case.payload, focus=case.focus)
    if case.tool == "local_test_ideas":
        return await server.local_test_ideas(case.payload, test_framework=case.focus or "unknown")
    if case.tool == "local_plan_check":
        return await server.local_plan_check(case.payload)
    if case.tool == "local_reason_check":
        return await server.local_reason_check(case.payload)
    if case.tool == "agy_compress_diff":
        return await server.agy_compress_diff(case.payload, focus=case.focus)
    if case.tool == "local_generate_walkthrough":
        return await server.local_generate_walkthrough(
            commit_or_branch="HEAD~1",
            task_description=case.payload,
            write_to_file=False,
        )
    raise ValueError(f"Unknown tool: {case.tool}")


def fallback_prefix(output: str) -> str:
    first = output.strip().splitlines()[0] if output.strip() else ""
    return first if first.startswith("[agy_") else ""


def has_unquoted_think_marker(output: str) -> bool:
    without_fences = re.sub(r"```.*?```", "", output, flags=re.DOTALL)
    without_inline_code = re.sub(r"`[^`]*`", "", without_fences)
    return bool(server.THINK_MARKER_RE.search(without_inline_code))


async def evaluate_case(
    case: LiveCase,
    *,
    supports_model_flag: bool,
    settings_model: str,
) -> LiveResult:
    started = time.perf_counter()
    selected_tier = selected_tier_for(case)
    try:
        output = await call_case(case)
        error = ""
        status = "pass"
    except Exception as exc:
        output = ""
        error = f"{type(exc).__name__}: {exc}"
        status = "error"
    latency_ms = round((time.perf_counter() - started) * 1000)
    artifact_tokens = artifact_tokens_for(case)
    output_tokens = server.estimate_tokens(output)
    reduction_pct = (
        round((artifact_tokens - output_tokens) / artifact_tokens, 3)
        if artifact_tokens
        else 0.0
    )
    prefix = fallback_prefix(output)
    if prefix and case.expected_tier == "antigravity":
        status = "cloud_fallback"
    if selected_tier != case.expected_tier:
        status = "route_mismatch"
    if status == "pass" and not model_selection_available(
        case,
        supports_model_flag=supports_model_flag,
        settings_model=settings_model,
    ):
        status = "model_selection_unavailable"
    think_leak = has_unquoted_think_marker(output)
    if think_leak:
        status = "think_leak"

    missing_terms = [
        term
        for term in case.required_terms
        if term.lower() not in output.lower()
    ]
    usefulness_notes: list[str] = []
    if missing_terms:
        usefulness_notes.append("missing required terms: " + ", ".join(missing_terms))
    if case.min_reduction_pct and reduction_pct < case.min_reduction_pct:
        usefulness_notes.append(
            f"reduction {reduction_pct:.1%} below minimum {case.min_reduction_pct:.1%}"
        )
    if status == "pass" and usefulness_notes:
        status = "usefulness_fail"

    return LiveResult(
        name=case.name,
        tool=case.tool,
        expected_tier=case.expected_tier,
        selected_tier=selected_tier,
        expected_model_role=case.expected_model_role,
        artifact_tokens_est=artifact_tokens,
        output_tokens_est=output_tokens,
        reduction_pct=reduction_pct,
        latency_ms=latency_ms,
        status=status,
        fallback_prefix=prefix,
        think_leak=think_leak,
        missing_required_terms=missing_terms,
        usefulness_notes=usefulness_notes,
        output_preview=" ".join(output.split())[:320],
        error=error,
    )


def render_markdown(results: dict[str, object]) -> str:
    cases = results.get("cases", [])
    unavailable_cases = [c for c in cases if c.get("status") == "model_selection_unavailable"]

    lines = [
        "# Hybrid MCP Live Evaluation",
        "",
        f"- Generated: `{results['generated_at']}`",
        f"- Code model: `{results['model']}`",
        f"- Plan model: `{results['plan_model']}`",
        f"- Reason model: `{results['reason_model']}`",
        f"- Routing threshold: `{results['routing_threshold_tokens']}` tokens",
        f"- agy per-call model selection: `{results['agy_model_flag_supported']}`",
        f"- agy settings model: `{results['agy_settings_model']}`",
        "",
    ]

    if unavailable_cases:
        lines.extend([
            "> [!WARNING]",
            "> **Model Selection Unavailable Detected!**",
            "> Some plan/reason cloud checks expected Claude/Opus thinking routes, but the Antigravity CLI does not support per-call model selection (or configured defaults differ). The following cases were routed to agy default but could not be confirmed as Claude/Opus Thinking:",
        ])
        for c in unavailable_cases:
            lines.append(f"> - `{c['name']}` (`{c['tool']}`)")
        lines.extend(["", ""])

    lines.extend([
        "## Preflight",
        "",
        "```json",
        json.dumps(results["preflight"], indent=2),
        "```",
        "",
        "## Cases",
        "",
        "| Case | Tool | Tokens | Expected | Selected | Status | Latency | Reduction | Fallback |",
        "| --- | --- | ---: | --- | --- | --- | ---: | ---: | --- |",
    ])
    for case in results["cases"]:
        lines.append(
            f"| `{case['name']}` | `{case['tool']}` | {case['artifact_tokens_est']} | "
            f"`{case['expected_tier']}` | `{case['selected_tier']}` | `{case['status']}` | "
            f"{case['latency_ms']} ms | {case['reduction_pct']:.1%} | `{case['fallback_prefix']}` |"
        )
    lines.extend(["", "## Output Previews", ""])
    for case in results["cases"]:
        lines.extend(
            [
                f"### {case['name']}",
                "",
                f"- Status: `{case['status']}`",
                f"- Usefulness notes: {'; '.join(case['usefulness_notes']) if case['usefulness_notes'] else '(none)'}",
                f"- Preview: {case['output_preview'] or case['error'] or '(empty)'}",
                "",
            ]
        )
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run live hybrid Qwen/cloud MCP checks.")
    parser.add_argument("--output-dir", default="", help="Report directory. Defaults to .local_ollama_mcp/eval_runs/live_<timestamp>.")
    parser.add_argument("--skip-cloud", action="store_true", help="Only run local Qwen cases.")
    parser.add_argument("--no-qwen-only", action="store_true", help="Do not force plan/reason tools to qwen3.5:9b for local cases.")
    args = parser.parse_args()

    if not args.no_qwen_only:
        server.configure_models(
            code_model="qwen3.5:9b",
            plan_model="qwen3.5:9b",
            reason_model="qwen3.5:9b",
            warm_model="qwen3.5:9b",
        )

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else ROOT_DIR / ".local_ollama_mcp" / "eval_runs" / f"live_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    preflight = {
        "agy_model_flag_supported": await server.agy_supports_model_flag(),
        "agy_settings_model": antigravity_settings_model(),
        "agy_usage": run_capture([server.AGY_BIN, "/usage"], timeout=server.AGY_USAGE_TIMEOUT),
        "ollama_ps_before": run_capture(["ollama", "ps"], timeout=10),
        "nvidia_smi_before": run_capture(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,clocks.gr",
                "--format=csv",
            ],
            timeout=10,
        ),
    }

    case_results = []
    for case in build_cases(skip_cloud=args.skip_cloud):
        case_results.append(
            asdict(
                await evaluate_case(
                    case,
                    supports_model_flag=bool(preflight["agy_model_flag_supported"]),
                    settings_model=str(preflight["agy_settings_model"]),
                )
            )
        )

    preflight["ollama_ps_after"] = run_capture(["ollama", "ps"], timeout=10)
    preflight["nvidia_smi_after"] = run_capture(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,clocks.gr",
            "--format=csv",
        ],
        timeout=10,
    )

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": server.CODE_MODEL,
        "plan_model": server.PLAN_MODEL,
        "reason_model": server.REASON_MODEL,
        "routing_threshold_tokens": server.AGY_ROUTING_MIN_TOKENS,
        "agy_model_flag_supported": preflight["agy_model_flag_supported"],
        "agy_settings_model": preflight["agy_settings_model"],
        "preflight": preflight,
        "cases": case_results,
    }

    json_path = output_dir / "results.json"
    markdown_path = output_dir / "report.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(results), encoding="utf-8")

    status_counts: dict[str, int] = {}
    for result in case_results:
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1

    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    print(json.dumps(status_counts, sort_keys=True))

    unavailable_count = status_counts.get("model_selection_unavailable", 0)
    if unavailable_count > 0:
        print("\n" + "!" * 80)
        print(f"WARNING: {unavailable_count} cases returned 'model_selection_unavailable'.")
        print("Claude/Opus Thinking routes could not be honestly verified because")
        print("per-call model selection is not supported or misconfigured.")
        print("!" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
