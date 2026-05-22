#!/usr/bin/env python3
"""
Evaluate qwen2.5-coder:7b-instruct-q5_K_M as a local MCP helper for Codex.

The harness is intentionally self-contained: it calls the existing server.py
helpers directly, scores local outputs against expert baseline facts, and
estimates how much raw context would be avoided by sending the local output to
the cloud model instead of the original artifact.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import textwrap
import time

import server


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
THINK_RE = re.compile(r"</?think\b", re.IGNORECASE)

CLOUD_ONLY_PROMPT = """You are GPT-5.5/Codex. Complete this coding-assistant task from the raw artifact.

Task:
{task}

Raw artifact:
{artifact}
"""

LOCAL_ASSISTED_PROMPT = """You are GPT-5.5/Codex. Complete this coding-assistant task using the local MCP summary first.
Verify the local output for plausibility and ask for raw context only if critical details are missing.

Task:
{task}

Local MCP output:
{local_output}
"""


@dataclass(frozen=True)
class ExpectedFact:
    label: str
    pattern: str
    required: bool = True


@dataclass(frozen=True)
class EvalCase:
    name: str
    category: str
    tool: str
    task: str
    artifact: str
    expected_facts: tuple[ExpectedFact, ...]
    focus: str = ""
    test_framework: str = "unknown"
    expected_recommendation: str = "use_local"
    forbidden_facts: tuple[ExpectedFact, ...] = ()


@dataclass
class CaseResult:
    name: str
    category: str
    tool: str
    recommendation: str
    expected_recommendation: str
    decision_matches_expected: bool
    accuracy_score: float
    required_facts_hit: int
    required_facts_total: int
    optional_facts_hit: int
    optional_facts_total: int
    forbidden_facts_hit: int
    forbidden_fact_labels: list[str]
    missing_required_facts: list[str]
    missing_optional_facts: list[str]
    think_leak: bool
    raw_cloud_tokens_est: int
    assisted_cloud_tokens_est: int
    local_output_tokens_est: int
    estimated_cloud_token_reduction: int
    estimated_cloud_token_reduction_pct: float
    compression_ratio: float
    latency_ms: int
    output_preview: str
    local_output: str


def estimate_tokens(text: str) -> int:
    """Cheap, stable token estimate for relative cloud-token comparisons."""
    return len(TOKEN_RE.findall(text))


def one_line(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def fact_hit(output: str, fact: ExpectedFact) -> bool:
    return re.search(fact.pattern, output, re.IGNORECASE | re.DOTALL) is not None


def classify_result(
    *,
    case: EvalCase,
    accuracy_score: float,
    required_facts_hit: int,
    required_facts_total: int,
    think_leak: bool,
    token_reduction_pct: float,
    forbidden_facts_hit: int,
) -> str:
    if case.expected_recommendation == "skip_local":
        if token_reduction_pct < 0.20:
            return "skip_local"
        if accuracy_score >= 0.80 and not think_leak and forbidden_facts_hit == 0:
            return "optional_local"
        return "skip_local"

    if think_leak:
        return "raw_cloud"
    if forbidden_facts_hit:
        if accuracy_score >= 0.60 and token_reduction_pct > 0:
            return "verify_raw"
        return "raw_cloud"
    if required_facts_total and required_facts_hit < required_facts_total:
        if accuracy_score >= 0.60 and token_reduction_pct > 0:
            return "verify_raw"
        return "raw_cloud"
    if token_reduction_pct >= 0.40 and accuracy_score >= 0.80:
        return "use_local"
    if token_reduction_pct > 0 and accuracy_score >= 0.60:
        return "verify_raw"
    return "raw_cloud"


def build_cases() -> list[EvalCase]:
    server_noise = "\n".join(
        f"# context line {index}: existing MCP helper behavior, timeout handling, and telemetry notes"
        for index in range(40)
    )
    billing_noise = "\n".join(
        f"@@ unchanged billing fixture {index}\n def fixture_{index}(items):\n     return [item.sku for item in items]"
        for index in range(35)
    )
    retry_noise = "\n".join(
        f"def helper_{index}(value):\n    # unrelated retry utility preserved for compatibility\n    return str(value).strip()\n"
        for index in range(30)
    )
    log_noise = "\n".join(
        f"INFO request_id=abc poll={index} provider import still waiting; duplicate retry line"
        for index in range(70)
    )

    server_diff = f"""{server_noise}
diff --git a/server.py b/server.py
@@
-        "options": {{"temperature": temperature, "num_ctx": 8192}}
+        "think": False,
+        "keep_alive": keep_alive,
+        "options": {{"temperature": temperature, "num_ctx": 4096, "num_predict": 512}}
@@
-        return data["message"]["content"].strip()
+        return strip_thinking(data["message"].get("content", ""))
@@
+async def local_warm_model(model=CODE_MODEL, keep_alive="2h"):
+    # warm the model and check ollama ps for 100% GPU residency
+    ...
{server_noise}
"""

    total_diff = f"""diff --git a/billing.py b/billing.py
{billing_noise}
@@
 def invoice_total(items):
     total = 0
     for item in items:
-        total += item.price * item.quantity
+        total = item.price * item.quantity
     return total
@@
 def apply_discount(total, discount):
-    return total - discount
+    return max(0, total - discount)
{billing_noise}
"""

    retry_after_code = f"""from email.utils import parsedate_to_datetime
from datetime import datetime, timezone

{retry_noise}

def parse_retry_after(value, now=None):
    if now is None:
        now = datetime.now(timezone.utc)
    if value.isdigit():
        return int(value)
    retry_at = parsedate_to_datetime(value)
    return max(0, int((retry_at - now).total_seconds()))

{retry_noise}
"""

    noisy_logs = f"""
{log_noise}
INFO request_id=abc retrying provider import
WARN request_id=abc duplicate source URL ignored
ERROR service.py:218 Place is required.
INFO request_id=abc retrying provider import
INFO request_id=abc retrying provider import
WARN ui/DialogContent.tsx:44 DialogContent requires a DialogTitle for accessibility.
ERROR service.py:218 Place is required.
DEBUG payload={{"raw_json": "... 900 repeated fields omitted ..."}}
ERROR pipeline/findhelpDirectory.ts:91 directory_expansion capped at 50 provider candidates
{log_noise}
"""

    small_code = "def add(a, b):\n    return a + b\n"

    return [
        EvalCase(
            name="server_diff_summary",
            category="summarization",
            tool="local_summarize",
            task="Summarize implementation details Codex must preserve while editing the local Ollama MCP server.",
            artifact=server_diff,
            focus="implementation details Codex would need to preserve while editing",
            expected_facts=(
                ExpectedFact("num_ctx reduced to 4096", r"4096|num_ctx"),
                ExpectedFact("num_predict added or capped", r"num_predict|predict"),
                ExpectedFact("keep_alive preserved", r"keep[_ -]?alive|resident|warm"),
                ExpectedFact("thinking stripped or disabled", r"think|strip_thinking|scratchpad|disabled"),
            ),
            forbidden_facts=(
                ExpectedFact("contradicts think false addition", r"remove[^.\n]*think[^.\n]*false|remove[^.\n]*\"think\""),
            ),
        ),
        EvalCase(
            name="invoice_diff_review",
            category="code_review",
            tool="local_code_review",
            task="Review the diff for likely bugs and identify the most important regression.",
            artifact=total_diff,
            focus="bugs and regressions",
            expected_facts=(
                ExpectedFact("total is overwritten instead of accumulated", r"overwrit|reset|replace|accumulat|\+=|total\s*="),
                ExpectedFact("loop or multiple items affected", r"loop|item|multiple|last"),
                ExpectedFact("discount clamp is likely intentional or less risky", r"discount|clamp|max\(0|less risky|not.*issue", required=False),
            ),
            forbidden_facts=(
                ExpectedFact("treats added discount clamp as the main bug", r"apply_discount[^.\n]*(does not|doesn't|bug|issue).*negative|negative discounts"),
                ExpectedFact("generic division by zero claim", r"division by zero"),
            ),
        ),
        EvalCase(
            name="retry_after_test_ideas",
            category="test_ideas",
            tool="local_test_ideas",
            task="Generate concise test ideas for parse_retry_after.",
            artifact=retry_after_code,
            test_framework="pytest",
            expected_facts=(
                ExpectedFact("numeric seconds case", r"numeric|digit|seconds|integer"),
                ExpectedFact("HTTP date future case", r"future|date|parsedate|HTTP"),
                ExpectedFact("past date clamps to zero", r"past|zero|clamp|0"),
                ExpectedFact("invalid input raises or is handled", r"invalid|malformed|raise|error"),
            ),
            forbidden_facts=(
                ExpectedFact("writes full test code instead of concise ideas", r"```|def test_|import pytest"),
            ),
        ),
        EvalCase(
            name="noisy_log_compression",
            category="compression",
            tool="local_summarize",
            task="Compress noisy logs while preserving exact actionable errors and affected files.",
            artifact=noisy_logs,
            focus="exact errors, affected files, and repeated noise to ignore",
            expected_facts=(
                ExpectedFact("Place is required exact error", r"Place is required"),
                ExpectedFact("DialogTitle accessibility warning", r"DialogTitle|accessibility"),
                ExpectedFact("directory_expansion cap", r"directory_expansion|capped|50"),
                ExpectedFact("repeated noise collapsed", r"repeat|duplicate|multiple|noise", required=False),
            ),
        ),
        EvalCase(
            name="tiny_input_negative_control",
            category="negative_control",
            tool="local_summarize",
            task="Summarize a tiny helper function.",
            artifact=small_code,
            focus="implementation behavior",
            expected_recommendation="skip_local",
            expected_facts=(
                ExpectedFact("addition behavior", r"add|sum|\+"),
            ),
        ),
    ]


async def call_tool(case: EvalCase) -> str:
    if case.tool == "local_summarize":
        return await server.local_summarize(case.artifact, focus=case.focus)
    if case.tool == "local_code_review":
        return await server.local_code_review(case.artifact, focus=case.focus)
    if case.tool == "local_test_ideas":
        return await server.local_test_ideas(case.artifact, test_framework=case.test_framework)
    raise ValueError(f"unknown tool: {case.tool}")


async def evaluate_case(case: EvalCase) -> CaseResult:
    started = time.perf_counter()
    output = await call_tool(case)
    latency_ms = round((time.perf_counter() - started) * 1000)

    required = [fact for fact in case.expected_facts if fact.required]
    optional = [fact for fact in case.expected_facts if not fact.required]
    missing_required = [fact.label for fact in required if not fact_hit(output, fact)]
    missing_optional = [fact.label for fact in optional if not fact_hit(output, fact)]
    forbidden_hits = [fact.label for fact in case.forbidden_facts if fact_hit(output, fact)]
    required_hits = len(required) - len(missing_required)
    optional_hits = len(optional) - len(missing_optional)
    total_facts = len(required) + len(optional)
    hit_count = required_hits + optional_hits
    accuracy_score = hit_count / total_facts if total_facts else 1.0

    raw_cloud = CLOUD_ONLY_PROMPT.format(task=case.task, artifact=case.artifact)
    assisted_cloud = LOCAL_ASSISTED_PROMPT.format(task=case.task, local_output=output)
    raw_tokens = estimate_tokens(raw_cloud)
    assisted_tokens = estimate_tokens(assisted_cloud)
    local_tokens = estimate_tokens(output)
    token_reduction = raw_tokens - assisted_tokens
    token_reduction_pct = token_reduction / raw_tokens if raw_tokens else 0.0
    think_leak = THINK_RE.search(output) is not None

    recommendation = classify_result(
        case=case,
        accuracy_score=accuracy_score,
        required_facts_hit=required_hits,
        required_facts_total=len(required),
        think_leak=think_leak,
        token_reduction_pct=token_reduction_pct,
        forbidden_facts_hit=len(forbidden_hits),
    )

    return CaseResult(
        name=case.name,
        category=case.category,
        tool=case.tool,
        recommendation=recommendation,
        expected_recommendation=case.expected_recommendation,
        decision_matches_expected=recommendation == case.expected_recommendation
        or (case.expected_recommendation == "use_local" and recommendation == "verify_raw"),
        accuracy_score=round(accuracy_score, 3),
        required_facts_hit=required_hits,
        required_facts_total=len(required),
        optional_facts_hit=optional_hits,
        optional_facts_total=len(optional),
        forbidden_facts_hit=len(forbidden_hits),
        forbidden_fact_labels=forbidden_hits,
        missing_required_facts=missing_required,
        missing_optional_facts=missing_optional,
        think_leak=think_leak,
        raw_cloud_tokens_est=raw_tokens,
        assisted_cloud_tokens_est=assisted_tokens,
        local_output_tokens_est=local_tokens,
        estimated_cloud_token_reduction=token_reduction,
        estimated_cloud_token_reduction_pct=round(token_reduction_pct, 3),
        compression_ratio=round(assisted_tokens / raw_tokens, 3) if raw_tokens else 0.0,
        latency_ms=latency_ms,
        output_preview=one_line(output),
        local_output=output,
    )


def render_markdown(results: dict) -> str:
    summary = results["summary"]
    lines = [
        "# Local Ollama MCP Evaluation",
        "",
        f"- Generated: `{results['generated_at']}`",
        f"- Model: `{results['model']}`",
        f"- Cases: `{summary['case_count']}`",
        f"- Useful local recommendations: `{summary['use_local_count']}`",
        f"- Optional local recommendations: `{summary['optional_local_count']}`",
        f"- Verify-raw recommendations: `{summary['verify_raw_count']}`",
        f"- Skip-local recommendations: `{summary['skip_local_count']}`",
        f"- Raw-cloud recommendations: `{summary['raw_cloud_count']}`",
        f"- Aggregate estimated cloud-token reduction: `{summary['aggregate_token_reduction_pct']:.1%}`",
        f"- Average accuracy score: `{summary['average_accuracy_score']:.1%}`",
        f"- Think leakage observed: `{'yes' if summary['think_leak_count'] else 'no'}`",
        f"- Contradiction/format risk flags: `{summary['forbidden_fact_hit_count']}`",
        "",
        "## Status Before",
        "",
        "```text",
        results["status_before"].strip(),
        "```",
        "",
        "## Warm Result",
        "",
        "```text",
        results["warm_result"].strip(),
        "```",
        "",
        "## Case Results",
        "",
        "| Case | Tool | Recommendation | Accuracy | Token Reduction | Latency | Missing Required Facts | Risk Flags |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]

    for case in results["cases"]:
        missing = ", ".join(case["missing_required_facts"]) or "-"
        risks = ", ".join(case["forbidden_fact_labels"]) or "-"
        lines.append(
            "| {name} | `{tool}` | `{recommendation}` | {accuracy:.0%} | {reduction:.0%} | {latency} ms | {missing} | {risks} |".format(
                name=case["name"],
                tool=case["tool"],
                recommendation=case["recommendation"],
                accuracy=case["accuracy_score"],
                reduction=case["estimated_cloud_token_reduction_pct"],
                latency=case["latency_ms"],
                missing=missing,
                risks=risks,
            )
        )

    lines.extend(["", "## Output Previews", ""])
    for case in results["cases"]:
        lines.extend(
            [
                f"### {case['name']}",
                "",
                f"- Recommendation: `{case['recommendation']}`",
                f"- Raw cloud tokens est: `{case['raw_cloud_tokens_est']}`",
                f"- Assisted cloud tokens est: `{case['assisted_cloud_tokens_est']}`",
                f"- Local output tokens est: `{case['local_output_tokens_est']}`",
                "",
                "```text",
                case["local_output"].strip(),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "- `use_local` means Qwen preserved required facts and reduced estimated cloud input tokens by at least 40%.",
            "- `verify_raw` means Qwen helped, but GPT-5.5/Codex should inspect the original artifact before final judgment.",
            "- `skip_local` means the raw input is small enough that local preprocessing is not worth adding to the workflow.",
            "- Risk flags catch contradictions, generic false positives, or verbose outputs that violate the helper contract.",
            "- Token counts are stable estimates for relative comparison, not billable provider counts.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize(case_results: list[CaseResult]) -> dict:
    raw_total = sum(result.raw_cloud_tokens_est for result in case_results)
    assisted_total = sum(result.assisted_cloud_tokens_est for result in case_results)
    return {
        "case_count": len(case_results),
        "use_local_count": sum(result.recommendation == "use_local" for result in case_results),
        "optional_local_count": sum(result.recommendation == "optional_local" for result in case_results),
        "verify_raw_count": sum(result.recommendation == "verify_raw" for result in case_results),
        "skip_local_count": sum(result.recommendation == "skip_local" for result in case_results),
        "raw_cloud_count": sum(result.recommendation == "raw_cloud" for result in case_results),
        "think_leak_count": sum(result.think_leak for result in case_results),
        "forbidden_fact_hit_count": sum(result.forbidden_facts_hit for result in case_results),
        "decision_match_count": sum(result.decision_matches_expected for result in case_results),
        "average_accuracy_score": sum(result.accuracy_score for result in case_results) / len(case_results),
        "aggregate_raw_cloud_tokens_est": raw_total,
        "aggregate_assisted_cloud_tokens_est": assisted_total,
        "aggregate_token_reduction": raw_total - assisted_total,
        "aggregate_token_reduction_pct": (raw_total - assisted_total) / raw_total if raw_total else 0.0,
    }


async def run_eval(args: argparse.Namespace) -> dict:
    if not args.no_status:
        status_before = await server.local_ollama_status()
    else:
        status_before = "skipped"

    if not args.no_warm:
        warm_result = await server.local_warm_model()
    else:
        warm_result = "skipped"

    case_results = []
    for case in build_cases():
        case_results.append(await evaluate_case(case))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": server.CODE_MODEL,
        "num_ctx": server.DEFAULT_NUM_CTX,
        "method": "local output scored against expert baseline facts; token counts are regex estimates",
        "status_before": status_before,
        "warm_result": warm_result,
        "summary": summarize(case_results),
        "cases": [asdict(result) for result in case_results],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate qwen2.5-coder local MCP usefulness for reducing cloud context."
    )
    parser.add_argument("--output-dir", default=".", help="Directory for JSON and Markdown reports.")
    parser.add_argument("--json-name", default="local_mcp_eval_results.json")
    parser.add_argument("--markdown-name", default="local_mcp_eval_report.md")
    parser.add_argument("--no-warm", action="store_true", help="Skip local_warm_model before cases.")
    parser.add_argument("--no-status", action="store_true", help="Skip local_ollama_status before cases.")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = await run_eval(args)
    json_path = output_dir / args.json_name
    markdown_path = output_dir / args.markdown_name
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(results), encoding="utf-8")

    summary = results["summary"]
    print(
        textwrap.dedent(
            f"""
            Wrote {json_path}
            Wrote {markdown_path}

            Cases: {summary['case_count']}
            use_local: {summary['use_local_count']}
            optional_local: {summary['optional_local_count']}
            verify_raw: {summary['verify_raw_count']}
            skip_local: {summary['skip_local_count']}
            raw_cloud: {summary['raw_cloud_count']}
            aggregate token reduction: {summary['aggregate_token_reduction_pct']:.1%}
            average accuracy: {summary['average_accuracy_score']:.1%}
            think leakage: {'yes' if summary['think_leak_count'] else 'no'}
            """
        ).strip()
    )


if __name__ == "__main__":
    asyncio.run(main())
