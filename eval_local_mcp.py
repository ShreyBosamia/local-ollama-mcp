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
from collections import Counter
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


def latest_outcomes_by_task(records: list[dict]) -> dict[str, dict]:
    outcomes: dict[str, dict] = {}
    for record in records:
        if record.get("record_type") == "outcome" and record.get("task_id"):
            outcomes[str(record["task_id"])] = record
    return outcomes


def is_safe_to_rely(tool_record: dict, outcome: dict | None) -> str:
    if outcome is None:
        return "unlabeled"
    risk_flags = set(tool_record.get("risk_flags") or [])
    if outcome.get("outcome") == "useful" and not {"tool_error", "think_leak"} & risk_flags:
        return "yes"
    return "no"


def evaluate_ledger(path: Path) -> dict:
    records = server.read_ledger_records(path)
    tool_records = [record for record in records if record.get("record_type") == "tool_call"]
    outcomes = latest_outcomes_by_task(records)
    cases = []

    for record in tool_records:
        task_id = str(record.get("task_id", ""))
        outcome = outcomes.get(task_id)
        token_estimates = record.get("token_estimates") or {}
        input_tokens = int(token_estimates.get("input") or 0)
        local_tokens = int(token_estimates.get("local_output") or 0)
        context_reduction = input_tokens - local_tokens
        cases.append(
            {
                "task_id": task_id,
                "timestamp": record.get("timestamp"),
                "tool": record.get("tool_name"),
                "model": record.get("model"),
                "recommendation": record.get("recommendation"),
                "outcome": outcome.get("outcome") if outcome else "unlabeled",
                "safe_to_rely": is_safe_to_rely(record, outcome),
                "risk_flags": record.get("risk_flags") or [],
                "latency_ms": record.get("latency_ms") or 0,
                "raw_context_tokens_est": input_tokens,
                "local_output_tokens_est": local_tokens,
                "estimated_context_reduction": context_reduction,
                "estimated_context_reduction_pct": round(context_reduction / input_tokens, 3)
                if input_tokens
                else 0.0,
                "accepted_solution_tokens_est": estimate_tokens(str(outcome.get("accepted_solution", "")))
                if outcome
                else 0,
            }
        )

    input_total = sum(case["raw_context_tokens_est"] for case in cases)
    local_total = sum(case["local_output_tokens_est"] for case in cases)
    labeled = [case for case in cases if case["outcome"] != "unlabeled"]
    safe = [case for case in cases if case["safe_to_rely"] == "yes"]
    latency_values = [int(case["latency_ms"]) for case in cases if case["latency_ms"]]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": server.CODE_MODEL,
        "num_ctx": server.DEFAULT_NUM_CTX,
        "method": "captured ledger analysis; token counts are regex estimates",
        "ledger_path": str(path),
        "summary": {
            "record_count": len(records),
            "tool_record_count": len(tool_records),
            "outcome_record_count": sum(record.get("record_type") == "outcome" for record in records),
            "labeled_tool_record_count": len(labeled),
            "safe_to_rely_count": len(safe),
            "safe_to_rely_labeled_pct": len(safe) / len(labeled) if labeled else 0.0,
            "aggregate_raw_context_tokens_est": input_total,
            "aggregate_local_output_tokens_est": local_total,
            "aggregate_context_reduction": input_total - local_total,
            "aggregate_context_reduction_pct": (input_total - local_total) / input_total
            if input_total
            else 0.0,
            "average_latency_ms": round(sum(latency_values) / len(latency_values))
            if latency_values
            else 0,
            "recommendation_counts": dict(Counter(case["recommendation"] for case in cases)),
            "outcome_counts": dict(Counter(case["outcome"] for case in cases)),
            "risk_flag_counts": dict(
                Counter(flag for case in cases for flag in case.get("risk_flags", []))
            ),
        },
        "cases": cases,
    }


def render_ledger_markdown(results: dict) -> str:
    summary = results["summary"]
    lines = [
        "# Local Ollama MCP Ledger Evaluation",
        "",
        f"- Generated: `{results['generated_at']}`",
        f"- Ledger: `{results['ledger_path']}`",
        f"- Tool records: `{summary['tool_record_count']}`",
        f"- Labeled tool records: `{summary['labeled_tool_record_count']}`",
        f"- Safe-to-rely labeled rate: `{summary['safe_to_rely_labeled_pct']:.1%}`",
        f"- Raw context tokens avoided est: `{summary['aggregate_context_reduction']}`",
        f"- Aggregate context reduction: `{summary['aggregate_context_reduction_pct']:.1%}`",
        f"- Local output tokens est: `{summary['aggregate_local_output_tokens_est']}`",
        f"- Average latency: `{summary['average_latency_ms']} ms`",
        "",
        "## Recommendation Counts",
        "",
        "```json",
        json.dumps(summary["recommendation_counts"], indent=2, sort_keys=True),
        "```",
        "",
        "## Outcome Counts",
        "",
        "```json",
        json.dumps(summary["outcome_counts"], indent=2, sort_keys=True),
        "```",
        "",
        "## Captured Cases",
        "",
        "| Task ID | Tool | Recommendation | Outcome | Safe To Rely | Reduction | Latency | Risk Flags |",
        "| --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for case in results["cases"]:
        risks = ", ".join(case["risk_flags"]) or "-"
        task_id = str(case["task_id"])[:12]
        lines.append(
            "| {task_id} | `{tool}` | `{recommendation}` | `{outcome}` | `{safe}` | {reduction:.0%} | {latency} ms | {risks} |".format(
                task_id=task_id,
                tool=case["tool"],
                recommendation=case["recommendation"],
                outcome=case["outcome"],
                safe=case["safe_to_rely"],
                reduction=case["estimated_context_reduction_pct"],
                latency=case["latency_ms"],
                risks=risks,
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `safe_to_rely` requires an accepted `useful` outcome and no hard risk flags.",
            "- Unlabeled records are useful for context-reduction and latency metrics but are not training-ready.",
            "- Token counts are stable estimates for comparing workflow context size, not provider billing counts.",
            "",
        ]
    )
    return "\n".join(lines)


async def run_eval(args: argparse.Namespace) -> dict:
    if args.from_ledger:
        return evaluate_ledger(Path(args.ledger_path) if args.ledger_path else server.ledger_path())

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
    parser.add_argument("--from-ledger", action="store_true", help="Analyze captured ledger records instead of running synthetic cases.")
    parser.add_argument("--ledger-path", default="", help="Ledger path for --from-ledger; defaults to LOCAL_MCP_LEDGER_PATH or .local_ollama_mcp/ledger.jsonl.")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = await run_eval(args)
    json_path = output_dir / args.json_name
    markdown_path = output_dir / args.markdown_name
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    markdown = render_ledger_markdown(results) if args.from_ledger else render_markdown(results)
    markdown_path.write_text(markdown, encoding="utf-8")

    summary = results["summary"]
    if args.from_ledger:
        print(
            textwrap.dedent(
                f"""
                Wrote {json_path}
                Wrote {markdown_path}

                Tool records: {summary['tool_record_count']}
                labeled tool records: {summary['labeled_tool_record_count']}
                safe to rely: {summary['safe_to_rely_count']}
                aggregate context reduction: {summary['aggregate_context_reduction_pct']:.1%}
                average latency: {summary['average_latency_ms']} ms
                """
            ).strip()
        )
        return

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
