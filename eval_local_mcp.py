#!/usr/bin/env python3
"""
Evaluate the configured local MCP helper model for Codex.

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
import subprocess
import textwrap
import time

import server


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
THINK_RE = re.compile(r"</?think\b", re.IGNORECASE)
LOCAL_ROUTING_MIN_TOKENS = 120
LOCAL_ROUTING_MAX_TOKENS = 4000
DEFAULT_RUN_INDEX = ".local_ollama_mcp/eval_runs/index.jsonl"
DEFAULT_CASE_DIR = ".local_ollama_mcp/eval_cases"
DEFAULT_ARTIFACT_DIR = ".local_ollama_mcp/eval_artifacts"
DEFAULT_DASHBOARD_PATH = "EVAL_DASHBOARD.md"
SUITES = ("synthetic", "reasoning", "artifacts", "pipeline", "standard", "all")


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
    source: str = "built_in"
    focus: str = ""
    test_framework: str = "unknown"
    expected_recommendation: str = ""
    forbidden_facts: tuple[ExpectedFact, ...] = ()
    max_output_tokens: int = 0
    max_output_chars: int = 0
    max_bullets: int = 0
    require_json: bool = False


@dataclass
class CaseResult:
    name: str
    category: str
    tool: str
    model: str
    source: str
    recommendation: str
    expected_recommendation: str
    decision_matches_expected: bool
    accuracy_score: float
    structure_score: float
    compression_score: float
    usefulness_score: float
    required_facts_hit: int
    required_facts_total: int
    optional_facts_hit: int
    optional_facts_total: int
    forbidden_facts_hit: int
    forbidden_fact_labels: list[str]
    missing_required_facts: list[str]
    missing_optional_facts: list[str]
    structure_violations: list[str]
    think_leak: bool
    raw_cloud_tokens_est: int
    assisted_cloud_tokens_est: int
    local_output_tokens_est: int
    estimated_cloud_token_reduction: int
    estimated_cloud_token_reduction_pct: float
    routing_decision: str
    confidence_score: float
    artifact_tokens_est: int
    cloud_tokens_avoided_est: int
    compression_ratio: float
    latency_ms: int
    tokens_per_second: float
    local_judge_score: float | None
    local_judge_notes: str
    output_preview: str
    local_output: str



@dataclass(frozen=True)
class RoutingResult:
    routing_decision: str
    confidence_score: float
    artifact_tokens_est: int
    local_output_tokens_est: int
    cloud_tokens_avoided_est: int
    risk_flags: tuple[str, ...]
    reduction_pct: float


def estimate_tokens(text: str) -> int:
    """Cheap, stable token estimate for relative cloud-token comparisons."""
    return len(TOKEN_RE.findall(text))


def one_line(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def fact_hit(output: str, fact: ExpectedFact) -> bool:
    return re.search(fact.pattern, output, re.IGNORECASE | re.DOTALL) is not None


def tool_model(tool: str) -> str:
    if tool == "local_reason_check":
        return server.REASON_MODEL
    if tool == "local_plan_check":
        return server.PLAN_MODEL
    return server.CODE_MODEL


def bullet_count(text: str) -> int:
    count = 0
    for line in text.splitlines():
        if re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", line):
            count += 1
    return count


def structure_result(output: str, case: EvalCase) -> tuple[float, list[str], bool]:
    violations: list[str] = []
    think_leak = THINK_RE.search(output) is not None
    if think_leak:
        violations.append("think_leak")
    if not output.strip():
        violations.append("empty_output")
    output_tokens = estimate_tokens(output)
    if case.max_output_tokens and output_tokens > case.max_output_tokens:
        violations.append(f"max_output_tokens:{output_tokens}>{case.max_output_tokens}")
    if case.max_output_chars and len(output) > case.max_output_chars:
        violations.append(f"max_output_chars:{len(output)}>{case.max_output_chars}")
    if case.max_bullets:
        count = bullet_count(output)
        if count > case.max_bullets:
            violations.append(f"max_bullets:{count}>{case.max_bullets}")
    if case.require_json:
        try:
            json.loads(output)
        except json.JSONDecodeError:
            violations.append("invalid_json")

    if think_leak or "empty_output" in violations:
        return 0.0, violations, think_leak
    score = 1.0 - (0.18 * len(violations))
    return round(clamp(score), 3), violations, think_leak


def compression_score_for(raw_tokens: int, assisted_tokens: int) -> float:
    if raw_tokens <= 0:
        return 0.0
    reduction_pct = (raw_tokens - assisted_tokens) / raw_tokens
    return round(clamp(reduction_pct / 0.40), 3)


def usefulness_score_for(
    *,
    accuracy_score: float,
    structure_score: float,
    compression_score: float,
    forbidden_facts_hit: int,
    think_leak: bool,
) -> float:
    score = (accuracy_score * 0.55) + (structure_score * 0.25) + (compression_score * 0.20)
    if forbidden_facts_hit:
        score *= 0.35
    if think_leak:
        score = 0.0
    return round(clamp(score), 3)


def route_local_artifact(
    *,
    artifact: str,
    local_output: str,
    required_facts_hit: int,
    required_facts_total: int,
    forbidden_facts_hit: int = 0,
    think_leak: bool = False,
    tool_error: bool = False,
) -> RoutingResult:
    artifact_tokens = estimate_tokens(artifact)
    local_tokens = estimate_tokens(local_output)
    reduction = artifact_tokens - local_tokens
    reduction_pct = reduction / artifact_tokens if artifact_tokens else 0.0
    required_ratio = required_facts_hit / required_facts_total if required_facts_total else 1.0
    risk_flags: list[str] = []
    if tool_error:
        risk_flags.append("tool_error")
    if think_leak:
        risk_flags.append("think_leak")
    if forbidden_facts_hit:
        risk_flags.append("contradiction")
    if required_facts_total and required_facts_hit < required_facts_total:
        risk_flags.append("missing_required_facts")

    hard_risk = bool({"tool_error", "think_leak", "contradiction"} & set(risk_flags))
    confidence = max(0.0, min(1.0, (required_ratio * 0.75) + (max(0.0, reduction_pct) * 0.25)))
    if hard_risk:
        confidence *= 0.25
    elif "missing_required_facts" in risk_flags:
        confidence *= 0.65

    if artifact_tokens < LOCAL_ROUTING_MIN_TOKENS:
        decision = "skip_local"
    elif artifact_tokens > LOCAL_ROUTING_MAX_TOKENS or hard_risk:
        decision = "raw_cloud"
    elif reduction_pct >= 0.40 and required_ratio >= 1.0:
        decision = "use_local"
    elif reduction_pct > 0 and local_output.strip():
        decision = "verify_raw"
    else:
        decision = "raw_cloud"

    return RoutingResult(
        routing_decision=decision,
        confidence_score=round(confidence, 3),
        artifact_tokens_est=artifact_tokens,
        local_output_tokens_est=local_tokens,
        cloud_tokens_avoided_est=max(0, reduction) if decision == "use_local" else 0,
        risk_flags=tuple(sorted(set(risk_flags))),
        reduction_pct=round(reduction_pct, 3),
    )


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
    artifact_tokens = estimate_tokens(case.artifact)
    if artifact_tokens < LOCAL_ROUTING_MIN_TOKENS:
        return "skip_local"
    if artifact_tokens > LOCAL_ROUTING_MAX_TOKENS or think_leak or forbidden_facts_hit:
        return "raw_cloud"
    if required_facts_total and required_facts_hit < required_facts_total:
        return "verify_raw" if accuracy_score >= 0.60 and token_reduction_pct > 0 else "raw_cloud"
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
            expected_recommendation="use_local",
            max_bullets=6,
            max_output_tokens=220,
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
            expected_recommendation="use_local",
            max_bullets=5,
            max_output_tokens=280,
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
            expected_recommendation="use_local",
            max_bullets=7,
            max_output_tokens=260,
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
            expected_recommendation="use_local",
            max_bullets=6,
            max_output_tokens=240,
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
            max_bullets=6,
            max_output_tokens=120,
            expected_facts=(
                ExpectedFact("addition behavior", r"add|sum|\+"),
            ),
        ),
    ]


def build_reasoning_cases() -> list[EvalCase]:
    noisy_problem = "\n".join(
        [
            "Provider import failed after a UI review approval.",
            "INFO retrying provider import",
            "ERROR service.py:218 Place is required.",
            "WARN ui/DialogContent.tsx:44 DialogContent requires a DialogTitle for accessibility.",
            "ERROR pipeline/findhelpDirectory.ts:91 directory_expansion capped at 50 provider candidates",
            *[f"DEBUG duplicate retry line {index}" for index in range(90)],
        ]
    )
    return [
        EvalCase(
            name="deepseek_reasoning_think_strip",
            category="reasoning",
            tool="local_reason_check",
            task="Ask DeepSeek-R1 for concise next checks without exposing hidden reasoning.",
            artifact=noisy_problem,
            expected_recommendation="use_local",
            max_bullets=5,
            max_output_tokens=220,
            expected_facts=(
                ExpectedFact("Place is required remains visible", r"Place is required|place"),
                ExpectedFact("DialogTitle warning remains visible", r"DialogTitle|accessib"),
                ExpectedFact("directory expansion cap remains visible", r"directory_expansion|cap|50"),
            ),
            forbidden_facts=(
                ExpectedFact("think tag leaked", r"</?think\b"),
            ),
        )
    ]


def build_pipeline_cases() -> list[EvalCase]:
    # 1. React Component Tree Refactoring & Prop-Drill Auditing case
    react_noise = "\n".join(
        f"    // Section block {i} representing nested dashboard layouts and static markup bloat\n"
        f"    const renderContentBlock_{i} = () => <div className=\"p-4 border\">Static Block {i}</div>;"
        for i in range(1, 100)
    )
    react_artifact = f"""import React, {{ useState, useEffect, useContext, createContext }} from 'react';
import {{ AuthContext }} from '../context/AuthContext';

interface DashboardProps {{
    userId: string;
    theme: 'dark' | 'light';
    onLogout: () => void;
}}

interface Metric {{
    id: string;
    label: string;
    value: number;
    delta: number;
}}

export const DashboardView: React.FC<DashboardProps> = ({{ userId, theme, onLogout }}) => {{
    const auth = useContext(AuthContext);
    const [metricsData, setMetricsData] = useState<Metric[]>([]);
    const [activeTab, setActiveTab] = useState<string>('overview');

{react_noise}

    return (
        <div>
            <HeaderSection userId={{userId}} />
            <ControlPanel activeTab={{activeTab}} setActiveTab={{setActiveTab}} />
            <MetricsGrid metricsData={{metricsData}} />
        </div>
    );
}};

const HeaderSection: React.FC<{{ userId: string }}> = ({{ userId }}) => {{
    return <header>User ID: {{userId}}</header>;
}};

const ControlPanel: React.FC<{{ activeTab: string; setActiveTab: (tab: string) => void }}> = ({{ activeTab, setActiveTab }}) => {{
    return <button onClick={{() => setActiveTab('overview')}}>Tab</button>;
}};

const MetricsGrid: React.FC<{{ metricsData: Metric[] }}> = ({{ metricsData }}) => {{
    return <div>{{metricsData.map(m => <MetricCard key={{m.id}} metric={{m}} />)}}</div>;
}};

const MetricCard: React.FC<{{ metric: Metric }}> = ({{ metric }}) => {{
    return <div>{{metric.label}}: ${{metric.value}} <SparklineChart metricId={{metric.id}} delta={{metric.delta}} /></div>;
}};

const SparklineChart: React.FC<{{ metricId: string; delta: number }}> = ({{ metricId, delta }}) => {{
    return <div>Delta: {{delta}}%</div>;
}};
"""

    # 2. Vite/Webpack Bundle Audit & Asset Size Bloat Detection case
    vite_noise = "\n".join(
        f"dist/assets/chunk-detail-block-{i:03d}-F87d9a8c.js  {0.4 * i:.1f} kB │ gzip: {0.08 * i:.2f} kB"
        for i in range(1, 100)
    )
    vite_artifact = f"""vite v5.2.11 building for production...
transforming (4512) index.html
✓ 4810 modules transformed.
rendering chunks...

{vite_noise}

dist/assets/index-D3g2.js                892.4 kB │ gzip: 184.2 kB
✓ built asset dist/assets/index-D3g2.js (main app bundle)
dist/assets/vendor-legacy-F89a.js       1204.8 kB │ gzip: 391.2 kB
✓ built asset dist/assets/vendor-legacy-F89a.js (compatibility layer)

[warn] 'moment' is imported by 'dist/assets/index-D3g2.js', but is not in vendor config. This creates duplicate packages.
[warn] Dynamic import of './LazyChart' could not be resolved statically; inlined instead. This spikes chunk sizes.
[warn] Bundle size exceeds recommended limit of 500 kB. Please split large libraries.
"""

    # 3. PostgreSQL Lock & Deadlock case
    pg_noise = "\n".join(
        f"2026-05-27 19:12:{i%60:02d}.{i*4:03d} UTC [1421] [0x7f83ad29c9] pid={i*10} connection: client connected from 127.0.0.1"
        for i in range(1, 150)
    )
    pg_artifact = f"""{pg_noise}
2026-05-27 19:15:32.481 UTC [1421] [0x7f83ad29c9] pid=1421 ERROR: deadlock detected
2026-05-27 19:15:32.481 UTC [1421] [0x7f83ad29c9] pid=1421 DETAIL: Process 1421 waits for ShareLock on transaction 8219318; blocked by process 1429.
    Process 1429 waits for ExclusiveLock on relation 49210 of database 16384; blocked by process 1421.
    Process 1421: UPDATE orders SET status = 'completed', updated_at = NOW() WHERE id = 'order_892183';
    Process 1429: UPDATE inventory SET quantity = quantity - 1 WHERE sku = 'PROD_SKU_8921';
2026-05-27 19:15:32.482 UTC [1421] [0x7f83ad29c9] pid=1421 STATEMENT: UPDATE orders SET status = 'completed', updated_at = NOW() WHERE id = 'order_892183';

2026-05-27 19:16:01.121 UTC [1433] [0x7f83b248e8] pid=1433 LOG: duration: 4821.192 ms  statement: SELECT * FROM transactions WHERE status = 'pending' AND updated_at < NOW() - INTERVAL '1 day';
2026-05-27 19:16:01.123 UTC [1433] [0x7f83b248e8] pid=1433 LOG: filter scan node details: Sequential Scan on transactions  (cost=0.00..128912.44 rows=5402123 width=254)
"""

    # 4. Telemetry Monitor case
    telemetry_noise = "\n".join(
        f"2026-05-27T19:20:{i%60:02d}Z - telemetry - [GPU] temperature.gpu=48, power.draw=32.4W, clocks.gr=1350MHz"
        for i in range(1, 100)
    )
    telemetry_artifact = f"""{telemetry_noise}
2026-05-27T19:25:01Z - telemetry - [WARNING] VRAM utilization has exceeded critical threshold!
2026-05-27T19:25:01Z - telemetry - [VRAM] memory.total=8192MB, memory.used=7910MB, memory.free=282MB (98.8% allocation)
2026-05-27T19:25:02Z - ollama - [INFO] loading model qwen2.5-coder:7b-instruct-q5_K_M
2026-05-27T19:25:05Z - ollama - [WARNING] context limit expanded to 6144. Prompt requires layers offload.
2026-05-27T19:25:06Z - ollama - [INFO] offloaded 28 / 32 transformer layers to GPU. 4 layers offloaded to system memory (CPU Fallback).
2026-05-27T19:25:12Z - performance - [METRIC] processing velocity decreased to 8.4 tokens/second (down from 42.0 tok/sec)
2026-05-27T19:25:20Z - telemetry - [GPU] temperature.gpu=78, power.draw=148.2W, clocks.gr=1860MHz (high thermal profile detected)
"""

    # 5. External API & Framework documentation case
    doc_artifact = """
# Framework Actions API (Beta)

Welcome to the comprehensive installation and API reference guide for modern server-side operations.

![Architecture Diagram](https://raw.githubusercontent.com/framework/actions/main/assets/architecture.svg)

## Quick Start Setup Steps
1. Prepare node environment and configure package manager.
- Standard setup list item 1
- Standard setup list item 2
- Standard setup list item 3
- Standard setup list item 4
- Standard setup list item 5
- Standard setup list item 6
- Standard setup list item 7
- Standard setup list item 8
- Standard setup list item 9
- Standard setup list item 10
- Standard setup list item 11

## API Reference Specifications
```typescript
import { createContext } from 'react';

export interface ActionConfig<T> {
  id: string;
  resolver: (payload: T) => Promise<ActionResponse>;
  optimisticUpdate?: (draft: Draft<State>) => void;
}
export type ActionResponse = { success: boolean; data?: any; error?: string };
```

## Hook Method Signatures
```typescript
export function useActionState<T>(action: ActionConfig<T>): [State, (payload: T) => void, boolean] {
  // Complex custom execution state management
  return [{} as any, () => {}, false];
}
```
"""

    return [
        EvalCase(
            name="react_prop_drill_pipeline",
            category="pipeline",
            tool="local_summarize",
            task="map out component parent-child hierarchy, prop drill paths, and state definitions",
            artifact=react_artifact,
            focus="map component relations and state",
            expected_recommendation="use_local",
            expected_facts=(
                ExpectedFact("DashboardView component details", r"DashboardView"),
                ExpectedFact("prop drilling components identified", r"MetricsGrid|MetricCard|SparklineChart|prop[ -]drill"),
                ExpectedFact("useContext context identified", r"useContext|AuthContext"),
            ),
        ),
        EvalCase(
            name="vite_bundle_audit_pipeline",
            category="pipeline",
            tool="local_summarize",
            task="summarize Vite bundle bottlenecks, identifying chunks exceeding 500kB and module dependency spikes",
            artifact=vite_artifact,
            focus="Vite warnings and large sizes",
            expected_recommendation="use_local",
            expected_facts=(
                ExpectedFact("index-D3g2.js bundle listed", r"index-D3g2\.js|892\.4"),
                ExpectedFact("vendor-legacy-F89a.js bundle listed", r"vendor-legacy-F89a\.js|1204\.8"),
                ExpectedFact("moment library duplicate warning", r"moment|duplicate"),
            ),
        ),
        EvalCase(
            name="postgres_lock_trace_pipeline",
            category="pipeline",
            tool="local_summarize",
            task="identify lock contention patterns, deadlocked tables, and queries triggering sequential scans",
            artifact=pg_artifact,
            focus="deadlock details and lock statements",
            expected_recommendation="use_local",
            expected_facts=(
                ExpectedFact("deadlock between processes", r"deadlock|process 1421|process 1429"),
                ExpectedFact("UPDATE orders SQL blocked statement", r"UPDATE orders"),
                ExpectedFact("Sequential scan on transactions identified", r"Sequential Scan|Seq Scan|transactions"),
            ),
        ),
        EvalCase(
            name="telemetry_vram_safety_pipeline",
            category="pipeline",
            tool="local_summarize",
            task="extract telemetry anomalies, VRAM utilization spikes, and GPU model context limits",
            artifact=telemetry_artifact,
            focus="VRAM and context limits",
            expected_recommendation="use_local",
            expected_facts=(
                ExpectedFact("VRAM utilization exceeded threshold", r"VRAM|98\.8%|7910"),
                ExpectedFact("CPU Fallback layers loaded to system memory", r"layers offload|CPU Fallback|system memory"),
                ExpectedFact("GPU core temperature spiked", r"temperature|78"),
            ),
        ),
        EvalCase(
            name="framework_api_docs_pipeline",
            category="pipeline",
            tool="local_summarize",
            task="extract public API interfaces, parameter definitions, and usage syntax",
            artifact=doc_artifact,
            focus="API configurations and method signatures",
            expected_recommendation="use_local",
            expected_facts=(
                ExpectedFact("ActionConfig interface definition", r"ActionConfig"),
                ExpectedFact("useActionState method signature", r"useActionState"),
            ),
        ),
    ]


def expected_fact_from_dict(raw: dict) -> ExpectedFact:
    return ExpectedFact(
        label=str(raw["label"]),
        pattern=str(raw["pattern"]),
        required=bool(raw.get("required", True)),
    )


def resolve_artifact_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return Path.cwd() / path


def eval_case_from_dict(raw: dict, base_dir: Path, source: str) -> EvalCase:
    artifact = str(raw.get("artifact", ""))
    if raw.get("artifact_path"):
        artifact_path = resolve_artifact_path(str(raw["artifact_path"]), base_dir)
        artifact = artifact_path.read_text(encoding="utf-8", errors="replace")
    expected_facts = tuple(expected_fact_from_dict(item) for item in raw.get("expected_facts", []))
    forbidden_facts = tuple(expected_fact_from_dict(item) for item in raw.get("forbidden_facts", []))
    return EvalCase(
        name=str(raw["name"]),
        category=str(raw.get("category", "external")),
        tool=str(raw.get("tool", "local_summarize")),
        task=str(raw.get("task", "")),
        artifact=artifact,
        expected_facts=expected_facts,
        source=source,
        focus=str(raw.get("focus", "")),
        test_framework=str(raw.get("test_framework", "unknown")),
        expected_recommendation=str(raw.get("expected_recommendation", "")),
        forbidden_facts=forbidden_facts,
        max_output_tokens=int(raw.get("max_output_tokens") or 0),
        max_output_chars=int(raw.get("max_output_chars") or 0),
        max_bullets=int(raw.get("max_bullets") or 0),
        require_json=bool(raw.get("require_json", False)),
    )


def load_case_file(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    if not path.exists():
        return cases
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL case: {exc.msg}") from exc
            cases.append(eval_case_from_dict(raw, path.parent, str(path)))
    return cases


def load_external_cases(args: argparse.Namespace) -> list[EvalCase]:
    paths: list[Path] = []
    for case_file in args.case_file:
        paths.append(Path(case_file))
    case_dir = Path(args.case_dir)
    if case_dir.exists():
        paths.extend(sorted(case_dir.glob("*.jsonl")))

    cases: list[EvalCase] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        cases.extend(load_case_file(path))
    return cases


def discover_artifact_cases(artifact_dir: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    diff_dir = artifact_dir / "diffs"
    if diff_dir.exists():
        for path in sorted(diff_dir.glob("*.diff")):
            cases.append(
                EvalCase(
                    name=f"artifact_diff_{path.stem}",
                    category="artifact",
                    tool="local_code_review",
                    task="Review this saved git diff for likely regressions.",
                    artifact=path.read_text(encoding="utf-8", errors="replace"),
                    expected_facts=(),
                    source=str(path),
                    focus="bugs, regressions, missing tests",
                    max_bullets=5,
                    max_output_tokens=280,
                )
            )
    log_dir = artifact_dir / "logs"
    if log_dir.exists():
        for pattern in ("*.txt", "*.log"):
            for path in sorted(log_dir.glob(pattern)):
                cases.append(
                    EvalCase(
                        name=f"artifact_log_{path.stem}",
                        category="artifact",
                        tool="local_summarize",
                        task="Summarize this saved terminal log while preserving actionable errors.",
                        artifact=path.read_text(encoding="utf-8", errors="replace"),
                        expected_facts=(),
                        source=str(path),
                        focus="exact errors, affected files, and repeated noise to ignore",
                        max_bullets=6,
                        max_output_tokens=240,
                    )
                )
    return cases


def build_suite_cases(args: argparse.Namespace) -> list[EvalCase]:
    cases: list[EvalCase] = []
    if args.suite in {"synthetic", "all"}:
        cases.extend(build_cases())
    if args.suite in {"reasoning", "all"}:
        cases.extend(build_reasoning_cases())
    if args.suite in {"pipeline", "all"}:
        cases.extend(build_pipeline_cases())
    if args.suite in {"standard", "all"}:
        standard_path = Path(__file__).parent / ".local_ollama_mcp" / "eval_cases" / "standard_benchmarks.jsonl"
        if standard_path.exists():
            cases.extend(load_case_file(standard_path))
    if args.suite in {"artifacts", "all"}:
        cases.extend(load_external_cases(args))
        cases.extend(discover_artifact_cases(Path(args.artifact_dir)))
    return cases
    

async def call_tool(case: EvalCase) -> str:
    artifact = case.artifact
    if case.category == "pipeline":
        if case.name == "react_prop_drill_pipeline":
            artifact = await server.extract_regex_lines(
                artifact,
                pattern=r"(interface|type)\s+\w+Props|const\s+\w+:\s*React\.FC|use(State|Context|Reducer|Memo|Effect)\(",
                case_insensitive=True,
                context_lines=2
            )
        elif case.name == "vite_bundle_audit_pipeline":
            artifact = await server.extract_regex_lines(
                artifact,
                pattern=r"(?i)warning|chunk|split|\b\d+(\.\d+)?\s*(kB|mB|B)\b|dist/assets/",
                case_insensitive=True,
                context_lines=1
            )
            artifact = await server.trim_markdown_payload(
                artifact,
                max_code_block_lines=8,
                max_list_items=5,
                remove_images=True
            )
        elif case.name == "postgres_lock_trace_pipeline":
            artifact = await server.clean_server_logs(
                artifact,
                remove_timestamps=True,
                remove_hex_hashes=True,
                deduplicate_consecutive=True
            )
            artifact = await server.extract_regex_lines(
                artifact,
                pattern=r"(?i)deadlock|exclusive\s+lock|lock\s+shared|duration:|seq\s+scan|exceeded\s+threshold",
                case_insensitive=True,
                context_lines=3
            )
        elif case.name == "telemetry_vram_safety_pipeline":
            artifact = await server.extract_regex_lines(
                artifact,
                pattern=r"(?i)vram|gpu\s+100%|offload|cpu\s+fallback|temperature|exhausted|context\s+limit|oom",
                case_insensitive=True,
                context_lines=1
            )
            artifact = await server.clean_server_logs(
                artifact,
                remove_timestamps=True,
                remove_hex_hashes=True,
                deduplicate_consecutive=True
            )
        elif case.name == "framework_api_docs_pipeline":
            artifact = await server.trim_markdown_payload(
                artifact,
                max_code_block_lines=8,
                max_list_items=4,
                remove_images=True
            )
            artifact = await server.extract_regex_lines(
                artifact,
                pattern=r"(export\s+(class|interface|type|const|function)|import\s+.*?from)",
                case_insensitive=False,
                context_lines=1
            )

    if case.tool == "local_summarize":
        return await server.local_summarize(artifact, focus=case.focus)
    if case.tool == "local_code_review":
        return await server.local_code_review(artifact, focus=case.focus)
    if case.tool == "local_test_ideas":
        return await server.local_test_ideas(artifact, test_framework=case.test_framework)
    if case.tool == "local_reason_check":
        return await server.local_reason_check(artifact)
    if case.tool == "local_map_project_structure":
        max_depth = int(case.focus) if case.focus.isdigit() else 3
        return await server.local_map_project_structure(max_depth=max_depth)
    if case.tool == "local_extract_signatures":
        return await server.local_extract_signatures(artifact)
    if case.tool == "local_lint_audit":
        return await server.local_lint_audit(artifact)
    raise ValueError(f"unknown tool: {case.tool}")



async def local_judge_case(case: EvalCase, output: str, judge_model: str) -> tuple[float | None, str]:
    rubric = {
        "task": case.task,
        "required_facts": [fact.label for fact in case.expected_facts if fact.required],
        "forbidden_facts": [fact.label for fact in case.forbidden_facts],
    }
    prompt = f"""
Score this local MCP output as a helper artifact for Codex.

Rubric:
{json.dumps(rubric, sort_keys=True)}

Factual Coverage: The output should cover the target required facts accurately.
Conciseness: No preamble, conversational fluff, or chatty introductions.
Formatting: Output must stay within bounds, utilize simple lists/bullets, or JSON as requested.

Output to evaluate:
{output}

Return a JSON object matching this schema:
{{
  "score": <float between 0.0 and 1.0 representing overall factual coverage and quality>,
  "notes": "<one short sentence explaining the score rationale>"
}}
"""
    try:
        raw = await server.ask_ollama(
            judge_model,
            prompt,
            temperature=0,
            num_predict=200,
            num_ctx=server.DEFAULT_NUM_CTX,
            keep_alive="0",
            system="You are an expert AI system judge. Return strict JSON only. Do not include markdown code block syntax. Do not include conversational text or chain-of-thought thinking.",
        )
        
        # Self-correcting JSON parser: strip markdown code blocks and reasoning tags
        clean_raw = raw.strip()
        if "</think>" in clean_raw:
            clean_raw = clean_raw.split("</think>")[-1].strip()
        if "```json" in clean_raw:
            clean_raw = clean_raw.split("```json")[-1].split("```")[0].strip()
        elif "```" in clean_raw:
            clean_raw = clean_raw.split("```")[-1].split("```")[0].strip()

        try:
            parsed = json.loads(clean_raw)
        except json.JSONDecodeError:
            # Absolute fallback using regex to parse score and notes from malformed string
            score_match = re.search(r'"score"\s*:\s*([0-9.]+)', clean_raw)
            notes_match = re.search(r'"notes"\s*:\s*"([^"]+)"', clean_raw)
            if score_match:
                parsed = {
                    "score": float(score_match.group(1)),
                    "notes": notes_match.group(1) if notes_match else "Fallback parsed"
                }
            else:
                raise ValueError("Regex parse failed")
                
    except Exception as exc:
        return None, f"local judge unavailable: {type(exc).__name__}: {exc}"

    score = parsed.get("score")
    notes = str(parsed.get("notes", "")).strip()
    if not isinstance(score, (int, float)):
        return None, "local judge returned non-numeric score"
    return round(clamp(float(score)), 3), one_line(notes, 160)


async def evaluate_case(
    case: EvalCase,
    *,
    use_local_judge: bool = False,
    judge_model: str = "",
) -> CaseResult:
    started = time.perf_counter()
    output = await call_tool(case)
    latency_ms = round((time.perf_counter() - started) * 1000)
    local_tokens = estimate_tokens(output)
    latency_sec = latency_ms / 1000.0
    tokens_per_second = round(local_tokens / latency_sec, 2) if latency_sec > 0.0 else 0.0

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
    token_reduction = raw_tokens - assisted_tokens
    token_reduction_pct = token_reduction / raw_tokens if raw_tokens else 0.0
    structure_score, structure_violations, think_leak = structure_result(output, case)
    compression_score = compression_score_for(raw_tokens, assisted_tokens)
    usefulness_score = usefulness_score_for(
        accuracy_score=accuracy_score,
        structure_score=structure_score,
        compression_score=compression_score,
        forbidden_facts_hit=len(forbidden_hits),
        think_leak=think_leak,
    )

    routing = route_local_artifact(
        artifact=case.artifact,
        local_output=output,
        required_facts_hit=required_hits,
        required_facts_total=len(required),
        forbidden_facts_hit=len(forbidden_hits),
        think_leak=think_leak,
    )
    recommendation = routing.routing_decision
    local_judge_score = None
    local_judge_notes = "skipped"
    if use_local_judge:
        local_judge_score, local_judge_notes = await local_judge_case(
            case,
            output,
            judge_model or server.CODE_MODEL,
        )
    expected = case.expected_recommendation
    decision_matches_expected = (
        True
        if not expected
        else recommendation == expected or (expected == "use_local" and recommendation == "verify_raw")
    )

    return CaseResult(
        name=case.name,
        category=case.category,
        tool=case.tool,
        model=tool_model(case.tool),
        source=case.source,
        recommendation=recommendation,
        expected_recommendation=expected,
        decision_matches_expected=decision_matches_expected,
        accuracy_score=round(accuracy_score, 3),
        structure_score=structure_score,
        compression_score=compression_score,
        usefulness_score=usefulness_score,
        required_facts_hit=required_hits,
        required_facts_total=len(required),
        optional_facts_hit=optional_hits,
        optional_facts_total=len(optional),
        forbidden_facts_hit=len(forbidden_hits),
        forbidden_fact_labels=forbidden_hits,
        missing_required_facts=missing_required,
        missing_optional_facts=missing_optional,
        structure_violations=structure_violations,
        think_leak=think_leak,
        raw_cloud_tokens_est=raw_tokens,
        assisted_cloud_tokens_est=assisted_tokens,
        local_output_tokens_est=local_tokens,
        estimated_cloud_token_reduction=token_reduction,
        estimated_cloud_token_reduction_pct=round(token_reduction_pct, 3),
        routing_decision=routing.routing_decision,
        confidence_score=routing.confidence_score,
        artifact_tokens_est=routing.artifact_tokens_est,
        cloud_tokens_avoided_est=routing.cloud_tokens_avoided_est,
        compression_ratio=round(assisted_tokens / raw_tokens, 3) if raw_tokens else 0.0,
        latency_ms=latency_ms,
        tokens_per_second=tokens_per_second,
        local_judge_score=local_judge_score,
        local_judge_notes=local_judge_notes,
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
        f"- Average usefulness score: `{summary['average_usefulness_score']:.1%}`",
        f"- Average structure score: `{summary['average_structure_score']:.1%}`",
        f"- Average latency: `{summary['average_latency_ms']} ms`",
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
        "| Case | Tool | Route | Useful | Accuracy | Structure | Token Reduction | Latency | Missing Required Facts | Violations |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]

    for case in results["cases"]:
        missing = ", ".join(case["missing_required_facts"]) or "-"
        violations = ", ".join(case["structure_violations"] + case["forbidden_fact_labels"]) or "-"
        lines.append(
            "| {name} | `{tool}` | `{recommendation}` | {useful:.0%} | {accuracy:.0%} | {structure:.0%} | {reduction:.0%} | {latency} ms | {missing} | {violations} |".format(
                name=case["name"],
                tool=case["tool"],
                recommendation=case["recommendation"],
                useful=case["usefulness_score"],
                accuracy=case["accuracy_score"],
                structure=case["structure_score"],
                reduction=case["estimated_cloud_token_reduction_pct"],
                latency=case["latency_ms"],
                missing=missing,
                violations=violations,
            )
        )

    lines.extend(["", "## Output Previews", ""])
    for case in results["cases"]:
        lines.extend(
            [
                f"### {case['name']}",
                "",
                f"- Recommendation: `{case['recommendation']}`",
                f"- Source: `{case['source']}`",
                f"- Model: `{case['model']}`",
                f"- Accuracy score: `{case['accuracy_score']}`",
                f"- Structure score: `{case['structure_score']}`",
                f"- Compression score: `{case['compression_score']}`",
                f"- Usefulness score: `{case['usefulness_score']}`",
                f"- Local judge score: `{case['local_judge_score'] if case['local_judge_score'] is not None else 'n/a'}`",
                f"- Local judge notes: `{case['local_judge_notes']}`",
                f"- Raw cloud tokens est: `{case['raw_cloud_tokens_est']}`",
                f"- Assisted cloud tokens est: `{case['assisted_cloud_tokens_est']}`",
                f"- Artifact tokens est: `{case['artifact_tokens_est']}`",
                f"- Local output tokens est: `{case['local_output_tokens_est']}`",
                f"- Cloud tokens avoided est: `{case['cloud_tokens_avoided_est']}`",
                f"- Confidence score: `{case['confidence_score']}`",
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
            "- `usefulness_score` is deterministic and combines fact coverage, structure, and compression. Local judge scores are optional metadata only.",
            "- Token counts are stable estimates for relative comparison, not billable provider counts.",
            "- Before large branch pushes or review summaries, route local diffs/logs through `local_code_review` or `local_summarize` and send only accepted local summaries to cloud.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize(case_results: list[CaseResult]) -> dict:
    raw_total = sum(result.raw_cloud_tokens_est for result in case_results)
    assisted_total = sum(result.assisted_cloud_tokens_est for result in case_results)
    count = len(case_results)
    return {
        "case_count": count,
        "use_local_count": sum(result.recommendation == "use_local" for result in case_results),
        "optional_local_count": sum(result.recommendation == "optional_local" for result in case_results),
        "verify_raw_count": sum(result.recommendation == "verify_raw" for result in case_results),
        "skip_local_count": sum(result.recommendation == "skip_local" for result in case_results),
        "raw_cloud_count": sum(result.recommendation == "raw_cloud" for result in case_results),
        "think_leak_count": sum(result.think_leak for result in case_results),
        "forbidden_fact_hit_count": sum(result.forbidden_facts_hit for result in case_results),
        "decision_match_count": sum(result.decision_matches_expected for result in case_results),
        "average_latency_ms": round(
            sum(result.latency_ms for result in case_results) / count
        )
        if count
        else 0,
        "average_tokens_per_second": round(
            sum(result.tokens_per_second for result in case_results) / count, 2
        )
        if count
        else 0.0,
        "average_accuracy_score": sum(result.accuracy_score for result in case_results) / count
        if count
        else 0.0,
        "average_structure_score": sum(result.structure_score for result in case_results) / count
        if count
        else 0.0,
        "average_compression_score": sum(result.compression_score for result in case_results) / count
        if count
        else 0.0,
        "average_usefulness_score": sum(result.usefulness_score for result in case_results) / count
        if count
        else 0.0,
        "aggregate_raw_cloud_tokens_est": raw_total,
        "aggregate_assisted_cloud_tokens_est": assisted_total,
        "aggregate_token_reduction": raw_total - assisted_total,
        "aggregate_token_reduction_pct": (raw_total - assisted_total) / raw_total if raw_total else 0.0,
    }


def read_run_index(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def clean_run_row(row: dict) -> dict:
    return {
        "timestamp": row.get("timestamp"),
        "suite": row.get("suite"),
        "case_count": row.get("case_count"),
        "use_local_count": row.get("use_local_count"),
        "verify_raw_count": row.get("verify_raw_count"),
        "skip_local_count": row.get("skip_local_count"),
        "raw_cloud_count": row.get("raw_cloud_count"),
        "average_accuracy_score": row.get("average_accuracy_score"),
        "average_structure_score": row.get("average_structure_score"),
        "average_usefulness_score": row.get("average_usefulness_score"),
        "average_latency_ms": row.get("average_latency_ms"),
        "think_leak_count": row.get("think_leak_count"),
        "aggregate_token_reduction_pct": row.get("aggregate_token_reduction_pct"),
    }


def number_value(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def sum_metric(rows: list[dict], key: str) -> int:
    return int(sum(number_value(row.get(key)) or 0 for row in rows))


def average_metric(rows: list[dict], key: str) -> float | None:
    values = [number_value(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def latest_row(rows: list[dict]) -> dict:
    sortable = [row for row in rows if row.get("timestamp")]
    return max(sortable, key=lambda row: str(row.get("timestamp"))) if sortable else (rows[-1] if rows else {})


def format_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def format_latency(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f} ms"


def format_dashboard_row(suite: str, group: list[dict]) -> str:
    latest = latest_row(group)
    report_path = str(latest.get("markdown_path") or "")
    report_cell = f"[report]({report_path})" if report_path else "-"
    return (
        "| {suite} | {latest} | {runs} | {cases} | {accuracy} | {structure} | "
        "{usefulness} | {latency} | {use_local} | {raw_cloud} | {think_leaks} | {report} |"
    ).format(
        suite=suite,
        latest=latest.get("timestamp", "n/a"),
        runs=len(group),
        cases=sum_metric(group, "case_count"),
        accuracy=format_pct(average_metric(group, "average_accuracy_score")),
        structure=format_pct(average_metric(group, "average_structure_score")),
        usefulness=format_pct(average_metric(group, "average_usefulness_score")),
        latency=format_latency(average_metric(group, "average_latency_ms")),
        use_local=sum_metric(group, "use_local_count"),
        raw_cloud=sum_metric(group, "raw_cloud_count"),
        think_leaks=sum_metric(group, "think_leak_count"),
        report=report_cell,
    )


def render_eval_dashboard(rows: list[dict]) -> str:
    rows = [row for row in rows if row.get("suite")]
    latest = latest_row(rows)
    lines = [
        "# Local Ollama MCP Evaluation Dashboard",
        "",
        f"- Runs tracked: `{len(rows)}`",
        f"- Latest run: `{latest.get('timestamp', 'n/a')}`",
        "",
        "| Suite | Latest Run | Runs | Cases | Avg Accuracy | Avg Structure | Avg Usefulness | Avg Latency | Use Local | Raw Cloud | Think Leaks | Latest Local Report |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for suite in sorted({str(row.get("suite")) for row in rows}):
        group = [row for row in rows if row.get("suite") == suite]
        lines.append(format_dashboard_row(suite, group))
    lines.extend(
        [
            "",
            "Local report links point at this machine's generated reports and are intentionally not included in clean exports.",
            "",
        ]
    )
    return "\n".join(lines)


def write_eval_dashboard(index_path: Path, dashboard_path: Path) -> None:
    dashboard_path.write_text(render_eval_dashboard(read_run_index(index_path)), encoding="utf-8")


def export_clean_runs(index_path: Path, clean_dir: Path) -> list[Path]:
    rows = [clean_run_row(row) for row in read_run_index(index_path) if row.get("suite")]
    clean_dir.mkdir(parents=True, exist_ok=True)
    json_path = clean_dir / "eval_runs_clean.json"
    jsonl_path = clean_dir / "eval_runs_clean.jsonl"
    dashboard_path = clean_dir / "eval_dashboard_clean.md"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n")
    dashboard_path.write_text(render_eval_dashboard(rows), encoding="utf-8")
    return [json_path, jsonl_path, dashboard_path]


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
        input_tokens = int(record.get("artifact_tokens_est") or token_estimates.get("input") or 0)
        local_tokens = int(record.get("local_output_tokens_est") or token_estimates.get("local_output") or 0)
        context_reduction = input_tokens - local_tokens
        cases.append(
            {
                "task_id": task_id,
                "timestamp": record.get("timestamp"),
                "tool": record.get("tool_name"),
                "model": record.get("model"),
                "recommendation": record.get("routing_decision") or record.get("recommendation"),
                "routing_decision": record.get("routing_decision") or record.get("recommendation"),
                "confidence_score": record.get("confidence_score"),
                "outcome": outcome.get("outcome") if outcome else "unlabeled",
                "safe_to_rely": is_safe_to_rely(record, outcome),
                "risk_flags": record.get("risk_flags") or [],
                "latency_ms": record.get("latency_ms") or 0,
                "raw_context_tokens_est": input_tokens,
                "local_output_tokens_est": local_tokens,
                "estimated_context_reduction": context_reduction,
                "cloud_tokens_avoided_est": int(record.get("cloud_tokens_avoided_est") or 0),
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
        "| Task ID | Tool | Route | Confidence | Outcome | Safe To Rely | Reduction | Latency | Risk Flags |",
        "| --- | --- | --- | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for case in results["cases"]:
        risks = ", ".join(case["risk_flags"]) or "-"
        task_id = str(case["task_id"])[:12]
        lines.append(
            "| {task_id} | `{tool}` | `{recommendation}` | {confidence} | `{outcome}` | `{safe}` | {reduction:.0%} | {latency} ms | {risks} |".format(
                task_id=task_id,
                tool=case["tool"],
                recommendation=case["recommendation"],
                confidence=(
                    f"{float(case['confidence_score']):.0%}"
                    if isinstance(case.get("confidence_score"), (int, float))
                    else "n/a"
                ),
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
            "- Before large branch pushes or review summaries, route local diffs/logs through `local_code_review` or `local_summarize` and send only accepted local summaries to cloud.",
            "",
        ]
    )
    return "\n".join(lines)


def run_git_command(args: list[str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = (completed.stdout + completed.stderr).strip()
    return completed.returncode == 0, output


def build_routing_artifacts(log_file: Path | None = None) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    for name, args in (
        ("git_diff_cached", ["diff", "--cached"]),
        ("git_diff_worktree", ["diff"]),
    ):
        ok, output = run_git_command(args)
        artifacts.append(
            {
                "name": name,
                "tool": "local_code_review",
                "artifact": output,
                "error": "" if ok else output,
            }
        )
    if log_file:
        try:
            text = log_file.read_text(encoding="utf-8", errors="replace")
            artifacts.append(
                {
                    "name": f"log_file:{log_file}",
                    "tool": "local_summarize",
                    "artifact": "\n".join(text.splitlines()[-200:]),
                    "error": "",
                }
            )
        except OSError as exc:
            artifacts.append(
                {
                    "name": f"log_file:{log_file}",
                    "tool": "local_summarize",
                    "artifact": "",
                    "error": str(exc),
                }
            )
    return artifacts


def synthetic_routing_cases() -> list[dict[str, object]]:
    sub_4k_diff = "\n".join(
        [
            "diff --git a/app.py b/app.py",
            "@@",
            "-    total += item.price * item.quantity",
            "+    total = item.price * item.quantity",
            "     return total",
            *[f" context line {index}: unchanged invoice helper" for index in range(180)],
        ]
    )
    oversized_diff = "\n".join(
        ["diff --git a/huge.py b/huge.py", "@@"] + [f"+value_{index} = {index}" for index in range(4200)]
    )
    return [
        {
            "name": "synthetic_sub4k_code_review",
            "tool": "local_code_review",
            "artifact": sub_4k_diff,
            "local_output": "Finding: total is overwritten inside the item loop instead of accumulated; verify multi-item invoices.",
            "required_facts_hit": 1,
            "required_facts_total": 1,
            "expected": "use_local",
        },
        {
            "name": "synthetic_oversized_diff",
            "tool": "local_code_review",
            "artifact": oversized_diff,
            "local_output": "Oversized diff should be summarized in smaller chunks before cloud review.",
            "required_facts_hit": 1,
            "required_facts_total": 1,
            "expected": "raw_cloud",
        },
    ]


def evaluate_routing_artifact(item: dict[str, object]) -> dict[str, object]:
    artifact = str(item.get("artifact") or "")
    local_output = str(item.get("local_output") or "")
    if not local_output and artifact:
        local_output = one_line(artifact, 400)
    routing = route_local_artifact(
        artifact=artifact,
        local_output=local_output,
        required_facts_hit=int(item.get("required_facts_hit") or (1 if artifact else 0)),
        required_facts_total=int(item.get("required_facts_total") or (1 if artifact else 0)),
        tool_error=bool(item.get("error")),
    )
    return {
        "name": item.get("name"),
        "tool": item.get("tool"),
        "routing_decision": routing.routing_decision,
        "confidence_score": routing.confidence_score,
        "artifact_tokens_est": routing.artifact_tokens_est,
        "local_output_tokens_est": routing.local_output_tokens_est,
        "cloud_tokens_avoided_est": routing.cloud_tokens_avoided_est,
        "reduction_pct": routing.reduction_pct,
        "risk_flags": list(routing.risk_flags),
        "expected": item.get("expected"),
        "decision_matches_expected": item.get("expected") in {None, routing.routing_decision},
        "error": item.get("error") or "",
    }


def run_routing_check(args: argparse.Namespace) -> dict:
    log_file = Path(args.log_file) if args.log_file else None
    artifacts = build_routing_artifacts(log_file)
    real_cases = [evaluate_routing_artifact(item) for item in artifacts]
    synthetic_cases = [evaluate_routing_artifact(item) for item in synthetic_routing_cases()]
    all_cases = real_cases + synthetic_cases
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": server.CODE_MODEL,
        "num_ctx": server.DEFAULT_NUM_CTX,
        "method": "deterministic local routing check; does not invoke Ollama",
        "summary": {
            "case_count": len(all_cases),
            "real_artifact_count": len(real_cases),
            "synthetic_case_count": len(synthetic_cases),
            "use_local_count": sum(case["routing_decision"] == "use_local" for case in all_cases),
            "verify_raw_count": sum(case["routing_decision"] == "verify_raw" for case in all_cases),
            "skip_local_count": sum(case["routing_decision"] == "skip_local" for case in all_cases),
            "raw_cloud_count": sum(case["routing_decision"] == "raw_cloud" for case in all_cases),
            "decision_match_count": sum(case["decision_matches_expected"] for case in synthetic_cases),
        },
        "cases": all_cases,
    }


def render_routing_markdown(results: dict) -> str:
    summary = results["summary"]
    lines = [
        "# Local Ollama MCP Routing Check",
        "",
        f"- Generated: `{results['generated_at']}`",
        f"- Model: `{results['model']}`",
        f"- num_ctx: `{results['num_ctx']}`",
        f"- Cases: `{summary['case_count']}`",
        f"- Synthetic decision matches: `{summary['decision_match_count']}/{summary['synthetic_case_count']}`",
        "",
        "| Case | Tool | Route | Confidence | Artifact Tokens | Local Tokens | Avoided | Risks |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in results["cases"]:
        risks = ", ".join(case["risk_flags"]) or "-"
        lines.append(
            "| {name} | `{tool}` | `{route}` | {confidence:.0%} | {artifact} | {local} | {avoided} | {risks} |".format(
                name=case["name"],
                tool=case["tool"],
                route=case["routing_decision"],
                confidence=case["confidence_score"],
                artifact=case["artifact_tokens_est"],
                local=case["local_output_tokens_est"],
                avoided=case["cloud_tokens_avoided_est"],
                risks=risks,
            )
        )
    lines.extend(
        [
            "",
            "## Guard",
            "",
            "Before large branch pushes or review summaries, route local diffs/logs through `local_code_review` or `local_summarize` and send only accepted local summaries to cloud.",
            "",
        ]
    )
    return "\n".join(lines)


async def run_eval(args: argparse.Namespace) -> dict:
    if args.from_ledger:
        return evaluate_ledger(Path(args.ledger_path) if args.ledger_path else server.ledger_path())
    if args.routing_check:
        return run_routing_check(args)

    if not args.no_status:
        status_before = await server.local_ollama_status()
    else:
        status_before = "skipped"

    if not args.no_warm:
        warm_result = await server.local_warm_model()
    else:
        warm_result = "skipped"

    case_results = []
    for case in build_suite_cases(args):
        case_results.append(
            await evaluate_case(
                case,
                use_local_judge=args.local_judge,
                judge_model=args.judge_model,
            )
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": server.CODE_MODEL,
        "reason_model": server.REASON_MODEL,
        "num_ctx": server.DEFAULT_NUM_CTX,
        "suite": args.suite,
        "method": "local output scored against expert baseline facts; token counts are regex estimates",
        "status_before": status_before,
        "warm_result": warm_result,
        "summary": summarize(case_results),
        "cases": [asdict(result) for result in case_results],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate local MCP usefulness for reducing cloud context."
    )
    parser.add_argument("--suite", choices=SUITES, default="synthetic", help="Offline eval suite to run.")
    parser.add_argument("--model", default="", help="Override default local Ollama model to evaluate.")
    parser.add_argument("--plan-model", default="", help="Override the local planning model for this run.")
    parser.add_argument("--reason-model", default="", help="Override the local reasoning model for this run.")
    parser.add_argument("--warm-model", default="", help="Override the local warm model for this run.")
    parser.add_argument("--output-dir", default=".", help="Directory for JSON and Markdown reports.")
    parser.add_argument("--json-name", default="local_mcp_eval_results.json")
    parser.add_argument("--markdown-name", default="local_mcp_eval_report.md")
    parser.add_argument("--no-warm", action="store_true", help="Skip local_warm_model before cases.")
    parser.add_argument("--no-status", action="store_true", help="Skip local_ollama_status before cases.")
    parser.add_argument("--from-ledger", action="store_true", help="Analyze captured ledger records instead of running synthetic cases.")
    parser.add_argument("--ledger-path", default="", help="Ledger path for --from-ledger; defaults to LOCAL_MCP_LEDGER_PATH or .local_ollama_mcp/ledger.jsonl.")
    parser.add_argument("--routing-check", action="store_true", help="Run deterministic local routing checks without invoking Ollama.")
    parser.add_argument("--log-file", default="", help="Optional recent command/log file to include in --routing-check.")
    parser.add_argument("--case-file", action="append", default=[], help="External JSONL eval case file.")
    parser.add_argument("--case-dir", default=DEFAULT_CASE_DIR, help="Directory of external JSONL eval cases.")
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR, help="Directory with diffs/ and logs/ artifacts.")
    parser.add_argument("--local-judge", action="store_true", help="Add optional non-authoritative local model judge scores.")
    parser.add_argument("--judge-model", default=server.CODE_MODEL, help="Local Ollama model for --local-judge.")
    parser.add_argument("--run-index-path", default=DEFAULT_RUN_INDEX, help="Append one summary row per run.")
    parser.add_argument("--write-dashboard", action="store_true", help="Write the aggregate eval dashboard from the run index without running cases.")
    parser.add_argument("--dashboard-path", default=DEFAULT_DASHBOARD_PATH, help="Dashboard path for --write-dashboard.")
    parser.add_argument("--export-clean", default="", metavar="CLEAN_DIR", help="Write anonymous aggregate eval exports into CLEAN_DIR without raw outputs or local paths.")
    return parser.parse_args()


def current_git_commit() -> str:
    ok, output = run_git_command(["rev-parse", "--short", "HEAD"])
    return output.strip() if ok else "unknown"


def append_run_index(results: dict, json_path: Path, markdown_path: Path, index_path: Path) -> None:
    if results.get("method", "").startswith("captured ledger"):
        suite = "ledger"
    elif results.get("method", "").startswith("deterministic local routing"):
        suite = "routing"
    else:
        suite = results.get("suite", "synthetic")
    summary = results.get("summary", {})
    row = {
        "timestamp": results.get("generated_at"),
        "git_commit": current_git_commit(),
        "suite": suite,
        "model": results.get("model"),
        "reason_model": results.get("reason_model"),
        "num_ctx": results.get("num_ctx"),
        "case_count": summary.get("case_count") or summary.get("tool_record_count") or 0,
        "use_local_count": summary.get("use_local_count"),
        "verify_raw_count": summary.get("verify_raw_count"),
        "skip_local_count": summary.get("skip_local_count"),
        "raw_cloud_count": summary.get("raw_cloud_count"),
        "average_accuracy_score": summary.get("average_accuracy_score"),
        "average_structure_score": summary.get("average_structure_score"),
        "average_usefulness_score": summary.get("average_usefulness_score"),
        "average_latency_ms": summary.get("average_latency_ms"),
        "average_tokens_per_second": summary.get("average_tokens_per_second", 0.0),
        "think_leak_count": summary.get("think_leak_count"),
        "aggregate_token_reduction_pct": summary.get("aggregate_token_reduction_pct")
        or summary.get("aggregate_context_reduction_pct"),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n")


async def main() -> None:
    args = parse_args()
    server.configure_models(
        code_model=args.model or None,
        plan_model=args.plan_model or None,
        reason_model=args.reason_model or None,
        warm_model=args.warm_model or None,
    )
    run_index_path = Path(args.run_index_path)
    if args.write_dashboard:
        dashboard_path = Path(args.dashboard_path)
        write_eval_dashboard(run_index_path, dashboard_path)
        print(f"Wrote {dashboard_path}")
        return
    if args.export_clean:
        written = export_clean_runs(run_index_path, Path(args.export_clean))
        for path in written:
            print(f"Wrote {path}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = await run_eval(args)
    json_path = output_dir / args.json_name
    markdown_path = output_dir / args.markdown_name
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.from_ledger:
        markdown = render_ledger_markdown(results)
    elif args.routing_check:
        markdown = render_routing_markdown(results)
    else:
        markdown = render_markdown(results)
    markdown_path.write_text(markdown, encoding="utf-8")
    append_run_index(results, json_path, markdown_path, run_index_path)

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
    if args.routing_check:
        print(
            textwrap.dedent(
                f"""
                Wrote {json_path}
                Wrote {markdown_path}

                Cases: {summary['case_count']}
                use_local: {summary['use_local_count']}
                verify_raw: {summary['verify_raw_count']}
                skip_local: {summary['skip_local_count']}
                raw_cloud: {summary['raw_cloud_count']}
                synthetic matches: {summary['decision_match_count']}/{summary['synthetic_case_count']}
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
            average usefulness: {summary['average_usefulness_score']:.1%}
            average latency: {summary['average_latency_ms']} ms
            think leakage: {'yes' if summary['think_leak_count'] else 'no'}
            """
        ).strip()
    )


if __name__ == "__main__":
    asyncio.run(main())
