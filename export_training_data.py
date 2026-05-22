#!/usr/bin/env python3
"""
Export labeled local MCP ledger records into future training/eval JSONL files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import server


FAILURE_OUTCOMES = {"needs_raw_verification", "misleading", "too_verbose"}


def warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def compact(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def prompt_for_record(record: dict[str, Any]) -> str:
    tool_name = record.get("tool_name", "local_tool")
    input_payload = compact(record.get("input", {}))
    return "\n".join(
        [
            f"Tool: {tool_name}",
            "Produce the concise local MCP output Codex should receive for this input.",
            "",
            "Input:",
            input_payload,
        ]
    )


def latest_outcomes_by_task(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    outcomes: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("record_type") != "outcome":
            continue
        task_id = record.get("task_id")
        if not task_id:
            warn("skipping malformed outcome without task_id")
            continue
        outcomes[str(task_id)] = record
    return outcomes


def read_records_with_warnings(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        warn(f"ledger does not exist: {path}")
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                warn(f"skipping malformed JSON on line {line_number}: {exc.msg}")
                continue
            if not isinstance(record, dict):
                warn(f"skipping malformed non-object record on line {line_number}")
                continue
            records.append(record)
    return records


def joined_labeled_records(records: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    outcomes = latest_outcomes_by_task(records)
    joined = []
    for record in records:
        if record.get("record_type") != "tool_call":
            continue
        task_id = record.get("task_id")
        if not task_id:
            warn("skipping malformed tool_call without task_id")
            continue
        outcome = outcomes.get(str(task_id))
        if not outcome:
            warn(f"skipping unlabeled tool_call {task_id}")
            continue
        if not compact(outcome.get("accepted_solution", "")):
            warn(f"skipping labeled tool_call {task_id} without accepted_solution")
            continue
        joined.append((record, outcome))
    return joined


def sft_example(record: dict[str, Any], outcome: dict[str, Any]) -> dict[str, Any] | None:
    if outcome.get("outcome") != "useful":
        return None
    return {
        "task_id": record.get("task_id"),
        "messages": [
            {
                "role": "system",
                "content": "You are a local MCP helper for Codex. Be concise, factual, and do not include chain-of-thought.",
            },
            {"role": "user", "content": prompt_for_record(record)},
            {"role": "assistant", "content": compact(outcome.get("accepted_solution", ""))},
        ],
        "metadata": {
            "tool_name": record.get("tool_name"),
            "model": record.get("model"),
            "outcome": outcome.get("outcome"),
            "recommendation": record.get("recommendation"),
        },
    }


def preference_example(record: dict[str, Any], outcome: dict[str, Any]) -> dict[str, Any] | None:
    if outcome.get("outcome") not in FAILURE_OUTCOMES:
        return None
    rejected = compact(record.get("local_output", ""))
    if not rejected:
        return None
    return {
        "task_id": record.get("task_id"),
        "prompt": prompt_for_record(record),
        "chosen": compact(outcome.get("accepted_solution", "")),
        "rejected": rejected,
        "metadata": {
            "tool_name": record.get("tool_name"),
            "model": record.get("model"),
            "outcome": outcome.get("outcome"),
            "risk_flags": record.get("risk_flags") or [],
        },
    }


def eval_example(record: dict[str, Any], outcome: dict[str, Any]) -> dict[str, Any] | None:
    if outcome.get("outcome") not in FAILURE_OUTCOMES:
        return None
    return {
        "task_id": record.get("task_id"),
        "name": f"{record.get('tool_name', 'local_tool')}_{str(record.get('task_id', ''))[:12]}",
        "tool_name": record.get("tool_name"),
        "input": record.get("input", {}),
        "expected_output": compact(outcome.get("accepted_solution", "")),
        "previous_local_output": compact(record.get("local_output", "")),
        "failure_label": outcome.get("outcome"),
        "risk_flags": record.get("risk_flags") or [],
    }


def build_examples(format_name: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    builders = {
        "sft": sft_example,
        "preference": preference_example,
        "eval": eval_example,
    }
    builder = builders[format_name]
    examples = []
    for record, outcome in joined_labeled_records(records):
        example = builder(record, outcome)
        if example is None:
            warn(
                "skipping task {task_id} for {format_name} export due to outcome {outcome}".format(
                    task_id=record.get("task_id"),
                    format_name=format_name,
                    outcome=outcome.get("outcome"),
                )
            )
            continue
        examples.append(example)
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export labeled local MCP ledger records.")
    parser.add_argument("--format", choices=("sft", "preference", "eval"), required=True)
    parser.add_argument(
        "--ledger-path",
        default="",
        help="Ledger path; defaults to LOCAL_MCP_LEDGER_PATH or .local_ollama_mcp/ledger.jsonl.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSONL path; defaults to .local_ollama_mcp/<format>_export.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ledger_path = Path(args.ledger_path) if args.ledger_path else server.ledger_path()
    output_path = Path(args.output) if args.output else Path(".local_ollama_mcp") / f"{args.format}_export.jsonl"

    records = read_records_with_warnings(ledger_path)
    examples = build_examples(args.format, records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, sort_keys=True, ensure_ascii=False, default=str) + "\n")

    print(f"wrote {len(examples)} {args.format} examples to {output_path}")


if __name__ == "__main__":
    main()
