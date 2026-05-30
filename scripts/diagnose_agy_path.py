#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import server  # noqa: E402


EXPECTED_SECTIONS = (
    "CHANGED_FILES:",
    "BEHAVIOR_CHANGES:",
    "RISKY_LINES:",
    "REMOVED_LOGIC:",
)


@dataclass(frozen=True)
class ProbeResult:
    name: str
    form: str
    prompt_tokens_est: int
    prompt_chars: int
    timeout_sec: float
    elapsed_sec: float
    exit_code: int | None
    timed_out: bool
    matched_sections: bool
    stdout_snippet: str
    stderr_snippet: str


def snippet(text: str, limit: int = 500) -> str:
    text = text.strip().replace("\r", "")
    return text[:limit]


def has_expected_sections(output: str) -> bool:
    return all(section in output for section in EXPECTED_SECTIONS)


def build_diff_prompt(target_prompt_tokens: int) -> str:
    lines: list[str] = []
    idx = 0
    prompt = ""
    while True:
        lines.append(
            f"+ changed line {idx}: update payment reconciliation branch "
            f"with repeated customer invoice refund ledger retry context {idx % 17}"
        )
        idx += 1
        if idx % 100:
            continue
        diff = "\n".join(lines)
        prompt = (
            f"{server._AGY_DIFF_COMPRESS_INSTRUCTIONS}\n\n"
            "FOCUS: bugs, regressions, API surface changes\n\n"
            f"<diff>\n{diff}\n</diff>"
        )
        if server.estimate_tokens(prompt) >= target_prompt_tokens:
            return prompt


async def run_probe(
    *,
    name: str,
    cmd: list[str],
    prompt: str,
    stdin: bool,
    timeout_sec: float,
    expect_sections: bool,
) -> ProbeResult:
    started = time.perf_counter()
    proc: asyncio.subprocess.Process | None = None
    stdout = b""
    stderr = b""
    timed_out = False
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if stdin else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ},
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=prompt.encode("utf-8") if stdin else None),
            timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        timed_out = True
        await server.kill_process(proc)
    except FileNotFoundError as exc:
        stderr = str(exc).encode("utf-8")

    elapsed = time.perf_counter() - started
    out = stdout.decode(errors="replace")
    err = stderr.decode(errors="replace")
    if expect_sections:
        matched = has_expected_sections(out)
    else:
        matched = out.strip() == "OK"

    return ProbeResult(
        name=name,
        form="stdin" if stdin else "argument",
        prompt_tokens_est=server.estimate_tokens(prompt),
        prompt_chars=len(prompt),
        timeout_sec=timeout_sec,
        elapsed_sec=round(elapsed, 3),
        exit_code=proc.returncode if proc else None,
        timed_out=timed_out,
        matched_sections=matched,
        stdout_snippet=snippet(out),
        stderr_snippet=snippet(err),
    )


async def run_diagnostics(targets: list[int], timeout_sec: float, tiny_timeout_sec: float) -> list[ProbeResult]:
    tiny_prompt = "Return exactly OK."
    print_timeout = server.agy_print_timeout_arg(tiny_timeout_sec)
    results = [
        await run_probe(
            name="tiny-argument",
            cmd=[server.AGY_BIN, "--print-timeout", print_timeout, "--print", tiny_prompt],
            prompt=tiny_prompt,
            stdin=False,
            timeout_sec=tiny_timeout_sec + 5,
            expect_sections=False,
        ),
        await run_probe(
            name="tiny-stdin",
            cmd=[server.AGY_BIN, "--print-timeout", print_timeout, "--print", "-"],
            prompt=tiny_prompt,
            stdin=True,
            timeout_sec=tiny_timeout_sec + 5,
            expect_sections=False,
        ),
    ]

    large_print_timeout = server.agy_print_timeout_arg(timeout_sec)
    for target in targets:
        prompt = build_diff_prompt(target)
        results.append(
            await run_probe(
                name=f"compress-{target}",
                cmd=[server.AGY_BIN, "--print-timeout", large_print_timeout, "--print", "-"],
                prompt=prompt,
                stdin=True,
                timeout_sec=timeout_sec + 5,
                expect_sections=True,
            )
        )
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated agy --print probes for the Gemini diff-compression path."
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        nargs="+",
        default=[4000, 12000, 30000],
        help="Estimated prompt token sizes for generated diff-compression probes.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=server.AGY_DIFF_COMPRESS_TIMEOUT,
        help="Python and agy print timeout budget for generated compression probes.",
    )
    parser.add_argument(
        "--tiny-timeout",
        type=float,
        default=60,
        help="Python and agy print timeout budget for tiny OK probes.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit newline-delimited JSON instead of a compact text table.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = asyncio.run(
        run_diagnostics(
            targets=args.target_tokens,
            timeout_sec=args.timeout,
            tiny_timeout_sec=args.tiny_timeout,
        )
    )

    if args.json:
        for result in results:
            print(json.dumps(asdict(result), sort_keys=True))
    else:
        print("diagnose_agy_path: isolated agy probes; production telemetry is not read")
        for result in results:
            status = "PASS" if result.exit_code == 0 and not result.timed_out and result.matched_sections else "FAIL"
            print(
                f"{status} {result.name} form={result.form} "
                f"tokens={result.prompt_tokens_est} chars={result.prompt_chars} "
                f"elapsed={result.elapsed_sec}s exit={result.exit_code} "
                f"timed_out={result.timed_out} matched={result.matched_sections}"
            )
            if result.stdout_snippet:
                print(f"  stdout: {result.stdout_snippet}")
            if result.stderr_snippet:
                print(f"  stderr: {result.stderr_snippet}")

    return 0 if all(
        result.exit_code == 0 and not result.timed_out and result.matched_sections
        for result in results
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
