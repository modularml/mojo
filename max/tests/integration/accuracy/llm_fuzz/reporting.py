# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""
Reporting: terminal output with color, summary stats, and JSON/CSV export.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone

from _version import __version__ as _version
from scenarios import ScenarioResult, Verdict

# ANSI colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

VERDICT_COLORS = {
    Verdict.PASS: GREEN,
    Verdict.FAIL: RED,
    Verdict.INTERESTING: YELLOW,
    Verdict.ERROR: RED,
}

VERDICT_ICONS = {
    Verdict.PASS: "✓",
    Verdict.FAIL: "✗",
    Verdict.INTERESTING: "?",
    Verdict.ERROR: "!",
}


def print_header(text: str) -> None:
    width = 80
    print(f"\n{BOLD}{CYAN}{'═' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'═' * width}{RESET}\n")


def print_banner() -> None:
    """Print the LLM FUZZ ASCII art banner."""
    art = (
        "██╗     ██╗     ███╗   ███╗   ███████╗██╗   ██╗███████╗███████╗\n"
        "██║     ██║     ████╗ ████║   ██╔════╝██║   ██║╚══███╔╝╚══███╔╝\n"
        "██║     ██║     ██╔████╔██║   █████╗  ██║   ██║  ███╔╝   ███╔╝\n"
        "██║     ██║     ██║╚██╔╝██║   ██╔══╝  ██║   ██║ ███╔╝   ███╔╝\n"
        "███████╗███████╗██║ ╚═╝ ██║   ██║     ╚██████╔╝███████╗███████╗\n"
        "╚══════╝╚══════╝╚═╝     ╚═╝   ╚═╝      ╚═════╝ ╚══════╝╚══════╝"
    )
    print(f"\n{BOLD}{CYAN}{'─' * 68}{RESET}")
    for line in art.splitlines():
        print(f"  {BOLD}{CYAN}{line}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 68}{RESET}")
    print(
        f"  {DIM}Adversarial fuzz testing for LLM inference endpoints{RESET}\n"
    )


def print_scenario_header(name: str, description: str) -> None:
    print(f"\n{BOLD}▶ {name}{RESET}")
    print(f"  {DIM}{description}{RESET}")
    print(f"  {DIM}{'─' * 70}{RESET}")


def print_result(result: ScenarioResult, verbose: bool = False) -> None:
    color = VERDICT_COLORS[result.verdict]
    icon = VERDICT_ICONS[result.verdict]

    status_str = (
        f"[{result.status_code}]" if result.status_code is not None else "[---]"
    )
    time_str = (
        f"{result.elapsed_ms:>7.0f}ms" if result.elapsed_ms else "       -"
    )

    print(
        f"  {color}{icon} {result.verdict.value:>7}{RESET}  "
        f"{DIM}{status_str:>5}{RESET}  "
        f"{DIM}{time_str}{RESET}  "
        f"{result.test_name}"
    )

    if verbose or result.verdict in (Verdict.FAIL, Verdict.INTERESTING):
        for prefix, field, color in [
            ("", result.detail, DIM),
            ("error: ", result.error, RED),
            ("request: ", result.request_body, DIM),
            ("body: ", result.response_body, DIM),
        ]:
            if not field:
                continue
            field = field.replace("\n", " ")
            print(f"    {color}↳ {prefix}{field}{RESET}")


def print_summary(
    all_results: list[ScenarioResult], elapsed_total: float
) -> None:
    total = len(all_results)
    by_verdict: dict[Verdict, int] = {}
    for r in all_results:
        by_verdict[r.verdict] = by_verdict.get(r.verdict, 0) + 1

    passes = by_verdict.get(Verdict.PASS, 0)
    fails = by_verdict.get(Verdict.FAIL, 0)
    interesting = by_verdict.get(Verdict.INTERESTING, 0)
    errors = by_verdict.get(Verdict.ERROR, 0)

    # Group failures by scenario
    fail_by_scenario: dict[str, list[ScenarioResult]] = {}
    for r in all_results:
        if r.verdict == Verdict.FAIL:
            fail_by_scenario.setdefault(r.scenario_name, []).append(r)

    interesting_by_scenario: dict[str, list[ScenarioResult]] = {}
    for r in all_results:
        if r.verdict == Verdict.INTERESTING:
            interesting_by_scenario.setdefault(r.scenario_name, []).append(r)

    print_header("LLM FUZZ SUMMARY")

    print(f"  Total tests:  {BOLD}{total}{RESET}")
    print(f"  Duration:     {elapsed_total:.1f}s")
    print()
    print(f"  {GREEN}✓ PASS:        {passes}{RESET}")
    print(f"  {RED}✗ FAIL:        {fails}{RESET}")
    print(f"  {YELLOW}? INTERESTING: {interesting}{RESET}")
    print(f"  {RED}! ERROR:       {errors}{RESET}")

    if fail_by_scenario:
        print(f"\n{BOLD}{RED}FAILURES:{RESET}")
        for scenario, results in sorted(fail_by_scenario.items()):
            print(f"  {RED}▸ {scenario}:{RESET}")
            for r in results:
                print(
                    f"    {RED}✗ {r.test_name}{RESET}  {DIM}{r.detail}{RESET}"
                )

    if interesting_by_scenario:
        print(f"\n{BOLD}{YELLOW}WORTH INVESTIGATING:{RESET}")
        for scenario, results in sorted(interesting_by_scenario.items()):
            print(f"  {YELLOW}▸ {scenario}:{RESET}")
            for r in results[:5]:
                print(
                    f"    {YELLOW}? {r.test_name}{RESET}  {DIM}{r.detail}{RESET}"
                )
            if len(results) > 5:
                print(f"    {DIM}... and {len(results) - 5} more{RESET}")

    # Overall verdict
    print()
    if fails == 0 and errors == 0:
        print(f"  {GREEN}{BOLD}OVERALL: ENDPOINT SURVIVED ALL TESTS ✓{RESET}")
    elif fails <= 3:
        print(
            f"  {YELLOW}{BOLD}OVERALL: SOME WEAKNESSES FOUND ({fails} failures){RESET}"
        )
    else:
        print(
            f"  {RED}{BOLD}OVERALL: ENDPOINT HAS SIGNIFICANT ISSUES ({fails} failures){RESET}"
        )

    print()


def print_progress(
    elapsed_sec: float,
    total_sec: float,
    total_requests: int,
    errors: int,
    latencies: list[float],
) -> None:
    """Print a single-line progress update for endurance mode."""
    elapsed_m = int(elapsed_sec) // 60
    elapsed_s = int(elapsed_sec) % 60
    total_m = int(total_sec) // 60
    total_s = int(total_sec) % 60
    error_pct = (errors / total_requests * 100) if total_requests > 0 else 0

    if latencies:
        sorted_l = sorted(latencies)
        p50 = sorted_l[len(sorted_l) // 2]
        p99 = sorted_l[int(len(sorted_l) * 0.99)]
        lat_str = f"p50: {p50:.0f}ms | p99: {p99:.0f}ms"
    else:
        lat_str = "p50: -ms | p99: -ms"

    color = RED if error_pct > 5 else (YELLOW if error_pct > 1 else GREEN)
    print(
        f"\r  {DIM}[{elapsed_m:02d}:{elapsed_s:02d}/{total_m:02d}:{total_s:02d}]{RESET} "
        f"{total_requests:,} requests | "
        f"{color}{error_pct:.1f}% errors{RESET} | "
        f"{lat_str}",
        end="",
        flush=True,
    )


def export_json(
    results: list[ScenarioResult],
    path: str,
    *,
    url: str = "",
    model: str = "",
    elapsed_sec: float = 0.0,
) -> None:
    # -- meta --
    meta = {
        "tool": "llm-fuzz",
        "version": _version,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "url": url,
        "model": model,
        "elapsed_sec": round(elapsed_sec, 1),
    }

    # -- summary --
    verdict_counts: Counter[Verdict] = Counter(r.verdict for r in results)
    total = len(results)
    pass_count = verdict_counts.get(Verdict.PASS, 0)
    pass_rate = round(pass_count / total * 100, 1) if total > 0 else 0.0

    # Per-scenario breakdown
    scenario_totals: dict[str, Counter[Verdict]] = {}
    for r in results:
        scenario_totals.setdefault(r.scenario_name, Counter())[r.verdict] += 1

    scenarios_summary: dict[str, dict[str, int]] = {}
    for name, counts in sorted(scenario_totals.items()):
        scenarios_summary[name] = {
            "total": sum(counts.values()),
            "pass": counts.get(Verdict.PASS, 0),
            "fail": counts.get(Verdict.FAIL, 0),
        }

    summary = {
        "total": total,
        "pass": pass_count,
        "fail": verdict_counts.get(Verdict.FAIL, 0),
        "interesting": verdict_counts.get(Verdict.INTERESTING, 0),
        "error": verdict_counts.get(Verdict.ERROR, 0),
        "pass_rate": pass_rate,
        "scenarios": scenarios_summary,
    }

    # -- results --
    result_dicts = []
    for r in results:
        result_dicts.append(
            {
                "scenario_name": r.scenario_name,
                "test_name": r.test_name,
                "verdict": r.verdict.value
                if hasattr(r.verdict, "value")
                else str(r.verdict),
                "status_code": r.status_code,
                "elapsed_ms": r.elapsed_ms,
                "detail": r.detail,
                "error": r.error,
            }
        )

    payload = {
        "meta": meta,
        "summary": summary,
        "results": result_dicts,
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"{DIM}Results exported to {path}{RESET}")


def export_csv(results: list[ScenarioResult], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scenario",
                "test",
                "verdict",
                "status_code",
                "elapsed_ms",
                "detail",
                "error",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.scenario_name,
                    r.test_name,
                    r.verdict.value
                    if hasattr(r.verdict, "value")
                    else str(r.verdict),
                    r.status_code,
                    f"{r.elapsed_ms:.1f}",
                    r.detail,
                    r.error,
                ]
            )
    print(f"{DIM}Results exported to {path}{RESET}")


def compare_with_baseline(
    current_results: list[ScenarioResult], baseline_path: str
) -> None:
    """Compare current results against a baseline file and print a regression report.

    Supports .json (list of result dicts or enriched format) and .jsonl (lines with
    "event": "test_result").
    """
    baseline_verdicts = _load_baseline(baseline_path)

    current_verdicts: dict[str, str] = {}
    for r in current_results:
        key = f"{r.scenario_name}:{r.test_name}"
        current_verdicts[key] = r.verdict.value

    baseline_only = set(baseline_verdicts) - set(current_verdicts)
    current_only = set(current_verdicts) - set(baseline_verdicts)
    common = set(baseline_verdicts) & set(current_verdicts)

    new_failures: list[str] = []
    recovered: list[str] = []
    unchanged = 0

    for key in sorted(common):
        old = baseline_verdicts[key]
        new = current_verdicts[key]
        if old == "PASS" and new == "FAIL":
            new_failures.append(key)
        elif old == "FAIL" and new == "PASS":
            recovered.append(key)
        elif old == new:
            unchanged += 1

    print_header("BASELINE COMPARISON")

    if new_failures:
        print(f"  {RED}{BOLD}NEW FAILURES ({len(new_failures)}):{RESET}")
        for key in new_failures:
            print(f"    {RED}✗ {key}{RESET}  {DIM}was PASS, now FAIL{RESET}")
        print()

    if recovered:
        print(f"  {GREEN}{BOLD}RECOVERED ({len(recovered)}):{RESET}")
        for key in recovered:
            print(f"    {GREEN}✓ {key}{RESET}  {DIM}was FAIL, now PASS{RESET}")
        print()

    if current_only:
        print(f"  {CYAN}{BOLD}NEW TESTS ({len(current_only)}):{RESET}")
        for key in sorted(current_only):
            verdict = current_verdicts[key]
            color = GREEN if verdict == "PASS" else RED
            print(f"    {color}  {key}{RESET}  {DIM}{verdict}{RESET}")
        print()

    if baseline_only:
        print(f"  {YELLOW}{BOLD}REMOVED TESTS ({len(baseline_only)}):{RESET}")
        for key in sorted(baseline_only):
            print(f"    {DIM}  {key}  (was {baseline_verdicts[key]}){RESET}")
        print()

    print(
        f"  {BOLD}Summary:{RESET} "
        f"{len(new_failures)} regressions, "
        f"{len(recovered)} recoveries, "
        f"{len(current_only)} new tests, "
        f"{len(baseline_only)} removed, "
        f"{unchanged} unchanged"
    )
    print()


def _load_baseline(path: str) -> dict[str, str]:
    """Load baseline verdicts from a .json or .jsonl file.

    Returns a dict mapping "scenario:test" -> verdict string.
    """
    verdicts: dict[str, str] = {}

    with open(path) as f:
        if path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("event") != "test_result":
                    continue
                sn, tn, v = (
                    obj.get("scenario_name"),
                    obj.get("test_name"),
                    obj.get("verdict"),
                )
                if not sn or not tn or not v:
                    continue
                verdicts[f"{sn}:{tn}"] = str(v)
        else:
            data = json.load(f)
            if isinstance(data, dict) and "results" in data:
                items = data["results"]
            elif isinstance(data, list):
                items = data
            else:
                items = []
            for obj in items:
                sn, tn = obj.get("scenario_name"), obj.get("test_name")
                if not sn or not tn:
                    continue
                verdict = obj.get("verdict", "")
                if hasattr(verdict, "value"):
                    verdict = verdict.value
                verdicts[f"{sn}:{tn}"] = str(verdict)

    return verdicts
