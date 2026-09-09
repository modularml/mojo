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
"""llm-fuzz — Unified fuzz and correctness testing for LLM inference endpoints.

Sends adversarial, malformed, and edge-case requests to OpenAI-compatible
chat completion endpoints to find crashes, hangs, state corruption, and
resource leaks. Also validates correctness of tool calling, structured output,
streaming protocol, and model-specific behavior.

Usage (with ``LLM_FUZZ="./bazelw run //max/tests/integration/accuracy:llm-fuzz --"``):
    $LLM_FUZZ --url http://localhost:8000 --model my-model
    $LLM_FUZZ --url http://localhost:8000 --model my-model --scenarios malformed_payloads,streaming_attacks
    $LLM_FUZZ --url http://localhost:8000 --model my-model --tags crash --verbose
    $LLM_FUZZ --url http://localhost:8000 --model my-model --validation-only
    $LLM_FUZZ --url http://localhost:8200 --model kimi-k2.5 --model-profile kimi-k2.5
    $LLM_FUZZ --url http://localhost:8000 --model my-model --list
    $LLM_FUZZ --url https://api.openai.com --api-key sk-... --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Any

from _version import __version__
from client import FuzzClient, RunConfig
from model_config import MODEL_PROFILES, build_model_config
from reporting import (
    BOLD,
    CYAN,
    DIM,
    GREEN,
    RED,
    RESET,
    YELLOW,
    compare_with_baseline,
    export_csv,
    export_json,
    print_banner,
    print_header,
    print_result,
    print_scenario_header,
    print_summary,
)
from run_log import RunLog
from scenarios import (
    BaseScenario,
    CircuitBreaker,
    ScenarioResult,
    Verdict,
    get_all_scenarios,
)
from validator_client import (
    ValidatorClient,
    make_validator_config,
)


def _format_elapsed(seconds: float) -> str:
    """Format seconds into 'Xm Ys' or 'Xs' if under a minute."""
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes}m {secs:02d}s"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="llm-fuzz",
        description="Unified fuzz and correctness testing for LLM inference endpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --url http://localhost:8000 --model llama3
  %(prog)s --url http://localhost:8000 --model llama3 --scenarios malformed_payloads
  %(prog)s --url http://localhost:8000 --model llama3 --tags crash,streaming
  %(prog)s --url http://localhost:8000 --model llama3 --validation-only
  %(prog)s --url http://localhost:8200 --model kimi-k2.5 --model-profile kimi-k2.5
  %(prog)s --url https://api.openai.com --api-key $OPENAI_API_KEY --model gpt-4o-mini
  %(prog)s --list
        """,
    )
    p.add_argument(
        "-V", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    p.add_argument(
        "--url",
        required="--list" not in sys.argv
        and "--version" not in sys.argv
        and "-V" not in sys.argv,
        help="Base URL of the endpoint (e.g. http://localhost:8000)",
    )
    p.add_argument(
        "--model", default="default", help="Model name to use in requests"
    )
    p.add_argument("--api-key", default="", help="API key (Bearer token)")
    p.add_argument(
        "--timeout",
        type=float,
        default=30,
        help="Default request timeout in seconds",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=200,
        help="Max concurrent requests",
    )

    p.add_argument(
        "--scenarios",
        help="Comma-separated list of scenario names to run (default: all)",
    )
    p.add_argument(
        "--tags",
        help="Comma-separated tags to filter scenarios (runs scenarios matching ANY tag)",
    )
    p.add_argument(
        "--exclude",
        help=(
            "Comma-separated scenarios to exclude. Use scenario:test to "
            "exclude a single test instead of the whole scenario."
        ),
    )

    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output for all tests; with --export-json, scenarios may include request/response bodies",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List all available scenarios and exit",
    )
    p.add_argument(
        "--export-json", metavar="PATH", help="Export results to JSON file"
    )
    p.add_argument(
        "--export-csv", metavar="PATH", help="Export results to CSV file"
    )
    p.add_argument(
        "--compare",
        metavar="PATH",
        help="Compare results against a baseline JSON/JSONL file",
    )
    log_group = p.add_mutually_exclusive_group()
    log_group.add_argument(
        "--no-log",
        action="store_true",
        help="Disable automatic JSONL run logging",
    )
    log_group.add_argument(
        "--log-file",
        metavar="PATH",
        help="Write the JSONL run log to PATH (default: logs/run-{timestamp}.jsonl)",
    )
    p.add_argument(
        "--no-health-check",
        action="store_true",
        help="Skip initial health check",
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="Run each scenario N times to detect flaky tests (default: 1, must be >= 1)",
    )

    p.add_argument(
        "--endurance",
        action="store_true",
        help="Enable endurance mode (long-running soak tests only)",
    )
    p.add_argument(
        "--endurance-duration",
        type=float,
        default=5,
        metavar="MINUTES",
        help="Endurance test duration in minutes (default: 5)",
    )
    p.add_argument(
        "--endurance-intensity",
        choices=["low", "medium", "high"],
        default="medium",
        help="Request rate: low=5/s, medium=20/s, high=100/s",
    )

    # Scenario type filtering
    mode_group = p.add_argument_group(
        "scenario modes", "Filter by scenario type"
    )
    mode_excl = mode_group.add_mutually_exclusive_group()
    mode_excl.add_argument(
        "--fuzz-only",
        action="store_true",
        help="Run only fuzz scenarios (skip validation)",
    )
    mode_excl.add_argument(
        "--validation-only",
        action="store_true",
        help="Run only validation scenarios (skip fuzz)",
    )

    # Model-specific testing
    profile_group = p.add_argument_group(
        "model profiles", "Model-specific test suites"
    )
    profile_group.add_argument(
        "--model-profile",
        choices=list(MODEL_PROFILES.keys()),
        default=None,
        help="Activate model-specific test scenarios (e.g. kimi-k2.5, glm-5.1, gemma4)",
    )
    profile_group.add_argument(
        "--image-stress-nodes",
        type=int,
        default=1,
        metavar="N",
        help="Scale image_stress concurrency for an N-node deployment (default: 1)",
    )
    profile_group.add_argument(
        "--k2vv-mode",
        choices=["quick", "full"],
        default="quick",
        help="K2VV benchmark mode: quick=500 samples, full=2000 (default: quick)",
    )

    # Circuit breaker
    p.add_argument(
        "--circuit-breaker",
        type=int,
        default=5,
        metavar="N",
        help="Stop after N consecutive server failures (default: 5, 0=disabled)",
    )

    model_group = p.add_argument_group(
        "model configuration",
        "Model-specific parameters for adaptive test sizing",
    )
    model_group.add_argument(
        "--max-context-length",
        type=int,
        default=None,
        metavar="TOKENS",
        help="Override max context window (max_position_embeddings). Takes priority over HuggingFace config.",
    )
    model_group.add_argument(
        "--max-num-tokens",
        type=int,
        default=None,
        metavar="TOKENS",
        help="Override engine max_num_tokens (deployment-level, not in HF config). Default: 4096.",
    )
    model_group.add_argument(
        "--no-hf-fetch",
        action="store_true",
        help="Disable automatic HuggingFace config fetch (uses defaults or manual overrides only)",
    )

    return p.parse_args()


def list_scenarios() -> None:
    """Print all registered scenarios."""
    scenarios = get_all_scenarios()
    print_header("AVAILABLE SCENARIOS")

    fuzz_scenarios = []
    validation_scenarios = []
    model_scenarios = []

    for name, cls in sorted(scenarios.items()):
        if cls.model_filter:
            model_scenarios.append((name, cls))
        elif cls.scenario_type == "validation":
            validation_scenarios.append((name, cls))
        else:
            fuzz_scenarios.append((name, cls))

    def _print_group(
        title: str, items: list[tuple[str, type[BaseScenario]]]
    ) -> None:
        if not items:
            return
        print(f"  {BOLD}{CYAN}{title}{RESET}")
        for name, cls in items:
            tags = ", ".join(cls.tags) if cls.tags else "none"
            marker = ""
            if cls.requires_validator:
                marker = f" {YELLOW}[requires openai SDK]{RESET}"
            print(f"    {BOLD}{name}{RESET}{marker}")
            print(f"      {cls.description}")
            print(f"      {DIM}tags: {tags}{RESET}")
        print()

    _print_group("Fuzz Scenarios (adversarial)", fuzz_scenarios)
    _print_group("Validation Scenarios (correctness)", validation_scenarios)
    _print_group("Model-Specific Scenarios", model_scenarios)

    print(f"  {DIM}Total: {len(scenarios)} scenarios{RESET}")
    if model_scenarios:
        profiles = sorted(
            {cls.model_filter for _, cls in model_scenarios if cls.model_filter}
        )
        print(f"  {DIM}Model profiles: {', '.join(profiles)}{RESET}")
    print()


def parse_exclude(spec: str | None) -> tuple[set[str], set[str]]:
    """Split an --exclude value into whole-scenario names and scenario:test
    names (the tokens containing a ":")."""
    exclude_groups: set[str] = set()
    exclude_tests: set[str] = set()
    for token in (spec or "").split(","):
        token = token.strip()
        if not token:
            continue
        (exclude_tests if ":" in token else exclude_groups).add(token)
    return exclude_groups, exclude_tests


def select_scenarios(
    args: argparse.Namespace, exclude_groups: set[str]
) -> list[Any]:
    """Select which scenarios to run based on CLI args."""
    all_scenarios = get_all_scenarios()

    if args.scenarios:
        names = [n.strip() for n in args.scenarios.split(",")]
        selected = []
        for n in names:
            if n not in all_scenarios:
                print(f"{RED}Unknown scenario: {n}{RESET}")
                print(f"Available: {', '.join(sorted(all_scenarios.keys()))}")
                sys.exit(1)
            selected.append(all_scenarios[n])
        return selected

    selected = list(all_scenarios.values())

    if args.tags:
        tags = {t.strip() for t in args.tags.split(",")}
        selected = [cls for cls in selected if tags & set(cls.tags)]
        if not selected:
            print(f"{RED}No scenarios match tags: {args.tags}{RESET}")
            sys.exit(1)

    if exclude_groups:
        selected = [s for s in selected if s.name not in exclude_groups]

    # Filter by scenario type
    if args.fuzz_only:
        selected = [s for s in selected if s.scenario_type == "fuzz"]
    elif args.validation_only:
        selected = [s for s in selected if s.scenario_type == "validation"]

    # Filter model-specific scenarios: exclude unless --model-profile matches
    model_profile = args.model_profile
    if model_profile:
        selected = [
            s
            for s in selected
            if s.model_filter is None or s.model_filter == model_profile
        ]
    else:
        selected = [s for s in selected if s.model_filter is None]

    return selected


async def run(args: argparse.Namespace) -> int:
    model_config = build_model_config(
        args.model,
        no_hf_fetch=args.no_hf_fetch,
        max_context_length=args.max_context_length,
        max_num_tokens=args.max_num_tokens,
    )

    config = RunConfig(
        base_url=args.url,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
        max_concurrency=args.max_concurrency,
        endurance_duration_sec=args.endurance_duration * 60,
        endurance_intensity=args.endurance_intensity,
        model_config=model_config,
        verbose=args.verbose,
        image_stress_nodes=args.image_stress_nodes,
    )

    exclude_groups: set[str] = set()
    exclude_tests: set[str] = set()
    if args.exclude:
        exclude_groups, exclude_tests = parse_exclude(args.exclude)

    scenario_classes = select_scenarios(args, exclude_groups)

    # In endurance mode, only run scenarios tagged with "endurance"
    if args.endurance:
        scenario_classes = [
            s for s in scenario_classes if "endurance" in getattr(s, "tags", [])
        ]
        if not scenario_classes:
            print(
                f"{RED}No endurance scenarios found. Add scenarios with the 'endurance' tag.{RESET}"
            )
            return 1

    # Check if any validation scenarios need the openai SDK
    needs_validator = any(s.requires_validator for s in scenario_classes)
    validator = None
    if needs_validator:
        try:
            vc = make_validator_config(config)
            validator = ValidatorClient(vc)
            if not validator.is_available():
                print(
                    f"{YELLOW}Warning: Validator client cannot reach server. Validation scenarios may fail.{RESET}"
                )
        except RuntimeError as e:
            print(f"{YELLOW}Warning: {e}{RESET}")
            print(
                f"{YELLOW}Skipping scenarios that require the openai SDK.{RESET}"
            )
            scenario_classes = [
                s for s in scenario_classes if not s.requires_validator
            ]

    # Store validator and extra config for scenarios to access
    config.validator = validator
    config.k2vv_mode = args.k2vv_mode

    print_banner()

    # Initialize run log
    run_log: RunLog | None = None
    if not args.no_log:
        run_log = RunLog(log_file=args.log_file)

    try:
        return await _run_body(
            args, config, scenario_classes, run_log, validator, exclude_tests
        )
    finally:
        if run_log:
            run_log.close()
        if validator:
            validator.close()


async def _clear_prefix_cache(client: FuzzClient) -> None:
    """Best-effort clear of the server KV/prefix cache between repeats.

    Lets each ``--repeat`` run start from a clean cache so runs are independent:
    prefix caching otherwise makes a repeated identical request hit the cache
    and take a different path. Requires the server (or orchestrator) to expose
    ``POST /reset_prefix_cache`` with prefix caching enabled; otherwise this is
    a no-op with a note. Never fails the run.
    """
    try:
        resp = await client.post_to_path("/reset_prefix_cache", {})
    except Exception as exc:
        print(f"  {DIM}prefix-cache reset skipped ({exc}){RESET}")
        return
    if 200 <= resp.status < 300:
        print(f"  {DIM}cleared prefix cache before run{RESET}")
    else:
        print(
            f"  {DIM}prefix-cache reset not applied "
            f"(HTTP {resp.status or 'no response'}); continuing{RESET}"
        )


async def _run_body(
    args: argparse.Namespace,
    config: RunConfig,
    scenario_classes: list[Any],
    run_log: RunLog | None,
    validator: ValidatorClient | None,
    exclude_tests: set[str],
) -> int:
    print(f"  Target:       {BOLD}{config.base_url}{RESET}")
    print(f"  Model:        {config.model}")
    mc = config.model_config
    print(
        f"  Model Config: {DIM}source={mc.source}, context={mc.max_position_embeddings:,}, max_num_tokens={mc.max_num_tokens:,}{RESET}"
    )
    if args.model_profile:
        print(f"  Profile:      {BOLD}{args.model_profile}{RESET}")
    mode = "all"
    if args.fuzz_only:
        mode = "fuzz only"
    elif args.validation_only:
        mode = "validation only"
    elif args.endurance:
        mode = "endurance"
    print(f"  Mode:         {mode}")
    repeat = args.repeat
    if repeat < 1:
        print(f"{RED}--repeat must be >= 1, got {repeat}{RESET}")
        sys.exit(1)
    print(f"  Scenarios:    {len(scenario_classes)}")
    if repeat > 1:
        print(f"  Repeat:       {repeat}x (flakiness detection)")
    print(f"  Timeout:      {config.timeout}s")
    print(f"  Concurrency:  {config.max_concurrency}")
    if args.circuit_breaker > 0:
        print(f"  Circuit Brk:  {args.circuit_breaker} consecutive failures")
    print()

    if run_log:
        run_log.log_run_start(
            url=config.base_url,
            model=config.model,
            scenario_count=len(scenario_classes),
            timeout=config.timeout,
            max_concurrency=config.max_concurrency,
        )

    circuit_breaker = CircuitBreaker(threshold=args.circuit_breaker)

    async with FuzzClient(config) as client:
        # Initial health check
        if not args.no_health_check:
            print(
                f"  {DIM}Running initial health check...{RESET}",
                end=" ",
                flush=True,
            )
            health = await client.health_check()
            if health.status == 200:
                print(f"{GREEN}OK{RESET} ({health.elapsed_ms:.0f}ms)")
            elif health.status == 0:
                print(f"{RED}FAILED{RESET}")
                print(f"\n  {RED}Cannot reach endpoint: {health.error}{RESET}")
                print(
                    f"  {DIM}Check that the server is running at {config.base_url}{RESET}"
                )
                print(
                    f"  {DIM}Use --no-health-check to skip this check{RESET}\n"
                )
                sys.exit(1)
            else:
                print(f"{RED}Status {health.status}{RESET}")
                print(
                    f"  {DIM}Server responded but health check got {health.status}.{RESET}"
                )
                print(
                    f"  {DIM}Proceeding anyway (use --no-health-check to suppress)...{RESET}\n"
                )

        # Run scenarios (with optional repeat for flakiness detection)
        all_results: list[ScenarioResult] = []
        n_pass = n_fail = n_interesting = 0
        t_start = time.perf_counter()
        circuit_tripped = False
        total_scenarios = len(scenario_classes) * repeat
        scenario_idx = 0

        for run_idx in range(repeat):
            if circuit_tripped:
                break
            if repeat > 1:
                circuit_breaker.reset()
                print(
                    f"\n  {BOLD}{CYAN}--- Run {run_idx + 1}/{repeat} ---{RESET}\n"
                )
                # Clear the server KV/prefix cache between repeats so each run
                # is an independent sample (prefix caching otherwise carries
                # state across runs).
                if run_idx > 0:
                    await _clear_prefix_cache(client)

            for scenario_cls in scenario_classes:
                if circuit_breaker.tripped:
                    print(
                        f"\n  {RED}{BOLD}CIRCUIT BREAKER TRIPPED: {args.circuit_breaker} consecutive failures.{RESET}"
                    )
                    print(
                        f"  {DIM}Server appears to be down. Stopping execution.{RESET}\n"
                    )
                    circuit_tripped = True
                    break

                scenario_idx += 1
                pct = int(scenario_idx / total_scenarios * 100)
                elapsed = time.perf_counter() - t_start
                n_tests = len(all_results)
                print(
                    f"\n  {DIM}[{scenario_idx}/{total_scenarios}] ({pct}%) {scenario_cls.name}  |  "
                    f"elapsed: {_format_elapsed(elapsed)}  |  "
                    f"{n_tests} tests so far ({n_pass} pass, {n_fail} fail, {n_interesting} interesting){RESET}"
                )

                scenario = scenario_cls()
                if run_log:
                    run_log.log_scenario_start(scenario.name)
                scenario_t0 = time.perf_counter()

                if repeat == 1:
                    print_scenario_header(scenario.name, scenario.description)
                else:
                    print_scenario_header(
                        scenario.name,
                        f"[{run_idx + 1}/{repeat}] {scenario.description}",
                    )

                try:
                    scenario_results = await scenario.run(client, config)
                    if not args.no_health_check and not any(
                        "health_check" in r.test_name for r in scenario_results
                    ):
                        health = await client.health_check()
                        scenario_results.append(
                            scenario.make_result(
                                scenario.name,
                                f"post_{scenario.name}_health_check",
                                Verdict.PASS
                                if health.status == 200
                                else Verdict.FAIL,
                                status_code=health.status,
                                detail="Server still healthy"
                                if health.status == 200
                                else f"Server unhealthy after scenario: {health.error}",
                                error=health.error or "",
                            )
                        )
                    if exclude_tests:
                        scenario_results = [
                            r
                            for r in scenario_results
                            if f"{r.scenario_name}:{r.test_name}"
                            not in exclude_tests
                        ]
                    all_results.extend(scenario_results)
                    s_pass = s_fail = 0
                    for r in scenario_results:
                        if r.verdict == Verdict.PASS:
                            n_pass += 1
                            s_pass += 1
                        elif r.verdict == Verdict.FAIL:
                            n_fail += 1
                            s_fail += 1
                        elif r.verdict == Verdict.INTERESTING:
                            n_interesting += 1
                        print_result(r, verbose=args.verbose)
                        if run_log:
                            run_log.log_test_result(r)
                        circuit_breaker.record(r)
                    if run_log:
                        scenario_elapsed_ms = (
                            time.perf_counter() - scenario_t0
                        ) * 1000
                        run_log.log_scenario_end(
                            scenario.name, scenario_elapsed_ms, s_pass, s_fail
                        )
                except Exception as e:
                    import traceback

                    print(f"  {RED}SCENARIO CRASHED: {e}{RESET}")
                    traceback.print_exc()
                    from scenarios import ScenarioResult

                    err_result = ScenarioResult(
                        scenario_name=scenario.name,
                        test_name="__scenario_exception__",
                        verdict=Verdict.ERROR,
                        error=str(e),
                    )
                    all_results.append(err_result)
                    if run_log:
                        run_log.log_test_result(err_result)
                        scenario_elapsed_ms = (
                            time.perf_counter() - scenario_t0
                        ) * 1000
                        run_log.log_scenario_end(
                            scenario.name, scenario_elapsed_ms, 0, 0
                        )
                    circuit_breaker.record(err_result)

        elapsed_total = time.perf_counter() - t_start

        # Summary
        print_summary(all_results, elapsed_total)

        if run_log:
            n_errors = sum(1 for r in all_results if r.verdict == Verdict.ERROR)
            run_log.log_run_end(
                {
                    "total_tests": len(all_results),
                    "pass_count": n_pass,
                    "fail_count": n_fail,
                    "interesting_count": n_interesting,
                    "error_count": n_errors,
                    "elapsed_sec": round(elapsed_total, 1),
                }
            )

        if args.compare:
            compare_with_baseline(all_results, args.compare)

        # Final timing summary
        tps = len(all_results) / elapsed_total if elapsed_total > 0 else 0
        print(
            f"  {DIM}Total elapsed: {_format_elapsed(elapsed_total)}  |  {tps:.1f} tests/sec{RESET}\n"
        )

        if circuit_tripped:
            print(
                f"  {RED}{BOLD}NOTE: Execution was stopped early by circuit breaker.{RESET}\n"
            )

        # Flakiness report when --repeat > 1
        if repeat > 1:
            from collections import Counter

            test_verdicts: dict[str, Counter[Verdict]] = {}
            for r in all_results:
                key = f"{r.scenario_name}:{r.test_name}"
                test_verdicts.setdefault(key, Counter())[r.verdict] += 1
            flaky = {
                key: counts
                for key, counts in test_verdicts.items()
                if len(counts) > 1 and Verdict.PASS in counts
            }
            if flaky:
                print(f"\n  {BOLD}{RED}FLAKY TESTS ({len(flaky)}):{RESET}")
                for key, counts in sorted(flaky.items()):
                    parts = [f"{v.value}={c}" for v, c in counts.most_common()]
                    rate = (
                        counts.get(Verdict.PASS, 0) / sum(counts.values()) * 100
                    )
                    print(
                        f"    {key}: {', '.join(parts)} ({rate:.0f}% pass rate)"
                    )
            else:
                print(
                    f"\n  {GREEN}No flaky tests detected across {repeat} runs.{RESET}"
                )

        # Export
        if args.export_json:
            export_json(
                all_results,
                args.export_json,
                url=config.base_url,
                model=config.model,
                elapsed_sec=elapsed_total,
            )
        if args.export_csv:
            export_csv(all_results, args.export_csv)

    if run_log:
        print(f"  {DIM}Run log: {run_log.path}{RESET}")

    # Exit code: 1 if any FAILs
    return 1 if n_fail > 0 else 0


def main() -> None:
    args = parse_args()

    if args.list:
        list_scenarios()
        return

    exit_code = asyncio.run(run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
