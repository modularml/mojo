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
"""Run the llm-fuzz suite against a managed MAX Serve instance.

Mirrors the structure of pipelines_lm_eval.py but invokes the in-tree
``llm-fuzz`` bazel binary, then parses the structured JSONL log to
produce a Slack payload and a GitHub step-summary chunk.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import click
import requests
from pipelines_lm_eval import PipelineSitter

# Make python.runfiles optional for Docker environments.
try:
    import python.runfiles

    HAVE_RUNFILES = True
except ImportError:
    HAVE_RUNFILES = False

logger = logging.getLogger("pipelines_llm_fuzz")

# Status buckets emitted by llm-fuzz. Order is the canonical reporting order.
_STATUSES = ("PASS", "FAIL", "INTERESTING", "ERROR")
_STATUS_EMOJI = {
    "PASS": ":white_check_mark:",
    "FAIL": ":x:",
    "INTERESTING": ":grey_question:",
    "ERROR": ":bangbang:",
}

# Map status -> per-status field names that appear in run_end events.
_SUMMARY_KEYS: dict[str, tuple[str, ...]] = {
    "PASS": ("pass_count", "PASS"),
    "FAIL": ("fail_count", "FAIL"),
    "INTERESTING": ("interesting_count", "INTERESTING"),
    "ERROR": ("error_count", "ERROR"),
}

# Crash-detection loop tunables. Wake up every poll interval to check
# that MAX Serve is still alive and answering /health; trip after this
# many consecutive failures.
_HEALTH_POLL_INTERVAL_S = 60.0
_HEALTH_PROBE_TIMEOUT_S = 60.0
_HEALTH_FAIL_THRESHOLD = 5


@dataclass(frozen=True)
class TestResult:
    """A single non-PASS test outcome lifted out of llm-fuzz's JSONL log."""

    name: str
    detail: str

    @classmethod
    def from_event(cls, event: Mapping[str, object]) -> TestResult:
        raw_name = event.get("test_name") or event.get("name") or "(unnamed)"
        raw_detail = (
            event.get("message")
            or event.get("error")
            or event.get("detail")
            or ""
        )
        name = raw_name if isinstance(raw_name, str) else "(unnamed)"
        detail = raw_detail if isinstance(raw_detail, str) else ""
        if len(detail) > 240:
            detail = detail[:237] + "..."
        return cls(name=name, detail=detail)


@dataclass
class FuzzSummary:
    """Aggregate parsed from llm-fuzz's structured JSONL log."""

    counts: dict[str, int] = field(
        default_factory=lambda: {s: 0 for s in _STATUSES}
    )
    by_scenario: Mapping[str, Mapping[str, list[TestResult]]] = field(
        default_factory=dict
    )
    total_duration: float | None = None
    total_tests: int = 0
    early_stop_reason: str | None = None
    log_present: bool = False


def _must_rlocation_str(runfiles: python.runfiles.Runfiles, rloc: str) -> str:
    loc = runfiles.Rlocation(rloc)
    if loc is None:
        raise FileNotFoundError(
            f"Required rlocation {rloc!r} could not be resolved"
        )
    return loc


def _resolve_pipelines_program() -> list[str]:
    """Resolve the MAX Serve binary path (Bazel runfiles or Docker fallback).

    The same Python entrypoint serves both //max/tests/integration/accuracy:pipelines-llm-fuzz
    (public, OSS pipelines binary) and //max_private:pipelines-llm-fuzz
    (internal binary that ships extra reasoning/tool parsers). Each Bazel
    target pins exactly one of the two paths via its ``data`` dep, so we
    pick whichever the runfiles tree actually contains.
    """
    if HAVE_RUNFILES:
        runfiles = python.runfiles.Create()
        if runfiles is not None:
            for rloc in (
                "_main/max_private/max_private",
                "_main/max/python/max/_entrypoints/pipelines",
            ):
                loc = runfiles.Rlocation(rloc)
                if loc is not None and Path(loc).exists():
                    return [loc]
            raise FileNotFoundError(
                "Neither max_private nor pipelines binary found in runfiles;"
                " invoke via //max_private:pipelines-llm-fuzz or"
                " //max/tests/integration/accuracy:pipelines-llm-fuzz"
            )
    return ["/opt/venv/bin/python", "-m", "max._entrypoints.pipelines"]


def _resolve_fuzz_program() -> str:
    """Resolve the llm-fuzz binary path via Bazel runfiles."""
    if HAVE_RUNFILES:
        runfiles = python.runfiles.Create()
        if runfiles is not None:
            return _must_rlocation_str(
                runfiles, "_main/max/tests/integration/accuracy/llm-fuzz"
            )
    raise RuntimeError(
        "llm-fuzz binary requires Bazel runfiles; run via"
        " //max/tests/integration/accuracy:pipelines-llm-fuzz"
    )


def _parse_run_end_event(
    event: Mapping[str, object], summary: FuzzSummary
) -> int | None:
    """Apply a run_end event's fields to ``summary`` in place.

    Returns the ``total_tests`` field if present, so the caller can
    prefer the tool's authoritative count over the per-test tally.
    """
    for status, keys in _SUMMARY_KEYS.items():
        for key in keys:
            val = event.get(key)
            if isinstance(val, int):
                summary.counts[status] = val
                break
    for dkey in ("elapsed_sec", "duration", "elapsed_seconds"):
        val = event.get(dkey)
        if isinstance(val, (int, float)):
            summary.total_duration = float(val)
            break
    stopped_early = event.get("stopped_early")
    if isinstance(stopped_early, str):
        summary.early_stop_reason = stopped_early
    for tkey in ("total_tests", "total"):
        val = event.get(tkey)
        if isinstance(val, int):
            return val
    return None


def _parse_test_event(
    event: Mapping[str, object],
    summary: FuzzSummary,
    by_scenario: defaultdict[str, defaultdict[str, list[TestResult]]],
) -> None:
    """Accumulate one per-test result into ``summary`` and ``by_scenario``."""
    verdict_raw = (
        event.get("verdict") or event.get("status") or event.get("result") or ""
    )
    verdict = verdict_raw.upper() if isinstance(verdict_raw, str) else ""
    if verdict not in summary.counts:
        return
    summary.counts[verdict] += 1
    if verdict in ("FAIL", "INTERESTING", "ERROR"):
        scenario_raw = (
            event.get("scenario_name")
            or event.get("scenario")
            or event.get("group")
            or event.get("category")
            or "uncategorized"
        )
        scenario = (
            scenario_raw if isinstance(scenario_raw, str) else "uncategorized"
        )
        by_scenario[scenario][verdict].append(TestResult.from_event(event))


def _summarize_jsonl(log_path: Path) -> FuzzSummary:
    """Parse llm-fuzz's structured JSONL log into an aggregate summary.

    Defensive against schema drift: any event whose ``verdict`` (or
    ``status``/``result``) field matches a known bucket counts; the
    ``run_end`` event's authoritative counts overwrite the per-test
    tally when present. Missing fields are treated as unknown rather
    than raising.
    """
    summary = FuzzSummary()
    if not log_path.exists():
        return summary
    summary.log_present = True

    by_scenario: defaultdict[str, defaultdict[str, list[TestResult]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    total_tests_field: int | None = None

    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line: %.120s", line)
                continue
            if not isinstance(event, dict):
                continue

            event_type_raw = event.get("event") or event.get("type") or ""
            event_type = (
                event_type_raw.lower()
                if isinstance(event_type_raw, str)
                else ""
            )

            if event_type in ("run_end", "summary", "final", "totals"):
                t = _parse_run_end_event(event, summary)
                if t is not None:
                    total_tests_field = t
            else:
                _parse_test_event(event, summary, by_scenario)

    summary.by_scenario = by_scenario
    summary.total_tests = (
        sum(summary.counts.values())
        if total_tests_field is None
        else total_tests_field
    )
    return summary


def _verdict_emoji(counts: dict[str, int]) -> str:
    """Return an emoji label for the run's overall verdict."""
    if counts.get("ERROR", 0) > 0:
        return ":rotating_light:"
    if counts.get("FAIL", 0) > 0:
        return ":warning:"
    if counts.get("INTERESTING", 0) > 0:
        return ":grey_question:"
    return ":white_check_mark:"


def _describe_failure(item: TestResult) -> str:
    if item.detail:
        return f"`{item.name}` — {item.detail}"
    return f"`{item.name}`"


def _emit_step_summary(
    summary: FuzzSummary,
    *,
    pipeline: str,
    model_path: str,
    output_path: Path,
) -> None:
    """Write a markdown step-summary chunk."""
    counts = summary.counts
    lines = [f"## llm-fuzz: `{pipeline}`", ""]
    lines.append(f"- **Model path:** `{model_path}`")
    if summary.total_duration is not None:
        lines.append(f"- **Duration:** {summary.total_duration:.1f}s")
    if summary.early_stop_reason:
        lines.append(f"- **Stopped early:** {summary.early_stop_reason}")
    if not summary.log_present:
        lines.append("- **Note:** llm-fuzz log was not produced.")
    lines.extend(["", "| Status | Count |", "|---|---|"])
    lines.append(f"| Total | {summary.total_tests} |")
    for status in _STATUSES:
        lines.append(f"| {status} | {counts[status]} |")

    failures = summary.by_scenario
    if failures:
        lines.extend(["", "### Failures by scenario", ""])
        for scenario, by_status in sorted(failures.items()):
            for status, items in by_status.items():
                if status not in ("FAIL", "ERROR"):
                    continue
                lines.append(f"- **{scenario} ({status}):**")
                for item in items[:10]:
                    lines.append(f"  - {_describe_failure(item)}")
                if len(items) > 10:
                    lines.append(f"  - ... and {len(items) - 10} more")

        interesting_present = any(
            "INTERESTING" in by_status for by_status in failures.values()
        )
        if interesting_present:
            lines.extend(["", "### Worth investigating", ""])
            for scenario, by_status in sorted(failures.items()):
                items = by_status.get("INTERESTING", [])
                if not items:
                    continue
                lines.append(f"- **{scenario}:**")
                for item in items[:10]:
                    lines.append(f"  - {_describe_failure(item)}")
                if len(items) > 10:
                    lines.append(f"  - ... and {len(items) - 10} more")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _emit_slack_blocks(
    summary: FuzzSummary,
    *,
    pipeline: str,
    model_path: str,
    output_path: Path,
) -> None:
    """Write the Slack blocks JSON consumed by push-status-to-slack-channel.

    The composite action appends this list to its own header/status blocks,
    so we emit a JSON array (not a wrapping object).
    """
    counts = summary.counts
    emoji = _verdict_emoji(counts)
    parts = [f"{emoji} *llm-fuzz: `{pipeline}`*"]
    parts.append(f"*Model path:* `{model_path}`")
    counts_line = " | ".join(
        f"{_STATUS_EMOJI[s]} {s}: {counts[s]}" for s in _STATUSES
    )
    parts.append(f"*Total:* {summary.total_tests}    {counts_line}")
    if summary.total_duration is not None:
        parts.append(f"*Duration:* {summary.total_duration:.1f}s")
    if summary.early_stop_reason:
        parts.append(f"*Stopped early:* {summary.early_stop_reason}")
    if not summary.log_present:
        parts.append(":warning: llm-fuzz log was not produced.")

    blocks: list[object] = [
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(parts)},
        },
    ]

    # Append failure groupings if any.
    fail_lines: list[str] = []
    for scenario, by_status in sorted(summary.by_scenario.items()):
        for status in ("FAIL", "ERROR"):
            items = by_status.get(status, [])
            if not items:
                continue
            fail_lines.append(f"*{scenario}* ({status}, {len(items)}):")
            for item in items[:5]:
                fail_lines.append(f"  • {_describe_failure(item)}")
            if len(items) > 5:
                fail_lines.append(f"  • _... and {len(items) - 5} more_")
    if fail_lines:
        # Slack section text caps at 3000 chars; truncate defensively.
        text = "\n".join(fail_lines)
        if len(text) > 2900:
            text = text[:2897] + "..."
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": text},
            }
        )

    # Trailing newline lets the workflow inline this file inside a
    # `<<EOF` heredoc without a defensive `echo` after the cat.
    output_path.write_text(
        json.dumps(blocks, indent=2) + "\n", encoding="utf-8"
    )


def _emit_summary_json(
    summary: FuzzSummary,
    *,
    output_path: Path,
) -> None:
    """Write machine-readable summary.json consumed by the CI monitor.

    Skipped when the fuzz log was never produced (server crashed before fuzz
    ran), so the monitor can distinguish a crash from a zero-failure run.
    """
    if not summary.log_present:
        return
    failures: list[dict[str, str]] = [
        {
            "category": scenario,
            "name": item.name,
            "reason": item.detail,
            "kind": status,
        }
        for scenario, by_status in sorted(summary.by_scenario.items())
        for status, items in by_status.items()
        if status in ("FAIL", "INTERESTING", "ERROR")
        for item in items
    ]
    data: dict[str, object] = {
        "total": summary.total_tests,
        "duration": summary.total_duration or 0.0,
        "pass": summary.counts.get("PASS", 0),
        "fail": summary.counts.get("FAIL", 0),
        "interesting": summary.counts.get("INTERESTING", 0),
        "error": summary.counts.get("ERROR", 0),
        "failures": failures,
    }
    output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _run_fuzz_with_crash_detection(
    cmd: list[str],
    *,
    sitter: PipelineSitter,
    health_probe_url: str | None,
) -> tuple[int, bool]:
    """Run llm-fuzz, polling serve health and killing on crash/hang.

    Returns ``(exit_code, mechanical_failure)``. ``mechanical_failure``
    is True only when we had to kill the fuzz subprocess because the
    serve side died or hung — that is, the run did not complete by the
    tool's own choice. When False, the tool exited on its own and the
    exit code is its verdict (1 means "found failures", 0 means "all
    clean").
    """
    consecutive_health_failures = 0
    proc = subprocess.Popen(cmd)
    try:
        while True:
            # Wake on subprocess exit or after the poll interval —
            # whichever comes first — so a finished run is detected
            # immediately rather than after a full sleep.
            try:
                rc = proc.wait(timeout=_HEALTH_POLL_INTERVAL_S)
            except subprocess.TimeoutExpired:
                pass
            else:
                logger.info("llm-fuzz exited with status code %s", rc)
                return (rc, False)
            if not sitter.is_alive():
                logger.error(
                    "MAX Serve died while llm-fuzz was running; terminating"
                )
                return (1, True)
            if health_probe_url is not None:
                try:
                    resp = requests.get(
                        health_probe_url, timeout=_HEALTH_PROBE_TIMEOUT_S
                    )
                    resp.raise_for_status()
                    consecutive_health_failures = 0
                except Exception:
                    consecutive_health_failures += 1
                    logger.warning(
                        "Health probe failed (%d/%d): %s",
                        consecutive_health_failures,
                        _HEALTH_FAIL_THRESHOLD,
                        health_probe_url,
                    )
                    if consecutive_health_failures >= _HEALTH_FAIL_THRESHOLD:
                        logger.error(
                            "Health probe failed %d times consecutively;"
                            " server appears hung",
                            consecutive_health_failures,
                        )
                        return (1, True)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("llm-fuzz did not terminate, sending SIGKILL")
                proc.kill()
                proc.wait(timeout=5)


@click.command()
@click.option("--pipelines-arg", "pipelines_args", multiple=True)
@click.option("--pipelines-probe-port", type=int, default=8000)
@click.option("--pipelines-probe-timeout", type=float, default=5000.0)
@click.option(
    "--pipelines-health-timeout",
    type=int,
    default=120,
    help=(
        "Heartbeat timeout in seconds for detecting model-worker hangs."
        " Set to 0 to disable."
    ),
)
@click.option(
    "--model",
    "model_path",
    type=str,
    help=(
        "Model identifier passed to llm-fuzz --model (HF repo or local"
        " path). Required unless --emit-only is set."
    ),
)
@click.option("--model-profile", type=str)
@click.option("--scenarios", type=str, default="")
@click.option(
    "--exclude",
    type=str,
    default="",
    help=(
        "Comma-separated scenarios to exclude, forwarded to llm-fuzz"
        " --exclude. Use scenario:test for a single test."
    ),
)
@click.option("--k2vv-mode", type=str, default="")
@click.option("--circuit-breaker", type=int, default=None)
@click.option("--extra-fuzz-arg", "extra_fuzz_args", multiple=True)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("./fuzz-output"),
)
@click.option(
    "--emit-only",
    is_flag=True,
    help=(
        "Skip serve and fuzz subprocess; regenerate step-summary.md and"
        " slack-blocks.json from an existing run.jsonl in --output-dir."
        " Used for local dry-runs of the summarizer against captured logs."
    ),
)
@click.option("--pipeline-name", type=str, default="")
def main(
    pipelines_args: Sequence[str],
    pipelines_probe_port: int,
    pipelines_probe_timeout: float,
    pipelines_health_timeout: int,
    model_path: str | None,
    model_profile: str | None,
    scenarios: str,
    exclude: str,
    k2vv_mode: str,
    circuit_breaker: int | None,
    extra_fuzz_args: Sequence[str],
    output_dir: Path,
    emit_only: bool,
    pipeline_name: str,
) -> None:
    """Manage MAX Serve, run llm-fuzz, summarize, and exit with the fuzz rc."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(name)s: %(message)s",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.jsonl"

    if emit_only:
        summary = _summarize_jsonl(log_path)
        _emit_step_summary(
            summary,
            pipeline=pipeline_name or "(unspecified)",
            model_path=model_path or "(unspecified)",
            output_path=output_dir / "step-summary.md",
        )
        _emit_slack_blocks(
            summary,
            pipeline=pipeline_name or "(unspecified)",
            model_path=model_path or "(unspecified)",
            output_path=output_dir / "slack-blocks.json",
        )
        sys.exit(0)

    if model_path is None or model_profile is None:
        logger.error(
            "--model and --model-profile are required unless --emit-only is set"
        )
        sys.exit(2)

    fuzz_program = _resolve_fuzz_program()
    fuzz_cmd = [
        fuzz_program,
        "--url",
        f"http://localhost:{pipelines_probe_port}",
        "--model",
        model_path,
        "--model-profile",
        model_profile,
        "--log-file",
        str(log_path.resolve()),
    ]
    if scenarios:
        fuzz_cmd.extend(["--scenarios", scenarios])
    if exclude:
        fuzz_cmd.extend(["--exclude", exclude])
    if k2vv_mode:
        fuzz_cmd.extend(["--k2vv-mode", k2vv_mode])
    if circuit_breaker is not None:
        fuzz_cmd.extend(["--circuit-breaker", str(circuit_breaker)])
    fuzz_cmd.extend(extra_fuzz_args)

    pipelines_program = _resolve_pipelines_program()
    pipelines_env: dict[str, str] = {}
    if pipelines_health_timeout > 0:
        pipelines_env = {
            "MAX_SERVE_USE_HEARTBEAT": "true",
            "MAX_SERVE_MW_HEALTH_FAIL": str(pipelines_health_timeout),
        }

    fuzz_rc = 1
    mechanical_failure = True
    try:
        with PipelineSitter(
            pipelines_program + list(pipelines_args),
            extra_env=pipelines_env,
        ) as sitter:
            sitter.wait_for_alive(
                probe_port=pipelines_probe_port,
                timeout=pipelines_probe_timeout,
            )
            health_url = (
                f"http://127.0.0.1:{pipelines_probe_port}/health"
                if pipelines_health_timeout > 0
                else None
            )
            logger.info("Running llm-fuzz: %s", fuzz_cmd)
            fuzz_rc, mechanical_failure = _run_fuzz_with_crash_detection(
                fuzz_cmd,
                sitter=sitter,
                health_probe_url=health_url,
            )
    finally:
        # Always summarize whatever JSONL we have, even on failure.
        summary = _summarize_jsonl(log_path)
        _emit_step_summary(
            summary,
            pipeline=pipeline_name or "(unspecified)",
            model_path=model_path,
            output_path=output_dir / "step-summary.md",
        )
        _emit_slack_blocks(
            summary,
            pipeline=pipeline_name or "(unspecified)",
            model_path=model_path,
            output_path=output_dir / "slack-blocks.json",
        )
        _emit_summary_json(
            summary,
            output_path=output_dir / "summary.json",
        )

    # Propagate fuzz exit code so the job conclusion reflects failures.
    # Mechanical failures (serve crashed/hung) always exit non-zero.
    sys.exit(fuzz_rc if not mechanical_failure else (fuzz_rc or 1))


if __name__ == "__main__":
    main()
