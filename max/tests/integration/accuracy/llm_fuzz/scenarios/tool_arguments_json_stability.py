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
Scenario: repeated forced tool calls — arguments must be valid JSON

Sends the same OpenAI-compatible chat completion many times with
tool_choice forcing a function. Each response with tool_calls must have
function.arguments that parse as JSON.

Motivated by: intermittent non-JSON strings in tool call arguments under
identical requests (temp 0).

Override run count: LLM_FUZZ_TOOL_ARGS_JSON_RUNS (default 32).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

from scenarios import (
    BaseScenario,
    ScenarioResult,
    Verdict,
    register_scenario,
)

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig


def _forced_search_payload(model: str) -> dict[str, Any]:
    """Same body on every request (flakiness detection)."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Use search tool only"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "other",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                    },
                },
            },
        ],
        "tool_choice": {"type": "function", "function": {"name": "search"}},
        # Reasoning models take a variable-length <think> block before the
        # forced tool call; 128 truncated the longer path mid-reasoning (no
        # tool_calls emitted). 512 lets it finish and reach the tool call.
        "max_tokens": 512,
        "temperature": 0,
    }


def _validate_tool_arguments_json(resp_body: str) -> tuple[bool, str]:
    """Return (ok, detail)."""
    try:
        data = json.loads(resp_body)
    except json.JSONDecodeError as e:
        return False, f"response body is not json: {e}"

    choices = data.get("choices")
    if not choices:
        return False, "no choices in response"
    msg = choices[0].get("message") or {}
    tool_calls = msg.get("tool_calls")
    if not tool_calls:
        return False, "no tool_calls in message"

    for i, tc in enumerate(tool_calls):
        fn = tc.get("function") or {}
        name = fn.get("name", "")
        args = fn.get("arguments")
        if args is None:
            return False, f"tool_calls[{i}] ({name!r}) missing arguments"
        if not isinstance(args, str):
            return (
                False,
                f"tool_calls[{i}] ({name!r}) arguments not a string: {type(args).__name__}",
            )
        if not args.strip():
            return False, f"tool_calls[{i}] ({name!r}) empty arguments"
        try:
            print(f"args: {args}")
            json.loads(args)
        except json.JSONDecodeError as e:
            snippet = args[:240] + ("…" if len(args) > 240 else "")
            return (
                False,
                f"tool_calls[{i}] ({name!r}) arguments not valid json: {e}; args={snippet!r}",
            )

    return True, ""


@register_scenario
class ToolArgumentsJsonStability(BaseScenario):
    name = "tool_arguments_json_stability"
    description = (
        "Repeated identical forced tool_choice requests; every tool "
        "call arguments string must be valid JSON"
    )
    tags = ["validation", "tools", "function_calling", "json", "flaky"]
    scenario_type = "validation"

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        model = config.model
        n_runs = int(os.environ.get("LLM_FUZZ_TOOL_ARGS_JSON_RUNS", "32"))
        if n_runs < 1:
            n_runs = 32

        payload = _forced_search_payload(model)
        payload_json = json.dumps(payload)

        log_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "logs"
        )
        os.makedirs(log_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        log_file = os.path.join(
            log_dir, f"tool_arguments_json_{timestamp_str}.jsonl"
        )

        ok_count = 0
        bad_json_count = 0
        api_errors = 0
        failure_samples: list[tuple[int, str, str]] = []

        with open(log_file, "w", encoding="utf-8") as logf:
            logf.write(
                json.dumps(
                    {
                        "event": "run_start",
                        "timestamp": datetime.now().isoformat(),
                        "runs": n_runs,
                        "model": model,
                        "payload": payload,
                    }
                )
                + "\n"
            )

            for i in range(n_runs):
                resp = await client.post_json(
                    payload, timeout=config.timeout * 4
                )
                run = i + 1

                if resp.error or resp.status != 200:
                    api_errors += 1
                    err = resp.error or f"HTTP {resp.status}"
                    logf.write(
                        json.dumps(
                            {
                                "event": "api_error",
                                "timestamp": datetime.now().isoformat(),
                                "run": run,
                                "error": err,
                                "status": resp.status,
                            }
                        )
                        + "\n"
                    )
                    failure_samples.append((run, "api_error", err))
                    continue

                ok, detail = _validate_tool_arguments_json(resp.body)
                if ok:
                    ok_count += 1
                else:
                    bad_json_count += 1
                    failure_samples.append((run, "invalid_tool_args", detail))
                    logf.write(
                        json.dumps(
                            {
                                "event": "invalid_tool_args",
                                "timestamp": datetime.now().isoformat(),
                                "run": run,
                                "detail": detail,
                                "response_excerpt": resp.body[:2000],
                            }
                        )
                        + "\n"
                    )

                if run % 10 == 0 or run == n_runs:
                    print(
                        f"    Progress: {run}/{n_runs} | "
                        f"ok={ok_count} bad_args={bad_json_count} api_errors={api_errors}"
                    )

            logf.write(
                json.dumps(
                    {
                        "event": "run_end",
                        "timestamp": datetime.now().isoformat(),
                        "ok": ok_count,
                        "bad_json": bad_json_count,
                        "api_errors": api_errors,
                    }
                )
                + "\n"
            )

        print(f"    Log: {log_file}")

        detail = (
            f"runs={n_runs} ok={ok_count} bad_tool_args_json={bad_json_count} "
            f"api_errors={api_errors}"
        )
        if failure_samples:
            lines = [
                f"  run {r}: {kind} — {d}" for r, kind, d in failure_samples[:8]
            ]
            if len(failure_samples) > 8:
                lines.append(
                    f"  … and {len(failure_samples) - 8} more (see log)"
                )
            detail += "\n" + "\n".join(lines)

        if api_errors > 0 or bad_json_count > 0:
            verdict = Verdict.FAIL
        else:
            verdict = Verdict.PASS

        results.append(
            self.make_result(
                self.name,
                f"forced_search_args_json_{n_runs}_runs",
                verdict,
                detail=detail,
                request_body=payload_json[:8000],
            )
        )

        health = await client.health_check()
        results.append(
            self.make_result(
                self.name,
                "post_tool_arguments_json_health_check",
                Verdict.PASS if health.status == 200 else Verdict.FAIL,
                status_code=health.status,
                detail="Server still healthy after scenario"
                if health.status == 200
                else (health.error or f"HTTP {health.status}"),
            )
        )

        return results
