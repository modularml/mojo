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
"""Validation: advanced tool calling correctness.

Tests streaming argument accumulation, SO/TC mode switching, concurrent
tool calling, edge-case tool definitions (empty description, special chars),
tool_choice=none enforcement, and sequential soak for memory leak detection.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from helpers import (
    budget_exhausted,
    make_tool,
    stream_budget_exhausted,
)

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig


@register_scenario
class TCAdvanced(BaseScenario):
    name = "tc_advanced"
    description = "Advanced tool calling: streaming accumulation, SO/TC switching, concurrency, edge cases, soak"
    tags = ["validation", "tool_calling", "advanced"]
    requires_validator = True
    scenario_type = "validation"

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        v = config.validator
        if not v:
            results.append(
                self.make_result(
                    self.name,
                    "setup",
                    Verdict.ERROR,
                    detail="No validator client available",
                )
            )
            return results
        loop = asyncio.get_running_loop()

        weather_tool = make_tool(
            "get_weather",
            {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        )

        # ------------------------------------------------------------------ 1
        # streaming_args_accumulation
        # ------------------------------------------------------------------
        try:
            tool_x = make_tool(
                "tool_x",
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                    "required": ["name", "count"],
                    "additionalProperties": False,
                },
            )
            result = await loop.run_in_executor(
                None,
                lambda: v.tc_chat_stream(
                    [
                        {
                            "role": "user",
                            "content": "Call tool_x with name='streaming_test' and count=42.",
                        }
                    ],
                    [tool_x],
                    tool_choice="required",
                    max_tokens=1024,
                ),
            )
            if stream_budget_exhausted(result):
                results.append(
                    self.make_result(
                        self.name,
                        "streaming_args_accumulation",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                errors: list[str] = []
                if not result["tool_calls"]:
                    errors.append("No tool calls in stream")
                else:
                    tc = result["tool_calls"][0]
                    try:
                        args = json.loads(tc["arguments"])
                        if "name" not in args:
                            errors.append("Missing 'name' in accumulated args")
                        if "count" not in args:
                            errors.append("Missing 'count' in accumulated args")
                    except json.JSONDecodeError as e:
                        errors.append(f"Accumulated args not valid JSON: {e}")
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "streaming_args_accumulation",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "streaming_args_accumulation",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        # ------------------------------------------------------------------ 2
        # so_tc_mode_switch
        # ------------------------------------------------------------------
        try:
            so_schema = {
                "type": "object",
                "properties": {"v": {"type": "integer"}},
                "required": ["v"],
                "additionalProperties": False,
            }
            ping_tool = make_tool(
                "ping",
                {
                    "type": "object",
                    "properties": {"host": {"type": "string"}},
                    "required": ["host"],
                    "additionalProperties": False,
                },
            )
            errors = []
            for i in range(5):
                if i % 2 == 0:
                    # Structured output request
                    def _so_call(idx: int = i) -> Any:
                        return v.so_chat(
                            [
                                {
                                    "role": "user",
                                    "content": f"Return v={idx}.",
                                }
                            ],
                            so_schema,
                            max_tokens=256,
                        )

                    resp = await loop.run_in_executor(None, _so_call)
                    if not budget_exhausted(resp):
                        try:
                            data = json.loads(resp.choices[0].message.content)
                            if "v" not in data:
                                errors.append(
                                    f"SO round {i}: missing 'v' in response"
                                )
                        except (json.JSONDecodeError, TypeError) as e:
                            errors.append(f"SO round {i}: {e}")
                else:
                    # Tool calling request
                    def _tc_call(idx: int = i) -> Any:
                        return v.tc_chat(
                            [
                                {
                                    "role": "user",
                                    "content": f"Ping host_{idx}.",
                                }
                            ],
                            [ping_tool],
                            tool_choice="required",
                            max_tokens=256,
                        )

                    resp = await loop.run_in_executor(None, _tc_call)
                    if not budget_exhausted(resp):
                        tcs = resp.choices[0].message.tool_calls
                        if not tcs:
                            errors.append(f"TC round {i}: no tool calls")
                        else:
                            try:
                                json.loads(tcs[0].function.arguments)
                            except json.JSONDecodeError as e:
                                errors.append(
                                    f"TC round {i}: invalid args JSON: {e}"
                                )
                            # Check for SO leaking into TC (content should be None/empty)
                            content = resp.choices[0].message.content
                            if content and content.strip().startswith("{"):
                                errors.append(
                                    f"TC round {i}: content looks like leaked SO JSON"
                                )
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            results.append(
                self.make_result(
                    self.name,
                    "so_tc_mode_switch",
                    verdict,
                    detail="; ".join(errors) or "5 alternations OK",
                )
            )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "so_tc_mode_switch", Verdict.ERROR, error=str(e)
                )
            )

        # ------------------------------------------------------------------ 3
        # concurrent_tc_4_workers
        # ------------------------------------------------------------------
        try:
            tools = [
                make_tool(
                    "search",
                    {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                        "additionalProperties": False,
                    },
                ),
                make_tool(
                    "calc",
                    {
                        "type": "object",
                        "properties": {"expr": {"type": "string"}},
                        "required": ["expr"],
                        "additionalProperties": False,
                    },
                ),
            ]

            def _run_tc(i: int) -> tuple[int, str | None]:
                """Returns (index, error_or_none)."""
                try:
                    resp = v.tc_chat(
                        [
                            {
                                "role": "user",
                                "content": f"Request {i}: search 'test{i}' and calculate '{i}+1'.",
                            }
                        ],
                        tools,
                        tool_choice="required",
                        max_tokens=1024,
                    )
                    if budget_exhausted(resp):
                        return (i, None)  # skip, not an error
                    tcs = resp.choices[0].message.tool_calls
                    if not tcs:
                        return (i, f"worker {i}: no tool calls")
                    for tc in tcs:
                        json.loads(tc.function.arguments)
                    return (i, None)
                except json.JSONDecodeError as e:
                    return (i, f"worker {i}: invalid JSON args: {e}")
                except Exception as e:
                    return (i, f"worker {i}: {e}")

            concurrent_results = await loop.run_in_executor(
                None,
                lambda: list(_concurrent_map(_run_tc, range(4), max_workers=4)),
            )
            errors = [err for _, err in concurrent_results if err]
            verdict = Verdict.PASS if not errors else Verdict.FAIL
            results.append(
                self.make_result(
                    self.name,
                    "concurrent_tc_4_workers",
                    verdict,
                    detail="; ".join(errors) or "4 concurrent OK",
                )
            )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "concurrent_tc_4_workers",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        # ------------------------------------------------------------------ 4
        # empty_tool_description
        # ------------------------------------------------------------------
        try:
            empty_desc_tool = {
                "type": "function",
                "function": {
                    "name": "empty_desc",
                    "description": "",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                        "required": ["x"],
                        "additionalProperties": False,
                    },
                },
            }
            resp = await loop.run_in_executor(
                None,
                lambda: v.tc_chat(
                    [
                        {
                            "role": "user",
                            "content": "Call the empty_desc tool with x='test'.",
                        }
                    ],
                    [empty_desc_tool],
                    tool_choice="required",
                    max_tokens=1024,
                ),
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "empty_tool_description",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                tcs = resp.choices[0].message.tool_calls
                errors = []
                if not tcs:
                    errors.append("No tool call with empty description")
                else:
                    try:
                        json.loads(tcs[0].function.arguments)
                    except json.JSONDecodeError as e:
                        errors.append(f"Invalid JSON args: {e}")
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "empty_tool_description",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "empty_tool_description",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        # ------------------------------------------------------------------ 5
        # special_chars_in_tool_name
        # ------------------------------------------------------------------
        try:
            special_tool = make_tool(
                "fetch_data_v2_final",
                {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                    "additionalProperties": False,
                },
            )
            resp = await loop.run_in_executor(
                None,
                lambda: v.tc_chat(
                    [
                        {
                            "role": "user",
                            "content": "Call fetch_data_v2_final with url='http://example.com'.",
                        }
                    ],
                    [special_tool],
                    tool_choice="required",
                    max_tokens=1024,
                ),
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "special_chars_in_tool_name",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                tcs = resp.choices[0].message.tool_calls
                errors = []
                if not tcs:
                    errors.append("No tool call for underscore/number name")
                elif tcs[0].function.name != "fetch_data_v2_final":
                    errors.append(
                        f"Wrong name: expected fetch_data_v2_final, got {tcs[0].function.name}"
                    )
                else:
                    try:
                        json.loads(tcs[0].function.arguments)
                    except json.JSONDecodeError as e:
                        errors.append(f"Invalid JSON args: {e}")
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "special_chars_in_tool_name",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "special_chars_in_tool_name",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        # ------------------------------------------------------------------ 6
        # tool_choice_none
        # ------------------------------------------------------------------
        try:
            resp = await loop.run_in_executor(
                None,
                lambda: v.tc_chat(
                    [
                        {
                            "role": "user",
                            "content": "What is the weather in Paris?",
                        }
                    ],
                    [weather_tool],
                    tool_choice="none",
                    max_tokens=1024,
                ),
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "tool_choice_none",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                tcs = resp.choices[0].message.tool_calls
                if tcs:
                    results.append(
                        self.make_result(
                            self.name,
                            "tool_choice_none",
                            Verdict.FAIL,
                            detail=f"tool_choice=none but got {len(tcs)} tool call(s)",
                        )
                    )
                else:
                    content = resp.choices[0].message.content
                    if not content or not content.strip():
                        results.append(
                            self.make_result(
                                self.name,
                                "tool_choice_none",
                                Verdict.INTERESTING,
                                detail="No tool calls (correct) but also no text content",
                            )
                        )
                    else:
                        results.append(
                            self.make_result(
                                self.name,
                                "tool_choice_none",
                                Verdict.PASS,
                                detail="No tool calls, text response OK",
                            )
                        )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "tool_choice_none", Verdict.ERROR, error=str(e)
                )
            )

        # ------------------------------------------------------------------ 7
        # sequential_soak_20
        # ------------------------------------------------------------------
        try:
            soak_tools = [
                make_tool(
                    f"soak_{i}",
                    {
                        "type": "object",
                        "properties": {f"v_{i}": {"type": "string"}},
                        "required": [f"v_{i}"],
                        "additionalProperties": False,
                    },
                )
                for i in range(5)
            ]
            errors = []
            success_count = 0
            for i in range(20):
                tool_idx = i % len(soak_tools)
                tool_name = f"soak_{tool_idx}"
                param_name = f"v_{tool_idx}"
                try:

                    def _tc_soak_call(
                        tn: str = tool_name,
                        pn: str = param_name,
                        idx: int = i,
                    ) -> Any:
                        return v.tc_chat(
                            [
                                {
                                    "role": "user",
                                    "content": f"Call {tn} with {pn}='iter_{idx}'.",
                                }
                            ],
                            soak_tools,
                            tool_choice="required",
                            max_tokens=512,
                        )

                    resp = await loop.run_in_executor(None, _tc_soak_call)
                    if budget_exhausted(resp):
                        continue
                    tcs = resp.choices[0].message.tool_calls
                    if not tcs:
                        errors.append(f"iter {i}: no tool calls")
                    else:
                        try:
                            json.loads(tcs[0].function.arguments)
                            success_count += 1
                        except json.JSONDecodeError as e:
                            errors.append(f"iter {i}: invalid JSON args: {e}")
                except Exception as e:
                    errors.append(f"iter {i}: {e}")

            if errors:
                verdict = Verdict.FAIL
                detail = (
                    f"{success_count}/20 OK; failures: {'; '.join(errors[:5])}"
                )
                if len(errors) > 5:
                    detail += f" ... and {len(errors) - 5} more"
            else:
                verdict = Verdict.PASS
                detail = f"{success_count}/20 sequential requests OK"
            results.append(
                self.make_result(
                    self.name, "sequential_soak_20", verdict, detail=detail
                )
            )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "sequential_soak_20", Verdict.ERROR, error=str(e)
                )
            )

        return results


def _concurrent_map(
    fn: Callable[[Any], Any],
    args_iter: Iterable[Any],
    *,
    max_workers: int = 4,
) -> list[Any]:
    """Run fn(arg) concurrently and return collected results."""
    collected = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn, arg): arg for arg in args_iter}
        for future in as_completed(futures):
            collected.append(future.result())
    return collected
