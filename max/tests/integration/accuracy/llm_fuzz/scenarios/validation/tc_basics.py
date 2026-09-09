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
"""Validation: basic tool calling correctness.

Tests single and parallel tool calls, tool_choice modes, multi-turn
conversations with tool results, large tool results, many-tool selection,
and parameter edge cases (zero params, nested objects).
"""

from __future__ import annotations

import asyncio
import json
import random
from typing import TYPE_CHECKING

from helpers import (
    budget_exhausted,
    make_tool,
    stream_budget_exhausted,
)

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig


@register_scenario
class TCBasics(BaseScenario):
    name = "tc_basics"
    description = "Basic tool calling correctness: choice modes, parallel calls, multi-turn, many tools, parameter shapes"
    tags = ["validation", "tool_calling"]
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
        stock_tool = make_tool(
            "get_stock_price",
            {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
                "additionalProperties": False,
            },
        )

        # ------------------------------------------------------------------ 1
        # single_tool_auto
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
                    tool_choice="auto",
                    max_tokens=1024,
                ),
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "single_tool_auto",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                tcs = resp.choices[0].message.tool_calls
                errors: list[str] = []
                if not tcs:
                    errors.append("No tool calls returned")
                else:
                    try:
                        args = json.loads(tcs[0].function.arguments)
                        if not isinstance(args, dict):
                            errors.append(f"Args not dict: {type(args)}")
                    except json.JSONDecodeError as e:
                        errors.append(f"Invalid JSON args: {e}")
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "single_tool_auto",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "single_tool_auto", Verdict.ERROR, error=str(e)
                )
            )

        # ------------------------------------------------------------------ 2
        # single_tool_required
        # ------------------------------------------------------------------
        try:
            resp = await loop.run_in_executor(
                None,
                lambda: v.tc_chat(
                    [{"role": "user", "content": "Tell me a joke."}],
                    [weather_tool],
                    tool_choice="required",
                    max_tokens=1024,
                ),
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "single_tool_required",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                tcs = resp.choices[0].message.tool_calls
                if not tcs:
                    results.append(
                        self.make_result(
                            self.name,
                            "single_tool_required",
                            Verdict.FAIL,
                            detail="tool_choice=required but no tool calls",
                        )
                    )
                else:
                    try:
                        json.loads(tcs[0].function.arguments)
                        results.append(
                            self.make_result(
                                self.name, "single_tool_required", Verdict.PASS
                            )
                        )
                    except json.JSONDecodeError as e:
                        results.append(
                            self.make_result(
                                self.name,
                                "single_tool_required",
                                Verdict.FAIL,
                                detail=f"Invalid JSON args: {e}",
                            )
                        )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "single_tool_required",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        # ------------------------------------------------------------------ 3
        # single_tool_named
        # ------------------------------------------------------------------
        try:
            resp = await loop.run_in_executor(
                None,
                lambda: v.tc_chat(
                    [{"role": "user", "content": "Do something."}],
                    [weather_tool, stock_tool],
                    tool_choice={
                        "type": "function",
                        "function": {"name": "get_weather"},
                    },
                    max_tokens=1024,
                ),
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "single_tool_named",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                tcs = resp.choices[0].message.tool_calls
                errors = []
                if not tcs:
                    errors.append("No tool calls returned")
                elif tcs[0].function.name != "get_weather":
                    errors.append(
                        f"Expected get_weather, got {tcs[0].function.name}"
                    )
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "single_tool_named",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "single_tool_named", Verdict.ERROR, error=str(e)
                )
            )

        # ------------------------------------------------------------------ 4
        # parallel_tc_two_tools
        # ------------------------------------------------------------------
        try:
            resp = await loop.run_in_executor(
                None,
                lambda: v.tc_chat(
                    [
                        {
                            "role": "user",
                            "content": "What's the weather in Paris AND the stock price of AAPL?",
                        }
                    ],
                    [weather_tool, stock_tool],
                    tool_choice="auto",
                    max_tokens=1024,
                ),
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "parallel_tc_two_tools",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                tcs = resp.choices[0].message.tool_calls
                errors = []
                if not tcs or len(tcs) < 2:
                    errors.append(
                        f"Expected 2+ tool calls, got {len(tcs) if tcs else 0}"
                    )
                else:
                    ids = [tc.id for tc in tcs]
                    if len(ids) != len(set(ids)):
                        errors.append(f"Duplicate tool call IDs: {ids}")
                    names = {tc.function.name for tc in tcs}
                    if len(names) < 2:
                        errors.append(
                            f"Expected 2+ different tool names, got {names}"
                        )
                    for i, tc in enumerate(tcs):
                        try:
                            json.loads(tc.function.arguments)
                        except json.JSONDecodeError as e:
                            errors.append(f"TC[{i}] invalid JSON: {e}")
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "parallel_tc_two_tools",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "parallel_tc_two_tools",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        # ------------------------------------------------------------------ 5
        # parallel_tc_correct_indices
        # ------------------------------------------------------------------
        try:
            lookup_a = make_tool(
                "lookup_a",
                {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                    "additionalProperties": False,
                },
            )
            lookup_b = make_tool(
                "lookup_b",
                {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                    "additionalProperties": False,
                },
            )
            result = await loop.run_in_executor(
                None,
                lambda: v.tc_chat_stream(
                    [
                        {
                            "role": "user",
                            "content": "Look up 'alpha' with lookup_a AND 'beta' with lookup_b.",
                        }
                    ],
                    [lookup_a, lookup_b],
                    tool_choice="auto",
                    max_tokens=1024,
                ),
            )
            if stream_budget_exhausted(result):
                results.append(
                    self.make_result(
                        self.name,
                        "parallel_tc_correct_indices",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                errors = []
                if len(result["tool_calls"]) < 2:
                    errors.append(
                        f"Expected 2+ streaming TCs, got {len(result['tool_calls'])}"
                    )
                else:
                    indices = sorted(result["first_tc_chunks"].keys())
                    expected = list(range(len(indices)))
                    if indices != expected:
                        errors.append(
                            f"Non-sequential indices: {indices}, expected {expected}"
                        )
                    for i, tc in enumerate(result["tool_calls"]):
                        try:
                            json.loads(tc["arguments"])
                        except json.JSONDecodeError as e:
                            errors.append(f"TC[{i}] invalid JSON: {e}")
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "parallel_tc_correct_indices",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "parallel_tc_correct_indices",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        # ------------------------------------------------------------------ 6
        # multi_turn_with_result
        # ------------------------------------------------------------------
        try:
            messages = [
                {"role": "user", "content": "What's the weather in Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_001",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_001",
                    "content": '{"temperature": 20, "unit": "C", "condition": "sunny"}',
                },
                {"role": "user", "content": "Now check London."},
            ]
            resp = await loop.run_in_executor(
                None,
                lambda: v.tc_chat(
                    messages,
                    [weather_tool],
                    tool_choice="required",
                    max_tokens=1024,
                ),
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "multi_turn_with_result",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                tcs = resp.choices[0].message.tool_calls
                errors = []
                if not tcs:
                    errors.append("No tool call after providing tool result")
                else:
                    try:
                        args = json.loads(tcs[0].function.arguments)
                        if "city" not in args:
                            errors.append(f"Missing 'city' in args: {args}")
                    except json.JSONDecodeError as e:
                        errors.append(f"Invalid JSON args: {e}")
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "multi_turn_with_result",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "multi_turn_with_result",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        # ------------------------------------------------------------------ 7
        # large_tool_result_8k
        # ------------------------------------------------------------------
        try:
            search_tool = make_tool(
                "search",
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            )
            big_result = json.dumps(
                {
                    "results": [
                        {
                            "title": f"Result {i}",
                            "content": "x" * 200,
                            "score": round(random.random(), 4),
                        }
                        for i in range(30)
                    ],
                }
            )  # ~8K chars
            messages = [
                {"role": "user", "content": "Search for machine learning."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_big",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": '{"query": "machine learning"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_big",
                    "content": big_result,
                },
                {"role": "user", "content": "Now search for deep learning."},
            ]
            resp = await loop.run_in_executor(
                None,
                lambda: v.tc_chat(
                    messages,
                    [search_tool],
                    tool_choice="required",
                    max_tokens=512,
                ),
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "large_tool_result_8k",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                tcs = resp.choices[0].message.tool_calls
                if not tcs:
                    results.append(
                        self.make_result(
                            self.name,
                            "large_tool_result_8k",
                            Verdict.FAIL,
                            detail="No tool call after 8K result",
                        )
                    )
                else:
                    try:
                        json.loads(tcs[0].function.arguments)
                        results.append(
                            self.make_result(
                                self.name, "large_tool_result_8k", Verdict.PASS
                            )
                        )
                    except json.JSONDecodeError as e:
                        results.append(
                            self.make_result(
                                self.name,
                                "large_tool_result_8k",
                                Verdict.FAIL,
                                detail=f"Invalid JSON args: {e}",
                            )
                        )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "large_tool_result_8k",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        # ------------------------------------------------------------------ 8
        # 20_tools_auto
        # ------------------------------------------------------------------
        try:
            twenty_tools = [
                make_tool(
                    f"tool_{i}",
                    {
                        "type": "object",
                        "properties": {f"arg_{i}": {"type": "string"}},
                        "required": [f"arg_{i}"],
                        "additionalProperties": False,
                    },
                )
                for i in range(20)
            ]
            resp = await loop.run_in_executor(
                None,
                lambda: v.tc_chat(
                    [
                        {
                            "role": "user",
                            "content": "Call tool_7 with arg_7='hello'.",
                        }
                    ],
                    twenty_tools,
                    tool_choice="auto",
                    max_tokens=1024,
                ),
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "20_tools_auto",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                tcs = resp.choices[0].message.tool_calls
                errors = []
                if not tcs:
                    errors.append("No tool call with 20 tools defined")
                else:
                    try:
                        json.loads(tcs[0].function.arguments)
                    except json.JSONDecodeError as e:
                        errors.append(f"Invalid JSON args: {e}")
                    # Check the model picked a valid tool name
                    valid_names = {f"tool_{i}" for i in range(20)}
                    if tcs[0].function.name not in valid_names:
                        errors.append(
                            f"Selected tool '{tcs[0].function.name}' not in defined tools"
                        )
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "20_tools_auto",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "20_tools_auto", Verdict.ERROR, error=str(e)
                )
            )

        # ------------------------------------------------------------------ 9
        # 20_tools_required
        # ------------------------------------------------------------------
        try:
            twenty_tools_req = [
                make_tool(
                    f"fn_{i}",
                    {
                        "type": "object",
                        "properties": {f"p_{i}": {"type": "string"}},
                        "required": [f"p_{i}"],
                        "additionalProperties": False,
                    },
                )
                for i in range(20)
            ]
            resp = await loop.run_in_executor(
                None,
                lambda: v.tc_chat(
                    [{"role": "user", "content": "Use fn_3 with p_3='test'."}],
                    twenty_tools_req,
                    tool_choice="required",
                    max_tokens=1024,
                ),
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "20_tools_required",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                tcs = resp.choices[0].message.tool_calls
                errors = []
                if not tcs:
                    errors.append(
                        "tool_choice=required but no tool calls with 20 tools"
                    )
                else:
                    try:
                        json.loads(tcs[0].function.arguments)
                    except json.JSONDecodeError as e:
                        errors.append(f"Invalid JSON args: {e}")
                    valid_names = {f"fn_{i}" for i in range(20)}
                    if tcs[0].function.name not in valid_names:
                        errors.append(
                            f"Selected tool '{tcs[0].function.name}' not in defined tools"
                        )
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "20_tools_required",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "20_tools_required", Verdict.ERROR, error=str(e)
                )
            )

        # ------------------------------------------------------------------ 10
        # zero_param_tool
        # ------------------------------------------------------------------
        try:
            no_params_tool = make_tool(
                "no_params",
                {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            )
            resp = await loop.run_in_executor(
                None,
                lambda: v.tc_chat(
                    [{"role": "user", "content": "Call the no_params tool."}],
                    [no_params_tool],
                    tool_choice="required",
                    max_tokens=1024,
                ),
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "zero_param_tool",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                tcs = resp.choices[0].message.tool_calls
                errors = []
                if not tcs:
                    errors.append("No tool call for zero-param tool")
                else:
                    try:
                        args = json.loads(tcs[0].function.arguments)
                        if args != {}:
                            errors.append(
                                f"Expected empty args {{}}, got {args}"
                            )
                    except json.JSONDecodeError as e:
                        errors.append(f"Invalid JSON args: {e}")
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "zero_param_tool",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name, "zero_param_tool", Verdict.ERROR, error=str(e)
                )
            )

        # ------------------------------------------------------------------ 11
        # nested_object_params
        # ------------------------------------------------------------------
        try:
            nested_tool = make_tool(
                "deep_tool",
                {
                    "type": "object",
                    "properties": {
                        "config": {
                            "type": "object",
                            "properties": {
                                "database": {
                                    "type": "object",
                                    "properties": {
                                        "host": {"type": "string"},
                                        "port": {"type": "integer"},
                                    },
                                    "required": ["host", "port"],
                                    "additionalProperties": False,
                                },
                            },
                            "required": ["database"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["config"],
                    "additionalProperties": False,
                },
            )
            resp = await loop.run_in_executor(
                None,
                lambda: v.tc_chat(
                    [
                        {
                            "role": "user",
                            "content": "Configure database: host='db.example.com', port=5432.",
                        }
                    ],
                    [nested_tool],
                    tool_choice="required",
                    max_tokens=1024,
                ),
            )
            if budget_exhausted(resp):
                results.append(
                    self.make_result(
                        self.name,
                        "nested_object_params",
                        Verdict.INTERESTING,
                        detail="Budget exhausted",
                    )
                )
            else:
                tcs = resp.choices[0].message.tool_calls
                errors = []
                if not tcs:
                    errors.append("No tool call for nested-param tool")
                else:
                    try:
                        args = json.loads(tcs[0].function.arguments)
                        db = args.get("config", {}).get("database", {})
                        if not db.get("host"):
                            errors.append("Missing config.database.host")
                        if not isinstance(db.get("port"), int):
                            errors.append(
                                f"config.database.port not int: {db.get('port')!r}"
                            )
                    except json.JSONDecodeError as e:
                        errors.append(f"Invalid JSON args: {e}")
                verdict = Verdict.PASS if not errors else Verdict.FAIL
                results.append(
                    self.make_result(
                        self.name,
                        "nested_object_params",
                        verdict,
                        detail="; ".join(errors) or "OK",
                    )
                )
        except Exception as e:
            results.append(
                self.make_result(
                    self.name,
                    "nested_object_params",
                    Verdict.ERROR,
                    error=str(e),
                )
            )

        return results
