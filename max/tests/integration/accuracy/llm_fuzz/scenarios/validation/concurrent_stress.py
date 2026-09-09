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
"""Concurrent stress validation: output correctness under concurrent load.

Tests that concurrent requests return correct, non-contaminated responses.
Unlike s04 (crash resilience), this scenario verifies that each concurrent
request gets the right response type, the right tool call, the right schema,
and correct usage metadata -- even under load.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from helpers import (
    budget_exhausted,
    make_tool,
    stream_budget_exhausted,
)

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig
    from validator_client import ValidatorClient

# ---------------------------------------------------------------------------
# Shared tool definitions
# ---------------------------------------------------------------------------

_WEATHER_TOOL = make_tool(
    "get_weather",
    {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    },
    description="Get the current weather for a city",
)

_STOCK_TOOL = make_tool(
    "get_stock_price",
    {
        "type": "object",
        "properties": {"ticker": {"type": "string"}},
        "required": ["ticker"],
        "additionalProperties": False,
    },
    description="Get the current stock price for a ticker symbol",
)

_TRANSLATE_TOOL = make_tool(
    "translate_text",
    {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "target_language": {"type": "string"},
        },
        "required": ["text", "target_language"],
        "additionalProperties": False,
    },
    description="Translate text to a target language",
)

_CALCULATOR_TOOL = make_tool(
    "calculator",
    {
        "type": "object",
        "properties": {
            "expression": {"type": "string"},
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
    description="Evaluate a mathematical expression",
)

# Schemas for concurrent SO tests -- each has a distinct shape
_PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}

_COLOR_SCHEMA = {
    "type": "object",
    "properties": {
        "color": {"type": "string"},
        "hex_code": {"type": "string"},
    },
    "required": ["color", "hex_code"],
    "additionalProperties": False,
}

_COORDINATE_SCHEMA = {
    "type": "object",
    "properties": {
        "latitude": {"type": "number"},
        "longitude": {"type": "number"},
    },
    "required": ["latitude", "longitude"],
    "additionalProperties": False,
}

_BOOK_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "author": {"type": "string"},
        "year": {"type": "integer"},
    },
    "required": ["title", "author", "year"],
    "additionalProperties": False,
}


@register_scenario
class ConcurrentStress(BaseScenario):
    """Concurrent request correctness -- no cross-contamination under load."""

    name = "concurrent_stress"
    description = (
        "Verify output correctness under concurrent load: mixed request types, "
        "tool call isolation, structured output isolation, streaming usage, "
        "and burst-then-verify sequencing"
    )
    tags = ["validation", "concurrency", "stress"]
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

        for test_fn in [
            self._test_concurrent_mixed_10,
            self._test_concurrent_tc_isolation,
            self._test_concurrent_so_isolation,
            self._test_concurrent_streaming_usage,
            self._test_burst_then_verify,
        ]:
            try:
                sub_results = await loop.run_in_executor(None, test_fn, v)
                results.extend(sub_results)
            except Exception as e:
                test_name = test_fn.__name__.removeprefix("_test_")
                results.append(
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.ERROR,
                        error=str(e),
                    )
                )

        return results

    # ------------------------------------------------------------------
    # 1. concurrent_mixed_10
    # ------------------------------------------------------------------

    def _test_concurrent_mixed_10(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """10 concurrent requests mixing plain chat, TC, and SO. Verify no cross-contamination."""
        test_name = "concurrent_mixed_10"

        # Build 10 requests: indices 0-3 plain chat, 4-6 tool calling, 7-9 structured output
        def plain_chat(idx: int) -> tuple[str, Any]:
            resp = v.chat(
                [
                    {
                        "role": "user",
                        "content": f"Say hello, request number {idx}.",
                    }
                ],
                max_tokens=128,
            )
            return ("plain", resp)

        def tc_request(idx: int) -> tuple[str, Any]:
            resp = v.tc_chat(
                [
                    {
                        "role": "user",
                        "content": f"What is the weather in city_{idx}?",
                    }
                ],
                [_WEATHER_TOOL],
                tool_choice="required",
                max_tokens=256,
            )
            return ("tc", resp)

        def so_request(idx: int) -> tuple[str, Any]:
            resp = v.so_chat(
                [
                    {
                        "role": "user",
                        "content": f"Return name='Person{idx}' and age={20 + idx}.",
                    }
                ],
                _PERSON_SCHEMA,
                max_tokens=256,
            )
            return ("so", resp)

        def dispatch(idx: int) -> tuple[str, Any]:
            if idx < 4:
                return plain_chat(idx)
            elif idx < 7:
                return tc_request(idx)
            else:
                return so_request(idx)

        args_list = [(i,) for i in range(10)]
        concurrent_results = v.concurrent_run(
            dispatch, args_list, max_workers=10
        )

        errors: list[str] = []
        for idx, result, err in concurrent_results:
            if err is not None:
                errors.append(f"request[{idx}]: exception: {err}")
                continue

            req_type, resp = result

            if req_type == "plain":
                if budget_exhausted(resp):
                    continue
                content = resp.choices[0].message.content
                if content is None or len(content.strip()) == 0:
                    errors.append(f"request[{idx}] (plain): empty content")
                # Plain chat should NOT have tool_calls
                if resp.choices[0].message.tool_calls:
                    errors.append(
                        f"request[{idx}] (plain): unexpected tool_calls in plain chat"
                    )

            elif req_type == "tc":
                if budget_exhausted(resp):
                    continue
                tcs = resp.choices[0].message.tool_calls
                if not tcs:
                    errors.append(
                        f"request[{idx}] (tc): no tool_calls returned"
                    )
                elif tcs[0].function.name != "get_weather":
                    errors.append(
                        f"request[{idx}] (tc): wrong function name {tcs[0].function.name!r}"
                    )

            elif req_type == "so":
                if budget_exhausted(resp):
                    continue
                content = resp.choices[0].message.content
                try:
                    data = json.loads(content)
                    if not isinstance(data, dict):
                        errors.append(
                            f"request[{idx}] (so): response not a dict"
                        )
                    elif "name" not in data or "age" not in data:
                        errors.append(
                            f"request[{idx}] (so): missing required fields, got keys={list(data.keys())}"
                        )
                except (json.JSONDecodeError, TypeError) as e:
                    errors.append(f"request[{idx}] (so): invalid JSON: {e}")

        if errors:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"{len(errors)} error(s): {'; '.join(errors[:5])}",
                )
            ]
        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail="All 10 concurrent mixed requests returned correct response types",
            )
        ]

    # ------------------------------------------------------------------
    # 2. concurrent_tc_isolation
    # ------------------------------------------------------------------

    def _test_concurrent_tc_isolation(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """4 concurrent TC requests each asking for a different tool. Verify no mixing."""
        test_name = "concurrent_tc_isolation"

        tools = [_WEATHER_TOOL, _STOCK_TOOL, _TRANSLATE_TOOL, _CALCULATOR_TOOL]
        prompts = [
            "What is the weather in Tokyo?",
            "What is the stock price for AAPL?",
            "Translate 'hello' to French.",
            "Calculate the result of 2+2.",
        ]
        expected_names = [
            "get_weather",
            "get_stock_price",
            "translate_text",
            "calculator",
        ]

        def tc_call(idx: int) -> Any:
            return v.tc_chat(
                [{"role": "user", "content": prompts[idx]}],
                [tools[idx]],
                tool_choice="required",
                max_tokens=256,
            )

        args_list = [(i,) for i in range(4)]
        concurrent_results = v.concurrent_run(tc_call, args_list, max_workers=4)

        errors: list[str] = []
        for idx, result, err in concurrent_results:
            if err is not None:
                errors.append(
                    f"tc[{idx}] ({expected_names[idx]}): exception: {err}"
                )
                continue
            resp = result
            if budget_exhausted(resp):
                continue
            tcs = resp.choices[0].message.tool_calls
            if not tcs:
                errors.append(
                    f"tc[{idx}] ({expected_names[idx]}): no tool_calls"
                )
                continue
            actual_name = tcs[0].function.name
            if actual_name != expected_names[idx]:
                errors.append(
                    f"tc[{idx}]: expected {expected_names[idx]!r}, got {actual_name!r} (cross-contamination?)"
                )

        if errors:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"{len(errors)} error(s): {'; '.join(errors)}",
                )
            ]
        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail="All 4 concurrent TC requests called the correct tool",
            )
        ]

    # ------------------------------------------------------------------
    # 3. concurrent_so_isolation
    # ------------------------------------------------------------------

    def _test_concurrent_so_isolation(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """4 concurrent SO requests each with a different schema. Verify each matches its own schema."""
        test_name = "concurrent_so_isolation"

        schemas = [
            _PERSON_SCHEMA,
            _COLOR_SCHEMA,
            _COORDINATE_SCHEMA,
            _BOOK_SCHEMA,
        ]
        prompts = [
            "Return name='Alice' and age=30.",
            "Return color='blue' and hex_code='#0000FF'.",
            "Return latitude=48.8566 and longitude=2.3522.",
            "Return title='1984', author='George Orwell', and year=1949.",
        ]
        required_keys_per_schema = [
            {"name", "age"},
            {"color", "hex_code"},
            {"latitude", "longitude"},
            {"title", "author", "year"},
        ]
        schema_labels = ["person", "color", "coordinate", "book"]

        def so_call(idx: int) -> Any:
            return v.so_chat(
                [{"role": "user", "content": prompts[idx]}],
                schemas[idx],
                schema_name=schema_labels[idx],
                max_tokens=256,
            )

        args_list = [(i,) for i in range(4)]
        concurrent_results = v.concurrent_run(so_call, args_list, max_workers=4)

        errors: list[str] = []
        for idx, result, err in concurrent_results:
            label = schema_labels[idx]
            if err is not None:
                errors.append(f"so[{idx}] ({label}): exception: {err}")
                continue
            resp = result
            if budget_exhausted(resp):
                continue
            content = resp.choices[0].message.content
            try:
                data = json.loads(content)
            except (json.JSONDecodeError, TypeError) as e:
                errors.append(f"so[{idx}] ({label}): invalid JSON: {e}")
                continue
            if not isinstance(data, dict):
                errors.append(f"so[{idx}] ({label}): response not a dict")
                continue
            missing = required_keys_per_schema[idx] - set(data.keys())
            if missing:
                errors.append(
                    f"so[{idx}] ({label}): missing keys {missing}, got {set(data.keys())} (cross-contamination?)"
                )

        if errors:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"{len(errors)} error(s): {'; '.join(errors)}",
                )
            ]
        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail="All 4 concurrent SO requests matched their own schemas",
            )
        ]

    # ------------------------------------------------------------------
    # 4. concurrent_streaming_usage
    # ------------------------------------------------------------------

    def _test_concurrent_streaming_usage(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """4 concurrent streaming requests with include_usage. Verify all return non-zero token counts."""
        test_name = "concurrent_streaming_usage"

        prompts = [
            "Explain what a CPU is in one sentence.",
            "Explain what RAM is in one sentence.",
            "Explain what a GPU is in one sentence.",
            "Explain what an SSD is in one sentence.",
        ]

        def streaming_call(idx: int) -> dict[str, Any]:
            return v.chat_stream(
                [{"role": "user", "content": prompts[idx]}],
                max_tokens=256,
                stream_options={"include_usage": True},
            )

        args_list = [(i,) for i in range(4)]
        concurrent_results = v.concurrent_run(
            streaming_call, args_list, max_workers=4
        )

        errors: list[str] = []
        for idx, result, err in concurrent_results:
            if err is not None:
                errors.append(f"stream[{idx}]: exception: {err}")
                continue
            if stream_budget_exhausted(result):
                continue

            usage = result.get("usage")
            if usage is None:
                errors.append(
                    f"stream[{idx}]: no usage chunk returned despite include_usage=True"
                )
                continue

            prompt_tokens = getattr(usage, "prompt_tokens", None) or 0
            completion_tokens = getattr(usage, "completion_tokens", None) or 0

            if prompt_tokens == 0:
                errors.append(f"stream[{idx}]: prompt_tokens=0")
            if completion_tokens == 0:
                errors.append(f"stream[{idx}]: completion_tokens=0")

        if errors:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"{len(errors)} error(s): {'; '.join(errors)}",
                )
            ]
        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail="All 4 concurrent streaming requests returned non-zero usage",
            )
        ]

    # ------------------------------------------------------------------
    # 5. burst_then_verify
    # ------------------------------------------------------------------

    def _test_burst_then_verify(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """Send 10 rapid concurrent requests, then 1 careful request. Verify no queue corruption."""
        test_name = "burst_then_verify"

        # Phase 1: burst of 10 concurrent requests (fire-and-forget style, but we collect results)
        def burst_call(idx: int) -> Any:
            return v.chat(
                [
                    {
                        "role": "user",
                        "content": f"Burst request number {idx}. Reply with just 'ok'.",
                    }
                ],
                max_tokens=64,
            )

        args_list = [(i,) for i in range(10)]
        burst_results = v.concurrent_run(burst_call, args_list, max_workers=10)

        # Check burst results -- we mainly care that they completed without server errors
        burst_errors: list[str] = []
        for idx, _, err in burst_results:
            if err is not None:
                burst_errors.append(f"burst[{idx}]: {err}")

        # Phase 2: careful request immediately after burst
        try:
            careful_schema = {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["answer", "confidence"],
                "additionalProperties": False,
            }
            careful_resp = v.so_chat(
                [
                    {
                        "role": "user",
                        "content": (
                            "After a burst of requests, return answer='post_burst_ok' and confidence=0.99."
                        ),
                    }
                ],
                careful_schema,
                schema_name="careful_check",
                max_tokens=256,
            )
        except Exception as e:
            detail = f"Careful request after burst failed: {e}"
            if burst_errors:
                detail += f"; also {len(burst_errors)} burst error(s)"
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=detail,
                )
            ]

        if budget_exhausted(careful_resp):
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail="budget exhausted on careful request",
                )
            ]

        content = careful_resp.choices[0].message.content
        errors: list[str] = []
        try:
            data = json.loads(content)
            if not isinstance(data, dict):
                errors.append(
                    f"careful response not a dict: {type(data).__name__}"
                )
            else:
                if "answer" not in data:
                    errors.append("missing 'answer' field")
                if "confidence" not in data:
                    errors.append("missing 'confidence' field")
        except (json.JSONDecodeError, TypeError) as e:
            errors.append(f"careful response invalid JSON: {e}")

        if burst_errors:
            errors.append(f"{len(burst_errors)} burst request(s) failed")

        if errors:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"{len(errors)} error(s): {'; '.join(errors)}",
                    response_body=content or "",
                )
            ]
        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail="Careful request after 10-request burst returned correct structured output",
            )
        ]
