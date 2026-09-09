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
"""Validation scenario for streaming tool calling protocol compliance.

Tests that streaming tool call responses follow the OpenAI API spec:
correct chunk structure, finish_reason semantics, sequential indices,
and consistency with non-streaming responses.

Based on ralph battle test Category 1 (Streaming Protocol Compliance)
and parallel tool call index verification.

Refs: vLLM #18412 (missing id), #16340 (missing type), #31437 (name overwrite)
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from helpers import (
    budget_exhausted,
    make_tool,
    stream_budget_exhausted,
    validate_json_args,
)

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig
    from validator_client import ValidatorClient

# ---------------------------------------------------------------------------
# Tool definitions used across tests
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

_COMPARE_TOOL = make_tool(
    "search_fn",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "count": {"type": "integer"},
        },
        "required": ["query", "count"],
        "additionalProperties": False,
    },
    description="Search for results by query with a count limit",
)

_LOOKUP_A = make_tool(
    "lookup_a",
    {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
        "additionalProperties": False,
    },
    description="Look up information with lookup_a",
)

_LOOKUP_B = make_tool(
    "lookup_b",
    {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
        "additionalProperties": False,
    },
    description="Look up information with lookup_b",
)


@register_scenario
class TCStreamingProtocol(BaseScenario):
    """Streaming tool call protocol compliance per the OpenAI API spec."""

    name = "tc_streaming_protocol"
    description = (
        "Verify streaming tool call responses follow OpenAI protocol: "
        "first-chunk fields, finish_reason semantics, no post-finish deltas, "
        "streaming vs sync consistency, and parallel TC indices"
    )
    tags = ["validation", "tool_calling", "streaming"]
    requires_validator = True
    scenario_type = "validation"
    model_filter = None

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        validator = config.validator
        if not validator:
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

        # Run each sub-test, collecting results
        for test_fn in [
            self._test_first_chunk_has_id,
            self._test_first_chunk_has_type,
            self._test_first_chunk_has_name,
            self._test_finish_reason_is_tool_calls,
            self._test_no_duplicate_finish_reason,
            self._test_no_content_after_finish,
            self._test_streaming_vs_sync_consistency,
            self._test_parallel_tc_sequential_indices,
        ]:
            try:
                sub_results = await loop.run_in_executor(
                    None, test_fn, validator
                )
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
    # Individual test methods (sync, run inside executor)
    # ------------------------------------------------------------------

    def _test_first_chunk_has_id(
        self, validator: ValidatorClient
    ) -> list[ScenarioResult]:
        """First streaming TC chunk must carry a non-empty id."""
        test_name = "first_chunk_has_id"
        result = validator.tc_chat_stream(
            [{"role": "user", "content": "What is the weather in Paris?"}],
            [_WEATHER_TOOL],
            tool_choice="required",
            max_tokens=1024,
        )
        if stream_budget_exhausted(result):
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail="budget exhausted",
                )
            ]

        if not result["tool_calls"]:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail="No tool calls produced in streaming response",
                )
            ]

        for idx, first_tc in result["first_tc_chunks"].items():
            if not first_tc.id:
                return [
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.FAIL,
                        detail=f"tool_call[{idx}]: missing id in first chunk",
                    )
                ]
            if len(first_tc.id) < 3:
                return [
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.FAIL,
                        detail=f"tool_call[{idx}]: id too short ({first_tc.id!r})",
                    )
                ]

        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail=f"All {len(result['first_tc_chunks'])} first chunks have valid id",
            )
        ]

    def _test_first_chunk_has_type(
        self, validator: ValidatorClient
    ) -> list[ScenarioResult]:
        """First streaming TC chunk must have type='function'."""
        test_name = "first_chunk_has_type"
        result = validator.tc_chat_stream(
            [{"role": "user", "content": "What is the weather in Paris?"}],
            [_WEATHER_TOOL],
            tool_choice="required",
            max_tokens=1024,
        )
        if stream_budget_exhausted(result):
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail="budget exhausted",
                )
            ]

        if not result["tool_calls"]:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail="No tool calls produced in streaming response",
                )
            ]

        for idx, first_tc in result["first_tc_chunks"].items():
            tc_type = getattr(first_tc, "type", None)
            if tc_type != "function":
                return [
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.FAIL,
                        detail=f"tool_call[{idx}]: type={tc_type!r}, expected 'function'",
                    )
                ]

        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail="All first chunks have type='function'",
            )
        ]

    def _test_first_chunk_has_name(
        self, validator: ValidatorClient
    ) -> list[ScenarioResult]:
        """First streaming TC chunk must include a non-empty function name."""
        test_name = "first_chunk_has_name"
        result = validator.tc_chat_stream(
            [{"role": "user", "content": "What is the weather in Paris?"}],
            [_WEATHER_TOOL],
            tool_choice="required",
            max_tokens=1024,
        )
        if stream_budget_exhausted(result):
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail="budget exhausted",
                )
            ]

        if not result["tool_calls"]:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail="No tool calls produced in streaming response",
                )
            ]

        for idx, first_tc in result["first_tc_chunks"].items():
            fn = first_tc.function
            if not fn or not fn.name:
                return [
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.FAIL,
                        detail=f"tool_call[{idx}]: missing function.name in first chunk",
                    )
                ]

        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail="All first chunks have function.name set",
            )
        ]

    def _test_finish_reason_is_tool_calls(
        self, validator: ValidatorClient
    ) -> list[ScenarioResult]:
        """With tool_choice='required', finish_reason must be 'tool_calls'."""
        test_name = "finish_reason_is_tool_calls"
        result = validator.tc_chat_stream(
            [{"role": "user", "content": "Use the tool."}],
            [make_tool("do_thing")],
            tool_choice="required",
            max_tokens=1024,
        )
        if stream_budget_exhausted(result):
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail="budget exhausted",
                )
            ]

        fr = result["finish_reason"]
        if fr == "tool_calls":
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail="finish_reason='tool_calls' as expected",
                )
            ]

        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.FAIL,
                detail=f"Expected finish_reason='tool_calls', got {fr!r}",
            )
        ]

    def _test_no_duplicate_finish_reason(
        self, validator: ValidatorClient
    ) -> list[ScenarioResult]:
        """finish_reason must appear in exactly one chunk, not duplicated."""
        test_name = "no_duplicate_finish_reason"

        # Use the raw streaming API so we can iterate chunks ourselves
        stream = validator.tc_chat(
            [{"role": "user", "content": "Use the tool."}],
            [make_tool("check_fr")],
            tool_choice="required",
            max_tokens=1024,
            stream=True,
        )

        fr_count = 0
        for chunk in stream:
            if chunk.choices and chunk.choices[0].finish_reason:
                fr_count += 1

        if fr_count == 1:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail="finish_reason appeared exactly once",
                )
            ]
        if fr_count == 0:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.INTERESTING,
                    detail="finish_reason never appeared in any chunk",
                )
            ]
        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.FAIL,
                detail=f"finish_reason appeared {fr_count} times, expected exactly 1",
            )
        ]

    def _test_no_content_after_finish(
        self, validator: ValidatorClient
    ) -> list[ScenarioResult]:
        """No tool_call deltas should appear after finish_reason is set."""
        test_name = "no_content_after_finish"

        stream = validator.tc_chat(
            [{"role": "user", "content": "Use the tool."}],
            [make_tool("check_post_finish")],
            tool_choice="required",
            max_tokens=1024,
            stream=True,
        )

        finished = False
        post_finish_deltas = 0
        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finished = True
                continue
            if finished and choice.delta.tool_calls:
                post_finish_deltas += 1

        if post_finish_deltas == 0:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail="No tool_call deltas after finish_reason",
                )
            ]
        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.FAIL,
                detail=f"{post_finish_deltas} tool_call delta(s) arrived after finish_reason",
            )
        ]

    def _test_streaming_vs_sync_consistency(
        self, validator: ValidatorClient
    ) -> list[ScenarioResult]:
        """Streaming and non-streaming should produce the same tool call structure."""
        test_name = "streaming_vs_sync_consistency"
        messages = [
            {
                "role": "user",
                "content": "Search for 'vllm performance' with 5 results.",
            }
        ]

        # Non-streaming request
        resp = validator.tc_chat(
            messages,
            [_COMPARE_TOOL],
            tool_choice="required",
            max_tokens=1024,
        )
        if budget_exhausted(resp):
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail="budget exhausted",
                )
            ]

        ns_tc = resp.choices[0].message.tool_calls
        if not ns_tc:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.INTERESTING,
                    detail="Non-streaming produced no tool calls despite tool_choice=required",
                )
            ]

        ns_name = ns_tc[0].function.name
        try:
            ns_args = json.loads(ns_tc[0].function.arguments)
        except (json.JSONDecodeError, TypeError) as e:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Non-streaming args not valid JSON: {e}",
                )
            ]

        # Streaming request
        stream_result = validator.tc_chat_stream(
            messages,
            [_COMPARE_TOOL],
            tool_choice="required",
            max_tokens=1024,
        )
        if stream_budget_exhausted(stream_result):
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail="budget exhausted",
                )
            ]

        if not stream_result["tool_calls"]:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail="Streaming produced no tool calls while non-streaming did",
                )
            ]

        s_tc = stream_result["tool_calls"][0]
        s_name = s_tc["name"]
        try:
            s_args = json.loads(s_tc["arguments"])
        except (json.JSONDecodeError, TypeError) as e:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Streaming accumulated args not valid JSON: {e}",
                )
            ]

        # Compare structure: same function name
        if ns_name != s_name:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Function name mismatch: non-stream={ns_name!r}, stream={s_name!r}",
                )
            ]

        # Compare structure: same keys in arguments
        ns_keys = set(ns_args.keys())
        s_keys = set(s_args.keys())
        if ns_keys != s_keys:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Argument key mismatch: non-stream={ns_keys}, stream={s_keys}",
                )
            ]

        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail=f"Both produced {ns_name}() with keys {ns_keys}",
            )
        ]

    def _test_parallel_tc_sequential_indices(
        self, validator: ValidatorClient
    ) -> list[ScenarioResult]:
        """Parallel tool calls in streaming must have sequential 0-based indices."""
        test_name = "parallel_tc_sequential_indices"

        stream_result = validator.tc_chat_stream(
            [
                {
                    "role": "user",
                    "content": (
                        "Look up 'alpha' with lookup_a AND 'beta' with lookup_b. "
                        "You must call both tools."
                    ),
                }
            ],
            [_LOOKUP_A, _LOOKUP_B],
            tool_choice="auto",
            max_tokens=1024,
        )
        if stream_budget_exhausted(stream_result):
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail="budget exhausted",
                )
            ]

        tc_count = len(stream_result["tool_calls"])
        if tc_count < 2:
            # Model chose not to call both tools -- not a protocol violation
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.INTERESTING,
                    detail=f"Only {tc_count} tool call(s) produced, expected 2+ for parallel index test",
                )
            ]

        # Verify indices are sequential starting from 0
        indices = sorted(stream_result["first_tc_chunks"].keys())
        expected = list(range(len(indices)))
        if indices != expected:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Non-sequential indices: got {indices}, expected {expected}",
                )
            ]

        # Verify unique IDs across parallel calls
        ids = [tc["id"] for tc in stream_result["tool_calls"]]
        if len(ids) != len(set(ids)):
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Duplicate tool call IDs in parallel calls: {ids}",
                )
            ]

        # Verify each tool call's accumulated arguments are valid JSON
        _, parse_errors = validate_json_args(stream_result["tool_calls"])
        if parse_errors:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Parallel TC args parse errors: {'; '.join(parse_errors)}",
                )
            ]

        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail=f"{tc_count} parallel tool calls with sequential indices {expected} and unique IDs",
            )
        ]
