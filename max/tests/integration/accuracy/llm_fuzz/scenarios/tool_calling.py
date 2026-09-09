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
Scenarios: Tool calling attacks
Target: Crashes from malformed tool definitions, empty args, huge schemas.
Known issues: vLLM #19419 (empty args crash), #27641 (streaming divergence).
Regression coverage:
- MXSERV-81 (JSON-string arguments in multi-turn tool calls).
- Tool schemas containing ``oneOf`` / ``const`` that Kimi's HF
  tokenizer cannot parse.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from helpers import STRUCTURAL_LEAK_MARKERS

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario
from scenarios._kimi_fixtures import PRODUCTION_ONEOF_TOOL

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig


@register_scenario
class ToolCallingAttacks(BaseScenario):
    name = "tool_calling"
    description = "Malformed tools, empty args, huge definitions, streaming vs non-streaming divergence"
    tags = ["tools", "function_calling", "crash"]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results = []
        model = config.model

        def with_tools(
            tools: Any,
            content: str = "What's the weather in Paris?",
            **extra: Any,
        ) -> dict[str, Any]:
            p = {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "tools": tools,
                "max_tokens": 200,
            }
            p.update(extra)
            return p

        valid_tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name",
                        },
                    },
                    "required": ["location"],
                },
            },
        }

        # ----- 1. Valid tool call (baseline) -----
        resp = await client.post_json(with_tools([valid_tool]))
        results.append(
            self.make_result(
                self.name,
                "valid_tool_baseline",
                Verdict.PASS if resp.status == 200 else Verdict.FAIL,
                status_code=resp.status,
            )
        )

        # ----- 2. Malformed tool definitions -----
        # Each entry: (tools, expected) where expected is:
        #   "reject"  → 400 = PASS, 200 = FAIL
        #   "accept"  → 200 = PASS, 400 = INTERESTING
        #   "either"  → 200 or 400 = PASS
        # 500 is always FAIL (server crash).
        malformed_tools = {
            # Empty array: normalized to no-tools, 200 is correct
            "empty_tools_array": ([], "accept"),
            # Missing type: OpenAI spec lists `type` as required
            "tool_missing_type": (
                [{"function": valid_tool["function"]}],
                "reject",
            ),
            # Missing function: structurally invalid, must reject
            "tool_missing_function": ([{"type": "function"}], "reject"),
            # Wrong type value: must reject
            "tool_wrong_type": (
                [{"type": "invalid", "function": valid_tool["function"]}],
                "reject",
            ),
            # Missing name: must reject (required field)
            "tool_missing_name": (
                [
                    {
                        "type": "function",
                        "function": {"description": "test", "parameters": {}},
                    }
                ],
                "reject",
            ),
            # Empty name: must reject (breaks grammar/ID generation)
            "tool_empty_name": (
                [
                    {
                        "type": "function",
                        "function": {"name": "", "parameters": {}},
                    }
                ],
                "reject",
            ),
            # Whitespace in name: must reject (breaks token parsing)
            "tool_name_special_chars": (
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "fn-with spaces & symbols!",
                            "parameters": {},
                        },
                    }
                ],
                "reject",
            ),
            # Null parameters: normalized to empty schema, 200 is correct
            "tool_null_parameters": (
                [
                    {
                        "type": "function",
                        "function": {"name": "test", "parameters": None},
                    }
                ],
                "accept",
            ),
            # Parameters as string: must reject
            "tool_parameters_not_object": (
                [
                    {
                        "type": "function",
                        "function": {"name": "test", "parameters": "string"},
                    }
                ],
                "reject",
            ),
            # Non-dict tool elements: must reject
            "tool_is_string": (["not a tool object"], "reject"),
            "tool_is_number": ([42], "reject"),
            "tool_is_null": ([None], "reject"),
            # Duplicate names: accepted (OpenAI accepts too)
            "duplicate_tool_names": ([valid_tool, valid_tool], "accept"),
        }

        for name, (tools, expected) in malformed_tools.items():
            resp = await client.post_json(with_tools(tools))
            if resp.error == "TIMEOUT" or resp.status >= 500:
                v = Verdict.FAIL
            elif expected == "reject":
                if 400 <= resp.status < 500:
                    v = Verdict.PASS
                elif resp.status == 200:
                    v = Verdict.FAIL  # should have been rejected
                else:
                    v = Verdict.INTERESTING
            elif expected == "accept":
                if resp.status == 200:
                    v = Verdict.PASS
                elif 400 <= resp.status < 500:
                    v = Verdict.INTERESTING  # stricter than required
                else:
                    v = Verdict.INTERESTING
            else:  # "either"
                v = (
                    Verdict.PASS
                    if resp.status in (200, 400)
                    else Verdict.INTERESTING
                )
            results.append(
                self.make_result(
                    self.name,
                    f"malformed_{name}",
                    v,
                    status_code=resp.status,
                    detail=f"Status {resp.status}"
                    + (f" error: {resp.error}" if resp.error else ""),
                )
            )

        # ----- 3. Huge tool definitions -----
        huge_tool = {
            "type": "function",
            "function": {
                "name": "mega_function",
                "description": "x" * 50000,
                "parameters": {
                    "type": "object",
                    "properties": {
                        f"param_{i}": {
                            "type": "string",
                            "description": f"Param {i}",
                        }
                        for i in range(500)
                    },
                },
            },
        }
        resp = await client.post_json(with_tools([huge_tool]))
        results.append(
            self.make_result(
                self.name,
                "huge_tool_definition",
                Verdict.FAIL
                if resp.status >= 500 or resp.error == "TIMEOUT"
                else Verdict.PASS,
                status_code=resp.status,
                detail=f"500 params, 50k desc: status {resp.status}",
            )
        )

        # ----- 4. Many tools -----
        many_tools = [
            {
                "type": "function",
                "function": {
                    "name": f"function_{i}",
                    "description": f"Function number {i}",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                    },
                },
            }
            for i in range(200)
        ]
        resp = await client.post_json(with_tools(many_tools))
        results.append(
            self.make_result(
                self.name,
                "200_tools",
                Verdict.FAIL
                if resp.status >= 500 or resp.error == "TIMEOUT"
                else Verdict.PASS,
                status_code=resp.status,
            )
        )

        # ----- 5. Tool call with empty arguments (vLLM #19419) -----
        # Simulate tool response with empty args
        tool_response_payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Call the test function"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {"name": "test_fn", "arguments": ""},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_abc123",
                    "content": "result",
                },
            ],
            "tools": [valid_tool],
            "max_tokens": 50,
        }
        resp = await client.post_json(
            tool_response_payload, timeout=config.timeout * 0.5
        )
        results.append(
            self.make_result(
                self.name,
                "empty_tool_arguments",
                Verdict.FAIL if resp.status >= 500 else Verdict.PASS,
                status_code=resp.status,
                detail="Tests vLLM #19419 pattern",
            )
        )

        # ----- 6. Malformed tool_calls in assistant message -----
        bad_tool_calls = {
            "null_arguments": {"name": "test", "arguments": None},
            "invalid_json_arguments": {
                "name": "test",
                "arguments": "{invalid json",
            },
            "arguments_is_number": {"name": "test", "arguments": "42"},
            "missing_name": {"arguments": "{}"},
            "empty_function": {},
        }

        for tc_name, fn_data in bad_tool_calls.items():
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_test",
                                "type": "function",
                                "function": fn_data,
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_test",
                        "content": "ok",
                    },
                ],
                "max_tokens": 50,
            }
            resp = await client.post_json(payload)
            results.append(
                self.make_result(
                    self.name,
                    f"bad_tool_call_{tc_name}",
                    Verdict.FAIL if resp.status >= 500 else Verdict.PASS,
                    status_code=resp.status,
                )
            )

        # ----- 7. Streaming vs non-streaming tool call divergence -----
        # vLLM #27641: streaming produces different tool call output
        tool_payload = with_tools(
            [valid_tool], content="Get me the weather in Tokyo"
        )
        resp_sync = await client.post_json(tool_payload)
        resp_stream = await client.post_streaming(tool_payload)

        if resp_sync.status == 200 and resp_stream.status == 200:
            try:
                sync_data = json.loads(resp_sync.body)
                sync_calls = (
                    sync_data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("tool_calls", [])
                )

                # Parse streaming chunks for tool calls
                stream_has_tools = False
                for c in resp_stream.chunks or []:
                    if c == "[DONE]":
                        continue
                    try:
                        chunk_data = json.loads(c)
                        delta = chunk_data.get("choices", [{}])[0].get(
                            "delta", {}
                        )
                        if delta.get("tool_calls"):
                            stream_has_tools = True
                            break
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

                if sync_calls and not stream_has_tools:
                    verdict = Verdict.INTERESTING
                    detail = "Sync has tool calls, streaming doesn't"
                elif not sync_calls and stream_has_tools:
                    verdict = Verdict.INTERESTING
                    detail = "Streaming has tool calls, sync doesn't"
                else:
                    verdict = Verdict.PASS
                    detail = "Consistent tool calling behavior"
            except (json.JSONDecodeError, KeyError, IndexError):
                verdict = Verdict.PASS
                detail = "Could not compare (parse error)"
        else:
            verdict = Verdict.PASS
            detail = f"Sync={resp_sync.status}, Stream={resp_stream.status}"

        results.append(
            self.make_result(
                self.name,
                "streaming_vs_sync_tool_divergence",
                verdict,
                detail=detail,
            )
        )

        # ----- 8. tool_choice forcing -----
        force_payloads = {
            "tool_choice_none": with_tools([valid_tool], tool_choice="none"),
            "tool_choice_auto": with_tools([valid_tool], tool_choice="auto"),
            "tool_choice_required": with_tools(
                [valid_tool], tool_choice="required"
            ),
            "tool_choice_specific": with_tools(
                [valid_tool],
                tool_choice={
                    "type": "function",
                    "function": {"name": "get_weather"},
                },
            ),
            "tool_choice_nonexistent": with_tools(
                [valid_tool],
                tool_choice={
                    "type": "function",
                    "function": {"name": "nonexistent_fn"},
                },
            ),
            "tool_choice_invalid": with_tools(
                [valid_tool], tool_choice="invalid_value"
            ),
            "tool_choice_empty_obj": with_tools([valid_tool], tool_choice={}),
        }

        for name, payload in force_payloads.items():
            resp = await client.post_json(payload, timeout=config.timeout * 0.5)
            results.append(
                self.make_result(
                    self.name,
                    name,
                    Verdict.FAIL if resp.status >= 500 else Verdict.PASS,
                    status_code=resp.status,
                )
            )

        # ----- 9. parallel_tool_calls parameter -----
        resp = await client.post_json(
            with_tools([valid_tool], parallel_tool_calls=True)
        )
        results.append(
            self.make_result(
                self.name,
                "parallel_tool_calls_true",
                Verdict.FAIL if resp.status >= 500 else Verdict.PASS,
                status_code=resp.status,
            )
        )

        resp = await client.post_json(
            with_tools([valid_tool], parallel_tool_calls=False)
        )
        results.append(
            self.make_result(
                self.name,
                "parallel_tool_calls_false",
                Verdict.FAIL if resp.status >= 500 else Verdict.PASS,
                status_code=resp.status,
            )
        )

        # ----- 10. Multi-turn with JSON-string tool_calls.arguments -----
        # Regression test for MXSERV-81: OpenAI API sends function.arguments as
        # a JSON-serialized string, but some chat templates iterate it as a dict.
        # Server must normalize string args to dict before templating.
        multi_turn_json_args = {
            # Basic case: single tool call with JSON string arguments
            "json_string_args_basic": {
                "model": model,
                "messages": [
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
                                    "arguments": '{"location": "Paris"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_001",
                        "content": "20°C, sunny",
                    },
                ],
                "tools": [valid_tool],
                "max_tokens": 100,
            },
            # Complex nested JSON string arguments
            "json_string_args_nested": {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Configure the system"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_002",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": json.dumps(
                                        {
                                            "location": "Paris",
                                            "options": {
                                                "units": "celsius",
                                                "include_forecast": True,
                                            },
                                        }
                                    ),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_002",
                        "content": "configured",
                    },
                ],
                "tools": [valid_tool],
                "max_tokens": 100,
            },
            # Multiple tool calls with JSON string arguments
            "json_string_args_parallel": {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Check multiple cities"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_003a",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Paris"}',
                                },
                            },
                            {
                                "id": "call_003b",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "London"}',
                                },
                            },
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_003a",
                        "content": "Paris: 20°C",
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_003b",
                        "content": "London: 15°C",
                    },
                ],
                "tools": [valid_tool],
                "max_tokens": 100,
            },
            # Empty JSON object as string (common for no-arg tools)
            "json_string_args_empty_obj": {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Get the time"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_004",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_004",
                        "content": "result",
                    },
                ],
                "tools": [valid_tool],
                "max_tokens": 100,
            },
            # Multi-turn: tool call → tool result → another turn requesting tools
            "json_string_args_multi_turn": {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Check weather in Paris"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_005",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Paris"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_005",
                        "content": "Paris: 20°C, sunny",
                    },
                    {
                        "role": "assistant",
                        "content": "The weather in Paris is 20°C and sunny!",
                    },
                    {"role": "user", "content": "Now check London"},
                ],
                "tools": [valid_tool],
                "tool_choice": "required",
                "max_tokens": 150,
            },
        }

        for name, payload in multi_turn_json_args.items():
            resp = await client.post_json(payload)
            results.append(
                self.make_result(
                    self.name,
                    name,
                    Verdict.FAIL
                    if resp.status >= 500 or resp.error == "TIMEOUT"
                    else Verdict.PASS,
                    status_code=resp.status,
                    detail=f"Status {resp.status}"
                    + (f" error: {resp.error}" if resp.error else ""),
                )
            )

        # Also test streaming variants of key cases
        for name_suffix, payload in [
            (
                "json_string_args_basic",
                multi_turn_json_args["json_string_args_basic"],
            ),
            (
                "json_string_args_parallel",
                multi_turn_json_args["json_string_args_parallel"],
            ),
        ]:
            resp_stream = await client.post_streaming(payload)
            results.append(
                self.make_result(
                    self.name,
                    f"{name_suffix}_streaming",
                    Verdict.FAIL
                    if resp_stream.status >= 500
                    or resp_stream.error == "TIMEOUT"
                    else Verdict.PASS,
                    status_code=resp_stream.status,
                    detail=f"Status {resp_stream.status}"
                    + (
                        f" error: {resp_stream.error}"
                        if resp_stream.error
                        else ""
                    ),
                )
            )

        # ----- 11. Grammar enforcement: no prose leak across transitions -----
        # Test that reasoning and grammar transitions do not leak text into the content field.
        # If ``tool_choice="required"`` then we should continue to constrain decoding even after a tool call completes.
        # Another source of unintentional unconstrained decoding is failing to update the bitmask if a transition
        # is triggered during draft token FSM advancement.
        #
        # Both bugs are silent — ``tool_calls`` is
        # well-formed, but ``message.content`` carries leaked prose. The
        # contract for ``tool_choice="required"`` is that the model MUST
        # call a tool with no preamble: ``content`` should be empty.
        def _assert_required_clean(
            name: str, resp_body: str, status: int
        ) -> ScenarioResult:
            """Verdict for required-mode regression tests.

            FAIL conditions in priority order:
              * server error (status >= 500 or timeout)
              * response not parseable JSON
              * message.content non-empty (prefix or suffix prose leak)
              * tool_calls missing or empty
              * any tool_call.function.arguments not valid JSON
            """
            if status >= 500:
                return self.make_result(
                    self.name,
                    name,
                    Verdict.FAIL,
                    status_code=status,
                    detail=f"server error: status {status}",
                )
            try:
                resp = json.loads(resp_body)
            except json.JSONDecodeError as e:
                return self.make_result(
                    self.name,
                    name,
                    Verdict.FAIL,
                    status_code=status,
                    detail=f"response body not JSON: {e}",
                )
            try:
                message = resp["choices"][0]["message"]
            except (KeyError, IndexError, TypeError) as e:
                return self.make_result(
                    self.name,
                    name,
                    Verdict.FAIL,
                    status_code=status,
                    detail=f"missing choices[0].message: {e}",
                )
            content = message.get("content") or ""
            tool_calls = message.get("tool_calls") or []
            if content:
                # The fingerprint of either leak: prose in ``content``
                # for a required-mode response. Truncate to keep the
                # report manageable.
                return self.make_result(
                    self.name,
                    name,
                    Verdict.FAIL,
                    status_code=status,
                    detail=(
                        "prose leaked into content under "
                        f"tool_choice='required': {content[:120]!r}"
                    ),
                )
            if not tool_calls:
                return self.make_result(
                    self.name,
                    name,
                    Verdict.FAIL,
                    status_code=status,
                    detail="tool_choice='required' produced no tool_calls",
                )
            for tc in tool_calls:
                args = tc.get("function", {}).get("arguments")
                if args is None:
                    return self.make_result(
                        self.name,
                        name,
                        Verdict.FAIL,
                        status_code=status,
                        detail=f"tool_call missing function.arguments: {tc}",
                    )
                try:
                    json.loads(args)
                except json.JSONDecodeError as e:
                    return self.make_result(
                        self.name,
                        name,
                        Verdict.FAIL,
                        status_code=status,
                        detail=(
                            "tool_call.function.arguments not valid JSON: "
                            f"{e}; raw={args!r}"
                        ),
                    )
            return self.make_result(
                self.name, name, Verdict.PASS, status_code=status
            )

        # Reasoning-heavy prompt: gives the model an obvious incentive
        # to think before calling, exercising the ``</think>`` → tool
        # section transition that Option 2 fixed.
        required_payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": ("What is the weather like in Coquitlam today?"),
                }
            ],
            "tools": [valid_tool],
            "tool_choice": "required",
            "max_tokens": 256,
        }
        resp = await client.post_json(required_payload)
        results.append(
            _assert_required_clean(
                "required_no_content_leak", resp.body, resp.status
            )
        )

        # Streaming variant: same contract, exercised through the
        # streaming response path which assembles ``content`` and
        # ``tool_calls`` from incremental deltas.
        resp_stream = await client.post_streaming(required_payload)
        results.append(
            self.make_result(
                self.name,
                "required_no_content_leak_streaming",
                Verdict.FAIL
                if resp_stream.status >= 500 or resp_stream.error == "TIMEOUT"
                else Verdict.PASS,
                status_code=resp_stream.status,
                detail=f"Status {resp_stream.status}"
                + (f" error: {resp_stream.error}" if resp_stream.error else ""),
            )
        )

        # Auto-mode tool call: verifies the helper's mid-window
        # transition handling for the ``<|tool_calls_section_begin|>``
        # start-token path. Content may be non-empty (auto allows
        # pre-call prose), so the assertion is just well-formed tool
        # calls.
        auto_payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Use the get_weather tool to look up Coquitlam, BC."
                    ),
                }
            ],
            "tools": [valid_tool],
            "tool_choice": "auto",
            "max_tokens": 256,
        }
        resp = await client.post_json(auto_payload)
        verdict = Verdict.PASS
        detail = ""
        if resp.status >= 500:
            verdict = Verdict.FAIL
            detail = f"server error: status {resp.status}"
        else:
            try:
                body = json.loads(resp.body)
                tcs = (
                    body.get("choices", [{}])[0]
                    .get("message", {})
                    .get("tool_calls")
                    or []
                )
                if not tcs:
                    # Model declined to call; that's a valid auto
                    # outcome and not what this scenario targets.
                    verdict = Verdict.PASS
                else:
                    for tc in tcs:
                        json.loads(tc["function"]["arguments"])
            except (
                json.JSONDecodeError,
                KeyError,
                IndexError,
                TypeError,
            ) as e:
                verdict = Verdict.FAIL
                detail = f"malformed auto-mode tool_call: {e}"
        results.append(
            self.make_result(
                self.name,
                "auto_tool_call_valid_json",
                verdict,
                status_code=resp.status,
                detail=detail,
            )
        )

        # ----- 12. Tool schemas with oneOf / const constructs -----
        # A tool whose ``parameters`` contains ``oneOf`` or
        # a bare ``{"const": X}`` makes Kimi's HF tokenizer code
        # (``tool_declaration_ts.py:_parse_parameter_type``) raise.
        # ``_sanitize_kimi_tool_schemas`` rewrites both
        # constructs into Kimi-supported equivalents before forwarding
        # to the delegate.
        #
        # End-to-end signal: with the sanitizer in place,
        # ``tool_choice="required"`` should produce a valid tool call.
        # Without it, ``message.content`` would be non-empty (prose
        # explaining the answer) and ``tool_calls`` would be missing.
        # The schema lives in ``scenarios/_kimi_fixtures.py`` so the
        # freeze-repro scenario (``kimi_freeze_repro``) and this test
        # cannot drift.
        one_of_payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Place the item 'apple' at the end of the list."
                    ),
                }
            ],
            "tools": [PRODUCTION_ONEOF_TOOL],
            "tool_choice": "required",
            "max_tokens": 256,
        }
        resp = await client.post_json(one_of_payload)
        results.append(
            _assert_required_clean(
                "tool_schema_with_oneof_and_const",
                resp.body,
                resp.status,
            )
        )

        # ----- 13. Interleaved thinking + multiple tool-call sections -----
        # Kimi K2.5 interleaves <think>...</think> reasoning with multiple
        # <|tool_calls_section_begin|>...<|tool_calls_section_end|> sections in
        # one turn. The constrained-decoding grammar caps sections at 1
        # previously, which (a) desynced the async matcher in tool_choice=auto
        # — the prod ``Async matcher rejected token 163595`` — and (b) could
        # leak structural / reasoning markers into message.content. This
        # multi-step agentic prompt encourages several tool calls; the
        # invariant we assert holds no matter how many the model emits:
        #   * no server error,
        #   * tool_call arguments are valid JSON,
        #   * NO structural / reasoning markers leak into message.content.
        # Marker leakage is the fingerprint of a matcher/parser desync or a
        # reasoning-span split that dropped a block — exactly the regressions
        # this change guards. ``STRUCTURAL_LEAK_MARKERS`` is the union across
        # tool-calling models, so this check is meaningful for whichever model
        # is served.
        leak_markers = STRUCTURAL_LEAK_MARKERS

        def _assert_no_marker_leak(
            name: str, resp_body: str, status: int
        ) -> ScenarioResult:
            if status >= 500:
                return self.make_result(
                    self.name,
                    name,
                    Verdict.FAIL,
                    status_code=status,
                    detail=f"server error: status {status}",
                )
            try:
                body = json.loads(resp_body)
                message = body["choices"][0]["message"]
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                return self.make_result(
                    self.name,
                    name,
                    Verdict.FAIL,
                    status_code=status,
                    detail=f"malformed response: {e}",
                )
            content = message.get("content") or ""
            reasoning = message.get("reasoning") or ""
            leaked = [m for m in leak_markers if m in content or m in reasoning]
            if leaked:
                return self.make_result(
                    self.name,
                    name,
                    Verdict.FAIL,
                    status_code=status,
                    detail=(
                        f"structural/reasoning markers leaked into "
                        f"content/reasoning: {leaked}"
                    ),
                )
            for tc in message.get("tool_calls") or []:
                args = tc.get("function", {}).get("arguments")
                if args is not None:
                    try:
                        json.loads(args)
                    except json.JSONDecodeError as e:
                        return self.make_result(
                            self.name,
                            name,
                            Verdict.FAIL,
                            status_code=status,
                            detail=f"tool_call arguments not valid JSON: {e}",
                        )
            return self.make_result(
                self.name, name, Verdict.PASS, status_code=status
            )

        interleaved_payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Plan a trip: first get the weather in Paris, then "
                        "based on that decide whether to also get the weather "
                        "in London. Use the get_weather tool for each city, "
                        "thinking step by step between calls."
                    ),
                }
            ],
            "tools": [valid_tool],
            "tool_choice": "auto",
            "chat_template_kwargs": {"thinking": True, "enable_thinking": True},
            "max_tokens": 512,
        }
        resp = await client.post_json(
            interleaved_payload, timeout=config.timeout * 2
        )
        results.append(
            _assert_no_marker_leak(
                "interleaved_thinking_multi_section", resp.body, resp.status
            )
        )

        # Streaming variant: reassemble content/reasoning and check the same
        # no-marker-leak invariant on the streamed deltas.
        resp_stream = await client.post_streaming(
            interleaved_payload, read_timeout=config.timeout * 4
        )
        stream_content_parts: list[str] = []
        stream_reasoning_parts: list[str] = []
        stream_ok = resp_stream.status == 200
        for raw in resp_stream.chunks or []:
            if raw == "[DONE]":
                break
            try:
                obj = json.loads(raw)
                delta = obj.get("choices", [{}])[0].get("delta", {})
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                continue
            if delta.get("content"):
                stream_content_parts.append(delta["content"])
            rc = delta.get("reasoning_content") or delta.get("reasoning")
            if rc:
                stream_reasoning_parts.append(rc)
        stream_blob = "".join(stream_content_parts) + "".join(
            stream_reasoning_parts
        )
        stream_leaked = [m for m in leak_markers if m in stream_blob]
        if resp_stream.status >= 500 or resp_stream.error == "TIMEOUT":
            stream_verdict = Verdict.FAIL
            stream_detail = f"stream status {resp_stream.status}"
        elif stream_leaked:
            stream_verdict = Verdict.FAIL
            stream_detail = (
                f"markers leaked into streamed text: {stream_leaked}"
            )
        else:
            stream_verdict = Verdict.PASS if stream_ok else Verdict.INTERESTING
            stream_detail = f"status {resp_stream.status}"
        results.append(
            self.make_result(
                self.name,
                "interleaved_thinking_multi_section_streaming",
                stream_verdict,
                status_code=resp_stream.status,
                detail=stream_detail,
            )
        )

        # ----- 14. Orphaned tool-role messages (trimmed histories) -----
        # Agent frameworks (LangChain-style trimming) routinely resend a tool
        # result without the assistant ``tool_calls`` turn that produced it.
        # A strict chat template (e.g. MiniMax M3) calls ``raise_exception``;
        # the server must translate that into a clean 4xx, never a 500. Kimi's
        # template tolerates the shape and returns 200 -- also a pass. Only a
        # 5xx (or timeout) is a failure. Regression coverage for CENG-789.
        orphaned_tool_histories = {
            # Tool result with no preceding assistant turn at all.
            "tool_after_user": [
                {"role": "user", "content": "hi"},
                {
                    "role": "tool",
                    "tool_call_id": "call_x",
                    "content": "result",
                },
            ],
            # Tool result as the very first message.
            "tool_as_first_message": [
                {
                    "role": "tool",
                    "tool_call_id": "call_x",
                    "content": "result",
                },
            ],
            # Assistant turn present but with no matching ``tool_calls`` entry,
            # so the tool_call_id dangles.
            "dangling_tool_call_id": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "sure, one moment"},
                {
                    "role": "tool",
                    "tool_call_id": "call_missing",
                    "content": "result",
                },
            ],
        }

        for case_name, messages in orphaned_tool_histories.items():
            payload = {
                "model": model,
                "messages": messages,
                "tools": [valid_tool],
                "max_tokens": 10,
            }
            resp = await client.post_json(payload)
            results.append(
                self.make_result(
                    self.name,
                    f"orphaned_{case_name}",
                    Verdict.FAIL
                    if resp.status >= 500 or resp.error == "TIMEOUT"
                    else Verdict.PASS,
                    status_code=resp.status,
                    detail=f"Status {resp.status}"
                    + (f" error: {resp.error}" if resp.error else ""),
                )
            )
            # The streaming path returns HTTP 200 and reports the failure in an
            # error frame, so a status check alone would miss a 5xx regression;
            # inspect the frames for a >=500 error code.
            resp_stream = await client.post_streaming(payload)
            stream_5xx = False
            for chunk in resp_stream.chunks or []:
                if chunk == "[DONE]":
                    continue
                try:
                    err = json.loads(chunk).get("error")
                except (json.JSONDecodeError, AttributeError):
                    continue
                if err and str(err.get("code", "")).startswith("5"):
                    stream_5xx = True
                    break
            results.append(
                self.make_result(
                    self.name,
                    f"orphaned_{case_name}_streaming",
                    Verdict.FAIL
                    if resp_stream.status >= 500
                    or resp_stream.error == "TIMEOUT"
                    or stream_5xx
                    else Verdict.PASS,
                    status_code=resp_stream.status,
                    detail=(
                        "stream error frame (5xx)"
                        if stream_5xx
                        else f"Status {resp_stream.status}"
                    ),
                )
            )

        return results
