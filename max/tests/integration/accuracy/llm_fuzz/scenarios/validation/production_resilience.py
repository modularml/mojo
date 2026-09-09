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
"""Production resilience: edge cases that occur in real deployments.

Tests proper handling of max_tokens truncation, empty/minimal prompts,
streaming vs non-streaming consistency, token count accuracy, long
multi-turn conversations, and tool error / oversized tool result handling.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from helpers import (
    collect_stream,
    make_tool,
)

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig
    from validator_client import ValidatorClient


@register_scenario
class ProductionResilience(BaseScenario):
    name = "production_resilience"
    description = (
        "Production resilience: max_tokens truncation, empty/minimal prompts, "
        "streaming vs non-streaming parity, token counts, long conversations, "
        "tool errors and oversized tool results"
    )
    tags = ["validation", "production"]
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
            self._test_max_tokens_truncation,
            self._test_max_tokens_json_truncation,
            self._test_empty_user_message,
            self._test_single_char_prompt,
            self._test_streaming_vs_nonstreaming_match,
            self._test_token_count_sanity,
            self._test_streaming_token_count,
            self._test_long_conversation_10_turns,
            self._test_tool_error_in_history,
            self._test_oversized_tool_result,
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
    # Individual tests (sync, run inside executor)
    # ------------------------------------------------------------------

    def _test_max_tokens_truncation(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """Set max_tokens=5 with a long prompt. finish_reason must be 'length'."""
        test_name = "max_tokens_truncation"
        resp = v.chat(
            [
                {
                    "role": "user",
                    "content": (
                        "Write a detailed essay about the history of computing, "
                        "covering all major milestones from the abacus to modern AI."
                    ),
                }
            ],
            max_tokens=5,
        )
        fr = resp.choices[0].finish_reason
        if fr == "length":
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail="finish_reason='length' as expected with max_tokens=5",
                )
            ]
        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.FAIL,
                detail=f"Expected finish_reason='length', got {fr!r}",
            )
        ]

    def _test_max_tokens_json_truncation(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """Structured output with max_tokens=10. Should truncate, not crash."""
        test_name = "max_tokens_json_truncation"
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "items": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "description", "items"],
            "additionalProperties": False,
        }
        resp = v.so_chat(
            [
                {
                    "role": "user",
                    "content": (
                        "Return a title, a long description (at least 500 words), "
                        "and a list of 20 items."
                    ),
                }
            ],
            schema,
            max_tokens=10,
        )
        fr = resp.choices[0].finish_reason
        if fr == "length":
            content = resp.choices[0].message.content or ""
            # Output may be incomplete JSON -- that is expected with truncation
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail=f"finish_reason='length', truncated output ({len(content)} chars)",
                )
            ]
        if fr == "stop":
            # Some servers may still complete within the budget
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail="finish_reason='stop' (server completed within tight budget)",
                )
            ]
        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.INTERESTING,
                detail=f"Unexpected finish_reason={fr!r} with max_tokens=10",
            )
        ]

    def _test_empty_user_message(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """Send empty string as user content. Should not crash (200 or 400)."""
        test_name = "empty_user_message"
        try:
            resp = v.chat(
                [{"role": "user", "content": ""}],
                max_tokens=64,
            )
            # Got a 200 response -- that is acceptable
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail=f"Server returned 200, finish_reason={resp.choices[0].finish_reason!r}",
                )
            ]
        except Exception as e:
            err_str = str(e).lower()
            # 400-level errors are acceptable -- the server handled it gracefully
            if "400" in err_str or "bad request" in err_str or "422" in err_str:
                return [
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.PASS,
                        detail=f"Server returned expected error for empty message: {e}",
                    )
                ]
            # 500-level or connection errors are failures
            if "500" in err_str or "502" in err_str or "503" in err_str:
                return [
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.FAIL,
                        detail=f"Server crashed on empty message: {e}",
                    )
                ]
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.ERROR,
                    error=f"Unexpected error on empty message: {e}",
                )
            ]

    def _test_single_char_prompt(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """Send 'x' as user content. Should produce a valid response."""
        test_name = "single_char_prompt"
        resp = v.chat(
            [{"role": "user", "content": "x"}],
            max_tokens=128,
        )
        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning = (
            getattr(msg, "reasoning_content", None)
            or getattr(msg, "reasoning", None)
            or ""
        )
        has_output = bool(content or reasoning)
        fr = resp.choices[0].finish_reason
        if fr in ("stop", "length") and has_output:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail=f"Got valid response ({len(content)} chars), finish_reason={fr!r}",
                )
            ]
        if fr in ("stop", "length"):
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.INTERESTING,
                    detail=f"Response was empty with finish_reason={fr!r}",
                )
            ]
        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.FAIL,
                detail=f"Unexpected finish_reason={fr!r}",
            )
        ]

    def _test_streaming_vs_nonstreaming_match(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """Same request via streaming and non-streaming with temperature=0, seed=42.

        Content should be similar (both valid), though not necessarily identical.
        """
        test_name = "streaming_vs_nonstreaming_match"
        messages = [
            {
                "role": "user",
                "content": "What is 2 + 2? Reply with just the number.",
            }
        ]
        kwargs: dict[str, Any] = {"temperature": 0, "seed": 42}

        # Non-streaming
        ns_resp = v.chat(messages, max_tokens=64, **kwargs)
        ns_content = ns_resp.choices[0].message.content or ""
        ns_fr = ns_resp.choices[0].finish_reason

        # Streaming
        s_result = v.chat_stream(messages, max_tokens=64, **kwargs)
        s_content = s_result["content"]
        s_fr = s_result["finish_reason"]

        issues: list[str] = []
        if not ns_content.strip():
            issues.append("non-streaming produced empty content")
        if not s_content.strip():
            issues.append("streaming produced empty content")
        if ns_fr not in ("stop", "length"):
            issues.append(f"non-streaming finish_reason={ns_fr!r}")
        if s_fr not in ("stop", "length"):
            issues.append(f"streaming finish_reason={s_fr!r}")

        if issues:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail="; ".join(issues),
                )
            ]

        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail=(
                    f"non-streaming: {ns_content.strip()!r} ({ns_fr}), "
                    f"streaming: {s_content.strip()!r} ({s_fr})"
                ),
            )
        ]

    def _test_token_count_sanity(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """Non-streaming response: verify usage token counts are sane."""
        test_name = "token_count_sanity"
        resp = v.chat(
            [{"role": "user", "content": "Tell me a short joke."}],
            max_tokens=256,
        )
        usage = resp.usage
        if usage is None:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail="Response has no usage field",
                )
            ]

        issues: list[str] = []
        if usage.prompt_tokens <= 0:
            issues.append(f"prompt_tokens={usage.prompt_tokens} (expected > 0)")
        if usage.completion_tokens <= 0:
            issues.append(
                f"completion_tokens={usage.completion_tokens} (expected > 0)"
            )
        expected_total = usage.prompt_tokens + usage.completion_tokens
        if usage.total_tokens != expected_total:
            issues.append(
                f"total_tokens={usage.total_tokens} != "
                f"prompt({usage.prompt_tokens}) + completion({usage.completion_tokens}) = {expected_total}"
            )

        if issues:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail="; ".join(issues),
                )
            ]
        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail=(
                    f"prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, "
                    f"total={usage.total_tokens}"
                ),
            )
        ]

    def _test_streaming_token_count(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """Streaming with stream_options include_usage. Verify usage chunk has non-zero counts."""
        test_name = "streaming_token_count"
        stream = v.chat(
            [{"role": "user", "content": "Name three colors."}],
            max_tokens=128,
            stream=True,
            stream_options={"include_usage": True},
        )
        result = collect_stream(stream)
        usage = result.get("usage")

        if usage is None:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail="No usage chunk in streaming response with include_usage=True",
                )
            ]

        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        issues: list[str] = []
        if prompt_tokens <= 0:
            issues.append(f"prompt_tokens={prompt_tokens} (expected > 0)")
        if completion_tokens <= 0:
            issues.append(
                f"completion_tokens={completion_tokens} (expected > 0)"
            )

        if issues:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail="; ".join(issues),
                )
            ]
        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail=f"streaming usage: prompt={prompt_tokens}, completion={completion_tokens}",
            )
        ]

    def _test_long_conversation_10_turns(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """10-turn conversation, each turn adds to messages. Model should still respond at turn 10."""
        test_name = "long_conversation_10_turns"
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "You are a helpful math tutor. Keep answers brief.",
            },
        ]

        prompts = [
            "What is 1 + 1?",
            "Now add 3 to that result.",
            "Multiply by 2.",
            "Subtract 4.",
            "What is the square root of that?",
            "Round to the nearest integer.",
            "Is that a prime number?",
            "What is the next prime after it?",
            "Multiply those two primes together.",
            "Is the result even or odd?",
        ]

        last_content = ""
        for i, prompt in enumerate(prompts):
            messages.append({"role": "user", "content": prompt})
            try:
                resp = v.chat(messages, max_tokens=256)
            except Exception as e:
                return [
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.FAIL,
                        detail=f"Turn {i + 1} failed with error: {e}",
                    )
                ]

            content = resp.choices[0].message.content or ""
            reasoning = (
                getattr(resp.choices[0].message, "reasoning_content", None)
                or getattr(resp.choices[0].message, "reasoning", None)
                or ""
            )
            if not content and not reasoning:
                return [
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.FAIL,
                        detail=f"Turn {i + 1}: model produced empty response",
                    )
                ]

            messages.append({"role": "assistant", "content": content})
            last_content = content

        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.PASS,
                detail=f"All 10 turns produced responses. Last: {last_content[:80]!r}",
            )
        ]

    def _test_tool_error_in_history(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """Send a tool result with content='Error: service unavailable'. Model should still respond."""
        test_name = "tool_error_in_history"
        tool = make_tool(
            "fetch_data",
            {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
                "additionalProperties": False,
            },
        )
        messages = [
            {
                "role": "user",
                "content": "Fetch data from https://api.example.com/data",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_err_001",
                        "type": "function",
                        "function": {
                            "name": "fetch_data",
                            "arguments": json.dumps(
                                {"url": "https://api.example.com/data"}
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_err_001",
                "content": "Error: service unavailable",
            },
        ]

        resp = v.chat(messages, tools=[tool], max_tokens=512)
        msg = resp.choices[0].message
        content = msg.content or ""
        has_tc = bool(msg.tool_calls)
        reasoning = (
            getattr(msg, "reasoning_content", None)
            or getattr(msg, "reasoning", None)
            or ""
        )

        if content or has_tc or reasoning:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail=(
                        f"Model responded after tool error "
                        f"(content={len(content)} chars, tool_calls={has_tc})"
                    ),
                )
            ]
        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.FAIL,
                detail="Model produced no output after receiving tool error",
            )
        ]

    def _test_oversized_tool_result(
        self, v: ValidatorClient
    ) -> list[ScenarioResult]:
        """Tool result with 16K characters. Should not crash."""
        test_name = "oversized_tool_result"
        tool = make_tool(
            "get_report",
            {
                "type": "object",
                "properties": {"report_id": {"type": "string"}},
                "required": ["report_id"],
                "additionalProperties": False,
            },
        )

        # Build a 16K character tool result
        large_content = json.dumps(
            {
                "report_id": "rpt_001",
                "sections": [
                    {
                        "title": f"Section {i}",
                        "body": f"This is the content of section {i}. " * 40,
                    }
                    for i in range(40)
                ],
            }
        )
        # Pad to ensure we hit at least 16K
        if len(large_content) < 16384:
            large_content += " " * (16384 - len(large_content))

        messages = [
            {"role": "user", "content": "Get report rpt_001 and summarize it."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_big_001",
                        "type": "function",
                        "function": {
                            "name": "get_report",
                            "arguments": json.dumps({"report_id": "rpt_001"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_big_001",
                "content": large_content,
            },
        ]

        try:
            resp = v.chat(messages, tools=[tool], max_tokens=512)
        except Exception as e:
            err_str = str(e).lower()
            # 400-level is acceptable (server rejected oversized input gracefully)
            if "400" in err_str or "422" in err_str or "too large" in err_str:
                return [
                    self.make_result(
                        self.name,
                        test_name,
                        Verdict.PASS,
                        detail=f"Server rejected oversized tool result gracefully: {e}",
                    )
                ]
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.FAIL,
                    detail=f"Server crashed on oversized tool result ({len(large_content)} chars): {e}",
                )
            ]

        msg = resp.choices[0].message
        content = msg.content or ""
        has_tc = bool(msg.tool_calls)
        reasoning = (
            getattr(msg, "reasoning_content", None)
            or getattr(msg, "reasoning", None)
            or ""
        )

        if content or has_tc or reasoning:
            return [
                self.make_result(
                    self.name,
                    test_name,
                    Verdict.PASS,
                    detail=(
                        f"Server handled {len(large_content)}-char tool result "
                        f"(content={len(content)} chars, tool_calls={has_tc})"
                    ),
                )
            ]
        return [
            self.make_result(
                self.name,
                test_name,
                Verdict.INTERESTING,
                detail=f"Server returned empty response for {len(large_content)}-char tool result",
            )
        ]
