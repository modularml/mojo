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
Scenario: reasoning-enabled model skips thinking — content must not leak into reasoning

When ``enable_thinking`` is on but the model decides the answer is simple
enough to skip its thinking phase, the response content must appear in
``message.content``, not in ``message.reasoning``.

Uses a multi-turn tool-calling conversation where the final assistant turn
is a short prose answer that models typically produce without reasoning.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig

# Multi-turn conversation that elicits a short answer after tool results.
# The model has already done the arithmetic via tools; the final turn is
# a simple summary that models often produce without a thinking block.
REPRODUCER_MESSAGES: list[dict[str, Any]] = [
    {
        "role": "system",
        "content": (
            "You are a US-based personal finance advisor. When the user asks "
            "a numeric question, USE the calculator tool \u2014 never do "
            "arithmetic in your head. When asked for current rates, USE "
            "web_search or get_market_rate. Think step-by-step before "
            "answering. When the user asks for a 'budget', return ONLY the "
            "requested JSON."
        ),
    },
    {
        "role": "user",
        "content": (
            "I make $3,500/month after tax. My monthly expenses are: "
            "$1,100 rent, $300 food, $80 transport, $150 utilities, "
            "$100 insurance, $70 entertainment. What's my monthly surplus? "
            "Use the calculator."
        ),
    },
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_51ffd468207b430dac538958",
                "function": {
                    "arguments": '{"expression": "1100 + 300 + 80 + 150 + 100 + 70"}',
                    "name": "calculator",
                },
                "type": "function",
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_51ffd468207b430dac538958",
        "content": '{"value": 1800}',
    },
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_a25a4d400ac64a5db4d1aef7",
                "function": {
                    "arguments": '{"expression": "3500 - 1800"}',
                    "name": "calculator",
                },
                "type": "function",
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_a25a4d400ac64a5db4d1aef7",
        "content": '{"value": 1700}',
    },
    {"role": "assistant", "content": "Your monthly surplus is $1,700."},
    {
        "role": "user",
        "content": (
            "Now summarize the above as a JSON object matching this shape: "
            "monthly_income, monthly_expenses, monthly_surplus, notes. "
            "Use the calculator if needed."
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"monthly_income":3500,"monthly_expenses":1800,'
            '"monthly_surplus":1700,"notes":"Surplus calculated based on '
            'provided monthly expenses."}'
        ),
    },
    {
        "role": "user",
        "content": (
            "Explain in 3-4 sentences what 'pay yourself first' means and "
            "whether my current surplus enables it. No JSON, just prose."
        ),
    },
    {
        "role": "assistant",
        "content": (
            '"Paying yourself first" is a financial strategy where you '
            "prioritize your savings and investments by allocating a set "
            "amount of money to them as soon as you receive your paycheck, "
            "before paying any other bills. This ensures that your future "
            "financial goals are met consistently rather than simply saving "
            "whatever happens to be left over at the end of the month. With "
            "a monthly surplus of $1,700, you are in an excellent position "
            "to implement this strategy. You can easily automate a "
            "significant portion of that surplus into a savings or investment "
            "account to build wealth efficiently."
        ),
    },
    {
        "role": "user",
        "content": (
            "What's the current HYSA APY? Use the get_market_rate tool. "
            "Then tell me, in one sentence, what $20,000 in such an HYSA "
            "earns in a year. Use the calculator for the math."
        ),
    },
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_33c8776d409c4f9dbceb45b3",
                "function": {
                    "arguments": '{"instrument": "hysa_apy"}',
                    "name": "get_market_rate",
                },
                "type": "function",
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_33c8776d409c4f9dbceb45b3",
        "content": '{"instrument": "hysa_apy", "rate_pct": 4.5}',
    },
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_37f30e1ecc45481b87feb556",
                "function": {
                    "arguments": '{"expression": "20000 * 0.045"}',
                    "name": "calculator",
                },
                "type": "function",
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_37f30e1ecc45481b87feb556",
        "content": '{"value": 900.0}',
    },
]

REPRODUCER_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression and return its numeric result.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information (rates, prices, news).",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_rate",
            "description": "Get the current rate for a financial instrument.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instrument": {
                        "type": "string",
                        "enum": [
                            "hysa_apy",
                            "mortgage_30yr",
                            "fed_funds",
                            "10yr_treasury",
                        ],
                    }
                },
                "required": ["instrument"],
            },
        },
    },
]


def _make_payload(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": REPRODUCER_MESSAGES,
        "tools": REPRODUCER_TOOLS,
        "tool_choice": "auto",
        "chat_template_kwargs": {"thinking": True, "enable_thinking": True},
        "max_tokens": 1024,
        "temperature": 0,
    }


@register_scenario
class ReasoningSkippedContentRouting(BaseScenario):
    name = "reasoning_skipped_content_routing"
    description = (
        "when thinking is enabled but model skips reasoning, "
        "content must not leak into the reasoning field"
    )
    tags = ["reasoning", "thinking", "streaming", "regression"]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        model = config.model
        payload = _make_payload(model)

        # ----- 1. Non-streaming: content vs reasoning field check -----
        resp = await client.post_json(payload, timeout=config.timeout * 2)
        if resp.status == 200:
            try:
                body = json.loads(resp.body)
                msg = body["choices"][0]["message"]
                content = (msg.get("content") or "").strip()
                reasoning = (msg.get("reasoning") or "").strip()
                has_tool_calls = bool(msg.get("tool_calls"))

                if content or has_tool_calls:
                    verdict = Verdict.PASS
                    detail = f"content present ({len(content)} chars)"
                    if has_tool_calls:
                        detail += " + tool_calls"
                elif reasoning:
                    verdict = Verdict.FAIL
                    detail = (
                        f"content empty but reasoning has {len(reasoning)} "
                        f"chars — content leaked into reasoning field "
                        f"(snippet: {reasoning[:120]!r})"
                    )
                else:
                    verdict = Verdict.INTERESTING
                    detail = "both content and reasoning are empty"
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                verdict = Verdict.FAIL
                detail = f"malformed response: {e}"
        else:
            verdict = (
                Verdict.FAIL if resp.status >= 500 else Verdict.INTERESTING
            )
            detail = f"HTTP {resp.status}"

        results.append(
            self.make_result(
                self.name,
                "nonstreaming_content_routing",
                verdict,
                status_code=resp.status,
                elapsed_ms=resp.elapsed_ms,
                detail=detail,
            )
        )

        # ----- 2. Streaming: same check on reassembled chunks -----
        stream_payload = {**payload, "stream": True}
        resp = await client.post_streaming(
            stream_payload, read_timeout=config.timeout * 4
        )
        if resp.status == 200:
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            has_tool_calls = False
            for raw in resp.chunks or []:
                if raw == "[DONE]":
                    break
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                ct = delta.get("content") or ""
                rc = (
                    delta.get("reasoning_content")
                    or delta.get("reasoning")
                    or ""
                )
                if ct:
                    content_parts.append(ct)
                if rc:
                    reasoning_parts.append(rc)
                if delta.get("tool_calls"):
                    has_tool_calls = True

            content = "".join(content_parts).strip()
            reasoning = "".join(reasoning_parts).strip()

            if content or has_tool_calls:
                verdict = Verdict.PASS
                detail = f"streaming content present ({len(content)} chars)"
                if has_tool_calls:
                    detail += " + tool_calls"
            elif reasoning:
                verdict = Verdict.FAIL
                detail = (
                    f"streaming content empty but reasoning has "
                    f"{len(reasoning)} chars — leaked "
                    f"(snippet: {reasoning[:120]!r})"
                )
            else:
                verdict = Verdict.INTERESTING
                detail = "streaming: both content and reasoning are empty"
        else:
            verdict = (
                Verdict.FAIL if resp.status >= 500 else Verdict.INTERESTING
            )
            detail = f"streaming HTTP {resp.status}"

        results.append(
            self.make_result(
                self.name,
                "streaming_content_routing",
                verdict,
                status_code=resp.status,
                elapsed_ms=resp.elapsed_ms,
                detail=detail,
            )
        )

        return results
