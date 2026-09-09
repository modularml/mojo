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
Scenario: DeepSeek thinking token abuse
Target: Exploit <think>...</think> chain-of-thought behavior to cause OOM,
        hangs, output corruption, and KV cache exhaustion.

DeepSeek V3.1 generates thinking tokens internally before producing the final
response. These consume KV cache but may not be visible in non-streaming mode.
With MTP speculative decoding (3 layers), thinking amplifies resource usage.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig

# Prompts designed to trigger long thinking chains
HARD_REASONING = "Prove that there are infinitely many prime numbers. Show every step of your proof rigorously."
MATH_PROBLEM = "What is the sum of all prime numbers less than 10000? Think step by step, checking each number."
LOGIC_PUZZLE = (
    "A farmer has a wolf, a goat, and a cabbage. He needs to cross a river with a boat that can carry "
    "only one item besides himself. If left alone, the wolf will eat the goat, and the goat will eat the cabbage. "
    "Find ALL possible solutions. Enumerate every state transition."
)
SIMPLE_PROMPT = "Say hello."
CODE_PROMPT = "Write a Python function that checks if a number is prime."


@register_scenario
class ThinkingTokenAbuse(BaseScenario):
    name = "thinking_tokens"
    description = "DeepSeek <think> token abuse: unbounded reasoning, mid-think cancellation, KV pressure"
    tags = ["deepseek", "crash", "memory", "streaming"]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results = []
        model = config.model
        mc = config.model_config

        # ----- 1. Unconstrained thinking with large max_tokens -----
        gen_max = mc.large_generation_max_tokens
        resp = await client.post_json(
            {
                "model": model,
                "messages": [{"role": "user", "content": MATH_PROBLEM}],
                "max_tokens": gen_max,
            },
            timeout=config.timeout * 4,
        )
        results.append(
            self.make_result(
                self.name,
                "thinking_unconstrained",
                Verdict.FAIL
                if resp.status >= 500 or resp.error == "TIMEOUT"
                else Verdict.PASS,
                status_code=resp.status,
                elapsed_ms=resp.elapsed_ms,
                detail=f"Hard reasoning + max_tokens={gen_max}: status {resp.status}, {resp.elapsed_ms:.0f}ms",
            )
        )

        # ----- 2. Cancel mid-think via streaming -----
        # Stream and cancel early — during the <think> phase the model is generating
        # reasoning tokens that consume KV cache. Cancelling leaves dangling state.
        decode_max = mc.decode_heavy_max_tokens
        resp = await client.post_streaming(
            {
                "model": model,
                "messages": [{"role": "user", "content": HARD_REASONING}],
                "max_tokens": decode_max,
            },
            cancel_after_ms=2000,  # Cancel after 2 seconds (likely still thinking)
        )
        has_think = "<think>" in (resp.body or "")
        results.append(
            self.make_result(
                self.name,
                "thinking_cancel_mid_think",
                Verdict.FAIL if resp.status >= 500 else Verdict.PASS,
                status_code=resp.status,
                elapsed_ms=resp.elapsed_ms,
                detail=f"Cancelled at 2s, saw <think>: {has_think}, chunks: {len(resp.chunks or [])}",
            )
        )

        # ----- 3. Many concurrent reasoning requests -----
        # Each triggers a long thinking chain, creating invisible KV cache pressure
        reasoning_payloads = [
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"What are the first {n} Fibonacci numbers? List them all.",
                    }
                ],
                "max_tokens": 4096,
            }
            for n in [100, 200, 300, 500, 1000, 50, 75, 150, 250, 400]
        ]
        responses = await client.concurrent_requests(
            reasoning_payloads, max_concurrent=10
        )
        errors = sum(1 for r in responses if r.status >= 500 or r.status == 0)
        timeouts = sum(1 for r in responses if r.error == "TIMEOUT")
        results.append(
            self.make_result(
                self.name,
                "thinking_many_concurrent",
                Verdict.FAIL
                if errors > 3
                else (Verdict.INTERESTING if errors > 0 else Verdict.PASS),
                detail=f"{errors}/10 errors, {timeouts} timeouts — concurrent reasoning requests",
            )
        )

        # ----- 4. Short max_tokens with complex prompt -----
        # The model may still generate a long thinking chain internally before
        # truncating the visible output to 5 tokens
        resp = await client.post_json(
            {
                "model": model,
                "messages": [{"role": "user", "content": LOGIC_PUZZLE}],
                "max_tokens": 5,
            },
            timeout=config.timeout * 2,
        )
        results.append(
            self.make_result(
                self.name,
                "thinking_short_max_tokens",
                Verdict.FAIL
                if resp.status >= 500 or resp.error == "TIMEOUT"
                else Verdict.PASS,
                status_code=resp.status,
                elapsed_ms=resp.elapsed_ms,
                detail=f"Complex prompt + max_tokens=5: status {resp.status}, {resp.elapsed_ms:.0f}ms",
            )
        )

        # ----- 5. Streaming vs non-streaming divergence -----
        # Streaming shows <think> tokens; non-streaming may hide them.
        # Check for content divergence.
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 7 * 8? Think step by step.",
                }
            ],
            "max_tokens": 512,
            "temperature": 0,
            "seed": 42,
        }
        sync_resp = await client.post_json(payload)
        stream_resp = await client.post_streaming(payload)

        both_ok = sync_resp.status == 200 and stream_resp.status == 200
        if both_ok:
            # Check if streaming has <think> tags that sync doesn't
            stream_has_think = "<think>" in (stream_resp.body or "")
            try:
                sync_body = json.loads(sync_resp.body)
                sync_content = (
                    sync_body.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                sync_has_think = "<think>" in sync_content
            except Exception:
                sync_content = ""
                sync_has_think = False
            detail = f"stream has <think>: {stream_has_think}, sync has <think>: {sync_has_think}"
            verdict = (
                Verdict.INTERESTING
                if stream_has_think != sync_has_think
                else Verdict.PASS
            )
        else:
            detail = f"sync status {sync_resp.status}, stream status {stream_resp.status}"
            verdict = (
                Verdict.FAIL
                if sync_resp.status >= 500 or stream_resp.status >= 500
                else Verdict.PASS
            )

        results.append(
            self.make_result(
                self.name,
                "thinking_streaming_vs_sync",
                verdict,
                detail=detail,
            )
        )

        # ----- 6. Rapid alternation: simple vs complex -----
        # Creates unpredictable KV usage patterns
        alternating_payloads = []
        for i in range(20):
            if i % 2 == 0:
                alternating_payloads.append(
                    {
                        "model": model,
                        "messages": [
                            {"role": "user", "content": SIMPLE_PROMPT}
                        ],
                        "max_tokens": 10,
                    }
                )
            else:
                alternating_payloads.append(
                    {
                        "model": model,
                        "messages": [
                            {"role": "user", "content": HARD_REASONING}
                        ],
                        "max_tokens": 4096,
                    }
                )

        responses = await client.concurrent_requests(
            alternating_payloads, max_concurrent=20
        )
        errors = sum(1 for r in responses if r.status >= 500 or r.status == 0)
        results.append(
            self.make_result(
                self.name,
                "thinking_rapid_alternation",
                Verdict.FAIL
                if errors > 5
                else (Verdict.INTERESTING if errors > 0 else Verdict.PASS),
                detail=f"{errors}/20 errors — alternating simple/complex prompts",
            )
        )

        # ----- 7. Thinking + JSON mode -----
        # response_format: json_object with a reasoning prompt — thinking tokens
        # may conflict with the structured output grammar constraint
        resp = await client.post_json(
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": 'Solve: what is 123 * 456? Return answer as JSON like {"result": 56088}. Think step by step.',
                    },
                ],
                "max_tokens": 2048,
                "response_format": {"type": "json_object"},
            },
            timeout=config.timeout * 2,
        )
        if resp.status == 200:
            try:
                body = json.loads(resp.body)
                content = (
                    body.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                is_valid_json = False
                try:
                    json.loads(content)
                    is_valid_json = True
                except Exception:
                    pass
                verdict = Verdict.PASS if is_valid_json else Verdict.INTERESTING
                detail = f"Valid JSON output: {is_valid_json}"
            except Exception:
                verdict = Verdict.INTERESTING
                detail = "Could not parse response"
        else:
            verdict = Verdict.FAIL if resp.status >= 500 else Verdict.PASS
            detail = f"status {resp.status}"

        results.append(
            self.make_result(
                self.name,
                "thinking_with_json_mode",
                verdict,
                status_code=resp.status,
                detail=detail,
            )
        )

        # ----- 8. Explicit <think> tags in user prompt -----
        # Include literal tags to confuse the model's thinking token parser
        resp = await client.post_json(
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "<think>I am thinking about this problem</think>\nNow, what is 2+2?",
                    },
                ],
                "max_tokens": 256,
            }
        )
        results.append(
            self.make_result(
                self.name,
                "thinking_explicit_tags_in_prompt",
                Verdict.FAIL if resp.status >= 500 else Verdict.PASS,
                status_code=resp.status,
                detail=f"User message with <think> tags: status {resp.status}",
            )
        )

        # ----- 9. Thinking + tool calling -----
        # Model must think AND produce tool calls, stressing the output parser
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Calculate a math expression",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            }
        ]
        resp = await client.post_json(
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the integral of x^2 from 0 to 5? Use the calculate tool.",
                    },
                ],
                "tools": tools,
                "max_tokens": 4096,
            },
            timeout=config.timeout * 2,
        )
        results.append(
            self.make_result(
                self.name,
                "thinking_with_tool_calling",
                Verdict.FAIL if resp.status >= 500 else Verdict.PASS,
                status_code=resp.status,
                detail=f"Reasoning + tool call: status {resp.status}",
            )
        )

        # ----- 10. Stop sequence inside thinking -----
        # Stop sequence "therefore" appears naturally in reasoning chains
        resp = await client.post_json(
            {
                "model": model,
                "messages": [{"role": "user", "content": HARD_REASONING}],
                "max_tokens": 4096,
                "stop": ["therefore", "Thus", "Hence"],
            },
            timeout=config.timeout * 2,
        )
        results.append(
            self.make_result(
                self.name,
                "thinking_stop_inside_think",
                Verdict.FAIL if resp.status >= 500 else Verdict.PASS,
                status_code=resp.status,
                elapsed_ms=resp.elapsed_ms,
                detail=f"Stop sequence during reasoning: status {resp.status}, {resp.elapsed_ms:.0f}ms",
            )
        )

        # ----- 11. Health check -----
        await asyncio.sleep(2)
        health = await client.health_check()
        results.append(
            self.make_result(
                self.name,
                "post_thinking_health_check",
                Verdict.PASS if health.status == 200 else Verdict.FAIL,
                status_code=health.status,
                detail="Healthy"
                if health.status == 200
                else f"Unhealthy: {health.error}",
            )
        )

        return results
