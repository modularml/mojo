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
Scenarios: Resource exhaustion
Target: OOM, KV cache exhaustion, context window overflow, memory leaks.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig


@register_scenario
class ResourceExhaustion(BaseScenario):
    name = "resource_exhaustion"
    description = "Context window overflow, KV cache pressure, prompt_logprobs memory, sustained load"
    tags = ["memory", "oom", "kv_cache", "context", "crash"]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results = []
        model = config.model
        mc = config.model_config

        # ----- 1. Context window boundary probing -----
        # Try increasingly large inputs to find the limit and see how the server handles overflow
        context_sizes = mc.context_boundary_probe_sizes
        last_success = 0

        for size in context_sizes:
            # ~4 chars per token rough estimate
            content = "word " * size
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 5,
            }
            resp = await client.post_json(payload, timeout=config.timeout * 2)

            if resp.status == 200:
                last_success = size
                verdict = Verdict.PASS
                detail = f"Accepted ~{size} tokens"
            elif 400 <= resp.status < 500:
                verdict = Verdict.PASS
                detail = f"Properly rejected at ~{size} tokens (last success: ~{last_success})"
            elif resp.status >= 500:
                verdict = Verdict.FAIL
                detail = f"Server error at ~{size} tokens"
            elif resp.error == "TIMEOUT":
                verdict = Verdict.FAIL
                detail = f"Hung at ~{size} tokens"
            else:
                verdict = Verdict.INTERESTING
                detail = f"Status {resp.status} at ~{size} tokens"

            results.append(
                self.make_result(
                    self.name,
                    f"context_{size}_tokens",
                    verdict,
                    status_code=resp.status,
                    elapsed_ms=resp.elapsed_ms,
                    detail=detail,
                )
            )

            # Don't continue if it's already failing hard
            if resp.status >= 500 or resp.error == "TIMEOUT":
                break

        # ----- 2. max_tokens requesting more than context allows -----
        medium_input = mc.medium_input_tokens
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "word " * medium_input}],
            "max_tokens": mc.max_position_embeddings * 2,
        }
        resp = await client.post_json(payload)
        results.append(
            self.make_result(
                self.name,
                "max_tokens_exceeds_context",
                Verdict.FAIL if resp.status >= 500 else Verdict.PASS,
                status_code=resp.status,
                detail=f"Large input + huge max_tokens: status {resp.status}",
            )
        )

        # ----- 3. prompt_logprobs memory amplification -----
        # vLLM issue #10325: prompt_logprobs can 5x memory usage
        logprob_input = medium_input // 2
        logprob_payload = {
            "model": model,
            "messages": [{"role": "user", "content": "word " * logprob_input}],
            "max_tokens": 10,
            "logprobs": True,
            "top_logprobs": 5,
        }
        resp = await client.post_json(logprob_payload)
        results.append(
            self.make_result(
                self.name,
                "logprobs_memory_amplification",
                Verdict.FAIL
                if resp.status >= 500 or resp.error == "TIMEOUT"
                else Verdict.PASS,
                status_code=resp.status,
                detail=f"{logprob_input} tokens + logprobs: status {resp.status}",
            )
        )

        # ----- 4. Concurrent large context requests (KV cache pressure) -----
        large_input = mc.large_input_tokens
        large_payloads = [
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": "word " * large_input}
                ],
                "max_tokens": 100,
            }
            for _ in range(20)
        ]
        responses = await client.concurrent_requests(
            large_payloads, max_concurrent=20
        )
        errors = sum(1 for r in responses if r.status >= 500 or r.status == 0)
        results.append(
            self.make_result(
                self.name,
                "concurrent_large_context_20",
                Verdict.FAIL
                if errors > 5
                else (Verdict.INTERESTING if errors > 0 else Verdict.PASS),
                detail=f"{errors}/20 failed under KV cache pressure",
            )
        )

        # ----- 5. Mixed short+long requests (prefill-decode interference) -----
        mixed_payloads = []
        for i in range(40):
            if i % 2 == 0:
                # Long context
                mixed_payloads.append(
                    {
                        "model": model,
                        "messages": [
                            {"role": "user", "content": "word " * medium_input}
                        ],
                        "max_tokens": 50,
                    }
                )
            else:
                # Short context
                mixed_payloads.append(
                    {
                        "model": model,
                        "messages": [{"role": "user", "content": "Say hi"}],
                        "max_tokens": 10,
                    }
                )

        responses = await client.concurrent_requests(
            mixed_payloads, max_concurrent=40
        )
        errors = sum(1 for r in responses if r.status >= 500 or r.status == 0)
        timeouts = sum(1 for r in responses if r.error == "TIMEOUT")

        # Check if short requests took abnormally long (prefill interference)
        short_latencies = [
            r.elapsed_ms
            for i, r in enumerate(responses)
            if i % 2 == 1 and r.status == 200
        ]
        # NOTE: long_latencies (i % 2 == 0) is captured but never compared
        # against short_latencies — likely an unfinished prefill-interference
        # check. Left as dead code removal until the comparison is implemented.

        if short_latencies:
            avg_short = sum(short_latencies) / len(short_latencies)
            max_short = max(short_latencies)
        else:
            avg_short = max_short = 0

        detail = f"{errors} errors, {timeouts} timeouts / 40"
        if avg_short > 0:
            detail += (
                f" | Short req avg={avg_short:.0f}ms max={max_short:.0f}ms"
            )

        results.append(
            self.make_result(
                self.name,
                "mixed_short_long_interference",
                Verdict.FAIL
                if errors > 10
                else (
                    Verdict.INTERESTING
                    if errors > 0 or max_short > 10000
                    else Verdict.PASS
                ),
                detail=detail,
            )
        )

        # ----- 6. Many multi-turn conversations (memory accumulation) -----
        long_conversation = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. " * 100,
                },
            ]
            + [
                {
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": f"Message {i}. " * 50,
                }
                for i in range(100)
            ],
            "max_tokens": 10,
        }
        resp = await client.post_json(long_conversation)
        results.append(
            self.make_result(
                self.name,
                "100_turn_conversation",
                Verdict.FAIL
                if resp.status >= 500 or resp.error == "TIMEOUT"
                else Verdict.PASS,
                status_code=resp.status,
                detail=f"100-turn conversation: status {resp.status}",
            )
        )

        # ----- 7. Sustained load (mini soak test) -----
        results.extend(
            await self._mini_soak(
                client, model, duration_sec=15, concurrency=10
            )
        )

        # ----- 8. Post-attack health check -----
        await asyncio.sleep(3)
        health = await client.health_check()
        results.append(
            self.make_result(
                self.name,
                "post_exhaustion_health_check",
                Verdict.PASS if health.status == 200 else Verdict.FAIL,
                status_code=health.status,
                detail="Healthy"
                if health.status == 200
                else f"Unhealthy: {health.error}",
            )
        )

        return results

    async def _mini_soak(
        self,
        client: FuzzClient,
        model: str,
        duration_sec: float,
        concurrency: int,
    ) -> list[ScenarioResult]:
        """Run sustained concurrent requests for a set duration."""
        end_time = time.perf_counter() + duration_sec
        total = 0
        errors = 0
        sem = asyncio.Semaphore(concurrency)

        async def _req() -> None:
            nonlocal total, errors
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "word " * 500}],
                "max_tokens": 20,
            }
            async with sem:
                r = await client.post_json(payload)
                total += 1
                if r.status >= 500 or r.status == 0:
                    errors += 1

        tasks = []
        while time.perf_counter() < end_time:
            tasks.append(asyncio.create_task(_req()))
            await asyncio.sleep(0.05)  # ~20 req/sec

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        error_rate = errors / total if total > 0 else 0
        verdict = (
            Verdict.FAIL
            if error_rate > 0.2
            else (Verdict.INTERESTING if error_rate > 0 else Verdict.PASS)
        )

        return [
            self.make_result(
                self.name,
                f"mini_soak_{duration_sec}s",
                verdict,
                detail=f"{errors}/{total} failed ({error_rate * 100:.1f}%) over {duration_sec}s at concurrency {concurrency}",
            )
        ]
