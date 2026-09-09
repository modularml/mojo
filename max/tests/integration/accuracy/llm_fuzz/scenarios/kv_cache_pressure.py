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
Scenario: KV cache pressure
Target: OOM via prefix cache pollution, context window boundary probing,
        concurrent KV cache exhaustion, cache eviction races, and page
        boundary allocation edge cases.

Test sizes adapt to model config (context window, max_num_tokens).
Use --max-context-length / --max-num-tokens or let the tool auto-detect
from HuggingFace.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig


@register_scenario
class KVCachePressure(BaseScenario):
    name = "kv_cache_pressure"
    description = "Prefix cache pollution, context boundary probing, concurrent KV exhaustion"
    tags = ["kv_cache", "memory", "oom", "crash"]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results = []
        model = config.model
        mc = config.model_config

        # ----- 1. Prefix cache pollution with unique system prompts -----
        # Each request uses a unique long system prompt, filling prefix cache with
        # entries that will never be reused (vLLM #35191)
        unique_payloads = []
        for i in range(50):
            # Each ~4K tokens (unique so prefix cache can't reuse)
            sys_prompt = (
                f"System configuration #{i}: "
                + f"unique-context-{i}-{'word ' * 1000}"
            )
            unique_payloads.append(
                {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": "Reply OK"},
                    ],
                    "max_tokens": 5,
                }
            )

        responses = await client.concurrent_requests(
            unique_payloads, max_concurrent=10
        )
        errors = sum(1 for r in responses if r.status >= 500 or r.status == 0)
        timeouts = sum(1 for r in responses if r.error == "TIMEOUT")
        results.append(
            self.make_result(
                self.name,
                "prefix_cache_pollution_50_unique",
                Verdict.FAIL
                if errors > 10
                else (Verdict.INTERESTING if errors > 0 else Verdict.PASS),
                detail=f"{errors}/50 errors, {timeouts} timeouts — unique system prompts to pollute prefix cache",
            )
        )

        # ----- 2. Prefix cache thrashing -----
        # Alternate between two long system prompts rapidly to force eviction/refill cycles
        prompt_a = "You are assistant Alpha. " * 500  # ~2K tokens
        prompt_b = "You are assistant Bravo. " * 500  # ~2K tokens
        thrash_payloads = []
        for i in range(40):
            prompt = prompt_a if i % 2 == 0 else prompt_b
            thrash_payloads.append(
                {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": "Reply OK"},
                    ],
                    "max_tokens": 5,
                }
            )

        responses = await client.concurrent_requests(
            thrash_payloads, max_concurrent=10
        )
        errors = sum(1 for r in responses if r.status >= 500 or r.status == 0)
        results.append(
            self.make_result(
                self.name,
                "prefix_cache_thrashing",
                Verdict.FAIL
                if errors > 5
                else (Verdict.INTERESTING if errors > 0 else Verdict.PASS),
                detail=f"{errors}/40 errors — alternating system prompts to thrash prefix cache",
            )
        )

        # ----- 3. Context window boundary: near max_seq_len -----
        # Probe near the model's context window boundary.
        for target_tokens in mc.context_near_boundary_sizes:
            content = (
                "w " * target_tokens
            )  # ~2 chars per "word" = roughly 1 token each
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 5,
            }
            resp = await client.post_json(payload, timeout=config.timeout * 4)

            if resp.status == 200:
                verdict = Verdict.PASS
                detail = f"Accepted ~{target_tokens} tokens"
            elif 400 <= resp.status < 500:
                verdict = Verdict.PASS
                detail = f"Properly rejected at ~{target_tokens} tokens"
            elif resp.status >= 500:
                verdict = Verdict.FAIL
                detail = f"Server error at ~{target_tokens} tokens"
            elif resp.error == "TIMEOUT":
                verdict = Verdict.FAIL
                detail = f"Hung at ~{target_tokens} tokens"
            else:
                verdict = Verdict.INTERESTING
                detail = f"Status {resp.status} at ~{target_tokens} tokens"

            results.append(
                self.make_result(
                    self.name,
                    f"context_boundary_{target_tokens}",
                    verdict,
                    status_code=resp.status,
                    elapsed_ms=resp.elapsed_ms,
                    detail=detail,
                )
            )
            # Stop if server is crashing
            if resp.status >= 500 or resp.error == "TIMEOUT":
                break

        # ----- 4. Context + 1: exceed max_seq_len -----
        over_limit = mc.context_over_limit_size
        content = "w " * over_limit
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 5,
        }
        resp = await client.post_json(payload, timeout=config.timeout * 2)
        results.append(
            self.make_result(
                self.name,
                "context_plus_one",
                Verdict.FAIL if resp.status >= 500 else Verdict.PASS,
                status_code=resp.status,
                detail=f"~{over_limit:,} tokens (over limit): status {resp.status}",
            )
        )

        # ----- 5. Concurrent requests exceeding KV cache capacity -----
        # Send 10 large requests at once to overflow KV cache
        large_input = mc.large_input_tokens
        large_payloads = [
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": "word " * large_input}
                ],
                "max_tokens": 100,
            }
            for _ in range(10)
        ]
        responses = await client.concurrent_requests(
            large_payloads, max_concurrent=10
        )
        errors = sum(1 for r in responses if r.status >= 500 or r.status == 0)
        ok = sum(1 for r in responses if r.status == 200)
        rejected = sum(
            1 for r in responses if 400 <= r.status < 500 or r.status == 429
        )
        results.append(
            self.make_result(
                self.name,
                "kv_cache_fill_concurrent_10",
                Verdict.FAIL
                if errors > 5
                else (Verdict.INTERESTING if errors > 0 else Verdict.PASS),
                detail=f"{ok} ok, {rejected} rejected, {errors} errors — 10 large requests (~{large_input} tokens each)",
            )
        )

        # ----- 6. Fill KV cache, then send tiny request -----
        # Tests preemption/eviction: large requests fill cache, small request should still work
        large_tasks = [
            client.post_json(
                {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": "word " * large_input}
                    ],
                    "max_tokens": 500,
                },
                timeout=config.timeout * 2,
            )
            for _ in range(5)
        ]
        # Start large requests
        large_futures = [asyncio.create_task(t) for t in large_tasks]
        await asyncio.sleep(2)  # Let them start prefilling

        # Now send a tiny request
        small_resp = await client.post_json(
            {
                "model": model,
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 5,
            }
        )

        # Wait for large requests to finish
        await asyncio.gather(*large_futures, return_exceptions=True)

        results.append(
            self.make_result(
                self.name,
                "kv_cache_fill_then_small",
                Verdict.FAIL
                if small_resp.status >= 500 or small_resp.error == "TIMEOUT"
                else Verdict.PASS,
                status_code=small_resp.status,
                elapsed_ms=small_resp.elapsed_ms,
                detail=f"Small request after cache fill: status {small_resp.status}, {small_resp.elapsed_ms:.0f}ms",
            )
        )

        # ----- 7. Mixed prefill lengths -----
        # Concurrent requests with wildly different sizes stress the chunked prefill scheduler
        mixed_sizes = mc.mixed_prefill_sizes
        mixed_payloads = []
        for size in mixed_sizes:
            mixed_payloads.append(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "word " * size}],
                    "max_tokens": 10,
                }
            )
        responses = await client.concurrent_requests(
            mixed_payloads, max_concurrent=6
        )
        errors = sum(1 for r in responses if r.status >= 500 or r.status == 0)
        size_desc = "/".join(str(s) for s in mixed_sizes)
        results.append(
            self.make_result(
                self.name,
                "mixed_prefill_lengths",
                Verdict.FAIL
                if errors > 2
                else (Verdict.INTERESTING if errors > 0 else Verdict.PASS),
                detail=f"{errors}/{len(mixed_payloads)} errors — mixed [{size_desc}] token requests",
            )
        )

        # ----- 8. Long generation KV growth -----
        # Short prompt but huge max_tokens: KV cache must grow during decode
        gen_max = mc.large_generation_max_tokens
        resp = await client.post_json(
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Count from 1 to 10000, one number per line.",
                    }
                ],
                "max_tokens": gen_max,
            },
            timeout=config.timeout * 4,
        )
        results.append(
            self.make_result(
                self.name,
                "long_generation_kv_growth",
                Verdict.FAIL
                if resp.status >= 500 or resp.error == "TIMEOUT"
                else Verdict.PASS,
                status_code=resp.status,
                elapsed_ms=resp.elapsed_ms,
                detail=f"Short prompt + max_tokens={gen_max}: status {resp.status}",
            )
        )

        # ----- 9. Repeated exact prefix under load -----
        # 20 concurrent requests with identical system prompt but different user messages
        # Tests prefix cache hit optimization under concurrent load
        shared_system = (
            "You are a helpful math tutor who explains step by step. " * 200
        )
        prefix_payloads = [
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": shared_system},
                    {"role": "user", "content": f"What is {i} + {i * 2}?"},
                ],
                "max_tokens": 50,
            }
            for i in range(20)
        ]
        responses = await client.concurrent_requests(
            prefix_payloads, max_concurrent=20
        )
        errors = sum(1 for r in responses if r.status >= 500 or r.status == 0)
        results.append(
            self.make_result(
                self.name,
                "repeated_exact_prefix_under_load",
                Verdict.FAIL
                if errors > 5
                else (Verdict.INTERESTING if errors > 0 else Verdict.PASS),
                detail=f"{errors}/20 errors — identical prefix, different suffixes",
            )
        )

        # ----- 10. Page boundary probing -----
        # KV cache uses paged attention with fixed block sizes (commonly 16-512
        # tokens).  Requests that land exactly at a block boundary, or spill by
        # one token into a new block, exercise edge cases in the block allocator.
        page_sizes = [32, 64, 128, 256, 512]

        for ps in page_sizes:
            for offset, label in [(-1, "minus1"), (0, "exact"), (1, "plus1")]:
                n_tokens = ps + offset
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": "w " * n_tokens}],
                    "max_tokens": 1,
                }
                resp = await client.post_json(payload)
                if (
                    resp.status >= 500
                    or resp.status == 0
                    or resp.error == "TIMEOUT"
                ):
                    verdict = Verdict.FAIL
                elif resp.status == 200 or 400 <= resp.status < 500:
                    verdict = Verdict.PASS
                else:
                    verdict = Verdict.INTERESTING
                results.append(
                    self.make_result(
                        self.name,
                        f"page_boundary_{ps}_{label}",
                        verdict,
                        status_code=resp.status,
                        elapsed_ms=resp.elapsed_ms,
                        detail=f"~{n_tokens} tokens (page {ps}{'+' if offset > 0 else ''}{offset or ''}): "
                        f"status {resp.status}, {resp.elapsed_ms:.0f}ms",
                    )
                )
                if (
                    resp.status >= 500
                    or resp.status == 0
                    or resp.error == "TIMEOUT"
                ):
                    break
            else:
                continue
            break  # stop probing if server is crashing

        # ----- 11. Decode growth across page boundary -----
        # Short prefill that fits within a page, then decode crosses into the
        # next block — forces on-demand block allocation during generation.
        for ps in page_sizes:
            n_tokens = ps - 5 if ps > 5 else 1
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "w " * n_tokens}],
                "max_tokens": 16,
            }
            resp = await client.post_json(payload)
            if (
                resp.status >= 500
                or resp.status == 0
                or resp.error == "TIMEOUT"
            ):
                verdict = Verdict.FAIL
            elif resp.status == 200 or 400 <= resp.status < 500:
                verdict = Verdict.PASS
            else:
                verdict = Verdict.INTERESTING
            results.append(
                self.make_result(
                    self.name,
                    f"decode_across_page_{ps}",
                    verdict,
                    status_code=resp.status,
                    elapsed_ms=resp.elapsed_ms,
                    detail=f"~{n_tokens} prefill + 16 decode (crosses page {ps}): "
                    f"status {resp.status}",
                )
            )
            if (
                resp.status >= 500
                or resp.status == 0
                or resp.error == "TIMEOUT"
            ):
                break

        # ----- 12. Concurrent requests at same page boundary -----
        # 10 simultaneous requests at each exact page size stresses the block
        # allocator's concurrent allocation path.
        for ps in page_sizes:
            payloads = [
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "w " * ps}],
                    "max_tokens": 1,
                }
                for _ in range(10)
            ]
            responses = await client.concurrent_requests(
                payloads, max_concurrent=10
            )
            errors = sum(
                1 for r in responses if r.status >= 500 or r.status == 0
            )
            results.append(
                self.make_result(
                    self.name,
                    f"concurrent_page_boundary_{ps}",
                    Verdict.FAIL
                    if errors > 3
                    else (Verdict.INTERESTING if errors > 0 else Verdict.PASS),
                    detail=f"{errors}/10 errors — 10 concurrent requests at page size {ps}",
                )
            )
            if errors > 5:
                break  # server likely unhealthy

        # ----- 13. Mixed page-aligned concurrent burst -----
        # Aligned and off-by-one sizes together create worst-case fragmentation.
        mixed_page_sizes = []
        for ps in page_sizes:
            mixed_page_sizes.extend([ps, ps + 1])
        mixed_page_payloads = [
            {
                "model": model,
                "messages": [{"role": "user", "content": "w " * s}],
                "max_tokens": 5,
            }
            for s in mixed_page_sizes
        ]
        responses = await client.concurrent_requests(
            mixed_page_payloads, max_concurrent=10
        )
        errors = sum(1 for r in responses if r.status >= 500 or r.status == 0)
        results.append(
            self.make_result(
                self.name,
                "mixed_page_aligned_burst",
                Verdict.FAIL
                if errors > 3
                else (Verdict.INTERESTING if errors > 0 else Verdict.PASS),
                detail=f"{errors}/{len(mixed_page_sizes)} errors — mixed aligned/+1 sizes {mixed_page_sizes}",
            )
        )

        # ----- 14. Streaming cancel at page boundary -----
        # Start streaming near a page boundary and cancel during decode, right
        # as the engine would allocate a new block.
        for ps in [128, 256, 512]:
            n_tokens = ps - 2
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "w " * n_tokens}],
                "max_tokens": 200,
                "stream": True,
            }
            resp = await client.post_streaming(payload, cancel_after_chunks=3)
            if resp.status >= 500 or (resp.status == 0 and not resp.cancelled):
                verdict = Verdict.FAIL
            elif resp.cancelled or resp.status == 200:
                verdict = Verdict.PASS
            else:
                verdict = Verdict.INTERESTING
            results.append(
                self.make_result(
                    self.name,
                    f"stream_cancel_page_boundary_{ps}",
                    verdict,
                    status_code=resp.status,
                    elapsed_ms=resp.elapsed_ms,
                    detail=f"~{n_tokens} prefill, cancel after 3 chunks (page {ps}): "
                    f"status {resp.status}",
                )
            )

        # ----- 15. Health check -----
        await asyncio.sleep(3)
        health = await client.health_check()
        results.append(
            self.make_result(
                self.name,
                "post_kv_pressure_health_check",
                Verdict.PASS if health.status == 200 else Verdict.FAIL,
                status_code=health.status,
                detail="Healthy"
                if health.status == 200
                else f"Unhealthy: {health.error}",
            )
        )

        return results
