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
Scenario: Disaggregated inference pipeline stall
Target: Exploit the separation between PrefillEngine and DecodeEngine workers
        in Dynamo's disaggregated architecture to create pipeline imbalances,
        stalls, and KV transfer failures.

Dynamo uses:
- PrefillEngine workers for initial prompt processing
- DecodeEngine workers for autoregressive token generation
- KVBM (KV Block Manager) for KV cache transfer between stages
- KV router mode for request routing
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig


@register_scenario
class PipelineStall(BaseScenario):
    name = "pipeline_stall"
    description = "Prefill/decode pipeline imbalance, KV transfer disruption, burst attacks"
    tags = ["dynamo", "crash", "concurrency", "pipeline"]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results = []
        model = config.model
        mc = config.model_config

        # ----- 1. Prefill flood: overwhelm prefill stage -----
        # Many large-context requests simultaneously to saturate prefill workers
        prefill_size = mc.prefill_flood_size
        prefill_heavy = [
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": "word " * prefill_size}
                ],
                "max_tokens": 10,
            }
            for _ in range(10)
        ]
        responses = await client.concurrent_requests(
            prefill_heavy, max_concurrent=10
        )
        errors = sum(1 for r in responses if r.status >= 500 or r.status == 0)
        timeouts = sum(1 for r in responses if r.error == "TIMEOUT")
        latencies = [r.elapsed_ms for r in responses if r.status == 200]
        max_lat = max(latencies) if latencies else 0

        results.append(
            self.make_result(
                self.name,
                "prefill_flood",
                Verdict.FAIL
                if errors > 3
                else (Verdict.INTERESTING if timeouts > 0 else Verdict.PASS),
                detail=f"{errors}/10 errors, {timeouts} timeouts, max_latency={max_lat:.0f}ms — prefill-heavy flood",
            )
        )

        # ----- 2. Long decode block: occupy decode workers -----
        # Short prompts with huge max_tokens tie up decode workers
        decode_max = mc.decode_heavy_max_tokens
        decode_heavy = [
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Count from 1 to {n}. Write each number on a new line.",
                    }
                ],
                "max_tokens": decode_max,
            }
            for n in [5000, 5000, 5000, 5000, 5000]
        ]
        responses = await client.concurrent_requests(
            decode_heavy, max_concurrent=5
        )
        errors = sum(1 for r in responses if r.status >= 500 or r.status == 0)
        timeouts = sum(1 for r in responses if r.error == "TIMEOUT")
        results.append(
            self.make_result(
                self.name,
                "long_decode_block",
                Verdict.FAIL
                if errors > 2
                else (Verdict.INTERESTING if timeouts > 0 else Verdict.PASS),
                detail=f"{errors}/5 errors, {timeouts} timeouts — decode-heavy (max_tokens={decode_max})",
            )
        )

        # ----- 3. Mixed prefill/decode pressure -----
        # Half prefill-heavy (large context, small output), half decode-heavy (small context, large output)
        large_input = mc.large_input_tokens
        mixed = []
        for i in range(20):
            if i % 2 == 0:
                # Prefill-heavy
                mixed.append(
                    {
                        "model": model,
                        "messages": [
                            {"role": "user", "content": "word " * large_input}
                        ],
                        "max_tokens": 10,
                    }
                )
            else:
                # Decode-heavy
                mixed.append(
                    {
                        "model": model,
                        "messages": [
                            {"role": "user", "content": "Write a long story."}
                        ],
                        "max_tokens": mc.decode_heavy_max_tokens,
                    }
                )

        responses = await client.concurrent_requests(mixed, max_concurrent=20)
        errors = sum(1 for r in responses if r.status >= 500 or r.status == 0)
        prefill_lats = [
            r.elapsed_ms
            for i, r in enumerate(responses)
            if i % 2 == 0 and r.status == 200
        ]
        decode_lats = [
            r.elapsed_ms
            for i, r in enumerate(responses)
            if i % 2 == 1 and r.status == 200
        ]
        detail = f"{errors}/20 errors"
        if prefill_lats:
            detail += (
                f" | prefill avg={sum(prefill_lats) / len(prefill_lats):.0f}ms"
            )
        if decode_lats:
            detail += (
                f" | decode avg={sum(decode_lats) / len(decode_lats):.0f}ms"
            )

        results.append(
            self.make_result(
                self.name,
                "mixed_prefill_decode_pressure",
                Verdict.FAIL
                if errors > 5
                else (Verdict.INTERESTING if errors > 0 else Verdict.PASS),
                detail=detail,
            )
        )

        # ----- 4. Streaming cancel during KV transfer window -----
        # The first 100-500ms is when KV cache transfers from prefill to decode.
        # Cancelling during this window can leave dangling KV cache entries.
        medium_input = mc.medium_input_tokens
        cancel_results = []
        for cancel_ms in [50, 100, 200, 500]:
            resp = await client.post_streaming(
                {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": "word " * medium_input}
                    ],
                    "max_tokens": 100,
                },
                cancel_after_ms=cancel_ms,
            )
            cancel_results.append(resp)

        errors = sum(1 for r in cancel_results if r.status >= 500)
        results.append(
            self.make_result(
                self.name,
                "streaming_cancel_during_kv_transfer",
                Verdict.FAIL if errors > 0 else Verdict.PASS,
                detail=f"Cancel at 50/100/200/500ms: {errors}/4 server errors",
            )
        )

        # ----- 5. Burst after idle -----
        # Let system idle, then hammer it — cold pipeline start
        await asyncio.sleep(10)

        burst_payloads = [
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": "word " * medium_input}
                ],
                "max_tokens": 100,
            }
            for _ in range(20)
        ]
        t0 = time.perf_counter()
        responses = await client.concurrent_requests(
            burst_payloads, max_concurrent=20
        )
        burst_elapsed = (time.perf_counter() - t0) * 1000
        errors = sum(1 for r in responses if r.status >= 500 or r.status == 0)
        results.append(
            self.make_result(
                self.name,
                "burst_after_idle",
                Verdict.FAIL
                if errors > 5
                else (Verdict.INTERESTING if errors > 0 else Verdict.PASS),
                detail=f"{errors}/20 errors after 10s idle, burst took {burst_elapsed:.0f}ms",
            )
        )

        # ----- 6. Sequential large then small -----
        # One massive context request, immediately followed by many tiny requests.
        # Small requests should not be head-of-line blocked.
        massive_input = min(mc.max_position_embeddings // 3, 50000)
        large_task = asyncio.create_task(
            client.post_json(
                {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": "word " * massive_input}
                    ],
                    "max_tokens": 10,
                },
                timeout=config.timeout * 4,
            )
        )

        await asyncio.sleep(1)  # Let large request start

        small_payloads = [
            {
                "model": model,
                "messages": [{"role": "user", "content": "Say OK"}],
                "max_tokens": 5,
            }
            for _ in range(10)
        ]
        small_responses = await client.concurrent_requests(
            small_payloads, max_concurrent=10
        )
        small_lats = [r.elapsed_ms for r in small_responses if r.status == 200]
        small_errors = sum(
            1 for r in small_responses if r.status >= 500 or r.status == 0
        )

        # Wait for large to finish
        large_resp = await large_task

        avg_small = sum(small_lats) / len(small_lats) if small_lats else 0
        max_small = max(small_lats) if small_lats else 0

        results.append(
            self.make_result(
                self.name,
                "sequential_large_then_small",
                Verdict.FAIL
                if small_errors > 5
                else (
                    Verdict.INTERESTING if max_small > 30000 else Verdict.PASS
                ),
                detail=f"Small: {small_errors}/10 errors, avg={avg_small:.0f}ms, max={max_small:.0f}ms | Large: status {large_resp.status}",
            )
        )

        # ----- 7. Health endpoint during inference load -----
        # Health checks should always respond quickly, even under heavy load
        async def health_poll() -> list[Any]:
            results_h = []
            for _ in range(50):
                r = await client.get_path(
                    "/health", timeout=config.timeout * 0.17
                )
                results_h.append(r)
                await asyncio.sleep(0.1)  # 10/sec
            return results_h

        async def inference_load() -> list[Any]:
            payloads = [
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "word " * 2000}],
                    "max_tokens": 100,
                }
                for _ in range(10)
            ]
            return await client.concurrent_requests(payloads, max_concurrent=10)

        health_task = asyncio.create_task(health_poll())
        load_task = asyncio.create_task(inference_load())

        gather_results = await asyncio.gather(health_task, load_task)
        health_results_list: list[Any] = gather_results[0]
        health_ok = sum(1 for r in health_results_list if r.status == 200)
        health_slow = sum(1 for r in health_results_list if r.elapsed_ms > 5000)
        results.append(
            self.make_result(
                self.name,
                "health_during_load",
                Verdict.FAIL
                if health_ok < 25
                else (Verdict.INTERESTING if health_slow > 5 else Verdict.PASS),
                detail=f"{health_ok}/50 healthy, {health_slow} slow (>5s) during inference load",
            )
        )

        # ----- 8. Post-pipeline health check -----
        await asyncio.sleep(3)
        health = await client.health_check()
        results.append(
            self.make_result(
                self.name,
                "post_pipeline_health_check",
                Verdict.PASS if health.status == 200 else Verdict.FAIL,
                status_code=health.status,
                detail="Healthy"
                if health.status == 200
                else f"Unhealthy: {health.error}",
            )
        )

        return results
