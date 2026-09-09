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
Scenario: Endurance soak tests
Target: Memory leaks, KV cache fragmentation, latency degradation, and
        error rate growth over sustained load.

Uses configurable duration via --endurance-duration (default 5 minutes)
and intensity via --endurance-intensity (low=5/s, medium=20/s, high=100/s).
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from reporting import print_progress

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig

INTENSITY_RPS = {"low": 5, "medium": 20, "high": 100}


@register_scenario
class EnduranceSoak(BaseScenario):
    name = "endurance_soak"
    description = "Sustained load for leak/degradation detection (5+ minutes)"
    tags = ["endurance", "memory", "resource_leak"]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results = []
        model = config.model
        duration = config.endurance_duration_sec
        rps = INTENSITY_RPS.get(config.endurance_intensity, 20)

        # ----- 1. Steady-state endurance -----
        results.extend(
            await self._endurance_test(
                client,
                model,
                duration,
                rps,
                test_name="endurance_steady_state",
                payload_fn=lambda: {
                    "model": model,
                    "messages": [{"role": "user", "content": "word " * 200}],
                    "max_tokens": 50,
                },
                description="Steady-state normal requests",
            )
        )

        # ----- 2. Streaming with random cancellation endurance -----
        results.extend(
            await self._streaming_endurance(
                client,
                model,
                min(duration, 120),
                max(rps // 4, 2),
            )
        )

        # ----- 3. Mixed workload endurance -----
        def mixed_payload() -> dict[str, Any]:
            r = random.random()
            if r < 0.6:
                # Short request (60%)
                return {
                    "model": model,
                    "messages": [{"role": "user", "content": "Say hi"}],
                    "max_tokens": 10,
                }
            elif r < 0.9:
                # Medium request (30%)
                return {
                    "model": model,
                    "messages": [{"role": "user", "content": "word " * 1000}],
                    "max_tokens": 100,
                }
            else:
                # Long request (10%)
                return {
                    "model": model,
                    "messages": [{"role": "user", "content": "word " * 4000}],
                    "max_tokens": 200,
                }

        results.extend(
            await self._endurance_test(
                client,
                model,
                min(duration, 120),
                rps,
                test_name="endurance_mixed_workload",
                payload_fn=mixed_payload,
                description="Mixed short/medium/long requests",
            )
        )

        return results

    async def _endurance_test(
        self,
        client: FuzzClient,
        model: str,
        duration_sec: float,
        rps: int,
        test_name: str,
        payload_fn: Callable[[], dict[str, Any]],
        description: str,
    ) -> list[ScenarioResult]:
        """Run sustained requests for a duration, tracking error rate and latency per window."""
        results = []
        window_sec = 10
        sem = asyncio.Semaphore(min(rps * 2, 200))

        all_latencies = []
        all_errors = 0
        all_total = 0
        window_data = []  # list of (error_count, total_count, latencies)

        current_window_errors = 0
        current_window_total = 0
        current_window_latencies = []
        window_start = time.perf_counter()

        end_time = time.perf_counter() + duration_sec
        t_start = time.perf_counter()

        async def _req() -> None:
            nonlocal \
                all_errors, \
                all_total, \
                current_window_errors, \
                current_window_total
            async with sem:
                r = await client.post_json(payload_fn())
                all_total += 1
                current_window_total += 1
                if r.status >= 500 or r.status == 0:
                    all_errors += 1
                    current_window_errors += 1
                if r.status == 200:
                    all_latencies.append(r.elapsed_ms)
                    current_window_latencies.append(r.elapsed_ms)

        tasks = []
        interval = 1.0 / rps
        last_progress = 0.0

        while time.perf_counter() < end_time:
            tasks.append(asyncio.create_task(_req()))
            await asyncio.sleep(interval)

            # Check window boundary
            now = time.perf_counter()
            if now - window_start >= window_sec:
                window_data.append(
                    (
                        current_window_errors,
                        current_window_total,
                        list(current_window_latencies),
                    )
                )
                current_window_errors = 0
                current_window_total = 0
                current_window_latencies = []
                window_start = now

            # Print progress every 30 seconds
            elapsed = now - t_start
            if elapsed - last_progress >= 30:
                print_progress(
                    elapsed,
                    duration_sec,
                    all_total,
                    all_errors,
                    all_latencies[-100:],
                )
                last_progress = elapsed

        # Collect remaining
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Final window
        if current_window_total > 0:
            window_data.append(
                (
                    current_window_errors,
                    current_window_total,
                    list(current_window_latencies),
                )
            )

        print()  # Newline after progress

        # Overall result
        error_rate = all_errors / all_total if all_total > 0 else 0
        results.append(
            self.make_result(
                self.name,
                test_name,
                Verdict.FAIL
                if error_rate > 0.05
                else (
                    Verdict.INTERESTING if error_rate > 0.01 else Verdict.PASS
                ),
                detail=f"{description}: {all_errors}/{all_total} ({error_rate * 100:.1f}%) over {duration_sec:.0f}s at {rps} req/s",
            )
        )

        # Error rate window analysis: FAIL if any window degrades beyond 5%
        # when the first window was clean
        if len(window_data) >= 3:
            first_rate = window_data[0][0] / max(window_data[0][1], 1)
            worst_rate = 0.0
            worst_idx = 0
            for i, (errs, total, _) in enumerate(window_data):
                rate = errs / max(total, 1)
                if rate > worst_rate:
                    worst_rate = rate
                    worst_idx = i

            degraded = first_rate < 0.01 and worst_rate > 0.05
            results.append(
                self.make_result(
                    self.name,
                    f"{test_name}_error_windows",
                    Verdict.FAIL if degraded else Verdict.PASS,
                    detail=f"Window analysis: first={first_rate * 100:.1f}%, worst={worst_rate * 100:.1f}% (window {worst_idx})",
                )
            )

        # Latency degradation analysis
        if len(window_data) >= 3:
            first_lats = window_data[0][2]
            last_lats = window_data[-1][2]

            if first_lats and last_lats:
                first_p99 = (
                    sorted(first_lats)[int(len(first_lats) * 0.99)]
                    if len(first_lats) > 1
                    else first_lats[0]
                )
                last_p99 = (
                    sorted(last_lats)[int(len(last_lats) * 0.99)]
                    if len(last_lats) > 1
                    else last_lats[0]
                )
                ratio = last_p99 / first_p99 if first_p99 > 0 else 0

                results.append(
                    self.make_result(
                        self.name,
                        f"{test_name}_latency_degradation",
                        Verdict.FAIL
                        if ratio > 3
                        else (
                            Verdict.INTERESTING if ratio > 2 else Verdict.PASS
                        ),
                        detail=f"p99 first window: {first_p99:.0f}ms, last window: {last_p99:.0f}ms, ratio: {ratio:.1f}x",
                    )
                )

        return results

    async def _streaming_endurance(
        self,
        client: FuzzClient,
        model: str,
        duration_sec: float,
        rps: int,
    ) -> list[ScenarioResult]:
        """Sustained streaming with random cancellations."""
        results = []
        sem = asyncio.Semaphore(min(rps * 2, 50))
        total = 0
        errors = 0

        end_time = time.perf_counter() + duration_sec
        interval = 1.0 / rps

        async def _stream_req() -> None:
            nonlocal total, errors
            async with sem:
                cancel_ms = random.choice([None, 500, 1000, 2000, None, None])
                r = await client.post_streaming(
                    {
                        "model": model,
                        "messages": [
                            {
                                "role": "user",
                                "content": "Tell me about the weather.",
                            }
                        ],
                        "max_tokens": 256,
                    },
                    cancel_after_ms=cancel_ms,
                )
                total += 1
                if r.status >= 500 or (r.status == 0 and r.error != "TIMEOUT"):
                    errors += 1

        tasks = []
        while time.perf_counter() < end_time:
            tasks.append(asyncio.create_task(_stream_req()))
            await asyncio.sleep(interval)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        error_rate = errors / total if total > 0 else 0
        results.append(
            self.make_result(
                self.name,
                "endurance_streaming_leak",
                Verdict.FAIL
                if error_rate > 0.05
                else (
                    Verdict.INTERESTING if error_rate > 0.01 else Verdict.PASS
                ),
                detail=f"Streaming+cancel: {errors}/{total} ({error_rate * 100:.1f}%) over {duration_sec:.0f}s",
            )
        )

        return results
