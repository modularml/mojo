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
Scenarios: Concurrency attacks
Target: Race conditions, queue overflow, OOM under concurrent load.
"""

from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING, Any

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig


@register_scenario
class ConcurrencyAttacks(BaseScenario):
    name = "concurrency_attacks"
    description = (
        "Thundering herd, burst, ramp, and mixed concurrent request patterns"
    )
    tags = ["concurrency", "load", "race", "crash"]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results = []
        model = config.model

        def simple_payload(
            content: str = "Say hi", max_tokens: int = 5
        ) -> dict[str, Any]:
            return {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
            }

        # ----- Test 1: Thundering herd — 100 identical requests at once -----
        results.extend(
            await self._thundering_herd(client, simple_payload(), 100)
        )

        # ----- Test 2: 100 requests with varying sizes -----
        varied = []
        for _ in range(100):
            size = random.choice([1, 10, 100, 500, 2000])
            varied.append(
                simple_payload(
                    content="x " * size,
                    max_tokens=random.choice([1, 5, 50, 200]),
                )
            )
        results.extend(
            await self._concurrent_batch(client, "varied_sizes_100", varied)
        )

        # ----- Test 3: Interleaved streaming and non-streaming -----
        results.extend(await self._mixed_streaming(client, model, 50))

        # ----- Test 4: Rapid sequential (no waiting for response) -----
        results.extend(await self._rapid_fire(client, simple_payload(), 200))

        # ----- Test 5: Burst-pause-burst -----
        results.extend(await self._burst_pause_burst(client, simple_payload()))

        # ----- Test 6: Gradually increasing concurrency -----
        results.extend(await self._ramp_up(client, simple_payload()))

        # ----- Test 7: Duplicate request IDs (if supported) -----
        dup_payload = simple_payload()
        dup_payload["user"] = "same-user-id"
        results.extend(
            await self._concurrent_batch(
                client, "duplicate_user_ids_50", [dup_payload] * 50
            )
        )

        # ----- Post-attack health check -----
        await asyncio.sleep(2)
        health = await client.health_check()
        verdict = Verdict.PASS if health.status == 200 else Verdict.FAIL
        results.append(
            self.make_result(
                self.name,
                "post_concurrency_health_check",
                verdict,
                status_code=health.status,
                detail="Server healthy"
                if health.status == 200
                else f"Server unhealthy after concurrency attacks: {health.error}",
            )
        )

        return results

    async def _thundering_herd(
        self, client: FuzzClient, payload: dict[str, Any], count: int
    ) -> list[ScenarioResult]:
        """Fire `count` identical requests simultaneously."""
        results = []
        responses = await client.concurrent_requests(
            [payload] * count, max_concurrent=count
        )

        errors = sum(1 for r in responses if r.status >= 500 or r.status == 0)
        timeouts = sum(1 for r in responses if r.error == "TIMEOUT")
        successes = sum(1 for r in responses if r.status == 200)
        rejected = sum(1 for r in responses if 400 <= r.status < 500)
        max_latency = max(r.elapsed_ms for r in responses) if responses else 0

        if errors > count * 0.5:
            verdict = Verdict.FAIL
            detail = f"Over 50% server errors ({errors}/{count})"
        elif timeouts > count * 0.3:
            verdict = Verdict.FAIL
            detail = f"Over 30% timeouts ({timeouts}/{count})"
        elif errors > 0:
            verdict = Verdict.INTERESTING
            detail = (
                f"{errors} server errors, {timeouts} timeouts out of {count}"
            )
        else:
            verdict = Verdict.PASS
            detail = f"{successes} OK, {rejected} rejected, max latency {max_latency:.0f}ms"

        results.append(
            self.make_result(
                self.name,
                f"thundering_herd_{count}",
                verdict,
                detail=detail,
                elapsed_ms=max_latency,
            )
        )
        return results

    async def _concurrent_batch(
        self,
        client: FuzzClient,
        name: str,
        payloads: list[dict[str, Any]],
    ) -> list[ScenarioResult]:
        responses = await client.concurrent_requests(payloads)
        errors = sum(1 for r in responses if r.status >= 500 or r.status == 0)
        timeouts = sum(1 for r in responses if r.error == "TIMEOUT")
        max_latency = max(r.elapsed_ms for r in responses) if responses else 0

        if errors > len(payloads) * 0.3:
            verdict = Verdict.FAIL
        elif errors > 0:
            verdict = Verdict.INTERESTING
        else:
            verdict = Verdict.PASS

        return [
            self.make_result(
                self.name,
                name,
                verdict,
                detail=f"{errors} errors, {timeouts} timeouts / {len(payloads)} total, max {max_latency:.0f}ms",
                elapsed_ms=max_latency,
            )
        ]

    async def _mixed_streaming(
        self, client: FuzzClient, model: str, count: int
    ) -> list[ScenarioResult]:
        """Interleave streaming and non-streaming requests."""

        async def _req(i: int) -> Any:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": f"Count to {i}"}],
                "max_tokens": 20,
            }
            if i % 2 == 0:
                return await client.post_streaming(payload)
            else:
                return await client.post_json(payload)

        tasks = [_req(i) for i in range(count)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        exceptions = sum(1 for r in responses if isinstance(r, BaseException))
        real_responses = [
            r for r in responses if not isinstance(r, BaseException)
        ]
        errors = sum(
            1 for r in real_responses if r.status >= 500 or r.status == 0
        )

        if errors + exceptions > count * 0.3:
            verdict = Verdict.FAIL
        elif errors + exceptions > 0:
            verdict = Verdict.INTERESTING
        else:
            verdict = Verdict.PASS

        return [
            self.make_result(
                self.name,
                f"mixed_streaming_{count}",
                verdict,
                detail=f"{errors} errors, {exceptions} exceptions / {count}",
            )
        ]

    async def _rapid_fire(
        self, client: FuzzClient, payload: dict[str, Any], count: int
    ) -> list[ScenarioResult]:
        """Fire requests as fast as possible, no concurrency limit."""
        sem = asyncio.Semaphore(count)
        errors = 0
        total = 0

        async def _fire() -> None:
            nonlocal errors, total
            async with sem:
                r = await client.post_json(
                    payload, timeout=client.config.timeout * 0.5
                )
                total += 1
                if r.status >= 500 or r.status == 0:
                    errors += 1

        await asyncio.gather(*[_fire() for _ in range(count)])

        verdict = (
            Verdict.FAIL
            if errors > count * 0.3
            else (Verdict.INTERESTING if errors > 0 else Verdict.PASS)
        )
        return [
            self.make_result(
                self.name,
                f"rapid_fire_{count}",
                verdict,
                detail=f"{errors}/{total} failed",
            )
        ]

    async def _burst_pause_burst(
        self, client: FuzzClient, payload: dict[str, Any]
    ) -> list[ScenarioResult]:
        """3 bursts of 50 with 2-second pauses."""
        all_errors = 0
        total = 0
        for _ in range(3):
            responses = await client.concurrent_requests([payload] * 50)
            all_errors += sum(
                1 for r in responses if r.status >= 500 or r.status == 0
            )
            total += len(responses)
            await asyncio.sleep(2)

        verdict = (
            Verdict.FAIL
            if all_errors > total * 0.3
            else (Verdict.INTERESTING if all_errors > 0 else Verdict.PASS)
        )
        return [
            self.make_result(
                self.name,
                "burst_pause_burst_3x50",
                verdict,
                detail=f"{all_errors}/{total} failed across 3 bursts",
            )
        ]

    async def _ramp_up(
        self, client: FuzzClient, payload: dict[str, Any]
    ) -> list[ScenarioResult]:
        """Gradually increase concurrency: 1, 5, 10, 25, 50, 100."""
        findings = []
        for n in [1, 5, 10, 25, 50, 100]:
            responses = await client.concurrent_requests(
                [payload] * n, max_concurrent=n
            )
            errors = sum(
                1 for r in responses if r.status >= 500 or r.status == 0
            )
            if errors > 0:
                findings.append(f"n={n}: {errors} errors")
            await asyncio.sleep(1)

        if findings:
            verdict = (
                Verdict.INTERESTING if len(findings) <= 2 else Verdict.FAIL
            )
            detail = "; ".join(findings)
        else:
            verdict = Verdict.PASS
            detail = "All concurrency levels handled cleanly"

        return [
            self.make_result(
                self.name,
                "ramp_up_1_to_100",
                verdict,
                detail=detail,
            )
        ]
