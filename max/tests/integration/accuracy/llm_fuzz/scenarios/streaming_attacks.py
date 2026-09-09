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
Scenarios: Streaming attacks
Target: Resource leaks, hangs, and crashes from streaming abuse.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario

if TYPE_CHECKING:
    from client import FuzzClient, RawResponse, RunConfig


@register_scenario
class StreamingAttacks(BaseScenario):
    name = "streaming_attacks"
    description = "Cancel mid-stream, abandon connections, open/close rapidly, streaming edge cases"
    tags = ["streaming", "cancel", "resource_leak", "crash"]

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results = []
        model = config.model

        def stream_payload(
            content: str = "Tell me a very long story about a dragon. Go into great detail.",
            max_tokens: int = 200,
        ) -> dict[str, Any]:
            return {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "stream": True,
            }

        # ----- 1. Cancel immediately (0 chunks) -----
        resp = await client.post_streaming(
            stream_payload(), cancel_after_chunks=0
        )
        results.append(self._assess(resp, "cancel_immediately"))

        # ----- 2. Cancel after first chunk -----
        resp = await client.post_streaming(
            stream_payload(), cancel_after_chunks=1
        )
        results.append(self._assess(resp, "cancel_after_1_chunk"))

        # ----- 3. Cancel after 3 chunks -----
        resp = await client.post_streaming(
            stream_payload(), cancel_after_chunks=3
        )
        results.append(self._assess(resp, "cancel_after_3_chunks"))

        # ----- 4. Cancel after 100ms -----
        resp = await client.post_streaming(
            stream_payload(), cancel_after_ms=100
        )
        results.append(self._assess(resp, "cancel_after_100ms"))

        # ----- 5. Cancel after 500ms -----
        resp = await client.post_streaming(
            stream_payload(), cancel_after_ms=500
        )
        results.append(self._assess(resp, "cancel_after_500ms"))

        # ----- 6. Cancel after 50ms (very fast) -----
        resp = await client.post_streaming(stream_payload(), cancel_after_ms=50)
        results.append(self._assess(resp, "cancel_after_50ms"))

        # ----- 7. Many concurrent stream cancels (cancel storm) -----
        results.extend(await self._cancel_storm(client, stream_payload, 50))

        # ----- 8. Many concurrent streams reading to completion -----
        results.extend(
            await self._concurrent_full_streams(client, stream_payload, 50)
        )

        # ----- 9. Cancel at random points -----
        results.extend(
            await self._random_cancel_points(client, stream_payload, 30)
        )

        # ----- 10. Open stream with max_tokens=1 (almost nothing to stream) -----
        resp = await client.post_streaming(stream_payload(max_tokens=1))
        v = Verdict.PASS if resp.status == 200 else Verdict.FAIL
        results.append(
            self.make_result(
                self.name,
                "stream_max_tokens_1",
                v,
                status_code=resp.status,
                detail=f"Got {len(resp.chunks or [])} chunks",
            )
        )

        # ----- 11. Stream with stream_options (usage reporting) -----
        usage_payload = stream_payload()
        usage_payload["stream_options"] = {"include_usage": True}
        resp = await client.post_streaming(usage_payload)
        v = Verdict.PASS if resp.status in (200, 400) else Verdict.FAIL
        results.append(
            self.make_result(
                self.name,
                "stream_with_usage",
                v,
                status_code=resp.status,
                detail=f"Status {resp.status}",
            )
        )

        # ----- 12. Stream then immediately send non-stream on same connection -----
        await client.post_streaming(stream_payload(), cancel_after_chunks=2)
        resp2 = await client.post_json(
            {
                "model": model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            }
        )
        v = Verdict.PASS if resp2.status == 200 else Verdict.FAIL
        results.append(
            self.make_result(
                self.name,
                "stream_then_non_stream",
                v,
                status_code=resp2.status,
                detail="Non-stream after cancelled stream"
                + (f" error: {resp2.error}" if resp2.error else ""),
            )
        )

        # ----- 13. Streaming with empty content -----
        resp = await client.post_streaming(stream_payload(content=""))
        v = Verdict.PASS if resp.status in (200, 400) else Verdict.FAIL
        results.append(
            self.make_result(
                self.name,
                "stream_empty_content",
                v,
                status_code=resp.status,
            )
        )

        # ----- 14. Health check after all streaming abuse -----
        await asyncio.sleep(3)
        health = await client.health_check()
        results.append(
            self.make_result(
                self.name,
                "post_streaming_health_check",
                Verdict.PASS if health.status == 200 else Verdict.FAIL,
                status_code=health.status,
                detail="Healthy"
                if health.status == 200
                else f"Unhealthy: {health.error}",
            )
        )

        return results

    def _assess(self, resp: RawResponse, name: str) -> ScenarioResult:
        if resp.error and "TIMEOUT" not in resp.error and resp.cancelled:
            # Connection was cancelled on purpose, that's fine
            verdict = Verdict.PASS
            detail = (
                f"Cancelled cleanly, {len(resp.chunks or [])} chunks received"
            )
        elif resp.error == "TIMEOUT":
            verdict = Verdict.FAIL
            detail = "Server hung during streaming"
        elif resp.status >= 500:
            verdict = Verdict.FAIL
            detail = f"Server error {resp.status} during streaming"
        elif resp.status == 0 and not resp.cancelled:
            verdict = Verdict.FAIL
            detail = f"Connection lost: {resp.error}"
        else:
            verdict = Verdict.PASS
            detail = f"Status {resp.status}, {len(resp.chunks or [])} chunks"

        return self.make_result(
            self.name,
            name,
            verdict,
            status_code=resp.status,
            elapsed_ms=resp.elapsed_ms,
            detail=detail,
        )

    async def _cancel_storm(
        self,
        client: FuzzClient,
        payload_fn: Callable[..., dict[str, Any]],
        count: int,
    ) -> list[ScenarioResult]:
        """Open many streams and cancel them all rapidly."""

        async def _cancel_one(i: int) -> RawResponse:
            cancel_at = random.choice([0, 1, 2, 3])
            return await client.post_streaming(
                payload_fn(), cancel_after_chunks=cancel_at
            )

        responses = await asyncio.gather(
            *[_cancel_one(i) for i in range(count)], return_exceptions=True
        )
        exceptions = sum(1 for r in responses if isinstance(r, BaseException))
        real = [r for r in responses if not isinstance(r, BaseException)]
        server_errors = sum(1 for r in real if r.status >= 500)

        if server_errors + exceptions > count * 0.3:
            verdict = Verdict.FAIL
        elif server_errors + exceptions > 0:
            verdict = Verdict.INTERESTING
        else:
            verdict = Verdict.PASS

        return [
            self.make_result(
                self.name,
                f"cancel_storm_{count}",
                verdict,
                detail=f"{server_errors} server errors, {exceptions} exceptions / {count}",
            )
        ]

    async def _concurrent_full_streams(
        self,
        client: FuzzClient,
        payload_fn: Callable[..., dict[str, Any]],
        count: int,
    ) -> list[ScenarioResult]:
        """Open many streams and read them all to completion."""
        responses = await asyncio.gather(
            *[
                client.post_streaming(payload_fn(max_tokens=50))
                for _ in range(count)
            ],
            return_exceptions=True,
        )
        exceptions = sum(1 for r in responses if isinstance(r, BaseException))
        real = [r for r in responses if not isinstance(r, BaseException)]
        server_errors = sum(1 for r in real if r.status >= 500)

        if server_errors + exceptions > count * 0.3:
            verdict = Verdict.FAIL
        elif server_errors + exceptions > 0:
            verdict = Verdict.INTERESTING
        else:
            verdict = Verdict.PASS

        return [
            self.make_result(
                self.name,
                f"concurrent_full_streams_{count}",
                verdict,
                detail=f"{server_errors} errors, {exceptions} exceptions / {count}",
            )
        ]

    async def _random_cancel_points(
        self,
        client: FuzzClient,
        payload_fn: Callable[..., dict[str, Any]],
        count: int,
    ) -> list[ScenarioResult]:
        """Cancel at random millisecond offsets."""

        async def _one() -> RawResponse:
            ms = random.randint(10, 2000)
            return await client.post_streaming(payload_fn(), cancel_after_ms=ms)

        responses = await asyncio.gather(
            *[_one() for _ in range(count)], return_exceptions=True
        )
        exceptions = sum(1 for r in responses if isinstance(r, BaseException))
        real = [r for r in responses if not isinstance(r, BaseException)]
        server_errors = sum(1 for r in real if r.status >= 500)

        if server_errors + exceptions > count * 0.3:
            verdict = Verdict.FAIL
        elif server_errors + exceptions > 0:
            verdict = Verdict.INTERESTING
        else:
            verdict = Verdict.PASS

        return [
            self.make_result(
                self.name,
                f"random_cancel_{count}",
                verdict,
                detail=f"{server_errors} errors, {exceptions} exceptions / {count}",
            )
        ]
