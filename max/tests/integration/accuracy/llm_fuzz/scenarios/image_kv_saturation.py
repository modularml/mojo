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
Scenario: vision tokens filling the KV cache
Target: CENG-880's acceptance criterion, which ``image_stress`` does not reach.

``image_stress`` sends the largest payload one *request* may legally carry.
That is a single-request ceiling, and CENG-880 does not close on it. Its bar is
aggregate: run a *batch* that fills G0 KV cache upwards of 90% with vision
tokens -- roughly 1M tokens per DP rank, so ~2M in flight together -- without
an OOM. One request cannot get there no matter how large, because the served
window caps it at ~1M.

So this scenario ramps concurrency instead of size. Each request carries the
largest legal image-token payload, and the ramp puts 1, 2, then 4 of them in
flight at once. Two is the interesting rung: it is the point where aggregate
vision tokens first exceed a single rank's capacity and the staging buffers
for two multi-chunk prefills have to coexist. Four deliberately overshoots.

The ramp matters as much as the ceiling. Tyler's original ask was to scale
"from 1 of these at a time to multiple in a batch together", and the failure
mode he named is exactly the one a single-request test cannot see: a change
that never OOMs on one big request but OOMs as soon as two are batched.

A deployment whose window is too small to hold one max-size image cannot
reach the criterion at any concurrency. That is reported as INTERESTING with
the shortfall named, rather than passing quietly on a bar it never cleared.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from client import FuzzClient

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario
from scenarios._image_fixtures import (
    IMAGE_MAX_COUNT,
    MAX_IMAGE_SIDE,
    MAX_IMAGE_TOKENS,
    image_part,
    image_payload,
)
from scenarios._image_stress_common import (
    HEAVY_TIMEOUT_SEC,
    PAYLOAD_OVERHEAD_TOKENS,
    BatchTally,
    LivenessProbe,
    failure_verdict,
    probe_result,
    served_context_window,
)

if TYPE_CHECKING:
    from client import RunConfig

# Concurrency rungs, in units of "largest legal single request". 1 is the
# control that ``image_stress`` already covers; 2 is the first rung whose
# aggregate exceeds one DP rank; 4 overshoots so a pod that survives 2 by
# luck rather than by sizing still gets pushed.
RAMP = (1, 2, 4)

# G0 capacity CENG-880 was written against: ~1M tokens per DP rank across 2
# ranks. Used only to report how close a rung came to the criterion -- the
# real capacity moves with quantization and fp8 KV, and the ticket says so.
CENG880_TARGET_TOKENS = 2_000_000

# Fraction of that target the ticket calls "filled".
CENG880_FILL_FRACTION = 0.9


@register_scenario
class ImageKVSaturation(BaseScenario):
    name = "image_kv_saturation"
    description = (
        "Concurrent max-payload image requests, ramped 1->2->4, to fill KV "
        "cache with vision tokens (CENG-880 acceptance criterion)"
    )
    tags = ["vision", "image", "oom", "memory", "crash", "kvcache"]
    # The per-image limits are MiniMax-M3's. Run elsewhere with an explicit
    # `--scenarios image_kv_saturation`, which bypasses profile filtering.
    model_filter = "minimax-m3"

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        window, source = await served_context_window(client, config)
        per_request = min(
            IMAGE_MAX_COUNT,
            (window - PAYLOAD_OVERHEAD_TOKENS) // MAX_IMAGE_TOKENS,
        )
        if per_request < 1:
            return [
                self.make_result(
                    self.name,
                    "kv_saturation_ramp",
                    Verdict.INTERESTING,
                    detail=(
                        f"{source} context window of {window} cannot hold one "
                        f"max-size image ({MAX_IMAGE_TOKENS} tokens) -- this "
                        "deployment cannot reach the CENG-880 criterion"
                    ),
                )
            ]

        results: list[ScenarioResult] = []
        for rung in RAMP:
            results.extend(
                await self._rung(
                    client, config, rung, per_request, window, source
                )
            )
            # A rung that wedged or crashed the pod makes every later rung
            # unattributable: the next OOM would be reported against a
            # concurrency the server never actually reached healthy.
            if any(r.verdict == Verdict.FAIL for r in results[-2:]):
                results.append(
                    self.make_result(
                        self.name,
                        "kv_saturation_ramp_halted",
                        Verdict.INTERESTING,
                        detail=(
                            f"stopped after concurrency {rung}; higher rungs "
                            "would not be attributable"
                        ),
                    )
                )
                break
        return results

    async def _rung(
        self,
        client: FuzzClient,
        config: RunConfig,
        concurrency: int,
        per_request: int,
        window: int,
        source: str,
    ) -> list[ScenarioResult]:
        """Puts ``concurrency`` max-payload requests in flight together."""
        nodes = max(1, config.image_stress_nodes)
        in_flight = concurrency * nodes
        aggregate = in_flight * per_request * MAX_IMAGE_TOKENS
        payloads = [
            image_payload(
                config.model,
                [
                    image_part(
                        MAX_IMAGE_SIDE,
                        MAX_IMAGE_SIDE,
                        f"kvsat-{concurrency}-{req}-{img}",
                        "high",
                    )
                    for img in range(per_request)
                ],
            )
            for req in range(in_flight)
        ]

        async with LivenessProbe(config) as probe:
            responses = await client.concurrent_requests(
                payloads,
                max_concurrent=in_flight,
                timeout=HEAVY_TIMEOUT_SEC,
            )

        fill = aggregate / CENG880_TARGET_TOKENS
        context = (
            f"{in_flight} concurrent requests x {per_request} max-size images "
            f"= ~{aggregate} vision tokens in flight "
            f"({fill:.0%} of the {CENG880_TARGET_TOKENS}-token G0 target; "
            f"{source} window {window})"
        )
        test = f"kv_saturation_x{concurrency}"

        tally = BatchTally().add(responses)
        # An OOM kills the pod outright, so it shows up as a mass failure or as
        # a stall; a lone reset from the network path is neither, and must not
        # halt the ramp on this rung's behalf.
        result = failure_verdict(
            self.name, test, tally, config, context=context, probe=probe
        )
        if result is None:
            if tally.client_errors or tally.other:
                # Every request here is inside the served window by
                # construction, so a 4xx means the deployment rejected work the
                # criterion requires it to accept -- not a crash, but not a
                # pass either. Nor is a status no bucket claims.
                verdict, detail = (
                    Verdict.INTERESTING,
                    f"{tally.summary()} -- rung not fully served ({context})",
                )
            elif fill < CENG880_FILL_FRACTION and concurrency == RAMP[-1]:
                # Survived every rung but never got near the bar: the window is
                # too small to saturate G0, so a PASS here would overstate what
                # was proven.
                verdict, detail = (
                    Verdict.INTERESTING,
                    f"{tally.summary()} -- survived, but peak fill "
                    f"{fill:.0%} never reached "
                    f"{CENG880_FILL_FRACTION:.0%} of the target ({context})",
                )
            else:
                verdict, detail = (
                    Verdict.PASS,
                    f"{tally.summary()} ({context})",
                )
            result = self.make_result(self.name, test, verdict, detail=detail)

        return [
            result,
            probe_result(self.name, f"{test}_liveness", probe, config),
        ]
