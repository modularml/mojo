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
Scenario: huge video request batches
Target: the video decode, sampling and encode path -- the vendor limits at
        their boundaries, the OOM on large frame counts, and the deadlock the
        image scenarios hunt, reached through a costlier front end.

Video reuses the vision encoder the image path already stresses, but
everything upstream of it is new work, and that is where this scenario aims.
A video request decodes an entire container before a single patch is encoded:
frames are demuxed, sampled down to the requested rate, resized, and only then
handed to the encoder. Decode runs on a thread pool
(``MODULAR_VIDEO_DECODE_THREADS``, 16 by default), so a batch of concurrent
video requests contends for host CPU in a way image requests never do.

Three properties make the limits worth probing rather than assuming:

* The frame cap is not enforced. ``sample_frame_indices`` accepts
  ``max_frames`` and ignores it -- the cap was removed so long clips keep
  their final frame -- so the sampled frame count is bounded only by the
  clip's own length and the requested ``fps``. A long clip sampled at its
  native rate is the largest single request the server will accept.
* The pixel cap is ``w * h * frames``, checked after the resize rules. At the
  default 672 tier that is 666 frames; the axis derives the count rather than
  hard-coding it, because the tier moves with ``detail``.
* ``fps`` has both an absolute range (0.2 to 5.0) and a relative one: it may
  not exceed the container's own rate, because the vendor never invents
  frames. The relative rule is the one a synthetic fixture can get wrong, so
  it gets its own case.

Grading is shared with the image scenarios: the liveness probe decides whether
the pod wedged, and a single dropped connection never claims the hang. See
``failure_verdict``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from client import FuzzClient

from scenarios import BaseScenario, ScenarioResult, Verdict, register_scenario
from scenarios._image_stress_common import (
    CONCURRENCY_PER_NODE,
    HEAVY_TIMEOUT_SEC,
    PAYLOAD_OVERHEAD_TOKENS,
    BatchTally,
    LivenessProbe,
    failure_verdict,
    post_once,
    post_stress_liveness,
    probe_result,
    served_context_window,
    single_response_verdict,
)
from scenarios._video_fixtures import (
    FPS_MAX,
    FPS_MIN,
    VIDEO_MAX_COUNT,
    VIDEO_MAX_LONG_SIDE,
    VIDEO_MAX_TOTAL_PIXELS,
    estimate_tokens,
    frames_for_pixels,
    video_part,
    video_payload,
)

if TYPE_CHECKING:
    from client import RunConfig

# Frame counts for the concurrency axes. Short clips: the deadlock the image
# scenarios reproduce is triggered by concurrent cache misses, not by scale,
# and long clips would turn a concurrency test into a decode-throughput test.
CONCURRENT_FRAMES = [4, 6, 8, 8, 12, 16]

# Frame size for the concurrency axes -- the default tier, so no resize work
# is skipped and no request is unusually cheap.
TYPICAL_FRAME = 672

# Concurrency rungs for the ramp: one request at a time, then several
# together, which is the scaling the ticket asks about.
RAMP = [1, 2, 4]

# Videos per request on the concurrency axes. Well under the vendor's 20, so
# the axis measures concurrency rather than per-request size.
VIDEOS_PER_REQUEST = 2


@register_scenario
class VideoStress(BaseScenario):
    name = "video_stress"
    description = (
        "Huge video batches: max count/frames/pixels, fps boundaries, and "
        "concurrent cache-miss batches under a liveness probe"
    )
    tags = ["vision", "video", "hang", "crash", "oom", "memory", "concurrency"]
    # The limits encoded here are MiniMax-M3 vendor limits. Run it against
    # another video model with an explicit `--scenarios video_stress`, which
    # bypasses profile filtering.
    model_filter = "minimax-m3"

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []

        # Boundary axes first: cheap to attribute, and a pod already wedged by
        # them makes the concurrency results meaningless.
        results.append(await self._max_video_count(client, config))
        results.append(await self._over_max_video_count(client, config))
        results.append(await self._max_frame_size(client, config))
        results.append(await self._at_pixel_cap(client, config))
        results.append(await self._over_pixel_cap(client, config))
        results.append(await self._max_total_video_tokens(client, config))
        results.extend(await self._fps_boundaries(client, config))
        results.append(await self._skewed_aspect_ratio(client, config))

        # Concurrency axis: the MXSERV-395 shape, reached through decode.
        results.extend(await self._concurrent_uncached_batches(client, config))

        results.append(await post_stress_liveness(client, self.name))
        return results

    # ------------------------------------------------------------------
    # Count and size boundaries
    # ------------------------------------------------------------------

    async def _max_video_count(
        self, client: FuzzClient, config: RunConfig
    ) -> ScenarioResult:
        """20 videos -- the vendor per-request ceiling, exactly at the limit."""
        parts = [
            video_part(504, 504, 4, 1.0, f"count-{i}")
            for i in range(VIDEO_MAX_COUNT)
        ]
        resp, dropped = await post_once(
            client,
            video_payload(config.model, parts),
            timeout=HEAVY_TIMEOUT_SEC,
        )
        return single_response_verdict(
            self.name,
            "max_video_count",
            resp,
            expect_reject=False,
            context=f"{VIDEO_MAX_COUNT} videos at the vendor limit",
            dropped_first=dropped,
        )

    async def _over_max_video_count(
        self, client: FuzzClient, config: RunConfig
    ) -> ScenarioResult:
        """21 videos -- one past the limit. Must be a clean 4xx."""
        parts = [
            video_part(504, 504, 4, 1.0, f"over-count-{i}")
            for i in range(VIDEO_MAX_COUNT + 1)
        ]
        resp, dropped = await post_once(
            client,
            video_payload(config.model, parts),
            timeout=HEAVY_TIMEOUT_SEC,
        )
        return single_response_verdict(
            self.name,
            "over_max_video_count",
            resp,
            expect_reject=True,
            context=f"{VIDEO_MAX_COUNT + 1} videos, one past the vendor limit",
            dropped_first=dropped,
        )

    async def _max_frame_size(
        self, client: FuzzClient, config: RunConfig
    ) -> ScenarioResult:
        """Frames at the largest video tier.

        Needs ``detail="high"``; the video default tier is 672, less than half
        the image default, so without it the frames are downscaled and the
        axis tests nothing.
        """
        part = video_part(
            VIDEO_MAX_LONG_SIDE,
            VIDEO_MAX_LONG_SIDE,
            8,
            1.0,
            "max-frame",
            detail="high",
        )
        resp, dropped = await post_once(
            client,
            video_payload(config.model, [part]),
            timeout=HEAVY_TIMEOUT_SEC,
        )
        return single_response_verdict(
            self.name,
            "max_frame_size",
            resp,
            expect_reject=False,
            context=(
                f"1 video, 8 frames at {VIDEO_MAX_LONG_SIDE}x"
                f"{VIDEO_MAX_LONG_SIDE} (detail=high)"
            ),
            dropped_first=dropped,
        )

    async def _at_pixel_cap(
        self, client: FuzzClient, config: RunConfig
    ) -> ScenarioResult:
        """The longest clip the pixel cap allows, and the OOM candidate.

        Every frame is decoded and resized before the cap is even checked
        against the post-resize size, so this is the largest amount of decode
        work one legal request can ask for.
        """
        frames = frames_for_pixels(
            TYPICAL_FRAME, TYPICAL_FRAME, VIDEO_MAX_TOTAL_PIXELS
        )
        part = video_part(
            TYPICAL_FRAME, TYPICAL_FRAME, frames, 1.0, "pixel-cap"
        )
        resp, dropped = await post_once(
            client,
            video_payload(config.model, [part]),
            timeout=HEAVY_TIMEOUT_SEC,
        )
        return single_response_verdict(
            self.name,
            "at_pixel_cap",
            resp,
            expect_reject=False,
            context=(
                f"1 video, {frames} frames at {TYPICAL_FRAME}x{TYPICAL_FRAME} "
                f"-- {frames * TYPICAL_FRAME * TYPICAL_FRAME:,} pixels, just "
                f"under the {VIDEO_MAX_TOTAL_PIXELS:,} cap"
            ),
            dropped_first=dropped,
        )

    async def _over_pixel_cap(
        self, client: FuzzClient, config: RunConfig
    ) -> ScenarioResult:
        """Past the pixel cap. The rejection is not the interesting part.

        The count check passes and the container is decoded in full before
        anything measures the pixel total, so a 5xx or a timeout here is the
        OOM, not a validation failure.
        """
        frames = (
            frames_for_pixels(
                TYPICAL_FRAME, TYPICAL_FRAME, VIDEO_MAX_TOTAL_PIXELS
            )
            + 60
        )
        part = video_part(
            TYPICAL_FRAME, TYPICAL_FRAME, frames, 1.0, "over-pixel-cap"
        )
        resp, dropped = await post_once(
            client,
            video_payload(config.model, [part]),
            timeout=HEAVY_TIMEOUT_SEC,
        )
        return single_response_verdict(
            self.name,
            "over_pixel_cap",
            resp,
            expect_reject=True,
            context=(
                f"1 video, {frames} frames at {TYPICAL_FRAME}x{TYPICAL_FRAME} "
                f"-- {frames * TYPICAL_FRAME * TYPICAL_FRAME:,} pixels, past "
                f"the {VIDEO_MAX_TOTAL_PIXELS:,} cap, decoded in full first"
            ),
            dropped_first=dropped,
        )

    async def _max_total_video_tokens(
        self, client: FuzzClient, config: RunConfig
    ) -> ScenarioResult:
        """The largest legal video-token payload the served window allows.

        Derived rather than fixed, for the reason the image axis derives its
        count: a deployment serving a fraction of the architectural window
        rejects a fixed count on context length, which says nothing about the
        video path and buries the axis under a result that never changes.
        """
        window, source = await served_context_window(client, config)
        per_frame_pair = estimate_tokens(
            TYPICAL_FRAME, TYPICAL_FRAME, 2, 1.0, 1.0
        )
        budget = window - PAYLOAD_OVERHEAD_TOKENS
        # Two frames per temporal patch, so frames scale at twice the patches.
        frames = max(2, (budget // max(1, per_frame_pair)) * 2)
        capped = min(
            frames,
            frames_for_pixels(
                TYPICAL_FRAME, TYPICAL_FRAME, VIDEO_MAX_TOTAL_PIXELS
            ),
        )
        if capped < 2:
            return self.make_result(
                self.name,
                "max_total_video_tokens",
                Verdict.INTERESTING,
                detail=(
                    f"{source} context window of {window} cannot hold one "
                    f"temporal patch ({per_frame_pair} tokens) -- this axis "
                    "cannot run against this deployment"
                ),
            )
        tokens = estimate_tokens(TYPICAL_FRAME, TYPICAL_FRAME, capped, 1.0, 1.0)
        part = video_part(
            TYPICAL_FRAME, TYPICAL_FRAME, capped, 1.0, "max-tokens", fps=1.0
        )
        resp, dropped = await post_once(
            client,
            video_payload(config.model, [part]),
            timeout=HEAVY_TIMEOUT_SEC,
        )
        pixel_bound = capped < frames
        return single_response_verdict(
            self.name,
            "max_total_video_tokens",
            resp,
            expect_reject=False,
            context=(
                f"1 video, {capped} frames, ~{tokens:,} vision tokens -- the "
                f"most the {source} window of {window} allows"
                + (
                    " (pixel cap bound this before the window did)"
                    if pixel_bound
                    else ""
                )
            ),
            dropped_first=dropped,
        )

    # ------------------------------------------------------------------
    # Sampling-rate boundaries
    # ------------------------------------------------------------------

    async def _fps_boundaries(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        """The absolute fps range, and the rate relative to the container.

        ``fps`` above the container's own rate is the case a synthetic fixture
        can accidentally send everywhere, and it is a clean 4xx: the vendor
        samples what was recorded and never interpolates.
        """
        cases: list[tuple[str, dict[str, Any], bool, str]] = [
            (
                "fps_at_floor",
                video_part(504, 504, 8, 1.0, "fps-floor", fps=FPS_MIN),
                False,
                f"fps={FPS_MIN} at the supported floor",
            ),
            (
                "fps_at_ceiling",
                video_part(504, 504, 30, FPS_MAX, "fps-ceil", fps=FPS_MAX),
                False,
                f"fps={FPS_MAX} at the supported ceiling, native {FPS_MAX}",
            ),
            (
                "fps_below_floor",
                video_part(504, 504, 8, 1.0, "fps-under", fps=FPS_MIN / 2),
                True,
                f"fps={FPS_MIN / 2} below the {FPS_MIN} floor",
            ),
            (
                "fps_above_ceiling",
                video_part(504, 504, 30, FPS_MAX, "fps-over", fps=FPS_MAX * 2),
                True,
                f"fps={FPS_MAX * 2} above the {FPS_MAX} ceiling",
            ),
            (
                "fps_above_native",
                video_part(504, 504, 8, 1.0, "fps-native", fps=4.0),
                True,
                "fps=4.0 against a 1.0 fps container -- frames cannot be "
                "invented",
            ),
        ]
        results = []
        for test, part, expect_reject, context in cases:
            resp, dropped = await post_once(
                client,
                video_payload(config.model, [part]),
                timeout=HEAVY_TIMEOUT_SEC,
            )
            results.append(
                single_response_verdict(
                    self.name,
                    test,
                    resp,
                    expect_reject=expect_reject,
                    context=context,
                    dropped_first=dropped,
                )
            )
        return results

    async def _skewed_aspect_ratio(
        self, client: FuzzClient, config: RunConfig
    ) -> ScenarioResult:
        """200:1 frames, where the resize rules leave one patch of height.

        The long-side downscale drops the short side to roughly 18 px, and
        because the two resize rules are mutually exclusive the 112 px floor
        never fires to rescue it -- the same shape the image axis covers, but
        applied to every frame of a clip.
        """
        part = video_part(112, 22400, 4, 1.0, "skew", detail="high")
        resp, dropped = await post_once(
            client,
            video_payload(config.model, [part]),
            timeout=HEAVY_TIMEOUT_SEC,
        )
        return single_response_verdict(
            self.name,
            "skewed_aspect_ratio",
            resp,
            expect_reject=False,
            context="4 frames at 112x22400 (200:1), detail=high",
            dropped_first=dropped,
        )

    # ------------------------------------------------------------------
    # Concurrency axis -- the MXSERV-395 shape
    # ------------------------------------------------------------------

    async def _concurrent_uncached_batches(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        """Concurrent multi-video requests, every frame a cache miss.

        Scales with ``--image-stress-nodes`` so a multi-node deployment is
        driven at the same per-node pressure a single pod sees. Decode runs on
        its own thread pool, so this contends for host CPU as well as for the
        encoder the image axis already saturates.
        """
        nodes = max(1, config.image_stress_nodes)
        concurrency = nodes * CONCURRENCY_PER_NODE
        payloads = []
        for i in range(concurrency * 2):
            parts = [
                video_part(
                    TYPICAL_FRAME,
                    TYPICAL_FRAME,
                    CONCURRENT_FRAMES[(i + v) % len(CONCURRENT_FRAMES)],
                    1.0,
                    f"conc-{i}-{v}",
                )
                for v in range(VIDEOS_PER_REQUEST)
            ]
            payloads.append(video_payload(config.model, parts))

        async with LivenessProbe(config) as probe:
            responses = await client.concurrent_requests(
                payloads, max_concurrent=concurrency
            )

        tally = BatchTally().add(responses)
        context = (
            f"{len(payloads)} requests at concurrency {concurrency} "
            f"({nodes} node(s) x {CONCURRENCY_PER_NODE}), "
            f"{VIDEOS_PER_REQUEST} videos each, every frame byte-unique"
        )
        failed = failure_verdict(
            self.name,
            "concurrent_uncached_videos",
            tally,
            config,
            context=context,
            probe=probe,
        )
        if failed is None:
            failed = self.make_result(
                self.name,
                "concurrent_uncached_videos",
                Verdict.INTERESTING
                if tally.client_errors or tally.other
                else Verdict.PASS,
                detail=(
                    f"{tally.summary()} -- not every request was served "
                    f"({context})"
                    if tally.client_errors or tally.other
                    else f"{tally.summary()} ({context})"
                ),
            )
        return [
            failed,
            probe_result(
                self.name, "concurrent_uncached_videos_liveness", probe, config
            ),
        ]


@register_scenario
class VideoStressRamp(BaseScenario):
    """One max-payload video request at a time, then several together.

    The boundary axes above establish that a single largest-legal request
    survives. This asks the question the axes cannot: whether several of them
    in flight together still do. Vision activations for a long clip have to
    fit alongside the KV cache, and the decode thread pool is shared across
    requests, so the failure this looks for appears only at a concurrency the
    single-request axes never reach.

    The ramp halts on the first rung that fails, because a rung that wedged or
    crashed the pod makes every later one unattributable -- the next failure
    would be reported against a concurrency the server never reached healthy.
    """

    name = "video_stress_ramp"
    description = (
        "Max-payload video requests ramped 1 -> 2 -> 4 concurrent, with a "
        "liveness probe"
    )
    tags = ["vision", "video", "oom", "memory", "crash", "concurrency"]
    model_filter = "minimax-m3"

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        frames = frames_for_pixels(
            TYPICAL_FRAME, TYPICAL_FRAME, VIDEO_MAX_TOTAL_PIXELS
        )
        results: list[ScenarioResult] = []
        for rung in RAMP:
            results.extend(await self._rung(client, config, rung, frames))
            if any(r.verdict == Verdict.FAIL for r in results[-2:]):
                results.append(
                    self.make_result(
                        self.name,
                        "video_ramp_halted",
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
        frames: int,
    ) -> list[ScenarioResult]:
        """Puts ``concurrency`` max-payload video requests in flight together."""
        nodes = max(1, config.image_stress_nodes)
        in_flight = concurrency * nodes
        payloads = [
            video_payload(
                config.model,
                [
                    video_part(
                        TYPICAL_FRAME,
                        TYPICAL_FRAME,
                        frames,
                        1.0,
                        f"ramp-{concurrency}-{req}",
                    )
                ],
            )
            for req in range(in_flight)
        ]

        async with LivenessProbe(config) as probe:
            responses = await client.concurrent_requests(
                payloads, max_concurrent=in_flight, timeout=HEAVY_TIMEOUT_SEC
            )

        test = f"video_ramp_x{concurrency}"
        pixels = in_flight * frames * TYPICAL_FRAME * TYPICAL_FRAME
        context = (
            f"{in_flight} concurrent requests x {frames} frames at "
            f"{TYPICAL_FRAME}x{TYPICAL_FRAME} = ~{pixels:,} pixels in flight"
        )
        tally = BatchTally().add(responses)
        # An OOM kills the pod outright, so it shows up as a mass failure or a
        # stall; a lone reset is neither, and must not halt the ramp on this
        # rung's behalf.
        result = failure_verdict(
            self.name, test, tally, config, context=context, probe=probe
        )
        if result is None:
            result = self.make_result(
                self.name,
                test,
                Verdict.INTERESTING
                if tally.client_errors or tally.other
                else Verdict.PASS,
                detail=(
                    f"{tally.summary()} -- rung not fully served ({context})"
                    if tally.client_errors or tally.other
                    else f"{tally.summary()} ({context})"
                ),
            )
        return [
            result,
            probe_result(self.name, f"{test}_liveness", probe, config),
        ]
