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
"""Tests for the maxserve.num_requests_queued gauge.

The gauge is a synchronous OTel ``Gauge``: every call replaces the
previously reported value rather than accumulating. ``BatchMetrics.publish_metrics``
calls ``METRICS.reqs_queued(self.num_pending_reqs)`` once per scheduler
iteration so the gauge mirrors the "Pending: N reqs" value emitted in
scheduler logs.

These tests assert that after each scheduler iteration the most recent
recorded value matches the actual depth of the CE / prefill queue.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from tests.serve.scheduler.common import (
    create_paged_scheduler,
    enqueue_request,
    run_until_completion,
)


class _GaugeRecorder:
    """Captures METRICS.reqs_queued(value) snapshots.

    Replaces the METRICS singleton inside ``scheduler.utils`` so the
    snapshots emitted from ``BatchMetrics.publish_metrics`` are captured
    while every other METRICS method becomes a no-op.
    """

    def __init__(self) -> None:
        self.values: list[int] = []

    @property
    def last(self) -> int | None:
        return self.values[-1] if self.values else None

    def reqs_queued(self, value: int) -> None:
        self.values.append(value)

    def __getattr__(self, _name: str) -> Any:
        return MagicMock()


def _patch_metrics(recorder: _GaugeRecorder) -> Any:
    """Patches the METRICS binding used by ``BatchMetrics.publish_metrics``."""
    return patch("max.serve.scheduler.utils.METRICS", recorder)


def test_publish_metrics_emits_snapshot() -> None:
    """Every scheduler iteration publishes the current queue depth.

    Drains 3 requests; the first iteration's snapshot should reflect
    however many requests are still in CE after batch construction. By
    the time the run completes, the gauge converges to 0.
    """
    recorder = _GaugeRecorder()
    with _patch_metrics(recorder):
        scheduler, request_queue = create_paged_scheduler(
            max_seq_len=128,
            num_blocks=64,
            max_batch_size=4,
            page_size=8,
        )
        for _ in range(3):
            enqueue_request(request_queue, prompt_len=16, max_seq_len=128)

        # Single iteration: drains, builds the CE batch, runs the
        # pipeline, and publishes batch metrics. publish_metrics fires
        # once per non-empty batch, so we expect at least one snapshot.
        scheduler.run_iteration()
        assert recorder.values, (
            "publish_metrics should have emitted at least one snapshot"
        )

        # Drain the rest of the work. The final published snapshot
        # must report zero pending requests.
        run_until_completion(scheduler)
        # The very last iteration that actually published metrics had
        # an empty CE queue (all requests in TG / done).
        assert recorder.last == 0
        assert len(scheduler.batch_constructor.all_ce_reqs) == 0


def test_snapshot_tracks_pending_count() -> None:
    """With max_batch_size=2 and 4 enqueued requests, the first CE
    iteration should publish a snapshot whose value equals the count of
    requests still waiting in CE after that iteration's batch (2).
    """
    recorder = _GaugeRecorder()
    with _patch_metrics(recorder):
        scheduler, request_queue = create_paged_scheduler(
            max_seq_len=128,
            num_blocks=64,
            max_batch_size=2,
            page_size=8,
        )
        # max_batch_size * dp * 2 = 4 — ensures the drain pulls every
        # request in a single pass.
        n = scheduler.max_items_per_drain
        assert n == 4
        for _ in range(n):
            enqueue_request(request_queue, prompt_len=16, max_seq_len=128)

        scheduler.run_iteration()

        # The first CE batch admits 2; the remaining 2 stay in ce_reqs
        # and that is what publish_metrics sampled as num_pending_reqs.
        assert recorder.last == len(scheduler.batch_constructor.all_ce_reqs)
        assert recorder.last == 2
