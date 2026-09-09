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
"""Unit tests for OverlapTextGenerationPipeline's completed-batch stats.

Under overlap scheduling, a batch's outputs are synchronized one ``execute()``
call after it was enqueued, so wall-clock time measured by the scheduler
describes the previously enqueued batch. ``_record_completed_batch_stats``
estimates the completed batch's execution time host-side — from the later of
its enqueue timestamp and the previous batch's sync timestamp until its own
sync — and ``take_completed_batch_stats`` hands the record to the scheduler
exactly once.

These tests exercise the timing/attribution logic in isolation using
stand-in objects; the methods only touch the attributes provided.
"""

from types import SimpleNamespace
from typing import Any

import pytest
from max.pipelines.lib import OverlapTextGenerationPipeline
from max.pipelines.modeling.types import BatchType


def _fake_pipeline() -> SimpleNamespace:
    return SimpleNamespace(
        _last_sync_monotonic=None,
        _completed_batch_stats=None,
    )


def _fake_batch(
    enqueue_monotonic: float,
    batch_type: BatchType = BatchType.CE,
    batch_size: int = 3,
    input_tokens: int = 4096,
    context_tokens: int = 8192,
) -> SimpleNamespace:
    return SimpleNamespace(
        enqueue_monotonic=enqueue_monotonic,
        inputs=SimpleNamespace(
            batch_type=batch_type,
            batch_size=batch_size,
            flat_batch=[object()] * batch_size,
            input_tokens=input_tokens,
            context_tokens=context_tokens,
        ),
    )


def _record(
    pipeline: Any,
    batch: Any,
    spec_decode_metrics: Any,
    sync_monotonic: float,
) -> None:
    OverlapTextGenerationPipeline._record_completed_batch_stats(
        pipeline, batch, spec_decode_metrics, sync_monotonic
    )


def _take(pipeline: Any) -> Any:
    return OverlapTextGenerationPipeline.take_completed_batch_stats(pipeline)


def test_saturated_gpu_uses_previous_sync_as_start_bound() -> None:
    """Back-to-back batches: the batch could only start executing when the
    previous batch finished, so its execution time is sync-to-sync."""
    pipe = _fake_pipeline()
    pipe._last_sync_monotonic = 10.5
    _record(pipe, _fake_batch(enqueue_monotonic=10.0), None, 10.8)
    stats = _take(pipe)
    assert stats is not None
    assert stats.execution_time_s == pytest.approx(0.3)
    assert stats.batch_type == BatchType.CE
    assert stats.batch_size == 3
    assert stats.num_input_tokens == 4096
    assert stats.num_context_tokens == 8192
    assert stats.num_output_tokens is None


def test_idle_gpu_uses_enqueue_time_as_start_bound() -> None:
    """First batch after idle: no previous sync; time runs from enqueue."""
    pipe = _fake_pipeline()
    _record(pipe, _fake_batch(enqueue_monotonic=10.0), None, 10.4)
    stats = _take(pipe)
    assert stats.execution_time_s == pytest.approx(0.4)


def test_stale_previous_sync_uses_enqueue_time_as_start_bound() -> None:
    """Previous sync happened long before this batch was enqueued (idle
    gap): the enqueue timestamp is the tighter start bound."""
    pipe = _fake_pipeline()
    pipe._last_sync_monotonic = 3.0
    _record(pipe, _fake_batch(enqueue_monotonic=10.0), None, 10.4)
    assert _take(pipe).execution_time_s == pytest.approx(0.4)


def test_execution_time_clamped_non_negative() -> None:
    pipe = _fake_pipeline()
    pipe._last_sync_monotonic = 11.0
    _record(pipe, _fake_batch(enqueue_monotonic=10.0), None, 10.9)
    assert _take(pipe).execution_time_s == 0.0


def test_spec_decode_stats_recorded() -> None:
    pipe = _fake_pipeline()
    spec = SimpleNamespace(
        output_tokens=13,
        draft_tokens_generated=24,
        draft_tokens_accepted=12,
        avg_acceptance_length=1.5,
        num_speculative_tokens=3,
        acceptance_rate_per_position=[0.75, 0.5, 0.25],
    )
    _record(pipe, _fake_batch(10.0, batch_type=BatchType.TG), spec, 10.2)
    stats = _take(pipe)
    assert stats.batch_type == BatchType.TG
    assert stats.num_output_tokens == 13
    assert stats.draft_tokens_generated == 24
    assert stats.draft_tokens_accepted == 12
    assert stats.avg_acceptance_length == 1.5
    assert stats.max_acceptance_length == 3
    assert stats.acceptance_rate_per_position == [0.75, 0.5, 0.25]


def test_take_pops_stats() -> None:
    pipe = _fake_pipeline()
    _record(pipe, _fake_batch(10.0), None, 10.1)
    assert _take(pipe) is not None
    assert _take(pipe) is None


def test_last_sync_advances_between_batches() -> None:
    """Each sync becomes the start bound for the next batch's estimate."""
    pipe = _fake_pipeline()
    _record(pipe, _fake_batch(enqueue_monotonic=10.0), None, 10.2)
    _record(pipe, _fake_batch(enqueue_monotonic=10.1), None, 10.9)
    assert _take(pipe).execution_time_s == pytest.approx(0.7)
