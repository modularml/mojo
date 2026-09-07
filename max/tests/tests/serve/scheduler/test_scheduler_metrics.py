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

import io
import json
import logging
import re
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

from max.pipelines.lib.vision_encoder_cache import (
    VideoEncoderMetrics,
    VisionEncoderMetrics,
)
from max.pipelines.modeling.types import (
    BatchType,
    CompletedBatchStats,
    TextGenerationInputs,
)
from max.serve.scheduler.utils import (
    BatchMetrics,
    SchedulerLogger,
    publish_completed_batch_metrics,
)
from pythonjsonlogger import jsonlogger
from tests.serve.scheduler.common import (
    FakeOverlapPipeline,
    create_paged_scheduler,
    enqueue_request,
)


def _make_metrics(**overrides: Any) -> BatchMetrics:
    base = dict[str, Any](
        batch_type=BatchType.CE,
        batch_size=1,
        max_batch_size=2,
        terminated_reqs=4,
        num_pending_reqs=5,
        num_input_tokens=6,
        max_batch_input_tokens=7,
        num_context_tokens=8,
        max_batch_total_tokens=9,
        batch_creation_time_s=10.0,
        batch_execution_time_s=11.0,
        prompt_throughput=12.0,
        generation_throughput=13.0,
        total_preemption_count=14,
        used_kv_pct=0.15,
        total_kv_blocks=16,
        cache_hit_rate=0.17,
        cache_hit_tokens=18,
        cache_miss_tokens=19,
        device_blocks_served=0,
        used_host_kv_pct=0.20,
        total_host_kv_bytes=21 * 1024,
        h2d_bytes_copied=22 * 1024,
        d2h_bytes_copied=23 * 1024,
        disk_bytes_read=0,
        disk_bytes_written=0,
        used_disk_kv_pct=0.0,
        total_disk_kv_bytes=0,
        inflight_disk_ops=0,
        draft_tokens_generated=0,
        draft_tokens_accepted=0,
        avg_acceptance_length=0.0,
        max_acceptance_length=0,
        acceptance_rate_per_position=[],
        nixl_read_latency_avg_ms=0.0,
        nixl_write_latency_avg_ms=0.0,
        rpc_acquire_latency_avg_ms=0.0,
        rpc_read_latency_avg_ms=0.0,
        num_new_admissions=1,
    )

    base.update(overrides)
    return BatchMetrics(**base)


def test_metric_to_string() -> None:
    metrics = BatchMetrics(
        batch_type=BatchType.CE,
        batch_size=1,
        max_batch_size=2,
        terminated_reqs=4,
        num_pending_reqs=5,
        num_input_tokens=6,
        max_batch_input_tokens=7,
        num_context_tokens=8,
        max_batch_total_tokens=9,
        batch_creation_time_s=10.0,
        batch_execution_time_s=11.0,
        prompt_throughput=12.0,
        generation_throughput=13.0,
        total_preemption_count=14,
        used_kv_pct=0.15,
        total_kv_blocks=16,
        cache_hit_rate=0.17,
        cache_hit_tokens=18,
        cache_miss_tokens=19,
        device_blocks_served=0,
        used_host_kv_pct=0.20,
        total_host_kv_bytes=21 * 1024,
        h2d_bytes_copied=22 * 1024,
        d2h_bytes_copied=23 * 1024,
        disk_bytes_read=0,
        disk_bytes_written=0,
        used_disk_kv_pct=0.0,
        total_disk_kv_bytes=0,
        inflight_disk_ops=0,
        draft_tokens_generated=0,
        draft_tokens_accepted=0,
        avg_acceptance_length=0.0,
        max_acceptance_length=0,
        acceptance_rate_per_position=[],
        nixl_read_latency_avg_ms=0.0,
        nixl_write_latency_avg_ms=0.0,
        rpc_acquire_latency_avg_ms=0.0,
        rpc_read_latency_avg_ms=0.0,
        num_new_admissions=1,
    )

    assert (
        metrics.pretty_format()
        == r"Executed CE batch with 1 reqs | Terminated: 4 reqs, Pending: 5 reqs | Input Tokens: 6/7 toks | Context Tokens: 8/9 toks | Prompt Tput: 12.0 tok/s, Generation Tput: 13.0 tok/s | Batch creation: 10.00s, Execution: 11.00s | KVCache usage: 15.0% of 16 blocks, Cache hit rate: 17.0% (18 hit, 19 miss) | Host KVCache Usage: 20.0% of 21.00 KiB, Copied: 22.00 KiB H2D, 23.00 KiB D2H | All Preemptions: 14 reqs"
    )

    metrics.total_kv_blocks = 0
    metrics.total_host_kv_bytes = 0
    assert (
        metrics.pretty_format()
        == r"Executed CE batch with 1 reqs | Terminated: 4 reqs, Pending: 5 reqs | Input Tokens: 6/7 toks | Context Tokens: 8/9 toks | Prompt Tput: 12.0 tok/s, Generation Tput: 13.0 tok/s | Batch creation: 10.00s, Execution: 11.00s | All Preemptions: 14 reqs"
    )

    metrics.draft_tokens_generated = 10
    metrics.draft_tokens_accepted = 5
    metrics.avg_acceptance_length = 2.5
    metrics.max_acceptance_length = 3
    assert (
        metrics.pretty_format()
        == r"Executed CE batch with 1 reqs | Terminated: 4 reqs, Pending: 5 reqs | Input Tokens: 6/7 toks | Context Tokens: 8/9 toks | Prompt Tput: 12.0 tok/s, Generation Tput: 13.0 tok/s | Batch creation: 10.00s, Execution: 11.00s | Draft Tokens: 5/10 (50.00%) accepted, Acceptance Len: 2.50 / 3 toks (accepted drafts/step; 3.50 toks/step incl bonus) | All Preemptions: 14 reqs"
    )

    # Test with per-position acceptance rates
    metrics.acceptance_rate_per_position = [0.90, 0.75, 0.50]
    assert (
        metrics.pretty_format()
        == r"Executed CE batch with 1 reqs | Terminated: 4 reqs, Pending: 5 reqs | Input Tokens: 6/7 toks | Context Tokens: 8/9 toks | Prompt Tput: 12.0 tok/s, Generation Tput: 13.0 tok/s | Batch creation: 10.00s, Execution: 11.00s | Draft Tokens: 5/10 (50.00%) accepted, Acceptance Len: 2.50 / 3 toks (accepted drafts/step; 3.50 toks/step incl bonus), Per-Pos: [p0=90%, p1=75%, p2=50%] | All Preemptions: 14 reqs"
    )


def test_metric_to_string_with_disk_kv() -> None:
    # When the tiered connector is active, the log line shows Disk: read/written
    # counts inside the host clause and a separate Disk KVCache Usage clause.
    metrics = _make_metrics(
        disk_bytes_read=24 * 1024,
        disk_bytes_written=25 * 1024,
        used_disk_kv_pct=0.30,
        total_disk_kv_bytes=100 * 1024,
        inflight_disk_ops=99,
    )

    formatted = metrics.pretty_format()
    assert (
        "Host KVCache Usage: 20.0% of 21.00 KiB, "
        "Copied: 22.00 KiB H2D, 23.00 KiB D2H, "
        "Disk: 24.00 KiB read, 25.00 KiB written | "
        "Disk KVCache Usage: 30.0% of 100.00 KiB, "
        "Inflight Disk Ops: 99 |"
    ) in formatted


def _overlap_metrics_overrides(**overrides: Any) -> dict[str, Any]:
    """Overrides that clear the state clauses so overlap-format tests can
    assert full strings without the KV/host-KV sections."""
    base = dict[str, Any](
        overlap_active=True,
        total_kv_blocks=0,
        used_kv_pct=0.0,
        num_new_admissions=0,
        cache_hit_rate=0.0,
        cache_hit_tokens=0,
        cache_miss_tokens=0,
        device_blocks_served=0,
        total_host_kv_bytes=0,
        used_host_kv_pct=0.0,
        h2d_bytes_copied=0,
        d2h_bytes_copied=0,
    )
    base.update(overrides)
    return base


def test_metric_to_string_overlap_scheduler() -> None:
    # Under the overlap scheduler, the batch enqueued this iteration and the
    # batch whose outputs were synchronized this iteration are different: the
    # line renders one coherent clause per batch (Enqueued/Completed) instead
    # of mixing their fields under a single identity with the old
    # "Previous Execution:" caveat label.
    completed = CompletedBatchStats(
        batch_type=BatchType.CE,
        batch_size=3,
        num_input_tokens=4000,
        num_context_tokens=8000,
        execution_time_s=0.25,
    )
    metrics = _make_metrics(
        **_overlap_metrics_overrides(
            batch_type=BatchType.TG, completed=completed
        )
    )

    assert metrics.pretty_format() == (
        "Enqueued TG batch with 1 reqs | Pending: 5 reqs | "
        "Input Tokens: 6/7 toks | Context Tokens: 8/9 toks | "
        "Batch creation: 10.00s | "
        "Completed CE batch with 3 reqs | Terminated: 4 reqs | "
        "Completed Input Tokens: 4000 toks | "
        "Prompt Tput: 16.0K tok/s, Generation Tput: 12.0 tok/s | "
        "Execution: 250.00ms | "
        "All Preemptions: 14 reqs"
    )
    assert "Previous Execution" not in metrics.pretty_format()


def test_metric_to_string_overlap_token_clauses_are_distinguishable() -> None:
    """The enqueued and completed clauses each carry their own token count
    under distinct labels, so a log consumer can correlate the completed
    batch's duration with its own size rather than the enqueued batch's."""
    completed = CompletedBatchStats(
        batch_type=BatchType.CE,
        batch_size=3,
        num_input_tokens=4000,
        num_context_tokens=8000,
        execution_time_s=0.25,
    )
    formatted = _make_metrics(
        **_overlap_metrics_overrides(
            batch_type=BatchType.TG, completed=completed
        )
    ).pretty_format()
    # Enqueued batch: 6 of a 7-token budget. Completed batch: its own 4000.
    assert "Input Tokens: 6/7 toks | " in formatted
    assert "Completed Input Tokens: 4000 toks | " in formatted
    # A bare "Input Tokens:" match must not pick up the completed clause.
    assert re.findall(r"(?<!Completed )Input Tokens:\s+(\S+)", formatted) == [
        "6/7"
    ]


def test_metric_to_string_overlap_no_completed_batch() -> None:
    """First overlap iteration: nothing completed yet, so no execution time,
    throughput, or termination clause may appear under the enqueued batch's
    identity."""
    metrics = _make_metrics(**_overlap_metrics_overrides())
    formatted = metrics.pretty_format()
    assert formatted == (
        "Enqueued CE batch with 1 reqs | Pending: 5 reqs | "
        "Input Tokens: 6/7 toks | Context Tokens: 8/9 toks | "
        "Batch creation: 10.00s | "
        "All Preemptions: 14 reqs"
    )


def test_metric_to_string_overlap_drain_iteration() -> None:
    """Drain iteration (empty inputs flushing the final deferred batch): the
    line describes only the completed batch instead of a misleading
    zero-request enqueued batch."""
    completed = CompletedBatchStats(
        batch_type=BatchType.CE,
        batch_size=2,
        num_input_tokens=1000,
        num_context_tokens=2000,
        execution_time_s=0.5,
    )
    metrics = _make_metrics(
        **_overlap_metrics_overrides(
            batch_type=BatchType.TG, batch_size=0, completed=completed
        )
    )
    formatted = metrics.pretty_format()
    assert formatted.startswith("Completed CE batch with 2 reqs | ")
    assert "Enqueued" not in formatted


def test_metric_to_string_overlap_completed_spec_decode() -> None:
    """Spec-decode acceptance stats describe the completed batch and render
    inside its clause."""
    completed = CompletedBatchStats(
        batch_type=BatchType.TG,
        batch_size=4,
        num_input_tokens=4,
        num_context_tokens=100,
        execution_time_s=0.1,
        num_output_tokens=12,
        draft_tokens_generated=10,
        draft_tokens_accepted=5,
        avg_acceptance_length=2.5,
        max_acceptance_length=3,
    )
    metrics = _make_metrics(
        **_overlap_metrics_overrides(
            batch_type=BatchType.TG, completed=completed
        )
    )
    formatted = metrics.pretty_format()
    assert (
        "Draft Tokens: 5/10 (50.00%) accepted, "
        "Acceptance Len: 2.50 / 3 toks "
        "(accepted drafts/step; 3.50 toks/step incl bonus) | " in formatted
    )
    # Generation throughput uses the completed batch's output tokens.
    assert "Generation Tput: 120.0 tok/s" in formatted


def test_metric_to_string_continuation_only_ce_batch() -> None:
    # A CE batch with only chunked-prefill continuations of already-admitted
    # requests has num_new_admissions=0. In that case the cache-hit clause
    # would otherwise be a misleading "0.0% (0 hit, N miss)" reading and is
    # suppressed; the KVCache usage clause is still shown.
    metrics = BatchMetrics(
        batch_type=BatchType.CE,
        batch_size=1,
        max_batch_size=2,
        terminated_reqs=0,
        num_pending_reqs=0,
        num_input_tokens=1862,
        max_batch_input_tokens=4096,
        num_context_tokens=50545,
        max_batch_total_tokens=262144,
        batch_creation_time_s=0.001,
        batch_execution_time_s=0.2,
        prompt_throughput=9100.0,
        generation_throughput=4.9,
        total_preemption_count=0,
        used_kv_pct=0.101,
        total_kv_blocks=13359,
        cache_hit_rate=0.0,
        cache_hit_tokens=0,
        cache_miss_tokens=0,
        device_blocks_served=0,
        used_host_kv_pct=0.0,
        total_host_kv_bytes=0,
        h2d_bytes_copied=0,
        d2h_bytes_copied=0,
        disk_bytes_read=0,
        disk_bytes_written=0,
        inflight_disk_ops=0,
        used_disk_kv_pct=0.0,
        total_disk_kv_bytes=0,
        draft_tokens_generated=0,
        draft_tokens_accepted=0,
        avg_acceptance_length=0.0,
        max_acceptance_length=0,
        acceptance_rate_per_position=[],
        nixl_read_latency_avg_ms=0.0,
        nixl_write_latency_avg_ms=0.0,
        rpc_acquire_latency_avg_ms=0.0,
        rpc_read_latency_avg_ms=0.0,
        num_new_admissions=0,
    )

    formatted = metrics.pretty_format()
    assert "KVCache usage: 10.1% of 13359 blocks |" in formatted
    assert "Cache hit rate" not in formatted
    assert "hit," not in formatted
    assert "miss)" not in formatted


def test_to_log_extra_required_fields() -> None:
    extra = _make_metrics().to_log_extra()

    assert extra["event"] == "batch_metrics"
    assert extra["batch_type"] == "CE"

    assert extra["batch_size"] == 1
    assert extra["num_input_tokens"] == 6
    assert extra["max_batch_input_tokens"] == 7
    assert extra["max_batch_total_tokens"] == 9
    assert extra["batch_execution_time_ms"] == 11000.0
    assert extra["batch_creation_time_ms"] == 10000.0

    assert extra["used_kv_pct"] == 0.15
    assert extra["total_kv_blocks"] == 16
    assert extra["num_new_admissions"] == 1
    assert extra["cache_hit_rate"] == 0.17
    assert extra["cache_hit_tokens"] == 18
    assert extra["cache_miss_tokens"] == 19

    assert extra["used_host_kv_pct"] == 0.20
    assert extra["total_host_kv_bytes"] == 21 * 1024

    # ensure data is flat
    for k, v in extra.items():
        assert not isinstance(v, (list, dict)), (
            f"{k} is nested ({type(v).__name__})"
        )


def test_to_log_extra_covers_every_pretty_format_number() -> None:
    """Every number the log line prints must also be a queryable attribute.

    A value that lives only in the message text cannot be faceted or graphed
    in a log backend, so this pins down the full set for a batch that
    exercises every clause at once.
    """
    metrics = _make_metrics(
        batch_type=BatchType.TG,
        disk_bytes_read=24 * 1024,
        disk_bytes_written=25 * 1024,
        used_disk_kv_pct=0.30,
        total_disk_kv_bytes=100 * 1024,
        inflight_disk_ops=99,
        draft_tokens_generated=95,
        draft_tokens_accepted=42,
        avg_acceptance_length=2.21,
        max_acceptance_length=5,
        acceptance_rate_per_position=[0.79, 0.47],
        dp_active_tokens=8008,
        dp_step_capacity_tokens=16000,
        cross_replica_blocks_copied=3,
        cross_replica_bytes_copied=4096,
    )
    extra = metrics.to_log_extra()

    assert extra["inflight_disk_ops"] == 99
    assert extra["disk_bytes_read"] == 24 * 1024
    assert extra["disk_bytes_written"] == 25 * 1024

    assert extra["max_acceptance_length"] == 5
    assert extra["draft_token_acceptance_rate"] == 42 / 95
    # Per-position rates flatten to indexed keys rather than an array.
    assert extra["acceptance_rate_p0"] == 0.79
    assert extra["acceptance_rate_p1"] == 0.47
    assert "acceptance_rate_p2" not in extra

    assert extra["dp_active_tokens"] == 8008
    assert extra["dp_step_capacity_tokens"] == 16000
    assert extra["cross_replica_blocks_copied"] == 3
    assert extra["cross_replica_bytes_copied"] == 4096

    assert extra["device_blocks_served"] == 0

    for key, value in extra.items():
        assert not isinstance(value, (list, dict)), (
            f"{key} is nested ({type(value).__name__})"
        )


def test_to_log_extra_gating_continuation_only_ce() -> None:
    extra = _make_metrics(
        num_new_admissions=0,
        cache_hit_rate=0.0,
        cache_hit_tokens=0,
        cache_miss_tokens=0,
        device_blocks_served=0,
        total_host_kv_bytes=0,
        used_host_kv_pct=0.0,
        h2d_bytes_copied=0,
        d2h_bytes_copied=0,
    ).to_log_extra()

    assert "used_kv_pct" in extra

    assert "cache_hit_rate" not in extra
    assert "cache_hit_tokens" not in extra
    assert "cache_miss_tokens" not in extra
    assert "num_new_admissions" not in extra
    assert "device_blocks_served" not in extra

    assert "used_host_kv_pct" not in extra
    assert "total_host_kv_bytes" not in extra
    assert "h2d_bytes_copied" not in extra
    assert "d2h_bytes_copied" not in extra

    assert "inflight_disk_ops" not in extra

    assert "draft_tokens_generated" not in extra
    assert "draft_token_acceptance_rate" not in extra
    assert "max_acceptance_length" not in extra
    assert "acceptance_rate_p0" not in extra
    assert "nixl_read_latency_avg_ms" not in extra

    assert "dp_active_tokens" not in extra
    assert "cross_replica_blocks_copied" not in extra


def test_to_log_extra_overlap_completed_fields() -> None:
    """Under overlap, per-iteration execution fields are omitted (they would
    describe the wrong batch) and the completed batch's fields are emitted
    under completed_* names with its own identity."""
    completed = CompletedBatchStats(
        batch_type=BatchType.CE,
        batch_size=3,
        num_input_tokens=4000,
        num_context_tokens=8000,
        execution_time_s=0.25,
    )
    extra = _make_metrics(
        batch_type=BatchType.TG, overlap_active=True, completed=completed
    ).to_log_extra()

    assert extra["overlap_active"] is True
    assert extra["batch_type"] == "TG"
    # Misattributed per-iteration values must be absent.
    assert "batch_execution_time_ms" not in extra
    assert "prompt_throughput" not in extra
    assert "generation_throughput" not in extra
    assert "terminated_reqs" not in extra
    # Completed-batch fields carry the completed batch's identity.
    assert extra["completed_batch_type"] == "CE"
    assert extra["completed_batch_size"] == 3
    assert extra["completed_execution_time_ms"] == 250.0
    assert extra["completed_prompt_throughput"] == 16000.0
    assert extra["completed_generation_throughput"] == 12.0
    assert extra["completed_terminated_reqs"] == 4

    # Data stays flat for the JSON formatter.
    for k, v in extra.items():
        assert not isinstance(v, (list, dict)), (
            f"{k} is nested ({type(v).__name__})"
        )


def test_to_log_extra_overlap_without_completed_omits_execution() -> None:
    """Overlap iteration with nothing completed: no execution/throughput
    fields at all."""
    extra = _make_metrics(overlap_active=True).to_log_extra()
    assert extra["overlap_active"] is True
    assert "batch_execution_time_ms" not in extra
    assert "prompt_throughput" not in extra
    assert "generation_throughput" not in extra
    assert "terminated_reqs" not in extra
    assert not any(k.startswith("completed_") for k in extra)


def test_to_log_extra_serializes_via_jsonlogger() -> None:
    metrics = _make_metrics()

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(
        jsonlogger.JsonFormatter("%(levelname)s %(message)s", timestamp=True)
    )

    test_logger = logging.getLogger(
        "max.serve.test_batch_metrics_structured_emission"
    )
    test_logger.handlers = [handler]
    test_logger.setLevel(logging.INFO)
    test_logger.propagate = False

    test_logger.info(metrics.pretty_format(), extra=metrics.to_log_extra())

    payload = json.loads(buf.getvalue().strip().splitlines()[-1])

    assert payload["event"] == "batch_metrics"
    assert payload["batch_type"] == "CE"
    assert payload["batch_size"] == 1
    assert payload["batch_execution_time_ms"] == 11000.0
    assert "Executed CE batch" in payload["message"]


def test_publish_metrics_default_path() -> None:
    """CE batch with device KV + host KV + cache hits active.
    Spec-decode and dKV are off by default in ``_make_metrics``; this test
    pins down the always-on batch fields, the active-subsystem calls, and
    the absence of inactive-subsystem calls.
    """
    metrics = (
        _make_metrics()
    )  # CE, total_kv_blocks=16, host KV active, num_new_admissions=1
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        metrics.publish_metrics()
    mock_metrics.batch_size.assert_called_once_with(1, batch_type="CE")
    mock_metrics.batch_input_tokens.assert_called_once_with(6, batch_type="CE")
    mock_metrics.batch_context_tokens.assert_called_once_with(
        8, batch_type="CE"
    )
    mock_metrics.batch_terminated_reqs.assert_called_once_with(
        4, batch_type="CE"
    )
    mock_metrics.batch_pending_reqs.assert_called_once_with(5, batch_type="CE")
    mock_metrics.batch_prompt_throughput.assert_called_once_with(
        12.0, batch_type="CE"
    )
    mock_metrics.batch_generation_throughput.assert_called_once_with(
        13.0, batch_type="CE"
    )
    mock_metrics.batch_creation_time.assert_called_once_with(
        10000.0, batch_type="CE"
    )
    mock_metrics.batch_execution_time.assert_called_once_with(
        11000.0, batch_type="CE"
    )
    # Device KV cluster.
    mock_metrics.cache_num_total_blocks.assert_called_once_with(16)
    mock_metrics.cache_used_kv_pct.assert_called_once_with(15.0)
    # Cache-hit clause (CE + num_new_admissions=1). No connector in this
    # fixture, so every hit token is tagged to the device tier.
    mock_metrics.cache_hits.assert_called_once_with(18, tier="g0")
    mock_metrics.cache_misses.assert_called_once_with(19)
    # Host KV clause (total_host_kv_bytes nonzero).
    mock_metrics.cache_used_host_kv_pct.assert_called_once_with(20.0)
    mock_metrics.cache_h2d_bytes_copied.assert_called_once_with(22 * 1024)
    mock_metrics.cache_d2h_bytes_copied.assert_called_once_with(23 * 1024)
    # Inactive subsystems must not emit anything.
    mock_metrics.spec_decode_avg_acceptance_length.assert_not_called()
    mock_metrics.spec_decode_acceptance_rate_per_position.assert_not_called()
    mock_metrics.dkv_nixl_read_latency.assert_not_called()
    mock_metrics.dkv_nixl_read_gib_per_s.assert_not_called()
    mock_metrics.dkv_nixl_write_latency.assert_not_called()
    mock_metrics.dkv_nixl_write_gib_per_s.assert_not_called()
    mock_metrics.dkv_rpc_acquire_latency.assert_not_called()
    mock_metrics.dkv_rpc_read_latency.assert_not_called()
    # Disk KV gated off (total_disk_kv_bytes=0).
    mock_metrics.cache_used_disk_kv_pct.assert_not_called()


def test_publish_metrics_subsystem_gating() -> None:
    """TG batch with spec-decode + dKV active, host KV / cache-hits off.
    Inverse of the default-path test: pins down that the per-subsystem
    guards are honored independently.
    """
    metrics = _make_metrics(
        batch_type=BatchType.TG,
        total_kv_blocks=0,
        used_kv_pct=0.0,
        num_new_admissions=0,
        cache_hit_rate=0.0,
        cache_hit_tokens=0,
        cache_miss_tokens=0,
        device_blocks_served=0,
        total_host_kv_bytes=0,
        used_host_kv_pct=0.0,
        h2d_bytes_copied=0,
        d2h_bytes_copied=0,
        draft_tokens_generated=10,
        draft_tokens_accepted=5,
        avg_acceptance_length=2.5,
        max_acceptance_length=3,
        acceptance_rate_per_position=[0.9, 0.5],
        nixl_read_latency_avg_ms=4.0,
        nixl_write_latency_avg_ms=5.0,
        nixl_read_gib_per_s=1.5,
        nixl_write_gib_per_s=2.5,
        rpc_acquire_latency_avg_ms=0.0,
        rpc_read_latency_avg_ms=0.0,
    )
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        metrics.publish_metrics()
    # Always-on path uses the TG label.
    mock_metrics.batch_size.assert_called_once_with(1, batch_type="TG")
    mock_metrics.batch_execution_time.assert_called_once_with(
        11000.0, batch_type="TG"
    )
    # Device KV gated off (total_kv_blocks=0).
    mock_metrics.cache_used_kv_pct.assert_not_called()
    # Cache-hit clause off (TG + num_new_admissions=0).
    mock_metrics.cache_hits.assert_not_called()
    mock_metrics.cache_misses.assert_not_called()
    # Host KV gated off.
    mock_metrics.cache_used_host_kv_pct.assert_not_called()
    mock_metrics.cache_h2d_bytes_copied.assert_not_called()
    mock_metrics.cache_d2h_bytes_copied.assert_not_called()
    # Spec-decode active.
    mock_metrics.spec_decode_avg_acceptance_length.assert_called_once_with(2.5)
    assert mock_metrics.spec_decode_acceptance_rate_per_position.call_count == 2
    # dKV NIXL active (latency + GiB/s emitted as paired values under one guard).
    mock_metrics.dkv_nixl_read_latency.assert_called_once_with(4.0)
    mock_metrics.dkv_nixl_read_gib_per_s.assert_called_once_with(1.5)
    mock_metrics.dkv_nixl_write_latency.assert_called_once_with(5.0)
    mock_metrics.dkv_nixl_write_gib_per_s.assert_called_once_with(2.5)
    # RPC inactive (rpc_*_avg_ms=0.0).
    mock_metrics.dkv_rpc_acquire_latency.assert_not_called()
    mock_metrics.dkv_rpc_read_latency.assert_not_called()
    # Disk KV gated off (total_disk_kv_bytes=0).
    mock_metrics.cache_used_disk_kv_pct.assert_not_called()


def test_publish_metrics_disk_kv_active() -> None:
    """Batch with disk KV cache active emits the disk usage metric."""
    metrics = _make_metrics(
        total_disk_kv_bytes=100 * 1024,
        used_disk_kv_pct=0.30,
    )
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        metrics.publish_metrics()
    mock_metrics.cache_used_disk_kv_pct.assert_called_once_with(30.0)


# ---------------------------------------------------------------------------
# Vision encoder metrics tests
# ---------------------------------------------------------------------------


def _make_vision_metrics(**overrides: Any) -> VisionEncoderMetrics:
    base = dict[str, Any](
        num_images_total=4,
        num_images_encoded=3,
        num_images_cached=1,
        num_patches_encoded=1200,
        num_tokens_encoded=256,
    )
    base.update(overrides)
    return VisionEncoderMetrics(**base)


def test_vision_metrics_cache_hit_rate() -> None:
    vm = _make_vision_metrics()
    assert vm.cache_hit_rate == 0.25
    # No images -> avoid divide-by-zero, report 0.0.
    assert VisionEncoderMetrics().cache_hit_rate == 0.0


def test_metric_to_string_with_vision() -> None:
    # Vision info is appended inline to the language model batch line.
    metrics = _make_metrics(vision_metrics=_make_vision_metrics())
    assert (
        "Vision Encoder: 3 imgs, 1200 patches, 256 toks encoded, "
        "cache hit rate 25.0% (1 hit, 3 miss) |"
    ) in metrics.pretty_format()


def test_metric_to_string_no_vision_clause_when_absent() -> None:
    # No vision metrics at all (text-only model).
    assert "Vision Encoder" not in _make_metrics().pretty_format()
    # Vision metrics present but with no images (guarded off).
    empty = _make_metrics(vision_metrics=VisionEncoderMetrics())
    assert "Vision Encoder" not in empty.pretty_format()


def test_to_log_extra_vision() -> None:
    extra = _make_metrics(vision_metrics=_make_vision_metrics()).to_log_extra()
    assert extra["vision_images_total"] == 4
    assert extra["vision_images_encoded"] == 3
    assert extra["vision_images_cached"] == 1
    assert extra["vision_patches_encoded"] == 1200
    assert extra["vision_tokens_encoded"] == 256
    assert extra["vision_cache_hit_rate"] == 0.25

    # Absent for text-only batches.
    assert "vision_images_total" not in _make_metrics().to_log_extra()


def test_publish_metrics_vision_active() -> None:
    metrics = _make_metrics(vision_metrics=_make_vision_metrics())
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        metrics.publish_metrics()
    mock_metrics.vision_images_encoded.assert_called_once_with(3)
    mock_metrics.vision_images_cached.assert_called_once_with(1)
    mock_metrics.vision_patches_encoded.assert_called_once_with(1200)
    mock_metrics.vision_tokens_encoded.assert_called_once_with(256)
    mock_metrics.vision_cache_hit_rate.assert_called_once_with(25.0)


def test_publish_metrics_vision_gated_off() -> None:
    # No vision metrics -> no vision emissions.
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        _make_metrics().publish_metrics()
    mock_metrics.vision_images_encoded.assert_not_called()
    mock_metrics.vision_cache_hit_rate.assert_not_called()


# ---------------------------------------------------------------------------
# Video encoder metrics tests
# ---------------------------------------------------------------------------


def _make_video_metrics(**overrides: Any) -> VideoEncoderMetrics:
    base = dict[str, Any](
        num_clips_total=4,
        num_clips_encoded=3,
        num_clips_cached=1,
        frame_counts=[16, 32, 8],
        num_tokens_encoded=512,
        encoding_time_ms=42.5,
    )
    base.update(overrides)
    return VideoEncoderMetrics(**base)


def test_video_metrics_cache_hit_rate() -> None:
    vm = _make_video_metrics()
    assert vm.cache_hit_rate == 0.25
    # No clips -> avoid divide-by-zero, report 0.0.
    assert VideoEncoderMetrics().cache_hit_rate == 0.0


def test_metric_to_string_with_video() -> None:
    metrics = _make_metrics(video_metrics=_make_video_metrics())
    assert (
        "Video Encoder: 3 clips, 56 frames, 512 toks encoded, 42.5ms, "
        "cache hit rate 25.0% (1 hit, 3 miss) |"
    ) in metrics.pretty_format()


def test_metric_to_string_no_video_clause_when_absent() -> None:
    assert "Video Encoder" not in _make_metrics().pretty_format()
    empty = _make_metrics(video_metrics=VideoEncoderMetrics())
    assert "Video Encoder" not in empty.pretty_format()


def test_to_log_extra_video() -> None:
    extra = _make_metrics(video_metrics=_make_video_metrics()).to_log_extra()
    assert extra["video_clips_total"] == 4
    assert extra["video_clips_encoded"] == 3
    assert extra["video_clips_cached"] == 1
    assert extra["video_frames_encoded"] == 56
    assert extra["video_tokens_encoded"] == 512
    assert extra["video_encoding_time_ms"] == 42.5
    assert extra["video_cache_hit_rate"] == 0.25

    assert "video_clips_total" not in _make_metrics().to_log_extra()


def test_publish_metrics_video_active() -> None:
    metrics = _make_metrics(video_metrics=_make_video_metrics())
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        metrics.publish_metrics()
    mock_metrics.video_clips_encoded.assert_called_once_with(3)
    mock_metrics.video_tokens_encoded.assert_called_once_with(512)
    mock_metrics.video_encoding_time_milliseconds.assert_called_once_with(42.5)
    assert mock_metrics.video_frames_per_clip.call_count == 3
    mock_metrics.video_frames_per_clip.assert_any_call(16)
    mock_metrics.video_frames_per_clip.assert_any_call(32)
    mock_metrics.video_frames_per_clip.assert_any_call(8)


def test_publish_metrics_video_gated_off() -> None:
    # No video metrics -> no video emissions.
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        _make_metrics().publish_metrics()
    mock_metrics.video_clips_encoded.assert_not_called()
    mock_metrics.video_frames_per_clip.assert_not_called()


# ---------------------------------------------------------------------------
# _SpeculativeDecodingMetrics tests
# ---------------------------------------------------------------------------


def _make_spec_metrics(
    num_speculative_tokens: int,
    accepted_per_position: list[int],
    num_verifications: int,
) -> Any:
    from max.pipelines.speculative.utils import _SpeculativeDecodingMetrics

    return _SpeculativeDecodingMetrics(
        num_speculative_tokens=num_speculative_tokens,
        accepted_per_position=accepted_per_position,
        num_verifications=num_verifications,
    )


def test_spec_decode_metrics_output_tokens() -> None:
    metrics = _make_spec_metrics(
        num_speculative_tokens=3,
        accepted_per_position=[4, 3, 1],
        num_verifications=5,
    )
    assert metrics.output_tokens == 13


def test_spec_decode_metrics_output_tokens_zero_verifications() -> None:
    metrics = _make_spec_metrics(
        num_speculative_tokens=3,
        accepted_per_position=[0, 0, 0],
        num_verifications=0,
    )
    assert metrics.output_tokens == 0


def test_spec_decode_metrics_properties() -> None:
    metrics = _make_spec_metrics(
        num_speculative_tokens=3,
        accepted_per_position=[6, 4, 2],
        num_verifications=8,
    )
    assert metrics.draft_tokens_accepted == 12
    assert metrics.draft_tokens_generated == 24
    assert metrics.acceptance_rate == 0.5
    assert metrics.avg_acceptance_length == 1.5
    assert metrics.acceptance_rate_per_position == [0.75, 0.5, 0.25]
    assert metrics.output_tokens == 20


def test_spec_decode_metrics_per_position_empty_when_no_verifications() -> None:
    # Regression: this returned [0.0] * num_speculative_tokens, injecting an
    # all-zero row per zero-verification batch into every rate consumer.
    metrics = _make_spec_metrics(
        num_speculative_tokens=3,
        accepted_per_position=[0, 0, 0],
        num_verifications=0,
    )
    assert metrics.acceptance_rate_per_position == []


# ---------------------------------------------------------------------------
# BatchMetrics.create spec-decode tests
# ---------------------------------------------------------------------------


def _mock_inputs(
    batch_size: int, batch_type: BatchType
) -> TextGenerationInputs[Any]:
    contexts = [MagicMock() for _ in range(batch_size)]
    for context in contexts:
        context._is_padding_ctx = False
        context.tokens.active_length = 0
        context.tokens.processed_length = 0
        context.tokens.generated_length = 1

    inputs: TextGenerationInputs[Any] = TextGenerationInputs(batches=[contexts])
    inputs.input_tokens = 100
    inputs.batch_type = batch_type
    inputs.context_tokens = 500
    return inputs


def _mock_sch_config(dp: int = 1) -> MagicMock:
    config = MagicMock()
    config.max_batch_size = 32
    config.target_tokens_per_batch_ce = 4096
    config.max_batch_total_tokens = 0
    config.data_parallel_degree = dp
    return config


def test_batch_metrics_create_tg_with_spec_decode() -> None:
    """TG batch with spec decode uses output_tokens / time for generation throughput."""
    inputs = _mock_inputs(batch_size=4, batch_type=BatchType.TG)
    spec_metrics = _make_spec_metrics(
        num_speculative_tokens=3,
        accepted_per_position=[4, 3, 1],
        num_verifications=4,
    )
    # output_tokens = 8 + 4 = 12
    metrics = BatchMetrics.create(
        sch_config=_mock_sch_config(),
        inputs=inputs,
        kv_cache=None,
        batch_creation_time_s=0.001,
        batch_execution_time_s=0.1,
        num_pending_reqs=0,
        num_terminated_reqs=0,
        total_preemption_count=0,
        batch_spec_decode_metrics=spec_metrics,
    )
    assert metrics.generation_throughput == 12 / 0.1
    assert metrics.draft_tokens_generated == spec_metrics.draft_tokens_generated
    assert metrics.draft_tokens_accepted == spec_metrics.draft_tokens_accepted
    assert metrics.avg_acceptance_length == spec_metrics.avg_acceptance_length
    assert metrics.max_acceptance_length == 3
    assert (
        metrics.acceptance_rate_per_position
        == spec_metrics.acceptance_rate_per_position
    )


def test_batch_metrics_create_ce_with_spec_decode_uses_standard_formula() -> (
    None
):
    """CE batch uses standard throughput formula even when stale spec_metrics leak from a previous TG batch."""
    inputs = _mock_inputs(batch_size=2, batch_type=BatchType.CE)
    spec_metrics = _make_spec_metrics(
        num_speculative_tokens=3,
        accepted_per_position=[4, 3, 1],
        num_verifications=4,
    )
    metrics = BatchMetrics.create(
        sch_config=_mock_sch_config(),
        inputs=inputs,
        kv_cache=None,
        batch_creation_time_s=0.001,
        batch_execution_time_s=0.1,
        num_pending_reqs=0,
        num_terminated_reqs=0,
        total_preemption_count=0,
        batch_spec_decode_metrics=spec_metrics,
    )
    assert metrics.generation_throughput == 2 * 1 / 0.1

    # Acceptance metrics describe the decode/verify step, so a CE batch must
    # not report them even when stale spec_metrics leak from a previous TG
    # batch. The zeroed draft fields make every downstream consumer drop the
    # spec-decode info.
    assert metrics.draft_tokens_generated == 0
    assert metrics.draft_tokens_accepted == 0
    assert metrics.avg_acceptance_length == 0.0
    assert metrics.max_acceptance_length == 0
    assert metrics.acceptance_rate_per_position == []
    formatted = metrics.pretty_format()
    assert "Draft Tokens" not in formatted
    assert "Acceptance Len" not in formatted
    assert "draft_tokens_generated" not in metrics.to_log_extra()


def test_batch_metrics_create_no_spec_decode() -> None:
    """Without spec decode metrics, standard throughput formula and zero draft fields."""
    inputs = _mock_inputs(batch_size=4, batch_type=BatchType.TG)
    metrics = BatchMetrics.create(
        sch_config=_mock_sch_config(),
        inputs=inputs,
        kv_cache=None,
        batch_creation_time_s=0.001,
        batch_execution_time_s=0.1,
        num_pending_reqs=0,
        num_terminated_reqs=0,
        total_preemption_count=0,
    )
    assert metrics.generation_throughput == 4 * 1 / 0.1
    assert metrics.draft_tokens_generated == 0
    assert metrics.draft_tokens_accepted == 0
    assert metrics.avg_acceptance_length == 0.0
    assert metrics.max_acceptance_length == 0
    assert metrics.acceptance_rate_per_position == []


def test_batch_metrics_create_excludes_padding_contexts() -> None:
    inputs = _mock_inputs(batch_size=4, batch_type=BatchType.TG)
    inputs.flat_batch[-1]._is_padding_ctx = True
    spec_metrics = _make_spec_metrics(
        num_speculative_tokens=3,
        accepted_per_position=[3, 2, 1],
        num_verifications=3,
    )
    metrics = BatchMetrics.create(
        sch_config=_mock_sch_config(),
        inputs=inputs,
        kv_cache=None,
        batch_creation_time_s=0.001,
        batch_execution_time_s=0.1,
        num_pending_reqs=0,
        num_terminated_reqs=0,
        total_preemption_count=0,
        batch_spec_decode_metrics=spec_metrics,
    )
    assert metrics.batch_size == 3
    assert metrics.generation_throughput == (6 + 3) / 0.1


def test_publish_metrics_zero_verification_batch_skips_per_position_rates() -> (
    None
):
    """A TG batch whose verify step ran zero verifications must contribute
    nothing to the per-position rate histograms, matching the
    draft_tokens_generated > 0 gate on the acceptance-length histogram.

    Regression: the zero-verification batch published a full row of 0% rates,
    diluting every per-position average by the fraction of such batches.
    """
    zero = _make_spec_metrics(
        num_speculative_tokens=3,
        accepted_per_position=[0, 0, 0],
        num_verifications=0,
    )
    real = _make_spec_metrics(
        num_speculative_tokens=3,
        accepted_per_position=[4, 3, 1],
        num_verifications=4,
    )
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        for spec_metrics in (zero, real):
            BatchMetrics.create(
                sch_config=_mock_sch_config(),
                inputs=_mock_inputs(batch_size=4, batch_type=BatchType.TG),
                kv_cache=None,
                batch_creation_time_s=0.001,
                batch_execution_time_s=0.1,
                num_pending_reqs=0,
                num_terminated_reqs=0,
                total_preemption_count=0,
                batch_spec_decode_metrics=spec_metrics,
            ).publish_metrics()

    # Both spec-decode histograms must observe the same batch population:
    # only the batch that actually verified.
    acc_len_calls = (
        mock_metrics.spec_decode_avg_acceptance_length.call_args_list
    )
    per_pos_calls = (
        mock_metrics.spec_decode_acceptance_rate_per_position.call_args_list
    )
    assert len(acc_len_calls) == 1
    assert len(per_pos_calls) == 3

    # With matched populations the identity holds: the per-position rates sum
    # to the average acceptance length (accepted drafts per verification).
    rate_sum = sum(c.kwargs["acceptance_rate"] / 100 for c in per_pos_calls)
    assert rate_sum == acc_len_calls[0].args[0] == 2.0


# ---------------------------------------------------------------------------
# Overlap-scheduling execution-metric attribution tests
# ---------------------------------------------------------------------------


def test_publish_metrics_defers_execution_metrics() -> None:
    """With defer_execution_metrics=True (overlap scheduling), execution-time,
    throughput and termination metrics are suppressed — they are published
    separately from CompletedBatchStats — while the current batch's
    composition metrics are still emitted.
    """
    metrics = _make_metrics()
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        metrics.publish_metrics(defer_execution_metrics=True)
    mock_metrics.batch_execution_time.assert_not_called()
    mock_metrics.batch_prompt_throughput.assert_not_called()
    mock_metrics.batch_generation_throughput.assert_not_called()
    # Terminations are counted from the synchronized outputs, so they belong
    # to the completed batch, not the one enqueued this iteration.
    mock_metrics.batch_terminated_reqs.assert_not_called()
    # Current-batch composition metrics are unaffected.
    mock_metrics.batch_size.assert_called_once_with(1, batch_type="CE")
    mock_metrics.batch_input_tokens.assert_called_once_with(6, batch_type="CE")
    mock_metrics.batch_creation_time.assert_called_once_with(
        10000.0, batch_type="CE"
    )


def test_publish_metrics_reports_terminated_reqs_without_overlap() -> None:
    """Without overlap the batch completed synchronously, so its terminations
    are published under its own type."""
    metrics = _make_metrics()
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        metrics.publish_metrics()
    mock_metrics.batch_terminated_reqs.assert_called_once_with(
        4, batch_type="CE"
    )


def _make_completed_stats(**overrides: Any) -> CompletedBatchStats:
    base = dict[str, Any](
        batch_type=BatchType.CE,
        batch_size=2,
        num_input_tokens=4096,
        num_context_tokens=8192,
        execution_time_s=0.25,
    )
    base.update(overrides)
    return CompletedBatchStats(**base)


def test_publish_completed_batch_metrics_ce() -> None:
    """Execution metrics carry the completed batch's label and are computed
    from that batch's own token counts and duration."""
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        publish_completed_batch_metrics(_make_completed_stats(), 3)
    mock_metrics.batch_execution_time.assert_called_once_with(
        250.0, batch_type="CE"
    )
    mock_metrics.batch_prompt_throughput.assert_called_once_with(
        4096 / 0.25, batch_type="CE"
    )
    mock_metrics.batch_generation_throughput.assert_called_once_with(
        2 / 0.25, batch_type="CE"
    )
    mock_metrics.batch_terminated_reqs.assert_called_once_with(
        3, batch_type="CE"
    )
    mock_metrics.di_early_sync_time.assert_not_called()


def test_publish_completed_batch_metrics_tg_spec_decode() -> None:
    """TG batches with known output tokens (spec decode) use them for
    generation throughput, mirroring BatchMetrics.create."""
    stats = _make_completed_stats(batch_type=BatchType.TG, num_output_tokens=12)
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        publish_completed_batch_metrics(stats, 3)
    mock_metrics.batch_execution_time.assert_called_once_with(
        250.0, batch_type="TG"
    )
    mock_metrics.batch_generation_throughput.assert_called_once_with(
        12 / 0.25, batch_type="TG"
    )


def test_publish_completed_batch_metrics_ce_ignores_output_tokens() -> None:
    """CE batches use the standard batch_size formula even when stale spec
    metrics report output tokens, mirroring BatchMetrics.create."""
    stats = _make_completed_stats(num_output_tokens=12)
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        publish_completed_batch_metrics(stats, 3)
    mock_metrics.batch_generation_throughput.assert_called_once_with(
        2 / 0.25, batch_type="CE"
    )


def test_publish_completed_batch_metrics_zero_duration_skipped() -> None:
    """A zero-duration record cannot produce meaningful throughput; nothing
    is published."""
    stats = _make_completed_stats(execution_time_s=0.0)
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        publish_completed_batch_metrics(stats, 3)
    mock_metrics.batch_execution_time.assert_not_called()
    mock_metrics.batch_prompt_throughput.assert_not_called()
    mock_metrics.batch_generation_throughput.assert_not_called()
    mock_metrics.batch_terminated_reqs.assert_not_called()


def test_publish_completed_batch_metrics_emits_early_sync_time() -> None:
    """When the early-sync guard fired for the completed batch, its
    duration is published as a di_* metric."""
    stats = _make_completed_stats(early_sync_duration_s=0.012)
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        publish_completed_batch_metrics(stats, 3)
    mock_metrics.di_early_sync_time.assert_called_once_with(12.0)


def test_log_metrics_overlap_coalesces_completed_batch_into_transaction() -> (
    None
):
    """Overlap path: publish_metrics and publish_completed_batch_metrics run
    inside one transaction, so the deferred completed-batch execution metric
    coalesces into the per-iteration packet instead of emitting on its own.

    Regression: publish_completed_batch_metrics used to run outside any
    transaction, producing extra individual cross-process packets per
    iteration.
    """
    depth = 0
    emit_depths: list[int] = []

    @contextmanager
    def _fake_txn() -> Any:
        nonlocal depth
        depth += 1
        try:
            yield
        finally:
            depth -= 1

    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        mock_metrics.transaction.side_effect = _fake_txn
        # In the overlap path, batch_execution_time is emitted only by
        # publish_completed_batch_metrics; record the transaction depth when it
        # fires (must be > 0, i.e. inside the coalescing transaction).
        mock_metrics.batch_execution_time.side_effect = lambda *args, **kwargs: (
            emit_depths.append(depth)
        )
        SchedulerLogger().log_metrics(
            sch_config=_mock_sch_config(),
            inputs=_mock_inputs(batch_size=2, batch_type=BatchType.TG),
            kv_cache=None,
            batch_creation_time_s=0.001,
            batch_execution_time_s=0.1,
            num_pending_reqs=0,
            num_terminated_reqs=0,
            total_preemption_count=0,
            overlap_active=True,
            completed_batch_stats=_make_completed_stats(),
        )
    assert emit_depths == [1]


class _ExecMetricsRecorder:
    """Captures labeled execution/creation metric calls; no-ops the rest."""

    def __init__(self) -> None:
        self.execution_calls: list[tuple[float, str]] = []
        self.creation_calls: list[tuple[float, str]] = []
        self.prompt_throughput_calls: list[tuple[float, str]] = []
        self.terminated_calls: list[tuple[int, str]] = []

    def batch_execution_time(self, ms: float, batch_type: str) -> None:
        self.execution_calls.append((ms, batch_type))

    def batch_creation_time(self, ms: float, batch_type: str) -> None:
        self.creation_calls.append((ms, batch_type))

    def batch_prompt_throughput(self, tps: float, batch_type: str) -> None:
        self.prompt_throughput_calls.append((tps, batch_type))

    def batch_terminated_reqs(self, value: int, batch_type: str) -> None:
        self.terminated_calls.append((value, batch_type))

    def __getattr__(self, _name: str) -> Any:
        return MagicMock()


def test_scheduler_overlap_attributes_execution_metrics_to_completed_batch() -> (
    None
):
    """Under overlap scheduling, execution-time and throughput telemetry must
    be labeled with the batch that actually completed (one iteration lagged),
    not the batch enqueued in the current iteration.

    Regression test: previously a CE batch's execution time was published
    under the next iteration's batch type (usually TG), corrupting the CE/TG
    split of maxserve.batch_execution_time and both throughput histograms.
    """
    recorder = _ExecMetricsRecorder()
    with patch("max.serve.scheduler.utils.METRICS", recorder):
        scheduler, request_queue = create_paged_scheduler(
            max_seq_len=128,
            num_blocks=64,
            max_batch_size=4,
            page_size=8,
        )
        scheduler.pipeline = FakeOverlapPipeline(
            kv_manager=scheduler.batch_constructor.kv_cache,
            max_seq_len=128,
        )
        enqueue_request(request_queue, prompt_len=16, max_seq_len=128)

        # Iteration 1: the CE batch is enqueued but has not completed. No
        # execution metrics may be published; the current batch's composition
        # metrics still are.
        scheduler.run_iteration()
        assert recorder.execution_calls == []
        assert recorder.prompt_throughput_calls == []
        assert recorder.terminated_calls == []
        assert recorder.creation_calls
        assert recorder.creation_calls[-1][1] == "CE"

        # Iteration 2: the CE batch completes while a TG batch is enqueued.
        # Execution metrics must carry the CE label with the completed
        # batch's duration and token counts.
        scheduler.run_iteration()
        expected_ms = FakeOverlapPipeline.FAKE_EXECUTION_TIME_S * 1000
        assert recorder.execution_calls == [(expected_ms, "CE")]
        assert recorder.prompt_throughput_calls == [
            (16 / FakeOverlapPipeline.FAKE_EXECUTION_TIME_S, "CE")
        ]
        # Terminations are derived from the CE batch's outputs, so they carry
        # the CE label rather than the enqueued TG batch's.
        assert [bt for _, bt in recorder.terminated_calls] == ["CE"]
        assert recorder.creation_calls[-1][1] == "TG"


# ---------------------------------------------------------------------------
# DP active-token / context-token occupancy tests
# ---------------------------------------------------------------------------


def _mock_ctx(
    active_length: int, padding: bool = False, processed_length: int = 0
) -> MagicMock:
    ctx = MagicMock()
    ctx.tokens.active_length = active_length
    ctx.tokens.processed_length = processed_length
    # Explicit False: an auto-created Mock attribute would be truthy and the
    # context would be skipped as a padding dummy.
    ctx._is_padding_ctx = padding
    return ctx


def _mock_dp_inputs(
    rank_batches: list[list[MagicMock]],
    batch_type: BatchType = BatchType.CE,
) -> TextGenerationInputs[Any]:
    # Real inputs rather than a mock: the per-replica token sums the metrics
    # read are frozen in __post_init__, so that production path must run.
    inputs: TextGenerationInputs[Any] = TextGenerationInputs(
        batches=rank_batches
    )
    # __post_init__ infers batch_type from the contexts' generated_length,
    # which mock contexts don't model; override it like the DP padder does.
    inputs.batch_type = batch_type
    return inputs


def _create_dp_metrics(
    inputs: TextGenerationInputs[Any], dp: int
) -> BatchMetrics:
    return BatchMetrics.create(
        sch_config=_mock_sch_config(dp=dp),
        inputs=inputs,
        kv_cache=None,
        batch_creation_time_s=0.001,
        batch_execution_time_s=0.1,
        num_pending_reqs=0,
        num_terminated_reqs=0,
        total_preemption_count=0,
    )


def test_dp_occupancy_imbalanced_ce() -> None:
    """The motivating case: one rank prefills 8k tokens while the other has
    a handful of decodes -> ~50% occupancy at DP2."""
    inputs = _mock_dp_inputs([[_mock_ctx(8000)], [_mock_ctx(8)]])
    metrics = _create_dp_metrics(inputs, dp=2)
    assert metrics.dp_active_token_occupancy_pct == 100.0 * 8008 / 16000
    assert metrics.dp_active_tokens == 8008
    assert metrics.dp_step_capacity_tokens == 16000


def test_dp_occupancy_balanced() -> None:
    inputs = _mock_dp_inputs(
        [[_mock_ctx(4000)], [_mock_ctx(2000), _mock_ctx(2000)]]
    )
    metrics = _create_dp_metrics(inputs, dp=2)
    assert metrics.dp_active_token_occupancy_pct == 100.0


def test_dp_occupancy_skipped_at_dp1() -> None:
    inputs = _mock_dp_inputs([[_mock_ctx(8000)]])
    metrics = _create_dp_metrics(inputs, dp=1)
    assert metrics.dp_active_token_occupancy_pct is None
    assert metrics.dp_active_tokens == 0
    assert metrics.dp_step_capacity_tokens == 0


def test_dp_occupancy_skipped_on_empty_batch() -> None:
    inputs = _mock_dp_inputs([[], []])
    metrics = _create_dp_metrics(inputs, dp=2)
    assert metrics.dp_active_token_occupancy_pct is None
    assert metrics.dp_active_tokens == 0
    assert metrics.dp_step_capacity_tokens == 0


def test_dp_occupancy_excludes_padding_dummies() -> None:
    """Padding dummies keep device-graph shapes valid but are not scheduler
    placement decisions; a padded-out rank counts as zero load."""
    inputs = _mock_dp_inputs([[_mock_ctx(8000)], [_mock_ctx(1, padding=True)]])
    metrics = _create_dp_metrics(inputs, dp=2)
    assert metrics.dp_active_token_occupancy_pct == 50.0
    assert metrics.dp_active_tokens == 8000
    assert metrics.dp_step_capacity_tokens == 16000


def test_dp_occupancy_missing_replicas_count_as_zero() -> None:
    inputs = _mock_dp_inputs([[_mock_ctx(8000)]])
    metrics = _create_dp_metrics(inputs, dp=2)
    assert metrics.dp_active_token_occupancy_pct == 50.0


def test_publish_metrics_dp_occupancy() -> None:
    metrics = _make_metrics(dp_active_token_occupancy_pct=50.0)
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        metrics.publish_metrics()
    mock_metrics.dp_active_token_occupancy.assert_called_once_with(
        50.0, batch_type="CE"
    )


def test_publish_metrics_dp_occupancy_skipped_when_unset() -> None:
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        _make_metrics().publish_metrics()
    mock_metrics.dp_active_token_occupancy.assert_not_called()


def test_publish_metrics_dp_token_counters() -> None:
    metrics = _make_metrics(
        dp_active_tokens=8008, dp_step_capacity_tokens=16000
    )
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        metrics.publish_metrics()
    mock_metrics.dp_active_tokens.assert_called_once_with(8008, batch_type="CE")
    mock_metrics.dp_step_capacity_tokens.assert_called_once_with(
        16000, batch_type="CE"
    )


def test_publish_metrics_dp_token_counters_skipped_when_zero() -> None:
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        _make_metrics().publish_metrics()
    mock_metrics.dp_active_tokens.assert_not_called()
    mock_metrics.dp_step_capacity_tokens.assert_not_called()


def test_dp_occupancy_in_log_line_and_extra() -> None:
    # A CE log line shows the active-token occupancy; the context value is
    # still carried in the structured extra.
    metrics = _make_metrics(
        dp_active_token_occupancy_pct=50.0,
        dp_context_token_occupancy_pct=75.0,
    )
    assert "DP Occupancy: 50.0% | " in metrics.pretty_format()
    extra = metrics.to_log_extra()
    assert extra["dp_active_token_occupancy_pct"] == 50.0
    assert extra["dp_context_token_occupancy_pct"] == 75.0

    # Absent at DP1 (field defaults to None).
    plain = _make_metrics()
    assert "DP Occupancy" not in plain.pretty_format()
    assert "dp_active_token_occupancy_pct" not in plain.to_log_extra()


def test_dp_context_occupancy_imbalanced_tg() -> None:
    """The motivating case: decode ranks with even request counts but wildly
    uneven context lengths -> uneven KV/attention load. Active occupancy is
    100% (one token per request) while context occupancy shows the skew."""
    inputs = _mock_dp_inputs(
        [
            [_mock_ctx(1, processed_length=8000)],
            [_mock_ctx(1, processed_length=8)],
        ],
        batch_type=BatchType.TG,
    )
    metrics = _create_dp_metrics(inputs, dp=2)
    assert metrics.dp_active_token_occupancy_pct == 100.0
    assert metrics.dp_context_token_occupancy_pct == 100.0 * 8008 / 16000


def test_dp_context_occupancy_balanced() -> None:
    inputs = _mock_dp_inputs(
        [
            [_mock_ctx(1, processed_length=4000)],
            [
                _mock_ctx(1, processed_length=2000),
                _mock_ctx(1, processed_length=2000),
            ],
        ],
        batch_type=BatchType.TG,
    )
    metrics = _create_dp_metrics(inputs, dp=2)
    assert metrics.dp_context_token_occupancy_pct == 100.0


def test_dp_context_occupancy_skipped_at_dp1() -> None:
    inputs = _mock_dp_inputs([[_mock_ctx(1, processed_length=8000)]])
    metrics = _create_dp_metrics(inputs, dp=1)
    assert metrics.dp_context_token_occupancy_pct is None


def test_dp_context_occupancy_skipped_on_fresh_prefill() -> None:
    """A fresh prefill batch has no processed tokens on any rank, so there
    is no context load to balance; the active-token metric still reports."""
    inputs = _mock_dp_inputs([[_mock_ctx(8000)], [_mock_ctx(8)]])
    metrics = _create_dp_metrics(inputs, dp=2)
    assert metrics.dp_active_token_occupancy_pct is not None
    assert metrics.dp_context_token_occupancy_pct is None


def test_dp_context_occupancy_excludes_padding_dummies() -> None:
    inputs = _mock_dp_inputs(
        [
            [_mock_ctx(1, processed_length=8000)],
            [_mock_ctx(1, padding=True, processed_length=8000)],
        ],
        batch_type=BatchType.TG,
    )
    metrics = _create_dp_metrics(inputs, dp=2)
    assert metrics.dp_context_token_occupancy_pct == 50.0


def test_dp_context_occupancy_missing_replicas_count_as_zero() -> None:
    inputs = _mock_dp_inputs(
        [[_mock_ctx(1, processed_length=8000)]], batch_type=BatchType.TG
    )
    metrics = _create_dp_metrics(inputs, dp=2)
    assert metrics.dp_context_token_occupancy_pct == 50.0


def test_publish_metrics_dp_context_occupancy() -> None:
    metrics = _make_metrics(dp_context_token_occupancy_pct=50.0)
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        metrics.publish_metrics()
    mock_metrics.dp_context_token_occupancy.assert_called_once_with(
        50.0, batch_type="CE"
    )


def test_publish_metrics_dp_context_occupancy_skipped_when_unset() -> None:
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        _make_metrics().publish_metrics()
    mock_metrics.dp_context_token_occupancy.assert_not_called()


def test_dp_context_occupancy_in_log_line_and_extra() -> None:
    # A TG log line shows the context-token occupancy under the same
    # "DP Occupancy" label; the active value stays in the structured extra.
    metrics = _make_metrics(
        batch_type=BatchType.TG,
        dp_active_token_occupancy_pct=100.0,
        dp_context_token_occupancy_pct=50.0,
    )
    assert "DP Occupancy: 50.0% | " in metrics.pretty_format()
    extra = metrics.to_log_extra()
    assert extra["dp_active_token_occupancy_pct"] == 100.0
    assert extra["dp_context_token_occupancy_pct"] == 50.0

    # Absent at DP1 (field defaults to None).
    plain = _make_metrics(batch_type=BatchType.TG)
    assert "DP Occupancy" not in plain.pretty_format()
    assert "dp_context_token_occupancy_pct" not in plain.to_log_extra()


def test_cache_hit_tiers_sum_to_the_untagged_total() -> None:
    """CLIN-1785: ``maxserve.cache.hits`` carries a ``tier`` attribute.

    The whole design rests on one invariant: the per-tier points SUM to the
    counter's untagged total. That is what lets the dashboard replace a
    cross-service subtraction (which went negative) with a direct grouped query
    while every ungrouped query keeps reading the same number.

    MAX resolves two tiers: ``g0`` for the device prefix cache and ``external``
    for the KV connector, whose host/disk split does not cross the connector
    boundary. Reporting those as ``g1`` would charge tokens to a tier that may
    not have served them.
    """
    metrics = _make_metrics(
        cache_hit_tokens=100,
        cache_hit_external_tokens=30,
        cache_miss_tokens=20,
    )
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        metrics.publish_metrics()

    tiered = {
        call.kwargs["tier"]: call.args[0]
        for call in mock_metrics.cache_hits.call_args_list
    }
    assert tiered == {"g0": 70, "external": 30}
    # THE invariant: the tiers partition the hit total exactly.
    assert sum(tiered.values()) == metrics.cache_hit_tokens
    # Every point carries a tier; an untagged one would double any ungrouped
    # query's total.
    assert all(
        "tier" in call.kwargs for call in mock_metrics.cache_hits.call_args_list
    )
    # Misses stay untagged: a missed token was served by no tier.
    mock_metrics.cache_misses.assert_called_once_with(20)


def test_cache_hits_without_a_connector_emit_device_tier_only() -> None:
    """No connector means no external tokens, and no idle ``external`` series.

    A zero-token tier is skipped rather than stamped, so a deployment with no
    dKV tier never mints a flat-zero series the dashboard would have to filter.
    """
    metrics = _make_metrics(
        cache_hit_tokens=64,
        cache_hit_external_tokens=0,
        cache_miss_tokens=0,
    )
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        metrics.publish_metrics()

    mock_metrics.cache_hits.assert_called_once_with(64, tier="g0")
    mock_metrics.dkv_read_blocks.assert_not_called()


def test_cold_cache_still_stamps_the_device_tier() -> None:
    """A CE batch that admitted requests but hit nothing still emits g0=0.

    Before the tier label this recorded ``cache_hits(0)``, which creates and
    keeps the series alive. Skipping every empty tier would leave
    ``maxserve_cache_hits_tokens_total`` absent on a cold server, and
    ``rate()``/``sum()`` over an absent series returns no data rather than
    zero, which reads as a broken panel instead of a cold cache.
    """
    metrics = _make_metrics(
        cache_hit_tokens=0,
        cache_hit_external_tokens=0,
        cache_miss_tokens=512,
    )
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        metrics.publish_metrics()

    mock_metrics.cache_hits.assert_called_once_with(0, tier="g0")
    mock_metrics.cache_misses.assert_called_once_with(512)


def test_dkv_read_blocks_publishes_outside_the_cache_hit_clause() -> None:
    """``dkv_read_blocks`` must not be gated on new admissions.

    It is a per-window delta that ``reset_metrics`` clears after every batch,
    so a window published under any other batch type would lose its count for
    good. That matters because a load posted on a CE iteration can have its
    blocks accounted on a later ``wait_for_loads``, which also runs on TG
    iterations: under the old placement those blocks were dropped, letting the
    counter read *below* ``cache.hits{tier=external}`` despite being
    documented as an upper bound on it.
    """
    metrics = _make_metrics(
        batch_type=BatchType.TG,
        num_new_admissions=0,
        dkv_read_blocks=7,
    )
    with patch("max.serve.scheduler.utils.METRICS") as mock_metrics:
        metrics.publish_metrics()

    mock_metrics.dkv_read_blocks.assert_called_once_with(7)
    # The cache-hit clause is genuinely skipped on a TG batch, so this proves
    # the counter is published from outside it rather than alongside it.
    mock_metrics.cache_hits.assert_not_called()


def test_create_sums_the_external_share_across_admissions() -> None:
    """CLIN-1785: ``BatchMetrics.create`` carries the per-tier split through.

    The other ``create`` tests build contexts with bare ``MagicMock``s, whose
    auto-created ``_cache_metrics_emitted`` attribute is truthy, so the
    admission loop skips every one and its body never runs. That leaves the
    accumulation of ``cached_prefix_external_length`` unexercised: drop it and
    ``g0`` silently absorbs the connector's tokens, which is the pre-change
    reading this metric exists to replace.

    Two admitted requests, each with half its cached prefix served externally,
    so the batch total must be the sum and the device remainder the complement.
    """
    contexts = []
    for _ in range(2):
        ctx = MagicMock()
        ctx._is_padding_ctx = False
        # Explicit False: the auto-created attribute is truthy and would make
        # the admission loop skip this context entirely.
        ctx._cache_metrics_emitted = False
        ctx.tokens.active_length = 8
        ctx.tokens.processed_length = 0
        ctx.tokens.generated_length = 0
        ctx.tokens.prompt_length = 100
        ctx.cached_prefix_length = 40
        ctx.cached_prefix_external_length = 15
        contexts.append(ctx)

    inputs: TextGenerationInputs[Any] = TextGenerationInputs(batches=[contexts])
    object.__setattr__(inputs, "batch_type", BatchType.CE)

    metrics = BatchMetrics.create(
        sch_config=_mock_sch_config(),
        inputs=inputs,
        kv_cache=None,
        batch_creation_time_s=0.001,
        batch_execution_time_s=0.1,
        num_pending_reqs=0,
        num_terminated_reqs=0,
        total_preemption_count=0,
    )

    assert metrics.num_new_admissions == 2
    assert metrics.cache_hit_tokens == 80
    assert metrics.cache_hit_external_tokens == 30
    # The device share is the complement, and the two partition the hit total.
    assert metrics.cache_hit_tokens - metrics.cache_hit_external_tokens == 50
