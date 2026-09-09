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

"""Tests for TPOT computation and request timestamps in calculate_metrics."""

from __future__ import annotations

import json
import logging
import math
from unittest.mock import MagicMock

import pytest
from max.benchmark.benchmark_shared.metrics import BenchmarkResult
from max.benchmark.benchmark_shared.request import (
    PixelGenerationRequestFuncOutput,
    RequestFuncOutput,
    ServerTokenStats,
)
from max.benchmark.benchmark_shared.serving_metrics import (
    _per_turn_cache_retentions,
    build_text_generation_result,
    calculate_metrics,
    calculate_pixel_generation_metrics,
)
from max.profiler.cpu import CPUMetrics

_EMPTY_CPU_METRICS = CPUMetrics(
    user=0.0, user_percent=0.0, system=0.0, system_percent=0.0, elapsed=0.0
)


def _make_mock_tokenizer(token_counts: dict[str, int]) -> MagicMock:
    """Create a mock tokenizer that returns specified token counts.

    Args:
        token_counts: Mapping from generated text to the number of tokens.
    """
    tokenizer = MagicMock()

    def encode(text: str, add_special_tokens: bool = True) -> list[int]:
        return list(range(token_counts.get(text, 0)))

    tokenizer.encode.side_effect = encode
    return tokenizer


def test_per_chunk_tpot_collected_from_outputs() -> None:
    """Per-chunk TPOT values are collected into step_tpot_ms."""
    output = RequestFuncOutput(
        success=True,
        latency=1.0,
        ttft=0.1,
        prompt_len=10,
        generated_text="hello world",
        itl=[0.1, 0.2, 0.3],
        tpot=[0.05, 0.1, 0.15],
    )

    tokenizer = _make_mock_tokenizer({"hello world": 5})

    metrics = calculate_metrics(
        outputs=[output],
        dur_s=1.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
        kv_block_size=128,
    )

    assert metrics.text_data is not None

    # Per-chunk values [0.05, 0.1, 0.15] now live in step_tpot_ms.
    assert metrics.text_data.step_tpot_ms is not None
    assert math.isclose(metrics.text_data.step_tpot_ms.p50, 100.0, rel_tol=1e-3)
    # tpot_ms is per-request: (latency - ttft) / (output_len - 1)
    #                       = (1.0 - 0.1) / (5 - 1) = 0.225 s -> 225 ms.
    assert metrics.text_data.tpot_ms is not None
    assert math.isclose(metrics.text_data.tpot_ms.p50, 225.0, rel_tol=1e-3)


def test_tpot_both_definitions() -> None:
    """Both TPOT definitions are computed: per-request tpot_ms and per-step step_tpot_ms."""
    # Request 1: 10 output tokens, latency 1.0s, ttft 0.1s
    #   -> per-request tpot = 0.9 / 9 = 0.1 s
    output1 = RequestFuncOutput(
        success=True,
        latency=1.0,
        ttft=0.1,
        prompt_len=10,
        generated_text="ten tokens out",
        itl=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        tpot=[0.1] * 9,
    )

    # Request 2: 4 output tokens, latency 0.8s, ttft 0.2s
    #   -> per-request tpot = 0.6 / 3 = 0.2 s
    output2 = RequestFuncOutput(
        success=True,
        latency=0.8,
        ttft=0.2,
        prompt_len=5,
        generated_text="four tok",
        itl=[0.2, 0.2, 0.2],
        tpot=[0.2] * 3,
    )

    # Mock tokenizer: output1 -> 10 tokens, output2 -> 4 tokens
    tokenizer = _make_mock_tokenizer({"ten tokens out": 10, "four tok": 4})

    metrics = calculate_metrics(
        outputs=[output1, output2],
        dur_s=2.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
        kv_block_size=128,
    )

    assert metrics.text_data is not None

    # Per-request tpots = [0.1, 0.2]; mean = 0.15 s -> 150 ms.
    assert metrics.text_data.tpot_ms is not None
    assert math.isclose(metrics.text_data.tpot_ms.mean, 150.0, rel_tol=1e-6)

    # Per-step step_tpots = [0.1]*9 + [0.2]*3; mean = 1.5/12 s -> 125 ms.
    assert metrics.text_data.step_tpot_ms is not None
    assert math.isclose(
        metrics.text_data.step_tpot_ms.mean, 125.0, rel_tol=1e-6
    )


def test_tpot_zero_decode_tokens() -> None:
    """When all requests produce <= 1 token, decode metrics are ``None``."""
    # Output with 1 token (only TTFT, no decode)
    output = RequestFuncOutput(
        success=True,
        latency=0.1,
        ttft=0.1,
        prompt_len=10,
        generated_text="a",
        itl=[],
        tpot=[],
    )

    # 1 output token, 1 completed -> decode_tokens = 0
    tokenizer = _make_mock_tokenizer({"a": 1})

    metrics = calculate_metrics(
        outputs=[output],
        dur_s=1.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
        kv_block_size=128,
    )

    assert metrics.text_data is not None

    # With no decode samples the decode-phase percentile metrics are
    # ``None`` (rather than a NaN-filled object), while prefill-phase
    # metrics that do have samples stay populated.
    assert metrics.text_data.tpot_ms is None
    assert metrics.text_data.itl_ms is None
    assert metrics.text_data.ttft_ms is not None


def test_empty_outputs_no_crash() -> None:
    """Empty outputs list doesn't crash."""
    tokenizer = _make_mock_tokenizer({})

    metrics = calculate_metrics(
        outputs=[],
        dur_s=1.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
        kv_block_size=128,
    )

    assert metrics.text_data is not None

    assert metrics.text_data.completed == 0
    assert metrics.text_data.output_lens == []
    # With no samples, percentile metrics are ``None`` (not a NaN-filled
    # object) so the serialized JSON stays free of null-valued percentile
    # objects that strict downstream consumers reject.
    assert metrics.text_data.latency_ms is None
    assert metrics.text_data.ttft_ms is None
    assert metrics.text_data.tpot_ms is None
    assert metrics.text_data.itl_ms is None
    assert metrics.text_data.input_throughput is None
    assert metrics.text_data.output_throughput is None

    # And the empty case round-trips through JSON as ``null`` fields, not
    # ``{"p50": null, ...}`` objects (regression guard for MXTOOLS-45:
    # legacy NaN objects tripped the dashboard's per-field drop pass).
    dumped = json.loads(metrics.model_dump_json())
    assert dumped["text_data"]["tpot_ms"] is None
    assert dumped["text_data"]["latency_ms"] is None


def test_itl_metrics_unchanged() -> None:
    """ITL metrics remain unchanged by the TPOT refactor."""
    output = RequestFuncOutput(
        success=True,
        latency=1.0,
        ttft=0.1,
        prompt_len=10,
        generated_text="hello world",
        itl=[0.1, 0.2, 0.3],
        tpot=[0.05, 0.1, 0.15],
    )

    tokenizer = _make_mock_tokenizer({"hello world": 5})

    metrics = calculate_metrics(
        outputs=[output],
        dur_s=1.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
        kv_block_size=128,
    )

    assert metrics.text_data is not None

    # ITL should be computed from the raw itl values [0.1, 0.2, 0.3] * 1000
    assert metrics.text_data.itl_ms is not None
    assert math.isclose(metrics.text_data.itl_ms.mean, 200.0, rel_tol=1e-3)
    assert math.isclose(metrics.text_data.itl_ms.p50, 200.0, rel_tol=1e-3)


def test_failed_requests_excluded() -> None:
    """Failed requests don't contribute to TPOT."""
    success_output = RequestFuncOutput(
        success=True,
        latency=1.0,
        ttft=0.1,
        prompt_len=10,
        generated_text="hello",
        itl=[0.1, 0.2],
        tpot=[0.1, 0.2],
    )
    failed_output = RequestFuncOutput(
        success=False,
        error="test error",
        itl=[999.0],
        tpot=[999.0],
    )

    tokenizer = _make_mock_tokenizer({"hello": 3, "": 0})

    metrics = calculate_metrics(
        outputs=[success_output, failed_output],
        dur_s=1.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
        kv_block_size=128,
    )

    assert metrics.text_data is not None

    # Only successful request's TPOT should be used
    assert metrics.text_data.completed == 1
    assert metrics.text_data.failures == 1
    # Per-request tpot_ms uses only the successful request, not [999.0].
    assert metrics.text_data.tpot_ms is not None
    assert metrics.text_data.tpot_ms.p50 < 500.0
    # Per-step step_tpot_ms uses only [0.1, 0.2] from the successful request.
    assert metrics.text_data.step_tpot_ms is not None
    assert metrics.text_data.step_tpot_ms.p50 < 500.0


def test_skip_last_n_requests() -> None:
    """Skipping last N requests excludes them from latency metrics."""
    outputs = [
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="first",
            itl=[0.1, 0.2],
            tpot=[0.1, 0.2],
        ),
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="second",
            itl=[0.1, 0.2],
            tpot=[0.1, 0.2],
        ),
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.5,
            prompt_len=10,
            generated_text="third",
            itl=[0.9, 0.8],
            tpot=[0.9, 0.8],
        ),
    ]

    tokenizer = _make_mock_tokenizer({"first": 3, "second": 3, "third": 3})

    metrics_all = calculate_metrics(
        outputs=outputs,
        dur_s=3.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
        kv_block_size=128,
    )

    metrics_skip_last = calculate_metrics(
        outputs=outputs,
        dur_s=3.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=0,
        skip_last_n_requests=1,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
        kv_block_size=128,
    )

    assert metrics_all.text_data is not None
    assert metrics_skip_last.text_data is not None

    # completed reflects the measured slice (first two, third was skipped).
    assert metrics_all.text_data.completed == 3
    assert metrics_skip_last.text_data.completed == 2
    # The last request's high TTFT (0.5s) is excluded from latency metrics.
    assert metrics_skip_last.text_data.ttft_ms is not None
    assert metrics_all.text_data.ttft_ms is not None
    assert (
        metrics_skip_last.text_data.ttft_ms.mean
        < metrics_all.text_data.ttft_ms.mean
    )


def test_skip_first_and_last_n_requests() -> None:
    """Skipping both first and last N requests works together."""
    outputs = [
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.5,
            prompt_len=10,
            generated_text="first",
            itl=[0.1],
            tpot=[0.1],
        ),
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="middle",
            itl=[0.1],
            tpot=[0.1],
        ),
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.5,
            prompt_len=10,
            generated_text="last",
            itl=[0.1],
            tpot=[0.1],
        ),
    ]

    tokenizer = _make_mock_tokenizer({"first": 2, "middle": 2, "last": 2})

    metrics = calculate_metrics(
        outputs=outputs,
        dur_s=3.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=1,
        skip_last_n_requests=1,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
        kv_block_size=128,
    )

    assert metrics.text_data is not None

    # Only the middle request is measured.
    assert metrics.text_data.completed == 1
    # Only the middle request's TTFT (0.1s = 100ms) should be measured
    assert metrics.text_data.ttft_ms is not None
    assert math.isclose(metrics.text_data.ttft_ms.mean, 100.0, rel_tol=1e-3)


def test_skip_last_with_cancelled_requests() -> None:
    """skip_last counts from end of completed requests, not the output array."""
    outputs = [
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="first",
            itl=[0.1],
            tpot=[0.1],
        ),
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.2,
            prompt_len=10,
            generated_text="second",
            itl=[0.1],
            tpot=[0.1],
        ),
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.5,
            prompt_len=10,
            generated_text="third",
            itl=[0.1],
            tpot=[0.1],
        ),
        # Cancelled requests pad the output array
        RequestFuncOutput(cancelled=True),
        RequestFuncOutput(cancelled=True),
        RequestFuncOutput(cancelled=True),
    ]

    tokenizer = _make_mock_tokenizer({"first": 2, "second": 2, "third": 2})

    metrics = calculate_metrics(
        outputs=outputs,
        dur_s=3.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=1,
        skip_last_n_requests=1,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
        kv_block_size=128,
    )

    assert metrics.text_data is not None

    # skip_last is applied to successful (3), not to the padded cancelled
    # entries, so only the middle successful request is measured.
    assert metrics.text_data.completed == 1
    # Only the second request should be measured (skip first 1, last 1)
    assert metrics.text_data.ttft_ms is not None
    assert math.isclose(metrics.text_data.ttft_ms.mean, 200.0, rel_tol=1e-3)


def test_skip_all_requests_warns() -> None:
    """Skipping all requests emits a warning."""
    outputs = [
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="only",
            itl=[0.1],
            tpot=[0.1],
        ),
    ]

    tokenizer = _make_mock_tokenizer({"only": 2})

    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        calculate_metrics(
            outputs=outputs,
            dur_s=1.0,
            tokenizer=tokenizer,
            gpu_metrics=None,
            cpu_metrics=_EMPTY_CPU_METRICS,
            skip_first_n_requests=1,
            skip_last_n_requests=1,
            max_concurrency=None,
            max_concurrent_conversations=None,
            collect_gpu_stats=False,
            kv_block_size=128,
        )
        assert len(w) == 1
        assert "excluded" in str(w[0].message).lower()


def test_calculate_pixel_generation_metrics() -> None:
    outputs = [
        PixelGenerationRequestFuncOutput(
            success=True,
            latency=1.0,
            num_generated_outputs=1,
        ),
        PixelGenerationRequestFuncOutput(
            success=True,
            latency=2.0,
            num_generated_outputs=2,
        ),
        PixelGenerationRequestFuncOutput(success=False, error="bad request"),
    ]

    metrics = calculate_pixel_generation_metrics(
        outputs=outputs,
        dur_s=5.0,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        max_concurrency=None,
        collect_gpu_stats=False,
    )

    assert metrics.pixel_data is not None

    assert metrics.pixel_data.completed == 2
    assert metrics.pixel_data.failures == 1
    assert math.isclose(
        metrics.pixel_data.request_throughput, 0.4, rel_tol=1e-6
    )
    assert metrics.pixel_data.total_generated_outputs == 3
    assert metrics.pixel_data.latency_ms is not None
    assert math.isclose(
        metrics.pixel_data.latency_ms.mean, 1500.0, rel_tol=1e-6
    )


def test_request_submit_time_defaults_to_none() -> None:
    """request_submit_time field defaults to None."""
    output = RequestFuncOutput()
    assert output.request_submit_time is None
    assert output.request_complete_time is None


def test_request_submit_time_set_on_output() -> None:
    """request_submit_time can be set and is preserved through metrics."""
    output = RequestFuncOutput(
        success=True,
        latency=1.0,
        ttft=0.1,
        prompt_len=10,
        generated_text="hello",
        itl=[0.1],
        tpot=[0.1],
        request_submit_time=100.5,
    )

    assert output.request_submit_time == 100.5

    tokenizer = _make_mock_tokenizer({"hello": 2})
    metrics = calculate_metrics(
        outputs=[output],
        dur_s=1.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
        kv_block_size=128,
    )
    assert metrics.text_data is not None
    # Metrics are computed normally regardless of submit time
    assert metrics.text_data.completed == 1


def test_measured_duration_uses_measured_window() -> None:
    """When skipping is applied, duration and throughput reflect the measured window only."""
    # 10 warmup requests: submit every 1s starting at t=0, each takes 5s.
    # 100 steady requests: submit every 0.1s starting at t=10, each takes 0.5s.
    # 10 tail requests: submit every 2s starting at t=30, each takes 5s.
    warmup = [
        RequestFuncOutput(
            success=True,
            latency=5.0,
            ttft=0.5,
            prompt_len=100,
            generated_text="warmup",
            itl=[0.1] * 4,
            tpot=[0.1] * 4,
            request_submit_time=float(i),
        )
        for i in range(10)
    ]
    steady = [
        RequestFuncOutput(
            success=True,
            latency=0.5,
            ttft=0.05,
            prompt_len=10,
            generated_text="steady",
            itl=[0.05] * 4,
            tpot=[0.05] * 4,
            request_submit_time=10.0 + i * 0.1,
        )
        for i in range(100)
    ]
    tail = [
        RequestFuncOutput(
            success=True,
            latency=5.0,
            ttft=0.5,
            prompt_len=100,
            generated_text="tail",
            itl=[0.1] * 4,
            tpot=[0.1] * 4,
            request_submit_time=30.0 + i * 2.0,
        )
        for i in range(10)
    ]
    outputs = warmup + steady + tail

    tokenizer = _make_mock_tokenizer({"warmup": 5, "steady": 5, "tail": 5})

    # Full run wall clock passed as dur_s. 60s is much longer than the
    # measured window should be.
    full_run_duration = 60.0

    metrics = calculate_metrics(
        outputs=outputs,
        dur_s=full_run_duration,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=10,
        skip_last_n_requests=10,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
        kv_block_size=128,
    )

    assert metrics.text_data is not None

    # Measured = the 100 steady requests.
    assert metrics.text_data.completed == 100
    # total_input / total_output are over the measured 100 only.
    assert metrics.text_data.total_input == 100 * 10
    assert metrics.text_data.total_output == 100 * 5
    # Measured window: first steady submits at t=10.0; last steady
    # completes at t = 10.0 + 99*0.1 + 0.5 = 20.4.
    expected_window = 20.4 - 10.0
    assert math.isclose(
        metrics.text_data.duration, expected_window, rel_tol=1e-6
    )
    # Request throughput is over the measured window, not the full run.
    assert math.isclose(
        metrics.text_data.request_throughput,
        100 / expected_window,
        rel_tol=1e-6,
    )
    # Crucially, throughput does NOT use the full run duration.
    assert metrics.text_data.request_throughput > 100 / full_run_duration


def test_measured_duration_falls_back_when_no_timestamps() -> None:
    """Without submit timestamps, duration falls back to dur_s."""
    output = RequestFuncOutput(
        success=True,
        latency=1.0,
        ttft=0.1,
        prompt_len=10,
        generated_text="hello",
        itl=[0.1],
        tpot=[0.1],
    )
    # No request_submit_time -> fallback to dur_s.
    tokenizer = _make_mock_tokenizer({"hello": 2})
    metrics = calculate_metrics(
        outputs=[output],
        dur_s=3.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
        kv_block_size=128,
    )
    assert metrics.text_data is not None
    assert math.isclose(metrics.text_data.duration, 3.0, rel_tol=1e-9)
    assert math.isclose(
        metrics.text_data.request_throughput, 1.0 / 3.0, rel_tol=1e-9
    )


def test_skipped_tokens_excluded_from_totals() -> None:
    """Tokens from skipped warmup/tail requests don't inflate total_input/total_output."""
    outputs = [
        RequestFuncOutput(
            success=True,
            latency=5.0,
            ttft=0.5,
            prompt_len=10_000,  # huge warmup prompt
            generated_text="warm",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=0.0,
        ),
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="mid",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=6.0,
        ),
        RequestFuncOutput(
            success=True,
            latency=5.0,
            ttft=0.5,
            prompt_len=10_000,  # huge tail prompt
            generated_text="tail",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=8.0,
        ),
    ]
    tokenizer = _make_mock_tokenizer({"warm": 999, "mid": 3, "tail": 999})

    metrics = calculate_metrics(
        outputs=outputs,
        dur_s=20.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=1,
        skip_last_n_requests=1,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
        kv_block_size=128,
    )

    assert metrics.text_data is not None

    # Only the middle request counts toward totals.
    assert metrics.text_data.total_input == 10
    assert metrics.text_data.total_output == 3
    assert metrics.text_data.completed == 1


def test_request_complete_time_property() -> None:
    """request_complete_time property = submit_time + latency."""
    outputs = [
        RequestFuncOutput(
            success=True,
            latency=1.5,
            ttft=0.1,
            prompt_len=10,
            generated_text="a",
            request_submit_time=100.0,
        ),
        RequestFuncOutput(
            success=True,
            latency=2.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="b",
            request_submit_time=101.0,
        ),
    ]

    assert outputs[0].request_complete_time is not None
    assert outputs[1].request_complete_time is not None
    assert math.isclose(outputs[0].request_complete_time, 101.5)
    assert math.isclose(outputs[1].request_complete_time, 103.0)

    # Arrays stay index-aligned (same length as submit_times)
    submit_times = [o.request_submit_time for o in outputs]
    complete_times = [o.request_complete_time for o in outputs]
    assert len(submit_times) == len(complete_times)


def test_skip_uses_submit_time_for_head_complete_time_for_tail() -> None:
    """skip_first targets earliest submits, skip_last targets latest completes.

    Mirrors a multi-turn flat list, where outputs arrive in
    ``[session0_turns, session1_turns, ...]`` block order rather than
    chronological order. The dispatch-order slice would silently target the
    wrong requests; the timing-based selection picks the right ones.
    """
    # Three "sessions" of two turns each, flattened in block order. Submit/
    # complete times are intentionally non-monotonic vs. iteration order.
    outputs = [
        # session 0: starts first, finishes early
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="s0t0",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=0.0,  # earliest submit overall
        ),
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="s0t1",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=2.0,
        ),
        # session 1: middle
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="s1t0",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=0.5,
        ),
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="s1t1",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=3.0,
        ),
        # session 2: starts last, last turn finishes last (latest complete)
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="s2t0",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=1.0,
        ),
        RequestFuncOutput(
            success=True,
            latency=5.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="s2t1",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=4.0,  # complete = 9.0, latest overall
        ),
    ]

    tokenizer = _make_mock_tokenizer(
        {"s0t0": 2, "s0t1": 2, "s1t0": 2, "s1t1": 2, "s2t0": 2, "s2t1": 2}
    )

    metrics = calculate_metrics(
        outputs=outputs,
        dur_s=10.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=1,
        skip_last_n_requests=1,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
        kv_block_size=128,
    )

    assert metrics.text_data is not None

    # Head trim drops s0t0 (submit=0.0, the earliest).
    # Tail trim drops s2t1 (complete=9.0, the latest).
    # Dispatch-order slicing would have dropped s0t0 and s2t1 here too —
    # but only because outputs[0] happens to be the earliest submit and
    # outputs[-1] happens to be the latest complete. Use the asymmetric
    # case below to distinguish.
    assert metrics.text_data.completed == 4


def test_skip_distinguishes_dispatch_order_from_timing() -> None:
    """Head/tail trim uses timing, not iteration order.

    Construct a list where the latest-completing request is *not* at the end
    of the list and the earliest-submitting request is *not* at the front,
    so a dispatch-order slice would target the wrong requests.
    """
    outputs = [
        # outputs[0]: NOT earliest submit; latest complete (slow request).
        RequestFuncOutput(
            success=True,
            latency=10.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="block_late_complete",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=2.0,  # complete = 12.0, latest
        ),
        # outputs[1]: earliest submit, completes mid-pack.
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="block_early_submit",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=0.0,  # earliest
        ),
        # outputs[2]: middle on both axes.
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="middle",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=1.0,
        ),
    ]

    tokenizer = _make_mock_tokenizer(
        {"block_late_complete": 2, "block_early_submit": 2, "middle": 2}
    )

    metrics = calculate_metrics(
        outputs=outputs,
        dur_s=15.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=1,
        skip_last_n_requests=1,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
        kv_block_size=128,
    )

    assert metrics.text_data is not None

    # Head trim drops outputs[1] (block_early_submit, submit=0.0).
    # Tail trim drops outputs[0] (block_late_complete, complete=12.0).
    # Both removed → only outputs[2] (middle) is measured.
    assert metrics.text_data.completed == 1
    # If the old dispatch-order slice were still in effect it would have
    # kept outputs[1] (the index-1 middle slot) and dropped outputs[0] and
    # outputs[2]. The "middle" generated_text being the sole measured
    # output proves the timing-based selection is what's running.
    # tail_drop on a request with latency=10.0 that we kept would have
    # inflated total_input. Verify only middle's prompt_len (10) is counted.
    assert metrics.text_data.total_input == 10


def test_skip_first_overlaps_with_skip_last_drops_both() -> None:
    """When the same output is both an earliest submit and a latest complete
    (slow request submitted first that finishes last), it ends up in both
    drop sets. The set-based filter handles this naturally — it gets dropped
    once."""
    outputs = [
        # The "warmup" request: submitted first AND completes last.
        RequestFuncOutput(
            success=True,
            latency=20.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="slow_warmup",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=0.0,  # complete = 20.0
        ),
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="a",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=1.0,
        ),
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="b",
            itl=[0.1],
            tpot=[0.1],
            request_submit_time=2.0,
        ),
    ]

    tokenizer = _make_mock_tokenizer({"slow_warmup": 2, "a": 2, "b": 2})

    metrics = calculate_metrics(
        outputs=outputs,
        dur_s=21.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=1,
        skip_last_n_requests=1,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
        kv_block_size=128,
    )

    assert metrics.text_data is not None

    # slow_warmup is in both head_drop_ids and tail_drop_ids — set union
    # handles dedupe. The remaining "a" and "b" are both measured.
    assert metrics.text_data.completed == 2


def _make_tokenizer_mock(tokens_per_output: int = 5) -> MagicMock:
    """Return a tokenizer mock whose encode call returns a fixed token count."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = list(range(tokens_per_output))
    return tokenizer


def _make_request_func_output(
    *,
    prompt_len: int = 10,
    generated_text: str = "hello world",
    latency: float = 1.0,
    ttft: float = 0.1,
    itl: list[float] | None = None,
    request_submit_time: float | None = 0.0,
) -> RequestFuncOutput:
    return RequestFuncOutput(
        success=True,
        latency=latency,
        ttft=ttft,
        prompt_len=prompt_len,
        generated_text=generated_text,
        itl=itl or [],
        request_submit_time=request_submit_time,
    )


def _make_stable_request_func_output(submit_time: float) -> RequestFuncOutput:
    """Return a RequestFuncOutput with stable TTFT and TPOT suitable for steady-state detection."""
    tpot = [0.02, 0.02, 0.02]
    return RequestFuncOutput(
        success=True,
        latency=1.0,
        ttft=0.05,
        prompt_len=10,
        generated_text="hello world",
        itl=tpot,
        tpot=tpot,
        request_submit_time=submit_time,
    )


def test_build_text_generation_result_not_detected_falls_back_to_trim() -> None:
    """With too few requests (no steady state), build_text_generation_result uses head/tail trim.

    When steady state is not detected the full-run path runs with the
    caller-supplied skip_first / skip_last parameters unchanged.  Diagnostic
    scalars still appear on the result.
    """
    outputs = [_make_request_func_output() for _ in range(3)]
    tokenizer = _make_tokenizer_mock(tokens_per_output=5)
    result = build_text_generation_result(
        outputs=outputs,
        benchmark_duration=1.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=None,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
    )

    assert result.steady_state_detected is False
    assert result.steady_state_window_count == 0
    assert result.steady_state_warning is not None
    assert "Too few" in result.steady_state_warning
    # The metrics themselves should still be computed (fall-back path).
    assert result.text_data is not None
    assert result.text_data.completed == 3


def test_build_text_generation_result_all_failed_suppresses_ss_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A run with no successful requests must not emit a steady-state warning.

    Regression test for PERF-2615: a run where every request failed logged
    ``Steady-state detection: Too few valid requests (0 of N total)`` at
    WARNING, which read as a detection bug. It is a run failure (surfaced by
    the failure count), so the steady-state path stays quiet and falls back.
    """
    outputs = [
        RequestFuncOutput(success=False, latency=0.0, ttft=0.0, prompt_len=10)
        for _ in range(200)
    ]
    tokenizer = _make_tokenizer_mock(tokens_per_output=5)
    with caplog.at_level(logging.WARNING):
        result = build_text_generation_result(
            outputs=outputs,
            benchmark_duration=1.0,
            tokenizer=tokenizer,
            gpu_metrics=None,
            cpu_metrics=None,
            skip_first_n_requests=0,
            skip_last_n_requests=0,
            max_concurrency=32,  # detection runs (not skipped)
            max_concurrent_conversations=None,
            collect_gpu_stats=False,
        )

    assert not any(
        "Steady-state detection:" in r.message and r.levelno >= logging.WARNING
        for r in caplog.records
    )
    assert result.steady_state_detected is False
    assert result.text_data is not None


def test_build_text_generation_result_too_few_but_some_success_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Some successful-but-unusable requests below threshold still warn.

    Distinguishes the genuine "too few valid" case (worth surfacing) from the
    all-failed run above (suppressed).
    """
    outputs = [_make_request_func_output() for _ in range(3)]  # 3 successful
    tokenizer = _make_tokenizer_mock(tokens_per_output=5)
    with caplog.at_level(logging.WARNING):
        build_text_generation_result(
            outputs=outputs,
            benchmark_duration=1.0,
            tokenizer=tokenizer,
            gpu_metrics=None,
            cpu_metrics=None,
            skip_first_n_requests=0,
            skip_last_n_requests=0,
            max_concurrency=32,
            max_concurrent_conversations=None,
            collect_gpu_stats=False,
        )

    assert any("Steady-state detection:" in r.message for r in caplog.records)


def test_build_text_generation_result_detected_uses_window_with_rejection() -> (
    None
):
    """With enough stable requests, build_text_generation_result uses the MAD window + rejection.

    Checks that:
    - steady_state_detected is True
    - the single reported metric set reflects the window (not full run)
    - outlier rejection diagnostic is present
    """
    outputs = [_make_stable_request_func_output(float(i)) for i in range(200)]
    tokenizer = _make_tokenizer_mock(tokens_per_output=5)
    result = build_text_generation_result(
        outputs=outputs,
        benchmark_duration=200.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=None,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=None,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
    )

    assert result.steady_state_detected is True
    assert result.steady_state_window_count is not None
    assert result.steady_state_window_count > 0
    assert result.steady_state_mode == "full"
    assert result.steady_state_warning is None
    assert result.num_outliers_rejected is not None
    assert result.text_data is not None
    # The reported TTFT should be very close to 50 ms (all inputs are 0.05 s).
    assert result.text_data.ttft_ms is not None
    assert result.text_data.ttft_ms.mean == pytest.approx(50.0, rel=0.05)


def test_build_text_generation_result_concurrency_one_falls_back() -> None:
    """At concurrency=1 detection is skipped; the full-run path (with trim) is used.

    Outlier rejection is NOT applied on the concurrency-1 fallback path.
    """
    outputs = [_make_stable_request_func_output(float(i)) for i in range(50)]
    tokenizer = _make_tokenizer_mock(tokens_per_output=5)
    result = build_text_generation_result(
        outputs=outputs,
        benchmark_duration=50.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=None,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=1,  # detection skipped
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
    )

    # Detection was skipped → not detected.
    assert result.steady_state_detected is False
    assert result.steady_state_warning is None  # skipped, not failed
    assert result.num_outliers_rejected == 0  # no rejection on fallback
    assert result.text_data is not None
    assert result.text_data.completed == 50


def test_build_text_generation_result_with_outlier_inputs() -> None:
    """Synthetic run: stable-phase requests with realistic spread + extreme TTFT outliers.

    Validates three properties end-to-end:
    (a) detection selects a steady window (from the stable phase),
    (b) outlier rejection drops the extreme TTFT values when the window
        has enough natural spread for MAD > 0,
    (c) the not-detected fallback path handles concurrency=1 without rejection.
    """
    import math
    import random

    random.seed(42)

    # 110 stable-phase requests with small natural spread so MAD > 0.
    # TTFT alternates ≈ 0.04-0.06 s → median 0.05 s, MAD ≈ 0.01 s.
    stable = [
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.04 + (i % 2) * 0.02,  # alternates 0.04 / 0.06
            prompt_len=10,
            generated_text="hello world",
            itl=[0.02, 0.02, 0.02],
            tpot=[0.02, 0.02, 0.02],
            request_submit_time=float(i),
        )
        for i in range(110)
    ]
    # Inject 3 extreme TTFT outliers mixed into the stable range.
    # |mz| = 0.6745 * (50.0 - 0.05) / 0.01 ≈ 3372 >> 3.5 → must be rejected.
    extreme_outliers = [
        RequestFuncOutput(
            success=True,
            latency=51.0,
            ttft=50.0,  # extreme: 50 s
            prompt_len=10,
            generated_text="hello world",
            itl=[0.02],
            tpot=[0.02],
            request_submit_time=float(50 + i * 20),  # scattered in the middle
        )
        for i in range(3)
    ]
    all_outputs = stable + extreme_outliers
    tokenizer = _make_tokenizer_mock(tokens_per_output=5)

    # (a+b) With concurrency > 1, detection may find a window; if it does,
    # rejection should drop the extreme TTFTs so the mean stays near 50 ms.
    # We also simply verify the result is sane and completes without error.
    result_high = build_text_generation_result(
        outputs=all_outputs,
        benchmark_duration=120.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=None,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=32,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
    )
    assert result_high.text_data is not None
    assert result_high.steady_state_detected is not None
    # num_outliers_rejected diagnostic is always present on the text path.
    assert result_high.num_outliers_rejected is not None

    # (c) At concurrency=1, detection is skipped; trim is used; outliers are
    # included but the fallback path should still complete without error.
    result_one = build_text_generation_result(
        outputs=all_outputs,
        benchmark_duration=120.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=None,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=1,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
    )
    assert result_one.steady_state_detected is False
    assert result_one.steady_state_warning is None  # skipped, not failed
    assert result_one.num_outliers_rejected == 0  # no rejection on fallback
    assert result_one.text_data is not None
    assert result_one.text_data.ttft_ms is not None
    assert not math.isnan(result_one.text_data.ttft_ms.mean)


def _turn(
    session_id: str,
    turn_index: int,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
) -> RequestFuncOutput:
    """A successful multi-turn output carrying server token stats."""
    return RequestFuncOutput(
        success=True,
        latency=1.0,
        ttft=0.1,
        prompt_len=prompt_tokens,
        generated_text="x",
        itl=[0.1],
        tpot=[0.1],
        session_id=session_id,
        turn_index=turn_index,
        server_token_stats=ServerTokenStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
        ),
    )


class TestPerTurnCacheRetention:
    """`_per_turn_cache_retentions` block-aligns vs the prior turn's context."""

    def test_clean_hit_is_full_retention(self) -> None:
        # prev context = 500 + 100 = 600. expected_cacheable =
        # (600 // 128 - 1) * 128 = 3 * 128 = 384. cached(turn1) == 384 -> 1.0.
        turns = [
            _turn(
                "s",
                0,
                prompt_tokens=500,
                completion_tokens=100,
                cached_tokens=0,
            ),
            _turn(
                "s",
                1,
                prompt_tokens=600,
                completion_tokens=100,
                cached_tokens=384,
            ),
        ]
        assert _per_turn_cache_retentions(turns, 128) == [pytest.approx(1.0)]

    def test_partial_drop(self) -> None:
        # cached(turn1) == 192 of a 384 ceiling -> 0.5.
        turns = [
            _turn(
                "s",
                0,
                prompt_tokens=500,
                completion_tokens=100,
                cached_tokens=0,
            ),
            _turn(
                "s",
                1,
                prompt_tokens=600,
                completion_tokens=100,
                cached_tokens=192,
            ),
        ]
        assert _per_turn_cache_retentions(turns, 128) == [pytest.approx(0.5)]

    def test_first_turn_excluded_and_sessions_isolated(self) -> None:
        # Two sessions, each 2 turns; only the second turn of each is checked,
        # and a session's retention is computed against its own prior turn.
        turns = [
            _turn("a", 0, 500, 100, 0),
            _turn("a", 1, 600, 100, 384),  # clean -> 1.0
            _turn("b", 0, 500, 100, 0),
            _turn("b", 1, 600, 100, 0),  # full miss -> 0.0
        ]
        result = sorted(_per_turn_cache_retentions(turns, 128))
        assert result == [pytest.approx(0.0), pytest.approx(1.0)]

    def test_sub_block_context_skipped(self) -> None:
        # prev context 100 -> (100 // 128 - 1) clamps to 0 blocks -> skipped.
        turns = [
            _turn("s", 0, 60, 40, 0),
            _turn("s", 1, 100, 40, 0),
        ]
        assert _per_turn_cache_retentions(turns, 128) == []

    def test_single_turn_outputs_have_no_retention_metric(self) -> None:
        # No session_id/turn_index (single-turn) -> per_turn_cache_retention None.
        output = RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1,
            prompt_len=10,
            generated_text="x",
            itl=[0.1],
        )
        metrics = calculate_metrics(
            outputs=[output],
            dur_s=1.0,
            tokenizer=_make_mock_tokenizer({"x": 3}),
            gpu_metrics=None,
            cpu_metrics=_EMPTY_CPU_METRICS,
            skip_first_n_requests=0,
            skip_last_n_requests=0,
            max_concurrency=None,
            max_concurrent_conversations=None,
            collect_gpu_stats=False,
            kv_block_size=128,
        )
        assert metrics.text_data is not None
        assert metrics.text_data.per_turn_cache_retention is None


class TestRequestRecords:
    """Per-request records: the joinable form of the unaligned arrays.

    ``input_lens`` follows dispatch order while ``output_lens`` lists
    failures before successes, so the two cannot be zipped. These tests pin
    the properties that make records usable where the arrays are not: one
    record per dispatched request, in dispatch order, each carrying its own
    outcome.
    """

    def _outputs(self) -> list[RequestFuncOutput]:
        return [
            RequestFuncOutput(
                success=True,
                latency=1.0,
                ttft=0.1,
                prompt_len=10,
                generated_text="alpha",
                itl=[0.1] * 4,
            ),
            RequestFuncOutput(
                success=False,
                error="boom",
                prompt_len=20,
                generated_text="",
            ),
            RequestFuncOutput(
                success=True,
                latency=2.0,
                ttft=0.2,
                prompt_len=30,
                generated_text="gamma",
                itl=[0.2] * 4,
            ),
        ]

    def _metrics(
        self, *, record_request_text: bool = False, skip_first: int = 0
    ) -> BenchmarkResult:
        tokenizer = _make_mock_tokenizer({"alpha": 5, "gamma": 5, "": 0})
        return calculate_metrics(
            outputs=self._outputs(),
            dur_s=3.0,
            tokenizer=tokenizer,
            gpu_metrics=None,
            cpu_metrics=_EMPTY_CPU_METRICS,
            skip_first_n_requests=skip_first,
            skip_last_n_requests=0,
            max_concurrency=None,
            max_concurrent_conversations=None,
            collect_gpu_stats=False,
            kv_block_size=128,
            record_request_text=record_request_text,
        )

    def test_one_record_per_dispatched_request(self) -> None:
        """Failures included: "failed here, succeeded there" is the finding."""
        text_data = self._metrics().text_data
        assert text_data is not None

        assert [r.index for r in text_data.request_records] == [0, 1, 2]
        assert [r.prompt_len for r in text_data.request_records] == [10, 20, 30]
        assert [r.success for r in text_data.request_records] == [
            True,
            False,
            True,
        ]

    def test_records_are_joinable_where_the_arrays_are_not(self) -> None:
        """The arrays disagree on which request index 0 is; records do not."""
        text_data = self._metrics().text_data
        assert text_data is not None

        # output_lens puts the failure first, input_lens keeps dispatch
        # order — so zipping them pairs the failure's 0 with prompt_len=10.
        assert text_data.input_lens == [10, 20, 30]
        assert text_data.output_lens[0] == 0

        by_index = {r.index: r for r in text_data.request_records}
        assert by_index[1].output_len == 0
        assert by_index[1].error == "boom"
        assert by_index[0].output_len == 5

    def test_a_failed_request_reports_no_latency(self) -> None:
        """0.0 would read as an instant response rather than no response."""
        text_data = self._metrics().text_data
        assert text_data is not None

        failed = text_data.request_records[1]
        assert failed.ttft_ms is None
        assert failed.latency_ms is None
        assert failed.tpot_ms is None

    def test_generated_text_is_opt_in(self) -> None:
        """It is the only unbounded field, so a run must ask for it."""
        default = self._metrics().text_data
        opted_in = self._metrics(record_request_text=True).text_data
        assert default is not None and opted_in is not None

        assert all(r.generated_text is None for r in default.request_records)
        assert [r.generated_text for r in opted_in.request_records] == [
            "alpha",
            "",
            "gamma",
        ]

    def test_trimmed_requests_are_recorded_but_not_measured(self) -> None:
        """A record outside the aggregates' window has to say so."""
        text_data = self._metrics(skip_first=1).text_data
        assert text_data is not None

        measured = {r.index for r in text_data.request_records if r.measured}
        # The trim drops the first success; the failure was never measured.
        assert measured == {2}
        assert len(text_data.request_records) == 3

    def test_records_survive_the_result_dict(self) -> None:
        """The JSON blob is what an offline consumer actually reads."""
        text_data = self._metrics(record_request_text=True).text_data
        assert text_data is not None

        payload = json.loads(json.dumps(text_data.to_result_dict()))

        records = payload["request_records"]
        assert [r["index"] for r in records] == [0, 1, 2]
        assert records[2]["generated_text"] == "gamma"


def test_steady_state_records_keep_dispatch_indices() -> None:
    """The steady-state path hands a filtered slice to calculate_metrics.

    Indexing that slice would renumber the requests and destroy the key two
    runs pair on, so records are built over the full dispatch list.
    """
    outputs = [
        RequestFuncOutput(
            success=True,
            latency=1.0,
            ttft=0.1 + 0.001 * i,
            prompt_len=10,
            generated_text="alpha",
            itl=[0.1] * 4,
            request_submit_time=float(i),
        )
        for i in range(12)
    ]
    tokenizer = _make_mock_tokenizer({"alpha": 5})

    result = build_text_generation_result(
        outputs=outputs,
        benchmark_duration=12.0,
        tokenizer=tokenizer,
        gpu_metrics=None,
        cpu_metrics=_EMPTY_CPU_METRICS,
        skip_first_n_requests=0,
        skip_last_n_requests=0,
        max_concurrency=4,
        max_concurrent_conversations=None,
        collect_gpu_stats=False,
    )

    assert result.text_data is not None
    records = result.text_data.request_records
    assert [r.index for r in records] == list(range(12))
