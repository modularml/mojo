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

"""Tests for ``BenchmarkResult.result_groups`` and ``build_result_groups``."""

from __future__ import annotations

from max.benchmark.benchmark_shared.metrics import (
    BenchmarkResult,
    PixelGenAggregates,
    StandardPercentileMetrics,
    TextGenAggregates,
    build_result_groups,
)
from max.benchmark.benchmark_shared.model_csv import columns_for_type


def _text_result(**kwargs: object) -> BenchmarkResult:
    defaults: dict[str, object] = {
        "task_type": "text",
        "max_concurrency": 8,
        "peak_gpu_memory_mib": [40000.0, 40500.0],
        "available_gpu_memory_mib": [39000.0, 38500.0],
        "gpu_utilization": [95.0, 97.0],
        "server_startup_time": 12.3,
        "steady_state_detected": True,
        "steady_state_window_count": 3,
        "steady_state_mode": "mad",
        "num_outliers_rejected": 2,
        "text_data": TextGenAggregates(
            duration=60.0,
            completed=100,
            failures=0,
            request_throughput=1.66,
            total_input=5000,
            total_output=8000,
            nonempty_response_chunks=8000,
            latency_ms=StandardPercentileMetrics(
                [0.114, 0.115], scale_factor=1000.0, unit="ms"
            ),
            ttft_ms=StandardPercentileMetrics(
                [0.055, 0.056], scale_factor=1000.0, unit="ms"
            ),
            tpot_ms=StandardPercentileMetrics(
                [0.0082, 0.0083], scale_factor=1000.0, unit="ms"
            ),
            itl_ms=StandardPercentileMetrics(
                [0.0082, 0.0083], scale_factor=1000.0, unit="ms"
            ),
            max_input=100,
            max_output=200,
            max_total=300,
            global_cached_token_rate=0.42,
        ),
    }
    defaults.update(kwargs)
    return BenchmarkResult(**defaults)  # type: ignore[arg-type]


def _pixel_result() -> BenchmarkResult:
    return BenchmarkResult(
        task_type="pixel",
        max_concurrency=1,
        pixel_data=PixelGenAggregates(
            duration=30.0,
            completed=10,
            failures=0,
            request_throughput=0.33,
            total_generated_outputs=10,
            latency_ms=StandardPercentileMetrics(
                [1.0, 1.1], scale_factor=1000.0, unit="ms"
            ),
        ),
    )


def test_result_auto_fills_result_groups() -> None:
    result = _text_result()
    assert result.result_groups is not None
    assert result.result_groups.summary.completed == 100


def test_text_groups_cover_expected_sections() -> None:
    groups = _text_result().result_groups
    assert groups is not None

    assert groups.summary.task_type == "text"
    assert groups.summary.completed == 100
    assert groups.summary.mean_ttft_ms is not None

    assert groups.gpu_stats is not None
    assert groups.gpu_stats.peak_gpu_memory_mib == [40000.0, 40500.0]

    assert groups.latency_stats is not None
    assert groups.latency_stats.ttft_ms is not None
    assert groups.throughput_stats is not None
    assert groups.cache_stats is not None
    assert groups.cache_stats.global_cached_token_rate == 0.42

    assert groups.diagnostics is not None
    assert groups.diagnostics.steady_state_detected is True
    assert groups.diagnostics.num_outliers_rejected == 2


def test_pixel_groups_have_no_text_only_sections() -> None:
    groups = _pixel_result().result_groups
    assert groups is not None

    assert groups.summary.total_generated_outputs == 10
    assert groups.summary.mean_ttft_ms is None
    assert groups.gpu_stats is None
    assert groups.throughput_stats is None
    assert groups.cache_stats is None
    assert groups.latency_stats is not None
    assert groups.latency_stats.latency_ms is not None
    # No startup / steady-state / error payload → diagnostics stays None.
    assert groups.diagnostics is None


def test_diagnostics_omits_empty_per_request_errors() -> None:
    result = _text_result(
        text_data=TextGenAggregates(
            duration=60.0,
            completed=2,
            failures=1,
            request_throughput=0.03,
            total_input=10,
            total_output=10,
            nonempty_response_chunks=10,
            errors=["", "boom", ""],
            max_input=5,
            max_output=5,
            max_total=10,
            global_cached_token_rate=0.0,
        )
    )
    assert result.result_groups is not None
    assert result.result_groups.diagnostics is not None
    assert result.result_groups.diagnostics.errors == ["boom"]


def test_to_result_dict_includes_result_groups() -> None:
    payload = _text_result().to_result_dict()
    groups = payload["result_groups"]
    assert isinstance(groups, dict)
    assert groups["summary"]["completed"] == 100


def test_model_dump_includes_result_groups() -> None:
    payload = _text_result().model_dump(mode="json")

    assert payload["task_type"] == "text"
    assert payload["text_data"]["completed"] == 100
    groups = payload["result_groups"]
    assert groups["summary"]["completed"] == 100
    assert groups["latency_stats"]["ttft_ms"]["p99"] is not None

    # Round-trip keeps the stored groups (no need to re-derive).
    restored = BenchmarkResult.model_validate(payload)
    assert restored.result_groups is not None
    assert restored.result_groups.summary.completed == 100


def test_build_result_groups_matches_stored_field() -> None:
    result = _text_result()
    assert result.result_groups == build_result_groups(result)


def test_result_groups_csv_column_stays_opaque() -> None:
    columns = list(columns_for_type(BenchmarkResult))
    assert "result_groups" in columns
    assert not any(c.startswith("result_groups.") for c in columns)
