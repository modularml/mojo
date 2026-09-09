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
"""Tests for benchmark_shared.serving_result_output."""

from __future__ import annotations

import re

import pytest
from max.benchmark.benchmark_shared.metrics import (
    BenchmarkResult,
    PercentileMetrics,
    StandardPercentileMetrics,
    TextGenAggregates,
    ThroughputMetrics,
)
from max.benchmark.benchmark_shared.serving_result_output import (
    PercentileRow,
    elide_data_uris_in_string,
    format_gpu_statistics_title,
    format_percentile_table,
    print_benchmark_summary,
)


def test_elide_data_uris_in_string() -> None:
    """Test that elide_data_uris_in_string correctly elides base64 data URIs."""

    # fmt: off

    # Basic case
    sample = "'image': 'data:image/jpeg;base64,/9j/4AAQSASDEEAE'"
    expected = "'image': 'data:image/jpeg;base64,...(hash: 783e7013, 16 bytes)...'"
    assert elide_data_uris_in_string(sample) == expected

    # Two data URIs in a single string
    sample = "data:image/jpeg;base64,/9j/4AAQSASDEEAE + data:image/jpeg;base64,/9j/4AAQSASDEEAE"
    expected = "data:image/jpeg;base64,...(hash: 783e7013, 16 bytes)... + data:image/jpeg;base64,...(hash: 783e7013, 16 bytes)..."
    assert elide_data_uris_in_string(sample) == expected

    # Still elides even if it results in longer string
    sample = "data:image/jpeg;base64,ABC"
    expected = "data:image/jpeg;base64,...(hash: b5d4045c, 3 bytes)..."
    assert elide_data_uris_in_string(sample) == expected

    # Does not elide if invalid characters in data
    sample = "data:image/jpeg;base64,ദ്ദി(˵ •̀ ᴗ - ˵ ) ✧"
    expected = "data:image/jpeg;base64,ദ്ദി(˵ •̀ ᴗ - ˵ ) ✧"
    assert elide_data_uris_in_string(sample) == expected

    # Does not elide if data uri type is empty
    sample = "data:;base64,ABC"
    expected = "data:;base64,ABC"
    assert elide_data_uris_in_string(sample) == expected

    # `data:` is present in string but not part of data uri
    sample = "Here is some data: 'data:image/jpeg;base64,AAAAAAAAASTUFF=='"
    expected = "Here is some data: 'data:image/jpeg;base64,...(hash: 6c6e1584, 16 bytes)...'"
    assert elide_data_uris_in_string(sample) == expected

    # `;base64` is present in string but not part of data uri
    sample = ";base64"
    expected = ";base64"
    assert elide_data_uris_in_string(sample) == expected

    # String is empty
    sample = ""
    expected = ""
    assert elide_data_uris_in_string(sample) == expected

    # fmt: on


def _pm(
    mean: float,
    std: float,
    p50: float,
    p90: float,
    p95: float,
    p99: float,
) -> PercentileMetrics:
    return PercentileMetrics(
        mean=mean, std=std, p50=p50, p90=p90, p95=p95, p99=p99
    )


def test_format_percentile_table() -> None:
    """The rendered table matches exactly, byte for byte."""
    table = format_percentile_table(
        [
            PercentileRow(
                "TTFT (ms)",
                _pm(15539.31, 4200.50, 15068.37, 28000.00, 31000.00, 33034.17),
            ),
            PercentileRow(
                "TPOT (ms)",
                _pm(34.23, 18.10, 28.47, 60.20, 95.40, 138.55),
            ),
            PercentileRow(
                "ITL (ms)",
                _pm(26.76, 22.40, 5.42, 48.90, 120.30, 228.45),
            ),
            PercentileRow(
                "Request Latency (ms)",
                _pm(20345.10, 5120.30, 19980.40, 30120.50, 33450.10, 35880.90),
            ),
            PercentileRow(
                "Input throughput (tok/s)",
                _pm(2180.51, 210.30, 2200.10, 1980.40, 1900.20, 1820.60),
            ),
            PercentileRow(
                "Output throughput (tok/s)",
                _pm(2301.89, 180.70, 2320.50, 2100.30, 2010.80, 1950.40),
            ),
        ]
    )

    expected = (
        "┌───────────────────────────┬──────────┬─────────┬──────────┬──────────┬──────────┬──────────┐\n"
        "│ Metric                    │     Mean │     Std │      P50 │      P90 │      P95 │      P99 │\n"
        "├───────────────────────────┼──────────┼─────────┼──────────┼──────────┼──────────┼──────────┤\n"
        "│ TTFT (ms)                 │ 15539.31 │ 4200.50 │ 15068.37 │ 28000.00 │ 31000.00 │ 33034.17 │\n"
        "│ TPOT (ms)                 │    34.23 │   18.10 │    28.47 │    60.20 │    95.40 │   138.55 │\n"
        "│ ITL (ms)                  │    26.76 │   22.40 │     5.42 │    48.90 │   120.30 │   228.45 │\n"
        "│ Request Latency (ms)      │ 20345.10 │ 5120.30 │ 19980.40 │ 30120.50 │ 33450.10 │ 35880.90 │\n"
        "│ Input throughput (tok/s)  │  2180.51 │  210.30 │  2200.10 │  1980.40 │  1900.20 │  1820.60 │\n"
        "│ Output throughput (tok/s) │  2301.89 │  180.70 │  2320.50 │  2100.30 │  2010.80 │  1950.40 │\n"
        "└───────────────────────────┴──────────┴─────────┴──────────┴──────────┴──────────┴──────────┘"
    )
    assert table == expected


def test_format_percentile_table_accepts_metric_wrappers() -> None:
    """The helper accepts the computed StandardPercentileMetrics / ThroughputMetrics wrappers."""
    latency = StandardPercentileMetrics([0.5, 0.6, 0.7], scale_factor=1000.0)
    tput = ThroughputMetrics([50.0, 60.0, 70.0], unit="tok/s")

    table = format_percentile_table(
        [
            PercentileRow("Request Latency (ms)", latency),
            PercentileRow("Output throughput (tok/s)", tput),
        ]
    )
    lines = table.splitlines()
    assert any("Request Latency (ms)" in ln for ln in lines)
    assert any("Output throughput (tok/s)" in ln for ln in lines)


def test_format_gpu_statistics_title_names_device_count() -> None:
    """The section title says the percentile rows are aggregated across devices."""
    assert format_gpu_statistics_title(1) == (
        "GPU Statistics (aggregated across 1 device)"
    )
    assert format_gpu_statistics_title(2) == (
        "GPU Statistics (aggregated across 2 devices)"
    )
    assert format_gpu_statistics_title(8) == (
        "GPU Statistics (aggregated across 8 devices)"
    )


def _gpu_benchmark_result() -> BenchmarkResult:
    return BenchmarkResult(
        task_type="text",
        max_concurrency=8,
        peak_gpu_memory_mib=[40000.0, 40500.0],
        available_gpu_memory_mib=[39000.0, 38500.0],
        gpu_utilization=[95.0, 97.0],
        text_data=TextGenAggregates(
            duration=60.0,
            completed=100,
            failures=0,
            request_throughput=1.66,
            total_input=5000,
            total_output=8000,
            nonempty_response_chunks=8000,
            max_input=100,
            max_output=200,
            max_total=300,
            global_cached_token_rate=0.42,
        ),
    )


def test_print_benchmark_summary_tabulates_gpu_stats(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """GPU stats reuse the shared percentile table, not one block per device."""
    print_benchmark_summary(
        metrics=_gpu_benchmark_result(),
        request_rate=float("inf"),
        max_concurrency=8,
        achieved_request_rate=1.66,
        collect_gpu_stats=True,
        collect_cpu_stats=False,
    )
    out = capsys.readouterr().out
    assert "GPU Statistics (aggregated across 2 devices)" in out
    assert "For per-GPU metrics, dump the result JSON instead" in out
    assert "Peak GPU memory (MiB)" in out
    assert "GPU 0 peak memory" not in out
    assert "GPU 1 peak memory" not in out
    # Same Mean/Std/P50/P90/P95/P99 columns as latency / throughput.
    peak = re.search(
        r"│\s*Peak GPU memory \(MiB\)\s*│([^│]+)│([^│]+)│([^│]+)│"
        r"([^│]+)│([^│]+)│([^│]+)│",
        out,
    )
    assert peak is not None
    assert float(peak.group(1).strip()) == 40250.0
    util = re.search(
        r"│\s*GPU utilization \(%\)\s*│([^│]+)│([^│]+)│([^│]+)│"
        r"([^│]+)│([^│]+)│([^│]+)│",
        out,
    )
    assert util is not None
    assert float(util.group(1).strip()) == 96.0
