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

"""Tests for the Metrics validation hierarchy."""

from __future__ import annotations

import json

import numpy as np
import pytest
from max.benchmark.benchmark_shared.metrics import (
    BenchmarkResult,
    PercentileMetrics,
    PixelGenAggregates,
    RatePercentileMetrics,
    SpecDecodeStats,
    StandardPercentileMetrics,
    TextGenAggregates,
    ThroughputMetrics,
)
from max.benchmark.benchmark_shared.percentile_metrics import (
    compute_confidence_info,
    finite_or_none,
    json_safe,
)
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# PercentileMetrics construction and formatting
# ---------------------------------------------------------------------------


def test_percentile_metrics_basic_creation() -> None:
    """Test basic creation of PercentileMetrics."""
    metrics = PercentileMetrics(
        mean=10.0,
        std=2.0,
        p50=9.5,
        p90=12.0,
        p95=14.0,
        p99=18.0,
        unit="ms",
    )
    assert metrics.mean == 10.0
    assert metrics.std == 2.0
    assert metrics.p50 == 9.5
    assert metrics.p90 == 12.0
    assert metrics.p95 == 14.0
    assert metrics.p99 == 18.0
    assert metrics.unit == "ms"


def test_percentile_metrics_creation_without_unit() -> None:
    """Test creating PercentileMetrics without unit."""
    metrics = PercentileMetrics(
        mean=10.0, std=2.0, p50=9.5, p90=12.0, p95=14.0, p99=18.0
    )
    assert metrics.unit is None


def test_percentile_metrics_str_representation() -> None:
    """Test string representation of PercentileMetrics."""
    metrics = PercentileMetrics(
        mean=10.5,
        std=2.3,
        p50=9.8,
        p90=12.7,
        p95=14.2,
        p99=18.9,
    )
    result = str(metrics)

    # All stats render on a single line in a table-aligned key=value form.
    assert "\n" not in result
    assert "Mean=10.50" in result
    assert "Std=2.30" in result
    assert "P50=9.80" in result
    assert "P90=12.70" in result
    assert "P95=14.20" in result
    assert "P99=18.90" in result


def test_percentile_metrics_format_with_prefix() -> None:
    """Test format_with_prefix method."""
    metrics = PercentileMetrics(
        mean=10.0,
        std=2.0,
        p50=9.5,
        p90=12.0,
        p95=14.0,
        p99=18.0,
        unit="ms",
    )
    result = metrics.format_with_prefix("latency")

    # The prefix and unit label the whole single line; each stat is a column.
    assert "\n" not in result
    assert "latency (ms)" in result
    assert "Mean=10.00" in result
    assert "Std=2.00" in result
    assert "P50=9.50" in result
    assert "P90=12.00" in result
    assert "P95=14.00" in result
    assert "P99=18.00" in result


def test_percentile_metrics_format_with_prefix_override_unit() -> None:
    """Test format_with_prefix with unit override."""
    metrics = PercentileMetrics(
        mean=10.0,
        std=2.0,
        p50=9.5,
        p90=12.0,
        p95=14.0,
        p99=18.0,
        unit="ms",
    )
    result = metrics.format_with_prefix("latency", unit="seconds")

    assert "\n" not in result
    assert "latency (seconds)" in result
    assert "Mean=10.00" in result
    assert "P99=18.00" in result


def test_percentile_metrics_format_with_prefix_no_unit() -> None:
    """Test format_with_prefix without unit."""
    metrics = PercentileMetrics(
        mean=10.0, std=2.0, p50=9.5, p90=12.0, p95=14.0, p99=18.0
    )
    result = metrics.format_with_prefix("metric")

    assert "\n" not in result
    assert "metric" in result
    assert "Mean=10.00" in result
    assert "P99=18.00" in result
    assert "(ms)" not in result
    assert "(seconds)" not in result


# ---------------------------------------------------------------------------
# StandardPercentileMetrics construction and data validation
# ---------------------------------------------------------------------------


def test_standard_percentile_metrics_basic_functionality() -> None:
    """Test basic StandardPercentileMetrics functionality."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    metrics = StandardPercentileMetrics(data)

    assert metrics.mean == pytest.approx(5.5, rel=1e-10)
    assert metrics.p50 == pytest.approx(5.5, rel=1e-10)

    assert metrics.p90 == pytest.approx(np.percentile(data, 90), rel=1e-10)
    assert metrics.p95 == pytest.approx(np.percentile(data, 95), rel=1e-10)
    assert metrics.p99 == pytest.approx(np.percentile(data, 99), rel=1e-10)


def test_standard_percentile_metrics_scale_factor() -> None:
    """Test scale factor functionality."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    scale_factor = 1000.0

    metrics = StandardPercentileMetrics(data, scale_factor=scale_factor)

    assert metrics.mean == pytest.approx(3.0 * scale_factor, rel=1e-10)
    assert metrics.p50 == pytest.approx(3.0 * scale_factor, rel=1e-10)
    assert metrics.p90 == pytest.approx(
        np.percentile(data, 90) * scale_factor, rel=1e-10
    )


def test_standard_percentile_metrics_with_unit() -> None:
    """Test StandardPercentileMetrics with unit."""
    assert StandardPercentileMetrics([1.0, 2.0, 3.0], unit="ms").unit == "ms"


def test_standard_percentile_metrics_str_representation() -> None:
    """Test string representation uses 'metric' prefix."""
    result = str(StandardPercentileMetrics([1.0, 2.0, 3.0]))
    assert "metric" in result.lower()


def test_standard_percentile_metrics_empty_data_assertion() -> None:
    """Test that empty data raises assertion error."""
    with pytest.raises(AssertionError, match="data must not be empty"):
        StandardPercentileMetrics([])


def test_standard_percentile_metrics_non_list_data_assertion() -> None:
    """Test that non-list data raises assertion error."""
    with pytest.raises(AssertionError, match="data must be a list"):
        StandardPercentileMetrics((1.0, 2.0, 3.0))  # type: ignore


def test_standard_percentile_metrics_non_float_data_assertion() -> None:
    """Test that non-float data raises assertion error."""
    with pytest.raises(AssertionError, match="data must contain only floats"):
        StandardPercentileMetrics([1, 2, 3])


def test_standard_percentile_metrics_single_value() -> None:
    """Test with single value in data."""
    metrics = StandardPercentileMetrics([5.0])

    assert metrics.mean == 5.0
    assert metrics.std == 0.0
    assert metrics.p50 == 5.0
    assert metrics.p90 == 5.0
    assert metrics.p95 == 5.0
    assert metrics.p99 == 5.0


# ---------------------------------------------------------------------------
# ThroughputMetrics construction, reversed percentiles, and data validation
# ---------------------------------------------------------------------------


def test_throughput_metrics_basic_functionality() -> None:
    """Test basic ThroughputMetrics functionality."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    metrics = ThroughputMetrics(data)

    assert metrics.mean == pytest.approx(5.5, rel=1e-10)
    assert metrics.p50 == pytest.approx(5.5, rel=1e-10)


def test_throughput_metrics_reversed_percentiles() -> None:
    """Test that percentiles are reversed for throughput (lower percentiles for p90, p95, p99)."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    metrics = ThroughputMetrics(data)

    assert metrics.p90 == pytest.approx(np.percentile(data, 10), rel=1e-10)
    assert metrics.p95 == pytest.approx(np.percentile(data, 5), rel=1e-10)
    assert metrics.p99 == pytest.approx(np.percentile(data, 1), rel=1e-10)


def test_throughput_metrics_vs_standard_percentiles() -> None:
    """Test that throughput percentiles are lower than standard percentiles."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    throughput_metrics = ThroughputMetrics(data)
    standard_metrics = StandardPercentileMetrics(data)

    assert throughput_metrics.p90 < standard_metrics.p90
    assert throughput_metrics.p95 < standard_metrics.p95
    assert throughput_metrics.p99 < standard_metrics.p99


def test_throughput_metrics_scale_factor() -> None:
    """Test scale factor functionality."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    scale_factor = 1000.0

    metrics = ThroughputMetrics(data, scale_factor=scale_factor)

    assert metrics.mean == pytest.approx(3.0 * scale_factor, rel=1e-10)
    assert metrics.p50 == pytest.approx(3.0 * scale_factor, rel=1e-10)
    assert metrics.p90 == pytest.approx(
        np.percentile(data, 10) * scale_factor, rel=1e-10
    )


def test_throughput_metrics_with_unit() -> None:
    """Test ThroughputMetrics with unit."""
    assert ThroughputMetrics([1.0, 2.0, 3.0], unit="tok/s").unit == "tok/s"


def test_throughput_metrics_str_representation() -> None:
    """Test string representation uses 'throughput' prefix."""
    result = str(ThroughputMetrics([1.0, 2.0, 3.0]))
    assert "throughput" in result.lower()


def test_throughput_metrics_empty_data_assertion() -> None:
    """Test that empty data raises assertion error."""
    with pytest.raises(AssertionError, match="data must not be empty"):
        ThroughputMetrics([])


def test_throughput_metrics_non_list_data_assertion() -> None:
    """Test that non-list data raises assertion error."""
    with pytest.raises(AssertionError, match="data must be a list"):
        ThroughputMetrics((1.0, 2.0, 3.0))  # type: ignore


def test_throughput_metrics_non_float_data_assertion() -> None:
    """Test that non-float data raises assertion error."""
    with pytest.raises(AssertionError, match="data must contain only floats"):
        ThroughputMetrics([1, 2, 3])


def test_throughput_metrics_single_value() -> None:
    """Test with single value in data."""
    metrics = ThroughputMetrics([5.0])

    assert metrics.mean == 5.0
    assert metrics.std == 0.0
    assert metrics.p50 == 5.0
    assert metrics.p90 == 5.0
    assert metrics.p95 == 5.0
    assert metrics.p99 == 5.0


# ---------------------------------------------------------------------------
# Integration: StandardPercentileMetrics and ThroughputMetrics together
# ---------------------------------------------------------------------------


def test_both_metrics_with_same_data() -> None:
    """Both metric types compute mean/median the same but differ on percentiles."""
    data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

    standard = StandardPercentileMetrics(data, scale_factor=1000.0, unit="ms")
    throughput = ThroughputMetrics(data, scale_factor=1.0, unit="tok/s")

    assert standard.mean == throughput.mean * 1000.0
    assert standard.p50 == throughput.p50 * 1000.0

    assert standard.p90 > throughput.p90 * 1000.0
    assert standard.p95 > throughput.p95 * 1000.0
    assert standard.p99 > throughput.p99 * 1000.0


def test_edge_case_large_dataset() -> None:
    """Both metric types handle large datasets and maintain expected orderings."""
    np.random.seed(42)
    data = np.random.normal(50.0, 10.0, 1000).tolist()

    standard = StandardPercentileMetrics(data)
    throughput = ThroughputMetrics(data)

    assert isinstance(standard.mean, float)
    assert isinstance(throughput.mean, float)
    assert standard.mean == pytest.approx(throughput.mean, rel=1e-10)

    assert standard.p99 > standard.p95 > standard.p90
    assert throughput.p90 > throughput.p95 > throughput.p99


# ---------------------------------------------------------------------------
# PercentileMetrics.validate_metrics()
# ---------------------------------------------------------------------------


def test_percentile_metrics_valid() -> None:
    """Healthy PercentileMetrics passes validation."""
    pm = PercentileMetrics(
        mean=10.0, std=1.0, p50=9.5, p90=15.0, p95=18.0, p99=20.0
    )
    ok, errors = pm.validate_metrics()
    assert ok is True
    assert errors == []


def test_percentile_metrics_nan_mean() -> None:
    """NaN mean is flagged by PercentileMetrics.validate_metrics()."""
    pm = PercentileMetrics(
        mean=float("nan"), std=1.0, p50=9.5, p90=15.0, p95=18.0, p99=20.0
    )
    ok, errors = pm.validate_metrics()
    assert ok is False
    assert len(errors) == 1
    assert "mean" in errors[0].lower()


def test_percentile_metrics_zero_mean() -> None:
    """Zero mean is flagged by PercentileMetrics.validate_metrics()."""
    pm = PercentileMetrics(
        mean=0.0, std=1.0, p50=9.5, p90=15.0, p95=18.0, p99=20.0
    )
    ok, errors = pm.validate_metrics()
    assert ok is False
    assert len(errors) == 1


def test_percentile_metrics_inf_mean() -> None:
    """Infinite mean is flagged by PercentileMetrics.validate_metrics()."""
    pm = PercentileMetrics(
        mean=float("inf"), std=0.0, p50=0.0, p90=0.0, p95=0.0, p99=0.0
    )
    ok, _ = pm.validate_metrics()
    assert ok is False


def test_percentile_metrics_negative_mean() -> None:
    """Negative mean is flagged by PercentileMetrics.validate_metrics()."""
    pm = PercentileMetrics(
        mean=-5.0, std=1.0, p50=9.5, p90=15.0, p95=18.0, p99=20.0
    )
    ok, _ = pm.validate_metrics()
    assert ok is False


# ---------------------------------------------------------------------------
# ThroughputMetrics.validate_metrics()
# ---------------------------------------------------------------------------


def test_throughput_metrics_valid() -> None:
    """Healthy ThroughputMetrics passes validation."""
    tm = ThroughputMetrics([50.0, 60.0], unit="tok/s")
    ok, errors = tm.validate_metrics()
    assert ok is True
    assert errors == []


def test_throughput_metrics_nan() -> None:
    """NaN input produces a validation error via delegation."""
    tm = ThroughputMetrics([float("nan")], unit="tok/s")
    ok, errors = tm.validate_metrics()
    assert ok is False
    assert len(errors) == 1
    assert "mean" in errors[0].lower()


# ---------------------------------------------------------------------------
# StandardPercentileMetrics.validate_metrics()
# ---------------------------------------------------------------------------


def test_standard_percentile_metrics_valid() -> None:
    """Healthy StandardPercentileMetrics passes validation."""
    spm = StandardPercentileMetrics(
        [0.05, 0.06], scale_factor=1000.0, unit="ms"
    )
    ok, errors = spm.validate_metrics()
    assert ok is True
    assert errors == []


def test_standard_percentile_metrics_nan() -> None:
    """NaN input produces a validation error via delegation."""
    spm = StandardPercentileMetrics(
        [float("nan")], scale_factor=1000.0, unit="ms"
    )
    ok, errors = spm.validate_metrics()
    assert ok is False
    assert len(errors) == 1
    assert "mean" in errors[0].lower()


# ---------------------------------------------------------------------------
# RatePercentileMetrics.validate_metrics()
# ---------------------------------------------------------------------------


def test_rate_percentile_metrics_in_range_passes() -> None:
    """Mean inside [0, 100] (percent mode) passes validation."""
    rpm = RatePercentileMetrics([0.2, 0.5, 0.8], as_percent=True)
    ok, errors = rpm.validate_metrics()
    assert ok is True
    assert errors == []


def test_rate_percentile_metrics_above_upper_bound_flagged() -> None:
    """Mean above scale_factor (e.g. cached > prompt bug) is flagged."""
    rpm = RatePercentileMetrics([1.5], as_percent=True)
    ok, errors = rpm.validate_metrics()
    assert ok is False
    assert any("outside [0, 100" in e for e in errors)


def test_rate_percentile_metrics_negative_flagged() -> None:
    """Negative mean is flagged."""
    rpm = RatePercentileMetrics([-0.1], as_percent=True)
    ok, _ = rpm.validate_metrics()
    assert ok is False


def test_rate_percentile_metrics_fraction_mode_bound() -> None:
    """as_percent=False enforces [0, 1] bound."""
    rpm = RatePercentileMetrics([0.95], as_percent=False)
    ok, errors = rpm.validate_metrics()
    assert ok is True
    assert errors == []

    rpm_oob = RatePercentileMetrics([1.5], as_percent=False)
    ok, errors = rpm_oob.validate_metrics()
    assert ok is False
    assert any("outside [0, 1" in e for e in errors)


# ---------------------------------------------------------------------------
# BenchmarkResult.validate_metrics()
# ---------------------------------------------------------------------------


def _make_metrics(
    *,
    completed: int = 10,
    failures: int = 0,
    excluded_successful: int = 0,
    total_input: int = 500,
    total_output: int = 200,
    max_output: int = 50,
    request_throughput: float = 5.0,
    output_throughput_values: list[float] | None = None,
    tpot_values: list[float] | None = None,
    itl_values: list[float] | None = None,
    ttft_values: list[float] | None = None,
    latency_values: list[float] | None = None,
) -> BenchmarkResult:
    """Build a text-gen BenchmarkResult with defaults that pass validation.

    Individual fields can be overridden to inject specific degenerate values.
    """
    output_throughput_values = output_throughput_values or [50.0]
    tpot_values = tpot_values or [0.01]
    itl_values = itl_values or [0.01]
    ttft_values = ttft_values or [0.05]
    latency_values = latency_values or [0.5]

    return BenchmarkResult(
        task_type="text",
        max_concurrency=1,
        peak_gpu_memory_mib=[],
        available_gpu_memory_mib=[],
        gpu_utilization=[],
        text_data=TextGenAggregates(
            duration=10.0,
            completed=completed,
            failures=failures,
            excluded_successful=excluded_successful,
            request_throughput=request_throughput,
            latency_ms=StandardPercentileMetrics(
                latency_values, scale_factor=1000.0, unit="ms"
            ),
            total_input=total_input,
            total_output=total_output,
            nonempty_response_chunks=total_output,
            input_throughput=ThroughputMetrics([100.0], unit="tok/s"),
            output_throughput=ThroughputMetrics(
                output_throughput_values, unit="tok/s"
            ),
            ttft_ms=StandardPercentileMetrics(
                ttft_values, scale_factor=1000.0, unit="ms"
            ),
            tpot_ms=StandardPercentileMetrics(
                tpot_values, scale_factor=1000.0, unit="ms"
            ),
            itl_ms=StandardPercentileMetrics(
                itl_values, scale_factor=1000.0, unit="ms"
            ),
            max_input=100,
            max_output=max_output,
            max_total=150,
            global_cached_token_rate=0.35,
            per_turn_cached_token_rate=RatePercentileMetrics(
                [0.35], as_percent=True
            ),
        ),
    )


def test_healthy_metrics_pass_validation() -> None:
    """Healthy metrics produce no validation errors."""
    ok, errors = _make_metrics().validate_metrics()
    assert ok is True
    assert errors == []


def test_failures_detected() -> None:
    """Any failed requests are flagged."""
    ok, errors = _make_metrics(failures=3).validate_metrics()
    assert ok is False
    assert any("failures=3" in e for e in errors)


def test_zero_completed_detected() -> None:
    """Zero completed requests are flagged."""
    ok, errors = _make_metrics(completed=0).validate_metrics()
    assert ok is False
    assert any("completed=0" in e for e in errors)


def test_all_excluded_by_skip_reports_insufficient_data() -> None:
    """Successes eaten by the skip windows report one insufficient-data error.

    Mirrors the Meta-Llama-3.1 multi-turn-v2 mc=64 failure: 91 successful turns, all
    excluded by skip_first=64 + skip_last=64. The server completed requests,
    so "No requests completed" / "No output tokens generated" would misread
    as a dead server.
    """
    ok, errors = _make_metrics(
        completed=0,
        excluded_successful=91,
        total_output=0,
        request_throughput=0.0,
    ).validate_metrics()
    assert ok is False
    assert len(errors) == 1
    assert "Insufficient data" in errors[0]
    assert "91" in errors[0]
    assert not any("No requests completed" in e for e in errors)
    assert not any("total_output" in e for e in errors)


def test_partial_exclusion_does_not_mask_other_errors() -> None:
    """A nonzero measured set keeps normal validation even with exclusions."""
    ok, errors = _make_metrics(
        completed=5, excluded_successful=100, total_output=0
    ).validate_metrics()
    assert ok is False
    assert any("total_output=0" in e for e in errors)
    assert not any("Insufficient data" in e for e in errors)


def test_zero_output_tokens_detected() -> None:
    """Zero total output tokens are flagged."""
    ok, errors = _make_metrics(total_output=0).validate_metrics()
    assert ok is False
    assert any("total_output=0" in e for e in errors)


def test_zero_request_throughput_detected() -> None:
    """Zero request throughput is flagged."""
    ok, errors = _make_metrics(request_throughput=0.0).validate_metrics()
    assert ok is False
    assert any("request_throughput" in e for e in errors)


def test_zero_cache_rate_passes() -> None:
    """A 0% per-turn cache hit rate is valid (cold cache, not a benchmark error)."""
    metrics = _make_metrics()
    assert metrics.text_data is not None
    metrics.text_data.per_turn_cached_token_rate = RatePercentileMetrics(
        [0.0, 0.0, 0.0], as_percent=True
    )
    ok, errors = metrics.validate_metrics()
    assert ok is True
    assert errors == []


def test_nan_request_throughput_detected() -> None:
    """NaN request throughput is flagged."""
    ok, errors = _make_metrics(
        request_throughput=float("nan")
    ).validate_metrics()
    assert ok is False
    assert any("request_throughput" in e for e in errors)


def test_nan_output_throughput_detected() -> None:
    """NaN output throughput mean is flagged with field prefix."""
    ok, errors = _make_metrics(
        output_throughput_values=[float("nan")]
    ).validate_metrics()
    assert ok is False
    assert any("output_throughput" in e for e in errors)


def test_nan_ttft_detected() -> None:
    """NaN TTFT mean is flagged with field prefix."""
    ok, errors = _make_metrics(ttft_values=[float("nan")]).validate_metrics()
    assert ok is False
    assert any("ttft_ms" in e for e in errors)


def test_nan_latency_detected() -> None:
    """NaN latency mean is flagged with field prefix."""
    ok, errors = _make_metrics(latency_values=[float("nan")]).validate_metrics()
    assert ok is False
    assert any("latency_ms" in e for e in errors)


def test_all_degenerate_reports_all_errors() -> None:
    """A fully degenerate benchmark reports errors for every checked field."""
    ok, errors = _make_metrics(
        completed=0,
        failures=5,
        total_output=0,
        request_throughput=0.0,
        output_throughput_values=[float("nan")],
        ttft_values=[float("nan")],
        latency_values=[float("nan")],
    ).validate_metrics()
    assert ok is False
    assert len(errors) == 7
    assert any("failures" in e for e in errors)
    assert any("completed" in e for e in errors)
    assert any("total_output" in e for e in errors)
    assert any("request_throughput" in e for e in errors)
    assert any("output_throughput" in e for e in errors)
    assert any("ttft_ms" in e for e in errors)
    assert any("latency_ms" in e for e in errors)


def test_inf_throughput_detected() -> None:
    """Infinite throughput is flagged as invalid."""
    ok, errors = _make_metrics(
        request_throughput=float("inf")
    ).validate_metrics()
    assert ok is False
    assert any("request_throughput" in e for e in errors)


def test_negative_throughput_detected() -> None:
    """Negative throughput is flagged as invalid."""
    ok, errors = _make_metrics(request_throughput=-1.0).validate_metrics()
    assert ok is False
    assert any("request_throughput" in e for e in errors)


def test_sub_metric_errors_prefixed() -> None:
    """Sub-metric errors are prefixed with the field name."""
    ok, errors = _make_metrics(
        output_throughput_values=[float("nan")]
    ).validate_metrics()
    assert ok is False
    assert any(e.startswith("output_throughput: ") for e in errors)


def test_prefill_only_skips_decode_phase_metrics() -> None:
    """Prefill-only workloads (max_output<=1) tolerate degenerate decode-phase metrics."""
    ok, errors = _make_metrics(
        total_output=40,
        max_output=1,
        output_throughput_values=[0.0],
        tpot_values=[float("nan")],
        itl_values=[float("nan")],
    ).validate_metrics()
    assert ok is True
    assert errors == []


def test_prefill_only_still_validates_non_decode_metrics() -> None:
    """Prefill-only skips decode-phase metrics but still catches other failures."""
    ok, errors = _make_metrics(
        max_output=1,
        output_throughput_values=[0.0],
        tpot_values=[float("nan")],
        itl_values=[float("nan")],
        request_throughput=0.0,
    ).validate_metrics()
    assert ok is False
    assert any("request_throughput" in e for e in errors)
    assert not any("output_throughput" in e for e in errors)


# ---------------------------------------------------------------------------
# Confidence interval computation
# ---------------------------------------------------------------------------


def test_confidence_info_known_data() -> None:
    """CI on 100 identical values should be very narrow (high confidence)."""
    data = [0.05] * 100
    ci = compute_confidence_info(data, scaled_mean=50.0, scale_factor=1000.0)
    assert ci is not None
    assert ci.sample_size == 100
    assert ci.confidence == "high"
    assert ci.ci_relative_width < 0.01


def test_confidence_info_insufficient_data() -> None:
    """Fewer than 5 samples should be classified as insufficient_data."""
    data = [0.05, 0.06, 0.04]
    ci = compute_confidence_info(data, scaled_mean=50.0, scale_factor=1000.0)
    assert ci is not None
    assert ci.confidence == "insufficient_data"
    assert ci.sample_size == 3


def test_confidence_info_single_sample() -> None:
    """A single sample should return None."""
    ci = compute_confidence_info([0.05], scaled_mean=50.0, scale_factor=1000.0)
    assert ci is None


def test_confidence_info_wide_ci() -> None:
    """High-variance data should produce low confidence."""
    import random

    random.seed(42)
    data = [random.uniform(0.01, 1.0) for _ in range(10)]
    mean = sum(data) / len(data) * 1000.0
    ci = compute_confidence_info(data, scaled_mean=mean, scale_factor=1000.0)
    assert ci is not None
    assert ci.confidence == "low"


def test_confidence_info_nan_mean() -> None:
    """NaN mean should return None."""
    ci = compute_confidence_info(
        [0.05, 0.06], scaled_mean=float("nan"), scale_factor=1000.0
    )
    assert ci is None


def test_standard_percentile_metrics_has_confidence() -> None:
    """StandardPercentileMetrics should populate confidence_info."""
    spm = StandardPercentileMetrics(
        [0.05, 0.06, 0.04, 0.05, 0.055] * 20,
        scale_factor=1000.0,
        unit="ms",
    )
    assert spm.confidence_info is not None
    assert spm.confidence_info.sample_size == 100


def test_throughput_metrics_has_confidence() -> None:
    """ThroughputMetrics should populate confidence_info."""
    tm = ThroughputMetrics([50.0, 60.0, 55.0] * 10, unit="tok/s")
    assert tm.confidence_info is not None
    assert tm.confidence_info.sample_size == 30


def test_confidence_warnings_empty_for_healthy() -> None:
    """Healthy metrics should produce no confidence warnings."""
    m = _make_metrics(
        ttft_values=[0.05 + i * 0.001 for i in range(50)],
        latency_values=[0.5 + i * 0.01 for i in range(50)],
        output_throughput_values=[50.0 + i * 0.5 for i in range(50)],
    )
    assert m.confidence_warnings() == []


def test_confidence_warnings_for_low_confidence() -> None:
    """Low-confidence metrics should produce warnings."""
    import random

    random.seed(99)
    m = _make_metrics(
        ttft_values=[random.uniform(0.01, 1.0) for _ in range(6)],
    )
    warns = m.confidence_warnings()
    assert any("ttft_ms" in w for w in warns)


# ---------------------------------------------------------------------------
# PercentileMetrics.to_flat_dict / confidence_to_flat_dict
# ---------------------------------------------------------------------------


def test_percentile_to_flat_dict() -> None:
    pm = PercentileMetrics(
        mean=10.0, std=1.0, p50=9.5, p90=15.0, p95=18.0, p99=20.0
    )
    d = pm.to_flat_dict("ttft_ms")
    assert d == {
        "mean_ttft_ms": 10.0,
        "std_ttft_ms": 1.0,
        "median_ttft_ms": 9.5,
        "p90_ttft_ms": 15.0,
        "p95_ttft_ms": 18.0,
        "p99_ttft_ms": 20.0,
    }


def test_confidence_to_flat_dict_present() -> None:
    from max.benchmark.benchmark_shared.metrics import ConfidenceInfo

    ci = ConfidenceInfo(
        ci_lower=9.0,
        ci_upper=11.0,
        ci_relative_width=0.2,
        confidence="medium",
        sample_size=30,
    )
    pm = PercentileMetrics(
        mean=10.0,
        std=1.0,
        p50=9.5,
        p90=15.0,
        p95=18.0,
        p99=20.0,
        confidence_info=ci,
    )
    d = pm.confidence_to_flat_dict("ttft_ms")
    assert d["ttft_ms_confidence"] == "medium"
    assert d["ttft_ms_ci_lower"] == 9.0
    assert d["ttft_ms_sample_size"] == 30


def test_confidence_to_flat_dict_absent() -> None:
    pm = PercentileMetrics(
        mean=10.0, std=1.0, p50=9.5, p90=15.0, p95=18.0, p99=20.0
    )
    assert pm.confidence_to_flat_dict("x") == {}


def test_standard_percentile_delegates_to_flat_dict() -> None:
    spm = StandardPercentileMetrics([0.05, 0.06], scale_factor=1000.0)
    d = spm.to_flat_dict("ttft_ms")
    assert "mean_ttft_ms" in d
    assert "p99_ttft_ms" in d


# ---------------------------------------------------------------------------
# BenchmarkMetrics.to_result_dict
# ---------------------------------------------------------------------------


def test_benchmark_metrics_to_result_dict_keys() -> None:
    m = _make_metrics()
    d = m.to_result_dict()
    assert d["duration"] == 10.0
    assert d["completed"] == 10
    assert d["total_input_tokens"] == 500
    assert d["total_output_tokens"] == 200
    # (500 + 200) tokens * 60 / 10.0 s = 4200 tokens/min.
    assert d["aggregate_tokens_per_minute"] == 4200.0
    assert d["request_throughput"] == 5.0
    assert "mean_ttft_ms" in d
    assert "p99_latency_ms" in d
    assert "mean_input_throughput" in d
    assert "p99_output_throughput" in d


def test_aggregate_tokens_per_minute_derived_and_serialized() -> None:
    """The derived TPM is a real field, so it flows through model_dump (JSON)."""
    result = _make_metrics(total_input=500, total_output=200)
    text_data = result.text_data
    assert text_data is not None
    # Derived from token totals and duration, regardless of construction path.
    assert text_data.aggregate_tokens_per_minute == 4200.0
    # A stored field (not a computed_field), so it appears in model_dump — the
    # payload the results-publication JSON reporter serializes.
    dumped = result.model_dump()
    assert dumped["text_data"]["aggregate_tokens_per_minute"] == 4200.0


def test_aggregate_tokens_per_minute_nan_for_nonpositive_duration() -> None:
    """Degenerate runs with non-positive duration yield NaN, not a crash."""
    text_data = TextGenAggregates(
        duration=0.0,
        completed=0,
        failures=0,
        request_throughput=0.0,
        total_input=0,
        total_output=0,
        nonempty_response_chunks=0,
        max_input=0,
        max_output=0,
        max_total=0,
        global_cached_token_rate=0.0,
    )
    assert np.isnan(text_data.aggregate_tokens_per_minute)


# ---------------------------------------------------------------------------
# Pixel-gen BenchmarkResult.to_result_dict
# ---------------------------------------------------------------------------


def test_pixel_metrics_to_result_dict() -> None:
    from max.benchmark.benchmark_shared.metrics import PixelGenAggregates

    pm = BenchmarkResult(
        task_type="pixel",
        max_concurrency=2,
        peak_gpu_memory_mib=[],
        available_gpu_memory_mib=[],
        gpu_utilization=[],
        pixel_data=PixelGenAggregates(
            duration=5.0,
            completed=8,
            failures=0,
            request_throughput=1.6,
            latency_ms=StandardPercentileMetrics(
                [0.5, 0.6], scale_factor=1000.0
            ),
            total_generated_outputs=8,
        ),
    )
    d = pm.to_result_dict()
    assert d["total_generated_outputs"] == 8
    assert "mean_latency_ms" in d
    assert "p99_latency_ms" in d


def test_benchmark_result_rejects_text_only_fields_for_pixel_task() -> None:
    """Diagnostic steady-state scalars on a pixel task raise ValidationError."""
    with pytest.raises(ValidationError):
        BenchmarkResult(
            task_type="pixel",
            max_concurrency=1,
            steady_state_detected=False,
            pixel_data=PixelGenAggregates(
                duration=1.0,
                completed=1,
                failures=0,
                request_throughput=1.0,
                latency_ms=StandardPercentileMetrics(
                    [0.5], scale_factor=1000.0
                ),
                total_generated_outputs=1,
            ),
        )


def test_benchmark_result_to_result_dict_includes_diagnostic_scalars() -> None:
    """to_result_dict emits steady-state diagnostic scalars and spec_decode fields."""
    result = _make_metrics()
    result = result.model_copy(
        update={
            "steady_state_detected": True,
            "steady_state_window_count": 120,
            "steady_state_mode": "full",
            "steady_state_warning": None,
            "num_outliers_rejected": 3,
            "spec_decode_stats": SpecDecodeStats(
                acceptance_rate=55.0,
                acceptance_length=2.0,
            ),
        }
    )
    d = result.to_result_dict()
    assert d["steady_state_detected"] is True
    assert d["steady_state_window_count"] == 120
    assert d["steady_state_mode"] == "full"
    assert d["steady_state_warning"] is None
    assert d["num_outliers_rejected"] == 3
    assert d["spec_decode_acceptance_rate"] == 55.0


def test_benchmark_result_to_result_dict_omits_diagnostics_when_none() -> None:
    """When steady_state_detected is None the diagnostic keys are absent."""
    result = _make_metrics()
    d = result.to_result_dict()
    assert "steady_state_detected" not in d
    assert "steady_state_window_count" not in d
    assert "num_outliers_rejected" not in d


def test_finite_or_none() -> None:
    assert finite_or_none(1.5) == 1.5
    assert finite_or_none(None) is None
    assert finite_or_none(float("nan")) is None
    assert finite_or_none(float("inf")) is None


def test_json_safe_tree_round_trips_strict_json() -> None:
    tree = {
        "ok": 1.0,
        "bad": float("nan"),
        "nested": [float("inf"), (2.0, float("-inf"))],
    }
    text = json.dumps(json_safe(tree), allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["nested"][1] == [2.0, None]
