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
"""Unit tests for benchmark server metrics module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from max.benchmark.benchmark_shared.metrics import (
    BenchmarkResult,
    PrefillDecodeStats,
    RatePercentileMetrics,
    SpecDecodeMetrics,
    SpecDecodeStats,
    StandardPercentileMetrics,
    TextGenAggregates,
    ThroughputMetrics,
    calculate_spec_decode_stats,
)
from max.benchmark.benchmark_shared.server_metrics import (
    HistogramData,
    ParsedMetrics,
    _format_metric_key,
    collect_benchmark_metrics,
    compute_metrics_delta,
    fetch_and_parse_metrics,
    get_metrics_url,
    parse_metrics,
    parse_spec_decode_metrics,
    print_server_metrics,
)

if TYPE_CHECKING:
    from unittest.mock import MagicMock


@pytest.fixture
def sample_metrics() -> str:
    """Representative Prometheus metrics with all types including labeled histograms."""
    return """# HELP requests Total requests
# TYPE requests counter
requests 100.0
# HELP memory_bytes Current memory
# TYPE memory_bytes gauge
memory_bytes 1024.0
# HELP latency Latency
# TYPE latency histogram
latency_bucket{le="0.1"} 50.0
latency_bucket{le="+Inf"} 100.0
latency_sum 25.0
latency_count 100.0
# HELP maxserve_batch_execution_time_milliseconds Batch execution time
# TYPE maxserve_batch_execution_time_milliseconds histogram
maxserve_batch_execution_time_milliseconds_bucket{batch_type="CE",le="100"} 10.0
maxserve_batch_execution_time_milliseconds_bucket{batch_type="CE",le="+Inf"} 15.0
maxserve_batch_execution_time_milliseconds_sum{batch_type="CE"} 900.0
maxserve_batch_execution_time_milliseconds_count{batch_type="CE"} 15.0
maxserve_batch_execution_time_milliseconds_bucket{batch_type="TG",le="100"} 80.0
maxserve_batch_execution_time_milliseconds_bucket{batch_type="TG",le="+Inf"} 100.0
maxserve_batch_execution_time_milliseconds_sum{batch_type="TG"} 2000.0
maxserve_batch_execution_time_milliseconds_count{batch_type="TG"} 100.0
"""


def _make_metrics(
    metrics_by_endpoint: dict[str, ParsedMetrics],
) -> BenchmarkResult:
    """Minimal text-gen BenchmarkResult carrying only the fields under test."""
    return BenchmarkResult(
        task_type="text",
        max_concurrency=10,
        peak_gpu_memory_mib=[],
        available_gpu_memory_mib=[],
        gpu_utilization=[],
        metrics_by_endpoint=metrics_by_endpoint,
        text_data=TextGenAggregates(
            duration=10.0,
            completed=100,
            failures=0,
            request_throughput=10.0,
            latency_ms=StandardPercentileMetrics([1.0]),
            total_input=1000,
            total_output=500,
            nonempty_response_chunks=500,
            input_throughput=ThroughputMetrics([1.0]),
            output_throughput=ThroughputMetrics([1.0]),
            ttft_ms=StandardPercentileMetrics([0.1]),
            tpot_ms=StandardPercentileMetrics([0.01]),
            itl_ms=StandardPercentileMetrics([0.01]),
            max_input=100,
            max_output=50,
            max_total=150,
            global_cached_token_rate=0.35,
            per_turn_cached_token_rate=RatePercentileMetrics(
                [0.35], as_percent=True
            ),
        ),
    )


def test_get_histogram_truth() -> None:
    """Test HistogramData truthiness based on having data."""
    # Empty histogram is falsy
    hist_empty = HistogramData(buckets=[], sum=0.0, count=0.0)
    assert not hist_empty

    # Histogram with data is truthy
    hist_with_buckets = HistogramData(
        buckets=[("100", 1.0)], sum=0.0, count=100.0
    )
    assert hist_with_buckets


def test_histogram_mean() -> None:
    """Test HistogramData calculates mean correctly."""
    hist = HistogramData(buckets=[], sum=150.0, count=10.0)
    assert hist.mean == 15.0

    hist_zero = HistogramData(buckets=[], sum=0.0, count=0.0)
    assert hist_zero.mean == 0.0


def test_get_metrics_url_supported_backends() -> None:
    """Test get_metrics_url returns correct URLs for supported backends."""
    url = get_metrics_url("modular", "http://localhost:8000")
    assert url == "http://localhost:8001/metrics"

    # Test vllm backend (uses same port as base_url)
    url = get_metrics_url("vllm", "http://localhost:8000")
    assert url == "http://localhost:8000/metrics"


def test_get_metrics_url_ipv6() -> None:
    """Bracketed IPv6 base_url must produce bracketed metrics URL."""
    url = get_metrics_url("modular", "http://[fdc3:5d58:ceb7::8:91fa]:8000")
    assert url == "http://[fdc3:5d58:ceb7::8:91fa]:8001/metrics"

    url = get_metrics_url("vllm", "http://[::1]:8000")
    assert url == "http://[::1]:8000/metrics"


def test_parse_metrics_success(sample_metrics: str) -> None:
    """Test parse_metrics extracts all metric types."""
    result = parse_metrics(sample_metrics)

    # Should parse all three metric types
    assert len(result.counters) > 0
    assert len(result.gauges) > 0
    assert len(result.histograms) > 0
    assert result.raw_text == sample_metrics


@patch("max.benchmark.benchmark_shared.server_metrics.requests.get")
def test_fetch_and_parse_http_error(mock_get: MagicMock) -> None:
    """Test that HTTP errors raise HTTPError."""
    import requests

    mock_response = type("Response", (), {"status_code": 500})()
    mock_get.return_value = mock_response

    with pytest.raises(requests.HTTPError, match="Failed to fetch metrics"):
        fetch_and_parse_metrics(
            backend="modular", base_url="http://localhost:8000"
        )


def test_print_server_metrics(
    sample_metrics: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that print_server_metrics outputs formatted metrics."""
    metrics = parse_metrics(sample_metrics)
    print_server_metrics(metrics)

    captured = capsys.readouterr()
    # Check section headers
    assert "Server Metrics (from Prometheus)" in captured.out
    assert "Counters:" in captured.out
    assert "Gauges:" in captured.out
    assert "Histograms:" in captured.out
    # Check specific metrics
    assert "requests" in captured.out
    assert "memory_bytes" in captured.out
    # Check labeled histogram breakdown
    assert "Breakdown:" in captured.out


def test_compute_metrics_delta() -> None:
    """Test that compute_metrics_delta correctly computes deltas."""
    baseline = ParsedMetrics(
        counters={"requests": 100.0},
        gauges={"memory_bytes": 1024.0},
        histograms={
            "latency": HistogramData(
                buckets=[("0.1", 50.0), ("+Inf", 100.0)],
                sum=25.0,
                count=100.0,
            )
        },
        raw_text="",
    )

    final = ParsedMetrics(
        counters={"requests": 250.0},
        gauges={"memory_bytes": 2048.0},
        histograms={
            "latency": HistogramData(
                buckets=[("0.1", 125.0), ("+Inf", 250.0)],
                sum=75.0,
                count=250.0,
            )
        },
        raw_text="",
    )

    delta = compute_metrics_delta(baseline, final)

    # Counters should be delta (final - baseline)
    assert delta.counters["requests"] == 150.0

    # Gauges should be final value
    assert delta.gauges["memory_bytes"] == 2048.0

    # Histograms should be delta
    assert delta.histograms["latency"].sum == 50.0
    assert delta.histograms["latency"].count == 150.0
    assert dict(delta.histograms["latency"].buckets) == {
        "0.1": 75.0,
        "+Inf": 150.0,
    }


def test_compute_metrics_delta_with_labeled_histograms() -> None:
    """Test delta computation preserves histogram labels."""
    baseline = ParsedMetrics(
        counters={},
        gauges={},
        histograms={
            'batch_time{type="prefill"}': HistogramData(
                buckets=[("100", 5.0), ("+Inf", 10.0)],
                sum=500.0,
                count=10.0,
            ),
            'batch_time{type="decode"}': HistogramData(
                buckets=[("100", 40.0), ("+Inf", 50.0)],
                sum=1000.0,
                count=50.0,
            ),
        },
        raw_text="",
    )

    final = ParsedMetrics(
        counters={},
        gauges={},
        histograms={
            'batch_time{type="prefill"}': HistogramData(
                buckets=[("100", 25.0), ("+Inf", 30.0)],
                sum=1500.0,
                count=30.0,
            ),
            'batch_time{type="decode"}': HistogramData(
                buckets=[("100", 140.0), ("+Inf", 150.0)],
                sum=3000.0,
                count=150.0,
            ),
        },
        raw_text="",
    )

    delta = compute_metrics_delta(baseline, final)

    # Check deltas for both labeled histograms
    prefill_delta = delta.get_histogram("batch_time", {"type": "prefill"})
    assert prefill_delta is not None
    assert prefill_delta.sum == 1000.0  # 1500 - 500
    assert prefill_delta.count == 20.0  # 30 - 10
    assert prefill_delta.mean == 50.0
    assert dict(prefill_delta.buckets) == {
        "100": 20.0,  # 25 - 5
        "+Inf": 20.0,  # 30 - 10
    }

    decode_delta = delta.get_histogram("batch_time", {"type": "decode"})
    assert decode_delta is not None
    assert decode_delta.sum == 2000.0  # 3000 - 1000
    assert decode_delta.count == 100.0  # 150 - 50
    assert decode_delta.mean == 20.0
    assert dict(decode_delta.buckets) == {
        "100": 100.0,  # 140 - 40
        "+Inf": 100.0,  # 150 - 50
    }


@pytest.mark.parametrize(
    "name,labels,expected",
    [
        ("metric_name", {}, "metric_name"),
        ("metric_name", {"label": "value"}, 'metric_name{label="value"}'),
        # Multiple labels should be sorted alphabetically
        (
            "metric_name",
            {"b": "val2", "a": "val1"},
            'metric_name{a="val1",b="val2"}',
        ),
    ],
)
def test_format_metric_key(name: str, labels: dict, expected: str) -> None:  # type: ignore[type-arg]
    """Test formatting metric keys with various label combinations."""
    assert _format_metric_key(name, labels) == expected


def test_parse_metrics_with_labeled_histogram(sample_metrics: str) -> None:
    """Test parsing histograms with labels (e.g., batch_type)."""
    metrics = parse_metrics(sample_metrics)

    # Should create separate histogram for each label value
    assert (
        'maxserve_batch_execution_time_milliseconds{batch_type="CE"}'
        in metrics.histograms
    )
    assert (
        'maxserve_batch_execution_time_milliseconds{batch_type="TG"}'
        in metrics.histograms
    )

    prefill = metrics.histograms[
        'maxserve_batch_execution_time_milliseconds{batch_type="CE"}'
    ]
    assert prefill.count == 15.0
    assert prefill.sum == 900.0
    assert prefill.mean == 60.0

    decode = metrics.histograms[
        'maxserve_batch_execution_time_milliseconds{batch_type="TG"}'
    ]
    assert decode.count == 100.0
    assert decode.sum == 2000.0
    assert decode.mean == 20.0


def test_get_histogram_helper() -> None:
    """Test ParsedMetrics.get_histogram() helper method."""
    # Create metrics with labeled histograms
    prefill_hist = HistogramData(buckets=[], sum=900.0, count=15.0)
    decode_hist = HistogramData(buckets=[], sum=2000.0, count=100.0)

    metrics = ParsedMetrics(
        counters={},
        gauges={},
        histograms={
            'maxserve_batch_execution_time_milliseconds{batch_type="CE"}': prefill_hist,
            'maxserve_batch_execution_time_milliseconds{batch_type="TG"}': decode_hist,
        },
        raw_text="",
    )

    # Test accessing with labels
    result_prefill = metrics.get_histogram(
        "maxserve_batch_execution_time_milliseconds", {"batch_type": "CE"}
    )
    assert result_prefill is not None
    assert result_prefill.mean == 60.0

    result_decode = metrics.get_histogram(
        "maxserve_batch_execution_time_milliseconds", {"batch_type": "TG"}
    )
    assert result_decode is not None
    assert result_decode.mean == 20.0

    # Test accessing non-existent metric returns None
    result_missing = metrics.get_histogram(
        "maxserve_batch_execution_time_milliseconds", {"batch_type": "missing"}
    )
    assert result_missing is None


def test_metrics_by_endpoint_defaults_to_empty() -> None:
    """With no endpoints scraped, convenience batch-time properties return
    safe defaults (None / 0) rather than raising."""
    metrics = _make_metrics({})

    assert dict(metrics.metrics_by_endpoint) == {}
    assert metrics.mean_prefill_batch_time_ms is None
    assert metrics.mean_decode_batch_time_ms is None
    assert metrics.prefill_batch_count == 0
    assert metrics.decode_batch_count == 0


class TestCollectBenchmarkMetrics:
    """Exercises the single collection path: empty urls -> auto-derive one
    endpoint; populated urls -> scrape each label as given."""

    PROM = "# HELP r Total\n# TYPE r counter\nr 1.0\n"

    @patch("max.benchmark.benchmark_shared.server_metrics.fetch_metrics")
    def test_empty_urls_synthesizes_default(
        self, mock_fetch: MagicMock
    ) -> None:
        """Empty mapping -> one endpoint derived from backend + base_url."""
        mock_fetch.return_value = self.PROM
        result = collect_benchmark_metrics(
            urls={}, backend="vllm", base_url="http://10.0.0.1:8000"
        )
        mock_fetch.assert_called_once_with("http://10.0.0.1:8000/metrics")
        assert list(result.keys()) == ["server"]
        assert result["server"].counters["r"] == 1.0

    @patch("max.benchmark.benchmark_shared.server_metrics.fetch_metrics")
    def test_explicit_urls_used_as_is(self, mock_fetch: MagicMock) -> None:
        """Populated mapping -> each label scraped from its exact URL."""
        mock_fetch.return_value = self.PROM
        result = collect_benchmark_metrics(
            urls={"engine": "http://10.0.0.2:9090/metrics"},
            backend="vllm",
            base_url="http://10.0.0.1:8000",
        )
        mock_fetch.assert_called_once_with("http://10.0.0.2:9090/metrics")
        assert list(result.keys()) == ["engine"]

    @patch("max.benchmark.benchmark_shared.server_metrics.fetch_metrics")
    def test_baseline_computes_delta_per_label(
        self, mock_fetch: MagicMock
    ) -> None:
        """Counter growing from baseline -> final must produce the delta, not
        the raw final value. Baseline reads 10.0, final reads 35.0, delta=25.0."""
        urls = {"eng": "http://10.0.0.2:9090/metrics"}
        mock_fetch.return_value = "# TYPE r counter\nr 10.0\n"
        baseline = collect_benchmark_metrics(
            urls=urls, backend="vllm", base_url="http://10.0.0.1:8000"
        )
        assert baseline["eng"].counters["r"] == 10.0

        mock_fetch.return_value = "# TYPE r counter\nr 35.0\n"
        final = collect_benchmark_metrics(
            urls=urls,
            backend="vllm",
            base_url="http://10.0.0.1:8000",
            baseline=baseline,
        )
        assert final["eng"].counters["r"] == 25.0

    @patch("max.benchmark.benchmark_shared.server_metrics.fetch_metrics")
    def test_partial_failure_does_not_drop_ok_endpoints(
        self, mock_fetch: MagicMock
    ) -> None:
        """Exception on one endpoint's fetch is logged + skipped; the other
        endpoints' results are still returned. Keyed by URL so the test isn't
        coupled to dict iteration order."""
        responses: dict[str, str | Exception] = {
            "http://ok:8001/metrics": "# TYPE r counter\nr 7.0\n",
            "http://bad:8001/metrics": ConnectionError("refused"),
        }

        def fake_fetch(url: str) -> str:
            value = responses[url]
            if isinstance(value, Exception):
                raise value
            return value

        mock_fetch.side_effect = fake_fetch
        result = collect_benchmark_metrics(
            urls={
                "ok": "http://ok:8001/metrics",
                "bad": "http://bad:8001/metrics",
            },
            backend="vllm",
            base_url="http://10.0.0.1:8000",
        )
        assert set(result.keys()) == {"ok"}
        assert result["ok"].counters["r"] == 7.0


def test_batch_histograms_found_on_non_first_endpoint() -> None:
    """`mean_prefill_batch_time_ms` scans across endpoints — it must find the
    engine's MAX-serve histogram even when the orchestrator (no such
    histogram) is listed first in the mapping."""
    orchestrator = ParsedMetrics(
        counters={"requests": 42.0}, gauges={}, histograms={}, raw_text=""
    )
    engine_prefill = HistogramData(
        buckets=[("100", 10.0)], sum=500.0, count=10.0
    )
    engine_decode = HistogramData(
        buckets=[("100", 80.0)], sum=2000.0, count=100.0
    )
    engine = ParsedMetrics(
        counters={},
        gauges={},
        histograms={
            'maxserve_batch_execution_time_milliseconds{batch_type="CE"}': engine_prefill,
            'maxserve_batch_execution_time_milliseconds{batch_type="TG"}': engine_decode,
        },
        raw_text="",
    )
    metrics = _make_metrics({"orchestrator": orchestrator, "engine-0": engine})

    assert metrics.mean_prefill_batch_time_ms == 50.0
    assert metrics.mean_decode_batch_time_ms == 20.0
    assert metrics.prefill_batch_count == 10
    assert metrics.decode_batch_count == 100


class TestMetricsResultOutput:
    """E2E: Prometheus response -> collect_benchmark_metrics -> to_result_dict JSON."""

    @patch("max.benchmark.benchmark_shared.server_metrics.fetch_metrics")
    def test_single_endpoint_emits_both_keys_with_matching_content(
        self, mock_fetch: MagicMock, sample_metrics: str
    ) -> None:
        """Auto-derive (urls={}) produces one endpoint labeled `server`.
        Both `server_metrics` and `server_metrics_by_endpoint.server` appear
        in the output JSON and carry the same counter/gauge/histogram data."""
        mock_fetch.return_value = sample_metrics
        final = collect_benchmark_metrics(
            urls={}, backend="vllm", base_url="http://10.0.0.1:8000"
        )
        result = _make_metrics(final).to_result_dict()

        sm = result["server_metrics"]
        by_ep = result["server_metrics_by_endpoint"]
        assert isinstance(sm, dict)
        assert isinstance(by_ep, dict)
        assert set(by_ep.keys()) == {"server"}

        assert isinstance(by_ep["server"], dict)
        assert sm["counters"] == by_ep["server"]["counters"]
        assert sm["histograms"] == by_ep["server"]["histograms"]
        assert set(sm.keys()) >= {
            "prefill_batch_execution_time_ms",
            "decode_batch_count",
        }

    @patch("max.benchmark.benchmark_shared.server_metrics.fetch_metrics")
    def test_multi_endpoint_attributes_per_label_and_mirrors_first(
        self, mock_fetch: MagicMock
    ) -> None:
        """Orchestrator and engine expose different metric families.
        Output must attribute each to its label, and `server_metrics` must
        mirror the FIRST endpoint (orchestrator), not the engine."""
        orch_prom = (
            "# TYPE orchestrator_requests counter\norchestrator_requests 42.0\n"
        )
        engine_prom = (
            "# TYPE engine_generation_tokens counter\n"
            "engine_generation_tokens 1000.0\n"
        )
        responses = {
            "http://orch:8001/metrics": orch_prom,
            "http://engine:8001/metrics": engine_prom,
        }
        mock_fetch.side_effect = lambda url: responses[url]

        final = collect_benchmark_metrics(
            urls={
                "orch": "http://orch:8001/metrics",
                "engine-0": "http://engine:8001/metrics",
            },
            backend="vllm",
            base_url="http://10.0.0.1:8000",
        )
        result = _make_metrics(final).to_result_dict()

        by_ep = result["server_metrics_by_endpoint"]
        assert isinstance(by_ep, dict)
        assert set(by_ep.keys()) == {"orch", "engine-0"}
        assert isinstance(by_ep["orch"], dict)
        assert isinstance(by_ep["engine-0"], dict)
        assert by_ep["orch"]["counters"] == {"orchestrator_requests": 42.0}
        assert by_ep["engine-0"]["counters"] == {
            "engine_generation_tokens": 1000.0
        }

        # server_metrics mirrors the FIRST inserted endpoint (orch), which is
        # how existing BigQuery/analysis consumers keep working.
        sm = result["server_metrics"]
        assert isinstance(sm, dict)
        assert sm["counters"] == by_ep["orch"]["counters"]
        assert sm["counters"] != by_ep["engine-0"]["counters"]


# ---------------------------------------------------------------------------
# Spec decode metrics parsing and statistics
# ---------------------------------------------------------------------------


def test_parse_spec_decode_metrics_matches_vllm_format() -> None:
    """Spec decode counters are parsed from vLLM Prometheus text."""
    metrics_text = """# HELP vllm:spec_decode_num_drafts Number of spec decoding drafts.
# TYPE vllm:spec_decode_num_drafts counter
vllm:spec_decode_num_drafts 12
# HELP vllm:spec_decode_num_draft_tokens Number of draft tokens.
# TYPE vllm:spec_decode_num_draft_tokens counter
vllm:spec_decode_num_draft_tokens 40
# HELP vllm:spec_decode_num_accepted_tokens Number of accepted tokens.
# TYPE vllm:spec_decode_num_accepted_tokens counter
vllm:spec_decode_num_accepted_tokens 21
# HELP vllm:spec_decode_num_accepted_tokens_per_pos Accepted tokens per position.
# TYPE vllm:spec_decode_num_accepted_tokens_per_pos counter
vllm:spec_decode_num_accepted_tokens_per_pos{position="0"} 12
vllm:spec_decode_num_accepted_tokens_per_pos{position="1"} 7
vllm:spec_decode_num_accepted_tokens_per_pos{position="2"} 2
"""

    parsed = parse_spec_decode_metrics(metrics_text)

    assert parsed is not None
    assert parsed.num_drafts == 12
    assert parsed.num_draft_tokens == 40
    assert parsed.num_accepted_tokens == 21
    assert parsed.accepted_per_pos == {0: 12, 1: 7, 2: 2}


def test_parse_spec_decode_metrics_returns_none_when_absent() -> None:
    """Metrics parsing returns None when no spec decode counters exist."""
    parsed = parse_spec_decode_metrics(
        "# HELP requests Total requests\n# TYPE requests counter\nrequests 10\n"
    )

    assert parsed is None


def test_parse_spec_decode_metrics_handles_maxserve_histogram() -> None:
    """MAX Serve's per-position acceptance histogram is parsed into running sums/counts."""
    metrics_text = """# HELP maxserve_spec_decode_acceptance_rate_per_position Per-position acceptance.
# TYPE maxserve_spec_decode_acceptance_rate_per_position histogram
maxserve_spec_decode_acceptance_rate_per_position_sum{position="0"} 8400.0
maxserve_spec_decode_acceptance_rate_per_position_count{position="0"} 100
maxserve_spec_decode_acceptance_rate_per_position_sum{position="1"} 5000.0
maxserve_spec_decode_acceptance_rate_per_position_count{position="1"} 100
"""

    parsed = parse_spec_decode_metrics(metrics_text)

    assert parsed is not None
    assert parsed.num_drafts == 0
    assert parsed.num_draft_tokens == 0
    assert parsed.per_pos_rate_sum == {0: 8400.0, 1: 5000.0}
    assert parsed.per_pos_rate_count == {0: 100, 1: 100}


def test_parse_spec_decode_metrics_handles_maxserve_avg_acceptance_length() -> (
    None
):
    """MAX Serve's avg acceptance length histogram is parsed for bench deltas."""
    metrics_text = """# HELP maxserve_spec_decode_avg_acceptance_length Avg len.
# TYPE maxserve_spec_decode_avg_acceptance_length histogram
maxserve_spec_decode_avg_acceptance_length_sum 25.0
maxserve_spec_decode_avg_acceptance_length_count 5
"""

    parsed = parse_spec_decode_metrics(metrics_text)

    assert parsed is not None
    assert parsed.avg_acceptance_length_sum == 25.0
    assert parsed.avg_acceptance_length_count == 5.0


def test_calculate_spec_decode_stats_from_maxserve_avg_length_histogram_only() -> (
    None
):
    """Acceptance length is derived from the avg-acceptance-length histogram delta."""
    before = SpecDecodeMetrics(
        avg_acceptance_length_sum=10.0, avg_acceptance_length_count=2.0
    )
    after = SpecDecodeMetrics(
        avg_acceptance_length_sum=30.0, avg_acceptance_length_count=5.0
    )

    stats = calculate_spec_decode_stats(before, after)

    assert stats is not None
    assert stats.acceptance_rate is None
    assert stats.acceptance_length == pytest.approx((30.0 - 10.0) / (5.0 - 2.0))


def test_calculate_spec_decode_stats_matches_vllm_math() -> None:
    """Acceptance math uses benchmark-window deltas like vLLM bench serve."""
    before = SpecDecodeMetrics(
        num_drafts=100,
        num_draft_tokens=320,
        num_accepted_tokens=150,
        accepted_per_pos={0: 100, 1: 40, 2: 10},
    )
    after = SpecDecodeMetrics(
        num_drafts=112,
        num_draft_tokens=356,
        num_accepted_tokens=174,
        accepted_per_pos={0: 112, 1: 48, 2: 14},
    )

    stats = calculate_spec_decode_stats(before, after)

    assert stats is not None
    assert stats.num_drafts == 12
    assert stats.draft_tokens == 36
    assert stats.accepted_tokens == 24
    assert stats.acceptance_rate == pytest.approx((24 / 36) * 100)
    assert stats.acceptance_length == pytest.approx(1 + 24 / 12)
    assert stats.per_position_acceptance_rates == pytest.approx(
        [12 / 12, 8 / 12, 4 / 12]
    )


def test_calculate_spec_decode_stats_from_maxserve_histogram_only() -> None:
    """Without aggregate counters, per-position rates surface from histogram deltas."""
    before = SpecDecodeMetrics(
        per_pos_rate_sum={0: 8000.0, 1: 4000.0},
        per_pos_rate_count={0: 100, 1: 100},
    )
    after = SpecDecodeMetrics(
        per_pos_rate_sum={0: 16800.0, 1: 9000.0},
        per_pos_rate_count={0: 200, 1: 200},
    )

    stats = calculate_spec_decode_stats(before, after)

    assert stats is not None
    # Window per-position acceptance: (8800/100)% / 100 = 0.88; (5000/100)% / 100 = 0.50
    assert stats.per_position_acceptance_rates == pytest.approx([0.88, 0.50])
    assert stats.num_drafts is None
    assert stats.draft_tokens is None
    assert stats.accepted_tokens is None
    assert stats.acceptance_rate == pytest.approx(69.0)
    assert stats.acceptance_length is None


def test_spec_decode_stats_to_result_dict_uses_vllm_json_keys() -> None:
    """Spec decode stats are serialized under vLLM-compatible keys."""
    stats = SpecDecodeStats(
        num_drafts=5,
        draft_tokens=18,
        accepted_tokens=9,
        acceptance_rate=50.0,
        acceptance_length=2.8,
        per_position_acceptance_rates=[1.0, 0.6, 0.2],
    )

    assert stats.to_result_dict() == {
        "spec_decode_acceptance_rate": 50.0,
        "spec_decode_acceptance_length": 2.8,
        "spec_decode_num_drafts": 5,
        "spec_decode_draft_tokens": 18,
        "spec_decode_accepted_tokens": 9,
        "spec_decode_per_position_acceptance_rates": [1.0, 0.6, 0.2],
    }


def test_spec_decode_stats_to_result_dict_omits_missing_aggregates() -> None:
    """JSON result only includes fields the backend actually exposed."""
    stats = SpecDecodeStats(per_position_acceptance_rates=[0.88, 0.50])

    assert stats.to_result_dict() == {
        "spec_decode_per_position_acceptance_rates": [0.88, 0.50],
    }


def test_spec_decode_metrics_iadd() -> None:
    """SpecDecodeMetrics.__iadd__ merges all fields by summation."""
    a = SpecDecodeMetrics(
        num_drafts=10,
        num_draft_tokens=30,
        num_accepted_tokens=20,
        accepted_per_pos={0: 10, 1: 7},
        per_pos_rate_sum={0: 800.0, 1: 500.0},
        per_pos_rate_count={0: 10, 1: 10},
        avg_acceptance_length_sum=5.0,
        avg_acceptance_length_count=2.0,
    )
    b = SpecDecodeMetrics(
        num_drafts=5,
        num_draft_tokens=15,
        num_accepted_tokens=11,
        accepted_per_pos={0: 5, 2: 3},
        per_pos_rate_sum={0: 400.0, 2: 300.0},
        per_pos_rate_count={0: 5, 2: 5},
        avg_acceptance_length_sum=3.0,
        avg_acceptance_length_count=1.0,
    )

    a += b

    assert a.num_drafts == 15
    assert a.num_draft_tokens == 45
    assert a.num_accepted_tokens == 31
    assert a.accepted_per_pos == {0: 15, 1: 7, 2: 3}
    assert a.per_pos_rate_sum == {0: 1200.0, 1: 500.0, 2: 300.0}
    assert a.per_pos_rate_count == {0: 15, 1: 10, 2: 5}
    assert a.avg_acceptance_length_sum == pytest.approx(8.0)
    assert a.avg_acceptance_length_count == pytest.approx(3.0)


def _hist(sum_: float, count: float) -> HistogramData:
    """HistogramData with no buckets (mean is derived from sum/count)."""
    return HistogramData(buckets=[], sum=sum_, count=count)


def _endpoint_with_batch_histograms() -> ParsedMetrics:
    """Endpoint exposing all five CE/TG batch histograms the validator reads.

    Distinct sum/count per (metric, batch_type) so attribution of CE->prefill
    and TG->decode is unambiguous.
    """
    return ParsedMetrics(
        counters={},
        gauges={},
        histograms={
            'maxserve_batch_context_tokens{batch_type="CE"}': _hist(300.0, 3.0),
            'maxserve_batch_context_tokens{batch_type="TG"}': _hist(20.0, 2.0),
            'maxserve_batch_creation_time_milliseconds{batch_type="CE"}': _hist(
                50.0, 5.0
            ),
            'maxserve_batch_creation_time_milliseconds{batch_type="TG"}': _hist(
                80.0, 4.0
            ),
            'maxserve_batch_prompt_throughput_tokens_per_second{batch_type="CE"}': _hist(
                4000.0, 2.0
            ),
            'maxserve_batch_prompt_throughput_tokens_per_second{batch_type="TG"}': _hist(
                300.0, 3.0
            ),
            'maxserve_batch_input_tokens{batch_type="CE"}': _hist(600.0, 3.0),
            'maxserve_batch_input_tokens{batch_type="TG"}': _hist(40.0, 2.0),
            'maxserve_batch_generation_throughput_tokens_per_second{batch_type="CE"}': _hist(
                150.0, 3.0
            ),
            'maxserve_batch_generation_throughput_tokens_per_second{batch_type="TG"}': _hist(
                900.0, 3.0
            ),
        },
        raw_text="",
    )


def test_prefill_decode_stats_to_result_dict_populates_all_fields() -> None:
    """Every histogram present -> mean/count/sum derived for all five metrics."""
    stats = PrefillDecodeStats(
        context_tokens=_hist(300.0, 3.0),
        creation_time_milliseconds=_hist(50.0, 5.0),
        prompt_throughput_tokens_per_second=_hist(4000.0, 2.0),
        input_tokens=_hist(600.0, 3.0),
        generation_throughput_tokens_per_second=_hist(150.0, 3.0),
    )

    d = stats.to_result_dict()
    assert d["maxserve_batch_context_tokens_mean"] == 100.0
    assert d["maxserve_batch_context_tokens_count"] == 3.0
    assert d["maxserve_batch_context_tokens_sum"] == 300.0
    assert d["maxserve_batch_creation_time_milliseconds_mean"] == 10.0
    assert (
        d["maxserve_batch_prompt_throughput_tokens_per_second_mean"] == 2000.0
    )
    assert d["maxserve_batch_input_tokens_mean"] == 200.0
    assert d["maxserve_batch_input_tokens_sum"] == 600.0
    assert (
        d["maxserve_batch_generation_throughput_tokens_per_second_mean"] == 50.0
    )


def test_prefill_decode_stats_all_none_is_all_none() -> None:
    """Regression: a missing batch type must not crash on ``.mean`` access.

    Every histogram absent -> every derived field None (the validator leaves a
    histogram unset when an endpoint never emitted that CE/TG metric)."""
    stats = PrefillDecodeStats()

    assert stats == PrefillDecodeStats()
    assert all(value is None for value in stats.to_result_dict().values())


def test_prefill_decode_stats_partial_presence() -> None:
    """Only the histograms that exist are derived; the rest stay None."""
    stats = PrefillDecodeStats(
        context_tokens=_hist(300.0, 3.0),
        input_tokens=_hist(600.0, 3.0),
    )

    d = stats.to_result_dict()
    assert d["maxserve_batch_context_tokens_mean"] == 100.0
    assert d["maxserve_batch_input_tokens_mean"] == 200.0
    assert d["maxserve_batch_creation_time_milliseconds_mean"] is None
    assert d["maxserve_batch_prompt_throughput_tokens_per_second_mean"] is None
    assert (
        d["maxserve_batch_generation_throughput_tokens_per_second_mean"] is None
    )


def test_prefill_decode_stats_to_result_dict_keys_and_values() -> None:
    """``to_result_dict`` emits the flat 15-key layout consumers expect."""
    stats = PrefillDecodeStats(
        context_tokens=_hist(300.0, 3.0),
    )

    d = stats.to_result_dict()
    assert d["maxserve_batch_context_tokens_mean"] == 100.0
    assert d["maxserve_batch_context_tokens_count"] == 3.0
    assert d["maxserve_batch_context_tokens_sum"] == 300.0
    assert set(d.keys()) == {
        "maxserve_batch_context_tokens_mean",
        "maxserve_batch_context_tokens_count",
        "maxserve_batch_context_tokens_sum",
        "maxserve_batch_creation_time_milliseconds_mean",
        "maxserve_batch_creation_time_milliseconds_count",
        "maxserve_batch_creation_time_milliseconds_sum",
        "maxserve_batch_generation_throughput_tokens_per_second_mean",
        "maxserve_batch_generation_throughput_tokens_per_second_count",
        "maxserve_batch_generation_throughput_tokens_per_second_sum",
        "maxserve_batch_input_tokens_mean",
        "maxserve_batch_input_tokens_count",
        "maxserve_batch_input_tokens_sum",
        "maxserve_batch_prompt_throughput_tokens_per_second_mean",
        "maxserve_batch_prompt_throughput_tokens_per_second_count",
        "maxserve_batch_prompt_throughput_tokens_per_second_sum",
    }


def test_derive_prefill_decode_stats_attributes_ce_to_prefill_tg_to_decode() -> (
    None
):
    """Regression for the HistogramMetric enum-name mismatch that raised
    ``AttributeError`` while building any result with server metrics.

    CE histograms feed ``prefill_stats``; TG histograms feed ``decode_stats``.
    """
    result = _make_metrics({"server": _endpoint_with_batch_histograms()})

    assert result.prefill_stats is not None
    assert result.decode_stats is not None

    prefill = result.prefill_stats.to_result_dict()
    decode = result.decode_stats.to_result_dict()

    assert prefill["maxserve_batch_context_tokens_mean"] == 100.0
    assert prefill["maxserve_batch_input_tokens_mean"] == 200.0
    assert (
        prefill["maxserve_batch_prompt_throughput_tokens_per_second_mean"]
        == 2000.0
    )
    assert (
        prefill["maxserve_batch_generation_throughput_tokens_per_second_mean"]
        == 50.0
    )

    assert decode["maxserve_batch_context_tokens_mean"] == 10.0
    assert decode["maxserve_batch_input_tokens_mean"] == 20.0
    assert (
        decode["maxserve_batch_prompt_throughput_tokens_per_second_mean"]
        == 100.0
    )
    assert (
        decode["maxserve_batch_generation_throughput_tokens_per_second_mean"]
        == 300.0
    )


def test_derive_prefill_decode_stats_scans_past_first_endpoint() -> None:
    """The validator scans all endpoints, so an orchestrator with no batch
    histograms listed first must not shadow the engine's histograms."""
    orchestrator = ParsedMetrics(
        counters={"requests": 1.0}, gauges={}, histograms={}, raw_text=""
    )
    result = _make_metrics(
        {"orch": orchestrator, "engine-0": _endpoint_with_batch_histograms()}
    )

    assert result.prefill_stats is not None
    assert (
        result.prefill_stats.to_result_dict()[
            "maxserve_batch_context_tokens_mean"
        ]
        == 100.0
    )
    assert result.decode_stats is not None
    assert (
        result.decode_stats.to_result_dict()[
            "maxserve_batch_context_tokens_mean"
        ]
        == 10.0
    )


def test_derive_prefill_decode_stats_none_without_endpoints() -> None:
    """No ``metrics_by_endpoint`` -> stats stay None (nothing to derive)."""
    result = _make_metrics({})

    assert result.prefill_stats is None
    assert result.decode_stats is None


def test_to_result_dict_includes_prefill_decode_stats_when_present() -> None:
    """Derived stats surface as nested dicts in ``to_result_dict`` output."""
    result = _make_metrics({"server": _endpoint_with_batch_histograms()})

    d = result.to_result_dict()
    assert isinstance(d["prefill_stats"], dict)
    assert isinstance(d["decode_stats"], dict)
    assert d["prefill_stats"]["maxserve_batch_context_tokens_mean"] == 100.0
    assert d["decode_stats"]["maxserve_batch_context_tokens_mean"] == 10.0


def test_to_result_dict_omits_prefill_decode_stats_when_absent() -> None:
    """Without server metrics the keys are omitted entirely (not None)."""
    result = _make_metrics({})

    d = result.to_result_dict()
    assert "prefill_stats" not in d
    assert "decode_stats" not in d
