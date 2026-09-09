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
import itertools
import pickle
from unittest import mock

import pytest
from max.serve.config import Settings
from max.serve.telemetry import common, metrics
from opentelemetry.metrics import get_meter_provider
from opentelemetry.metrics._internal.instrument import (
    _ProxyGauge,
    _ProxyHistogram,
    _ProxyInstrument,
)
from opentelemetry.sdk.metrics._internal.aggregation import (
    ExplicitBucketHistogramAggregation,
    ExponentialBucketHistogramAggregation,
)
from opentelemetry.sdk.metrics._internal.point import (
    Buckets,
    ExponentialHistogram,
    ExponentialHistogramDataPoint,
    HistogramDataPoint,
    Metric,
    MetricsData,
    NumberDataPoint,
    ResourceMetrics,
    ScopeMetrics,
    Sum,
)
from opentelemetry.sdk.metrics._internal.point import (
    Histogram as ExplicitHistogramData,
)
from opentelemetry.sdk.metrics.export import AggregationTemporality
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.util.instrumentation import InstrumentationScope

_meter = get_meter_provider().get_meter("testing")


def test_correct_metric_names() -> None:
    for name, inst in metrics.SERVE_METRICS.items():
        if isinstance(inst, _ProxyInstrument):
            assert name == inst._name
        else:
            assert name == inst.name


def test_max_measurement() -> None:
    m = metrics.MaxMeasurement("maxserve.itl", 1)
    m.commit()


def test_time_per_output_token_measurement() -> None:
    common.configure_metrics(Settings())
    assert "maxserve.time_per_output_token" in metrics.SERVE_METRICS
    m = metrics.MaxMeasurement("maxserve.time_per_output_token", 1.5)
    m.commit()  # Should not raise


def test_serialization() -> None:
    measurements = [
        metrics.MaxMeasurement("maxserve.itl", 1),
        metrics.MaxMeasurement("maxserve.itl", -3.4),
        metrics.MaxMeasurement("maxserve.itl", 1, attributes={"att1": "val"}),
    ]
    for m in measurements:
        b = pickle.dumps(m)
        m2 = pickle.loads(b)

        assert m.instrument_name == m2.instrument_name
        assert m.value == m2.value
        assert m.attributes == m2.attributes
        assert m.time_unix_nano == m2.time_unix_nano


def test_reject_unknown_metric() -> None:
    m = metrics.MaxMeasurement("bogus", 1)
    with pytest.raises(metrics.UnknownMetric):
        m.commit()


def test_instrument_called() -> None:
    common.configure_metrics(Settings())
    itl = metrics.SERVE_METRICS["maxserve.itl"]
    assert isinstance(itl, _ProxyInstrument)
    assert itl._real_instrument is not None
    with mock.patch.object(
        itl._real_instrument, "_measurement_consumer"
    ) as mock_consumer:
        # make _real_instrument None & verify that the measurement does _not_ get consumed
        with mock.patch.object(itl, "_real_instrument", None):
            metrics.MaxMeasurement("maxserve.itl", 1).commit()
            assert mock_consumer.consume_measurement.call_count == 0

        # put things back together and verify that it does get consumed
        metrics.MaxMeasurement("maxserve.itl", 1).commit()
        # make sure the consumer got called
        assert mock_consumer.consume_measurement.call_count == 1


def test_model_load_time_with_component_attribute() -> None:
    """Pins down the ``component`` tag on the model_load_time histogram.

    The model worker records the per-phase startup breakdown (build, compile,
    init, ...) on the same histogram as the untagged model-load aggregate,
    split by the ``component`` tag.
    """
    common.configure_metrics(Settings())
    assert "maxserve.model_load_time" in metrics.SERVE_METRICS

    # Untagged aggregate.
    metrics.MaxMeasurement("maxserve.model_load_time", 1234.5).commit()

    # Per-phase records, tagged by component.
    for component in ("build", "compile", "total"):
        metrics.MaxMeasurement(
            "maxserve.model_load_time",
            100.0,
            attributes={"component": component},
        ).commit()  # Should not raise


def test_batch_execution_time_with_attributes() -> None:
    """Test that batch_execution_time metric works with batch_type attribute."""
    common.configure_metrics(Settings())

    # Test with CE (prefill) batch type
    m_ce = metrics.MaxMeasurement(
        "maxserve.batch_execution_time", 100.5, attributes={"batch_type": "CE"}
    )
    m_ce.commit()  # Should not raise

    # Test with TG (decode) batch type
    m_tg = metrics.MaxMeasurement(
        "maxserve.batch_execution_time", 50.2, attributes={"batch_type": "TG"}
    )
    m_tg.commit()  # Should not raise


def test_tokens_per_request_histograms() -> None:
    """Test that per-request token histogram metrics can be recorded."""
    common.configure_metrics(Settings())

    # Verify metrics exist in SERVE_METRICS
    assert "maxserve.input_tokens_per_request" in metrics.SERVE_METRICS
    assert "maxserve.output_tokens_per_request" in metrics.SERVE_METRICS

    # Test recording input tokens per request
    m_input = metrics.MaxMeasurement("maxserve.input_tokens_per_request", 256)
    m_input.commit()  # Should not raise

    # Test recording output tokens per request
    m_output = metrics.MaxMeasurement("maxserve.output_tokens_per_request", 128)
    m_output.commit()  # Should not raise


def _is_histogram(inst: object) -> bool:
    """True if a SERVE_METRICS instrument is a Histogram.

    Entries in SERVE_METRICS are created at import time (before any real meter
    provider is configured), so they are always proxy instruments.
    """
    return isinstance(inst, _ProxyHistogram)


def test_all_histograms_have_explicit_buckets() -> None:
    """Every primary histogram must have tuned bucket boundaries.

    The default latency-ms buckets are wrong for non-latency histograms
    (percentages, token counts, throughput, ...), so common.py assigns
    per-metric buckets by exact instrument name. Guard against a new histogram
    being added without a matching bucket View (which would silently fall back
    to the SDK default buckets), and against stale map entries. Excludes the
    exponential-histogram shadow metrics (metrics.HISTOGRAM_SHADOW_SUFFIX),
    which are deliberately not in this map -- they're covered by the glob
    View in _histogram_views instead.
    """
    histogram_names = {
        name
        for name, inst in metrics.SERVE_METRICS.items()
        if _is_histogram(inst)
        and not name.endswith(metrics.HISTOGRAM_SHADOW_SUFFIX)
    }
    mapped = set(common.HISTOGRAM_BUCKETS_BY_METRIC)

    missing = histogram_names - mapped
    assert not missing, (
        f"Histograms missing explicit bucket Views: {sorted(missing)}"
    )

    stale = mapped - histogram_names
    assert not stale, (
        f"HISTOGRAM_BUCKETS_BY_METRIC references non-histograms: {sorted(stale)}"
    )


def test_histogram_buckets_are_strictly_increasing() -> None:
    """OTEL rejects unsorted boundaries, and duplicates waste Prometheus series.

    The log-spaced ladders round to three significant digits, so this also
    guards against a future range/ratio change rounding two boundaries together.
    """
    for name, buckets in common.HISTOGRAM_BUCKETS_BY_METRIC.items():
        assert all(lo < hi for lo, hi in itertools.pairwise(buckets)), (
            f"{name} boundaries are not strictly increasing: {buckets}"
        )


def test_log_spaced_bucket_ranges() -> None:
    """Pins the endpoints and step ratio of each log-spaced ladder.

    Latency reaches 8 hours so slow compiles and long agentic requests land in a
    real bucket rather than the overflow bucket, and generic counts reach 100M
    for batch-level token totals over very long contexts. A coarser step ratio
    than these would make quantile interpolation too imprecise to act on.
    """
    for buckets, first, last, max_ratio in (
        (common.HISTOGRAM_LATENCY_BUCKETS_MS, 1.0, 8 * 60 * 60 * 1000.0, 1.5),
        (common.HISTOGRAM_COUNT_BUCKETS, 1.0, 100_000_000.0, 1.5),
        (common.HISTOGRAM_THROUGHPUT_TOKENS_BUCKETS, 10.0, 1_000_000.0, 1.5),
        (common.HISTOGRAM_BATCH_SIZE_BUCKETS, 1.0, 512.0, 1.5),
        (common.HISTOGRAM_GIB_PER_S_BUCKETS, 0.1, 800.0, 2.0),
    ):
        # The leading 0 keeps exact zeros out of the smallest real bucket.
        assert buckets[0] == 0.0
        assert buckets[1] == first
        assert buckets[-1] == last
        # Checked on the ladder as a whole, since rounding boundaries to three
        # significant digits nudges individual steps either side of the ratio.
        ratio = (buckets[-1] / buckets[1]) ** (1 / (len(buckets) - 2))
        assert ratio <= max_ratio


def test_percent_buckets_tighten_near_saturation() -> None:
    """Percent buckets step linearly by 5, and by 1 above 90.

    Utilization (KV cache usage, occupancy, hit rates) is only actionable near
    saturation, so a geometric ladder would put the resolution at the wrong end
    and leave 85% and 99% sharing a bucket. The count stays modest because
    several of these metrics carry a label (batch_type, draft position) that
    multiplies every boundary into more Prometheus series.
    """
    buckets = common.HISTOGRAM_PERCENT_BUCKETS
    assert (buckets[0], buckets[-1]) == (0.0, 100.0)
    assert len(buckets) == 29
    pairs = list(itertools.pairwise(buckets))
    assert {hi - lo for lo, hi in pairs if hi <= 90.0} == {5.0}
    assert {hi - lo for lo, hi in pairs if hi > 90.0} == {1.0}


def test_every_histogram_has_an_exponential_shadow() -> None:
    """Every primary histogram gets a same-named exponential shadow metric.

    Created unconditionally at import time (see metrics.py) so
    configure_histogram_shadow_emission can be toggled purely at commit
    time, without touching instrument setup.
    """
    for name in common.HISTOGRAM_BUCKETS_BY_METRIC:
        shadow_name = name + metrics.HISTOGRAM_SHADOW_SUFFIX
        assert shadow_name in metrics.SERVE_METRICS, (
            f"{name} has no exponential shadow metric ({shadow_name})"
        )
        assert _is_histogram(metrics.SERVE_METRICS[shadow_name])


def test_histogram_views_default_uses_explicit_buckets_only() -> None:
    """With no OTLP metrics endpoint configured, only Prometheus's buckets.

    Verifies the default (flag-off) path stays on today's per-metric,
    hand-tuned ``ExplicitBucketHistogramAggregation`` Views, matched by
    exact instrument name -- and that no exponential View is added, since
    the shadow metrics are never committed to in this mode.
    """
    views = common._histogram_views(Settings())

    assert len(views) == len(common.HISTOGRAM_BUCKETS_BY_METRIC)
    for view in views:
        assert view._instrument_name in common.HISTOGRAM_BUCKETS_BY_METRIC
        assert isinstance(view._aggregation, ExplicitBucketHistogramAggregation)


def test_histogram_views_with_otlp_endpoint_adds_exponential_shadow() -> None:
    """Setting an OTLP metrics endpoint adds an exponential shadow View.

    The explicit-bucket Views (Prometheus's) stay exactly as in the default
    case -- this is additive, not a replacement -- plus one more View,
    matched by a name glob rather than per-metric, covering every shadow
    histogram at once with a self-calibrating aggregation (see MXSERV-258).
    """
    views = common._histogram_views(
        Settings(otlp_metrics_endpoint="http://localhost:4318")
    )

    explicit_views = [
        v
        for v in views
        if isinstance(v._aggregation, ExplicitBucketHistogramAggregation)
    ]
    exponential_views = [
        v
        for v in views
        if isinstance(v._aggregation, ExponentialBucketHistogramAggregation)
    ]

    assert len(explicit_views) == len(common.HISTOGRAM_BUCKETS_BY_METRIC)
    assert len(exponential_views) == 1
    assert exponential_views[0]._instrument_name == (
        "*" + metrics.HISTOGRAM_SHADOW_SUFFIX
    )


def test_histogram_shadow_emission_mirrors_commits() -> None:
    """configure_histogram_shadow_emission gates the shadow commit.

    Disabled (the default): committing a primary histogram measurement
    doesn't touch its shadow. Enabled: it does, with the same value.
    """
    metrics.configure_histogram_shadow_emission(False)
    try:
        with mock.patch(
            "max.serve.telemetry.metrics._commit_measurement"
        ) as mock_commit:
            metrics.MaxMeasurement("maxserve.itl", 12.5).commit()
            assert mock_commit.call_count == 1
            assert mock_commit.call_args[0][0] == "maxserve.itl"

        metrics.configure_histogram_shadow_emission(True)
        with mock.patch(
            "max.serve.telemetry.metrics._commit_measurement"
        ) as mock_commit:
            metrics.MaxMeasurement("maxserve.itl", 12.5).commit()
            assert mock_commit.call_count == 2
            committed_names = {
                call.args[0] for call in mock_commit.call_args_list
            }
            assert committed_names == {
                "maxserve.itl",
                "maxserve.itl" + metrics.HISTOGRAM_SHADOW_SUFFIX,
            }
    finally:
        metrics.configure_histogram_shadow_emission(False)


def _synthetic_metrics_data(
    *data: Sum | ExplicitHistogramData | ExponentialHistogram,
) -> MetricsData:
    """Builds a minimal MetricsData tree wrapping the given data payloads.

    Exercises common._drop_metrics_matching (shared by both reader guards)
    directly, without needing a real MeterProvider/exporter -- these are
    frozen dataclasses, cheap to construct by hand.
    """
    return MetricsData(
        resource_metrics=[
            ResourceMetrics(
                resource=Resource.create({}),
                schema_url="",
                scope_metrics=[
                    ScopeMetrics(
                        scope=InstrumentationScope("test"),
                        schema_url="",
                        metrics=[
                            Metric(
                                name=f"metric_{i}",
                                description=None,
                                unit=None,
                                data=payload,
                            )
                            for i, payload in enumerate(data)
                        ],
                    )
                ],
            )
        ]
    )


def _sum_payload() -> Sum:
    return Sum(
        data_points=[
            NumberDataPoint(
                attributes={},
                start_time_unix_nano=0,
                time_unix_nano=1,
                value=1,
            )
        ],
        aggregation_temporality=AggregationTemporality.CUMULATIVE,
        is_monotonic=True,
    )


def _explicit_histogram_payload() -> ExplicitHistogramData:
    return ExplicitHistogramData(
        data_points=[
            HistogramDataPoint(
                attributes={},
                start_time_unix_nano=0,
                time_unix_nano=1,
                count=1,
                sum=1.0,
                bucket_counts=[1, 0],
                explicit_bounds=[1.0],
                min=1.0,
                max=1.0,
            )
        ],
        aggregation_temporality=AggregationTemporality.CUMULATIVE,
    )


def _exponential_histogram_payload() -> ExponentialHistogram:
    return ExponentialHistogram(
        data_points=[
            ExponentialHistogramDataPoint(
                attributes={},
                start_time_unix_nano=0,
                time_unix_nano=1,
                count=1,
                sum=1.0,
                scale=1,
                zero_count=0,
                positive=Buckets(offset=0, bucket_counts=[1]),
                negative=Buckets(offset=0, bucket_counts=[]),
                flags=0,
                min=1.0,
                max=1.0,
            )
        ],
        aggregation_temporality=AggregationTemporality.CUMULATIVE,
    )


def _metric_data_types(filtered: MetricsData | None) -> list[type]:
    assert filtered is not None
    return [
        type(m.data)
        for rm in filtered.resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics
    ]


def test_drop_metrics_matching_keeps_and_drops_by_data_type() -> None:
    """common._drop_metrics_matching is the shared primitive behind both
    reader guards: _SkipExponentialHistogramsPrometheusReader filters out
    ExponentialHistogram data (so Prometheus never sees a shape it can't
    render), and _ExponentialShadowOnlyReader filters out the classic
    Histogram data (so the OTLP endpoint carries only the shadow metric,
    per MXSERV-258). Exercised directly against a synthetic MetricsData
    tree containing all three payload shapes at once.
    """
    data = _synthetic_metrics_data(
        _sum_payload(),
        _explicit_histogram_payload(),
        _exponential_histogram_payload(),
    )

    # Mirrors _SkipExponentialHistogramsPrometheusReader's filter.
    prometheus_bound = common._drop_metrics_matching(
        data, lambda m: isinstance(m.data, ExponentialHistogram)
    )
    assert set(_metric_data_types(prometheus_bound)) == {
        Sum,
        ExplicitHistogramData,
    }

    # Mirrors _ExponentialShadowOnlyReader's filter.
    otlp_bound = common._drop_metrics_matching(
        data, lambda m: isinstance(m.data, ExplicitHistogramData)
    )
    assert set(_metric_data_types(otlp_bound)) == {
        Sum,
        ExponentialHistogram,
    }

    # Neither guard drops everything, and a None input passes through as
    # None (both readers' _receive_metrics skip the export call in that
    # case, matching the base SDK's own no-op-on-None convention).
    assert common._drop_metrics_matching(None, lambda m: True) is None


def test_block_level_metrics_are_gauges() -> None:
    """num_used_blocks / num_total_blocks are instantaneous levels (gauges).

    Emitting an absolute level into a counter would sum the levels into a
    meaningless running total, so these must be gauges (LastValue).
    """
    for name in (
        "maxserve.cache.num_used_blocks",
        "maxserve.cache.num_total_blocks",
    ):
        inst = metrics.SERVE_METRICS[name]
        assert isinstance(inst, _ProxyGauge), (
            f"{name} should be a gauge, got {type(inst)}"
        )


def test_disk_byte_counters_record() -> None:
    """The disk-tier block transfer counters exist and record without raising."""
    common.configure_metrics(Settings())
    for name in (
        "maxserve.cache.disk_bytes_read",
        "maxserve.cache.disk_bytes_written",
    ):
        assert name in metrics.SERVE_METRICS
        metrics.MaxMeasurement(name, 7).commit()  # Should not raise


def test_tool_call_conformance_error_counter() -> None:
    """The tool-call conformance counter records with a bounded 'outcome' tag."""
    common.configure_metrics(Settings())
    assert "maxserve.tool_call.conformance_errors" in metrics.SERVE_METRICS
    for outcome in ("invalid_json", "unknown_tool", "schema_mismatch"):
        metrics.METRICS.tool_call_conformance_error(outcome)  # Should not raise


def test_response_format_conformance_error_counter() -> None:
    """The response_format conformance counter records with a bounded 'outcome' tag."""
    common.configure_metrics(Settings())
    assert (
        "maxserve.response_format.conformance_errors" in metrics.SERVE_METRICS
    )
    for outcome in ("invalid_json", "schema_mismatch"):
        metrics.METRICS.response_format_conformance_error(
            outcome
        )  # Should not raise


def test_structured_output_grammar_rejection_counter() -> None:
    """The 400-rejection counter records with a bounded 'kind' tag."""
    common.configure_metrics(Settings())
    assert (
        "maxserve.structured_output.grammar_rejections" in metrics.SERVE_METRICS
    )
    for kind in ("tool_grammar", "json_schema"):
        metrics.METRICS.structured_output_grammar_rejection(
            kind
        )  # Should not raise


def test_batch_metrics_with_batch_type_attribute() -> None:
    """Pins down the ``batch_type`` label on the new graduated batch histograms.
    ``maxserve.batch_prompt_throughput`` is representative of the batch-level
    instruments declared in this PR; the per-instrument plumbing is identical
    so a single test guards the whole class.
    """
    common.configure_metrics(Settings())
    assert "maxserve.batch_prompt_throughput" in metrics.SERVE_METRICS
    metrics.MaxMeasurement(
        "maxserve.batch_prompt_throughput",
        9100.0,
        attributes={"batch_type": "CE"},
    ).commit()
    metrics.MaxMeasurement(
        "maxserve.batch_prompt_throughput",
        4.9,
        attributes={"batch_type": "TG"},
    ).commit()
