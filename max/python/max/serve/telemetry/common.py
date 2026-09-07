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

from __future__ import annotations

import dataclasses
import logging
import logging.handlers
import math
import os
import platform
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from time import time

import numpy as np
import requests
from max.profiler import set_gpu_profiling_state
from max.serve.config import KernelTraceLevel, Settings
from max.serve.telemetry.metrics import (
    HISTOGRAM_SHADOW_SUFFIX,
    configure_histogram_shadow_emission,
)
from opentelemetry._logs import set_logger_provider
from opentelemetry.context import Context as OtelContext
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import Counter, Histogram, MeterProvider
from opentelemetry.sdk.metrics._internal.aggregation import (
    ExplicitBucketHistogramAggregation,
    ExponentialBucketHistogramAggregation,
)
from opentelemetry.sdk.metrics._internal.point import (
    ExponentialHistogram,
    Metric,
)
from opentelemetry.sdk.metrics._internal.point import (
    Histogram as ExplicitHistogramData,
)
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    MetricReader,
    MetricsData,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.metrics.view import View
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import set_tracer_provider
from pythonjsonlogger import jsonlogger

otelBaseUrl = "https://telemetry.modular.com:443"

request_trace_ctx: ContextVar[OtelContext | None] = ContextVar(
    "max.serve.request_trace_ctx", default=None
)
"""The OTel context extracted from the current request's inbound W3C
traceparent/tracestate headers (or None). Set once per request by the route
handler in openai_routes.py before it calls into the pipeline, so it is
already populated by the time TextContext is constructed in llm.py — even
though route handlers and the pipeline live in different modules, ContextVar
values propagate through the whole async call chain within the same task."""


def _getCloudProvider() -> str:
    providers = ["amazon", "google", "microsoft", "oracle"]
    path = "/sys/class/dmi/id/"
    if os.path.isdir(path):
        for idFile in os.listdir(path):
            try:
                with open(idFile) as file:
                    contents = file.read().lower()
                    for provider in providers:
                        if provider in contents:
                            return provider
            except Exception:
                pass
    return ""


def _getWebUserId() -> str:
    try:
        idFile = os.path.expanduser("~") + "/.modular/webUserId"
        with open(idFile) as file:
            return file.readline().rstrip("\n")
    except Exception:
        return ""


logs_resource = Resource.create(
    {
        "event.domain": "serve",
        "telemetry.session": uuid.uuid4().hex,
        "web.user.id": _getWebUserId(),
        "enduser.id": os.environ.get("MODULAR_USER_ID", ""),
        "os.type": platform.system(),
        "os.version": platform.release(),
        "cpu.description": platform.processor(),
        "cpu.arch": platform.architecture()[0],
        "system.cloud": _getCloudProvider(),
        "deployment.id": os.environ.get("MAX_SERVE_DEPLOYMENT_ID", ""),
    }
)

metrics_resource = Resource.create(
    {
        "enduser.id": os.environ.get("MODULAR_USER_ID", ""),
        "deployment.id": os.environ.get("MAX_SERVE_DEPLOYMENT_ID", ""),
    }
)


def _log_spaced_buckets(
    start: float, stop: float, max_ratio: float
) -> tuple[float, ...]:
    """Builds a 0 boundary plus geometrically spaced boundaries up to ``stop``."""
    steps = math.ceil(math.log(stop / start) / math.log(max_ratio))
    return (0.0,) + tuple(
        float(f"{bound:.3g}")
        for bound in np.exp(
            np.linspace(math.log(start), math.log(stop), steps + 1)
        )
    )


# Latency metrics (milliseconds). Log-spaced from 1ms out to 8 hours, which
# covers slow ops (request time, model load, compile) and long agentic requests.
HISTOGRAM_LATENCY_BUCKETS_MS: tuple[float, ...] = _log_spaced_buckets(
    1.0, 8 * 60 * 60 * 1000.0, 1.5
)

# Percentages / utilization ratios (0-100). Linear rather than geometric, since
# utilization is only actionable near saturation: 5% steps through the bulk,
# tightening to 1% above 90% so a nearly-full KV cache tier resolves to a
# single point instead of a range. Kept to 29 boundaries because several of
# these metrics carry a label that multiplies the series count (batch_type,
# draft position).
HISTOGRAM_PERCENT_BUCKETS: tuple[float, ...] = tuple(
    float(pct) for pct in (*range(0, 91, 5), *range(91, 101))
)

# Token counts per request/batch and other unbounded counts. Log-spaced from 1
# up to 100M, so batch-level token totals over very long contexts stay on scale.
HISTOGRAM_COUNT_BUCKETS: tuple[float, ...] = _log_spaced_buckets(
    1.0, 100_000_000.0, 1.5
)

# Batch size (number of requests in a batch), from a single request up to the
# largest batch the scheduler forms.
HISTOGRAM_BATCH_SIZE_BUCKETS: tuple[float, ...] = _log_spaced_buckets(
    1.0, 512.0, 1.5
)

# Mean speculative-decode acceptance length per batch. Because it is an average
# it takes fractional values, so use fine 0.25-wide buckets from 0 to 16 tokens
# (boundaries 0.25, 0.50, ..., 16.0).
HISTOGRAM_ACCEPTANCE_LENGTH_BUCKETS: tuple[float, ...] = tuple(
    i * 0.25 for i in range(1, 65)
)

# Throughput in tokens/second, from a slow single request to a large prefill
# batch.
HISTOGRAM_THROUGHPUT_TOKENS_BUCKETS: tuple[float, ...] = _log_spaced_buckets(
    10.0, 1_000_000.0, 1.5
)

# Transfer throughput in GiB/s (e.g. NIXL over fast fabric). Coarser steps than
# the other ladders: this spans four decades and only feeds transfer dashboards.
HISTOGRAM_GIB_PER_S_BUCKETS: tuple[float, ...] = _log_spaced_buckets(
    0.1, 800.0, 2.0
)

# Per-metric histogram bucket boundaries, matched by exact instrument name.
#
# Each Histogram instrument is matched to exactly one View by its exact name, so
# there is no risk of an instrument matching multiple Views (which would emit
# duplicate Prometheus series). Any histogram not listed here falls back to the
# OTEL SDK default buckets, so keep this in sync with the Histogram instruments
# in metrics.py SERVE_METRICS. A unit test
# (test_all_histograms_have_explicit_buckets) enforces that.
HISTOGRAM_BUCKETS_BY_METRIC: dict[str, tuple[float, ...]] = {
    # Latency / time (ms)
    "maxserve.request_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.input_processing_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.output_processing_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.time_to_first_token": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.response_queue_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.model_load_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.itl": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.time_per_output_token": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.batch_execution_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.batch_creation_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.di.decode_admission_queue_wait_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.di.decode_send_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.di.dispatch_rtt": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.di.prefill_span": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.di.reply_rtt": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.di.prefill_queue_wait_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.di.decode_postprocess_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.di.early_sync_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.di.handoff_to_first_token_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.structured_output.grammar_build_time": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.video.encoding_time_milliseconds": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.dkv.nixl_read_latency": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.dkv.nixl_write_latency": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.dkv.rpc_acquire_latency": HISTOGRAM_LATENCY_BUCKETS_MS,
    "maxserve.dkv.rpc_read_latency": HISTOGRAM_LATENCY_BUCKETS_MS,
    # Percentages
    "maxserve.dp_active_token_occupancy": HISTOGRAM_PERCENT_BUCKETS,
    "maxserve.dp_context_token_occupancy": HISTOGRAM_PERCENT_BUCKETS,
    "maxserve.cache.request_prefix_coverage": HISTOGRAM_PERCENT_BUCKETS,
    "maxserve.cache.used_kv_pct": HISTOGRAM_PERCENT_BUCKETS,
    "maxserve.cache.used_host_kv_pct": HISTOGRAM_PERCENT_BUCKETS,
    "maxserve.cache.used_disk_kv_pct": HISTOGRAM_PERCENT_BUCKETS,
    "maxserve.spec_decode.acceptance_rate_per_position": HISTOGRAM_PERCENT_BUCKETS,
    "maxserve.vision.cache_hit_rate": HISTOGRAM_PERCENT_BUCKETS,
    # Generic counts (tokens / occupancy)
    "maxserve.input_tokens_per_request": HISTOGRAM_COUNT_BUCKETS,
    "maxserve.output_tokens_per_request": HISTOGRAM_COUNT_BUCKETS,
    "maxserve.batch_input_tokens": HISTOGRAM_COUNT_BUCKETS,
    "maxserve.batch_context_tokens": HISTOGRAM_COUNT_BUCKETS,
    "maxserve.batch_terminated_reqs": HISTOGRAM_COUNT_BUCKETS,
    "maxserve.batch_pending_reqs": HISTOGRAM_COUNT_BUCKETS,
    "maxserve.requests_awaiting_admission": HISTOGRAM_COUNT_BUCKETS,
    "maxserve.responses_buffered": HISTOGRAM_COUNT_BUCKETS,
    "maxserve.di.ce_preempted_tg_pending_count": HISTOGRAM_COUNT_BUCKETS,
    # Batch size
    "maxserve.batch_size": HISTOGRAM_BATCH_SIZE_BUCKETS,
    # MiniMax-M3's video processor samples up to 512 frames per clip
    # (see max_private/minimax_m3/vision_processor.py); this bucket set's
    # upper bound (512) matches exactly.
    "maxserve.video.frames_per_clip": HISTOGRAM_BATCH_SIZE_BUCKETS,
    # Throughput (tokens/s)
    "maxserve.batch_prompt_throughput": HISTOGRAM_THROUGHPUT_TOKENS_BUCKETS,
    "maxserve.batch_generation_throughput": HISTOGRAM_THROUGHPUT_TOKENS_BUCKETS,
    # Transfer throughput (GiB/s)
    "maxserve.dkv.nixl_read_gib_per_s": HISTOGRAM_GIB_PER_S_BUCKETS,
    "maxserve.dkv.nixl_write_gib_per_s": HISTOGRAM_GIB_PER_S_BUCKETS,
    # Acceptance length (tokens)
    "maxserve.spec_decode.avg_acceptance_length": HISTOGRAM_ACCEPTANCE_LENGTH_BUCKETS,
}


def get_log_level(settings: Settings) -> int | str | None:
    otlp_level: int | str | None = (
        logging.getLevelName(settings.logs_otlp_level)
        if settings.logs_otlp_level
        else None
    )

    if settings.disable_telemetry:
        otlp_level = None

    return otlp_level


# Create a logger that buffers logs in memory and prints them to the console in batches.
# The returned logger is a no op if logging has not yet been configured.
def get_batch_logger(
    parent_logger: logging.Logger, capacity: int = 10
) -> logging.Logger:
    batch_logger = logging.getLogger(parent_logger.name)
    console_handlers = [
        h
        for h in logging.getLogger().handlers
        if type(h) is logging.StreamHandler
    ]
    memory_handler = logging.handlers.MemoryHandler(
        capacity=capacity,
        target=console_handlers[0] if len(console_handlers) > 0 else None,
    )
    batch_logger.addHandler(memory_handler)
    batch_logger.propagate = False
    return batch_logger


# Force a flush of the batch given logger.
def flush_batch_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        if type(handler) is logging.handlers.MemoryHandler:
            handler.flush()


COLOR_MAP = {
    "green": "\033[92m",
    "blue": "\033[94m",
    "red": "\033[91m",
}


class PrefixFormatter(logging.Formatter):
    """Custom formatter that adds a prefix to log messages."""

    def __init__(self, prefix: str, base_formatter: logging.Formatter):
        super().__init__()
        self.prefix = prefix
        self.base_formatter = base_formatter

    def format(self, record: logging.LogRecord) -> str:
        # Format the message using the base formatter
        formatted_message = self.base_formatter.format(record)
        # Add the prefix to the message
        return f"{self.prefix} {formatted_message}"


# Configure logging to console and OTEL.  This should be called before any
# 3rd party imports whose logging you wish to capture.
# Note that the color is not propagated to subprocesses. eg: ModelWorker
def configure_logging(
    settings: Settings, color: str | None = None, silent: bool = True
) -> None:
    otlp_level = get_log_level(settings)
    egress_enabled = not settings.disable_telemetry

    logging_handlers: list[logging.Handler] = []

    # Set up log filtering
    # ``uvicorn`` owns the HTTP error log: an exception escaping the ASGI app,
    # a malformed request, and the cancellation of in-flight requests when the
    # graceful-shutdown drain expires are all reported there and nowhere else.
    # Dropping them left connection-level failures with no server-side trace at
    # all. The ``uvicorn`` logger is pinned to WARNING below, so this admits
    # warnings and errors without the per-request ``uvicorn.access`` stream.
    components_to_log = [
        "root",
        "max._entrypoints",
        "max.pipelines",
        "max.serve",
        "uvicorn",
    ]
    try:
        if settings.logs_enable_components is not None:
            components = settings.logs_enable_components.split(",")
            components_to_log.extend(components)
    except Exception:
        print(
            "ERROR: Failed to parse logging components setting!  Using default."
        )

    def LogFilter(record: logging.LogRecord) -> bool:
        # Check if the logger name starts with any of the allowed component prefixes
        # This handles hierarchical logger names like "max.pipelines.architectures.llama3"
        return any(
            record.name == component or record.name.startswith(component + ".")
            for component in components_to_log
        )

    # Create a console handler
    if settings.logs_console_level is not None:
        if color is not None:
            if color not in COLOR_MAP:
                raise ValueError(f"Invalid color: {color}")
            color_code = COLOR_MAP[color]
            color_terminator = "\033[0m"
        else:
            color_code = ""
            color_terminator = ""

        console_handler = logging.StreamHandler()
        console_formatter: logging.Formatter
        if settings.structured_logging:
            console_formatter = jsonlogger.JsonFormatter(
                f"{color_code}%(levelname)s: %(message)s %(request_id)s %(batch_id)s{color_terminator}",
                timestamp=True,
            )
        else:
            console_formatter = logging.Formatter(
                (
                    f"{color_code}%(asctime)s.%(msecs)03d %(levelname)s:{color_terminator} %(message)s"
                ),
                datefmt="%H:%M:%S",
            )

        # Apply log prefix if provided
        if settings.log_prefix is not None:
            console_formatter = PrefixFormatter(
                settings.log_prefix, console_formatter
            )

        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(settings.logs_console_level)
        console_handler.addFilter(LogFilter)

        logging_handlers.append(console_handler)

    if (
        settings.logs_file_level is not None
        and settings.logs_file_path is not None
    ):
        # Create a file handler
        file_handler = logging.FileHandler(settings.logs_file_path)
        file_formatter: logging.Formatter
        if settings.structured_logging:
            file_formatter = jsonlogger.JsonFormatter(
                "%(levelname)s %(message)s %(request_id)s %(batch_id)s",
                timestamp=True,
            )
        else:
            file_formatter = logging.Formatter(
                ("%(asctime)s.%(msecs)03d %(levelname)s: %(message)s"),
                datefmt="%y:%m:%d-%H:%M:%S",
            )

        # Apply log prefix if provided
        if settings.log_prefix is not None:
            file_formatter = PrefixFormatter(
                settings.log_prefix, file_formatter
            )

        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(settings.logs_file_level)
        file_handler.addFilter(LogFilter)
        logging_handlers.append(file_handler)

    if egress_enabled and otlp_level is not None:
        # Create an OTEL handler
        logger_provider = LoggerProvider(logs_resource)
        set_logger_provider(logger_provider)
        exporter = OTLPLogExporter(endpoint=otelBaseUrl + "/v1/logs")
        logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(exporter)
        )
        otlp_handler = LoggingHandler(
            level=logging.getLevelName(otlp_level),
            logger_provider=logger_provider,
        )
        otlp_handler.addFilter(LogFilter)
        logging_handlers.append(otlp_handler)

    # Configure root logger level
    logger = logging.getLogger()
    if len(logging_handlers) > 0:
        # Clear existing handlers to prevent duplicates when configure_logging is called multiple times
        logger.handlers.clear()

        logger_level = min(h.level for h in logging_handlers)
        logger.setLevel(logger_level)
        for handler in logging_handlers:
            logger.addHandler(handler)

        # TODO use FastAPIInstrumentor once Motel supports traces.
        # For now, manually configure uvicorn.
        logging.getLogger("uvicorn").setLevel(logging.WARNING)
        # Explicit levels to reduce noise
        logging.getLogger("sse_starlette.sse").setLevel(
            max(logger_level, logging.INFO)
        )

    if not silent:
        logger.info(
            "Logging initialized: Console: %s, File: %s, Telemetry: %s",
            settings.logs_console_level,
            settings.logs_file_level,
            otlp_level,
        )


def _drop_metrics_matching(
    metrics_data: MetricsData | None,
    should_drop: Callable[[Metric], bool],
) -> MetricsData | None:
    """Rebuilds the MetricsData tree without metrics matching ``should_drop``.

    MetricsData/ResourceMetrics/ScopeMetrics are frozen dataclasses, so this
    reconstructs the tree rather than mutating it in place.
    """
    if metrics_data is None:
        return None
    return dataclasses.replace(
        metrics_data,
        resource_metrics=[
            dataclasses.replace(
                resource_metrics,
                scope_metrics=[
                    dataclasses.replace(
                        scope_metrics,
                        metrics=[
                            m
                            for m in scope_metrics.metrics
                            if not should_drop(m)
                        ],
                    )
                    for scope_metrics in resource_metrics.scope_metrics
                ],
            )
            for resource_metrics in metrics_data.resource_metrics
        ],
    )


class _SkipExponentialHistogramsPrometheusReader(PrometheusMetricReader):
    """Drops exponential-histogram data points instead of crashing on them.

    The classic Prometheus text exposition format (and this exporter) only
    knows how to render explicit-bucket histograms:
    ``_translate_to_prometheus`` assumes every non-``HistogramDataPoint`` has
    a ``.value`` attribute, which ``ExponentialHistogramDataPoint`` doesn't
    have, so it would raise ``AttributeError`` and 500 the whole ``/metrics``
    scrape. The exponential-histogram shadow metrics
    (metrics.HISTOGRAM_SHADOW_SUFFIX) only exist when
    ``MAX_SERVE_OTLP_METRICS_ENDPOINT`` is set, and are meant for that
    endpoint only (see _ExponentialShadowOnlyReader below) — this keeps
    Prometheus showing exactly the explicit-bucket histograms it always has,
    unaffected either way.
    """

    def _receive_metrics(
        self,
        metrics_data: MetricsData,
        timeout_millis: float = 10_000,
        **kwargs: object,
    ) -> None:
        filtered = _drop_metrics_matching(
            metrics_data, lambda m: isinstance(m.data, ExponentialHistogram)
        )
        if filtered is not None:
            super()._receive_metrics(filtered, timeout_millis, **kwargs)


class _ExponentialShadowOnlyReader(PeriodicExportingMetricReader):
    """Drops explicit-bucket histogram data points; keeps everything else.

    The complement of _SkipExponentialHistogramsPrometheusReader: the
    explicit-bucket histograms already reach Prometheus unchanged (above),
    so this reader — used only for the opt-in
    ``MAX_SERVE_OTLP_METRICS_ENDPOINT`` — carries just the exponential
    shadow histograms (plus counters/gauges) to that endpoint, keeping the
    two histogram representations on separate destinations for MXSERV-258
    side-by-side comparison rather than duplicating the explicit-bucket one
    there too.
    """

    def _receive_metrics(
        self,
        metrics_data: MetricsData,
        timeout_millis: float = 10_000,
        **kwargs: object,
    ) -> None:
        filtered = _drop_metrics_matching(
            metrics_data, lambda m: isinstance(m.data, ExplicitHistogramData)
        )
        if filtered is not None:
            super()._receive_metrics(filtered, timeout_millis, **kwargs)


def _histogram_views(settings: Settings) -> list[View]:
    # One View per histogram, matched by its exact instrument name, so each
    # metric gets bucket boundaries tuned to its actual range (latency in
    # ms, percentages, token/occupancy counts, batch sizes, throughput,
    # ...). Always active, regardless of MAX_SERVE_OTLP_METRICS_ENDPOINT:
    # Prometheus keeps showing exactly these, unchanged.
    #
    # Matching by exact name guarantees every Histogram matches at most one
    # of these Views, so we never get duplicate Prometheus series from
    # overlapping Views. Any histogram missing from the map falls back to
    # the OTEL SDK default buckets; test_all_histograms_have_explicit_buckets
    # guards against that.
    views: list[View] = [
        View(
            instrument_name=name,
            aggregation=ExplicitBucketHistogramAggregation(buckets),
        )
        for name, buckets in HISTOGRAM_BUCKETS_BY_METRIC.items()
    ]

    if settings.otlp_metrics_endpoint:
        # Exponential shadow of every histogram (metrics.HISTOGRAM_SHADOW_SUFFIX),
        # matched by name glob so new histograms are covered automatically.
        # Self-calibrating: no per-metric bucket boundaries to hand-tune, and
        # (unlike the explicit-bucket histograms above) mergeable across
        # replicas, so a Datadog-side p99 across the fleet is a mathematically
        # valid percentile rather than an average/max of independent
        # per-replica estimates. Reaches only the OTLP endpoint
        # (_ExponentialShadowOnlyReader), side by side with the
        # explicit-bucket histograms on Prometheus, for MXSERV-258 comparison.
        views.append(
            View(
                instrument_name="*" + HISTOGRAM_SHADOW_SUFFIX,
                aggregation=ExponentialBucketHistogramAggregation(),
            )
        )

    return views


def configure_metrics(settings: Settings) -> None:
    egress_enabled = not settings.disable_telemetry
    configure_histogram_shadow_emission(bool(settings.otlp_metrics_endpoint))

    meter_readers: list[MetricReader] = [
        _SkipExponentialHistogramsPrometheusReader(True)
    ]
    if egress_enabled:
        meter_readers.append(
            PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=otelBaseUrl + "/v1/metrics")
            )
        )
    if settings.otlp_metrics_endpoint:
        meter_readers.append(
            _ExponentialShadowOnlyReader(
                OTLPMetricExporter(
                    endpoint=settings.otlp_metrics_endpoint,
                    preferred_temporality={
                        Histogram: AggregationTemporality.DELTA,
                        Counter: AggregationTemporality.DELTA,
                    },
                )
            )
        )

    set_meter_provider(
        MeterProvider(
            metric_readers=meter_readers,
            resource=metrics_resource,
            views=_histogram_views(settings),
        )
    )

    logger = logging.getLogger()
    if settings.disable_telemetry:
        logger.info("Metrics disabled.")
    else:
        logger.info("Metrics initialized.")


def configure_tracing(settings: Settings) -> None:
    if not settings.disable_telemetry:
        # If the user set either standard OTel env var (e.g. for an
        # in-cluster DD agent), let OTLPSpanExporter resolve the endpoint
        # itself: its own env-var handling appends the signal-specific
        # "/v1/traces" path to OTEL_EXPORTER_OTLP_ENDPOINT, which a plain
        # os.environ.get() read here would not. Only fall back to the shared
        # Modular telemetry endpoint when neither var is set.
        user_configured_endpoint = os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT"
        ) or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        exporter = (
            OTLPSpanExporter()
            if user_configured_endpoint
            else OTLPSpanExporter(endpoint=otelBaseUrl + "/v1/traces")
        )
        provider = TracerProvider(resource=logs_resource)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        set_tracer_provider(provider)

    logger = logging.getLogger()
    if settings.disable_telemetry:
        logger.info("Tracing disabled.")
    else:
        logger.info("Tracing initialized.")


_kernel_trace_level = KernelTraceLevel.OFF


def batch_spans_enabled() -> bool:
    """Returns whether ``max.batch`` spans should be emitted, i.e. the model
    worker was configured with ``kernel_trace_level`` at ``batch`` or above."""
    return _kernel_trace_level >= KernelTraceLevel.BATCH


def configure_kernel_tracing(settings: Settings) -> None:
    """Configures GPU kernel-trace capture based on ``kernel_trace_level``.

    Must be called in the model worker process before ``InferenceSession``
    is constructed so that the libkineto auto-start picks up the enabled
    flag. Also records the level read by :func:`batch_spans_enabled`, so it
    must run before the scheduler starts.

    Args:
        settings: Server settings carrying ``kernel_trace_level``.
    """
    global _kernel_trace_level
    level = settings.kernel_trace_level
    _kernel_trace_level = level
    if level < KernelTraceLevel.OP:
        return

    if level == KernelTraceLevel.KERNEL:
        # Full libkineto kernel timeline: enable detailed NVTX tracing and
        # the libkineto auto-start that fires on InferenceSession construction.
        set_gpu_profiling_state("detailed")
        os.environ.setdefault("MODULAR_MAX_DEBUG_PROFILING_ENABLED", "true")
    else:
        # OP level: op-level NVTX user-annotation ranges only.
        set_gpu_profiling_state("on")

    logging.getLogger("max.serve").info(
        "Kernel tracing initialized: level=%s", level.value
    )


# Send a simple one-time structured log, avoiding the buggy OTEL SDK
# (see MAXSERV-904)
def send_telemetry_log(model_name: str) -> None:
    request_body = f"""{{
  "resourceLogs": [
    {{
      "resource": {{
        "attributes": [
          {{"key": "deployment.model", "value": {{"stringValue": "{model_name}"}}}},
          {{"key": "web.user.id", "value": {{"stringValue": "{logs_resource.attributes["web.user.id"]}"}}}},
          {{"key": "enduser.id", "value": {{"stringValue": "{logs_resource.attributes["enduser.id"]}"}}}},
          {{"key": "deployment.id", "value": {{"stringValue": "{logs_resource.attributes["deployment.id"]}"}}}},
          {{"key": "os.type", "value": {{"stringValue": "{logs_resource.attributes["os.type"]}"}}}},
          {{"key": "os.version", "value": {{"stringValue": "{logs_resource.attributes["os.version"]}"}}}},
          {{"key": "cpu.description", "value": {{"stringValue": "{logs_resource.attributes["cpu.description"]}"}}}},
          {{"key": "cpu.arch", "value": {{"stringValue": "{logs_resource.attributes["cpu.arch"]}"}}}},
          {{"key": "system.cloud", "value": {{"stringValue": "{logs_resource.attributes["system.cloud"]}"}}}},
          {{"key": "service.name", "value": {{"stringValue": "unknown_service"}}}},
          {{"key": "telemetry.sdk.language", "value": {{"stringValue": "python"}}}},
          {{"key": "telemetry.sdk.version", "value": {{"stringValue": "0.0.0"}}}},
          {{"key": "telemetry.sdk.name", "value": {{"stringValue": "opentelemetry"}}}}
        ]
      }},
      "scopeLogs": [
        {{
          "logRecords": [
            {{
              "attributes": [
                {{"key": "event.domain", "value": {{"stringValue": "modular"}}}},
                {{"key": "event.name", "value": {{"stringValue": "serve.telemetry.log"}}}}
              ],
              "body": {{"stringValue": ""}},
              "observedTimeUnixNano": "{int(time() * 1_000_000_000)}",
              "severityNumber": 9,
              "severityText": "INFO"
            }}
          ],
          "scope": {{"name": "modular_logger"}}
        }}
      ]
    }}
  ]
}}"""

    requests.post(
        otelBaseUrl + "/v1/logs",
        data=request_body,
        headers={"Content-Type": "application/json"},
        timeout=2,
    )
