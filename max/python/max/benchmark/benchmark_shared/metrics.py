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

"""Metrics classes for benchmark serving.

The lightweight percentile-metric types
(:class:`Metrics`, :class:`ConfidenceInfo`, :class:`PercentileMetrics`)
live in :mod:`max.benchmark.benchmark_shared.percentile_metrics` so
consumers that only need the type definitions can import them without
pulling in this module's heavier dependency surface (numpy, the rest
of MAX, transformers, etc.). They are re-exported here so existing
``from max.benchmark.benchmark_shared.metrics import …`` callers
continue to work unchanged.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np
from max.benchmark.benchmark_shared.percentile_metrics import (
    ConfidenceInfo,
    ConfidenceLevel,
    Metrics,
    PercentileMetrics,
    _is_finite_and_positive,
    compute_confidence_info,
)
from max.benchmark.benchmark_shared.request import ServerTokenStats
from max.benchmark.benchmark_shared.result_groups import (
    BenchmarkResultGroups,
    CacheStatsGroup,
    DiagnosticsGroup,
    GpuStatsGroup,
    LatencyStatsGroup,
    SummaryGroup,
    ThroughputStatsGroup,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ConfidenceInfo",
    "ConfidenceLevel",
    "Metrics",
    "PercentileMetrics",
    "build_result_groups",
]

if TYPE_CHECKING:
    from max.profiler.cpu import CPUMetrics

    from .server_metrics import (
        HistogramData,
        ParsedMetrics,
    )


def _validate_data(data: list[float]) -> None:
    """Validate input data for metrics calculations."""
    assert isinstance(data, list), "data must be a list"
    assert len(data) > 0, "data must not be empty"
    assert all(isinstance(x, float) for x in data), (
        "data must contain only floats"
    )


def _calculate_basic_stats(
    data: list[float], scale_factor: float
) -> dict[str, float]:
    """Calculate basic statistics (mean, std, p50) with scaling."""
    return {
        "mean": float(np.mean(data)) * scale_factor,
        "std": float(np.std(data)) * scale_factor,
        "p50": float(np.median(data)) * scale_factor,
    }


class BatchType(str, Enum):
    """Type of batch."""

    CE = "CE"
    """Context encoding batch."""
    TG = "TG"
    """Token generation batch."""


class HistogramMetric(str, Enum):
    BATCH_CONTEXT_TOKENS = "maxserve_batch_context_tokens"
    BATCH_CREATION_TIME_MS = "maxserve_batch_creation_time_milliseconds"
    BATCH_GEN_THROUGHPUT = (
        "maxserve_batch_generation_throughput_tokens_per_second"
    )
    BATCH_INPUT_TOKENS = "maxserve_batch_input_tokens"
    BATCH_PROMPT_THROUGHPUT = (
        "maxserve_batch_prompt_throughput_tokens_per_second"
    )
    BATCH_SIZE = "maxserve_batch_size"
    CACHE_REQUEST_PREFIX_COVERAGE_PCT = (
        "maxserve_cache_request_prefix_coverage_percent"
    )
    CACHE_USED_KV_PCT = "maxserve_cache_used_kv_pct_percent"
    INPUT_PROCESSING_TIME_MS = "maxserve_input_processing_time_milliseconds"
    INPUT_TOKENS_PER_REQUEST = "maxserve_input_tokens_per_request_tokens"
    ITL_MS = "maxserve_itl_milliseconds"
    OUTPUT_PROCESSING_TIME_MS = "maxserve_output_processing_time_milliseconds"
    OUTPUT_TOKENS_PER_REQUEST = "maxserve_output_tokens_per_request_tokens"
    REQUEST_TIME_MS = "maxserve_request_time_milliseconds"
    TIME_TO_FIRST_TOKEN_MS = "maxserve_time_to_first_token_milliseconds"
    MAXSERVE_BATCH_EXECUTION_TIME_MILLISECONDS = (
        "maxserve_batch_execution_time_milliseconds"
    )


@dataclass(init=False)
class ThroughputMetrics(PercentileMetrics):
    """
    Container for throughput-based metrics with automatic percentile calculations.

    For throughput metrics, percentiles are reversed because smaller values
    are worse for throughput (e.g., p99 represents the 1st percentile).

    Structured as a ``PercentileMetrics`` subclass (rather than wrapping one in a
    private attribute) so the computed stats live in dataclass fields: JSON /
    ``model_dump`` serializable, and expandable into per-stat columns by the
    schema-driven CSV flattener. The public API (``mean``, ``p99``,
    ``to_flat_dict``, ``format_with_prefix``, ``validate_metrics``, ...) is
    inherited from ``PercentileMetrics`` unchanged.
    """

    def __init__(
        self,
        data: list[float],
        scale_factor: float = 1.0,
        unit: str | None = None,
    ) -> None:
        """
        Initialize throughput metrics with automatic percentile calculations.

        Args:
            data: List of throughput values to calculate percentiles from.
            scale_factor: Factor to multiply all values by (e.g., for unit conversion).
            unit: Unit string to display (e.g., "tok/s", "req/s", "MB/s").
        """
        _validate_data(data)

        # Calculate basic stats and reversed percentiles for throughput
        basic_stats = _calculate_basic_stats(data, scale_factor)
        percentiles = self._calculate_throughput_percentiles(data, scale_factor)

        ci = compute_confidence_info(data, basic_stats["mean"], scale_factor)
        super().__init__(
            unit=unit, confidence_info=ci, **basic_stats, **percentiles
        )

    @staticmethod
    def _calculate_throughput_percentiles(
        data: list[float], scale_factor: float
    ) -> dict[str, float]:
        """Calculate throughput percentiles (reversed: bottom 10%, 5%, 1%)."""
        return {
            "p90": float(np.percentile(data, 10)) * scale_factor,  # Bottom 10%
            "p95": float(np.percentile(data, 5)) * scale_factor,  # Bottom 5%
            "p99": float(np.percentile(data, 1)) * scale_factor,  # Bottom 1%
        }

    def __str__(self) -> str:
        """Return a formatted string representation of throughput metrics in table format."""
        return self.format_with_prefix(prefix="throughput")


@dataclass(init=False)
class StandardPercentileMetrics(PercentileMetrics):
    """
    Container for standard percentile-based metrics with automatic calculations.

    For standard metrics, higher percentiles represent worse performance
    (e.g., p99 represents the 99th percentile).

    Structured as a ``PercentileMetrics`` subclass (see ``ThroughputMetrics``)
    so the computed stats are serializable dataclass fields; the public API is
    inherited from ``PercentileMetrics`` unchanged.
    """

    def __init__(
        self,
        data: list[float],
        scale_factor: float = 1.0,
        unit: str | None = None,
    ) -> None:
        """
        Initialize standard percentile metrics with automatic calculations.

        Args:
            data: List of values to calculate percentiles from.
            scale_factor: Factor to multiply all values by (e.g., 1000 for ms conversion).
            unit: Unit string to display (e.g., "ms", "s", "MB/s").
        """
        _validate_data(data)

        # Calculate basic stats and standard percentiles
        basic_stats = _calculate_basic_stats(data, scale_factor)
        percentiles = self._calculate_standard_percentiles(data, scale_factor)

        ci = compute_confidence_info(data, basic_stats["mean"], scale_factor)
        super().__init__(
            unit=unit, confidence_info=ci, **basic_stats, **percentiles
        )

    @staticmethod
    def _calculate_standard_percentiles(
        data: list[float], scale_factor: float
    ) -> dict[str, float]:
        """Calculate standard percentiles (90th, 95th, 99th)."""
        return {
            "p90": float(np.percentile(data, 90)) * scale_factor,
            "p95": float(np.percentile(data, 95)) * scale_factor,
            "p99": float(np.percentile(data, 99)) * scale_factor,
        }

    def __str__(self) -> str:
        """Return a formatted string representation of standard percentile metrics in table format."""
        return self.format_with_prefix(prefix="metric")


@dataclass(init=False)
class RatePercentileMetrics(StandardPercentileMetrics):
    """Bounded ratio in [0, 1]; mean of per-item ratios.

    Stored and displayed either as a fraction in [0, 1] (``as_percent=False``)
    or a percentage in [0, 100] (``as_percent=True``). Validation enforces the
    corresponding upper bound, which catches both negative values and values
    above the representation's maximum (e.g. ``cached_tokens > prompt_tokens``).
    """

    def __init__(
        self,
        data: list[float],
        *,
        as_percent: bool = True,
    ) -> None:
        scale_factor = 100.0 if as_percent else 1.0
        unit = "%" if as_percent else None
        super().__init__(data, scale_factor=scale_factor, unit=unit)

    def validate_metrics(self) -> tuple[bool, list[str]]:
        # Upper bound follows the stored representation: 100 for a percentage
        # (unit "%"), 1 for a fraction. Catches negatives and values above the
        # representation's maximum.
        upper_bound = 100.0 if self.unit == "%" else 1.0
        m = self.mean
        if not math.isfinite(m):
            return False, [f"Invalid mean: {m}"]
        if m < 0 or m > upper_bound:
            return False, [f"Mean {m} outside [0, {upper_bound}]"]
        return True, []


@dataclass
class PrefillDecodeStats:
    """Metrics specific to prefill and decode operations."""

    context_tokens: HistogramData | None = None
    creation_time_milliseconds: HistogramData | None = None
    generation_throughput_tokens_per_second: HistogramData | None = None
    input_tokens: HistogramData | None = None
    prompt_throughput_tokens_per_second: HistogramData | None = None

    def to_result_dict(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for f in dataclasses.fields(self):
            histogram: HistogramData | None = getattr(self, f.name)
            prefix = f"maxserve_batch_{f.name}"
            result[f"{prefix}_mean"] = (
                histogram.mean if histogram is not None else None
            )
            result[f"{prefix}_count"] = (
                histogram.count if histogram is not None else None
            )
            result[f"{prefix}_sum"] = (
                histogram.sum if histogram is not None else None
            )
        return result


@dataclass
class LoRAMetrics:
    """Metrics specific to LoRA operations."""

    load_times_ms: list[float] = field(default_factory=list)
    unload_times_ms: list[float] = field(default_factory=list)
    swap_times_ms: list[float] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    total_loads: int = 0
    total_unloads: int = 0
    total_swaps: int = 0

    def to_result_dict(self) -> dict[str, object]:
        # TODO: Not clear why only these metrics are exposed.
        # It will probably be moot once to_result_dict is removed, though.
        return {
            "total_loads": self.total_loads,
            "total_unloads": self.total_unloads,
            "load_times_ms": self.load_times_ms,
            "unload_times_ms": self.unload_times_ms,
        }


class BaseBenchmarkMetrics(BaseModel, Metrics):
    """Shared fields and logic for all benchmark metric containers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    duration: float
    completed: int
    failures: int
    max_concurrency: int
    request_throughput: float

    latency_ms: StandardPercentileMetrics

    peak_gpu_memory_mib: list[float]
    available_gpu_memory_mib: list[float]
    gpu_utilization: list[float]

    cpu_metrics: CPUMetrics | None = None

    metrics_by_endpoint: Mapping[str, ParsedMetrics] = Field(
        default_factory=dict
    )

    def to_result_dict(self) -> dict[str, object]:
        """Serialize aggregate metrics to a flat dict.

        Produces the key layout that the upload script and the
        ``--result-filename`` JSON expect (e.g. ``mean_ttft_ms``,
        ``p99_latency_ms``, ``ttft_ms_confidence``, …).

        Subclasses should call ``super().to_result_dict()`` and merge in
        their own fields.
        """
        d: dict[str, object] = {
            "duration": self.duration,
            "completed": self.completed,
            "failures": self.failures,
            "max_concurrency": self.max_concurrency,
            "request_throughput": self.request_throughput,
            "peak_gpu_memory_mib": self.peak_gpu_memory_mib,
            "available_gpu_memory_mib": self.available_gpu_memory_mib,
            "gpu_utilization": self.gpu_utilization,
        }
        if self.cpu_metrics is not None:
            d["cpu_metrics"] = dataclasses.asdict(self.cpu_metrics)
        for name in type(self).model_fields:
            val = getattr(self, name)
            if isinstance(val, (StandardPercentileMetrics, ThroughputMetrics)):
                d.update(val.to_flat_dict(name))
                d.update(val.confidence_to_flat_dict(name))
        if self.metrics_by_endpoint:
            # Backwards compat: `server_metrics` mirrors the first endpoint so
            # existing BigQuery/analysis consumers keep working.
            # `server_metrics_by_endpoint` carries the full per-endpoint breakdown.
            first_pm = next(iter(self.metrics_by_endpoint.values()))
            d["server_metrics"] = first_pm.to_dict()
            d["server_metrics_by_endpoint"] = {
                label: pm.to_dict()
                for label, pm in self.metrics_by_endpoint.items()
            }
        return d

    def validate_metrics(self) -> tuple[bool, list[str]]:
        """Validate common metric invariants.

        Subclasses should call ``super().validate_metrics()`` and extend the
        error list with their own checks.
        """
        errors: list[str] = []

        if self.failures > 0:
            errors.append(f"Some requests failed (failures={self.failures})")

        if self.completed <= 0:
            errors.append(f"No requests completed (completed={self.completed})")

        if not _is_finite_and_positive(self.request_throughput):
            errors.append(
                "Invalid throughput:"
                f" request_throughput={self.request_throughput}"
            )

        for name in type(self).model_fields:
            val = getattr(self, name)
            if isinstance(val, Metrics):
                ok, sub_errors = val.validate_metrics()
                if not ok:
                    for err in sub_errors:
                        errors.append(f"{name}: {err}")

        return len(errors) == 0, errors


class RequestRecord(BaseModel):
    """One request's own outcome, keyed by its dispatch index.

    The aggregates carry several per-request arrays (``input_lens``,
    ``output_lens``, ``ttfts``, ``request_submit_times``, …), but they are
    *not* mutually aligned: ``output_lens`` lists failures before
    successes while ``input_lens`` follows dispatch order, so
    ``input_lens[i]`` and ``output_lens[i]`` can describe different
    requests. Nothing can join them back together, which makes them
    unusable for any analysis that needs a request as a unit — comparing
    one run's request against another run's same request, above all.

    A record is that unit. ``index`` is the request's position in dispatch
    order, stable across runs of the same workload and seed, so two runs
    pair on it. Every request gets exactly one record, including failures
    and cancellations, because "this request failed here and succeeded
    there" is itself the finding.

    ``generated_text`` is opt-in (``--record-request-text``): it is the
    only unbounded field here, and a long-context run would multiply the
    result file by the size of its own output. Correctness comparison
    across runs needs it; a latency dashboard does not.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int
    """Position in dispatch order; the identity two runs pair on."""

    prompt_len: int
    output_len: int
    """Generated token count. 0 for a request that failed or was cancelled,
    which is the count it produced, not a missing value."""

    success: bool
    cancelled: bool
    measured: bool
    """Whether this request is inside the window the aggregates were
    computed over. A request can succeed and still be excluded by the
    skip-first/skip-last trim or by steady-state detection, and a reader
    comparing a record against a percentile needs to know which."""

    error: str = ""
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    """Mean time per output token; None below two generated tokens, where
    it is undefined rather than zero."""

    latency_ms: float | None = None
    submit_time: float | None = None
    complete_time: float | None = None
    """Run-relative ``perf_counter`` stamps, as the aggregates use."""

    session_id: str | None = None
    turn_index: int | None = None
    """Multi-turn provenance; None for single-turn workloads."""

    generated_text: str | None = None
    """None unless ``--record-request-text`` was set."""


# Workload-specific aggregates. ``BenchmarkResult`` (below) holds at
# most one per record, selected by ``task_type``; failed runs leave both
# ``None``. Composing them as nested pydantic objects (rather than
# mostly-Optional flat fields on the parent) lets consumers narrow once and
# access required fields directly.


class _CompletedRunBase(BaseModel):
    """Aggregates required for any completed benchmark iteration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    duration: float
    completed: int
    failures: int
    request_throughput: float
    # Successful requests dropped from the measured set by the skip_first /
    # skip_last trim windows. ``completed`` counts only what remains, so
    # without this field a slow iteration whose few successes all fell inside
    # the windows is indistinguishable from one where the server did nothing.
    excluded_successful: int = 0
    # ``None`` when the iteration produced no measured samples (e.g. every
    # request failed or all were skipped). Emitting ``None`` rather than a
    # NaN-filled metric keeps the serialized JSON free of null-valued
    # percentile objects: ``model_dump_json`` renders ``NaN`` as ``null``
    # (and BigQuery does the same on ingest), which strict downstream
    # consumers — the benchmark-visibility dashboard — otherwise reject
    # field-by-field and surface as spurious "dropped metric" warnings.
    latency_ms: StandardPercentileMetrics | None = None

    errors: list[str] = Field(default_factory=list)
    request_submit_times: list[float | None] = Field(default_factory=list)
    request_complete_times: list[float | None] = Field(default_factory=list)

    def to_result_dict(self) -> dict[str, object]:
        """Serialize aggregate fields to a flat dict.

        Subclasses extend with their workload-specific fields.
        """
        d: dict[str, object] = {
            "duration": self.duration,
            "completed": self.completed,
            "failures": self.failures,
            "request_throughput": self.request_throughput,
            "excluded_successful": self.excluded_successful,
            "errors": self.errors,
            "request_submit_times": self.request_submit_times,
            "request_complete_times": self.request_complete_times,
        }
        if self.latency_ms is not None:
            d.update(self.latency_ms.to_flat_dict("latency_ms"))
            d.update(self.latency_ms.confidence_to_flat_dict("latency_ms"))
        return d

    def all_measured_excluded(self) -> bool:
        """Return True when successes existed but the skip windows ate them all.

        In this state every derived metric (throughput, latency, token
        counts) is degenerate for lack of samples, not because the server
        did nothing — validation reports it as a single insufficient-data
        error instead of one error per degenerate metric.
        """
        return self.completed <= 0 and self.excluded_successful > 0

    def validate_metrics(self) -> tuple[bool, list[str]]:
        """Validate common aggregate invariants.

        Subclasses extend with their workload-specific checks.
        """
        if self.all_measured_excluded():
            return False, [
                f"Insufficient data: all {self.excluded_successful}"
                " successful requests were excluded by the skip_first /"
                " skip_last windows, leaving no measured samples. The server"
                " completed requests, but too few for this concurrency"
                " level's trim; no metrics can be concluded from this run."
            ]
        errors: list[str] = []
        if self.failures > 0:
            errors.append(f"Some requests failed (failures={self.failures})")
        if self.completed <= 0:
            errors.append(f"No requests completed (completed={self.completed})")
        if not _is_finite_and_positive(self.request_throughput):
            errors.append(
                "Invalid throughput:"
                f" request_throughput={self.request_throughput}"
            )
        if self.latency_ms is not None:
            ok, sub_errors = self.latency_ms.validate_metrics()
            if not ok:
                errors.extend(f"latency_ms: {e}" for e in sub_errors)
        return len(errors) == 0, errors

    def confidence_warnings(self) -> list[str]:
        """Return warnings for metrics with low or insufficient confidence.

        Default implementation returns ``[]``; workload-specific subclasses
        with comparable percentile fields override.
        """
        return []


class TextGenAggregates(_CompletedRunBase):
    """Aggregates from a successful text-generation iteration."""

    total_input: int
    total_output: int
    nonempty_response_chunks: int
    max_concurrent_conversations: int | None = None

    # All metric fields below are ``None`` when the iteration produced no
    # samples to derive them from (see ``latency_ms`` on the base class):
    # a fully-failed iteration leaves every field ``None``, while a
    # prefill-only / single-output-token iteration leaves just the
    # decode-phase fields (``tpot_ms`` / ``itl_ms`` / ``step_tpot_ms``)
    # ``None``. This avoids serializing NaN-filled percentile objects.
    input_throughput: ThroughputMetrics | None = None
    output_throughput: ThroughputMetrics | None = Field(
        default=None, json_schema_extra={"phase": "decode"}
    )
    ttft_ms: StandardPercentileMetrics | None = None
    tpot_ms: StandardPercentileMetrics | None = Field(
        default=None, json_schema_extra={"phase": "decode"}
    )
    # Per-step TPOT: ITL / tokens_per_step for each decode step.
    # Only populated when chunk-level text is available for re-tokenization.
    step_tpot_ms: StandardPercentileMetrics | None = Field(
        default=None, json_schema_extra={"phase": "decode"}
    )
    itl_ms: StandardPercentileMetrics | None = Field(
        default=None, json_schema_extra={"phase": "decode"}
    )

    max_input: int
    max_output: int
    max_total: int

    # Run-level aggregate throughput: (input + output) tokens per minute over
    # wall-clock duration. Derived in ``_derive_aggregate_tokens_per_minute``
    # so it is a real stored field (not a ``@computed_field``) and therefore
    # flows through the field-based CSV reporter as well as the JSON reporters,
    # rather than existing only inside ``to_result_dict``. ``nan`` when
    # duration is non-positive (degenerate / fully-skipped runs).
    aggregate_tokens_per_minute: float = float("nan")

    # Global: SUM(cached_tokens) / SUM(prompt_tokens).
    global_cached_token_rate: float
    # Per-turn cached_tokens / prompt_tokens; None when usage data is
    # unavailable.
    per_turn_cached_token_rate: RatePercentileMetrics | None = None
    # Turn-by-turn KV cache retention: cached_tokens(turn N) vs the
    # block-aligned context carried from turn N-1 in the same session. Catches
    # cached-token drop between turns. None for single-turn / no usage data.
    per_turn_cache_retention: RatePercentileMetrics | None = None

    # ``skip_first_n_requests`` / ``skip_last_n_requests`` are inputs and
    # shouldn't be part of the output metrics, but are included for
    # compatibility. These should be removed once results publication is in
    # use.
    skip_first_n_requests: int = 0
    skip_last_n_requests: int = 0
    # input_lens covers all outputs (including cancelled); output_lens covers
    # only non-cancelled outputs (failures get 0, not None — they failed,
    # they did not produce a zero-length response). The two lists are not
    # aligned index-for-index: failures appear first in output_lens, then
    # successes.
    input_lens: list[int] = Field(default_factory=list)
    output_lens: list[int] = Field(default_factory=list)
    ttfts: list[float] = Field(default_factory=list)
    # One record per request in dispatch order, the joinable form of the
    # arrays above. ``csv_mode=opaque`` keeps type-driven CSV from
    # exploding it into a column per request.
    request_records: list[RequestRecord] = Field(
        default_factory=list,
        json_schema_extra={"csv_mode": "opaque"},
    )
    # Empty when the server did not report per-request cached_tokens.
    per_turn_cached_token_rates: list[float] = Field(default_factory=list)
    # Per-turn cache retention fractions (one per checked turn, N>=2). Empty
    # for single-turn workloads or when usage data is unavailable.
    per_turn_cache_retentions: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _derive_aggregate_tokens_per_minute(self) -> TextGenAggregates:
        """Derive run-level TPM so console, JSON, and CSV share one source.

        Computed from the token totals and wall-clock duration rather than
        stored by callers, so every construction path (and JSON round-trips)
        yields a consistent value.
        """
        self.aggregate_tokens_per_minute = (
            (self.total_input + self.total_output) * 60.0 / self.duration
            if self.duration > 0
            else float("nan")
        )
        return self

    def to_result_dict(self) -> dict[str, object]:
        d = super().to_result_dict()
        d["total_input_tokens"] = self.total_input
        d["total_output_tokens"] = self.total_output
        # Aggregate across the entire benchmark run (not per-GPU): sum of
        # input + output tokens divided by wall-clock duration, in tokens/min.
        d["aggregate_tokens_per_minute"] = self.aggregate_tokens_per_minute
        d["max_concurrent_conversations"] = self.max_concurrent_conversations
        d["skip_first_n_requests"] = self.skip_first_n_requests
        d["skip_last_n_requests"] = self.skip_last_n_requests
        d["input_lens"] = self.input_lens
        d["output_lens"] = self.output_lens
        d["ttfts"] = self.ttfts
        d["request_records"] = [r.model_dump() for r in self.request_records]
        d["per_turn_cached_token_rates"] = self.per_turn_cached_token_rates
        d["per_turn_cache_retentions"] = self.per_turn_cache_retentions
        d["global_cached_token_rate"] = self.global_cached_token_rate
        for name, pm in [
            ("input_throughput", self.input_throughput),
            ("output_throughput", self.output_throughput),
        ]:
            if pm is not None:
                d.update(pm.to_flat_dict(name))
                d.update(pm.confidence_to_flat_dict(name))
        for name, spm in [
            ("ttft_ms", self.ttft_ms),
            ("tpot_ms", self.tpot_ms),
            ("itl_ms", self.itl_ms),
        ]:
            if spm is not None:
                d.update(spm.to_flat_dict(name))
                d.update(spm.confidence_to_flat_dict(name))
        if self.step_tpot_ms is not None:
            d.update(self.step_tpot_ms.to_flat_dict("step_tpot_ms"))
            d.update(self.step_tpot_ms.confidence_to_flat_dict("step_tpot_ms"))
        if self.per_turn_cached_token_rate is not None:
            d.update(
                self.per_turn_cached_token_rate.to_flat_dict(
                    "per_turn_cached_token_rate"
                )
            )
            d.update(
                self.per_turn_cached_token_rate.confidence_to_flat_dict(
                    "per_turn_cached_token_rate"
                )
            )
        if self.per_turn_cache_retention is not None:
            d.update(
                self.per_turn_cache_retention.to_flat_dict(
                    "per_turn_cache_retention"
                )
            )
            d.update(
                self.per_turn_cache_retention.confidence_to_flat_dict(
                    "per_turn_cache_retention"
                )
            )
        return d

    def validate_metrics(self) -> tuple[bool, list[str]]:
        ok, errors = super().validate_metrics()
        if self.all_measured_excluded():
            return ok, errors
        if self.total_output <= 0:
            errors.append(
                f"No output tokens generated (total_output={self.total_output})"
            )
        for name, m in [
            ("input_throughput", self.input_throughput),
            ("output_throughput", self.output_throughput),
            ("ttft_ms", self.ttft_ms),
            ("tpot_ms", self.tpot_ms),
            ("itl_ms", self.itl_ms),
            ("step_tpot_ms", self.step_tpot_ms),
        ]:
            # ``None`` means the metric had no samples this iteration
            # (e.g. decode metrics on a prefill-only run); the empty case
            # is already reported via ``completed`` / ``total_output``.
            if m is None:
                continue
            ok, sub_errors = m.validate_metrics()
            if not ok:
                errors.extend(f"{name}: {e}" for e in sub_errors)
        if self.per_turn_cached_token_rate is not None:
            ok, sub_errors = self.per_turn_cached_token_rate.validate_metrics()
            if not ok:
                errors.extend(
                    f"per_turn_cached_token_rate: {e}" for e in sub_errors
                )
        if self.per_turn_cache_retention is not None:
            ok, sub_errors = self.per_turn_cache_retention.validate_metrics()
            if not ok:
                errors.extend(
                    f"per_turn_cache_retention: {e}" for e in sub_errors
                )
        # Prefill-only workloads (max 1 output token per request) produce
        # no decode data, so decode-phase metrics are expected to be
        # degenerate.
        if self.max_output <= 1:
            decode_fields = {
                name
                for name, info in TextGenAggregates.model_fields.items()
                if isinstance(info.json_schema_extra, dict)
                and info.json_schema_extra.get("phase") == "decode"
            }
            errors = [
                e
                for e in errors
                if not any(e.startswith(f"{name}:") for name in decode_fields)
            ]
        return len(errors) == 0, errors

    def confidence_warnings(self) -> list[str]:
        warns: list[str] = []
        for name, metric in [
            ("ttft_ms", self.ttft_ms),
            ("tpot_ms", self.tpot_ms),
            ("output_throughput", self.output_throughput),
            ("step_tpot_ms", self.step_tpot_ms),
        ]:
            if metric is None:
                continue
            ci = getattr(metric, "confidence_info", None)
            if ci and ci.confidence in ("low", "insufficient_data"):
                warns.append(
                    f"{name}: {ci.confidence} confidence"
                    f" (CI width {ci.ci_relative_width:.0%} of mean,"
                    f" n={ci.sample_size})"
                )
        return warns


class PixelGenAggregates(_CompletedRunBase):
    """Aggregates from a successful pixel-generation iteration."""

    total_generated_outputs: int
    latencies: list[float] = Field(default_factory=list)
    # Per-request output counts; distinct from ``total_generated_outputs``,
    # which is the run-level sum of successful outputs only.
    num_generated_outputs: list[int] = Field(default_factory=list)

    def to_result_dict(self) -> dict[str, object]:
        d = super().to_result_dict()
        d["total_generated_outputs"] = self.total_generated_outputs
        d["latencies"] = self.latencies
        d["num_generated_outputs"] = self.num_generated_outputs
        return d


BenchmarkType = Literal["text", "pixel"]


class BenchmarkResult(BaseModel):
    """Per-iteration benchmark result for text- and pixel-generation tasks."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid", strict=True
    )
    task_type: BenchmarkType
    max_concurrency: int

    # Common per-iteration sampling. Populated regardless of success.
    peak_gpu_memory_mib: list[float] = Field(default_factory=list)
    available_gpu_memory_mib: list[float] = Field(default_factory=list)
    gpu_utilization: list[float] = Field(default_factory=list)
    cpu_metrics: CPUMetrics | None = None
    metrics_by_endpoint: Mapping[str, ParsedMetrics] = Field(
        default_factory=dict
    )
    prefill_stats: PrefillDecodeStats | None = None
    decode_stats: PrefillDecodeStats | None = None
    lora_metrics: LoRAMetrics | None = None

    # Run-level (not per-iteration) timing, captured once after the server
    # reports ready and stamped onto every iteration's result. ``None`` when
    # the harness didn't launch the server (e.g. benchmarking an external
    # endpoint) or startup capture failed.
    server_startup_time: float | None = None

    # Workload aggregates. Exactly the one matching ``task_type`` is set on
    # success; both stay ``None`` for failed iterations / dry runs.
    #
    # IMPORTANT: keep these as two *separate* Optional fields, NOT a combined
    # union ``aggregates: TextGenAggregates | PixelGenAggregates | None``.
    # The shared type-driven CSV column derivation in
    # ``max/python/max/benchmark/benchmark_shared/model_csv.py`` can only
    # expand ``Optional[SingleStructuredType]`` recursively into per-field
    # columns.  A two-type union returns ``None`` from
    # ``_unwrap_optional_structured_type``, causing ``flatten_model`` to fall
    # through to ``json.dumps`` and emit a single opaque JSON-blob column —
    # making the CSV output difficult to work with in spreadsheet tools.
    text_data: TextGenAggregates | None = None
    pixel_data: PixelGenAggregates | None = None

    # Text-generation-only fields. Stay ``None`` for pixel workloads.
    spec_decode_stats: SpecDecodeStats | None = None
    session_server_stats: dict[str, list[ServerTokenStats]] | None = None
    aggregate_server_stats: list[ServerTokenStats] | None = None

    # Lightweight steady-state detection diagnostics (scalars only — no
    # duplicate metric set). Populated by ``build_text_generation_result``
    # after MAD-based window detection. ``None`` for pixel workloads and
    # failed iterations.
    steady_state_detected: bool | None = None
    steady_state_window_count: int | None = None
    steady_state_mode: str | None = None
    steady_state_warning: str | None = None
    # Per-request TTFT outliers rejected in the steady-state window
    # (bounded by the request count). See build_text_generation_result.
    num_outliers_rejected: int | None = None

    # Namespaced presentation groups (summary / gpu / latency / …). Part of
    # the stored result — console, local JSON, and BigQuery all serialize
    # this field directly. Auto-filled when omitted so constructors and
    # historical blobs without the key still validate. ``csv_mode=opaque``
    # keeps type-driven CSV from expanding groups into duplicate columns
    # alongside ``text_data`` / ``pixel_data``.
    result_groups: BenchmarkResultGroups | None = Field(
        default=None,
        json_schema_extra={"csv_mode": "opaque"},
    )

    @model_validator(mode="after")
    def _check_data_matches_task_type(self) -> BenchmarkResult:
        if self.text_data is not None and self.task_type != "text":
            raise ValueError(f"text_data set but task_type={self.task_type!r}")
        if self.pixel_data is not None and self.task_type != "pixel":
            raise ValueError(f"pixel_data set but task_type={self.task_type!r}")
        text_only_fields = (
            self.spec_decode_stats,
            self.session_server_stats,
            self.aggregate_server_stats,
            self.steady_state_detected,
            self.steady_state_window_count,
            self.steady_state_mode,
            self.steady_state_warning,
            self.num_outliers_rejected,
        )
        if self.task_type != "text" and any(
            field is not None for field in text_only_fields
        ):
            raise ValueError(
                f"text-only result fields set but task_type={self.task_type!r}"
            )
        return self

    @model_validator(mode="after")
    def _derive_prefill_decode_stats(self) -> BenchmarkResult:
        """Auto-derive the prefill and decode stats from the metrics_by_endpoint."""
        if self.metrics_by_endpoint:
            context_tokens_ce = self._find_batch_histogram(
                BatchType.CE, HistogramMetric.BATCH_CONTEXT_TOKENS
            )
            context_tokens_tg = self._find_batch_histogram(
                BatchType.TG, HistogramMetric.BATCH_CONTEXT_TOKENS
            )
            creation_time_milliseconds_ce = self._find_batch_histogram(
                BatchType.CE,
                HistogramMetric.BATCH_CREATION_TIME_MS,
            )
            creation_time_milliseconds_tg = self._find_batch_histogram(
                BatchType.TG,
                HistogramMetric.BATCH_CREATION_TIME_MS,
            )
            prompt_throughput_tokens_per_second_ce = self._find_batch_histogram(
                BatchType.CE,
                HistogramMetric.BATCH_PROMPT_THROUGHPUT,
            )
            prompt_throughput_tokens_per_second_tg = self._find_batch_histogram(
                BatchType.TG,
                HistogramMetric.BATCH_PROMPT_THROUGHPUT,
            )
            input_tokens_ce = self._find_batch_histogram(
                BatchType.CE, HistogramMetric.BATCH_INPUT_TOKENS
            )
            input_tokens_tg = self._find_batch_histogram(
                BatchType.TG, HistogramMetric.BATCH_INPUT_TOKENS
            )
            generation_throughput_tokens_per_second_ce = (
                self._find_batch_histogram(
                    BatchType.CE,
                    HistogramMetric.BATCH_GEN_THROUGHPUT,
                )
            )
            generation_throughput_tokens_per_second_tg = (
                self._find_batch_histogram(
                    BatchType.TG,
                    HistogramMetric.BATCH_GEN_THROUGHPUT,
                )
            )
            self.prefill_stats = PrefillDecodeStats(
                context_tokens=context_tokens_ce,
                creation_time_milliseconds=creation_time_milliseconds_ce,
                generation_throughput_tokens_per_second=generation_throughput_tokens_per_second_ce,
                input_tokens=input_tokens_ce,
                prompt_throughput_tokens_per_second=prompt_throughput_tokens_per_second_ce,
            )
            self.decode_stats = PrefillDecodeStats(
                context_tokens=context_tokens_tg,
                creation_time_milliseconds=creation_time_milliseconds_tg,
                generation_throughput_tokens_per_second=generation_throughput_tokens_per_second_tg,
                input_tokens=input_tokens_tg,
                prompt_throughput_tokens_per_second=prompt_throughput_tokens_per_second_tg,
            )
        return self

    @model_validator(mode="after")
    def _ensure_result_groups(self) -> BenchmarkResult:
        """Populate ``result_groups`` once when the caller omitted them."""
        if self.result_groups is None:
            # Assign in place: returning ``model_copy`` from an ``after``
            # validator is ignored for ``__init__`` construction in pydantic.
            self.result_groups = build_result_groups(self)
        return self

    @property
    def aggregates(self) -> _CompletedRunBase | None:
        """Return whichever workload-specific aggregates are populated."""
        return self.text_data or self.pixel_data

    def _find_batch_histogram(
        self, batch_type: BatchType, property_name: HistogramMetric
    ) -> HistogramData | None:
        """First endpoint that exposes the MAX-serve batch-time histogram."""
        for pm in self.metrics_by_endpoint.values():
            hist = pm.get_histogram(
                property_name.value,
                {"batch_type": batch_type.value},
            )
            if hist:
                return hist
        return None

    @property
    def mean_prefill_batch_time_ms(self) -> float | None:
        """Mean prefill (context encoding) batch execution time in milliseconds."""
        hist = self._find_batch_histogram(
            BatchType.CE,
            HistogramMetric.MAXSERVE_BATCH_EXECUTION_TIME_MILLISECONDS,
        )
        return hist.mean if hist else None

    @property
    def mean_decode_batch_time_ms(self) -> float | None:
        """Mean decode (token generation) batch execution time in milliseconds."""
        hist = self._find_batch_histogram(
            BatchType.TG,
            HistogramMetric.MAXSERVE_BATCH_EXECUTION_TIME_MILLISECONDS,
        )
        return hist.mean if hist else None

    @property
    def prefill_batch_count(self) -> int:
        """Total number of prefill (context encoding) batches executed."""
        hist = self._find_batch_histogram(
            BatchType.CE,
            HistogramMetric.MAXSERVE_BATCH_EXECUTION_TIME_MILLISECONDS,
        )
        return int(hist.count) if hist else 0

    @property
    def decode_batch_count(self) -> int:
        """Total number of decode (token generation) batches executed."""
        hist = self._find_batch_histogram(
            BatchType.TG,
            HistogramMetric.MAXSERVE_BATCH_EXECUTION_TIME_MILLISECONDS,
        )
        return int(hist.count) if hist else 0

    def to_result_dict(self) -> dict[str, object]:
        """Serialize aggregate metrics to a flat dict.

        Produces the key layout that the upload script and the
        ``--result-filename`` JSON expect (e.g. ``mean_ttft_ms``,
        ``p99_latency_ms``, ``ttft_ms_confidence``, …).
        """
        d: dict[str, object] = {
            "max_concurrency": self.max_concurrency,
            "peak_gpu_memory_mib": self.peak_gpu_memory_mib,
            "available_gpu_memory_mib": self.available_gpu_memory_mib,
            "gpu_utilization": self.gpu_utilization,
        }
        if self.cpu_metrics is not None:
            d["cpu_metrics"] = dataclasses.asdict(self.cpu_metrics)
        if self.metrics_by_endpoint:
            # Backwards compat: ``server_metrics`` mirrors the first endpoint
            # so existing BigQuery / analysis consumers keep working.
            # ``server_metrics_by_endpoint`` carries the full per-endpoint
            # breakdown.
            first_pm = next(iter(self.metrics_by_endpoint.values()))
            d["server_metrics"] = first_pm.to_dict()
            d["server_metrics_by_endpoint"] = {
                label: pm.to_dict()
                for label, pm in self.metrics_by_endpoint.items()
            }

        if self.prefill_stats is not None:
            d["prefill_stats"] = self.prefill_stats.to_result_dict()
        if self.decode_stats is not None:
            d["decode_stats"] = self.decode_stats.to_result_dict()

        agg = self.aggregates
        if agg is not None:
            d.update(agg.to_result_dict())
            # Prefill/decode batch stats live on parent histograms (not on
            # the aggregate); enrich server_metrics here to avoid threading
            # parent state through agg.to_result_dict().
            if self.task_type == "text" and "server_metrics" in d:
                assert isinstance(d["server_metrics"], dict)
                d["server_metrics"].update(
                    {
                        "prefill_batch_execution_time_ms": self.mean_prefill_batch_time_ms,
                        "prefill_batch_count": self.prefill_batch_count,
                        "decode_batch_execution_time_ms": self.mean_decode_batch_time_ms,
                        "decode_batch_count": self.decode_batch_count,
                    }
                )

        if self.lora_metrics is not None:
            d["lora_metrics"] = self.lora_metrics.to_result_dict()
        # Lightweight steady-state detection diagnostics (no duplicate metric
        # set — scalars only for observability / PERF-2615 telemetry).
        if self.steady_state_detected is not None:
            d["steady_state_detected"] = self.steady_state_detected
            d["steady_state_window_count"] = self.steady_state_window_count
            d["steady_state_mode"] = self.steady_state_mode
            d["steady_state_warning"] = self.steady_state_warning
            d["num_outliers_rejected"] = self.num_outliers_rejected
        if self.spec_decode_stats is not None:
            d.update(self.spec_decode_stats.to_result_dict())
        if self.session_server_stats is not None:
            d["session_server_stats"] = {
                sid: [dataclasses.asdict(s) for s in stats]
                for sid, stats in self.session_server_stats.items()
            }
        if self.aggregate_server_stats is not None:
            d["aggregate_server_stats"] = [
                dataclasses.asdict(s) for s in self.aggregate_server_stats
            ]
        # Namespaced presentation view — same object BigQuery embeds via
        # ``model_dump_json`` and the local ``--result-filename`` JSON carries.
        if self.result_groups is not None:
            d["result_groups"] = self.result_groups.model_dump(mode="json")
        return d

    def validate_metrics(self) -> tuple[bool, list[str]]:
        """Validate that aggregates are populated with sensible values.

        Returns ``(True, [])`` for failed iterations / dry runs that have no
        aggregates to check.
        """
        agg = self.aggregates
        if agg is None:
            return True, []
        return agg.validate_metrics()

    def confidence_warnings(self) -> list[str]:
        """Return warnings for metrics with low or insufficient confidence.

        Returns ``[]`` for pixel-gen workloads (no comparable percentile
        fields) and for failed iterations.
        """
        agg = self.aggregates
        if agg is None:
            return []
        return agg.confidence_warnings()


def build_result_groups(result: BenchmarkResult) -> BenchmarkResultGroups:
    """Builds the namespaced groups for a :class:`BenchmarkResult`.

    Used by ``BenchmarkResult``'s model validator when ``result_groups`` is
    omitted. Pure projection: reads whichever of ``text_data`` /
    ``pixel_data`` is populated and reshapes already-present fields into
    named groups. Adds no new data. Schema types live in the lightweight
    ``:result_groups`` target; the factory stays here beside the concrete
    result type.
    """
    text_data = result.text_data
    pixel_data = result.pixel_data
    agg = text_data or pixel_data

    summary = SummaryGroup(
        task_type=result.task_type,
        max_concurrency=result.max_concurrency,
        duration=agg.duration if agg else None,
        completed=agg.completed if agg else None,
        failures=agg.failures if agg else None,
        request_throughput=agg.request_throughput if agg else None,
        aggregate_tokens_per_minute=(
            text_data.aggregate_tokens_per_minute if text_data else None
        ),
        mean_ttft_ms=(
            text_data.ttft_ms.mean if text_data and text_data.ttft_ms else None
        ),
        mean_tpot_ms=(
            text_data.tpot_ms.mean if text_data and text_data.tpot_ms else None
        ),
        mean_itl_ms=(
            text_data.itl_ms.mean if text_data and text_data.itl_ms else None
        ),
        total_generated_outputs=(
            pixel_data.total_generated_outputs if pixel_data else None
        ),
    )

    gpu_stats = (
        GpuStatsGroup(
            peak_gpu_memory_mib=result.peak_gpu_memory_mib,
            available_gpu_memory_mib=result.available_gpu_memory_mib,
            gpu_utilization=result.gpu_utilization,
        )
        if (
            result.peak_gpu_memory_mib
            or result.available_gpu_memory_mib
            or result.gpu_utilization
        )
        else None
    )

    latency_stats: LatencyStatsGroup | None = None
    throughput_stats: ThroughputStatsGroup | None = None
    cache_stats: CacheStatsGroup | None = None
    if text_data is not None:
        latency_stats = LatencyStatsGroup(
            latency_ms=text_data.latency_ms,
            ttft_ms=text_data.ttft_ms,
            tpot_ms=text_data.tpot_ms,
            itl_ms=text_data.itl_ms,
            step_tpot_ms=text_data.step_tpot_ms,
        )
        throughput_stats = ThroughputStatsGroup(
            input_throughput=text_data.input_throughput,
            output_throughput=text_data.output_throughput,
        )
        cache_stats = CacheStatsGroup(
            global_cached_token_rate=text_data.global_cached_token_rate,
            per_turn_cached_token_rate=text_data.per_turn_cached_token_rate,
            per_turn_cache_retention=text_data.per_turn_cache_retention,
        )
    elif pixel_data is not None:
        latency_stats = LatencyStatsGroup(latency_ms=pixel_data.latency_ms)

    # ``agg.errors`` is one entry per request (empty string on success);
    # keep only real messages, matching the dashboard aggregates mirror.
    real_errors = [e for e in (agg.errors if agg else []) if e]
    diagnostics: DiagnosticsGroup | None = None
    if (
        result.server_startup_time is not None
        or result.steady_state_detected is not None
        or result.steady_state_window_count is not None
        or result.steady_state_mode is not None
        or result.steady_state_warning is not None
        or result.num_outliers_rejected is not None
        or real_errors
    ):
        diagnostics = DiagnosticsGroup(
            server_startup_time=result.server_startup_time,
            steady_state_detected=result.steady_state_detected,
            steady_state_window_count=result.steady_state_window_count,
            steady_state_mode=result.steady_state_mode,
            steady_state_warning=result.steady_state_warning,
            num_outliers_rejected=result.num_outliers_rejected,
            errors=real_errors,
        )

    return BenchmarkResultGroups(
        summary=summary,
        gpu_stats=gpu_stats,
        latency_stats=latency_stats,
        throughput_stats=throughput_stats,
        cache_stats=cache_stats,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Speculative decoding metrics
# ---------------------------------------------------------------------------


@dataclass
class SpecDecodeMetrics:
    """Speculative decoding metrics scraped from a Prometheus endpoint.

    Two backend shapes are supported:

    - vLLM-style counters (``vllm:spec_decode_*``): ``num_drafts``,
      ``num_draft_tokens``, ``num_accepted_tokens``, ``accepted_per_pos``.
    - MAX-style histogram (``maxserve_spec_decode_acceptance_rate_per_position``):
      ``per_pos_rate_sum`` / ``per_pos_rate_count`` give running sums and counts
      of observed acceptance-rate samples per position. Window averages are
      computed via deltas.
    - MAX-style histogram (``maxserve_spec_decode_avg_acceptance_length``):
      ``avg_acceptance_length_sum`` / ``avg_acceptance_length_count`` track
      observations of per-batch average acceptance length (tokens).

    A backend may populate either group; missing values default to 0/empty.
    """

    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    accepted_per_pos: dict[int, int] = field(default_factory=dict)
    per_pos_rate_sum: dict[int, float] = field(default_factory=dict)
    per_pos_rate_count: dict[int, int] = field(default_factory=dict)
    avg_acceptance_length_sum: float = 0.0
    avg_acceptance_length_count: float = 0.0

    def __iadd__(self, other: SpecDecodeMetrics) -> SpecDecodeMetrics:
        self.num_drafts += other.num_drafts
        self.num_draft_tokens += other.num_draft_tokens
        self.num_accepted_tokens += other.num_accepted_tokens
        for pos, n in other.accepted_per_pos.items():
            self.accepted_per_pos[pos] = self.accepted_per_pos.get(pos, 0) + n
        for pos, s in other.per_pos_rate_sum.items():
            self.per_pos_rate_sum[pos] = self.per_pos_rate_sum.get(pos, 0.0) + s
        for pos, c in other.per_pos_rate_count.items():
            self.per_pos_rate_count[pos] = (
                self.per_pos_rate_count.get(pos, 0) + c
            )
        self.avg_acceptance_length_sum += other.avg_acceptance_length_sum
        self.avg_acceptance_length_count += other.avg_acceptance_length_count
        return self


@dataclass
class SpecDecodeStats:
    """Speculative decoding statistics for a benchmark window.

    Fields are ``None`` when the underlying metric was not exposed by the
    backend in the scraped Prometheus output.
    """

    num_drafts: int | None = None
    """Number of draft sequences generated."""
    draft_tokens: int | None = None
    """Total number of draft tokens generated."""
    accepted_tokens: int | None = None
    """Total number of draft tokens accepted."""
    acceptance_rate: float | None = None
    """Percentage of draft tokens accepted."""
    acceptance_length: float | None = None
    """Average number of tokens accepted per draft (including verified token)."""
    per_position_acceptance_rates: list[float] = field(default_factory=list)
    """Acceptance rate at each draft position as a fraction (0-1).

    Empty when no per-position data was exposed.
    """

    def to_result_dict(self) -> dict[str, object]:
        """Return a flat dict of spec-decode keys for the benchmark result.

        Only fields the backend actually exposed are emitted; missing aggregates
        are omitted rather than written as ``None``.
        """
        result: dict[str, object] = {}
        if self.acceptance_rate is not None:
            result["spec_decode_acceptance_rate"] = self.acceptance_rate
        if self.acceptance_length is not None:
            result["spec_decode_acceptance_length"] = self.acceptance_length
        if self.num_drafts is not None:
            result["spec_decode_num_drafts"] = int(self.num_drafts)
        if self.draft_tokens is not None:
            result["spec_decode_draft_tokens"] = int(self.draft_tokens)
        if self.accepted_tokens is not None:
            result["spec_decode_accepted_tokens"] = int(self.accepted_tokens)
        if self.per_position_acceptance_rates:
            result["spec_decode_per_position_acceptance_rates"] = (
                self.per_position_acceptance_rates
            )
        return result


def calculate_spec_decode_stats(
    metrics_before: SpecDecodeMetrics,
    metrics_after: SpecDecodeMetrics,
) -> SpecDecodeStats | None:
    """Compute benchmark-window speculative decoding stats from metric deltas.

    Aggregate counters (``num_drafts``, ``num_draft_tokens``,
    ``num_accepted_tokens``) are computed when the backend exposed them
    (vLLM-style). Per-position acceptance rates are computed from either the
    vLLM ``num_accepted_tokens_per_pos`` counter or the MAX-style
    ``maxserve_spec_decode_acceptance_rate_per_position`` histogram, whichever
    is available.

    When only MAX histograms are present (no vLLM-style counters), an aggregate
    **acceptance_rate** (0--100, matching the printed benchmark column) is the
    count-weighted mean of per-position acceptance-rate observations pooled
    across positions. **acceptance_length** can additionally be taken from the
    ``maxserve_spec_decode_avg_acceptance_length`` histogram delta mean.

    Args:
        metrics_before: Snapshot taken before the benchmark window.
        metrics_after: Snapshot taken after the benchmark window.

    Returns:
        A ``SpecDecodeStats`` with whatever fields are derivable, or ``None``
        when no spec-decode metrics moved during the window.
    """
    delta_drafts = metrics_after.num_drafts - metrics_before.num_drafts
    delta_draft_tokens = (
        metrics_after.num_draft_tokens - metrics_before.num_draft_tokens
    )
    delta_accepted = (
        metrics_after.num_accepted_tokens - metrics_before.num_accepted_tokens
    )

    per_pos_rates: list[float] = []
    pooled_acceptance_rate_percent: float | None = None
    if delta_drafts > 0 and (
        metrics_before.accepted_per_pos or metrics_after.accepted_per_pos
    ):
        positions = sorted(
            set(metrics_before.accepted_per_pos.keys())
            | set(metrics_after.accepted_per_pos.keys())
        )
        for pos in positions:
            before_val = metrics_before.accepted_per_pos.get(pos, 0)
            after_val = metrics_after.accepted_per_pos.get(pos, before_val)
            per_pos_rates.append((after_val - before_val) / delta_drafts)
    elif metrics_before.per_pos_rate_count or metrics_after.per_pos_rate_count:
        positions = sorted(
            set(metrics_before.per_pos_rate_sum.keys())
            | set(metrics_after.per_pos_rate_sum.keys())
        )
        total_sum_delta = 0.0
        total_count_delta = 0.0
        for pos in positions:
            sum_delta = metrics_after.per_pos_rate_sum.get(
                pos, 0.0
            ) - metrics_before.per_pos_rate_sum.get(pos, 0.0)
            count_delta = metrics_after.per_pos_rate_count.get(
                pos, 0
            ) - metrics_before.per_pos_rate_count.get(pos, 0)
            if count_delta > 0:
                total_sum_delta += sum_delta
                total_count_delta += count_delta
                # Histogram observations are recorded as percentages (0-100);
                # normalize to a 0-1 fraction for parity with the vLLM path.
                per_pos_rates.append((sum_delta / count_delta) / 100.0)
        if total_count_delta > 0:
            pooled_acceptance_rate_percent = total_sum_delta / total_count_delta

    al_sum_delta = (
        metrics_after.avg_acceptance_length_sum
        - metrics_before.avg_acceptance_length_sum
    )
    al_count_delta = (
        metrics_after.avg_acceptance_length_count
        - metrics_before.avg_acceptance_length_count
    )
    acceptance_length_from_max_hist: float | None = None
    if al_count_delta > 0:
        acceptance_length_from_max_hist = al_sum_delta / al_count_delta

    has_aggregates = delta_draft_tokens > 0
    if (
        not has_aggregates
        and not per_pos_rates
        and pooled_acceptance_rate_percent is None
        and acceptance_length_from_max_hist is None
    ):
        return None

    if has_aggregates:
        acceptance_rate = (delta_accepted / delta_draft_tokens) * 100
        acceptance_length = (
            1 + delta_accepted / delta_drafts if delta_drafts > 0 else None
        )
        return SpecDecodeStats(
            num_drafts=delta_drafts,
            draft_tokens=delta_draft_tokens,
            accepted_tokens=delta_accepted,
            acceptance_rate=acceptance_rate,
            acceptance_length=acceptance_length,
            per_position_acceptance_rates=per_pos_rates,
        )
    return SpecDecodeStats(
        acceptance_rate=pooled_acceptance_rate_percent,
        acceptance_length=acceptance_length_from_max_hist,
        per_position_acceptance_rates=per_pos_rates,
    )


# Resolve forward references on the pydantic models. ``CPUMetrics``,
# ``HistogramData``, and ``ParsedMetrics`` are kept under ``TYPE_CHECKING`` to
# avoid a circular import (``server_metrics`` imports ``SpecDecodeMetrics`` from
# this module), so we re-import them here once all of this module's classes are
# defined and call ``model_rebuild()`` so pydantic can resolve the annotations.
# ``HistogramData`` backs the ``PrefillDecodeStats`` fields referenced by
# ``BenchmarkResult.prefill_stats``/``decode_stats``.
from max.profiler.cpu import CPUMetrics

from .server_metrics import HistogramData, ParsedMetrics

BaseBenchmarkMetrics.model_rebuild()
BenchmarkResult.model_rebuild()
