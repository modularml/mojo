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

"""Namespaced group schemas for :class:`BenchmarkResult.result_groups`.

Carved out of the full ``benchmark_shared`` library (bazel target
``:result_groups``) so consumers that only need the group type shapes can
import them without the runner's heavy transitive deps. Runtime deps are
stdlib + pydantic + :mod:`percentile_metrics` only.

``BenchmarkResult.result_groups`` is the stored presentation view of an
iteration: an always-populated ``summary`` group followed by named
supporting-detail groups (GPU stats, latency stats, …), each ``None`` when
the underlying result has nothing to report for it.

The factory that fills the field lives next to ``BenchmarkResult`` in
``metrics.build_result_groups`` (it needs the concrete result type). The
serving benchmark serializes ``result_groups`` with the rest of the model
(console summary, local result JSON, BigQuery ``result`` blob). The
benchmark-visibility dashboard renders the stored groups directly — its
Pydantic mirrors are held to these schemas by ``test_schema_sync``.
"""

from __future__ import annotations

from typing import Literal

from max.benchmark.benchmark_shared.percentile_metrics import PercentileMetrics
from pydantic import BaseModel, ConfigDict, Field

# Keep in sync with ``metrics.BenchmarkType`` (defined there to avoid a
# circular import: this module must stay importable from ``metrics``).
BenchmarkType = Literal["text", "pixel"]


class SummaryGroup(BaseModel):
    """Headline metrics for a serving benchmark iteration.

    The one group renderers show by default; every other group is opt-in
    supporting detail.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_type: BenchmarkType | None = None
    max_concurrency: int | None = None
    duration: float | None = None
    completed: int | None = None
    failures: int | None = None
    request_throughput: float | None = None
    aggregate_tokens_per_minute: float | None = None
    mean_ttft_ms: float | None = None
    mean_tpot_ms: float | None = None
    mean_itl_ms: float | None = None
    total_generated_outputs: int | None = None


class GpuStatsGroup(BaseModel):
    """Per-device GPU utilization / memory sampling over the iteration.

    Each field is one value per GPU. The console prints those lists
    through the shared percentile table; JSON keeps the per-GPU series.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    peak_gpu_memory_mib: list[float] = Field(default_factory=list)
    available_gpu_memory_mib: list[float] = Field(default_factory=list)
    gpu_utilization: list[float] = Field(default_factory=list)


class LatencyStatsGroup(BaseModel):
    """Percentile latency distributions for the iteration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    latency_ms: PercentileMetrics | None = None
    ttft_ms: PercentileMetrics | None = None
    tpot_ms: PercentileMetrics | None = None
    itl_ms: PercentileMetrics | None = None
    step_tpot_ms: PercentileMetrics | None = None


class ThroughputStatsGroup(BaseModel):
    """Percentile throughput distributions for the iteration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_throughput: PercentileMetrics | None = None
    output_throughput: PercentileMetrics | None = None


class CacheStatsGroup(BaseModel):
    """Prefix-cache hit-rate / retention metrics (text workloads only)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    global_cached_token_rate: float | None = None
    per_turn_cached_token_rate: PercentileMetrics | None = None
    per_turn_cache_retention: PercentileMetrics | None = None


class DiagnosticsGroup(BaseModel):
    """Steady-state detection and error diagnostics for the iteration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    server_startup_time: float | None = None
    steady_state_detected: bool | None = None
    steady_state_window_count: int | None = None
    steady_state_mode: str | None = None
    steady_state_warning: str | None = None
    num_outliers_rejected: int | None = None
    errors: list[str] = Field(default_factory=list)


class BenchmarkResultGroups(BaseModel):
    """Namespaced groups stored on :class:`BenchmarkResult.result_groups`.

    ``summary`` is always populated; the remaining groups are ``None``
    when the underlying result has nothing to report for that group
    (e.g. ``cache_stats`` on a pixel-generation iteration).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: SummaryGroup
    gpu_stats: GpuStatsGroup | None = None
    latency_stats: LatencyStatsGroup | None = None
    throughput_stats: ThroughputStatsGroup | None = None
    cache_stats: CacheStatsGroup | None = None
    diagnostics: DiagnosticsGroup | None = None
