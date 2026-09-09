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

"""Metrics calculation for serving benchmarks."""

from __future__ import annotations

import logging
import statistics
import warnings
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from itertools import pairwise
from typing import TYPE_CHECKING, TypeGuard

from max.benchmark.benchmark_shared.metrics import (
    BenchmarkResult,
    PixelGenAggregates,
    RatePercentileMetrics,
    RequestRecord,
    SpecDecodeStats,
    StandardPercentileMetrics,
    TextGenAggregates,
    ThroughputMetrics,
)
from max.benchmark.benchmark_shared.request import (
    BaseRequestFuncOutput,
    PixelGenerationRequestFuncOutput,
    RequestFuncOutput,
    measured_window_duration,
)
from max.benchmark.benchmark_shared.steady_state import (
    detect_steady_state,
    reject_metric_outliers,
)
from max.profiler.cpu import CPUMetrics
from transformers import PreTrainedTokenizerBase

if TYPE_CHECKING:
    from max.benchmark.benchmark_shared.server_metrics import ParsedMetrics
    from max.profiler.gpu import GPUStats

logger = logging.getLogger(__name__)

# Server and client token counts can legitimately differ slightly because the
# server applies the chat template and may count special tokens differently.
# 5% tolerates small systematic gaps while still surfacing real mismatches
# (e.g. wrong tokenizer, prompt truncation, or model-side reprocessing).
_INPUT_TOKEN_DISCREPANCY_THRESHOLD = 0.05


def compute_output_len(
    tokenizer: PreTrainedTokenizerBase,
    output: RequestFuncOutput,
) -> int:
    return len(
        tokenizer.encode(output.generated_text, add_special_tokens=False)
    )


def _warn_on_request_failures(
    outputs: Sequence[BaseRequestFuncOutput],
    completed: int,
    failures: int,
    failed_responses: Sequence[BaseRequestFuncOutput],
) -> None:
    if len(outputs) == 0:
        warnings.warn(
            "No responses were received from the server.", stacklevel=2
        )

    if failures != 0:
        warnings.warn(
            (
                "Some requests failed. The responses returned are displayed "
                "below. Please check server logs for more information."
            ),
            stacklevel=2,
        )
        for failed_response in failed_responses:
            logger.error(f"Failed :: {failed_response}")

    if completed == 0:
        warnings.warn(
            (
                "All requests failed. This is likely due to a misconfiguration "
                "on the benchmark arguments."
            ),
            stacklevel=2,
        )


def _aggregate_gpu_stats(
    collect_gpu_stats: bool,
    gpu_metrics: list[dict[str, GPUStats]] | None,
) -> tuple[list[float], list[float], list[float]]:
    peak_gpu_memory_mib: list[float] = []
    available_gpu_memory_mib: list[float] = []
    gpu_utilization: list[float] = []

    if not collect_gpu_stats or not gpu_metrics:
        return peak_gpu_memory_mib, available_gpu_memory_mib, gpu_utilization

    # Simplification: We assume that whatever devices are available at the
    # start of benchmarking stays the same throughout the run. If someone is
    # hotplugging GPUs during a benchmark this may not be true.
    all_devices = list(gpu_metrics[0].keys())
    if not all_devices:
        logger.warning("No GPUs found, so there are no GPU stats to report")
        return peak_gpu_memory_mib, available_gpu_memory_mib, gpu_utilization

    bytes_per_mib = 1024 * 1024
    for device_name in all_devices:
        peak_gpu_memory_mib.append(
            max(
                snapshot[device_name].memory.used_bytes
                for snapshot in gpu_metrics
            )
            / bytes_per_mib
        )
        available_gpu_memory_mib.append(
            min(
                snapshot[device_name].memory.free_bytes
                for snapshot in gpu_metrics
            )
            / bytes_per_mib
        )
        gpu_utilization.append(
            statistics.mean(
                snapshot[device_name].utilization.gpu_usage_percent
                for snapshot in gpu_metrics
            )
        )

    return peak_gpu_memory_mib, available_gpu_memory_mib, gpu_utilization


def _per_turn_cache_retentions(
    measured_outputs: list[RequestFuncOutput], block_size: int
) -> list[float]:
    """Per-turn KV cache retention across multi-turn sessions.

    For each session (grouped by ``session_id``, ordered by ``turn_index``) and
    each turn ``N >= 2``, the previous turn's full context (prompt + completion)
    is an exact prefix of turn ``N``'s prompt and should be served from the
    prefix cache. We compare the server-reported ``cached_tokens`` of turn ``N``
    against the block-aligned ceiling of that prior context — full
    ``block_size`` blocks minus the in-progress block the engine never reuses —
    so a clean hit reads ~1.0 and a real drop (eviction, or a turn routed to a
    cold replica) reads low. Single-turn requests contribute nothing.

    Returns the list of per-turn retention fractions (one per checked turn).
    """
    by_session: dict[str, list[RequestFuncOutput]] = {}
    for o in measured_outputs:
        if o.session_id is None or o.turn_index is None:
            continue
        by_session.setdefault(o.session_id, []).append(o)

    retentions: list[float] = []
    for turns in by_session.values():
        turns.sort(key=lambda o: o.turn_index or 0)
        for prev, cur in pairwise(turns):
            prev_prompt = prev.server_token_stats.prompt_tokens or 0
            prev_completion = prev.server_token_stats.completion_tokens or 0
            expected_prefix = prev_prompt + prev_completion
            expected_blocks = max(0, expected_prefix // block_size - 1)
            expected_cacheable = expected_blocks * block_size
            if expected_cacheable <= 0:
                continue
            cached = cur.server_token_stats.cached_tokens
            retentions.append(min(1.0, cached / expected_cacheable))
    return retentions


def _build_request_records(
    outputs: Sequence[RequestFuncOutput],
    tokenizer: PreTrainedTokenizerBase,
    measured_ids: AbstractSet[int],
    *,
    record_request_text: bool,
) -> list[RequestRecord]:
    """One record per request, in dispatch order.

    Every request gets a record — failed and cancelled included — so a
    reader can tell a request that produced nothing from one that is
    absent. ``measured_ids`` holds ``id()`` of the outputs the aggregates
    were computed over, which is how a record says whether the trim or
    steady-state window kept it; identity is the right key because the
    measured list holds the same objects.
    """
    records: list[RequestRecord] = []
    for index, o in enumerate(outputs):
        counted = o.success and not o.cancelled
        output_len = compute_output_len(tokenizer, o) if counted else 0
        # Undefined below two tokens: there is no inter-token interval to
        # average, and reporting 0.0 would pull a mean toward a value no
        # request achieved.
        tpot_ms = (
            (o.latency - o.ttft) / (output_len - 1) * 1000.0
            if counted and output_len > 1
            else None
        )
        records.append(
            RequestRecord(
                index=index,
                prompt_len=o.prompt_len,
                output_len=output_len,
                success=o.success,
                cancelled=o.cancelled,
                measured=id(o) in measured_ids,
                error=o.error,
                ttft_ms=o.ttft * 1000.0 if counted else None,
                tpot_ms=tpot_ms,
                latency_ms=o.latency * 1000.0 if counted else None,
                submit_time=o.request_submit_time,
                complete_time=o.request_complete_time,
                session_id=o.session_id,
                turn_index=o.turn_index,
                generated_text=o.generated_text
                if record_request_text
                else None,
            )
        )
    return records


def calculate_metrics(
    outputs: Sequence[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    gpu_metrics: list[dict[str, GPUStats]] | None,
    cpu_metrics: CPUMetrics | None,
    skip_first_n_requests: int,
    skip_last_n_requests: int,
    max_concurrency: int | None,
    max_concurrent_conversations: int | None,
    collect_gpu_stats: bool,
    kv_block_size: int,
    metrics_by_endpoint: Mapping[str, ParsedMetrics] | None = None,
    *,
    reject_outliers: bool = False,
    record_request_text: bool = False,
    all_outputs: Sequence[RequestFuncOutput] | None = None,
) -> BenchmarkResult:
    actual_output_lens: list[int] = []
    failures = 0
    failed_responses: list[RequestFuncOutput] = []
    itls: list[float] = []
    tpots: list[float] = []
    step_tpots: list[float] = []
    ttfts: list[float] = []
    latencies: list[float] = []
    input_throughputs: list[float] = []
    output_throughputs: list[float] = []
    per_turn_cached_token_rates: list[float] = []
    total_server_cached_tokens: int = 0
    total_server_prompt_tokens: int = 0

    successful: list[tuple[RequestFuncOutput, int]] = []
    for o in outputs:
        if o.cancelled:
            continue
        if o.success:
            successful.append((o, compute_output_len(tokenizer, o)))
        else:
            actual_output_lens.append(0)
            failures += 1
            failed_responses.append(o)

    total_successful = len(successful)

    for _, output_len in successful:
        actual_output_lens.append(output_len)

    # Pick head / tail to drop using request timing rather than dispatch
    # order. For multi-turn the flat list arrives in
    # ``[session0_turns, session1_turns, ...]`` block order, so a
    # dispatch-order slice would silently target the wrong requests.
    # Sorting by submit time for the head and complete time for the tail
    # gives "first N sent, last N completed" uniformly across single-turn
    # and multi-turn flows.
    head_drop_ids: set[int] = set()
    tail_drop_ids: set[int] = set()
    if skip_first_n_requests > 0:
        by_submit = sorted(
            successful,
            key=lambda pair: pair[0].request_submit_time or 0.0,
        )
        head_drop_ids = {
            id(pair[0]) for pair in by_submit[:skip_first_n_requests]
        }
    if skip_last_n_requests > 0:
        by_complete = sorted(
            successful,
            key=lambda pair: pair[0].request_complete_time or 0.0,
        )
        tail_drop_ids = {
            id(pair[0]) for pair in by_complete[-skip_last_n_requests:]
        }
    measured = [
        pair
        for pair in successful
        if id(pair[0]) not in head_drop_ids and id(pair[0]) not in tail_drop_ids
    ]

    # Aggregate token/chunk stats over the measured slice only. Skipped
    # warmup/tail requests contribute neither their tokens nor their wall
    # time to throughput metrics, so TPM-style numbers reflect the
    # intended steady-state portion of the run.
    total_input_client_calculated = 0
    total_output = 0
    nonempty_response_chunks = 0
    max_input = 0
    max_output = 0
    max_total = 0
    for o, output_len in measured:
        total_input_client_calculated += o.prompt_len
        total_output += output_len
        nonempty_response_chunks += 1 if o.ttft != 0 else 0
        nonempty_response_chunks += len(o.itl)
        max_input = max(max_input, o.prompt_len)
        max_output = max(max_output, output_len)
        max_total = max(max_total, o.prompt_len + output_len)

        if output_len > 1:
            tpots.append((o.latency - o.ttft) / (output_len - 1))
        step_tpots += o.tpot
        itls += o.itl
        ttfts.append(o.ttft)
        if o.ttft > 0:
            input_throughputs.append(o.prompt_len / o.ttft)
        if (o.latency - o.ttft) > 0:
            output_throughputs.append((output_len - 1) / (o.latency - o.ttft))
        latencies.append(o.latency)
        if o.server_token_stats.prompt_tokens:
            per_turn_cached_token_rates.append(
                o.server_token_stats.cached_tokens
                / o.server_token_stats.prompt_tokens
            )
            total_server_cached_tokens += o.server_token_stats.cached_tokens
            total_server_prompt_tokens += o.server_token_stats.prompt_tokens

    # Records span every request the run dispatched, not the window the
    # aggregates were computed over: the steady-state path passes a
    # filtered slice as ``outputs``, and indexing that would renumber the
    # requests and destroy the key two runs pair on.
    measured_ids = {id(pair[0]) for pair in measured}
    request_records = _build_request_records(
        outputs if all_outputs is None else all_outputs,
        tokenizer,
        measured_ids,
        record_request_text=record_request_text,
    )

    if not measured:
        total_input = 0
    elif total_server_prompt_tokens == 0:
        warnings.warn(
            "Server did not report prompt_tokens; using client-calculated token"
            " counts. Input token count and cache rate metrics may not be accurate.",
            stacklevel=2,
        )
        total_input = total_input_client_calculated
    else:
        discrepancy = (
            abs(total_server_prompt_tokens - total_input_client_calculated)
            / total_input_client_calculated
        )
        if discrepancy > _INPUT_TOKEN_DISCREPANCY_THRESHOLD:
            warnings.warn(
                f"Server-reported total input tokens ({total_server_prompt_tokens})"
                f" differs from client-calculated count ({total_input_client_calculated})"
                f" by {discrepancy:.1%}. Using server-reported value.",
                stacklevel=2,
            )
        total_input = total_server_prompt_tokens
        logger.info(
            "Using server-reported prompt_tokens for total input token count."
        )

    _warn_on_request_failures(
        outputs=outputs,
        completed=total_successful,
        failures=failures,
        failed_responses=failed_responses,
    )

    measured_count = len(measured)
    if measured_count == 0 and total_successful > 0:
        warnings.warn(
            (
                f"All {total_successful} successful requests were excluded"
                f" by skip_first_n_requests={skip_first_n_requests} (first"
                f" submitted) and skip_last_n_requests={skip_last_n_requests}"
                " (last completed). Consider running a longer benchmark."
            ),
            stacklevel=2,
        )
    elif 0 < measured_count < 10:
        warnings.warn(
            (
                f"Only {measured_count} requests remain after skipping the"
                f" first {skip_first_n_requests} submitted and last"
                f" {skip_last_n_requests} completed."
                " Results may not be reliable."
                " Consider running a longer benchmark."
            ),
            stacklevel=2,
        )

    # Duration over the measured window: first measured submit to last
    # measured complete. Mirrors the steady-state block's window math so
    # skipped warmup/tail wall time does not pollute throughput.
    measured_duration = measured_window_duration(
        (o for o, _ in measured), fallback=dur_s
    )

    (
        peak_gpu_memory_mib,
        available_gpu_memory_mib,
        gpu_utilization,
    ) = _aggregate_gpu_stats(
        collect_gpu_stats=collect_gpu_stats,
        gpu_metrics=gpu_metrics,
    )

    global_cached_token_rate: float = (
        total_server_cached_tokens / total_input if total_input > 0 else 0.0
    )
    per_turn_cached_token_rate: RatePercentileMetrics | None = (
        RatePercentileMetrics(per_turn_cached_token_rates, as_percent=True)
        if len(per_turn_cached_token_rates) > 0
        else None
    )

    # Turn-by-turn KV cache retention: how much of the previous turn's context
    # is still cached on the next turn (catches cached-token drop). Only
    # meaningful for multi-turn sessions; empty for single-turn workloads.
    per_turn_cache_retentions = _per_turn_cache_retentions(
        [o for o, _ in measured], kv_block_size
    )
    per_turn_cache_retention: RatePercentileMetrics | None = (
        RatePercentileMetrics(per_turn_cache_retentions, as_percent=True)
        if len(per_turn_cache_retentions) > 0
        else None
    )

    # Per-request outlier rejection via Iglewicz-Hoaglin modified z-score.
    # Applied independently to each latency series so one noisy metric
    # doesn't mask a clean one. Only active when the caller opts in
    # (steady-state window path); the default head/tail trim path leaves
    # all series unchanged.
    if reject_outliers:
        ttft_keep = reject_metric_outliers(ttfts)
        tpot_keep = reject_metric_outliers(tpots)
        itl_keep = reject_metric_outliers(itls)
        latency_keep = reject_metric_outliers(latencies)
        ttfts = [v for v, k in zip(ttfts, ttft_keep, strict=False) if k]
        tpots = [v for v, k in zip(tpots, tpot_keep, strict=False) if k]
        itls = [v for v, k in zip(itls, itl_keep, strict=False) if k]
        latencies = [
            v for v, k in zip(latencies, latency_keep, strict=False) if k
        ]

    text_data = TextGenAggregates(
        duration=measured_duration,
        completed=measured_count,
        failures=failures,
        request_throughput=measured_count / measured_duration,
        excluded_successful=total_successful - measured_count,
        # A metric is ``None`` when it has no samples (empty data list),
        # rather than a NaN-filled placeholder: NaN serializes to JSON
        # ``null`` (both via ``model_dump_json`` and BigQuery ingest),
        # which strict consumers reject. ``None`` is the honest "no data"
        # representation and renders as an empty cell downstream.
        latency_ms=StandardPercentileMetrics(
            latencies, scale_factor=1000.0, unit="ms"
        )
        if latencies
        else None,
        errors=[o.error for o in outputs],
        request_submit_times=[o.request_submit_time for o in outputs],
        request_complete_times=[o.request_complete_time for o in outputs],
        total_input=total_input,
        total_output=total_output,
        nonempty_response_chunks=nonempty_response_chunks,
        max_concurrent_conversations=max_concurrent_conversations,
        # Use specialized metric classes that handle percentile calculations automatically
        input_throughput=ThroughputMetrics(input_throughputs, unit="tok/s")
        if input_throughputs
        else None,
        output_throughput=ThroughputMetrics(output_throughputs, unit="tok/s")
        if output_throughputs
        else None,
        ttft_ms=StandardPercentileMetrics(ttfts, scale_factor=1000.0, unit="ms")
        if ttfts
        else None,
        tpot_ms=StandardPercentileMetrics(tpots, scale_factor=1000.0, unit="ms")
        if tpots
        else None,
        step_tpot_ms=StandardPercentileMetrics(
            step_tpots, scale_factor=1000.0, unit="ms"
        )
        if step_tpots
        else None,
        itl_ms=StandardPercentileMetrics(itls, scale_factor=1000.0, unit="ms")
        if itls
        else None,
        max_input=max_input,
        max_output=max_output,
        max_total=max_total,
        global_cached_token_rate=global_cached_token_rate,
        per_turn_cached_token_rate=per_turn_cached_token_rate,
        per_turn_cache_retention=per_turn_cache_retention,
        skip_first_n_requests=skip_first_n_requests,
        skip_last_n_requests=skip_last_n_requests,
        input_lens=[o.prompt_len for o in outputs],
        output_lens=actual_output_lens,
        ttfts=[o.ttft for o in outputs],
        request_records=request_records,
        per_turn_cached_token_rates=per_turn_cached_token_rates,
        per_turn_cache_retentions=per_turn_cache_retentions,
    )

    return BenchmarkResult(
        task_type="text",
        max_concurrency=max_concurrency or len(outputs),
        peak_gpu_memory_mib=peak_gpu_memory_mib,
        available_gpu_memory_mib=available_gpu_memory_mib,
        gpu_utilization=gpu_utilization,
        cpu_metrics=cpu_metrics,
        metrics_by_endpoint=metrics_by_endpoint or {},
        text_data=text_data,
    )


def calculate_pixel_generation_metrics(
    outputs: Sequence[PixelGenerationRequestFuncOutput],
    dur_s: float,
    gpu_metrics: list[dict[str, GPUStats]] | None,
    cpu_metrics: CPUMetrics | None,
    max_concurrency: int | None,
    collect_gpu_stats: bool,
    metrics_by_endpoint: Mapping[str, ParsedMetrics] | None = None,
) -> BenchmarkResult:
    completed = 0
    failures = 0
    latencies: list[float] = []
    total_generated_outputs = 0
    failed_responses: list[PixelGenerationRequestFuncOutput] = []
    successful: list[PixelGenerationRequestFuncOutput] = []

    for output in outputs:
        if output.cancelled:
            continue
        if output.success:
            completed += 1
            latencies.append(output.latency)
            total_generated_outputs += output.num_generated_outputs
            successful.append(output)
        else:
            failures += 1
            failed_responses.append(output)

    _warn_on_request_failures(
        outputs=outputs,
        completed=completed,
        failures=failures,
        failed_responses=failed_responses,
    )
    (
        peak_gpu_memory_mib,
        available_gpu_memory_mib,
        gpu_utilization,
    ) = _aggregate_gpu_stats(
        collect_gpu_stats=collect_gpu_stats,
        gpu_metrics=gpu_metrics,
    )

    # Use the first-submit -> last-complete window so setup/teardown
    # around the actual requests doesn't inflate the denominator.
    measured_duration = measured_window_duration(successful, fallback=dur_s)

    pixel_data = PixelGenAggregates(
        duration=measured_duration,
        completed=completed,
        failures=failures,
        request_throughput=completed / measured_duration,
        # ``None`` when no requests succeeded (see the text-gen path).
        latency_ms=StandardPercentileMetrics(
            latencies, scale_factor=1000.0, unit="ms"
        )
        if latencies
        else None,
        errors=[o.error for o in outputs],
        request_submit_times=[o.request_submit_time for o in outputs],
        request_complete_times=[o.request_complete_time for o in outputs],
        total_generated_outputs=total_generated_outputs,
        latencies=[o.latency for o in outputs],
        num_generated_outputs=[o.num_generated_outputs for o in outputs],
    )

    return BenchmarkResult(
        task_type="pixel",
        max_concurrency=max_concurrency or len(outputs),
        peak_gpu_memory_mib=peak_gpu_memory_mib,
        available_gpu_memory_mib=available_gpu_memory_mib,
        gpu_utilization=gpu_utilization,
        cpu_metrics=cpu_metrics,
        metrics_by_endpoint=metrics_by_endpoint or {},
        pixel_data=pixel_data,
    )


def _is_pixel_generation_outputs(
    outputs: Sequence[BaseRequestFuncOutput],
) -> TypeGuard[Sequence[PixelGenerationRequestFuncOutput]]:
    return all(
        isinstance(output, PixelGenerationRequestFuncOutput)
        for output in outputs
    )


def _is_text_generation_outputs(
    outputs: Sequence[BaseRequestFuncOutput],
) -> TypeGuard[Sequence[RequestFuncOutput]]:
    return all(isinstance(output, RequestFuncOutput) for output in outputs)


def build_pixel_generation_result(
    *,
    outputs: Sequence[BaseRequestFuncOutput],
    benchmark_duration: float,
    gpu_metrics: list[dict[str, GPUStats]] | None,
    cpu_metrics: CPUMetrics | None,
    max_concurrency: int | None,
    collect_gpu_stats: bool,
    metrics_by_endpoint: Mapping[str, ParsedMetrics] | None = None,
) -> BenchmarkResult:
    """Compute metrics and build the result dict for pixel-generation tasks."""
    if not _is_pixel_generation_outputs(outputs):
        raise TypeError(
            "Expected all outputs to be PixelGenerationRequestFuncOutput"
            " in pixel-generation benchmark flow."
        )
    metrics = calculate_pixel_generation_metrics(
        outputs=outputs,
        dur_s=benchmark_duration,
        gpu_metrics=gpu_metrics,
        cpu_metrics=cpu_metrics,
        max_concurrency=max_concurrency,
        collect_gpu_stats=collect_gpu_stats,
        metrics_by_endpoint=metrics_by_endpoint,
    )
    return metrics


def build_text_generation_result(
    *,
    outputs: Sequence[BaseRequestFuncOutput],
    benchmark_duration: float,
    tokenizer: PreTrainedTokenizerBase | None,
    gpu_metrics: list[dict[str, GPUStats]] | None,
    cpu_metrics: CPUMetrics | None,
    skip_first_n_requests: int,
    skip_last_n_requests: int,
    max_concurrency: int | None,
    max_concurrent_conversations: int | None,
    collect_gpu_stats: bool,
    metrics_by_endpoint: Mapping[str, ParsedMetrics] | None = None,
    spec_decode_stats: SpecDecodeStats | None = None,
    kv_block_size: int = 128,
    record_request_text: bool = False,
) -> BenchmarkResult:
    """Compute metrics and build the result for text-generation tasks.

    Uses a single reported metric set selected by the following strategy:

    - Run MAD-based steady-state detection (``detect_steady_state``).
    - If detection succeeds: compute metrics over the detected steady-state
      window with ``skip_first=0``, ``skip_last=0``, and per-request
      outlier rejection via the Iglewicz-Hoaglin modified z-score
      (``reject_outliers=True``). Duration is the wall-clock span of the
      window (first submit to last complete of the window requests).
    - If detection is skipped (concurrency==1) or fails: fall back to the
      full-run path with the caller-supplied head/tail trim and no outlier
      rejection. This guarantees identical behavior to the previous
      implementation for non-steady-state runs.

    Lightweight detection diagnostics (``steady_state_detected``,
    ``steady_state_window_count``, ``steady_state_mode``,
    ``steady_state_warning``, ``num_outliers_rejected``) are attached to
    the returned ``BenchmarkResult`` for observability without duplicating
    the full metric set.
    """
    if not _is_text_generation_outputs(outputs):
        raise TypeError(
            "Expected all outputs to be RequestFuncOutput"
            " in text-generation benchmark flow."
        )

    steady = detect_steady_state(outputs, max_concurrency=max_concurrency)

    # Persist detection mode for downstream consumers; None when detection
    # was skipped (concurrency=1) so the default "full" isn't mistaken for
    # a real result.
    mode: str | None = (
        steady.mode if (steady.detected or steady.warning is not None) else None
    )

    num_outliers_rejected = 0

    if steady.detected:
        # Build the steady-state sub-list and compute its wall-clock span.
        ss_index_set = set(steady.steady_state_indices)
        ss_outputs = [
            out
            for i, out in enumerate(outputs)
            if i in ss_index_set and out.success and not out.cancelled
        ]
        ss_valid = [
            out
            for out in ss_outputs
            if out.request_submit_time is not None
            and out.request_complete_time is not None
        ]

        if len(ss_valid) >= 2:
            ss_valid.sort(key=lambda o: o.request_submit_time or 0.0)
            first_submit = ss_valid[0].request_submit_time
            last_complete = ss_valid[-1].request_complete_time
            assert first_submit is not None and last_complete is not None
            ss_duration = max(last_complete - first_submit, 1e-9)
        else:
            ss_duration = benchmark_duration

        # Log detection details before calling calculate_metrics so any
        # downstream warnings appear after the info line.
        assert steady.start_index is not None and steady.end_index is not None
        dispatch_span = steady.end_index - steady.start_index
        span_note = (
            f" spans {dispatch_span} positions"
            if dispatch_span != steady.steady_state_count
            else ""
        )
        mode_note = (
            " [TTFT-only fallback; TPOT absent across run]"
            if steady.mode == "ttft_only"
            else ""
        )
        logger.info(
            "Post-run steady-state analysis (informational, reported as"
            f" ss_* metrics): window of {steady.steady_state_count} valid"
            f" requests (dispatch range [{steady.start_index},"
            f" {steady.end_index}){span_note};"
            f" {steady.total_requests} total valid in the run)"
            f"{mode_note}"
        )

        text_metrics = calculate_metrics(
            outputs=ss_outputs,
            dur_s=ss_duration,
            tokenizer=tokenizer,
            gpu_metrics=gpu_metrics,
            cpu_metrics=cpu_metrics,
            skip_first_n_requests=0,
            skip_last_n_requests=0,
            max_concurrency=max_concurrency,
            max_concurrent_conversations=max_concurrent_conversations,
            collect_gpu_stats=collect_gpu_stats,
            metrics_by_endpoint=metrics_by_endpoint,
            kv_block_size=kv_block_size,
            reject_outliers=True,
            record_request_text=record_request_text,
            all_outputs=outputs,
        )

        # Diagnostic: number of per-request TTFT outliers rejected in the
        # steady-state window. TTFT is the headline steady-state metric and
        # is one value per request, so this count is bounded by the request
        # count and directly interpretable (a step-level count over per-chunk
        # ITL/TPOT series would be far larger than the request count and
        # misleading as a headline diagnostic).
        _active = [o for o in ss_outputs if o.success and not o.cancelled]
        _ss_ttfts = [o.ttft for o in _active if o.ttft is not None]
        num_outliers_rejected = sum(
            1 for keep in reject_metric_outliers(_ss_ttfts) if not keep
        )

    else:
        # Detection did not run or did not converge. Only surface this at
        # WARNING level when the run actually produced successful requests.
        # When there are none (a serving/run failure, e.g. the "0 of 10000
        # valid" case in PERF-2615), the steady-state detector is the wrong
        # messenger: the failure is already reflected in the failure count and
        # the reported metrics, so a "too few valid requests" warning is
        # misleading noise on top of an already-failed run. Genuine cases with
        # some successful-but-unusable requests still warn.
        n_success = sum(1 for o in outputs if o.success and not o.cancelled)
        if steady.warning and n_success > 0:
            logger.warning(f"Steady-state detection: {steady.warning}")
        elif n_success == 0:
            logger.info(
                "Steady-state detection skipped: run produced no successful"
                " requests; reporting full-run metrics over all outputs."
            )

        text_metrics = calculate_metrics(
            outputs=outputs,
            dur_s=benchmark_duration,
            tokenizer=tokenizer,
            gpu_metrics=gpu_metrics,
            cpu_metrics=cpu_metrics,
            skip_first_n_requests=skip_first_n_requests,
            skip_last_n_requests=skip_last_n_requests,
            max_concurrency=max_concurrency,
            max_concurrent_conversations=max_concurrent_conversations,
            collect_gpu_stats=collect_gpu_stats,
            metrics_by_endpoint=metrics_by_endpoint,
            kv_block_size=kv_block_size,
            reject_outliers=False,
            record_request_text=record_request_text,
        )

    for warn in text_metrics.confidence_warnings():
        logger.warning(f"Confidence: {warn}")

    return text_metrics.model_copy(
        update={
            "steady_state_detected": steady.detected,
            "steady_state_window_count": steady.steady_state_count,
            "steady_state_mode": mode,
            "steady_state_warning": steady.warning,
            "num_outliers_rejected": num_outliers_rejected,
            "spec_decode_stats": spec_decode_stats,
            "task_type": "text",
        }
    )
