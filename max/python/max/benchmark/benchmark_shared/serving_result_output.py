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

"""Result output helpers for serving benchmarks.

Printing, formatting, JSON/YAML persistence, and related utilities.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import statistics
from collections.abc import Sequence
from datetime import datetime
from typing import NamedTuple

import numpy as np
import yaml
from max.benchmark.benchmark_shared.config import (
    Backend,
    BenchmarkTask,
    ServingBenchmarkConfig,
)
from max.benchmark.benchmark_shared.datasets.chat_judge import (
    ChatJudgeChatSamples,
)
from max.benchmark.benchmark_shared.datasets.types import (
    ChatSamples,
    RequestSamples,
    Samples,
)
from max.benchmark.benchmark_shared.lora_benchmark_manager import (
    LoRABenchmarkManager,
)
from max.benchmark.benchmark_shared.metrics import (
    BenchmarkResult,
    LoRAMetrics,
    PercentileMetrics,
    StandardPercentileMetrics,
    ThroughputMetrics,
)
from max.benchmark.benchmark_shared.percentile_metrics import (
    json_safe,
)
from max.benchmark.benchmark_shared.server_metrics import print_server_metrics
from max.benchmark.benchmark_shared.utils import print_section
from tabulate import tabulate

logger = logging.getLogger(__name__)


def hash_string(s: str) -> str:
    """Hash a string using SHA-256. This is stable and deterministic across runs.

    hexdigest is a 64-character string of hexadecimal digits. We only return the
    first 8 characters to keep the output concise.
    """
    return hashlib.sha256(s.encode()).hexdigest()[:8]


def elide_data_uris_in_string(data_uri: str) -> str:
    """Elides the base64 data URIs parts of the string.

    Eg: elide_data_uris_in_string("'image': 'data:image/jpeg;base64,/9j/4AAQSASDEEAE'")
                               -> "'image': 'data:image/jpeg;base64,...(hash: 783e7013, 16 bytes)...'"
    """

    def _match_replacer(m: re.Match[str]) -> str:
        uri_prefix = m.group(1)
        uri_data = m.group(2)
        return f"{uri_prefix}...(hash: {hash_string(uri_data)}, {len(uri_data)} bytes)..."

    return re.sub(
        r"(data:[a-z/]+;base64,)([A-Za-z0-9+/=]+)",
        _match_replacer,
        data_uri,
    )


def print_input_prompts(samples: Samples) -> None:
    """Helper function to print input prompts."""
    if isinstance(samples, ChatSamples):
        raise NotImplementedError(
            "Printing out multi-turn chats is not supported."
        )

    print("Input prompts:")
    for req_id, request in enumerate(samples.requests):
        prompt_info = {
            "req_id": req_id,
            "output_len": request.output_len,
            "prompt_len": request.prompt_len,
            "prompt": request.prompt_formatted,
            "encoded_images": request.encoded_images,
        }
        # We turn the entire prompt_info dict into a string and then elide the
        # data URIs. The alternative approach of only applying the transformation
        # to a stringified version of the `request.prompt_formatted` field will
        # lead to double-escaping of special characters which is not desirable.
        print(elide_data_uris_in_string(str(prompt_info)))


def _format_distribution_table(
    values: Sequence[float | int], label: str
) -> str:
    """Format distribution statistics as a mini-table with header and value rows."""
    _STAT_COLUMNS = (
        "min",
        "max",
        "mean",
        "std",
        "p5",
        "p25",
        "p50",
        "p75",
        "p95",
        "p99",
    )
    _COL_WIDTH = 10

    arr = np.array(values, dtype=float)
    p5, p25, p50, p75, p95, p99 = np.percentile(arr, [5, 25, 50, 75, 95, 99])
    stat_values = (
        np.min(arr),
        np.max(arr),
        np.mean(arr),
        np.std(arr),
        p5,
        p25,
        p50,
        p75,
        p95,
        p99,
    )
    header = "    " + "".join(f"{name:<{_COL_WIDTH}}" for name in _STAT_COLUMNS)
    row = "    " + "".join(f"{v:<{_COL_WIDTH}.2f}" for v in stat_values)
    return f"  {label}:\n{header}\n{row}"


def print_workload_stats(samples: Samples) -> None:
    """Print workload distribution statistics and exit.

    For single-turn workloads, prints input/output length stats.
    For multi-turn workloads, additionally prints num_turns and
    delay_until_next_message stats.
    """
    print_section(title=" Workload Statistics ", char="=")

    if isinstance(samples, RequestSamples):
        input_lens = [r.prompt_len for r in samples.requests]
        output_lens = [
            r.output_len for r in samples.requests if r.output_len is not None
        ]

        print(f"  {'Total requests:':<30} {len(samples.requests)}")
        print()
        print(_format_distribution_table(input_lens, "Input length"))
        if output_lens:
            print()
            print(_format_distribution_table(output_lens, "Output length"))
        else:
            print()
            print("  Output length:  not specified (server-determined)")

    elif isinstance(samples, ChatSamples):
        sessions = samples.chat_sessions
        num_turns_list = [s.num_turns for s in sessions]

        all_input_lens: list[int] = []
        all_output_lens: list[int] = []
        all_delays: list[float] = []
        sessions_with_system = 0
        system_prompt_tokens: list[int] = []

        is_chat_judge = isinstance(samples, ChatJudgeChatSamples)
        for session in sessions:
            current_context_length = 0
            session_system_tokens = 0
            for msg in session.messages:
                if msg.source == "system":
                    sessions_with_system += 1
                    system_prompt_tokens.append(msg.num_tokens)
                    session_system_tokens = msg.num_tokens
                elif is_chat_judge:
                    # chat-judge sends [system?, user_i] per turn without
                    # accumulating prior turns, so per-turn input length is
                    # system_prompt + this user turn's content.
                    all_input_lens.append(
                        session_system_tokens + msg.num_tokens
                    )
                    if msg.delay_until_next_message is not None:
                        all_delays.append(msg.delay_until_next_message)
                else:
                    current_context_length += msg.num_tokens
                    if msg.source == "user":
                        all_input_lens.append(current_context_length)
                    else:
                        all_output_lens.append(msg.num_tokens)
                    if msg.delay_until_next_message is not None:
                        all_delays.append(msg.delay_until_next_message)

        total_prefix = sum(s.prefix_turns for s in sessions)
        total_turns = sum(num_turns_list)
        print(f"  {'Total sessions:':<30} {len(sessions)}")
        if sessions_with_system:
            avg_sys = sum(system_prompt_tokens) / sessions_with_system
            print(
                f"  {'Sessions with explicit system role:':<30}"
                f" {sessions_with_system} / {len(sessions)}"
                f" (avg {avg_sys:.1f} tokens)"
            )
        if total_prefix > 0:
            print(
                f"  {'Total turns (measured):':<30}"
                f" {total_turns - total_prefix}"
            )
            print(f"  {'Total turns (warmup):':<30} {total_prefix}")
        else:
            print(f"  {'Total turns (across all):':<30} {total_turns}")
        print()
        print(
            _format_distribution_table(
                all_input_lens, "Input length (per turn)"
            )
        )
        print()
        if all_output_lens:
            print(
                _format_distribution_table(
                    all_output_lens, "Output length (per turn)"
                )
            )
        else:
            print("  Output length (per turn):  none recorded")
        print()
        print(
            _format_distribution_table(num_turns_list, "Num turns per session")
        )
        if all_delays:
            print()
            print(
                _format_distribution_table(
                    all_delays, "Delay between chat turns (ms)"
                )
            )
        else:
            print()
            print("  Delay until next msg:  none configured")

    print("=" * 50)


def print_lora_benchmark_results(metrics: LoRAMetrics) -> None:
    """Print LoRA benchmark statistics if available."""

    print_section(title=" LoRA Adapter Benchmark Results ", char="=")
    print("{:<40} {:<10}".format("Total LoRA loads:", metrics.total_loads))
    print("{:<40} {:<10}".format("Total LoRA unloads:", metrics.total_unloads))

    def print_float_metric(metric: str, value: float) -> None:
        print(f"{metric:<40} {value:<10.2f}")

    def print_action_section(action: str, times_ms: list[float]) -> None:
        if not times_ms:
            return
        print_section(title=f"LoRA {action.title()} Times")
        print_float_metric(f"Mean {action} time:", statistics.mean(times_ms))
        print_float_metric(
            f"Median {action} time:", statistics.median(times_ms)
        )
        print_float_metric(f"Min {action} time:", min(times_ms))
        print_float_metric(f"Max {action} time:", max(times_ms))
        if len(times_ms) > 1:
            print_float_metric(
                f"Std dev {action} time:", statistics.stdev(times_ms)
            )

    print_action_section("load", metrics.load_times_ms)
    print_action_section("unload", metrics.unload_times_ms)


# Metric containers exposing the six percentile fields consumed by the summary
# table. ``StandardPercentileMetrics`` (and its ``RatePercentileMetrics``
# subclass) and ``ThroughputMetrics`` delegate the fields to an inner
# ``PercentileMetrics`` via ``__getattr__``.
_PercentileLike = (
    PercentileMetrics | StandardPercentileMetrics | ThroughputMetrics
)


class PercentileRow(NamedTuple):
    """A labeled metric to render as one row of the percentile summary table."""

    label: str
    metrics: _PercentileLike


def format_gpu_statistics_title(n_devices: int) -> str:
    """Section title stating GPU rows are derived across the sampled devices."""
    assert n_devices > 0, "GPU statistics title requires at least one device"
    device_label = "device" if n_devices == 1 else "devices"
    return f"GPU Statistics (aggregated across {n_devices} {device_label})"


def format_percentile_table(rows: Sequence[PercentileRow]) -> str:
    """Render per-metric percentile stats as one table, one row per metric."""
    headers = ["Metric", "Mean", "Std", "P50", "P90", "P95", "P99"]
    table = [
        [
            row.label,
            f"{row.metrics.mean:.2f}",
            f"{row.metrics.std:.2f}",
            f"{row.metrics.p50:.2f}",
            f"{row.metrics.p90:.2f}",
            f"{row.metrics.p95:.2f}",
            f"{row.metrics.p99:.2f}",
        ]
        for row in rows
    ]
    return tabulate(
        table,
        headers=headers,
        tablefmt="simple_outline",
        colalign=("left", *("right",) * (len(headers) - 1)),
        disable_numparse=True,
    )


def print_benchmark_summary(
    metrics: BenchmarkResult,
    request_rate: float,
    max_concurrency: int | None,
    achieved_request_rate: float,
    collect_gpu_stats: bool,
    collect_cpu_stats: bool,
    lora_manager: LoRABenchmarkManager | None = None,
) -> None:
    """Print benchmark summary for text-generation and pixel-generation."""
    groups = metrics.result_groups
    assert groups is not None, (
        "print_benchmark_summary called before result_groups was populated"
    )
    summary = groups.summary
    assert metrics.aggregates is not None, (
        "print_benchmark_summary called on a metrics record with no aggregates"
    )
    # Summary scalars are filled from aggregates by ``build_result_groups``.
    assert summary.completed is not None
    assert summary.failures is not None
    assert summary.duration is not None
    assert summary.request_throughput is not None

    print_section(title=" Serving Benchmark Result ", char="=")
    print("{:<40} {:<10}".format("Successful requests:", summary.completed))
    print("{:<40} {:<10}".format("Failed requests:", summary.failures))
    print(
        "{:<40} {:<10.2f}".format("Benchmark duration (s):", summary.duration)
    )
    if metrics.text_data is not None:
        t = metrics.text_data
        print("{:<40} {:<10}".format("Total input tokens:", t.total_input))
        print("{:<40} {:<10}".format("Total generated tokens:", t.total_output))
        # We found that response chunks can be empty in content and the token number
        # can be different with the re-tokenization in one pass or chunk-by-chunk.
        # Let's count the number of nonempty_response_chunks for all serving backends.
        # With the move to zero-overhead single step scheduling, this should generally
        # exactly match the number of requested output tokens.
        print(
            "{:<40} {:<10}".format(
                "Total nonempty serving response chunks:",
                t.nonempty_response_chunks,
            )
        )
    elif summary.total_generated_outputs is not None:
        print(
            "{:<40} {:<10}".format(
                "Total generated outputs:",
                summary.total_generated_outputs,
            )
        )
    offline_benchmark = math.isinf(request_rate) and max_concurrency is None
    print(
        "{:<40} {:<10.5f}".format(
            "Input request rate (req/s):",
            float("inf") if offline_benchmark else achieved_request_rate,
        )
    )
    print(
        "{:<40} {:<10.5f}".format(
            "Request throughput (req/s):", summary.request_throughput
        )
    )
    print("{:<40} {:<10}".format("Max Concurrency:", summary.max_concurrency))
    if (
        metrics.text_data is not None
        and metrics.text_data.max_concurrent_conversations is not None
    ):
        print(
            "{:<40} {:<10}".format(
                "Max Concurrent Conversations:",
                metrics.text_data.max_concurrent_conversations,
            )
        )

    if summary.aggregate_tokens_per_minute is not None:
        print(
            "{:<40} {:<10.2f}".format(
                "Total TPM (input+output, whole bench):",
                summary.aggregate_tokens_per_minute,
            )
        )

    latency = groups.latency_stats
    if latency is not None:
        latency_rows = [
            PercentileRow(label, pm)
            for label, pm in (
                ("TTFT (ms)", latency.ttft_ms),
                ("TPOT (ms)", latency.tpot_ms),
                ("Step TPOT (ms)", latency.step_tpot_ms),
                ("ITL (ms)", latency.itl_ms),
                ("Request Latency (ms)", latency.latency_ms),
            )
            if pm is not None
        ]
        if latency_rows:
            print_section(title="Latency Stats")
            print(format_percentile_table(latency_rows))
            if latency.tpot_ms is not None:
                print("  TPOT = (latency - TTFT) / (output_tokens - 1)")
            if latency.step_tpot_ms is not None:
                print("  Step TPOT = ITL / tokens_per_step, per decode step")

    throughput = groups.throughput_stats
    if throughput is not None:
        throughput_rows = [
            PercentileRow(label, pm)
            for label, pm in (
                ("Input throughput (tok/s)", throughput.input_throughput),
                ("Output throughput (tok/s)", throughput.output_throughput),
            )
            if pm is not None
        ]
        if throughput_rows:
            print_section(title="Throughput Stats")
            print(format_percentile_table(throughput_rows))
            print(
                "  Throughput P90/P95/P99 are reversed"
                " (bottom 10/5/1%; higher is better)"
            )

    cache = groups.cache_stats
    if cache is not None:
        print_section(title="Cache Stats")
        if cache.global_cached_token_rate is not None:
            print(
                "{:<40} {:<10.2f}".format(
                    "Global Cached Token Rate:",
                    cache.global_cached_token_rate * 100,
                )
                + "%"
            )
        cache_rows = [
            PercentileRow(label, pm)
            for label, pm in (
                (
                    "Per-Turn Cached Token Rate (%)",
                    cache.per_turn_cached_token_rate,
                ),
                (
                    "Per-Turn Cache Retention (%)",
                    cache.per_turn_cache_retention,
                ),
            )
            if pm is not None
        ]
        if cache_rows:
            print(format_percentile_table(cache_rows))

    if metrics.text_data is not None:
        t = metrics.text_data
        print_section(title="Token Stats")
        print("{:<40} {:<10}".format("Max input tokens:", t.max_input))
        print("{:<40} {:<10}".format("Max output tokens:", t.max_output))
        print("{:<40} {:<10}".format("Max total tokens:", t.max_total))

    spec_decode_stats = metrics.spec_decode_stats
    if spec_decode_stats is not None:
        print_section(title="Speculative Decoding")
        if spec_decode_stats.acceptance_rate is not None:
            print(
                "{:<40} {:<10.2f}".format(
                    "Acceptance rate (%):", spec_decode_stats.acceptance_rate
                )
            )
        if spec_decode_stats.acceptance_length is not None:
            print(
                "{:<40} {:<10.2f}".format(
                    "Acceptance length:", spec_decode_stats.acceptance_length
                )
            )
        if spec_decode_stats.num_drafts is not None:
            print(
                "{:<40} {:<10}".format(
                    "Drafts:", int(spec_decode_stats.num_drafts)
                )
            )
        if spec_decode_stats.draft_tokens is not None:
            print(
                "{:<40} {:<10}".format(
                    "Draft tokens:", int(spec_decode_stats.draft_tokens)
                )
            )
        if spec_decode_stats.accepted_tokens is not None:
            print(
                "{:<40} {:<10}".format(
                    "Accepted tokens:", int(spec_decode_stats.accepted_tokens)
                )
            )
        per_pos_rates = spec_decode_stats.per_position_acceptance_rates
        if per_pos_rates:
            print("Per-position acceptance (%):")
            for pos, rate in enumerate(per_pos_rates):
                print(
                    "{:<40} {:<10.2f}".format(f"  Position {pos}:", rate * 100)
                )

    # Per-GPU series stay in result_groups.gpu_stats JSON; the console
    # reuses the same percentile table as latency / throughput.
    gpu = groups.gpu_stats
    if collect_gpu_stats and gpu is not None:
        gpu_rows = [
            PercentileRow(
                label,
                StandardPercentileMetrics(
                    [float(v) for v in values], unit=unit
                ),
            )
            for label, values, unit in (
                ("Peak GPU memory (MiB)", gpu.peak_gpu_memory_mib, "MiB"),
                (
                    "Available GPU memory (MiB)",
                    gpu.available_gpu_memory_mib,
                    "MiB",
                ),
                ("GPU utilization (%)", gpu.gpu_utilization, "%"),
            )
            if values
        ]
        if gpu_rows:
            n_devices = max(
                len(gpu.peak_gpu_memory_mib),
                len(gpu.available_gpu_memory_mib),
                len(gpu.gpu_utilization),
            )
            print_section(title=format_gpu_statistics_title(n_devices))
            print(format_percentile_table(gpu_rows))
            print(
                "  For per-GPU metrics, dump the result JSON instead"
                " (--result-filename)"
            )
    if collect_cpu_stats:
        cpu = metrics.cpu_metrics
        print_section(title="CPU Statistics")
        print(
            "{:<40} {:<10.2f}".format(
                "CPU utilization user (%):",
                cpu.user_percent if cpu else 0.0,
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "CPU utilization system (%):",
                cpu.system_percent if cpu else 0.0,
            )
        )
    diag = groups.diagnostics
    if diag is not None:
        diag_rows = [
            (label, value)
            for label, value in (
                ("Server startup time (s):", diag.server_startup_time),
                ("Steady-state detected:", diag.steady_state_detected),
                ("Steady-state windows:", diag.steady_state_window_count),
                ("Outliers rejected:", diag.num_outliers_rejected),
            )
            if value is not None
        ]
        if diag_rows or diag.steady_state_warning:
            print_section(title="Diagnostics")
            for label, value in diag_rows:
                if isinstance(value, float):
                    print(f"{label:<40} {value:<10.2f}")
                else:
                    print(f"{label:<40} {value:<10}")
            if diag.steady_state_warning:
                print(f"Steady-state warning: {diag.steady_state_warning}")
    print("=" * 50)
    if lora_manager:
        print_lora_benchmark_results(lora_manager.metrics)
    for label, pm in metrics.metrics_by_endpoint.items():
        if len(metrics.metrics_by_endpoint) > 1:
            print(f"\n--- Metrics: {label} ---")
        print_server_metrics(pm)
    print("=" * 50)


def _parse_metadata(metadata: list[str] | None) -> dict[str, str]:
    """Parse ``KEY=VALUE`` metadata strings into a flat dict.

    The special key ``server_cpu`` is mapped to ``cpu``.
    """
    result: dict[str, str] = {}
    for item in metadata or ():
        if "=" not in item:
            raise ValueError(
                "Invalid metadata format. Please use KEY=VALUE format."
            )
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        result["cpu" if key == "server_cpu" else key] = value
    return result


def save_result_json(
    result_filename: str | None,
    args: ServingBenchmarkConfig,
    benchmark_result: BenchmarkResult,
    *,
    benchmark_task: BenchmarkTask,
    model_id: str,
    tokenizer_id: str,
    request_rate: float,
    record_max_concurrency: int | None,
) -> None:
    """Persist benchmark results to *result_filename*."""
    if not result_filename:
        return
    backend: Backend = args.backend
    client_args = args.model_dump()
    rr_client = str(request_rate) if request_rate < float("inf") else "inf"
    client_args["request_rate"] = rr_client
    client_args["max_concurrency"] = record_max_concurrency
    result_json: dict[str, object] = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "backend": backend,
        "benchmark_task": benchmark_task,
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "num_prompts": (
            agg.completed
            if (agg := benchmark_result.aggregates) is not None
            else 0
        ),
        "dataset_name": args.dataset_name,
        "client_args": client_args,
        "request_rate": (
            request_rate if request_rate < float("inf") else "inf"
        ),
        "burstiness": args.burstiness,
        "max_concurrency": record_max_concurrency,
        "max_concurrent_conversations": args.max_concurrent_conversations,
        **_parse_metadata(args.metadata),
        **benchmark_result.to_result_dict(),
    }
    logger.info(f"Writing file: {result_filename}")
    if os.path.isfile(result_filename):
        logger.warning(
            "This is going to overwrite an existing file.  "
            f"The existing file will be moved to {result_filename}.orig."
        )
        os.rename(result_filename, f"{result_filename}.orig")
    with open(result_filename, "w") as outfile:
        # NaN latencies are real in partially-failed runs; scrub to null so
        # the file stays consumable by strict JSON parsers (jq, JSON.parse).
        json.dump(json_safe(result_json), outfile, allow_nan=False)


def save_output_lengths(
    args: ServingBenchmarkConfig,
    benchmark_result: BenchmarkResult,
    benchmark_task: BenchmarkTask,
    *,
    iteration_config: tuple[int | None, float] | None = None,
) -> None:
    """Save output lengths to a YAML file if configured."""
    if not args.record_output_lengths:
        return
    if benchmark_task != "text-generation":
        logger.warning(
            "--record-output-lengths is only supported for text-generation"
        )
        return
    args_to_save = (
        "backend",
        "burstiness",
        "dataset_name",
        "dataset_path",
        "endpoint",
        "max_concurrency",
        "max_output_len",
        "model",
        "request_rate",
        "seed",
        "temperature",
        "top_k",
        "top_p",
    )
    output_lens_dict: dict[str, object] = {}
    args_dict = args.model_dump()
    if iteration_config is not None:
        mc_snap, rr_snap = iteration_config
        args_dict["max_concurrency"] = mc_snap
        args_dict["request_rate"] = rr_snap
    output_lens_dict["args"] = {x: args_dict[x] for x in args_to_save}
    text_data = benchmark_result.text_data
    output_lens_dict["output_lengths"] = (
        text_data.output_lens if text_data is not None else []
    )
    with open(args.record_output_lengths, "w") as f:
        yaml.dump(output_lens_dict, f)
