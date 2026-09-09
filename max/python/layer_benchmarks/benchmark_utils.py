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

"""Shared utilities for layer-wise GPU benchmarks using CUDA events."""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from max._core.profiler import Trace
from max.driver import Accelerator, Buffer, DeviceEvent
from max.engine import Model
from tabulate import tabulate

BENCHMARK_NVTX_RANGE = "testbed/benchmark"


def measure_gpu_latency(
    fn: Callable[[], object],
    device: Accelerator,
    num_iterations: int = 50,
    num_warmup: int = 5,
) -> Sequence[float]:
    """Measure GPU execution latency using CUDA events.

    Args:
        fn: Callable that submits GPU work (e.g. compiled_model.execute).
        device: The GPU device to record events on.
        num_iterations: Number of timed iterations.
        num_warmup: Number of warmup iterations (not timed).

    Returns:
        Sequence of per-iteration latencies in milliseconds.
    """
    stream = device.default_queue
    latencies: list[float] = []

    # Warmup (outside NVTX range).
    for wi in range(num_warmup):
        with Trace(f"warmup/{wi}"):
            start = DeviceEvent(device, enable_timing=True)
            end = DeviceEvent(device, enable_timing=True)
            stream.record_event(start)
            fn()
            stream.record_event(end)
            end.synchronize()

    # Timed iterations wrapped in NVTX ranges for nsys capture.
    # Trace is a no-op when profiling is not enabled.
    for i in range(num_iterations):
        with Trace(f"iteration/{i}"):
            start = DeviceEvent(device, enable_timing=True)
            end = DeviceEvent(device, enable_timing=True)
            stream.record_event(start)
            fn()
            stream.record_event(end)
            end.synchronize()
            latencies.append(start.elapsed_time(end))

    return latencies


def measure_gpu_latency_cuda_graph(
    model: Model,
    execute_args: list[Buffer],
    device: Accelerator,
    graph_key: int,
    num_iterations: int = 50,
    num_warmup: int = 5,
) -> Sequence[float]:
    """Measure GPU execution latency using CUDA graph capture/replay.

    Captures the model execution into a CUDA graph once, then times
    repeated replays. Use this for decode-style workloads where
    production uses CUDA graphs to eliminate kernel launch overhead.

    Args:
        model: Compiled model with capture() and replay() methods.
        execute_args: Arguments to pass to model.capture/replay.
        device: The GPU device to record events on.
        graph_key: Unique integer key for this CUDA graph capture.
        num_iterations: Number of timed replay iterations.
        num_warmup: Number of warmup replay iterations (not timed).

    Returns:
        Sequence of per-iteration latencies in milliseconds.
    """
    model.capture(graph_key, *execute_args)
    device.synchronize()

    return measure_gpu_latency(
        lambda: model.replay(graph_key, *execute_args),
        device,
        num_iterations=num_iterations,
        num_warmup=num_warmup,
    )


@dataclass
class BenchmarkStats:
    """Statistics computed from benchmark latency measurements."""

    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    median_ms: float
    p95_ms: float
    n: int


def compute_stats(latencies: Sequence[float]) -> BenchmarkStats:
    """Compute benchmark statistics from a sequence of latencies in ms."""
    arr = np.array(latencies)
    return BenchmarkStats(
        mean_ms=float(np.mean(arr)),
        std_ms=float(np.std(arr)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        median_ms=float(np.median(arr)),
        p95_ms=float(np.percentile(arr, 95)),
        n=len(latencies),
    )


def _fmt_ms(val: float) -> str:
    """Format a millisecond value with appropriate precision."""
    if val < 1.0:
        return f"{val:.3f} ms"
    elif val < 100.0:
        return f"{val:.2f} ms"
    else:
        return f"{val:.1f} ms"


def print_results_table(
    title: str,
    results: list[tuple[str, BenchmarkStats]],
) -> None:
    """Print a formatted results table to stdout using tabulate.

    Args:
        title: Table title (e.g. "AttentionWithRope [llama3-8b]").
        results: List of (label, stats) tuples.
    """
    headers = ["Shape", "Mean", "Std", "Min", "Max", "Median", "P95", "N"]

    rows: list[list[str]] = []
    for label, stats in results:
        row = [label]
        for val in [
            stats.mean_ms,
            stats.std_ms,
            stats.min_ms,
            stats.max_ms,
            stats.median_ms,
            stats.p95_ms,
        ]:
            row.append(_fmt_ms(val))
        row.append(str(stats.n))
        rows.append(row)

    print(f"\n  {title}\n")
    print(
        tabulate(
            rows,
            headers=headers,
            tablefmt="simple_outline",
            colalign=("left", *("right",) * 7),
        )
    )
    print()


def dump_kbench_csv(
    bench_name: str,
    results: list[tuple[str, BenchmarkStats]],
    output_path: str | Path,
) -> None:
    """Write benchmark results as multi-row CSV in kbench's expected format.

    Each row's name includes ``/input_id:<label>`` so kbench can distinguish
    multiple measurements from a single subprocess invocation.

    Args:
        bench_name: Base benchmark name (e.g. "attention_with_rope").
        results: List of (label, stats) tuples from compute_stats().
        output_path: Path to write the CSV file.
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "met (ms)", "iters"])
        for label, stats in results:
            name = f"{bench_name}/input_id:{label}"
            writer.writerow([name, stats.mean_ms, stats.n])
