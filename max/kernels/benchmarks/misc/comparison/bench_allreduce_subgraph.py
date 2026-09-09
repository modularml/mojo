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

"""Benchmark comparing NCCL vs SGLang custom allreduce.

This benchmark compares:
- NCCL's allreduce (via torch.distributed)
- SGLang's custom_all_reduce (cross_device_reduce_1stage/2stage)

SGLang's custom allreduce is optimized for small message sizes (<256KB) and uses
P2P communication with NVLink for low latency.

Run the benchmarks using kbench:
    kbench bench_allreduce_subgraph.yaml

Usage:
    bazel run install_sglang
    bazel run bench_allreduce_subgraph --run_under="mpirun -np 8" -- --num_gpus=8 --num_bytes=16384
"""

from __future__ import annotations

import argparse
import os
import traceback
from typing import Any

import torch
import torch.distributed as dist
from bencher_utils import Bench, ThroughputMeasure, check_mpirun

# Try importing SGLang's custom allreduce
_sgl_kernel_available = False
_CustomAllreduceClass: type | None = None

# First try the high-level wrapper from sglang package (preferred)
try:
    from sglang.srt.distributed.device_communicators.custom_all_reduce import (  # type: ignore[import-not-found]
        CustomAllreduce,
    )

    _CustomAllreduceClass = CustomAllreduce
except (ImportError, RuntimeError):
    traceback.print_exc()
    print(
        "[bench_allreduce_subgraph] sglang CustomAllreduce unavailable,"
        " falling back to NCCL-only"
    )

# Fall back to low-level sgl_kernel API if high-level not available
if _CustomAllreduceClass is None:
    import importlib.util

    _sgl_kernel_available = importlib.util.find_spec("sgl_kernel") is not None


def _bench_cuda_events(
    fn: Any,
    num_warmups: int = 10,
    num_iters: int = 100,
    reduce_max: bool = True,
) -> float:
    """Simple CUDA event-based benchmarking for distributed scenarios.

    Args:
        fn: Function to benchmark
        num_warmups: Number of warmup iterations
        num_iters: Number of test iterations
        reduce_max: If True and distributed is initialized, gather max time across all ranks.
                   For collective operations like allreduce, the operation is only complete
                   when the slowest GPU finishes, so we report max time across all ranks.

    Returns:
        Average time in seconds (max across all ranks if reduce_max=True)
    """
    # Synchronize all processes before warmup
    if dist.is_initialized():
        dist.barrier()

    # Warmup
    for _ in range(num_warmups):
        fn()
    torch.cuda.synchronize()

    # Synchronize all processes before timing
    if dist.is_initialized():
        dist.barrier()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    # Get local average time in seconds
    local_time_s = start_event.elapsed_time(end_event) / num_iters / 1000.0

    # For collective operations, report the max time across all ranks
    # This is important because allreduce is only complete when ALL GPUs finish
    if reduce_max and dist.is_initialized():
        # Use a tensor to gather max time across all ranks
        time_tensor = torch.tensor(
            [local_time_s], dtype=torch.float64, device="cuda"
        )
        dist.all_reduce(time_tensor, op=dist.ReduceOp.MAX)
        return time_tensor.item()

    return local_time_s


_sglang_parallel_state_initialized = False


def _init_sglang_parallel_state() -> None:
    """Initialize SGLang's parallel state if not already done."""
    global _sglang_parallel_state_initialized
    if _sglang_parallel_state_initialized:
        return

    try:
        from sglang.srt.distributed import (  # type: ignore[import-not-found]
            parallel_state,
        )

        # First initialize the distributed environment (creates world group)
        parallel_state.init_distributed_environment()

        # Then initialize model parallel groups with tensor parallelism matching world size
        world_size = dist.get_world_size()
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=world_size
        )
        _sglang_parallel_state_initialized = True
    except Exception as e:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(f"  [SGLang] Failed to initialize parallel state: {e}")


def bench_sglang_allreduce(
    num_gpus: int,
    num_bytes: int,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[float, float] | None:
    """Benchmark SGLang's custom allreduce.

    Note: SGLang's custom allreduce requires torch.distributed to be initialized
    and works best with NVLink-connected GPUs.

    Args:
        num_gpus: Number of GPUs to use
        num_bytes: Number of bytes to allreduce
        dtype: Data type for the tensors

    Returns:
        Tuple of (time_seconds, bandwidth_gbps) or None if not available
    """
    if not _sgl_kernel_available and _CustomAllreduceClass is None:
        return None

    # SGLang's custom allreduce requires distributed initialization
    if not dist.is_initialized():
        return None

    # Initialize SGLang's parallel state (required for CustomAllreduce)
    _init_sglang_parallel_state()

    bytes_per_element = 2 if dtype in (torch.bfloat16, torch.float16) else 4
    num_elements = num_bytes // bytes_per_element

    # Create input tensor
    input_tensor = torch.randn(num_elements, dtype=dtype, device="cuda")

    try:
        if _CustomAllreduceClass is not None:
            # Use high-level CustomAllreduce wrapper from sglang package
            rank = dist.get_rank()

            # Create a CPU group for coordination (CustomAllreduce needs this)
            try:
                cpu_group = dist.new_group(backend="gloo")
            except Exception:
                cpu_group = dist.group.WORLD

            device = torch.device(f"cuda:{rank}")

            custom_ar = _CustomAllreduceClass(
                cpu_group, device, max_size=num_bytes * 2
            )

            if custom_ar.disabled:
                return None

            def run_kernel() -> torch.Tensor:
                return custom_ar.custom_all_reduce(input_tensor)

        elif _sgl_kernel_available:
            # Low-level sgl_kernel API requires multi-process IPC setup
            # which is complex - recommend using sglang[all] instead
            return None

        # Benchmark using CUDA events
        time_s = _bench_cuda_events(run_kernel, num_warmups=10, num_iters=100)

        # Calculate bandwidth
        # AllReduce: each GPU sends and receives (N-1)/N of the data
        # Bus bandwidth = 2 * data_size * (N-1) / N / time
        busbw = 2 * num_bytes * (num_gpus - 1) / num_gpus

        return time_s, busbw

    except Exception as e:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(f"SGLang benchmark failed: {e}")
        return None


def bench_nccl_allreduce(
    num_gpus: int,
    num_bytes: int,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[float, float] | None:
    """Benchmark NCCL allreduce as baseline.

    Args:
        num_gpus: Number of GPUs to use
        num_bytes: Number of bytes to allreduce
        dtype: Data type for the tensors

    Returns:
        Tuple of (time_seconds, bandwidth_gbps) or None if not available
    """
    if not dist.is_initialized():
        return None

    bytes_per_element = 2 if dtype in (torch.bfloat16, torch.float16) else 4
    num_elements = num_bytes // bytes_per_element

    input_tensor = torch.randn(num_elements, dtype=dtype, device="cuda")

    def run_kernel() -> None:
        dist.all_reduce(input_tensor, op=dist.ReduceOp.SUM)

    try:
        time_s = _bench_cuda_events(run_kernel, num_warmups=10, num_iters=100)

        algbw = num_bytes
        busbw = 2 * algbw * (num_gpus - 1) / num_gpus

        return time_s, busbw

    except Exception as e:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(f"NCCL benchmark failed: {e}")
        return None


def run_benchmark(
    num_gpus: int,
    num_bytes: int,
    dtype: torch.dtype = torch.bfloat16,
    backend: str = "nccl",
) -> tuple[float, float] | None:
    """Run allreduce comparison benchmark.

    All ranks must call this function for distributed benchmarks to work.
    Only rank 0 prints results.

    Args:
        num_gpus: Number of GPUs
        num_bytes: List of buffer sizes to test
        dtype: Data type
        backend: Backend to benchmark: nccl, sglang
    """
    # All ranks must participate in distributed benchmarks
    if backend == "nccl":
        result = bench_nccl_allreduce(num_gpus, num_bytes, dtype)
    elif backend == "sglang":
        result = bench_sglang_allreduce(num_gpus, num_bytes, dtype)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AllReduce benchmark: NCCL vs SGLang"
    )
    parser.add_argument(
        "--num-gpus",
        "--num_gpus",
        type=int,
        help="Number of GPUs",
    )
    parser.add_argument(
        "--num-bytes",
        "--num_bytes",
        type=int,
        default=16 * 1024,
        help="Buffer size in bytes to test",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        choices=["nccl", "sglang"],
        help="Select backend engine",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.csv",
        help="Output path",
    )

    args, _ = parser.parse_known_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    # Initialize distributed env if running under mpirun
    pe_rank = check_mpirun()
    if pe_rank != -1:
        torch.cuda.set_device(pe_rank)
        dist.init_process_group(
            backend="nccl", device_id=torch.device(f"cuda:{pe_rank}")
        )

    if not dist.is_initialized():
        print("Error: This benchmark requires mpirun for multi-GPU execution.")
        return

    # Check the number of GPUs
    assert args.num_gpus <= int(os.environ["OMPI_COMM_WORLD_SIZE"])

    # All ranks must call run_benchmark for distributed benchmarks
    result = run_benchmark(
        num_gpus=args.num_gpus,
        num_bytes=args.num_bytes,
        dtype=dtype_map[args.dtype],
        backend=args.backend,
    )

    if dist.is_initialized():
        dist.destroy_process_group()

    name = "bench_allreduce_subgraph"
    met_sec, bytes = result or [0, 0]
    bytes = args.num_bytes
    bytes_per_sec = ThroughputMeasure(Bench.bytes, bytes)

    b = Bench(
        name,
        iters=1,
        met=met_sec,
        metric_list=[bytes_per_sec],
    )

    # TODO: move this to bencher_utils.py
    out = args.output.replace(".csv", f"{pe_rank}.csv")
    if pe_rank == 0:
        b.dump_report(output_path=out)


if __name__ == "__main__":
    main()
