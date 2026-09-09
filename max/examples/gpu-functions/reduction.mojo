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

from std.math import ceildiv
from std.atomic import Atomic
from std.random import randint
from std.sys import has_accelerator, size_of

from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.bit import log2_floor
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.primitives import warp
from max.gpu.globals import WARP_SIZE
from max.gpu.host import DeviceContext, DeviceBuffer
from std.memory import unsafe_stack_allocation
from std.testing import assert_equal
from max.benchmark import bencher_iter_custom

# Initialize parameters
# To achieve high bandwidth increase SIZE to large value
comptime TPB = 512
comptime BATCH_SIZE = 8  # needs to be power of 2
comptime SIZE = 1 << 12
comptime NUM_BLOCKS = ceildiv(SIZE, TPB * BATCH_SIZE)
comptime dtype = DType.int32


def sum_kernel[
    size: Int, batch_size: Int
](output: Pointer[Int32, MutAnyOrigin], a: Pointer[Int32, MutAnyOrigin],):
    """Efficient reduction of the vector a."""
    comptime KERNEL_TPB: Int = 512
    var sums = unsafe_stack_allocation[
        KERNEL_TPB,
        Scalar[dtype],
        address_space=.SHARED,
    ]()

    var global_tid = block_idx.x * block_dim.x + thread_idx.x
    var tid = thread_idx.x
    var threads_in_grid = KERNEL_TPB * NUM_BLOCKS
    var sum: Int32 = 0

    for i in range(global_tid, size, threads_in_grid):
        var idx = i * batch_size
        # Load in a vectorized fashion and reduce the loaded SIMD vector
        if idx < size:
            sum += a.unsafe_load[width=batch_size](idx).reduce_add()
    sums[unsafe_offset=tid] = sum
    barrier()

    # Reduce until the first warp

    var active_threads = KERNEL_TPB
    comptime KERNEL_LOG_TPB = log2_floor(KERNEL_TPB)

    comptime for power in range(1, KERNEL_LOG_TPB - log2_floor(WARP_SIZE) + 1):
        active_threads >>= 1
        if tid < active_threads:
            sums[unsafe_offset=tid] += sums[unsafe_offset=tid + active_threads]
        barrier()

    # Reduce the warp and accumulate via atomic addition
    if tid < WARP_SIZE:
        var warp_sum: Int32 = sums[unsafe_offset=tid][0]
        warp_sum = warp.sum(warp_sum)

        if tid == 0:
            _ = Atomic.fetch_add(output, warp_sum)


struct SumKernelBenchmarkParams:
    @__allow_legacy_any_origin_fields
    var out_ptr: Pointer[Int32, MutAnyOrigin]

    @__allow_legacy_any_origin_fields
    var a_ptr: Pointer[Int32, MutAnyOrigin]

    def __init__(
        out self,
        out_ptr: Pointer[mut=True, Int32, _],
        a_ptr: Pointer[mut=True, Int32, _],
    ):
        self.out_ptr = out_ptr.as_unsafe_any_origin()
        self.a_ptr = a_ptr.as_unsafe_any_origin()


# Benchmark function for sum_kernel
@always_inline
def sum_kernel_benchmark(
    mut b: Bencher, input_data: SumKernelBenchmarkParams
) raises:
    @always_inline
    def kernel_launch_sum(ctx: DeviceContext) raises {imm}:
        comptime kernel = sum_kernel[SIZE, BATCH_SIZE]
        var out_ptr = input_data.out_ptr
        var a_ptr = input_data.a_ptr
        var out_buffer = DeviceBuffer[dtype](ctx, out_ptr, 1, owning=False)
        var a_buffer = DeviceBuffer[dtype](ctx, a_ptr, SIZE, owning=False)
        ctx.enqueue_function[kernel](
            out_buffer,
            a_buffer,
            grid_dim=NUM_BLOCKS,
            block_dim=TPB,
        )

    var bench_ctx = DeviceContext()
    bencher_iter_custom(b, kernel_launch_sum, bench_ctx)


def main() raises:
    comptime assert has_accelerator(), "This example requires a supported GPU"

    with DeviceContext() as ctx:
        # Allocate memory on the device
        comptime kernel = sum_kernel[SIZE, BATCH_SIZE]
        var out = ctx.enqueue_create_buffer[dtype](1)
        out.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(0)

        # Initialise a with random integers between 0 and 10
        with a.map_to_host() as a_host:
            randint[dtype](a_host.as_span(), low=0, high=10)

        # Call the kernel
        ctx.enqueue_function[kernel](
            out,
            a,
            grid_dim=NUM_BLOCKS,
            block_dim=TPB,
        )
        ctx.synchronize()

        # Calculate the sum in a sequential fashion on the host
        # for correctness check
        var expected = ctx.enqueue_create_host_buffer[dtype](1)
        expected.enqueue_fill(0)
        with a.map_to_host() as a_host:
            for i in range(SIZE):
                expected[0] += a_host[i]

        # Assert the correctness of the kernel
        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            assert_equal(out_host[0], expected[0])

        var out_ptr = out.unsafe_ptr()
        var a_ptr = a.unsafe_ptr()

        # Benchmark performance
        var bench = Bench(BenchConfig(max_iters=50000))
        bench.bench_with_input(
            sum_kernel_benchmark,
            BenchId("sum_kernel_benchmark", "gpu"),
            SumKernelBenchmarkParams(out_ptr, a_ptr),
            [ThroughputMeasure(BenchMetric.bytes, SIZE * size_of[dtype]())],
        )
        # Pretty print in table format
        print(bench)
