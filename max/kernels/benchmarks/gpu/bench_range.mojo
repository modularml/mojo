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
"""Benchmarks the per-element cost of a `range()` loop inside a GPU kernel.

Every strided loop in every kernel pays whatever `_StridedRange.__next__()`
costs, so a change there lands on the whole kernel library at once and no single
kernel's benchmark moves enough to name it. These cases run the loop with almost
nothing in it: the working set is one L1-resident buffer read warp-uniformly, so
what the timing reports is the cursor and the addressing derived from it.

The strided cases are what the wrap check touches; the zero-based and sequential
cases are the control, and they must not move with it.

```
bazel run //max/kernels/benchmarks:gpu/bench_range
```
"""

from std.math import ceildiv

from max.benchmark import bencher_iter_custom
from max.gpu.host import DeviceContext
from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.builtin._closure import __ownership_keepalive
from max.gpu import global_idx
from internal_utils import update_bench_config_args

comptime DTYPE = DType.int32

# One 16 KiB buffer, read at the same indices by every thread, so the loads
# broadcast out of L1 and leave the loop itself as the cost being timed.
comptime SIZE = 4096
comptime STEP = 4

comptime BLOCK_DIM = 256
comptime GRID_DIM = 1024

# Extents arrive as `Int64` because `Int` is not `DevicePassable`. Widening
# rather than narrowing to `Int32` matters: a sign-extended `Int32` bound would
# let the compiler prove the cursor can never leave the dtype and delete the
# very check these cases exist to measure.


def zero_start_kernel(
    dst: Pointer[Scalar[DTYPE], MutAnyOrigin],
    src: Pointer[Scalar[DTYPE], ImmutAnyOrigin],
    n_arg: Int64,
):
    var n = Int(n_arg)
    var acc = Scalar[DTYPE](0)
    for i in range(n):
        acc += src[unsafe_offset=i]
    dst[unsafe_offset=Int(global_idx.x)] = acc


def sequential_kernel(
    dst: Pointer[Scalar[DTYPE], MutAnyOrigin],
    src: Pointer[Scalar[DTYPE], ImmutAnyOrigin],
    n_arg: Int64,
):
    var n = Int(n_arg)
    var acc = Scalar[DTYPE](0)
    for i in range(0, n):
        acc += src[unsafe_offset=i]
    dst[unsafe_offset=Int(global_idx.x)] = acc


def strided_static_step_kernel(
    dst: Pointer[Scalar[DTYPE], MutAnyOrigin],
    src: Pointer[Scalar[DTYPE], ImmutAnyOrigin],
    n_arg: Int64,
):
    var n = Int(n_arg)
    var acc = Scalar[DTYPE](0)
    for i in range(0, n, STEP):
        acc += src[unsafe_offset=i]
    dst[unsafe_offset=Int(global_idx.x)] = acc


def strided_dynamic_step_kernel(
    dst: Pointer[Scalar[DTYPE], MutAnyOrigin],
    src: Pointer[Scalar[DTYPE], ImmutAnyOrigin],
    n_arg: Int64,
    step_arg: Int64,
):
    var n = Int(n_arg)
    var acc = Scalar[DTYPE](0)
    for i in range(0, n, Int(step_arg)):
        acc += src[unsafe_offset=i]
    dst[unsafe_offset=Int(global_idx.x)] = acc


def strided_negative_step_kernel(
    dst: Pointer[Scalar[DTYPE], MutAnyOrigin],
    src: Pointer[Scalar[DTYPE], ImmutAnyOrigin],
    n_arg: Int64,
):
    var n = Int(n_arg)
    var acc = Scalar[DTYPE](0)
    for i in range(n - 1, -1, -STEP):
        acc += src[unsafe_offset=i]
    dst[unsafe_offset=Int(global_idx.x)] = acc


# A runtime step off a thread-dependent start, the cursor shape a grid-stride
# loop has. The extents are the same as the other cases rather than a real grid
# partition, which at this thread count would leave most threads with nothing to
# iterate and nothing to time.
def grid_stride_kernel(
    dst: Pointer[Scalar[DTYPE], MutAnyOrigin],
    src: Pointer[Scalar[DTYPE], ImmutAnyOrigin],
    n_arg: Int64,
    stride_arg: Int64,
):
    var n = Int(n_arg)
    var acc = Scalar[DTYPE](0)
    var stride = Int(stride_arg)
    for i in range(Int(global_idx.x) % stride, n, stride):
        acc += src[unsafe_offset=i]
    dst[unsafe_offset=Int(global_idx.x)] = acc


@no_inline
def bench_range(mut b: Bench, ctx: DeviceContext) raises:
    var threads = GRID_DIM * BLOCK_DIM
    var src_host = List(length=SIZE, fill=Scalar[DTYPE](1))
    var src = ctx.enqueue_create_buffer[DTYPE](SIZE)
    var dst = ctx.enqueue_create_buffer[DTYPE](threads)
    ctx.enqueue_copy(src, src_host)

    # `elements` counts the loop trips the whole grid retires, so the reported
    # throughput is per cursor step and comparable across the cases.
    var strided_elems = threads * ceildiv(SIZE, STEP)

    def run[
        LaunchFn: def(DeviceContext) raises -> None
    ](launch: LaunchFn, name: StaticString, elems: Int) raises:
        def bench_func(mut bencher: Bencher) raises {imm launch, imm ctx}:
            bencher_iter_custom(bencher, launch, ctx)

        b.bench_function(
            bench_func,
            BenchId(name),
            [ThroughputMeasure(BenchMetric.elements, elems)],
        )

    def launch_zero_start(c: DeviceContext) raises {imm src, imm dst}:
        c.enqueue_function[zero_start_kernel](
            dst, src, Int64(SIZE), grid_dim=GRID_DIM, block_dim=BLOCK_DIM
        )

    def launch_sequential(c: DeviceContext) raises {imm src, imm dst}:
        c.enqueue_function[sequential_kernel](
            dst, src, Int64(SIZE), grid_dim=GRID_DIM, block_dim=BLOCK_DIM
        )

    def launch_static_step(c: DeviceContext) raises {imm src, imm dst}:
        c.enqueue_function[strided_static_step_kernel](
            dst, src, Int64(SIZE), grid_dim=GRID_DIM, block_dim=BLOCK_DIM
        )

    def launch_dynamic_step(c: DeviceContext) raises {imm src, imm dst}:
        c.enqueue_function[strided_dynamic_step_kernel](
            dst,
            src,
            Int64(SIZE),
            Int64(STEP),
            grid_dim=GRID_DIM,
            block_dim=BLOCK_DIM,
        )

    def launch_negative_step(c: DeviceContext) raises {imm src, imm dst}:
        c.enqueue_function[strided_negative_step_kernel](
            dst, src, Int64(SIZE), grid_dim=GRID_DIM, block_dim=BLOCK_DIM
        )

    def launch_grid_stride(c: DeviceContext) raises {imm src, imm dst}:
        c.enqueue_function[grid_stride_kernel](
            dst,
            src,
            Int64(SIZE),
            Int64(STEP),
            grid_dim=GRID_DIM,
            block_dim=BLOCK_DIM,
        )

    run(launch_zero_start, "range_zero_start", threads * SIZE)
    run(launch_sequential, "range_sequential", threads * SIZE)
    run(launch_static_step, "range_strided_static_step", strided_elems)
    run(launch_dynamic_step, "range_strided_dynamic_step", strided_elems)
    run(launch_negative_step, "range_strided_negative_step", strided_elems)
    run(launch_grid_stride, "range_grid_stride", strided_elems)

    ctx.synchronize()
    __ownership_keepalive(src, dst)
    _ = src_host^


def main() raises:
    var b = Bench()
    update_bench_config_args(b)
    with DeviceContext() as ctx:
        bench_range(b, ctx)
    b.dump_report()
