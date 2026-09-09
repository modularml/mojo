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
"""Benchmarks scatter_nd on GPU.

Two modes:
- `row`: slice-style scatter — data [rows, cols], indices [num_idx, 1],
  updates [num_idx, cols].
- `elem`: element-style scatter — data [rows, cols], indices [num_idx, 2],
  updates [num_idx].

The timed region is the full op, which includes the kernel's internal
data->output device copy; reported bytes are 2*data + 2*updates + indices,
so the achievable ceiling is the device-to-device memcpy bandwidth.
"""

from std.sys import get_defined_dtype, get_defined_int, size_of

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    BenchId,
    BenchMetric,
    Bencher,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext
from internal_utils import arg_parse
from layout import TileTensor, row_major
from nn.gather_scatter import scatter_nd_generator

comptime itype = DType.int64


@no_inline
def run_row_scatter[
    dtype: DType, rows: Int, cols: Int, num_idx: Int
](mut m: Bench, ctx: DeviceContext) raises:
    var data_dev = ctx.enqueue_create_buffer[dtype](rows * cols)
    var out_dev = ctx.enqueue_create_buffer[dtype](rows * cols)
    var upd_dev = ctx.enqueue_create_buffer[dtype](num_idx * cols)
    var idx_dev = ctx.enqueue_create_buffer[itype](num_idx)

    # Indices must be valid; pseudo-random rows via a Knuth hash.
    var idx_host = ctx.enqueue_create_host_buffer[itype](num_idx)
    ctx.synchronize()
    for i in range(num_idx):
        idx_host[i] = Scalar[itype]((i * 2654435761) % rows)
    ctx.enqueue_copy(idx_dev, idx_host)
    ctx.synchronize()

    var data_tt = TileTensor(data_dev, row_major[rows, cols]())
    var out_tt = TileTensor(out_dev, row_major[rows, cols]())
    var upd_tt = TileTensor(upd_dev, row_major[num_idx, cols]())
    var idx_tt = TileTensor(idx_dev, row_major[num_idx, 1]())

    @always_inline
    def bench_func(
        mut b: Bencher,
    ) raises {var data_tt, var out_tt, var upd_tt, var idx_tt, imm}:
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises {imm}:
            scatter_nd_generator[target="gpu"](
                data_tt, idx_tt, upd_tt, out_tt, ctx
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    comptime data_bytes = rows * cols * size_of[dtype]()
    comptime upd_bytes = num_idx * cols * size_of[dtype]()
    comptime idx_bytes = num_idx * size_of[itype]()
    comptime num_bytes = 2 * data_bytes + 2 * upd_bytes + idx_bytes
    m.bench_function(
        bench_func,
        BenchId(
            "scatter_nd",
            input_id=String(
                "row/", dtype, "/", rows, "x", cols, "/idx=", num_idx
            ),
        ),
        [ThroughputMeasure(BenchMetric.bytes, num_bytes)],
    )
    ctx.synchronize()

    _ = data_dev
    _ = out_dev
    _ = upd_dev
    _ = idx_dev
    _ = idx_host


@no_inline
def run_elem_scatter[
    dtype: DType, rows: Int, cols: Int, num_idx: Int
](mut m: Bench, ctx: DeviceContext) raises:
    var data_dev = ctx.enqueue_create_buffer[dtype](rows * cols)
    var out_dev = ctx.enqueue_create_buffer[dtype](rows * cols)
    var upd_dev = ctx.enqueue_create_buffer[dtype](num_idx)
    var idx_dev = ctx.enqueue_create_buffer[itype](num_idx * 2)

    var idx_host = ctx.enqueue_create_host_buffer[itype](num_idx * 2)
    ctx.synchronize()
    for i in range(num_idx):
        idx_host[2 * i] = Scalar[itype]((i * 2654435761) % rows)
        idx_host[2 * i + 1] = Scalar[itype]((i * 40503 + 17) % cols)
    ctx.enqueue_copy(idx_dev, idx_host)
    ctx.synchronize()

    var data_tt = TileTensor(data_dev, row_major[rows, cols]())
    var out_tt = TileTensor(out_dev, row_major[rows, cols]())
    var upd_tt = TileTensor(upd_dev, row_major[num_idx]())
    var idx_tt = TileTensor(idx_dev, row_major[num_idx, 2]())

    @always_inline
    def bench_func(
        mut b: Bencher,
    ) raises {var data_tt, var out_tt, var upd_tt, var idx_tt, imm}:
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises {imm}:
            scatter_nd_generator[target="gpu"](
                data_tt, idx_tt, upd_tt, out_tt, ctx
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    comptime data_bytes = rows * cols * size_of[dtype]()
    comptime upd_bytes = num_idx * size_of[dtype]()
    comptime idx_bytes = num_idx * 2 * size_of[itype]()
    comptime num_bytes = 2 * data_bytes + 2 * upd_bytes + idx_bytes
    m.bench_function(
        bench_func,
        BenchId(
            "scatter_nd",
            input_id=String(
                "elem/", dtype, "/", rows, "x", cols, "/idx=", num_idx
            ),
        ),
        [ThroughputMeasure(BenchMetric.bytes, num_bytes)],
    )
    ctx.synchronize()

    _ = data_dev
    _ = out_dev
    _ = upd_dev
    _ = idx_dev
    _ = idx_host


def main() raises:
    var mode = arg_parse("mode", "row")
    comptime dtype = get_defined_dtype["dtype", .float32]()
    comptime rows = get_defined_int["rows", 131072]()
    comptime cols = get_defined_int["cols", 1024]()
    comptime num_idx = get_defined_int["num_idx", 4096]()

    var m = Bench()
    with DeviceContext() as ctx:
        if mode == "row":
            run_row_scatter[dtype, rows, cols, num_idx](m, ctx)
        elif mode == "elem":
            run_elem_scatter[dtype, rows, cols, num_idx](m, ctx)
        else:
            raise Error("unknown mode: " + mode)
    m.dump_report()
