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
"""Benchmarks `argmax_gpu`, the top-k K=1 path it replaced, and the
`algorithm.rowwise` reduction `ops.argmax` actually launches.

All variants run in the same process so they see identical clocks, which
matters on hosts where the SM clock cannot be pinned.
"""

from std.random import random_float64, seed
from std.sys import get_defined_dtype, size_of
from std.sys.info import size_of as _size_of

from algorithm.reductions import reduce_argmax
from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext
from internal_utils import arg_parse
from layout import Coord, TileTensor, coord_to_index_list, row_major
from nn.argmaxmin_gpu import argmax_gpu
from nn.topk import topk_gpu
from std.utils.index import IndexList


def bench_argmax[
    dtype: DType, out_idx_type: DType
](ctx: DeviceContext, mut m: Bench, batch: Int, num_elements: Int) raises:
    var in_host = ctx.enqueue_create_host_buffer[dtype](batch * num_elements)
    ctx.synchronize()
    for i in range(batch * num_elements):
        in_host[i] = random_float64(-1e4, 1e4).cast[dtype]()

    var in_dev = ctx.enqueue_create_buffer[dtype](batch * num_elements)
    var out_dev = ctx.enqueue_create_buffer[out_idx_type](batch)
    var vals_dev = ctx.enqueue_create_buffer[dtype](batch)
    ctx.enqueue_copy(in_dev, in_host)
    ctx.synchronize()

    var in_tensor = TileTensor(
        in_dev, row_major(Coord(IndexList[2](batch, num_elements)))
    )
    var out_tensor = TileTensor(
        out_dev, row_major(Coord(IndexList[2](batch, 1)))
    )
    var vals_tensor = TileTensor(
        vals_dev, row_major(Coord(IndexList[2](batch, 1)))
    )

    var num_bytes = batch * num_elements * size_of[Scalar[dtype]]()
    var suffix = String("/N=", num_elements, "/batch=", batch, "/", dtype)

    @always_inline
    def bench_streaming(mut b: Bencher) raises {imm}:
        @always_inline
        def launch(ctx: DeviceContext) raises {imm}:
            argmax_gpu(ctx, in_tensor, out_tensor)

        bencher_iter_custom(b, launch, ctx)

    m.bench_function(
        bench_streaming,
        BenchId(String("argmax-streaming", suffix)),
        [ThroughputMeasure(BenchMetric.bytes, num_bytes)],
    )

    @always_inline
    def bench_topk_k1(mut b: Bencher) raises {imm}:
        @always_inline
        def launch(ctx: DeviceContext) raises {imm}:
            topk_gpu[sampling=False, largest=True](
                ctx, 1, in_tensor, vals_tensor, out_tensor
            )

        bencher_iter_custom(b, launch, ctx)

    m.bench_function(
        bench_topk_k1,
        BenchId(String("argmax-topk-k1", suffix)),
        [ThroughputMeasure(BenchMetric.bytes, num_bytes)],
    )

    var in_shape = Coord(IndexList[2](batch, num_elements))

    @always_inline
    def launch_rowwise(
        ctx: DeviceContext,
    ) raises {mut out_tensor, imm}:
        @always_inline
        def input_fn[
            width: Int, alignment: Int
        ](coords: Coord) {var in_tensor} -> SIMD[dtype, width]:
            return in_tensor.load[width=width](coords)

        @always_inline
        def output_fn[
            width: SIMDLength
        ](coords: Coord, val: SIMD[.int64, width]) {var out_tensor}:
            out_tensor.store[width=Int(width)](coords, val.cast[out_idx_type]())

        reduce_argmax[dtype, target="gpu", reduce_dim=1](
            input_fn, output_fn, in_shape, ctx
        )

    @always_inline
    def bench_rowwise(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, launch_rowwise, ctx)

    m.bench_function(
        bench_rowwise,
        BenchId(String("argmax-rowwise", suffix)),
        [ThroughputMeasure(BenchMetric.bytes, num_bytes)],
    )

    _ = in_dev^
    _ = out_dev^
    _ = vals_dev^


def main() raises:
    seed(24301)
    var batch = Int(arg_parse("batch", 1))
    var num_elements = Int(arg_parse("N", 262144))

    comptime dtype = get_defined_dtype["dtype", .bfloat16]()

    var m = Bench()
    with DeviceContext() as ctx:
        bench_argmax[dtype, DType.int64](ctx, m, batch, num_elements)
    m.dump_report()
