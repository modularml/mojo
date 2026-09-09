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

from std.random import randn
from std.sys import simd_width_of, size_of

from max.algorithm.functional import elementwise
from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext
from std.memory import alloc, dealloc

from std.utils import IndexList

from layout import TileTensor, Coord, row_major


def bench_add[
    unroll_by: Int, rank: Int
](mut b: Bench, shape: IndexList[rank], ctx: DeviceContext) raises:
    comptime type = DType.float32
    var size = shape.flattened_length()
    var input0_ptr = ctx.enqueue_create_buffer[type](size)
    var input1_ptr = ctx.enqueue_create_buffer[type](size)
    var output_ptr = ctx.enqueue_create_buffer[type](size)

    var input0_host = alloc[Scalar[type]]({count = size}).into_managed()
    var input0_ptr_host = input0_host.unsafe_ptr()
    var input1_host = alloc[Scalar[type]]({count = size}).into_managed()
    var input1_ptr_host = input1_host.unsafe_ptr()
    var output_host = alloc[Scalar[type]]({count = size}).into_managed()
    var output_ptr_host = output_host.unsafe_ptr()

    randn(input0_host.unsafe_span())
    randn(input1_host.unsafe_span())
    randn(output_host.unsafe_span())
    ctx.enqueue_copy(input0_ptr, input0_host.unsafe_span())
    ctx.enqueue_copy(input1_ptr, input1_host.unsafe_span())
    ctx.enqueue_copy(output_ptr, output_host.unsafe_span())

    var input0 = TileTensor(input0_ptr, row_major(Coord(shape)))
    var input1 = TileTensor(input1_ptr, row_major(Coord(shape)))
    var output = TileTensor(output_ptr, row_major(Coord(shape)))

    @always_inline
    def add[simd_width: Int, alignment: Int = 1](idx: Coord) {var}:
        comptime assert input0.flat_rank >= idx.flat_rank
        var val = input0.load[width=simd_width](idx) + input1.load[
            width=simd_width
        ](idx)
        output.store[width=simd_width](idx, val)

    @always_inline
    def bench_func(mut b: Bencher, shape: IndexList[rank]) raises {imm}:
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises {imm}:
            elementwise[simd_width=unroll_by, target="gpu"](
                add, Coord(shape), ctx
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    b.bench_with_input(
        bench_func,
        BenchId("add", String(shape)),
        shape,
        # TODO: Pick relevant benchmetric.
        [ThroughputMeasure(BenchMetric.elements, size * size_of[type]() * 3)],
    )

    ctx.enqueue_copy(output_host.unsafe_span(), output_ptr)

    comptime nelts = simd_width_of[type]()
    for i in range(0, size, nelts):
        if not (
            output_ptr_host.unsafe_load[width=nelts](i).eq(
                input0_ptr_host.unsafe_load[width=nelts](i)
                + input1_ptr_host.unsafe_load[width=nelts](i)
            )
        ).reduce_and():
            raise Error(t"mismatch at flattened idx {i}")

    dealloc(input0_host^)
    dealloc(input1_host^)
    dealloc(output_host^)


def main() raises:
    var b = Bench()
    with DeviceContext() as ctx:
        bench_add[unroll_by=4](b, IndexList[4](2, 4, 1024, 1024), ctx)
        bench_add[unroll_by=1](b, IndexList[4](2, 4, 1024, 1024), ctx)
        b.dump_report()
