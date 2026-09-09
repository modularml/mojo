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

from std.random import random_float64
from std.sys import (
    is_defined,
    get_defined_dtype,
    get_defined_string,
    simd_width_of,
)

from max.benchmark import bencher_iter_custom
from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from max.gpu.host import DeviceContext, get_gpu_target
from internal_utils import get_defined_shape, int_list_to_tuple
from layout import Coord, TileTensor, coord_to_index_list, row_major
from nn.softmax import softmax, softmax_inline, softmax_with_temperature

from std.utils.coord import ComptimeInt
from std.utils.index import IndexList


def bench_softmax_gpu[
    rank: Int, //, dtype: DType, shape: IndexList[rank]
](ctx: DeviceContext, mut b: Bench, fn_name: String) raises:
    comptime cols = shape[rank - 1]
    comptime total = shape.flattened_length()

    var data_h = List(length=total, fill=Scalar[dtype](0))

    for i in range(total):
        data_h[i] = Scalar[dtype](random_float64(-1, 1).cast[dtype]())

    var data_d = ctx.enqueue_create_buffer[dtype](total)
    var out_d = ctx.enqueue_create_buffer[dtype](total)

    var data_buf = TileTensor(data_d, row_major(Coord(shape))).as_immut()
    var out_buf = TileTensor(out_d, row_major(Coord(shape)))

    ctx.enqueue_copy(data_d, data_h)

    # The no-lambda `softmax_inline` overload defaults to target="cpu".
    @__parameter
    @__copy_capture(data_buf)
    def input_fn[_simd_width: Int](coords: Coord) -> SIMD[dtype, _simd_width]:
        return data_buf.load[width=_simd_width, alignment=1](coords)

    @always_inline
    def kernel_launch(ctx: DeviceContext) raises {mut out_buf, mut data_d, imm}:
        softmax_inline[
            dtype,
            simd_width_of[dtype, target=get_gpu_target()](),
            rank,
            input_fn,
            target="gpu",
        ](
            Coord(shape),
            out_buf,
            rank - 1,
            ctx,
        )

    @always_inline
    def bench_fn(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, kernel_launch, ctx)

    b.bench_function(
        bench_fn,
        BenchId("softmax", input_id=String(fn_name, "/", dtype, "/", shape)),
    )

    # The `algorithm.rowwise` overload — what `mo.reduce.softmax` launches.
    # Benched in the same process so both arms see identical clocks.
    @always_inline
    def kernel_launch_rowwise(ctx: DeviceContext) raises {mut out_buf, imm}:
        @always_inline
        def rowwise_input_fn[
            width: Int, alignment: Int
        ](coords: Coord) {var data_buf} -> SIMD[dtype, width]:
            return data_buf.load[width=width](coords)

        softmax[dtype, rank, target="gpu", reduce_dim=rank - 1](
            rowwise_input_fn,
            Coord(shape),
            ComptimeInt[cols](),
            out_buf,
            rank - 1,
            context=ctx,
        )

    @always_inline
    def bench_fn_rowwise(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, kernel_launch_rowwise, ctx)

    b.bench_function(
        bench_fn_rowwise,
        BenchId(
            "softmax", input_id=String("softmax_rowwise/", dtype, "/", shape)
        ),
    )

    ctx.synchronize()

    _ = data_d
    _ = out_d
    _ = data_h^


def bench_softmax_with_temperature_gpu[
    dtype: DType, shape: IndexList
](
    ctx: DeviceContext,
    mut b: Bench,
    fn_name: String,
    temperature: Float32,
) raises:
    comptime total = shape.flattened_length()
    comptime rows = shape[0]

    var data_h = List(length=total, fill=Scalar[dtype](0))
    for i in range(total):
        data_h[i] = Scalar[dtype](random_float64(-1, 1).cast[dtype]())

    var data_d = ctx.enqueue_create_buffer[dtype](total)
    var out_d = ctx.enqueue_create_buffer[dtype](total)

    var data_buf = TileTensor(data_d, row_major(Coord(shape))).as_immut()
    var out_buf = TileTensor(out_d, row_major(Coord(shape)))

    ctx.enqueue_copy(data_d, data_h)

    var temp = temperature

    @always_inline
    def kernel_launch(ctx: DeviceContext) raises {mut out_buf, imm}:
        softmax_with_temperature(ctx, data_buf, out_buf, temp)

    @always_inline
    def bench_fn(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, kernel_launch, ctx)

    b.bench_function(
        bench_fn,
        BenchId(
            "softmax_with_temperature",
            input_id=String(fn_name, "/", dtype, "/", shape, "/T=", temp),
        ),
    )

    ctx.synchronize()

    _ = data_d
    _ = out_d
    _ = data_h^


def main() raises:
    comptime dtype = get_defined_dtype["dtype", .bfloat16]()
    comptime shape = int_list_to_tuple[
        get_defined_shape["shape", "256x256"]()
    ]()
    var m = Bench(BenchConfig(num_repetitions=1))
    with DeviceContext() as ctx:
        comptime if is_defined["temperature"]():
            var temperature = Float32(
                atof(get_defined_string["temperature", "1.0"]())
            )
            bench_softmax_with_temperature_gpu[dtype, shape](
                ctx, m, "softmax_with_temperature_gpu", temperature
            )
        else:
            bench_softmax_gpu[dtype, shape](ctx, m, "softmax_gpu")

    m.dump_report()
