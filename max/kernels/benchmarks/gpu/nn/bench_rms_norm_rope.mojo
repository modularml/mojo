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
from std.sys import get_defined_dtype

from max.benchmark import bencher_iter_custom
from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from max.gpu.host import DeviceContext
from internal_utils import get_defined_shape, int_list_to_tuple
from layout import Coord, Idx, TileTensor, row_major
from nn.normalization import rms_norm_rope

from std.utils.index import Index, IndexList


def bench_rms_norm_rope_gpu[
    rank: Int, //, dtype: DType, shape: IndexList[rank]
](ctx: DeviceContext, mut b: Bench, fn_name: String) raises:
    comptime cols = shape[rank - 1]
    comptime rows = shape.flattened_length() // cols

    var data_h = List(length=rows * cols, fill=Scalar[dtype](0))
    var gamma_h = List(length=cols, fill=Scalar[dtype](0))
    var cos_h = List(length=rows * cols, fill=Scalar[dtype](0))
    var sin_h = List(length=rows * cols, fill=Scalar[dtype](0))

    for i in range(rows * cols):
        data_h[i] = Scalar[dtype](random_float64(-1, 1).cast[dtype]())
        cos_h[i] = Scalar[dtype](random_float64(-1, 1).cast[dtype]())
        sin_h[i] = Scalar[dtype](random_float64(-1, 1).cast[dtype]())

    for i in range(cols):
        gamma_h[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()

    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)
    var cos_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var sin_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var output_d = ctx.enqueue_create_buffer[dtype](rows * cols)

    var param_shape = Index(cols)

    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var output_buf = TileTensor(output_d, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_d, row_major(Coord(param_shape)))
    var cos_vals = TileTensor(cos_d, row_major(Coord(shape)))
    var sin_vals = TileTensor(sin_d, row_major(Coord(shape)))
    var epsilon = Scalar[dtype](0.001)
    var weight_offset = Scalar[dtype](0.0)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(cos_d, cos_h)
    ctx.enqueue_copy(sin_d, sin_h)

    @always_inline
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var data_buf} -> SIMD[dtype, width]:
        var idx = data_buf.layout(coords)
        return data_buf.raw_load[width=width, alignment=alignment](idx)

    @always_inline
    def cos_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var cos_vals} -> SIMD[dtype, width]:
        var idx = cos_vals.layout(coords)
        return cos_vals.raw_load[width=width, alignment=alignment](idx)

    @always_inline
    def sin_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var sin_vals} -> SIMD[dtype, width]:
        var idx = sin_vals.layout(coords)
        return sin_vals.raw_load[width=width, alignment=alignment](idx)

    @always_inline
    def output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) {var output_buf} -> None:
        var idx = output_buf.layout(coords)
        output_buf.raw_store[width=width, alignment=alignment](idx, val)

    @always_inline
    def bench_fn(
        mut b: Bencher,
    ) raises {
        var gamma,
        var epsilon,
        var weight_offset,
        var input_fn,
        var cos_fn,
        var sin_fn,
        var output_fn,
        imm,
    }:
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises {imm}:
            rms_norm_rope[
                dtype,
                dtype,
                dtype,
                rank,
                target="gpu",
                multiply_before_cast=False,
            ](
                input_fn,
                cos_fn,
                sin_fn,
                output_fn,
                Coord(shape),
                Int(cols),
                gamma,
                epsilon,
                weight_offset,
                ctx,
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    b.bench_function(
        bench_fn,
        BenchId(
            "rms_norm_rope",
            input_id=String(fn_name, "/", dtype, "/", shape),
        ),
    )

    ctx.synchronize()

    _ = data_d
    _ = gamma_d
    _ = cos_d
    _ = sin_d
    _ = output_d
    _ = data_h^
    _ = gamma_h^
    _ = cos_h^
    _ = sin_h^


def main() raises:
    comptime dtype = get_defined_dtype["dtype", .bfloat16]()
    comptime shape = int_list_to_tuple[
        get_defined_shape["shape", "32x2048x12x128"]()
    ]()

    var m = Bench(BenchConfig(num_repetitions=1))
    with DeviceContext() as ctx:
        bench_rms_norm_rope_gpu[dtype, shape](ctx, m, "rms_norm_rope_gpu")

    m.dump_report()
