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
from nn.normalization import rms_norm_fused_residual_add

from std.utils.index import Index, IndexList


def bench_rms_norm_fused_residual_add_gpu[
    rank: Int, //, dtype: DType, shape: IndexList[rank]
](ctx: DeviceContext, mut b: Bench, fn_name: String) raises:
    comptime cols = shape[rank - 1]
    comptime rows = shape.flattened_length() // cols

    var data_h = List(length=rows * cols, fill=Scalar[dtype](0))
    var residual_h = List(length=rows * cols, fill=Scalar[dtype](0))
    var gamma1_h = List(length=cols, fill=Scalar[dtype](0))
    var gamma2_h = List(length=cols, fill=Scalar[dtype](0))

    for i in range(rows * cols):
        data_h[i] = Scalar[dtype](random_float64(0, 100).cast[dtype]())
        residual_h[i] = Scalar[dtype](random_float64(0, 100).cast[dtype]())

    for i in range(cols):
        gamma1_h[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()
        gamma2_h[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()

    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var residual_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var output_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var residual_output_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma1_d = ctx.enqueue_create_buffer[dtype](cols)
    var gamma2_d = ctx.enqueue_create_buffer[dtype](cols)

    var param_shape = Index(cols)

    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var residual_buf = TileTensor(residual_d, row_major(Coord(shape)))
    var output_buf = TileTensor(output_d, row_major(Coord(shape)))
    var residual_output_buf = TileTensor(
        residual_output_d, row_major(Coord(shape))
    )
    var gamma1 = TileTensor(gamma1_d, row_major(Coord(param_shape)))
    var gamma2 = TileTensor(gamma2_d, row_major(Coord(param_shape)))
    var epsilon1 = Float32(0.001)
    var epsilon2 = Float32(0.001)
    var weight_offset1 = Scalar[dtype](0.0)
    var weight_offset2 = Scalar[dtype](0.0)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(residual_d, residual_h)
    ctx.enqueue_copy(gamma1_d, gamma1_h)
    ctx.enqueue_copy(gamma2_d, gamma2_h)

    # `Output0Fn` precedes `OutputResidualFn`, the reverse of the deleted
    # `..._gpu` kernel's order.
    @always_inline
    def input_fn[
        width: Int
    ](coords: Coord) {var data_buf} -> SIMD[dtype, width]:
        return data_buf.load[width=width](coords)

    @always_inline
    def residual_input_fn[
        width: Int
    ](coords: Coord) {var residual_buf} -> SIMD[dtype, width]:
        return residual_buf.load[width=width](coords)

    @always_inline
    def output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) {var output_buf} -> None:
        output_buf.store[alignment=alignment](coords, val)

    @always_inline
    def residual_output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) {var residual_output_buf} -> None:
        residual_output_buf.store[alignment=alignment](coords, val)

    @always_inline
    def bench_fn(
        mut b: Bencher,
    ) raises {
        var gamma1,
        var epsilon1,
        var weight_offset1,
        var gamma2,
        var epsilon2,
        var weight_offset2,
        var input_fn,
        var residual_input_fn,
        var output_fn,
        var residual_output_fn,
        imm,
    }:
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises {imm}:
            rms_norm_fused_residual_add[
                dtype,
                rank,
                target="gpu",
                multiply_before_cast=True,
            ](
                input_fn,
                residual_input_fn,
                output_fn,
                residual_output_fn,
                Coord(shape),
                Int(cols),
                gamma1,
                epsilon1.cast[dtype](),
                weight_offset1,
                gamma2,
                epsilon2.cast[dtype](),
                weight_offset2,
                ctx,
            )

        bencher_iter_custom(b, kernel_launch, ctx)

    b.bench_function(
        bench_fn,
        BenchId(
            "rms_norm_fused_residual_add",
            input_id=String(fn_name, "/", dtype, "/", shape),
        ),
    )

    ctx.synchronize()

    _ = data_d
    _ = residual_d
    _ = output_d
    _ = residual_output_d
    _ = gamma1_d
    _ = gamma2_d
    _ = data_h^
    _ = residual_h^
    _ = gamma1_h^
    _ = gamma2_h^


def main() raises:
    comptime dtype = get_defined_dtype["dtype", .bfloat16]()
    comptime shape = int_list_to_tuple[get_defined_shape["shape", "1x5376"]()]()

    var m = Bench(BenchConfig(num_repetitions=1))
    with DeviceContext() as ctx:
        bench_rms_norm_fused_residual_add_gpu[dtype, shape](
            ctx, m, "rms_norm_fused_residual_add_gpu"
        )

    m.dump_report()
