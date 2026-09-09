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
    align_of,
    get_defined_bool,
    get_defined_dtype,
    get_defined_string,
)

from max.benchmark import bencher_iter_custom
from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from max.gpu.host import DeviceContext
from internal_utils import get_defined_shape, int_list_to_tuple
from layout import Coord, TileTensor, coord, row_major
from nn.normalization import layer_norm, rms_norm_gpu
from std.utils.coord import ComptimeInt

from std.utils.index import Index, IndexList


# The shape `Coord` is passed in (built by the caller) rather than constructed
# here: a statically-shaped `Coord` (outer dims as `ComptimeInt`, what the graph
# compiler's `input.shape_coord()` yields) folds the per-row row->n-D divisors to
# magic-multiply + shift, while an all-runtime `Coord` emits a runtime integer
# divide. Same kernel either way; passing the `Coord` in lets one bench function
# measure both arms and isolate the static-divisor folding. `shape` (comptime)
# still drives host buffer sizing and the `BenchId`.
def bench_layer_norm_gpu[
    rank: Int,
    //,
    dtype: DType,
    shape: IndexList[rank],
    static_shape: Bool = False,
](
    ctx: DeviceContext,
    mut b: Bench,
    fn_name: String,
    shape_coord: Coord,
) raises:
    comptime cols = shape[rank - 1]
    comptime rows = shape.flattened_length() // cols

    var data_h = List(length=rows * cols, fill=Scalar[dtype](0))
    var gamma_h = List(length=cols, fill=Scalar[dtype](0))
    var beta_h = List(length=cols, fill=Scalar[dtype](0))

    for i in range(rows * cols):
        var val = Scalar[dtype](random_float64(0, 100).cast[dtype]())
        data_h[i] = val

    for i in range(cols):
        gamma_h[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()
        beta_h[i] = (Float64(i) / Float64(cols)).cast[dtype]()

    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    # Distinct output buffer: `input_fn` (reads) and `output_fn` (writes) are
    # separate value-closure args and must reference distinct buffer origins
    # (mirrors test_layer_norm.mojo's `run_layer_norm_gpu`).
    var out_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)
    var beta_d = ctx.enqueue_create_buffer[dtype](cols)

    var param_shape = Index(cols)

    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var out_buf = TileTensor(out_d, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_d, row_major(Coord(param_shape)))
    var beta = TileTensor(beta_d, row_major(Coord(param_shape)))
    var epsilon = Scalar[dtype](0)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(beta_d, beta_h)

    # `layer_norm` takes gamma/beta as `TileTensor`s directly (no
    # `gamma_fn`); `input_fn`/`output_fn` are unified closures matching its
    # (width, alignment) `Coord` signatures.
    @always_inline
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var data_buf} -> SIMD[dtype, width]:
        var idx = data_buf.layout(coords)

        return data_buf.raw_load[width=width, alignment=alignment](idx)

    @always_inline
    def output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) {var out_buf} -> None:
        var idx = out_buf.layout(coords)

        out_buf.raw_store[width=width, alignment=alignment](idx, val)

    @always_inline
    def kernel_launch(
        ctx: DeviceContext,
    ) raises {imm}:
        comptime if static_shape:
            layer_norm[dtype, rank, target="gpu"](
                input_fn,
                output_fn,
                shape_coord,
                ComptimeInt[cols](),
                gamma,
                beta,
                epsilon,
                ctx,
            )
        else:
            layer_norm[dtype, rank, target="gpu"](
                input_fn,
                output_fn,
                shape_coord,
                Int(cols),
                gamma,
                beta,
                epsilon,
                ctx,
            )

    @always_inline
    def bench_fn(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, kernel_launch, ctx)

    comptime shape_tag = "static" if static_shape else "dynamic"
    b.bench_function(
        bench_fn,
        BenchId(
            "layer_norm",
            input_id=String(fn_name, shape_tag, dtype, shape, sep="/"),
        ),
    )

    ctx.synchronize()

    _ = data_d
    _ = out_d
    _ = gamma_d
    _ = beta_d
    _ = data_h^
    _ = gamma_h^
    _ = beta_h^


def bench_rms_norm_gpu[
    rank: Int, //, dtype: DType, shape: IndexList[rank]
](ctx: DeviceContext, mut b: Bench, fn_name: String) raises:
    comptime cols = shape[rank - 1]
    comptime rows = shape.flattened_length() // cols

    var data_h = List(length=rows * cols, fill=Scalar[dtype](0))
    var gamma_h = List(length=cols, fill=Scalar[dtype](0))

    for i in range(rows * cols):
        var val = Scalar[dtype](random_float64(0, 100).cast[dtype]())
        data_h[i] = val

    for i in range(cols):
        gamma_h[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()

    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)

    var param_shape = Index(cols)

    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_d, row_major(Coord(param_shape)))
    var epsilon = Float32(0.001)
    var weight_offset = Scalar[dtype](0.0)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)

    # `rms_norm_gpu` migrated to a `Coord` shape boundary (softmax PR #88203).
    @__copy_capture(data_buf)
    @always_inline
    @__parameter
    def input_fn[width: Int](coords: Coord) -> SIMD[dtype, width]:
        var idx = data_buf.layout(coords)

        # Match the MOGG lambda contract (reductions.mojo passes
        # `element_alignment=width` to `_lambda_load`): vector loads are
        # aligned to the full SIMD width.
        return data_buf.raw_load[
            width=width, alignment=align_of[SIMD[dtype, width]]()
        ](idx)

    @always_inline
    @__copy_capture(data_buf)
    @__parameter
    def identity_output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) -> None:
        var idx = data_buf.layout(coords)
        data_buf.raw_store[width=width, alignment=alignment](idx, val)

    @always_inline
    def kernel_launch(ctx: DeviceContext) raises {mut data_d, imm}:
        rms_norm_gpu[
            rank, input_fn, identity_output_fn, multiply_before_cast=True
        ](
            Coord(shape),
            gamma,
            epsilon,
            weight_offset,
            ctx,
        )

    @always_inline
    def bench_fn(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, kernel_launch, ctx)

    b.bench_function(
        bench_fn,
        BenchId("rms_norm", input_id=String(fn_name, "/", dtype, "/", shape)),
    )

    ctx.synchronize()

    _ = data_d
    _ = gamma_d
    _ = data_h^
    _ = gamma_h^


def main() raises:
    comptime dtype = get_defined_dtype["dtype", .bfloat16]()
    comptime shape = int_list_to_tuple[
        get_defined_shape["shape", "256x256"]()
    ]()
    # Which kernel to benchmark. Default preserves the historical dispatch
    # (rank-2 -> layer_norm, rank-3 -> rms_norm); pass `kernel=layer_norm` to
    # benchmark `layer_norm_gpu` at rank-3 (e.g. `[B, S, H]` vision shapes),
    # which the default dispatch could not reach.
    comptime kernel = get_defined_string["kernel", "auto"]()
    # For `layer_norm`, also run a statically-shaped arm so the static-divisor
    # folding path (the optimization that motivated the `Coord` boundary) is
    # measured, not just the runtime-divide path.
    comptime want_static = get_defined_bool["static", True]()

    var m = Bench(BenchConfig(num_repetitions=1))
    with DeviceContext() as ctx:
        comptime if kernel == "rms_norm" or (
            kernel == "auto" and len(shape) == 3
        ):
            bench_rms_norm_gpu[dtype, shape](ctx, m, "rms_norm_gpu")
        else:
            # `layer_norm`: always run the dynamic (runtime-divide) arm; add the
            # static (folded-divisor) arm so the two are directly comparable. The
            # `Coord`s are built here (at `main` scope, where `shape[i]` folds at
            # comptime): `Coord(shape)` erases dims to runtime; `coord[...]`
            # keeps them as `ComptimeInt`.
            bench_layer_norm_gpu[dtype, shape, static_shape=False](
                ctx, m, "layer_norm_gpu", Coord(shape)
            )
            comptime if want_static:
                comptime if len(shape) == 2:
                    bench_layer_norm_gpu[dtype, shape, static_shape=True](
                        ctx, m, "layer_norm_gpu", coord[shape[0], shape[1]]
                    )
                elif len(shape) == 3:
                    bench_layer_norm_gpu[dtype, shape, static_shape=True](
                        ctx,
                        m,
                        "layer_norm_gpu",
                        coord[shape[0], shape[1], shape[2]],
                    )

    m.dump_report()
