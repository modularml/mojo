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

from std.collections import Optional
from std.sys import (
    get_defined_bool,
    get_defined_dtype,
    get_defined_int,
    align_of,
)

import linalg.matmul.vendor.blas as vendor_blas
from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext
from internal_utils import arg_parse, CacheBustingBuffer
from internal_utils._utils import InitializationType
from layout import CoordLike, Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu import _matmul_gpu
from linalg.utils import elementwise_compute_lambda_type
from linalg.matmul.gpu.amd.amd_ping_pong_matmul import (
    amd_ping_pong_matmul as ping_pong_matmul,
)
from std.utils import IndexList


def _get_run_name[
    dtype: DType,
    *,
    transpose_b: Bool,
    cache_busting: Bool,
    use_vendor_blas: Bool,
](shape_c: Coord, shape_a: Coord, shape_b: Coord) -> String:
    var vendor_str = "vendor_matmul" if use_vendor_blas else "matmul"
    var type_str = String("(", dtype, ") : ")
    # M
    var m_str = String(shape_c[0], "_dynamic")
    # N
    var n_str = String(
        shape_c[1],
        "_dynamic" if not shape_c.element_types[1].is_static_value else "",
    )
    # K
    var k_str = String(
        shape_a[1],
        "_dynamic" if not shape_a.element_types[1].is_static_value else "",
    )

    var transpose_b_str = String(
        "/transpose_b=", "True" if transpose_b else "False"
    )
    var cache_busting_str = String(
        "/cache_busting=", "True" if cache_busting else "False"
    )
    return String(
        vendor_str,
        type_str,
        m_str,
        " x ",
        n_str,
        " x ",
        k_str,
        transpose_b_str,
        cache_busting_str,
    )


def bench_matmul[
    dtype: DType,
    *,
    cache_busting: Bool,
    use_vendor_blas: Bool,
    transpose_b: Bool = False,
    epilogue: Bool = False,
](
    ctx: DeviceContext,
    mut b: Bench,
    shape_c: Coord,
    shape_a: Coord,
    shape_b: Coord,
    init_type: InitializationType,
) raises:
    # Choose a size larger than the two times the L2 cache
    # 128 MiB is larger that twice the L2 cache on the A100, A10, and L4.
    # update: using 512 to be 2x the infinity cache on MI300x
    @always_inline
    def get_size(shape: Coord) -> Int:
        return Int(shape[0].value()) * Int(shape[1].value())

    comptime simd_size = 4

    # Benchmark with the same data type for C as A and B
    comptime c_dtype = dtype

    var cb_a = CacheBustingBuffer[dtype](get_size(shape_a), simd_size, ctx)
    var cb_b = CacheBustingBuffer[dtype](get_size(shape_b), simd_size, ctx)
    var cb_c = CacheBustingBuffer[c_dtype](get_size(shape_c), simd_size, ctx)

    cb_a.init_on_device(init_type, ctx)
    cb_b.init_on_device(init_type, ctx)

    @always_inline
    def bench_func(
        mut b: Bencher,
    ) {
        var cb_a,
        var cb_b,
        var cb_c,
        var shape_c,
        var shape_a,
        var shape_b,
        imm,
    }:
        @always_inline
        def kernel_launch(ctx: DeviceContext, iteration: Int) raises {imm}:
            var tensor_a = TileTensor(
                cb_a.offset_ptr(iteration), row_major(shape_a)
            )
            var tensor_b = TileTensor(
                cb_b.offset_ptr(iteration), row_major(shape_b)
            )
            var tensor_c = TileTensor(
                cb_c.offset_ptr(iteration), row_major(shape_c)
            )

            @__parameter
            @always_inline
            @__copy_capture(tensor_c)
            def test_lambda_add_coords_prod[
                _dtype: DType,
                width: SIMDLength,
                *,
                alignment: Int = align_of[SIMD[_dtype, width]](),
            ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
                _dtype, width
            ]:
                # FIXME(mojo-compiler): restore the comptime-assert below
                # once the Mojo `Int.__ge__` comptime-cycle bug is fixed.
                # The assert is always true here (tensor_c is the 2D output
                # TileTensor, so flat_rank == 2 by construction) but triggers
                # a stdlib parse-order cycle: resolving Int.__ge__ from a
                # comptime context inside this parametric capturing closure
                # transitively requires the EqualityComparable.__eq__ default
                # body, whose `comptime for i in range(...)` re-resolves
                # Int.__ge__ via the range iterator's `nth`. Manifests as
                # "attempt to resolve a recursive reference to declaration
                # 'Int.__ge__'". `> 1` / `== 2` substitutions don't trigger;
                # only `>=` in this comptime + parametric-capturing-closure
                # combination does. See commit e65f549ada3 for the audit.
                # comptime assert tensor_c.flat_rank >= 2
                var x = tensor_c.load[width=width](Coord(idx)).cast[_dtype]()
                var y = val * x
                return y

            comptime optional_lambda_fn = Optional[
                elementwise_compute_lambda_type
            ](test_lambda_add_coords_prod) if epilogue else None

            comptime if use_vendor_blas:
                vendor_blas.matmul[use_tf32=True](
                    ctx,
                    tensor_c,
                    tensor_a,
                    tensor_b,
                    c_row_major=True,
                    transpose_b=transpose_b,
                )
            else:
                comptime use_ping_pong_matmul = get_defined_bool[
                    "use_ping_pong_matmul", True
                ]()
                comptime enable_swizzle = get_defined_bool[
                    "enable_swizzle", True
                ]()

                comptime if use_ping_pong_matmul:
                    ping_pong_matmul[enable_swizzle=enable_swizzle](
                        tensor_a,
                        tensor_b,
                        tensor_c,
                        ctx,
                    )
                else:
                    _matmul_gpu[
                        use_tensor_core=True,
                        transpose_b=transpose_b,
                        elementwise_compute_lambda_fn=optional_lambda_fn,
                    ](
                        tensor_c,
                        tensor_a,
                        tensor_b,
                        ctx,
                    )

        bencher_iter_custom(b, kernel_launch, ctx)

    var flops = ThroughputMeasure(
        BenchMetric.flops,
        # Flop: 2*M*N*K. Use A and C shapes since they're not transposed.
        2
        * Int(shape_c[0].value())
        * Int(shape_c[1].value())
        * Int(shape_a[1].value()),
    )
    b.bench_function(
        bench_func,
        BenchId(
            _get_run_name[
                dtype,
                transpose_b=transpose_b,
                cache_busting=cache_busting,
                use_vendor_blas=use_vendor_blas,
            ](shape_c, shape_a, shape_b)
        ),
        # TODO: Pick relevant benchmetric
        [flops],
    )


def create_matmul_bench[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    dtype: DType,
    *,
    transpose_b: Bool,
    cache_busting: Bool,
    use_vendor_blas: Bool,
    epilogue: Bool,
](
    ctx: DeviceContext,
    mut b: Bench,
    m: MType,
    n: NType,
    k: KType,
    init_type: InitializationType,
) raises:
    var b_shape = Coord(
        Idx[NType.static_value if transpose_b else KType.static_value],
        Idx[KType.static_value if transpose_b else NType.static_value],
    )

    bench_matmul[
        dtype,
        transpose_b=transpose_b,
        cache_busting=cache_busting,
        use_vendor_blas=use_vendor_blas,
        epilogue=epilogue,
    ](
        ctx,
        b,
        Coord(m, n),
        Coord(m, k),
        b_shape,
        init_type,
    )


def main() raises:
    comptime dtype = get_defined_dtype["dtype", .bfloat16]()

    var M = Int(arg_parse("M", 1))
    comptime N = get_defined_int["N", 1]()
    comptime K = get_defined_int["K", 1]()
    var init_type = InitializationType.from_str(
        arg_parse("init_type", "uniform_distribution")
    )
    comptime cache_busting = True
    comptime transpose_b = True
    comptime use_vendor_blas = get_defined_bool["use_vendor_blas", False]()
    comptime epilogue = get_defined_bool["epilogue", False]()

    var m = Bench()
    with DeviceContext() as ctx:
        create_matmul_bench[
            dtype,
            transpose_b=transpose_b,
            cache_busting=cache_busting,
            use_vendor_blas=use_vendor_blas,
            epilogue=epilogue,
        ](
            ctx,
            m,
            M,
            Idx[N],
            Idx[K],
            init_type,
        )

    m.dump_report()
