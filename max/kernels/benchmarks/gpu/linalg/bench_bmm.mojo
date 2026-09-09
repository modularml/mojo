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

from std.sys import (
    get_defined_bool,
    get_defined_dtype,
    get_defined_int,
    simd_width_of,
)
from std.sys.info import has_amd_gpu_accelerator

from layout import Coord, TileTensor, row_major, CoordLike, Idx
import linalg.matmul.vendor.blas as vendor_blas
from max.algorithm.functional import elementwise
from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext, get_gpu_target
from internal_utils import arg_parse
from internal_utils._utils import (
    InitializationType,
    init_vector_launch,
)
from linalg.bmm import _batched_matmul_gpu

from std.utils import IndexList


def _ri(v: Int) -> Int64:
    return Int64(v)


def _get_run_name[
    c_type: DType,
    a_type: DType,
    *,
    transpose_b: Bool,
    use_vendor_blas: Bool,
    lambda_fn: Optional[epilogue_func_type] = None,
](b: Int, m: Int, n: Int, k: Int) -> String:
    var vendor_str = "vendor_bmm" if use_vendor_blas else "bmm"
    var type_str = String(
        "(in=", String(a_type), ",out=", String(c_type), ") : "
    )
    # B
    var b_str = String(b, "" if b else "_dynamic")
    # M
    var m_str = String(m, "" if m else "_dynamic")
    # N
    var n_str = String(n, "" if n else "_dynamic")
    # K
    var k_str = String(k, "" if k else "_dynamic")

    var transpose_b_str = String("/transpose_b=", transpose_b)

    return String(
        vendor_str,
        type_str,
        b_str,
        " x ",
        m_str,
        " x ",
        n_str,
        " x ",
        k_str,
        transpose_b_str,
    )


comptime epilogue_func_type = def[
    dtype: DType, width: SIMDLength, *, alignment: Int = 1
](SIMD[dtype, width]) capturing -> SIMD[dtype, width]


@always_inline
@__parameter
def elementwise_epilogue_fn[
    dtype: DType,
    width: SIMDLength,
    *,
    alignment: Int = 1,
](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return val + 2


def bench_bmm[
    BType: CoordLike,
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    c_type: DType,
    a_type: DType,
    /,
    *,
    use_vendor_blas: Bool = False,
    transpose_b: Bool = False,
    lambda_fn: Optional[epilogue_func_type] = None,
](
    ctx: DeviceContext,
    mut bench: Bench,
    b: BType,
    m: MType,
    n: NType,
    k: KType,
    init_type: InitializationType,
) raises:
    var a_size = Int(b.value()) * Int(m.value()) * Int(k.value())
    var b_size = (
        Int(b.value())
        * Int(n.value())
        * Int(k.value()) if transpose_b else Int(b.value())
        * Int(k.value())
        * Int(n.value())
    )
    var c_size = Int(b.value()) * Int(m.value()) * Int(n.value())

    var a_device_buffer = ctx.enqueue_create_buffer[a_type](a_size)
    var b_device_buffer = ctx.enqueue_create_buffer[a_type](b_size)
    var c_device_buffer = ctx.enqueue_create_buffer[c_type](c_size)

    var a_device = TileTensor(
        a_device_buffer,
        row_major(Coord(b, m, k)),
    ).as_unsafe_any_origin()

    var b_device = TileTensor(
        b_device_buffer,
        row_major(
            Coord(
                b,
                Idx[NType.static_value if transpose_b else KType.static_value],
                Idx[KType.static_value if transpose_b else NType.static_value],
            )
        ),
    ).as_unsafe_any_origin()
    var c_device = TileTensor(
        c_device_buffer,
        row_major(Coord(b, m, n)),
    ).as_unsafe_any_origin()

    # Initialize data on the device
    init_vector_launch[a_type](a_device_buffer, a_size, init_type, ctx)
    init_vector_launch[a_type](b_device_buffer, b_size, init_type, ctx)

    @__parameter
    @always_inline
    @__copy_capture(c_device)
    def epilogue_fn[
        dtype: DType,
        width: SIMDLength,
        rank: Int,
        *,
        alignment: Int = 1,
    ](idx: IndexList[rank], val: SIMD[dtype, width]) capturing -> None:
        comptime func = lambda_fn.value()
        var update_val = func(val)
        c_device.store_linear(idx, update_val.cast[c_device.dtype]())

    comptime pack_size = simd_width_of[c_type, target=get_gpu_target()]()

    @always_inline
    def func[simd_width: Int, alignment: Int = 1](idx: Coord) {var}:
        var val = c_device.load[width=simd_width](idx)
        comptime element_lambda = lambda_fn.value()
        var update_val = element_lambda(val)

        c_device.store(
            idx,
            update_val,
        )

    @always_inline
    def bench_func(
        mut bench: Bencher,
    ) {var a_device, var b_device, var c_device, imm}:
        @always_inline
        def kernel_launch(ctx: DeviceContext, iteration: Int) raises {imm}:
            comptime if use_vendor_blas:
                comptime if has_amd_gpu_accelerator():
                    var c_buffer = TileTensor(
                        c_device.ptr, row_major(Coord(m, n))
                    )
                    var a_buffer = TileTensor(
                        a_device.ptr, row_major(Coord(m, k))
                    )
                    var b_buffer = TileTensor(
                        b_device.ptr,
                        row_major(
                            Coord(
                                Idx[
                                    NType.static_value if transpose_b else KType.static_value
                                ],
                                Idx[
                                    KType.static_value if transpose_b else NType.static_value
                                ],
                            )
                        ),
                    )

                    vendor_blas.matmul[use_tf32=True](
                        ctx,
                        c_buffer,
                        a_buffer,
                        b_buffer,
                        c_row_major=True,
                        transpose_b=transpose_b,
                        batch_size=Int(b.value()),
                    )
                else:
                    # Fallback vendor BMM for non-AMD GPUs
                    for i in range(Int(b.value())):
                        var c_ptr = c_device.ptr + (
                            i * Int(m.value()) * Int(n.value())
                        )
                        var a_ptr = a_device.ptr + (
                            i * Int(m.value()) * Int(k.value())
                        )
                        var b_ptr = b_device.ptr + (
                            i * Int(k.value()) * Int(n.value())
                        )

                        var c_buffer = TileTensor(c_ptr, row_major(Coord(m, n)))
                        var a_buffer = TileTensor(a_ptr, row_major(Coord(m, k)))
                        var b_buffer = TileTensor(
                            b_ptr,
                            row_major(
                                Coord(
                                    Idx[
                                        NType.static_value if transpose_b else KType.static_value
                                    ],
                                    Idx[
                                        KType.static_value if transpose_b else NType.static_value
                                    ],
                                )
                            ),
                        )

                        vendor_blas.matmul[use_tf32=True](
                            ctx,
                            c_buffer,
                            a_buffer,
                            b_buffer,
                            c_row_major=True,
                            transpose_b=transpose_b,
                        )
                ctx.synchronize()

                # Epilogue
                comptime if lambda_fn:
                    elementwise[pack_size, target="gpu"](
                        func,
                        (b, m, n),
                        ctx,
                    )
            else:
                comptime if lambda_fn:
                    _batched_matmul_gpu[
                        transpose_b=transpose_b,
                        elementwise_epilogue_fn=epilogue_fn,
                    ](
                        c_device,
                        a_device,
                        b_device,
                        ctx,
                    )
                else:
                    _batched_matmul_gpu[transpose_b=transpose_b](
                        c_device,
                        a_device,
                        b_device,
                        ctx,
                    )

        bencher_iter_custom(bench, kernel_launch, ctx)

    bench.bench_function(
        bench_func,
        BenchId(
            _get_run_name[
                c_type,
                a_type,
                transpose_b=transpose_b,
                use_vendor_blas=use_vendor_blas,
                lambda_fn=lambda_fn,
            ](Int(b.value()), Int(m.value()), Int(n.value()), Int(k.value()))
        ),
        # TODO: Pick relevant benchmetric
        [
            ThroughputMeasure(
                BenchMetric.flops,
                2
                * Int(b.value())
                * Int(m.value())
                * Int(n.value())
                * Int(k.value()),
            )
        ],
    )

    # Retain our buffers till the end.
    _ = a_device_buffer^
    _ = b_device_buffer^
    _ = c_device_buffer^


def create_bmm_bench[
    BType: CoordLike,
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    c_type: DType,
    a_type: DType,
    *,
    transpose_b: Bool,
    use_vendor_blas: Bool,
    lambda_fn: Optional[epilogue_func_type] = None,
](
    ctx: DeviceContext,
    mut bench: Bench,
    b: BType,
    m: MType,
    n: NType,
    k: KType,
    init_type: InitializationType,
) raises:
    bench_bmm[
        c_type,
        a_type,
        transpose_b=transpose_b,
        use_vendor_blas=use_vendor_blas,
        lambda_fn=lambda_fn,
    ](
        ctx,
        bench,
        b,
        m,
        n,
        k,
        init_type,
    )


def main() raises:
    comptime a_type = get_defined_dtype["atype", .bfloat16]()
    comptime c_type = get_defined_dtype["ctype", .bfloat16]()

    var b = Int(arg_parse("B", 1))
    var m = Int(arg_parse("M", 1))
    comptime N = get_defined_int["N", 128]()
    comptime K = get_defined_int["K", 128]()
    var init_type = InitializationType.from_str(
        arg_parse("init_type", "uniform_distribution")
    )
    comptime transpose_b = True
    comptime use_vendor_blas = get_defined_bool["use_vendor_blas", False]()

    var bench = Bench()
    with DeviceContext() as ctx:
        create_bmm_bench[
            c_type,
            a_type,
            transpose_b=transpose_b,
            use_vendor_blas=use_vendor_blas,
        ](
            ctx,
            bench,
            b,
            m,
            Idx[N],
            Idx[K],
            init_type,
        )

    bench.dump_report()
