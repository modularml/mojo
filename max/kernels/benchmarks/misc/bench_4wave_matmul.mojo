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
from max.gpu import global_idx, grid_dim, block_dim, thread_idx, block_idx
from max.gpu.host import DeviceContext
from max.gpu.primitives import block
from std.memory import dealloc
from internal_utils import (
    CacheBustingBuffer,
    arg_parse,
    pytorch_like_tolerances_for,
)
from std.random import rand
from internal_utils._utils import InitializationType, init_vector_launch
from layout import (
    TileTensor,
    Idx,
    Coord,
    CoordLike,
    row_major,
)
from linalg.matmul.gpu.amd.amd_4wave_matmul import structured_4wave_matmul
from linalg.utils import (
    elementwise_compute_lambda_type,
    elementwise_epilogue_type,
)
from std.utils import IndexList


def _verify_buffers_gpu[
    c_type: DType, BLOCK_SIZE: Int
](
    output: ImmPointer[Scalar[c_type], ImmutAnyOrigin],
    reference: ImmPointer[Scalar[c_type], ImmutAnyOrigin],
    length_dev: Int32,
    atol: Float32,
    rtol: Float32,
    result: MutPointer[Float32, MutAnyOrigin],
):
    """GPU kernel that computes verification metrics in one pass.

    Each block computes partial reductions and writes 5 Float32 values:
      [0] abs_diff_sum — for relative difference metric
      [1] abs_ref_sum  — for relative difference metric
      [2] max_violation — max(|x-y| - (atol + rtol*|y|)), <=0 means pass
      [3] out_nz — 1.0 if any output element is nonzero
      [4] ref_nz — 1.0 if any reference element is nonzero
    """
    var length = Int(length_dev)
    # Per-thread accumulators
    var abs_diff_sum: Float32 = 0
    var abs_ref_sum: Float32 = 0
    var max_violation = Float32.MIN_FINITE
    var out_nz: Float32 = 0
    var ref_nz: Float32 = 0

    # Grid-stride loop
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < length:
        var x = output[i].cast[.float32]()
        var y = reference[i].cast[.float32]()
        abs_diff_sum += abs(x - y)
        abs_ref_sum += abs(y)
        max_violation = max(max_violation, abs(x - y) - (atol + rtol * abs(y)))
        if x != 0:
            out_nz = 1.0
        if y != 0:
            ref_nz = 1.0
        i += stride

    # Block-wide reductions
    abs_diff_sum = block.sum[block_size=BLOCK_SIZE](abs_diff_sum)
    abs_ref_sum = block.sum[block_size=BLOCK_SIZE](abs_ref_sum)
    max_violation = block.max[block_size=BLOCK_SIZE](max_violation)
    out_nz = block.max[block_size=BLOCK_SIZE](out_nz)
    ref_nz = block.max[block_size=BLOCK_SIZE](ref_nz)

    # Each block writes its partial results
    if thread_idx.x == 0:
        var base = block_idx.x * 5
        result[base + 0] = abs_diff_sum
        result[base + 1] = abs_ref_sum
        result[base + 2] = max_violation
        result[base + 3] = out_nz
        result[base + 4] = ref_nz


def verify_matmul[
    c_type: DType,
    a_type: DType,
    *,
    transpose_b: Bool = False,
    init_on_gpu: Bool = True,
](
    ctx: DeviceContext,
    c_shape: Coord,
    a_shape: Coord,
    b_shape: Coord,
    init_type: InitializationType,
) raises:
    var c_size = Int(c_shape[0].value()) * Int(c_shape[1].value())
    var a_size = Int(a_shape[0].value()) * Int(a_shape[1].value())
    var b_size = Int(b_shape[0].value()) * Int(b_shape[1].value())

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_device_nd = TileTensor(a_device, row_major(a_shape))
    var b_device = ctx.enqueue_create_buffer[a_type](b_size)
    var b_device_nd = TileTensor(b_device, row_major(b_shape))
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_nd = TileTensor(c_device, row_major(c_shape))
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_ref_nd = TileTensor(c_device_ref, row_major(c_shape))

    # Initialize matmul operands
    comptime if not init_on_gpu:
        var a_host_ptr_alloc = alloc[Scalar[a_type]](
            {count = a_size}
        ).into_managed()
        var a_host_ptr = a_host_ptr_alloc.unsafe_ptr()
        var b_host_ptr_alloc = alloc[Scalar[a_type]](
            {count = b_size}
        ).into_managed()
        var b_host_ptr = b_host_ptr_alloc.unsafe_ptr()
        var a_host = TileTensor(a_host_ptr, row_major(a_shape))
        var b_host = TileTensor(b_host_ptr, row_major(b_shape))

        comptime if a_type.is_float8():
            rand(a_host.ptr, a_host.num_elements())
            rand(b_host.ptr, b_host.num_elements())
        else:
            if init_type == InitializationType.zero:
                _ = a_host.fill(0)
                _ = b_host.fill(0)
            elif init_type == InitializationType.one:
                _ = a_host.fill(1)
                _ = b_host.fill(1)
            elif init_type == InitializationType.uniform_distribution:
                rand(a_host.ptr, a_host.num_elements())
                rand(b_host.ptr, b_host.num_elements())
            elif init_type == InitializationType.arange:
                for i in range(a_host.num_elements()):
                    a_host.raw_store(i, Scalar[a_type](i))
                for i in range(b_host.num_elements()):
                    b_host.raw_store(i, Scalar[a_type](i))
        # Move operands to the Device
        ctx.enqueue_copy(a_device, a_host_ptr)
        ctx.enqueue_copy(b_device, b_host_ptr)
        ctx.synchronize()
        dealloc(a_host_ptr_alloc^)
        dealloc(b_host_ptr_alloc^)
    else:
        init_vector_launch[a_type](a_device, a_size, init_type, ctx)
        init_vector_launch[a_type](b_device, b_size, init_type, ctx)

    vendor_blas.matmul[use_tf32=True](
        ctx,
        c_device_ref_nd,
        a_device_nd,
        b_device_nd,
        c_row_major=True,
        transpose_b=transpose_b,
    )

    structured_4wave_matmul(a_device_nd, b_device_nd, c_device_nd, ctx)

    # Launch GPU verification kernel
    comptime NUM_BLOCKS = 32
    comptime BLOCK_SIZE = 256

    var rtol: Float32
    var atol: Float32
    comptime if a_type.is_float8():
        rtol = 1e-2
        atol = 1e-2
    else:
        var rtol64: Float64
        var atol64: Float64
        rtol64, atol64 = pytorch_like_tolerances_for[.bfloat16]()
        rtol = Float32(rtol64)
        atol = Float32(atol64)

    var result_device = ctx.enqueue_create_buffer[.float32](NUM_BLOCKS * 5)

    comptime kernel = _verify_buffers_gpu[c_type, BLOCK_SIZE]
    ctx.enqueue_function[kernel](
        c_device,
        c_device_ref,
        Int32(c_size),
        atol,
        rtol,
        result_device,
        grid_dim=NUM_BLOCKS,
        block_dim=BLOCK_SIZE,
    )

    # Copy back only NUM_BLOCKS * 5 Float32 values
    var result_host_alloc = alloc[Float32](
        {count = NUM_BLOCKS * 5}
    ).into_managed()
    var result_host = result_host_alloc.unsafe_ptr()
    ctx.enqueue_copy(result_host, result_device)
    ctx.synchronize()

    # Reduce partial results from all blocks
    var total_abs_diff: Float32 = 0
    var total_abs_ref: Float32 = 0
    var worst_violation = Float32.MIN_FINITE
    var any_out_nz: Float32 = 0
    var any_ref_nz: Float32 = 0

    for b_idx in range(NUM_BLOCKS):
        var base = b_idx * 5
        total_abs_diff += result_host[base + 0]
        total_abs_ref += result_host[base + 1]
        worst_violation = max(worst_violation, result_host[base + 2])
        any_out_nz = max(any_out_nz, result_host[base + 3])
        any_ref_nz = max(any_ref_nz, result_host[base + 4])

    dealloc(result_host_alloc^)

    # Check zero/nonzero expectations
    var c_is_zeros = any_out_nz == 0
    var c_ref_is_zeros = any_ref_nz == 0

    if init_type == InitializationType.zero:
        if not c_is_zeros:
            raise "matmul verification failed: kernel output should be all zeros for zero input"
        if not c_ref_is_zeros:
            raise "matmul verification failed: vendor BLAS output should be all zeros for zero input"
    else:
        if c_is_zeros:
            raise "matmul verification failed: kernel output is all zeros"
        if c_ref_is_zeros:
            raise "matmul verification failed: vendor BLAS output is all zeros"

    # Check relative difference: sum(|x-y|) / sum(|y|) <= 0.001
    if total_abs_ref > 0:
        var rel_diff = total_abs_diff / total_abs_ref
        if rel_diff > 0.001:
            raise String(
                "matmul verification failed (relative_difference): ",
                rel_diff,
                " > 0.001",
            )

    # Check element-wise tolerance: max(|x-y| - (atol + rtol*|y|)) <= 0
    if worst_violation > 0:
        raise String(
            (
                "matmul verification failed (element-wise tolerance): worst"
                " violation = "
            ),
            worst_violation,
        )

    print("\n=== TEST PASSED ===\n")


def _get_run_name[
    c_type: DType,
    a_type: DType,
    *,
    transpose_b: Bool,
    cache_busting: Bool,
    use_vendor_blas: Bool,
](shape_c: Coord, shape_a: Coord, shape_b: Coord) -> String:
    var vendor_str = "vendor_matmul" if use_vendor_blas else "matmul"
    var type_str = String(
        "(in=", String(a_type), ",out=", String(c_type), ") : "
    )
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
    c_type: DType,
    a_type: DType,
    *,
    cache_busting: Bool,
    use_vendor_blas: Bool,
    transpose_b: Bool = False,
    enable_compute_epilogue: Bool = False,
    enable_normal_epilogue: Bool = False,
](
    ctx: DeviceContext,
    mut b: Bench,
    shape_c: Coord,
    shape_a: Coord,
    shape_b: Coord,
    init_type: InitializationType,
    verify: Bool,
    run_benchmark: Bool = True,
) raises:
    # Choose a size larger than the two times the L2 cache
    # 128 MiB is larger that twice the L2 cache on the A100, A10, and L4.
    # update: using 512 to be 2x the infinity cache on MI300x
    @always_inline
    def get_size(shape: Coord) -> Int:
        return Int(shape[0].value()) * Int(shape[1].value())

    comptime simd_size = 4
    var cb_a = CacheBustingBuffer[a_type](get_size(shape_a), simd_size, ctx)
    var cb_b = CacheBustingBuffer[a_type](get_size(shape_b), simd_size, ctx)
    var cb_c = CacheBustingBuffer[c_type](get_size(shape_c), simd_size, ctx)
    # TODO: remove init_on_gpu flag and the loading on CPU
    comptime init_on_gpu = True

    comptime if not init_on_gpu:
        var a_host_ptr_alloc = alloc[Scalar[a_type]](
            {count = cb_a.alloc_size()}
        ).into_managed()
        var a_host_ptr = a_host_ptr_alloc.unsafe_ptr()
        var b_host_ptr_alloc = alloc[Scalar[a_type]](
            {count = cb_b.alloc_size()}
        ).into_managed()
        var b_host_ptr = b_host_ptr_alloc.unsafe_ptr()
        var a_host = TileTensor(a_host_ptr, row_major(cb_a.alloc_size()))
        var b_host = TileTensor(b_host_ptr, row_major(cb_b.alloc_size()))

        comptime if a_type.is_float8():
            rand(a_host.ptr, a_host.num_elements())
            rand(b_host.ptr, b_host.num_elements())
        else:
            if init_type == InitializationType.zero:
                _ = a_host.fill(0)
                _ = b_host.fill(0)
            elif init_type == InitializationType.one:
                _ = a_host.fill(1)
                _ = b_host.fill(1)
            elif init_type == InitializationType.uniform_distribution:
                rand(a_host.ptr, a_host.num_elements())
                rand(b_host.ptr, b_host.num_elements())
            elif init_type == InitializationType.arange:
                for i in range(a_host.num_elements()):
                    a_host.raw_store(i, Scalar[a_type](i))
                for i in range(b_host.num_elements()):
                    b_host.raw_store(i, Scalar[a_type](i))

        ctx.enqueue_copy(cb_a.device_buffer(), a_host_ptr)
        ctx.enqueue_copy(cb_b.device_buffer(), b_host_ptr)
        ctx.synchronize()
        dealloc(a_host_ptr_alloc^)
        dealloc(b_host_ptr_alloc^)
    else:
        cb_a.init_on_device(init_type, ctx)
        cb_b.init_on_device(init_type, ctx)

    # Helper to run vendor BLAS matmul - used by both benchmark and verification
    def run_vendor_blas(
        ctx: DeviceContext,
        tensor_a: TileTensor[a_type, ...],
        tensor_b: TileTensor[a_type, ...],
        tensor_c: TileTensor[mut=True, c_type, ...],
    ) raises:
        vendor_blas.matmul[use_tf32=True](
            ctx,
            tensor_c,
            tensor_a,
            tensor_b,
            c_row_major=True,
            transpose_b=transpose_b,
        )

    @__copy_capture(
        cb_a,
        cb_b,
        cb_c,
    )
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
        comptime assert tensor_c.flat_rank >= 2

        @__parameter
        @always_inline
        @__copy_capture(tensor_c)
        def test_lambda_add_coords_prod[
            _dtype: DType,
            width: Int,
            *,
            alignment: Int = align_of[SIMD[_dtype, width]](),
        ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
            _dtype, width
        ]:
            var x = tensor_c.load[width=width](Coord(idx)).cast[_dtype]()
            var y = val * x
            return y

        comptime optional_compute_lambda_fn = Optional[
            elementwise_compute_lambda_type
        ](test_lambda_add_coords_prod) if enable_compute_epilogue else None

        @always_inline
        @__parameter
        @__copy_capture(tensor_c)
        def normal_elementwise_epilogue[
            dtype: DType, width: Int, *, alignment: Int = 1
        ](idx: IndexList[2], val: SIMD[dtype, width]) capturing -> None:
            tensor_c.store[width=width]((idx[0], idx[1]), val.cast[c_type]())

        comptime optional_normal_lambda_fn = Optional[
            elementwise_epilogue_type
        ](normal_elementwise_epilogue) if enable_normal_epilogue else None

        comptime if use_vendor_blas:
            run_vendor_blas(ctx, tensor_a, tensor_b, tensor_c)
        else:
            # 4-wave-simple kernel — direct call (no dispatcher).
            # transpose_b=True is hardcoded in the kernel (FP8 layout).
            structured_4wave_matmul(tensor_a, tensor_b, tensor_c, ctx)

    @always_inline
    def bench_func(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, kernel_launch, ctx)

    var flops = ThroughputMeasure(
        BenchMetric.flops,
        # Flop: 2*M*N*K. Use A and C shapes since they're not transposed.
        2
        * Int(shape_c[0].value())
        * Int(shape_c[1].value())
        * Int(shape_a[1].value()),
    )
    if run_benchmark:
        b.bench_function(
            bench_func,
            BenchId(
                _get_run_name[
                    c_type,
                    a_type,
                    transpose_b=transpose_b,
                    cache_busting=cache_busting,
                    use_vendor_blas=use_vendor_blas,
                ](shape_c, shape_a, shape_b)
            ),
            # TODO: Pick relevant benchmetric
            [flops],
        )
    else:
        kernel_launch(ctx, 0)

    # Verification: compare our kernel output against vendor BLAS as reference.
    # The benchmark already wrote our kernel's output to buffer_c at offset 0
    # (iteration 0 uses offset 0), so we just need to run vendor BLAS once.
    comptime if not use_vendor_blas and not enable_compute_epilogue and not enable_normal_epilogue:
        if verify:
            verify_matmul[
                c_type,
                a_type,
                transpose_b=transpose_b,
                init_on_gpu=init_on_gpu,
            ](ctx, shape_c, shape_a, shape_b, init_type)


def create_matmul_bench[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    c_type: DType,
    a_type: DType,
    *,
    transpose_b: Bool,
    cache_busting: Bool,
    use_vendor_blas: Bool,
    enable_compute_epilogue: Bool,
    enable_normal_epilogue: Bool,
](
    ctx: DeviceContext,
    mut b: Bench,
    m: MType,
    n: NType,
    k: KType,
    init_type: InitializationType,
    verify: Bool,
    run_benchmark: Bool,
) raises:
    var b_shape = Coord(
        Idx[NType.static_value if transpose_b else KType.static_value],
        Idx[KType.static_value if transpose_b else NType.static_value],
    )

    bench_matmul[
        c_type,
        a_type,
        transpose_b=transpose_b,
        cache_busting=cache_busting,
        use_vendor_blas=use_vendor_blas,
        enable_compute_epilogue=enable_compute_epilogue,
        enable_normal_epilogue=enable_normal_epilogue,
    ](
        ctx,
        b,
        Coord(m, n),
        Coord(m, k),
        b_shape,
        init_type,
        verify,
        run_benchmark=run_benchmark,
    )


def main() raises:
    # 4-wave kernel is FP8-only; default the bench dtype accordingly so
    # `bazel build` (no `-D dtype=...`) compiles cleanly. Override to
    # other FP8 variants via `-D dtype=...` at the command line.
    comptime a_type = get_defined_dtype["dtype", .float8_e4m3fn]()
    comptime c_type = get_defined_dtype["ctype", .bfloat16]()

    var M = Int(arg_parse("M", 1024))
    comptime N = get_defined_int["N", 16384]()
    comptime K = get_defined_int["K", 512]()
    var init_type = InitializationType.from_str(
        arg_parse("init_type", "uniform_distribution")
    )
    var verify = arg_parse("verify", True) if not c_type.is_float8() else False
    comptime cache_busting = True
    comptime transpose_b = True
    comptime use_vendor_blas = get_defined_bool["use_vendor_blas", False]()
    comptime enable_compute_epilogue = get_defined_bool[
        "enable_compute_epilogue", False
    ]()
    comptime enable_normal_epilogue = get_defined_bool[
        "enable_normal_epilogue", False
    ]()
    var run_benchmark = arg_parse("run_benchmark", True)

    var m = Bench()
    with DeviceContext() as ctx:
        create_matmul_bench[
            c_type,
            a_type,
            transpose_b=transpose_b,
            cache_busting=cache_busting,
            use_vendor_blas=use_vendor_blas,
            enable_compute_epilogue=enable_compute_epilogue,
            enable_normal_epilogue=enable_normal_epilogue,
        ](
            ctx,
            m,
            M,
            Idx[N],
            Idx[K],
            init_type,
            verify,
            run_benchmark=run_benchmark,
        )

    if run_benchmark:
        m.dump_report()
