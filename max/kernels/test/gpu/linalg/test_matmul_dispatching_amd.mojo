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
"""Test _matmul_gpu dispatch paths on AMD GPUs.

Exercises the full dispatch logic in __init__.mojo for FP8 and BF16,
validating against vendor BLAS with random data and relative tolerance.
Covers the same shapes as test_ping_pong_shapes.mojo but goes through
the production dispatch path instead of calling ping_pong_matmul directly.
"""

from std.sys import align_of, get_defined_bool

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.host import DeviceContext
from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from layout._fillers import random
from linalg.matmul.gpu import _matmul_gpu

from std.testing import assert_true
from std.utils.index import IndexList


def test_dispatch_dynamic_m[
    a_type: DType,
    c_type: DType,
    N: Int,
    K: Int,
](ctx: DeviceContext, m: Int) raises:
    """Test _matmul_gpu dispatch with runtime M, validated against vendor BLAS.
    """
    comptime b_type = a_type

    var a_shape = row_major(Coord(m, Idx[K]))
    var b_shape = row_major(Coord(Idx[N], Idx[K]))
    var c_shape = row_major(Coord(Int(m), Idx[N]))

    var a_size = m * K
    var b_size = N * K
    var c_size = m * N

    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_ref_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)

    var a_host = TileTensor(a_host_ptr, row_major(Coord(m, Idx[K])))
    random(a_host)

    var b_host = TileTensor(b_host_ptr, row_major[N, K]())
    random(b_host)

    var a_dev = ctx.enqueue_create_buffer[a_type](a_size)
    var b_dev = ctx.enqueue_create_buffer[b_type](b_size)
    var c_dev = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ref_dev = ctx.enqueue_create_buffer[c_type](c_size)

    ctx.enqueue_copy(a_dev, a_host_ptr)
    ctx.enqueue_copy(b_dev, b_host_ptr)
    ctx.enqueue_copy(c_dev, c_host_ptr)
    ctx.enqueue_copy(c_ref_dev, c_ref_host_ptr)

    var a_tensor = TileTensor(a_dev, a_shape)
    var b_tensor = TileTensor(b_dev, b_shape)
    var c_tensor = TileTensor(c_dev, c_shape)
    var c_ref_tensor = TileTensor(c_ref_dev, c_shape)

    _matmul_gpu[
        use_tensor_core=True,
        transpose_b=True,
    ](c_tensor, a_tensor, b_tensor, ctx)

    vendor_blas.matmul(
        ctx,
        c_ref_tensor,
        a_tensor,
        b_tensor,
        c_row_major=True,
        transpose_b=True,
    )

    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_dev)
    ctx.enqueue_copy(c_ref_host_ptr, c_ref_dev)
    ctx.synchronize()

    var rtol = Float32(0.01)
    var atol = Float32(1.0)
    var errors = 0
    var max_abs_err = Float32(0.0)
    var max_rel_err = Float32(0.0)

    for i in range(c_size):
        var actual = c_host_ptr[i].cast[.float32]()
        var expected = c_ref_host_ptr[i].cast[.float32]()
        var abs_err = abs(actual - expected)
        var ref_mag = max(abs(expected), Float32(1.0))
        var rel_err = abs_err / ref_mag
        if abs_err > max_abs_err:
            max_abs_err = abs_err
        if rel_err > max_rel_err:
            max_rel_err = rel_err
        if abs_err > atol and rel_err > rtol:
            errors += 1

    print(
        "  M=",
        m,
        " errors=",
        errors,
        "/",
        c_size,
        " max_abs=",
        max_abs_err,
        " max_rel=",
        max_rel_err,
    )

    assert_true(errors == 0, msg=String("FAILED:", errors, "errors"))

    ctx.synchronize()
    _ = a_dev^
    _ = b_dev^
    _ = c_dev^
    _ = c_ref_dev^
    ctx.synchronize()


def test_oob_diagnostic[
    a_type: DType,
    c_type: DType,
    M: Int,
    N: Int,
    K: Int,
    alloc_M: Int,
](ctx: DeviceContext) raises:
    """Detect OOB reads/writes along M by using oversized buffers with sentinels.

    Allocates buffers padded along M only (N-dimension padding is tested via
    the epilogue test since C's stride=N prevents correct N-padding detection).
    - A: [alloc_M, K] with poison in rows [M, alloc_M)
    - C: [alloc_M, N] with sentinel in rows [M, alloc_M)
    The kernel sees [M, N, K] but the allocations extend past M.
    """
    comptime assert alloc_M > M, "alloc_M must be larger than M"
    comptime assert alloc_M % 256 == 0, "alloc_M should be 256-aligned"

    comptime b_type = a_type
    comptime sentinel = Float32(-999.0)
    comptime poison = Float32(99.0)

    var a_alloc_size = alloc_M * K
    var b_size = N * K
    var c_alloc_size = alloc_M * N
    var c_valid_size = M * N

    # Host allocations (oversized for A and C)
    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_alloc_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_alloc_size)
    var c_ref_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_valid_size)

    # Initialize A: random for rows [0, M), poison for rows [M, alloc_M)
    var a_host = TileTensor(a_host_ptr, row_major[alloc_M, K]())
    random(a_host)
    for i in range(M * K, alloc_M * K):
        a_host_ptr[i] = Scalar[a_type](poison)

    # Initialize B: random
    var b_host = TileTensor(b_host_ptr, row_major[N, K]())
    random(b_host)

    # Initialize C: kernel writes the full valid region [0, M); sentinel out
    # the trailing rows [M, alloc_M) to detect OOB writes.
    for i in range(c_valid_size, c_alloc_size):
        c_host_ptr[i] = Scalar[c_type](sentinel)

    # Device allocations
    var a_dev = ctx.enqueue_create_buffer[a_type](a_alloc_size)
    var b_dev = ctx.enqueue_create_buffer[b_type](b_size)
    var c_dev = ctx.enqueue_create_buffer[c_type](c_alloc_size)
    var c_ref_dev = ctx.enqueue_create_buffer[c_type](c_valid_size)

    ctx.enqueue_copy(a_dev, a_host_ptr)
    ctx.enqueue_copy(b_dev, b_host_ptr)
    ctx.enqueue_copy(c_dev, c_host_ptr)
    ctx.enqueue_copy(c_ref_dev, c_ref_host_ptr)

    # TileTensors: kernel sees [M, N/K] but backed by larger allocation
    var a_tensor = TileTensor(a_dev, row_major(Coord(Idx[M], Idx[K])))
    var b_tensor = TileTensor(b_dev, row_major(Coord(Idx[N], Idx[K])))
    var c_tensor = TileTensor(c_dev, row_major(Coord(Idx[M], Idx[N])))
    var c_ref_tensor = TileTensor(c_ref_dev, row_major(Coord(Idx[M], Idx[N])))

    # Run kernel under test
    _matmul_gpu[
        use_tensor_core=True,
        transpose_b=True,
    ](c_tensor, a_tensor, b_tensor, ctx)

    # Reference
    vendor_blas.matmul(
        ctx,
        c_ref_tensor,
        a_tensor,
        b_tensor,
        c_row_major=True,
        transpose_b=True,
    )

    ctx.synchronize()

    # Copy back full C (including sentinel region) and reference
    ctx.enqueue_copy(c_host_ptr, c_dev)
    ctx.enqueue_copy(c_ref_host_ptr, c_ref_dev)
    ctx.synchronize()

    # --- Check 1: sentinel region (rows M..alloc_M) should be untouched ---
    var oob_writes = 0
    for i in range(c_valid_size, c_alloc_size):
        var val = c_host_ptr[i].cast[.float32]()
        if val != sentinel:
            oob_writes += 1
            if oob_writes <= 5:
                var row = i // N
                var col = i % N
                print(
                    "  OOB WRITE [",
                    row,
                    ",",
                    col,
                    "]: expected sentinel",
                    sentinel,
                    "got",
                    val,
                )
    if oob_writes > 5:
        print("  ... and", oob_writes - 5, "more OOB writes")

    # --- Check 2: valid region (rows 0..M) should match reference ---
    var rtol = Float32(0.01)
    var atol = Float32(1.0)
    var accuracy_errors = 0
    var max_abs_err = Float32(0.0)
    var max_rel_err = Float32(0.0)

    for i in range(c_valid_size):
        var actual = c_host_ptr[i].cast[.float32]()
        var expected = c_ref_host_ptr[i].cast[.float32]()
        var abs_err = abs(actual - expected)
        var ref_mag = max(abs(expected), Float32(1.0))
        var rel_err = abs_err / ref_mag

        if abs_err > max_abs_err:
            max_abs_err = abs_err
        if rel_err > max_rel_err:
            max_rel_err = rel_err

        if abs_err > atol and rel_err > rtol:
            accuracy_errors += 1
            if accuracy_errors <= 5:
                var row = i // N
                var col = i % N
                print(
                    "  ACCURACY [",
                    row,
                    ",",
                    col,
                    "]: got",
                    actual,
                    "expected",
                    expected,
                    "abs=",
                    abs_err,
                    "rel=",
                    rel_err,
                )
    if accuracy_errors > 5:
        print("  ... and", accuracy_errors - 5, "more accuracy errors")

    print(
        "\n    oob_writes=",
        oob_writes,
        "/",
        c_alloc_size - c_valid_size,
        " accuracy_errors=",
        accuracy_errors,
        "/",
        c_valid_size,
        " max_abs=",
        max_abs_err,
        " max_rel=",
        max_rel_err,
    )

    assert_true(
        oob_writes == 0,
        msg=String("OOB WRITES DETECTED:", oob_writes, "elements corrupted"),
    )
    assert_true(
        accuracy_errors == 0,
        msg=String(
            "ACCURACY FAILED:", accuracy_errors, "errors (poison read?)"
        ),
    )

    ctx.synchronize()
    _ = a_dev^
    _ = b_dev^
    _ = c_dev^
    _ = c_ref_dev^
    ctx.synchronize()


def test_oob_epilogue[
    a_type: DType,
    c_type: DType,
    M: Int,
    N: Int,
    K: Int,
    alloc_M: Int,
    alloc_N: Int = N,
](ctx: DeviceContext) raises:
    """Detect OOB writes through the epilogue lambda path along M and/or N.

    The epilogue writes to a separate output buffer using global (m, n)
    coordinates. The output buffer is [alloc_M, alloc_N] filled with sentinel,
    so any write outside the valid [M, N] region is detectable regardless of
    the kernel's C tensor stride.
    """
    comptime assert alloc_M >= M, "alloc_M must be >= M"
    comptime assert alloc_N >= N, "alloc_N must be >= N"
    comptime assert (
        alloc_M > M or alloc_N > N
    ), "at least one dimension must be padded"

    comptime b_type = a_type
    comptime sentinel = Float32(-999.0)
    comptime poison = Float32(99.0)

    var a_alloc_size = alloc_M * K
    var b_alloc_size = alloc_N * K
    var c_size = M * N
    var out_alloc_size = alloc_M * alloc_N

    # Host allocations
    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_alloc_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](b_alloc_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    # Output: sentinel everywhere — epilogue should only write [0..M, 0..N].
    var out_host_ptr = ctx.enqueue_create_host_buffer[c_type](out_alloc_size)
    for i in range(out_alloc_size):
        out_host_ptr[i] = Scalar[c_type](sentinel)
    var c_ref_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)

    # A: random for [0, M), poison for [M, alloc_M)
    var a_host = TileTensor(a_host_ptr, row_major[alloc_M, K]())
    random(a_host)

    for i in range(M * K, alloc_M * K):
        a_host_ptr[i] = Scalar[a_type](poison)

    # B: random for [0, N), poison for [N, alloc_N) (transpose_b layout)
    var b_host = TileTensor(b_host_ptr, row_major[alloc_N, K]())

    random(b_host)

    for i in range(N * K, alloc_N * K):
        b_host_ptr[i] = Scalar[b_type](poison)

    # Device allocations
    var a_dev = ctx.enqueue_create_buffer[a_type](a_alloc_size)
    var b_dev = ctx.enqueue_create_buffer[b_type](b_alloc_size)
    var c_dev = ctx.enqueue_create_buffer[c_type](c_size)
    var out_dev = ctx.enqueue_create_buffer[c_type](out_alloc_size)
    var c_ref_dev = ctx.enqueue_create_buffer[c_type](c_size)

    ctx.enqueue_copy(a_dev, a_host_ptr)
    ctx.enqueue_copy(b_dev, b_host_ptr)
    ctx.enqueue_copy(c_dev, c_host_ptr)
    ctx.enqueue_copy(out_dev, out_host_ptr)
    ctx.enqueue_copy(c_ref_dev, c_ref_host_ptr)

    # TileTensors: kernel sees [M, N, K]
    var a_tensor = TileTensor(a_dev, row_major(Coord(Idx[M], Idx[K])))
    var b_tensor = TileTensor(b_dev, row_major(Coord(Idx[N], Idx[K])))
    var c_tensor = TileTensor(c_dev, row_major(Coord(Idx[M], Idx[N])))
    var c_ref_tensor = TileTensor(c_ref_dev, row_major(Coord(Idx[M], Idx[N])))

    # Output buffer: [alloc_M, alloc_N] so OOB writes in both dims are visible
    var out_tensor = TileTensor(
        out_dev.unsafe_ptr(),
        row_major(Coord(Idx[alloc_M], Idx[alloc_N])),
    )

    # Epilogue writes to out_tensor using global (m, n) coordinates
    @__parameter
    @always_inline
    @__copy_capture(out_tensor)
    def epilogue_fn[
        _dtype: DType,
        width: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> None:
        var coord = Coord(idx)
        comptime assert coord.flat_rank == out_tensor.flat_rank
        out_tensor.store[width=width, alignment=alignment](
            coord, rebind[SIMD[c_type, width]](val)
        )

    _matmul_gpu[
        use_tensor_core=True,
        transpose_b=True,
        elementwise_lambda_fn=epilogue_fn,
    ](c_tensor, a_tensor, b_tensor, ctx)

    vendor_blas.matmul(
        ctx,
        c_ref_tensor,
        a_tensor,
        b_tensor,
        c_row_major=True,
        transpose_b=True,
    )

    ctx.synchronize()

    ctx.enqueue_copy(out_host_ptr, out_dev)
    ctx.enqueue_copy(c_ref_host_ptr, c_ref_dev)
    ctx.synchronize()

    # --- Check 1: sentinel region [alloc_M, alloc_N] outside [M, N] ---
    var oob_writes = 0
    for row in range(alloc_M):
        for col in range(alloc_N):
            if row < M and col < N:
                continue
            var val = out_host_ptr[row * alloc_N + col].cast[.float32]()
            if val != sentinel:
                oob_writes += 1
                if oob_writes <= 5:
                    print(
                        "  OOB WRITE [",
                        row,
                        ",",
                        col,
                        "]: expected sentinel",
                        sentinel,
                        "got",
                        val,
                    )
    if oob_writes > 5:
        print("  ... and", oob_writes - 5, "more OOB writes")

    # --- Check 2: valid region [0..M, 0..N] matches reference ---
    var rtol = Float32(0.01)
    var atol = Float32(1.0)
    var accuracy_errors = 0
    var max_abs_err = Float32(0.0)
    var max_rel_err = Float32(0.0)

    for row in range(M):
        for col in range(N):
            var actual = out_host_ptr[row * alloc_N + col].cast[.float32]()
            var expected = c_ref_host_ptr[row * N + col].cast[.float32]()
            var abs_err = abs(actual - expected)
            var ref_mag = max(abs(expected), Float32(1.0))
            var rel_err = abs_err / ref_mag

            if abs_err > max_abs_err:
                max_abs_err = abs_err
            if rel_err > max_rel_err:
                max_rel_err = rel_err

            if abs_err > atol and rel_err > rtol:
                accuracy_errors += 1
                if accuracy_errors <= 5:
                    print(
                        "  ACCURACY [",
                        row,
                        ",",
                        col,
                        "]: got",
                        actual,
                        "expected",
                        expected,
                        "abs=",
                        abs_err,
                        "rel=",
                        rel_err,
                    )
    if accuracy_errors > 5:
        print("  ... and", accuracy_errors - 5, "more accuracy errors")

    var sentinel_count = alloc_M * alloc_N - M * N
    print(
        "\n    oob_writes=",
        oob_writes,
        "/",
        sentinel_count,
        " accuracy_errors=",
        accuracy_errors,
        "/",
        c_size,
        " max_abs=",
        max_abs_err,
        " max_rel=",
        max_rel_err,
    )

    assert_true(
        oob_writes == 0,
        msg=String("EPILOGUE OOB WRITES:", oob_writes, "elements corrupted"),
    )
    assert_true(
        accuracy_errors == 0,
        msg=String("EPILOGUE ACCURACY FAILED:", accuracy_errors, "errors"),
    )

    ctx.synchronize()
    _ = a_dev^
    _ = b_dev^
    _ = c_dev^
    _ = out_dev^
    _ = c_ref_dev^
    ctx.synchronize()


def test_oob_epilogue_dynamic_m[
    a_type: DType,
    c_type: DType,
    N: Int,
    K: Int,
](ctx: DeviceContext, m: Int) raises:
    """OOB epilogue test with runtime M dimension.

    N and K are comptime (required by the kernel), M is runtime.
    Allocates with padding above M for OOB detection.
    """
    comptime b_type = a_type
    comptime sentinel = Float32(-999.0)
    comptime poison = Float32(99.0)

    # Pad M up to next multiple of 256 + 256 buffer
    var alloc_m = ((m + 255) // 256 + 1) * 256

    var a_alloc_size = alloc_m * K
    var b_size = N * K
    var c_size = m * N
    var out_alloc_size = alloc_m * N

    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_alloc_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    # Output: sentinel everywhere — epilogue should only write [0..M, 0..N].
    var out_host_ptr = ctx.enqueue_create_host_buffer[c_type](out_alloc_size)
    for i in range(out_alloc_size):
        out_host_ptr[i] = Scalar[c_type](sentinel)
    var c_ref_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)

    # A: random for [0, M), poison for [M, alloc_m)
    var a_host = TileTensor(a_host_ptr, row_major(Coord(alloc_m, Idx[K])))
    random(a_host)

    for i in range(m * K, alloc_m * K):
        a_host_ptr[i] = Scalar[a_type](poison)

    # B: random
    var b_host = TileTensor(b_host_ptr, row_major[N, K]())
    random(b_host)

    var a_dev = ctx.enqueue_create_buffer[a_type](a_alloc_size)
    var b_dev = ctx.enqueue_create_buffer[b_type](b_size)
    var c_dev = ctx.enqueue_create_buffer[c_type](c_size)
    var out_dev = ctx.enqueue_create_buffer[c_type](out_alloc_size)
    var c_ref_dev = ctx.enqueue_create_buffer[c_type](c_size)

    ctx.enqueue_copy(a_dev, a_host_ptr)
    ctx.enqueue_copy(b_dev, b_host_ptr)
    ctx.enqueue_copy(c_dev, c_host_ptr)
    ctx.enqueue_copy(out_dev, out_host_ptr)
    ctx.enqueue_copy(c_ref_dev, c_ref_host_ptr)

    # Dynamic M: TileTensors with runtime M dimension
    var a_tensor = TileTensor(a_dev, row_major(Coord(Int(m), Idx[K])))
    var b_tensor = TileTensor(b_dev, row_major(Coord(Idx[N], Idx[K])))
    var c_tensor = TileTensor(c_dev, row_major(Coord(Int(m), Idx[N])))
    var c_ref_tensor = TileTensor(c_ref_dev, row_major(Coord(Int(m), Idx[N])))

    var out_tensor = TileTensor(
        out_dev.unsafe_ptr(), row_major(Coord(Int(alloc_m), Idx[N]))
    )

    @__parameter
    @always_inline
    @__copy_capture(out_tensor)
    def epilogue_fn[
        _dtype: DType,
        width: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> None:
        var coord = Coord(idx)
        comptime assert coord.flat_rank == out_tensor.flat_rank
        out_tensor.store[width=width, alignment=alignment](
            coord, rebind[SIMD[c_type, width]](val)
        )

    _matmul_gpu[
        use_tensor_core=True,
        transpose_b=True,
        elementwise_lambda_fn=epilogue_fn,
    ](c_tensor, a_tensor, b_tensor, ctx)

    vendor_blas.matmul(
        ctx,
        c_ref_tensor,
        a_tensor,
        b_tensor,
        c_row_major=True,
        transpose_b=True,
    )

    ctx.synchronize()

    ctx.enqueue_copy(out_host_ptr, out_dev)
    ctx.enqueue_copy(c_ref_host_ptr, c_ref_dev)
    ctx.synchronize()

    # Check OOB writes
    var oob_writes = 0
    for i in range(c_size, out_alloc_size):
        var val = out_host_ptr[i].cast[.float32]()
        if val != sentinel:
            oob_writes += 1
            if oob_writes <= 3:
                print(
                    "  OOB [",
                    i // N,
                    ",",
                    i % N,
                    "]: sentinel",
                    sentinel,
                    "got",
                    val,
                )

    # Check accuracy
    var rtol = Float32(0.01)
    var atol = Float32(1.0)
    var accuracy_errors = 0
    var max_abs_err = Float32(0.0)
    var max_rel_err = Float32(0.0)

    for i in range(c_size):
        var actual = out_host_ptr[i].cast[.float32]()
        var expected = c_ref_host_ptr[i].cast[.float32]()
        var abs_err = abs(actual - expected)
        var ref_mag = max(abs(expected), Float32(1.0))
        var rel_err = abs_err / ref_mag
        if abs_err > max_abs_err:
            max_abs_err = abs_err
        if rel_err > max_rel_err:
            max_rel_err = rel_err
        if abs_err > atol and rel_err > rtol:
            accuracy_errors += 1

    print(
        "  M=",
        m,
        " oob=",
        oob_writes,
        " acc_err=",
        accuracy_errors,
        "/",
        c_size,
        " max_abs=",
        max_abs_err,
        " max_rel=",
        max_rel_err,
    )

    assert_true(oob_writes == 0, msg=String("OOB:", oob_writes))
    assert_true(accuracy_errors == 0, msg=String("ACC:", accuracy_errors))

    ctx.synchronize()
    _ = a_dev^
    _ = b_dev^
    _ = c_dev^
    _ = out_dev^
    _ = c_ref_dev^
    ctx.synchronize()


def main() raises:
    with DeviceContext() as ctx:
        print("Testing _matmul_gpu AMD dispatch paths")
        print("=" * 60)

        # ============================================================
        # BF16 tests (N=4096, K=4096)
        # ============================================================
        print("\nBF16 - Various M values (N=4096, K=4096):")
        var bf16_m: List[Int] = [4096, 1000, 300, 100, 16, 128, 192]
        for i in range(len(bf16_m)):
            test_dispatch_dynamic_m[.bfloat16, .bfloat16, 4096, 4096](
                ctx, bf16_m[i]
            )

        # ============================================================
        # FP8 M=256, N=256, Test K % BK != 0
        # ============================================================

        print("\nFP8 M=256, N=256, Test K % BK != 0")
        print("  K = 128,", end=" ")
        test_dispatch_dynamic_m[.float8_e4m3fn, .float32, 256, 128](ctx, 256)

        print("  K = 192,", end=" ")
        test_dispatch_dynamic_m[.float8_e4m3fn, .float32, 256, 192](ctx, 256)

        print("  K = 256,", end=" ")
        test_dispatch_dynamic_m[.float8_e4m3fn, .float32, 256, 256](ctx, 256)

        print("  K = 320,", end=" ")
        test_dispatch_dynamic_m[.float8_e4m3fn, .float32, 256, 320](ctx, 256)

        print("  K = 384,", end=" ")
        test_dispatch_dynamic_m[.float8_e4m3fn, .float32, 256, 384](ctx, 256)

        # ============================================================
        # FP8 N=4096 K=4096 (covers standard GEMM, pingpong, skinny)
        # ============================================================
        print("\nFP8 - Various M values (N=4096, K=4096):")
        var fp8_4096_m: List[Int] = [
            1,
            16,
            32,
            64,
            75,
            100,
            127,
            128,
            200,
            256,
            288,
            300,
            512,
            960,
            992,
            1000,
            1001,
            1024,
            1088,
            2048,
            4096,
        ]
        for i in range(len(fp8_4096_m)):
            test_dispatch_dynamic_m[.float8_e4m3fn, .float32, 4096, 4096](
                ctx, fp8_4096_m[i]
            )

        # ============================================================
        # FP8 N=16384 K=2048 (output proj shape)
        # ============================================================
        print("\nFP8 - Various M values (N=16384, K=2048):")
        var fp8_16384_m: List[Int] = [300, 750, 8192]
        for i in range(len(fp8_16384_m)):
            test_dispatch_dynamic_m[.float8_e4m3fn, .float32, 16384, 2048](
                ctx, fp8_16384_m[i]
            )

        # ============================================================
        # FP8 N=2304 K=16384 (fused QKV shape)
        # ============================================================
        print("\nFP8 - Various M values (N=2304, K=16384):")
        var fp8_2304_m: List[Int] = [16, 75, 300, 600, 1024]
        for i in range(len(fp8_2304_m)):
            test_dispatch_dynamic_m[.float8_e4m3fn, .float32, 2304, 16384](
                ctx, fp8_2304_m[i]
            )

        # ============================================================
        # FP8 N=2048 K=2048 (small square)
        # ============================================================
        print("\nFP8 - Various M values (N=2048, K=2048):")
        test_dispatch_dynamic_m[.float8_e4m3fn, .float32, 2048, 2048](ctx, 2048)

        # ============================================================
        # OOB diagnostic: detect reads/writes past M
        # ============================================================
        print("\nFP8 - OOB diagnostic (padded buffers):")

        print("  M=300 alloc=1024 N=4096 K=16384...", end="")
        test_oob_diagnostic[
            .float8_e4m3fn,
            .float32,
            300,
            4096,
            16384,
            alloc_M=1024,
        ](ctx)
        print(" PASSED")

        print("  M=500 alloc=1024 N=4096 K=16384...", end="")
        test_oob_diagnostic[
            .float8_e4m3fn,
            .float32,
            500,
            16384,
            16384,
            alloc_M=1024,
        ](ctx)
        print(" PASSED")

        print("  M=100 alloc=512 N=4096 K=16384...", end="")
        test_oob_diagnostic[
            .float8_e4m3fn,
            .float32,
            100,
            4096,
            16384,
            alloc_M=512,
        ](ctx)
        print(" PASSED")

        print("  M=256 alloc=512 N=4096 K=16384 (aligned, control)...", end="")
        test_oob_diagnostic[
            .float8_e4m3fn,
            .float32,
            256,
            4096,
            16384,
            alloc_M=512,
        ](ctx)
        print(" PASSED")

        # ============================================================
        # OOB diagnostic with epilogue lambda (production path)
        # ============================================================
        print("\nFP8 - OOB epilogue diagnostic (production write path):")

        print("  M=256 alloc=512 N=256 K=256...", end="")
        test_oob_epilogue[
            .float8_e4m3fn,
            .float32,
            256,
            256,
            256,
            alloc_M=512,
        ](ctx)
        print(" PASSED")

        print("  M=256 alloc=512 N=256 K=16384...", end="")
        test_oob_epilogue[
            .float8_e4m3fn,
            .float32,
            256,
            256,
            16384,
            alloc_M=512,
        ](ctx)
        print(" PASSED")

        print("  M=300 alloc=512 N=256 K=16384...", end="")
        test_oob_epilogue[
            .float8_e4m3fn,
            .float32,
            300,
            256,
            16384,
            alloc_M=512,
        ](ctx)
        print(" PASSED")

        print("  M=300 alloc=1024 N=4096 K=16384...", end="")
        test_oob_epilogue[
            .float8_e4m3fn,
            .float32,
            300,
            4096,
            16384,
            alloc_M=1024,
        ](ctx)
        print(" PASSED")

        print("  M=500 alloc=1024 N=4096 K=16384...", end="")
        test_oob_epilogue[
            .float8_e4m3fn,
            .float32,
            500,
            4096,
            16384,
            alloc_M=1024,
        ](ctx)
        print(" PASSED")

        print("  M=100 alloc=512 N=4096 K=16384...", end="")
        test_oob_epilogue[
            .float8_e4m3fn,
            .float32,
            100,
            4096,
            16384,
            alloc_M=512,
        ](ctx)
        print(" PASSED")

        print("  M=256 alloc=512 N=4096 K=16384 (aligned, control)...", end="")
        test_oob_epilogue[
            .float8_e4m3fn,
            .float32,
            256,
            4096,
            16384,
            alloc_M=512,
        ](ctx)
        print(" PASSED")

        # ============================================================
        # OOB epilogue: N-dimension padding (sharded B)
        # ============================================================
        print("\nFP8 - OOB epilogue (N-dimension padding):")

        print("  M=256 N=3000 allocN=4096 K=16384...", end="")
        test_oob_epilogue[
            .float8_e4m3fn,
            .float32,
            256,
            3000,
            16384,
            alloc_M=256,
            alloc_N=4096,
        ](ctx)
        print(" PASSED")

        print("  M=256 N=2304 allocN=4096 K=16384 (N aligned)...", end="")
        test_oob_epilogue[
            .float8_e4m3fn,
            .float32,
            256,
            2304,
            16384,
            alloc_M=256,
            alloc_N=4096,
        ](ctx)
        print(" PASSED")

        print("  M=256 N=500 allocN=1024 K=16384...", end="")
        test_oob_epilogue[
            .float8_e4m3fn,
            .float32,
            256,
            500,
            16384,
            alloc_M=256,
            alloc_N=1024,
        ](ctx)
        print(" PASSED")

        print(
            "  M=300 N=3000 allocM=1024 allocN=4096 K=16384 (both padded)...",
            end="",
        )
        test_oob_epilogue[
            .float8_e4m3fn,
            .float32,
            300,
            3000,
            16384,
            alloc_M=1024,
            alloc_N=4096,
        ](ctx)
        print(" PASSED")

        comptime run_llama3_sizes = get_defined_bool[
            "RUN_LLAMA3_SIZES", False
        ]()
        comptime if run_llama3_sizes:
            # ============================================================
            # Llama3-405B TP=4: all observed M values × all layer shapes
            # ============================================================
            # TP=4 weight shapes (transpose_b, so N = output dim):
            #   Fused QKV:   N=4608,  K=16384
            #   Output proj: N=16384, K=4096
            #   Gate/Up:     N=13312, K=16384
            #   Down proj:   N=16384, K=13312

            var runtime_m_values: List[Int] = [
                140,
                154,
                158,
                159,
                162,
                165,
                166,
                174,
                175,
                179,
                187,
                195,
                199,
                204,
                211,
                215,
                217,
                220,
                222,
                225,
                229,
                232,
                235,
                237,
                238,
                241,
                243,
                246,
                257,
                258,
                261,
                272,
                273,
                277,
                282,
                2702,
                8192,
            ]

            # --- Fused QKV: N=4608, K=16384 ---
            print("\nFP8 - Llama3-405B TP=4: Fused QKV (N=4608, K=16384):")
            for i in range(len(runtime_m_values)):
                test_oob_epilogue_dynamic_m[
                    .float8_e4m3fn, .float32, 4608, 16384
                ](ctx, runtime_m_values[i])
            print(" PASSED")

            # --- Output proj: N=16384, K=4096 ---
            print("\nFP8 - Llama3-405B TP=4: Output proj (N=16384, K=4096):")
            for i in range(len(runtime_m_values)):
                test_oob_epilogue_dynamic_m[
                    .float8_e4m3fn, .float32, 16384, 4096
                ](ctx, runtime_m_values[i])
            print(" PASSED")

            # --- Gate/Up proj: N=13312, K=16384 ---
            print("\nFP8 - Llama3-405B TP=4: Gate/Up proj (N=13312, K=16384):")
            for i in range(len(runtime_m_values)):
                test_oob_epilogue_dynamic_m[
                    .float8_e4m3fn, .float32, 13312, 16384
                ](ctx, runtime_m_values[i])
            print(" PASSED")

            # --- Down proj: N=16384, K=13312 ---
            print("\nFP8 - Llama3-405B TP=4: Down proj (N=16384, K=13312):")
            for i in range(len(runtime_m_values)):
                test_oob_epilogue_dynamic_m[
                    .float8_e4m3fn, .float32, 16384, 13312
                ](ctx, runtime_m_values[i])
            print(" PASSED")

        print("\n" + "=" * 60)
        print("All tests passed!")
