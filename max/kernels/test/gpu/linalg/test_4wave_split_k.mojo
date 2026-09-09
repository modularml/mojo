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
"""AMD 4-wave split-K matmul correctness tests (FP8 only).

Two cases exercised from `main()`:

  - **plain**: standard split-K (no lambda), output written to `c`.
  - **epilogue**: split-K with `elementwise_lambda_fn`. Each output
    cell is reduced from `num_splits` f32 partials, then the lambda
    fires ONCE on the final reduced value (not per partial). Verifies
    both lambda accuracy vs vendor BLAS and lambda exclusivity (`c`
    must stay at sentinel — no leak through the no-lambda code path).

Compile-time configuration:
  mojo -D M=64 -D N=4096 -D K=4096 -D NUM_SPLITS=4 test_4wave_split_k.mojo
"""

from std.sys import align_of, get_defined_int

from layout import Coord, TileTensor, row_major
from max.gpu.host import DeviceContext
from std.utils import IndexList
import linalg.matmul.vendor.blas as vendor_blas
from std.testing import assert_equal
from std.random import random_float64
from linalg.matmul.gpu.amd.amd_4wave_split_k_matmul import (
    amd_4wave_split_k_matmul,
    SplitKWorkspace,
)

comptime TEST_M = get_defined_int["M", 64]()
comptime TEST_N = get_defined_int["N", 512]()
comptime TEST_K = get_defined_int["K", 512]()
comptime TEST_NUM_SPLITS = get_defined_int["NUM_SPLITS", 2]()


def test_4wave_split_k[
    M: Int,
    N: Int,
    K: Int,
    num_splits: Int,
    enable_swizzle: Bool,
](ctx: DeviceContext) raises:
    """Test 4-wave split-K FP8 kernel against vendor BLAS (no-lambda path)."""
    comptime in_dtype = DType.float8_e4m3fn

    var device_a = ctx.enqueue_create_buffer[in_dtype](M * K)
    var device_b = ctx.enqueue_create_buffer[in_dtype](N * K)
    var device_c = ctx.enqueue_create_buffer[.float32](M * N)
    var device_c_ref = ctx.enqueue_create_buffer[.float32](M * N)

    with device_a.map_to_host() as host_a, device_b.map_to_host() as host_b:
        for i in range(M * K):
            host_a[i] = random_float64(-0.5, 0.5).cast[in_dtype]()
        for i in range(K * N):
            host_b[i] = random_float64(-0.5, 0.5).cast[in_dtype]()

    var a_tt = TileTensor(device_a, row_major[M, K]())
    var b_tt = TileTensor(device_b, row_major[N, K]())
    var c_tt = TileTensor(device_c, row_major[M, N]())

    ctx.enqueue_memset(device_c_ref, 0)
    var c_ref_tt = TileTensor(device_c_ref, row_major[M, N]())
    vendor_blas.matmul(
        ctx,
        c_ref_tt.to_layout_tensor(),
        a_tt.to_layout_tensor(),
        b_tt.to_layout_tensor(),
        c_row_major=True,
        transpose_b=True,
    )

    ctx.enqueue_memset(device_c, 0)

    var workspace = SplitKWorkspace[num_splits](ctx, M * N)
    amd_4wave_split_k_matmul[
        num_splits=num_splits, enable_swizzle=enable_swizzle
    ](a_tt, b_tt, c_tt, ctx, workspace=workspace)

    with device_c.map_to_host() as host_c, device_c_ref.map_to_host() as host_c_ref:
        var errors = 0
        var printed = 0
        var max_rel_err = Float32(0.0)
        var rel_tol = Float32(0.05)  # FP8 16x16x128 precision
        var abs_tol = Float32(1e-5)
        for i in range(M * N):
            var actual = host_c[i]
            var expected = host_c_ref[i]
            var diff = abs(actual - expected)
            var denom = max(abs(expected), abs_tol)
            var rel_err = diff / denom
            max_rel_err = max(max_rel_err, rel_err)
            if rel_err > rel_tol:
                if printed < 10:
                    var row, col = divmod(i, N)
                    print(
                        "Mismatch at (",
                        row,
                        ",",
                        col,
                        "): actual=",
                        actual,
                        " expected=",
                        expected,
                        " rel_err=",
                        rel_err,
                    )
                    printed += 1
                errors += 1

        print("  Max relative error:", max_rel_err)
        if errors != 0:
            print("  Error count:", errors, "out of", M * N)
        assert_equal(errors, 0)


def test_4wave_split_k_epilogue[
    M: Int,
    N: Int,
    K: Int,
    num_splits: Int,
    enable_swizzle: Bool,
](ctx: DeviceContext) raises:
    """Test 4-wave split-K FP8 lambda path against vendor BLAS.

    Verifies (1) lambda accuracy and (2) lambda exclusivity: the
    kernel must NOT also write to `c` via the no-lambda code path.
    """
    comptime in_dtype = DType.float8_e4m3fn
    comptime out_dtype = DType.float32

    var device_a = ctx.enqueue_create_buffer[in_dtype](M * K)
    var device_b = ctx.enqueue_create_buffer[in_dtype](N * K)
    # `c` is the kernel's nominal output; the lambda routes writes to
    # `out_tt` instead. `c` stays at sentinel and proves the lambda
    # branch did not also write to `c` via the no-lambda code path.
    comptime sentinel_c = Float32(-7.0)
    var device_c = ctx.enqueue_create_buffer[out_dtype](M * N)
    var device_out = ctx.enqueue_create_buffer[out_dtype](M * N)
    var device_c_ref = ctx.enqueue_create_buffer[out_dtype](M * N)

    with device_a.map_to_host() as host_a, device_b.map_to_host() as host_b:
        for i in range(M * K):
            host_a[i] = random_float64(-0.5, 0.5).cast[in_dtype]()
        for i in range(K * N):
            host_b[i] = random_float64(-0.5, 0.5).cast[in_dtype]()

    with device_c.map_to_host() as host_c:
        for i in range(M * N):
            host_c[i] = sentinel_c

    var a_tt = TileTensor(device_a, row_major[M, K]())
    var b_tt = TileTensor(device_b, row_major[N, K]())
    var c_tt = TileTensor(device_c, row_major[M, N]())
    # Capture the raw pointer: `@__copy_capture` byte-copies, so a
    # `DeviceBuffer`-backed tile would reach the device as a host reference.
    var out_tt = TileTensor(device_out.unsafe_ptr(), row_major[M, N]())

    ctx.enqueue_memset(device_c_ref, 0)
    var c_ref_tt = TileTensor(device_c_ref, row_major[M, N]())
    vendor_blas.matmul(
        ctx,
        c_ref_tt.to_layout_tensor(),
        a_tt.to_layout_tensor(),
        b_tt.to_layout_tensor(),
        c_row_major=True,
        transpose_b=True,
    )

    ctx.enqueue_memset(device_out, 0)

    @__parameter
    @always_inline
    @__copy_capture(out_tt)
    def epilogue_fn[
        _dtype: DType,
        width: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> None:
        var coord = Coord(idx)
        comptime assert coord.flat_rank == out_tt.flat_rank
        out_tt.store[width=width, alignment=alignment](
            coord, rebind[SIMD[out_dtype, width]](val)
        )

    var workspace = SplitKWorkspace[num_splits](ctx, M * N)
    amd_4wave_split_k_matmul[
        num_splits=num_splits,
        enable_swizzle=enable_swizzle,
        elementwise_lambda_fn=epilogue_fn,
    ](a_tt, b_tt, c_tt, ctx, workspace=workspace)

    with device_out.map_to_host() as host_out, device_c_ref.map_to_host() as host_c_ref, device_c.map_to_host() as host_c:
        # 1. Lambda accuracy: out_tt vs vendor BLAS reference
        var errors = 0
        var printed = 0
        var max_rel_err = Float32(0.0)
        var rel_tol = Float32(0.05)
        var abs_tol = Float32(1e-5)
        for i in range(M * N):
            var actual = host_out[i]
            var expected = host_c_ref[i]
            var diff = abs(actual - expected)
            var denom = max(abs(expected), abs_tol)
            var rel_err = diff / denom
            max_rel_err = max(max_rel_err, rel_err)
            if rel_err > rel_tol:
                if printed < 10:
                    var row, col = divmod(i, N)
                    print(
                        "Mismatch at (",
                        row,
                        ",",
                        col,
                        "): actual=",
                        actual,
                        " expected=",
                        expected,
                        " rel_err=",
                        rel_err,
                    )
                    printed += 1
                errors += 1
        print("  Max relative error (lambda):", max_rel_err)
        if errors != 0:
            print("  Error count:", errors, "out of", M * N)
        assert_equal(errors, 0)

        # 2. Lambda exclusivity: c_tt should still be sentinel.
        var leaks = 0
        for i in range(M * N):
            if host_c[i] != sentinel_c:
                leaks += 1
        if leaks != 0:
            print(
                "  WARNING: kernel wrote to c (",
                leaks,
                " cells != sentinel). Lambda path should leave c untouched.",
            )
        assert_equal(leaks, 0)


def main() raises:
    with DeviceContext() as ctx:
        print(
            "Running AMD 4-wave split-K Tests (FP8): M=",
            TEST_M,
            " N=",
            TEST_N,
            " K=",
            TEST_K,
            " num_splits=",
            TEST_NUM_SPLITS,
            sep="",
        )

        # ---- plain split-K (no lambda) ----
        print("  plain / no swizzle...")
        test_4wave_split_k[
            TEST_M, TEST_N, TEST_K, TEST_NUM_SPLITS, enable_swizzle=False
        ](ctx)
        print("  PASSED: plain, no swizzle")

        print("  plain / with swizzle...")
        test_4wave_split_k[
            TEST_M, TEST_N, TEST_K, TEST_NUM_SPLITS, enable_swizzle=True
        ](ctx)
        print("  PASSED: plain, with swizzle")

        # ---- split-K + fused-epilogue (lambda) ----
        print("  epilogue / no swizzle...")
        test_4wave_split_k_epilogue[
            TEST_M, TEST_N, TEST_K, TEST_NUM_SPLITS, enable_swizzle=False
        ](ctx)
        print("  PASSED: epilogue, no swizzle")

        print("  epilogue / with swizzle...")
        test_4wave_split_k_epilogue[
            TEST_M, TEST_N, TEST_K, TEST_NUM_SPLITS, enable_swizzle=True
        ](ctx)
        print("  PASSED: epilogue, with swizzle")

        print("==== ALL AMD 4-wave split-K tests passed ====")
