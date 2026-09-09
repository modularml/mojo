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
"""AMD 4-wave matmul fused-epilogue correctness test.

Drives the lambda branch of `structured_4wave_matmul`: each lane writes one
SIMD-of-`c_frag_size` to the lambda which forwards to a separate
output buffer. Compares against vendor BLAS for accuracy and against
a sentinel-filled OOB region for write safety.

Iterates dtype (FP8 / BF16 / FP16) inside `main()` — single BUILD target
per file, HK MHA-style parametrization.

Compile-time configuration:
  mojo -D M=128 -D N=512 -D K=512 test_4wave_epilogue.mojo
"""

from std.sys import align_of, get_defined_int

from layout import Coord, Idx, TileTensor, row_major
from max.gpu.host import DeviceContext
from std.utils import IndexList
import linalg.matmul.vendor.blas as vendor_blas
from std.testing import assert_equal
from std.random import random_float64
from linalg.matmul.gpu.amd.amd_4wave_matmul import structured_4wave_matmul

comptime TEST_M = get_defined_int["M", 128]()
comptime TEST_N = get_defined_int["N", 512]()
comptime TEST_K = get_defined_int["K", 512]()


def test_4wave_epilogue[
    in_dtype: DType,
    M: Int,
    N: Int,
    K: Int,
    enable_swizzle: Bool,
](ctx: DeviceContext) raises:
    """Test 4-wave lambda path against vendor BLAS at `in_dtype`."""
    comptime out_dtype = DType.float32

    var device_a = ctx.enqueue_create_buffer[in_dtype](M * K)
    var device_b = ctx.enqueue_create_buffer[in_dtype](N * K)
    # `c` is the kernel's nominal output, but the lambda routes writes
    # to `out_dev` instead — `c` stays at sentinel and proves the
    # lambda path didn't accidentally also write to `c`.
    comptime sentinel_c = Float32(-7.0)
    var device_c = ctx.enqueue_create_buffer[out_dtype](M * N)
    var device_out = ctx.enqueue_create_buffer[out_dtype](M * N)
    var device_c_ref = ctx.enqueue_create_buffer[out_dtype](M * N)

    with device_a.map_to_host() as host_a, device_b.map_to_host() as host_b:
        for i in range(M * K):
            host_a[i] = random_float64(-0.5, 0.5).cast[in_dtype]()
        for i in range(K * N):
            host_b[i] = random_float64(-0.5, 0.5).cast[in_dtype]()

    # Initialize c with sentinel (lambda path should NOT touch it).
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

    # Capturing lambda: writes the SIMD into `out_tt` at the supplied
    # global coords. 4wave hits this with `width=c_frag_size=4`.
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

    structured_4wave_matmul[
        enable_swizzle=enable_swizzle,
        elementwise_lambda_fn=epilogue_fn,
    ](a_tt, b_tt, c_tt, ctx)

    with device_out.map_to_host() as host_out, device_c_ref.map_to_host() as host_c_ref, device_c.map_to_host() as host_c:
        # --- 1. Lambda accuracy: out_tt vs vendor BLAS reference ---
        # Tolerance: PyTorch-like `|diff| <= abs_tol + rel_tol * |expected|`
        # (matches test_4wave_matmul.mojo). The cancellation-sensitive
        # cells in fp16 (very small |expected|) need an abs_tol floor;
        # `diff / max(|expected|, abs_tol)` collapses to noise there.
        var errors = 0
        var printed = 0
        var max_rel_err = Float32(0.0)
        comptime rel_tol = Float32(0.05) if in_dtype.is_float8() else (
            Float32(1.6e-2) if in_dtype == .bfloat16 else Float32(1e-3)
        )
        comptime abs_tol = Float32(0.01) if in_dtype.is_float8() else Float32(
            1e-5
        )
        for i in range(M * N):
            var actual = host_out[i]
            var expected = host_c_ref[i]
            var diff = abs(actual - expected)
            var threshold = abs_tol + rel_tol * abs(expected)
            var rel_err = diff / max(abs(expected), Float32(1e-5))
            max_rel_err = max(max_rel_err, rel_err)
            if diff > threshold:
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
                        " diff=",
                        diff,
                        " threshold=",
                        threshold,
                    )
                    printed += 1
                errors += 1
        print("  Max relative error (lambda):", max_rel_err)
        if errors != 0:
            print("  Error count:", errors, "out of", M * N)
        assert_equal(errors, 0)

        # --- 2. Lambda exclusivity: c_tt should still be sentinel ---
        # The kernel writes via the lambda, NOT into `c`. If this
        # check fires, the kernel has a "shadow" store path or my
        # `comptime if Bool(elementwise_lambda_fn)` branch is leaking
        # to the no-lambda store as well.
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


def run_dtype_sweep[dtype: DType](ctx: DeviceContext) raises:
    """Run swizzle on/off cases for one dtype."""
    print("  Shape: M=", TEST_M, " N=", TEST_N, " K=", TEST_K, sep="")
    print("  no swizzle...")
    test_4wave_epilogue[dtype, TEST_M, TEST_N, TEST_K, enable_swizzle=False](
        ctx
    )
    print("  PASSED: no swizzle")
    print("  with swizzle...")
    test_4wave_epilogue[dtype, TEST_M, TEST_N, TEST_K, enable_swizzle=True](ctx)
    print("  PASSED: with swizzle")


def main() raises:
    with DeviceContext() as ctx:
        print("Running AMD 4-wave epilogue Kernel Tests")

        # In-main dtype iteration (HK MHA pattern). One BUILD target
        # per file; three dtype specializations compile into one binary.
        print("-- dtype=float8_e4m3fn --")
        run_dtype_sweep[.float8_e4m3fn](ctx)
        print("-- dtype=bfloat16 --")
        run_dtype_sweep[.bfloat16](ctx)
        print("-- dtype=float16 --")
        run_dtype_sweep[.float16](ctx)
        print("==== AMD 4-wave epilogue tests passed ====")

        print("==== AMD 4-wave epilogue Tests passed ====")
