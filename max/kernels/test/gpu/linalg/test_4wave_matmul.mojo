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
"""AMD 4-wave matmul correctness test vs vendor BLAS.

Drives `structured_4wave_matmul` (the AMD MI355X 4-wave matmul
launcher) against `linalg.matmul.vendor.blas` on the same shape and
checks element-wise tolerance. Runs all three supported dtypes (FP8,
BF16, FP16) from `main()` — single BUILD target per file, HK MHA-style
parametrization. Optional shape overrides via `-D M=`, `-D N=`, `-D K=`;
defaults to M=128 N=512 K=512.

  mojo -D M=128 -D N=4096 -D K=4096 test_4wave_matmul.mojo
"""

from std.sys import get_defined_int, get_defined_string

from layout import Idx, Coord, TileTensor, row_major
from max.gpu.host import DeviceContext
import linalg.matmul.vendor.blas as vendor_blas
from std.testing import assert_equal
from std.random import random_float64, seed
from linalg.matmul.gpu.amd.amd_4wave_matmul import structured_4wave_matmul

comptime TEST_M = get_defined_int["M", 128]()
comptime TEST_N = get_defined_int["N", 512]()
comptime TEST_K = get_defined_int["K", 512]()
comptime DUMP_GPU_ASM = get_defined_string["DUMP_GPU_ASM", ""]()
# Number of times to repeat each shape. Use `-D REPEAT=100` to catch
# sporadic failures. Each repetition reseeds the RNG with a different
# value, so we exercise distinct input patterns each pass.
comptime REPEAT = get_defined_int["REPEAT", 1]()


def test_4wave_matmul[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
    enable_swizzle: Bool,
](ctx: DeviceContext, seed_value: Int = 20260513) raises:
    """Test 4-wave matmul kernel against vendor BLAS at `dtype`.

    `seed_value` controls the PRNG seed used to generate inputs. The
    REPEAT loop in `main` advances this per repetition so each pass
    exercises distinct input patterns; CI failures are reproducible
    by setting -D REPEAT=N and reading the failing seed from output.
    """
    seed(seed_value)
    var device_a = ctx.enqueue_create_buffer[dtype](M * K)
    var device_b = ctx.enqueue_create_buffer[dtype](N * K)
    var device_c = ctx.enqueue_create_buffer[.float32](M * N)
    var device_c_ref = ctx.enqueue_create_buffer[.float32](M * N)

    with device_a.map_to_host() as host_a, device_b.map_to_host() as host_b:
        for i in range(M * K):
            host_a[i] = random_float64(-0.5, 0.5).cast[dtype]()
        for i in range(K * N):
            host_b[i] = random_float64(-0.5, 0.5).cast[dtype]()

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

    structured_4wave_matmul[
        enable_swizzle=enable_swizzle, dump_asm_path=DUMP_GPU_ASM
    ](
        a_tt,
        b_tt,
        c_tt,
        ctx,
    )

    with device_c.map_to_host() as host_c, device_c_ref.map_to_host() as host_c_ref:
        var errors = 0
        var printed = 0
        var max_rel_err = Float32(0.0)
        # Tolerance matches `bench_matmul.mojo`'s `--verify` path
        # (`pytorch_like_tolerances_for` in `internal_utils/_testing.mojo`),
        # using the standard `|diff| <= abs_tol + rel_tol * |expected|`
        # formula. The previous formula
        # `diff / max(|expected|, abs_tol) <= rel_tol` collapsed to
        # `diff <= rel_tol * abs_tol` (5e-8 for fp16) at near-zero
        # cells — well below fp16's achievable precision for
        # cancelling sums, which caused sporadic CI failures.
        comptime rel_tol = Float32(0.01) if dtype.is_float8() else (
            Float32(1.6e-2) if dtype == .bfloat16 else Float32(1e-3)
        )
        comptime abs_tol = Float32(0.01) if dtype.is_float8() else Float32(1e-5)
        for i in range(M * N):
            var actual = host_c[i]
            var expected = host_c_ref[i]
            var diff = abs(actual - expected)
            var threshold = abs_tol + rel_tol * abs(expected)
            # Track rel_err using a non-degenerate denom for reporting
            # only — the pass/fail decision is `diff <= threshold`.
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

        print("  Max relative error:", max_rel_err)
        if errors != 0:
            print("  Error count:", errors, "out of", M * N)
        assert_equal(errors, 0)


def run_dtype_sweep[dtype: DType](ctx: DeviceContext, seed_value: Int) raises:
    """Run all shape cases for one dtype at the given seed."""

    # Parametric smoke shape (overridable via -D M= -D N= -D K=).
    print("  Shape: M=", TEST_M, " N=", TEST_N, " K=", TEST_K, sep="")
    test_4wave_matmul[dtype, TEST_M, TEST_N, TEST_K, enable_swizzle=False](
        ctx, seed_value
    )
    test_4wave_matmul[dtype, TEST_M, TEST_N, TEST_K, enable_swizzle=True](
        ctx, seed_value
    )
    print("  PASSED")

    # Multi-block: M=1024 forces multiple BM=128 row blocks.
    print("  Shape: M=1024 N=2048 K=2048")
    test_4wave_matmul[dtype, 1024, 2048, 2048, enable_swizzle=True](
        ctx, seed_value
    )
    print("  PASSED")

    # Off-square (MHA/MLA prefill-style): tall-skinny M with small N.
    # K must be a multiple of 2*BK = 256.
    print("  Shape: M=4096 N=128 K=256")
    test_4wave_matmul[dtype, 4096, 128, 256, enable_swizzle=True](
        ctx, seed_value
    )
    print("  PASSED")


def main() raises:
    with DeviceContext() as ctx:
        print("Running AMD 4-wave Kernel Tests (REPEAT=", REPEAT, ")", sep="")

        for rep in range(REPEAT):
            # Different seed each repetition so distinct input patterns
            # exercise the kernel. Print on every iter so a CI failure
            # report tells us which seed reproduced it.
            var seed_value = 20260513 + rep
            if REPEAT > 1:
                print(
                    "--- rep ",
                    rep,
                    "/",
                    REPEAT,
                    " (seed=",
                    seed_value,
                    ") ---",
                    sep="",
                )

            # In-main dtype iteration (HK MHA pattern). Three separate
            # specializations compile into one binary; one BUILD target
            # runs all three.
            print("-- dtype=float8_e4m3fn --")
            run_dtype_sweep[.float8_e4m3fn](ctx, seed_value)
            print("-- dtype=bfloat16 --")
            run_dtype_sweep[.bfloat16](ctx, seed_value)
            print("-- dtype=float16 --")
            run_dtype_sweep[.float16](ctx, seed_value)

        print("==== AMD 4-wave tests passed ====")
