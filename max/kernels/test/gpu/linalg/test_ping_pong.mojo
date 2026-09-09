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
"""AMD Ping-Pong Matmul Tests.

Supports both BF16 and FP8 via compile-time flag:
  mojo -D FP8=false test_ping_pong.mojo  # BF16 (default)
  mojo -D FP8=true test_ping_pong.mojo   # FP8
"""

from std.sys import get_defined_bool, get_defined_int

from layout import Idx, Coord, TileTensor, row_major
from max.gpu.host import DeviceContext
import linalg.matmul.vendor.blas as vendor_blas
from std.testing import assert_equal
from std.random import random_float64
from linalg.matmul.gpu.amd.amd_ping_pong_matmul import (
    amd_ping_pong_matmul as ping_pong_matmul,
)

# Compile-time test configuration
comptime TEST_SIZE = get_defined_int["test_size", 4 * 1024]()
comptime TEST_RUNS = get_defined_int["test_runs", 1]()
comptime USE_FP8 = get_defined_bool["FP8", False]()
comptime input_dtype = DType.float8_e4m3fn if USE_FP8 else DType.bfloat16


def test_ping_pong_kernel_amd[
    in_dtype: DType,
    M: Int,
    N: Int,
    K: Int,
    enable_swizzle: Bool,
](ctx: DeviceContext) raises:
    """Test ping-pong kernel with parameterized input dtype."""
    var device_a = ctx.enqueue_create_buffer[in_dtype](M * K)
    var device_b = ctx.enqueue_create_buffer[in_dtype](N * K)
    var device_c = ctx.enqueue_create_buffer[.float32](M * N)
    var device_c_ref = ctx.enqueue_create_buffer[.float32](M * N)

    with device_a.map_to_host() as host_a, device_b.map_to_host() as host_b:
        # Use random floats for both BF16 and FP8 to expose precision bugs.
        # Small range [-0.5, 0.5] keeps values representable in low-precision
        # formats while ensuring non-trivial floating-point accumulation.
        for i in range(M * K):
            host_a[i] = random_float64(-0.5, 0.5).cast[in_dtype]()
        for i in range(K * N):
            host_b[i] = random_float64(-0.5, 0.5).cast[in_dtype]()

    var a_tt = TileTensor(device_a, row_major[M, K]())
    var b_tt = TileTensor(device_b, row_major[N, K]())
    var c_tt = TileTensor(device_c, row_major[M, N]())

    ctx.enqueue_memset(device_c_ref, 0)

    # Compute reference
    var c_ref_tt = TileTensor(device_c_ref, row_major[M, N]())
    vendor_blas.matmul(
        ctx,
        c_ref_tt.to_layout_tensor(),
        a_tt.to_layout_tensor(),
        b_tt.to_layout_tensor(),
        c_row_major=True,
        transpose_b=True,
    )

    for test_run in range(TEST_RUNS):
        print("Test run", test_run + 1)

        ctx.enqueue_memset(device_c, 0)

        # Run kernel under test
        ping_pong_matmul[enable_swizzle=enable_swizzle](
            a_tt,
            b_tt,
            c_tt,
            ctx,
        )

        # Validate results using relative error
        with device_c.map_to_host() as host_c, device_c_ref.map_to_host() as host_c_ref:
            var errors = 0
            var printed = 0
            var max_rel_err = Float32(0.0)
            # FP8 16x16x128 MMA has lower precision than BF16 16x16x32
            var rel_tol = Float32(
                0.05
            ) if in_dtype == .float8_e4m3fn else Float32(0.01)
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
                print("First row actual vs ref:")
                for j in range(min(16, N)):
                    print(host_c[j], host_c_ref[j])

            assert_equal(errors, 0)


def main() raises:
    with DeviceContext() as ctx:
        print("Running AMD Ping-Pong Kernel Tests")
        print("  Input dtype:", input_dtype)
        print("  Matrix M:", TEST_SIZE, " N/K:", TEST_SIZE)

        # Test without swizzle
        print("  Testing without swizzle...")
        test_ping_pong_kernel_amd[
            input_dtype,
            TEST_SIZE,
            TEST_SIZE,
            TEST_SIZE,
            enable_swizzle=False,
        ](ctx)
        print("  PASSED: No swizzle")

        # Test with swizzle
        print("  Testing with swizzle...")
        test_ping_pong_kernel_amd[
            input_dtype,
            TEST_SIZE,
            TEST_SIZE,
            TEST_SIZE,
            enable_swizzle=True,
        ](ctx)
        print("  PASSED: With swizzle")

        print("==== AMD Ping-Pong Kernel Tests passed ====")
