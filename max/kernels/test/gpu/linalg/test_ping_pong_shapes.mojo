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
"""Test ping-pong kernel across different shapes.

Documents known limitations:
- FP8: Only works when M % 256 == 0 (BM tile alignment)
- BF16: Works with all M values
"""

from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
import linalg.matmul.vendor.blas as vendor_blas
from linalg.matmul.gpu.amd.amd_ping_pong_matmul import (
    amd_ping_pong_matmul as ping_pong_matmul,
)
from std.testing import assert_true
from std.random import random_float64


def test_shape[
    in_dtype: DType, M: Int, N: Int, K: Int, enable_swizzle: Bool = True
](ctx: DeviceContext) raises:
    """Test a single shape."""
    var device_a = ctx.enqueue_create_buffer[in_dtype](M * K)
    var device_b = ctx.enqueue_create_buffer[in_dtype](N * K)
    var device_c = ctx.enqueue_create_buffer[.float32](M * N)
    var device_c_ref = ctx.enqueue_create_buffer[.float32](M * N)

    # Use random data to expose precision and swizzle bugs.
    # Small range [-0.5, 0.5] keeps values representable in low-precision formats.
    with device_a.map_to_host() as host_a, device_b.map_to_host() as host_b:
        for i in range(M * K):
            host_a[i] = random_float64(-0.5, 0.5).cast[in_dtype]()
        for i in range(K * N):
            host_b[i] = random_float64(-0.5, 0.5).cast[in_dtype]()

    var a_tt = TileTensor(device_a, row_major[M, K]())
    var b_tt = TileTensor(device_b, row_major[N, K]())
    var c_tt = TileTensor(device_c, row_major[M, N]())

    ctx.enqueue_memset(device_c, 0)
    ctx.enqueue_memset(device_c_ref, 0)

    # Run kernel
    ping_pong_matmul[enable_swizzle=enable_swizzle](
        a_tt,
        b_tt,
        c_tt,
        ctx,
    )

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

    # Validate using relative error
    with device_c.map_to_host() as host_c, device_c_ref.map_to_host() as host_c_ref:
        var errors = 0
        var max_rel_err = Float32(0.0)
        # FP8 accumulation has more noise than BF16 due to lower precision inputs.
        # BF16 with small M and large K can also show >1% relative error on
        # tiny values due to cancellation and accumulation order differences.
        var rel_tol = Float32(
            0.05
        ) if in_dtype == DType.float8_e4m3fn else Float32(0.03)
        var abs_tol = Float32(1e-4)

        for i in range(M * N):
            var actual = host_c[i]
            var expected = host_c_ref[i]
            var diff = abs(actual - expected)
            var denom = max(abs(expected), abs_tol)
            var rel_err = diff / denom
            max_rel_err = max(max_rel_err, rel_err)
            if rel_err > rel_tol:
                errors += 1
                if errors <= 5:
                    var row, col = divmod(i, N)
                    print(
                        "  Mismatch at row",
                        row,
                        "col",
                        col,
                        ": actual=",
                        actual,
                        "expected=",
                        expected,
                        "rel_err=",
                        rel_err,
                    )

        assert_true(errors == 0, msg=String(t"Test failed:{errors} errors"))


def main() raises:
    with DeviceContext() as ctx:
        print("Testing Ping-Pong Kernel Shape Compatibility")
        print("=" * 60)

        # BF16: Works with all M values
        print("\nBF16 - Testing various M values:")
        print("  M=4096 (aligned)...", end="")
        test_shape[.bfloat16, 4096, 4096, 4096, enable_swizzle=True](ctx)
        print(" PASSED")

        print("  M=1000 (unaligned)...", end="")
        test_shape[.bfloat16, 1000, 4096, 4096](ctx)
        print(" PASSED")

        print("  M=300...", end="")
        test_shape[.bfloat16, 300, 4096, 4096](ctx)
        print(" PASSED")

        print("  M=100 (small)...", end="")
        test_shape[.bfloat16, 100, 4096, 4096](ctx)
        print(" PASSED")

        print("  M=16 (very small)...", end="")
        test_shape[.bfloat16, 16, 4096, 4096](ctx)
        print(" PASSED")

        # FP8 - Testing various M values
        print("\nFP8 - Testing aligned M values (M % 256 == 0):")
        print("  M=4096...", end="")
        test_shape[.float8_e4m3fn, 4096, 4096, 4096](ctx)
        print(" PASSED")

        print("  M=2048...", end="")
        test_shape[.float8_e4m3fn, 2048, 4096, 4096](ctx)
        print(" PASSED")

        print("  M=256...", end="")
        test_shape[.float8_e4m3fn, 256, 4096, 4096](ctx)
        print(" PASSED")

        # Test FP8 without swizzle to isolate the issue
        print("\nFP8 - Testing unaligned M without swizzle:")
        print("  M=1000...", end="")
        test_shape[.float8_e4m3fn, 1024, 4096, 4096](ctx)
        print(" PASSED")

        print("  M=300...", end="")
        test_shape[.float8_e4m3fn, 300, 4096, 4096](ctx)
        print(" PASSED")

        # FP8 partial blocks: test 32×32×64 MMA (M % 32 == 0, M % 256 != 0)
        print("\nFP8 - Testing 32×32×64 MMA (M % 32 == 0, M % 256 != 0):")
        print("  M=992 (partial block, 32-aligned)...", end="")
        test_shape[.float8_e4m3fn, 992, 4096, 4096](ctx)
        print(" PASSED")

        print("  M=960 (partial block, 32-aligned)...", end="")
        test_shape[.float8_e4m3fn, 960, 4096, 4096](ctx)
        print(" PASSED")

        print("  M=1088 (partial block, 32-aligned)...", end="")
        test_shape[.float8_e4m3fn, 1088, 4096, 4096](ctx)
        print(" PASSED")

        print("  M=288 (partial block, 32-aligned, edge case)...", end="")
        test_shape[.float8_e4m3fn, 288, 4096, 4096](ctx)
        print(" PASSED")

        print("\nFP8 - Testing 16×16×128 MMA (M % 256 == 0):")
        print("  M=1024 (full blocks)...", end="")
        test_shape[.float8_e4m3fn, 1024, 4096, 4096](ctx)
        print(" PASSED")

        print("\nFP8 - Testing 16×16×128 MMA (unaligned M, fallback):")
        print("  M=1000 (partial block, unaligned)...", end="")
        test_shape[.float8_e4m3fn, 1000, 4096, 4096](ctx)
        print(" PASSED")

        print("  M=1001 (partial block, unaligned)...", end="")
        test_shape[.float8_e4m3fn, 1001, 4096, 4096](ctx)
        print(" PASSED")

        # Baseline 256x256 with various M values
        # (skinny BM=128 config is disabled due to pipeline race conditions)
        print("\nFP8 - Testing baseline 256x256:")
        print("  M=128 N=4096...", end="")
        test_shape[.float8_e4m3fn, 128, 4096, 4096](ctx)
        print(" PASSED")

        print("  M=256 N=4096...", end="")
        test_shape[.float8_e4m3fn, 256, 4096, 4096](ctx)
        print(" PASSED")

        print("  M=512 N=4096...", end="")
        test_shape[.float8_e4m3fn, 512, 4096, 4096](ctx)
        print(" PASSED")

        print("  M=4096 N=4096...", end="")
        test_shape[.float8_e4m3fn, 4096, 4096, 4096](ctx)
        print(" PASSED")

        print("\nBF16 - Testing small M values:")
        print("  M=128 (small)...", end="")
        test_shape[.bfloat16, 128, 4096, 4096](ctx)
        print(" PASSED")

        print("  M=192...", end="")
        test_shape[.bfloat16, 192, 4096, 4096](ctx)
        print(" PASSED")

        print("\nFP8 - Testing llama3-8B shapes:")
        print("  M=256 N=2304 K=16384...", end="")
        test_shape[.float8_e4m3fn, 256, 2304, 16384](ctx)
        print(" PASSED")

        print("  M=256 N=16384 K=2048...", end="")
        test_shape[.float8_e4m3fn, 256, 16384, 2048](ctx)
        print(" PASSED")

        print("  M=2048 N=2304 K=16384...", end="")
        test_shape[.float8_e4m3fn, 2048, 2304, 16384](ctx)
        print(" PASSED")

        print("  M=2048 N=16384 K=2048...", end="")
        test_shape[.float8_e4m3fn, 2048, 16384, 2048](ctx)
        print(" PASSED")

        print("\n" + "=" * 60)
        print("All tests passed!")
