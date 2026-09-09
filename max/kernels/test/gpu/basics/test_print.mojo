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

from std.reflection import source_location
from max.gpu.host import DeviceContext


# CHECK-LABEL: == test_gpu_print_formattable
def test_gpu_print_formattable() raises:
    print("== test_gpu_print_formattable")

    def do_print(x_dev: Int32, y: Float64):
        var x = Int(x_dev)
        # ==============================
        # Test printing primitive types
        # ==============================

        # CHECK: Hello I got 42 7.2
        print("Hello I got", x, y)

        # CHECK: Printing Int zero  = 0
        print("Printing Int zero  =", Int(0))

        #
        # Test printing SIMD values
        #

        var simd = SIMD[.float64, 4](0.0, -1.0, Float64.MIN, Float64.MAX_FINITE)
        # CHECK: SIMD values are: [0.0, -1.0, -inf, 1.7976931348623157e+308]
        print("SIMD values are:", simd)

        # CHECK: test_print.mojo:43:30
        print(source_location())

        # ------------------------------
        # Test printing bfloat16
        # ------------------------------

        # Note:
        #   The difference in content for the bfloat16 values tested below is
        #   expected due to precision loss inherent in shrinking down to
        #   a 16 bit type.

        def print_casts(value: Float32):
            var a = value.cast[.float64]()
            var b = value.cast[.bfloat16]()
            var c = value.cast[.bfloat16]().cast[.float64]()

            print("  original fp32:", value)
            print("        => fp64:", a)
            print("        => bf16:", b)  # Test formatting bfloat16 directly.
            print("=> bf16 => fp64:", c)  # Test formatting bfloat16 indirectly.

        # CHECK-LABEL: === value_a ===
        # CHECK:   original fp32: 0.502364
        # CHECK:         => fp64: 0.5023639798164368
        # CHECK:         => bf16: 0.50390625
        # CHECK: => bf16 => fp64: 0.50390625
        print("=== value_a ===")
        print_casts(Float32(0.502364))

        # CHECK-LABEL: === value_b ===
        # CHECK:   original fp32: 0.501858
        # CHECK:         => fp64: 0.5018579959869385
        # CHECK:         => bf16: 0.5
        # CHECK: => bf16 => fp64: 0.5
        print("=== value_b ===")
        print_casts(Float32(0.501858))

    with DeviceContext() as ctx:
        comptime kernel = do_print
        ctx.enqueue_function[kernel](
            Int32(42), Float64(7.2), grid_dim=1, block_dim=1
        )
        # Ensure queued function finished before proceeding.
        ctx.synchronize()


def main() raises:
    test_gpu_print_formattable()
