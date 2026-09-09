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

"""Test FP8 E4M3FN to E4M3FNUZ conversion kernel."""

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from linalg.fp8_quantization import convert_e4m3fn_to_e4m3fnuz
from std.testing import assert_equal


# CHECK-LABEL: test_convert_e4m3fn_to_e4m3fnuz_basic
def test_convert_e4m3fn_to_e4m3fnuz_basic() raises:
    print("== test_convert_e4m3fn_to_e4m3fnuz_basic")
    var ctx = DeviceContext()

    # Test with 5 values: 4 regular values + 1 special -128 bit pattern
    var m = 1
    var n = 5
    var total_size = m * n

    # Create device buffers
    var device_in = ctx.enqueue_create_buffer[.float8_e4m3fn](total_size)
    var device_out = ctx.enqueue_create_buffer[.float8_e4m3fnuz](total_size)

    # Initialize input data on host
    with device_in.map_to_host() as host_in:
        # Regular values - these should pass through unchanged (same bits, different interpretation)
        host_in[0] = Float8_e4m3fn(1.0)
        host_in[1] = Float8_e4m3fn(2.0)
        host_in[2] = Float8_e4m3fn(-1.0)
        host_in[3] = Float8_e4m3fn(0.0)
        host_in[4] = Float8_e4m3fn(
            -0.0
        )  # Special 0x80 bit pattern - this should be converted to 0.0

    # Create TileTensors for GPU operations
    var shape = Coord(Int64(m), Int64(n))
    var in_tt = TileTensor(device_in, row_major(shape))
    var out_tt = TileTensor(device_out, row_major(shape))

    convert_e4m3fn_to_e4m3fnuz(in_tt, out_tt, ctx)
    ctx.synchronize()

    # Verify results: regular values should be unchanged in bits, -128 should become 0
    # E4M3FN -> E4M3FNUZ conversion halves values (different exponent bias)
    with device_out.map_to_host() as host_out:
        assert_equal(
            host_out[0].cast[.float32](),
            Float32(0.5),
            msg="1.0 -> 0.5",
        )
        assert_equal(
            host_out[1].cast[.float32](),
            Float32(1.0),
            msg="2.0 -> 1.0",
        )
        assert_equal(
            host_out[2].cast[.float32](),
            Float32(-0.5),
            msg="-1.0 -> -0.5",
        )
        assert_equal(
            host_out[3].cast[.float32](),
            Float32(0.0),
            msg="0.0 -> 0.0",
        )
        assert_equal(
            host_out[4].cast[.float32](),
            Float32(0.0),
            msg="-0.0 -> 0.0 (special case)",
        )


def main() raises:
    test_convert_e4m3fn_to_e4m3fnuz_basic()
