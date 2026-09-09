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
from std.memory.alloc import alloc, dealloc, Layout
from std.testing import assert_equal


# start-invert-red-channel
def invert_red_channel(ptr: Pointer[mut=True, UInt8, _], pixel_count: Int):
    # Number of values loaded or stored at a time
    comptime simd_width = 8
    # Bytes per pixel, which is also the stride size
    comptime bpp = 3
    for i in range(0, pixel_count * bpp, simd_width * bpp):
        var red_values = ptr.unsafe_offset(i).unsafe_strided_load[
            width=simd_width
        ](bpp)
        # Invert values and store them in their original locations
        ptr.unsafe_offset(i).unsafe_strided_store[width=simd_width](
            ~red_values, bpp
        )


# end-invert-red-channel


def main() raises:
    # Create an 8-pixel RGB image (8 * 3 = 24 bytes) with red=100, G=0, B=0
    comptime pixel_count = 8
    comptime bpp = 3
    var allocation = alloc(Layout[UInt8](count=pixel_count * bpp))
    var img = allocation.unsafe_ptr()
    for i in range(pixel_count):
        img.unsafe_offset(i * bpp).unsafe_write(UInt8(100))  # R
        img.unsafe_offset(i * bpp + 1).unsafe_write(UInt8(0))  # G
        img.unsafe_offset(i * bpp + 2).unsafe_write(UInt8(0))  # B

    invert_red_channel(img, pixel_count)

    # Copy the red values out before releasing the allocation: an assertion
    # that raises would abandon the allocation without deallocating it.
    var red_values = List[UInt8](capacity=pixel_count)
    for i in range(pixel_count):
        red_values.append(img[unsafe_offset=i * bpp])

    dealloc(allocation^)

    # After inversion, red values should be ~100 = 155 (bitwise NOT of 100)
    for value in red_values:
        assert_equal(value, 155)
