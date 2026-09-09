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

from std.os import abort

from std.builtin._format_float import _write_float
from std.builtin.simd import Float8_e4m3fn, Float8_e5m2
from max.gpu.host import DeviceContext
from std.memory import unsafe_memcmp, unsafe_memcpy


struct Buffer[capacity: Int](Defaultable, Writer):
    var data: Array[UInt8, Self.capacity]
    var pos: Int

    def __init__(out self):
        self.data = Array[UInt8, Self.capacity](fill=0)
        self.pos = 0

    def write_string(mut self, string: StringSlice):
        var data_ptr: Pointer[
            UInt8, origin_of(self.data)
        ] = self.data.unsafe_ptr()
        for i, byte in enumerate(string.bytes()):
            (data_ptr + self.pos)[i] = byte
        self.pos += string.byte_length()


def check_float[
    dtype: DType, //, expected: StaticString
](f8: Scalar[dtype]) where dtype.is_floating_point():
    var f8_str = Buffer[expected.byte_length()]()
    _write_float(f8_str, f8)
    for i, byte in enumerate(expected.bytes()):
        if byte != f8_str.data[i]:
            abort()


def check_8e5m2[expected: StaticString](f8: Float8_e5m2):
    check_float[expected](f8)


def check_8e4m3[expected: StaticString](f8: Float8_e4m3fn):
    check_float[expected](f8)


def test_format_float8_e5m2():
    check_8e5m2["0.0"](0)
    check_8e5m2["0.125"](0.125)
    check_8e5m2["1.25"](1.25)
    check_8e5m2["1.52587890625e-05"](1.52587890625e-05)
    check_8e5m2["-57344.0"](-57344)
    check_8e5m2["-0.0001068115234375"](-0.0001068115234375)
    check_8e5m2["nan"](FloatLiteral.nan)
    check_8e5m2["inf"](FloatLiteral.infinity)
    check_8e5m2["-inf"](FloatLiteral.negative_infinity)
    check_8e5m2["-0.0"](FloatLiteral.negative_zero)


def test_format_float8_e4m3fn():
    check_8e4m3["0.0"](0)
    check_8e4m3["0.001953125"](0.001953125)
    check_8e4m3["-0.01953125"](-0.01953125)
    check_8e4m3["0.001953125"](0.001953125)
    check_8e4m3["0.02734375"](0.02734375)
    check_8e4m3["0.029296875"](0.029296875)
    check_8e4m3["0.03125"](0.03125)
    check_8e4m3["0.25"](0.25)
    check_8e4m3["208.0"](208)
    check_8e4m3["-12.0"](-12)
    check_8e4m3["-104.0"](-104)


def main() raises:
    # TODO(KERN-1259): Add tests for fnuz types when they're working
    with DeviceContext() as ctx:
        print("== test_format_float8_e5m2")
        comptime kernel_0 = test_format_float8_e5m2
        ctx.enqueue_function[kernel_0](grid_dim=1, block_dim=1)

        print("== test_format_float8_e4m3fn")
        comptime kernel_1 = test_format_float8_e4m3fn
        ctx.enqueue_function[kernel_1](grid_dim=1, block_dim=1)
        ctx.synchronize()
