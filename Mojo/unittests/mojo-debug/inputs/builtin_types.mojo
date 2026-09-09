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

from std.collections.string import StaticString


def getFloat() -> Float32:
    return 4.125


@fieldwise_init
struct ARegisterPassableStruct(TrivialRegisterPassable):
    var int: Int
    var f32: Float32
    var another_int: Int
    var float16: Float16
    var uint8: UInt8
    var simd: SIMD[.float16, 4]
    var none: None
    var uint16: UInt16
    var int32: Int32

    def __init__(out self):
        self.int = -101
        self.f32 = 24.125
        self.another_int = 101
        self.float16 = 25.125
        self.uint8 = 123
        self.simd = SIMD[.float16, 4](-0.125, -1.5, -1, 5.725)
        self.none = None
        self.uint16 = 123
        self.int32 = 485


struct AStruct:
    var int: UInt8
    var tuple: Tuple[Int, Int8, Float32]

    def __init__(out self):
        self.int = 12
        self.tuple = Tuple[Int, Int8, Float32](1, 87, 123.125)


@fieldwise_init("implicit")
struct ParamStruct[T: TrivialRegisterPassable]:
    var t: Self.T


def keep_alive[*Ts: AnyType](*args: *Ts):
    pass


comptime AFloatOrBoolOrSimd = __mlir_type[
    `!kgen.variant<`,
    Float64,
    `, `,
    Bool,
    `, `,
    SIMD[.int, 2],
    `>`,
]


def main():
    var none = None

    var a_var_index = __mlir_op.`index.constant`[value=__mlir_attr.`48:index`]()

    var a_register_passable_struct = ARegisterPassableStruct()

    var a_struct = AStruct()

    var p_struct_int = ParamStruct[Int](8)

    var p_struct_string_slice = ParamStruct(StaticString("hello"))

    var an_int: Int = 123

    var a_literal_float = 3.125

    var a_float = Float32(3.125)

    var another_float = getFloat()

    var `^ uncommon name` = 1123123

    var a_string = "fofofo"

    var a_list = [1, 2, 3]

    var a_simd = SIMD[.float16, 4](1.125, 2.5, 0, -3.725)

    # fmt: off
    var b_simd = SIMD[.int64, 32](
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16,
    )
    # fmt: on

    var c_simd = SIMD[.int, 2](5, 6)

    var u8_max = UInt8(255)
    var u64_max = UInt64.MAX

    var a_float_or_bool_or_simd = __mlir_op.`kgen.variant.create`[
        _type=AFloatOrBoolOrSimd,
        index=Int(2).__mlir_index__(),
    ](c_simd)

    print("breakpoint")  # breakpoint

    keep_alive(
        a_var_index,
        a_register_passable_struct,
        a_struct,
        p_struct_int,
        p_struct_string_slice,
        an_int,
        a_literal_float,
        a_float,
        another_float,
        `^ uncommon name`,
        a_string,
        a_list,
        a_simd,
        b_simd,
        c_simd,
        u8_max,
        u64_max,
        a_float_or_bool_or_simd,
        none,
    )
