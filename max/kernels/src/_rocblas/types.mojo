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

from std.collections import OptionalReg
from std.os import abort


@fieldwise_init
struct Handle(Defaultable, Equatable, TrivialRegisterPassable):
    var _value: OptionalReg[OpaquePointer[MutUntrackedOrigin]]

    def __init__(out self):
        self._value = None


@fieldwise_init
struct Operation(Equatable, TrivialRegisterPassable):
    var _value: Int32

    comptime NONE = Self(111)
    comptime TRANSPOSE = Self(112)
    comptime CONJUGATE_TRANSPOSE = Self(113)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct Fill(Equatable, TrivialRegisterPassable):
    var _value: Int32

    comptime UPPER = Self(121)
    comptime LOWER = Self(122)
    comptime FULL = Self(123)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct Diagonal(Equatable, TrivialRegisterPassable):
    var _value: Int32

    comptime NON_UNIT = Self(131)
    comptime DIAGONAL_UNIT = Self(132)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct Side(Equatable, TrivialRegisterPassable):
    var _value: Int32

    comptime LEFT = Self(141)
    comptime RIGHT = Self(142)
    comptime BOTH = Self(143)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct DataType(Equatable, TrivialRegisterPassable):
    var _value: Int32

    comptime F16_R = Self(150)
    comptime F32_R = Self(151)
    comptime F64_R = Self(152)
    comptime F16_C = Self(153)
    comptime F32_C = Self(154)
    comptime F64_C = Self(155)

    comptime I8_R = Self(160)
    comptime U8_R = Self(161)
    comptime I32_R = Self(162)
    comptime U32_R = Self(163)

    comptime I8_C = Self(164)
    comptime U8_C = Self(165)
    comptime I32_C = Self(166)
    comptime U32_C = Self(167)

    comptime BF16_R = Self(168)
    comptime BF16_C = Self(169)
    comptime F8_R = Self(170)
    comptime BF8_R = Self(171)

    comptime INVALID = Self(255)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __init__(out self, dtype: DType) raises:
        if dtype == .float16:
            self = Self.F16_R
        elif dtype == .bfloat16:
            self = Self.BF16_R
        elif dtype == .float32:
            self = Self.F32_R
        elif dtype == .float64:
            self = Self.F64_R
        else:
            raise Error(
                "the dtype '", dtype, "' is not currently handled by rocBLAS"
            )

    def __int__(self) -> Int:
        return Int(self._value)


struct ComputeType(Equatable, TrivialRegisterPassable):
    var _value: Int32

    comptime F32 = Self(300)
    comptime F8_F8_F32 = Self(301)
    comptime F8_BF8_F32 = Self(302)
    comptime BF8_F8_F32 = Self(303)
    comptime BF8_BF8_F32 = Self(304)
    comptime INVALID = Self(455)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct Status(Equatable, TrivialRegisterPassable, Writable):
    var _value: Int32

    comptime SUCCESS = Self(0)
    comptime INVALID_HANDLE = Self(1)
    comptime NOT_IMPLEMENTED = Self(2)
    comptime INVALID_POINTER = Self(3)
    comptime INVALID_SIZE = Self(4)
    comptime MEMORY_ERROR = Self(5)
    comptime INTERNAL_ERROR = Self(6)
    comptime PERF_DEGRADED = Self(7)
    comptime SIZE_QUERY_MISMATCH = Self(8)
    comptime SIZE_INCREASED = Self(9)
    comptime SIZE_UNCHANGED = Self(10)
    comptime INVALID_VALUE = Self(11)
    comptime CONTINUE = Self(12)
    comptime CHECK_NUMERICS_FAIL = Self(13)
    comptime EXCLUDED_FROM_BUILD = Self(14)
    comptime ARCH_MISMATCH = Self(15)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __int__(self) -> Int:
        return Int(self._value)

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.SUCCESS:
            return writer.write_string("SUCCESS")
        if self == Self.INVALID_HANDLE:
            return writer.write_string("INVALID_HANDLE")
        if self == Self.NOT_IMPLEMENTED:
            return writer.write_string("NOT_IMPLEMENTED")
        if self == Self.INVALID_POINTER:
            return writer.write_string("INVALID_POINTER")
        if self == Self.INVALID_SIZE:
            return writer.write_string("INVALID_SIZE")
        if self == Self.MEMORY_ERROR:
            return writer.write_string("MEMORY_ERROR")
        if self == Self.INTERNAL_ERROR:
            return writer.write_string("INTERNAL_ERROR")
        if self == Self.PERF_DEGRADED:
            return writer.write_string("PERF_DEGRADED")
        if self == Self.SIZE_QUERY_MISMATCH:
            return writer.write_string("SIZE_QUERY_MISMATCH")
        if self == Self.SIZE_INCREASED:
            return writer.write_string("SIZE_INCREASED")
        if self == Self.SIZE_UNCHANGED:
            return writer.write_string("SIZE_UNCHANGED")
        if self == Self.INVALID_VALUE:
            return writer.write_string("INVALID_VALUE")
        if self == Self.CONTINUE:
            return writer.write_string("CONTINUE")
        if self == Self.CHECK_NUMERICS_FAIL:
            return writer.write_string("CHECK_NUMERICS_FAIL")
        if self == Self.EXCLUDED_FROM_BUILD:
            return writer.write_string("EXCLUDED_FROM_BUILD")
        if self == Self.ARCH_MISMATCH:
            return writer.write_string("ARCH_MISMATCH")

        abort("unreachable: invalid Status entry")


@fieldwise_init
struct PointerMode(Equatable, TrivialRegisterPassable):
    var _value: Int32

    comptime HOST = Self(0)
    comptime DEVICE = Self(1)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct MallocBase(TrivialRegisterPassable):
    var _value: Int32


@fieldwise_init
struct Algorithm(Equatable, TrivialRegisterPassable):
    var _value: Int32

    comptime STANDARD = Self(0)
    comptime SOLUTION_INDEX = Self(1)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct GEAMExOp(Equatable, TrivialRegisterPassable):
    var _value: Int32

    comptime MIN_PLUS = Self(0)
    comptime PLUS_MIN = Self(1)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    def __int__(self) -> Int:
        return Int(self._value)
