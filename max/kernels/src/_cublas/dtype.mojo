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


struct Property(Equatable, TrivialRegisterPassable, Writable):
    var _value: Int32
    comptime MAJOR_VERSION = Self(0)
    comptime MINOR_VERSION = Self(1)
    comptime PATCH_LEVEL = Self(2)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.MAJOR_VERSION:
            return writer.write_string("MAJOR_VERSION")
        if self == Self.MINOR_VERSION:
            return writer.write_string("MINOR_VERSION")
        if self == Self.PATCH_LEVEL:
            return writer.write_string("PATCH_LEVEL")
        abort("invalid Property entry")

    def __int__(self) -> Int:
        return Int(self._value)


struct DataType(Equatable, TrivialRegisterPassable, Writable):
    var _value: Int32
    comptime R_16F = Self(2)
    comptime C_16F = Self(6)
    comptime R_16BF = Self(14)
    comptime C_16BF = Self(15)
    comptime R_32F = Self(0)
    comptime C_32F = Self(4)
    comptime R_64F = Self(1)
    comptime C_64F = Self(5)
    comptime R_8I = Self(3)
    comptime C_8I = Self(7)
    comptime R_8U = Self(8)
    comptime C_8U = Self(9)
    comptime R_32I = Self(10)
    comptime C_32I = Self(11)
    comptime R_4I = Self(16)
    comptime C_4I = Self(17)
    comptime R_4U = Self(18)
    comptime C_4U = Self(19)
    comptime R_16I = Self(20)
    comptime C_16I = Self(21)
    comptime R_16U = Self(22)
    comptime C_16U = Self(23)
    comptime R_32U = Self(12)
    comptime C_32U = Self(13)
    comptime R_64I = Self(24)
    comptime C_64I = Self(25)
    comptime R_64U = Self(26)
    comptime C_64U = Self(27)
    comptime R_8F_E4M3 = Self(28)
    comptime R_8F_UE4M3 = Self.R_8F_E4M3
    comptime R_8F_E5M2 = Self(29)
    comptime R_8F_UE8M0 = Self(30)
    comptime R_6F_E2M3 = Self(31)
    comptime R_6F_E3M2 = Self(32)
    comptime R_4F_E2M1 = Self(33)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.R_16F:
            return writer.write_string("R_16F")
        if self == Self.C_16F:
            return writer.write_string("C_16F")
        if self == Self.R_16BF:
            return writer.write_string("R_16BF")
        if self == Self.C_16BF:
            return writer.write_string("C_16BF")
        if self == Self.R_32F:
            return writer.write_string("R_32F")
        if self == Self.C_32F:
            return writer.write_string("C_32F")
        if self == Self.R_64F:
            return writer.write_string("R_64F")
        if self == Self.C_64F:
            return writer.write_string("C_64F")
        if self == Self.R_8I:
            return writer.write_string("R_8I")
        if self == Self.C_8I:
            return writer.write_string("C_8I")
        if self == Self.R_8U:
            return writer.write_string("R_8U")
        if self == Self.C_8U:
            return writer.write_string("C_8U")
        if self == Self.R_32I:
            return writer.write_string("R_32I")
        if self == Self.C_32I:
            return writer.write_string("C_32I")
        if self == Self.R_4I:
            return writer.write_string("R_4I")
        if self == Self.C_4I:
            return writer.write_string("C_4I")
        if self == Self.R_4U:
            return writer.write_string("R_4U")
        if self == Self.C_4U:
            return writer.write_string("C_4U")
        if self == Self.R_16I:
            return writer.write_string("R_16I")
        if self == Self.C_16I:
            return writer.write_string("C_16I")
        if self == Self.R_16U:
            return writer.write_string("R_16U")
        if self == Self.C_16U:
            return writer.write_string("C_16U")
        if self == Self.R_32U:
            return writer.write_string("R_32U")
        if self == Self.C_32U:
            return writer.write_string("C_32U")
        if self == Self.R_64I:
            return writer.write_string("R_64I")
        if self == Self.C_64I:
            return writer.write_string("C_64I")
        if self == Self.R_64U:
            return writer.write_string("R_64U")
        if self == Self.C_64U:
            return writer.write_string("C_64U")
        if self == Self.R_8F_E4M3:
            return writer.write_string("R_8F_E4M3")
        if self == Self.R_8F_E5M2:
            return writer.write_string("R_8F_E5M2")
        if self == Self.R_8F_UE4M3:
            return writer.write_string("R_8F_UE4M3")
        if self == Self.R_8F_UE8M0:
            return writer.write_string("R_8F_UE8M0")
        if self == Self.R_6F_E2M3:
            return writer.write_string("R_6F_E2M3")
        if self == Self.R_6F_E3M2:
            return writer.write_string("R_6F_E3M2")
        if self == Self.R_4F_E2M1:
            return writer.write_string("R_4F_E2M1")

        abort("invalid DataType entry")

    def __int__(self) -> Int:
        return Int(self._value)
