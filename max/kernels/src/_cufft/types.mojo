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


@fieldwise_init
struct LibraryProperty(Equatable, TrivialRegisterPassable, Writable):
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
        abort("invalid LibraryProperty entry")

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct Status(Equatable, Identifiable, TrivialRegisterPassable, Writable):
    var _value: Int8
    comptime CUFFT_INVALID_PLAN = Self(1)
    comptime CUFFT_SUCCESS = Self(0)
    comptime CUFFT_ALLOC_FAILED = Self(2)
    comptime CUFFT_INVALID_TYPE = Self(3)
    comptime CUFFT_INVALID_VALUE = Self(4)
    comptime CUFFT_INTERNAL_ERROR = Self(5)
    comptime CUFFT_EXEC_FAILED = Self(6)
    comptime CUFFT_SETUP_FAILED = Self(7)
    comptime CUFFT_INVALID_SIZE = Self(8)
    comptime CUFFT_UNALIGNED_DATA = Self(9)
    comptime CUFFT_INCOMPLETE_PARAMETER_LIST = Self(10)
    comptime CUFFT_INVALID_DEVICE = Self(11)
    comptime CUFFT_PARSE_ERROR = Self(12)
    comptime CUFFT_NO_WORKSPACE = Self(13)
    comptime CUFFT_NOT_IMPLEMENTED = Self(14)
    comptime CUFFT_LICENSE_ERROR = Self(15)
    comptime CUFFT_NOT_SUPPORTED = Self(16)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUFFT_SUCCESS:
            return writer.write_string("CUFFT_SUCCESS")
        if self is Self.CUFFT_INVALID_PLAN:
            return writer.write_string("CUFFT_INVALID_PLAN")
        if self is Self.CUFFT_ALLOC_FAILED:
            return writer.write_string("CUFFT_ALLOC_FAILED")
        if self is Self.CUFFT_INVALID_TYPE:
            return writer.write_string("CUFFT_INVALID_TYPE")
        if self is Self.CUFFT_INVALID_VALUE:
            return writer.write_string("CUFFT_INVALID_VALUE")
        if self is Self.CUFFT_INTERNAL_ERROR:
            return writer.write_string("CUFFT_INTERNAL_ERROR")
        if self is Self.CUFFT_EXEC_FAILED:
            return writer.write_string("CUFFT_EXEC_FAILED")
        if self is Self.CUFFT_SETUP_FAILED:
            return writer.write_string("CUFFT_SETUP_FAILED")
        if self is Self.CUFFT_INVALID_SIZE:
            return writer.write_string("CUFFT_INVALID_SIZE")
        if self is Self.CUFFT_UNALIGNED_DATA:
            return writer.write_string("CUFFT_UNALIGNED_DATA")
        if self is Self.CUFFT_INCOMPLETE_PARAMETER_LIST:
            return writer.write_string("CUFFT_INCOMPLETE_PARAMETER_LIST")
        if self is Self.CUFFT_INVALID_DEVICE:
            return writer.write_string("CUFFT_INVALID_DEVICE")
        if self is Self.CUFFT_PARSE_ERROR:
            return writer.write_string("CUFFT_PARSE_ERROR")
        if self is Self.CUFFT_NO_WORKSPACE:
            return writer.write_string("CUFFT_NO_WORKSPACE")
        if self is Self.CUFFT_NOT_IMPLEMENTED:
            return writer.write_string("CUFFT_NOT_IMPLEMENTED")
        if self is Self.CUFFT_LICENSE_ERROR:
            return writer.write_string("CUFFT_LICENSE_ERROR")
        if self is Self.CUFFT_NOT_SUPPORTED:
            return writer.write_string("CUFFT_NOT_SUPPORTED")
        abort("invalid cufftResult_t entry")

    def write_repr_to(self, mut writer: Some[Writer]):
        t"cufftResult_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


#  CUFFT defines and supports the following data types


@fieldwise_init
struct Type(Equatable, Identifiable, TrivialRegisterPassable, Writable):
    var _value: Int8
    comptime CUFFT_R2C = Self(0x2A)
    comptime CUFFT_C2R = Self(0x2C)
    comptime CUFFT_C2C = Self(0x29)
    comptime CUFFT_D2Z = Self(0x6A)
    comptime CUFFT_Z2D = Self(0x6C)
    comptime CUFFT_Z2Z = Self(0x69)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUFFT_R2C:
            return writer.write_string("CUFFT_R2C")
        if self is Self.CUFFT_C2R:
            return writer.write_string("CUFFT_C2R")
        if self is Self.CUFFT_C2C:
            return writer.write_string("CUFFT_C2C")
        if self is Self.CUFFT_D2Z:
            return writer.write_string("CUFFT_D2Z")
        if self is Self.CUFFT_Z2D:
            return writer.write_string("CUFFT_Z2D")
        if self is Self.CUFFT_Z2Z:
            return writer.write_string("CUFFT_Z2Z")
        abort("invalid cufftType_t entry")

    def write_repr_to(self, mut writer: Some[Writer]):
        t"cufftType_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct Compatibility(
    Equatable, Identifiable, TrivialRegisterPassable, Writable
):
    var _value: Int8
    comptime CUFFT_COMPATIBILITY_FFTW_PADDING = Self(0)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.CUFFT_COMPATIBILITY_FFTW_PADDING:
            return writer.write_string("CUFFT_COMPATIBILITY_FFTW_PADDING")
        abort("invalid cufftCompatibility_t entry")

    def write_repr_to(self, mut writer: Some[Writer]):
        t"cufftCompatibility_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)


@fieldwise_init
struct Property(Equatable, Identifiable, TrivialRegisterPassable, Writable):
    var _value: Int8
    comptime NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT = Self(0)
    comptime NVFFT_PLAN_PROPERTY_INT64_MAX_NUM_HOST_THREADS = Self(1)

    def __init__(out self, value: Int):
        self._value = Int8(value)

    def __is__(self, other: Self) -> Bool:
        return self == other

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self is Self.NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT:
            return writer.write_string("NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT")
        if self is Self.NVFFT_PLAN_PROPERTY_INT64_MAX_NUM_HOST_THREADS:
            return writer.write(
                "NVFFT_PLAN_PROPERTY_INT64_MAX_NUM_HOST_THREADS"
            )
        abort("invalid cufftProperty_t entry")

    def write_repr_to(self, mut writer: Some[Writer]):
        t"cufftProperty_t({self})".write_to(writer)

    def __int__(self) -> Int:
        return Int(self._value)
