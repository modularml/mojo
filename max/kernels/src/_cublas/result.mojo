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


struct Result(Equatable, TrivialRegisterPassable, Writable):
    var _value: Int32
    comptime SUCCESS = Self(0)
    comptime NOT_INITIALIZED = Self(1)
    comptime ALLOC_FAILED = Self(3)
    comptime INVALID_VALUE = Self(7)
    comptime ARCH_MISMATCH = Self(8)
    comptime MAPPING_ERROR = Self(11)
    comptime EXECUTION_FAILED = Self(13)
    comptime INTERNAL_ERROR = Self(14)
    comptime NOT_SUPPORTED = Self(15)
    comptime LICENSE_ERROR = Self(16)

    def __init__(out self, value: Int):
        self._value = Int32(value)

    @inline(.never)
    def write_to(self, mut writer: Some[Writer]):
        if self == Self.SUCCESS:
            return writer.write_string("SUCCESS")
        if self == Self.NOT_INITIALIZED:
            return writer.write_string("NOT_INITIALIZED")
        if self == Self.ALLOC_FAILED:
            return writer.write_string("ALLOC_FAILED")
        if self == Self.INVALID_VALUE:
            return writer.write_string("INVALID_VALUE")
        if self == Self.ARCH_MISMATCH:
            return writer.write_string("ARCH_MISMATCH")
        if self == Self.MAPPING_ERROR:
            return writer.write_string("MAPPING_ERROR")
        if self == Self.EXECUTION_FAILED:
            return writer.write_string("EXECUTION_FAILED")
        if self == Self.INTERNAL_ERROR:
            return writer.write_string("INTERNAL_ERROR")
        if self == Self.NOT_SUPPORTED:
            return writer.write_string("NOT_SUPPORTED")
        if self == Self.LICENSE_ERROR:
            return writer.write_string("LICENSE_ERROR")

        abort("unreachable: invalid Result entry")

    def __int__(self) -> Int:
        return Int(self._value)
