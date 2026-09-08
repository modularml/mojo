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

from std.ffi import c_int, c_long, c_long_long, c_ulong, c_ulong_long
from std.sys.info import CompilationTarget, is_64bit

from std.testing import assert_equal, assert_true
from std.testing import TestSuite

#
# Reference:
#     https://en.wikipedia.org/wiki/64-bit_computing#64-bit_data_models
#


def test_c_int_type() raises:
    if is_64bit() and (
        CompilationTarget.is_macos() or CompilationTarget.is_linux()
    ):
        # `int` is always 32 bits on the modern 64-bit OSes.
        assert_equal(c_int.dtype, DType.int32)
    else:
        assert_true(False, "platform c_int size is untested")


def test_c_long_types() raises:
    if is_64bit() and (
        CompilationTarget.is_macos() or CompilationTarget.is_linux()
    ):
        # `long` is 64 bits on macOS and Linux.
        assert_equal(c_long.dtype, DType.int64)
        assert_equal(c_ulong.dtype, DType.uint64)
    else:
        assert_true(False, "platform c_long and c_ulong size is untested")


def test_c_long_long_types() raises:
    if is_64bit() and (
        CompilationTarget.is_macos() or CompilationTarget.is_linux()
    ):
        assert_equal(c_long_long.dtype, DType.int64)
        assert_equal(c_ulong_long.dtype, DType.uint64)
    else:
        assert_true(
            False, "platform c_long_long and c_ulong_long size is untested"
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
