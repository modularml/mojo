# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# RUN: %mojo %s

from sys.ffi import c_int, c_long, c_long_long
from sys.info import is_32bit, is_64bit, os_is_linux, os_is_macos, os_is_windows

from testing import assert_equal, assert_true

#
# Reference:
#     https://en.wikipedia.org/wiki/64-bit_computing#64-bit_data_models
#


def test_c_int_type():
    if is_64bit() and (os_is_macos() or os_is_linux() or os_is_windows()):
        # `int` is always 32 bits on the modern 64-bit OSes.
        assert_equal(c_int.type, DType.int32)
    else:
        assert_true(False, "platform c_int size is untested")


def test_c_long_type():
    if is_64bit() and (os_is_macos() or os_is_linux()):
        # `long` is 64 bits on macOS and Linux.
        assert_equal(c_long.type, DType.int64)
    elif is_64bit() and os_is_windows():
        # `long` is 32 bits only on Windows.
        assert_equal(c_long.type, DType.int32)
    else:
        assert_true(False, "platform c_long size is untested")


def test_c_long_long_type():
    if is_64bit() and (os_is_macos() or os_is_linux() or os_is_windows()):
        assert_equal(c_long_long.type, DType.int64)
    else:
        assert_true(False, "platform c_long_long size is untested")


def main():
    test_c_int_type()
    test_c_long_type()
    test_c_long_long_type()
