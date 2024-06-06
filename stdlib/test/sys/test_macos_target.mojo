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
#
# This file is only run on macos targets.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: darwin
# RUN: %mojo %s


from sys import (
    is_big_endian,
    is_little_endian,
    os_is_linux,
    os_is_macos,
    os_is_windows,
)
from sys.info import _macos_version

from testing import assert_false, assert_true


fn test_os_query() raises:
    assert_true(os_is_macos())

    assert_false(os_is_linux())

    assert_false(os_is_windows())

    # The mac systems are either arm64 or intel, so they are always little
    # endian at the moment.

    assert_true(is_little_endian())

    assert_false(is_big_endian())


def test_os_version():
    var major = 0
    var minor = 0
    var patch = 0

    major, minor, patch = _macos_version()

    assert_true(major >= 13)
    assert_true(minor >= 0)
    assert_true(patch >= 0)


def main():
    test_os_query()
    test_os_version()
