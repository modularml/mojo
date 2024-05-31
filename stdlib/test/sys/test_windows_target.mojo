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
# This file is only run on windows targets.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: windows
# RUN: mojo.exe %s


from os._windows import (
    get_last_error_code,
    last_operation_succeeded,
    reset_last_error,
)
from sys import external_call, os_is_linux, os_is_macos, os_is_windows

from testing import assert_false, assert_true, assert_equal


def test_os_query():
    assert_false(os_is_macos())
    assert_false(os_is_linux())
    assert_true(os_is_windows())


def test_last_error():
    reset_last_error()

    assert_equal(get_last_error_code(), 0)

    assert_true(last_operation_succeeded())

    # GetProcessId takes the handle to a process and returns its id. If the
    # handle is null this will fail and returns an invalid handle error (error
    # code 6).
    var succeeded = external_call["GetProcessId", Int](UnsafePointer[Int]())

    assert_equal(succeeded, 0)

    assert_false(last_operation_succeeded())
    assert_equal(get_last_error_code(), 6)


def main():
    test_os_query()
    test_last_error()
