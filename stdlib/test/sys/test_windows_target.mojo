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
# RUN: mojo.exe %s | FileCheck %s


from os._windows import (
    get_last_error_code,
    last_operation_succeeded,
    reset_last_error,
)
from sys import external_call, os_is_linux, os_is_macos, os_is_windows

from memory import Pointer


# CHECK-LABEL: test_os_query
fn test_os_query():
    print("== test_os_query\n")

    # CHECK: False
    print(os_is_macos())

    # CHECK: False
    print(os_is_linux())

    # CHECK: True
    print(os_is_windows())


# CHECK-LABEL: test_last_error
fn test_last_error():
    print("== test_last_error\n")

    reset_last_error()

    # CHECK: 0
    print(get_last_error_code())

    # CHECK: True
    print(last_operation_succeeded())

    # GetProcessId takes the handle to a process and returns its id. If the
    # handle is null this will fail and returns an invalid handle error (error
    # code 6).
    var succeeded = external_call["GetProcessId", Int](Pointer[Int].get_null())

    # CHECK: 0
    print(succeeded)

    # CHECK: False
    print(last_operation_succeeded())

    # CHECK: 6
    print(get_last_error_code())


fn main():
    test_os_query()
    test_last_error()
