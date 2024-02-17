# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
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
from sys import external_call
from sys.info import os_is_linux, os_is_macos, os_is_windows

from memory.unsafe import Pointer


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
    let succeeded = external_call["GetProcessId", Int](Pointer[Int].get_null())

    # CHECK: 0
    print(succeeded)

    # CHECK: False
    print(last_operation_succeeded())

    # CHECK: 6
    print(get_last_error_code())


fn main():
    test_os_query()
    test_last_error()
