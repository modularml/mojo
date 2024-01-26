# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# This file is only run on macos targets.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: darwin
# RUN: %mojo -debug-level full %s | FileCheck %s


from sys.info import (
    is_big_endian,
    is_little_endian,
    os_is_linux,
    os_is_macos,
    os_is_windows,
)


# CHECK-LABEL: test_os_query
fn test_os_query():
    print("== test_os_query")

    # CHECK: True
    print(os_is_macos())

    # CHECK: False
    print(os_is_linux())

    # CHECK: False
    print(os_is_windows())

    # The mac systems are either arm64 or intel, so they are always little
    # endian at the moment.

    # CHECK: True
    print(is_little_endian())

    # CHECK: False
    print(is_big_endian())


fn main():
    test_os_query()
