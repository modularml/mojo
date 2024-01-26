# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# This file is only run on linux targets.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: linux
# RUN: %mojo -debug-level full %s | FileCheck %s


from sys.info import os_is_linux, os_is_macos


# CHECK-LABEL: test_os_query
fn test_os_query():
    print("== test_os_query")

    # CHECK: False
    print(os_is_macos())

    # CHECK: True
    print(os_is_linux())


fn main():
    test_os_query()
