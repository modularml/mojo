# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: is_debug
# RUN: %mojo -debug-level full %s | FileCheck %s

from sys._build import is_debug_build, is_release_build


# CHECK-OK-LABEL: test_is_debug
fn test_is_debug():
    print("== test_is_debug")

    # CHECK: True
    print(is_debug_build())

    # CHECK: False
    print(is_release_build())


fn main():
    test_is_debug()
