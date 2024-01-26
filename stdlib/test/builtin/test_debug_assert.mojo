# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# This file only tests the debug_assert function
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -D DEBUG -debug-level full %s | FileCheck %s -check-prefix=CHECK-OK


# CHECK-OK-LABEL: test_ok
fn main():
    print("== test_ok")
    debug_assert(True, "ok")
    # CHECK-OK: is reached
    print("is reached")
