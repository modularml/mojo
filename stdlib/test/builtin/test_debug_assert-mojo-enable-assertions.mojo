# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# This file only tests the debug_assert function
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_not
# RUN: not --crash %mojo -D MOJO_ENABLE_ASSERTIONS -debug-level full %s 2>&1 | FileCheck %s


# CHECK-LABEL: test_fail
fn main():
    print("== test_fail")
    # CHECK: Assert Error: fail
    debug_assert(False, "fail")
    # CHECK-NOT: is never reached
    print("is never reached")
