# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# This file only tests the debug_assert function
#
# ===----------------------------------------------------------------------=== #
# RUN: not --crash %mojo -D MOJO_ENABLE_ASSERTIONS -debug-level full %s 2>&1 | FileCheck %s -check-prefix=CHECK-FAIL


# CHECK-FAIL-LABEL: test_fail
fn main():
    print("== test_fail")
    debug_assert(False, "fail")
    # CHECK-FAIL-NOT: is never reached
    print("is never reached")
