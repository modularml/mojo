# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# This file only tests the debug_assert function
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -D ASSERT_WARNING -debug-level full %s | FileCheck %s -check-prefix=CHECK-WARN


# CHECK-WARN: test_ok
fn main():
    print("== test_ok")
    # CHECK-WARN: Assert Warning: failed, but we don't terminate
    debug_assert(False, "failed, but we don't terminate")
    # CHECK-WARN: is reached
    print("is reached")
