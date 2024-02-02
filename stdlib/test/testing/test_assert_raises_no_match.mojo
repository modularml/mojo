# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: not %mojo -debug-level full %s 2>&1 | FileCheck %s -check-prefix=CHECK-FAIL

from testing import assert_raises


# CHECK-FAIL-LABEL: test_assert_raises_no_match
fn test_assert_raises_no_match() raises:
    print("== test_assert_raises_no_match")
    # CHECK-FAIL-NOT: is never reached
    # CHECK: AssertionError
    with assert_raises(contains="Some"):
        raise "OtherError"
    # CHECK-FAIL-NOT: is never reached
    print("is never reached")


def main():
    test_assert_raises_no_match()
