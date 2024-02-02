# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from testing import assert_raises


# CHECK-LABEL: test_assert_raises_catches_error
fn test_assert_raises_catches_error() raises:
    print("== test_assert_raises_catches_error")
    with assert_raises():
        raise "SomeError"
    # CHECK: 🔥
    print("🔥")


# CHECK-LABEL: test_assert_raises_catches_matched_error
fn test_assert_raises_catches_matched_error() raises:
    print("== test_assert_raises_catches_matched_error")
    with assert_raises(contains="Some"):
        raise "SomeError"
    # CHECK: 🔥
    print("🔥")

    with assert_raises(contains="Error"):
        raise "SomeError"
    # CHECK: 🔥
    print("🔥")

    with assert_raises(contains="eE"):
        raise "SomeError"
    # CHECK: 🔥
    print("🔥")


def main():
    test_assert_raises_catches_error()
    test_assert_raises_catches_matched_error()
