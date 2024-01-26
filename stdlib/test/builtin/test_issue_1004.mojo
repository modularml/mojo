# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s
# Test for https://github.com/modularml/mojo/issues/1004

from testing import assert_equal


fn foo(x: String) raises:
    raise Error("Failed on: " + x)


def main():
    try:
        foo("Hello")
    except e:
        # CHECK: Failed on: Hello
        print(e)
        assert_equal(str(e), "Failed on: Hello")
