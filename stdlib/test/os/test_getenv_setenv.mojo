# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: linux || darwin
# RUN: TEST_MYVAR=MyValue %mojo -debug-level full %s

from os import getenv, setenv

from testing import assert_equal


def test_getenv():
    print("== test_getenv")

    assert_equal(getenv("TEST_MYVAR"), "MyValue")

    assert_equal(getenv("TEST_MYVAR", "DefaultValue"), "MyValue")

    assert_equal(getenv("NON_EXISTENT_VAR", "DefaultValue"), "DefaultValue")


# CHECK-OK-LABEL: test_setenv
def test_setenv():
    print("== test_setenv")

    assert_equal(setenv("NEW_VAR", "FOO", True), True)
    assert_equal(getenv("NEW_VAR"), "FOO")

    assert_equal(setenv("NEW_VAR", "BAR", False), True)
    assert_equal(getenv("NEW_VAR"), "FOO")

    assert_equal(setenv("NEW_VAR", "BAR", True), True)
    assert_equal(getenv("NEW_VAR", "BAR"), "BAR")

    assert_equal(setenv("=", "INVALID", True), False)


def main():
    test_getenv()
    test_setenv()
