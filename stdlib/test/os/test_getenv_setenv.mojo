# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: linux || darwin
# RUN: TEST_MYVAR=MyValue mojo %s | FileCheck %s

from os import getenv, setenv


# CHECK-OK-LABEL: test_getenv
fn test_getenv():
    print("== test_getenv")

    # CHECK: MyValue
    print(getenv("TEST_MYVAR"))

    # CHECK: MyValue
    print(getenv("TEST_MYVAR", "DefaultValue"))

    # CHECK: DefaultValue
    print(getenv("NON_EXISTENT_VAR", "DefaultValue"))


# CHECK-OK-LABEL: test_setenv
fn test_setenv():
    print("== test_setenv")

    # CHECK: True
    print(setenv("NEW_VAR", "FOO", True))
    # CHECK: FOO
    print(getenv("NEW_VAR"))

    # CHECK: True
    print(setenv("NEW_VAR", "BAR", False))
    # CHECK: FOO
    print(getenv("NEW_VAR"))

    # CHECK: True
    print(setenv("NEW_VAR", "BAR", True))
    # CHECK: BAR
    print(getenv("NEW_VAR", "BAR"))

    # CHECK: False
    print(setenv("=", "INVALID", True))


fn main():
    test_getenv()
    test_setenv()
