# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from math import max, min


# CHECK-LABEL: test_int
fn test_int():
    print("== test_int")
    # CHECK: 3
    print(3)
    # CHECK: 6
    print(3 + 3)

    # CHECK: 3
    print(4 - 1)
    # CHECK: 5
    print(6 - 1)


# CHECK-LABEL: test_floordiv
fn test_floordiv():
    print("== test_floordiv")

    # CHECK: 1
    print(2 // 2)

    # CHECK: 0
    print(2 // 3)

    # CHECK: -1
    print(2 // -2)

    # CHECK: -50
    print(99 // -2)


# CHECK-LABEL: test_mod
fn test_mod():
    print("== test_mod")

    # CHECK: 0
    print(99 % 1)
    # CHECK: 0
    print(99 % 3)
    # CHECK: -1
    print(99 % -2)
    # CHECK: 3
    print(99 % 8)
    # CHECK: -5
    print(99 % -8)
    # CHECK: 0
    print(2 % -1)
    # CHECK: 0
    print(2 % -2)
    # CHECK: -1
    print(3 % -2)
    # CHECK: 1
    print(-3 % 2)


# CHECK-LABEL: test_bit_width
fn test_bit_width():
    print("== test_bit_width")

    # CHECK: 1
    print((0)._bit_width())
    # CHECK: 1
    print((-1)._bit_width())
    # CHECK: 9
    print((255)._bit_width())
    # CHECK: 9
    print((-256)._bit_width())


fn main():
    test_int()
    test_floordiv()
    test_mod()
    test_bit_width()
