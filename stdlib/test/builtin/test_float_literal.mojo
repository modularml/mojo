# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s
from math import round

from testing import *


fn round10(x: Float64) -> Float64:
    return (round(Float64(x * 10)) / 10).value


def test_boolean_comparable():
    var f1 = FloatLiteral(0.0)
    assert_false(f1)

    var f2 = FloatLiteral(2.0)
    assert_true(f2)

    var f3 = FloatLiteral(1.0)
    assert_true(f2)


def test_equality():
    var f1 = FloatLiteral(4.4)
    var f2 = FloatLiteral(4.4)
    var f3 = FloatLiteral(42.0)
    assert_equal(f1, f2)
    assert_not_equal(f1, f3)


def main():
    # CHECK: == test_double
    print("== test_double")

    # CHECK-NEXT: 8.8
    print(FloatLiteral(4.4) / 0.5)
    # CHECK-NEXT: 8.0
    print(FloatLiteral(4.4) // 0.5)
    # CHECK-NEXT: -9.0
    print(FloatLiteral(-4.4) // 0.5)
    # CHECK-NEXT: -9.0
    print(FloatLiteral(4.4) // -0.5)
    # CHECK-NEXT: 8.0
    print(FloatLiteral(-4.4) // -0.5)

    # CHECK-NEXT: 0.4
    print(round10(FloatLiteral(4.4) % 0.5))
    # CHECK-NEXT: 0.1
    print(round10(FloatLiteral(-4.4) % 0.5))
    # CHECK-NEXT: -0.1
    print(round10(FloatLiteral(4.4) % -0.5))
    # CHECK-NEXT: -0.4
    print(round10(FloatLiteral(-4.4) % -0.5))
    # CHECK-NEXT: 0.1
    print(round10(3.1 % 1.0))

    # CHECK-NEXT: 42.95
    print(FloatLiteral(4.5) ** 2.5)
    # CHECK-NEXT: 0.023
    print(FloatLiteral(4.5) ** -2.5)
    # TODO (https://github.com/modularml/modular/issues/33045): Float64/SIMD has
    # issues with negative numbers raised to fractional powers.
    # CHECK-NEXT_DISABLED: -42.95
    # print(FloatLiteral(-4.5) ** 2.5)
    # CHECK-NEXT_DISABLED: -0.023
    # print(FloatLiteral(-4.5) ** -2.5)

    # CHECK-NEXT: -4
    print(int(FloatLiteral(-4.0)))

    # CHECK-NEXT: -4
    print(int(FloatLiteral(-4.5)))

    # CHECK-NEXT: -4
    print(int(FloatLiteral(-4.3)))

    # CHECK-NEXT: 4
    print(int(FloatLiteral(4.5)))

    # CHECK-NEXT: 4
    print(int(FloatLiteral(4.0)))

    test_boolean_comparable()
    test_equality()
