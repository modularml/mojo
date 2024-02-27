# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s
from math import round


fn round10(x: Float64) -> Float64:
    return (round(Float64(x * 10)) / 10).value


fn main():
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
    # CHECK-NEXT: -42.95
    print(FloatLiteral(-4.5) ** 2.5)
    # CHECK-NEXT: 0.023
    print(FloatLiteral(4.5) ** -2.5)
    # CHECK-NEXT: -0.023
    print(FloatLiteral(-4.5) ** -2.5)

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
