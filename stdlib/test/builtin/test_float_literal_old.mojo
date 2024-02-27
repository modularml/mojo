# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s
from math import round


fn round10(x: FloatLiteralOld) -> FloatLiteralOld:
    return (round(Float64(x * 10)) / 10).value


fn main():
    # CHECK: == test_double
    print("== test_double")

    # CHECK-NEXT: 8.8
    print(FloatLiteralOld(4.4) / 0.5)
    # CHECK-NEXT: 8.0
    print(FloatLiteralOld(4.4) // 0.5)
    # CHECK-NEXT: -9.0
    print(FloatLiteralOld(-4.4) // 0.5)
    # CHECK-NEXT: -9.0
    print(FloatLiteralOld(4.4) // -0.5)
    # CHECK-NEXT: 8.0
    print(FloatLiteralOld(-4.4) // -0.5)

    # CHECK-NEXT: 0.4
    print(round10(FloatLiteralOld(4.4) % 0.5))
    # CHECK-NEXT: 0.1
    print(round10(FloatLiteralOld(-4.4) % 0.5))
    # CHECK-NEXT: -0.1
    print(round10(FloatLiteralOld(4.4) % -0.5))
    # CHECK-NEXT: -0.4
    print(round10(FloatLiteralOld(-4.4) % -0.5))
    # CHECK-NEXT: 0.1
    print(round10(3.1 % 1.0))

    # CHECK-NEXT: 42.95
    print(FloatLiteralOld(4.5) ** 2.5)
    # CHECK-NEXT: -42.95
    print(FloatLiteralOld(-4.5) ** 2.5)
    # CHECK-NEXT: 0.023
    print(FloatLiteralOld(4.5) ** -2.5)
    # CHECK-NEXT: -0.023
    print(FloatLiteralOld(-4.5) ** -2.5)

    # CHECK-NEXT: -4
    print(int(FloatLiteralOld(-4.0)))

    # CHECK-NEXT: -4
    print(int(FloatLiteralOld(-4.5)))

    # CHECK-NEXT: -4
    print(int(FloatLiteralOld(-4.3)))

    # CHECK-NEXT: 4
    print(int(FloatLiteralOld(4.5)))

    # CHECK-NEXT: 4
    print(int(FloatLiteralOld(4.0)))
