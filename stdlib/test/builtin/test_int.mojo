# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from math import max, min, divmod


# CHECK-LABEL: test_int
fn test_int():
    print("== test_int")
    # CHECK: 3
    print(Int(3))
    # CHECK: 5
    print(Int(Int(5)))
    # CHECK: 6
    print(Int(3) + Int(3))

    var n = Int(5)
    let d = Int(2)
    # CHECK: 2.5
    print(n / d)
    # CHECK: 2
    n /= d
    print(n)

    # CHECK: 1
    print(Int(3) ** Int(0))
    # CHECK: 27
    print(Int(3) ** Int(3))
    # CHECK: 81
    print(Int(3) ** Int(4))
    # CHECK: 3
    print(Int(4) - Int(1))
    # CHECK: 5
    print(Int(6) - Int(1))

    let a = 0
    let b = a + Int(1)
    # CHECK: True
    print(min(a, b) == a)
    # CHECK: True
    print(max(a, b) == b)


# CHECK-LABEL: test_floordiv
fn test_floordiv():
    print("== test_floordiv")

    # CHECK: 1
    print(Int(2) // Int(2))

    # CHECK: 0
    print(Int(2) // Int(3))

    # CHECK: -1
    print(Int(2) // Int(-2))

    # CHECK: -50
    print(Int(99) // Int(-2))

    # CHECK: -1
    print(Int(-1) // Int(10))


# CHECK-LABEL: test_mod
fn test_mod():
    print("== test_mod")

    # CHECK: 0
    print(Int(99) % Int(1))
    # CHECK: 0
    print(Int(99) % Int(3))
    # CHECK: -1
    print(Int(99) % Int(-2))
    # CHECK: 3
    print(Int(99) % Int(8))
    # CHECK: -5
    print(Int(99) % Int(-8))
    # CHECK: 0
    print(Int(2) % Int(-1))
    # CHECK: 0
    print(Int(2) % Int(-2))
    # CHECK: -1
    print(Int(3) % Int(-2))
    # CHECK: 1
    print(Int(-3) % Int(2))


# CHECK-LABEL: test_divmod
fn test_divmod():
    print("== test_divmod")

    # CHECK: (-2, 1)
    print(divmod(-3, 2))


fn main():
    test_int()
    test_floordiv()
    test_mod()
    test_divmod()
