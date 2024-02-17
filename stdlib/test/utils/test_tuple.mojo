# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from utils.index import StaticIntTuple
from utils.static_tuple import StaticTuple


# CHECK-LABEL: test_static_tuple
fn test_static_tuple():
    print("== test_static_tuple")

    # CHECK: 1
    let tup1 = StaticTuple[1, Int](1)
    print(tup1[0])

    # CHECK: 1 1
    let tup2 = StaticTuple[2, Int](1)
    print(tup2[0], tup2[1])

    # CHECK: 1 2 3
    let tup3 = StaticTuple[3, Int](1, 2, 3)
    print(tup3[0], tup3[1], tup3[2])

    # CHECK: 1 1 1
    print(tup3[0], tup3[Int(0)], tup3[Int64(0)])


# CHECK-LABEL: test_static_int_tuple
fn test_static_int_tuple():
    print("== test_static_int_tuple")
    # CHECK: (1,)
    print(StaticIntTuple[1](1))
    # CHECK: (2, 2, 2)
    print(StaticIntTuple[3](2))
    # CHECK: (4, 10, 18)
    print(StaticIntTuple[3](1, 2, 3) * StaticIntTuple[3](4, 5, 6))
    # CHECK: (-3, -3, -3, -3)
    print(StaticIntTuple[4](1, 2, 3, 4) - StaticIntTuple[4](4, 5, 6, 7))
    # CHECK: (3, 2)
    print(StaticIntTuple[2](10, 11) // StaticIntTuple[2](3, 4))
    # Note: index comparison is intended for access bound checking, which is
    #  usually all-element semantic, i.e. true if true for all positions.
    # CHECK: True
    print(StaticIntTuple[5](1, 2, 3, 4, 5) < StaticIntTuple[5](4, 5, 6, 7, 8))
    # CHECK: False
    print(StaticIntTuple[4](3, 5, -1, -2) > StaticIntTuple[4](0, 0, 0, 0))
    # CHECK: 4
    print(len(StaticIntTuple[4](3, 5, -1, -2)))

    # CHECK: (1, 2)
    print(StaticIntTuple[2]((1, 2)))

    # CHECK: (1, 2, 3, 4)
    print(StaticIntTuple[4]((1, 2, 3, 4)))


# CHECK-LABEL: test_tuple_literal
fn test_tuple_literal():
    print("== test_tuple_literal\n")
    # CHECK: 4
    print(len((1, 2, (3, 4), 5)))
    # CHECK: 0
    print(len(()))
    return


fn main():
    test_static_tuple()
    test_static_int_tuple()
    test_tuple_literal()
