# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from utils.index import StaticIntTuple
from utils.static_tuple import StaticTuple

from testing import assert_equal, assert_true, assert_false


def test_static_tuple():
    print("== test_static_tuple")

    var tup1 = StaticTuple[1, Int](1)
    assert_equal(tup1[0], 1)

    var tup2 = StaticTuple[2, Int](1)
    assert_equal(tup2[0], 1)
    assert_equal(tup2[1], 1)

    var tup3 = StaticTuple[3, Int](1, 2, 3)
    assert_equal(tup3[0], 1)
    assert_equal(tup3[1], 2)
    assert_equal(tup3[2], 3)

    assert_equal(tup3[0], 1)
    assert_equal(tup3[Int(0)], 1)
    assert_equal(tup3[Int64(0)], 1)


def test_static_int_tuple():
    print("== test_static_int_tuple")
    assert_equal(str(StaticIntTuple[1](1)), "(1,)")

    assert_equal(str(StaticIntTuple[3](2)), "(2, 2, 2)")

    assert_equal(
        str(StaticIntTuple[3](1, 2, 3) * StaticIntTuple[3](4, 5, 6)),
        "(4, 10, 18)",
    )

    assert_equal(
        str(StaticIntTuple[4](1, 2, 3, 4) - StaticIntTuple[4](4, 5, 6, 7)),
        "(-3, -3, -3, -3)",
    )

    assert_equal(
        str(StaticIntTuple[2](10, 11) // StaticIntTuple[2](3, 4)), "(3, 2)"
    )

    # Note: index comparison is intended for access bound checking, which is
    #  usually all-element semantic, i.e. true if true for all positions.
    assert_true(
        StaticIntTuple[5](1, 2, 3, 4, 5) < StaticIntTuple[5](4, 5, 6, 7, 8)
    )

    assert_false(
        StaticIntTuple[4](3, 5, -1, -2) > StaticIntTuple[4](0, 0, 0, 0)
    )

    assert_equal(len(StaticIntTuple[4](3, 5, -1, -2)), 4)

    assert_equal(str(StaticIntTuple[2]((1, 2))), "(1, 2)")

    assert_equal(str(StaticIntTuple[4]((1, 2, 3, 4))), "(1, 2, 3, 4)")


def test_tuple_literal():
    print("== test_tuple_literal\n")
    assert_equal(len((1, 2, (3, 4), 5)), 4)
    assert_equal(len(()), 0)


def main():
    test_static_tuple()
    test_static_int_tuple()
    test_tuple_literal()
