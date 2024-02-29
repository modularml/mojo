# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from math import divmod, max, min

from testing import *


def test_constructors():
    var i1 = Int(3)  # Constructible from IntLiteral
    var i2 = Int(Int(5))  # Constructible from Int


def test_add():
    assert_equal(6, Int(3) + Int(3))


def test_sub():
    assert_equal(3, Int(4) - Int(1))
    assert_equal(5, Int(6) - Int(1))


def test_div():
    var n = Int(5)
    var d = Int(2)
    assert_equal(2.5, n / d)
    n /= d
    assert_equal(2, n)


def test_pow():
    assert_equal(1, Int(3) ** Int(0))
    assert_equal(27, Int(3) ** Int(3))
    assert_equal(81, Int(3) ** Int(4))


def test_int():
    var a = 0
    var b = a + Int(1)
    assert_equal(a, min(a, b))
    assert_equal(b, max(a, b))


def test_floordiv():
    assert_equal(1, Int(2) // Int(2))
    assert_equal(0, Int(2) // Int(3))
    assert_equal(-1, Int(2) // Int(-2))
    assert_equal(-50, Int(99) // Int(-2))
    assert_equal(-1, Int(-1) // Int(10))


def test_mod():
    assert_equal(0, Int(99) % Int(1))
    assert_equal(0, Int(99) % Int(3))
    assert_equal(-1, Int(99) % Int(-2))
    assert_equal(3, Int(99) % Int(8))
    assert_equal(-5, Int(99) % Int(-8))
    assert_equal(0, Int(2) % Int(-1))
    assert_equal(0, Int(2) % Int(-2))
    assert_equal(-1, Int(3) % Int(-2))
    assert_equal(1, Int(-3) % Int(2))


def test_divmod():
    assert_equal(StaticIntTuple[2](-2, 1), divmod(-3, 2))


def main():
    test_constructors()
    test_add()
    test_sub()
    test_div()
    test_pow()
    test_int()
    test_floordiv()
    test_mod()
    test_divmod()
