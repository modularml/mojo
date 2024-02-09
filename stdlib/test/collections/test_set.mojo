# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from collections.set import Set
from collections.dict import EqualityComparable

from testing import *


fn assert_equal[T: EqualityComparable](lhs: T, rhs: T) raises:
    if not lhs == rhs:
        raise Error("AssertionError: values not equal, can't stringify :(")


def test_set_construction():
    _ = Set[Int]()
    _ = Set[String]()
    _ = Set[Int](1, 2, 3)
    _ = Set(Set[Int](1, 2, 3))


def test_len():
    let s1 = Set[Int]()
    assert_equal(0, len(s1))

    let s2 = Set[Int](1, 2, 3)
    assert_equal(3, len(s2))


def test_in():
    let s1 = Set[Int]()
    assert_false(0 in s1)
    assert_false(1 in s1)

    let s2 = Set[Int](1, 2, 3)
    assert_false(0 in s2)
    assert_true(1 in s2)
    assert_true(2 in s2)
    assert_true(3 in s2)
    assert_false(4 in s2)


def test_equal():
    let s1 = Set[Int]()
    let s2 = Set[Int](1, 2, 3)

    assert_true(s1 == s1)
    assert_true(s2 == s2)
    assert_true(s1 == Set[Int]())
    assert_true(s2 == Set[Int](3, 2, 1))
    assert_false(s1 == s2)
    assert_false(s2 == Set[Int](1, 2, 2))
    assert_false(s2 == Set[Int](1, 2, 4))


def test_bool():
    assert_false(Set[Int]())
    assert_false(Set[Int](DynamicVector[Int]()))
    assert_true(Set[Int](1))
    assert_true(Set[Int](1, 2, 3))


def test_intersection():
    assert_equal(Set[Int]() & Set[Int](), Set[Int]())
    assert_equal(Set[Int]() & Set[Int](1, 2, 3), Set[Int]())
    assert_equal(Set[Int](1, 2, 3) & Set[Int](1, 2, 3), Set[Int](1, 2, 3))
    assert_equal(Set[Int](1, 2, 3) & Set[Int](), Set[Int]())
    assert_equal(Set[Int](1, 2, 3) & Set[Int](3, 4), Set[Int](3))

    assert_equal(Set[Int]().intersection(Set[Int]()), Set[Int]())
    assert_equal(Set[Int]().intersection(Set[Int](1, 2, 3)), Set[Int]())
    assert_equal(
        Set[Int](1, 2, 3).intersection(Set[Int](1, 2, 3)), Set[Int](1, 2, 3)
    )
    assert_equal(Set[Int](1, 2, 3).intersection(Set[Int]()), Set[Int]())
    assert_equal(Set[Int](1, 2, 3).intersection(Set[Int](3, 4)), Set[Int](3))

    var x = Set[Int]()
    x &= Set[Int](1, 2, 3)
    assert_equal(x, Set[Int]())

    x = Set[Int]()
    x &= Set[Int]()
    assert_equal(x, Set[Int]())

    x = Set[Int](1, 2, 3)
    x &= Set[Int]()
    assert_equal(x, Set[Int]())

    x = Set[Int](1, 2, 3)
    x &= Set[Int](1, 2, 3)
    assert_equal(x, Set[Int](1, 2, 3))

    x = Set[Int](1, 2)
    x &= Set[Int](2, 3)
    assert_equal(x, Set[Int](2))


def test_union():
    assert_equal(Set[Int]() | Set[Int](), Set[Int]())
    assert_equal(Set[Int]() | Set[Int](1, 2, 3), Set[Int](1, 2, 3))
    assert_equal(Set[Int](1, 2, 3) | Set[Int](1, 2, 3), Set[Int](1, 2, 3))
    assert_equal(Set[Int](1, 2, 3) | Set[Int](), Set[Int](1, 2, 3))
    assert_equal(Set[Int](1, 2, 3) | Set[Int](3, 4), Set[Int](1, 2, 3, 4))

    assert_equal(Set[Int]().union(Set[Int]()), Set[Int]())
    assert_equal(Set[Int]().union(Set[Int](1, 2, 3)), Set[Int](1, 2, 3))
    assert_equal(Set[Int](1, 2, 3).union(Set[Int](1, 2, 3)), Set[Int](1, 2, 3))
    assert_equal(Set[Int](1, 2, 3).union(Set[Int]()), Set[Int](1, 2, 3))
    assert_equal(Set[Int](1, 2, 3).union(Set[Int](3, 4)), Set[Int](1, 2, 3, 4))

    var x = Set[Int]()
    x |= Set[Int](1, 2, 3)
    assert_equal(x, Set[Int](1, 2, 3))

    x = Set[Int]()
    x |= Set[Int]()
    assert_equal(x, Set[Int]())

    x = Set[Int](1, 2, 3)
    x |= Set[Int]()
    assert_equal(x, Set[Int](1, 2, 3))

    x = Set[Int](1, 2, 3)
    x |= Set[Int](1, 2, 3)
    assert_equal(x, Set[Int](1, 2, 3))

    x = Set[Int](1, 2)
    x |= Set[Int](2, 3)
    assert_equal(x, Set[Int](1, 2, 3))


def test_subtract():
    let s1 = Set[Int]()
    let s2 = Set[Int](1, 2, 3)

    assert_equal(s1 - s1, s1)
    assert_equal(s1 - s2, s1)
    assert_equal(s2 - s2, s1)
    assert_equal(s2 - s1, s2)
    assert_equal(s2 - Set[Int](3, 4), Set[Int](1, 2))


def test_remove_all():
    var x = Set[Int]()
    x.remove_all(Set[Int]())
    assert_equal(x, Set[Int]())

    x = Set[Int](1, 2, 3)
    x.remove_all(Set[Int](1, 2, 3))
    assert_equal(x, Set[Int]())

    x = Set[Int](1, 2, 3)
    x.remove_all(Set[Int]())
    assert_equal(x, Set[Int](1, 2, 3))

    x = Set[Int](1, 2, 3)
    x.remove_all(Set[Int](3, 4))
    assert_equal(x, Set[Int](1, 2))

    x = Set[Int]()
    x -= Set[Int]()
    assert_equal(x, Set[Int]())

    x = Set[Int](1, 2, 3)
    x -= Set[Int](1, 2, 3)
    assert_equal(x, Set[Int]())

    x = Set[Int](1, 2, 3)
    x -= Set[Int]()
    assert_equal(x, Set[Int](1, 2, 3))

    x = Set[Int](1, 2, 3)
    x -= Set[Int](3, 4)
    assert_equal(x, Set[Int](1, 2))


def test_iter():
    var sum = 0
    for e in Set[Int]():
        sum += e[]

    assert_equal(sum, 0)

    sum = 0
    for e in Set[Int](1, 2, 3):
        sum += e[]

    assert_equal(sum, 6)


def test_add():
    var s = Set[Int]()
    s.add(1)
    assert_equal(s, Set[Int](1))

    s.add(2)
    assert_equal(s, Set[Int](1, 2))

    s.add(3)
    assert_equal(s, Set[Int](1, 2, 3))

    # 1 is already in the set
    s.add(1)
    assert_equal(s, Set[Int](1, 2, 3))


def test_remove():
    var s = Set[Int](1, 2, 3)
    s.remove(1)
    assert_equal(s, Set[Int](2, 3))

    s.remove(2)
    assert_equal(s, Set[Int](3))

    s.remove(3)
    assert_equal(s, Set[Int]())

    with assert_raises():
        # 1 not in the set, should raise
        s.remove(1)


def test_pop_insertion_order():
    var s = Set[Int](1, 2, 3)
    assert_equal(s.pop(), 1)
    assert_equal(s, Set[Int](2, 3))

    s.add(4)

    assert_equal(s.pop(), 2)
    assert_equal(s, Set[Int](3, 4))

    assert_equal(s.pop(), 3)
    assert_equal(s, Set[Int](4))

    assert_equal(s.pop(), 4)
    assert_equal(s, Set[Int]())

    with assert_raises():
        s.pop()  # pop from empty set raises


fn test[name: String, test_fn: fn () raises -> object]() raises:
    var name_val = name  # FIXME(#26974): Can't pass 'name' directly.
    print_no_newline("Test", name_val, "...")
    try:
        _ = test_fn()
    except e:
        print("FAIL")
        raise e
    print("PASS")


def main():
    test["test_set_construction", test_set_construction]()
    test["test_len", test_len]()
    test["test_in", test_in]()
    test["test_equal", test_equal]()
    test["test_bool", test_bool]()
    test["test_intersection", test_intersection]()
    test["test_union", test_union]()
    test["test_subtract", test_subtract]()
    test["test_remove_all", test_remove_all]()
    test["test_iter", test_iter]()
    test["test_add", test_add]()
    test["test_remove", test_remove]()
    test["test_pop_insertion_order", test_pop_insertion_order]()
