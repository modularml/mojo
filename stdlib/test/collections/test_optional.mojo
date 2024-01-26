# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from collections.optional import Optional

from testing import *


def test_basic():
    let a = Optional(1)
    let b = Optional[Int](None)

    assert_true(a.__bool__())
    assert_false(b.__bool__())

    assert_equal(1, a.value())

    # TODO(27776): can't inline these, they need to be mutable lvalues
    var a1 = a.or_else(2)
    var b1 = b.or_else(2)

    assert_equal(1, a1.value())
    assert_equal(2, b1.value())

    assert_equal(1, (a ^).take())


def main():
    test_basic()
