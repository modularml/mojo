# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from utils._optional import Optional
from testing import *


def main():
    print("== test_optional")

    var val: Optional[Int] = None
    assert_false(val.__bool__())

    val = 15
    assert_true(val.__bool__())

    assert_equal(val.value(), 15)

    assert_true(val or Bool(False))
    assert_true(val and Bool(True))

    assert_true(Bool(False) or val)
    assert_true(Bool(True) and val)
