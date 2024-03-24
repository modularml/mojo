# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo --debug-level full %s

from testing import assert_equal

from utils.inlined_string import _ArrayMem


def main():
    test_array_mem()


def test_array_mem():
    var array = _ArrayMem[Int, 4](1)

    assert_equal(array.SIZE, 4)
    assert_equal(len(array), 4)

    # ==================================
    # Test pointer operations
    # ==================================

    var ptr = array.as_ptr()
    assert_equal(ptr[0], 1)
    assert_equal(ptr[1], 1)
    assert_equal(ptr[2], 1)
    assert_equal(ptr[3], 1)
