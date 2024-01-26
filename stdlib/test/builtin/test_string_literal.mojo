# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from testing import (
    assert_equal,
    assert_not_equal,
    assert_false,
    assert_true,
)


def test_hash():
    # Test a couple basic hash behaviors.
    # `test_hash.test_hash_bytes` has more comprehensive tests.
    assert_not_equal(0, hash("test"))
    assert_not_equal(hash("a"), hash("b"))
    assert_equal(hash("a"), hash("a"))
    assert_equal(hash("b"), hash("b"))


def main():
    assert_equal(4, len("four"))
    assert_equal("fivesix", "five" + "six")
    assert_not_equal("five", "six")
    assert_equal("five", "five")

    assert_true("not_empty")
    assert_false("")

    test_hash()
