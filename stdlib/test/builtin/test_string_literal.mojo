# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from testing import assert_equal, assert_false, assert_not_equal, assert_true


def test_basics():
    assert_equal(4, len("four"))
    assert_equal("fivesix", "five" + "six")
    assert_not_equal("five", "six")
    assert_equal("five", "five")

    assert_true("not_empty")
    assert_false("")


def test_contains():
    assert_true("abc" in "abcde")
    assert_true("bc" in "abcde")
    assert_true("xy" not in "abcde")


def test_find():
    assert_equal(0, "Hello world".find(""))
    assert_equal(0, "Hello world".find("Hello"))
    assert_equal(2, "Hello world".find("llo"))
    assert_equal(6, "Hello world".find("world"))
    assert_equal(-1, "Hello world".find("universe"))

    assert_equal(-1, "abc".find("abcd"))


def test_rfind():
    # Basic usage.
    assert_equal("hello world".rfind("world"), 6)
    assert_equal("hello world".rfind("bye"), -1)

    # Repeated substrings.
    assert_equal("ababab".rfind("ab"), 4)

    # Empty string and substring.
    assert_equal("".rfind("ab"), -1)
    assert_equal("foo".rfind(""), 3)

    assert_equal(-1, "abc".rfind("abcd"))


def test_hash():
    # Test a couple basic hash behaviors.
    # `test_hash.test_hash_bytes` has more comprehensive tests.
    assert_not_equal(0, hash("test"))
    assert_not_equal(hash("a"), hash("b"))
    assert_equal(hash("a"), hash("a"))
    assert_equal(hash("b"), hash("b"))


def main():
    test_basics()
    test_contains()
    test_find()
    test_rfind()
    test_hash()
