# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from testing import (
    assert_equal,
    assert_not_equal,
    assert_true,
    assert_false,
    assert_raises,
)


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

    assert_equal(3, "...a".find("a", 0))
    assert_equal(3, "...a".find("a", 1))
    assert_equal(3, "...a".find("a", 2))
    assert_equal(3, "...a".find("a", 3))

    # Test find() support for negative start positions
    assert_equal(4, "Hello world".find("o", -10))
    assert_equal(7, "Hello world".find("o", -5))

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

    # Test that rfind(start) returned pos is absolute, not relative to specifed
    # start. Also tests positive and negative start offsets.
    assert_equal("hello world".rfind("l", 5), 9)
    assert_equal("hello world".rfind("l", -5), 9)
    assert_equal("hello world".rfind("w", -3), -1)
    assert_equal("hello world".rfind("w", -5), 6)

    assert_equal(-1, "abc".rfind("abcd"))


fn test_comparison_operators() raises:
    # Test less than and greater than
    assert_true(StringLiteral.__lt__("abc", "def"))
    assert_false(StringLiteral.__lt__("def", "abc"))
    assert_false(StringLiteral.__lt__("abc", "abc"))
    assert_true(StringLiteral.__lt__("ab", "abc"))
    assert_true(StringLiteral.__gt__("abc", "ab"))
    assert_false(StringLiteral.__gt__("abc", "abcd"))

    # Test less than or equal to and greater than or equal to
    assert_true(StringLiteral.__le__("abc", "def"))
    assert_true(StringLiteral.__le__("abc", "abc"))
    assert_false(StringLiteral.__le__("def", "abc"))
    assert_true(StringLiteral.__ge__("abc", "abc"))
    assert_false(StringLiteral.__ge__("ab", "abc"))
    assert_true(StringLiteral.__ge__("abcd", "abc"))

    # Test case sensitivity in comparison (assuming ASCII order)
    assert_true(StringLiteral.__gt__("abc", "ABC"))
    assert_false(StringLiteral.__le__("abc", "ABC"))

    # Test comparisons involving empty strings
    assert_true(StringLiteral.__lt__("", "abc"))
    assert_false(StringLiteral.__lt__("abc", ""))
    assert_true(StringLiteral.__le__("", ""))
    assert_true(StringLiteral.__ge__("", ""))


def test_hash():
    # Test a couple basic hash behaviors.
    # `test_hash.test_hash_bytes` has more comprehensive tests.
    assert_not_equal(0, hash("test"))
    assert_not_equal(hash("a"), hash("b"))
    assert_equal(hash("a"), hash("a"))
    assert_equal(hash("b"), hash("b"))


def test_intable():
    assert_equal(int("123"), 123)

    with assert_raises():
        _ = int("hi")


fn test_repr() raises:
    # Usual cases
    assert_equal(StringLiteral.__repr__("hello"), "'hello'")

    # Escape cases
    assert_equal(StringLiteral.__repr__("\0"), r"'\x00'")
    assert_equal(StringLiteral.__repr__("\x06"), r"'\x06'")
    assert_equal(StringLiteral.__repr__("\x09"), r"'\t'")
    assert_equal(StringLiteral.__repr__("\n"), r"'\n'")
    assert_equal(StringLiteral.__repr__("\x0d"), r"'\r'")
    assert_equal(StringLiteral.__repr__("\x0e"), r"'\x0e'")
    assert_equal(StringLiteral.__repr__("\x1f"), r"'\x1f'")
    assert_equal(StringLiteral.__repr__(" "), "' '")
    assert_equal(StringLiteral.__repr__("'"), '"\'"')
    assert_equal(StringLiteral.__repr__("A"), "'A'")
    assert_equal(StringLiteral.__repr__("\\"), r"'\\'")
    assert_equal(StringLiteral.__repr__("~"), "'~'")
    assert_equal(StringLiteral.__repr__("\x7f"), r"'\x7f'")


def test_format_args():
    with assert_raises(contains="Index 1 not in *args"):
        print("A {0} B {1}".format("First"))

    with assert_raises(
        contains="Automatic indexing require more args in *args"
    ):
        print("A {} B {}".format("First"))

    with assert_raises(
        contains="Cannot both use manual and automatic indexing for *args"
    ):
        print("A {} B {1}".format("First", "Second"))

    with assert_raises(contains="Index second not in kwargs"):
        print("A {first} B {second}".format(first="A"))

    assert_equal(
        "A {} B {First} {Second} {} {Third} {} C".format(
            "Hello",
            "World",
            "🔥",
            First=str(True),
            Second=str(1.125),
            Third=str(123),
        ),
        "A Hello B True 1.125 World 123 🔥 C",
    )

    assert_equal(
        "{0} {Second} {First} {1} {Second} {0}".format(
            "🔥",
            "Mojo",
            First="Love",
            Second="❤️‍🔥",
        ),
        "🔥 ❤️‍🔥 Love Mojo ❤️‍🔥 🔥",
    )

    assert_equal("{0} {1}".format("🔥", "Mojo"), "🔥 Mojo")

    assert_equal("{0} {1}".format("{1}", "Mojo"), "{1} Mojo")

    # Does not work in the parameter domain
    # alias A = "Love"
    # alias B = String("❤️‍🔥")
    # alias C = "🔥"
    # alias D = String("Mojo")
    # alias Result = "{0} {Second} {First} {1} {Second} {0}".format(
    #    C,
    #    D,
    #    First=A,
    #    Second=B
    # )
    # @parameter
    # if Result != "🔥 ❤️‍🔥 Love Mojo ❤️‍🔥 🔥":
    #    raise "Assertion failed (alias): " + Result


def main():
    test_basics()
    test_contains()
    test_find()
    test_rfind()
    test_comparison_operators()
    test_hash()
    test_intable()
    test_repr()
    test_format_args()
