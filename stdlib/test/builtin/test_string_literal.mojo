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

from sys.ffi import C_char

from testing import (
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_raises,
    assert_true,
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


def test_layout():
    #
    # Test empty StringLiteral contents
    #

    var empty = "".unsafe_ptr()

    # An empty string literal is stored as just the NUL terminator.
    assert_true(int(empty) != 0)
    # TODO(MSTDL-596): This seems to hang?
    # assert_equal(empty[0], 0)

    #
    # Test non-empty StringLiteral C string
    #

    var ptr: UnsafePointer[C_char] = "hello".unsafe_cstr_ptr()

    assert_equal(ptr[0], ord("h"))
    assert_equal(ptr[1], ord("e"))
    assert_equal(ptr[2], ord("l"))
    assert_equal(ptr[3], ord("l"))
    assert_equal(ptr[4], ord("o"))
    assert_equal(ptr[5], 0)  # Verify NUL terminated


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


def main():
    test_basics()
    test_contains()
    test_find()
    test_rfind()
    test_comparison_operators()
    test_hash()
    test_intable()
    test_layout()
    test_repr()
