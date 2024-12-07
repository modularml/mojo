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

from sys.ffi import c_char

from memory import UnsafePointer
from testing import (
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_raises,
    assert_true,
)


def test_add():
    assert_equal("five", StringLiteral.__add__("five", ""))
    assert_equal("six", StringLiteral.__add__("", "six"))
    assert_equal("fivesix", StringLiteral.__add__("five", "six"))


fn add_literal(
    owned original: StringLiteral, add: StringLiteral, n: Int
) -> StringLiteral:
    for _ in range(n):
        original += add
    return original


def test_iadd():
    alias original = "mojo"
    alias concat = add_literal(original, "!", 4)
    assert_equal("mojo!!!!", concat)


def test_mul():
    alias `3`: Int = 3
    alias `u3`: UInt = 3
    alias static_concat_0 = "mojo" * 3
    alias static_concat_1 = "mojo" * `3`
    alias static_concat_2 = "mojo" * `u3`
    assert_equal(static_concat_0, static_concat_1)
    assert_equal(static_concat_1, static_concat_2)
    assert_equal("mojomojomojo", static_concat_0)
    assert_equal(static_concat_0, String("mojo") * 3)
    var dynamic_concat = "mojo" * 3
    assert_equal("mojomojomojo", dynamic_concat)
    assert_equal(static_concat_0, dynamic_concat)


def test_equality():
    assert_false(StringLiteral.__eq__("five", "six"))
    assert_true(StringLiteral.__eq__("six", "six"))

    assert_true(StringLiteral.__ne__("five", "six"))
    assert_false(StringLiteral.__ne__("six", "six"))

    var hello = str("hello")
    var hello_ref = hello.as_string_slice()

    assert_false(StringLiteral.__eq__("goodbye", hello))
    assert_true(StringLiteral.__eq__("hello", hello))

    assert_false(StringLiteral.__eq__("goodbye", hello_ref))
    assert_true(StringLiteral.__eq__("hello", hello_ref))


def test_len():
    assert_equal(0, StringLiteral.__len__(""))
    assert_equal(4, StringLiteral.__len__("four"))


def test_bool():
    assert_true(StringLiteral.__bool__("not_empty"))
    assert_false(StringLiteral.__bool__(""))


def test_contains():
    assert_true(StringLiteral.__contains__("abcde", "abc"))
    assert_true(StringLiteral.__contains__("abcde", "bc"))
    assert_false(StringLiteral.__contains__("abcde", "xy"))


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

    # Test that rfind(start) returned pos is absolute, not relative to specified
    # start. Also tests positive and negative start offsets.
    assert_equal("hello world".rfind("l", 5), 9)
    assert_equal("hello world".rfind("l", -5), 9)
    assert_equal("hello world".rfind("w", -3), -1)
    assert_equal("hello world".rfind("w", -5), 6)

    assert_equal(-1, "abc".rfind("abcd"))


def test_replace():
    assert_equal("".replace("", "hello world"), "")
    assert_equal("hello world".replace("", "something"), "hello world")
    assert_equal("hello world".replace("world", ""), "hello ")
    assert_equal("hello world".replace("world", "mojo"), "hello mojo")
    assert_equal(
        "hello world hello world".replace("world", "mojo"),
        "hello mojo hello mojo",
    )


def test_startswith():
    var str = "Hello world"

    assert_true(str.startswith("Hello"))
    assert_false(str.startswith("Bye"))

    assert_true(str.startswith("llo", 2))
    assert_true(str.startswith("llo", 2, -1))
    assert_false(str.startswith("llo", 2, 3))


def test_endswith():
    var str = "Hello world"

    assert_true(str.endswith(""))
    assert_true(str.endswith("world"))
    assert_true(str.endswith("ld"))
    assert_false(str.endswith("universe"))

    assert_true(str.endswith("ld", 2))
    assert_true(str.endswith("llo", 2, 5))
    assert_false(str.endswith("llo", 2, 3))


def test_comparison_operators():
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
    assert_not_equal(0, StringLiteral.__hash__("test"))
    assert_not_equal(StringLiteral.__hash__("a"), StringLiteral.__hash__("b"))
    assert_equal(StringLiteral.__hash__("a"), StringLiteral.__hash__("a"))
    assert_equal(StringLiteral.__hash__("b"), StringLiteral.__hash__("b"))


def test_indexing():
    var s = "hello"
    assert_equal(s[False], "h")
    assert_equal(s[int(1)], "e")
    assert_equal(s[2], "l")


def test_intable():
    assert_equal(StringLiteral.__int__("123"), 123)

    with assert_raises():
        _ = StringLiteral.__int__("hi")


def test_join():
    assert_equal("".join(), "")
    assert_equal("".join("a", "b", "c"), "abc")
    assert_equal(" ".join("a", "b", "c"), "a b c")
    assert_equal(" ".join("a", "b", "c", ""), "a b c ")
    assert_equal(" ".join("a", "b", "c", " "), "a b c  ")

    var sep = ","
    var s = String("abc")
    assert_equal(sep.join(s, s, s, s), "abc,abc,abc,abc")
    assert_equal(sep.join(1, 2, 3), "1,2,3")
    assert_equal(sep.join(1, "abc", 3), "1,abc,3")


def test_isdigit():
    assert_true("123".isdigit())
    assert_false("abc".isdigit())
    assert_false("123abc".isdigit())
    # TODO: Uncomment this when PR3439 is merged
    # assert_false("".isdigit())


def test_islower():
    assert_true("hello".islower())
    assert_false("Hello".islower())
    assert_false("HELLO".islower())
    assert_false("123".islower())
    assert_false("".islower())


def test_isupper():
    assert_true("HELLO".isupper())
    assert_false("Hello".isupper())
    assert_false("hello".isupper())
    assert_false("123".isupper())
    assert_false("".isupper())


def test_iter():
    # Test iterating over a string
    var s = "one"
    var i = 0
    for c in s:
        if i == 0:
            assert_equal(c, "o")
        elif i == 1:
            assert_equal(c, "n")
        elif i == 2:
            assert_equal(c, "e")


def test_layout():
    # Test empty StringLiteral contents
    var empty = "".unsafe_ptr()
    # An empty string literal is stored as just the NUL terminator.
    assert_true(int(empty) != 0)
    # TODO(MSTDL-596): This seems to hang?
    # assert_equal(empty[0], 0)

    # Test non-empty StringLiteral C string
    var ptr: UnsafePointer[c_char] = "hello".unsafe_cstr_ptr()
    assert_equal(ptr[0], ord("h"))
    assert_equal(ptr[1], ord("e"))
    assert_equal(ptr[2], ord("l"))
    assert_equal(ptr[3], ord("l"))
    assert_equal(ptr[4], ord("o"))
    assert_equal(ptr[5], 0)  # Verify NUL terminated


def test_lower_upper():
    assert_equal("hello".lower(), "hello")
    assert_equal("HELLO".lower(), "hello")
    assert_equal("Hello".lower(), "hello")
    assert_equal("hello".upper(), "HELLO")
    assert_equal("HELLO".upper(), "HELLO")
    assert_equal("Hello".upper(), "HELLO")


def test_repr():
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


def test_strip():
    assert_equal("".strip(), "")
    assert_equal("  ".strip(), "")
    assert_equal("  hello".strip(), "hello")
    assert_equal("hello  ".strip(), "hello")
    assert_equal("  hello  ".strip(), "hello")
    assert_equal("  hello  world  ".strip(" "), "hello  world")
    assert_equal("_wrap_hello world_wrap_".strip("_wrap_"), "hello world")
    assert_equal("  hello  world  ".strip("  "), "hello  world")
    assert_equal("  hello  world  ".lstrip(), "hello  world  ")
    assert_equal("  hello  world  ".rstrip(), "  hello  world")
    assert_equal(
        "_wrap_hello world_wrap_".lstrip("_wrap_"), "hello world_wrap_"
    )
    assert_equal(
        "_wrap_hello world_wrap_".rstrip("_wrap_"), "_wrap_hello world"
    )


def test_count():
    var str = "Hello world"

    assert_equal(12, str.count(""))
    assert_equal(1, str.count("Hell"))
    assert_equal(3, str.count("l"))
    assert_equal(1, str.count("ll"))
    assert_equal(1, str.count("ld"))
    assert_equal(0, str.count("universe"))

    assert_equal(String("aaaaa").count("a"), 5)
    assert_equal(String("aaaaaa").count("aa"), 3)


def test_rjust():
    assert_equal("hello".rjust(4), "hello")
    assert_equal("hello".rjust(8), "   hello")
    assert_equal("hello".rjust(8, "*"), "***hello")


def test_ljust():
    assert_equal("hello".ljust(4), "hello")
    assert_equal("hello".ljust(8), "hello   ")
    assert_equal("hello".ljust(8, "*"), "hello***")


def test_center():
    assert_equal("hello".center(4), "hello")
    assert_equal("hello".center(8), " hello  ")
    assert_equal("hello".center(8, "*"), "*hello**")


def test_split():
    var d = "hello world".split()
    assert_true(len(d) == 2)
    assert_true(d[0] == "hello")
    assert_true(d[1] == "world")
    d = "hello \t\n\n\v\fworld".split("\n")
    assert_true(len(d) == 3)
    assert_true(d[0] == "hello \t" and d[1] == "" and d[2] == "\v\fworld")

    # should split into empty strings between separators
    d = "1,,,3".split(",")
    assert_true(len(d) == 4)
    assert_true(d[0] == "1" and d[1] == "" and d[2] == "" and d[3] == "3")
    d = "abababaaba".split("aba")
    assert_true(len(d) == 4)
    assert_true(d[0] == "" and d[1] == "b" and d[2] == "" and d[3] == "")

    # should split into maxsplit + 1 items
    d = "1,2,3".split(",", 0)
    assert_true(len(d) == 1)
    assert_true(d[0] == "1,2,3")
    d = "1,2,3".split(",", 1)
    assert_true(len(d) == 2)
    assert_true(d[0] == "1" and d[1] == "2,3")

    assert_true(len("".split()) == 0)
    assert_true(len(" ".split()) == 0)
    assert_true(len("".split(" ")) == 1)
    assert_true(len(" ".split(" ")) == 2)
    assert_true(len("  ".split(" ")) == 3)
    assert_true(len("   ".split(" ")) == 4)

    with assert_raises():
        _ = "".split("")

    # Matches should be properly split in multiple case
    var d2 = " "
    var in2 = "modcon is coming soon"
    var res2 = in2.split(d2)
    assert_equal(len(res2), 4)
    assert_equal(res2[0], "modcon")
    assert_equal(res2[1], "is")
    assert_equal(res2[2], "coming")
    assert_equal(res2[3], "soon")

    # No match from the delimiter
    var d3 = "x"
    var in3 = "hello world"
    var res3 = in3.split(d3)
    assert_equal(len(res3), 1)
    assert_equal(res3[0], "hello world")

    # Multiple character delimiter
    var d4 = "ll"
    var in4 = "hello"
    var res4 = in4.split(d4)
    assert_equal(len(res4), 2)
    assert_equal(res4[0], "he")
    assert_equal(res4[1], "o")


def test_splitlines():
    alias L = List[String]
    # Test with no line breaks
    assert_equal("hello world".splitlines(), L("hello world"))

    # Test with line breaks
    assert_equal("hello\nworld".splitlines(), L("hello", "world"))
    assert_equal("hello\rworld".splitlines(), L("hello", "world"))
    assert_equal("hello\r\nworld".splitlines(), L("hello", "world"))

    # Test with multiple different line breaks
    s1 = "hello\nworld\r\nmojo\rlanguage\r\n"
    hello_mojo = L("hello", "world", "mojo", "language")
    assert_equal(s1.splitlines(), hello_mojo)
    assert_equal(
        s1.splitlines(keepends=True),
        L("hello\n", "world\r\n", "mojo\r", "language\r\n"),
    )

    # Test with an empty string
    assert_equal("".splitlines(), L())
    # test \v \f \x1c \x1d
    s2 = "hello\vworld\fmojo\x1clanguage\x1d"
    assert_equal(s2.splitlines(), hello_mojo)
    assert_equal(
        s2.splitlines(keepends=True),
        L("hello\v", "world\f", "mojo\x1c", "language\x1d"),
    )

    # test \x1c \x1d \x1e
    s3 = "hello\x1cworld\x1dmojo\x1elanguage\x1e"
    assert_equal(s3.splitlines(), hello_mojo)
    assert_equal(
        s3.splitlines(keepends=True),
        L("hello\x1c", "world\x1d", "mojo\x1e", "language\x1e"),
    )


def test_float_conversion():
    assert_equal(("4.5").__float__(), 4.5)
    assert_equal(float("4.5"), 4.5)
    with assert_raises():
        _ = ("not a float").__float__()


def test_string_literal_from_stringable():
    assert_equal(StringLiteral.get["hello"](), "hello")
    assert_equal(StringLiteral.get[String("hello")](), "hello")
    assert_equal(StringLiteral.get[42](), "42")
    assert_equal(
        StringLiteral.get[SIMD[DType.int64, 4](1, 2, 3, 4)](), "[1, 2, 3, 4]"
    )


def main():
    test_add()
    test_iadd()
    test_mul()
    test_equality()
    test_len()
    test_bool()
    test_contains()
    test_find()
    test_join()
    test_rfind()
    test_replace()
    test_comparison_operators()
    test_count()
    test_hash()
    test_indexing()
    test_intable()
    test_isdigit()
    test_islower()
    test_isupper()
    test_layout()
    test_lower_upper()
    test_repr()
    test_rjust()
    test_ljust()
    test_center()
    test_startswith()
    test_endswith()
    test_strip()
    test_split()
    test_splitlines()
    test_float_conversion()
    test_string_literal_from_stringable()
