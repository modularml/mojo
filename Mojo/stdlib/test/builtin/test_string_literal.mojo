# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from std.ffi import c_char

from std.testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)


def test_add() raises:
    assert_equal("five", StringLiteral.__add__("five", ""))
    assert_equal("six", StringLiteral.__add__("", "six"))
    assert_equal("fivesix", StringLiteral.__add__("five", "six"))


def test_mul() raises:
    comptime `3`: Int = 3
    comptime `u3`: UInt = 3
    comptime static_concat_0 = "mojo" * 3
    comptime static_concat_1 = "mojo" * `3`
    comptime static_concat_2 = "mojo" * Int(`u3`)
    assert_equal(static_concat_0, static_concat_1)
    assert_equal(static_concat_1, static_concat_2)
    assert_equal("mojomojomojo", static_concat_0)
    assert_equal(static_concat_0, "mojo" * 3)
    var dynamic_concat = "mojo" * 3
    assert_equal("mojomojomojo", dynamic_concat)
    assert_equal(static_concat_0, dynamic_concat)


def test_equality() raises:
    assert_false(StringLiteral.__eq__("five", "six"))
    assert_true(StringLiteral.__eq__("six", "six"))

    assert_true(StringLiteral.__ne__("five", "six"))
    assert_false(StringLiteral.__ne__("six", "six"))

    var hello = "hello"
    var hello_ref = StringSlice(hello)

    assert_false(StringLiteral.__eq__("goodbye", hello))
    assert_true(StringLiteral.__eq__("hello", hello))

    assert_false(StringLiteral.__eq__("goodbye", hello_ref))
    assert_true(StringLiteral.__eq__("hello", hello_ref))


def test_len() raises:
    assert_equal(0, "".byte_length())
    assert_equal(4, "four".byte_length())


def test_bool() raises:
    assert_true(StringLiteral.__bool__("not_empty"))
    assert_false(StringLiteral.__bool__(""))


def test_find() raises:
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


def test_rfind() raises:
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


def test_startswith() raises:
    var str = "Hello world"

    assert_true(str.startswith("Hello"))
    assert_false(str.startswith("Bye"))

    assert_true(str.startswith("llo", 2))
    assert_true(str.startswith("llo", 2, -1))
    assert_false(str.startswith("llo", 2, 3))


def test_endswith() raises:
    var str = "Hello world"

    assert_true(str.endswith(""))
    assert_true(str.endswith("world"))
    assert_true(str.endswith("ld"))
    assert_false(str.endswith("universe"))

    assert_true(str.endswith("ld", 2))
    assert_true(str.endswith("llo", 2, 5))
    assert_false(str.endswith("llo", 2, 3))


def test_comparison_operators() raises:
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

    # Test less than and greater than
    var def_slice = StringSlice("def")
    var abcd_slice = StringSlice("abc")
    assert_true(StringLiteral.__lt__("abc", def_slice))
    assert_false(StringLiteral.__lt__("def", abcd_slice[byte=0:3]))
    assert_false(StringLiteral.__lt__("abc", abcd_slice[byte=0:3]))
    assert_true(StringLiteral.__lt__("ab", abcd_slice[byte=0:3]))
    assert_true(StringLiteral.__gt__("abc", abcd_slice[byte=0:2]))
    assert_false(StringLiteral.__gt__("abc", abcd_slice))

    # Test less than or equal to and greater than or equal to
    assert_true(StringLiteral.__le__("abc", def_slice))
    assert_true(StringLiteral.__le__("abc", abcd_slice[byte=0:3]))
    assert_false(StringLiteral.__le__("def", abcd_slice[byte=0:3]))
    assert_true(StringLiteral.__ge__("abc", abcd_slice[byte=0:3]))
    assert_false(StringLiteral.__ge__("ab", abcd_slice[byte=0:3]))
    assert_true(StringLiteral.__ge__("abcd", abcd_slice[byte=0:3]))

    var abc_upper_slice = StringSlice("ABC")
    # Test case sensitivity in comparison (assuming ASCII order)
    assert_true(StringLiteral.__gt__("abc", abc_upper_slice))
    assert_false(StringLiteral.__le__("abc", abc_upper_slice))

    var empty_slice = StringSlice("")
    # Test comparisons involving empty strings
    assert_true(StringLiteral.__lt__("", abcd_slice[byte=0:3]))
    assert_false(StringLiteral.__lt__("abc", empty_slice))
    assert_true(StringLiteral.__le__("", empty_slice))
    assert_true(StringLiteral.__ge__("", empty_slice))


def test_indexing() raises:
    var s = "hello"
    assert_equal(s[byte=0], "h")
    assert_equal(s[byte=Int(1)], "e")
    assert_equal(s[byte=2], "l")


def test_intable() raises:
    assert_equal(StringLiteral.__int__("123"), 123)

    with assert_raises():
        _ = StringLiteral.__int__("hi")


def test_is_ascii_digit() raises:
    assert_true("123".is_ascii_digit())
    assert_false("abc".is_ascii_digit())
    assert_false("123abc".is_ascii_digit())
    # TODO: Uncomment this when PR3439 is merged
    # assert_false("".isdigit())


def test_islower() raises:
    assert_true("hello".islower())
    assert_false("Hello".islower())
    assert_false("HELLO".islower())
    assert_false("123".islower())
    assert_false("".islower())


def test_isupper() raises:
    assert_true("HELLO".isupper())
    assert_false("Hello".isupper())
    assert_false("hello".isupper())
    assert_false("123".isupper())
    assert_false("".isupper())


def test_iter() raises:
    # Test iterating over a string
    var i = 0
    for c in "one".codepoint_slices():
        if i == 0:
            assert_equal(c, "o")
        elif i == 1:
            assert_equal(c, "n")
        elif i == 2:
            assert_equal(c, "e")
        i += 1

    i = 0
    for c in "one".codepoints():
        if i == 0:
            assert_equal(c, Codepoint.ord("o"))
        elif i == 1:
            assert_equal(c, Codepoint.ord("n"))
        elif i == 2:
            assert_equal(c, Codepoint.ord("e"))
        i += 1


def test_layout() raises:
    # Test empty StringLiteral contents
    var empty = "".ptr()
    # An empty string literal is stored as just the NUL terminator.
    assert_true(Int(empty) != 0)
    # TODO(MSTDL-596): This seems to hang?
    # assert_equal(empty[0], 0)

    # Test non-empty StringLiteral C string
    var ptr = "hello".as_c_string_slice().ptr()
    assert_equal(ptr[unsafe_offset=0], Int8(ord("h")))
    assert_equal(ptr[unsafe_offset=1], Int8(ord("e")))
    assert_equal(ptr[unsafe_offset=2], Int8(ord("l")))
    assert_equal(ptr[unsafe_offset=3], Int8(ord("l")))
    assert_equal(ptr[unsafe_offset=4], Int8(ord("o")))
    assert_equal(ptr[unsafe_offset=5], 0)  # Verify NUL terminated


def test_lower_upper() raises:
    assert_equal("hello".lower(), "hello")
    assert_equal("HELLO".lower(), "hello")
    assert_equal("Hello".lower(), "hello")
    assert_equal("hello".upper(), "HELLO")
    assert_equal("HELLO".upper(), "HELLO")
    assert_equal("Hello".upper(), "HELLO")


def test_strip() raises:
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


def test_strip_mutable_chars() raises:
    # `chars` is read-only, so a mutable argument must not be borrowed mutably.
    var chars = String("_wrap_")
    assert_equal("_wrap_hello world_wrap_".lstrip(chars), "hello world_wrap_")
    assert_equal("_wrap_hello world_wrap_".rstrip(chars), "_wrap_hello world")
    assert_equal("_wrap_hello world_wrap_".strip(chars), "hello world")


def test_count() raises:
    var str = "Hello world"

    assert_equal(12, str.count(""))
    assert_equal(1, str.count("Hell"))
    assert_equal(3, str.count("l"))
    assert_equal(1, str.count("ll"))
    assert_equal(1, str.count("ld"))
    assert_equal(0, str.count("universe"))

    assert_equal(String("aaaaa").count("a"), 5)
    assert_equal(String("aaaaaa").count("aa"), 3)


def test_ascii_rjust() raises:
    assert_equal("hello".ascii_rjust(4), "hello")
    assert_equal("hello".ascii_rjust(8), "   hello")
    assert_equal("hello".ascii_rjust(8, "*"), "***hello")


def test_ascii_ljust() raises:
    assert_equal("hello".ascii_ljust(4), "hello")
    assert_equal("hello".ascii_ljust(8), "hello   ")
    assert_equal("hello".ascii_ljust(8, "*"), "hello***")


def test_center() raises:
    assert_equal("hello".ascii_center(4), "hello")
    assert_equal("hello".ascii_center(8), " hello  ")
    assert_equal("hello".ascii_center(8, "*"), "*hello**")


def test_float_conversion() raises:
    assert_equal(("4.5").__float__(), 4.5)
    assert_equal(Float64("4.5"), 4.5)
    with assert_raises():
        _ = ("not a float").__float__()


# If this compiles, then the format method does not raise.
def _test_format_does_not_raise():
    var _hello = "Hello, {}! I am {} years old.".format("world", 42)


def test_string_literal_codepoint_slices_reversed() raises:
    # Test ASCII
    var iter = "abc".codepoint_slices_reversed()
    assert_equal(iter.__next__(), "c")
    assert_equal(iter.__next__(), "b")
    assert_equal(iter.__next__(), "a")

    # Test concatenation
    var concat = String()
    for v in "abc".codepoint_slices_reversed():
        concat += v
    assert_equal(concat, "cba")

    # Test Unicode
    concat = String()
    for v in "test✅".codepoint_slices_reversed():
        concat += v
    assert_equal(concat, "✅tset")

    # Test empty string
    concat = String()
    for v in "".codepoint_slices_reversed():
        concat += v
    assert_equal(concat, "")


def test_unicode_escape_byte_layout() raises:
    # Verify that \u/\U escapes produce the correct UTF-8 byte layout when
    # embedded between ASCII characters. Checks: byte length, bytes
    # immediately before and after the escape, and all bytes of the escape.

    # Two-byte encoding (U+00E9, é): "abc" + C3 A9 + "def" = 8 bytes
    comptime s2 = "abc\u00E9def"
    assert_equal(s2.byte_length(), 8)
    var b2 = s2.as_bytes()
    assert_equal(Int(b2[2]), 0x63)  # 'c' before escape
    assert_equal(Int(b2[3]), 0xC3)  # first byte of U+00E9
    assert_equal(Int(b2[4]), 0xA9)  # second byte of U+00E9
    assert_equal(Int(b2[5]), 0x64)  # 'd' after escape

    # Three-byte encoding (U+4E2D, 中): "abc" + E4 B8 AD + "def" = 9 bytes
    comptime s3 = "abc\u4E2Ddef"
    assert_equal(s3.byte_length(), 9)
    var b3 = s3.as_bytes()
    assert_equal(Int(b3[2]), 0x63)  # 'c' before escape
    assert_equal(Int(b3[3]), 0xE4)  # first byte of U+4E2D
    assert_equal(Int(b3[4]), 0xB8)  # second byte of U+4E2D
    assert_equal(Int(b3[5]), 0xAD)  # third byte of U+4E2D
    assert_equal(Int(b3[6]), 0x64)  # 'd' after escape

    # Four-byte encoding (U+1F600, 😀): "abc" + F0 9F 98 80 + "def" = 10 bytes
    comptime s4 = "abc\U0001F600def"
    assert_equal(s4.byte_length(), 10)
    var b4 = s4.as_bytes()
    assert_equal(Int(b4[2]), 0x63)  # 'c' before escape
    assert_equal(Int(b4[3]), 0xF0)  # first byte of U+1F600
    assert_equal(Int(b4[4]), 0x9F)  # second byte of U+1F600
    assert_equal(Int(b4[5]), 0x98)  # third byte of U+1F600
    assert_equal(Int(b4[6]), 0x80)  # fourth byte of U+1F600
    assert_equal(Int(b4[7]), 0x64)  # 'd' after escape

    # Interleaved: "a" + 2-byte + "b" + 3-byte + "c" = 8 bytes
    comptime si = "a\u00E9b\u4E2Dc"
    assert_equal(si.byte_length(), 8)
    var bi = si.as_bytes()
    assert_equal(Int(bi[0]), 0x61)  # 'a'
    assert_equal(Int(bi[1]), 0xC3)  # first byte of U+00E9
    assert_equal(Int(bi[2]), 0xA9)  # second byte of U+00E9
    assert_equal(Int(bi[3]), 0x62)  # 'b' between escapes
    assert_equal(Int(bi[4]), 0xE4)  # first byte of U+4E2D
    assert_equal(Int(bi[5]), 0xB8)  # second byte of U+4E2D
    assert_equal(Int(bi[6]), 0xAD)  # third byte of U+4E2D
    assert_equal(Int(bi[7]), 0x63)  # 'c' at end


struct TakesStringLiteral[a: StringLiteral](TrivialRegisterPassable):
    def __init__(out self):
        pass


def concat[
    xa: StringLiteral
](x: TakesStringLiteral[xa]) -> TakesStringLiteral[xa + xa]:
    return {}


def test_concat_with_string_literal() raises:
    var a = TakesStringLiteral["hello"]()
    # This requires folding of the string append to know that "hello"+"hello" = "hellohello"
    _: TakesStringLiteral["hellohello"] = concat(a)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
