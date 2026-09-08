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

from std.collections.string import (
    ImmStringSpan,
    ImmStringSlice,
    MutStringSpan,
    MutStringSlice,
)
from std.collections.string.string_span import (
    _to_string_list,
    get_static_string,
)
from std.sys.info import size_of, simd_width_of

from std.testing import assert_equal, assert_false, assert_true, assert_raises
from std.testing import TestSuite

# ===----------------------------------------------------------------------=== #
# Reusable testing data
# ===----------------------------------------------------------------------=== #

comptime EVERY_CODEPOINT_LENGTH_STR = StringSlice("߷കൈ🔄!")
"""A string that contains at least one of 1-, 2-, 3-, and 4-byte UTF-8
sequences.

Visualized as:

```text
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                      ߷കൈ🔄!                    ┃
┣━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━┫
┃   ߷  ┃     ക     ┃     ൈ    ┃       🔄      ┃! ┃
┣━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━━━━━━╋━━┫
┃ 2039  ┃   3349    ┃   3400    ┃    128260     ┃33┃
┣━━━┳━━━╋━━━┳━━━┳━━━╋━━━┳━━━┳━━━╋━━━┳━━━┳━━━┳━━━╋━━┫
┃223┃183┃224┃180┃149┃224┃181┃136┃240┃159┃148┃132┃33┃
┗━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━┛
  0   1   2   3   4   5   6   7   8   9  10  11  12
```

For further visualization and analysis involving this sequence, see:
<https://connorgray.com/ephemera/project-log#2025-01-13>.
"""

# ===----------------------------------------------------------------------=== #
# Tests
# ===----------------------------------------------------------------------=== #


def test_string_span_layout() raises:
    # Test that the layout of `StringSpan` is the same as `llvm::StringRef`.
    # This is necessary for `StringSpan` to be validly bitcasted to and from
    # `llvm::StringRef`

    # StringSpan should be two words in size.
    assert_equal(size_of[StringSpan[MutAnyOrigin]](), 2 * size_of[Int]())
    assert_equal(
        size_of[StringSlice[MutAnyOrigin]](),
        size_of[StringSpan[MutAnyOrigin]](),
    )

    var string_span = StringSpan("")

    var base_ptr = Int(Pointer(to=string_span))
    var first_word_ptr = Int(Pointer(to=string_span._slice._data))
    var second_word_ptr = Int(Pointer(to=string_span._slice._len))

    # 1st field should be at 0-byte offset from base ptr
    assert_equal(first_word_ptr - base_ptr, 0)
    # 2nd field should at 1-word offset from base ptr
    assert_equal(second_word_ptr - base_ptr, size_of[Int]())


def test_constructors() raises:
    def some_func_immut(b: StringSlice[mut=False, ...]) raises:
        assert_false(b.mut)

    def some_func_mut(b: StringSlice[mut=True, ...]) raises:
        assert_true(b.mut)

    var a = "123"
    some_func_immut(a)
    some_func_mut(StringSlice(a))


def test_string_slice_from_string_static_repr() raises:
    """Tests special case of StringSlice from String pointing to static data."""

    def is_static_string(s: String) -> Bool:
        return not s._is_inline() and not s._is_ref_counted()

    var s1 = String("foo")
    assert_true(is_static_string(s1))

    # Borrowing as immutable doesn't change `s1`
    var str1 = StringSlice[mut=False, ...](s1)
    assert_equal(str1, "foo")
    assert_true(is_static_string(s1))

    # Borrowing as mutable changes `s1` to a different representation
    var str2 = StringSlice[mut=True, ...](s1)
    assert_equal(str2, "foo")
    assert_false(is_static_string(s1))


def test_string_literal_byte_span() raises:
    comptime slc = "Hello".as_bytes()

    assert_equal(len(slc), 5)
    assert_equal(slc[0], Byte(ord("H")))
    assert_equal(slc[1], Byte(ord("e")))
    assert_equal(slc[2], Byte(ord("l")))
    assert_equal(slc[3], Byte(ord("l")))
    assert_equal(slc[4], Byte(ord("o")))


def test_string_byte_span() raises:
    var string = "Hello"
    var str_slice = string.unsafe_as_bytes_mut()

    assert_equal(len(str_slice), 5)
    assert_equal(str_slice[0], Byte(ord("H")))
    assert_equal(str_slice[1], Byte(ord("e")))
    assert_equal(str_slice[2], Byte(ord("l")))
    assert_equal(str_slice[3], Byte(ord("l")))
    assert_equal(str_slice[4], Byte(ord("o")))

    # ----------------------------------
    # Test subslicing
    # ----------------------------------

    # Slice the whole thing
    var sub1 = str_slice[:5]
    assert_equal(len(sub1), 5)
    assert_equal(sub1[0], Byte(ord("H")))
    assert_equal(sub1[1], Byte(ord("e")))
    assert_equal(sub1[2], Byte(ord("l")))
    assert_equal(sub1[3], Byte(ord("l")))
    assert_equal(sub1[4], Byte(ord("o")))

    # Slice the end
    var sub2 = str_slice[2:5]
    assert_equal(len(sub2), 3)
    assert_equal(sub2[0], Byte(ord("l")))
    assert_equal(sub2[1], Byte(ord("l")))
    assert_equal(sub2[2], Byte(ord("o")))

    # Slice the first element
    var sub3 = str_slice[0:1]
    assert_equal(len(sub3), 1)
    assert_equal(sub3[0], Byte(ord("H")))

    #
    # Test mutation through slice
    #

    sub1[0] = Byte(ord("J"))
    assert_equal(string, "Jello")

    sub2[2] = Byte(ord("y"))
    assert_equal(string, "Jelly")

    # ----------------------------------
    # Test empty subslicing
    # ----------------------------------

    var sub4 = str_slice[0:0]
    assert_equal(len(sub4), 0)

    var sub5 = str_slice[2:2]
    assert_equal(len(sub5), 0)

    # Empty slices still have a pointer value
    assert_equal(Int(sub5.unsafe_ptr()) - Int(sub4.unsafe_ptr()), 2)

    # ----------------------------------
    # Test out of range slicing
    # ----------------------------------

    # TODO: Improve error reporting for invalid slice bounds.

    # assert_equal(
    #     # str_slice[3:6]
    #     str_slice._try_slice(slice(3, 6)).unwrap[String](),
    #     String("Slice end is out of bounds"),
    # )

    # assert_equal(
    #     # str_slice[5:6]
    #     str_slice._try_slice(slice(5, 6)).unwrap[String](),
    #     String("Slice start is out of bounds"),
    # )

    # assert_equal(
    #     # str_slice[5:5]
    #     str_slice._try_slice(slice(5, 5)).unwrap[String](),
    #     String("Slice start is out of bounds"),
    # )

    # --------------------------------------------------------
    # Test that malformed partial slicing of codepoints raises
    # --------------------------------------------------------

    # These test what happens if you try to subslice a string in a way that
    # would leave the byte contents of the string containing partial encoded
    # codepoint sequences, invalid UTF-8. Consider a string with the following
    # content, containing both 1-byte and a 4-byte UTF-8 sequence:
    #
    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃          Hi👋!          ┃ String
    # ┣━━┳━━━┳━━━━━━━━━━━━━━━┳━━┫
    # ┃H ┃ i ┃       👋      ┃! ┃ Codepoint Characters
    # ┣━━╋━━━╋━━━━━━━━━━━━━━━╋━━┫
    # ┃72┃105┃    128075     ┃33┃ Codepoints
    # ┣━━╋━━━╋━━━┳━━━┳━━━┳━━━╋━━┫
    # ┃72┃105┃240┃159┃145┃139┃33┃ Bytes
    # ┗━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━┛
    #  0   1   2   3   4   5   6
    var unicode_str1 = StringSlice("Hi👋!")

    # Test slicing 0:{0–7}
    assert_equal(unicode_str1[byte=0:0], "")
    assert_equal(unicode_str1[byte=0:1], "H")
    assert_equal(unicode_str1[byte=0:2], "Hi")
    assert_equal(unicode_str1[byte=0:6], "Hi👋")
    assert_equal(unicode_str1[byte=0:7], "Hi👋!")

    # -------------------------------------------------------------------
    # Test that slicing through combining codepoint graphemes is allowed
    # -------------------------------------------------------------------

    # The "ö" is a user-perceived character (grapheme) that is composed of two
    # codepoints. This test tests that we can use string slicing to divide that
    # grapheme into constituent codepoints.
    #
    # ┏━━━━━━━━━━━━━━━┓
    # ┃      yö       ┃ String
    # ┣━━━┳━━━┳━━━━━━━┫
    # ┃ y ┃ o ┃   ̈    ┃ Codepoint Characters
    # ┣━━━╋━━━╋━━━━━━━┫
    # ┃121┃111┃  776  ┃ Codepoints
    # ┣━━━╋━━━╋━━━┳━━━┫
    # ┃121┃111┃204┃136┃ Bytes
    # ┗━━━┻━━━┻━━━┻━━━┛
    #   0   1   2   3
    var unicode_str2 = StringSlice("yö")

    assert_equal(unicode_str2[byte=0:1], "y")
    assert_equal(unicode_str2[byte=0:2], "yo")
    assert_equal(unicode_str2[byte=0:4], unicode_str2)
    # NOTE: This renders weirdly, but is a single-codepoint string containing
    #   <https://www.compart.com/en/unicode/U+0308>.
    assert_equal(unicode_str2[byte=2:4], "̈")


def test_heap_string_from_string_slice() raises:
    comptime static_str = StringSlice("Hello")

    comptime heap_string = String(static_str)

    assert_equal(heap_string, "Hello")


def test_string_substring() raises:
    var string = "Hello"
    var str_slice = StringSlice(string)

    assert_equal(str_slice.byte_length(), 5)
    assert_equal(str_slice[byte=0], "H")
    assert_equal(str_slice[byte=1], "e")
    assert_equal(str_slice[byte=2], "l")
    assert_equal(str_slice[byte=3], "l")
    assert_equal(str_slice[byte=4], "o")

    # ----------------------------------
    # Test subslicing
    # ----------------------------------

    # Slice the whole thing
    var sub1 = str_slice[byte=:5]
    assert_equal(sub1.byte_length(), 5)
    assert_equal(sub1[byte=0], "H")
    assert_equal(sub1[byte=1], "e")
    assert_equal(sub1[byte=2], "l")
    assert_equal(sub1[byte=3], "l")
    assert_equal(sub1[byte=4], "o")

    # Slice the end
    var sub2 = str_slice[byte=2:5]
    assert_equal(sub2.byte_length(), 3)
    assert_equal(sub2[byte=0], "l")
    assert_equal(sub2[byte=1], "l")
    assert_equal(sub2[byte=2], "o")

    # Slice the first element
    var sub3 = str_slice[byte=0:1]
    assert_equal(sub3.byte_length(), 1)
    assert_equal(sub3[byte=0], "H")
    assert_equal(sub3[byte=sub3.byte_length() - 1], "H")

    # ----------------------------------
    # Test empty subslicing
    # ----------------------------------

    var sub4 = str_slice[byte=0:0]
    assert_equal(sub4.byte_length(), 0)

    var sub5 = str_slice[byte=2:2]
    assert_equal(sub5.byte_length(), 0)

    # Empty slices still have a pointer value
    assert_equal(
        Int(sub5.as_bytes().unsafe_ptr()) - Int(sub4.as_bytes().unsafe_ptr()),
        2,
    )


def test_slice_len() raises:
    assert_equal(5, StringSlice("12345").byte_length())
    assert_equal(4, StringSlice("1234").byte_length())
    assert_equal(3, StringSlice("123").byte_length())
    assert_equal(2, StringSlice("12").byte_length())
    assert_equal(1, StringSlice("1").byte_length())
    assert_equal(0, StringSlice("").byte_length())

    # String length is in bytes, not codepoints.
    var s0 = "ನಮಸ್ಕಾರ"
    assert_equal(s0.byte_length(), 21)
    assert_equal(len(s0.codepoints()), 7)

    # For ASCII string, the byte and codepoint length are the same:
    var s1 = "abc"
    assert_equal(s1.byte_length(), 3)
    assert_equal(len(s1.codepoints()), 3)


def test_slice_count_codepoints() raises:
    var s0 = StringSlice("")
    assert_equal(s0.byte_length(), 0)
    assert_equal(s0.count_codepoints(), 0)

    var s1 = StringSlice("foo")
    assert_equal(s1.byte_length(), 3)
    assert_equal(s1.count_codepoints(), 3)

    # This string contains 1-, 2-, 3-, and 4-byte codepoint sequences.
    var s2 = EVERY_CODEPOINT_LENGTH_STR
    assert_equal(s2.byte_length(), 13)
    assert_equal(s2.count_codepoints(), 5)

    # Just a bit of Zalgo text.
    var s3 = StringSlice("H̵͙̖̼̬̬̲̱͊̇̅͂̍͐͌͘͜͝")
    assert_equal(s3.byte_length(), 37)
    assert_equal(s3.count_codepoints(), 19)

    # Character length is codepoints, not graphemes
    # This is thumbs up + a skin tone modifier codepoint.
    var s4 = StringSlice("👍🏻")
    assert_equal(s4.byte_length(), 8)
    assert_equal(s4.count_codepoints(), 2)
    # TODO: assert_equal(s4.grapheme_count(), 1)


def test_slice_eq() raises:
    var str1: String = "12345"
    var str2: String = "12345"
    var str3: StaticString = "12345"
    var str4: String = "abc"
    var str5: String = "abcdef"
    var str6: StaticString = "abcdef"

    # eq

    # FIXME: the origin of the StringSlice origin should be the data in the
    # string, not the string itself.
    # assert_true(StringSlice(str1).__eq__(str1))
    assert_true(StringSlice(str1).__eq__(str2))
    assert_true(StringSlice(str2).__eq__(StringSlice(str2)))
    assert_true(StringSlice(str1).__eq__(str3))

    # ne

    assert_true(StringSlice(str1).__ne__(str4))
    assert_true(StringSlice(str1).__ne__(str5))
    assert_true(StringSlice(str1).__ne__(StringSlice(str5)))
    assert_true(StringSlice(str1).__ne__(str6))


def test_slice_bool() raises:
    var str1: String = "abc"
    assert_true(StringSlice(str1).__bool__())
    var str2: String = ""
    assert_true(not StringSlice(str2).__bool__())


comptime REPR_MAPPINGS = [
    # Empty string
    ("", "''"),
    # Standard single-byte printable characters
    ("hello", "'hello'"),
    ("ABC123", "'ABC123'"),
    # Boundary cases for printable ASCII range
    (" ", "' '"),  # 0x20 - first printable ASCII
    ("!", "'!'"),  # 0x21 - first printable non-whitespace
    ("~", "'~'"),  # 0x7E - last printable ASCII
    # Special escape sequences
    ("\\", r"'\\'"),  # backslash - must be escaped
    ("\t", r"'\t'"),  # tab (0x09) - special escape
    ("\n", r"'\n'"),  # newline (0x0A) - special escape
    ("\r", r"'\r'"),  # carriage return (0x0D) - special escape
    ('"', "'\"'"),  # double quote - not escaped, but good to verify
    ("'", r"'\''"),  # single quote - escaped
    # Control characters - testing edge cases of different formatting rules
    ("\x00", r"'\x00'"),  # null - first control char (0x00-0x0F: \x0X format)
    ("\x06", r"'\x06'"),  # ACK - sample from 0x00-0x0F range
    ("\x0f", r"'\x0f'"),  # SI - last of 0x00-0x0F range (tests \x0X format)
    ("\x10", r"'\x10'"),  # DLE - first of 0x10-0x1F range (tests \xXX format)
    ("\x1b", r"'\x1b'"),  # ESC - sample from 0x10-0x1F range
    ("\x1f", r"'\x1f'"),  # US - last control char before space (boundary test)
    ("\x7f", r"'\x7f'"),  # DEL - non-printable after tilde (boundary test)
    # Multi-byte UTF-8 characters (test each byte length)
    ("Ö", "'Ö'"),  # 2-byte single char
    ("café", "'café'"),  # 2-byte mixed with ASCII
    ("你好", "'你好'"),  # 3-byte
    ("🔥", "'🔥'"),  # 4-byte emoji
    ("hello 🔥!", "'hello 🔥!'"),  # 4-byte mixed with ASCII
    # Mixed content - escapes with normal text
    ("hello\nworld", r"'hello\nworld'"),
    ('quote"here', "'quote\"here'"),
    ("back\\slash", r"'back\\slash'"),
    ("path\\to\\file", r"'path\\to\\file'"),
    # Mixed content - control chars embedded in text
    ("start\x00end", r"'start\x00end'"),
    ("data\x1fmore", r"'data\x1fmore'"),
    # Multiple escapes in one string
    ("\n\t\r", r"'\n\t\r'"),
    ('\\"', "'\\\\\"'"),
    # Boundary testing - transitions between character types
    ("\x1f ", r"'\x1f '"),  # control → printable
    (" \x7f", r"' \x7f'"),  # printable → DEL
    ("~\x7f", r"'~\x7f'"),  # last printable → DEL
]


def test_slice_write_repr_to() raises:
    for item in materialize[REPR_MAPPINGS]():
        var string = String()
        StringSlice.write_repr_to(item[0], string)
        assert_equal(string, item[1])


def test_find() raises:
    var haystack = StringSlice("abcdefg")
    var haystack_with_special_chars = StringSlice("abcdefg@#$")
    var haystack_repeated_chars = StringSlice("aaaaaaaaaaaaaaaaaaaaaaaa")

    assert_equal(haystack.find(StringSlice("a")), 0)
    assert_equal(haystack.find(StringSlice("ab")), 0)
    assert_equal(haystack.find(StringSlice("abc")), 0)
    assert_equal(haystack.find(StringSlice("bcd")), 1)
    assert_equal(haystack.find(StringSlice("de")), 3)
    assert_equal(haystack.find(StringSlice("fg")), 5)
    assert_equal(haystack.find(StringSlice("g")), 6)
    assert_equal(haystack.find(StringSlice("z")), -1)
    assert_equal(haystack.find(StringSlice("zzz")), -1)

    assert_equal(haystack.find(StringSlice("@#$")), -1)
    assert_equal(haystack_with_special_chars.find(StringSlice("@#$")), 7)

    assert_equal(haystack_repeated_chars.find(StringSlice("aaa")), 0)
    assert_equal(haystack_repeated_chars.find(StringSlice("AAa")), -1)

    assert_equal(haystack.find(StringSlice("hijklmnopqrstuvwxyz")), -1)

    assert_equal(StringSlice(String()).find(StringSlice("abc")), -1)


def test_find_compile_time() raises:
    comptime haystack = StringSlice("abcdefg")
    comptime haystack_with_special_chars = StringSlice("abcdefg@#$")
    comptime haystack_repeated_chars = StringSlice("aaaaaaaaaaaaaaaaaaaaaaaa")

    comptime c1 = haystack.find(StringSlice("a"))
    comptime c2 = haystack.find(StringSlice("ab"))
    comptime c3 = haystack.find(StringSlice("abc"))
    comptime c4 = haystack.find(StringSlice("bcd"))
    comptime c5 = haystack.find(StringSlice("de"))
    comptime c6 = haystack.find(StringSlice("fg"))
    comptime c7 = haystack.find(StringSlice("g"))
    comptime c8 = haystack.find(StringSlice("z"))
    comptime c9 = haystack.find(StringSlice("zzz"))
    comptime c10 = haystack.find(StringSlice("@#$"))
    comptime c11 = haystack_with_special_chars.find(StringSlice("@#$"))
    comptime c12 = haystack_repeated_chars.find(StringSlice("aaa"))
    comptime c13 = haystack_repeated_chars.find(StringSlice("AAa"))
    comptime c14 = haystack.find(StringSlice("hijklmnopqrstuvwxyz"))
    comptime c15 = StringSlice(String()).find(StringSlice("abc"))

    assert_equal(c1, 0)
    assert_equal(c2, 0)
    assert_equal(c3, 0)
    assert_equal(c4, 1)
    assert_equal(c5, 3)
    assert_equal(c6, 5)
    assert_equal(c7, 6)
    assert_equal(c8, -1)
    assert_equal(c9, -1)
    assert_equal(c10, -1)
    assert_equal(c11, 7)
    assert_equal(c12, 0)
    assert_equal(c13, -1)
    assert_equal(c14, -1)
    assert_equal(c15, -1)


def test_find_mstdl_2258() raises:
    # Bug - when searching for a needle with the following conditions:
    # - needle length is longer than the SIMD width
    # - the haystack prefix is larger than UInt16.MAX
    # - the haystack postfix is at least as long as the SIMD width
    # then the search would fail due to integer overflow in offset calculation.
    comptime simd_width = simd_width_of[DType.bool]()

    var needle = "z" * (simd_width + 1)
    var prefix = "a" * (Int(UInt16.MAX) + 1)
    var postfix = "a" * simd_width
    var haystack = prefix + needle + postfix

    var expected_pos = prefix.byte_length()
    var found = haystack.find(needle)
    assert_equal(found, expected_pos)
    assert_true(needle in haystack)


def test_is_codepoint_boundary() raises:
    var abc = StringSlice("abc")
    assert_equal(abc.byte_length(), 3)
    assert_true(abc.is_codepoint_boundary(0))
    assert_true(abc.is_codepoint_boundary(1))
    assert_true(abc.is_codepoint_boundary(2))
    assert_true(abc.is_codepoint_boundary(3))

    var thumb = StringSlice("👍")
    assert_equal(thumb.byte_length(), 4)
    assert_true(thumb.is_codepoint_boundary(0))
    assert_false(thumb.is_codepoint_boundary(1))
    assert_false(thumb.is_codepoint_boundary(2))
    assert_false(thumb.is_codepoint_boundary(3))

    var empty = StringSlice("")
    assert_equal(empty.byte_length(), 0)
    assert_true(empty.is_codepoint_boundary(0))
    # Also tests that positions greater then the length don't raise/abort.
    assert_false(empty.is_codepoint_boundary(1))


def test_comparison_operators() raises:
    var abc = StringSlice("abc")
    var de = StringSlice("de")
    var ABC = StringSlice("ABC")
    var ab = StringSlice("ab")
    var abcd = StringSlice("abcd")

    # Test equality and inequality
    assert_true(StringSlice.__eq__(abc, abc))
    assert_true(StringSlice.__eq__(abc, "abc"))
    assert_false(StringSlice.__eq__(abc, de))
    assert_false(StringSlice.__eq__(abc, "xyz"))

    # Test less than and greater than
    assert_true(StringSlice.__lt__(abc, de))
    assert_false(StringSlice.__lt__(de, abc))
    assert_false(StringSlice.__lt__(abc, abc))
    assert_true(StringSlice.__lt__(ab, abc))
    assert_true(StringSlice.__gt__(abc, ab))
    assert_false(StringSlice.__gt__(abc, abcd))

    # Test less than or equal to and greater than or equal to
    assert_true(StringSlice.__le__(abc, de))
    assert_true(StringSlice.__le__(abc, abc))
    assert_false(StringSlice.__le__(de, abc))
    assert_true(StringSlice.__ge__(abc, abc))
    assert_false(StringSlice.__ge__(ab, abc))
    assert_true(StringSlice.__ge__(abcd, abc))

    # Test case sensitivity in comparison (assuming ASCII order)
    assert_true(StringSlice.__gt__(abc, ABC))
    assert_false(StringSlice.__le__(abc, ABC))

    # Testing with implicit conversion
    assert_true(StringSlice.__lt__(abc, "defgh"))
    assert_false(StringSlice.__gt__(abc, "xyz"))
    assert_true(StringSlice.__ge__(abc, "abc"))
    assert_false(StringSlice.__le__(abc, "ab"))

    # Test comparisons involving empty strings
    assert_true(StringSlice.__lt__("", abc))
    assert_false(StringSlice.__lt__(abc, ""))
    assert_true(StringSlice.__le__("", ""))
    assert_true(StringSlice.__ge__("", ""))


def test_split() raises:
    comptime S = StaticString

    # Should add all whitespace-like chars as one
    # test all unicode separators
    var next_line = [UInt8(0xC2), 0x85]
    var unicode_line_sep = [UInt8(0xE2), 0x80, 0xA8]
    var unicode_paragraph_sep = [UInt8(0xE2), 0x80, 0xA9]
    # TODO add line and paragraph separator as StringLiteral once unicode
    # escape sequences are accepted
    var univ_sep_var = String(
        " ",
        "\t",
        "\n",
        "\r",
        "\v",
        "\f",
        "\x1c",
        "\x1d",
        "\x1e",
        String(unsafe_from_utf8=next_line),
        String(unsafe_from_utf8=unicode_line_sep),
        String(unsafe_from_utf8=unicode_paragraph_sep),
    )
    var s = univ_sep_var + "hello" + univ_sep_var + "world" + univ_sep_var
    assert_equal(StringSlice(s).split(), [StaticString("hello"), "world"])

    # should split into empty strings between separators
    assert_equal(S("1,,,3").split(","), [StaticString("1"), "", "", "3"])
    assert_equal(S(",,,").split(","), [StaticString(""), "", "", ""])
    assert_equal(S(" a b ").split(" "), [StaticString(""), "a", "b", ""])
    assert_equal(S("abababaaba").split("aba"), [StaticString(""), "b", "", ""])
    assert_true(len(S("").split()) == 0)
    assert_true(len(S(" ").split()) == 0)
    assert_true(len(S("").split(" ")) == 1)
    assert_true(len(S(",").split(",")) == 2)
    assert_true(len(S(" ").split(" ")) == 2)
    assert_true(len(S("").split("")) == 2)
    assert_true(len(S("  ").split(" ")) == 3)
    assert_true(len(S("   ").split(" ")) == 4)

    # should split into maxsplit + 1 items
    assert_equal(S("1,2,3").split(",", 0), [StaticString("1,2,3")])
    assert_equal(S("1,2,3").split(",", 1), [StaticString("1"), "2,3"])

    # Split in middle
    assert_equal(S("faang").split("n"), [StaticString("faa"), "g"])

    # No match from the delimiter
    assert_equal(S("hello world").split("x"), [StaticString("hello world")])

    # Multiple character delimiter
    assert_equal(S("hello").split("ll"), [StaticString("he"), "o"])

    var res: List = [StaticString(""), "bb", "", "", "", "bbb", ""]
    assert_equal(S("abbaaaabbba").split("a"), res)
    assert_equal(S("abbaaaabbba").split("a", 8), res)
    var s1 = S("abbaaaabbba").split("a", 5)
    assert_equal(s1, [StaticString(""), "bb", "", "", "", "bbba"])
    assert_equal(S("aaa").split("a", 0), [StaticString("aaa")])
    assert_equal(S("a").split("a"), [StaticString(""), ""])
    assert_equal(S("1,2,3").split("3", 0), [StaticString("1,2,3")])
    assert_equal(S("1,2,3").split("3", 1), [StaticString("1,2,"), ""])
    assert_equal(S("1,2,3,3").split("3", 2), [StaticString("1,2,"), ",", ""])
    assert_equal(
        S("1,2,3,3,3").split("3", 2), [StaticString("1,2,"), ",", ",3"]
    )

    assert_equal(S("Hello 🔥!").split(), [StaticString("Hello"), "🔥!"])

    var s2 = S("Лорем ипсум долор сит амет").split(" ")
    assert_equal(s2, [StaticString("Лорем"), "ипсум", "долор", "сит", "амет"])
    var s3 = S("Лорем ипсум долор сит амет").split("м")
    assert_equal(s3, [StaticString("Лоре"), " ипсу", " долор сит а", "ет"])

    assert_equal(S("123").split(""), [StaticString(""), "1", "2", "3", ""])
    assert_equal(S("").join(S("123").split("")), "123")
    assert_equal(S(",1,2,3,").split(","), S("123").split(""))
    assert_equal(S(",").join(S("123").split("")), ",1,2,3,")


def test_splitlines() raises:
    comptime S = StaticString

    # Test with no line breaks
    assert_equal(S("hello world").splitlines(), [StaticString("hello world")])

    # Test with line breaks
    assert_equal(
        S("hello\nworld").splitlines(), [StaticString("hello"), "world"]
    )
    assert_equal(
        S("hello\rworld").splitlines(), [StaticString("hello"), "world"]
    )
    assert_equal(
        S("hello\r\nworld").splitlines(), [StaticString("hello"), "world"]
    )

    # Test with multiple different line breaks
    var s1 = S("hello\nworld\r\nmojo\rlanguage\r\n")
    var hello_mojo: List = [StaticString("hello"), "world", "mojo", "language"]
    assert_equal(s1.splitlines(), hello_mojo)
    assert_equal(
        s1.splitlines(keepends=True),
        [StaticString("hello\n"), "world\r\n", "mojo\r", "language\r\n"],
    )

    # Test with an empty string
    assert_equal(S("").splitlines(), [])
    # test \v \f \x1c \x1d
    var s2 = S("hello\vworld\fmojo\x1clanguage\x1d")
    assert_equal(s2.splitlines(), hello_mojo)
    assert_equal(
        s2.splitlines(keepends=True),
        [StaticString("hello\v"), "world\f", "mojo\x1c", "language\x1d"],
    )

    # test \x1c \x1d \x1e
    var s3 = S("hello\x1cworld\x1dmojo\x1elanguage\x1e")
    assert_equal(s3.splitlines(), hello_mojo)
    assert_equal(
        s3.splitlines(keepends=True),
        [StaticString("hello\x1c"), "world\x1d", "mojo\x1e", "language\x1e"],
    )

    # test \x85 \u2028 \u2029
    var next_line = String(unsafe_from_utf8=[Byte(0xC2), 0x85])
    var unicode_line_sep = String(unsafe_from_utf8=[Byte(0xE2), 0x80, 0xA8])
    var unicode_paragraph_sep = String(
        unsafe_from_utf8=[Byte(0xE2), 0x80, 0xA9]
    )

    for ref u in [next_line, unicode_line_sep, unicode_paragraph_sep]:
        var item = StaticString("").join(
            ["hello", u, "world", u, "mojo", u, "language", u]
        )
        assert_equal(item.splitlines(), hello_mojo)
        assert_equal(
            _to_string_list(item.splitlines(keepends=True)),
            ["hello" + u, "world" + u, "mojo" + u, "language" + u],
        )


def test_rstrip() raises:
    # with default rstrip chars
    var empty_string = StringSlice("")
    assert_true(empty_string.rstrip() == "")

    var space_string = StringSlice(" \t\n\r\v\f  ")
    assert_true(space_string.rstrip() == "")

    var str0 = StringSlice("     n ")
    assert_true(str0.rstrip() == "     n")

    var str1 = StringSlice("string")
    assert_true(str1.rstrip() == "string")

    var str2 = StringSlice("something \t\n\t\v\f")
    assert_true(str2.rstrip() == "something")

    # with custom chars for rstrip
    var str3 = StringSlice("mississippi")
    assert_true(str3.rstrip("sip") == "m")

    var str4 = StringSlice("mississippimississippi \n ")
    assert_true(str4.rstrip("sip ") == "mississippimississippi \n")
    assert_true(str4.rstrip("sip \n") == "mississippim")

    # should strip off single codepoints
    var str5 = "😀smile😀"
    assert_true(str5.rstrip("😀") == "😀smile")

    # Ñ and Ò share the leading utf-8 byte of 0xc3
    var str6 = "eeeeÑ"
    assert_true(str6.rstrip("Ò") == "eeeeÑ")


def test_lstrip() raises:
    # with default lstrip chars
    var empty_string = StringSlice("")
    assert_true(empty_string.lstrip() == "")

    var space_string = StringSlice(" \t\n\r\v\f  ")
    assert_true(space_string.lstrip() == "")

    var str0 = StringSlice("     n ")
    assert_true(str0.lstrip() == "n ")

    var str1 = StringSlice("string")
    assert_true(str1.lstrip() == "string")

    var str2 = StringSlice(" \t\n\t\v\fsomething")
    assert_true(str2.lstrip() == "something")

    # with custom chars for lstrip
    var str3 = StringSlice("mississippi")
    assert_true(str3.lstrip("mis") == "ppi")

    var str4 = StringSlice(" \n mississippimississippi")
    assert_true(str4.lstrip("mis ") == "\n mississippimississippi")
    assert_true(str4.lstrip("mis \n") == "ppimississippi")

    var str5 = "😀smile😀"
    assert_true(str5.lstrip("😀") == "smile😀")

    # Ñ and Ò share the leading utf-8 byte of 0xc3
    var str6 = "Ñeeee"
    assert_true(str6.lstrip("Ò") == "Ñeeee")


def test_strip() raises:
    # with default strip chars
    var empty_string = StringSlice("")
    assert_true(empty_string.strip() == "")
    comptime comp_empty_string_stripped = StringSlice("").strip()
    assert_true(comp_empty_string_stripped == "")

    var space_string = StringSlice(" \t\n\r\v\f  ")
    assert_true(space_string.strip() == "")
    comptime comp_space_string_stripped = StringSlice(" \t\n\r\v\f  ").strip()
    assert_true(comp_space_string_stripped == "")

    var str0 = StringSlice("     n ")
    assert_true(str0.strip() == "n")
    comptime comp_str0_stripped = StringSlice("     n ").strip()
    assert_true(comp_str0_stripped == "n")

    var str1 = StringSlice("string")
    assert_true(str1.strip() == "string")
    comptime comp_str1_stripped = ("string").strip()
    assert_true(comp_str1_stripped == "string")

    var str2 = StringSlice(" \t\n\t\v\fsomething \t\n\t\v\f")
    comptime comp_str2_stripped = (" \t\n\t\v\fsomething \t\n\t\v\f").strip()
    assert_true(str2.strip() == "something")
    assert_true(comp_str2_stripped == "something")

    # with custom strip chars
    var str3 = StringSlice("mississippi")
    assert_true(str3.strip("mips") == "")
    assert_true(str3.strip("mip") == "ssiss")
    comptime comp_str3_stripped = StringSlice("mississippi").strip("mips")
    assert_true(comp_str3_stripped == "")

    var str4 = StringSlice(" \n mississippimississippi \n ")
    assert_true(str4.strip(" ") == "\n mississippimississippi \n")
    assert_true(str4.strip("\nmip ") == "ssissippimississ")

    comptime comp_str4_stripped = (
        StringSlice(" \n mississippimississippi \n ").strip(" ")
    )
    assert_true(comp_str4_stripped == "\n mississippimississippi \n")

    var str5 = "😀smile😀"
    assert_true(str5.strip("😀") == "smile")

    # Ñ and Ò share the leading utf-8 byte of 0xc3
    var str6 = "ÑeeeeÑ"
    assert_true(str6.strip("Ò") == "ÑeeeeÑ")


def test_strip_mutable_chars() raises:
    # `chars` is read-only, so a mutable argument must not be borrowed mutably;
    # otherwise it can't alias `self` and it can't be a mutable local at all
    # once `self` is an interior slice of it.
    var chars = String("hi")
    assert_equal(StringSlice("himojohi").lstrip(chars), "mojohi")
    assert_equal(StringSlice("himojohi").rstrip(chars), "himojo")
    assert_equal(StringSlice("himojohi").strip(chars), "mojo")

    var mut_chars = MutStringSpan(chars)
    assert_equal(StringSlice("himojohi").lstrip(mut_chars), "mojohi")
    assert_equal(StringSlice("himojohi").rstrip(mut_chars), "himojo")
    assert_equal(StringSlice("himojohi").strip(mut_chars), "mojo")

    var self_aliasing = StaticString("aabbaa")
    assert_equal(self_aliasing.lstrip(self_aliasing), "")
    assert_equal(self_aliasing.rstrip(self_aliasing), "")
    assert_equal(self_aliasing.strip(self_aliasing), "")


def test_startswith() raises:
    var empty = StringSlice("")
    assert_true(empty.startswith(""))
    assert_false(empty.startswith("a"))
    assert_false(empty.startswith("ab"))

    var a = StringSlice("a")
    assert_true(a.startswith(""))
    assert_true(a.startswith("a"))
    assert_false(a.startswith("ab"))

    var ab = StringSlice("ab")
    assert_true(ab.startswith(""))
    assert_true(ab.startswith("a"))
    assert_false(ab.startswith("b"))
    assert_true(ab.startswith("b", start=1))
    assert_true(ab.startswith("a", end=1))
    assert_true(ab.startswith("ab"))


def test_endswith() raises:
    var empty = StringSlice("")
    assert_true(empty.endswith(""))
    assert_false(empty.endswith("a"))
    assert_false(empty.endswith("ab"))

    var a = StringSlice("a")
    assert_true(a.endswith(""))
    assert_true(a.endswith("a"))
    assert_false(a.endswith("ab"))

    var ab = StringSlice("ab")
    assert_true(ab.endswith(""))
    assert_false(ab.endswith("a"))
    assert_true(ab.endswith("b"))
    assert_true(ab.endswith("b", start=1))
    assert_true(ab.endswith("a", end=1))
    assert_true(ab.endswith("ab"))


def test_isupper() raises:
    assert_true(StringSlice("ASDG").isupper())
    assert_false(StringSlice("AsDG").isupper())
    assert_true(StringSlice("ABC123").isupper())
    assert_false(StringSlice("1!").isupper())
    assert_true(StringSlice("É").isupper())
    assert_false(StringSlice("é").isupper())


def test_islower() raises:
    assert_true(StringSlice("asdfg").islower())
    assert_false(StringSlice("asdFDg").islower())
    assert_true(StringSlice("abc123").islower())
    assert_false(StringSlice("1!").islower())
    assert_true(StringSlice("é").islower())
    assert_false(StringSlice("É").islower())


def test_lower() raises:
    assert_equal(StringSlice("HELLO").lower(), "hello")
    assert_equal(StringSlice("hello").lower(), "hello")
    assert_equal(StringSlice("FoOBaR").lower(), "foobar")

    assert_equal(StringSlice("MOJO🔥").lower(), "mojo🔥")

    assert_equal(StringSlice("É").lower(), "é")
    assert_equal(StringSlice("é").lower(), "é")

    assert_equal(StringSlice("").lower(), "")
    # Deseret, whose case mappings are 4-byte sequences.
    assert_equal(StringSlice("𐐀").lower(), "𐐨")
    assert_equal(EVERY_CODEPOINT_LENGTH_STR.lower(), "߷കൈ🔄!")
    # U+023A lowercases to U+2C65, one byte longer than the input.
    assert_equal(StringSlice("Ⱥ").lower(), "ⱥ")


def test_upper() raises:
    assert_equal(StringSlice("hello").upper(), "HELLO")
    assert_equal(StringSlice("HELLO").upper(), "HELLO")
    assert_equal(StringSlice("FoOBaR").upper(), "FOOBAR")

    assert_equal(StringSlice("mojo🔥").upper(), "MOJO🔥")

    assert_equal(StringSlice("É").upper(), "É")
    assert_equal(StringSlice("é").upper(), "É")

    assert_equal(StringSlice("").upper(), "")
    assert_equal(StringSlice("𐐨").upper(), "𐐀")
    assert_equal(EVERY_CODEPOINT_LENGTH_STR.upper(), "߷കൈ🔄!")

    # Codepoints whose uppercase form is a sequence of 2 or 3 codepoints:
    # `ß` (U+00DF) becomes "SS", and `ΐ` (U+0390) becomes the decomposed
    # sequence U+0399 U+0308 U+0301.
    assert_equal(StringSlice("straße").upper(), "STRASSE")
    var upper_0390 = String(
        Codepoint.from_u32(0x0399).value(),
        Codepoint.from_u32(0x0308).value(),
        Codepoint.from_u32(0x0301).value(),
    )
    assert_equal(StringSlice("ΐ").upper(), upper_0390)

    # `ΐ` is 2 bytes but uppercases to 6, so a run of them outgrows the
    # capacity `to_uppercase` estimates from the input length.
    var input = String()
    var expected = String()
    for _ in range(64):
        input += "ΐ"
        expected += upper_0390
    assert_equal(StringSlice(input).upper(), expected)


def test_is_ascii_digit() raises:
    assert_false(StringSlice("").is_ascii_digit())
    assert_true(StringSlice("123").is_ascii_digit())
    assert_false(StringSlice("asdg").is_ascii_digit())
    assert_false(StringSlice("123asdg").is_ascii_digit())


def test_is_ascii_printable() raises:
    assert_true(StringSlice("aasdg").is_ascii_printable())
    assert_false(StringSlice("aa\nae").is_ascii_printable())
    assert_false(StringSlice("aa\tae").is_ascii_printable())


def test_ascii_rjust() raises:
    assert_equal(StringSlice("hello").ascii_rjust(4), "hello")
    assert_equal(StringSlice("hello").ascii_rjust(8), "   hello")
    assert_equal(StringSlice("hello").ascii_rjust(8, "*"), "***hello")


def test_ascii_ljust() raises:
    assert_equal(StringSlice("hello").ascii_ljust(4), "hello")
    assert_equal(StringSlice("hello").ascii_ljust(8), "hello   ")
    assert_equal(StringSlice("hello").ascii_ljust(8, "*"), "hello***")


def test_ascii_center() raises:
    assert_equal(StringSlice("hello").ascii_center(4), "hello")
    assert_equal(StringSlice("hello").ascii_center(8), " hello  ")
    assert_equal(StringSlice("hello").ascii_center(8, "*"), "*hello**")


def test_count() raises:
    var str = StringSlice("Hello world")

    assert_equal(12, str.count(""))
    assert_equal(1, str.count("Hell"))
    assert_equal(3, str.count("l"))
    assert_equal(1, str.count("ll"))
    assert_equal(1, str.count("ld"))
    assert_equal(0, str.count("universe"))

    assert_equal(StringSlice("aaaaa").count("a"), 5)
    assert_equal(StringSlice("aaaaaa").count("aa"), 3)


def test_replace() raises:
    assert_equal(StringSlice("").replace("", "hello world"), "")
    assert_equal(
        StringSlice("hello").replace("", "something"),
        "somethinghsomethingesomethinglsomethinglsomethingo",
    )
    assert_equal(StringSlice("hello world").replace("world", ""), "hello ")
    assert_equal(
        StringSlice("hello world").replace("world", "mojo"), "hello mojo"
    )
    assert_equal(
        StringSlice("hello world hello world").replace("world", "mojo"),
        "hello mojo hello mojo",
    )
    assert_equal(
        StringSlice("this is a test").replace(
            "this", "abcdefghijklmnopqrstuvwxyz_abcdefghijklmnopqrstuvwxyz"
        ),
        "abcdefghijklmnopqrstuvwxyz_abcdefghijklmnopqrstuvwxyz is a test",
    )
    assert_equal(
        StringSlice("🔥").replace("", "x"),
        "x🔥",
    )
    assert_equal(
        StringSlice("a🔥a🔥a").replace("🔥", "x"),
        "axaxa",
    )
    assert_equal(
        StringSlice("a🔥a🔥a").replace("a", "x"),
        "x🔥x🔥x",
    )


def test_join() raises:
    # TODO(MOCO-2908): This explicit origin should not be necessary; the
    #   compiler ought to infer some default "bottom" origin.
    assert_equal(StaticString("").join(Span[String, ImmutAnyOrigin]()), "")
    assert_equal(StaticString("").join(["a", "b", "c"]), "abc")
    assert_equal(StaticString(" ").join(["a", "b", "c"]), "a b c")
    assert_equal(StaticString(" ").join(["a", "b", "c", ""]), "a b c ")
    assert_equal(StaticString(" ").join(["a", "b", "c", " "]), "a b c  ")

    var sep = StaticString(",")
    var s = "abc"
    assert_equal(sep.join([s, s, s, s]), "abc,abc,abc,abc")
    assert_equal(sep.join([1, 2, 3]), "1,2,3")
    # TODO(MSTDL-2078): Continue supporting heterogenous StringSlice.join
    #   arguments, somehow?
    # assert_equal(sep.join([1, "abc", 3]), "1,abc,3")

    var s2 = StaticString(",").join([Byte(1), 2, 3])
    assert_equal(s2, "1,2,3")

    var s3 = StaticString(",").join([Byte(1), 2, 3, 4, 5, 6, 7, 8, 9])
    assert_equal(s3, "1,2,3,4,5,6,7,8,9")

    var s4 = StaticString(",").join(List[Byte]())
    assert_equal(s4, "")

    var s5 = StaticString(",").join([Byte(1)])
    assert_equal(s5, "1")


def test_string_slice_intern() raises:
    assert_equal(get_static_string["hello"](), "hello")
    assert_equal(get_static_string[String("hello")](), "hello")
    assert_equal(get_static_string[String(42)](), "42")
    comptime simd = SIMD[.int64, 4](1, 2, 3, 4)
    assert_equal(get_static_string[String(simd)](), "[1, 2, 3, 4]")
    # Test get_static_string with multiple string arguments.
    assert_equal(get_static_string["a", "b", "c"](), "abc")


# This is just a compile test
# it does not need to be run
def test_merge() raises:
    var a = ""
    var b = "hi"

    def cond(
        pred: Bool, a: StringSlice, b: StringSlice
    ) -> StringSlice[origin_of(a.origin, b.origin)]:
        return a if pred else b

    _ = cond(True, a, b)


def test_codepoint_indexing() raises:
    assert_equal(StringSlice("abc")[codepoint=0], "a")
    assert_equal(StringSlice("abc")[codepoint=2], "c")
    assert_equal(EVERY_CODEPOINT_LENGTH_STR[codepoint=0], "߷")
    assert_equal(EVERY_CODEPOINT_LENGTH_STR[codepoint=1], "ക")
    assert_equal(EVERY_CODEPOINT_LENGTH_STR[codepoint=2], "ൈ")
    assert_equal(EVERY_CODEPOINT_LENGTH_STR[codepoint=3], "🔄")
    assert_equal(EVERY_CODEPOINT_LENGTH_STR[codepoint=4], "!")
    assert_equal(StringSlice("🔄🔥🔄")[codepoint=0], "🔄")
    assert_equal(StringSlice("🔄🔥🔄")[codepoint=1], "🔥")
    assert_equal(StringSlice("🔄🔥🔄")[codepoint=2], "🔄")

    # ASCII followed by a multi-byte codepoint: the codepoint index no
    # longer lines up with the byte index once `🙂` (4 bytes) is reached.
    var mixed = StringSlice("ab🙂cd")
    assert_equal(mixed[codepoint=2], "🙂")
    assert_equal(mixed[codepoint=3], "c")


def test_codepoint_slicing() raises:
    assert_equal(StringSlice("abc")[codepoint=0:1], "a")
    assert_equal(StringSlice("abc")[codepoint=2:3], "c")
    assert_equal(EVERY_CODEPOINT_LENGTH_STR[codepoint=0:1], "߷")
    assert_equal(EVERY_CODEPOINT_LENGTH_STR[codepoint=1:2], "ക")
    assert_equal(EVERY_CODEPOINT_LENGTH_STR[codepoint=2:3], "ൈ")
    assert_equal(EVERY_CODEPOINT_LENGTH_STR[codepoint=3:4], "🔄")
    assert_equal(EVERY_CODEPOINT_LENGTH_STR[codepoint=4:5], "!")

    var elems = StringSlice("🔄🔥🔄")
    assert_equal(elems[codepoint=0:1], "🔄")
    assert_equal(elems[codepoint=1:2], "🔥")
    assert_equal(elems[codepoint=2:3], "🔄")

    assert_equal(elems[codepoint=:1], "🔄")
    assert_equal(elems[codepoint=:2], "🔄🔥")
    assert_equal(elems[codepoint=:3], "🔄🔥🔄")

    assert_equal(elems[codepoint=:], "🔄🔥🔄")

    assert_equal(elems[codepoint=0:], "🔄🔥🔄")
    assert_equal(elems[codepoint=1:], "🔥🔄")
    assert_equal(elems[codepoint=2:], "🔄")

    assert_equal(elems[codepoint=0:3], "🔄🔥🔄")
    assert_equal(elems[codepoint=1:3], "🔥🔄")
    assert_equal(elems[codepoint=0:2], "🔄🔥")

    assert_equal(elems[codepoint=0:0], "")
    assert_equal(elems[codepoint=1:1], "")
    assert_equal(elems[codepoint=2:2], "")

    # `start == count_codepoints()` is a valid (empty) boundary; a start
    # past that, or a start greater than end, aborts (see
    # test_string_slice_codepoint_grapheme_bounds_abort.mojo).
    assert_equal(elems[codepoint=3:], "")

    # ASCII followed by a multi-byte codepoint: the codepoint index no
    # longer lines up with the byte index once `🙂` (4 bytes) is reached.
    var mixed = StringSlice("ab🙂cd")
    assert_equal(mixed[codepoint=2:3], "🙂")
    assert_equal(mixed[codepoint=3:4], "c")


def test_grapheme_indexing() raises:
    assert_equal(StringSlice("abc")[grapheme=0], "a")
    assert_equal(StringSlice("abc")[grapheme=1], "b")
    assert_equal(StringSlice("abc")[grapheme=2], "c")
    assert_equal(StringSlice("🔄🔥🔄")[grapheme=0], "🔄")
    assert_equal(StringSlice("🔄🔥🔄")[grapheme=1], "🔥")
    assert_equal(StringSlice("🔄🔥🔄")[grapheme=2], "🔄")
    assert_equal(StringSlice("👨‍🚀🧑‍🌾क्षि")[grapheme=0], "👨‍🚀")
    assert_equal(StringSlice("👨‍🚀🧑‍🌾क्षि")[grapheme=1], "🧑‍🌾")
    assert_equal(StringSlice("👨‍🚀🧑‍🌾क्षि")[grapheme=2], "क्षि")


def test_mut_string_slice_alias() raises:
    def capitalize(s: MutStringSpan[_]):
        s.as_bytes()[0] -= Byte(ord("a") - ord("A"))

    def capitalize_compat(s: MutStringSlice[_]):
        capitalize(s)

    var canonical_data = String("hello")
    var compatibility_data = String("world")
    capitalize(canonical_data)
    capitalize_compat(compatibility_data)
    assert_equal(canonical_data, "Hello")
    assert_equal(compatibility_data, "World")


def test_imm_string_slice_alias() raises:
    def byte_sum(s: ImmStringSpan[_]) -> Int:
        var total = 0
        for b in s.as_bytes():
            total += Int(b)
        return total

    def byte_sum_compat(s: ImmStringSlice[_]) -> Int:
        return byte_sum(s)

    var mutable_data = String("abc")
    var immutable_data = StaticString("abc")

    # Both names work with mutable and immutable data.
    assert_equal(byte_sum(mutable_data), 294)
    assert_equal(byte_sum_compat(immutable_data), 294)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
