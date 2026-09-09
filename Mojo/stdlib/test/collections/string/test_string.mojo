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

from std.collections.string.string import (
    _calc_initial_buffer_size_int32,
    _calc_initial_buffer_size_int64,
)
from std.collections.string.string_span import _to_string_list
from std.math import isinf, isnan

from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.testing import (
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_raises,
    assert_true,
)
from std.testing import TestSuite


def test_constructors() raises:
    # Default construction
    assert_equal(0, String().byte_length())
    assert_true(not String())
    assert_not_equal("xyz", "abc")

    # Construction from Int
    var s0 = String(0)
    assert_equal("0", String(0))
    assert_equal(1, s0.byte_length())

    var s1 = String(123)
    assert_equal("123", String(123))
    assert_equal(3, s1.byte_length())

    # Construction from StringLiteral
    var s2 = "abc"
    assert_equal("abc", String(s2))
    assert_equal(3, s2.byte_length())

    # Construction with capacity
    var s4 = String(capacity_bytes=1)
    assert_true(s4.capacity_bytes() <= String.INLINE_CAPACITY)

    # Construction from Codepoint
    var s5 = String(Codepoint(65))
    assert_true(s5.capacity_bytes() <= String.INLINE_CAPACITY)
    assert_equal(s5, "A")


def test_copy() raises:
    var s0 = "find"
    var s1 = String(s0)
    s1.unsafe_as_bytes_mut().unsafe_ptr()[unsafe_offset=3] = Byte(ord("e"))
    assert_equal("find", s0)
    assert_equal("fine", s1)


def test_len() raises:
    # String length is in bytes, not codepoints.
    var s0 = "ನಮಸ್ಕಾರ"

    assert_equal(s0.byte_length(), 21)
    assert_equal(len(s0.codepoints()), 7)

    # For ASCII string, the byte and codepoint length are the same:
    var s1 = "abc"

    assert_equal(s1.byte_length(), 3)
    assert_equal(len(s1.codepoints()), 3)


def test_equality_operators() raises:
    var s0 = "abc"
    var s1 = "def"
    assert_equal(s0, s0)
    assert_not_equal(s0, s1)

    var s2 = "abc"
    assert_equal(s0, s2)
    # Explicitly invoke eq and ne operators
    assert_true(s0 == s2)
    assert_false(s0 != s2)

    # Is case sensitive
    var s3 = "ABC"
    assert_not_equal(s0, s3)

    # Implicit conversion can promote for eq and ne
    assert_equal(s0, "abc")
    assert_not_equal(s0, "notabc")


def test_comparison_operators() raises:
    var abc = "abc"
    var de = "de"
    var ABC = "ABC"
    var ab = "ab"
    var abcd = "abcd"

    # Test less than and greater than
    assert_true(String.__lt__(abc, de))
    assert_false(String.__lt__(de, abc))
    assert_false(String.__lt__(abc, abc))
    assert_true(String.__lt__(ab, abc))
    assert_true(String.__gt__(abc, ab))
    assert_false(String.__gt__(abc, abcd))

    # Test less than or equal to and greater than or equal to
    assert_true(String.__le__(abc, de))
    assert_true(String.__le__(abc, abc))
    assert_false(String.__le__(de, abc))
    assert_true(String.__ge__(abc, abc))
    assert_false(String.__ge__(ab, abc))
    assert_true(String.__ge__(abcd, abc))

    # Test case sensitivity in comparison (assuming ASCII order)
    assert_true(String.__gt__(abc, ABC))
    assert_false(String.__le__(abc, ABC))

    # Testing with implicit conversion
    assert_true(String.__lt__(abc, "defgh"))
    assert_false(String.__gt__(abc, "xyz"))
    assert_true(String.__ge__(abc, "abc"))
    assert_false(String.__le__(abc, "ab"))

    # Test comparisons involving empty strings
    assert_true(String.__lt__("", abc))
    assert_false(String.__lt__(abc, ""))
    assert_true(String.__le__("", ""))
    assert_true(String.__ge__("", ""))


def test_add() raises:
    var s1 = "123"
    var s2 = "abc"
    var s3 = s1 + s2
    assert_equal("123abc", s3)

    var s4 = "x"
    var s5 = s4.join([1, 2, 3])
    assert_equal("1x2x3", s5)

    var s6 = s4.join([s1, s2])
    assert_equal("123xabc", s6)

    var s7 = String()
    assert_equal("abc", s2 + s7)

    assert_equal("abcdef", s2 + "def")
    assert_equal("123abc", "123" + s2)


def test_add_string_slice() raises:
    var s1 = "123"
    var s2 = StringSlice("abc")
    var s3 = "abc"
    assert_equal("123abc", s1 + s2)
    assert_equal("123abc", s1 + s3)
    assert_equal("abc123", s2 + s1)
    assert_equal("abc123", s3 + s1)
    s1 += s2
    s1 += s3
    assert_equal("123abcabc", s1)


def test_string_join() raises:
    var sep = ","
    var s0 = "abc"
    var s1 = sep.join([s0, s0, s0, s0])
    assert_equal("abc,abc,abc,abc", s1)

    assert_equal(sep.join([1, 2, 3]), "1,2,3")

    # TODO(MSTDL-2078): Continue supporting heterogenous String.join
    #   arguments, somehow?
    # assert_equal(sep.join(1, "abc", 3), "1,abc,3")

    var s2 = ",".join([UInt8(1), 2, 3])
    assert_equal(s2, "1,2,3")

    var s3 = ",".join([UInt8(1), 2, 3, 4, 5, 6, 7, 8, 9])
    assert_equal(s3, "1,2,3,4,5,6,7,8,9")

    var s4 = ",".join(List[UInt8]())
    assert_equal(s4, "")

    var s5 = ",".join([UInt8(1)])
    assert_equal(s5, "1")

    var s6 = ",".join([String("1"), "2", "3"])
    assert_equal(s6, "1,2,3")


def test_ord() raises:
    # Regular ASCII
    assert_equal(ord("A"), 65)
    assert_equal(ord("Z"), 90)
    assert_equal(ord("0"), 48)
    assert_equal(ord("9"), 57)
    assert_equal(ord("a"), 97)
    assert_equal(ord("z"), 122)
    assert_equal(ord("!"), 33)

    # Multi byte character
    assert_equal(ord("α"), 945)
    assert_equal(ord("➿"), 10175)
    assert_equal(ord("🔥"), 128293)

    # Make sure they work in the parameter domain too
    comptime single_byte = ord("A")
    assert_equal(single_byte, 65)
    comptime single_byte2 = ord("!")
    assert_equal(single_byte2, 33)

    # TODO: change these to parameter domain when it work.
    var multi_byte = ord("α")
    assert_equal(multi_byte, 945)
    var multi_byte2 = ord("➿")
    assert_equal(multi_byte2, 10175)
    var multi_byte3 = ord("🔥")
    assert_equal(multi_byte3, 128293)

    # Test StringSlice overload
    assert_equal(ord(StringSlice("A")), 65)
    assert_equal(ord(StringSlice("α")), 945)
    assert_equal(ord(StringSlice("➿")), 10175)
    assert_equal(ord(StringSlice("🔥")), 128293)

    # `\xhh` and `\ooo` escapes denote Unicode code points (not raw bytes),
    # so values >= 0x80 are encoded to UTF-8 and `ord` recovers the original
    # value.
    assert_equal(ord("\x1c"), 0x1C)
    assert_equal(ord("\x1e"), 0x1E)
    assert_equal(ord("\x7f"), 0x7F)
    assert_equal(ord("\x85"), 0x85)  # U+0085 NEL
    assert_equal(ord("\xa0"), 0xA0)  # U+00A0 NBSP
    assert_equal(ord("\xff"), 0xFF)
    assert_equal(ord("\205"), 0x85)  # octal form of U+0085 NEL
    assert_equal(ord("\377"), 0xFF)  # octal form of U+00FF


def test_chr() raises:
    assert_equal("\0", chr(0))
    assert_equal("A", chr(65))
    assert_equal("a", chr(97))
    assert_equal("!", chr(33))
    assert_equal("α", chr(945))
    assert_equal("➿", chr(10175))
    assert_equal("🔥", chr(128293))


def test_string_indexing() raises:
    var str = "Hello Mojo!!"

    assert_equal("H", str[byte=0])
    assert_equal("!", str[byte=str.byte_length() - 1])
    assert_equal("H", str[byte=0])
    assert_equal("llo Mojo!!", str[byte=2:])
    assert_equal("lo Mojo!", str[byte = 3 : str.byte_length() - 1])
    assert_equal("lo Moj", str[byte = 3 : str.byte_length() - 3])
    assert_equal("Hello Mojo!!", str[byte=:])
    assert_equal("Hello Mojo!!", str[byte = : str.byte_length()])

    var str2 = "😌😃"
    assert_equal("😌", str2[byte=0])
    assert_equal("😌", str2[byte=0:4])
    assert_equal("😃", str2[byte=4])
    assert_equal("😃", str2[byte=4:])
    var str3 = "😌😃🥰😋"
    assert_equal("😃🥰", str3[byte=4:12])


def test_atol() raises:
    # base 10
    assert_equal(375, atol("375"))
    assert_equal(1, atol("001"))
    assert_equal(5, atol(" 005"))
    assert_equal(13, atol(" 013  "))
    assert_equal(-89, atol("-89"))
    assert_equal(-52, atol(" -52"))
    assert_equal(-69, atol(" -69  "))
    assert_equal(1_100_200, atol(" 1_100_200"))

    # other bases
    assert_equal(10, atol("A", 16))
    assert_equal(15, atol("f ", 16))
    assert_equal(255, atol(" FF", 16))
    assert_equal(255, atol(" 0xff ", 16))
    assert_equal(255, atol(" 0Xff ", 16))
    assert_equal(18, atol("10010", 2))
    assert_equal(18, atol("0b10010", 2))
    assert_equal(18, atol("0B10010", 2))
    assert_equal(10, atol("12", 8))
    assert_equal(10, atol("0o12", 8))
    assert_equal(10, atol("0O12", 8))
    assert_equal(35, atol("Z", 36))
    assert_equal(255, atol("0x_00_ff", 16))
    assert_equal(18, atol("0b0001_0010", 2))
    assert_equal(18, atol("0b_000_1001_0", 2))

    # Negative cases
    with assert_raises(
        contains="String is not convertible to integer with base 10: '9.03'"
    ):
        _ = atol("9.03")

    with assert_raises(
        contains="String is not convertible to integer with base 10: ' 10 1'"
    ):
        _ = atol(" 10 1")

    # start/end with underscore double underscores
    with assert_raises(
        contains="String is not convertible to integer with base 10: '5__5'"
    ):
        _ = atol("5__5")

    with assert_raises(
        contains="String is not convertible to integer with base 10: ' _5'"
    ):
        _ = atol(" _5")

    with assert_raises(
        contains="String is not convertible to integer with base 10: '5_'"
    ):
        _ = atol("5_")

    with assert_raises(
        contains="String is not convertible to integer with base 5: '5'"
    ):
        _ = atol("5", 5)

    with assert_raises(
        contains="String is not convertible to integer with base 10: '0x_ff'"
    ):
        _ = atol("0x_ff")

    with assert_raises(
        contains="String is not convertible to integer with base 3: '_12'"
    ):
        _ = atol("_12", 3)

    with assert_raises(contains="Base must be >= 2 and <= 36, or 0."):
        _ = atol("0", 1)

    with assert_raises(contains="Base must be >= 2 and <= 36, or 0."):
        _ = atol("0", 37)

    with assert_raises(
        contains="String is not convertible to integer with base 16: '_ff'"
    ):
        _ = atol("_ff", base=16)

    with assert_raises(
        contains="String is not convertible to integer with base 2: '  _01'"
    ):
        _ = atol("  _01", base=2)

    with assert_raises(
        contains="String is not convertible to integer with base 0: '   '"
    ):
        _ = atol("   ", 0)

    with assert_raises(
        contains="String is not convertible to integer with base 10: '   '"
    ):
        _ = atol("   ")

    with assert_raises(
        contains="String is not convertible to integer with base 0: '   +'"
    ):
        _ = atol("   +", 0)

    with assert_raises(
        contains="String is not convertible to integer with base 0: '-'"
    ):
        _ = atol("-", 0)

    with assert_raises(
        contains="String is not convertible to integer with base 10: '0x_ff'"
    ):
        _ = atol("0x_ff")

    with assert_raises(
        contains="String is not convertible to integer with base 10: ''"
    ):
        _ = atol("")

    with assert_raises(
        contains="String expresses an integer too large to store in Int."
    ):
        _ = atol("9223372036854775832")

    # Overflow detection must be exact at the Int boundaries: values one past
    # Int.MAX / Int.MIN raise instead of silently wrapping.
    with assert_raises(
        contains="String expresses an integer too large to store in Int."
    ):
        _ = atol("9223372036854775808")

    with assert_raises(
        contains="String expresses an integer too large to store in Int."
    ):
        _ = atol("-9223372036854775809")

    assert_equal(Int.MAX, atol("9223372036854775807"))
    assert_equal(Int.MIN, atol("-9223372036854775808"))
    assert_equal(Int.MAX, atol("0x7FFF_FFFF_FFFF_FFFF", 16))
    assert_equal(Int.MIN, atol("-0x8000_0000_0000_0000", 16))


def test_atol_base_0() raises:
    assert_equal(155, atol(" 155", base=0))
    assert_equal(155_155, atol("155_155 ", base=0))

    assert_equal(0, atol(" 0000", base=0))
    assert_equal(0, atol(" 000_000", base=0))

    assert_equal(3, atol("0b11", base=0))
    assert_equal(3, atol("0B1_1", base=0))

    assert_equal(63, atol("0o77", base=0))
    assert_equal(63, atol(" 0O7_7 ", base=0))

    assert_equal(17, atol("0x11", base=0))
    assert_equal(17, atol("0X1_1", base=0))

    assert_equal(0, atol("0X0", base=0))

    assert_equal(255, atol("0x_00_ff", base=0))

    assert_equal(18, atol("0b_0001_0010", base=0))
    assert_equal(18, atol("0b000_1001_0", base=0))

    assert_equal(10, atol("0o_000_12", base=0))
    assert_equal(10, atol("0o00_12", base=0))

    with assert_raises(
        contains="String is not convertible to integer with base 0: '  0x'"
    ):
        _ = atol("  0x", base=0)

    with assert_raises(
        contains="String is not convertible to integer with base 0: '  0b  '"
    ):
        _ = atol("  0b  ", base=0)

    with assert_raises(
        contains="String is not convertible to integer with base 0: '00100'"
    ):
        _ = atol("00100", base=0)

    with assert_raises(
        contains="String is not convertible to integer with base 0: '0r100'"
    ):
        _ = atol("0r100", base=0)

    with assert_raises(
        contains="String is not convertible to integer with base 0: '0xf__f'"
    ):
        _ = atol("0xf__f", base=0)

    with assert_raises(
        contains="String is not convertible to integer with base 0: '0of_'"
    ):
        _ = atol("0of_", base=0)


def test_atof() raises:
    assert_equal(375.0, atof("375.f"))
    assert_equal(1.0, atof("001."))
    assert_equal(+5.0, atof(" +005."))
    assert_equal(13.0, atof(" 013.f  "))
    assert_equal(-89.0, atof("-89"))
    assert_equal(-0.3, atof(" -0.3"))
    assert_equal(-69e3, atof(" -69E+3  "))
    assert_equal(123.2e1, atof(" 123.2E1  "))
    assert_equal(23e3, atof(" 23E3  "))
    assert_equal(989343e-13, atof(" 989343E-13  "))
    assert_equal(1.123, atof(" 1.123f"))
    assert_equal(0.78, atof(" .78 "))
    assert_equal(121234.0, atof(" 121234.  "))
    assert_equal(985031234.0, atof(" 985031234.F  "))
    assert_equal(FloatLiteral.negative_zero, atof("-0"))
    assert_equal(String(FloatLiteral.nan), String(atof("  nan")))
    assert_equal(FloatLiteral.infinity, atof(" inf "))
    assert_equal(FloatLiteral.negative_infinity, atof("-inf  "))

    # Tests for scientific notation bug fix (using buff[pos] instead of buff[start])
    assert_equal(
        1.23e-2, atof("1.23e-2")
    )  # Previously failed due to wrong buffer indexing
    assert_equal(
        4.56e2, atof("4.56e+2")
    )  # Previously failed due to wrong buffer indexing

    # Tests for case-insensitive NaN and infinity
    assert_true(isnan(atof("NaN")))
    assert_true(isnan(atof("nan")))
    assert_true(isinf(atof("Inf")))
    assert_true(isinf(atof("INFINITY")))
    assert_true(isinf(atof("infinity")))
    assert_true(isinf(atof("-INFINITY")))

    # Tests for leading decimal point (no digits before decimal)
    assert_equal(0.123, atof(".123"))
    assert_equal(-0.123, atof("-.123"))
    assert_equal(0.123, atof("+.123"))

    # Tests for large exponents (overflow handling)
    assert_equal(
        FloatLiteral.infinity, atof("1e309")
    )  # Overflows double precision
    assert_equal(0.0, atof("1e-325"))  # Underflows to zero

    # Negative cases
    with assert_raises(contains="String is not convertible to float: ''"):
        _ = atof("")

    with assert_raises(
        contains=(
            "String is not convertible to float: ' 123 asd'. "
            "The last character of '123 asd' should be a "
            "digit or dot to convert it to a float."
        )
    ):
        _ = atof(" 123 asd")

    with assert_raises(
        contains=(
            "String is not convertible to float: ' f.9123 '. "
            "The first character of 'f.9123' should be a "
            "digit or dot to convert it to a float."
        )
    ):
        _ = atof(" f.9123 ")

    with assert_raises(
        contains=(
            "String is not convertible to float: ' 989343E-1A3 '. "
            "Invalid character(s) in the number: '989343E-1A3'"
        )
    ):
        _ = atof(" 989343E-1A3 ")

    with assert_raises(
        contains=(
            "String is not convertible to float: ' 124124124_2134124124 '. "
            "Invalid character(s) in the number: '124124124_2134124124'"
        )
    ):
        _ = atof(" 124124124_2134124124 ")

    with assert_raises(
        contains=(
            "String is not convertible to float: ' 123.2E '. "
            "The last character of '123.2E' should be a digit "
            "or dot to convert it to a float."
        )
    ):
        _ = atof(" 123.2E ")

    with assert_raises(
        contains=(
            "String is not convertible to float: ' --958.23 '. "
            "The first character of '-958.23' should be a digit "
            "or dot to convert it to a float."
        )
    ):
        _ = atof(" --958.23 ")

    with assert_raises(
        contains=(
            "String is not convertible to float: ' ++94. '. "
            "The first character of '+94.' should be "
            "a digit or dot to convert it to a float."
        )
    ):
        _ = atof(" ++94. ")

    with assert_raises(contains="String is not convertible to float"):
        _ = atof(".")  # Just a decimal point with no digits

    with assert_raises(contains="String is not convertible to float"):
        _ = atof("e5")  # Exponent with no mantissa


def test_calc_initial_buffer_size_int32() raises:
    assert_equal(1, _calc_initial_buffer_size_int32(0))
    assert_equal(1, _calc_initial_buffer_size_int32(9))
    assert_equal(2, _calc_initial_buffer_size_int32(10))
    assert_equal(2, _calc_initial_buffer_size_int32(99))
    assert_equal(8, _calc_initial_buffer_size_int32(99999999))
    assert_equal(9, _calc_initial_buffer_size_int32(100000000))
    assert_equal(9, _calc_initial_buffer_size_int32(999999999))
    assert_equal(10, _calc_initial_buffer_size_int32(1000000000))
    assert_equal(10, _calc_initial_buffer_size_int32(4294967295))


def test_calc_initial_buffer_size_int64() raises:
    assert_equal(1, _calc_initial_buffer_size_int64(0))
    assert_equal(1, _calc_initial_buffer_size_int64(9))
    assert_equal(2, _calc_initial_buffer_size_int64(10))
    assert_equal(2, _calc_initial_buffer_size_int64(99))
    assert_equal(9, _calc_initial_buffer_size_int64(999999999))
    assert_equal(10, _calc_initial_buffer_size_int64(1000000000))
    assert_equal(10, _calc_initial_buffer_size_int64(9999999999))
    assert_equal(11, _calc_initial_buffer_size_int64(10000000000))
    assert_equal(20, _calc_initial_buffer_size_int64(18446744073709551615))


def test_contains() raises:
    var str = "Hello world"

    assert_true(str.__contains__(""))
    assert_true(str.__contains__("He"))
    assert_true("lo" in str)
    assert_true(str.__contains__(" "))
    assert_true(str.__contains__("ld"))

    assert_false(str.__contains__("below"))
    assert_true("below" not in str)


def test_find() raises:
    var str = "Hello world"

    assert_equal(0, str.find(""))
    assert_equal(0, str.find("Hello"))
    assert_equal(2, str.find("llo"))
    assert_equal(6, str.find("world"))
    assert_equal(-1, str.find("universe"))

    # Test find() offset is absolute, not relative (issue mojo/#1355)
    var str2 = "...a"
    assert_equal(3, str2.find("a", 0))
    assert_equal(3, str2.find("a", 1))
    assert_equal(3, str2.find("a", 2))
    assert_equal(3, str2.find("a", 3))

    # Test find() support for negative start positions
    assert_equal(4, str.find("o", -10))
    assert_equal(7, str.find("o", -5))

    assert_equal(-1, String("abc").find("abcd"))


def test_replace() raises:
    # Replace empty
    var s1 = "abc"
    assert_equal("xaxbxc", s1.replace("", "x"))
    assert_equal("->a->b->c", s1.replace("", "->"))

    var s2 = "Hello Python"
    assert_equal("Hello Mojo", s2.replace("Python", "Mojo"))
    assert_equal("HELLo Python", s2.replace("Hell", "HELL"))
    assert_equal("Hello Python", s2.replace("HELL", "xxx"))
    assert_equal("HellP oython", s2.replace("o P", "P o"))
    assert_equal("Hello Pything", s2.replace("thon", "thing"))
    assert_equal("He||o Python", s2.replace("ll", "||"))
    assert_equal("He--o Python", s2.replace("l", "-"))
    assert_equal("He-x--x-o Python", s2.replace("l", "-x-"))

    var s3 = "a   complex  test case  with some  spaces"
    assert_equal("a  complex test case with some spaces", s3.replace("  ", " "))


def test_rfind() raises:
    var s1 = "hello world"
    # Basic usage.
    assert_equal(s1.rfind("world"), 6)
    assert_equal(s1.rfind("bye"), -1)

    # Repeated substrings.
    assert_equal(String("ababab").rfind("ab"), 4)

    # Empty string and substring.
    assert_equal(String().rfind("ab"), -1)
    assert_equal(String("foo").rfind(""), 3)

    # Test that rfind(start) returned pos is absolute, not relative to specified
    # start. Also tests positive and negative start offsets.
    assert_equal(s1.rfind("l", 5), 9)
    assert_equal(s1.rfind("l", -5), 9)
    assert_equal(s1.rfind("w", -3), -1)
    assert_equal(s1.rfind("w", -5), 6)

    assert_equal(-1, String("abc").rfind("abcd"))

    # Special characters.
    # TODO(#26444): Support unicode strings.
    # assert_equal(String("こんにちは").rfind("にち"), 2)
    # assert_equal(String("🔥🔥").rfind("🔥"), 1)


def test_split() raises:
    comptime S = StaticString

    # Should add all whitespace-like chars as one
    # test all unicode separators
    var next_line: List[UInt8] = [0xC2, 0x85]
    var unicode_line_sep: List[UInt8] = [0xE2, 0x80, 0xA8]
    var unicode_paragraph_sep: List[UInt8] = [0xE2, 0x80, 0xA9]
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
    comptime S = String
    comptime L = List[StaticString]

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
    assert_equal(S("").splitlines(), L())
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

    for u in [next_line^, unicode_line_sep^, unicode_paragraph_sep^]:
        var item = StaticString("").join(
            ["hello", u, "world", u, "mojo", u, "language", u]
        )
        assert_equal(item.splitlines(), hello_mojo)
        assert_equal(
            _to_string_list(item.splitlines(keepends=True)),
            ["hello" + u, "world" + u, "mojo" + u, "language" + u],
        )


def test_isspace() raises:
    assert_false(String().isspace())

    # test all utf8 and unicode separators
    var next_line: List[UInt8] = [0xC2, 0x85]
    var unicode_line_sep: List[UInt8] = [0xE2, 0x80, 0xA8]
    var unicode_paragraph_sep: List[UInt8] = [0xE2, 0x80, 0xA9]
    # TODO add line and paragraph separator as StringLiteral once unicode
    # escape sequences are accepted
    var univ_sep_var = [
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
    ]

    for i in univ_sep_var:
        assert_true(i.isspace())

    for i in [String("not"), "space", "", "s", "a", "c"]:
        assert_false(i.isspace())

    for sep1 in univ_sep_var:
        var sep = String()
        for sep2 in univ_sep_var:
            sep += sep1
            sep += sep2
        assert_true(sep.isspace())
        _ = sep


def test_ascii_aliases() raises:
    assert_true("a" in String.ASCII_LOWERCASE)
    assert_true("b" in String.ASCII_LOWERCASE)
    assert_true("y" in String.ASCII_LOWERCASE)
    assert_true("z" in String.ASCII_LOWERCASE)

    assert_true("A" in String.ASCII_UPPERCASE)
    assert_true("B" in String.ASCII_UPPERCASE)
    assert_true("Y" in String.ASCII_UPPERCASE)
    assert_true("Z" in String.ASCII_UPPERCASE)

    assert_true("a" in String.ASCII_LETTERS)
    assert_true("b" in String.ASCII_LETTERS)
    assert_true("y" in String.ASCII_LETTERS)
    assert_true("z" in String.ASCII_LETTERS)
    assert_true("A" in String.ASCII_LETTERS)
    assert_true("B" in String.ASCII_LETTERS)
    assert_true("Y" in String.ASCII_LETTERS)
    assert_true("Z" in String.ASCII_LETTERS)

    assert_true("0" in String.DIGITS)
    assert_true("9" in String.DIGITS)

    assert_true("0" in String.HEX_DIGITS)
    assert_true("9" in String.HEX_DIGITS)
    assert_true("A" in String.HEX_DIGITS)
    assert_true("F" in String.HEX_DIGITS)

    assert_true("7" in String.OCT_DIGITS)
    assert_false("8" in String.OCT_DIGITS)

    assert_true("," in String.PUNCTUATION)
    assert_true("." in String.PUNCTUATION)
    assert_true("\\" in String.PUNCTUATION)
    assert_true("@" in String.PUNCTUATION)
    assert_true('"' in String.PUNCTUATION)
    assert_true("'" in String.PUNCTUATION)

    var text = "I love my Mom and Dad so much!!!\n"
    for cp in text.codepoint_slices():
        assert_true(cp in String.PRINTABLE)


def test_rstrip() raises:
    # with default rstrip chars
    var empty_string = String()
    assert_true(empty_string.rstrip() == "")

    var space_string = " \t\n\r\v\f  "
    assert_true(space_string.rstrip() == "")

    var str0 = "     n "
    assert_true(str0.rstrip() == "     n")

    var str1 = "string"
    assert_true(str1.rstrip() == "string")

    var str2 = "something \t\n\t\v\f"
    assert_true(str2.rstrip() == "something")

    # with custom chars for rstrip
    var str3 = "mississippi"
    assert_true(str3.rstrip("sip") == "m")

    var str4 = "mississippimississippi \n "
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
    var empty_string = String()
    assert_true(empty_string.lstrip() == "")

    var space_string = " \t\n\r\v\f  "
    assert_true(space_string.lstrip() == "")

    var str0 = "     n "
    assert_true(str0.lstrip() == "n ")

    var str1 = "string"
    assert_true(str1.lstrip() == "string")

    var str2 = " \t\n\t\v\fsomething"
    assert_true(str2.lstrip() == "something")

    # with custom chars for lstrip
    var str3 = "mississippi"
    assert_true(str3.lstrip("mis") == "ppi")

    var str4 = " \n mississippimississippi"
    assert_true(str4.lstrip("mis ") == "\n mississippimississippi")
    assert_true(str4.lstrip("mis \n") == "ppimississippi")

    # should strip off single codepoints
    var str5 = "😀smile😀"
    assert_true(str5.lstrip("😀") == "smile😀")

    # Ñ and Ò share the leading utf-8 byte of 0xc3
    var str6 = "Ñeeee"
    assert_true(str6.lstrip("Ò") == "Ñeeee")


def test_strip() raises:
    # with default strip chars
    var empty_string = String()
    assert_true(empty_string.strip() == "")
    comptime comp_empty_string_stripped = String(String().strip())
    assert_true(comp_empty_string_stripped == "")

    var space_string = " \t\n\r\v\f  "
    assert_true(space_string.strip() == "")
    comptime comp_space_string_stripped = String(" \t\n\r\v\f  ".strip())
    assert_true(comp_space_string_stripped == "")

    var str0 = "     n "
    assert_true(str0.strip() == "n")
    comptime comp_str0_stripped = String("     n ".strip())
    assert_true(comp_str0_stripped == "n")

    var str1 = "string"
    assert_true(str1.strip() == "string")
    comptime comp_str1_stripped = String("string".strip())
    assert_true(comp_str1_stripped == "string")

    var str2 = " \t\n\t\v\fsomething \t\n\t\v\f"
    comptime comp_str2_stripped = String(
        " \t\n\t\v\fsomething \t\n\t\v\f".strip()
    )
    assert_true(str2.strip() == "something")
    assert_true(comp_str2_stripped == "something")

    # with custom strip chars
    var str3 = "mississippi"
    assert_true(str3.strip("mips") == "")
    assert_true(str3.strip("mip") == "ssiss")
    comptime comp_str3_stripped = String("mississippi".strip("mips"))
    assert_true(comp_str3_stripped == "")

    var str4 = " \n mississippimississippi \n "
    assert_true(str4.strip(" ") == "\n mississippimississippi \n")
    assert_true(str4.strip("\nmip ") == "ssissippimississ")

    comptime comp_str4_stripped = String(
        " \n mississippimississippi \n ".strip(" ")
    )
    assert_true(comp_str4_stripped == "\n mississippimississippi \n")

    # should strip off single codepoints
    var str5 = "😀smile😀"
    assert_true(str5.strip("😀") == "smile")

    # Ñ and Ò share the leading utf-8 byte of 0xc3
    var str6 = "ÑeeeeÑ"
    assert_true(str6.strip("Ò") == "ÑeeeeÑ")


def test_strip_mutable_chars() raises:
    # `chars` is read-only, so a mutable argument must not be borrowed mutably;
    # otherwise it can't alias `self`, which is an interior slice of the string.
    var chars = String("hi")
    assert_equal(String("himojohi").lstrip(chars), "mojohi")
    assert_equal(String("himojohi").rstrip(chars), "himojo")
    assert_equal(String("himojohi").strip(chars), "mojo")

    var self_aliasing = String("aabbaa")
    assert_equal(self_aliasing.lstrip(self_aliasing), "")
    assert_equal(self_aliasing.rstrip(self_aliasing), "")
    assert_equal(self_aliasing.strip(self_aliasing), "")


def test_hash() raises:
    def assert_hash_equals_literal_hash[s: StaticString]() raises:
        assert_equal(hash(s), hash(String(s)))

    assert_hash_equals_literal_hash["a"]()
    assert_hash_equals_literal_hash["b"]()
    assert_hash_equals_literal_hash["c"]()
    assert_hash_equals_literal_hash["d"]()
    assert_hash_equals_literal_hash["this is a longer string"]()
    assert_hash_equals_literal_hash[
        """
Blue: We have to take the amulet to the Banana King.
Charlie: Oh, yes, The Banana King, of course. ABSOLUTELY NOT!
Pink: He, he's counting on us, Charlie! (Pink starts floating) ah...
Blue: If we don't give the amulet to the Banana King, the vortex will open and let out a thousand years of darkness.
Pink: No! Darkness! (Pink is floating in the air)"""
    ]()


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


def test_removeprefix() raises:
    assert_equal(StaticString("hello world").removeprefix(""), "hello world")
    assert_equal(StaticString("hello world").removeprefix("hello"), " world")
    assert_equal(
        StaticString("hello world").removeprefix("world"), "hello world"
    )
    assert_equal(StaticString("hello world").removeprefix("hello world"), "")
    assert_equal(
        StaticString("hello world").removeprefix("llo wor"), "hello world"
    )


def test_removesuffix() raises:
    assert_equal(StaticString("hello world").removesuffix(""), "hello world")
    assert_equal(StaticString("hello world").removesuffix("world"), "hello ")
    assert_equal(
        StaticString("hello world").removesuffix("hello"), "hello world"
    )
    assert_equal(StaticString("hello world").removesuffix("hello world"), "")
    assert_equal(
        StaticString("hello world").removesuffix("llo wor"), "hello world"
    )


def test_intable() raises:
    assert_equal(Int("123"), 123)

    with assert_raises():
        _ = Int("hi")


def test_string_mul() raises:
    assert_equal("*" * 0, "")
    assert_equal("!" * 10, "!!!!!!!!!!")
    assert_equal("ab" * 5, "ababababab")


def test_indexing() raises:
    var a = "abc"
    assert_equal(a[byte=0], "a")
    assert_equal(a[byte=Int(1)], "b")
    assert_equal(a[byte=2], "c")


def test_string_codepoints_iter() raises:
    var s = "abc"
    var iter = s.codepoints()
    assert_equal(iter.__next__(), Codepoint.ord("a"))
    assert_equal(iter.__next__(), Codepoint.ord("b"))
    assert_equal(iter.__next__(), Codepoint.ord("c"))
    with assert_raises():
        _ = iter.__next__()  # raises StopIteration


def test_string_char_slices_iter() raises:
    var s0 = "abc"
    var s0_iter = s0.codepoint_slices()
    assert_true(s0_iter.__next__() == "a")
    assert_true(s0_iter.__next__() == "b")
    assert_true(s0_iter.__next__() == "c")
    with assert_raises():
        _ = s0_iter.__next__()  # raises StopIteration

    var vs = "123"

    # Borrow immutably
    def conc(vs: String) -> String:
        var c = String()
        for v in vs.codepoint_slices():
            c += v
        return c

    assert_equal(123, atol(conc(vs)))

    var concat = String()
    for v in vs.codepoint_slices_reversed():
        concat += v
    assert_equal(321, atol(concat))

    # Borrow immutably
    for v in vs.codepoint_slices():
        concat += v

    assert_equal(321123, atol(concat))

    vs = "mojo🔥"
    var iterator = vs.codepoint_slices()
    assert_equal(5, len(iterator))
    var item = iterator.__next__()
    assert_equal("m", String(item))
    assert_equal(4, len(iterator))
    item = iterator.__next__()
    assert_equal("o", String(item))
    assert_equal(3, len(iterator))
    item = iterator.__next__()
    assert_equal("j", String(item))
    assert_equal(2, len(iterator))
    item = iterator.__next__()
    assert_equal("o", String(item))
    assert_equal(1, len(iterator))
    item = iterator.__next__()
    assert_equal("🔥", String(item))
    assert_equal(0, len(iterator))

    var items: List[String] = [
        "mojo🔥",
        "السلام عليكم",
        "Dobrý den",
        "Hello",
        "שָׁלוֹם",
        "नमस्ते",
        "こんにちは",
        "안녕하세요",
        "你好",
        "Olá",
        "Здравствуйте",
    ]
    var rev: List[String] = [
        "🔥ojom",
        "مكيلع مالسلا",
        "ned ýrboD",
        "olleH",
        "םֹולָׁש",
        "ेत्समन",
        "はちにんこ",
        "요세하녕안",
        "好你",
        "álO",
        "етйувтсвардЗ",
    ]
    var items_amount_characters = [5, 12, 9, 5, 7, 6, 5, 5, 2, 3, 12]
    for item_idx in range(len(items)):
        var item = items[item_idx]
        var ptr = item.as_bytes().unsafe_ptr()
        var amnt_characters = 0
        var byte_idx = 0
        for v in item.codepoint_slices():
            var byte_len = v.byte_length()
            for i in range(byte_len):
                assert_equal(
                    ptr[unsafe_offset=byte_idx + i],
                    v.as_bytes().unsafe_ptr()[unsafe_offset=i],
                )
            byte_idx += byte_len
            amnt_characters += 1

        assert_equal(amnt_characters, items_amount_characters[item_idx])
        var concat = String()
        for v in item.codepoint_slices_reversed():
            concat += v
        assert_equal(rev[item_idx], concat)


def test_format_args() raises:
    with assert_raises(contains="Index -1 not in *args"):
        _ = String("{-1} {0}").format("First")

    with assert_raises(contains="Index 1 not in *args"):
        _ = String("A {0} B {1}").format("First")

    with assert_raises(contains="Index 1 not in *args"):
        _ = String("A {1} B {0}").format("First")

    with assert_raises(contains="Index 1 not in *args"):
        _ = String("A {1} B {0}").format()

    with assert_raises(
        contains="Automatic indexing require more args in *args"
    ):
        _ = String("A {} B {}").format("First")

    with assert_raises(
        contains="Cannot both use manual and automatic indexing"
    ):
        _ = String("A {} B {1}").format("First", "Second")

    with assert_raises(contains="Index first not in kwargs"):
        _ = String("A {first} B {second}").format(1, 2)

    var s = " {} , {} {} !".format("Hello", "Beautiful", "World")
    assert_equal(s, " Hello , Beautiful World !")

    def curly(c: StaticString) -> String:
        return "there is a single curly " + c + " left unclosed or unescaped"

    with assert_raises(contains=curly("{")):
        _ = String("{ {}").format(1)

    with assert_raises(contains=curly("{")):
        _ = String("{ {0}").format(1)

    with assert_raises(contains=curly("{")):
        _ = String("{}{").format(1)

    with assert_raises(contains=curly("}")):
        _ = String("{}}").format(1)

    with assert_raises(contains=curly("{")):
        _ = String("{} {").format(1)

    with assert_raises(contains=curly("{")):
        _ = String("{").format(1)

    with assert_raises(contains=curly("}")):
        _ = String("}").format(1)

    with assert_raises(contains=curly("}")):
        _ = String("hello}world").format(42)

    with assert_raises(contains=""):
        _ = String("{}").format()

    assert_equal("}}".format(), "}")
    assert_equal("{{".format(), "{")

    assert_equal("{{}}{}{{}}".format("foo"), "{}foo{}")

    assert_equal("{{ {0}".format("foo"), "{ foo")
    assert_equal("{{{0}".format("foo"), "{foo")
    assert_equal("{{0}}".format("foo"), "{0}")
    assert_equal("{{}}".format("foo"), "{}")
    assert_equal("{{0}}".format("foo"), "{0}")
    assert_equal("{{{0}}}".format("foo"), "{foo}")

    var vinput = "{} {}"
    var output = vinput.format("123", 456)
    assert_equal(output.byte_length(), 7)

    var vinput2 = "{1}{0}"
    output = vinput2.format("123", 456)
    assert_equal(output.byte_length(), 6)
    assert_equal(output, "456123")

    var vinput3 = "123"
    output = vinput3.format()
    assert_equal(output.byte_length(), 3)

    var vinput4 = ""
    output = vinput4.format()
    assert_equal(output.byte_length(), 0)

    var res = "🔥 Mojo ❤️‍🔥 Mojo 🔥"
    assert_equal("{0} {1} ❤️‍🔥 {1} {0}".format("🔥", "Mojo"), res)

    assert_equal("{0} {1}".format(True, 1.125), "True 1.125")

    assert_equal("{0} {1}".format("{1}", "Mojo"), "{1} Mojo")
    assert_equal("{0} {1} {0} {1}".format("{1}", "Mojo"), "{1} Mojo {1} Mojo")


def test_format_conversion_flags() raises:
    assert_equal("{!r}".format(""), "''")
    var special_str = "a\nb\tc"
    assert_equal(
        "{} {!r}".format(special_str, special_str),
        "a\nb\tc 'a\\nb\\tc'",
    )
    assert_equal(
        "{!s} {!r}".format(special_str, special_str),
        "a\nb\tc 'a\\nb\\tc'",
    )

    var a = "Mojo"
    assert_equal("{} {!r}".format(a, a), "Mojo 'Mojo'")
    assert_equal("{!s} {!r}".format(a, a), "Mojo 'Mojo'")
    assert_equal("{0!s} {0!r}".format(a), "Mojo 'Mojo'")
    assert_equal("{0!s} {0!r}".format(a, "Mojo2"), "Mojo 'Mojo'")

    var b = 21.1
    assert_true(
        "21.1 Float64(2" in "{} {!r}".format(b, b),
    )
    assert_true(
        "21.1 Float64(2" in "{!s} {!r}".format(b, b),
    )

    var c = 1e100
    assert_equal(
        "{} {!r}".format(c, c),
        "1e+100 Float64(1e+100)",
    )
    assert_equal(
        "{!s} {!r}".format(c, c),
        "1e+100 Float64(1e+100)",
    )

    var d = 42
    assert_equal("{} {!r}".format(d, d), "42 Int(42)")
    assert_equal("{!s} {!r}".format(d, d), "42 Int(42)")

    assert_true("Mojo Float64(2" in "{} {!r} {} {!r}".format(a, b, c, d))
    assert_true("Mojo Float64(2" in "{!s} {!r} {!s} {!r}".format(a, b, c, d))

    var e = True
    assert_equal("{} {!r}".format(e, e), "True True")

    assert_true("Mojo Float64(2" in "{0} {1!r} {2} {3}".format(a, b, c, d))
    assert_true("Mojo Float64(2" in "{0!s} {1!r} {2} {3!s}".format(a, b, c, d))

    assert_equal(
        "{3} {2} {1} {0}".format(a, d, c, b),
        "21.1 1e+100 42 Mojo",
    )

    assert_true("'Mojo' 42 Float64(2" in "{0!r} {3} {1!r}".format(a, b, c, d))

    assert_true(
        "True 'Mojo' 42 Float64(2"
        in "{4} {0!r} {3} {1!r}".format(a, b, c, d, True)
    )

    with assert_raises(contains='Conversion flag "x" not recognized.'):
        _ = String("{!x}").format(1)

    with assert_raises(contains="Empty conversion flag."):
        _ = String("{!}").format(1)

    with assert_raises(contains='Conversion flag "rs" not recognized.'):
        _ = String("{!rs}").format(1)

    with assert_raises(contains='Conversion flag "r123" not recognized.'):
        _ = String("{!r123}").format(1)

    with assert_raises(contains='Conversion flag "r!" not recognized.'):
        _ = String("{!r!}").format(1)

    with assert_raises(contains='Conversion flag "x" not recognized.'):
        _ = String("{0!x}").format(1)

    with assert_raises(contains='Conversion flag "r:d" not recognized.'):
        _ = String("{!r:d}").format(1)


def test_float_conversion() raises:
    # This is basically just a wrapper around atof which is
    # more throughouly tested above
    assert_equal(String("4.5").__float__(), 4.5)
    assert_equal(Float64("4.5"), 4.5)
    with assert_raises():
        _ = Float64("not a float")


def test_slice_contains() raises:
    assert_true(StringSlice("hello world").__contains__("world"))
    assert_false(StringSlice("hello world").__contains__("not-found"))


def test_reserve() raises:
    var s = String()
    assert_equal(s.capacity_bytes(), 23)
    s.reserve_bytes(61)
    assert_equal(s.capacity_bytes(), 64)


def test_resize() raises:
    var s = String()
    s.resize(100, 0)
    for c in s.codepoints():
        assert_equal(c, Codepoint(0))
    var s2 = String("😀😃")
    s2.resize(4)
    assert_equal(s2, "😀")


def test_uninit_ctor() raises:
    var hello_len = "hello".byte_length()
    var s = String(unsafe_uninit_length=hello_len)
    unsafe_memcpy(
        dest=s.unsafe_as_bytes_mut().unsafe_ptr(),
        src=StaticString("hello").as_bytes().unsafe_ptr(),
        count=hello_len,
    )
    assert_equal(s, "hello")

    # Resize with uninitialized memory.
    var s2 = String()
    s2.resize(unsafe_uninit_length=hello_len)
    unsafe_memcpy(
        dest=s2.unsafe_as_bytes_mut().unsafe_ptr(),
        src=StaticString("hello").as_bytes().unsafe_ptr(),
        count=hello_len,
    )
    assert_equal(s2, "hello")
    assert_equal(s2._is_inline(), True)

    var s3 = String()
    var long: StaticString = "hellohellohellohellohellohellohellohellohellohel"
    s3.resize(unsafe_uninit_length=long.byte_length())
    unsafe_memcpy(
        dest=s3.unsafe_as_bytes_mut().unsafe_ptr(),
        src=long.as_bytes().unsafe_ptr(),
        count=long.byte_length(),
    )
    assert_equal(s3, long)
    assert_equal(s3._is_inline(), False)


def test_as_c_string_slice_empty() raises:
    var string = String()
    var cslice = string.as_c_string_slice()
    assert_equal(string.byte_length(), 0)
    assert_true(string.capacity_bytes() > 0)
    # Safe to index `string.byte_length()` as this has a nul terminator
    assert_equal(
        string.as_bytes().unsafe_ptr()[unsafe_offset=string.byte_length()], 0
    )
    assert_true(string.as_bytes() == cslice.as_bytes())


def test_as_c_string_slice_inlined() raises:
    var string = String("a")
    var cslice = string.as_c_string_slice()
    # Safe to index `string.byte_length()` as this has a nul terminator
    assert_equal(
        string.as_bytes().unsafe_ptr()[unsafe_offset=string.byte_length()], 0
    )
    assert_true(string.as_bytes() == cslice.as_bytes())


def test_as_c_string_slice_heap() raises:
    var string = String("abcdefghijlmnopqrstuvwxyz")
    var cslice = string.as_c_string_slice()
    # Safe to index `string.byte_length()` as this has a nul terminator
    assert_equal(
        string.as_bytes().unsafe_ptr()[unsafe_offset=string.byte_length()], 0
    )
    assert_true(string.as_bytes() == cslice.as_bytes())


def test_variadic_ctors() raises:
    var s = String("message", 42, 42.2, True, sep=", ")
    assert_equal(s, "message, 42, 42.2, True")

    def forward_variadic_pack[
        *Ts: Writable,
    ](*args: *Ts) -> String:
        return String(*args)

    var s3 = forward_variadic_pack(1, ", ", 2.0, ", ", "three")
    assert_equal(s3, "1, 2.0, three")


def test_sso() raises:
    # String literals are initially stored as indirect regardless of length
    var s = "hello"
    assert_equal(s.capacity_bytes(), 5)
    assert_equal(s.byte_length(), 5)
    assert_equal(s._is_inline(), False)
    assert_equal(s._has_nul_terminator(), True)
    assert_equal(s.as_bytes().unsafe_ptr()[unsafe_offset=s.byte_length()], 0)

    # Adding a single char should remove the nul terminator and inline it.
    s += "f"
    assert_equal(s.byte_length(), 6)
    assert_equal(s.capacity_bytes(), String.INLINE_CAPACITY)
    assert_equal(s._is_inline(), True)
    assert_equal(s._has_nul_terminator(), False)
    assert_equal(s, "hellof")

    # Test StringLiterals behave the same when above SSO capacity.
    comptime long = "hellohellohellohellohellohellohellohellohellohellohello"
    s = String(long)
    assert_equal(s.byte_length(), 55)
    assert_equal(s.capacity_bytes(), 55)
    assert_equal(s._is_inline(), False)
    assert_equal(s._has_nul_terminator(), True)
    assert_equal(s.as_bytes().unsafe_ptr()[unsafe_offset=s.byte_length()], 0)

    # Modifying it should remove the nul terminator.
    s += "f"
    assert_equal(s.byte_length(), 56)
    assert_true(s.capacity_bytes() >= 56)
    assert_equal(s._is_inline(), False)
    assert_equal(s._has_nul_terminator(), False)
    assert_equal(s, long + "f")

    # Empty strings are stored inline.
    s = String()
    assert_equal(s.capacity_bytes(), String.INLINE_CAPACITY)
    assert_equal(s._is_inline(), True)
    assert_equal(s._has_nul_terminator(), False)

    s += "f" * String.INLINE_CAPACITY
    assert_equal(s.byte_length(), String.INLINE_CAPACITY)
    assert_equal(s.capacity_bytes(), String.INLINE_CAPACITY)
    assert_equal(s._is_inline(), True)

    # One more byte.
    s += "f"

    # The capacity should be 2x the previous amount, rounded up to 8.
    comptime expected_capacity = (String.INLINE_CAPACITY * 2 + 7) & ~7
    assert_equal(s.capacity_bytes(), Int(expected_capacity))
    assert_equal(s._is_inline(), False)

    # Shrink down to small, but stays out of line to maintain our malloc.
    s.resize(4)
    assert_equal(s.capacity_bytes(), Int(expected_capacity))
    assert_equal(s._is_inline(), False)
    assert_equal(s, "ffff")

    # Copying the small out-of-line string should just bump the count.
    var s2 = s.copy()
    assert_equal(s2._is_inline(), False)
    assert_equal(s2, "ffff")

    # Stringizing short things should be inline.
    s = String(42)
    assert_equal(s, "42")
    assert_equal(s._is_inline(), True)


def test_copyinit() raises:
    comptime sizes = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    assert_equal(len(sizes), 10)
    var test_current_size = 1

    comptime for sizes_index in range(len(sizes)):
        comptime current_size = rebind[Int](sizes[sizes_index])
        var x = ""
        for i in range(current_size):
            x += String(i)[byte=0]
        var y = x
        assert_equal(test_current_size, current_size)
        assert_equal(y.byte_length(), current_size)
        # TODO: check pointer equality?
        test_current_size *= 2
    assert_equal(test_current_size, 1024)


def test_from_utf8_lossy() raises:
    # Test empty byte span
    var empty = String(from_utf8_lossy=List[Byte]())
    assert_equal(empty, "")
    assert_equal(empty.byte_length(), 0)

    # Test single ASCII character
    var single_char = String(from_utf8_lossy=[Byte(0x41)])
    assert_equal(single_char, "A")
    assert_equal(single_char.byte_length(), 1)

    # Test valid 2-byte sequence
    var two_byte = String(from_utf8_lossy=[Byte(0xC2), 0xA9])
    assert_equal(two_byte, "©")

    # Test valid 3-byte sequence
    var three_byte = String(from_utf8_lossy=[Byte(0xE2), 0x82, 0xAC])
    assert_equal(three_byte, "€")

    # Test valid 4-byte sequence
    var four_byte = String(from_utf8_lossy=[Byte(0xF0), 0x9F, 0x94, 0xA5])
    assert_equal(four_byte, "🔥")

    def replacement_bytes() -> List[Byte]:
        # The byte sequence for the replacement character '�'
        return [Byte(0xEF), 0xBF, 0xBD]

    # Test invalid sequence at the beginning
    var invalid_start = String(from_utf8_lossy=[Byte(0xFF), 0x61, 0x62, 0x63])
    assert_equal(invalid_start, "�abc")
    assert_true(
        invalid_start.as_bytes()
        == replacement_bytes() + [Byte(0x61), 0x62, 0x63]
    )

    # Test invalid sequence at the end
    var invalid_end = String(
        from_utf8_lossy=[Byte(0x6D), 0x6F, 0x6A, 0x6F, 0xF0, 0x90, 0x80]
    )
    assert_equal(invalid_end, "mojo�")
    assert_true(
        invalid_end.as_bytes()
        == List([Byte(0x6D), 0x6F, 0x6A, 0x6F]) + replacement_bytes()
    )

    # Test invalid sequence in the middle
    var invalid_middle = String(
        from_utf8_lossy=[Byte(0x61), 0x62, 0xC0, 0xAF, 0x63, 0x64]
    )
    assert_equal(invalid_middle, "ab��cd")
    assert_true(
        invalid_middle.as_bytes()
        == List([Byte(0x61), 0x62])
        + replacement_bytes()
        + replacement_bytes()
        + [Byte(0x63), 0x64]
    )

    # Test multiple invalid sequences
    var multiple_invalid = String(from_utf8_lossy=[Byte(0xFF), 0xFE, 0xFD])
    assert_equal(multiple_invalid, "���")
    assert_true(
        multiple_invalid.as_bytes()
        == replacement_bytes() + replacement_bytes() + replacement_bytes()
    )

    # Test valid text followed by invalid sequence
    var valid_then_invalid = String(
        from_utf8_lossy=[Byte(0x48), 0x65, 0x6C, 0x6C, 0x6F, 0xC0]
    )
    assert_equal(valid_then_invalid, "Hello�")
    assert_true(
        valid_then_invalid.as_bytes()
        == List([Byte(0x48), 0x65, 0x6C, 0x6C, 0x6F]) + replacement_bytes()
    )

    # Test invalid sequence followed by valid text
    var invalid_then_valid = String(
        from_utf8_lossy=[Byte(0xC0), 0x48, 0x65, 0x6C, 0x6C, 0x6F]
    )
    assert_equal(invalid_then_valid, "�Hello")
    assert_true(
        invalid_then_valid.as_bytes()
        == replacement_bytes() + [Byte(0x48), 0x65, 0x6C, 0x6C, 0x6F]
    )

    # Test overlong encoding (2-byte when 1-byte would suffice) - should be invalid
    var overlong_2byte = String(from_utf8_lossy=[Byte(0xC0), 0x80])
    assert_equal(overlong_2byte, "��")
    assert_true(
        overlong_2byte.as_bytes() == replacement_bytes() + replacement_bytes()
    )

    # Test continuation byte without start byte
    var orphan_continuation = String(from_utf8_lossy=[Byte(0x61), 0x80, 0x62])
    assert_equal(orphan_continuation, "a�b")
    assert_true(
        orphan_continuation.as_bytes()
        == List([Byte(0x61)]) + replacement_bytes() + [Byte(0x62)]
    )

    # Test truncated 2-byte sequence
    var truncated_2byte = String(from_utf8_lossy=[Byte(0x61), 0xC2])
    assert_equal(truncated_2byte, "a�")
    assert_true(
        truncated_2byte.as_bytes() == List([Byte(0x61)]) + replacement_bytes()
    )

    # Test truncated 3-byte sequence
    var truncated_3byte = String(from_utf8_lossy=[Byte(0x61), 0xE2, 0x82])
    assert_equal(truncated_3byte, "a�")
    assert_true(
        truncated_3byte.as_bytes() == List([Byte(0x61)]) + replacement_bytes()
    )

    # Test mixed valid and invalid with multi-byte characters
    var mixed = String(
        from_utf8_lossy=[
            Byte(0x48),
            0x65,
            0x6C,
            0x6C,
            0x6F,  # "Hello"
            0x20,  # space
            0xF0,
            0x9F,
            0x94,
            0xA5,  # 🔥
            0xFF,  # invalid
            0x20,  # space
            0xE2,
            0x82,
            0xAC,  # €
        ]
    )
    assert_equal(mixed, "Hello 🔥� €")
    assert_true(
        mixed.as_bytes()
        == List([Byte(0x48), 0x65, 0x6C, 0x6C, 0x6F])  # "Hello"
        + [Byte(0x20)]  # space
        + [Byte(0xF0), 0x9F, 0x94, 0xA5]  # 🔥
        + replacement_bytes()  # replacement character
        + [Byte(0x20)]  # space
        + [Byte(0xE2), 0x82, 0xAC]  # €
    )


def test_from_utf8() raises:
    # Test empty byte span
    var empty = String(from_utf8=List[Byte]())
    assert_equal(empty, "")
    assert_equal(empty.byte_length(), 0)

    # Test single ASCII character
    var single_char = String(from_utf8=[Byte(0x41)])
    assert_equal(single_char, "A")
    assert_equal(single_char.byte_length(), 1)

    # Test valid 4-byte sequence
    var four_byte = String(from_utf8=[Byte(0xF0), 0x9F, 0x94, 0xA5])
    assert_equal(four_byte, "🔥")

    var invalid_sequences: List[List[Byte]] = [
        [Byte(0xFF), 0x61, 0x62, 0x63],  # Invalid byte at the beginning
        [Byte(0x6D), 0x6F, 0x6A, 0x6F, 0xFF],  # Invalid byte at the end
        [Byte(0x61), 0x62, 0xFF, 0x63, 0x64],  # Invalid byte in the middle
        [Byte(0x61), 0xC2],  # Truncated 2-byte sequence
        [Byte(0x61), 0xE2, 0x82],  # Truncated 3-byte sequence
        [Byte(0x61), 0xF0, 0x9F, 0x94],  # Truncated 4-byte sequence
        [
            Byte(0xC0),
            0x80,
        ],  # Overlong encoding (2-byte when 1-byte would suffice)
        [Byte(0x61), 0x80, 0x62],  # Continuation byte without start byte
        [Byte(0xC2), 0x00],  # Invalid continuation byte
    ]

    for sequence in invalid_sequences:
        with assert_raises():
            _ = String(from_utf8=Span(sequence))


def test_append_codepoint() raises:
    var s = String()
    s.append(Codepoint.ord("a"))
    assert_equal(s, "a")
    assert_equal(s.byte_length(), 1)

    s.append(Codepoint.ord("€"))
    assert_equal(s, "a€")
    assert_equal(s.byte_length(), 4)

    s.append(Codepoint.ord("🔥"))
    assert_equal(s, "a€🔥")
    assert_equal(s.byte_length(), 8)


def test_capitalize() raises:
    # Basic
    assert_equal(String("hello world").capitalize(), "Hello world")
    # All upper
    assert_equal(String("HELLO").capitalize(), "Hello")
    # Mixed
    assert_equal(String("hELLO").capitalize(), "Hello")
    # Empty
    assert_equal(String("").capitalize(), "")
    # Non-alpha first char
    assert_equal(String("123abc").capitalize(), "123abc")
    # Single char
    assert_equal(String("a").capitalize(), "A")


def test_title() raises:
    # Basic
    assert_equal(String("hello world").title(), "Hello World")
    # Already title case
    assert_equal(String("Hello World").title(), "Hello World")
    # All caps
    assert_equal(String("HELLO WORLD").title(), "Hello World")
    # Empty
    assert_equal(String("").title(), "")
    # Hyphenated
    assert_equal(String("hello-world").title(), "Hello-World")
    # Apostrophe (Python behavior: apostrophe is non-alpha, so char after it is uppercased)
    assert_equal(String("it's a test").title(), "It'S A Test")
    # Digits
    assert_equal(String("123abc def").title(), "123Abc Def")
    # Single word
    assert_equal(String("hello").title(), "Hello")


def test_string_zero_init() raises:
    var s = String()
    assert_equal(Int(s._ptr_or_data), 0)
    assert_equal(s._len_or_data, 0)
    assert_equal(s._capacity_or_data, String.FLAG_IS_INLINE)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
