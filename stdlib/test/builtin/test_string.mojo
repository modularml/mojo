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
# RUN: %bare-mojo %s

# TODO: Replace %bare-mojo with %mojo
# when  https://github.com/modularml/mojo/issues/2751 is fixed.
from builtin.string import (
    _calc_initial_buffer_size_int32,
    _calc_initial_buffer_size_int64,
    _isspace,
)
from python import Python
from testing import (
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_raises,
    assert_true,
)

from utils import StringRef


@value
struct AString(Stringable):
    fn __str__(self: Self) -> String:
        return "a string"


def test_stringable():
    assert_equal(str("hello"), "hello")
    assert_equal(str(0), "0")
    assert_equal(str(StringRef("AAA")), "AAA")
    assert_equal(str(AString()), "a string")


def test_repr():
    # Usual cases
    assert_equal(String.__repr__("hello"), "'hello'")
    assert_equal(String.__repr__(str(0)), "'0'")

    # Escape cases
    assert_equal(String.__repr__("\0"), r"'\x00'")
    assert_equal(String.__repr__("\x06"), r"'\x06'")
    assert_equal(String.__repr__("\x09"), r"'\t'")
    assert_equal(String.__repr__("\n"), r"'\n'")
    assert_equal(String.__repr__("\x0d"), r"'\r'")
    assert_equal(String.__repr__("\x0e"), r"'\x0e'")
    assert_equal(String.__repr__("\x1f"), r"'\x1f'")
    assert_equal(String.__repr__(" "), "' '")
    assert_equal(String.__repr__("'"), '"\'"')
    assert_equal(String.__repr__("A"), "'A'")
    assert_equal(String.__repr__("\\"), r"'\\'")
    assert_equal(String.__repr__("~"), "'~'")
    assert_equal(String.__repr__("\x7f"), r"'\x7f'")


def test_constructors():
    # Default construction
    assert_equal(0, len(String()))
    assert_true(not String())

    # Construction from Int
    var s0 = str(0)
    assert_equal(str(0), "0")
    assert_equal(len(s0), 1)

    var s1 = str(123)
    assert_equal(str(123), "123")
    assert_equal(len(s1), 3)

    # Construction from StringLiteral
    var s2 = String("abc")
    assert_equal(str(s2), "abc")
    assert_equal(len(s2), 3)

    # Construction from UnsafePointer
    var ptr = UnsafePointer[UInt8].alloc(4)
    ptr[0] = ord("a")
    ptr[1] = ord("b")
    ptr[2] = ord("c")
    ptr[3] = 0
    var s3 = String(ptr, 4)
    assert_equal(s3, "abc")


def test_copy():
    var s0 = String("find")
    var s1 = str(s0)
    s1._buffer[3] = ord("e")
    assert_equal(s0, "find")
    assert_equal(s1, "fine")


def test_equality_operators():
    var s0 = String("abc")
    var s1 = String("def")
    assert_equal(s0, s0)
    assert_not_equal(s0, s1)

    var s2 = String("abc")
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


def test_comparison_operators():
    var abc = String("abc")
    var de = String("de")
    var ABC = String("ABC")
    var ab = String("ab")
    var abcd = String("abcd")

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


def test_add():
    var s1 = String("123")
    var s2 = String("abc")
    var s3 = s1 + s2
    assert_equal(s3, "123abc")

    var s4 = String("x")
    var s5 = s4.join(1, 2, 3)
    assert_equal(s5, "1x2x3")

    var s6 = s4.join(s1, s2)
    assert_equal(s6, "123xabc")

    var s7 = String()
    assert_equal(s2 + s7, "abc")

    assert_equal(s2 + "def", "abcdef")
    assert_equal("123" + s2, "123abc")

    var s8 = String("abc is ")
    var s9 = AString()
    assert_equal(str(s8) + str(s9), "abc is a string")


def test_string_join():
    var sep = String(",")
    var s0 = String("abc")
    var s1 = sep.join(s0, s0, s0, s0)
    assert_equal(s1, "abc,abc,abc,abc")

    assert_equal(sep.join(1, 2, 3), "1,2,3")

    assert_equal(sep.join(1, "abc", 3), "1,abc,3")

    var s2 = String(",").join(List[UInt8](1, 2, 3))
    assert_equal(s2, "1,2,3")

    var s3 = String(",").join(List[UInt8](1, 2, 3, 4, 5, 6, 7, 8, 9))
    assert_equal(s3, "1,2,3,4,5,6,7,8,9")

    var s4 = String(",").join(List[UInt8]())
    assert_equal(s4, "")

    var s5 = String(",").join(List[UInt8](1))
    assert_equal(s5, "1")


def test_string_literal_join():
    var s2 = ",".join(List[UInt8](1, 2, 3))
    assert_equal(s2, "1,2,3")

    var s3 = ",".join(List[UInt8](1, 2, 3, 4, 5, 6, 7, 8, 9))
    assert_equal(s3, "1,2,3,4,5,6,7,8,9")

    var s4 = ",".join(List[UInt8]())
    assert_equal(s4, "")

    var s5 = ",".join(List[UInt8](1))
    assert_equal(s5, "1")


def test_stringref():
    var a = StringRef("AAA")
    var b = StringRef("BBB")
    var c = StringRef("AAA")

    assert_equal(3, len(a))
    assert_equal(3, len(b))
    assert_equal(3, len(c))
    assert_equal(4, len("ABBA"))

    # Equality operators
    assert_not_equal(a, b)
    assert_not_equal(b, a)

    # Self equality
    assert_equal(a, a)

    # Value equality
    assert_equal(a, c)


def test_stringref_from_dtypepointer():
    var a = StringRef("AAA")
    var b = StringRef(a.data)
    assert_equal(3, len(a))
    assert_equal(3, len(b))
    assert_equal(a, b)


def test_stringref_strip():
    var a = StringRef("  mojo rocks  ")
    var b = StringRef("mojo  ")
    var c = StringRef("  mojo")
    var d = StringRef("")
    assert_equal(a.strip(), "mojo rocks")
    assert_equal(b.strip(), "mojo")
    assert_equal(c.strip(), "mojo")
    assert_equal(d.strip(), "")


def test_ord():
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
    alias single_byte = ord("A")
    assert_equal(single_byte, 65)
    alias single_byte2 = ord("!")
    assert_equal(single_byte2, 33)

    # TODO: change these to parameter domain when it work.
    var multi_byte = ord("α")
    assert_equal(multi_byte, 945)
    var multi_byte2 = ord("➿")
    assert_equal(multi_byte2, 10175)
    var multi_byte3 = ord("🔥")
    assert_equal(multi_byte3, 128293)

    # Test StringSlice overload
    assert_equal(ord("A".as_string_slice()), 65)
    assert_equal(ord("α".as_string_slice()), 945)
    assert_equal(ord("➿".as_string_slice()), 10175)
    assert_equal(ord("🔥".as_string_slice()), 128293)


def test_chr():
    assert_equal(chr(65), "A")
    assert_equal(chr(97), "a")
    assert_equal(chr(33), "!")
    assert_equal(chr(945), "α")
    assert_equal(chr(10175), "➿")
    assert_equal(chr(128293), "🔥")


def test_string_indexing():
    var str = String("Hello Mojo!!")

    assert_equal(str[0], "H")
    assert_equal(str[-1], "!")
    assert_equal(str[-len(str)], "H")
    assert_equal(str[2:], "llo Mojo!!")
    assert_equal(str[3:-1:1], "lo Mojo!")
    assert_equal(str[3:-3], "lo Moj")

    assert_equal(str[::-1], "!!ojoM olleH")

    assert_equal(str[2::-1], "leH")

    assert_equal(str[::-2], "!oo le")

    assert_equal(str[:-1:-2], "")
    assert_equal(str[-50::-1], "")
    assert_equal(str[-50::], "Hello Mojo!!")
    assert_equal(str[:-50:-1], "!!ojoM olleH")
    assert_equal(str[:50:], "Hello Mojo!!")
    assert_equal(str[::50], "H")
    assert_equal(str[::-50], "!")
    assert_equal(str[50::-50], "!")
    assert_equal(str[-50::50], "H")


def test_atol():
    # base 10
    assert_equal(375, atol(String("375")))
    assert_equal(1, atol(String("001")))
    assert_equal(5, atol(String(" 005")))
    assert_equal(13, atol(String(" 013  ")))
    assert_equal(-89, atol(String("-89")))
    assert_equal(-52, atol(String(" -52")))
    assert_equal(-69, atol(String(" -69  ")))
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
        _ = atol(String("9.03"))

    with assert_raises(
        contains="String is not convertible to integer with base 10: ' 10 1'"
    ):
        _ = atol(String(" 10 1"))

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
        contains="String is not convertible to integer with base 10: '0x_ff'"
    ):
        _ = atol("0x_ff")

    with assert_raises(
        contains="String is not convertible to integer with base 10: ''"
    ):
        _ = atol(String(""))

    with assert_raises(
        contains="String expresses an integer too large to store in Int."
    ):
        _ = atol(String("9223372036854775832"))


def test_atol_base_0():
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


def test_atof():
    assert_equal(375.0, atof(String("375.f")))
    assert_equal(1.0, atof(String("001.")))
    assert_equal(+5.0, atof(String(" +005.")))
    assert_equal(13.0, atof(String(" 013.f  ")))
    assert_equal(-89.0, atof(String("-89")))
    assert_equal(-0.3, atof(String(" -0.3")))
    assert_equal(-69e3, atof(String(" -69E+3  ")))
    assert_equal(123.2e1, atof(String(" 123.2E1  ")))
    assert_equal(23e3, atof(String(" 23E3  ")))
    assert_equal(989343e-13, atof(String(" 989343E-13  ")))
    assert_equal(1.123, atof(String(" 1.123f")))
    assert_equal(0.78, atof(String(" .78 ")))
    assert_equal(121234.0, atof(String(" 121234.  ")))
    assert_equal(985031234.0, atof(String(" 985031234.F  ")))
    assert_equal(FloatLiteral.negative_zero, atof(String("-0")))
    assert_equal(FloatLiteral.nan, atof(String("  nan")))
    assert_equal(FloatLiteral.infinity, atof(String(" inf ")))
    assert_equal(FloatLiteral.negative_infinity, atof(String("-inf  ")))

    # Negative cases
    with assert_raises(contains="String is not convertible to float: ''"):
        _ = atof(String(""))

    with assert_raises(
        contains="String is not convertible to float: ' 123 asd'"
    ):
        _ = atof(String(" 123 asd"))

    with assert_raises(
        contains="String is not convertible to float: ' f.9123 '"
    ):
        _ = atof(String(" f.9123 "))

    with assert_raises(
        contains="String is not convertible to float: ' 989343E-1A3 '"
    ):
        _ = atof(String(" 989343E-1A3 "))

    with assert_raises(
        contains="String is not convertible to float: ' 124124124_2134124124 '"
    ):
        _ = atof(String(" 124124124_2134124124 "))

    with assert_raises(
        contains="String is not convertible to float: ' 123.2E '"
    ):
        _ = atof(String(" 123.2E "))

    with assert_raises(
        contains="String is not convertible to float: ' --958.23 '"
    ):
        _ = atof(String(" --958.23 "))

    with assert_raises(
        contains="String is not convertible to float: ' ++94. '"
    ):
        _ = atof(String(" ++94. "))


def test_calc_initial_buffer_size_int32():
    assert_equal(1, _calc_initial_buffer_size_int32(0))
    assert_equal(1, _calc_initial_buffer_size_int32(9))
    assert_equal(2, _calc_initial_buffer_size_int32(10))
    assert_equal(2, _calc_initial_buffer_size_int32(99))
    assert_equal(8, _calc_initial_buffer_size_int32(99999999))
    assert_equal(9, _calc_initial_buffer_size_int32(100000000))
    assert_equal(9, _calc_initial_buffer_size_int32(999999999))
    assert_equal(10, _calc_initial_buffer_size_int32(1000000000))
    assert_equal(10, _calc_initial_buffer_size_int32(4294967295))


def test_calc_initial_buffer_size_int64():
    assert_equal(1, _calc_initial_buffer_size_int64(0))
    assert_equal(1, _calc_initial_buffer_size_int64(9))
    assert_equal(2, _calc_initial_buffer_size_int64(10))
    assert_equal(2, _calc_initial_buffer_size_int64(99))
    assert_equal(9, _calc_initial_buffer_size_int64(999999999))
    assert_equal(10, _calc_initial_buffer_size_int64(1000000000))
    assert_equal(10, _calc_initial_buffer_size_int64(9999999999))
    assert_equal(11, _calc_initial_buffer_size_int64(10000000000))
    assert_equal(20, _calc_initial_buffer_size_int64(18446744073709551615))


def test_contains():
    var str = String("Hello world")

    assert_true(str.__contains__(""))
    assert_true(str.__contains__("He"))
    assert_true("lo" in str)
    assert_true(str.__contains__(" "))
    assert_true(str.__contains__("ld"))

    assert_false(str.__contains__("below"))
    assert_true("below" not in str)


def test_find():
    var str = String("Hello world")

    assert_equal(0, str.find(""))
    assert_equal(0, str.find("Hello"))
    assert_equal(2, str.find("llo"))
    assert_equal(6, str.find("world"))
    assert_equal(-1, str.find("universe"))

    # Test find() offset is absolute, not relative (issue mojo/#1355)
    var str2 = String("...a")
    assert_equal(3, str2.find("a", 0))
    assert_equal(3, str2.find("a", 1))
    assert_equal(3, str2.find("a", 2))
    assert_equal(3, str2.find("a", 3))

    # Test find() support for negative start positions
    assert_equal(4, str.find("o", -10))
    assert_equal(7, str.find("o", -5))

    assert_equal(-1, String("abc").find("abcd"))


def test_count():
    var str = String("Hello world")

    assert_equal(12, str.count(""))
    assert_equal(1, str.count("Hell"))
    assert_equal(3, str.count("l"))
    assert_equal(1, str.count("ll"))
    assert_equal(1, str.count("ld"))
    assert_equal(0, str.count("universe"))

    assert_equal(String("aaaaa").count("a"), 5)
    assert_equal(String("aaaaaa").count("aa"), 3)


def test_replace():
    # Replace empty
    var s1 = String("abc")
    assert_equal(s1.replace("", "x"), "xaxbxc")
    assert_equal(s1.replace("", "->"), "->a->b->c")

    var s2 = String("Hello Python")
    assert_equal(s2.replace("Python", "Mojo"), "Hello Mojo")
    assert_equal(s2.replace("Hell", "HELL"), "HELLo Python")
    assert_equal(s2.replace("HELL", "xxx"), "Hello Python")
    assert_equal(s2.replace("o P", "P o"), "HellP oython")
    assert_equal(s2.replace("thon", "thing"), "Hello Pything")
    assert_equal(s2.replace("ll", "||"), "He||o Python")
    assert_equal(s2.replace("l", "-"), "He--o Python")
    assert_equal(s2.replace("l", "-x-"), "He-x--x-o Python")

    var s3 = String("a   complex  test case  with some  spaces")
    assert_equal(s3.replace("  ", " "), "a  complex test case with some spaces")


def test_rfind():
    # Basic usage.
    assert_equal(String("hello world").rfind("world"), 6)
    assert_equal(String("hello world").rfind("bye"), -1)

    # Repeated substrings.
    assert_equal(String("ababab").rfind("ab"), 4)

    # Empty string and substring.
    assert_equal(String("").rfind("ab"), -1)
    assert_equal(String("foo").rfind(""), 3)

    # Test that rfind(start) returned pos is absolute, not relative to specified
    # start. Also tests positive and negative start offsets.
    assert_equal(String("hello world").rfind("l", 5), 9)
    assert_equal(String("hello world").rfind("l", -5), 9)
    assert_equal(String("hello world").rfind("w", -3), -1)
    assert_equal(String("hello world").rfind("w", -5), 6)

    assert_equal(-1, String("abc").rfind("abcd"))

    # Special characters.
    # TODO(#26444): Support unicode strings.
    # assert_equal(String("こんにちは").rfind("にち"), 2)
    # assert_equal(String("🔥🔥").rfind("🔥"), 1)


def test_split():
    # empty separators default to whitespace
    var d = String("hello world").split()
    assert_true(len(d) == 2)
    assert_true(d[0] == "hello")
    assert_true(d[1] == "world")
    d = String("hello \t\n\n\v\fworld").split("\n")
    assert_true(len(d) == 3)
    assert_true(d[0] == "hello \t" and d[1] == "" and d[2] == "\v\fworld")

    # Should add all whitespace-like chars as one
    # test all unicode separators
    # 0 is to build a String with null terminator
    alias next_line = List[UInt8](0xC2, 0x85, 0)
    """TODO: \\x85"""
    alias unicode_line_sep = List[UInt8](0xE2, 0x80, 0xA8, 0)
    """TODO: \\u2028"""
    alias unicode_paragraph_sep = List[UInt8](0xE2, 0x80, 0xA9, 0)
    """TODO: \\u2029"""
    # TODO add line and paragraph separator as StringLiteral once unicode
    # escape secuences are accepted
    var univ_sep_var = (
        String(" ")
        + String("\t")
        + String("\n")
        + String("\r")
        + String("\v")
        + String("\f")
        + String("\x1c")
        + String("\x1d")
        + String("\x1e")
        + String(next_line)
        + String(unicode_line_sep)
        + String(unicode_paragraph_sep)
    )
    var s = univ_sep_var + "hello" + univ_sep_var + "world" + univ_sep_var
    d = s.split()
    assert_true(len(d) == 2)
    assert_true(d[0] == "hello" and d[1] == "world")

    # should split into empty strings between separators
    d = String("1,,,3").split(",")
    assert_true(len(d) == 4)
    assert_true(d[0] == "1" and d[1] == "" and d[2] == "" and d[3] == "3")
    d = String(",,,").split(",")
    assert_true(len(d) == 4)
    assert_true(d[0] == "" and d[1] == "" and d[2] == "" and d[3] == "")
    d = String(" a b ").split(" ")
    assert_true(len(d) == 4)
    assert_true(d[0] == "" and d[1] == "a" and d[2] == "b" and d[3] == "")
    d = String("abababaaba").split("aba")
    assert_true(len(d) == 4)
    assert_true(d[0] == "" and d[1] == "b" and d[2] == "" and d[3] == "")

    # should split into maxsplit + 1 items
    d = String("1,2,3").split(",", 0)
    assert_true(len(d) == 1)
    assert_true(d[0] == "1,2,3")
    d = String("1,2,3").split(",", 1)
    assert_true(len(d) == 2)
    assert_true(d[0] == "1" and d[1] == "2,3")

    assert_true(len(String("").split()) == 0)
    assert_true(len(String(" ").split()) == 0)
    assert_true(len(String("").split(" ")) == 1)
    assert_true(len(String(" ").split(" ")) == 2)
    assert_true(len(String("  ").split(" ")) == 3)
    assert_true(len(String("   ").split(" ")) == 4)

    with assert_raises():
        _ = String("").split("")

    # Split in middle
    var d1 = String("n")
    var in1 = String("faang")
    var res1 = in1.split(d1)
    assert_equal(len(res1), 2)
    assert_equal(res1[0], "faa")
    assert_equal(res1[1], "g")

    # Matches should be properly split in multiple case
    var d2 = String(" ")
    var in2 = String("modcon is coming soon")
    var res2 = in2.split(d2)
    assert_equal(len(res2), 4)
    assert_equal(res2[0], "modcon")
    assert_equal(res2[1], "is")
    assert_equal(res2[2], "coming")
    assert_equal(res2[3], "soon")

    # No match from the delimiter
    var d3 = String("x")
    var in3 = String("hello world")
    var res3 = in3.split(d3)
    assert_equal(len(res3), 1)
    assert_equal(res3[0], "hello world")

    # Multiple character delimiter
    var d4 = String("ll")
    var in4 = String("hello")
    var res4 = in4.split(d4)
    assert_equal(len(res4), 2)
    assert_equal(res4[0], "he")
    assert_equal(res4[1], "o")

    # related to #2879
    # TODO: replace string comparison when __eq__ is implemented for List
    assert_equal(
        String("abbaaaabbba").split("a").__str__(),
        "['', 'bb', '', '', '', 'bbb', '']",
    )
    assert_equal(
        String("abbaaaabbba").split("a", 8).__str__(),
        "['', 'bb', '', '', '', 'bbb', '']",
    )
    assert_equal(
        String("abbaaaabbba").split("a", 5).__str__(),
        "['', 'bb', '', '', '', 'bbba']",
    )
    assert_equal(String("aaa").split("a", 0).__str__(), "['aaa']")
    assert_equal(String("a").split("a").__str__(), "['', '']")
    assert_equal(String("1,2,3").split("3", 0).__str__(), "['1,2,3']")
    assert_equal(String("1,2,3").split("3", 1).__str__(), "['1,2,', '']")
    assert_equal(String("1,2,3,3").split("3", 2).__str__(), "['1,2,', ',', '']")
    assert_equal(
        String("1,2,3,3,3").split("3", 2).__str__(), "['1,2,', ',', ',3']"
    )


def test_splitlines():
    # Test with no line breaks
    var in1 = String("hello world")
    var res1 = in1.splitlines()
    assert_equal(len(res1), 1)
    assert_equal(res1[0], "hello world")

    # Test with \n line break
    var in2 = String("hello\nworld")
    var res2 = in2.splitlines()
    assert_equal(len(res2), 2)
    assert_equal(res2[0], "hello")
    assert_equal(res2[1], "world")

    # Test with \r\n line break
    var in3 = String("hello\r\nworld")
    var res3 = in3.splitlines()
    assert_equal(len(res3), 2)
    assert_equal(res3[0], "hello")
    assert_equal(res3[1], "world")

    # Test with \r line break
    var in4 = String("hello\rworld")
    var res4 = in4.splitlines()
    assert_equal(len(res4), 2)
    assert_equal(res4[0], "hello")
    assert_equal(res4[1], "world")

    # Test with multiple different line breaks
    var in5 = String("hello\nworld\r\nmojo\rlanguage")
    var res5 = in5.splitlines()
    assert_equal(len(res5), 4)
    assert_equal(res5[0], "hello")
    assert_equal(res5[1], "world")
    assert_equal(res5[2], "mojo")
    assert_equal(res5[3], "language")

    # Test with keepends=True
    var res6 = in5.splitlines(keepends=True)
    assert_equal(len(res6), 4)
    assert_equal(res6[0], "hello\n")
    assert_equal(res6[1], "world\r\n")
    assert_equal(res6[2], "mojo\r")
    assert_equal(res6[3], "language")

    # Test with an empty string
    var in7 = String("")
    var res7 = in7.splitlines()
    assert_equal(len(res7), 0)

    # test \v \f \x1c \x1d
    var in8 = String("hello\vworld\fmojo\x1clanguage\x1d")
    var res8 = in8.splitlines()
    assert_equal(len(res8), 4)
    assert_equal(res8[0], "hello")
    assert_equal(res8[1], "world")
    assert_equal(res8[2], "mojo")
    assert_equal(res8[3], "language")

    # test \x1e \x1d
    var in9 = String("hello\x1eworld\x1dmojo")
    var res9 = in9.splitlines()
    assert_equal(len(res9), 3)
    assert_equal(res9[0], "hello")
    assert_equal(res9[1], "world")
    assert_equal(res9[2], "mojo")

    # test \x85 \u2028 \u2029
    var next_line = List[UInt8](0xC2, 0x85, 0)
    """TODO: \\x85"""
    var unicode_line_sep = List[UInt8](0xE2, 0x80, 0xA8, 0)
    """TODO: \\u2028"""
    var unicode_paragraph_sep = List[UInt8](0xE2, 0x80, 0xA9, 0)
    """TODO: \\u2029"""

    for i in List(next_line, unicode_line_sep, unicode_paragraph_sep):
        var in9 = "hello\x1eworld" + String(i[]) + "mojo"
        var res9 = in9.splitlines()
        assert_equal(len(res9), 3)
        assert_equal(res9[0], "hello")
        assert_equal(res9[1], "world")
        assert_equal(res9[2], "mojo")

    # test with keepends=True
    var res10 = in8.splitlines(keepends=True)
    assert_equal(len(res10), 4)
    assert_equal(res10[0], "hello\v")
    assert_equal(res10[1], "world\f")
    assert_equal(res10[2], "mojo\x1c")
    assert_equal(res10[3], "language\x1d")

    var res11 = ("hello\x1eworld" + String(next_line) + "mojo").splitlines(
        keepends=True
    )
    assert_equal(len(res11), 3)
    assert_equal(res11[0], "hello\x1e")
    assert_equal(res11[1], "world" + String(next_line))
    assert_equal(res11[2], "mojo")


def test_isupper():
    assert_true(isupper(ord("A")))
    assert_true(isupper(ord("B")))
    assert_true(isupper(ord("Y")))
    assert_true(isupper(ord("Z")))

    assert_false(isupper(ord("A") - 1))
    assert_false(isupper(ord("Z") + 1))

    assert_false(isupper(ord("!")))
    assert_false(isupper(ord("0")))

    assert_true(String("ASDG").isupper())
    assert_false(String("AsDG").isupper())
    assert_true(String("ABC123").isupper())
    assert_false(String("1!").isupper())


def test_islower():
    assert_true(islower(ord("a")))
    assert_true(islower(ord("b")))
    assert_true(islower(ord("y")))
    assert_true(islower(ord("z")))

    assert_false(islower(ord("a") - 1))
    assert_false(islower(ord("z") + 1))

    assert_false(islower(ord("!")))
    assert_false(islower(ord("0")))

    assert_true(String("asdfg").islower())
    assert_false(String("asdFDg").islower())
    assert_true(String("abc123").islower())
    assert_false(String("1!").islower())


def test_lower():
    assert_equal(String("HELLO").lower(), "hello")
    assert_equal(String("hello").lower(), "hello")
    assert_equal(String("FoOBaR").lower(), "foobar")

    assert_equal(String("MOJO🔥").lower(), "mojo🔥")

    # TODO(#26444): Non-ASCII not supported yet
    assert_equal(String("É").lower(), "É")


def test_upper():
    assert_equal(String("hello").upper(), "HELLO")
    assert_equal(String("HELLO").upper(), "HELLO")
    assert_equal(String("FoOBaR").upper(), "FOOBAR")

    assert_equal(String("mojo🔥").upper(), "MOJO🔥")

    # TODO(#26444): Non-ASCII not supported yet
    assert_equal(String("É").upper(), "É")


def test_isspace():
    # checking true cases
    assert_true(_isspace(ord(" ")))
    assert_true(_isspace(ord("\n")))
    assert_true(_isspace("\n"))
    assert_true(_isspace(ord("\t")))
    assert_true(_isspace(ord("\r")))
    assert_true(_isspace(ord("\v")))
    assert_true(_isspace(ord("\f")))

    # Checking false cases
    assert_false(_isspace(ord("a")))
    assert_false(_isspace("a"))
    assert_false(_isspace(ord("u")))
    assert_false(_isspace(ord("s")))
    assert_false(_isspace(ord("t")))
    assert_false(_isspace(ord("i")))
    assert_false(_isspace(ord("n")))
    assert_false(_isspace(ord("z")))
    assert_false(_isspace(ord(".")))

    # test all utf8 and unicode separators
    # 0 is to build a String with null terminator
    alias next_line = List[UInt8](0xC2, 0x85, 0)
    """TODO: \\x85"""
    alias unicode_line_sep = List[UInt8](0xE2, 0x80, 0xA8, 0)
    """TODO: \\u2028"""
    alias unicode_paragraph_sep = List[UInt8](0xE2, 0x80, 0xA9, 0)
    """TODO: \\u2029"""
    # TODO add line and paragraph separator as StringLiteral once unicode
    # escape sequences are accepted
    var univ_sep_var = List[String](
        String(" "),
        String("\t"),
        String("\n"),
        String("\r"),
        String("\v"),
        String("\f"),
        String("\x1c"),
        String("\x1d"),
        String("\x1e"),
        String(next_line),
        String(unicode_line_sep),
        String(unicode_paragraph_sep),
    )

    for i in univ_sep_var:
        assert_true(i[].isspace())

    for i in List[String]("not", "space", "", "s", "a", "c"):
        assert_false(i[].isspace())

    for i in range(len(univ_sep_var)):
        var sep = String("")
        for j in range(len(univ_sep_var)):
            sep += univ_sep_var[i]
            sep += univ_sep_var[j]
        assert_true(sep.isspace())
        _ = sep


def test_ascii_aliases():
    assert_true(String("a") in String.ASCII_LOWERCASE)
    assert_true(String("b") in String.ASCII_LOWERCASE)
    assert_true(String("y") in String.ASCII_LOWERCASE)
    assert_true(String("z") in String.ASCII_LOWERCASE)

    assert_true(String("A") in String.ASCII_UPPERCASE)
    assert_true(String("B") in String.ASCII_UPPERCASE)
    assert_true(String("Y") in String.ASCII_UPPERCASE)
    assert_true(String("Z") in String.ASCII_UPPERCASE)

    assert_true(String("a") in String.ASCII_LETTERS)
    assert_true(String("b") in String.ASCII_LETTERS)
    assert_true(String("y") in String.ASCII_LETTERS)
    assert_true(String("z") in String.ASCII_LETTERS)
    assert_true(String("A") in String.ASCII_LETTERS)
    assert_true(String("B") in String.ASCII_LETTERS)
    assert_true(String("Y") in String.ASCII_LETTERS)
    assert_true(String("Z") in String.ASCII_LETTERS)

    assert_true(String("0") in String.DIGITS)
    assert_true(String("9") in String.DIGITS)

    assert_true(String("0") in String.HEX_DIGITS)
    assert_true(String("9") in String.HEX_DIGITS)
    assert_true(String("A") in String.HEX_DIGITS)
    assert_true(String("F") in String.HEX_DIGITS)

    assert_true(String("7") in String.OCT_DIGITS)
    assert_false(String("8") in String.OCT_DIGITS)

    assert_true(String(",") in String.PUNCTUATION)
    assert_true(String(".") in String.PUNCTUATION)
    assert_true(String("\\") in String.PUNCTUATION)
    assert_true(String("@") in String.PUNCTUATION)
    assert_true(String('"') in String.PUNCTUATION)
    assert_true(String("'") in String.PUNCTUATION)

    var text = String("I love my Mom and Dad so much!!!\n")
    for i in range(len(text)):
        assert_true(text[i] in String.PRINTABLE)


def test_rstrip():
    # with default rstrip chars
    var empty_string = String("")
    assert_true(empty_string.rstrip() == "")

    var space_string = String(" \t\n\r\v\f  ")
    assert_true(space_string.rstrip() == "")

    var str0 = String("     n ")
    assert_true(str0.rstrip() == "     n")

    var str1 = String("string")
    assert_true(str1.rstrip() == "string")

    var str2 = String("something \t\n\t\v\f")
    assert_true(str2.rstrip() == "something")

    # with custom chars for rstrip
    var str3 = String("mississippi")
    assert_true(str3.rstrip("sip") == "m")

    var str4 = String("mississippimississippi \n ")
    assert_true(str4.rstrip("sip ") == "mississippimississippi \n")
    assert_true(str4.rstrip("sip \n") == "mississippim")


def test_lstrip():
    # with default lstrip chars
    var empty_string = String("")
    assert_true(empty_string.lstrip() == "")

    var space_string = String(" \t\n\r\v\f  ")
    assert_true(space_string.lstrip() == "")

    var str0 = String("     n ")
    assert_true(str0.lstrip() == "n ")

    var str1 = String("string")
    assert_true(str1.lstrip() == "string")

    var str2 = String(" \t\n\t\v\fsomething")
    assert_true(str2.lstrip() == "something")

    # with custom chars for lstrip
    var str3 = String("mississippi")
    assert_true(str3.lstrip("mis") == "ppi")

    var str4 = String(" \n mississippimississippi")
    assert_true(str4.lstrip("mis ") == "\n mississippimississippi")
    assert_true(str4.lstrip("mis \n") == "ppimississippi")


def test_strip():
    # with default strip chars
    var empty_string = String("")
    assert_true(empty_string.strip() == "")
    alias comp_empty_string_stripped = String("").strip()
    assert_true(comp_empty_string_stripped == "")

    var space_string = String(" \t\n\r\v\f  ")
    assert_true(space_string.strip() == "")
    alias comp_space_string_stripped = String(" \t\n\r\v\f  ").strip()
    assert_true(comp_space_string_stripped == "")

    var str0 = String("     n ")
    assert_true(str0.strip() == "n")
    alias comp_str0_stripped = String("     n ").strip()
    assert_true(comp_str0_stripped == "n")

    var str1 = String("string")
    assert_true(str1.strip() == "string")
    alias comp_str1_stripped = String("string").strip()
    assert_true(comp_str1_stripped == "string")

    var str2 = String(" \t\n\t\v\fsomething \t\n\t\v\f")
    alias comp_str2_stripped = String(" \t\n\t\v\fsomething \t\n\t\v\f").strip()
    assert_true(str2.strip() == "something")
    assert_true(comp_str2_stripped == "something")

    # with custom strip chars
    var str3 = String("mississippi")
    assert_true(str3.strip("mips") == "")
    assert_true(str3.strip("mip") == "ssiss")
    alias comp_str3_stripped = String("mississippi").strip("mips")
    assert_true(comp_str3_stripped == "")

    var str4 = String(" \n mississippimississippi \n ")
    assert_true(str4.strip(" ") == "\n mississippimississippi \n")
    assert_true(str4.strip("\nmip ") == "ssissippimississ")

    alias comp_str4_stripped = String(" \n mississippimississippi \n ").strip(
        " "
    )
    assert_true(comp_str4_stripped == "\n mississippimississippi \n")


def test_hash():
    fn assert_hash_equals_literal_hash[s: StringLiteral]() raises:
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


def test_startswith():
    var str = String("Hello world")

    assert_true(str.startswith("Hello"))
    assert_false(str.startswith("Bye"))

    assert_true(str.startswith("llo", 2))
    assert_true(str.startswith("llo", 2, -1))
    assert_false(str.startswith("llo", 2, 3))


def test_endswith():
    var str = String("Hello world")

    assert_true(str.endswith(""))
    assert_true(str.endswith("world"))
    assert_true(str.endswith("ld"))
    assert_false(str.endswith("universe"))

    assert_true(str.endswith("ld", 2))
    assert_true(str.endswith("llo", 2, 5))
    assert_false(str.endswith("llo", 2, 3))


def test_removeprefix():
    assert_equal(String("hello world").removeprefix(""), String("hello world"))
    assert_equal(String("hello world").removeprefix("hello"), " world")
    assert_equal(String("hello world").removeprefix("world"), "hello world")
    assert_equal(String("hello world").removeprefix("hello world"), "")
    assert_equal(String("hello world").removeprefix("llo wor"), "hello world")


def test_removesuffix():
    assert_equal(String("hello world").removesuffix(""), String("hello world"))
    assert_equal(String("hello world").removesuffix("world"), "hello ")
    assert_equal(String("hello world").removesuffix("hello"), "hello world")
    assert_equal(String("hello world").removesuffix("hello world"), "")
    assert_equal(String("hello world").removesuffix("llo wor"), "hello world")


def test_intable():
    assert_equal(int(String("123")), 123)
    assert_equal(int(String("10"), base=8), 8)

    with assert_raises():
        _ = int(String("hi"))


def test_string_mul():
    assert_equal(String("*") * 0, "")
    assert_equal(String("!") * 10, String("!!!!!!!!!!"))
    assert_equal(String("ab") * 5, "ababababab")


def test_indexing():
    a = String("abc")
    assert_equal(a[False], "a")
    assert_equal(a[int(1)], "b")
    assert_equal(a[2], "c")


def test_string_iter():
    var vs = String("123")

    # Borrow immutably
    fn conc(vs: String) -> String:
        var c = String("")
        for v in vs:
            c += v
        return c

    assert_equal(123, atol(conc(vs)))

    concat = String("")
    for v in vs.__reversed__():
        concat += v
    assert_equal(321, atol(concat))

    # TODO: UnsafePointer does not have a store or __setitem__ method
    # for v in vs:
    #     v.unsafe_ptr().store(0, "1")

    # # Borrow immutably
    # for v in vs:
    #     concat += v

    # assert_equal(111, atol(concat))

    var idx = -1
    vs = String("mojo🔥")
    for item in vs:
        idx += 1
        if idx == 0:
            assert_equal(str(item), "m")
        elif idx == 1:
            assert_equal(str(item), "o")
        elif idx == 2:
            assert_equal(str(item), "j")
        elif idx == 3:
            assert_equal(str(item), "o")
        elif idx == 4:
            assert_equal(str(item), "🔥")
    assert_equal(4, idx)

    var items = List[String](
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
    )
    var rev = List[String](
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
    )
    var utf8_sequence_lengths = List(5, 12, 9, 5, 7, 6, 5, 5, 2, 3, 12)
    for item_idx in range(len(items)):
        var item = items[item_idx]
        var utf8_sequence_len = 0
        var byte_idx = 0
        for v in item:
            var byte_len = v.byte_length()
            assert_equal(item[byte_idx : byte_idx + byte_len], v)
            byte_idx += byte_len
            utf8_sequence_len += 1
        assert_equal(utf8_sequence_len, utf8_sequence_lengths[item_idx])
        var concat = String("")
        for v in item.__reversed__():
            concat += v
        assert_equal(rev[item_idx], concat)
        item_idx += 1


def test_format_args():
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

    assert_equal(
        String(" {} , {} {} !").format(
            "Hello",
            "Beautiful",
            "World",
        ),
        " Hello , Beautiful World !",
    )

    with assert_raises(
        contains="there is a single curly { left unclosed or unescaped"
    ):
        _ = String("{ {}").format(1)

    with assert_raises(
        contains="there is a single curly { left unclosed or unescaped"
    ):
        _ = String("{ {0}").format(1)

    with assert_raises(
        contains="there is a single curly { left unclosed or unescaped"
    ):
        _ = String("{}{").format(1)

    with assert_raises(
        contains="there is a single curly } left unclosed or unescaped"
    ):
        _ = String("{}}").format(1)

    with assert_raises(
        contains="there is a single curly { left unclosed or unescaped"
    ):
        _ = String("{} {").format(1)

    with assert_raises(
        contains="there is a single curly { left unclosed or unescaped"
    ):
        _ = String("{").format(1)

    with assert_raises(
        contains="there is a single curly } left unclosed or unescaped"
    ):
        _ = String("}").format(1)

    assert_equal(String("}}").format(), "}")
    assert_equal(String("{{").format(), "{")

    assert_equal(String("{{}}{}{{}}").format("foo"), "{}foo{}")

    assert_equal(String("{{ {0}").format("foo"), "{ foo")
    assert_equal(String("{{{0}").format("foo"), "{foo")
    assert_equal(String("{{0}}").format("foo"), "{0}")
    assert_equal(String("{{}}").format("foo"), "{}")
    assert_equal(String("{{0}}").format("foo"), "{0}")
    assert_equal(String("{{{0}}}").format("foo"), "{foo}")

    var vinput = "{} {}"
    var output = String(vinput).format("123", 456)
    assert_equal(len(output), 7)

    vinput = "{1}{0}"
    output = String(vinput).format("123", 456)
    assert_equal(len(output), 6)
    assert_equal(output, "456123")

    vinput = "123"
    output = String(vinput).format()
    assert_equal(len(output), 3)

    vinput = ""
    output = String(vinput).format()
    assert_equal(len(output), 0)

    assert_equal(
        String("{0} {1} ❤️‍🔥 {1} {0}").format(
            "🔥",
            "Mojo",
        ),
        "🔥 Mojo ❤️‍🔥 Mojo 🔥",
    )

    assert_equal(String("{0} {1}").format(True, 1.125), "True 1.125")

    assert_equal(String("{0} {1}").format("{1}", "Mojo"), "{1} Mojo")
    assert_equal(
        String("{0} {1} {0} {1}").format("{1}", "Mojo"), "{1} Mojo {1} Mojo"
    )


def test_format_conversion_flags():
    assert_equal(String("{!r}").format(""), "''")
    var special_str = "a\nb\tc"
    assert_equal(
        String("{} {!r}").format(special_str, special_str),
        "a\nb\tc 'a\\nb\\tc'",
    )
    assert_equal(
        String("{!s} {!r}").format(special_str, special_str),
        "a\nb\tc 'a\\nb\\tc'",
    )

    var a = "Mojo"
    assert_equal(String("{} {!r}").format(a, a), "Mojo 'Mojo'")
    assert_equal(String("{!s} {!r}").format(a, a), "Mojo 'Mojo'")
    assert_equal(String("{0!s} {0!r}").format(a), "Mojo 'Mojo'")

    var b = 21.1
    assert_true(
        "21.100000000000001 SIMD[DType.float64, 1](2"
        in String("{} {!r}").format(b, b),
    )
    assert_true(
        "21.100000000000001 SIMD[DType.float64, 1](2"
        in String("{!s} {!r}").format(b, b),
    )

    var c = 1e100
    assert_equal(
        String("{} {!r}").format(c, c),
        "1e+100 SIMD[DType.float64, 1](1.0000000000000000e+100)",
    )
    assert_equal(
        String("{!s} {!r}").format(c, c),
        "1e+100 SIMD[DType.float64, 1](1.0000000000000000e+100)",
    )

    var d = 42
    assert_equal(String("{} {!r}").format(d, d), "42 42")
    assert_equal(String("{!s} {!r}").format(d, d), "42 42")

    assert_true(
        "Mojo SIMD[DType.float64, 1](2"
        in String("{} {!r} {} {!r}").format(a, b, c, d)
    )
    assert_true(
        "Mojo SIMD[DType.float64, 1](2"
        in String("{!s} {!r} {!s} {!r}").format(a, b, c, d)
    )

    var e = True
    assert_equal(String("{} {!r}").format(e, e), "True True")

    assert_true(
        "Mojo SIMD[DType.float64, 1](2"
        in String("{0} {1!r} {2} {3}").format(a, b, c, d)
    )
    assert_true(
        "Mojo SIMD[DType.float64, 1](2"
        in String("{0!s} {1!r} {2} {3!s}").format(a, b, c, d)
    )

    assert_equal(
        String("{3} {2} {1} {0}").format(a, d, c, b),
        "21.100000000000001 1e+100 42 Mojo",
    )

    assert_true(
        "'Mojo' 42 SIMD[DType.float64, 1](2"
        in String("{0!r} {3} {1!r}").format(a, b, c, d)
    )

    assert_true(
        "True 'Mojo' 42 SIMD[DType.float64, 1](2"
        in String("{4} {0!r} {3} {1!r}").format(a, b, c, d, True)
    )

    with assert_raises(contains='Conversion flag "x" not recognised.'):
        _ = String("{!x}").format(1)

    with assert_raises(contains="Empty conversion flag."):
        _ = String("{!}").format(1)

    with assert_raises(contains='Conversion flag "rs" not recognised.'):
        _ = String("{!rs}").format(1)

    with assert_raises(contains='Conversion flag "r123" not recognised.'):
        _ = String("{!r123}").format(1)

    with assert_raises(contains='Conversion flag "r!" not recognised.'):
        _ = String("{!r!}").format(1)

    with assert_raises(contains='Conversion flag "x" not recognised.'):
        _ = String("{0!x}").format(1)

    with assert_raises(contains='Conversion flag "r:d" not recognised.'):
        _ = String("{!r:d}").format(1)


def test_isdigit():
    assert_true(isdigit(ord("1")))
    assert_false(isdigit(ord("g")))

    assert_true(String("123").isdigit())
    assert_false(String("asdg").isdigit())
    assert_false(String("123asdg").isdigit())


def test_isprintable():
    assert_true(isprintable(ord("a")))
    assert_false(isprintable(ord("\n")))
    assert_false(isprintable(ord("\t")))

    assert_true(String("aasdg").isprintable())
    assert_false(String("aa\nae").isprintable())
    assert_false(String("aa\tae").isprintable())


def test_rjust():
    assert_equal(String("hello").rjust(4), "hello")
    assert_equal(String("hello").rjust(8), "   hello")
    assert_equal(String("hello").rjust(8, "*"), "***hello")


def test_ljust():
    assert_equal(String("hello").ljust(4), "hello")
    assert_equal(String("hello").ljust(8), "hello   ")
    assert_equal(String("hello").ljust(8, "*"), "hello***")


def test_center():
    assert_equal(String("hello").center(4), "hello")
    assert_equal(String("hello").center(8), " hello  ")
    assert_equal(String("hello").center(8, "*"), "*hello**")


def main():
    test_constructors()
    test_copy()
    test_equality_operators()
    test_comparison_operators()
    test_add()
    test_stringable()
    test_repr()
    test_string_join()
    test_string_literal_join()
    test_stringref()
    test_stringref_from_dtypepointer()
    test_stringref_strip()
    test_ord()
    test_chr()
    test_string_indexing()
    test_atol()
    test_atol_base_0()
    test_atof()
    test_calc_initial_buffer_size_int32()
    test_calc_initial_buffer_size_int64()
    test_contains()
    test_find()
    test_count()
    test_replace()
    test_rfind()
    test_split()
    test_splitlines()
    test_isupper()
    test_islower()
    test_lower()
    test_upper()
    test_isspace()
    test_ascii_aliases()
    test_rstrip()
    test_lstrip()
    test_strip()
    test_hash()
    test_startswith()
    test_endswith()
    test_removeprefix()
    test_removesuffix()
    test_intable()
    test_string_mul()
    test_indexing()
    test_string_iter()
    test_format_args()
    test_format_conversion_flags()
    test_isdigit()
    test_isprintable()
    test_rjust()
    test_ljust()
    test_center()
