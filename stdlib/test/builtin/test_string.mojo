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
from testing import (
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_raises,
    assert_true,
)

from utils import StringRef
from python import Python


@value
struct AString(Stringable):
    fn __str__(self: Self) -> String:
        return "a string"


fn test_stringable() raises:
    assert_equal("hello", str("hello"))
    assert_equal("0", str(0))
    assert_equal("AAA", str(StringRef("AAA")))
    assert_equal("a string", str(AString()))


fn test_repr() raises:
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


fn test_constructors() raises:
    # Default construction
    assert_equal(0, len(String()))
    assert_true(not String())

    # Construction from Int
    var s0 = str(0)
    assert_equal("0", str(0))
    assert_equal(1, len(s0))

    var s1 = str(123)
    assert_equal("123", str(123))
    assert_equal(3, len(s1))

    # Construction from StringLiteral
    var s2 = String("abc")
    assert_equal("abc", str(s2))
    assert_equal(3, len(s2))

    # Construction from UnsafePointer
    var ptr = UnsafePointer[UInt8].alloc(4)
    ptr[0] = ord("a")
    ptr[1] = ord("b")
    ptr[2] = ord("c")
    ptr[3] = 0
    var s3 = String(ptr, 4)
    assert_equal(s3, "abc")

    # Construction from PythonObject
    var py = Python.evaluate("1 + 1")
    var s4 = String(py)
    assert_equal(s4, "2")


fn test_copy() raises:
    var s0 = String("find")
    var s1 = str(s0)
    s1._buffer[3] = ord("e")
    assert_equal("find", s0)
    assert_equal("fine", s1)


fn test_equality_operators() raises:
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


fn test_comparison_operators() raises:
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


fn test_add() raises:
    var s1 = String("123")
    var s2 = String("abc")
    var s3 = s1 + s2
    assert_equal("123abc", s3)

    var s4 = String("x")
    var s5 = s4.join(1, 2, 3)
    assert_equal("1x2x3", s5)

    var s6 = s4.join(s1, s2)
    assert_equal("123xabc", s6)

    var s7 = String()
    assert_equal("abc", s2 + s7)

    assert_equal("abcdef", s2 + "def")
    assert_equal("123abc", "123" + s2)

    var s8 = String("abc is ")
    var s9 = AString()
    assert_equal("abc is a string", str(s8) + str(s9))


fn test_string_join() raises:
    var sep = String(",")
    var s0 = String("abc")
    var s1 = sep.join(s0, s0, s0, s0)
    assert_equal("abc,abc,abc,abc", s1)

    assert_equal(sep.join(1, 2, 3), "1,2,3")

    assert_equal(sep.join(1, "abc", 3), "1,abc,3")


fn test_stringref() raises:
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


fn test_stringref_from_dtypepointer() raises:
    var a = StringRef("AAA")
    var b = StringRef(a.data)
    assert_equal(3, len(a))
    assert_equal(3, len(b))
    assert_equal(a, b)


fn test_stringref_strip() raises:
    var a = StringRef("  mojo rocks  ")
    var b = StringRef("mojo  ")
    var c = StringRef("  mojo")
    var d = StringRef("")
    assert_equal(a.strip(), "mojo rocks")
    assert_equal(b.strip(), "mojo")
    assert_equal(c.strip(), "mojo")
    assert_equal(d.strip(), "")


fn test_ord() raises:
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

    var multi_byte = ord("α")
    assert_equal(multi_byte, 945)
    var multi_byte2 = ord("➿")
    assert_equal(multi_byte2, 10175)
    var multi_byte3 = ord("🔥")
    assert_equal(multi_byte3, 128293)


fn test_chr() raises:
    assert_equal("A", chr(65))
    assert_equal("a", chr(97))
    assert_equal("!", chr(33))
    assert_equal("α", chr(945))
    assert_equal("➿", chr(10175))
    assert_equal("🔥", chr(128293))


fn test_string_indexing() raises:
    var str = String("Hello Mojo!!")

    assert_equal("H", str[0])
    assert_equal("!", str[-1])
    assert_equal("H", str[-len(str)])
    assert_equal("llo Mojo!!", str[2:])
    assert_equal("lo Mojo!", str[3:-1:1])
    assert_equal("lo Moj", str[3:-3])

    assert_equal("!!ojoM olleH", str[::-1])

    assert_equal("!!ojoM oll", str[2::-1])

    assert_equal("!oo le", str[::-2])

    assert_equal("!jMolH", str[:-1:-2])


fn test_atol() raises:
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

    with assert_raises(contains="Base must be >= 2 and <= 36, or 0."):
        _ = atol("0", 1)

    with assert_raises(contains="Base must be >= 2 and <= 36, or 0."):
        _ = atol("0", 37)

    with assert_raises(
        contains="String is not convertible to integer with base 10: ''"
    ):
        _ = atol(String(""))

    with assert_raises(
        contains="String expresses an integer too large to store in Int."
    ):
        _ = atol(String("9223372036854775832"))


fn test_atol_base_0() raises:
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
        contains="String is not convertible to integer with base 0: '0b_0'"
    ):
        _ = atol("0b_0", base=0)

    with assert_raises(
        contains="String is not convertible to integer with base 0: '0xf__f'"
    ):
        _ = atol("0xf__f", base=0)

    with assert_raises(
        contains="String is not convertible to integer with base 0: '0of_'"
    ):
        _ = atol("0of_", base=0)


fn test_atof() raises:
    assert_equal(375.0, atof(String("375.f")))
    assert_equal(1.0, atof(String("001.")))
    assert_equal(+5.0, atof(String(" +005.")))
    assert_equal(13.0, atof(String(" 013.f  ")))
    assert_equal(-89, atof(String("-89")))
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


fn test_calc_initial_buffer_size_int32() raises:
    assert_equal(1, _calc_initial_buffer_size_int32(0))
    assert_equal(1, _calc_initial_buffer_size_int32(9))
    assert_equal(2, _calc_initial_buffer_size_int32(10))
    assert_equal(2, _calc_initial_buffer_size_int32(99))
    assert_equal(8, _calc_initial_buffer_size_int32(99999999))
    assert_equal(9, _calc_initial_buffer_size_int32(100000000))
    assert_equal(9, _calc_initial_buffer_size_int32(999999999))
    assert_equal(10, _calc_initial_buffer_size_int32(1000000000))
    assert_equal(10, _calc_initial_buffer_size_int32(4294967295))


fn test_calc_initial_buffer_size_int64() raises:
    assert_equal(1, _calc_initial_buffer_size_int64(0))
    assert_equal(1, _calc_initial_buffer_size_int64(9))
    assert_equal(2, _calc_initial_buffer_size_int64(10))
    assert_equal(2, _calc_initial_buffer_size_int64(99))
    assert_equal(9, _calc_initial_buffer_size_int64(999999999))
    assert_equal(10, _calc_initial_buffer_size_int64(1000000000))
    assert_equal(10, _calc_initial_buffer_size_int64(9999999999))
    assert_equal(11, _calc_initial_buffer_size_int64(10000000000))
    assert_equal(20, _calc_initial_buffer_size_int64(18446744073709551615))


fn test_contains() raises:
    var str = String("Hello world")

    assert_true(str.__contains__(""))
    assert_true(str.__contains__("He"))
    assert_true("lo" in str)
    assert_true(str.__contains__(" "))
    assert_true(str.__contains__("ld"))

    assert_false(str.__contains__("bellow"))
    assert_true("bellow" not in str)


fn test_find() raises:
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


fn test_count() raises:
    var str = String("Hello world")

    assert_equal(12, str.count(""))
    assert_equal(1, str.count("Hell"))
    assert_equal(3, str.count("l"))
    assert_equal(1, str.count("ll"))
    assert_equal(1, str.count("ld"))
    assert_equal(0, str.count("universe"))

    assert_equal(String("aaaaa").count("a"), 5)
    assert_equal(String("aaaaaa").count("aa"), 3)


fn test_replace() raises:
    # Replace empty
    var s1 = String("abc")
    assert_equal("xaxbxc", s1.replace("", "x"))
    assert_equal("->a->b->c", s1.replace("", "->"))

    var s2 = String("Hello Python")
    assert_equal("Hello Mojo", s2.replace("Python", "Mojo"))
    assert_equal("HELLo Python", s2.replace("Hell", "HELL"))
    assert_equal("Hello Python", s2.replace("HELL", "xxx"))
    assert_equal("HellP oython", s2.replace("o P", "P o"))
    assert_equal("Hello Pything", s2.replace("thon", "thing"))
    assert_equal("He||o Python", s2.replace("ll", "||"))
    assert_equal("He--o Python", s2.replace("l", "-"))
    assert_equal("He-x--x-o Python", s2.replace("l", "-x-"))

    var s3 = String("a   complex  test case  with some  spaces")
    assert_equal("a  complex test case with some spaces", s3.replace("  ", " "))


fn test_rfind() raises:
    # Basic usage.
    assert_equal(String("hello world").rfind("world"), 6)
    assert_equal(String("hello world").rfind("bye"), -1)

    # Repeated substrings.
    assert_equal(String("ababab").rfind("ab"), 4)

    # Empty string and substring.
    assert_equal(String("").rfind("ab"), -1)
    assert_equal(String("foo").rfind(""), 3)

    # Test that rfind(start) returned pos is absolute, not relative to specifed
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


fn test_split() raises:
    # empty separators default to whitespace
    var d = String("hello world").split()
    assert_true(len(d) == 2)
    assert_true(d[0] == "hello")
    assert_true(d[1] == "world")
    d = String("hello \t\n\n\v\fworld").split("\n")
    assert_true(len(d) == 3)
    assert_true(d[0] == "hello \t" and d[1] == "" and d[2] == "\v\fworld")

    # Should add all whitespace-like chars as one
    alias utf8_spaces = String(" \t\n\r\v\f")
    var s = utf8_spaces + "hello" + utf8_spaces + "world" + utf8_spaces
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


fn test_splitlines() raises:
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

    # test \x1e \x85
    var in9 = String("hello\x1eworld\x85mojo")
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

    var res11 = in9.splitlines(keepends=True)
    assert_equal(len(res11), 3)
    assert_equal(res11[0], "hello\x1e")
    assert_equal(res11[1], "world\x85")
    assert_equal(res11[2], "mojo")


fn test_isupper() raises:
    assert_true(isupper(ord("A")))
    assert_true(isupper(ord("B")))
    assert_true(isupper(ord("Y")))
    assert_true(isupper(ord("Z")))

    assert_false(isupper(ord("A") - 1))
    assert_false(isupper(ord("Z") + 1))

    assert_false(isupper(ord("!")))
    assert_false(isupper(ord("0")))


fn test_islower() raises:
    assert_true(islower(ord("a")))
    assert_true(islower(ord("b")))
    assert_true(islower(ord("y")))
    assert_true(islower(ord("z")))

    assert_false(islower(ord("a") - 1))
    assert_false(islower(ord("z") + 1))

    assert_false(islower(ord("!")))
    assert_false(islower(ord("0")))


fn test_lower() raises:
    assert_equal(String("HELLO").lower(), "hello")
    assert_equal(String("hello").lower(), "hello")
    assert_equal(String("FoOBaR").lower(), "foobar")

    assert_equal(String("MOJO🔥").lower(), "mojo🔥")

    # TODO(#26444): Non-ASCII not supported yet
    assert_equal(String("É").lower(), "É")


fn test_upper() raises:
    assert_equal(String("hello").upper(), "HELLO")
    assert_equal(String("HELLO").upper(), "HELLO")
    assert_equal(String("FoOBaR").upper(), "FOOBAR")

    assert_equal(String("mojo🔥").upper(), "MOJO🔥")

    # TODO(#26444): Non-ASCII not supported yet
    assert_equal(String("É").upper(), "É")


fn test_isspace() raises:
    # checking true cases
    assert_true(_isspace(ord(" ")))
    assert_true(_isspace(ord("\n")))
    assert_true(_isspace(ord("\t")))
    assert_true(_isspace(ord("\r")))
    assert_true(_isspace(ord("\v")))
    assert_true(_isspace(ord("\f")))

    # Checking false cases
    assert_false(_isspace(ord("a")))
    assert_false(_isspace(ord("u")))
    assert_false(_isspace(ord("s")))
    assert_false(_isspace(ord("t")))
    assert_false(_isspace(ord("i")))
    assert_false(_isspace(ord("n")))
    assert_false(_isspace(ord("z")))
    assert_false(_isspace(ord(".")))

    # test all utf8 and unicode separators
    # 0 is to build a String with null terminator
    alias information_sep_four = List[UInt8](0x5C, 0x78, 0x31, 0x63, 0)
    """TODO: \\x1c"""
    alias information_sep_two = List[UInt8](0x5C, 0x78, 0x31, 0x65, 0)
    """TODO: \\x1e"""
    alias next_line = List[UInt8](0x78, 0x38, 0x35, 0)
    """TODO: \\x85"""
    alias unicode_line_sep = List[UInt8](
        0x20, 0x5C, 0x75, 0x32, 0x30, 0x32, 0x38, 0
    )
    """TODO: \\u2028"""
    alias unicode_paragraph_sep = List[UInt8](
        0x20, 0x5C, 0x75, 0x32, 0x30, 0x32, 0x39, 0
    )
    """TODO: \\u2029"""
    # TODO add line and paragraph separator as stringliteral once unicode
    # escape secuences are accepted
    var univ_sep_var = List[String](
        String(" "),
        String("\t"),
        String("\n"),
        String("\r"),
        String("\v"),
        String("\f"),
        String(next_line),
        String(information_sep_four),
        String(information_sep_two),
        String(unicode_line_sep),
        String(unicode_paragraph_sep),
    )

    for b in List[UInt8](0x20, 0x5C, 0x75, 0x32, 0x30, 0x32, 0x38, 0):
        var val = String(List[UInt8](b[], 0))
        if not (val in univ_sep_var):
            assert_false(val.isspace())

    for b in List[UInt8](0x20, 0x5C, 0x75, 0x32, 0x30, 0x32, 0x39, 0):
        var val = String(List[UInt8](b[], 0))
        if not (val in univ_sep_var):
            assert_false(val.isspace())

    for i in univ_sep_var:
        assert_true(i[].isspace())

    for i in List[String]("not", "space", "", "s", "a", "c"):
        assert_false(i[].isspace())


fn test_ascii_aliases() raises:
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


fn test_rstrip() raises:
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


fn test_lstrip() raises:
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


fn test_strip() raises:
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


fn test_hash() raises:
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


fn test_startswith() raises:
    var str = String("Hello world")

    assert_true(str.startswith("Hello"))
    assert_false(str.startswith("Bye"))

    assert_true(str.startswith("llo", 2))
    assert_true(str.startswith("llo", 2, -1))
    assert_false(str.startswith("llo", 2, 3))


fn test_endswith() raises:
    var str = String("Hello world")

    assert_true(str.endswith(""))
    assert_true(str.endswith("world"))
    assert_true(str.endswith("ld"))
    assert_false(str.endswith("universe"))

    assert_true(str.endswith("ld", 2))
    assert_true(str.endswith("llo", 2, 5))
    assert_false(str.endswith("llo", 2, 3))


def test_removeprefix():
    assert_equal(String("hello world").removeprefix("hello"), " world")
    assert_equal(String("hello world").removeprefix("world"), "hello world")
    assert_equal(String("hello world").removeprefix("hello world"), "")
    assert_equal(String("hello world").removeprefix("llo wor"), "hello world")


def test_removesuffix():
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


def main():
    test_constructors()
    test_copy()
    test_equality_operators()
    test_comparison_operators()
    test_add()
    test_stringable()
    test_repr()
    test_string_join()
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
