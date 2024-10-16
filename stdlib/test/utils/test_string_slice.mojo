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

from testing import assert_equal, assert_true, assert_false

from utils import Span, StringSlice
from utils._utf8_validation import _is_valid_utf8
from utils.string_slice import _count_utf8_continuation_bytes


fn test_string_literal_byte_span() raises:
    alias string: StringLiteral = "Hello"
    alias slc = string.as_bytes()

    assert_equal(len(slc), 5)
    assert_equal(slc[0], ord("H"))
    assert_equal(slc[1], ord("e"))
    assert_equal(slc[2], ord("l"))
    assert_equal(slc[3], ord("l"))
    assert_equal(slc[4], ord("o"))


fn test_string_byte_span() raises:
    var string = String("Hello")
    var str_slice = string.as_bytes()

    assert_equal(len(str_slice), 5)
    assert_equal(str_slice[0], ord("H"))
    assert_equal(str_slice[1], ord("e"))
    assert_equal(str_slice[2], ord("l"))
    assert_equal(str_slice[3], ord("l"))
    assert_equal(str_slice[4], ord("o"))

    # ----------------------------------
    # Test subslicing
    # ----------------------------------

    # Slice the whole thing
    var sub1 = str_slice[:5]
    assert_equal(len(sub1), 5)
    assert_equal(sub1[0], ord("H"))
    assert_equal(sub1[1], ord("e"))
    assert_equal(sub1[2], ord("l"))
    assert_equal(sub1[3], ord("l"))
    assert_equal(sub1[4], ord("o"))

    # Slice the end
    var sub2 = str_slice[2:5]
    assert_equal(len(sub2), 3)
    assert_equal(sub2[0], ord("l"))
    assert_equal(sub2[1], ord("l"))
    assert_equal(sub2[2], ord("o"))

    # Slice the first element
    var sub3 = str_slice[0:1]
    assert_equal(len(sub3), 1)
    assert_equal(sub3[0], ord("H"))

    #
    # Test mutation through slice
    #

    sub1[0] = ord("J")
    assert_equal(string, "Jello")

    sub2[2] = ord("y")
    assert_equal(string, "Jelly")

    # ----------------------------------
    # Test empty subslicing
    # ----------------------------------

    var sub4 = str_slice[0:0]
    assert_equal(len(sub4), 0)

    var sub5 = str_slice[2:2]
    assert_equal(len(sub5), 0)

    # Empty slices still have a pointer value
    assert_equal(int(sub5.unsafe_ptr()) - int(sub4.unsafe_ptr()), 2)

    # ----------------------------------
    # Test invalid slicing
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


fn test_heap_string_from_string_slice() raises:
    alias string_lit: StringLiteral = "Hello"

    alias static_str = string_lit.as_string_slice()

    alias heap_string = String(static_str)

    assert_equal(heap_string, "Hello")


fn test_slice_len() raises:
    alias str1: StringLiteral = "12345"
    alias str2: StringLiteral = "1234"
    alias str3: StringLiteral = "123"
    alias str4: StringLiteral = "12"
    alias str5: StringLiteral = "1"

    alias slice1 = str1.as_string_slice()
    alias slice2 = str2.as_string_slice()
    alias slice3 = str3.as_string_slice()
    alias slice4 = str4.as_string_slice()
    alias slice5 = str5.as_string_slice()

    assert_equal(5, len(slice1))
    assert_equal(4, len(slice2))
    assert_equal(3, len(slice3))
    assert_equal(2, len(slice4))
    assert_equal(1, len(slice5))


fn test_slice_eq() raises:
    var str1: String = "12345"
    var str2: String = "12345"
    var str3: StringLiteral = "12345"
    var str4: String = "abc"
    var str5: String = "abcdef"
    var str6: StringLiteral = "abcdef"

    # eq

    # FIXME: the origin of the StringSlice origin should be the data in the
    # string, not the string itself.
    # assert_true(str1.as_string_slice().__eq__(str1))
    assert_true(str1.as_string_slice().__eq__(str2))
    assert_true(str2.as_string_slice().__eq__(str2.as_string_slice()))
    assert_true(str1.as_string_slice().__eq__(str3))

    # ne

    assert_true(str1.as_string_slice().__ne__(str4))
    assert_true(str1.as_string_slice().__ne__(str5))
    assert_true(str1.as_string_slice().__ne__(str5.as_string_slice()))
    assert_true(str1.as_string_slice().__ne__(str6))


fn test_slice_bool() raises:
    var str1: String = "abc"
    assert_true(str1.as_string_slice().__bool__())
    var str2: String = ""
    assert_true(not str2.as_string_slice().__bool__())


fn test_utf8_validation() raises:
    var text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam
    varius tellus quis tincidunt dictum. Donec eros orci, ultricies ac metus non
    , rutrum faucibus neque. Nunc ultricies turpis ut lacus consequat dapibus.
    Nulla nec risus a purus volutpat blandit. Donec sit amet massa velit. Aenean
    fermentum libero eu pharetra placerat. Sed id molestie tellus. Fusce
    sollicitudin a purus ac placerat.
    Lorem Ipsum，也称乱数假文或者哑元文本， 是印刷及排版领域所常用的虚拟文字
    由于曾经一台匿名的打印机刻意打乱了一盒印刷字体从而造出一本字体样品书，Lorem
    Ipsum从西元15世纪起就被作为此领域的标准文本使用。它不仅延续了五个世纪，
    还通过了电子排版的挑战，其雏形却依然保存至今。在1960年代，”Leatraset”公司发布了印刷着
    Lorem Ipsum段落的纸张，从而广泛普及了它的使用。最近，计算机桌面出版软件
    למה אנו משתמשים בזה?
    זוהי עובדה מבוססת שדעתו של הקורא תהיה מוסחת על ידי טקטס קריא כאשר הוא יביט בפריסתו. המטרה בשימוש
     ב- Lorem Ipsum הוא שיש לו פחות או יותר תפוצה של אותיות, בניגוד למלל ' יסוי
    יסוי  יסוי', ונותן חזות קריאה יותר.הרבה הוצאות מחשבים ועורכי דפי אינטרנט משתמשים כיום ב-
    Lorem Ipsum כטקסט ברירת המחדל שלהם, וחיפוש של 'lorem ipsum' יחשוף אתרים רבים בראשית
    דרכם.גרסאות רבות נוצרו במהלך השנים, לעתים בשגגה
    Lorem Ipsum е едноставен модел на текст кој се користел во печатарската
    индустрија.
    Lorem Ipsum - це текст-"риба", що використовується в друкарстві та дизайні.
    Lorem Ipsum คือ เนื้อหาจำลองแบบเรียบๆ ที่ใช้กันในธุรกิจงานพิมพ์หรืองานเรียงพิมพ์
    มันได้กลายมาเป็นเนื้อหาจำลองมาตรฐานของธุรกิจดังกล่าวมาตั้งแต่ศตวรรษที่
    Lorem ipsum" في أي محرك بحث ستظهر العديد
     من المواقع الحديثة العهد في نتائج البحث. على مدى السنين
     ظهرت نسخ جديدة ومختلفة من نص لوريم إيبسوم، أحياناً عن طريق
     الصدفة، وأحياناً عن عمد كإدخال بعض العبارات الفكاهية إليها.
    """
    assert_true(_is_valid_utf8(text.unsafe_ptr(), text.byte_length()))
    assert_true(_is_valid_utf8(text.unsafe_ptr(), text.byte_length()))

    var positive = List[List[UInt8]](
        List[UInt8](0x0),
        List[UInt8](0x00),
        List[UInt8](0x66),
        List[UInt8](0x7F),
        List[UInt8](0x00, 0x7F),
        List[UInt8](0x7F, 0x00),
        List[UInt8](0xC2, 0x80),
        List[UInt8](0xDF, 0xBF),
        List[UInt8](0xE0, 0xA0, 0x80),
        List[UInt8](0xE0, 0xA0, 0xBF),
        List[UInt8](0xED, 0x9F, 0x80),
        List[UInt8](0xEF, 0x80, 0xBF),
        List[UInt8](0xF0, 0x90, 0xBF, 0x80),
        List[UInt8](0xF2, 0x81, 0xBE, 0x99),
        List[UInt8](0xF4, 0x8F, 0x88, 0xAA),
    )
    for item in positive:
        assert_true(_is_valid_utf8(item[].unsafe_ptr(), len(item[])))
        assert_true(_is_valid_utf8(item[].unsafe_ptr(), len(item[])))
    var negative = List[List[UInt8]](
        List[UInt8](0x80),
        List[UInt8](0xBF),
        List[UInt8](0xC0, 0x80),
        List[UInt8](0xC1, 0x00),
        List[UInt8](0xC2, 0x7F),
        List[UInt8](0xDF, 0xC0),
        List[UInt8](0xE0, 0x9F, 0x80),
        List[UInt8](0xE0, 0xC2, 0x80),
        List[UInt8](0xED, 0xA0, 0x80),
        List[UInt8](0xED, 0x7F, 0x80),
        List[UInt8](0xEF, 0x80, 0x00),
        List[UInt8](0xF0, 0x8F, 0x80, 0x80),
        List[UInt8](0xF0, 0xEE, 0x80, 0x80),
        List[UInt8](0xF2, 0x90, 0x91, 0x7F),
        List[UInt8](0xF4, 0x90, 0x88, 0xAA),
        List[UInt8](0xF4, 0x00, 0xBF, 0xBF),
        List[UInt8](
            0xC2, 0x80, 0x00, 0x00, 0xE1, 0x80, 0x80, 0x00, 0xC2, 0xC2, 0x80
        ),
        List[UInt8](0x00, 0xC2, 0xC2, 0x80, 0x00, 0x00, 0xE1, 0x80, 0x80),
        List[UInt8](0x00, 0x00, 0x00, 0xF1, 0x80, 0x00),
        List[UInt8](0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF1),
        List[UInt8](0x00, 0x00, 0x00, 0x00, 0xF1, 0x00, 0x80, 0x80),
        List[UInt8](0x00, 0x00, 0xF1, 0x80, 0xC2, 0x80, 0x00),
        List[UInt8](0x00, 0x00, 0xF0, 0x80, 0x80, 0x80),
    )
    for item in negative:
        assert_false(_is_valid_utf8(item[].unsafe_ptr(), len(item[])))
        assert_false(_is_valid_utf8(item[].unsafe_ptr(), len(item[])))


def test_find():
    haystack = str("abcdefg").as_string_slice()
    haystack_with_special_chars = str("abcdefg@#$").as_string_slice()
    haystack_repeated_chars = str("aaaaaaaaaaaaaaaaaaaaaaaa").as_string_slice()

    assert_equal(haystack.find(str("a").as_string_slice()), 0)
    assert_equal(haystack.find(str("ab").as_string_slice()), 0)
    assert_equal(haystack.find(str("abc").as_string_slice()), 0)
    assert_equal(haystack.find(str("bcd").as_string_slice()), 1)
    assert_equal(haystack.find(str("de").as_string_slice()), 3)
    assert_equal(haystack.find(str("fg").as_string_slice()), 5)
    assert_equal(haystack.find(str("g").as_string_slice()), 6)
    assert_equal(haystack.find(str("z").as_string_slice()), -1)
    assert_equal(haystack.find(str("zzz").as_string_slice()), -1)

    assert_equal(haystack.find(str("@#$").as_string_slice()), -1)
    assert_equal(
        haystack_with_special_chars.find(str("@#$").as_string_slice()), 7
    )

    assert_equal(haystack_repeated_chars.find(str("aaa").as_string_slice()), 0)
    assert_equal(haystack_repeated_chars.find(str("AAa").as_string_slice()), -1)

    assert_equal(
        haystack.find(str("hijklmnopqrstuvwxyz").as_string_slice()), -1
    )

    assert_equal(
        str("").as_string_slice().find(str("abc").as_string_slice()), -1
    )


alias GOOD_SEQUENCES = List[String](
    "a",
    "\xc3\xb1",
    "\xe2\x82\xa1",
    "\xf0\x90\x8c\xbc",
    "안녕하세요, 세상",
    "\xc2\x80",
    "\xf0\x90\x80\x80",
    "\xee\x80\x80",
    "very very very long string 🔥🔥🔥",
)


# TODO: later on, don't use String because
# it will likely refuse non-utf8 data.
alias BAD_SEQUENCES = List[String](
    "\xc3\x28",  # continuation bytes does not start with 10xx
    "\xa0\xa1",  # first byte is continuation byte
    "\xe2\x28\xa1",  # second byte should be continuation byte
    "\xe2\x82\x28",  # third byte should be continuation byte
    "\xf0\x28\x8c\xbc",  # second byte should be continuation byte
    "\xf0\x90\x28\xbc",  # third byte should be continuation byte
    "\xf0\x28\x8c\x28",  # fourth byte should be continuation byte
    "\xc0\x9f",  # overlong, could be just one byte
    "\xf5\xff\xff\xff",  # missing continuation bytes
    "\xed\xa0\x81",  # UTF-16 surrogate pair
    "\xf8\x90\x80\x80\x80",  # 5 bytes is too long
    "123456789012345\xed",  # Continuation bytes are missing
    "123456789012345\xf1",  # Continuation bytes are missing
    "123456789012345\xc2",  # Continuation bytes are missing
    "\xC2\x7F",  # second byte is not continuation byte
    "\xce",  # Continuation byte missing
    "\xce\xba\xe1",  # two continuation bytes missing
    "\xce\xba\xe1\xbd",  # One continuation byte missing
    "\xce\xba\xe1\xbd\xb9\xcf",  # fifth byte should be continuation byte
    "\xce\xba\xe1\xbd\xb9\xcf\x83\xce",  # missing continuation byte
    "\xce\xba\xe1\xbd\xb9\xcf\x83\xce\xbc\xce",  # missing continuation byte
    "\xdf",  # missing continuation byte
    "\xef\xbf",  # missing continuation byte
)


fn validate_utf8(slice: String) -> Bool:
    return _is_valid_utf8(slice.unsafe_ptr(), slice.byte_length())


def test_good_utf8_sequences():
    for sequence in GOOD_SEQUENCES:
        assert_true(validate_utf8(sequence[]))


def test_bad_utf8_sequences():
    for sequence in BAD_SEQUENCES:
        assert_false(validate_utf8(sequence[]))


def test_combination_good_utf8_sequences():
    # any combination of good sequences should be good
    for i in range(0, len(GOOD_SEQUENCES)):
        for j in range(i, len(GOOD_SEQUENCES)):
            var sequence = GOOD_SEQUENCES[i] + GOOD_SEQUENCES[j]
            assert_true(validate_utf8(sequence))


def test_combination_bad_utf8_sequences():
    # any combination of bad sequences should be bad
    for i in range(0, len(BAD_SEQUENCES)):
        for j in range(i, len(BAD_SEQUENCES)):
            var sequence = BAD_SEQUENCES[i] + BAD_SEQUENCES[j]
            assert_false(validate_utf8(sequence))


def test_combination_good_bad_utf8_sequences():
    # any combination of good and bad sequences should be bad
    for i in range(0, len(GOOD_SEQUENCES)):
        for j in range(0, len(BAD_SEQUENCES)):
            var sequence = GOOD_SEQUENCES[i] + BAD_SEQUENCES[j]
            assert_false(validate_utf8(sequence))


def test_combination_10_good_utf8_sequences():
    # any 10 combination of good sequences should be good
    for i in range(0, len(GOOD_SEQUENCES)):
        for j in range(i, len(GOOD_SEQUENCES)):
            var sequence = GOOD_SEQUENCES[i] * 10 + GOOD_SEQUENCES[j] * 10
            assert_true(validate_utf8(sequence))


def test_combination_10_good_10_bad_utf8_sequences():
    # any 10 combination of good and bad sequences should be bad
    for i in range(0, len(GOOD_SEQUENCES)):
        for j in range(0, len(BAD_SEQUENCES)):
            var sequence = GOOD_SEQUENCES[i] * 10 + BAD_SEQUENCES[j] * 10
            assert_false(validate_utf8(sequence))


def test_count_utf8_continuation_bytes():
    alias c = UInt8(0b1000_0000)
    alias b1 = UInt8(0b0100_0000)
    alias b2 = UInt8(0b1100_0000)
    alias b3 = UInt8(0b1110_0000)
    alias b4 = UInt8(0b1111_0000)

    def _test(amnt: Int, items: List[UInt8]):
        p = items.unsafe_ptr()
        span = Span[UInt8, StaticConstantOrigin](unsafe_ptr=p, len=len(items))
        assert_equal(amnt, _count_utf8_continuation_bytes(span))

    _test(5, List[UInt8](c, c, c, c, c))
    _test(2, List[UInt8](b2, c, b2, c, b1))
    _test(2, List[UInt8](b2, c, b1, b2, c))
    _test(2, List[UInt8](b2, c, b2, c, b1))
    _test(2, List[UInt8](b2, c, b1, b2, c))
    _test(2, List[UInt8](b1, b2, c, b2, c))
    _test(2, List[UInt8](b3, c, c, b1, b1))
    _test(2, List[UInt8](b1, b1, b3, c, c))
    _test(2, List[UInt8](b1, b3, c, c, b1))
    _test(3, List[UInt8](b1, b4, c, c, c))
    _test(3, List[UInt8](b4, c, c, c, b1))
    _test(3, List[UInt8](b3, c, c, b2, c))
    _test(3, List[UInt8](b2, c, b3, c, c))


def test_split():
    alias S = StringSlice

    # empty separators default to whitespace
    var d = S("hello world").split()
    assert_true(len(d) == 2)
    assert_true(d[0] == "hello")
    assert_true(d[1] == "world")
    d = S("hello \t\n\n\v\fworld").split("\n")
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
        S(" ")
        + S("\t")
        + S("\n")
        + S("\r")
        + S("\v")
        + S("\f")
        + S("\x1c")
        + S("\x1d")
        + S("\x1e")
        + S(next_line)
        + S(unicode_line_sep)
        + S(unicode_paragraph_sep)
    )
    var s = univ_sep_var + "hello" + univ_sep_var + "world" + univ_sep_var
    d = s.split()
    assert_true(len(d) == 2)
    assert_true(d[0] == "hello" and d[1] == "world")

    # should split into empty strings between separators
    d = S("1,,,3").split(",")
    assert_true(len(d) == 4)
    assert_true(d[0] == "1" and d[1] == "" and d[2] == "" and d[3] == "3")
    d = S(",,,").split(",")
    assert_true(len(d) == 4)
    assert_true(d[0] == "" and d[1] == "" and d[2] == "" and d[3] == "")
    d = S(" a b ").split(" ")
    assert_true(len(d) == 4)
    assert_true(d[0] == "" and d[1] == "a" and d[2] == "b" and d[3] == "")
    d = S("abababaaba").split("aba")
    assert_true(len(d) == 4)
    assert_true(d[0] == "" and d[1] == "b" and d[2] == "" and d[3] == "")

    # should split into maxsplit + 1 items
    d = S("1,2,3").split(",", 0)
    assert_true(len(d) == 1)
    assert_true(d[0] == "1,2,3")
    d = S("1,2,3").split(",", 1)
    assert_true(len(d) == 2)
    assert_true(d[0] == "1" and d[1] == "2,3")

    assert_true(len(S("").split()) == 0)
    assert_true(len(S(" ").split()) == 0)
    assert_true(len(S("").split(" ")) == 1)
    assert_true(len(S(" ").split(" ")) == 2)
    assert_true(len(S("  ").split(" ")) == 3)
    assert_true(len(S("   ").split(" ")) == 4)

    # Split in middle
    var d1 = S("n")
    var in1 = S("faang")
    var res1 = in1.split(d1)
    assert_equal(len(res1), 2)
    assert_equal(res1[0], "faa")
    assert_equal(res1[1], "g")

    # Matches should be properly split in multiple case
    var d2 = S(" ")
    var in2 = S("modcon is coming soon")
    var res2 = in2.split(d2)
    assert_equal(len(res2), 4)
    assert_equal(res2[0], "modcon")
    assert_equal(res2[1], "is")
    assert_equal(res2[2], "coming")
    assert_equal(res2[3], "soon")

    # No match from the delimiter
    var d3 = S("x")
    var in3 = S("hello world")
    var res3 = in3.split(d3)
    assert_equal(len(res3), 1)
    assert_equal(res3[0], "hello world")

    # Multiple character delimiter
    var d4 = S("ll")
    var in4 = S("hello")
    var res4 = in4.split(d4)
    assert_equal(len(res4), 2)
    assert_equal(res4[0], "he")
    assert_equal(res4[1], "o")

    # related to #2879
    # TODO: replace string comparison when __eq__ is implemented for List
    s = S("abbaaaabbba").split("a").__str__()
    assert_equal(s, "['', 'bb', '', '', '', 'bbb', '']")
    s = S("abbaaaabbba").split("a", 8).__str__()
    assert_equal(s, "['', 'bb', '', '', '', 'bbb', '']")
    s = S("abbaaaabbba").split("a", 5).__str__()
    assert_equal(s, "['', 'bb', '', '', '', 'bbba']")
    assert_equal(S("aaa").split("a", 0).__str__(), "['aaa']")
    assert_equal(S("a").split("a").__str__(), "['', '']")
    assert_equal(S("1,2,3").split("3", 0).__str__(), "['1,2,3']")
    assert_equal(S("1,2,3").split("3", 1).__str__(), "['1,2,', '']")
    assert_equal(S("1,2,3,3").split("3", 2).__str__(), "['1,2,', ',', '']")
    assert_equal(S("1,2,3,3,3").split("3", 2).__str__(), "['1,2,', ',', ',3']")

    var in5 = S("Hello 🔥!")
    var res5 = in5.split()
    assert_equal(len(res5), 2)
    assert_equal(res5[0], "Hello")
    assert_equal(res5[1], "🔥!")

    var in6 = S("Лорем ипсум долор сит амет")
    var res6 = in6.split(" ")
    assert_equal(len(res6), 5)
    assert_equal(res6[0], "Лорем")
    assert_equal(res6[1], "ипсум")
    assert_equal(res6[2], "долор")
    assert_equal(res6[3], "сит")
    assert_equal(res6[4], "амет")
    var res7 = in6.split("м")
    assert_equal(res7[0], "Лоре")
    assert_equal(res7[1], " ипсу")
    assert_equal(res7[2], " долор сит а")
    assert_equal(res7[3], "ет")

    assert_equal(S("123").split(""), List[String]("", "1", "2", "3", ""))
    assert_equal(S("").join(S("123").split("")), "123")
    assert_equal(S(",1,2,3,").split(","), S("123").split(""))
    assert_equal(S(",").join(S("123").split("")), ",1,2,3,")


def test_join():
    alias S = StringSlice
    l1 = List[UInt8](1, 2, 3, 4, 5, 6, 7, 8, 9)
    assert_equal(S(",").join(l1), "1,2,3,4,5,6,7,8,9")
    assert_equal(S(",").join(List[UInt8](1, 2, 3)), "1,2,3")
    assert_equal(S(",").join(List[UInt8]()), "")
    assert_equal(S(",").join(List[UInt8](1)), "1")
    l2 = List[S]("1", "2", "3")
    assert_equal(S(",").join(l2), "1,2,3")


fn main() raises:
    test_string_literal_byte_span()
    test_string_byte_span()
    test_heap_string_from_string_slice()
    test_slice_len()
    test_slice_eq()
    test_slice_bool()
    test_utf8_validation()
    test_find()
    test_good_utf8_sequences()
    test_bad_utf8_sequences()
    test_combination_good_utf8_sequences()
    test_combination_bad_utf8_sequences()
    test_combination_good_bad_utf8_sequences()
    test_combination_10_good_utf8_sequences()
    test_combination_10_good_10_bad_utf8_sequences()
    test_count_utf8_continuation_bytes()
    test_split()
    test_join()
