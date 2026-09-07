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

from std.collections.string._utf8 import (
    _count_utf8_continuation_bytes,
    _is_utf8_continuation_byte,
    _is_valid_utf8,
    _is_valid_utf8_comptime,
    _is_valid_utf8_runtime,
    _utf8_byte_type,
    BIGGEST_UTF8_FIRST_BYTE,
)

from std.testing import assert_equal, assert_false, assert_raises, assert_true
from std.testing import TestSuite

# ===----------------------------------------------------------------------=== #
# Reusable testing data
# ===----------------------------------------------------------------------=== #


comptime GOOD_SEQUENCES: List[List[Byte]] = [
    List("a".as_bytes()),
    [0xC3, 0xB1],  # U+00F1 (ñ)
    [0xE2, 0x82, 0xA1],  # U+20A1 (₡)
    [0xF0, 0x90, 0x8C, 0xBC],  # U+1033C (𐌼)
    List("안녕하세요, 세상".as_bytes()),
    [0xC2, 0x80],  # U+0080
    [0xF0, 0x90, 0x80, 0x80],  # U+10000
    [0xEE, 0x80, 0x80],  # U+E000
    List("very very very long string 🔥🔥🔥".as_bytes()),
    List(" τo".as_bytes()),
]


comptime BAD_SEQUENCES: List[List[Byte]] = [
    [0xC3, 0x28],  # continuation bytes does not start with 10xx
    [0xA0, 0xA1],  # first byte is continuation byte
    [0xE2, 0x28, 0xA1],  # second byte should be continuation byte
    [0xE2, 0x82, 0x28],  # third byte should be continuation byte
    [0xF0, 0x28, 0x8C, 0xBC],  # second byte should be continuation byte
    [0xF0, 0x90, 0x28, 0xBC],  # third byte should be continuation byte
    [0xF0, 0x28, 0x8C, 0x28],  # fourth byte should be continuation byte
    [0xC0, 0x9F],  # overlong, could be just one byte
    [0xF5, 0xFF, 0xFF, 0xFF],  # missing continuation bytes
    [0xED, 0xA0, 0x81],  # UTF-16 surrogate pair
    [0xF8, 0x90, 0x80, 0x80, 0x80],  # 5 bytes is too long
    List("123456789012345".as_bytes())
    + [0xED],  # Continuation bytes are missing
    List("123456789012345".as_bytes())
    + [0xF1],  # Continuation bytes are missing
    List("123456789012345".as_bytes())
    + [0xC2],  # Continuation bytes are missing
    [0xC2, 0x7F],  # second byte is not continuation byte
    [0xCE],  # Continuation byte missing
    [0xCE, 0xBA, 0xE1],  # two continuation bytes missing
    [0xCE, 0xBA, 0xE1, 0xBD],  # One continuation byte missing
    [
        0xCE,
        0xBA,
        0xE1,
        0xBD,
        0xB9,
        0xCF,
    ],  # fifth byte should be continuation byte
    [
        0xCE,
        0xBA,
        0xE1,
        0xBD,
        0xB9,
        0xCF,
        0x83,
        0xCE,
    ],  # missing continuation byte
    [
        0xCE,
        0xBA,
        0xE1,
        0xBD,
        0xB9,
        0xCF,
        0x83,
        0xCE,
        0xBC,
        0xCE,
    ],  # missing continuation byte
    [0xDF],  # missing continuation byte
    [0xEF, 0xBF],  # missing continuation byte
]

# ===----------------------------------------------------------------------=== #
# Tests
# ===----------------------------------------------------------------------=== #


def validate_utf8[span: Span[Byte, _]]() raises -> Bool:
    comptime comp_time = _is_valid_utf8_comptime(span)
    var runtime = _is_valid_utf8_runtime(span)
    assert_equal(comp_time, runtime)
    return comp_time


def validate_utf8(span: Span[Byte, _]) raises -> Bool:
    var comp_time = _is_valid_utf8_comptime(span)
    var runtime = _is_valid_utf8_runtime(span)
    assert_equal(comp_time, runtime)
    return comp_time


def test_utf8_validation() raises:
    comptime text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam
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
    assert_true(validate_utf8[text.as_bytes()]())

    comptime positive: List[List[UInt8]] = [
        [0x0],
        [0x00],
        [0x66],
        [0x7F],
        [0x00, 0x7F],
        [0x7F, 0x00],
        [0xC2, 0x80],
        [0xDF, 0xBF],
        [0xE0, 0xA0, 0x80],
        [0xE0, 0xA0, 0xBF],
        [0xED, 0x9F, 0x80],
        [0xEF, 0x80, 0xBF],
        [0xF0, 0x90, 0xBF, 0x80],
        [0xF2, 0x81, 0xBE, 0x99],
        [0xF4, 0x8F, 0x88, 0xAA],
    ]

    comptime for i in range(len(positive)):
        assert_true(validate_utf8[positive[i]]())

    comptime negative: List[List[UInt8]] = [
        [0x80],
        [0xBF],
        [0xC0, 0x80],
        [0xC1, 0x00],
        [0xC2, 0x7F],
        [0xDF, 0xC0],
        [0xE0, 0x9F, 0x80],
        [0xE0, 0xC2, 0x80],
        [0xED, 0xA0, 0x80],
        [0xED, 0x7F, 0x80],
        [0xEF, 0x80, 0x00],
        [0xF0, 0x8F, 0x80, 0x80],
        [0xF0, 0xEE, 0x80, 0x80],
        [0xF2, 0x90, 0x91, 0x7F],
        [0xF4, 0x90, 0x88, 0xAA],
        [0xF4, 0x00, 0xBF, 0xBF],
        [0xC2, 0x80, 0x00, 0x00, 0xE1, 0x80, 0x80, 0x00, 0xC2, 0xC2, 0x80],
        [0x00, 0xC2, 0xC2, 0x80, 0x00, 0x00, 0xE1, 0x80, 0x80],
        [0x00, 0x00, 0x00, 0xF1, 0x80, 0x00],
        [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF1],
        [0x00, 0x00, 0x00, 0x00, 0xF1, 0x00, 0x80, 0x80],
        [0x00, 0x00, 0xF1, 0x80, 0xC2, 0x80, 0x00],
        [0x00, 0x00, 0xF0, 0x80, 0x80, 0x80],
    ]

    comptime for i in range(len(negative)):
        assert_false(validate_utf8[negative[i]]())


def test_good_utf8_sequences() raises:
    comptime for i in range(len(GOOD_SEQUENCES)):
        assert_true(validate_utf8[GOOD_SEQUENCES[i]]())


def test_bad_utf8_sequences() raises:
    comptime for i in range(len(BAD_SEQUENCES)):
        assert_false(validate_utf8[BAD_SEQUENCES[i]]())


def test_stringslice_from_utf8() raises:
    for sequence in materialize[GOOD_SEQUENCES]():
        _ = StringSlice(from_utf8=Span(sequence))

    for sequence in materialize[BAD_SEQUENCES]():
        with assert_raises(contains="buffer is not valid UTF-8"):
            _ = StringSlice(from_utf8=Span(sequence))


def test_combination_good_utf8_sequences() raises:
    # any combination of good sequences should be good
    var good_sequence = materialize[GOOD_SEQUENCES]()
    for i in range(0, len(good_sequence)):
        for j in range(i, len(good_sequence)):
            var sequence = good_sequence[i] + good_sequence[j].copy()
            assert_true(validate_utf8(Span(sequence)))


def test_combination_bad_utf8_sequences() raises:
    # any combination of bad sequences should be bad
    var bad_sequence = materialize[BAD_SEQUENCES]()
    for i in range(0, len(bad_sequence)):
        for j in range(i, len(bad_sequence)):
            var sequence = bad_sequence[i] + bad_sequence[j].copy()
            assert_false(validate_utf8(Span(sequence)))


def test_combination_good_bad_utf8_sequences() raises:
    # any combination of good and bad sequences should be bad
    var good_sequence = materialize[GOOD_SEQUENCES]()
    var bad_sequence = materialize[BAD_SEQUENCES]()
    for i in range(0, len(good_sequence)):
        for j in range(0, len(bad_sequence)):
            var sequence = good_sequence[i] + bad_sequence[j].copy()
            assert_false(validate_utf8(Span(sequence)))


def test_combination_10_good_utf8_sequences() raises:
    # any 10 combination of good sequences should be good
    var good_sequence = materialize[GOOD_SEQUENCES]()
    for i in range(0, len(good_sequence)):
        for j in range(i, len(good_sequence)):
            var sequence = good_sequence[i] * 10 + good_sequence[j] * 10
            assert_true(validate_utf8(Span(sequence)))


def test_combination_10_good_10_bad_utf8_sequences() raises:
    # any 10 combination of good and bad sequences should be bad
    var good_sequence = materialize[GOOD_SEQUENCES]()
    var bad_sequence = materialize[BAD_SEQUENCES]()
    for i in range(0, len(good_sequence)):
        for j in range(0, len(bad_sequence)):
            var sequence = good_sequence[i] * 10 + bad_sequence[j] * 10
            assert_false(validate_utf8(Span(sequence)))


def test_count_utf8_continuation_bytes() raises:
    comptime c = UInt8(0b1000_0000)
    comptime b1 = UInt8(0b0100_0000)
    comptime b2 = UInt8(0b1100_0000)
    comptime b3 = UInt8(0b1110_0000)
    comptime b4 = UInt8(0b1111_0000)

    for i in range(c):
        assert_false(_is_utf8_continuation_byte(i))

    for i in range(c, b2):
        assert_true(_is_utf8_continuation_byte(i))

    for i in range(b2, UInt8.MAX):
        assert_false(_is_utf8_continuation_byte(i))

    def _test(amnt: Int, items: List[UInt8]) raises:
        var p = items.unsafe_ptr()
        var span = Span(unsafe_ptr=p, length=len(items))
        assert_equal(amnt, _count_utf8_continuation_bytes(span))

    _test(5, [c, c, c, c, c])
    _test(2, [b2, c, b2, c, b1])
    _test(2, [b2, c, b1, b2, c])
    _test(2, [b2, c, b2, c, b1])
    _test(2, [b2, c, b1, b2, c])
    _test(2, [b1, b2, c, b2, c])
    _test(2, [b3, c, c, b1, b1])
    _test(2, [b1, b1, b3, c, c])
    _test(2, [b1, b3, c, c, b1])
    _test(3, [b1, b4, c, c, c])
    _test(3, [b4, c, c, c, b1])
    _test(3, [b3, c, c, b2, c])
    _test(3, [b2, c, b3, c, c])


def test_utf8_byte_type() raises:
    for i in range(UInt8(0b1000_0000)):
        assert_equal(_utf8_byte_type(i), 0)
    for i in range(UInt8(0b1000_0000), UInt8(0b1100_0000)):
        assert_equal(_utf8_byte_type(i), 1)
    for i in range(UInt8(0b1100_0000), UInt8(0b1110_0000)):
        assert_equal(_utf8_byte_type(i), 2)
    for i in range(UInt8(0b1110_0000), UInt8(0b1111_0000)):
        assert_equal(_utf8_byte_type(i), 3)
    for i in range(UInt8(0b1111_0000), BIGGEST_UTF8_FIRST_BYTE + 1):
        assert_equal(_utf8_byte_type(i), 4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
