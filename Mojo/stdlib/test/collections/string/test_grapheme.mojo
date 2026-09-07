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
"""Tests for grapheme cluster segmentation and String grapheme APIs."""

from std.collections.string import StringSlice
from std.testing import (
    assert_equal,
    assert_true,
    TestSuite,
)


# ===----------------------------------------------------------------------=== #
# Helper: build strings from codepoint values
# ===----------------------------------------------------------------------=== #


def _string_from_codepoints(*cps: Int) -> String:
    """Build a String from a sequence of Unicode codepoint values."""
    var s = String()
    for i in range(len(cps)):
        s += chr(cps[i])
    return s


# ===----------------------------------------------------------------------=== #
# Basic grapheme counting
# ===----------------------------------------------------------------------=== #


def test_empty_string() raises:
    assert_equal(String("").count_graphemes(), 0)
    assert_equal(StringSlice("").count_graphemes(), 0)


def test_ascii() raises:
    assert_equal(String("Hello").count_graphemes(), 5)
    assert_equal(String("a").count_graphemes(), 1)
    assert_equal(String("abc").count_graphemes(), 3)


def test_ascii_matches_byte_length() raises:
    var s = String("Hello, World!")
    assert_equal(s.count_graphemes(), s.byte_length())


# ===----------------------------------------------------------------------=== #
# Combining marks
# ===----------------------------------------------------------------------=== #


def test_precomposed_accent() raises:
    # Precomposed "café" (é = U+00E9, single codepoint)
    assert_equal(String("café").count_graphemes(), 4)


def test_combining_accent() raises:
    # Decomposed: "cafe" + combining acute accent (U+0301)
    # 5 codepoints but 4 graphemes
    var decomposed = String("caf") + _string_from_codepoints(0x65, 0x0301)
    assert_equal(decomposed.count_codepoints(), 5)
    assert_equal(decomposed.count_graphemes(), 4)


def test_multiple_combining_marks() raises:
    # a + combining acute (U+0301) + combining cedilla (U+0327) = 1 grapheme
    var s = _string_from_codepoints(0x61, 0x0301, 0x0327)
    assert_equal(s.count_codepoints(), 3)
    assert_equal(s.count_graphemes(), 1)


def test_lone_combining_mark() raises:
    # A standalone combining mark is its own grapheme cluster
    var s = _string_from_codepoints(0x0301)
    assert_equal(s.count_graphemes(), 1)


# ===----------------------------------------------------------------------=== #
# Emoji
# ===----------------------------------------------------------------------=== #


def test_simple_emoji() raises:
    # U+1F600 (😀) - single emoji
    var s = chr(0x1F600)
    assert_equal(s.count_graphemes(), 1)


def test_emoji_with_skin_tone() raises:
    # Waving hand (U+1F44B) + medium skin tone (U+1F3FD) = 1 grapheme
    var s = _string_from_codepoints(0x1F44B, 0x1F3FD)
    assert_equal(s.count_codepoints(), 2)
    assert_equal(s.count_graphemes(), 1)


def test_emoji_zwj_sequence() raises:
    # Family: man + ZWJ + woman + ZWJ + girl + ZWJ + boy = 1 grapheme
    var family = _string_from_codepoints(
        0x1F468, 0x200D, 0x1F469, 0x200D, 0x1F467, 0x200D, 0x1F466
    )
    assert_equal(family.count_graphemes(), 1)


def test_flag_emoji() raises:
    # US flag: Regional Indicator U + Regional Indicator S = 1 grapheme
    var us_flag = _string_from_codepoints(0x1F1FA, 0x1F1F8)
    assert_equal(us_flag.count_codepoints(), 2)
    assert_equal(us_flag.count_graphemes(), 1)


def test_three_regional_indicators() raises:
    # Three RIs: first two pair into a flag, third is standalone
    var s = _string_from_codepoints(0x1F1FA, 0x1F1F8, 0x1F1E6)
    assert_equal(s.count_graphemes(), 2)


def test_four_regional_indicators() raises:
    # Four RIs: two pairs = 2 flag graphemes
    var s = _string_from_codepoints(0x1F1FA, 0x1F1F8, 0x1F1E6, 0x1F1E7)
    assert_equal(s.count_graphemes(), 2)


# ===----------------------------------------------------------------------=== #
# CR/LF
# ===----------------------------------------------------------------------=== #


def test_crlf() raises:
    # CR + LF = 1 grapheme
    assert_equal(String("\r\n").count_graphemes(), 1)


def test_cr_alone() raises:
    assert_equal(String("\r").count_graphemes(), 1)


def test_lf_alone() raises:
    assert_equal(String("\n").count_graphemes(), 1)


def test_cr_cr() raises:
    assert_equal(String("\r\r").count_graphemes(), 2)


def test_crlf_in_text() raises:
    assert_equal(String("a\r\nb").count_graphemes(), 3)


# ===----------------------------------------------------------------------=== #
# Grapheme iteration
# ===----------------------------------------------------------------------=== #


def test_iter_ascii() raises:
    var result = List[String]()
    for g in String("abc").graphemes():
        result.append(String(g))
    assert_equal(len(result), 3)
    assert_equal(result[0], "a")
    assert_equal(result[1], "b")
    assert_equal(result[2], "c")


def test_iter_combining() raises:
    # "caf" + e + combining acute: 4 graphemes
    var s = String("caf") + _string_from_codepoints(0x65, 0x0301)
    var result = List[String]()
    for g in s.graphemes():
        result.append(String(g))
    assert_equal(len(result), 4)
    assert_equal(result[0], "c")
    assert_equal(result[1], "a")
    assert_equal(result[2], "f")
    # The last grapheme is e + combining acute
    assert_equal(result[3], _string_from_codepoints(0x65, 0x0301))


def test_iter_empty() raises:
    var count = 0
    for _ in String("").graphemes():
        count += 1
    assert_equal(count, 0)


# ===----------------------------------------------------------------------=== #
# StringSlice grapheme APIs
# ===----------------------------------------------------------------------=== #


def test_string_slice_graphemes() raises:
    var s = StringSlice("Hello")
    assert_equal(s.count_graphemes(), 5)


def test_string_slice_iter() raises:
    var result = List[String]()
    for g in StringSlice("abc").graphemes():
        result.append(String(g))
    assert_equal(len(result), 3)


# ===----------------------------------------------------------------------=== #
# Hangul
# ===----------------------------------------------------------------------=== #


def test_hangul_syllable() raises:
    # Korean "한글" - each is a precomposed LVT syllable = 2 graphemes
    var s = _string_from_codepoints(0xD55C, 0xAE00)
    assert_equal(s.count_graphemes(), 2)


# ===----------------------------------------------------------------------=== #
# Mixed content
# ===----------------------------------------------------------------------=== #


def test_mixed_ascii_and_emoji() raises:
    var s = String("Hi ") + chr(0x1F44B)  # "Hi 👋"
    assert_equal(s.count_graphemes(), 4)  # H, i, space, 👋


def test_mixed_scripts() raises:
    # Latin + CJK + emoji
    var s = String("A") + chr(0x4E16) + chr(0x1F600)  # "A世😀"
    assert_equal(s.count_graphemes(), 3)


# ===----------------------------------------------------------------------=== #
# Indic conjunct clusters (GB9c)
# ===----------------------------------------------------------------------=== #


def test_indic_conjunct_devanagari() raises:
    # Devanagari: क (U+0915) + virama (U+094D) + ष (U+0937) = 1 grapheme
    # This exercises GB9c: Consonant + Linker + Consonant
    var s = _string_from_codepoints(0x0915, 0x094D, 0x0937)
    assert_equal(s.count_codepoints(), 3)
    assert_equal(s.count_graphemes(), 1)


def test_indic_conjunct_with_extend() raises:
    # Consonant + Extend + Linker + Consonant = 1 grapheme
    # क (U+0915) + nukta (U+093C) + virama (U+094D) + ष (U+0937)
    var s = _string_from_codepoints(0x0915, 0x093C, 0x094D, 0x0937)
    assert_equal(s.count_codepoints(), 4)
    assert_equal(s.count_graphemes(), 1)


def test_indic_conjunct_no_linker_breaks() raises:
    # Two consonants without a linker (virama) between them = 2 graphemes
    # क (U+0915) + ष (U+0937)
    var s = _string_from_codepoints(0x0915, 0x0937)
    assert_equal(s.count_graphemes(), 2)


# ===----------------------------------------------------------------------=== #
# Extended Pictographic sequences
# ===----------------------------------------------------------------------=== #


def test_keycap_sequence() raises:
    # Digit + U+FE0F (VS16) + U+20E3 (combining enclosing keycap) = 1 grapheme
    var s = _string_from_codepoints(0x31, 0xFE0F, 0x20E3)  # 1️⃣
    assert_equal(s.count_graphemes(), 1)


# ===----------------------------------------------------------------------=== #
# grapheme=: slice indexer
# ===----------------------------------------------------------------------=== #


def test_grapheme_slice_ascii() raises:
    var s = StringSlice("Hello")
    assert_equal(s[grapheme=0:5], "Hello")
    assert_equal(s[grapheme=0:1], "H")
    assert_equal(s[grapheme=1:4], "ell")
    assert_equal(s[grapheme=4:5], "o")
    assert_equal(s[grapheme=:], "Hello")
    assert_equal(s[grapheme=:3], "Hel")
    assert_equal(s[grapheme=2:], "llo")


def test_grapheme_slice_empty_ranges() raises:
    var s = StringSlice("Hello")
    assert_equal(s[grapheme=0:0], "")
    assert_equal(s[grapheme=3:3], "")
    assert_equal(s[grapheme=5:5], "")


def test_grapheme_slice_at_end_boundary() raises:
    var s = StringSlice("Hi")
    # `start`/`end` exactly at the grapheme count is a valid (empty)
    # boundary, matching `byte=` slicing. An end past the count, or a start
    # past the count, aborts instead of clamping (see
    # test_string_slice_codepoint_grapheme_bounds_abort.mojo).
    assert_equal(s[grapheme=2:2], "")
    assert_equal(s[grapheme=2:], "")
    assert_equal(s[grapheme=0:2], "Hi")


def test_grapheme_slice_empty_string() raises:
    var s = StringSlice("")
    assert_equal(s[grapheme=:], "")
    assert_equal(s[grapheme=0:0], "")


def test_grapheme_slice_combining_mark() raises:
    # "café" decomposed: 'c','a','f','e' + combining acute (U+0301).
    # 5 codepoints but 4 graphemes; the 4th grapheme spans 3 bytes.
    var e_acute = _string_from_codepoints(0x65, 0x0301)
    var decomposed = String("caf") + e_acute
    var s = StringSlice(decomposed)
    assert_equal(s.count_graphemes(), 4)
    assert_equal(s[grapheme=0:3], "caf")
    assert_equal(s[grapheme=3:4], e_acute)
    assert_equal(s[grapheme=3:], e_acute)
    assert_equal(s[grapheme=:4], String("caf") + e_acute)


def test_grapheme_slice_emoji_zwj() raises:
    # "a" + family ZWJ sequence + "b": 3 graphemes, but 9 codepoints.
    var family = _string_from_codepoints(
        0x1F468, 0x200D, 0x1F469, 0x200D, 0x1F467, 0x200D, 0x1F466
    )
    var s = String("a") + family + String("b")
    var slc = StringSlice(s)
    assert_equal(slc.count_graphemes(), 3)
    assert_equal(slc[grapheme=0:1], "a")
    assert_equal(slc[grapheme=1:2], family)
    assert_equal(slc[grapheme=2:3], "b")
    assert_equal(slc[grapheme=0:2], String("a") + family)


def test_grapheme_slice_flag_emoji() raises:
    # US flag: two Regional Indicators = 1 grapheme, 8 bytes.
    var flag = _string_from_codepoints(0x1F1FA, 0x1F1F8)
    var s = String("A") + flag + String("B")
    var slc = StringSlice(s)
    assert_equal(slc.count_graphemes(), 3)
    assert_equal(slc[grapheme=1:2], flag)
    assert_equal(slc[grapheme=0:3], String("A") + flag + String("B"))


def test_grapheme_slice_crlf() raises:
    # CR+LF is a single grapheme cluster.
    var s = StringSlice("a\r\nb")
    assert_equal(s.count_graphemes(), 3)
    assert_equal(s[grapheme=1:2], "\r\n")
    assert_equal(s[grapheme=0:2], "a\r\n")


# ===----------------------------------------------------------------------=== #
# Reverse grapheme iteration
# ===----------------------------------------------------------------------=== #


def _assert_reverse_matches(s: StringSlice) raises:
    """Assert that iterating s.graphemes_reversed() yields the same clusters
    as s.graphemes() in reverse order."""
    var fwd = List[String]()
    for g in s.graphemes():
        fwd.append(String(g))
    var rev = List[String]()
    for g in s.graphemes_reversed():
        rev.append(String(g))
    assert_equal(len(fwd), len(rev))
    for i in range(len(fwd)):
        assert_equal(
            fwd[len(fwd) - 1 - i],
            rev[i],
            msg=String(t"forward[{len(fwd) - 1 - i}] != reversed[{i}]"),
        )


def test_next_back_ascii() raises:
    var s = StringSlice("abc")
    var iter = s.graphemes()
    assert_equal(iter.next_back().value(), "c")
    assert_equal(iter.next_back().value(), "b")
    assert_equal(iter.next_back().value(), "a")
    assert_true(iter.next_back() is None)


def test_next_back_empty() raises:
    var iter = StringSlice("").graphemes()
    assert_true(iter.next_back() is None)


def test_peek_back_does_not_advance() raises:
    var iter = StringSlice("abc").graphemes()
    assert_equal(iter.peek_back().value(), "c")
    assert_equal(iter.peek_back().value(), "c")
    assert_equal(iter.next_back().value(), "c")
    assert_equal(iter.peek_back().value(), "b")


def test_next_back_combining_mark() raises:
    # "caf" + e + combining acute: 4 graphemes.
    # next_back() must return the full 2-codepoint cluster, not just the mark.
    var e_acute = _string_from_codepoints(0x65, 0x0301)
    var s = String("caf") + e_acute
    var slc = StringSlice(s)
    var iter = slc.graphemes()
    assert_equal(iter.next_back().value(), e_acute)
    assert_equal(iter.next_back().value(), "f")
    assert_equal(iter.next_back().value(), "a")
    assert_equal(iter.next_back().value(), "c")
    assert_true(iter.next_back() is None)


def test_next_back_emoji_zwj() raises:
    # Family ZWJ sequence (7 codepoints, 1 grapheme) between ASCII markers.
    var family = _string_from_codepoints(
        0x1F468, 0x200D, 0x1F469, 0x200D, 0x1F467, 0x200D, 0x1F466
    )
    var s = String("a") + family + String("b")
    var slc = StringSlice(s)
    var iter = slc.graphemes()
    assert_equal(iter.next_back().value(), "b")
    assert_equal(iter.next_back().value(), family)
    assert_equal(iter.next_back().value(), "a")


def test_next_back_flag_emoji() raises:
    # Two RIs = 1 flag grapheme.
    var flag = _string_from_codepoints(0x1F1FA, 0x1F1F8)
    var s = String("A") + flag + String("B")
    var slc = StringSlice(s)
    var iter = slc.graphemes()
    assert_equal(iter.next_back().value(), "B")
    assert_equal(iter.next_back().value(), flag)
    assert_equal(iter.next_back().value(), "A")


def test_next_back_crlf_single_grapheme() raises:
    # CR+LF must remain a single grapheme cluster on backward scan.
    var slc = StringSlice("a\r\nb")
    var iter = slc.graphemes()
    assert_equal(iter.next_back().value(), "b")
    assert_equal(iter.next_back().value(), "\r\n")
    assert_equal(iter.next_back().value(), "a")


def test_next_back_indic_conjunct() raises:
    # क + virama + ष = 1 grapheme via GB9c.
    var s = _string_from_codepoints(0x0915, 0x094D, 0x0937)
    var slc = StringSlice(s)
    var iter = slc.graphemes()
    assert_equal(
        iter.next_back().value(),
        _string_from_codepoints(0x0915, 0x094D, 0x0937),
    )
    assert_true(iter.next_back() is None)


def test_alternating_next_and_next_back() raises:
    # "abcde" -> a, e, b, d, c when alternating.
    var iter = StringSlice("abcde").graphemes()
    assert_equal(iter.next().value(), "a")
    assert_equal(iter.next_back().value(), "e")
    assert_equal(iter.next().value(), "b")
    assert_equal(iter.next_back().value(), "d")
    assert_equal(iter.next().value(), "c")
    assert_true(iter.next() is None)
    assert_true(iter.next_back() is None)


def test_graphemes_reversed_for_loop() raises:
    var s = StringSlice("abc")
    var result = List[String]()
    for g in s.graphemes_reversed():
        result.append(String(g))
    assert_equal(len(result), 3)
    assert_equal(result[0], "c")
    assert_equal(result[1], "b")
    assert_equal(result[2], "a")


def test_reverse_matches_forward_various() raises:
    _assert_reverse_matches(StringSlice(""))
    _assert_reverse_matches(StringSlice("a"))
    _assert_reverse_matches(StringSlice("Hello, World!"))
    _assert_reverse_matches(StringSlice("a\r\nb\r\nc"))

    # Combining mark
    var e_acute = _string_from_codepoints(0x65, 0x0301)
    _assert_reverse_matches(StringSlice(String("caf") + e_acute))

    # Emoji ZWJ family
    var family = _string_from_codepoints(
        0x1F468, 0x200D, 0x1F469, 0x200D, 0x1F467, 0x200D, 0x1F466
    )
    _assert_reverse_matches(StringSlice(String("a") + family + String("b")))

    # Flag emoji
    var flag = _string_from_codepoints(0x1F1FA, 0x1F1F8)
    _assert_reverse_matches(StringSlice(String("A") + flag + String("B")))

    # Mixed scripts
    var mixed = String("A") + chr(0x4E16) + chr(0x1F600)  # "A世😀"
    _assert_reverse_matches(StringSlice(mixed))

    # Indic conjunct
    var indic = _string_from_codepoints(0x0915, 0x094D, 0x0937)
    _assert_reverse_matches(StringSlice(indic))

    # Keycap sequence
    var keycap = _string_from_codepoints(0x31, 0xFE0F, 0x20E3)
    _assert_reverse_matches(StringSlice(keycap))


# ===----------------------------------------------------------------------=== #
# grapheme_indices()
# ===----------------------------------------------------------------------=== #


def test_grapheme_indices_ascii() raises:
    var s = StringSlice("abc")
    var offs = List[Int]()
    var parts = List[String]()
    for off, g in s.grapheme_indices():
        offs.append(off)
        parts.append(String(g))
    assert_equal(len(offs), 3)
    assert_equal(offs[0], 0)
    assert_equal(offs[1], 1)
    assert_equal(offs[2], 2)
    assert_equal(parts[0], "a")
    assert_equal(parts[1], "b")
    assert_equal(parts[2], "c")


def test_grapheme_indices_empty() raises:
    var count = 0
    for _, _ in StringSlice("").grapheme_indices():
        count += 1
    assert_equal(count, 0)


def test_grapheme_indices_combining_mark() raises:
    # "caf" + e+̀ (combining acute, 2 codepoints, 3 bytes).
    var e_acute = _string_from_codepoints(0x65, 0x0301)
    var s = String("caf") + e_acute
    var offs = List[Int]()
    var widths = List[Int]()
    for off, g in StringSlice(s).grapheme_indices():
        offs.append(off)
        widths.append(g.byte_length())
    assert_equal(len(offs), 4)
    assert_equal(offs[0], 0)
    assert_equal(offs[1], 1)
    assert_equal(offs[2], 2)
    assert_equal(offs[3], 3)
    assert_equal(widths[3], 3)  # e (1 byte) + combining acute (2 bytes)


def test_grapheme_indices_emoji_zwj() raises:
    # family ZWJ sequence = 1 grapheme spanning many bytes.
    var family = _string_from_codepoints(
        0x1F468, 0x200D, 0x1F469, 0x200D, 0x1F467, 0x200D, 0x1F466
    )
    var s = String("a") + family + String("b")
    var offs = List[Int]()
    var widths = List[Int]()
    for off, g in StringSlice(s).grapheme_indices():
        offs.append(off)
        widths.append(g.byte_length())
    assert_equal(len(offs), 3)
    assert_equal(offs[0], 0)
    assert_equal(offs[1], 1)
    assert_equal(widths[1], family.byte_length())
    assert_equal(offs[2], 1 + family.byte_length())
    assert_equal(widths[2], 1)


def test_grapheme_indices_crlf() raises:
    # CR+LF is a single grapheme spanning 2 bytes.
    var offs = List[Int]()
    var widths = List[Int]()
    for off, g in StringSlice("a\r\nb").grapheme_indices():
        offs.append(off)
        widths.append(g.byte_length())
    assert_equal(len(offs), 3)
    assert_equal(offs[0], 0)
    assert_equal(offs[1], 1)
    assert_equal(widths[1], 2)
    assert_equal(offs[2], 3)


# ===----------------------------------------------------------------------=== #
# nth_grapheme()
# ===----------------------------------------------------------------------=== #


def test_nth_grapheme_ascii() raises:
    var s = StringSlice("abc")
    assert_equal(s.graphemes().nth(0).value(), "a")
    assert_equal(s.graphemes().nth(1).value(), "b")
    assert_equal(s.graphemes().nth(2).value(), "c")
    assert_true(s.graphemes().nth(3) is None)
    assert_true(s.graphemes().nth(99) is None)


def test_nth_grapheme_empty() raises:
    var s = StringSlice("")
    assert_true(s.graphemes().nth(0) is None)


def test_nth_grapheme_combining_mark() raises:
    # 4 graphemes: c, a, f, e+́
    var e_acute = _string_from_codepoints(0x65, 0x0301)
    var s = String("caf") + e_acute
    var slc = StringSlice(s)
    assert_equal(slc.graphemes().nth(3).value(), e_acute)
    assert_true(slc.graphemes().nth(4) is None)


def test_nth_grapheme_flag_emoji() raises:
    var flag = _string_from_codepoints(0x1F1FA, 0x1F1F8)
    var s = String("A") + flag + String("B")
    var slc = StringSlice(s)
    assert_equal(slc.graphemes().nth(0).value(), "A")
    assert_equal(slc.graphemes().nth(1).value(), flag)
    assert_equal(slc.graphemes().nth(2).value(), "B")
    assert_true(slc.graphemes().nth(3) is None)


# ===----------------------------------------------------------------------=== #
# split_at_grapheme()
# ===----------------------------------------------------------------------=== #


def test_split_at_grapheme_ascii() raises:
    var s = StringSlice("Hello, World!")
    var prefix, suffix = s.split_at_grapheme(5)
    assert_equal(prefix, "Hello")
    assert_equal(suffix, ", World!")


def test_split_at_grapheme_zero() raises:
    var s = StringSlice("Hello")
    var prefix, suffix = s.split_at_grapheme(0)
    assert_equal(prefix, "")
    assert_equal(suffix, "Hello")


def test_split_at_grapheme_end() raises:
    var s = StringSlice("Hello")
    var prefix, suffix = s.split_at_grapheme(5)
    assert_equal(prefix, "Hello")
    assert_equal(suffix, "")


def test_split_at_grapheme_past_end() raises:
    var s = StringSlice("Hi")
    var prefix, suffix = s.split_at_grapheme(99)
    assert_equal(prefix, "Hi")
    assert_equal(suffix, "")


def test_split_at_grapheme_empty_string() raises:
    var s = StringSlice("")
    var prefix, suffix = s.split_at_grapheme(0)
    assert_equal(prefix, "")
    assert_equal(suffix, "")
    var p2, s2 = s.split_at_grapheme(5)
    assert_equal(p2, "")
    assert_equal(s2, "")


def test_split_at_grapheme_combining_mark() raises:
    # 4 graphemes: c, a, f, e+́. Splitting at 3 must put the whole
    # e+combining acute in the suffix, not split the cluster.
    var e_acute = _string_from_codepoints(0x65, 0x0301)
    var s = String("caf") + e_acute
    var prefix, suffix = StringSlice(s).split_at_grapheme(3)
    assert_equal(prefix, "caf")
    assert_equal(suffix, e_acute)


def test_split_at_grapheme_emoji_zwj() raises:
    # 3 graphemes: "a", family, "b". Split at 2 → ("a" + family, "b").
    var family = _string_from_codepoints(
        0x1F468, 0x200D, 0x1F469, 0x200D, 0x1F467, 0x200D, 0x1F466
    )
    var s = String("a") + family + String("b")
    var prefix, suffix = StringSlice(s).split_at_grapheme(2)
    assert_equal(prefix, String("a") + family)
    assert_equal(suffix, "b")


# ===----------------------------------------------------------------------=== #
# ASCII fast path
# ===----------------------------------------------------------------------=== #


def test_ascii_fast_path_count() raises:
    # Pure safe-ASCII (U+0020..U+007E): every byte is one grapheme.
    var s = (
        String(
            "The quick brown fox jumps over the lazy dog. 0123456789"
            " !?.,;:'\"-_()[]{}<>"
        )
        * 20
    )
    assert_equal(s.count_graphemes(), s.byte_length())


def test_ascii_fast_path_iter() raises:
    # Iterating pure safe-ASCII returns one byte per grapheme.
    var s = String("hello world")
    var result = List[String]()
    for g in s.graphemes():
        assert_equal(g.byte_length(), 1)
        result.append(String(g))
    assert_equal(len(result), s.byte_length())
    assert_equal(result[0], "h")
    assert_equal(result[4], "o")
    assert_equal(result[5], " ")


def test_ascii_fast_path_with_embedded_nonascii() raises:
    # ASCII run, then a combining-mark-carrying codepoint, then ASCII run.
    # The character right before the combining mark must NOT be split off
    # as its own grapheme.
    var e_acute = _string_from_codepoints(0x65, 0x0301)
    var s = String("caf") + e_acute + String("! ok")
    var slc = StringSlice(s)
    # Expected: c, a, f, e+́, !, space, o, k -> 8 graphemes
    assert_equal(slc.count_graphemes(), 8)
    var parts = List[String]()
    for g in slc.graphemes():
        parts.append(String(g))
    assert_equal(parts[0], "c")
    assert_equal(parts[1], "a")
    assert_equal(parts[2], "f")
    assert_equal(parts[3], e_acute)
    assert_equal(parts[4], "!")
    assert_equal(parts[5], " ")
    assert_equal(parts[6], "o")
    assert_equal(parts[7], "k")


def test_ascii_fast_path_with_control_chars() raises:
    # Tab (U+0009) is a Control codepoint, NOT in the fast-path range.
    # "a\tb" should be three graphemes.
    var s = StringSlice("a\tb")
    assert_equal(s.count_graphemes(), 3)
    var parts = List[String]()
    for g in s.graphemes():
        parts.append(String(g))
    assert_equal(parts[0], "a")
    assert_equal(parts[1], "\t")
    assert_equal(parts[2], "b")


# ===----------------------------------------------------------------------=== #
# Default iteration: __iter__ / __reversed__ yield grapheme clusters
# ===----------------------------------------------------------------------=== #


def check_default_iteration(result: List[String], *, forward: Bool) raises:
    assert_equal(len(result), 4)
    if forward:
        assert_equal(result, ["a", "👨‍👩‍👧‍👦", "🇺🇸", "b"])
    else:
        assert_equal(result, ["b", "🇺🇸", "👨‍👩‍👧‍👦", "a"])


def test_string_iter_default_yields_graphemes() raises:
    var s = String("a👨‍👩‍👧‍👦🇺🇸b")
    var result = List[String]()
    for g in s:
        result.append(String(g))
    check_default_iteration(result, forward=True)


def test_string_slice_iter_default_yields_graphemes() raises:
    var s = StringSlice("a👨‍👩‍👧‍👦🇺🇸b")
    var result = List[String]()
    for g in s:
        result.append(String(g))
    check_default_iteration(result, forward=True)


def test_string_literal_iter_default_yields_graphemes() raises:
    var result = List[String]()
    for g in "a👨‍👩‍👧‍👦🇺🇸b":
        result.append(String(g))
    check_default_iteration(result, forward=True)


def test_string_reversed_default_yields_graphemes() raises:
    var s = String("a👨‍👩‍👧‍👦🇺🇸b")
    var result = List[String]()
    for g in s.__reversed__():
        result.append(String(g))
    check_default_iteration(result, forward=False)


def test_string_slice_reversed_default_yields_graphemes() raises:
    var s = StringSlice("a👨‍👩‍👧‍👦🇺🇸b")
    var result = List[String]()
    for g in s.__reversed__():
        result.append(String(g))
    check_default_iteration(result, forward=False)


def test_string_literal_reversed_default_yields_graphemes() raises:
    var result = List[String]()
    for g in "a👨‍👩‍👧‍👦🇺🇸b".__reversed__():
        result.append(String(g))
    check_default_iteration(result, forward=False)


def test_string_and_string_slice_default_iteration_empty() raises:
    var count = 0
    for _g in String(""):
        count += 1
    for _g in StringSlice(""):
        count += 1
    assert_equal(count, 0)


# ===----------------------------------------------------------------------=== #
# Test runner
# ===----------------------------------------------------------------------=== #


def main() raises:
    var suite = TestSuite.discover_tests[__functions_in_module()]()
    suite^.run()
