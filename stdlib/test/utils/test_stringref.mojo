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

from testing import assert_equal, assert_false, assert_raises, assert_true

from utils import StringRef


def test_strref_from_start():
    var str = StringRef("Hello")

    assert_equal(5, len(str))

    assert_equal(str._from_start(0), "Hello")
    assert_equal(str._from_start(1), "ello")
    assert_equal(str._from_start(4), "o")
    assert_equal(str._from_start(5), "")
    assert_equal(str._from_start(10), "")

    assert_equal(str._from_start(-1), "o")
    assert_equal(str._from_start(-3), "llo")
    assert_equal(str._from_start(-5), "Hello")
    assert_equal(str._from_start(-10), "Hello")


fn test_comparison_operators() raises:
    var abc = StringRef("abc")
    var de = StringRef("de")
    var ABC = StringRef("ABC")
    var ab = StringRef("ab")
    var abcd = StringRef("abcd")

    # Test less than and greater than
    assert_true(StringRef.__lt__(abc, de))
    assert_false(StringRef.__lt__(de, abc))
    assert_false(StringRef.__lt__(abc, abc))
    assert_true(StringRef.__lt__(ab, abc))
    assert_true(StringRef.__gt__(abc, ab))
    assert_false(StringRef.__gt__(abc, abcd))

    # Test less than or equal to and greater than or equal to
    assert_true(StringRef.__le__(abc, de))
    assert_true(StringRef.__le__(abc, abc))
    assert_false(StringRef.__le__(de, abc))
    assert_true(StringRef.__ge__(abc, abc))
    assert_false(StringRef.__ge__(ab, abc))
    assert_true(StringRef.__ge__(abcd, abc))

    # Test case sensitivity in comparison (assuming ASCII order)
    assert_true(StringRef.__gt__(abc, ABC))
    assert_false(StringRef.__le__(abc, ABC))

    # Test comparisons involving empty strings
    assert_true(StringRef.__lt__("", abc))
    assert_false(StringRef.__lt__(abc, ""))
    assert_true(StringRef.__le__("", ""))
    assert_true(StringRef.__ge__("", ""))


def test_intable():
    assert_equal(int(StringRef("123")), 123)

    with assert_raises():
        _ = int(StringRef("hi"))


def test_indexing():
    a = StringRef("abc")
    assert_equal(a[False], "a")
    assert_equal(a[int(1)], "b")
    assert_equal(a[0], "a")


def test_find():
    haystack = StringRef("abcdefg")
    haystack_with_special_chars = StringRef("abcdefg@#$")
    haystack_repeated_chars = StringRef("aaaaaaaaaaaaaaaaaaaaaaaa")

    assert_equal(haystack.find("a"), 0)
    assert_equal(haystack.find("ab"), 0)
    assert_equal(haystack.find("abc"), 0)
    assert_equal(haystack.find("bcd"), 1)
    assert_equal(haystack.find("de"), 3)
    assert_equal(haystack.find("fg"), 5)
    assert_equal(haystack.find("g"), 6)
    assert_equal(haystack.find("z"), -1)
    assert_equal(haystack.find("zzz"), -1)

    assert_equal(haystack.find("@#$"), -1)
    assert_equal(haystack_with_special_chars.find("@#$"), 7)

    assert_equal(haystack_repeated_chars.find("aaa"), 0)
    assert_equal(haystack_repeated_chars.find("AAa"), -1)

    assert_equal(haystack.find("hijklmnopqrstuvwxyz"), -1)

    assert_equal(StringRef("").find("abc"), -1)


def test_endswith():
    var empty = StringRef("")
    assert_true(empty.endswith(""))
    assert_false(empty.endswith("a"))
    assert_false(empty.endswith("ab"))

    var a = StringRef("a")
    assert_true(a.endswith(""))
    assert_true(a.endswith("a"))
    assert_false(a.endswith("ab"))

    var ab = StringRef("ab")
    assert_true(ab.endswith(""))
    assert_false(ab.endswith("a"))
    assert_true(ab.endswith("b"))
    assert_true(ab.endswith("b", start=1))
    assert_true(ab.endswith("a", end=1))
    assert_true(ab.endswith("ab"))


fn test_stringref_split() raises:
    # Reject empty delimiters
    with assert_raises(
        contains="empty delimiter not allowed to be passed to split."
    ):
        _ = StringRef("hello").split("")

    # Split in middle
    var d1 = StringRef("n")
    var in1 = StringRef("faang")
    var res1 = in1.split(d1)
    assert_equal(len(res1), 2)
    assert_equal(res1[0], "faa")
    assert_equal(res1[1], "g")

    # Matches should be properly split in multiple case
    var d2 = StringRef(" ")
    var in2 = StringRef("modcon is coming soon")
    var res2 = in2.split(d2)
    assert_equal(len(res2), 4)
    assert_equal(res2[0], "modcon")
    assert_equal(res2[1], "is")
    assert_equal(res2[2], "coming")
    assert_equal(res2[3], "soon")

    # No match from the delimiter
    var d3 = StringRef("x")
    var in3 = StringRef("hello world")
    var res3 = in3.split(d3)
    assert_equal(len(res3), 1)
    assert_equal(res3[0], "hello world")

    # Multiple character delimiter
    var d4 = StringRef("ll")
    var in4 = StringRef("hello")
    var res4 = in4.split(d4)
    assert_equal(len(res4), 2)
    assert_equal(res4[0], "he")
    assert_equal(res4[1], "o")


def test_str_and_ref():
    assert_equal(StringRef("abc").__str__(), "abc")
    assert_equal(StringRef("abc").__repr__(), "StringRef('abc')")
    assert_equal(StringRef("\0").__repr__(), r"StringRef('\x00')")
    assert_equal(StringRef("\x09").__repr__(), r"StringRef('\t')")
    assert_equal(StringRef("\n").__repr__(), r"StringRef('\n')")
    assert_equal(StringRef("\x0d").__repr__(), r"StringRef('\r')")
    assert_equal(StringRef("'").__repr__(), 'StringRef("\'")')

    # Multi-byte characters.__repr__()
    assert_equal(
        StringRef("Örnsköldsvik").__repr__(), "StringRef('Örnsköldsvik')"
    )  # 2-byte
    assert_equal(StringRef("你好!").__repr__(), "StringRef('你好!')")  # 3-byte
    assert_equal(
        StringRef("hello 🔥!").__repr__(), "StringRef('hello 🔥!')"
    )  # 4-byte


def main():
    test_strref_from_start()
    test_stringref_split()
    test_comparison_operators()
    test_intable()
    test_indexing()
    test_find()
    test_endswith()
    test_str_and_ref()
