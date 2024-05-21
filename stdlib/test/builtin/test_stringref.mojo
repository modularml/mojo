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

from testing import assert_equal, assert_true, assert_false, assert_raises

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
    assert_equal(a[Int16(1)], "b")
    assert_equal(a[0], "a")


def main():
    test_strref_from_start()
    test_comparison_operators()
    test_intable()
    test_indexing()
