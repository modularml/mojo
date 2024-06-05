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
# REQUIRES: disabled
# RUN: %mojo --debug-level full %s

from testing import assert_equal, assert_true

from utils import InlineString
from utils.inline_string import _FixedString


def main():
    test_fixed_string()
    test_fixed_string_growth()

    test_small_string_construction()
    test_small_string_iadd()
    test_small_string_add()


def test_fixed_string():
    # Create from StringLiteral
    var s = _FixedString[50]("hello world")

    # Test conversion to String
    assert_equal(str(s), "hello world")

    # Test comparison with StringLiteral
    assert_equal(s, "hello world")

    try:
        var s2 = _FixedString[5]("hello world")
        abort("unreachable: Expected FixedString creation to fail")
    except e:
        assert_equal(
            e, "String literal (len=11) is longer than FixedString capacity (5)"
        )

    # -----------------------------------
    # Test an empty _FixedString
    # -----------------------------------

    var s3 = _FixedString[1]("")

    assert_equal(len(s3), 0)
    assert_equal(str(s3), "")


def test_fixed_string_growth():
    var s1 = _FixedString[10]()
    assert_equal(len(s1), 0)

    s1 += "hello "

    assert_equal(len(s1), 6)
    assert_equal(str(s1), "hello ")

    try:
        s1 += "world"
        assert_true(False, "expected exception to be thrown")
    except e:
        assert_equal(
            e,
            (
                "Insufficient capacity to append len=5 string to len=6"
                " FixedString with capacity=10"
            ),
        )

    # s1 should be unchanged by the failed append
    assert_equal(str(s1), "hello ")


def test_small_string_construction():
    # ==================================
    # Test construction from StringLiteral
    # ==================================

    var s1 = InlineString("hello world")
    var s2 = InlineString("the quick brown fox jumped over the lazy dog")

    assert_true(s1._is_small())
    assert_true(not s2._is_small())

    assert_equal(len(s1), 11)
    assert_equal(len(s2), 44)

    # ==================================
    # Test construction from String
    # ==================================

    var heap_s1 = String("hello")
    var heap_s1_addr = int(heap_s1.unsafe_ptr())

    var s3 = InlineString(heap_s1^)

    # Test that a InlineString constructed from a String uses the same
    # allocation as the original String (even if the String size is small
    # enough to fit inline).
    assert_equal(int(s3.unsafe_ptr()), heap_s1_addr)


def test_small_string_iadd():
    # ==================================
    # Test appending StringLiteral to InlineString
    # ==================================

    var s1 = InlineString("")

    assert_equal(len(s1), 0)
    assert_true(s1._is_small())

    #
    # Tests appending to a small-layout string and staying in the small layout
    #

    s1 += "Hello"

    assert_equal(len(s1), 5)
    assert_true(s1._is_small())

    #
    # Tests appending to a small-layout string and changing to the big layout
    #

    s1 += " world, how's it going?"

    assert_equal(len(s1), 28)
    assert_true(not s1._is_small())

    #
    # Tests appending to a big-layout string and staying in the big layout
    #

    s1 += " The End."

    assert_equal(len(s1), 37)
    assert_true(not s1._is_small())

    assert_equal(str(s1), "Hello world, how's it going? The End.")

    # ==================================
    # Test appending String to InlineString
    # ==================================

    var s2 = InlineString("")
    s2 += String("Hello, World!")

    assert_equal(str(s2), "Hello, World!")
    assert_equal(len(s2), 13)


def test_small_string_add():
    #
    # Test InlineString + StringLiteral
    #

    var s1: InlineString = InlineString("hello") + " world"

    assert_equal(str(s1), "hello world")
    assert_equal(len(s1), "11")

    #
    # Test InlineString + InlineString
    #

    var s2: InlineString = InlineString("hello") + InlineString(" world")

    assert_equal(str(s2), "hello world")
    assert_equal(len(s2), "11")

    #
    # Test InlineString + String
    #

    var s3: InlineString = InlineString("hello") + String(" world")

    assert_equal(str(s3), "hello world")
    assert_equal(len(s3), "11")
