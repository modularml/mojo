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
# RUN: %mojo -debug-level full %s

from testing import assert_equal

from memory.memory import memset_zero
from utils import StringSlice
from utils.write import (
    Writable,
    Writer,
    _write_hex,
    _hex_digits_to_hex_chars,
)
from utils.inline_string import _FixedString


@value
struct Point(Writable, Stringable):
    var x: Int
    var y: Int

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        writer.write("Point(", self.x, ", ", self.y, ")")

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)


def test_writer_of_string():
    #
    # Test write_to(String)
    #
    var s1 = String()
    Point(2, 7).write_to(s1)
    assert_equal(s1, "Point(2, 7)")

    #
    # Test writer.write(String, ..)
    #
    var s2 = String()
    s2.write(Point(3, 8))
    assert_equal(s2, "Point(3, 8)")


def test_string_write_seq():
    var s1 = String.write("Hello, ", "World!")
    assert_equal(s1, "Hello, World!")

    var s2 = String.write("point = ", Point(2, 7))
    assert_equal(s2, "point = Point(2, 7)")

    var s3 = String.write()
    assert_equal(s3, "")


def test_stringable_based_on_format():
    assert_equal(str(Point(10, 11)), "Point(10, 11)")


def test_writer_of_fixed_string():
    var s1 = _FixedString[100]()
    s1.write("Hello, World!")
    assert_equal(str(s1), "Hello, World!")


def test_write_int_padded():
    var s1 = String()

    Int(5).write_padded(s1, width=5)

    assert_equal(s1, "    5")

    Int(123).write_padded(s1, width=5)

    assert_equal(s1, "    5  123")

    # ----------------------------------
    # Test writing int larger than width
    # ----------------------------------

    var s2 = String()

    Int(12345).write_padded(s2, width=3)

    assert_equal(s2, "12345")


def test_hex_digits_to_hex_chars():
    items = List[Byte](0, 0, 0, 0, 0, 0, 0, 0, 0)
    alias S = StringSlice[__origin_of(items)]
    ptr = items.unsafe_ptr()
    _hex_digits_to_hex_chars(ptr, UInt32(ord("🔥")))
    assert_equal("0001f525", S(ptr=ptr, length=8))
    memset_zero(ptr, len(items))
    _hex_digits_to_hex_chars(ptr, UInt16(ord("你")))
    assert_equal("4f60", S(ptr=ptr, length=4))
    memset_zero(ptr, len(items))
    _hex_digits_to_hex_chars(ptr, UInt8(ord("Ö")))
    assert_equal("d6", S(ptr=ptr, length=2))
    _hex_digits_to_hex_chars(ptr, UInt8(0))
    assert_equal("00", S(ptr=ptr, length=2))
    _hex_digits_to_hex_chars(ptr, UInt16(0))
    assert_equal("0000", S(ptr=ptr, length=4))
    _hex_digits_to_hex_chars(ptr, UInt32(0))
    assert_equal("00000000", S(ptr=ptr, length=8))
    _hex_digits_to_hex_chars(ptr, ~UInt8(0))
    assert_equal("ff", S(ptr=ptr, length=2))
    _hex_digits_to_hex_chars(ptr, ~UInt16(0))
    assert_equal("ffff", S(ptr=ptr, length=4))
    _hex_digits_to_hex_chars(ptr, ~UInt32(0))
    assert_equal("ffffffff", S(ptr=ptr, length=8))


def test_write_hex():
    items = List[Byte](0, 0, 0, 0, 0, 0, 0, 0, 0)
    alias S = StringSlice[__origin_of(items)]
    ptr = items.unsafe_ptr()
    _write_hex[8](ptr, ord("🔥"))
    assert_equal(r"\U0001f525", S(ptr=ptr, length=10))
    memset_zero(ptr, len(items))
    _write_hex[4](ptr, ord("你"))
    assert_equal(r"\u4f60", S(ptr=ptr, length=6))
    memset_zero(ptr, len(items))
    _write_hex[2](ptr, ord("Ö"))
    assert_equal(r"\xd6", S(ptr=ptr, length=4))


def main():
    test_writer_of_string()
    test_string_write_seq()
    test_stringable_based_on_format()

    test_writer_of_fixed_string()

    test_write_int_padded()

    test_hex_digits_to_hex_chars()
    test_write_hex()
