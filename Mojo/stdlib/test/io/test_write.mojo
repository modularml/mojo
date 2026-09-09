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

from std.format import Writable, Writer
from std.format._utils import _hex_digits_to_hex_chars, _write_hex
from std.collections import Span
from std.memory.memory import unsafe_memset_zero
from std.testing import assert_equal, TestSuite


@fieldwise_init
struct Point(Writable):
    var x: Int
    var y: Int

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        writer.write("Point(", self.x, ", ", self.y, ")")


def test_writer_of_string() raises:
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


def test_string_ctor_seq() raises:
    var s1 = String("Hello, ", "World!")
    assert_equal(s1, "Hello, World!")

    var s2 = String("point = ", Point(2, 7))
    assert_equal(s2, "point = Point(2, 7)")


def test_stringable_based_on_format() raises:
    assert_equal(String(Point(10, 11)), "Point(10, 11)")


def test_write_int_padded() raises:
    var s1 = String()

    Int(5).write_padded(s1, width=5)

    assert_equal(s1, "    5")

    Int(-5).write_padded(s1, width=5)

    assert_equal(s1, "    5   -5")

    Int(123).write_padded(s1, width=5)

    assert_equal(s1, "    5   -5  123")

    Int(0).write_padded(s1, width=5)

    assert_equal(s1, "    5   -5  123    0")

    # ----------------------------------
    # Test writing int larger than width
    # ----------------------------------

    var s2 = String()

    Int(12345).write_padded(s2, width=3)

    assert_equal(s2, "12345")

    Int(-1).write_padded(s2, width=1)

    assert_equal(s2, "12345-1")

    Int(-1).write_padded(s2, width=0)

    assert_equal(s2, "12345-1-1")


def test_write_simd_padded() raises:
    # ----------------------------------
    # Test writing scalar Int32
    # ----------------------------------
    var s1 = String()

    Int32(5).write_padded(s1, width=5)

    assert_equal(s1, "    5")

    # Test negative integers - note that negative signs aren't counted
    Int32(-5).write_padded(s1, width=5)

    assert_equal(s1, "    5   -5")

    Int32(123).write_padded(s1, width=5)

    assert_equal(s1, "    5   -5  123")

    # ----------------------------------
    # Test writing scalar Int32 larger than width
    # ----------------------------------

    var s2 = String()

    Int32(12345).write_padded(s2, width=3)

    assert_equal(s2, "12345")

    Int32(-1).write_padded(s2, width=1)

    assert_equal(s2, "12345-1")

    Int32(-1).write_padded(s2, width=0)

    assert_equal(s2, "12345-1-1")

    # ----------------------------------
    # Test writing vector Int32
    # ----------------------------------

    var s3 = String()
    SIMD[.int32, 2](12345).write_padded(s3, width=3)
    assert_equal(s3, "[12345,12345]")

    s3 = String()
    SIMD[.int32, 2](12345).write_padded(s3, width=5)
    assert_equal(s3, "[12345,12345]")

    s3 = String()
    SIMD[.int32, 2](12345).write_padded(s3, width=6)
    assert_equal(s3, "[ 12345, 12345]")

    s3 = String()
    SIMD[.int32, 2](-12345).write_padded(s3, width=7)
    assert_equal(s3, "[ -12345, -12345]")

    s3 = String()
    SIMD[.int8, 4](127, 1, 10, 0).write_padded(s3, width=6)
    assert_equal(s3, "[   127,     1,    10,     0]")


def test_hex_digits_to_hex_chars() raises:
    var items: List[Byte] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    comptime S = StringSlice[origin_of(items)]
    var ptr = items.unsafe_ptr()
    ptr.unsafe_store(_hex_digits_to_hex_chars(UInt32(ord("🔥"))))
    assert_equal("0001f525", S(unsafe_from_utf8=Span(unsafe_ptr=ptr, length=8)))
    unsafe_memset_zero(ptr, len(items))
    ptr.unsafe_store(_hex_digits_to_hex_chars(UInt16(ord("你"))))
    assert_equal("4f60", S(unsafe_from_utf8=Span(unsafe_ptr=ptr, length=4)))
    unsafe_memset_zero(ptr, len(items))
    ptr.unsafe_store(_hex_digits_to_hex_chars(UInt8(ord("Ö"))))
    assert_equal("d6", S(unsafe_from_utf8=Span(unsafe_ptr=ptr, length=2)))
    ptr.unsafe_store(_hex_digits_to_hex_chars(UInt8(0)))
    assert_equal("00", S(unsafe_from_utf8=Span(unsafe_ptr=ptr, length=2)))
    ptr.unsafe_store(_hex_digits_to_hex_chars(UInt16(0)))
    assert_equal("0000", S(unsafe_from_utf8=Span(unsafe_ptr=ptr, length=4)))
    ptr.unsafe_store(_hex_digits_to_hex_chars(UInt32(0)))
    assert_equal("00000000", S(unsafe_from_utf8=Span(unsafe_ptr=ptr, length=8)))
    ptr.unsafe_store(_hex_digits_to_hex_chars(~UInt8(0)))
    assert_equal("ff", S(unsafe_from_utf8=Span(unsafe_ptr=ptr, length=2)))
    ptr.unsafe_store(_hex_digits_to_hex_chars(~UInt16(0)))
    assert_equal("ffff", S(unsafe_from_utf8=Span(unsafe_ptr=ptr, length=4)))
    ptr.unsafe_store(_hex_digits_to_hex_chars(~UInt32(0)))
    assert_equal("ffffffff", S(unsafe_from_utf8=Span(unsafe_ptr=ptr, length=8)))


def test_write_hex() raises:
    var s = String()
    _write_hex[amnt_hex_bytes=8](s, Int(ord("🔥")))
    assert_equal(r"\U0001f525", s)
    s = ""
    _write_hex[amnt_hex_bytes=4](s, Int(ord("你")))
    assert_equal(r"\u4f60", s)
    s = ""
    _write_hex[amnt_hex_bytes=2](s, Int(ord("Ö")))
    assert_equal(r"\xd6", s)


def test_closure_non_capturing() raises:
    def write_closure(mut writer: Some[Writer]):
        writer.write("Hello Mojo!")

    def write_non_capturing[
        func: def(mut writer: Some[Writer]) thin raises -> None
    ]() raises:
        var writer2 = String()
        func(writer2)

        assert_equal(writer2, "Hello Mojo!")

    write_non_capturing[write_closure]()


def _test_closure_capturing(mut writer: Some[Writer & Writable]) raises:
    def write_closure() capturing:
        writer.write("Hello Mojo!")

    def write_capturing[func: def() capturing -> None]():
        func()

    write_capturing[write_closure]()

    # Write result to concrete `String` type to pass to `assert_equal`
    var result = String()
    writer.write_to(result)
    assert_equal(result, "Hello Mojo!")


def test_closure_capturing() raises:
    var writer = String()
    _test_closure_capturing(writer)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
