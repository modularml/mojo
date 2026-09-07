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
# test_types_and_literals.mojo
# Tests for the Mojo "Types & Literals" cheat-sheet card.
#
# Not tested (compile-error claims or no portable runtime API to assert):
#   - "no implicit numeric conversion" is a compile error, not runtime
#   - \u and \U reject surrogate code points (U+D800 to U+DFFF) at compile time
#   - nan == nan is False (nan construction left out to keep this portable)
#   - Float8 needs a GPU; Int128 / Int256 are software-emulated
from std.testing import assert_equal
from std.sys.info import bit_width_of


def test_simd_is_the_foundation() raises:
    var v = SIMD[.float32, 4](1.0, 2.0, 3.0, 4.0)
    var d = v * 2.0
    assert_equal(d[1], 4.0)  # every lane scaled
    v[0] = 5.0  # write one lane
    assert_equal(v.reduce_add(), 14.0)  # 5 + 2 + 3 + 4


def test_conversions_explicit() raises:
    var i = 42
    assert_equal(Float64(i), 42.0)
    var g = Float64(i).cast[.int32]()  # between SIMD types
    assert_equal(g, Int32(42))


def test_integer_overflow_wraps() raises:
    # signed overflow wraps (two's complement)
    assert_equal(Int8(127) + 1, Int8(-128))


def test_float_to_int_truncates() raises:
    # truncates toward zero
    assert_equal(Int(Float64(3.9)), 3)


def test_bounds() raises:
    assert_equal(UInt8.MAX, UInt8(255))
    assert_equal(Int8.MIN, Int8(-128))


def test_bit_width() raises:
    # bit_width_of[Int]() from std.sys.info; 64 on this platform
    assert_equal(bit_width_of[Int](), 64)


def test_number_literals() raises:
    assert_equal(0xFF, 255)
    assert_equal(0o52, 42)
    assert_equal(0b1010, 10)
    assert_equal(1_000_000, 1000000)


def test_triple_quote_keeps_layout() raises:
    # newlines AND indentation are part of the string
    var s = """line one
    line two"""
    assert_equal(s, "line one\n    line two")


def test_adjacent_literals_join() raises:
    var same_line = "abcd"
    assert_equal(same_line, "abcd")
    var across_lines = "Content of line 1. Content of line 2."
    assert_equal(across_lines, "Content of line 1. Content of line 2.")


def test_unicode_escapes() raises:
    assert_equal("\u20AC", "€")  # lowercase \u, 4 digits (EURO)
    assert_equal("\U0001F44B", "👋")  # uppercase \U, 8 digits (above U+FFFF)


def test_tstring_interpolates() raises:
    var who = "Mojo"
    assert_equal(String(t"x = {who}"), "x = Mojo")
    assert_equal(String(t"sum = {1 + 2}"), "sum = 3")


def test_raw_tstring() raises:
    var who = "Mojo"
    # raw: backslash stays literal, interpolation still happens
    assert_equal(String(rt"raw\path {who}"), "raw\\path Mojo")


def test_collection_literals() raises:
    # A bracket literal infers a fixed-size Array; annotating gets a List, and
    # only the List grows.
    var fixed = [1, 2, 3]
    assert_equal(len(fixed), 3)
    var nums: List[Int] = [1, 2, 3]
    nums.append(4)
    assert_equal(len(nums), 4)
    var d = {"id": 1, "qty": 9}
    assert_equal(d["qty"], 9)
    var pair = (1, "a", 2.0)
    assert_equal(pair[0], 1)


def main() raises:
    test_simd_is_the_foundation()
    test_conversions_explicit()
    test_integer_overflow_wraps()
    test_float_to_int_truncates()
    test_bounds()
    test_bit_width()
    test_number_literals()
    test_triple_quote_keeps_layout()
    test_adjacent_literals_join()
    test_unicode_escapes()
    test_tstring_interpolates()
    test_raw_tstring()
    test_collection_literals()
