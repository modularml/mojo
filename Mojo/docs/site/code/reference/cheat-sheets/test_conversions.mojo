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
# test_conversions.mojo
# Tests for the Mojo "conversions" cheat-sheet card.
#
# Exercises the make / convert / access casts shown on the card so a claim that
# drifts stops compiling or fails an assert.
#
# Not tested (need a GPU/Python, or no portable runtime API to assert):
#   - parse FAILURES abort/raise paths beyond one sample (Int("3.5") raises)
#   - .value()/.take() on an empty Optional abort (can't assert an abort)
#   - @implicit conversions (a compile-time behavior, not a runtime value)
#   - Python conversions (need the Python runtime); Float8 (needs a GPU)
from std.testing import assert_equal, assert_true, assert_false


def test_numbers() raises:
    # make an Int: from Intable, a scalar, a String (raising)
    assert_equal(Int(True), 1)
    assert_equal(Int(Int32(5)), 5)
    assert_equal(Int("42"), 42)
    # Int is a SIMD scalar, so .cast works
    assert_equal(Int(5).cast[.int8](), Int8(5))
    # Int <-> UInt share bits: the reinterpretation round-trips
    assert_equal(Int(UInt(Int(-1))), -1)
    # make a Float64; Float -> Int truncates toward zero
    assert_equal(Float64(3), 3.0)
    assert_equal(Float64("1.5"), 1.5)
    assert_equal(Int(Float64(3.9)), 3)
    # Bool: from Boolable, from None
    assert_true(Bool(1))
    assert_true(Bool("x"))
    assert_false(Bool(""))
    assert_false(Bool(None))
    # number -> Bool / Float / String
    assert_equal(Int(True), 1)
    assert_equal(Int(False), 0)
    assert_equal(Float64(True), 1.0)
    assert_equal(String(True), "True")


def test_simd() raises:
    var v = SIMD[.int32, 4](7)  # splat one value to all lanes
    assert_equal(v.reduce_add(), 28)
    var v2 = SIMD[.int32, 4](1, 2, 3, 4)  # per-lane (N matches arg count)
    assert_equal(v2[2], 3)
    assert_equal(v2.cast[.float64]()[0], 1.0)  # new dtype, same lane count
    var s: Int32 = 9
    assert_equal(SIMD[.int32, 4](s).reduce_add(), 36)  # scalar -> splat


def test_string() raises:
    # make from any Writable
    assert_equal(String(42), "42")
    assert_equal(String(True), "True")
    # convert: parse
    assert_equal(Int("7"), 7)
    assert_equal(Float64("2.5"), 2.5)
    assert_true(Bool("x"))
    assert_false(Bool(String("")))
    # access: byte/codepoint index + slice; grapheme single-index only
    var s = String("abcde")
    assert_equal(s[byte=0], "a")
    assert_equal(s[byte=1:3], "bc")
    assert_equal(s[codepoint=1], "b")
    assert_equal(s[codepoint=1:3], "bc")
    assert_equal(s[grapheme=2], "c")
    # as_bytes is an iterator over the bytes
    assert_equal(len(String("abc").as_bytes()), 3)


def test_pointers() raises:
    # reach into a container's raw buffer, then vectorize it (the escape hatch)
    var r = List(range(4))
    var vec = r.unsafe_ptr().unsafe_load[width=4]()  # 4 elements -> one SIMD
    assert_equal(vec.reduce_add(), 6)  # 0+1+2+3
    assert_equal(r.unsafe_ptr()[unsafe_offset=0], 0)  # deref one element


def test_list() raises:
    var lst: List[Int] = [1, 2, 3]
    assert_equal(len(lst), 3)
    assert_equal(lst[0], 1)  # one element, by reference
    var view: Span[Int, _] = lst[0:2]  # a Span view, no copy
    assert_equal(len(view), 2)
    var filled = List[Int](length=4, fill=0)
    assert_equal(len(filled), 4)
    assert_equal(filled[2], 0)
    var r = List(range(5))  # range -> List
    assert_equal(len(r), 5)
    assert_equal(r[4], 4)


def test_dict() raises:
    var d = Dict[Int, String]()
    d[1] = "one"  # fill with d[k] = v
    d[2] = "two"
    assert_equal(d.setdefault(3, "three"), "three")  # insert-if-absent
    assert_true(d.get(1))  # Optional[V]
    assert_false(d.get(99))
    assert_true(1 in d)
    assert_equal(d.pop(2), "two")  # value, removes it
    assert_equal(len(List(d.values())), 2)  # materialize the iterator


def test_optional() raises:
    var o = Optional(5)  # from a value
    assert_true(Bool(o))
    assert_equal(o.value(), 5)  # ref
    assert_equal(o[], 5)  # ref via []
    var empty = Optional[Int](None)  # empty
    assert_false(Bool(empty))
    var empty2 = Optional[Int]()  # empty, explicit
    assert_false(Bool(empty2))
    assert_equal(empty.or_else(99), 99)  # value, or default
    var seven = Optional(7)
    assert_equal(seven.take(), 7)  # move value out


def main() raises:
    test_numbers()
    test_simd()
    test_string()
    test_pointers()
    test_list()
    test_dict()
    test_optional()
