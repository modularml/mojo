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
# tests.mojo
# Tests for types.mdx code examples.
#
# Not tested at runtime (intentional):
#   - `fatal()` is defined to type-check the `Never` example, but it's
#     never called: invoking `abort()` would terminate the test process.
#   - The Numeric types and Memory types sections delegate to other
#     pages and have no standalone runnable code.
from std.collections import Set
from std.os import abort
from std.testing import assert_equal, assert_true
from std.utils import Variant


# --- Built-in types and the prelude: set display vs. the Set type ---


def test_set_display_and_type() raises:
    var primes = {2, 3, 5, 7}  # set display: no import needed
    assert_equal(len(primes), 4)
    var empty = Set[Int]()  # naming Set needs the import
    assert_equal(len(empty), 0)


# --- String types ---


def test_string_lengths() raises:
    var wave = String("👋🏽")  # waving hand + skin-tone modifier
    assert_equal(wave.byte_length(), 8)
    assert_equal(wave.count_codepoints(), 2)
    assert_equal(wave.count_graphemes(), 1)


# --- Collection types: Optional ---


def test_optional() raises:
    var maybe: Optional[Int] = 5
    var present = False
    if maybe:
        present = True
    assert_true(present)
    assert_equal(maybe.value(), 5)

    var empty: Optional[Int] = None
    assert_equal(empty.or_else(0), 0)


# --- Collection types: Variant ---


def test_variant() raises:
    var v = Variant[Int, String](5)
    assert_equal(v[Int], 5)
    v.set[String]("text")
    assert_equal(v[String], "text")


# --- Other built-in types: Bool ---


def test_bool() raises:
    var flag = True
    var captured = False
    if flag:
        captured = True
    assert_true(captured)


# --- Other built-in types: Error ---


def parse(s: String) raises -> Int:
    raise Error("invalid input")


def test_error_raises() raises:
    var caught = String("")
    try:
        _ = parse("bad")
    except e:
        caught = String(e)
    assert_equal(caught, "invalid input")


# --- Other built-in types: Never (type-checked, not invoked) ---


def fatal() -> Never:
    abort()  # never returns


# --- Other built-in types: None ---


def implicit_greet(name: String):
    print("hello", name)
    # returns None implicitly


def explicit_greet(name: String) -> None:
    print("hello", name)
    # returns None explicitly


def test_none_returns() raises:
    implicit_greet("implicit")
    explicit_greet("explicit")


# --- Other built-in types: Slice ---


def test_slice_views() raises:
    var items: List = [0, 1, 2, 3, 4, 5]
    var middle = items[1:4]  # Span view, [1, 2, 3]
    assert_equal(len(middle), 3)
    assert_equal(middle[0], 1)
    assert_equal(middle[2], 3)

    var strided = items[::2]  # new List, [0, 2, 4]
    assert_equal(len(strided), 3)
    assert_equal(strided[0], 0)
    assert_equal(strided[2], 4)


def main() raises:
    test_set_display_and_type()
    test_string_lengths()
    test_optional()
    test_variant()
    test_bool()
    test_error_raises()
    test_none_returns()
    test_slice_views()
