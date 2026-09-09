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
# Tests for keywords.mdx code examples.
# Skip: CountingTool.get (intentional immutable-self error),
#        make_point with both `out` and `->` (intentional signature
#        error), bare identifier name lists (not standalone code).
from std.testing import assert_equal


# --- Regular identifiers ---


def test_regular_identifiers() raises:
    var foo = 1
    var _private = 2
    var MyStruct = 3
    var basic_value = 4
    assert_equal(foo, 1)
    assert_equal(_private, 2)
    assert_equal(MyStruct, 3)
    assert_equal(basic_value, 4)


# --- Escaped identifiers ---


def test_escaped_identifiers() raises:
    var `struct` = 10  # Use a keyword as a name
    var `日本語の変数` = 20  # Non-ASCII identifier
    var `my value` = 30  # Spaces in a name
    assert_equal(`struct`, 10)
    assert_equal(`日本語の変数`, 20)
    assert_equal(`my value`, 30)


# --- Identifiers are case-sensitive ---


def test_case_sensitivity() raises:
    var MyStruct = 1
    var mystruct = 2
    assert_equal(MyStruct, 1)
    assert_equal(mystruct, 2)


# --- Struct with `out self`, `mut self`, and `var` field ---
#     `get()` can't be tested as it intentionally errors


struct CountingTool:
    var value: Int

    def __init__(out self):  # out: self is the return value
        self.value = 0

    def increment(mut self):  # mut: modifies self in place
        self.value += 1


def test_counting_tool() raises:
    var t = CountingTool()
    assert_equal(t.value, 0)
    t.increment()
    assert_equal(t.value, 1)
    t.increment()
    assert_equal(t.value, 2)


# --- `var` creates a scoped mutable variable;
#     `ref` creates a scoped reference binding ---


def test_var_and_ref() raises:
    var count = 0
    count += 1
    assert_equal(count, 1)

    var data: List[Int] = [1, 2, 3]
    ref view = data
    assert_equal(view[0], 1)
    assert_equal(view[2], 3)


# --- `out` convention: out arg replaces explicit return type ---


struct Point(Copyable, Movable):
    var x: Int
    var y: Int

    def __init__(out self, x: Int, y: Int):
        self.x = x
        self.y = y


def make_point(out result: Point):
    result = Point(0, 0)


def test_make_point() raises:
    var p = make_point()
    assert_equal(p.x, 0)
    assert_equal(p.y, 0)


# --- `raises` clause in signatures ---


def validate(value: Int) raises:
    if value < 0:
        raise Error("must be non-negative")


def test_validate_ok() raises:
    validate(5)
    validate(0)


def test_validate_raises() raises:
    var raised = False
    try:
        validate(-1)
    except e:
        raised = True
    assert_equal(raised, True)


def main() raises:
    test_regular_identifiers()
    test_escaped_identifiers()
    test_case_sensitivity()
    test_counting_tool()
    test_var_and_ref()
    test_make_point()
    test_validate_ok()
    test_validate_raises()
