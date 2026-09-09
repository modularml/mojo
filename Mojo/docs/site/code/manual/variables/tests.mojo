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
# Tests for variables.mdx code examples.
#
# Not tested (the doc shows these as compile errors, so a passing build
# can't contain them):
#   - `count = "Nine?"` after `var count = 8` (type mismatch)
#   - `var second = first` where `first` is `List[Int]` (not
#     `ImplicitlyCopyable`)
#   - `ref item_ref = items[2]` rebinding an existing reference binding
#   - reading `source` after `var moved = source^` (uninitialized)
from std.testing import assert_equal

# --- Declaration forms: value, annotation, or both ---


def test_declaration_forms() raises:
    var greeting: String = "Hello World"
    var a = 5
    var b: Float64 = 3.14
    assert_equal(greeting, "Hello World")
    assert_equal(a, 5)
    assert_equal(b, 3.14)


# --- A name can be declared uninitialized, then assigned ---


def test_late_initialization() raises:
    var c: String
    c = "assigned later"
    assert_equal(c, "assigned later")


# --- Lexical scoping: an inner `var` shadows rather than reassigns ---


def test_variable_scopes() raises:
    var num = 1
    var dig = 1
    var seen_inside = 0
    if num == 1:
        var num = 2  # A new inner-scope "num", not the outer one
        seen_inside = num
        dig = 2  # Updates the outer-scope "dig"
    assert_equal(seen_inside, 2)
    assert_equal(num, 1)  # Outer "num" is untouched by the shadow
    assert_equal(dig, 2)


# --- Assigning a literal establishes ownership ---


def test_literal_establishes_ownership() raises:
    var owning_variable = "Owned value"
    assert_equal(owning_variable, "Owned value")


# --- Copy vs. transfer of an implicitly copyable value ---


def test_copy_and_transfer() raises:
    var source = String("Hello")
    var copied = source  # A copy
    var moved = source^  # A transfer
    assert_equal(copied, "Hello")
    assert_equal(moved, "Hello")


# --- `copy()` leaves the original intact ---


def test_explicit_copy() raises:
    var first: List[Int] = [1, 2, 3]
    var second = first.copy()
    second.append(4)
    assert_equal(len(first), 3)  # first is unchanged
    assert_equal(len(second), 4)


# --- Implicitly copyable types copy without an explicit signal ---


def test_implicit_copy() raises:
    var one_value = 15
    var another_value = one_value  # implicit copy
    another_value += 1
    assert_equal(one_value, 15)
    assert_equal(another_value, 16)


# --- The transfer sigil moves ownership ---


def test_transfer_sigil() raises:
    var first: List[Int] = [1, 2, 3]
    var second = first^
    assert_equal(len(second), 3)
    assert_equal(second[0], 1)


# --- Subscripting a collection returns a reference, not a copy ---


def test_reference_avoids_copy() raises:
    var animals: List[String] = ["Cats", "Dogs", "Zebras"]
    assert_equal(animals[2], "Zebras")


# --- Assigning that reference to a variable copies the value ---


def test_assignment_from_reference_copies() raises:
    var items: List[Int] = [99, 77, 33, 12]
    var item = items[1]
    item += 1
    assert_equal(item, 78)
    assert_equal(items[1], 77)  # The collection is untouched


# --- A `ref` binding writes through to the referenced element ---


def test_reference_binding() raises:
    var items: List[Int] = [99, 77, 33, 12]
    ref item_ref = items[1]
    item_ref += 1
    assert_equal(items[1], 78)


def main() raises:
    test_declaration_forms()
    test_late_initialization()
    test_variable_scopes()
    test_literal_establishes_ownership()
    test_copy_and_transfer()
    test_explicit_copy()
    test_implicit_copy()
    test_transfer_sigil()
    test_reference_avoids_copy()
    test_assignment_from_reference_copies()
    test_reference_binding()
