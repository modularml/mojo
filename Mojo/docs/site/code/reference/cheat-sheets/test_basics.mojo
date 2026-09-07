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
# test_basics.mojo
# Tests for the Mojo "Basics" cheat-sheet card.
#
# Not tested (compile-time errors — a runtime test cannot assert that code
# FAILS to compile; these are verified by hand and belong in a lit
# expected-error test if we want them automated):
#   - no implicit numeric conversion: `var f: Float64 = i` (i: Int) must error
#   - leading-zero decimals rejected: `0123` must error (use `0o` for octal)
from std.collections import Set
from std.testing import assert_equal


def risky() raises:
    # mirrors the Errors panel: a function that may raise is marked `raises`
    raise Error("boom")


def test_ref_writes_through() raises:
    # `ref` binds a name to a value it doesn't own; writing through it updates
    # the original.
    var data: List[Int] = [1, 2, 3]
    ref view = data[0]
    view = 99
    assert_equal(data[0], 99)


def test_literal_adapts() raises:
    # a numeric literal adapts to the annotated type
    var a: Float64 = 0.5
    assert_equal(a, 0.5)


def test_set_literal_and_dedup() raises:
    var primes = {1, 2, 3}  # set display
    assert_equal(len(primes), 3)

    var bag = Set[Int]()
    bag.add(4)
    bag.add(4)  # adding a duplicate is a no-op
    assert_equal(len(bag), 1)


def test_power() raises:
    assert_equal(2**10, 1024)


def test_errors_try_except() raises:
    # Errors panel: raise / try / except e binds the error
    var caught = String("")
    try:
        risky()
    except e:
        caught = String(e)
    assert_equal(caught, "boom")


def test_conversions() raises:
    # Built-in types note: cast explicitly, no implicit numeric conversion
    var i = 42
    assert_equal(Float64(i), 42.0)
    assert_equal(Int(Float64(2.9)), 2)  # truncates toward zero
    assert_equal(String(i), "42")


def test_list_ops() raises:
    # Lists panel
    var xs: List[Int] = [1, 2, 3]
    xs.append(4)
    assert_equal(xs[0], 1)
    assert_equal(len(xs), 4)


def main() raises:
    test_ref_writes_through()
    test_literal_adapts()
    test_set_literal_and_dedup()
    test_power()
    test_errors_try_except()
    test_conversions()
    test_list_ops()
