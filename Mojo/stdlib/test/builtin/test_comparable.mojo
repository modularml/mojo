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

from test_utils.reflection import SimplePoint, NestedStruct, EmptyStruct
from std.testing import assert_true, assert_false, TestSuite


@fieldwise_init
struct OldImpl(Defaultable, Equatable):
    var n: Int

    def __init__(out self):
        self.n = 0

    def __eq__(self, other: Self) -> Bool:
        return True


def param[e: Equatable](lhs: e, rhs: e) -> Bool:
    return lhs.__eq__(rhs)


def test_old_impl() raises:
    """Test that non-positional only implementations are handled correctly."""
    var p1 = OldImpl(n=1)
    var p2 = OldImpl()
    assert_true(p1 == p2)
    assert_true(p1.n == 1)
    assert_true(p2.n == 0)
    assert_true(param[OldImpl](p1, p2))


def test_default_eq_simple() raises:
    """Test the reflection-based default __eq__ with a simple struct."""
    var p1 = SimplePoint(1, 2)
    var p2 = SimplePoint(1, 2)
    var p3 = SimplePoint(1, 3)
    var p4 = SimplePoint(2, 2)

    assert_true(p1 == p2)
    assert_false(p1 != p2)
    assert_false(p1 == p3)
    assert_true(p1 != p3)
    assert_false(p1 == p4)


def test_default_eq_nested() raises:
    """Test the reflection-based default __eq__ with nested structs."""
    var s1 = NestedStruct(SimplePoint(1, 2), "hello")
    var s2 = NestedStruct(SimplePoint(1, 2), "hello")
    var s3 = NestedStruct(SimplePoint(1, 2), "world")
    var s4 = NestedStruct(SimplePoint(3, 4), "hello")

    assert_true(s1 == s2)
    assert_false(s1 == s3)
    assert_false(s1 == s4)


def test_default_eq_empty() raises:
    """Test the reflection-based default __eq__ with an empty struct."""
    var e1 = EmptyStruct()
    var e2 = EmptyStruct()

    assert_true(e1 == e2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
