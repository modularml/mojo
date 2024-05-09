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
# RUN: %mojo %s

from test_utils import MoveOnly
from testing import assert_equal


def test_swap_Int():
    var a: Int = 42
    var b: Int = 24

    swap(a, b)

    assert_equal(a, 24)
    assert_equal(b, 42)


def test_swap_MoveOnlyInt():
    var a: MoveOnly[Int] = 42
    var b: MoveOnly[Int] = 24

    swap(a, b)

    assert_equal(a.data, 24)
    assert_equal(b.data, 42)


def test_swap_String():
    var a: String = "Hello"
    var b: String = "World"

    swap(a, b)

    assert_equal(a, "World")
    assert_equal(b, "Hello")


def test_swap_Tuple_Int():
    var a = (1, 2, 3, 4)
    var b = (5, 6, 7, 8)

    swap(a, b)

    assert_equal(a[0], 5)
    assert_equal(a[1], 6)
    assert_equal(a[2], 7)
    assert_equal(a[3], 8)

    assert_equal(b[0], 1)
    assert_equal(b[1], 2)
    assert_equal(b[2], 3)
    assert_equal(b[3], 4)


def test_swap_Tuple_Mixed():
    var a = (1, String("Hello"), 3)
    var b = (4, String("World"), 6)

    swap(a, b)

    assert_equal(a[0], 4)
    assert_equal(a[1], "World")
    assert_equal(a[2], 6)

    assert_equal(b[0], 1)
    assert_equal(b[1], "Hello")
    assert_equal(b[2], 3)


def main():
    test_swap_Int()
    test_swap_String()
    test_swap_Tuple_Int()
    test_swap_Tuple_Mixed()
