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

from std.iter import peekable
from std.testing import *


def test_empty_peek() raises:
    var list = List[Int]()
    var iter = peekable(list)
    with assert_raises():
        _ = next(iter)  # raises StopIteration
    assert_false(iter.peek())


def test_peekable_with_peeking() raises:
    var list = [1, 2, 3]
    var iter = peekable(list)
    assert_equal(iter.peek()[][], 1)
    assert_equal(next(iter), 1)

    assert_equal(iter.peek()[][], 2)
    assert_equal(next(iter), 2)

    assert_equal(iter.peek()[][], 3)
    assert_equal(next(iter), 3)

    with assert_raises():
        _ = next(iter)  # raises StopIteration
    assert_false(iter.peek())


def test_peekable_without_peeking() raises:
    var list = [1, 2, 3]
    var iter = peekable(list)
    assert_equal(next(iter), 1)
    assert_equal(next(iter), 2)
    assert_equal(next(iter), 3)
    with assert_raises():
        _ = next(iter)  # raises StopIteration
    assert_false(iter.peek())


def test_peekable_peek_does_not_advance_iterator() raises:
    var list = [1]
    var iter = peekable(list)
    assert_equal(iter.peek()[][], 1)
    assert_equal(iter.peek()[][], 1)
    assert_equal(next(iter), 1)


def test_peekable_bounds() raises:
    var list = [1, 2, 3]
    var iter = peekable(list)

    var lower, upper = iter.bounds()
    assert_equal(lower, 3)
    assert_equal(upper.value(), 3)

    _ = iter.peek()
    lower, upper = iter.bounds()
    assert_equal(lower, 3)
    assert_equal(upper.value(), 3)

    _ = next(iter)
    lower, upper = iter.bounds()
    assert_equal(lower, 2)
    assert_equal(upper.value(), 2)


def test_peekable_owned() raises:
    var l: List[Int] = [1, 2, 3]
    var iter = peekable(l^)
    assert_equal(iter.peek()[][], 1)
    assert_equal(next(iter), 1)
    assert_equal(next(iter), 2)
    assert_equal(next(iter), 3)
    with assert_raises():
        _ = next(iter)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
