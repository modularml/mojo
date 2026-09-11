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

from std.testing import TestSuite, assert_equal


def test_count_basic() raises:
    var nums = [10, 20, 30, 40]
    assert_equal(iter(nums).count(), 4)


def test_count_empty() raises:
    var nums = List[Int]()
    assert_equal(iter(nums).count(), 0)


def test_count_partially_consumed() raises:
    var nums = [1, 2, 3, 4, 5]
    var it = iter(nums)
    _ = next(it)
    _ = next(it)
    assert_equal(it^.count(), 3)


@fieldwise_init
struct _KnownLengthIter(Copyable, Iterator):
    """Yields nothing, but reports a fixed length through `count()`."""

    comptime Element = Int
    var _length: Int

    def __next__(mut self) raises StopIteration -> Int:
        raise StopIteration()

    def count(var self) -> Int:
        return self._length


def test_count_can_be_overridden() raises:
    # The default implementation would walk the iterator and return 0.
    assert_equal(_KnownLengthIter(42).count(), 42)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
