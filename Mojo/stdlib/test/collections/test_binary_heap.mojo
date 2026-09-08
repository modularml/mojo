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

from std.testing.prop import PropTest
from std.testing import (
    assert_equal,
    assert_true,
    TestSuite,
)
from std.testing.prop.strategy import SIMD, List
from std.collections import BinaryHeap


def _is_heap(self: BinaryHeap[_], start: Int, end: Int) -> Bool:
    for idx in range(start * 2 + 1, end):
        if self._data[idx] > self._data[(idx - 1) // 2]:
            return False
    return True


def test_binary_heap() raises:
    var bheap = BinaryHeap[Int]()
    bheap.push(6)
    bheap.push(7)

    assert_equal(bheap.pop(), 7)
    assert_equal(bheap.pop(), 6)


def test_binary_heap_heap_property() raises:
    def properties(forward: List[Int]) raises:
        var heap = BinaryHeap[Int]()
        for i in forward:
            heap.push(i)
            assert_true(_is_heap(heap, 0, len(heap)))
        assert_true(_is_heap(heap, 0, len(heap)))
        var list = List[Int]()
        while len(heap) > 0:
            list.append(heap.pop())
            assert_true(_is_heap(heap, 0, len(heap)))

    PropTest().test[properties](List[Int].strategy(Int.strategy()))


def test_binary_heap_peek() raises:
    var heap = BinaryHeap[Int]()
    heap.push(3)
    heap.push(1)
    heap.push(5)
    heap.push(2)

    # peek returns the max without removing it
    assert_equal(heap.peek(), 5)
    assert_equal(len(heap), 4)
    assert_equal(heap.peek(), 5)

    # peek tracks the max after pops
    _ = heap.pop()
    assert_equal(heap.peek(), 3)
    assert_equal(len(heap), 3)

    # peek on a single-element heap
    var single = BinaryHeap[Int]()
    single.push(42)
    assert_equal(single.peek(), 42)
    assert_equal(len(single), 1)


def test_binary_heap_clear() raises:
    var heap = BinaryHeap[Int]()
    heap.push(3)
    heap.push(1)
    heap.push(5)

    heap.clear()
    assert_equal(len(heap), 0)

    # heap is still usable after clearing
    heap.push(10)
    heap.push(4)
    heap.push(7)
    assert_equal(len(heap), 3)
    assert_true(_is_heap(heap, 0, len(heap)))
    assert_equal(heap.pop(), 10)
    assert_equal(heap.pop(), 7)
    assert_equal(heap.pop(), 4)

    # clearing an already-empty heap is a no-op
    heap.clear()
    assert_equal(len(heap), 0)
    heap.clear()
    assert_equal(len(heap), 0)


def test_binary_heap_sorted() raises:
    var heap = BinaryHeap[Int]()
    heap.push(1)
    assert_true(_is_heap(heap, 0, len(heap)))
    heap.push(2)
    assert_true(_is_heap(heap, 0, len(heap)))
    heap.push(3)
    assert_true(_is_heap(heap, 0, len(heap)))
    heap.push(4)
    assert_true(_is_heap(heap, 0, len(heap)))
    heap.push(5)
    assert_true(_is_heap(heap, 0, len(heap)))

    assert_equal(heap.pop(), 5)
    assert_equal(heap.pop(), 4)
    assert_equal(heap.pop(), 3)
    assert_equal(heap.pop(), 2)
    assert_equal(heap.pop(), 1)
    assert_equal(len(heap), 0)


def test_binary_heap_reverse_sorted() raises:
    var heap = BinaryHeap[Int]()
    heap.push(5)
    assert_true(_is_heap(heap, 0, len(heap)))
    heap.push(4)
    assert_true(_is_heap(heap, 0, len(heap)))
    heap.push(3)
    assert_true(_is_heap(heap, 0, len(heap)))
    heap.push(2)
    assert_true(_is_heap(heap, 0, len(heap)))
    heap.push(1)
    assert_true(_is_heap(heap, 0, len(heap)))

    assert_equal(heap.pop(), 5)
    assert_equal(heap.pop(), 4)
    assert_equal(heap.pop(), 3)
    assert_equal(heap.pop(), 2)
    assert_equal(heap.pop(), 1)
    assert_equal(len(heap), 0)


def test_binary_heap_copy() raises:
    var heap = BinaryHeap[Int]()
    heap.push(1)
    heap.push(2)
    heap.push(3)
    heap.push(4)
    heap.push(5)

    var heap2 = BinaryHeap[Int](copy=heap)
    assert_equal(heap.pop(), 5)
    assert_equal(heap.pop(), 4)
    assert_equal(heap.pop(), 3)
    assert_equal(heap.pop(), 2)
    assert_equal(heap.pop(), 1)
    assert_equal(len(heap), 0)
    assert_equal(len(heap2), 5)
    assert_equal(heap2.pop(), 5)
    assert_equal(heap2.pop(), 4)
    assert_equal(heap2.pop(), 3)
    assert_equal(heap2.pop(), 2)
    assert_equal(heap2.pop(), 1)
    assert_equal(len(heap2), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
