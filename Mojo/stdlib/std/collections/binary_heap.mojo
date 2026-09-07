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
"""Defines a BinaryHeap type."""


# TODO: Copyable can be relaxed
struct BinaryHeap[T: Copyable & Comparable & Deinitable](
    Copyable,
    Defaultable,
    Sized,
):
    """A List-backed binary max-heap.

    push and pop have complexity `O(log(n))` where `n = len(self)`

    Examples:
        ```mojo
        from std.collections.binary_heap import BinaryHeap
        var heap = BinaryHeap[Int]()
        heap.push(5)
        heap.push(10)
        heap.push(3)

        print(heap.peek()) # 10
        print(heap.pop()) # 10
        print(heap.pop()) # 5
        print(heap.pop()) # 3
        ```

    Parameters:
        T: Element type.
    """

    var _data: List[Self.T]
    """Backing storage for heap"""

    def __len__(self) -> Int:
        """Gets the size of the binary heap.

        Returns:
            The number of elements in the heap.
        """
        return len(self._data)

    def __init__(out self):
        """Constructs an empty heap."""
        self._data = List[Self.T]()

    def __init__(out self, *, capacity: Int):
        """Constructs a heap with a given capacity.

        Args:
            capacity: The capacity of the heap.
        """
        self._data = List[Self.T](capacity=capacity)

    def clear(mut self):
        """Clears the elements in the heap."""
        self._data.clear()

    def _heapify_up(mut self, start: Int, var pos: Int):
        """Pulls a new element at `pos` up into the heap starting at `start`.

        Args:
            start: The start of the heap.
            pos: The position of the new item.
        """
        # note: leaves an uninitialized hole in the array
        var data_ptr = self._data.unsafe_ptr()
        var element = (data_ptr.unsafe_offset(pos)).unsafe_take_pointee()
        while pos > start:
            var parent = (pos - 1) // 2

            if element <= self._data[parent]:
                break

            # copy the parent element into the (vacant) hole
            (data_ptr.unsafe_offset(pos)).unsafe_write(
                (data_ptr.unsafe_offset(parent)).unsafe_take_pointee()
            )
            pos = parent

        # fill the hole back in
        (data_ptr.unsafe_offset(pos)).unsafe_write(element^)

    def _heapify_down(mut self, var pos: Int):
        """Restores the heap property.

        First sinks the element at "pos" down to the bottom of the heap
        then shifts it up to restore the heap property.
        """
        var start = pos
        var data_ptr = self._data.unsafe_ptr()
        var element = (data_ptr.unsafe_offset(pos)).unsafe_take_pointee()

        var child = 2 * pos + 1
        while child <= max(len(self) - 2, 0):
            child += (
                1 if data_ptr[unsafe_offset=child]
                <= data_ptr[unsafe_offset=child + 1] else 0
            )
            (data_ptr.unsafe_offset(pos)).unsafe_write(
                (data_ptr.unsafe_offset(child)).unsafe_take_pointee()
            )
            pos = child
            child = 2 * pos + 1

        if child == len(self) - 1:
            (data_ptr.unsafe_offset(pos)).unsafe_write(
                (data_ptr.unsafe_offset(child)).unsafe_take_pointee()
            )
            pos = child

        (data_ptr.unsafe_offset(pos)).unsafe_write(element^)

        self._heapify_up(start, pos)

    def push(mut self, var val: Self.T):
        """Adds a value to the heap.

        Args:
            val: The value to add.
        """
        self._data.append(val^)
        self._heapify_up(0, len(self) - 1)

    def pop(mut self) -> Self.T:
        """Removes the largest item from the heap and returns it.

        Aborts if the heap is empty.

        Returns:
            The largest item in the heap.
        """
        debug_assert[assert_mode="safe"](
            len(self) > 0, "BinaryHeap.pop requires a non-empty heap."
        )
        var item = self._data.pop()
        if len(self) > 0:
            swap(item, self._data[0])
            self._heapify_down(0)
        return item^

    def peek(imm self) -> ref[self._data[0]] Self.T:
        """Gets a reference to the largest element in the heap.

        Returns:
            The largest element in the heap.

        Examples:

        ```mojo
        from std.collections.binary_heap import BinaryHeap
        var heap = BinaryHeap[Int]()
        heap.push(1)
        heap.push(2)
        var two = heap.peek() # returns a reference to "2"
        ```
        """
        debug_assert[assert_mode="safe"](
            len(self) > 0, "BinaryHeap.peek requires a non-empty heap."
        )
        return self._data[0]
