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
"""Defines the heapq module.

Implementation currently tightly follows the Python implementation.
"""


fn heappush[T: ComparableCollectionElement](inout heap: List[T], owned item: T):
    """Push an element onto a heapified list.

    Parameters:
        T: A comparable collection element type.

    Args:
        heap: The heap to push to.
        item: The new item to be placed in the heap.
    """
    heap.append(item^)
    _siftdown(heap, 0, len(heap) - 1)


fn heappop[T: ComparableCollectionElement](inout heap: List[T]) -> T:
    """Pop an element from a heapified list.

    Parameters:
        T: A comparable collection element type.

    Args:
        heap: The heap to push to.

    Returns:
        The popped element.
    """
    var lastelt = heap.pop()
    if heap:
        var returnitem = heap[0]
        heap[0] = lastelt
        _siftup(heap, 0)
        return returnitem
    return lastelt


fn heapify[T: ComparableCollectionElement](inout x: List[T]):
    """Convert a list of elements into a binary heap.

    Parameters:
        T: A comparable collection element type.

    Args:
        x: The list to heapify.
    """
    for i in reversed(range(len(x) // 2)):
        _siftup(x, i)


fn _siftdown[
    T: ComparableCollectionElement
](inout heap: List[T], startpos: Int, owned pos: Int):
    var newitem = heap[pos]
    while pos > startpos:
        var parentpos = (pos - 1) >> 1
        var parent = heap[parentpos]
        if newitem < parent:
            heap[pos] = parent
            pos = parentpos
            continue
        break
    heap[pos] = newitem


fn _siftup[T: ComparableCollectionElement](inout heap: List[T], owned pos: Int):
    var endpos = len(heap)
    var startpos = pos
    var newitem = heap[pos]
    var childpos = 2 * pos + 1
    while childpos < endpos:
        var rightpos = childpos + 1
        if rightpos < endpos and not heap[childpos] < heap[rightpos]:
            childpos = rightpos
        heap[pos] = heap[childpos]
        pos = childpos
        childpos = 2 * pos + 1
    heap[pos] = newitem
    _siftdown(heap, startpos, pos)
