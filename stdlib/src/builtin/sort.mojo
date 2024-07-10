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
"""Implements the built-in `sort` function.

These are Mojo built-ins, so you don't need to import them.
"""

from collections import List
from sys import bitwidthof

from bit import count_leading_zeros
from memory import UnsafePointer

# ===----------------------------------------------------------------------===#
# sort
# ===----------------------------------------------------------------------===#


@value
struct _SortWrapper[type: CollectionElement](CollectionElement):
    var data: type


@always_inline
fn _insertion_sort[
    type: CollectionElement,
    cmp_fn: fn (_SortWrapper[type], _SortWrapper[type]) capturing -> Bool,
](array: UnsafePointer[type], size: Int):
    """Sort the array[start:end] slice"""

    for i in range(1, size):
        var value = array[i]
        var j = i

        # Find the placement of the value in the array, shifting as we try to
        # find the position. Throughout, we assume array[start:i] has already
        # been sorted.
        while j > 0 and not cmp_fn(array[j - 1], value):
            array[j] = array[j - 1]
            j -= 1

        array[j] = value


# put everything thats "<" to the left of pivot
@always_inline
fn _quicksort_partition_right[
    type: CollectionElement,
    cmp_fn: fn (_SortWrapper[type], _SortWrapper[type]) capturing -> Bool,
](array: UnsafePointer[type], size: Int) -> Int:
    var left = 1
    var right = size - 1
    var pivot_value = array[0]

    while True:
        # no need for left < right since quick sort pick median of 3 as pivot
        while cmp_fn(array[left], pivot_value):
            left += 1
        while left < right and not cmp_fn(array[right], pivot_value):
            right -= 1
        if left >= right:
            var pivot_pos = left - 1
            swap(array[pivot_pos], array[0])
            return pivot_pos
        swap(array[left], array[right])
        left += 1
        right -= 1


# put everything thats "<=" to the left of pivot
@always_inline
fn _quicksort_partition_left[
    type: CollectionElement,
    cmp_fn: fn (_SortWrapper[type], _SortWrapper[type]) capturing -> Bool,
](array: UnsafePointer[type], size: Int) -> Int:
    var left = 1
    var right = size - 1
    var pivot_value = array[0]

    while True:
        while left < right and not cmp_fn(pivot_value, array[left]):
            left += 1
        while cmp_fn(pivot_value, array[right]):
            right -= 1
        if left >= right:
            var pivot_pos = left - 1
            swap(array[pivot_pos], array[0])
            return pivot_pos
        swap(array[left], array[right])
        left += 1
        right -= 1


fn _heap_sort_fix_down[
    type: CollectionElement,
    cmp_fn: fn (_SortWrapper[type], _SortWrapper[type]) capturing -> Bool,
](array: UnsafePointer[type], size: Int, idx: Int):
    var i = idx
    var j = i * 2 + 1
    while j < size:  # has left child
        # if right child exist and has higher value, swap with right
        if i * 2 + 2 < size and cmp_fn(array[j], array[i * 2 + 2]):
            j = i * 2 + 2
        if not cmp_fn(array[i], array[j]):
            return
        swap(array[j], array[i])
        i = j
        j = i * 2 + 1


@always_inline
fn _heap_sort[
    type: CollectionElement,
    cmp_fn: fn (_SortWrapper[type], _SortWrapper[type]) capturing -> Bool,
](array: UnsafePointer[type], owned size: Int):
    # heapify
    for i in range(size // 2 - 1, -1, -1):
        _heap_sort_fix_down[type, cmp_fn](array, size, i)
    # sort
    while size > 1:
        size -= 1
        swap(array[0], array[size])
        _heap_sort_fix_down[type, cmp_fn](array, size, 0)


@always_inline
fn _estimate_initial_height(size: Int) -> Int:
    # Compute the log2 of the size rounded upward.
    var log2 = int(
        (bitwidthof[DType.index]() - 1) ^ count_leading_zeros(size | 1)
    )
    return max(2, log2)


@always_inline
fn _delegate_small_sort[
    type: CollectionElement,
    cmp_fn: fn (_SortWrapper[type], _SortWrapper[type]) capturing -> Bool,
](array: UnsafePointer[type], size: Int):
    if size == 2:
        _small_sort[2, type, cmp_fn](array)

        return
    if size == 3:
        _small_sort[3, type, cmp_fn](array)
        return

    if size == 4:
        _small_sort[4, type, cmp_fn](array)
        return

    if size == 5:
        _small_sort[5, type, cmp_fn](array)
        return


@always_inline
fn _quicksort[
    type: CollectionElement,
    cmp_fn: fn (_SortWrapper[type], _SortWrapper[type]) capturing -> Bool,
](array: UnsafePointer[type], size: Int):
    if size == 0:
        return
    var stack = List[Int](capacity=_estimate_initial_height(size))
    stack.append(0)
    stack.append(size)
    while len(stack) > 0:
        var end = stack.pop()
        var start = stack.pop()

        var len = end - start

        if len <= 5:
            _delegate_small_sort[type, cmp_fn](array + start, len)
            continue

        if len < 32:
            _insertion_sort[type, cmp_fn](array + start, len)
            continue

        # pick median of 3 as pivot
        _sort3[type, cmp_fn](array, (start + end) >> 1, start, end - 1)

        # if array[start - 1] == pivot_value, then everything in between will
        # be the same, so no need to recurse that interval
        # already have array[start - 1] <= array[start]
        if start > 0 and not cmp_fn(array[start - 1], array[start]):
            var pivot = start + _quicksort_partition_left[type, cmp_fn](
                array + start, len
            )
            if end > pivot + 2:
                stack.append(pivot + 1)
                stack.append(end)
            continue

        var pivot = start + _quicksort_partition_right[type, cmp_fn](
            array + start, len
        )

        if end > pivot + 2:
            stack.append(pivot + 1)
            stack.append(end)

        if pivot > start + 1:
            stack.append(start)
            stack.append(pivot)


# ===----------------------------------------------------------------------===#
# partition
# ===----------------------------------------------------------------------===#


@always_inline
fn _partition[
    type: CollectionElement,
    cmp_fn: fn (_SortWrapper[type], _SortWrapper[type]) capturing -> Bool,
](array: UnsafePointer[type], size: Int) -> Int:
    if size == 0:
        return size

    var pivot = size // 2

    var pivot_value = array[pivot]

    var left = 0
    var right = size - 2

    swap(array[pivot], array[size - 1])

    while left < right:
        if cmp_fn(array[left], pivot_value):
            left += 1
        elif not cmp_fn(array[right], pivot_value):
            right -= 1
        else:
            swap(array[left], array[right])

    if cmp_fn(array[right], pivot_value):
        right += 1
    swap(array[size - 1], array[right])
    return right


fn _partition[
    type: CollectionElement,
    cmp_fn: fn (_SortWrapper[type], _SortWrapper[type]) capturing -> Bool,
](array: UnsafePointer[type], k: Int, size: Int):
    var stack = List[Int](capacity=_estimate_initial_height(size))
    stack.append(0)
    stack.append(size)
    while len(stack) > 0:
        var end = stack.pop()
        var start = stack.pop()
        var pivot = start + _partition[type, cmp_fn](array + start, end - start)
        if pivot == k:
            break
        elif k < pivot:
            stack.append(start)
            stack.append(pivot)
        else:
            stack.append(pivot + 1)
            stack.append(end)


fn partition[
    type: CollectionElement,
    cmp_fn: fn (type, type) capturing -> Bool,
](array: UnsafePointer[type], k: Int, size: Int):
    """Partition the input buffer inplace such that first k elements are the
    largest (or smallest if cmp_fn is < operator) elements.
    The ordering of the first k elements is undefined.

    Parameters:
        type: Type of the underlying data.
        cmp_fn: Comparison functor of (type, type) capturing -> Bool type.

    Args:
        array: Input buffer.
        k: Index of the partition element.
        size: The length of the buffer.
    """

    @parameter
    fn _cmp_fn(lhs: _SortWrapper[type], rhs: _SortWrapper[type]) -> Bool:
        return cmp_fn(lhs.data, rhs.data)

    _partition[type, _cmp_fn](array, k, size)


fn partition[
    cmp_fn: fn (Int, Int) capturing -> Bool,
](array: UnsafePointer[Int], k: Int, size: Int):
    """Partition the input buffer inplace such that first k elements are the
    largest (or smallest if cmp_fn is < operator) elements.
    The ordering of the first k elements is undefined.

    Parameters:
        cmp_fn: Comparison functor of (type, type) capturing -> Bool type.

    Args:
        array: Input buffer.
        k: Index of the partition element.
        size: The length of the buffer.
    """

    @parameter
    fn _cmp_fn(lhs: _SortWrapper[Int], rhs: _SortWrapper[Int]) -> Bool:
        return cmp_fn(lhs.data, rhs.data)

    _partition[Int, _cmp_fn](array, k, size)


fn partition[
    type: DType,
    cmp_fn: fn (Scalar[type], Scalar[type]) capturing -> Bool,
](array: UnsafePointer[Scalar[type]], k: Int, size: Int):
    """Partition the input buffer inplace such that first k elements are the
    largest (or smallest if cmp_fn is < operator) elements.
    The ordering of the first k elements is undefined.

    Parameters:
        type: DType of the underlying data.
        cmp_fn: Comparison functor of (type, type) capturing -> Bool type.

    Args:
        array: Input buffer.
        k: Index of the partition element.
        size: The length of the buffer.
    """

    @parameter
    fn _cmp_fn(
        lhs: _SortWrapper[Scalar[type]], rhs: _SortWrapper[Scalar[type]]
    ) -> Bool:
        return cmp_fn(lhs.data, rhs.data)

    _partition[Scalar[type], _cmp_fn](array, k, size)


# ===----------------------------------------------------------------------===#
# sort
# ===----------------------------------------------------------------------===#


# Junction from public to private API
fn _sort[
    type: CollectionElement,
    cmp_fn: fn (_SortWrapper[type], _SortWrapper[type]) capturing -> Bool,
](ptr: UnsafePointer[type], len: Int):
    if len <= 5:
        _delegate_small_sort[type, cmp_fn](ptr, len)
        return

    if len < 32:
        _insertion_sort[type, cmp_fn](ptr, len)
        return

    _quicksort[type, cmp_fn](ptr, len)


# TODO (MSTDL-766): The Int and Scalar[type] overload should be remove
# (same for partition)
# Eventually we want a sort that takes a Span and one that takes a List with
# optional cmp_fn.
fn sort[
    type: CollectionElement,
    cmp_fn: fn (type, type) capturing -> Bool,
](ptr: UnsafePointer[type], len: Int):
    """Sort the list inplace.
    The function doesn't return anything, the list is updated inplace.

    Parameters:
        type: CollectionElement type of the underlying data.
        cmp_fn: The comparison function.

    Args:
        ptr: Pointer to the start of the memory to be sorted.
        len: Number of elements from ptr that to be sorted.
    """

    @parameter
    fn _cmp_fn(lhs: _SortWrapper[type], rhs: _SortWrapper[type]) -> Bool:
        return cmp_fn(lhs.data, rhs.data)

    _sort[type, _cmp_fn](ptr, len)


fn sort[
    type: CollectionElement,
    cmp_fn: fn (Int, Int) capturing -> Bool,
](ptr: UnsafePointer[Int], len: Int):
    """Sort the list inplace.
    The function doesn't return anything, the list is updated inplace.

    Parameters:
        type: CollectionElement type of the underlying data.
        cmp_fn: The comparison function.

    Args:
        ptr: Pointer to the start of the memory to be sorted.
        len: Number of elements from ptr that to be sorted.
    """

    @parameter
    fn _cmp_fn(lhs: _SortWrapper[Int], rhs: _SortWrapper[Int]) -> Bool:
        return cmp_fn(lhs.data, rhs.data)

    _sort[Int, _cmp_fn](ptr, len)


fn sort[
    type: DType,
    cmp_fn: fn (Scalar[type], Scalar[type]) capturing -> Bool,
](ptr: UnsafePointer[Scalar[type]], len: Int):
    """Sort the list inplace.
    The function doesn't return anything, the list is updated inplace.

    Parameters:
        type: CollectionElement type of the underlying data.
        cmp_fn: The comparison function.

    Args:
        ptr: Pointer to the start of the memory to be sorted.
        len: Number of elements from ptr that to be sorted.
    """

    @parameter
    fn _cmp_fn(
        lhs: _SortWrapper[Scalar[type]], rhs: _SortWrapper[Scalar[type]]
    ) -> Bool:
        return cmp_fn(lhs.data, rhs.data)

    _sort[Scalar[type], _cmp_fn](ptr, len)


fn sort(ptr: UnsafePointer[Int], len: Int):
    """Sort the list inplace.
    The function doesn't return anything, the list is updated inplace.

    Args:
        ptr: Pointer to the start of the memory to be sorted.
        len: Number of elements from ptr that to be sorted.
    """

    @parameter
    fn _cmp_fn(lhs: Int, rhs: Int) -> Bool:
        return lhs < rhs

    sort[Int, _cmp_fn](ptr, len)


fn sort[
    type: DType,
](ptr: UnsafePointer[Scalar[type]], len: Int):
    """Sort the list inplace.
    The function doesn't return anything, the list is updated inplace.

    Parameters:
        type: CollectionElement type of the underlying data.

    Args:
        ptr: Pointer to the start of the memory to be sorted.
        len: Number of elements from ptr that to be sorted.
    """

    @parameter
    fn _cmp_fn(lhs: Scalar[type], rhs: Scalar[type]) -> Bool:
        return lhs < rhs

    sort[type, _cmp_fn](ptr, len)


fn sort[
    type: CollectionElement,
    cmp_fn: fn (Int, Int) capturing -> Bool,
](inout list: List[Int]):
    """Sort the list inplace.
    The function doesn't return anything, the list is updated inplace.

    Parameters:
        type: CollectionElement type of the underlying data.
        cmp_fn: The comparison function.

    Args:
        list: Input list to sort.
    """

    sort[Int, cmp_fn](list.data, len(list))


fn sort[
    type: DType,
    cmp_fn: fn (Scalar[type], Scalar[type]) capturing -> Bool,
](inout list: List[Scalar[type]]):
    """Sort the list inplace.
    The function doesn't return anything, the list is updated inplace.

    Parameters:
        type: DType of the underlying data.
        cmp_fn: The comparison function.

    Args:
        list: Input list to sort.
    """

    sort[type, cmp_fn](list.data, len(list))


fn sort[
    type: CollectionElement,
    cmp_fn: fn (type, type) capturing -> Bool,
](inout list: List[type]):
    """Sort the list inplace.
    The function doesn't return anything, the list is updated inplace.

    Parameters:
        type: CollectionElement type of the underlying data.
        cmp_fn: The comparison function.

    Args:
        list: Input list to sort.
    """

    sort[type, cmp_fn](list.data, len(list))


fn sort(inout list: List[Int]):
    """Sort the list inplace.
    The function doesn't return anything, the list is updated inplace.

    Args:
        list: Input integer list to sort.
    """

    @parameter
    fn _cmp_fn(lhs: Int, rhs: Int) -> Bool:
        return lhs < rhs

    sort[Int, _cmp_fn](list.data, len(list))


fn sort[type: DType](inout list: List[Scalar[type]]):
    """Sort the list inplace.
    The function doesn't return anything, the list is updated inplace.

    Parameters:
        type: DType of the underlying data.

    Args:
        list: Input vector to sort.
    """

    @parameter
    fn _cmp_fn(lhs: Scalar[type], rhs: Scalar[type]) -> Bool:
        return lhs < rhs

    sort[type, _cmp_fn](list.data, len(list))


fn sort[type: ComparableCollectionElement](inout list: List[type]):
    """Sort list of the order comparable elements in-place.

    Parameters:
        type: The order comparable collection element type.

    Args:
        list: The list of the scalars which will be sorted in-place.
    """

    @parameter
    fn _cmp_fn(a: type, b: type) -> Bool:
        return a < b

    sort[type, _cmp_fn](list.data, len(list))


# ===----------------------------------------------------------------------===#
# sort networks
# ===----------------------------------------------------------------------===#


@always_inline
fn _sort2[
    type: CollectionElement,
    cmp_fn: fn (_SortWrapper[type], _SortWrapper[type]) capturing -> Bool,
](array: UnsafePointer[type], offset0: Int, offset1: Int):
    var a = array[offset0]
    var b = array[offset1]
    if not cmp_fn(a, b):
        array[offset0] = b
        array[offset1] = a


@always_inline
fn _sort3[
    type: CollectionElement,
    cmp_fn: fn (_SortWrapper[type], _SortWrapper[type]) capturing -> Bool,
](array: UnsafePointer[type], offset0: Int, offset1: Int, offset2: Int):
    _sort2[type, cmp_fn](array, offset0, offset1)
    _sort2[type, cmp_fn](array, offset1, offset2)
    _sort2[type, cmp_fn](array, offset0, offset1)


@always_inline
fn _sort_partial_3[
    type: CollectionElement,
    cmp_fn: fn (_SortWrapper[type], _SortWrapper[type]) capturing -> Bool,
](array: UnsafePointer[type], offset0: Int, offset1: Int, offset2: Int):
    var a = array[offset0]
    var b = array[offset1]
    var c = array[offset2]
    var r = cmp_fn(c, a)
    var t = c if r else a
    if r:
        array[offset2] = a
    if cmp_fn(b, t):
        array[offset0] = b
        array[offset1] = t
    elif r:
        array[offset0] = t


@always_inline
fn _small_sort[
    n: Int,
    type: CollectionElement,
    cmp_fn: fn (_SortWrapper[type], _SortWrapper[type]) capturing -> Bool,
](array: UnsafePointer[type]):
    @parameter
    if n == 2:
        _sort2[type, cmp_fn](array, 0, 1)
        return

    @parameter
    if n == 3:
        _sort2[type, cmp_fn](array, 1, 2)
        _sort_partial_3[type, cmp_fn](array, 0, 1, 2)
        return

    @parameter
    if n == 4:
        _sort2[type, cmp_fn](array, 0, 2)
        _sort2[type, cmp_fn](array, 1, 3)
        _sort2[type, cmp_fn](array, 0, 1)
        _sort2[type, cmp_fn](array, 2, 3)
        _sort2[type, cmp_fn](array, 1, 2)
        return

    @parameter
    if n == 5:
        _sort2[type, cmp_fn](array, 0, 1)
        _sort2[type, cmp_fn](array, 3, 4)
        _sort_partial_3[type, cmp_fn](array, 2, 3, 4)
        _sort2[type, cmp_fn](array, 1, 4)
        _sort_partial_3[type, cmp_fn](array, 0, 2, 3)
        _sort_partial_3[type, cmp_fn](array, 1, 2, 3)
        return
