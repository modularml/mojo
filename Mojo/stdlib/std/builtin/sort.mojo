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
"""Implements the built-in `sort` function.

These are Mojo built-ins, so you don't need to import them.
"""

from std.math import ceil

from std.sys import bit_width_of
from std.bit import count_leading_zeros
from std.collections import Span
from std.memory.alloc import alloc, dealloc, Layout

# ===-----------------------------------------------------------------------===#
# sort
# ===-----------------------------------------------------------------------===#

comptime insertion_sort_threshold = 32
"""Threshold below which insertion sort is used instead of quicksort."""


@always_inline
def _insertion_sort[
    T: Copyable,
    origin: MutOrigin,
    //,
](span: Span[T, origin], cmp_fn: Some[def(T, T) -> Bool]):
    """Sort the array[start:end] slice"""
    var array = span.as_ref().as_unsafe_any_origin()
    var size = len(span)

    for i in range(1, size):
        var value = array.unsafe_offset(i).unsafe_take_pointee()
        var j = i

        # Find the placement of the value in the array, shifting as we try to
        # find the position. Throughout, we assume array[start:i] has already
        # been sorted.
        while j > 0 and cmp_fn(value, array[unsafe_offset=j - 1]):
            array.unsafe_offset(j).unsafe_write_move_from(
                array.unsafe_offset(j - 1)
            )
            j -= 1

        array.unsafe_offset(j).unsafe_write(value^)


# put everything that's "<" to the left of pivot
@always_inline
def _quicksort_partition_right[
    T: Copyable,
    origin: MutOrigin,
    //,
](span: Span[T, origin], cmp_fn: Some[def(T, T) -> Bool]) -> Int:
    var size = len(span)

    var left = 1
    var right = size - 1
    ref pivot_value = span.unsafe_get(0)

    while True:
        # no need for left < right since quick sort pick median of 3 as pivot
        while cmp_fn(span.unsafe_get(left), pivot_value):
            left += 1
        while left < right and not cmp_fn(span.unsafe_get(right), pivot_value):
            right -= 1
        if left >= right:
            var pivot_pos = left - 1
            span.unsafe_swap_elements(pivot_pos, 0)
            return pivot_pos
        span.unsafe_swap_elements(left, right)
        left += 1
        right -= 1


# put everything that's "<=" to the left of pivot
@always_inline
def _quicksort_partition_left[
    T: Copyable,
    origin: MutOrigin,
    //,
](span: Span[T, origin], cmp_fn: Some[def(T, T) -> Bool]) -> Int:
    var size = len(span)

    var left = 1
    var right = size - 1
    ref pivot_value = span.unsafe_get(0)

    while True:
        while left < right and not cmp_fn(pivot_value, span.unsafe_get(left)):
            left += 1
        while cmp_fn(pivot_value, span.unsafe_get(right)):
            right -= 1
        if left >= right:
            var pivot_pos = left - 1
            span.unsafe_swap_elements(pivot_pos, 0)
            return pivot_pos
        span.unsafe_swap_elements(left, right)
        left += 1
        right -= 1


def _heap_sort_fix_down[
    T: Copyable,
    origin: MutOrigin,
    //,
](span: Span[T, origin], idx: Int, cmp_fn: Some[def(T, T) -> Bool]):
    var size = len(span)
    var i = idx
    var j = i * 2 + 1
    while j < size:  # has left child
        # if right child exist and has higher value, swap with right
        if i * 2 + 2 < size and cmp_fn(
            span.unsafe_get(j), span.unsafe_get(i * 2 + 2)
        ):
            j = i * 2 + 2
        if not cmp_fn(span.unsafe_get(i), span.unsafe_get(j)):
            return
        span.unsafe_swap_elements(j, i)
        i = j
        j = i * 2 + 1


@always_inline
def _heap_sort[
    T: Copyable,
    origin: MutOrigin,
    //,
](span: Span[T, origin], cmp_fn: Some[def(T, T) -> Bool]):
    var size = len(span)
    # heapify
    for i in range(size // 2 - 1, -1, -1):
        _heap_sort_fix_down(span, i, cmp_fn)
    # sort
    while size > 1:
        size -= 1
        span.unsafe_swap_elements(0, size)
        _heap_sort_fix_down(span, 0, cmp_fn)


@always_inline
def _estimate_initial_height(size: Int) -> Int:
    # Compute the log2 of the size rounded upward.
    var log2: Int = (bit_width_of[DType.int]() - 1) ^ count_leading_zeros(
        size | 1
    )
    # The number 1.3 was chosen by experimenting the max stack size for random
    # input. This also depends on insertion_sort_threshold
    return max(2, Int(ceil(1.3 * Float64(log2))))


@always_inline
def _delegate_small_sort[
    T: Copyable,
    origin: MutOrigin,
    //,
](span: Span[T, origin], cmp_fn: Some[def(T, T) -> Bool]):
    var size = len(span)
    if size == 2:
        _small_sort[2](span, cmp_fn)

        return
    if size == 3:
        _small_sort[3](span, cmp_fn)
        return

    if size == 4:
        _small_sort[4](span, cmp_fn)
        return

    if size == 5:
        _small_sort[5](span, cmp_fn)
        return


# FIXME (MSTDL-808): Using _Pair over Span results in 1-3% improvement
# struct _Pair[T: AnyType]:
#     var ptr: Pointer[T]
#     var len: Int


@always_inline
def _quicksort[
    T: Copyable,
    origin: MutOrigin,
    //,
    *,
    do_smallsort: Bool = False,
](span: Span[T, origin], cmp_fn: Some[def(T, T) -> Bool]):
    var size = len(span)
    if size == 0:
        return

    var stack = List[Span[T, origin]](capacity=_estimate_initial_height(size))
    stack.append(span)
    while len(stack) > 0:
        var interval = stack.pop()
        var len = len(interval)

        comptime if do_smallsort:
            if len <= 5:
                _delegate_small_sort(interval, cmp_fn)
                continue

        if len < insertion_sort_threshold:
            _insertion_sort(interval, cmp_fn)
            continue

        # pick median of 3 as pivot
        _sort3(interval, len >> 1, 0, len - 1, cmp_fn)

        # if ptr[-1] == pivot_value, then everything in between will
        # be the same, so no need to recurse that interval
        # already have array[-1] <= array[0]
        var interval_ptr = interval.unsafe_ptr()
        if interval_ptr > span.unsafe_ptr() and not cmp_fn(
            interval_ptr[unsafe_offset=-1], interval_ptr[unsafe_offset=0]
        ):
            var pivot = _quicksort_partition_left(interval, cmp_fn)
            if len > pivot + 2:
                stack.append(
                    interval.unsafe_subspan(
                        offset=pivot + 1, length=len - pivot - 1
                    )
                )
            continue

        var pivot = _quicksort_partition_right(interval, cmp_fn)

        if len > pivot + 2:
            stack.append(
                interval.unsafe_subspan(
                    offset=pivot + 1, length=len - pivot - 1
                )
            )

        if pivot > 1:
            stack.append(interval.unsafe_subspan(offset=0, length=pivot))


# ===-----------------------------------------------------------------------===#
# stable sort
# ===-----------------------------------------------------------------------===#


# This is being passed mutable origins that are taken from the same memory
# object, so of course they alias.  The caller guarantees they don't overlap.
@__unsafe_nested_origins_read_only
def _merge[
    T: Copyable,
    span_origin: MutOrigin,
    result_origin: MutOrigin,
    //,
](
    span1: Span[T, span_origin],
    span2: Span[T, span_origin],
    result: Span[T, result_origin],
    cmp_fn: Some[def(T, T) -> Bool],
):
    """Merge span1 and span2 into result using the given cmp_fn. The function
    will crash if result is not large enough to hold both span1 and span2.

    Note that if result contains data previously, its destructor will not be called.

    Parameters:
        T: Type of the spans.
        span_origin: Origin of the input spans.
        result_origin: Origin of the result Span.

    Args:
        span1: The first span to be merged.
        span2: The second span to be merged.
        result: The output span.
        cmp_fn: The comparison function.
    """
    var span1_size = len(span1)
    var span2_size = len(span2)
    var span1_ptr = span1.as_ref()
    var span2_ptr = span2.as_ref()
    var res_ptr = result.as_ref()

    assert span1_size + span2_size <= len(
        result
    ), "The merge result does not fit in the span provided"
    var i = 0
    var j = 0
    var k = 0
    while i < span1_size:
        if j == span2_size:
            while i < span1_size:
                res_ptr.unsafe_offset(k).unsafe_write_move_from(
                    span1_ptr.unsafe_offset(i)
                )
                k += 1
                i += 1
            return
        if cmp_fn(span2.unsafe_get(j), span1.unsafe_get(i)):
            res_ptr.unsafe_offset(k).unsafe_write_move_from(
                span2_ptr.unsafe_offset(j)
            )
            j += 1
        else:
            res_ptr.unsafe_offset(k).unsafe_write_move_from(
                span1_ptr.unsafe_offset(i)
            )
            i += 1
        k += 1

    while j < span2_size:
        res_ptr.unsafe_offset(k).unsafe_write_move_from(
            span2_ptr.unsafe_offset(j)
        )
        k += 1
        j += 1


def _stable_sort_impl[
    T: Copyable,
    span_life: MutOrigin,
    tmp_life: MutOrigin,
    //,
](
    span: Span[T, span_life],
    temp_buff: Span[T, tmp_life],
    cmp_fn: Some[def(T, T) -> Bool],
):
    var size = len(span)
    if size <= 1:
        return
    var i = 0
    while i < size:
        _insertion_sort(
            span.unsafe_subspan(
                offset=i, length=min(insertion_sort_threshold, size - i)
            ),
            cmp_fn,
        )
        i += insertion_sort_threshold
    var merge_size = insertion_sort_threshold
    while merge_size < size:
        var j = 0
        while j + merge_size < size:
            var span1 = span.unsafe_subspan(offset=j, length=merge_size)
            var span2 = span.unsafe_subspan(
                offset=j + merge_size,
                length=min(merge_size, max(size - (j + merge_size), 0)),
            )
            _merge(span1, span2, temp_buff, cmp_fn)
            for i in range(merge_size + len(span2)):
                Pointer(to=span.unsafe_get(j + i)).unsafe_write_move_from(
                    Pointer(to=temp_buff.unsafe_get(i))
                )
            j += 2 * merge_size
        merge_size *= 2


def _stable_sort[
    T: Copyable,
    origin: MutOrigin,
    //,
](span: Span[T, origin], cmp_fn: Some[def(T, T) -> Bool]):
    var layout = Layout[T](count=len(span))
    var temp_buff = alloc(layout)
    var temp_buff_span = temp_buff.unsafe_span()
    _stable_sort_impl(span, temp_buff_span, cmp_fn)
    dealloc(temp_buff^)


# ===-----------------------------------------------------------------------===#
# partition
# ===-----------------------------------------------------------------------===#


@always_inline
def _partition[
    T: Copyable,
    origin: MutOrigin,
    //,
](span: Span[T, origin], cmp_fn: Some[def(T, T) -> Bool]) -> Int:
    var size = len(span)
    if size <= 1:
        return 0

    var pivot = size // 2

    var left = 0
    var right = size - 2

    var pivot_index = size - 1
    span.unsafe_swap_elements(pivot, pivot_index)

    while left < right:
        if cmp_fn(span.unsafe_get(left), span.unsafe_get(pivot_index)):
            left += 1
        elif not cmp_fn(span.unsafe_get(right), span.unsafe_get(pivot_index)):
            right -= 1
        else:
            span.unsafe_swap_elements(left, right)

    if cmp_fn(span.unsafe_get(right), span.unsafe_get(pivot_index)):
        right += 1
    span.unsafe_swap_elements(pivot_index, right)
    return right


def _partition[
    T: Copyable,
    origin: MutOrigin,
    //,
](var span: Span[T, origin], var k: Int, cmp_fn: Some[def(T, T) -> Bool]):
    while True:
        var pivot = _partition(span, cmp_fn)
        if pivot == k:
            return
        elif k < pivot:
            span = span.unsafe_subspan(offset=0, length=pivot)
        else:
            span._data = span._data.unsafe_offset(pivot + 1)
            span._len -= pivot + 1
            k -= pivot + 1


def partition[
    T: Copyable,
    origin: MutOrigin,
    //,
](span: Span[T, origin], k: Int, cmp_fn: Some[def(T, T) -> Bool]):
    """Partition the input buffer inplace such that first k elements are the
    largest (or smallest if cmp_fn is < operator) elements.
    The ordering of the first k elements is undefined.

    Parameters:
        T: Type of the underlying data.
        origin: Origin of span.

    Args:
        span: Input buffer.
        k: Index of the partition element.
        cmp_fn: The comparison function.
    """

    _partition(span, k, cmp_fn)


# ===-----------------------------------------------------------------------===#
# sort
# ===-----------------------------------------------------------------------===#


# Junction from public to private API
def _sort[
    T: Copyable,
    origin: MutOrigin,
    //,
    *,
    stable: Bool = False,
    do_smallsort: Bool = False,
](span: Span[T, origin], cmp_fn: Some[def(T, T) -> Bool]):
    comptime if do_smallsort:
        if len(span) <= 5:
            _delegate_small_sort(span, cmp_fn)
            return

    if len(span) < insertion_sort_threshold:
        _insertion_sort(span, cmp_fn)
        return

    comptime if stable:
        _stable_sort(span, cmp_fn)
    else:
        _quicksort[do_smallsort=do_smallsort](span, cmp_fn)


def sort[
    T: Copyable,
    origin: MutOrigin,
    //,
    *,
    stable: Bool = False,
](span: Span[T, origin], cmp_fn: Some[def(T, T) -> Bool]):
    """Sort a span in-place.
    The function doesn't return anything, the span is updated in-place.

    Parameters:
        T: Type of the underlying data.
        origin: Origin of span.
        stable: Whether the sort should be stable.

    Args:
        span: The span to be sorted.
        cmp_fn: The comparison function.
    """

    _sort[stable=stable](span, cmp_fn)


def sort[
    T: Copyable & Comparable,
    origin: MutOrigin,
    //,
    *,
    stable: Bool = False,
](span: Span[T, origin]):
    """Sort a span of comparable elements in-place.

    Parameters:
        T: The order comparable collection element type.
        origin: Origin of span.
        stable: Whether the sort should be stable.

    Args:
        span: The span to be sorted.
    """

    def _cmp_fn(a: T, b: T) -> Bool:
        return a < b

    sort[stable=stable](span, _cmp_fn)


# ===-----------------------------------------------------------------------===#
# sort networks
# ===-----------------------------------------------------------------------===#


@always_inline
def _sort2[
    T: Copyable,
    origin: MutOrigin,
    //,
](
    span: Span[T, origin],
    offset0: Int,
    offset1: Int,
    cmp_fn: Some[def(T, T) -> Bool],
):
    if not cmp_fn(span.unsafe_get(offset0), span.unsafe_get(offset1)):
        span.unsafe_swap_elements(offset0, offset1)


@always_inline
def _sort3[
    T: Copyable,
    origin: MutOrigin,
    //,
](
    span: Span[T, origin],
    offset0: Int,
    offset1: Int,
    offset2: Int,
    cmp_fn: Some[def(T, T) -> Bool],
):
    _sort2(span, offset0, offset1, cmp_fn)
    _sort2(span, offset1, offset2, cmp_fn)
    _sort2(span, offset0, offset1, cmp_fn)


@always_inline
def _sort_partial_3[
    T: Copyable,
    origin: MutOrigin,
    //,
](
    span: Span[T, origin],
    offset0: Int,
    offset1: Int,
    offset2: Int,
    cmp_fn: Some[def(T, T) -> Bool],
):
    """Sorts [a, b, c] assuming [b, c] is already sorted."""
    if cmp_fn(span.unsafe_get(offset0), span.unsafe_get(offset1)):
        return

    span.unsafe_swap_elements(offset0, offset1)
    if not cmp_fn(span.unsafe_get(offset1), span.unsafe_get(offset2)):
        span.unsafe_swap_elements(offset1, offset2)


@always_inline
def _small_sort[
    T: Copyable,
    origin: MutOrigin,
    //,
    n: Int,
](span: Span[T, origin], cmp_fn: Some[def(T, T) -> Bool]):
    comptime if n == 2:
        _sort2(span, 0, 1, cmp_fn)
        return

    comptime if n == 3:
        _sort2(span, 1, 2, cmp_fn)
        _sort_partial_3(span, 0, 1, 2, cmp_fn)
        return

    comptime if n == 4:
        _sort2(span, 0, 2, cmp_fn)
        _sort2(span, 1, 3, cmp_fn)
        _sort2(span, 0, 1, cmp_fn)
        _sort2(span, 2, 3, cmp_fn)
        _sort2(span, 1, 2, cmp_fn)
        return

    comptime if n == 5:
        _sort2(span, 0, 1, cmp_fn)
        _sort2(span, 3, 4, cmp_fn)
        _sort_partial_3(span, 2, 3, 4, cmp_fn)
        _sort2(span, 1, 4, cmp_fn)
        _sort_partial_3(span, 0, 2, 3, cmp_fn)
        _sort_partial_3(span, 1, 2, 3, cmp_fn)
        return
