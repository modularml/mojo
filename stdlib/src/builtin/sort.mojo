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

# ===----------------------------------------------------------------------=== #
#  Scalar list sorting
# ===----------------------------------------------------------------------=== #


@always_inline
fn insertion_sort[dtype: DType](inout list: List[Scalar[dtype]]):
    """Sort list of scalars in-place with insertion sort algorithm.

    Parameters:
        dtype: The dtype of the scalar.

    Args:
        list: The list of the scalars which will be sorted in-place.
    """
    for i in range(1, len(list)):
        var key = list[i]
        var j = i - 1
        while j >= 0 and key < list[j]:
            list[j + 1] = list[j]
            j -= 1
        list[j + 1] = key


fn _quick_sort[dtype: DType](inout list: List[Scalar[dtype]], low: Int, high: Int):
    """Sort section of the list, between low and high, with quick sort algorithm in-place.

    Parameters:
        dtype: The dtype of the scalar.

    Args:
        list: The list of the scalars which will be sorted in-place.
        low: Int value identifying the lowest index of the list section to be sorted.
        high: Int value identifying the highest index of the list section to be sorted.
    """

    @always_inline
    @parameter
    fn _partition(low: Int, high: Int) -> Int:
        var pivot = list[high]
        var i = low - 1
        for j in range(low, high):
            if list[j] <= pivot:
                i += 1
                list[j], list[i] = list[i], list[j]
        list[i + 1], list[high] = list[high], list[i + 1]
        return i + 1

    if low < high:
        var pi = _partition(low, high)
        _quick_sort(list, low, pi - 1)
        _quick_sort(list, pi + 1, high)


@always_inline
fn quick_sort[dtype: DType](inout list: List[Scalar[dtype]]):
    """Sort list of scalars in-place with quick sort algorithm.

    Parameters:
        dtype: The dtype of the scalar.

    Args:
        list: The list of the scalars which will be sorted in-place.
    """
    _quick_sort(list, 0, len(list) - 1)


fn sort[dtype: DType, slist_ub: Int = 64](inout list: List[Scalar[dtype]]):
    """Sort list of scalars in-place. This function picks the best algorithm based on the list length.

    Parameters:
        dtype: The dtype of the scalar.
        slist_ub: The upper bound for a list size which is considered small.

    Args:
        list: The list of the scalars which will be sorted in-place.
    """
    var count = len(list)
    if count <= slist_ub:
        insertion_sort(list)  # small lists are best sorted with insertion sort
    else:
        quick_sort(list)  # medium lists are best sorted with quick sort


# ===----------------------------------------------------------------------=== #
#  Comparable elements list sorting
# ===----------------------------------------------------------------------=== #


@always_inline
fn insertion_sort[type: ComparableCollectionElement](inout list: List[type]):
    """Sort list of the order comparable elements in-place with insertion sort algorithm.

    Parameters:
        type: The order comparable collection element type.

    Args:
        list: The list of the order comparable elements which will be sorted in-place.
    """
    for i in range(1, len(list)):
        var key = list[i]
        var j = i - 1
        while j >= 0 and key < list[j]:
            list[j + 1] = list[j]
            j -= 1
        list[j + 1] = key


fn _quick_sort[
    type: ComparableCollectionElement
](inout list: List[type], low: Int, high: Int):
    """Sort section of the list, between low and high, with quick sort algorithm in-place.

    Parameters:
        type: The order comparable collection element type.

    Args:
        list: The list of the order comparable elements which will be sorted in-place.
        low: Int value identifying the lowest index of the list section to be sorted.
        high: Int value identifying the highest index of the list section to be sorted.
    """

    @always_inline
    @parameter
    fn _partition(low: Int, high: Int) -> Int:
        var pivot = list[high]
        var i = low - 1
        for j in range(low, high):
            if list[j] <= pivot:
                i += 1
                list[j], list[i] = list[i], list[j]
        list[i + 1], list[high] = list[high], list[i + 1]
        return i + 1

    if low < high:
        var pi = _partition(low, high)
        _quick_sort(list, low, pi - 1)
        _quick_sort(list, pi + 1, high)


@always_inline
fn quick_sort[type: ComparableCollectionElement](inout list: List[type]):
    """Sort list of the order comparable elements in-place with quick sort algorithm.

    Parameters:
        type: The order comparable collection element type.

    Args:
        list: The list of the order comparable elements which will be sorted in-place.
    """
    _quick_sort(list, 0, len(list) - 1)


fn sort[
    type: ComparableCollectionElement, slist_ub: Int = 64
](inout list: List[type]):
    """Sort list of the order comparable elements in-place. This function picks the best algorithm based on the list length.

    Parameters:
        type: The order comparable collection element type.
        slist_ub: The upper bound for a list size which is considered small.

    Args:
        list: The list of the scalars which will be sorted in-place.
    """
    var count = len(list)
    if count <= slist_ub:
        insertion_sort(list)  # small lists are best sorted with insertion sort
    else:
        quick_sort(list)  # others are best sorted with quick sort
