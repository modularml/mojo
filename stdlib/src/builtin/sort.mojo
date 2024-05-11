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
fn insertion_sort[D: DType](inout list: List[Scalar[D]]):
    """Sort list of scalars in place with insertion sort algorithm.

    Parameters:
        D: The dtype of the scalar.

    Args:
        list: The list of the scalars which will be sorted inpace.
    """
    for i in range(1, len(list)):
        var key = list[i]
        var j = i - 1
        while j >= 0 and key < list[j]:
            list[j + 1] = list[j]
            j -= 1
        list[j + 1] = key


fn _quick_sort[D: DType](inout list: List[Scalar[D]], low: Int, high: Int):
    """Sort section of the list, between low and high, with quick sort algorithm inplace.

    Parameters:
        D: The dtype of the scalar.

    Args:
        list: The list of the scalars which will be sorted inpace.
        low: Int value identifying the lowes index of the list section to be sorted.
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
fn quick_sort[D: DType](inout list: List[Scalar[D]]):
    """Sort list of scalars in place with quick sort algorithm.

    Parameters:
        D: The dtype of the scalar.

    Args:
        list: The list of the scalars which will be sorted inpace.
    """
    _quick_sort(list, 0, len(list) - 1)


@always_inline
fn _simd_prefix_sum[D: DType](inout ptr: DTypePointer[D], size: Int):
    """Compute prefix sum of a buffer employing SIMD.

    Parameters:
        D: The dtype of the underlying values.

    Args:
        ptr: A pointer to the start of the buffer.
        size: The size of the buffer.
    """

    @parameter
    @always_inline
    fn prefix_sum[loops: Int]():
        """Compute prefix sum.

        Parameters:
            loops: Number of loops to perform.
        """

        # Width of the chunk: 5 -> 32, 6 -> 64, 7 -> 128, 8 -> 256
        alias width = 1 << loops

        @always_inline
        @parameter
        fn prefix_sum_on_chunk(
            ptr: DTypePointer[D], carry_over: SIMD[D, 1]
        ) -> SIMD[D, width]:
            """Compute prefix sum on chunk.

            Say we have a list [1, 2, 3, 4, 5, 6, 7, 8] number of loops is 2,
            number of elements is 4

            First loop, chunk = [1, 2, 3, 4], carry_over = 0
              1, 2, 3, 4
            + 0, 1, 2, 3
            = 1, 3, 5, 7
            + 0, 0, 1, 2
            = 1, 3, 6, 9
            + 0, 0, 0, 0 # done with loops add carry_over
            = 1, 3, 6, 9

            Second loop, chunk = [5, 6, 7, 8], carry_over = 9
               5,  6,  7,  8
            +  0,  5,  6,  7
            =  5, 11, 13, 15
            +  0,  0,  5,  6
            =  5, 11, 18, 21
            +  9,  9,  9,  9 # done with loops add carry_over
            = 14, 20, 27, 30

            Args:
                ptr: A pointer to the start of the chunk.
                carry_over: Last value from previous chunk.
            """
            var chunk = ptr.load[width=width]()

            @parameter
            fn add[loop: Int]():
                """Add shifted chunk to itself.
                E.g. [1, 2, 3, 4] + [0, 1, 2, 3]

                Parameters:
                    loop: Loop index used to compute the shift
                """
                chunk += chunk.shift_right[1 << loop]()

            unroll[add, loops]()

            chunk += carry_over
            return chunk

        var last_value: Scalar[D] = 0
        var i = 0

        while i + width <= size:
            var part = prefix_sum_on_chunk(ptr.offset(i), last_value)
            last_value = part[width - 1]
            ptr.store(i, part)
            i += width

        @parameter
        fn add_rest[loop: Int]():
            """Perform prefix sum on the rest of the buffer.

            Parameters:
                loop: Loop index, used to compute the width
            """
            alias index = loop + 1
            alias w = width >> index
            if i + w <= size:
                var part = prefix_sum_on_chunk(ptr.offset(i), last_value)
                last_value = part[w - 1]
                ptr.store(i, part)
                i += w

        unroll[add_rest, loops]()

    @parameter
    if D.sizeof() == 1:
        prefix_sum[8]()  # 8 loops 256 elements per chunk
    elif D.sizeof() == 2:
        prefix_sum[7]()  # 7 loops 128 elements per chunk
    elif D.sizeof() == 4:
        prefix_sum[6]()  # 6 loops 64 elements per chunk
    else:
        prefix_sum[5]()  # 5 loops 32 elements per chunk


@always_inline
fn radix_sort[D: DType](inout list: List[Scalar[D]]):
    """Sort list of scalars in place with radix sort algorithm.

    Parameters:
        D: The dtype of the scalar.

    Args:
        list: The list of the scalars which will be sorted inpace.
    """

    @always_inline
    @parameter
    fn _radix_sort[CD: DType]():
        """Perform radix sort, by performing counting sort on every byte of the list element.

        Parameters:
            CD: The dtype of counts list.
        """

        @always_inline
        @parameter
        fn _counting_sort[byte_index: Int]():
            """Perform counting sort based only on the bytes at byte index of the list elements.

            Parameters:
                byte_index: The byte index to consider in this round.
            """

            @always_inline
            @parameter
            fn _ge_histogram_index(index: Int) -> Int:
                """Returns histogram index index, based on the element byte index and the dtype of the list element.
                For an unsigned int the index is just the byte at byte position.
                For signed int: -128 -> 0, 0 -> 128, 127 -> 255
                For float the logic is a bit more complex, but similar to signed int.

                Args:
                    index: The index of the element in the list
                """
                alias last_bit_8 = 1 << 7
                alias last_bit_16 = 1 << 15
                alias last_bit_32 = 1 << 31
                alias last_bit_64 = 1 << 63

                @parameter
                if D == DType.int8:
                    return (
                        int(
                            (bitcast[DType.uint8, 1](list[index]) ^ last_bit_8)
                            >> byte_index
                        )
                        & 255
                    )
                elif D == DType.int16:
                    return (
                        int(
                            (
                                bitcast[DType.uint16, 1](list[index])
                                ^ last_bit_16
                            )
                            >> byte_index
                        )
                        & 255
                    )
                elif D == DType.float16:
                    var f = bitcast[DType.uint16, 1](list[index])
                    var mask = bitcast[DType.uint16, 1](
                        -bitcast[DType.int16, 1](f >> 15) | last_bit_16
                    )
                    return int((f ^ mask) >> byte_index) & 255
                elif D == DType.int32:
                    return (
                        int(
                            (
                                bitcast[DType.uint32, 1](list[index])
                                ^ last_bit_32
                            )
                            >> byte_index
                        )
                        & 255
                    )
                elif D == DType.float32:
                    var f = bitcast[DType.uint32, 1](list[index])
                    var mask = bitcast[DType.uint32, 1](
                        -bitcast[DType.int32, 1](f >> 31) | last_bit_32
                    )
                    return int((f ^ mask) >> byte_index) & 255
                elif D == DType.int64:
                    return (
                        int(
                            (
                                bitcast[DType.uint64, 1](list[index])
                                ^ last_bit_64
                            )
                            >> byte_index
                        )
                        & 255
                    )
                elif D == DType.float64:
                    var f = bitcast[DType.uint64, 1](list[index])
                    var mask = bitcast[DType.uint64, 1](
                        -bitcast[DType.int64, 1](f >> 63) | last_bit_64
                    )
                    return int((f ^ mask) >> byte_index) & 255
                else:
                    return int(list[index] >> byte_index) & 255

            var size = len(list)
            var output = List[SIMD[D, 1]](capacity=size)
            memset_zero(output.data, size)
            output.resize(size)

            var histogram = stack_allocation[256, CD]()
            memset_zero(histogram, 256)

            for i in range(size):
                var index = _ge_histogram_index(i)
                histogram.store(index, histogram.load(index) + 1)

            _simd_prefix_sum[CD](histogram, 256)

            var i = size - 1
            while i >= 0:
                var index = _ge_histogram_index(i)
                output[int(histogram.load(index) - 1)] = list[i]
                histogram.store(index, histogram.load(index) - 1)
                i -= 1
            list = output

        @parameter
        fn call_counting_sort[index: Int]():
            _counting_sort[index * 8]()

        unroll[call_counting_sort, D.sizeof()]()

    var count = len(list)
    if count < int(UInt32.MAX):
        if count < int(UInt8.MAX):
            _radix_sort[DType.uint8]()
        else:
            _radix_sort[DType.uint16]()
    else:
        if count < int(UInt32.MAX):
            _radix_sort[DType.uint32]()
        else:
            _radix_sort[DType.uint64]()


fn sort[D: DType](inout list: List[Scalar[D]]):
    """Sort list of scalars in place. This function picks the best algorithm based on the list length.

    Parameters:
        D: The dtype of the scalar.

    Args:
        list: The list of the scalars which will be sorted inpace.
    """
    var count = len(list)
    if count <= 64:
        insertion_sort(list)  # small lists are best sorted with insertion sort
    elif count <= 250:
        quick_sort(list)  # medium lists are best sorted with quick sort
    else:
        radix_sort(list)  # large lists are best sorted with radix sort


# ===----------------------------------------------------------------------=== #
#  Comparable elements list sorting
# ===----------------------------------------------------------------------=== #


@always_inline
fn insertion_sort[D: ComparableCollectionElement](inout list: List[D]):
    """Sort list of the order comparable elements in place with insertion sort algorithm.

    Parameters:
        D: The order comparable collection element type.

    Args:
        list: The list of the order comparable elements which will be sorted inpace.
    """
    for i in range(1, len(list)):
        var key = list[i]
        var j = i - 1
        while j >= 0 and key < list[j]:
            list[j + 1] = list[j]
            j -= 1
        list[j + 1] = key


fn _quick_sort[
    D: ComparableCollectionElement
](inout list: List[D], low: Int, high: Int):
    """Sort section of the list, between low and high, with quick sort algorithm inplace.

    Parameters:
        D: The order comparable collection element type.

    Args:
        list: The list of the order comparable elements which will be sorted inpace.
        low: Int value identifying the lowes index of the list section to be sorted.
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
fn quick_sort[D: ComparableCollectionElement](inout list: List[D]):
    """Sort list of the order comparable elements in place with quick sort algorithm.

    Parameters:
        D: The order comparable collection element type.

    Args:
        list: The list of the order comparable elements which will be sorted inpace.
    """
    _quick_sort(list, 0, len(list) - 1)


fn sort[D: ComparableCollectionElement](inout list: List[D]):
    """Sort list of the order comparable elements in place. This function picks the best algorithm based on the list length.

    Parameters:
        D: The order comparable collection element type.

    Args:
        list: The list of the scalars which will be sorted inpace.
    """
    var count = len(list)
    if count <= 64:
        insertion_sort(list)  # small lists are best sorted with insertion sort
    else:
        quick_sort(list)  # others are best sorted with quick sort
