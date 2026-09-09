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
"""Provides CPU implementations of top-p (nucleus) and min-p sampling for autoregressive token generation."""


from std.math import iota

from std.memory import ThinAllocation, dealloc
from std.memory.alloc import Layout as AllocLayout
from std.random import random_float64
from layout import Coord, Idx, TileTensor, coord_to_index_list, row_major
from nn.softmax import softmax_inline

from std.utils import IndexList


@always_inline
def top_p_sampling[
    dtype: DType,
    out_idx_type: DType,
    //,
    _test_sort: Bool = False,
](
    top_ps: TileTensor[mut=False, dtype, ...],
    input_logits: TileTensor[mut=True, dtype, ...],
    out_token_ids: TileTensor[mut=True, out_idx_type, ...],
    temperature: Scalar[dtype] = 1,
) raises:
    """
    Naive CPU implementation of Top-P sampling for token selection.
    This function applies temperature scaling, softmax, a merge sort, and then
    samples tokens based on the cumulative probability mass (Top-P).

    Parameters:
        dtype: Element type of `input_logits`, `top_ps`, and `temperature`
            (inferred).
        out_idx_type: Element type of the `out_token_ids` tensor (inferred).
        _test_sort: When true, copies the sorted probabilities back into
            `input_logits` to verify descending order (defaults to false).
    Args:
        top_ps: Per-batch cumulative probability mass thresholds in the
            range (0, 1].
        input_logits: Rank-2 logits tensor of shape (batch, vocab) read
            for temperature scaling and softmax.
        out_token_ids: Rank-2 output tensor receiving the sampled token
            index at column 0 of each batch row.
        temperature: Positive scalar divisor applied to logits before the
            softmax (defaults to 1).
    """
    # TODO: Implement rank generalization
    comptime assert input_logits.rank == 2, "Only rank 2 tensors are supported"
    _topp_minp_sampling[is_top_p=True, _test_sort=_test_sort](
        top_ps, input_logits, out_token_ids, temperature
    )


@always_inline
def min_p_sampling[
    dtype: DType,
    out_idx_type: DType,
    //,
    _test_sort: Bool = False,
](
    min_ps: TileTensor[mut=False, dtype, ...],
    input_logits: TileTensor[mut=True, dtype, ...],
    out_token_ids: TileTensor[mut=True, out_idx_type, ...],
    temperature: Scalar[dtype] = 1,
) raises:
    """
    Naive CPU implementation of Min-P sampling for token selection.
    This function applies temperature scaling, softmax, a merge sort, and then
    samples tokens based on the calculated probability threshold (Min-P).

    Parameters:
        dtype: Element type of `input_logits`, `min_ps`, and `temperature`
            (inferred).
        out_idx_type: Element type of the `out_token_ids` tensor (inferred).
        _test_sort: When true, copies the sorted probabilities back into
            `input_logits` to verify descending order (defaults to false).
    Args:
        min_ps: Per-batch minimum probability thresholds in the range
            (0, 1).
        input_logits: Rank-2 logits tensor of shape (batch, vocab) read
            for temperature scaling and softmax.
        out_token_ids: Rank-2 output tensor receiving the sampled token
            index at column 0 of each batch row.
        temperature: Positive scalar divisor applied to logits before the
            softmax (defaults to 1).
    """
    _topp_minp_sampling[is_top_p=False, _test_sort=_test_sort](
        min_ps, input_logits, out_token_ids, temperature
    )


@always_inline
def _topp_minp_sampling[
    dtype: DType,
    out_idx_type: DType,
    //,
    is_top_p: Bool,
    _test_sort: Bool = False,
](
    p_thresholds: TileTensor[mut=False, dtype, ...],
    input_logits: TileTensor[mut=True, dtype, ...],
    out_token_ids: TileTensor[mut=True, out_idx_type, ...],
    temperature: Scalar[dtype] = 1,
) raises:
    """
    Naive CPU implementation of Top-P/Min-P sampling for token selection.
    This function applies temperature scaling, softmax, a merge sort, and then
    samples tokens based on either cumulative probability mass (Top-P) or
    minimum probability threshold (Min-P).

    Parameters:
        dtype: DType - The data type of the input logits, p_thresholds, and temperature.
        out_idx_type: DType - The data type for output token indices.
        is_top_p: Bool - Whether to use Top-P (True) or Min-P (False) sampling.
        _test_sort: Bool - For internal testing purposes to check if the
            sorted probs are in descending order. If true, copies the sorted
            probs back into input_logits.
    Args:
        p_thresholds: TileTensor[dtype] - Sampling thresholds, one per batch.
        input_logits: TileTensor[dtype] - Input logits (modified in-place).
        out_token_ids: TileTensor[out_idx_type] - Output sampled token IDs.
        temperature: Scalar[dtype] - Temperature for logits scaling.
    """
    comptime assert (
        input_logits.flat_rank == 2
    ), "Only rank 2 tensors are supported"
    comptime assert (
        out_token_ids.flat_rank == 2
    ), "Only rank 2 tensors are supported"
    comptime assert p_thresholds.flat_rank == 1
    var input_shape = coord_to_index_list(input_logits.layout.shape_coord())
    var batch_size = input_shape[0]
    var vocab_size = input_shape[1]

    var sorted_probs_alloc = alloc(
        AllocLayout[Scalar[dtype]](count=batch_size * vocab_size)
    ).into_managed()
    var sorted_probs_ptr: UnsafePointer[
        Scalar[dtype], origin_of(sorted_probs_alloc)
    ] = sorted_probs_alloc.unsafe_ptr()
    var sorted_probs = TileTensor(
        sorted_probs_ptr,
        row_major(Coord(batch_size, vocab_size)),
    )

    var sorted_ids_alloc = alloc(
        AllocLayout[Scalar[out_idx_type]](count=batch_size * vocab_size)
    ).into_managed()
    var sorted_ids_ptr: UnsafePointer[
        Scalar[out_idx_type], origin_of(sorted_ids_alloc)
    ] = sorted_ids_alloc.unsafe_ptr()
    var sorted_ids = TileTensor(
        sorted_ids_ptr,
        row_major(Coord(batch_size, vocab_size)),
    )

    comptime assert sorted_probs.element_size == out_token_ids.element_size
    comptime assert out_token_ids.element_size == 1

    # Initialize sorted_ids with iota values
    for batch_id in range(batch_size):
        iota(sorted_ids.ptr + (batch_id * vocab_size), vocab_size)
        # Copy input_logits to sorted_probs
        for i in range(vocab_size):
            var batch_offset = batch_id * vocab_size
            sorted_probs.raw_store(
                batch_offset + i, input_logits.raw_load(batch_offset + i)
            )

    @__parameter
    @__copy_capture(input_logits)
    def apply_temperature[
        _simd_width: Int
    ](coords: Coord) -> SIMD[dtype, _simd_width]:
        var val = input_logits.load[width=_simd_width](coords)
        return val / temperature

    softmax_inline[
        simd_width=1, rank=input_logits.rank, input_fn=apply_temperature
    ](
        input_logits.layout.shape_coord(),
        sorted_probs,
        axis=input_logits.rank - 1,
    )

    sort_buf_descending(sorted_probs, sorted_ids, vocab_size)

    # Copy sorted probs back to input_logits if testing
    comptime if _test_sort:
        for i in range(batch_size * vocab_size):
            input_logits.raw_store(i, sorted_probs.raw_load(i))

    # Process each batch
    for batch in range(batch_size):
        var p_threshold = p_thresholds[batch]

        comptime if is_top_p:
            # Sample using top-p (nucleus) sampling
            var r = p_threshold * random_float64().cast[dtype]()
            for i in range(vocab_size):
                r -= sorted_probs[batch, i]
                if r <= 0 or i == vocab_size - 1:
                    var sid = sorted_ids[batch, i]
                    out_token_ids[batch, 0] = sid
                    break
        else:
            # Sample using min-p sampling
            # Step 1: Filter out tokens with probabilities less than min-p threshold
            var sum_filtered_probs = SIMD[dtype, out_token_ids.element_size](
                0.0
            )
            var num_filtered_tokens = 0
            for i in range(vocab_size):
                if sorted_probs[batch, i][0] >= p_threshold[0]:
                    sum_filtered_probs += sorted_probs[batch, i]
                    num_filtered_tokens += 1
                else:
                    break

            # Step 2: Sample from normalized distribution of remaining tokens
            var r = sum_filtered_probs * random_float64().cast[dtype]()

            # Step 3: Select token based on normalized probabilities
            for i in range(num_filtered_tokens):
                r -= sorted_probs[batch, i]
                if r <= 0 or i == vocab_size - 1:
                    var sid = sorted_ids[batch, i]
                    out_token_ids[batch, 0] = sid
                    break

    dealloc(sorted_ids_alloc^)
    dealloc(sorted_probs_alloc^)


@always_inline
def sort_buf_descending[
    dtype: DType, out_idx_type: DType
](
    mut buf_keys: TileTensor[mut=True, dtype, ...],
    mut buf_ids: TileTensor[mut=True, out_idx_type, ...],
    vocab_size: Int,
):
    """Sort each batch separately in descending order using parallel merge sort.

    Parameters:
        dtype: Element type of `buf_keys` (inferred).
        out_idx_type: Element type of `buf_ids` (inferred).
    Args:
        buf_keys: Rank-2 keys sorted in place in descending order, one
            row per batch.
        buf_ids: Rank-2 indices carried alongside `buf_keys` so each key
            retains its original position.
        vocab_size: Number of elements per batch row; the total element
            count divided by this gives the batch count.
    """
    comptime assert buf_keys.rank == 2, "rank must be 2"
    var batch_size = buf_keys.num_elements() // vocab_size

    for batch_id in range(batch_size):
        var start = batch_id * vocab_size
        var end = start + vocab_size
        merge_sort_recursive(buf_keys, buf_ids, start, end)


def merge_sort_recursive[
    dtype: DType,
    out_idx_type: DType,
](
    mut buf_keys: TileTensor[mut=True, dtype, ...],
    mut buf_ids: TileTensor[mut=True, out_idx_type, ...],
    start: Int,
    end: Int,
):
    """
    Recursive merge sort implementation.

    Parameters:
        dtype: Element type of `buf_keys` (inferred).
        out_idx_type: Element type of `buf_ids` (inferred).
    Args:
        buf_keys: Rank-2 keys sorted in place in descending order, one
            row per batch.
        buf_ids: Rank-2 indices carried alongside `buf_keys` so each key
            retains its original position.
        start: Inclusive start index of the contiguous range to sort
            within the flattened buffer.
        end: Exclusive end index of the contiguous range to sort within
            the flattened buffer.
    """
    if end - start > 1:
        var mid = start + (end - start) // 2
        merge_sort_recursive(buf_keys, buf_ids, start, mid)
        merge_sort_recursive(buf_keys, buf_ids, mid, end)
        merge(buf_keys, buf_ids, start, mid, end)


@always_inline
def merge[
    dtype: DType, out_idx_type: DType
](
    mut buf_keys: TileTensor[mut=True, dtype, ...],
    mut buf_ids: TileTensor[mut=True, out_idx_type, ...],
    start: Int,
    mid: Int,
    end: Int,
):
    """
    Merge two sorted subarrays into one sorted array.

    Parameters:
        dtype: Element type of `buf_keys` (inferred).
        out_idx_type: Element type of `buf_ids` (inferred).
    Args:
        buf_keys: Rank-2 keys holding two adjacent sorted subranges that
            are merged in place in descending order.
        buf_ids: Rank-2 indices carried alongside `buf_keys` so each key
            retains its original position.
        start: Inclusive start index of the left sorted subrange.
        mid: Exclusive end of the left subrange and inclusive start of
            the right subrange.
        end: Exclusive end index of the right sorted subrange.
    """
    var left_size = mid - start
    var right_size = end - mid

    # Create temporary arrays
    var left_keys_ptr = alloc(AllocLayout[Scalar[dtype]](count=left_size))
    var right_keys_ptr = alloc(AllocLayout[Scalar[dtype]](count=right_size))
    var left_ids_ptr = alloc(AllocLayout[Scalar[out_idx_type]](count=left_size))
    var right_ids_ptr = alloc(
        AllocLayout[Scalar[out_idx_type]](count=right_size)
    )
    var left_keys_data: UnsafePointer[
        Scalar[dtype], origin_of(left_keys_ptr._alloc)
    ] = left_keys_ptr.unsafe_ptr()
    var right_keys_data: UnsafePointer[
        Scalar[dtype], origin_of(right_keys_ptr._alloc)
    ] = right_keys_ptr.unsafe_ptr()
    var left_ids_data: UnsafePointer[
        Scalar[out_idx_type], origin_of(left_ids_ptr._alloc)
    ] = left_ids_ptr.unsafe_ptr()
    var right_ids_data: UnsafePointer[
        Scalar[out_idx_type], origin_of(right_ids_ptr._alloc)
    ] = right_ids_ptr.unsafe_ptr()

    # Copy data to temporary arrays
    for i in range(left_size):
        left_keys_data[i] = buf_keys.raw_load(start + i)
        left_ids_data[i] = buf_ids.raw_load(start + i)
    for i in range(right_size):
        right_keys_data[i] = buf_keys.raw_load(mid + i)
        right_ids_data[i] = buf_ids.raw_load(mid + i)

    # Merge back into original array
    var i = 0  # Index for left subarray
    var j = 0  # Index for right subarray
    var k = start  # Index for merged array

    while i < left_size and j < right_size:
        if (
            left_keys_data[i] >= right_keys_data[j]
        ):  # Use >= for descending order
            buf_keys.raw_store(k, left_keys_data[i])
            buf_ids.raw_store(k, left_ids_data[i])
            i += 1
        else:
            buf_keys.raw_store(k, right_keys_data[j])
            buf_ids.raw_store(k, right_ids_data[j])
            j += 1
        k += 1

    # Copy remaining elements if any
    while i < left_size:
        buf_keys.raw_store(k, left_keys_data[i])
        buf_ids.raw_store(k, left_ids_data[i])
        i += 1
        k += 1

    while j < right_size:
        buf_keys.raw_store(k, right_keys_data[j])
        buf_ids.raw_store(k, right_ids_data[j])
        j += 1
        k += 1

    # Free temporary arrays
    dealloc(left_keys_ptr^)
    dealloc(right_keys_ptr^)
    dealloc(left_ids_ptr^)
    dealloc(right_ids_ptr^)
