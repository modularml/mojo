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
"""Provides GPU implementations of top-p (nucleus) and min-p sampling for autoregressive token generation."""


from std.math import ceildiv
from std.sys import bit_width_of

from std.builtin.dtype import _uint_type_of_width
from max.gpu import WARP_SIZE, block_idx, thread_idx
from max.gpu.sync import barrier
import max.gpu.primitives.warp as warp
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.host import Dim
from std.sys.info import has_apple_gpu_accelerator
from std.random import Random
from layout import Coord, Idx, TileTensor, row_major
from std.memory import bitcast, unsafe_stack_allocation
from nn.softmax import _softmax_gpu
from nn.topk import (
    TopK_2,
    _block_reduce_topk,
    _topk_dead_val,
)

from std.utils import IndexList

comptime DEBUG_FILE = False
comptime SEED = 42


@__name(t"topk_wrapper_{input_type}_{index_type}_{is_top_p}")
def topk_wrapper[
    input_type: DType,
    index_type: DType,
    *,
    is_top_p: Bool,
    block_size: Int,
    largest: Bool = True,
    _test_sort: Bool = False,
](
    K: Int32,
    num_elements: Int32,
    num_blocks_per_input: Int32,
    in_buffer: UnsafePointer[Scalar[input_type], ImmUntrackedOrigin],
    local_topk_vals: UnsafePointer[
        Scalar[input_type], MutUntrackedOrigin
    ],  # Output buffer of size num_blocks_per_input * K
    local_topk_idxs: UnsafePointer[
        Scalar[index_type], MutUntrackedOrigin
    ],  # Output buffer of size num_blocks_per_input * K
    p_threshold: UnsafePointer[Scalar[input_type], MutUntrackedOrigin],
    skip_sort: UnsafePointer[Scalar[.bool], MutUntrackedOrigin],
):
    """
    Copy of `Kernels/mojo/nn/topk.mojo:_topk_stage1` with the addition of
    max_vals and p_threshold arguments to determine if sorting is needed for
    top-p/min-p sampling.

    Parameters:
        input_type: DType - The data type of the elements.
        index_type: DType - The data type of the output indices.
        is_top_p: Bool - Whether this if for top-p sampling or min-p sampling.
        block_size: Int - The number of threads per block to use for the kernel.
        largest: Bool - Whether to find the maximum or minimum value.
        _test_sort: Bool - An internal test flag to not skip sort if testing.

    Arguments:
        K: Int - Number of top elements to select per block
        num_elements: Int - Size of last dimension of input buffer (vocab size)
        num_blocks_per_input: Int - Number of blocks used to process the input data
        in_buffer: Pointer[Scalar[input_type]] - Input buffer containing the elements to process
        local_topk_vals: Pointer[Scalar[input_type]] - Output buffer to store the local top-K values
        local_topk_idxs: Pointer[Scalar[index_type]] - Output buffer to store the indices of local top-K elements
        p_threshold: Pointer[Scalar[input_type]] - Threshold for top-p sampling if is_top_p is True else min-p coefficient
        skip_sort: Pointer[Scalar[.bool]] - Output buffer to store whether sorting is needed
    """
    var _K = Int(K)
    var _num_elements = Int(num_elements)
    var _num_blocks_per_input = Int(num_blocks_per_input)
    var tid = thread_idx.x
    var bid = block_idx.x

    var batch_id, block_lane = divmod(bid, _num_blocks_per_input)

    var _in_buffer = in_buffer + batch_id * _num_elements

    # # Allocate shared memory for the values and indices
    var topk_sram = unsafe_stack_allocation[
        block_size,
        TopK_2[input_type, largest],
        address_space=.SHARED,
    ]()

    # Pack the topk_vals and topk_idxs into shared memory
    var block_offset = block_lane * block_size
    var stride = block_size * _num_blocks_per_input
    topk_sram[tid] = TopK_2[input_type, largest]()
    for i in range(tid + block_offset, _num_elements, stride):
        topk_sram[tid].insert(_in_buffer[i], i)

    barrier()

    # Prepare for _K iterations to find the local top-_K elements
    for k in range(_K):
        # Initialize each thread with its own TopK_2 value and index
        var partial = topk_sram[tid]

        # Perform block-level reduction to find the maximum TopK_2
        var total = _block_reduce_topk[ascending=largest](partial)

        if tid == 0:
            # Store the local top-_K values and indices in global memory
            var vector_idx = total.p
            local_topk_vals[bid * _K + k] = total.u
            local_topk_idxs[bid * _K + k] = Int(vector_idx).cast[index_type]()

            comptime if is_top_p:
                # In top-p sampling, we check if the highest probability token exceeds
                # the probability threshold (p_threshold). If it does, we can skip sorting
                # since we'll just sample this token. Otherwise, we need to sort to find
                # all tokens that sum to p_threshold probability mass.
                skip_sort[batch_id] = (
                    total.u > p_threshold[batch_id]
                ) and not _test_sort
                # If we're testing sort, we can't skip sort
            else:
                # For min-p sampling, we calculate a dynamic threshold as:
                # threshold = min_p_coefficient * max_probability
                # This ensures we only consider tokens with probability at least
                # min_p_coefficient times the highest probability token.
                var p_threshold_val = p_threshold[batch_id] * total.u
                # update with actual min-p threshold
                p_threshold[batch_id] = p_threshold_val
                skip_sort[batch_id] = False

            # Remove the found maximum from consideration in the next iteration
            var orig_tid = (vector_idx - block_offset) % stride
            topk_sram[orig_tid].u = _topk_dead_val[input_type, largest]()

        barrier()


@always_inline
def normalize(value: BFloat16) -> UInt16:
    """
    Normalizes a bfloat16 value to an unsigned 16-bit integer for radix sort
    by flipping the sign bit for positive values and fully inverting negative
    values.
    """

    @always_inline
    def reinterpret(value: BFloat16) -> UInt16:
        # For unsigned integral types: No conversion needed, return as-is
        return bitcast[.uint16, 1](value)

    # Normalize bf16 values by flipping the sign bit for positive and fully
    # inverting negative numbers
    var bits = reinterpret(value)
    comptime sign_bit_mask = 0b1 << (bit_width_of[BFloat16]() - 1)
    if bits & UInt16(sign_bit_mask):
        # For negative numbers, flip all bits (two's complement behavior)
        return ~bits
    else:
        # For positive numbers, flip only the sign bit
        return bits ^ UInt16(sign_bit_mask)


@always_inline
def normalize_u32(value: UInt32) -> UInt32:
    """
    Returns a uint32 value unchanged since unsigned integers already sort
    correctly in radix sort.
    """
    return value


@always_inline
def normalize(value: Int32) -> UInt32:
    """
    Normalizes a signed 32-bit integer to unsigned by flipping the most
    significant bit so negative values sort before positive ones.
    """

    @always_inline
    def reinterpret(value: Int32) -> UInt32:
        # For signed integral types: Convert to unsigned int to ensure proper
        # comparison
        return value.cast[.uint32]()

    # For signed integers: Flip the most significant bit to ensure correct ordering
    # This makes negative numbers appear "smaller" than positive numbers in
    # unsigned comparison
    comptime sign_bit_mask = 0b1 << (bit_width_of[Int32]() - 1)

    return reinterpret(value) ^ UInt32(sign_bit_mask)


@always_inline
def normalize(value: UInt16) -> UInt16:
    """
    Returns a uint16 value unchanged since unsigned integers already sort
    correctly in radix sort.
    """
    return value


@always_inline
def normalize(value: Float32) -> UInt32:
    """
    Normalizes a float32 value to an unsigned 32-bit integer for radix sort
    by reinterpreting its bit pattern and flipping bits for negative values.
    """

    @always_inline
    def reinterpret(value: Float32) -> UInt32:
        # For floating-point types: Reinterpret the bit pattern as an unsigned int
        # This allows for comparison of floating-point values based on their binary
        # representation
        return bitcast[.uint32, 1](value)

    var bits = reinterpret(value)
    comptime sign_bit = bit_width_of[Float32]() - 1
    # Flip all bits if the value is negative (sign bit is 1)
    # This makes more negative numbers appear "smaller" in unsigned comparison
    return bits ^ ((-(bits >> UInt32(sign_bit))) | UInt32(0b1 << sign_bit))


@always_inline
def normalize(
    value: Scalar,
    out result: Scalar[_uint_type_of_width[bit_width_of[value.dtype]()]()],
):
    """
    Normalize the value to the appropriate unsigned integer type. This is needed
    for radix sort to work correctly.
    """
    comptime dtype = value.dtype

    comptime if dtype == .int32:
        return normalize(rebind[Int32](value)).cast[result.dtype]()
    elif dtype == .uint32:
        return normalize(rebind[UInt32](value)).cast[result.dtype]()
    elif dtype == .float32:
        return normalize(rebind[Float32](value)).cast[result.dtype]()
    # TODO: These below don't return uint32 so must generalize and fix
    elif dtype == .uint16:
        return normalize(rebind[UInt16](value)).cast[result.dtype]()
    elif dtype == .float16:
        return normalize(rebind[Float16](value)).cast[result.dtype]()
    elif dtype == .bfloat16:
        return normalize(rebind[BFloat16](value)).cast[result.dtype]()
    else:
        comptime assert False, "unhandled normalize type"


@always_inline
@__name(t"radix_sort_pairs_{dtype}_{out_idx_type}_{ascending}")
def radix_sort_pairs_kernel[
    dtype: DType,
    out_idx_type: DType,
    current_bit: Int,
    ascending: Bool = False,
    BLOCK_SIZE: Int = 256,  # found empirically
    NUM_BITS_PER_PASS: Int = 4,
](
    input_keys_: UnsafePointer[
        Scalar[dtype], MutUntrackedOrigin
    ],  # modifies input
    output_keys_: UnsafePointer[mut=True, Scalar[dtype], MutUntrackedOrigin],
    input_key_ids_: UnsafePointer[
        Scalar[out_idx_type], MutUntrackedOrigin
    ],  # modifies input
    output_key_ids_: UnsafePointer[Scalar[out_idx_type], MutUntrackedOrigin],
    num_keys: Int32,
    skip_sort: UnsafePointer[Scalar[.bool], MutUntrackedOrigin],
):
    """
    Radix pair sort kernel for (default) descending order.

    Parameters:
        dtype: DType - Data type.
        out_idx_type: DType - Output index type.
        current_bit: Int - Current bit to start sorting NUM_BITS_PER_PASS bits at.
        ascending: Bool - Whether to sort in ascending order.
        BLOCK_SIZE: Int - Block size.
        NUM_BITS_PER_PASS: Int - Number of bits per pass.

    Args:
        input_keys_: Input tensor values to sort.
        output_keys_: Output tensor values sorted in (default) descending order.
        input_key_ids_: Input tensor indices.
        output_key_ids_: Output tensor indices sorted in (default) descending order.
        num_keys: Number of keys to sort per batch.
        skip_sort: Whether sorting is skipped for this batch.

    Implementation based on:
    AMD. Introduction to GPU Radix Sort. GPUOpen, 2017. Available at:
    https://gpuopen.com/download/publications/Introduction_to_GPU_Radix_Sort.pdf.
    """
    var _num_keys = Int(num_keys)

    var tid = thread_idx.x
    var batch_id = block_idx.x
    var elems_per_thread = ceildiv(_num_keys, BLOCK_SIZE)
    comptime NUM_BUCKETS = 2**NUM_BITS_PER_PASS

    var input_keys = input_keys_ + batch_id * _num_keys
    var output_keys = output_keys_ + batch_id * _num_keys
    var input_key_ids = input_key_ids_ + batch_id * _num_keys
    var output_key_ids = output_key_ids_ + batch_id * _num_keys

    if skip_sort[batch_id]:
        return

    # Shared mem declarations
    var s_counts = unsafe_stack_allocation[
        BLOCK_SIZE * NUM_BUCKETS,
        Int32,
        address_space=.SHARED,
    ]()
    var total_counts = unsafe_stack_allocation[
        NUM_BUCKETS,
        Int32,
        address_space=.SHARED,
    ]()
    var total_offsets = unsafe_stack_allocation[
        (NUM_BUCKETS + 1),  # +1 extended size for descending
        Int32,
        address_space=.SHARED,
    ]()
    var total_offsets_descending = unsafe_stack_allocation[
        NUM_BUCKETS,
        Int32,
        address_space=.SHARED,
    ]()
    var s_thread_offsets = unsafe_stack_allocation[
        BLOCK_SIZE * NUM_BUCKETS,
        Int32,
        address_space=.SHARED,
    ]()

    # Initialize counts[NUM_BUCKETS]
    var counts_stack = Array[Int32, NUM_BUCKETS](fill=0)
    var counts_buf = TileTensor(counts_stack, row_major[NUM_BUCKETS]())
    var counts = counts_buf.ptr

    # Process elements and compute counts for each thread
    for index in range(tid * elems_per_thread, (tid + 1) * elems_per_thread):
        if index < _num_keys:
            var key = input_keys[index]
            var normalized_key = normalize(key)
            comptime KeyType = type_of(normalized_key)
            var radix = (normalized_key >> KeyType(current_bit)) & KeyType(
                NUM_BUCKETS - 1
            )
            counts[radix] += 1

    # Store counts[NUM_BUCKETS] per thread into shared memory s_counts
    comptime for i in range(NUM_BUCKETS):
        s_counts[tid * NUM_BUCKETS + i] = counts[i]
    barrier()

    # Compute total_counts[NUM_BUCKETS] by summing counts[NUM_BUCKETS] across threads
    if tid < NUM_BUCKETS:
        var sum = Int32(0)
        var bucket_offset = tid

        comptime for t in range(BLOCK_SIZE):
            sum += s_counts[t * NUM_BUCKETS + bucket_offset]
        total_counts[bucket_offset] = sum
    barrier()

    # Perform exclusive scan over total_counts[NUM_BUCKETS] to get total_offsets[NUM_BUCKETS]
    if tid == 0:
        total_offsets[0] = 0

        comptime for i in range(1, NUM_BUCKETS + 1):
            total_offsets[i] = total_offsets[i - 1] + total_counts[i - 1]

    # Compute per-thread starting offsets per radix value
    comptime for i in range(NUM_BUCKETS):
        s_thread_offsets[tid * NUM_BUCKETS + i] = s_counts[
            tid * NUM_BUCKETS + i
        ]
    barrier()

    # Perform exclusive scan over s_thread_offsets per radix value
    comptime for radix in range(NUM_BUCKETS):
        # Initialize the offset to 1, which will be used to determine the distance
        # between threads whose values will be reduced/summed.
        var offset = 1
        while offset < BLOCK_SIZE:
            # Initialize a temporary variable to store the value from the neighboring thread.
            var val = Int32(0)
            if tid >= offset:
                # If the current thread ID is greater than or equal to the offset,
                # fetch the value from the neighboring thread that is 'offset' positions behind.
                val = s_thread_offsets[(tid - offset) * NUM_BUCKETS + radix]

            # Synchronize all threads to ensure that the value fetching is complete.
            barrier()
            # Add the fetched value to the current thread's value.
            s_thread_offsets[tid * NUM_BUCKETS + radix] += val
            # Synchronize all threads to ensure that the addition is complete.
            barrier()
            # Double the offset for the next iteration to fetch values from farther threads.
            offset <<= 1

        # After the loop, set the first thread's offset to 0.
        if tid == 0:
            s_thread_offsets[tid * NUM_BUCKETS + radix] = 0
        else:
            # For all other threads, set the offset to the value of the previous thread.
            s_thread_offsets[tid * NUM_BUCKETS + radix] = s_thread_offsets[
                (tid - 1) * NUM_BUCKETS + radix
            ]
        # Synchronize all threads to ensure that the final offset values are set.
        barrier()

    # Compute total_offsets_descending[NUM_BUCKETS] if needed
    comptime if not ascending:
        if tid < NUM_BUCKETS:
            total_offsets_descending[tid] = (
                total_offsets[NUM_BUCKETS] - total_offsets[tid + 1]
            )
        barrier()

    # Each thread initializes local_offsets[NUM_BUCKETS] = 0
    var local_offsets_stack = Array[Int32, NUM_BUCKETS](fill=0)
    var local_offsets_buf = TileTensor(
        local_offsets_stack, row_major[NUM_BUCKETS]()
    )
    var local_offsets = local_offsets_buf.ptr

    # Now, each thread processes its elements, computes destination index, write to output
    for index in range(tid * elems_per_thread, (tid + 1) * elems_per_thread):
        if index < _num_keys:
            var key = input_keys[index]
            var normalized_key = normalize(key)
            comptime KeyType = type_of(normalized_key)
            var radix = Int(
                (normalized_key >> KeyType(current_bit))
                & KeyType(NUM_BUCKETS - 1)
            )

            # Adjust global_offset for ascending or descending order
            var global_offset: Int

            comptime if ascending:
                global_offset = Int(
                    total_offsets[radix]
                    + s_thread_offsets[tid * NUM_BUCKETS + radix]
                    + local_offsets[radix]
                )
            else:
                global_offset = Int(
                    total_offsets_descending[radix]
                    + s_thread_offsets[tid * NUM_BUCKETS + radix]
                    + local_offsets[radix]
                )

            output_keys[global_offset] = key

            comptime if current_bit == 0:
                output_key_ids[global_offset] = Scalar[out_idx_type](index)
            else:
                output_key_ids[global_offset] = input_key_ids[index]

            local_offsets[radix] += 1


struct DoubleBuffer[dtype: DType](ImplicitlyCopyable):
    """
    Holds two GPU buffers and alternates between them for double-buffered
    radix sort passes.

    The struct tracks which buffer is currently active and provides methods to
    access the current and alternate buffers, plus a swap operation to toggle
    between them.
    """

    var _d_buffers: Array[
        Optional[UnsafePointer[Scalar[Self.dtype], MutUntrackedOrigin]], 2
    ]
    var _selection: Int32
    var _size: Int

    def __init__(out self):
        self._d_buffers = {fill = None}
        self._selection = 0
        self._size = 0

    def __init__(
        out self,
        current: UnsafePointer[Scalar[Self.dtype], MutUntrackedOrigin],
        alternate: UnsafePointer[Scalar[Self.dtype], MutUntrackedOrigin],
        size: Int,
    ):
        self._d_buffers = [current, alternate]
        self._selection = 0
        self._size = size

    def __init__(out self, *, copy: Self):
        self._d_buffers = copy._d_buffers.copy()
        self._selection = copy._selection
        self._size = copy._size

    @always_inline
    def current(self, ctx: DeviceContext) -> DeviceBuffer[Self.dtype]:
        if self._d_buffers[self._selection]:
            return DeviceBuffer[Self.dtype](
                ctx,
                self._d_buffers[self._selection].unsafe_value(),
                self._size,
                owning=False,
            )
        else:
            return DeviceBuffer[Self.dtype].empty(ctx)

    @always_inline
    def alternate(self, ctx: DeviceContext) -> DeviceBuffer[Self.dtype]:
        if self._d_buffers[self._selection ^ 1]:
            return DeviceBuffer[Self.dtype](
                ctx,
                self._d_buffers[self._selection ^ 1].unsafe_value(),
                self._size,
                owning=False,
            )
        else:
            return DeviceBuffer[Self.dtype].empty(ctx)

    @always_inline
    def swap(mut self):
        self._selection ^= 1


@always_inline
def run_radix_sort_pairs_gpu[
    dtype: DType,
    out_idx_type: DType,
    ascending: Bool = False,
    BLOCK_SIZE: Int = 256,  # found empirically
    NUM_BITS_PER_PASS: Int = 4,
](
    ctx: DeviceContext,
    mut keys: DoubleBuffer[dtype, ...],
    mut key_ids: DoubleBuffer[out_idx_type, ...],
    skip_sort: UnsafePointer[mut=True, Scalar[.bool], _],
    in_shape: IndexList,
) raises:
    """
    Runs a multi-pass radix sort on key/index pairs across batches on the GPU
    using double buffering.

    Parameters:
        dtype: DType - Data type of the keys to sort.
        out_idx_type: DType - Data type of the output indices.
        ascending: Bool - Whether to sort in ascending order (default is descending).
        BLOCK_SIZE: Int - Number of threads per block (default 256, found empirically).
        NUM_BITS_PER_PASS: Int - Number of radix bits processed per pass (default 4).

    Args:
        ctx: DeviceContext - The GPU device context for enqueuing kernels.
        keys: DoubleBuffer[dtype] - Double buffer holding the keys to sort, swapped each pass.
        key_ids: DoubleBuffer[out_idx_type] - Double buffer holding the key indices, swapped each pass.
        skip_sort: Pointer[Scalar[.bool]] - Per-batch flag indicating whether sorting is skipped.
        in_shape: IndexList - Shape of the input tensor as [batch_size, vocab_size].
    """
    var batch_size = in_shape[0]
    var vocab_size = in_shape[1]

    var skip_sort_device = DeviceBuffer[.bool](
        ctx,
        skip_sort,
        batch_size,
        owning=False,
    )

    comptime for current_bit in range(
        0, bit_width_of[dtype](), NUM_BITS_PER_PASS
    ):
        comptime kernel = radix_sort_pairs_kernel[
            dtype, out_idx_type, current_bit, ascending, BLOCK_SIZE
        ]

        ctx.enqueue_function[kernel](
            keys.current(ctx),
            keys.alternate(ctx),
            key_ids.current(ctx),
            key_ids.alternate(ctx),
            Int32(vocab_size),
            skip_sort_device,
            grid_dim=Dim(batch_size),
            block_dim=Dim(BLOCK_SIZE),
        )
        keys.swap()
        key_ids.swap()


@always_inline
@__name(t"topp_minp_sampling_{dtype}_{out_idx_type}_{is_top_p}")
def topp_minp_sampling_kernel[
    dtype: DType,
    out_idx_type: DType,
    is_top_p: Bool,
](
    p_thresholds_: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    sorted_probs_: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    sorted_ids_: UnsafePointer[Scalar[out_idx_type], MutUntrackedOrigin],
    out_token_ids: UnsafePointer[Scalar[out_idx_type], MutUntrackedOrigin],
    skip_sort: UnsafePointer[Scalar[.bool], MutUntrackedOrigin],
    vocab_size: Int32,
):
    """
    Top P-Min P sampling kernel.

    Parameters:
        dtype: DType - scalar values dtype.
        out_idx_type: DType - output index type.
        is_top_p: Bool - Whether to use Top-P (True) or Min-P (False) sampling.
    Args:
        p_thresholds_: Top p or min-p calculated thresholds for each batch.
        sorted_probs_: Sorted probabilities in descending order.
        sorted_ids_: Sorted token ids in descending order.
        out_token_ids: Output token ids.
        skip_sort: Whether sorting was skipped for this batch.
        vocab_size: Number of tokens in the vocabulary per batch, used to
            iterate over the sorted probability and id buffers.
    """
    var _vocab_size = Int(vocab_size)
    var tid = thread_idx.x
    var batch_id = block_idx.x

    if skip_sort[batch_id]:
        # out_token_ids is already set by topk_wrapper
        return

    var p_threshold = p_thresholds_[batch_id]
    var sorted_probs = sorted_probs_ + batch_id * _vocab_size
    var sorted_ids = sorted_ids_ + batch_id * _vocab_size

    comptime if is_top_p:
        if tid == 0:
            var rng_state = Random(seed=SEED)
            var rng = rng_state.step_uniform()
            var r = p_threshold * rng[0].cast[dtype]()
            for i in range(_vocab_size):
                r -= sorted_probs[i]

                if r <= 0.0 or i == _vocab_size - 1:
                    comptime if DEBUG_FILE:
                        print("sorted_probs[i]: ", sorted_probs[i])
                        print("r: ", r)
                        print("p_threshold: ", p_threshold)

                    out_token_ids[batch_id] = sorted_ids[i]
                    break
    else:
        # Min-P sampling
        if tid == 0:
            var rng_state = Random(seed=SEED)
            var rng = rng_state.step_uniform()

            # Step 1: Filter out tokens with probabilities less than the min-p threshold
            var sum_filtered_probs = Scalar[dtype](0.0)
            var num_filtered_tokens = 0
            for i in range(_vocab_size):
                if sorted_probs[i] >= p_threshold:
                    sum_filtered_probs += sorted_probs[i]
                    num_filtered_tokens += 1
                else:
                    break

            # Step 2: Sample from normalized distribution of remaining tokens
            var r = sum_filtered_probs * rng[0].cast[dtype]()
            # Step 3: Select token based on normalized probabilities
            for i in range(num_filtered_tokens):
                r -= sorted_probs[i]

                if r <= 0.0 or i == _vocab_size - 1:
                    out_token_ids[batch_id] = sorted_ids[i]

                    comptime if DEBUG_FILE:
                        print("sorted_probs[i]: ", sorted_probs[i])
                        print("r: ", r)
                        print("p_threshold: ", p_threshold)
                    break


@always_inline
def _is_supported_dtype[dtype: DType]() -> Bool:
    """
    Check if the type is supported by the radix sort kernel.
    If not supported, need to add a normalize function for that
    numeric type.
    """
    if dtype in (DType.bfloat16, DType.float32):
        return True
    if dtype in (DType.uint16, DType.uint32, DType.int32):
        return True
    return False


@always_inline
def _topp_minp_sampling_gpu[
    dtype: DType,
    out_idx_type: DType,
    //,
    is_top_p: Bool,
    _test_sort: Bool = False,
](
    ctx: DeviceContext,
    p_thresholds: TileTensor[dtype, ...],
    input_logits: TileTensor[dtype, address_space=.GENERIC, ...],
    out_token_ids: TileTensor[
        mut=True, out_idx_type, address_space=.GENERIC, ...
    ],
    temperature: Scalar[dtype] = 1,
) raises:
    """
    GPU implementation of Top-P (nucleus) and Min-P sampling for token selection.
    This function applies temperature scaling, softmax, a radix sort, and then samples tokens
    based on either the cumulative probability mass (Top-P) or calculated probability threshold (Min-P).
    Token sampling algorithm details: https://www.notion.so/modularai/Token-sampler-1081044d37bb80c39932d6be9a4215d5


    Parameters:
        dtype: DType - The data type of the input logits, p_thresholds, and temperature.
        out_idx_type: DType - The data type for output token indices.
        is_top_p: Bool - Whether to use Top-P (True) or Min-P (False) sampling. If Min-P, the
            p_thresholds are used as min-p coefficients that determine the minimum probability
            threshold for token inclusion.
        _test_sort: Bool - For internal testing purposes to check if the
            sorted probs are in descending order.
    Args:
        ctx: DeviceContext
            The context for GPU execution.
        p_thresholds: TileTensor[type]
            Batch of p values (thresholds) for Top-P/Min-P sampling.
            For Top-P: cumulative probability threshold (e.g., 0.9 means sample from top 90%).
            For Min-P: min-p coefficients that determine the minimum probability threshold.
        input_logits: TileTensor[type]
            Input logits tensor of shape [batch_size, vocab_size].
        out_token_ids: TileTensor[out_idx_type]
            Output buffer for sampled token indices of shape [batch_size, 1].
        temperature: Scalar[type]
            Temperature for softmax scaling of logits (default=1.0).
            Higher values increase diversity, lower values make sampling more deterministic.

    The implementation follows these steps:
    1. Apply temperature scaling to the input logits
    2. Convert logits to probabilities using softmax
    3. Sort probability/index pairs in descending order
    4. For each sequence in the batch:
        - For Top-P: Sample from tokens that sum to the p_threshold of probability mass
        - For Min-P: Sample from tokens that exceed the minimum probability threshold
    5. Output the selected token indices

    Based on sampling implementations from:
    - TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/samplingTopPKernels.cu#L199-L323
    - InternLM: https://github.com/InternLM/lmdeploy/
    """
    comptime assert p_thresholds.rank == 1, "p_thresholds must be rank 1"
    comptime assert (
        input_logits.rank == 2 and out_token_ids.rank == 2
    ), "Only rank 2 tensors are supported"
    comptime assert _is_supported_dtype[dtype](), String(
        "Unsupported dtype: ", dtype
    )

    comptime BLOCK_SIZE = WARP_SIZE if has_apple_gpu_accelerator() else 256

    # Step 1; Apply temperature scaling to the logits and apply
    # softmax to get probabilities
    var input_shape = IndexList[input_logits.rank](
        Int(input_logits.dim[0]()), Int(input_logits.dim[1]())
    )
    var batch_size = input_shape[0]
    var vocab_size = input_shape[1]

    @__parameter
    @__copy_capture(input_logits)
    def apply_temperature[
        _simd_width: Int
    ](coords: Coord) -> SIMD[dtype, _simd_width]:
        var val = input_logits.load[width=_simd_width](coords)
        return val / temperature

    var input_size = input_logits.num_elements()
    # TODO: Should softmax be done in-place without needing this other buffer?
    var probs_buf = ctx.enqueue_create_buffer[dtype](input_size * 2)
    var input_probs = TileTensor(
        probs_buf,
        row_major(batch_size, vocab_size),
    )

    _softmax_gpu[
        dtype,
        1,
        input_logits.rank,
        apply_temperature,
    ](Coord(input_shape), input_probs, input_logits.rank - 1, ctx)

    # Step 2: Do a Top K=1 search on each vocab_size row of the
    #   probabilities tensor. This is to check if the most probable
    #   token exceeds P. If it does, we skip sorting by setting
    #   begin_offset_buf[bi] = offset_buf[bi]
    # materialize a vals buffer
    var max_vals = ctx.enqueue_create_buffer[dtype](batch_size)
    var skip_sort = ctx.enqueue_create_buffer[.bool](batch_size)

    comptime K = 1
    comptime num_blocks_per_input = 1
    comptime topk_kernel = topk_wrapper[
        input_type=dtype,
        index_type=out_idx_type,
        is_top_p=is_top_p,
        block_size=BLOCK_SIZE,
        _test_sort=_test_sort,
    ]

    ctx.enqueue_function[topk_kernel](
        Int32(K),
        Int32(vocab_size),
        Int32(num_blocks_per_input),
        probs_buf,
        max_vals,
        out_token_ids.to_device_buffer(ctx),
        p_thresholds.to_device_buffer(ctx),
        skip_sort,
        grid_dim=batch_size,
        block_dim=BLOCK_SIZE,
    )

    # Step 3: Apply a global sort on the input tensor of probs
    # Create the input_ids buffer
    var ids_buf = ctx.enqueue_create_buffer[out_idx_type](input_size * 2)
    var probs_double_buffer = DoubleBuffer(
        probs_buf.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        probs_buf.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        + input_size,
        input_size,
    )
    var keys_double_buffer = DoubleBuffer(
        ids_buf.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        ids_buf.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        + input_size,
        input_size,
    )

    run_radix_sort_pairs_gpu[BLOCK_SIZE=BLOCK_SIZE](
        ctx,
        probs_double_buffer,
        keys_double_buffer,
        skip_sort.unsafe_ptr(),
        input_shape,
    )

    comptime if _test_sort:
        # Copy output of sort & softmax back to original input tensor
        # for testing and debugging purposes
        ctx.enqueue_copy(
            # TODO: properly propagate mutability to input_logits
            input_logits.ptr.unsafe_mut_cast[True](),
            probs_buf.unsafe_ptr(),
            input_size,
        )

    # Step 4: Sample from the sorted probabilities by cumsumming
    comptime topp_minp_kernel = topp_minp_sampling_kernel[
        dtype, out_idx_type, is_top_p
    ]
    ctx.enqueue_function[topp_minp_kernel](
        p_thresholds.to_device_buffer(ctx),
        probs_buf,
        ids_buf,
        out_token_ids.to_device_buffer(ctx),
        skip_sort,
        Int32(vocab_size),
        grid_dim=Dim(batch_size),
        block_dim=Dim(BLOCK_SIZE),
    )
    _ = max_vals^
    _ = skip_sort^
    _ = probs_buf^
    _ = ids_buf^


@always_inline
def top_p_sampling_gpu[
    dtype: DType,
    out_idx_type: DType,
    //,
    _test_sort: Bool = False,
](
    ctx: DeviceContext,
    top_ps: TileTensor[dtype, ...],
    input_logits: TileTensor[dtype, address_space=.GENERIC, ...],
    out_token_ids: TileTensor[
        mut=True, out_idx_type, address_space=.GENERIC, ...
    ],
    temperature: Scalar[dtype] = 1,
) raises:
    """
    GPU implementation of Top-P sampling for token selection.
    This function applies temperature scaling, softmax, a radix sort, and then
    samples tokens based on the cumulative probability mass (Top-P).
    """
    # TODO: Implement rank generalization
    comptime assert top_ps.rank == 1, "top_ps must be of rank 1"
    comptime assert (
        input_logits.rank == 2 and out_token_ids.rank == 2
    ), "Only rank 2 tensors are supported"

    _topp_minp_sampling_gpu[is_top_p=True, _test_sort=_test_sort](
        ctx, top_ps, input_logits, out_token_ids, temperature
    )


@always_inline
def min_p_sampling_gpu[
    dtype: DType,
    out_idx_type: DType,
    //,
    _test_sort: Bool = False,
](
    ctx: DeviceContext,
    min_ps: TileTensor[dtype, address_space=.GENERIC, ...],
    input_logits: TileTensor[dtype, address_space=.GENERIC, ...],
    out_token_ids: TileTensor[
        mut=True, out_idx_type, address_space=.GENERIC, ...
    ],
    temperature: Scalar[dtype] = 1,
) raises:
    """
    GPU implementation of Min-P sampling for token selection.
    This function applies temperature scaling, softmax, a radix sort, and then
    samples tokens based on the calculated probability threshold (Min-P).
    """
    _topp_minp_sampling_gpu[is_top_p=False, _test_sort=_test_sort](
        ctx, min_ps, input_logits, out_token_ids, temperature
    )
