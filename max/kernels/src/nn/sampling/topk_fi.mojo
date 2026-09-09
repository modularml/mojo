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
"""Implements a float-integer packed top-K kernel that jointly tracks values and indices in a single register."""

from std.bit import log2_floor
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_dim,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from max.gpu.sync import barrier
from max.gpu.primitives import block
from max.gpu.primitives import warp
from max.gpu.primitives.grid_controls import (
    PDL,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
    PDLLevel,
)
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import external_memory
from std.sys.info import (
    has_amd_gpu_accelerator,
    has_apple_gpu_accelerator,
    is_amd_gpu,
    is_apple_gpu,
)
from layout import (
    ComptimeInt,
    Coord,
    Idx,
    DefaultEngine,
    TensorLayout,
    TensorEngine,
    TileTensor,
    coord_to_index_list,
    row_major,
)
from layout.tile_layout import Layout
from std.math import align_up, ceildiv, gcd, exp
from std.math.uutils import ufloordiv
from std.memory import unsafe_stack_allocation
from std.atomic import Atomic
from std.random import Random
from std.sys import align_of, simd_width_of, size_of
from max.runtime.tracing import Trace, TraceLevel, trace_arg
from std.utils.static_tuple import StaticTuple
from ..normalization import (
    _APPLE_STATIC_SHMEM_MAX_COUNT,
    _APPLE_STATIC_SHMEM_MAX_BYTES,
)
from .coop_row import (
    COOP_SLOT_FLOATS,
    CoopRow,
    coop_group_size,
    coop_row_words,
)

# Apple-only `topk_softmax_sample` cache budget. The kernel statically allocates
# `s_vals` (the top-k cache) plus auxiliary SMEM (`s_count` + block-reduction
# per-warp scratch). Allocating the full 32K bucket for the cache alone left no
# room for the auxiliary SMEM and overflowed Apple's 32K threadgroup limit
# (33932 > 32768). Reserve 2K of headroom for the auxiliary SMEM.
comptime _APPLE_STATIC_SHMEM_RESERVE_BYTES = 2 * 1024
comptime _APPLE_STATIC_SHMEM_CACHE_BYTES = (
    _APPLE_STATIC_SHMEM_MAX_BYTES - _APPLE_STATIC_SHMEM_RESERVE_BYTES
)
comptime _APPLE_STATIC_SHMEM_CACHE_COUNT = (
    _APPLE_STATIC_SHMEM_CACHE_BYTES // size_of[Float32]()
)


@always_inline
def _block_minmax[
    dtype: DType, //, *, block_size: Int, broadcast: Bool = True
](min_val: Scalar[dtype], max_val: Scalar[dtype]) -> Tuple[
    Scalar[dtype], Scalar[dtype]
]:
    """Fused block-level min and max reduction in a single barrier pass.
    Parameters:
        dtype: The data type of the values.
        block_size: The total number of threads in the block.
        broadcast: If True, broadcast results to all threads.
    Args:
        min_val: Thread-local value for the min reduction.
        max_val: Thread-local value for the max reduction.
    Returns:
        Tuple of (block_min, block_max).
    """

    @always_inline
    @__parameter
    def _reduce_fn[
        dtype: DType, width: SIMDLength, reduction_idx: Int
    ](v: SIMD[dtype, width]) -> Scalar[dtype]:
        comptime if reduction_idx == 0:
            return warp.min(v)
        else:
            return warp.max(v)

    var results = block._block_reduce[
        block_size,
        warp_reduce_fn=_reduce_fn,
        broadcast=broadcast,
    ](
        StaticTuple[Scalar[dtype], 2](min_val, max_val),
        initial_vals=StaticTuple[Scalar[dtype], 2](
            Scalar[dtype].MAX_FINITE, Scalar[dtype].MIN_FINITE
        ),
    )
    return (results[0], results[1])


@always_inline
def _block_reduce_pivot_bounds[
    block_size: Int, broadcast: Bool = True
](
    count0: Int32,
    count1: Int32,
    min_gt_low: Float32,
    max_le_high: Float32,
) -> Tuple[Int32, Int32, Float32, Float32]:
    """Fused block reduction for pivot-search loop: 2 sums + min + max.

    Performs all four reductions in a single 2-barrier pass by casting
    the Int32 counts to Float32 (exact for counts up to 2^23).
    """

    @always_inline
    @__parameter
    def _reduce_fn[
        dtype: DType, width: SIMDLength, reduction_idx: Int
    ](v: SIMD[dtype, width]) -> Scalar[dtype]:
        comptime if reduction_idx < 2:
            return warp.sum(v)
        elif reduction_idx == 2:
            return warp.min(v)
        else:
            return warp.max(v)

    var results = block._block_reduce[
        block_size,
        warp_reduce_fn=_reduce_fn,
        broadcast=broadcast,
    ](
        StaticTuple[Float32, 4](
            Float32(count0), Float32(count1), min_gt_low, max_le_high
        ),
        initial_vals=StaticTuple[Float32, 4](
            0, 0, Float32.MAX_FINITE, Float32.MIN_FINITE
        ),
    )
    return (
        Int32(results[0]),
        Int32(results[1]),
        results[2],
        results[3],
    )


@always_inline
def get_min_max_value[
    vec_size: Int,
    block_size: Int,
    dtype: DType,
](
    in_data: UnsafePointer[Scalar[dtype], _],
    row_idx: Int,
    d: Int,
) -> Tuple[
    Float32, Float32
]:
    """Compute the minimum and maximum values from input data using block reduction.

    Parameters:
        vec_size: Number of elements each thread processes per iteration (vectorization width).
        block_size: Number of threads per block.
        dtype: The dtype of the input data.

    Args:
        in_data: Pointer to input data buffer.
        row_idx: Row index for the current block (for 2D data access).
        d: Total number of elements in the row.

    Returns:
        Tuple containing [min_val, max_val].
    """
    var tx = thread_idx.x

    # Accumulate thread-local min/max across all chunks, then reduce once.
    var thread_max = Float32.MIN
    var thread_min = Float32.MAX

    var num_iterations = ceildiv(d, block_size * vec_size)
    for i in range(num_iterations):
        var in_data_vec = SIMD[.float32, vec_size](0)

        if (i * block_size + tx) * vec_size < d:
            var offset = row_idx * d + i * block_size * vec_size + tx * vec_size
            in_data_vec = in_data.load[width=vec_size](offset).cast[.float32]()

        thread_max = max(thread_max, in_data_vec.reduce_max())
        thread_min = min(thread_min, in_data_vec.reduce_min())

    var min_val, max_val = _block_minmax[block_size=block_size](
        thread_min, thread_max
    )

    return Tuple[Float32, Float32](min_val, max_val)


@__name(t"topk_mask_logits_{dtype}_{out_idx_type}")
def TopKMaskLogitsKernel[
    block_size: Int,
    vec_size: Int,
    dtype: DType,
    out_idx_type: DType,
    LogitsLayoutType: TensorLayout,
    logits_origin: ImmOrigin,
    MaskedLogitsLayoutType: TensorLayout,
    masked_logits_origin: MutOrigin,
](
    logits: TileTensor[dtype, LogitsLayoutType, logits_origin],
    masked_logits: TileTensor[
        dtype, MaskedLogitsLayoutType, masked_logits_origin
    ],
    top_k_arr: Optional[
        UnsafePointer[Scalar[out_idx_type], MutUntrackedOrigin]
    ],
    top_k_val: Int32,
    d: Int32,
):
    var _top_k_val = Int(top_k_val)
    var _d = Int(d)
    var bx = block_idx.x
    var tx = thread_idx.x
    var row_idx = bx

    var logits_ptr = logits.ptr + bx * _d
    var masked_logits_ptr = masked_logits.ptr + bx * _d

    var logits_row = TileTensor(logits_ptr, row_major(Idx[1], _d))
    var masked_logits_row = TileTensor(masked_logits_ptr, row_major(Idx[1], _d))

    with PDL():
        var k = _top_k_val
        if top_k_arr:
            k = Int(top_k_arr.unsafe_value()[bx])

        # Initialize pivot to negative infinity.
        var pivot = Float32.MIN

        var logits_vec = SIMD[.float32, vec_size]()

        if k < _d:
            var min_max = get_min_max_value[vec_size, block_size](
                logits.ptr, row_idx, _d
            )
            var min_val, max_val = min_max[0], min_max[1]

            # Initialize ternary search bounds.
            var low = Float32(
                min_val - 1 if min_val != Float32.MIN else Float32.MIN
            )
            var high = max_val

            while True:
                var pivot_0 = (high + 2 * low) / 3
                var pivot_1 = (2 * high + low) / 3

                # Accumulate thread-local counts across all chunks.
                var thread_count_0_total: Int32 = 0
                var thread_count_1_total: Int32 = 0
                var min_gt_low = high
                var max_le_high = low

                for i in range(ceildiv(_d, block_size * vec_size)):
                    if (i * block_size + tx) * vec_size < _d:
                        logits_vec = logits_row.load[width=vec_size](
                            (
                                Idx[0],
                                i * block_size * vec_size + tx * vec_size,
                            ),
                        ).cast[.float32]()

                    var probs_gt_pivot_0_count = SIMD[.int32, vec_size]()
                    var probs_gt_pivot_1_count = SIMD[.int32, vec_size]()

                    comptime for j in range(vec_size):
                        # Calculate the global index for this element in the row.
                        # Will only count if the index is within the valid range [0, _d).
                        var idx = (i * block_size + tx) * vec_size + j

                        # Count elements greater than pivot_0 (higher ternary search bound).
                        probs_gt_pivot_0_count[j] = Int32(1) if (
                            logits_vec[j] > pivot_0 and idx < _d
                        ) else Int32(0)
                        # Count elements greater than pivot_1 (lower ternary search bound).
                        probs_gt_pivot_1_count[j] = Int32(1) if (
                            logits_vec[j] > pivot_1 and idx < _d
                        ) else Int32(0)

                        # Track the minimum value that's greater than 'low'.
                        # Used to narrow the search range from below.
                        if logits_vec[j] > low and idx < _d:
                            min_gt_low = min(min_gt_low, logits_vec[j])
                        # Track the maximum value that's less than or equal to 'high'.
                        # Used to narrow the search range from above.
                        if logits_vec[j] <= high and idx < _d:
                            max_le_high = max(max_le_high, logits_vec[j])

                    # Accumulate thread-local counts (no block reduction per chunk).
                    thread_count_0_total += probs_gt_pivot_0_count.reduce_add()
                    thread_count_1_total += probs_gt_pivot_1_count.reduce_add()

                # Single block reduction after processing all chunks.
                var _pivot_results = _block_reduce_pivot_bounds[block_size](
                    thread_count_0_total,
                    thread_count_1_total,
                    min_gt_low,
                    max_le_high,
                )
                var aggregate_gt_pivot_0 = _pivot_results[0]
                var aggregate_gt_pivot_1 = _pivot_results[1]
                min_gt_low = _pivot_results[2]
                max_le_high = _pivot_results[3]

                # Update the search bounds based on the counts and the minimum/maximum values.
                if aggregate_gt_pivot_1 >= Int32(k):
                    low = pivot_1
                elif aggregate_gt_pivot_0 >= Int32(k):
                    low = pivot_0
                    high = min(pivot_1, max_le_high)
                else:
                    high = min(pivot_0, max_le_high)

                if min_gt_low == max_le_high:
                    break

            pivot = low

        for i in range(ceildiv(_d, block_size * vec_size)):
            logits_vec = 0
            if (i * block_size + tx) * vec_size < _d:
                logits_vec = logits_row.load[width=vec_size](
                    (
                        Idx[0],
                        i * block_size * vec_size + tx * vec_size,
                    )
                ).cast[.float32]()

            logits_vec = (logits_vec.gt(pivot)).select(logits_vec, Float32.MIN)

            if (i * block_size + tx) * vec_size < _d:
                masked_logits_row.store[width=vec_size](
                    (
                        Idx[0],
                        i * block_size * vec_size + tx * vec_size,
                    ),
                    logits_vec.cast[dtype](),
                )


def topk_mask_logits[
    dtype: DType,
    out_idx_type: DType,
    block_size: Int = 1024,
    TopKArrLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
](
    ctx: DeviceContext,
    logits: TileTensor[mut=False, dtype, ...],
    masked_logits: TileTensor[mut=True, dtype, ...],
    top_k_val: Int,
    top_k_arr: Optional[
        TileTensor[out_idx_type, TopKArrLayoutType, MutUntrackedOrigin]
    ] = None,
) raises:
    """Masks logits to keep only the top-k largest values per row.

    Launches `TopKMaskLogitsKernel` with one block per batch row. Elements below
    the k-th largest logit are set to the dtype's minimum value so downstream
    sampling ignores them.

    Parameters:
        dtype: Element type of the `logits` and `masked_logits` tensors.
        out_idx_type: Index type used for per-row top-k override values in
            `top_k_arr`.
        block_size: Number of threads per block (defaults to 1024).
        TopKArrLayoutType: Memory layout of the optional `top_k_arr` tensor.

    Args:
        ctx: Device context for kernel execution.
        logits: Input logits tensor [batch_size, d].
        masked_logits: Output buffer for masked logits, same shape as logits.
        top_k_val: Default number of largest logits to retain per row.
        top_k_arr: Optional per-row top-k values that override top_k_val.

    Raises:
        Error: If masked_logits shape does not match logits shape.
    """
    comptime assert logits.rank == 2, "logits rank must be 2"
    comptime assert (
        logits.rank == masked_logits.rank
    ), "logits.rank must match masked_logits.rank"

    var shape = coord_to_index_list(logits.layout.shape_coord())
    var batch_size = shape[0]
    var d = shape[1]

    def trace_information() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("logits", shape, dtype),
                    "top_k_val=" + String(top_k_val),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        "topk_mask_logits",
        Trace[TraceLevel.OP]._get_detail_str(trace_information),
        task_id=Int(ctx.id()),
    ):
        var out_shape = coord_to_index_list(masked_logits.layout.shape_coord())
        if shape[0] != out_shape[0] or shape[1] != out_shape[1]:
            raise Error("masked_logits shape must match logits shape")

        # Use up to 16 elements per vector to minimize the number of chunks
        # (and therefore the number of block-level reductions in inner loops).
        # GPU vector loads handle wider-than-native SIMD efficiently, and the
        # per-element idx < d guard handles non-aligned tails correctly.
        var vec_size = gcd(8, d)

        var top_k_ptr: Optional[
            UnsafePointer[Scalar[out_idx_type], MutUntrackedOrigin]
        ] = None
        if top_k_arr:
            top_k_ptr = top_k_arr.value().ptr

        @__parameter
        def launch_kernel[vec_size: Int]() raises:
            comptime kernel = TopKMaskLogitsKernel[
                block_size,
                vec_size,
                dtype,
                out_idx_type,
                LogitsLayoutType=logits.LayoutType,
                logits_origin=ImmOrigin(logits.origin),
                MaskedLogitsLayoutType=masked_logits.LayoutType,
                masked_logits_origin=masked_logits.origin,
            ]
            ctx.enqueue_function[kernel](
                logits.as_immut(),
                masked_logits,
                top_k_ptr,
                Int32(top_k_val),
                Int32(d),
                grid_dim=batch_size,
                block_dim=block_size,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )

        # Runtime dispatch to compile-time parameter.
        comptime for param_vec_size in [16, 8, 4, 2, 1]:
            if vec_size == param_vec_size:
                return launch_kernel[param_vec_size]()


@always_inline
def device_sampling_from_prob[
    vec_size: Int,
    block_size: Int,
    dtype: DType,
    deterministic: Bool = False,
](
    i: Int,
    d: Int,
    low: Float32,
    u: Float32,
    prob_vec: SIMD[.float32, vec_size],
    aggregate: Float32,
    sampled_id_sram: UnsafePointer[mut=True, Int, _, address_space=.SHARED],
) -> Tuple[Float32, Int]:
    """Device-level sampling from probability distribution with atomic operations.

    Parameters:
        vec_size: Number of elements each thread loads per vectorized
            access.
        block_size: Number of threads per block.
        dtype: Element type of the probability distribution.
        deterministic: If True, use deterministic sampling (defaults to
            False).

    Args:
        i: Chunk iteration index used to compute element offsets within
            the row.
        d: Total number of elements in the row (vocabulary size).
        low: Lower-bound threshold; only probabilities greater than this
            value participate in sampling.
        u: Target cumulative probability to locate, scaled by the
            remaining mass `q`.
        prob_vec: Vector of probabilities for the current chunk.
        aggregate: Running sum of filtered probabilities from previously
            processed chunks.
        sampled_id_sram: Shared-memory slot holding the sampled index,
            updated via atomic minimum.

    Returns:
        Tuple of (new_aggregate, thread_local_max_valid_idx).
        The caller is responsible for reducing max_valid_idx across the block
        after all chunks are processed.
    """

    var tx = thread_idx.x

    # Step 1: Filter probabilities based on predicate (prob > low).
    var prob_gt_threshold = SIMD[.float32, vec_size]()
    var valid = SIMD[.bool, vec_size]()

    comptime for j in range(vec_size):
        var idx = (i * block_size + tx) * vec_size + j
        var passes_pred = prob_vec[j] > low
        prob_gt_threshold[j] = prob_vec[j] if passes_pred else 0.0
        valid[j] = passes_pred and (idx < d)

    # Step 2: Block reduce to get sum of filtered probabilities.
    var thread_sum = prob_gt_threshold.reduce_add()

    var aggregate_local = block.sum[
        block_size=block_size,
        broadcast=True,
    ](thread_sum)

    # Step 3: Check if we found the sampled index in this chunk.
    if aggregate + aggregate_local > u:
        # Step 4: Thread-local prefix sum.
        # Intra-SIMD prefix sum using shift operations.
        var local_inclusive_cdf = prob_gt_threshold  # Start with the values

        comptime for i in range(log2_floor(vec_size)):
            # Shift right by 2^i positions (filling with zeros)
            # and add to accumulate prefix sums.
            local_inclusive_cdf += local_inclusive_cdf.shift_right[2**i]()

        # Step 5: Block-level exclusive scan.
        var thread_total = local_inclusive_cdf[vec_size - 1]
        var prefix_from_prev_threads = block.prefix_sum[
            dtype=.float32,
            block_size=block_size,
            exclusive=True,
        ](thread_total)

        # Step 6: Compute global inclusive CDF.
        var global_inclusive_cdf = (
            local_inclusive_cdf + prefix_from_prev_threads
        )

        # Step 7: Find first index where cumulative > u using atomic min.
        comptime for j in range(vec_size):
            var idx = (i * block_size + tx) * vec_size + j
            if (global_inclusive_cdf[j] + aggregate > u) and valid[j]:
                # Atomic min to ensure we get the smallest index across all threads.
                Atomic.min(sampled_id_sram.bitcast[Int32](), Int32(idx))
                break

        barrier()

    # Step 8: Compute thread-local max valid index (deferred to caller).
    var max_valid_idx = -1

    comptime for j in range(vec_size):
        var idx = (i * block_size + tx) * vec_size + j
        if valid[j]:
            max_valid_idx = idx

    return Tuple[Float32, Int](aggregate + aggregate_local, max_valid_idx)


struct ValueCount[T: DType](Defaultable, TrivialRegisterPassable):
    """A struct that holds a value and a count, used for block reductions.

    This is useful for computing both the sum of values and the count
    of elements that satisfy a condition in a single reduction pass.

    Parameters:
        T: The DType of the value field.
    """

    var value: Scalar[Self.T]
    var count: Int32

    def __init__(out self, value: Scalar[Self.T], count: Int32):
        # Initialize a ValueCount instance.
        self.value = value
        self.count = count

    def __init__(out self):
        # Zero-initialize a ValueCount instance.
        self.value = 0
        self.count = 0

    def __add__(self, other: Self) -> Self:
        # Add two ValueCount instances (element-wise).
        return {self.value + other.value, self.count + other.count}

    def __iadd__(mut self, other: Self):
        # In-place addition of another ValueCount.
        self.value += other.value
        self.count += other.count


@always_inline
def _warp_reduce_value_count[T: DType](val: ValueCount[T]) -> ValueCount[T]:
    """Warp-level reduction for ValueCount using shuffle operations.

    Reduces both value and count fields across all lanes in a warp.

    Parameters:
        T: DType of the value field.

    Args:
        val: The ValueCount from this thread's lane.

    Returns:
        ValueCount with both fields reduced across the warp (only valid in lane 0).
    """
    var result = val

    comptime limit = log2_floor(WARP_SIZE)

    # Reduce across warp lanes using shuffle_down.
    comptime for i in reversed(range(limit)):
        comptime offset = 1 << i
        result.value += warp.shuffle_down(result.value, UInt32(offset))
        result.count += warp.shuffle_down(result.count, UInt32(offset))
    return result


@always_inline
def _block_reduce_value_count[
    T: DType,
    broadcast: Bool = False,
](val: ValueCount[T]) -> ValueCount[T]:
    """Block-level reduction for ValueCount struct.

    Reduces both value and count fields across all threads in a block.

    Parameters:
        T: DType of the value field.
        broadcast: If True, all threads get the reduced result.
                   If False, only thread 0 has the correct result.

    Args:
        val: The ValueCount from this thread.

    Returns:
        ValueCount with both fields reduced across the entire block.
        If broadcast=True, all threads get the same result.
        If broadcast=False, only thread 0 has the valid result.
    """
    comptime MAX_BLOCK_SIZE = 1024
    comptime assert (
        MAX_BLOCK_SIZE % WARP_SIZE == 0
    ), "block size must be a multiple of the warp size"

    comptime value_width = simd_width_of[Scalar[T]]()
    comptime count_width = simd_width_of[DType.int32]()

    var value_sram = unsafe_stack_allocation[
        (MAX_BLOCK_SIZE // WARP_SIZE) * value_width,
        Scalar[T],
        address_space=.SHARED,
    ]()
    var count_sram = unsafe_stack_allocation[
        (MAX_BLOCK_SIZE // WARP_SIZE) * count_width,
        Int32,
        address_space=.SHARED,
    ]()

    var warp = warp_id()
    comptime num_warps_needed = MAX_BLOCK_SIZE // WARP_SIZE

    var warp_accum = _warp_reduce_value_count(val)

    # Store warp-level results in shared memory (only lane 0 of each warp).
    if lane_id() == 0 and warp < num_warps_needed:
        value_sram[warp * value_width] = warp_accum.value
        count_sram[warp * count_width] = warp_accum.count
    barrier()

    # Each warp has reduced its own ValueCount in smem (value_sram and count_sram).
    # Below we perform block-level reduction (across all warps) to get final result.
    # Only the first N threads from warp 0 will have valid results in the corresponding
    # smem slots above and participate in the final warp-level reduction (e.g. if
    # block_size = 1024 and WARP_SIZE = 32, then only the first 32 threads from warp 0
    # will have valid results).
    var block_accum: ValueCount[T]
    var thread_in_final_warp = thread_idx.x < ufloordiv(block_dim.x, WARP_SIZE)

    if thread_in_final_warp:
        block_accum = {
            value = value_sram[lane_id() * value_width],
            count = count_sram[lane_id() * count_width],
        }
    else:
        # Initialize unused threads with zeros (identity for sum).
        block_accum = {value = Scalar[T](0), count = 0}

    # Perform final warp-level reduction.
    var result = _warp_reduce_value_count(block_accum)

    comptime if broadcast:
        if thread_idx.x == 0:
            value_sram[0] = result.value
            count_sram[0] = result.count

        barrier()

        result = {
            value = value_sram[0],
            count = count_sram[0],
        }

    return result


@__name(
    t"topk_sampling_from_prob_{dtype}_{out_idx_type}_{deterministic}",
)
def TopKSamplingFromProbKernel[
    ProbsLayoutType: TensorLayout,
    probs_origin: ImmOrigin,
    OutputLayoutType: TensorLayout,
    output_origin: MutOrigin,
    block_size: Int,
    vec_size: Int,
    dtype: DType,
    out_idx_type: DType,
    deterministic: Bool,
](
    probs: TileTensor[dtype, ProbsLayoutType, probs_origin],
    output: TileTensor[out_idx_type, OutputLayoutType, output_origin],
    indices: Optional[UnsafePointer[Scalar[out_idx_type], MutUntrackedOrigin]],
    top_k_arr: Optional[
        UnsafePointer[Scalar[out_idx_type], MutUntrackedOrigin]
    ],
    top_k_val: Int32,
    d: Int32,
    rng_seed: UInt64,
    rng_offset: UInt64,
):
    """Kernel for top-k sampling from probability distribution.

    This kernel performs top-k sampling by:
    1. Using ternary search to find a pivot threshold.
    2. Rejecting samples iteratively until acceptance criteria is met.
    3. Sampling an index using uniform random numbers from Random generator.

    Parameters:
        ProbsLayoutType: Memory layout of the input `probs` tile.
        probs_origin: Origin tag for the immutable input `probs` tile.
        OutputLayoutType: Memory layout of the output `output` tile.
        output_origin: Origin tag for the mutable output `output` tile.
        block_size: Number of threads per block.
        vec_size: Number of elements each thread loads per vectorized
            access.
        dtype: Element type of the `probs` tensor.
        out_idx_type: Index type used for the sampled output indices.
        deterministic: If True, use deterministic sampling.

    Args:
        probs: Input probability distribution [batch_size, _d].
        output: Output sampled indices [batch_size].
        indices: Optional row indices for batch indexing [batch_size].
        top_k_arr: Optional per-row top_k values [batch_size].
        top_k_val: Default top_k value if top_k_arr is null.
        d: Vocabulary size.
        rng_seed: Random seed for Random number generator.
        rng_offset: Random offset for Random number generator.
    """
    var _top_k_val = Int(top_k_val)
    var _d = Int(d)
    comptime assert output.flat_rank == 1

    var bx = block_idx.x
    var tx = thread_idx.x

    with PDL():
        var generator = Random(seed=rng_seed, offset=UInt64(bx) + rng_offset)
        var k = _top_k_val
        if top_k_arr:
            k = Int(top_k_arr.unsafe_value().load(bx))
        var row_idx = bx
        if indices:
            row_idx = Int(indices.unsafe_value().load(bx))

        var probs_ptr = probs.ptr + row_idx * _d
        var probs_row = TileTensor(probs_ptr, row_major(Idx[1], _d))

        # The final sampled index, produced by whichever search path runs.
        var sampled_id = 0

        comptime if is_apple_gpu():
            # Single-thread-driven ternary search (Apple/Metal only).
            #
            # The block-collective `while low < high` loop (the non-Apple path
            # below) computes the Case decision (`count_0 < k`, `count_1 < k`)
            # PER THREAD from broadcast block reductions. On Metal, at the
            # ghost-warp geometry (block=1024 but only the first few warps carry
            # data), repeated/interleaved block collectives inside the loop —
            # specifically the `block.sum` + `block.prefix_sum` pair inside
            # `device_sampling_from_prob` — progressively DESYNCHRONIZE the
            # warps by a full loop iteration (verified: a single-cell publish +
            # barrier is uniform in isolation and stays uniform after
            # `block.sum`/`block.prefix_sum`/`block.max` individually, but BREAKS
            # once `device_sampling_from_prob` runs in the loop; ghost warps then
            # lag warp 0 by one iteration). With the warps desynced, `count_0`
            # differed per thread => different Case branches => divergent trip
            # count => in-body barriers on a partial threadgroup (UB),
            # compounding into out-of-top-K results. No per-iteration
            # publish/broadcast mechanism fixes this because the warps are
            # already drifting.
            #
            # The robust structural fix: run the ENTIRE search on a single
            # thread (tx==0) with sequential scans over the row — NO block
            # collectives in the loop — and publish only the final id. A
            # FIXED-BOUND `for` loop keeps every thread executing the same
            # iteration count; the other threads merely hit the per-iteration
            # `barrier()` and advance the RNG in lockstep. With no in-loop
            # collective there is nothing to desynchronize, so the search is
            # correct by construction. The vocab is one row per block and K is
            # small, so the sequential O(_d) scans per iteration are acceptable
            # for a sampler.
            #
            # MAX_ITERS bound: the ternary search strictly narrows [low, high]
            # each rejected iteration and converges in <=4 iterations in the
            # host replay. 64 is a large safety margin; the `done` flag no-ops
            # the rest.
            comptime MAX_ITERS = 64

            var done_sram = unsafe_stack_allocation[
                1, Int32, address_space=.SHARED
            ]()
            var out_id_sram = unsafe_stack_allocation[
                1, Int, address_space=.SHARED
            ]()

            # Initialize control once, uniformly.
            if tx == 0:
                done_sram[0] = 0
                out_id_sram[0] = 0
            barrier()

            # The ENTIRE ternary search runs on tx==0 with sequential scans (no
            # block collectives), then publishes only the final id. Every other
            # thread just participates in the per-iteration barrier so the loop
            # stays a uniform fixed-bound `for`. The non-tx0 threads advance the
            # SAME RNG stream so `u` stays identical, but only tx==0's draw is
            # used.
            var low: Float32 = 0.0
            var high: Float32 = 1.0
            var q: Float32 = 1.0

            for _it in range(MAX_ITERS):
                var done = done_sram[0] != 0
                var u = generator.step_uniform()[0] * q

                if tx == 0 and not done:
                    # Sequential CDF sample over the row: first index whose
                    # inclusive CDF (restricted to prob > low) exceeds u. Falls
                    # back to the last valid (prob > low) index if the mass is
                    # smaller than u (u very close to 1).
                    var cum: Float32 = 0.0
                    var search_id = _d
                    var last_valid_id = 0
                    for j in range(_d):
                        var pv = Float32(probs_row.load[width=1]((Idx[0], j)))
                        if pv > low:
                            last_valid_id = j
                            cum += pv
                            if cum > u and search_id == _d:
                                search_id = j
                    if search_id == _d:
                        search_id = last_valid_id

                    var pivot_0 = Float32(
                        probs_row.load[width=1]((Idx[0], search_id))
                    )
                    var pivot_1 = (pivot_0 + high) / 2.0

                    # Sequential counts of #{prob > pivot} and their prob mass.
                    var count_0: Int = 0
                    var value_0: Float32 = 0.0
                    var count_1: Int = 0
                    var value_1: Float32 = 0.0
                    for j in range(_d):
                        var pv = Float32(probs_row.load[width=1]((Idx[0], j)))
                        if pv > pivot_0:
                            count_0 += 1
                            value_0 += pv
                        if pv > pivot_1:
                            count_1 += 1
                            value_1 += pv

                    if count_0 < k:
                        # Case 1: pivot_0 accepted - found acceptable threshold.
                        out_id_sram[0] = search_id
                        done_sram[0] = 1
                    elif count_1 < k:
                        # Case 2: pivot_0 rejected, pivot_1 accepted.
                        low = pivot_0
                        high = pivot_1
                        q = value_0
                    else:
                        # Case 3: both pivots rejected.
                        low = pivot_1
                        q = value_1

                    # Bracket collapse: emit the current candidate as a fallback.
                    if low >= high:
                        out_id_sram[0] = search_id
                        done_sram[0] = 1
                barrier()

            sampled_id = out_id_sram[0]
        else:
            var sampled_id_sram = unsafe_stack_allocation[
                1, Int, address_space=.SHARED
            ]()
            var last_valid_id_sram = unsafe_stack_allocation[
                1, Int, address_space=.SHARED
            ]()

            var probs_vec: SIMD[.float32, vec_size]
            var aggregate: Float32
            var q: Float32 = 1.0
            var low: Float32 = 0.0
            var high: Float32 = 1.0

            # Seeded once, because the slot deliberately persists across
            # iterations as the "last index above `low`" fallback. It is
            # written only when some element qualifies, so without this seed a
            # row that is degenerate on the FIRST iteration reads whatever the
            # previous workgroup left in shared memory and uses it as an index.
            if tx == 0:
                last_valid_id_sram[0] = -1
            barrier()

            while low < high:
                if tx == 0:
                    sampled_id_sram[0] = _d
                barrier()

                var u = generator.step_uniform()[0] * q
                aggregate = 0.0
                var thread_max_valid = -1

                for i in range(ceildiv(_d, block_size * vec_size)):
                    probs_vec = 0
                    if (i * block_size + tx) * vec_size < _d:
                        probs_vec = probs_row.load[width=vec_size](
                            (Idx[0], ((i * block_size + tx) * vec_size))
                        ).cast[.float32]()

                    var result = device_sampling_from_prob[
                        vec_size, block_size, dtype, deterministic
                    ](
                        i,
                        _d,
                        low,
                        u,
                        probs_vec,
                        aggregate,
                        sampled_id_sram,
                    )
                    aggregate = result[0]
                    thread_max_valid = max(thread_max_valid, result[1])
                    if aggregate > u:
                        break

                # Reduce last_valid_id across block (single reduction after loop).
                var block_max_valid = block.max[
                    block_size=block_size,
                    broadcast=False,
                ](Int32(thread_max_valid))

                if tx == 0 and block_max_valid != -1:
                    last_valid_id_sram[0] = Int(block_max_valid)

                barrier()

                sampled_id = sampled_id_sram[0]
                if sampled_id == _d:
                    # This would happen when u is very close to 1 and the
                    # sum of probabilities is smaller than u. In this case
                    # we use the last valid index as the sampled id.
                    sampled_id = last_valid_id_sram[0]

                if sampled_id < 0:
                    # Degenerate row: nothing ever exceeded `low`, so there is
                    # no candidate to sample and the bracket cannot narrow
                    # (`low` would stay put and this would spin). Emit an
                    # in-range index and stop.
                    sampled_id = 0
                    break

                var pivot_0 = Float32(
                    probs_row.load[width=1]((Idx[0], sampled_id))
                )
                var pivot_1 = (pivot_0 + high) / 2.0

                # Accumulate thread-local value counts across all chunks.
                var thread_vc_0_total = ValueCount[.float32](0.0, 0)
                var thread_vc_1_total = ValueCount[.float32](0.0, 0)

                for i in range(ceildiv(_d, block_size * vec_size)):
                    probs_vec = 0
                    if (i * block_size + tx) * vec_size < _d:
                        probs_vec = probs_row.load[width=vec_size](
                            (Idx[0], ((i * block_size + tx) * vec_size))
                        ).cast[.float32]()

                    var probs_gt_pivot_0_values = SIMD[.float32, vec_size]()
                    var probs_gt_pivot_0_counts = SIMD[.int32, vec_size]()
                    var probs_gt_pivot_1_values = SIMD[.float32, vec_size]()
                    var probs_gt_pivot_1_counts = SIMD[.int32, vec_size]()

                    comptime for j in range(vec_size):
                        var idx = (i * block_size + tx) * vec_size + j
                        var is_valid = idx < _d

                        # For pivot_0.
                        var gt_pivot_0 = probs_vec[j] > pivot_0
                        probs_gt_pivot_0_values[j] = probs_vec[
                            j
                        ] if gt_pivot_0 else 0.0
                        probs_gt_pivot_0_counts[j] = Int32(1) if (
                            gt_pivot_0 and is_valid
                        ) else Int32(0)

                        # For pivot_1.
                        var gt_pivot_1 = probs_vec[j] > pivot_1
                        probs_gt_pivot_1_values[j] = probs_vec[
                            j
                        ] if gt_pivot_1 else 0.0
                        probs_gt_pivot_1_counts[j] = Int32(1) if (
                            gt_pivot_1 and is_valid
                        ) else Int32(0)

                    # Accumulate thread-local (no block reduction per chunk).
                    thread_vc_0_total += ValueCount[.float32](
                        probs_gt_pivot_0_values.reduce_add(),
                        probs_gt_pivot_0_counts.reduce_add(),
                    )
                    thread_vc_1_total += ValueCount[.float32](
                        probs_gt_pivot_1_values.reduce_add(),
                        probs_gt_pivot_1_counts.reduce_add(),
                    )

                # Reduce pivot_0 first; defer pivot_1 until needed.
                # For small K, acceptance (count_0 < k) is common, saving
                # the pivot_1 reduction (2 barriers) on the fast path.
                var aggregate_gt_pivot_0 = _block_reduce_value_count[
                    .float32, broadcast=True
                ](thread_vc_0_total)

                if aggregate_gt_pivot_0.count < Int32(k):
                    # Case 1: pivot_0 accepted - found acceptable threshold.
                    break

                # Only reduce pivot_1 when pivot_0 is rejected.
                var aggregate_gt_pivot_1 = _block_reduce_value_count[
                    .float32, broadcast=True
                ](thread_vc_1_total)

                if aggregate_gt_pivot_1.count < Int32(k):
                    # Case 2: pivot_0 rejected, pivot_1 accepted.
                    low = pivot_0
                    high = pivot_1
                    q = aggregate_gt_pivot_0.value
                else:
                    # Case 3: both pivots rejected.
                    low = pivot_1
                    q = aggregate_gt_pivot_1.value

            barrier()

        if tx == 0:
            output[bx] = Scalar[out_idx_type](sampled_id)


def topk_sampling_from_prob[
    dtype: DType,
    out_idx_type: DType,
    block_size: Int = 1024,
    TopKArrLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    IndicesLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
](
    ctx: DeviceContext,
    probs: TileTensor[mut=False, dtype, ...],
    output: TileTensor[mut=True, out_idx_type, ...],
    top_k_val: Int,
    deterministic: Bool = False,
    rng_seed: UInt64 = 0,
    rng_offset: UInt64 = 0,
    indices: Optional[
        TileTensor[out_idx_type, IndicesLayoutType, MutUntrackedOrigin]
    ] = None,
    top_k_arr: Optional[
        TileTensor[out_idx_type, TopKArrLayoutType, MutUntrackedOrigin]
    ] = None,
) raises:
    """Top-K sampling from probability distribution.

    Performs stochastic sampling from a probability distribution, considering only
    the top-k most probable tokens. Uses rejection sampling with ternary search
    to efficiently find appropriate samples.

    Parameters:
        dtype: Element type of the `probs` tensor.
        out_idx_type: Index type used for the sampled output indices.
        block_size: Number of threads per block (defaults to 1024).
        TopKArrLayoutType: Memory layout of the optional `top_k_arr` tensor.
        IndicesLayoutType: Memory layout of the optional `indices` tensor.

    Args:
        ctx: Device context for kernel execution.
        probs: Input probability distribution [batch_size, d].
        output: Output sampled indices [batch_size].
        top_k_val: Default top-k value (number of top tokens to consider).
        deterministic: Whether to use deterministic sampling.
        rng_seed: Random seed for Random number generator.
        rng_offset: Random offset for Random number generator.
        indices: Optional row indices for batch indexing [batch_size].
        top_k_arr: Optional per-row top-k values [batch_size].

    Raises:
        Error: If tensor ranks or shapes are invalid.
    """

    comptime assert probs.rank == 2, "probs rank must be 2"
    comptime assert output.rank == 1, "output rank must be 1"

    var shape = coord_to_index_list(probs.layout.shape_coord())
    var batch_size = shape[0]
    var d = shape[1]

    def trace_information() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("probs", shape, dtype),
                    "top_k_val=" + String(top_k_val),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        "topk_sampling_from_prob",
        Trace[TraceLevel.OP]._get_detail_str(trace_information),
        task_id=Int(ctx.id()),
    ):
        var out_shape = coord_to_index_list(output.layout.shape_coord())
        if out_shape[0] != batch_size:
            raise Error("output batch size must match probs batch size")

        # Use up to 16 elements per vector to minimize the number of chunks
        # (and therefore the number of block-level reductions in inner loops).
        # GPU vector loads handle wider-than-native SIMD efficiently, and the
        # per-element idx < d guard handles non-aligned tails correctly.
        var vec_size = gcd(8, d)

        var indices_ptr: Optional[
            UnsafePointer[Scalar[out_idx_type], MutUntrackedOrigin]
        ] = None
        if indices:
            indices_ptr = indices.value().ptr

        var top_k_ptr: Optional[
            UnsafePointer[Scalar[out_idx_type], MutUntrackedOrigin]
        ] = None
        if top_k_arr:
            top_k_ptr = top_k_arr.value().ptr

        @__parameter
        def launch_kernel[vec_size: Int, deterministic: Bool]() raises:
            comptime kernel = TopKSamplingFromProbKernel[
                probs.LayoutType,
                ImmOrigin(probs.origin),
                output.LayoutType,
                output.origin,
                block_size,
                vec_size,
                dtype,
                out_idx_type,
                deterministic,
            ]
            ctx.enqueue_function[kernel](
                probs.as_immut(),
                output,
                indices_ptr,
                top_k_ptr,
                Int32(top_k_val),
                Int32(d),
                rng_seed,
                rng_offset,
                grid_dim=batch_size,
                block_dim=block_size,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )

        # Runtime dispatch to compile-time parameter.
        @__parameter
        def dispatch_vec_size[deterministic: Bool]() raises:
            comptime for param_vec_size in [16, 8, 4, 2, 1]:
                if vec_size == param_vec_size:
                    return launch_kernel[param_vec_size, deterministic]()

        # Dispatch on deterministic flag.
        if deterministic:
            dispatch_vec_size[True]()
        else:
            dispatch_vec_size[False]()


@__name(t"apply_min_p_mask_{dtype}_{block_size}")
def apply_min_p_mask_kernel[
    dtype: DType,
    block_size: Int,
](
    probs: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    min_p_arr: UnsafePointer[Float32, ImmUntrackedOrigin],
    d: Int32,
):
    """Zero out probabilities below the per-row min_p threshold.

    Each block processes one batch row. Threads cooperatively find the
    row-wise max probability via a block reduction, compute the threshold
    as ``min_p * max_prob``, and then zero any element below it.

    Parameters:
        dtype: Element type of the `probs` buffer.
        block_size: Number of threads per block.

    Args:
        probs: Probability buffer [batch_size * _d], modified in-place.
        min_p_arr: Per-row min_p values [batch_size].
        d: Vocabulary size (row length).
    """
    var _d = Int(d)
    var tx = thread_idx.x
    var bx = block_idx.x
    var row_start = bx * _d

    var min_p_val = min_p_arr[bx]
    if min_p_val == 0.0:
        return

    # Pass 1: find thread-local max.
    var thread_max = Float32(-1e30)
    for i in range(tx, _d, block_size):
        thread_max = max(thread_max, Float32(probs[row_start + i]))

    # Block-level max reduction (broadcast result to all threads).
    var row_max = block.max[block_size=block_size, broadcast=True](thread_max)
    var threshold = min_p_val * row_max

    # Pass 2: zero out below threshold.
    for i in range(tx, _d, block_size):
        if Float32(probs[row_start + i]) < threshold:
            probs[row_start + i] = Scalar[dtype](0)


# The joint top-k/top-p cutoff search is bounded in practice by the value
# narrowing (min_gt_low == max_le_high); the iteration cap is defensive
# insurance against fp pathologies (NaN-laced rows), mirroring the Apple
# search path's fixed bound.
comptime _CUTOFF_SEARCH_MAX_ITERS = 64

# Eight lanes keep the six cutoff statistics in one aligned store.
comptime _COOP_STATS_WIDTH = 8


@always_inline
@__parameter
def _coop_max(x: SIMD, y: type_of(x)) -> type_of(x):
    return max(x, y)


@always_inline
@__parameter
def _coop_sum(x: SIMD, y: type_of(x)) -> type_of(x):
    return x + y


@always_inline
@__parameter
def _coop_cutoff_stats(x: SIMD, y: type_of(x)) -> type_of(x):
    """Combines cutoff statistics.

    Lanes 0 through 3 are sums, lane 4 is a minimum, and lane 5 is a maximum.
    """
    var r = x + y
    r[4] = min(x[4], y[4])
    r[5] = max(x[5], y[5])
    return r


@always_inline
def _block_reduce_cutoff_stats[
    block_size: Int, broadcast: Bool = True
](
    count0: Int32,
    count1: Int32,
    mass0: Float32,
    mass1: Float32,
    min_gt_low: Float32,
    max_le_high: Float32,
) -> Tuple[Int32, Int32, Float32, Float32, Float32, Float32]:
    """Fused block reduction for the cutoff search: 4 sums + min + max.

    Performs all six reductions in a single 2-barrier pass by casting the
    Int32 counts to Float32 (exact for counts up to 2^23).
    """

    @always_inline
    @__parameter
    def _reduce_fn[
        dtype: DType, width: SIMDLength, reduction_idx: Int
    ](v: SIMD[dtype, width]) -> Scalar[dtype]:
        comptime if reduction_idx < 4:
            return warp.sum(v)
        elif reduction_idx == 4:
            return warp.min(v)
        else:
            return warp.max(v)

    var results = block._block_reduce[
        block_size,
        warp_reduce_fn=_reduce_fn,
        broadcast=broadcast,
    ](
        StaticTuple[Float32, 6](
            Float32(count0),
            Float32(count1),
            mass0,
            mass1,
            min_gt_low,
            max_le_high,
        ),
        initial_vals=StaticTuple[Float32, 6](
            0, 0, 0, 0, Float32.MAX_FINITE, Float32.MIN_FINITE
        ),
    )
    return (
        Int32(results[0]),
        Int32(results[1]),
        results[2],
        results[3],
        results[4],
        results[5],
    )


@always_inline
def _block_reduce_topp_stats[
    block_size: Int, broadcast: Bool = True
](
    mass0: Float32,
    mass1: Float32,
    min_gt_low: Float32,
    max_le_high: Float32,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Reduces two masses and the cutoff bounds for a top-p-only search."""

    @always_inline
    @__parameter
    def _reduce_fn[
        dtype: DType, width: SIMDLength, reduction_idx: Int
    ](v: SIMD[dtype, width]) -> Scalar[dtype]:
        comptime if reduction_idx < 2:
            return warp.sum(v)
        elif reduction_idx == 2:
            return warp.min(v)
        else:
            return warp.max(v)

    var results = block._block_reduce[
        block_size,
        warp_reduce_fn=_reduce_fn,
        broadcast=broadcast,
    ](
        StaticTuple[Float32, 4](
            mass0,
            mass1,
            min_gt_low,
            max_le_high,
        ),
        initial_vals=StaticTuple[Float32, 4](
            0, 0, Float32.MAX_FINITE, Float32.MIN_FINITE
        ),
    )
    return (results[0], results[1], results[2], results[3])


@always_inline
def _sampling_rejection_loop_coop[
    vec_size: Int,
    block_size: Int,
    dtype: DType,
    deterministic: Bool,
    coop_size: Int,
    load_vec: def(Int) capturing[_] -> SIMD[.float32, vec_size],
    load_one: def(Int) capturing[_] -> Float32,
](
    d: Int,
    k: Int,
    p_eff: Float32,
    z: Float32,
    mut generator: Random,
    mut coop: CoopRow[coop_size],
    coop_ws: UnsafePointer[Int32, MutAnyOrigin],
    coop_table: UnsafePointer[mut=True, Float32, _, address_space=.SHARED],
    vec_begin: Int,
    vec_end: Int,
) -> Tuple[Int, Float32, Float32, Float32]:
    """Draws one token from a row split across blocks.

    Each block scans a contiguous slice. Group-wide values keep every block
    on the same branch and barrier sequence.
    """
    var tx = Int(thread_idx.x)
    var sampled_id_sram = unsafe_stack_allocation[
        1, Int, address_space=.SHARED
    ]()

    var sampled_id = 0
    var low = Float32(0)
    var high = Float32(1)
    var q = z
    var accepted_e = Float32(-1)

    while low < high:
        var u = generator.step_uniform()[0] * q

        # Float32 represents every vocabulary index exactly.
        var thread_sum = Float32(0)
        var thread_last = Float32(-1)
        for g in range(vec_begin + tx, vec_end, block_size):
            var v = load_vec(g * vec_size)
            comptime for j in range(vec_size):
                if v[j] > low:
                    thread_sum += v[j]
                    thread_last = Float32(g * vec_size + j)
        var slice_mass = block.sum[block_size=block_size, broadcast=True](
            thread_sum
        )
        var slice_last = block.max[block_size=block_size, broadcast=True](
            thread_last
        )
        var slice_last_val = Float32(0)
        if slice_last >= 0:
            slice_last_val = load_one(Int(slice_last))

        coop.gather[4](
            coop_ws,
            coop_table,
            SIMD[.float32, 4](slice_mass, slice_last, slice_last_val, 0),
        )

        # Rank order matches token order, so these are row-wide CDF prefixes.
        var prefix = Float32(0)
        var owner = -1
        var owner_base = Float32(0)
        var global_last = Float32(-1)
        var global_last_val = Float32(0)
        comptime for r in range(coop_size):
            var r_sum = coop_table[r * 4]
            var r_last = coop_table[r * 4 + 1]
            if owner < 0 and u < prefix + r_sum:
                owner = r
                owner_base = prefix
            prefix += r_sum
            if r_last > global_last:
                global_last = r_last
                global_last_val = coop_table[r * 4 + 2]

        if tx == 0:
            sampled_id_sram[0] = d
        barrier()
        var found = SIMD[.float32, 2](-1, 0)
        if coop.rank == owner:
            var aggregate = owner_base
            var slice_vecs = vec_end - vec_begin
            for i in range(ceildiv(slice_vecs, block_size)):
                var probs_vec = SIMD[.float32, vec_size](0)
                var g = i * block_size + tx
                if g < slice_vecs:
                    probs_vec = load_vec((vec_begin + g) * vec_size)
                var result = device_sampling_from_prob[
                    vec_size, block_size, dtype, deterministic
                ](i, d, low, u, probs_vec, aggregate, sampled_id_sram)
                aggregate = result[0]
                if aggregate > u:
                    break
            barrier()
            var sid = sampled_id_sram[0]
            if sid < d:
                var gid = sid + vec_begin * vec_size
                found[0] = Float32(gid)
                found[1] = load_one(gid)

        var comb = coop.combine[2, _coop_max](coop_ws, coop_table, found)

        var pivot_0: Float32
        if comb[0] >= 0:
            sampled_id = Int(comb[0])
            pivot_0 = comb[1]
        elif global_last >= 0:
            # Rounding can put `u` beyond the summed mass.
            sampled_id = Int(global_last)
            pivot_0 = global_last_val
        else:
            # Non-finite logits can leave the row without a valid token.
            sampled_id = 0
            break

        var pivot_1 = (pivot_0 + high) / 2.0

        var thread_count_0: Int32 = 0
        var thread_count_1: Int32 = 0
        var thread_mass_0 = Float32(0)
        var thread_mass_1 = Float32(0)
        for g in range(vec_begin + tx, vec_end, block_size):
            var v = load_vec(g * vec_size)
            comptime for j in range(vec_size):
                if v[j] > pivot_0:
                    thread_count_0 += 1
                    thread_mass_0 += v[j]
                if v[j] > pivot_1:
                    thread_count_1 += 1
                    thread_mass_1 += v[j]
        var stats = _block_reduce_cutoff_stats[block_size](
            thread_count_0,
            thread_count_1,
            thread_mass_0,
            thread_mass_1,
            Float32.MAX_FINITE,
            Float32.MIN_FINITE,
        )
        var combined = coop.combine[4, _coop_sum](
            coop_ws,
            coop_table,
            SIMD[.float32, 4](
                Float32(stats[0]),
                Float32(stats[1]),
                stats[2],
                stats[3],
            ),
        )
        var count_0 = Int32(combined[0])
        var count_1 = Int32(combined[1])
        var mass_0 = combined[2]
        var mass_1 = combined[3]

        # A zero top-p budget must still accept the argmax.
        if count_0 < Int32(k) and mass_0 <= p_eff:
            accepted_e = pivot_0
            break
        if count_1 < Int32(k) and mass_1 <= p_eff:
            low = pivot_0
            high = pivot_1
            q = mass_0
        else:
            low = pivot_1
            q = mass_1

    return Tuple[Int, Float32, Float32, Float32](sampled_id, low, accepted_e, q)


def _topp_budget(p: Float32, z: Float32) -> Float32:
    """Makes the top-p budget unbounded when ``p >= 1``.

    Independent reductions can disagree by a few ulps, so using ``z`` as the
    limit can reject a valid pivot.
    """
    return Float32.MAX if p >= 1.0 else p * z


@always_inline
def _topk_topp_cutoff_search[
    vec_size: Int,
    block_size: Int,
    load_dist: def(Int) capturing[_] -> SIMD[.float32, vec_size],
    track_count: Bool = True,
    coop_size: Int = 1,
](
    d: Int,
    k: Int32,
    p_eff: Float32,
    low_init: Float32,
    high_init: Float32,
    mass_above_low_init: Float32,
    mut coop: CoopRow[coop_size],
    coop_ws: Optional[UnsafePointer[Int32, MutAnyOrigin]],
    coop_table: UnsafePointer[mut=True, Float32, _, address_space=.SHARED],
    vec_begin: Int,
    vec_end: Int,
) -> Tuple[Float32, Float32]:
    """Finds the exact constraint-set cutoff and kept mass for joint top-k/top-p.

    A token with value ``t`` (in the working distribution domain served by
    ``load_dist``) survives the joint constraint iff ``count(> t) < k`` and
    ``mass(> t) <= p_eff`` — the same predicate
    ``TopKTopPSamplingFromProbKernel`` accepts samples with. That predicate is
    monotone in ``t``, so the surviving set is exactly ``{v : v > cutoff}``
    for the returned ``cutoff``, with all ties at the boundary value kept
    (this differs from an exact-count sorted rule only at exact fp ties).

    Dual-pivot ternary search with the value-narrowing termination from
    ``TopKMaskLogitsKernel``: bounds snap to actual data values via
    ``min_gt_low`` / ``max_le_high`` and the search ends when exactly one
    distinct value remains in ``(low, high]``, giving an exact cutoff with no
    epsilon reasoning. Top-p-only callers disable count tracking.

    Callers must guarantee the bracket invariants at entry: the predicate
    fails at ``low_init`` (some constraint violated), holds at ``high_init``,
    and ``mass_above_low_init == mass(> low_init)``. Rows where nothing fails
    (constraints disabled, or fewer than ``k`` positive-mass tokens) must be
    short-circuited by the caller instead of searched.

    With ``coop_size`` above one, blocks scan contiguous slices and combine
    statistics in rank order. Group-wide values keep every block on the same
    branch and barrier sequence.

    Returns:
        ``(cutoff, kept_mass)`` where ``kept_mass == mass(> cutoff)``. On the
        (defensive) iteration cap, returns the current bracket state, which
        keeps a superset of the constraint set but stays self-consistent.
    """
    var tx = thread_idx.x
    var low = low_init
    var high = high_init
    var mass_above_low = mass_above_low_init

    for _ in range(_CUTOFF_SEARCH_MAX_ITERS):
        var pivot_0: Float32
        var pivot_1: Float32
        comptime if not track_count:
            # A high top-p budget puts the cutoff near `low`; lower pivots
            # shrink that side of the bracket without changing the predicate.
            if 4 * p_eff > 3 * mass_above_low:
                pivot_0 = (high + 3 * low) / 4
                pivot_1 = (high + low) / 2
            else:
                pivot_0 = (high + 2 * low) / 3
                pivot_1 = (2 * high + low) / 3
        else:
            pivot_0 = (high + 2 * low) / 3
            pivot_1 = (2 * high + low) / 3

        # Accumulate thread-local counts/masses across all chunks.
        var thread_count_0: Int32 = 0
        var thread_count_1: Int32 = 0
        var thread_mass_0 = Float32(0)
        var thread_mass_1 = Float32(0)
        var min_gt_low = high
        var max_le_high = low

        # `vec_size` divides `d`, so every vector stays within the row.
        for g in range(vec_begin + Int(tx), vec_end, block_size):
            var v = load_dist(g * vec_size)

            comptime for j in range(vec_size):
                if v[j] > pivot_0:
                    comptime if track_count:
                        thread_count_0 += 1
                    thread_mass_0 += v[j]
                if v[j] > pivot_1:
                    comptime if track_count:
                        thread_count_1 += 1
                    thread_mass_1 += v[j]
                if v[j] > low:
                    min_gt_low = min(min_gt_low, v[j])
                if v[j] <= high:
                    max_le_high = max(max_le_high, v[j])

        # Single fused block reduction after processing all chunks.
        var count_0 = Int32(0)
        var count_1 = Int32(0)
        var mass_0: Float32
        var mass_1: Float32
        comptime if track_count:
            var stats = _block_reduce_cutoff_stats[block_size](
                thread_count_0,
                thread_count_1,
                thread_mass_0,
                thread_mass_1,
                min_gt_low,
                max_le_high,
            )
            count_0 = stats[0]
            count_1 = stats[1]
            mass_0 = stats[2]
            mass_1 = stats[3]
            min_gt_low = stats[4]
            max_le_high = stats[5]
        else:
            var stats = _block_reduce_topp_stats[block_size](
                thread_mass_0,
                thread_mass_1,
                min_gt_low,
                max_le_high,
            )
            mass_0 = stats[0]
            mass_1 = stats[1]
            min_gt_low = stats[2]
            max_le_high = stats[3]

        comptime if coop_size > 1:
            var combined = coop.combine[_COOP_STATS_WIDTH, _coop_cutoff_stats](
                coop_ws.unsafe_value(),
                coop_table,
                SIMD[.float32, _COOP_STATS_WIDTH](
                    Float32(count_0),
                    Float32(count_1),
                    mass_0,
                    mass_1,
                    min_gt_low,
                    max_le_high,
                    0,
                    0,
                ),
            )
            count_0 = Int32(combined[0])
            count_1 = Int32(combined[1])
            mass_0 = combined[2]
            mass_1 = combined[3]
            min_gt_low = combined[4]
            max_le_high = combined[5]

        # pivot_1 > pivot_0: if the constraint still fails above the higher
        # pivot it also fails above the lower one, so test high-to-low.
        comptime if track_count:
            if count_1 >= k or mass_1 > p_eff:
                low = pivot_1
                mass_above_low = mass_1
            elif count_0 >= k or mass_0 > p_eff:
                low = pivot_0
                mass_above_low = mass_0
                high = min(pivot_1, max_le_high)
            else:
                high = min(pivot_0, max_le_high)
        else:
            if mass_1 > p_eff:
                low = pivot_1
                mass_above_low = mass_1
            elif mass_0 > p_eff:
                low = pivot_0
                mass_above_low = mass_0
                high = min(pivot_1, max_le_high)
            else:
                high = min(pivot_0, max_le_high)

        # Exactly one distinct data value remains in (low, high]: every token
        # above `low` passes the predicate, every token at or below fails.
        if min_gt_low == max_le_high or high <= low:
            break

    return Tuple[Float32, Float32](low, mass_above_low)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
@__name(
    t"topk_topp_sampling_from_prob_{dtype}_{out_idx_type}_{deterministic}_{from_logits}_{emit_dist}_{dist_dtype}_{coop_size}",
)
def TopKTopPSamplingFromProbKernel[
    ProbsLayoutType: TensorLayout,
    probs_origin: ImmOrigin,
    OutputLayoutType: TensorLayout,
    output_origin: MutOrigin,
    block_size: Int,
    vec_size: Int,
    dtype: DType,
    out_idx_type: DType,
    deterministic: Bool,
    from_logits: Bool = False,
    emit_dist: Bool = False,
    dist_dtype: DType = .float32,
    coop_size: Int = 1,
    ProbsEngine: TensorEngine = DefaultEngine[element_width=1],
    OutputEngine: TensorEngine = DefaultEngine[element_width=1],
](
    probs: TileTensor[dtype, ProbsLayoutType, probs_origin, Engine=ProbsEngine],
    output: TileTensor[
        out_idx_type, OutputLayoutType, output_origin, Engine=OutputEngine
    ],
    out_dist: Optional[UnsafePointer[Scalar[dist_dtype], MutAnyOrigin]],
    indices: Optional[UnsafePointer[Scalar[out_idx_type], ImmutAnyOrigin]],
    top_k_arr: Optional[UnsafePointer[Scalar[out_idx_type], ImmutAnyOrigin]],
    top_k_val: Int32,
    top_p_arr: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
    top_p_val: Float32,
    d: Int32,
    rng_seed: Optional[UnsafePointer[UInt64, ImmutAnyOrigin]],
    rng_offset: UInt64,
    temperature: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
    min_p: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
    coop_ws: Optional[UnsafePointer[Int32, MutAnyOrigin]],
):
    """Kernel for joint top-k + top-p sampling from probability distribution.

    Identical to TopKSamplingFromProbKernel but additionally enforces a nucleus
    (top-p) constraint: a token is accepted only when both the count of tokens
    above the pivot is less than k AND the cumulative probability of those
    tokens is less than p.

    When top_p_val = 1.0 and top_p_arr is null, this degrades to top-k-only
    with zero overhead since sum < 1.0 is always true.

    When `from_logits` is True, `probs` contains raw logits and softmax with
    per-row temperature scaling is fused into the kernel: every load is
    transformed to `exp((logit - row_max) / temp)`, the unnormalized softmax
    value with the row maximum shifted to exactly 1.0. The pivot search over
    [0, 1] is unchanged; the total unnormalized mass `z` replaces the
    normalized distribution's implicit total of 1.0 in the initial CDF budget
    and scales the top-p threshold. The optional min-p mask is applied inline
    (in this domain the max "probability" is 1.0, so the mask threshold is
    simply `min_p`), matching `apply_min_p_mask_kernel` semantics in the
    normalized domain.

    When `emit_dist` is set, the kernel also writes the masked renormalized
    distribution it drew from to `out_dist`. Speculative decoding builds its
    rejection residual from that distribution, and reads the sampled token's
    own probability back out of it. Requires `from_logits`, and a non-Apple
    GPU because the cutoff search uses block collectives.

    Parameters:
        ProbsLayoutType: Memory layout of the input `probs` tile.
        probs_origin: Origin tag for the immutable input `probs` tile.
        OutputLayoutType: Memory layout of the output `output` tile.
        output_origin: Origin tag for the mutable output `output` tile.
        block_size: Number of threads per block.
        vec_size: Number of elements each thread loads per vectorized
            access.
        dtype: Element type of the `probs` tensor.
        out_idx_type: Index type used for the sampled output indices.
        deterministic: If True, use deterministic sampling.
        from_logits: If True, `probs` holds raw logits and softmax with
            per-row temperature scaling and min-p masking is fused into the
            kernel (defaults to False).
        emit_dist: If True, also write the masked distribution to
            `out_dist` (defaults to False).
        dist_dtype: Element type of `out_dist`.
        coop_size: Blocks sharing each row; 1 keeps the whole row in one
            block and compiles the cross-block traffic away.
        ProbsEngine: Engine of the input `probs` tile.
        OutputEngine: Engine of the output `output` tile.

    Args:
        probs: Input probability distribution [batch_size, _d].
        output: Output sampled indices [batch_size].
        out_dist: Output masked distribution [batch_size, _d]; required when
            `emit_dist` is set.
        indices: Optional row indices for batch indexing [batch_size].
        top_k_arr: Optional per-row top_k values [batch_size].
        top_k_val: Default top_k value if top_k_arr is null.
        top_p_arr: Optional per-row top_p values [batch_size].
        top_p_val: Default top_p value if top_p_arr is null.
        d: Vocabulary size.
        rng_seed: Optional per-row seed array [batch_size], indexed by
            row_idx. If null, defaults to 0.
        rng_offset: Random offset for Random number generator.
        temperature: Optional per-row temperature [batch_size]. Only used
            when `from_logits` is True; defaults to 1.0 per row.
        min_p: Optional per-row min-p thresholds [batch_size]. Only used
            when `from_logits` is True.
        coop_ws: Zeroed cross-block workspace; required when `coop_size`
            exceeds one, ignored otherwise.
    """
    comptime assert (
        not emit_dist or from_logits
    ), "out_dist requires from_logits"
    comptime assert (
        not emit_dist or not is_apple_gpu()
    ), "out_dist is not supported on Apple GPUs"
    # AMD benefits from replacing repeated exponentiation with an FP32
    # workspace; the final pass overwrites the cached weights in place.
    comptime cache_dist = (
        is_amd_gpu() and from_logits and emit_dist and dist_dtype == .float32
    )

    var _top_k_val = Int(top_k_val)
    var _d = Int(d)
    comptime assert output.flat_rank == 1

    comptime assert (
        coop_size == 1 or from_logits
    ), "the cooperative split needs the fused softmax"
    var n_vec = _d // vec_size
    var rank = Int(block_idx.x) % coop_size
    var bx = Int(block_idx.x) // coop_size
    var tx = thread_idx.x

    # Contiguous slices make block-rank order match token order.
    var slice_vec = ceildiv(n_vec, coop_size)
    var vec_begin = min(rank * slice_vec, n_vec)
    var vec_end = min(vec_begin + slice_vec, n_vec)

    var coop = CoopRow[coop_size](bx, rank)
    var coop_table = unsafe_stack_allocation[
        coop_size * COOP_SLOT_FLOATS,
        Float32,
        alignment=16,
        address_space=.SHARED,
    ]()

    var row_idx = bx
    if indices:
        row_idx = Int(indices.unsafe_value().load(bx))

    with PDL():
        var seed_val = UInt64(0)
        if rng_seed:
            seed_val = rng_seed.unsafe_value()[row_idx]

        # Offset is keyed on row_idx (the request's logical row), not bx (the
        # physical batch slot), so a request samples identically regardless of
        # where it lands in the batch. The per-row seed already decorrelates rows.
        var generator = Random(
            seed=seed_val, offset=UInt64(row_idx) + rng_offset
        )

        var k = _top_k_val
        if top_k_arr:
            k = Int(top_k_arr.unsafe_value().load(row_idx))
        if k == -1:
            k = _top_k_val
        if k <= 0 or k > _d:
            k = _d

        var p = top_p_val
        if top_p_arr:
            p = top_p_arr.unsafe_value()[row_idx]

        var probs_ptr = probs.ptr + row_idx * _d
        var probs_row = TileTensor(probs_ptr, row_major(Idx[1], _d))

        # From-logits mode: resolve per-row temperature / min-p and compute
        # the row max and total unnormalized softmax mass z in two uniform
        # passes. z defaults to 1.0 in from-prob mode so the CDF budget and
        # top-p scaling below are domain-independent.
        var inv_temp = Float32(1.0)
        var row_max = Float32(0.0)
        var min_p_thresh = Float32(0.0)
        var z = Float32(1.0)
        # Mass of the min-p-masked working distribution -- what `load_dist`
        # actually serves. Only the emitted distribution normalizes by it;
        # the sampling budget stays `z`.
        var masked_z = Float32(1.0)

        comptime if from_logits:
            var temp_val = Float32(1.0)
            if temperature:
                temp_val = temperature.unsafe_value()[row_idx]
            # Clamp to prevent division by zero on greedy (T=0) rows.
            inv_temp = 1.0 / max(temp_val, Float32(1e-6))
            if min_p:
                min_p_thresh = min_p.unsafe_value()[row_idx]

            # Every block needs the same row maximum because later control
            # flow contains group barriers.
            var thread_max = Float32.MIN
            for i in range(vec_begin + Int(tx), vec_end, block_size):
                var v = probs_row.load[width=vec_size](
                    (Idx[0], i * vec_size)
                ).cast[.float32]()
                thread_max = max(thread_max, v.reduce_max())
            row_max = block.max[block_size=block_size, broadcast=True](
                thread_max
            )
            comptime if coop_size > 1:
                row_max = coop.combine[1, _coop_max](
                    coop_ws.unsafe_value(),
                    coop_table,
                    SIMD[.float32, 1](row_max),
                )[0]

            # Pass 2: block sum of exp((logit - row_max) / temp). The full
            # (unmasked) mass is used, matching the separate-softmax path
            # where probabilities are normalized before min-p masking.
            var thread_sum = Float32(0.0)
            var thread_masked_sum = Float32(0.0)
            for i in range(vec_begin + Int(tx), vec_end, block_size):
                var v = probs_row.load[width=vec_size](
                    (Idx[0], i * vec_size)
                ).cast[.float32]()
                var e = exp((v - row_max) * inv_temp)
                comptime if cache_dist:
                    # Each thread rereads its own slice. Block reductions
                    # order the only cross-thread scalar read.
                    var dist_row = TileTensor(
                        out_dist.unsafe_value() + bx * _d,
                        row_major(Idx[1], _d),
                    )
                    dist_row.store[width=vec_size](
                        (Idx[0], i * vec_size), e.cast[dist_dtype]()
                    )
                thread_sum += e.reduce_add()
                comptime if emit_dist:
                    if min_p_thresh > 0:
                        comptime for j in range(vec_size):
                            if e[j] < min_p_thresh:
                                e[j] = 0
                        thread_masked_sum += e.reduce_add()
            z = block.sum[block_size=block_size, broadcast=True](thread_sum)
            comptime if emit_dist:
                # `min_p_thresh` is block-uniform (one row per block), so
                # every thread takes the same branch and the collective
                # stays legal. Without a mask the masked total is `z`.
                if min_p_thresh > 0:
                    masked_z = block.sum[block_size=block_size, broadcast=True](
                        thread_masked_sum
                    )
                else:
                    masked_z = z
            comptime if coop_size > 1:
                comptime if emit_dist:
                    var totals = coop.combine[2, _coop_sum](
                        coop_ws.unsafe_value(),
                        coop_table,
                        SIMD[.float32, 2](z, masked_z),
                    )
                    z = totals[0]
                    masked_z = totals[1]
                else:
                    z = coop.combine[1, _coop_sum](
                        coop_ws.unsafe_value(),
                        coop_table,
                        SIMD[.float32, 1](z),
                    )[0]

        @__parameter
        @always_inline
        def load_dist[width: Int](offset: Int) -> SIMD[.float32, width]:
            # Load `width` elements of the sampling distribution at `offset`.
            # In from-logits mode this is the unnormalized softmax value with
            # the min-p mask applied inline.
            comptime if from_logits:
                var e: SIMD[.float32, width]
                comptime if cache_dist:
                    var dist_row = TileTensor(
                        out_dist.unsafe_value() + bx * _d,
                        row_major(Idx[1], _d),
                    )
                    e = dist_row.load[width=width]((Idx[0], offset)).cast[
                        .float32
                    ]()
                else:
                    var v = probs_row.load[width=width]((Idx[0], offset)).cast[
                        .float32
                    ]()
                    e = exp((v - row_max) * inv_temp)
                # Same predicate as apply_min_p_mask_kernel (`< threshold`
                # zeroes; NaN compares false and is preserved).
                if min_p_thresh > 0:
                    comptime for j in range(width):
                        if e[j] < min_p_thresh:
                            e[j] = 0
                return e
            else:
                return probs_row.load[width=width]((Idx[0], offset)).cast[
                    .float32
                ]()

        # Top-p budget in the working domain (z == 1.0 in from-prob mode).
        var p_eff = _topp_budget(p, z)

        # The final sampled index, produced by whichever search path runs.
        var sampled_id = 0

        comptime if is_apple_gpu():
            # Single-thread-driven ternary search (Apple/Metal only; see the
            # detailed comment on TopKSamplingFromProbKernel — this is the same
            # fix with the joint top-k + top-p accept predicate). The ENTIRE
            # search runs sequentially on tx==0 with no in-loop block
            # collectives, so there is nothing to desynchronize the warps; a
            # FIXED-BOUND `for` loop keeps the trip count uniform and the other
            # threads merely hit the per-iteration barrier.
            comptime MAX_ITERS = 64

            var done_sram = unsafe_stack_allocation[
                1, Int32, address_space=.SHARED
            ]()
            var out_id_sram = unsafe_stack_allocation[
                1, Int, address_space=.SHARED
            ]()

            # Initialize control once, uniformly.
            if tx == 0:
                done_sram[0] = 0
                out_id_sram[0] = 0
            barrier()

            # Entire search on tx==0 (sequential scans, no block collectives);
            # all threads advance the RNG in lockstep and hit the per-iteration
            # barrier.
            var low: Float32 = 0.0
            var high: Float32 = 1.0
            var q: Float32 = z

            for _it in range(MAX_ITERS):
                var done = done_sram[0] != 0
                var u = generator.step_uniform()[0] * q

                if tx == 0 and not done:
                    # Sequential CDF sample over the row (prob > low).
                    var cum: Float32 = 0.0
                    var search_id = _d
                    var last_valid_id = 0
                    for j in range(_d):
                        var pv = Float32(load_dist[1](j))
                        if pv > low:
                            last_valid_id = j
                            cum += pv
                            if cum > u and search_id == _d:
                                search_id = j
                    if search_id == _d:
                        search_id = last_valid_id

                    var pivot_0 = Float32(load_dist[1](search_id))
                    var pivot_1 = (pivot_0 + high) / 2.0

                    # Sequential counts + prob mass for both pivots.
                    var count_0: Int = 0
                    var value_0: Float32 = 0.0
                    var count_1: Int = 0
                    var value_1: Float32 = 0.0
                    for j in range(_d):
                        var pv = Float32(load_dist[1](j))
                        if pv > pivot_0:
                            count_0 += 1
                            value_0 += pv
                        if pv > pivot_1:
                            count_1 += 1
                            value_1 += pv

                    if count_0 < k and value_0 <= p_eff:
                        # Case 1: pivot_0 accepted - count below k AND mass
                        # below p. Use <= so that p=0 correctly accepts the
                        # argmax.
                        out_id_sram[0] = search_id
                        done_sram[0] = 1
                    elif count_1 < k and value_1 <= p_eff:
                        # Case 2: pivot_0 rejected, pivot_1 accepted.
                        low = pivot_0
                        high = pivot_1
                        q = value_0
                    else:
                        # Case 3: both pivots rejected.
                        low = pivot_1
                        q = value_1

                    # Bracket collapse: emit the current candidate as a fallback.
                    if low >= high:
                        out_id_sram[0] = search_id
                        done_sram[0] = 1
                barrier()

            sampled_id = out_id_sram[0]
        else:
            # `emit_dist` reuses this bracket for its cutoff search.
            var q: Float32 = z
            var low: Float32 = 0.0
            var accepted_e = Float32(-1)

            comptime if coop_size > 1:

                @__parameter
                @always_inline
                def load_slice(offset: Int) -> SIMD[.float32, vec_size]:
                    return load_dist[vec_size](offset)

                @__parameter
                @always_inline
                def load_scalar(offset: Int) -> Float32:
                    return load_dist[1](offset)[0]

                var drawn = _sampling_rejection_loop_coop[
                    vec_size,
                    block_size,
                    dtype,
                    deterministic,
                    coop_size,
                    load_slice,
                    load_scalar,
                ](
                    _d,
                    k,
                    p_eff,
                    z,
                    generator,
                    coop,
                    coop_ws.unsafe_value(),
                    coop_table,
                    vec_begin,
                    vec_end,
                )
                sampled_id = drawn[0]
                low = drawn[1]
                accepted_e = drawn[2]
                q = drawn[3]
            else:
                var sampled_id_sram = unsafe_stack_allocation[
                    1, Int, address_space=.SHARED
                ]()
                var last_valid_id_sram = unsafe_stack_allocation[
                    1, Int, address_space=.SHARED
                ]()

                var probs_vec: SIMD[.float32, vec_size]
                var aggregate: Float32
                var high: Float32 = 1.0

                # A row without a candidate must not read stale shared memory.
                if tx == 0:
                    last_valid_id_sram[0] = -1
                barrier()

                while low < high:
                    if tx == 0:
                        sampled_id_sram[0] = _d
                    barrier()

                    var u = generator.step_uniform()[0] * q
                    aggregate = 0.0
                    var thread_max_valid = -1

                    for i in range(ceildiv(_d, block_size * vec_size)):
                        probs_vec = 0
                        if (i * block_size + tx) * vec_size < _d:
                            probs_vec = load_dist[vec_size](
                                (i * block_size + tx) * vec_size
                            )
                        var result = device_sampling_from_prob[
                            vec_size, block_size, dtype, deterministic
                        ](
                            i,
                            _d,
                            low,
                            u,
                            probs_vec,
                            aggregate,
                            sampled_id_sram,
                        )
                        aggregate = result[0]
                        thread_max_valid = max(thread_max_valid, result[1])
                        if aggregate > u:
                            break

                    var block_max_valid = block.max[
                        block_size=block_size,
                        broadcast=False,
                    ](Int32(thread_max_valid))

                    if tx == 0 and block_max_valid != -1:
                        last_valid_id_sram[0] = Int(block_max_valid)

                    barrier()

                    sampled_id = sampled_id_sram[0]
                    if sampled_id == _d:
                        sampled_id = last_valid_id_sram[0]

                    if sampled_id < 0:
                        # Non-finite logits can leave the row without a valid token.
                        sampled_id = 0
                        break

                    var pivot_0 = Float32(load_dist[1](sampled_id))
                    var pivot_1 = (pivot_0 + high) / 2.0

                    var thread_vc_0_total = ValueCount[.float32](0.0, 0)
                    var thread_vc_1_total = ValueCount[.float32](0.0, 0)

                    for i in range(ceildiv(_d, block_size * vec_size)):
                        probs_vec = 0
                        if (i * block_size + tx) * vec_size < _d:
                            probs_vec = load_dist[vec_size](
                                (i * block_size + tx) * vec_size
                            )
                        var probs_gt_pivot_0_values = SIMD[.float32, vec_size]()
                        var probs_gt_pivot_0_counts = SIMD[.int32, vec_size]()
                        var probs_gt_pivot_1_values = SIMD[.float32, vec_size]()
                        var probs_gt_pivot_1_counts = SIMD[.int32, vec_size]()

                        comptime for j in range(vec_size):
                            var idx = (i * block_size + tx) * vec_size + j
                            var is_valid = idx < _d

                            var gt_pivot_0 = probs_vec[j] > pivot_0
                            probs_gt_pivot_0_values[j] = probs_vec[
                                j
                            ] if gt_pivot_0 else 0.0
                            probs_gt_pivot_0_counts[j] = Int32(1) if (
                                gt_pivot_0 and is_valid
                            ) else Int32(0)

                            var gt_pivot_1 = probs_vec[j] > pivot_1
                            probs_gt_pivot_1_values[j] = probs_vec[
                                j
                            ] if gt_pivot_1 else 0.0
                            probs_gt_pivot_1_counts[j] = Int32(1) if (
                                gt_pivot_1 and is_valid
                            ) else Int32(0)

                        thread_vc_0_total += ValueCount[.float32](
                            probs_gt_pivot_0_values.reduce_add(),
                            probs_gt_pivot_0_counts.reduce_add(),
                        )
                        thread_vc_1_total += ValueCount[.float32](
                            probs_gt_pivot_1_values.reduce_add(),
                            probs_gt_pivot_1_counts.reduce_add(),
                        )

                    # Small top-k values usually accept the first pivot, so
                    # defer the second reduction.
                    var aggregate_gt_pivot_0 = _block_reduce_value_count[
                        .float32, broadcast=True
                    ](thread_vc_0_total)

                    if (
                        aggregate_gt_pivot_0.count < Int32(k)
                        and aggregate_gt_pivot_0.value <= p_eff
                    ):
                        # A zero top-p budget must still accept the argmax.
                        accepted_e = pivot_0
                        break

                    var aggregate_gt_pivot_1 = _block_reduce_value_count[
                        .float32, broadcast=True
                    ](thread_vc_1_total)

                    if (
                        aggregate_gt_pivot_1.count < Int32(k)
                        and aggregate_gt_pivot_1.value <= p_eff
                    ):
                        low = pivot_0
                        high = pivot_1
                        q = aggregate_gt_pivot_0.value
                    else:
                        low = pivot_1
                        q = aggregate_gt_pivot_1.value

            barrier()

            comptime if emit_dist:
                # The accept predicate is monotone in the token's own weight,
                # so the surviving set is {v : v > cutoff} and softmax over it
                # is v / kept_mass. A row that never accepted (non-finite
                # logits, or the bracket collapsing) has no constraint set to
                # report, so it stays zero -- callers read that as "no
                # distribution for this row".
                var cutoff = Float32.MAX_FINITE
                var kept_mass = Float32(1)

                if accepted_e >= 0:
                    if k == _d and p >= 1.0 and not min_p:
                        cutoff = 0
                        kept_mass = z
                    else:

                        @__parameter
                        @always_inline
                        def load_dist_vec(
                            offset: Int,
                        ) -> SIMD[.float32, vec_size]:
                            return load_dist[vec_size](offset)

                        # The search needs `mass(> low)` over the masked working
                        # distribution. A refined `q` came from `load_dist` sums
                        # and is exactly that; the initial budget is the unmasked
                        # `z`, which overstates it whenever min-p zeroed weight.
                        var mass_above_low = q if low > 0 else masked_z
                        var refined: Tuple[Float32, Float32]
                        if k == _d:
                            refined = _topk_topp_cutoff_search[
                                vec_size,
                                block_size,
                                load_dist_vec,
                                track_count=False,
                                coop_size=coop_size,
                            ](
                                _d,
                                Int32(k),
                                p_eff,
                                low,
                                accepted_e,
                                mass_above_low,
                                coop,
                                coop_ws,
                                coop_table,
                                vec_begin,
                                vec_end,
                            )
                        else:
                            refined = _topk_topp_cutoff_search[
                                vec_size,
                                block_size,
                                load_dist_vec,
                                coop_size=coop_size,
                            ](
                                _d,
                                Int32(k),
                                p_eff,
                                low,
                                accepted_e,
                                mass_above_low,
                                coop,
                                coop_ws,
                                coop_table,
                                vec_begin,
                                vec_end,
                            )
                        cutoff = refined[0]
                        kept_mass = refined[1]

                var dist_row = TileTensor(
                    out_dist.unsafe_value() + bx * _d,
                    row_major(Idx[1], _d),
                )
                for i in range(vec_begin + Int(tx), vec_end, block_size):
                    var e = load_dist[vec_size](i * vec_size)
                    var masked = (e.gt(cutoff)).select(
                        e / kept_mass, SIMD[.float32, vec_size](0)
                    )
                    dist_row.store[width=vec_size](
                        (Idx[0], i * vec_size), masked.cast[dist_dtype]()
                    )

        if tx == 0 and rank == 0:
            output[bx] = Scalar[out_idx_type](sampled_id)


def topk_topp_sampling_from_prob[
    dtype: DType,
    out_idx_type: DType,
    block_size: Int = 1024,
    from_logits: Bool = False,
    emit_dist: Bool = False,
    dist_dtype: DType = .float32,
    DistLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64, Int64].element_types,
        stride_types=Coord[Int64, ComptimeInt[1]].element_types,
    ],
    TopKArrLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    IndicesLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    TopPArrLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    SeedLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    TemperatureLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    MinPLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    TopKArrEngine: TensorEngine = DefaultEngine[element_width=1],
    IndicesEngine: TensorEngine = DefaultEngine[element_width=1],
    TopPArrEngine: TensorEngine = DefaultEngine[element_width=1],
    SeedEngine: TensorEngine = DefaultEngine[element_width=1],
    TemperatureEngine: TensorEngine = DefaultEngine[element_width=1],
    MinPEngine: TensorEngine = DefaultEngine[element_width=1],
](
    ctx: DeviceContext,
    probs: TileTensor[mut=False, dtype, ...],
    output: TileTensor[mut=True, out_idx_type, ...],
    top_k_val: Int,
    top_p_val: Float32 = 1.0,
    deterministic: Bool = False,
    rng_seed: Optional[
        TileTensor[.uint64, SeedLayoutType, ImmutAnyOrigin, Engine=SeedEngine]
    ] = None,
    rng_offset: UInt64 = 0,
    indices: Optional[
        TileTensor[
            out_idx_type,
            IndicesLayoutType,
            ImmutAnyOrigin,
            Engine=IndicesEngine,
        ]
    ] = None,
    top_k_arr: Optional[
        TileTensor[
            out_idx_type,
            TopKArrLayoutType,
            ImmutAnyOrigin,
            Engine=TopKArrEngine,
        ]
    ] = None,
    top_p_arr: Optional[
        TileTensor[
            .float32,
            TopPArrLayoutType,
            ImmutAnyOrigin,
            Engine=TopPArrEngine,
        ]
    ] = None,
    temperature: Optional[
        TileTensor[
            .float32,
            TemperatureLayoutType,
            ImmutAnyOrigin,
            Engine=TemperatureEngine,
        ]
    ] = None,
    min_p: Optional[
        TileTensor[.float32, MinPLayoutType, ImmutAnyOrigin, Engine=MinPEngine]
    ] = None,
    out_dist: Optional[
        TileTensor[dist_dtype, DistLayoutType, MutAnyOrigin]
    ] = None,
) raises:
    """Joint top-k + top-p sampling from probability distribution.

    Performs stochastic sampling considering only tokens that satisfy both the
    top-k count constraint AND the top-p nucleus constraint. When top_p_val is
    1.0 (default) this behaves identically to topk_sampling_from_prob.

    When `emit_dist` is set, the masked renormalized distribution is written
    to `out_dist` as well; see the kernel docstring.

    When `from_logits` is True, `probs` contains raw logits: softmax with
    per-row temperature scaling and the optional min-p mask are fused into
    the sampling kernel, avoiding the [batch_size, d] probability round-trip
    through global memory and the separate softmax / mask kernel launches.

    Parameters:
        dtype: Element type of the `probs` tensor.
        out_idx_type: Index type used for the sampled output indices.
        block_size: Number of threads per block (defaults to 1024).
        from_logits: If True, `probs` holds raw logits and softmax with
            per-row temperature scaling and min-p masking is fused into
            the kernel (defaults to False).
        emit_dist: If True, also write the masked renormalized distribution
            to `out_dist` (defaults to False).
        dist_dtype: Element type of `out_dist`.
        DistLayoutType: Memory layout of the optional `out_dist` tensor.
        TopKArrLayoutType: Memory layout of the optional `top_k_arr` tensor.
        IndicesLayoutType: Memory layout of the optional `indices` tensor.
        TopPArrLayoutType: Memory layout of the optional `top_p_arr` tensor.
        SeedLayoutType: Memory layout of the optional `rng_seed` tensor.
        TemperatureLayoutType: Memory layout of the optional `temperature`
            tensor.
        MinPLayoutType: Memory layout of the optional `min_p` tensor.
        TopKArrEngine: Engine of the optional `top_k_arr` tensor.
        IndicesEngine: Engine of the optional `indices` tensor.
        TopPArrEngine: Engine of the optional `top_p_arr` tensor.
        SeedEngine: Engine of the optional `rng_seed` tensor.
        TemperatureEngine: Engine of the optional `temperature`
            tensor.
        MinPEngine: Engine of the optional `min_p` tensor.

    Args:
        ctx: Device context for kernel execution.
        probs: Input probability distribution [batch_size, d], or raw logits
            when `from_logits` is True.
        output: Output sampled indices [batch_size].
        top_k_val: Default top-k value (number of top tokens to consider).
        top_p_val: Default top-p value (nucleus probability threshold).
        deterministic: Whether to use deterministic sampling.
        rng_seed: Optional per-row seed tensor [batch_size], indexed by the
            request's logical row (see `indices`). If None, defaults to 0.
        rng_offset: Random offset for Random number generator.
        indices: Optional row indices for batch indexing [batch_size].
        top_k_arr: Optional per-row top-k values [batch_size].
        top_p_arr: Optional per-row top-p values [batch_size].
        temperature: Optional per-row temperature values [batch_size]. Only
            used when `from_logits` is True; defaults to 1.0 per row.
        min_p: Optional per-row min-p thresholds [batch_size]. Only used
            when `from_logits` is True.
        out_dist: Output masked distribution [batch_size, d]. Required when
            `emit_dist` is set.

    Raises:
        Error: If tensor ranks or shapes are invalid.
    """

    comptime assert probs.rank == 2, "probs rank must be 2"
    comptime assert output.rank == 1, "output rank must be 1"

    var shape = coord_to_index_list(probs.layout.shape_coord())
    var batch_size = shape[0]
    var d = shape[1]

    def trace_information() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("probs", shape, dtype),
                    "top_k_val=" + String(top_k_val),
                    "top_p_val=" + String(top_p_val),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        "topk_topp_sampling_from_prob",
        Trace[TraceLevel.OP]._get_detail_str(trace_information),
        task_id=Int(ctx.id()),
    ):
        var out_shape = coord_to_index_list(output.layout.shape_coord())
        if out_shape[0] != batch_size:
            raise Error("output batch size must match probs batch size")

        # Use up to 8 elements per vector to minimize the number of chunks
        # (and therefore the number of block-level reductions in inner loops).
        # GPU vector loads handle wider-than-native SIMD efficiently, and the
        # per-element idx < d guard handles non-aligned tails correctly.
        var vec_size = gcd(8, d)

        var indices_ptr: Optional[
            UnsafePointer[Scalar[out_idx_type], ImmutAnyOrigin]
        ] = None
        if indices:
            indices_ptr = indices.unsafe_value().ptr

        var top_k_ptr: Optional[
            UnsafePointer[Scalar[out_idx_type], ImmutAnyOrigin]
        ] = None
        if top_k_arr:
            top_k_ptr = top_k_arr.unsafe_value().ptr

        var top_p_ptr: Optional[UnsafePointer[Float32, ImmutAnyOrigin]] = None
        if top_p_arr:
            top_p_ptr = top_p_arr.unsafe_value().ptr

        var seed_ptr: Optional[UnsafePointer[UInt64, ImmutAnyOrigin]] = None
        if rng_seed:
            seed_ptr = rng_seed.unsafe_value().ptr

        var temperature_ptr: Optional[
            UnsafePointer[Float32, ImmutAnyOrigin]
        ] = None
        if temperature:
            temperature_ptr = temperature.unsafe_value().ptr

        var min_p_ptr: Optional[UnsafePointer[Float32, ImmutAnyOrigin]] = None
        if min_p:
            min_p_ptr = min_p.unsafe_value().ptr

        var dist_ptr: Optional[
            UnsafePointer[Scalar[dist_dtype], MutAnyOrigin]
        ] = None

        comptime if emit_dist:
            if not out_dist:
                raise Error("out_dist is required when emit_dist is set")
            var dist_shape = coord_to_index_list(
                out_dist.unsafe_value().layout.shape_coord()
            )
            if dist_shape[0] != batch_size or dist_shape[1] != d:
                raise Error("out_dist shape must match probs shape")
            dist_ptr = out_dist.unsafe_value().ptr

        comptime coop_capable = from_logits and has_amd_gpu_accelerator()
        var coop = 1
        comptime if coop_capable:
            coop = coop_group_size(ctx, batch_size, block_size, d // vec_size)
        var coop_buf = ctx.enqueue_create_buffer[.int32](
            coop_row_words(batch_size, coop) if coop > 1 else 1
        )
        var coop_ptr: Optional[UnsafePointer[Int32, MutAnyOrigin]] = None
        if coop > 1:
            # Barrier counters must start at zero.
            ctx.enqueue_memset(coop_buf, 0)
            coop_ptr = coop_buf.unsafe_ptr().as_unsafe_any_origin()

        @__parameter
        def launch_kernel[
            vec_size: Int, deterministic: Bool, coop_size: Int
        ]() raises:
            comptime kernel = TopKTopPSamplingFromProbKernel[
                probs.LayoutType,
                ImmOrigin(probs.origin),
                output.LayoutType,
                output.origin,
                block_size,
                vec_size,
                dtype,
                out_idx_type,
                deterministic,
                from_logits,
                emit_dist,
                dist_dtype,
                coop_size,
                ProbsEngine=probs.Engine,
                OutputEngine=output.Engine,
            ]
            ctx.enqueue_function[kernel](
                probs.as_immut(),
                output,
                dist_ptr,
                indices_ptr,
                top_k_ptr,
                Int32(top_k_val),
                top_p_ptr,
                top_p_val,
                Int32(d),
                seed_ptr,
                rng_offset,
                temperature_ptr,
                min_p_ptr,
                coop_ptr,
                grid_dim=batch_size * coop_size,
                block_dim=block_size,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )

        @__parameter
        def dispatch_vec_size[deterministic: Bool]() raises:
            comptime for param_vec_size in [16, 8, 4, 2, 1]:
                if vec_size == param_vec_size:
                    comptime if coop_capable:
                        comptime for param_coop in [2, 4, 8, 16]:
                            if coop == param_coop:
                                return launch_kernel[
                                    param_vec_size, deterministic, param_coop
                                ]()
                    return launch_kernel[param_vec_size, deterministic, 1]()

        if deterministic:
            dispatch_vec_size[True]()
        else:
            dispatch_vec_size[False]()
        _ = coop_buf^


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
@__name(t"topk_softmax_sample_{dtype}_{out_idx_type}")
def topk_softmax_sample_kernel[
    block_size: Int,
    vec_size: Int,
    dtype: DType,
    out_idx_type: DType,
    LogitsLayoutType: TensorLayout,
    logits_origin: ImmOrigin,
    SampledLayoutType: TensorLayout,
    sampled_origin: MutOrigin,
](
    logits: TileTensor[dtype, LogitsLayoutType, logits_origin],
    sampled_indices: TileTensor[
        out_idx_type, SampledLayoutType, sampled_origin
    ],
    top_k_arr: Optional[
        UnsafePointer[Scalar[out_idx_type], MutUntrackedOrigin]
    ],
    top_k_val: Int32,
    temperature_val: Float32,
    temperature: Optional[UnsafePointer[Float32, MutUntrackedOrigin]],
    seed_val: UInt64,
    seed: Optional[UnsafePointer[UInt64, MutUntrackedOrigin]],
    d: Int32,
):
    var _top_k_val = Int(top_k_val)
    var _d = Int(d)
    comptime assert sampled_indices.flat_rank == 1

    var bx = block_idx.x
    var tx = thread_idx.x
    var row_idx = bx

    var logits_ptr = logits.ptr + bx * _d

    var logits_row = TileTensor(logits_ptr, row_major(Idx[1], _d))

    var k = _top_k_val
    if top_k_arr:
        k = Int(top_k_arr.unsafe_value()[bx])
    var temp_val = temperature_val
    if temperature:
        temp_val = max(temperature.unsafe_value()[bx], 1e-6)

    # Allocate shared memory for caching top-k elements.
    # Round up to ensure proper alignment for Int array.
    var k_rounded = align_up(k, WARP_SIZE)

    # On Apple the cache is a static allocation. Reserve headroom below the 32K
    # threadgroup limit for the kernel's auxiliary SMEM (`s_count` + the block
    # reductions' per-warp scratch); allocating the full 32K bucket for the
    # cache alone overflowed the limit (33932 > 32768). The host launcher's
    # guard (`_APPLE_STATIC_SHMEM_CACHE_BYTES`) bounds k to this reduced budget.
    var s_vals = unsafe_stack_allocation[
        _APPLE_STATIC_SHMEM_CACHE_COUNT,
        Float32,
        address_space=.SHARED,
    ]() if comptime (is_apple_gpu()) else external_memory[
        Float32,
        address_space=.SHARED,
        alignment=align_of[Float32](),
    ]()

    var s_idxs = (s_vals + k_rounded).bitcast[Int]()
    var s_count = unsafe_stack_allocation[1, Int, address_space=.SHARED]()

    with PDL():
        if tx == 0:
            s_count[0] = 0

        # PHASE 1: Find pivot (k-th largest) via ternary search.
        var pivot = Float32.MIN
        var max_logit: Float32
        var logits_vec = SIMD[.float32, vec_size]()

        if k < _d:
            var min_max = get_min_max_value[vec_size, block_size](
                logits.ptr, row_idx, _d
            )
            var min_val, max_val = min_max[0], min_max[1]

            max_logit = max_val

            # Initialize ternary search bounds.
            var low = Float32(
                min_val - 1 if min_val != Float32.MIN else Float32.MIN
            )
            var high = max_val

            while True:
                var pivot_0 = (high + 2 * low) / 3
                var pivot_1 = (2 * high + low) / 3

                # Accumulate thread-local counts across all chunks.
                var thread_count_0_total: Int32 = 0
                var thread_count_1_total: Int32 = 0
                var min_gt_low = high
                var max_le_high = low

                for i in range(ceildiv(_d, block_size * vec_size)):
                    if (i * block_size + tx) * vec_size < _d:
                        logits_vec = logits_row.load[width=vec_size](
                            (
                                Idx[0],
                                i * block_size * vec_size + tx * vec_size,
                            )
                        ).cast[.float32]()

                    var probs_gt_pivot_0_count = SIMD[.int32, vec_size]()
                    var probs_gt_pivot_1_count = SIMD[.int32, vec_size]()

                    comptime for j in range(vec_size):
                        var idx = (i * block_size + tx) * vec_size + j

                        probs_gt_pivot_0_count[j] = Int32(1) if (
                            logits_vec[j] > pivot_0 and idx < _d
                        ) else Int32(0)
                        probs_gt_pivot_1_count[j] = Int32(1) if (
                            logits_vec[j] > pivot_1 and idx < _d
                        ) else Int32(0)

                        if logits_vec[j] > low and idx < _d:
                            min_gt_low = min(min_gt_low, logits_vec[j])
                        if logits_vec[j] <= high and idx < _d:
                            max_le_high = max(max_le_high, logits_vec[j])

                    # Accumulate thread-local counts (no block reduction per chunk).
                    thread_count_0_total += probs_gt_pivot_0_count.reduce_add()
                    thread_count_1_total += probs_gt_pivot_1_count.reduce_add()

                # Single block reduction after processing all chunks.
                var _pivot_results = _block_reduce_pivot_bounds[block_size](
                    thread_count_0_total,
                    thread_count_1_total,
                    min_gt_low,
                    max_le_high,
                )
                var aggregate_gt_pivot_0 = _pivot_results[0]
                var aggregate_gt_pivot_1 = _pivot_results[1]
                min_gt_low = _pivot_results[2]
                max_le_high = _pivot_results[3]

                if aggregate_gt_pivot_1 >= Int32(k):
                    low = pivot_1
                elif aggregate_gt_pivot_0 >= Int32(k):
                    low = pivot_0
                    high = min(pivot_1, max_le_high)
                else:
                    high = min(pivot_0, max_le_high)

                if min_gt_low == max_le_high:
                    break

            pivot = low
        else:
            # If k >= _d, include all elements.
            var min_max = get_min_max_value[vec_size, block_size](
                logits.ptr, row_idx, _d
            )
            max_logit = min_max[1]

        barrier()

        # PHASE 2: Compute softmax sum and cache top-k elements.

        # All threads cooperatively collect elements > pivot.
        var thread_sum = Float32(0.0)

        # Use atomic counter in shared memory for write position.
        var s_write_idx = unsafe_stack_allocation[
            1, Int32, address_space=.SHARED
        ]()
        if tx == 0:
            s_write_idx[0] = Int32(0)

        barrier()

        # Each thread processes elements and atomically writes to shared memory.
        for i in range(tx, _d, block_size):
            var logit = logits_row.load[width=1]((Idx[0], i)).cast[.float32]()
            if logit > pivot:
                var exp_val = exp((logit - max_logit) / temp_val)

                # Atomically get write position and store.
                var pos = Int(Atomic.fetch_add(s_write_idx, Int32(1)))
                if pos < k:
                    s_vals[pos] = exp_val
                    s_idxs[pos] = i
                    thread_sum += exp_val

        var block_sum = block.sum[block_size=block_size, broadcast=True](
            thread_sum
        )

        barrier()

        if tx == 0:
            s_count[0] = min(Int(s_write_idx[0]), k)

        barrier()

        # PHASE 3: Sampling (thread 0 only).
        if tx == 0:
            var seed_val = seed_val
            if seed:
                seed_val = seed.unsafe_value()[bx]
            var rng_state = Random(seed=seed_val)
            var rng = rng_state.step_uniform()
            var r = block_sum * rng[0]

            var cached_count = s_count[0]
            for ki in range(cached_count):
                var exp_val = s_vals[ki]
                r -= exp_val
                if r <= 0.0 or ki == cached_count - 1:
                    sampled_indices[bx] = Scalar[out_idx_type](s_idxs[ki])
                    break


def topk_softmax_sample[
    dtype: DType,
    out_idx_type: DType,
    block_size: Int = 1024,
    TopKArrLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    TemperatureLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    SeedLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
](
    ctx: DeviceContext,
    logits: TileTensor[mut=False, dtype, address_space=.GENERIC, ...],
    sampled_indices: TileTensor[
        mut=True, out_idx_type, address_space=.GENERIC, ...
    ],
    top_k_val: Int,
    temperature_val: Float32 = 1.0,
    seed_val: UInt64 = 0,
    top_k_arr: Optional[
        TileTensor[out_idx_type, TopKArrLayoutType, MutUntrackedOrigin]
    ] = None,
    temperature: Optional[
        TileTensor[.float32, TemperatureLayoutType, MutUntrackedOrigin]
    ] = None,
    seed: Optional[
        TileTensor[.uint64, SeedLayoutType, MutUntrackedOrigin]
    ] = None,
) raises:
    """Samples token indices from top-K logits using softmax probabilities.

    This kernel performs single-pass top-K selection and categorical sampling:
    1. Finds the k-th largest logit via ternary search.
    2. Computes softmax over top-K elements and caches them in shared memory.
    3. Samples a single token index from the categorical distribution.

    Parameters:
        dtype: The data type of the input logits tensor.
        out_idx_type: The data type of the output sampled indices.
        block_size: The number of threads per block (default is 1024).
        TopKArrLayoutType: The layout type of the optional top_k_arr tensor.
        TemperatureLayoutType: The layout type of the optional temperature tensor.
        SeedLayoutType: The layout type of the optional seed tensor.

    Args:
        ctx: DeviceContext
            The context for GPU execution.
        logits:
            Input logits tensor with shape [batch_size, vocab_size].
        sampled_indices:
            Output buffer for sampled token indices with shape [batch_size].
        top_k_val: Int
            Default number of top elements to sample from for each batch element.
        temperature_val: Float32
            Temperature for softmax scaling (default is 1.0).
        seed_val: UInt64
            Seed for the random number generator (default is 0).
        top_k_arr:
            Optional per-batch top-K values. If provided, overrides top_k_val
            for each batch element.
        temperature:
            Optional per-batch temperature values. If provided, overrides
            temperature_val for each batch element.
        seed:
            Optional per-batch seed values. If provided, overrides seed_val
            for each batch element.
    """
    comptime assert logits.rank == 2, "logits rank must be 2"
    comptime assert sampled_indices.rank == 1, "sampled_indices rank must be 1"

    var shape = coord_to_index_list(logits.layout.shape_coord())
    var batch_size = shape[0]
    var d = shape[1]

    def trace_information() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("logits", shape, dtype),
                    "top_k_val=" + String(top_k_val),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        "topk_softmax_sample",
        Trace[TraceLevel.OP]._get_detail_str(trace_information),
        task_id=Int(ctx.id()),
    ):
        var out_shape = coord_to_index_list(
            sampled_indices.layout.shape_coord()
        )
        if shape[0] != out_shape[0]:
            raise Error("sampled_indices shape must be [batch_size]")

        # Use up to 16 elements per vector to minimize the number of chunks
        # (and therefore the number of block-level reductions in inner loops).
        # GPU vector loads handle wider-than-native SIMD efficiently, and the
        # per-element idx < d guard handles non-aligned tails correctly.
        var vec_size = gcd(8, d)

        var k_rounded = align_up(top_k_val, WARP_SIZE)
        var shared_mem_bytes = k_rounded * (size_of[Float32]() + size_of[Int]())
        comptime if has_apple_gpu_accelerator():
            if shared_mem_bytes > _APPLE_STATIC_SHMEM_CACHE_BYTES:
                raise Error(
                    t"shared memory of {shared_mem_bytes} exceeds static"
                    t" allocation capacity of"
                    t" {_APPLE_STATIC_SHMEM_CACHE_BYTES} when evaluating"
                    t" topk_softmax_sample with top_k_val={top_k_val}"
                    t" and vec_size={vec_size}. Consider reducing"
                    t" top_k_val or using a smaller block_size."
                )

        var top_k_ptr: Optional[
            UnsafePointer[Scalar[out_idx_type], MutUntrackedOrigin]
        ] = None
        if top_k_arr:
            top_k_ptr = top_k_arr.unsafe_value().ptr

        var temp_ptr: Optional[
            UnsafePointer[Float32, MutUntrackedOrigin]
        ] = None
        if temperature:
            temp_ptr = temperature.unsafe_value().ptr

        var seed_ptr: Optional[UnsafePointer[UInt64, MutUntrackedOrigin]] = None
        if seed:
            seed_ptr = seed.unsafe_value().ptr

        @__parameter
        def launch_kernel[vec_size: Int]() raises:
            comptime kernel = topk_softmax_sample_kernel[
                block_size,
                vec_size,
                dtype,
                out_idx_type,
                LogitsLayoutType=logits.LayoutType,
                logits_origin=ImmOrigin(logits.origin),
                SampledLayoutType=sampled_indices.LayoutType,
                sampled_origin=sampled_indices.origin,
            ]
            ctx.enqueue_function[kernel](
                logits.as_immut(),
                sampled_indices,
                top_k_ptr,
                Int32(top_k_val),
                temperature_val,
                temp_ptr,
                seed_val,
                seed_ptr,
                Int32(d),
                grid_dim=batch_size,
                block_dim=block_size,
                shared_mem_bytes=shared_mem_bytes,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )

        # Runtime dispatch to compile-time parameter.
        comptime for param_vec_size in [16, 8, 4, 2, 1]:
            if vec_size == param_vec_size:
                return launch_kernel[param_vec_size]()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
@__name(t"topk_topp_masked_probs_{dtype}_{coop_size}")
def TopKTopPMaskedProbsKernel[
    block_size: Int,
    vec_size: Int,
    dtype: DType,
    LogitsLayoutType: TensorLayout,
    logits_origin: ImmOrigin,
    coop_size: Int = 1,
](
    logits: TileTensor[dtype, LogitsLayoutType, logits_origin],
    probs_ptr: UnsafePointer[Float32, MutAnyOrigin],
    top_k_arr: Optional[UnsafePointer[Int64, ImmutAnyOrigin]],
    top_k_val: Int32,
    top_p_arr: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
    top_p_val: Float32,
    temperature: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
    d: Int32,
    coop_ws: Optional[UnsafePointer[Int32, MutAnyOrigin]],
):
    """Writes each row's top-k/top-p masked softmax.

    With `coop_size` above one, `coop_size` blocks share a row: each owns a
    contiguous slice and combines whole-row statistics through `coop_ws`. The
    launch must keep every block in a group resident.

    Parameters:
        block_size: Number of threads per block.
        vec_size: Number of elements each thread loads per access.
        dtype: Element type of `logits`.
        LogitsLayoutType: Memory layout of the `logits` tile.
        logits_origin: Origin tag for the immutable `logits` tile.
        coop_size: Blocks sharing each row; 1 keeps the whole row in one
            block and compiles the cross-block traffic away.

    Args:
        logits: Input logits [batch_size, d].
        probs_ptr: Output masked renormalized distribution [batch_size, d].
        top_k_arr: Optional per-row top-k [batch_size].
        top_k_val: Default top-k if `top_k_arr` is null.
        top_p_arr: Optional per-row top-p [batch_size].
        top_p_val: Default top-p if `top_p_arr` is null.
        temperature: Optional per-row temperature [batch_size].
        d: Vocabulary size.
        coop_ws: Zeroed cross-block workspace; required when `coop_size`
            exceeds one, ignored otherwise.
    """
    comptime assert (
        not is_apple_gpu()
    ), "TopKTopPMaskedProbsKernel is not supported on Apple GPUs"
    var _d = Int(d)
    var n_vec = _d // vec_size
    var rank = Int(block_idx.x) % coop_size
    var bx = Int(block_idx.x) // coop_size
    var tx = thread_idx.x

    # Contiguous slices make block-rank order match token order.
    var slice_vec = ceildiv(n_vec, coop_size)
    var vec_begin = min(rank * slice_vec, n_vec)
    var vec_end = min(vec_begin + slice_vec, n_vec)

    # The single-block specialization never dereferences `coop_ws`.
    var coop = CoopRow[coop_size](bx, rank)
    var coop_table = unsafe_stack_allocation[
        coop_size * COOP_SLOT_FLOATS,
        Float32,
        alignment=16,
        address_space=.SHARED,
    ]()

    wait_on_dependent_grids()
    launch_dependent_grids()

    var k = Int(top_k_val)
    if top_k_arr:
        k = Int(top_k_arr.unsafe_value().load(bx))
    if k <= 0 or k > _d:
        k = _d

    var p = top_p_val
    if top_p_arr:
        p = top_p_arr.unsafe_value()[bx]
    p = p.clamp(Float32(0.0), Float32(1.0))

    var temp_val = Float32(1.0)
    if temperature:
        temp_val = temperature.unsafe_value()[bx]
    # Clamp so a greedy (T=0) row cannot divide by zero.
    var inv_temp = 1.0 / max(temp_val, Float32(1e-6))

    var logits_row = TileTensor(logits.ptr + bx * _d, row_major(Idx[1], _d))

    # Every block needs the same row maximum because the search contains
    # group barriers.
    var thread_max = Float32.MIN
    for i in range(vec_begin + Int(tx), vec_end, block_size):
        var v = logits_row.load[width=vec_size]((Idx[0], i * vec_size)).cast[
            .float32
        ]()
        thread_max = max(thread_max, v.reduce_max())
    var m = block.max[block_size=block_size, broadcast=True](thread_max)
    comptime if coop_size > 1:
        m = coop.combine[1, _coop_max](
            coop_ws.unsafe_value(), coop_table, SIMD[.float32, 1](m)
        )[0]

    @__parameter
    @always_inline
    def compute_e(offset: Int) -> SIMD[.float32, vec_size]:
        var v = logits_row.load[width=vec_size]((Idx[0], offset)).cast[
            .float32
        ]()
        return exp((v - m) * inv_temp)

    var probs_row = TileTensor(probs_ptr + bx * _d, row_major(Idx[1], _d))

    # Total mass, plus how many tokens carry any. At most k positive tokens
    # need no top-k search: cutoff zero already keeps the full positive set.
    var thread_sum = Float32(0)
    var thread_pos: Int32 = 0
    for i in range(vec_begin + Int(tx), vec_end, block_size):
        var e = compute_e(i * vec_size)
        # Repeated exponentiation costs more than the wider cache traffic on
        # AMD; each thread reuses its own output slice until the final write.
        comptime if is_amd_gpu():
            probs_row.store[width=vec_size]((Idx[0], i * vec_size), e)
        thread_sum += e.reduce_add()
        if k < _d:
            comptime for j in range(vec_size):
                if e[j] > 0:
                    thread_pos += 1

    var z: Float32
    var positive_count = Int32(0)
    if k < _d:
        var total = _block_reduce_value_count[.float32, broadcast=True](
            ValueCount[.float32](thread_sum, thread_pos)
        )
        z = total.value
        positive_count = total.count
    else:
        z = block.sum[block_size=block_size, broadcast=True](thread_sum)
    comptime if coop_size > 1:
        # `k` is identical across the group, so this conditional is collective.
        var totals = coop.combine[2, _coop_sum](
            coop_ws.unsafe_value(),
            coop_table,
            SIMD[.float32, 2](z, Float32(positive_count)),
        )
        z = totals[0]
        positive_count = Int32(totals[1])
    var p_eff = _topp_budget(p, z)

    @__parameter
    @always_inline
    def load_e(offset: Int) -> SIMD[.float32, vec_size]:
        comptime if is_amd_gpu():
            return probs_row.load[width=vec_size]((Idx[0], offset))
        else:
            return compute_e(offset)

    var cut = Float32(0)
    var mass_s = z
    if positive_count > Int32(k) or z > p_eff:
        var refined: Tuple[Float32, Float32]
        if k == _d:
            refined = _topk_topp_cutoff_search[
                vec_size,
                block_size,
                load_e,
                track_count=False,
                coop_size=coop_size,
            ](
                _d,
                Int32(k),
                p_eff,
                0.0,
                1.0,
                z,
                coop,
                coop_ws,
                coop_table,
                vec_begin,
                vec_end,
            )
        else:
            refined = _topk_topp_cutoff_search[
                vec_size, block_size, load_e, coop_size=coop_size
            ](
                _d,
                Int32(k),
                p_eff,
                0.0,
                1.0,
                z,
                coop,
                coop_ws,
                coop_table,
                vec_begin,
                vec_end,
            )
        cut = refined[0]
        mass_s = refined[1]

    for i in range(vec_begin + Int(tx), vec_end, block_size):
        var e = load_e(i * vec_size)
        var masked = (e.gt(cut)).select(e / mass_s, SIMD[.float32, vec_size](0))
        probs_row.store[width=vec_size]((Idx[0], i * vec_size), masked)


def topk_topp_masked_probs[
    dtype: DType,
    block_size: Int = 1024,
    TopKArrLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    TopPArrLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    TemperatureLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64].element_types,
        stride_types=Coord[ComptimeInt[1]].element_types,
    ],
    ProbsLayoutType: TensorLayout = Layout[
        shape_types=Coord[Int64, Int64].element_types,
        stride_types=Coord[Int64, ComptimeInt[1]].element_types,
    ],
    TopKArrEngine: TensorEngine = DefaultEngine[element_width=1],
    TopPArrEngine: TensorEngine = DefaultEngine[element_width=1],
    TemperatureEngine: TensorEngine = DefaultEngine[element_width=1],
](
    ctx: DeviceContext,
    logits: TileTensor[mut=False, dtype, ...],
    probs: TileTensor[.float32, ProbsLayoutType, MutAnyOrigin],
    top_k_val: Int,
    top_p_val: Float32 = 1.0,
    top_k_arr: Optional[
        TileTensor[
            .int64, TopKArrLayoutType, ImmutAnyOrigin, Engine=TopKArrEngine
        ]
    ] = None,
    top_p_arr: Optional[
        TileTensor[
            .float32, TopPArrLayoutType, ImmutAnyOrigin, Engine=TopPArrEngine
        ]
    ] = None,
    temperature: Optional[
        TileTensor[
            .float32,
            TemperatureLayoutType,
            ImmutAnyOrigin,
            Engine=TemperatureEngine,
        ]
    ] = None,
) raises:
    """Computes per-row top-k/top-p masked softmax.

    See `TopKTopPMaskedProbsKernel` for what the output means.

    Parameters:
        dtype: Element type of `logits`.
        block_size: Threads per block.
        TopKArrLayoutType: Memory layout of `top_k_arr`.
        TopPArrLayoutType: Memory layout of `top_p_arr`.
        TemperatureLayoutType: Memory layout of `temperature`.
        ProbsLayoutType: Memory layout of `probs`.
        TopKArrEngine: Engine policy of `top_k_arr`.
        TopPArrEngine: Engine policy of `top_p_arr`.
        TemperatureEngine: Engine policy of `temperature`.

    Args:
        ctx: Device context.
        logits: Input logits [batch_size, d].
        probs: Output masked renormalized distribution [batch_size, d].
        top_k_val: Default top-k; `<= 0` or `> d` keeps every token.
        top_p_val: Default top-p threshold.
        top_k_arr: Optional per-row top-k [batch_size].
        top_p_arr: Optional per-row top-p [batch_size].
        temperature: Optional per-row temperature [batch_size]; 0 is clamped.

    Raises:
        Error: If the tensor shapes disagree.
    """
    comptime assert logits.rank == 2, "logits rank must be 2"

    var shape = coord_to_index_list(logits.layout.shape_coord())
    var batch_size = shape[0]
    var d = shape[1]

    def trace_information() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("logits", shape, dtype),
                    "top_k=" + String(top_k_val),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        "topk_topp_masked_probs",
        Trace[TraceLevel.OP]._get_detail_str(trace_information),
        task_id=Int(ctx.id()),
    ):
        var probs_shape = coord_to_index_list(probs.layout.shape_coord())
        if probs_shape[0] != batch_size or probs_shape[1] != d:
            raise Error("probs shape must match the logits shape")

        # Speculative decoding runs this with zero rows on every step that has
        # no drafts to verify, and a grid of 0 is not a legal launch.
        if batch_size == 0:
            return

        var vec_size = gcd(8, d)

        var top_k_ptr: Optional[UnsafePointer[Int64, ImmutAnyOrigin]] = None
        if top_k_arr:
            top_k_ptr = top_k_arr.unsafe_value().ptr

        var top_p_ptr: Optional[UnsafePointer[Float32, ImmutAnyOrigin]] = None
        if top_p_arr:
            top_p_ptr = top_p_arr.unsafe_value().ptr

        var temperature_ptr: Optional[
            UnsafePointer[Float32, ImmutAnyOrigin]
        ] = None
        if temperature:
            temperature_ptr = temperature.unsafe_value().ptr

        comptime coop_capable = has_amd_gpu_accelerator()
        var coop = 1
        comptime if coop_capable:
            coop = coop_group_size(ctx, batch_size, block_size, d // vec_size)
        var coop_buf = ctx.enqueue_create_buffer[.int32](
            coop_row_words(batch_size, coop) if coop > 1 else 1
        )
        var coop_ptr: Optional[UnsafePointer[Int32, MutAnyOrigin]] = None
        if coop > 1:
            # Barrier counters must start at zero.
            ctx.enqueue_memset(coop_buf, 0)
            coop_ptr = coop_buf.unsafe_ptr().as_unsafe_any_origin()

        @__parameter
        def launch_kernel[vec_size: Int, coop_size: Int]() raises:
            comptime kernel = TopKTopPMaskedProbsKernel[
                block_size,
                vec_size,
                dtype,
                logits.LayoutType,
                ImmOrigin(logits.origin),
                coop_size,
            ]
            ctx.enqueue_function[kernel](
                logits.as_immut(),
                probs.ptr,
                top_k_ptr,
                Int32(top_k_val),
                top_p_ptr,
                top_p_val,
                temperature_ptr,
                Int32(d),
                coop_ptr,
                grid_dim=batch_size * coop_size,
                block_dim=block_size,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )

        comptime for param_vec_size in [16, 8, 4, 2, 1]:
            if vec_size == param_vec_size:
                comptime if coop_capable:
                    comptime for param_coop in [2, 4, 8, 16]:
                        if coop == param_coop:
                            launch_kernel[param_vec_size, param_coop]()
                            _ = coop_buf^
                            return
                launch_kernel[param_vec_size, 1]()
                _ = coop_buf^
                return
