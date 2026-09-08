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
"""GPU block-level operations and utilities.

This module provides block-level operations for NVIDIA and AMD GPUs, including:

- Block-wide reductions:
  - sum: Compute sum across block
  - max: Find maximum value across block
  - min: Find minimum value across block
  - broadcast: Broadcast value to all threads

The module builds on warp-level operations from the warp module, extending them
to work across a full thread block (potentially multiple warps). It handles both
NVIDIA and AMD GPU architectures and supports various data types with SIMD
vectorization.

All operations support 1D blocks via the `block_size` parameter, as well as 2D
and 3D blocks via the `block_dim_x`, `block_dim_y`, and `block_dim_z` parameters.
For multi-dimensional blocks, thread linearization follows the standard row-major
order: `linear_id = x + y * dim_x + z * dim_x * dim_y`.
"""

from std.math import align_up, ceildiv
from std.math.uutils import ufloordiv

from std.memory import unsafe_stack_allocation
from std.utils.static_tuple import StaticTuple

from std._gpu import WARP_SIZE, lane_id, thread_idx, warp_id
from std._gpu.primitives import warp
from std.sys.info import is_apple_gpu

from max.gpu import barrier

# ===-----------------------------------------------------------------------===#
# Block Reduction Core
# ===-----------------------------------------------------------------------===#


@always_inline
def _block_reduce_with_padding[
    dtype: DType,
    num_reductions: Int,
    //,
    *,
    n_warps: Int,
    padding: Int,
    warp_reduce_fn: def[dtype: DType, width: SIMDLength, reduction_idx: Int](
        SIMD[dtype, width]
    ) capturing[_] -> Scalar[dtype],
    broadcast: Bool = False,
](
    vals: StaticTuple[Scalar[dtype], num_reductions],
    *,
    initial_vals: StaticTuple[Scalar[dtype], num_reductions],
    wid: Int,
) -> StaticTuple[Scalar[dtype], num_reductions]:
    comptime smem_stride = n_warps + padding
    # Add padding to avoid bank conflicts
    var shared_mem = unsafe_stack_allocation[
        num_reductions * smem_stride, dtype, address_space=.SHARED
    ]()

    var lid = lane_id()

    # Step 1: Perform warp-level reduction for each reduction.
    var warp_results = StaticTuple[Scalar[dtype], num_reductions]()
    comptime for i in range(num_reductions):
        warp_results[i] = warp_reduce_fn[reduction_idx=i](vals[i])

    @always_inline
    def compute_offset(offset: Int) -> Int:
        """Computes the offset with the padding if needed."""

        comptime if padding > 0:
            return offset + ufloordiv(offset, WARP_SIZE)
        else:
            return offset

    # Step 2: Store warp results to shared memory with padding consideration.
    # Each leader thread (lane 0) is responsible for its warp.
    # Account for padding when storing to avoid bank conflicts.
    if lid == 0:
        comptime for i in range(num_reductions):
            shared_mem[
                unsafe_offset=i * smem_stride + compute_offset(wid)
            ] = warp_results[i]

    barrier()

    # Step 3: Have the first warp reduce all warp results.
    if wid == 0:
        comptime for i in range(num_reductions):
            # Make sure that the "ghost" warps do not contribute to the
            # reduction.
            var block_val = initial_vals[i]
            # Load values from the shared memory (ith lane will have ith
            # warp's value). Account for padding when loading.
            if lid < n_warps:
                block_val = shared_mem[
                    unsafe_offset=i * smem_stride + compute_offset(lid)
                ]

            # Reduce across the first warp
            warp_results[i] = warp_reduce_fn[reduction_idx=i](block_val)

        comptime if broadcast:
            # Store the final results back to shared memory for broadcast
            if lid == 0:
                comptime for i in range(num_reductions):
                    shared_mem[unsafe_offset=i] = warp_results[i]

    comptime if broadcast:
        # Synchronize and broadcast the results to all threads
        barrier()
        # All threads read the final results from shared memory
        comptime for i in range(num_reductions):
            warp_results[i] = shared_mem[unsafe_offset=i]

    # The trailing shared-memory reads above (warp 0's per-warp loads, and
    # every thread's broadcast load) have no barrier after them, so a warp
    # that finishes this combine can begin the caller's next one and
    # overwrite the strip while a lagging warp still reads it. On
    # NVIDIA/AMD warps never
    # drift that far in practice; Metal's threadgroup scheduling loses this
    # race readily (measured: rms_norm block tier, run-to-run divergent
    # bytes on M5). Close the reuse window on Apple only, keeping other
    # targets' codegen byte-identical.
    comptime if is_apple_gpu():
        barrier()

    return warp_results


@always_inline
def _block_reduce[
    dtype: DType,
    num_reductions: Int,
    //,
    block_dim_x: Int,
    block_dim_y: Int = 1,
    block_dim_z: Int = 1,
    *,
    warp_reduce_fn: def[dtype: DType, width: SIMDLength, reduction_idx: Int](
        SIMD[dtype, width]
    ) capturing[_] -> Scalar[dtype],
    broadcast: Bool = False,
](
    vals: StaticTuple[Scalar[dtype], num_reductions],
    *,
    initial_vals: StaticTuple[Scalar[dtype], num_reductions],
) -> StaticTuple[Scalar[dtype], num_reductions]:
    """Performs a generic block-level reduction operation.

    This function implements a block-level reduction using warp-level operations
    and shared memory for inter-warp communication. All threads in the block
    participate to compute the final reduced values. Supports 1D, 2D, and 3D
    thread blocks; thread IDs are linearized in row-major order.

    Multiple reductions can be fused into a single call by setting
    `num_reductions` > 1. The `warp_reduce_fn` receives a compile-time
    `reduction_idx` parameter to dispatch between different reduction
    operations. Fusing reductions amortizes barrier synchronization costs.

    Parameters:
        dtype: The data type of the SIMD elements.
        num_reductions: The number of fused reductions to perform.
        block_dim_x: The number of threads along the X dimension.
        block_dim_y: The number of threads along the Y dimension (default: 1).
        block_dim_z: The number of threads along the Z dimension (default: 1).
        warp_reduce_fn: A function that performs warp-level reduction. Receives
            a compile-time `reduction_idx` parameter to select the reduction
            operation.
        broadcast: If True, the final reduced values are broadcast to all
            threads in the block. If False, only the first thread will have the
            complete results.

    Args:
        vals: The input values from each thread, one per reduction.
        initial_vals: The initial values for each reduction.

    Returns:
        A `StaticTuple` of reduced values. If broadcast is True, each thread
        in the block will receive the reduced values. Otherwise, only the first
        thread will have the complete results.
    """
    comptime block_size = block_dim_x * block_dim_y * block_dim_z
    comptime assert (
        block_size >= WARP_SIZE
    ), "Block size must be a greater than warp size"
    comptime assert (
        block_size % WARP_SIZE == 0
    ), "Block size must be a multiple of warp size"

    # Compute linearized thread and warp IDs for multi-dimensional blocks.
    # For 1D blocks (block_dim_y=1, block_dim_z=1) this reduces to
    # thread_idx.x // WARP_SIZE, matching the original warp_id() behaviour.
    var linear_tid = (
        thread_idx.x
        + thread_idx.y * block_dim_x
        + thread_idx.z * block_dim_x * block_dim_y
    )
    var wid = ufloordiv(linear_tid, WARP_SIZE)

    # Allocate shared memory for inter-warp communication.
    comptime n_warps = block_size // WARP_SIZE

    comptime if n_warps == 1:
        # Single warp optimization: no shared memory or barriers needed
        # Warp shuffle operations are sufficient and much faster
        var warp_results = StaticTuple[Scalar[dtype], num_reductions]()
        comptime for i in range(num_reductions):
            warp_results[i] = warp_reduce_fn[reduction_idx=i](vals[i])

            comptime if broadcast:
                # Use efficient warp broadcast (shuffle to lane 0)
                warp_results[i] = warp.broadcast(warp_results[i])

        return warp_results

    comptime if n_warps == 2:
        return _block_reduce_with_padding[
            n_warps=n_warps,
            padding=0,
            warp_reduce_fn=warp_reduce_fn,
            broadcast=broadcast,
        ](vals, initial_vals=initial_vals, wid=wid)

    # General case with bank conflict optimization
    # Add padding to avoid bank conflicts
    comptime padding = ceildiv(n_warps, WARP_SIZE) if n_warps > WARP_SIZE else 0
    return _block_reduce_with_padding[
        n_warps=n_warps,
        padding=padding,
        warp_reduce_fn=warp_reduce_fn,
        broadcast=broadcast,
    ](vals, initial_vals=initial_vals, wid=wid)


@always_inline
def _block_reduce[
    dtype: DType,
    //,
    block_dim_x: Int,
    block_dim_y: Int = 1,
    block_dim_z: Int = 1,
    *,
    warp_reduce_fn: def[dtype: DType, width: SIMDLength](
        SIMD[dtype, width]
    ) thin -> Scalar[dtype],
    broadcast: Bool = False,
](val: Scalar[dtype], *, initial_val: Scalar[dtype]) -> Scalar[dtype]:
    """Performs a single block-level reduction operation.

    This is a convenience overload that accepts a single warp-level reduction
    function (without a `reduction_idx` parameter) and a single value. It
    wraps the function and delegates to the multi-reduction overload.

    Parameters:
        dtype: The data type of the SIMD elements.
        block_dim_x: The number of threads along the X dimension.
        block_dim_y: The number of threads along the Y dimension (default: 1).
        block_dim_z: The number of threads along the Z dimension (default: 1).
        warp_reduce_fn: A function that performs warp-level reduction.
        broadcast: If True, the final reduced value is broadcast to all
            threads in the block. If False, only the first thread will have the
            complete result.

    Args:
        val: The input value from each thread to include in the reduction.
        initial_val: The initial value for the reduction.

    Returns:
        The reduced value. If broadcast is True, each thread in the block will
        receive the reduced value. Otherwise, only the first thread will have
        the complete result.
    """

    @always_inline
    @__parameter
    def _indexed_fn[
        dtype: DType, width: SIMDLength, reduction_idx: Int
    ](v: SIMD[dtype, width]) -> Scalar[dtype]:
        return warp_reduce_fn(v)

    return _block_reduce[
        block_dim_x,
        block_dim_y,
        block_dim_z,
        warp_reduce_fn=_indexed_fn,
        broadcast=broadcast,
    ](
        StaticTuple[Scalar[dtype], 1](val),
        initial_vals=StaticTuple[Scalar[dtype], 1](initial_val),
    )[
        0
    ]


# ===-----------------------------------------------------------------------===#
# Block Sum
# ===-----------------------------------------------------------------------===#


@always_inline
def sum[
    dtype: DType,
    width: SIMDLength,
    //,
    *,
    block_size: Int,
    broadcast: Bool = True,
](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Computes the sum of values across all threads in a block.

    Performs a parallel reduction using warp-level operations and shared memory
    to find the global sum across all threads in the block.

    Parameters:
        dtype: The data type of the SIMD elements.
        width: The number of elements in each SIMD vector.
        block_size: The total number of threads in the block.
        broadcast: If True, the final sum is broadcast to all threads in the
            block. If False, only the first thread will have the complete sum.

    Args:
        val: The SIMD value to reduce. Each thread contributes its value to the
             sum.

    Returns:
        If broadcast is True, each thread in the block will receive the final
        sum. Otherwise, only the first thread will have the complete sum.
    """

    return _block_reduce[
        block_size, warp_reduce_fn=warp.sum, broadcast=broadcast
    ](val.reduce_add(), initial_val=0)


@always_inline
def sum[
    dtype: DType,
    width: SIMDLength,
    //,
    *,
    block_dim_x: Int,
    block_dim_y: Int,
    block_dim_z: Int = 1,
    broadcast: Bool = True,
](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Computes the sum of values across all threads in a multi-dimensional block.

    Performs a parallel reduction using warp-level operations and shared memory
    to find the global sum across all threads in the block. Thread IDs are
    linearized in row-major order: `x + y * dim_x + z * dim_x * dim_y`.

    Parameters:
        dtype: The data type of the SIMD elements.
        width: The number of elements in each SIMD vector.
        block_dim_x: The number of threads along the X dimension.
        block_dim_y: The number of threads along the Y dimension.
        block_dim_z: The number of threads along the Z dimension (default: 1).
        broadcast: If True, the final sum is broadcast to all threads in the
            block. If False, only the first thread will have the complete sum.

    Args:
        val: The SIMD value to reduce. Each thread contributes its value to the
             sum.

    Returns:
        If broadcast is True, each thread in the block will receive the final
        sum. Otherwise, only the first thread will have the complete sum.
    """
    return _block_reduce[
        block_dim_x,
        block_dim_y,
        block_dim_z,
        warp_reduce_fn=warp.sum,
        broadcast=broadcast,
    ](val.reduce_add(), initial_val=0)


# ===-----------------------------------------------------------------------===#
# Block Max
# ===-----------------------------------------------------------------------===#


@always_inline
def max[
    dtype: DType,
    width: SIMDLength,
    //,
    *,
    block_size: Int,
    broadcast: Bool = True,
](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Computes the maximum value across all threads in a block.

    Performs a parallel reduction using warp-level operations and shared memory
    to find the global maximum across all threads in the block.

    Parameters:
        dtype: The data type of the SIMD elements.
        width: The number of elements in each SIMD vector.
        block_size: The total number of threads in the block.
        broadcast: If True, the final reduced value is broadcast to all
            threads in the block. If False, only the first thread will have the
            complete result.

    Args:
        val: The SIMD value to reduce. Each thread contributes its value to find
             the maximum.

    Returns:
        If broadcast is True, each thread in the block will receive the maximum
        value across the entire block. Otherwise, only the first thread will
        have the complete result.
    """

    return _block_reduce[
        block_size, warp_reduce_fn=warp.max, broadcast=broadcast
    ](val.reduce_max(), initial_val=Scalar[dtype].MIN_FINITE)


@always_inline
def max[
    dtype: DType,
    width: SIMDLength,
    //,
    *,
    block_dim_x: Int,
    block_dim_y: Int,
    block_dim_z: Int = 1,
    broadcast: Bool = True,
](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Computes the maximum value across all threads in a multi-dimensional block.

    Performs a parallel reduction using warp-level operations and shared memory
    to find the global maximum across all threads in the block. Thread IDs are
    linearized in row-major order: `x + y * dim_x + z * dim_x * dim_y`.

    Parameters:
        dtype: The data type of the SIMD elements.
        width: The number of elements in each SIMD vector.
        block_dim_x: The number of threads along the X dimension.
        block_dim_y: The number of threads along the Y dimension.
        block_dim_z: The number of threads along the Z dimension (default: 1).
        broadcast: If True, the final reduced value is broadcast to all
            threads in the block. If False, only the first thread will have the
            complete result.

    Args:
        val: The SIMD value to reduce. Each thread contributes its value to find
             the maximum.

    Returns:
        If broadcast is True, each thread in the block will receive the maximum
        value across the entire block. Otherwise, only the first thread will
        have the complete result.
    """
    return _block_reduce[
        block_dim_x,
        block_dim_y,
        block_dim_z,
        warp_reduce_fn=warp.max,
        broadcast=broadcast,
    ](val.reduce_max(), initial_val=Scalar[dtype].MIN_FINITE)


# ===-----------------------------------------------------------------------===#
# Block Min
# ===-----------------------------------------------------------------------===#


@always_inline
def min[
    dtype: DType,
    width: SIMDLength,
    //,
    *,
    block_size: Int,
    broadcast: Bool = True,
](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Computes the minimum value across all threads in a block.

    Performs a parallel reduction using warp-level operations and shared memory
    to find the global minimum across all threads in the block.

    Parameters:
        dtype: The data type of the SIMD elements.
        width: The number of elements in each SIMD vector.
        block_size: The total number of threads in the block.
        broadcast: If True, the final minimum is broadcast to all threads in the
            block. If False, only the first thread will have the complete min.

    Args:
        val: The SIMD value to reduce. Each thread contributes its value to find
             the minimum.

    Returns:
        If broadcast is True, each thread in the block will receive the minimum
        value across the entire block. Otherwise, only the first thread will
        have the complete result.
    """

    return _block_reduce[
        block_size, warp_reduce_fn=warp.min, broadcast=broadcast
    ](val.reduce_min(), initial_val=Scalar[dtype].MAX_FINITE)


@always_inline
def min[
    dtype: DType,
    width: SIMDLength,
    //,
    *,
    block_dim_x: Int,
    block_dim_y: Int,
    block_dim_z: Int = 1,
    broadcast: Bool = True,
](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Computes the minimum value across all threads in a multi-dimensional block.

    Performs a parallel reduction using warp-level operations and shared memory
    to find the global minimum across all threads in the block. Thread IDs are
    linearized in row-major order: `x + y * dim_x + z * dim_x * dim_y`.

    Parameters:
        dtype: The data type of the SIMD elements.
        width: The number of elements in each SIMD vector.
        block_dim_x: The number of threads along the X dimension.
        block_dim_y: The number of threads along the Y dimension.
        block_dim_z: The number of threads along the Z dimension (default: 1).
        broadcast: If True, the final minimum is broadcast to all threads in the
            block. If False, only the first thread will have the complete min.

    Args:
        val: The SIMD value to reduce. Each thread contributes its value to find
             the minimum.

    Returns:
        If broadcast is True, each thread in the block will receive the minimum
        value across the entire block. Otherwise, only the first thread will
        have the complete result.
    """
    return _block_reduce[
        block_dim_x,
        block_dim_y,
        block_dim_z,
        warp_reduce_fn=warp.min,
        broadcast=broadcast,
    ](val.reduce_min(), initial_val=Scalar[dtype].MAX_FINITE)


# ===-----------------------------------------------------------------------===#
# Block Broadcast
# ===-----------------------------------------------------------------------===#


@always_inline
def broadcast[
    dtype: DType, width: SIMDLength, //, *, block_size: Int
](val: SIMD[dtype, width], src_thread: Int = 0) -> SIMD[dtype, width]:
    """Broadcasts a value from a source thread to all threads in a block.

    This function takes a SIMD value from the specified source thread and
    copies it to all other threads in the block, effectively broadcasting
    the value across the entire block.

    Parameters:
        dtype: The data type of the SIMD elements.
        width: The number of elements in each SIMD vector.
        block_size: The total number of threads in the block.

    Args:
        val: The SIMD value to broadcast from the source thread.
        src_thread: The thread ID of the source thread (default: 0).

    Returns:
        A SIMD value where all threads contain a copy of the input value from
        the source thread.
    """
    comptime assert (
        block_size >= WARP_SIZE
    ), "Block size must be greater than or equal to warp size"
    comptime assert (
        block_size % WARP_SIZE == 0
    ), "Block size must be a multiple of warp size"

    comptime if block_size == WARP_SIZE:
        # Single warp - use warp shuffle for better performance
        return warp.broadcast(val)

    # Multi-warp block - use shared memory
    var shared_mem = unsafe_stack_allocation[
        width, dtype, address_space=.SHARED
    ]()

    # Source thread writes its value to shared memory
    if thread_idx.x == Int(src_thread):
        shared_mem.unsafe_store(val)

    barrier()

    # All threads read the same value from shared memory
    return shared_mem.unsafe_load[width=width]()


@always_inline
def broadcast[
    dtype: DType,
    width: SIMDLength,
    //,
    *,
    block_dim_x: Int,
    block_dim_y: Int,
    block_dim_z: Int = 1,
](val: SIMD[dtype, width], src_thread: Int = 0) -> SIMD[dtype, width]:
    """Broadcasts a value from a source thread to all threads in a multi-dimensional block.

    This function takes a SIMD value from the specified source thread (identified
    by its linearized thread ID) and copies it to all other threads in the block.
    Thread IDs are linearized in row-major order: `x + y * dim_x + z * dim_x * dim_y`.

    Parameters:
        dtype: The data type of the SIMD elements.
        width: The number of elements in each SIMD vector.
        block_dim_x: The number of threads along the X dimension.
        block_dim_y: The number of threads along the Y dimension.
        block_dim_z: The number of threads along the Z dimension (default: 1).

    Args:
        val: The SIMD value to broadcast from the source thread.
        src_thread: The linearized thread ID of the source thread (default: 0).

    Returns:
        A SIMD value where all threads contain a copy of the input value from
        the source thread.
    """
    comptime block_size = block_dim_x * block_dim_y * block_dim_z
    comptime assert (
        block_size >= WARP_SIZE
    ), "Block size must be greater than or equal to warp size"
    comptime assert (
        block_size % WARP_SIZE == 0
    ), "Block size must be a multiple of warp size"

    comptime if block_size == WARP_SIZE:
        return warp.broadcast(val)

    var shared_mem = unsafe_stack_allocation[
        width, dtype, address_space=.SHARED
    ]()

    var linear_tid = (
        thread_idx.x
        + thread_idx.y * block_dim_x
        + thread_idx.z * block_dim_x * block_dim_y
    )
    if linear_tid == Int(src_thread):
        shared_mem.unsafe_store(val)

    barrier()

    return shared_mem.unsafe_load[width=width]()


# ===-----------------------------------------------------------------------===#
# Block Prefix Sum
# ===-----------------------------------------------------------------------===#


@always_inline
def _prefix_sum[
    dtype: DType,
    //,
    *,
    block_size: Int,
    exclusive: Bool = False,
](val: Scalar[dtype], *, wid: Int) -> Scalar[dtype]:
    """Performs a prefix sum (scan) operation across all threads in a block."""
    comptime assert (
        block_size % WARP_SIZE == 0
    ), "Block size must be a multiple of warp size"

    # Allocate shared memory for inter-warp communication
    # We need one slot per warp to store warp-level scan results
    comptime n_warps = block_size // WARP_SIZE
    var warp_mem = unsafe_stack_allocation[
        align_up(n_warps, WARP_SIZE), dtype, address_space=.SHARED
    ]()

    var thread_result = warp.prefix_sum[exclusive=exclusive](val)

    # Step 2: Store last value from each warp to shared memory
    if lane_id() == WARP_SIZE - 1:
        var inclusive_warp_sum: Scalar[dtype] = thread_result

        comptime if exclusive:
            # For exclusive scan, thread_result is the sum of elements 0 to
            # WARP_SIZE-2. 'val' is the value of the element at WARP_SIZE-1.
            # Adding them gives the inclusive sum of the warp.
            inclusive_warp_sum += val

        warp_mem[unsafe_offset=wid] = inclusive_warp_sum

    barrier()

    # Step 3: Have the first warp perform a scan on the warp results
    var lid = lane_id()
    if wid == 0:
        var previous_warps_prefix = warp.prefix_sum[exclusive=False](
            warp_mem[unsafe_offset=lid]
        )
        if lid < n_warps:
            warp_mem[unsafe_offset=lid] = previous_warps_prefix
    barrier()

    # Step 4: Add the prefix from previous warps
    if wid > 0:
        thread_result += warp_mem[unsafe_offset=wid - 1]

    return thread_result


@always_inline
def prefix_sum[
    dtype: DType,
    //,
    *,
    block_size: Int,
    exclusive: Bool = False,
](val: Scalar[dtype]) -> Scalar[dtype]:
    """Performs a prefix sum (scan) operation across all threads in a 1D block.

    This function implements a block-level inclusive or exclusive scan,
    efficiently computing the cumulative sum for each thread based on
    thread indices.

    Parameters:
        dtype: The data type of the Scalar elements.
        block_size: The total number of threads in the block.
        exclusive: If True, perform exclusive scan instead of inclusive.

    Args:
        val: The Scalar value from each thread to include in the scan.

    Returns:
        A Scalar value containing the result of the scan operation for each
        thread.
    """
    return _prefix_sum[block_size=block_size, exclusive=exclusive](
        val, wid=warp_id()
    )


@always_inline
def prefix_sum[
    dtype: DType,
    //,
    *,
    block_dim_x: Int,
    block_dim_y: Int,
    block_dim_z: Int = 1,
    exclusive: Bool = False,
](val: Scalar[dtype]) -> Scalar[dtype]:
    """Performs a prefix sum (scan) operation across all threads in a multi-dimensional block.

    This function implements a block-level inclusive or exclusive scan for 2D
    and 3D thread blocks. Thread IDs are linearized in row-major order:
    `x + y * dim_x + z * dim_x * dim_y`.

    Parameters:
        dtype: The data type of the Scalar elements.
        block_dim_x: The number of threads along the X dimension.
        block_dim_y: The number of threads along the Y dimension.
        block_dim_z: The number of threads along the Z dimension (default: 1).
        exclusive: If True, perform exclusive scan instead of inclusive.

    Args:
        val: The Scalar value from each thread to include in the scan.

    Returns:
        A Scalar value containing the result of the scan operation for each
        thread.
    """
    comptime block_size = block_dim_x * block_dim_y * block_dim_z
    var linear_tid = (
        thread_idx.x
        + thread_idx.y * block_dim_x
        + thread_idx.z * block_dim_x * block_dim_y
    )
    return _prefix_sum[block_size=block_size, exclusive=exclusive](
        val, wid=ufloordiv(linear_tid, WARP_SIZE)
    )
