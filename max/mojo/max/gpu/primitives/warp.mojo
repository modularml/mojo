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
"""GPU warp-level operations and utilities.

This module provides warp-level operations for NVIDIA and AMD GPUs, including:

- Shuffle operations to exchange values between threads in a warp:
  - shuffle_idx: Copy value from source lane to other lanes
  - shuffle_up: Copy from lower lane IDs
  - shuffle_down: Copy from higher lane IDs
  - shuffle_xor: Exchange values in butterfly pattern

- Warp-wide reductions:
  - sum: Compute sum across warp
  - max: Find maximum value across warp
  - min: Find minimum value across warp
  - broadcast: Broadcast value to all lanes

The module handles both NVIDIA and AMD GPU architectures through architecture-specific
implementations of the core operations. It supports various data types including
integers, floats, and half-precision floats, with SIMD vectorization.
"""

import std._gpu.primitives.warp
from std._gpu.globals import WARP_SIZE
from std._gpu.primitives.warp import (
    _ReduceFn,
    # Reached directly by the MSA top-k kernels and the SM100 attention
    # kernels; a private name needs an explicit re-export.
    _dpp_move,
    _vote_nvidia_helper,
)


@always_inline("nodebug")
def shuffle_idx[
    dtype: DType, simd_width: SIMDLength, //
](val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[dtype, simd_width]:
    """Copies a value from a source lane to other lanes in a warp.

        Broadcasts a value from a source thread in a warp to all participating threads
        without using shared memory. This is a convenience wrapper that uses the full
        warp mask by default.

    Parameters:
        dtype: The data type of the SIMD elements (e.g. float32, int32, half).
        simd_width: The number of elements in each SIMD vector.

    Args:
        val: The SIMD value to be broadcast from the source lane.
        offset: The source lane ID to copy the value from.

    Returns:
        A SIMD vector where all lanes contain the value from the source lane specified by offset.

    Example:

        ```mojo
            from max.gpu.primitives.warp import shuffle_idx

            val = SIMD[.float32, 16](1.0)

            # Broadcast value from lane 0 to all lanes
            result = shuffle_idx(val, 0)

            # Get value from lane 5
            result = shuffle_idx(val, 5)
        ```
    """
    return std._gpu.primitives.warp.shuffle_idx(val, offset)


@always_inline("nodebug")
def shuffle_idx[
    dtype: DType, simd_width: SIMDLength, //
](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
    dtype, simd_width
]:
    """Copies a value from a source lane to other lanes in a warp with explicit mask control.

        Broadcasts a value from a source thread in a warp to participating threads specified by
        the mask. This provides fine-grained control over which threads participate in the shuffle
        operation.

    Parameters:
        dtype: The data type of the SIMD elements (e.g. float32, int32, half).
        simd_width: The number of elements in each SIMD vector.

    Args:
        mask: A bit mask specifying which lanes participate in the shuffle (1 bit per lane).
        val: The SIMD value to be broadcast from the source lane.
        offset: The source lane ID to copy the value from.

    Returns:
        A SIMD vector where participating lanes (set in mask) contain the value from the
        source lane specified by offset. Non-participating lanes retain their original values.

    Example:

        ```mojo
            from max.gpu.primitives.warp import shuffle_idx

            # Only broadcast to first 16 lanes
            var mask: UInt = 0xFFFF  # 16 ones
            var val = SIMD[.float32, 32](1.0)
            var result = shuffle_idx(mask, val, 5)
        ```
    """
    return std._gpu.primitives.warp.shuffle_idx(mask, val, offset)


@always_inline("nodebug")
def shuffle_up[
    dtype: DType, simd_width: SIMDLength, //
](val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[dtype, simd_width]:
    """Copies values from threads with lower lane IDs in the warp.

    Performs a shuffle operation where each thread receives a value from a thread with a
    lower lane ID, offset by the specified amount. Uses the full warp mask by default.

    For example, with offset=1:
    - Thread N gets value from thread N-1
    - Thread 1 gets value from thread 0
    - Thread 0 gets undefined value

    Parameters:
        dtype: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in each SIMD vector.

    Args:
        val: The SIMD value to be shuffled up the warp.
        offset: The number of lanes to shift values up by.

    Returns:
        The SIMD value from the thread offset lanes lower in the warp.
        Returns undefined values for threads where lane_id - offset < 0.
    """
    return std._gpu.primitives.warp.shuffle_up(val, offset)


@always_inline("nodebug")
def shuffle_up[
    dtype: DType, simd_width: SIMDLength, //
](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
    dtype, simd_width
]:
    """Copies values from threads with lower lane IDs in the warp.

    Performs a shuffle operation where each thread receives a value from a thread with a
    lower lane ID, offset by the specified amount. The operation is performed only for
    threads specified in the mask.

    For example, with offset=1:
    - Thread N gets value from thread N-1 if both threads are in the mask
    - Thread 1 gets value from thread 0 if both threads are in the mask
    - Thread 0 gets undefined value
    - Threads not in the mask get undefined values

    Parameters:
        dtype: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in each SIMD vector.

    Args:
        mask: The warp mask specifying which threads participate in the shuffle.
        val: The SIMD value to be shuffled up the warp.
        offset: The number of lanes to shift values up by.

    Returns:
        The SIMD value from the thread offset lanes lower in the warp.
        Returns undefined values for threads where lane_id - offset < 0 or
        threads not in the mask.
    """
    return std._gpu.primitives.warp.shuffle_up(mask, val, offset)


@always_inline("nodebug")
def shuffle_down[
    dtype: DType, simd_width: SIMDLength, //
](val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[dtype, simd_width]:
    """Copies values from threads with higher lane IDs in the warp.

    Performs a shuffle operation where each thread receives a value from a thread with a
    higher lane ID, offset by the specified amount. Uses the full warp mask by default.

    For example, with offset=1:
    - Thread 0 gets value from thread 1
    - Thread 1 gets value from thread 2
    - Thread N gets value from thread N+1
    - Last N threads get undefined values

    Parameters:
        dtype: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in each SIMD vector.

    Args:
        val: The SIMD value to be shuffled down the warp.
        offset: The number of lanes to shift values down by. Must be positive.

    Returns:
        The SIMD value from the thread offset lanes higher in the warp.
        Returns undefined values for threads where lane_id + offset >= WARP_SIZE.
    """
    return std._gpu.primitives.warp.shuffle_down(val, offset)


@always_inline("nodebug")
def shuffle_down[
    dtype: DType, simd_width: SIMDLength, //
](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
    dtype, simd_width
]:
    """Copies values from threads with higher lane IDs in the warp using a custom mask.

    Performs a shuffle operation where each thread receives a value from a thread with a
    higher lane ID, offset by the specified amount. The mask parameter controls which
    threads participate in the shuffle.

    For example, with offset=1:
    - Thread 0 gets value from thread 1
    - Thread 1 gets value from thread 2
    - Thread N gets value from thread N+1
    - Last N threads get undefined values

    Parameters:
        dtype: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in each SIMD vector.

    Args:
        mask: A bitmask controlling which threads participate in the shuffle.
             Only threads with their corresponding bit set will exchange values.
        val: The SIMD value to be shuffled down the warp.
        offset: The number of lanes to shift values down by. Must be positive.

    Returns:
        The SIMD value from the thread offset lanes higher in the warp.
        Returns undefined values for threads where lane_id + offset >= WARP_SIZE
        or where the corresponding mask bit is not set.
    """
    return std._gpu.primitives.warp.shuffle_down(mask, val, offset)


@always_inline("nodebug")
def shuffle_xor[
    dtype: DType, simd_width: SIMDLength, //
](val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[dtype, simd_width]:
    """Exchanges values between threads in a warp using a butterfly pattern.

    Performs a butterfly exchange pattern where each thread swaps values with another thread
    whose lane ID differs by a bitwise XOR with the given offset. This creates a butterfly
    communication pattern useful for parallel reductions and scans.

    Parameters:
        dtype: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in each SIMD vector.

    Args:
        val: The SIMD value to be exchanged with another thread.
        offset: The lane offset to XOR with the current thread's lane ID to determine
               the exchange partner. Common values are powers of 2 for butterfly patterns.

    Returns:
        The SIMD value from the thread at lane (current_lane XOR offset).
    """
    return std._gpu.primitives.warp.shuffle_xor(val, offset)


@always_inline("nodebug")
def shuffle_xor[
    dtype: DType, simd_width: SIMDLength, //
](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
    dtype, simd_width
]:
    """Exchanges values between threads in a warp using a butterfly pattern with masking.

    Performs a butterfly exchange pattern where each thread swaps values with another thread
    whose lane ID differs by a bitwise XOR with the given offset. The mask parameter allows
    controlling which threads participate in the exchange.

    Parameters:
        dtype: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in each SIMD vector.

    Args:
        mask: A bit mask specifying which threads participate in the exchange.
             Only threads with their corresponding bit set in the mask will exchange values.
        val: The SIMD value to be exchanged with another thread.
        offset: The lane offset to XOR with the current thread's lane ID to determine
               the exchange partner. Common values are powers of 2 for butterfly patterns.

    Returns:
        The SIMD value from the thread at lane (current_lane XOR offset) if both threads
        are enabled by the mask, otherwise the original value is preserved.

    Example:

        ```mojo
            from max.gpu.primitives.warp import shuffle_xor

            # Exchange values between even-numbered threads 4 lanes apart
            var mask: UInt = 0xAAAAAAAA  # Even threads only
            var val = SIMD[.float32, 16](42.0)  # Example value
            var result = shuffle_xor(mask, val, 4)
        ```
    """
    return std._gpu.primitives.warp.shuffle_xor(mask, val, offset)


@always_inline("nodebug")
def lane_group_reduce[
    val_type: DType,
    simd_width: SIMDLength,
    //,
    shuffle: def[dtype: DType, simd_width: SIMDLength](
        val: SIMD[dtype, simd_width], offset: UInt32
    ) thin -> SIMD[dtype, simd_width],
    func: _ReduceFn,
    num_lanes: Int,
    *,
    stride: Int = 1,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Performs a generic warp-level reduction operation using shuffle operations.

    This function implements a parallel reduction across threads in a warp using a butterfly
    pattern. It allows customizing both the shuffle operation and reduction function.

    Parameters:
        val_type: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in the SIMD vector.
        shuffle: A function that performs the warp shuffle operation. Takes a SIMD value and
                offset and returns the shuffled result.
        func: A binary function that combines two SIMD values during reduction. This defines
              the reduction operation (e.g. add, max, min).
        num_lanes: The number of lanes in a group. The reduction is done within each group. Must be a power of 2.
        stride: The stride between lanes participating in the reduction.

    Args:
        val: The SIMD value to reduce. Each lane contributes its value.

    Returns:
        A SIMD value containing the reduction result.

    Example:

        ```mojo
            from max.gpu.primitives.warp import lane_group_reduce, shuffle_down

            # Compute sum across 16 threads using shuffle down
            @__parameter
            def add[dtype: DType, width: SIMDLength](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[dtype, width]:
                return x + y
            var val = SIMD[.float32, 16](42.0)
            var result = lane_group_reduce[shuffle_down, add, num_lanes=16](val)
        ```
    """
    return std._gpu.primitives.warp.lane_group_reduce[
        shuffle, func, num_lanes, stride=stride
    ](val)


@always_inline("nodebug")
def reduce[
    val_type: DType,
    simd_width: SIMDLength,
    //,
    shuffle: def[dtype: DType, simd_width: SIMDLength](
        val: SIMD[dtype, simd_width], offset: UInt32
    ) thin -> SIMD[dtype, simd_width],
    func: _ReduceFn,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Performs a generic warp-wide reduction operation using shuffle operations.

    This is a convenience wrapper around lane_group_reduce that operates on the entire warp.
    It allows customizing both the shuffle operation and reduction function.

    Parameters:
        val_type: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in the SIMD vector.
        shuffle: A function that performs the warp shuffle operation. Takes a SIMD value and
                offset and returns the shuffled result.
        func: A binary function that combines two SIMD values during reduction. This defines
              the reduction operation (e.g. add, max, min).

    Args:
        val: The SIMD value to reduce. Each lane contributes its value.

    Returns:
        A SIMD value containing the reduction result broadcast to all lanes in the warp.

    Example:

    ```mojo
        from max.gpu.primitives.warp import reduce, shuffle_down

        # Compute warp-wide sum using shuffle down
        @__parameter
        def add[dtype: DType, width: SIMDLength](x: SIMD[dtype, width], y: SIMD[dtype, width]) capturing -> SIMD[dtype, width]:
            return x + y

        val = SIMD[.float32, 4](2.0, 4.0, 6.0, 8.0)
        result = reduce[shuffle_down, add](val)
    ```
    """
    return std._gpu.primitives.warp.reduce[shuffle, func](val)


@always_inline("nodebug")
def lane_group_sum[
    val_type: DType,
    simd_width: SIMDLength,
    //,
    num_lanes: Int,
    stride: Int = 1,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Computes the sum of values across a group of lanes and broadcasts to all lanes.

    This function performs a parallel reduction across a group of lanes to compute their sum.
    The result is broadcast to all participating lanes using optimized hardware-specific
    paths (AMD DPP, Blackwell redux, or butterfly shuffle pattern).

    Parameters:
        val_type: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in the SIMD vector.
        num_lanes: The number of threads participating in the reduction.
        stride: The stride between lanes participating in the reduction.

    Args:
        val: The SIMD value to reduce. Each lane contributes its value to the sum.

    Returns:
        A SIMD value where all participating lanes contain the sum found across the lane group.
        Non-participating lanes (lane_id >= num_lanes) retain their original values.
    """
    return std._gpu.primitives.warp.lane_group_sum[num_lanes, stride](val)


@always_inline("nodebug")
def sum(val: SIMD) -> Scalar[val.dtype]:
    """Computes the sum of values across all lanes in a warp.

    This is a convenience wrapper around `lane_group_sum` that operates on the
    entire warp. It performs a parallel reduction using warp shuffle operations
    to find the global sum across all lanes in the warp.

    Args:
        val: The SIMD value to reduce. Each lane contributes its value to the sum.

    Returns:
        The scalar sum of values across all lanes in the warp.
    """
    return std._gpu.primitives.warp.sum(val)


@always_inline("nodebug")
def prefix_sum[
    dtype: DType,
    //,
    intermediate_type: DType = dtype,
    *,
    output_type: DType = dtype,
    exclusive: Bool = False,
](x: Scalar[dtype]) -> Scalar[output_type]:
    """Computes a warp-level prefix sum (scan) operation.

    Performs an inclusive or exclusive prefix sum across threads in a warp using
    a parallel scan algorithm with warp shuffle operations. This implements an
    efficient parallel scan with logarithmic complexity.

    For example, if we have a warp with the following elements:
    $$$
    [x_0, x_1, x_2, x_3, x_4]
    $$$

    The prefix sum is:
    $$$
    [x_0, x_0 + x_1, x_0 + x_1 + x_2, x_0 + x_1 + x_2 + x_3, x_0 + x_1 + x_2 + x_3 + x_4]
    $$$

    Parameters:
        dtype: The data type of the input SIMD elements.
        intermediate_type: Type used for intermediate calculations (defaults to
                          input dtype).
        output_type: The desired output data type (defaults to input dtype).
        exclusive: If True, performs exclusive scan where each thread receives
                   the sum of all previous threads. If False (default), performs
                   inclusive scan where each thread receives the sum including
                   its own value.

    Args:
        x: The SIMD value to include in the prefix sum.

    Returns:
        A scalar containing the prefix sum at the current thread's position in
        the warp, cast to the specified output dtype.
    """
    return std._gpu.primitives.warp.prefix_sum[
        intermediate_type, output_type=output_type, exclusive=exclusive
    ](x)


@always_inline("nodebug")
def lane_group_max[
    val_type: DType,
    simd_width: SIMDLength,
    //,
    num_lanes: Int,
    stride: Int = 1,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Reduces a SIMD value to its maximum within a lane group and broadcasts to all lanes.

    This function performs a parallel reduction across a group of lanes to find the maximum value.
    The result is broadcast to all participating lanes using optimized hardware-specific
    paths (AMD DPP, Blackwell redux, or butterfly shuffle pattern).

    Parameters:
        val_type: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in the SIMD vector.
        num_lanes: The number of threads participating in the reduction.
        stride: The stride between lanes participating in the reduction.

    Args:
        val: The SIMD value to reduce. Each lane contributes its value to find the maximum.

    Returns:
        A SIMD value where all participating lanes contain the maximum value found across the lane group.
        Non-participating lanes (lane_id >= num_lanes) retain their original values.
    """
    return std._gpu.primitives.warp.lane_group_max[num_lanes, stride](val)


@always_inline("nodebug")
def max(val: SIMD) -> Scalar[val.dtype]:
    """Computes the maximum value across all lanes in a warp.

    This is a convenience wrapper around lane_group_max that operates on the entire warp.
    It performs a parallel reduction using warp shuffle operations to find the global maximum
    value across all lanes in the warp.

    Args:
        val: The SIMD value to reduce. Each lane contributes its value to find the maximum.

    Returns:
        The scalar maximum value across all lanes in the warp.
    """
    return std._gpu.primitives.warp.max(val)


@always_inline("nodebug")
def lane_group_min[
    val_type: DType,
    simd_width: SIMDLength,
    //,
    num_lanes: Int,
    stride: Int = 1,
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Reduces a SIMD value to its minimum within a lane group and broadcasts to all lanes.

    This function performs a parallel reduction across a group of lanes to find the minimum value.
    The result is broadcast to all participating lanes using optimized hardware-specific
    paths (AMD DPP, Blackwell redux, or butterfly shuffle pattern).

    Parameters:
        val_type: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in the SIMD vector.
        num_lanes: The number of threads participating in the reduction.
        stride: The stride between lanes participating in the reduction.

    Args:
        val: The SIMD value to reduce. Each lane contributes its value to find the minimum.

    Returns:
        A SIMD value where all participating lanes contain the minimum value found across the lane group.
        Non-participating lanes (lane_id >= num_lanes) retain their original values.
    """
    return std._gpu.primitives.warp.lane_group_min[num_lanes, stride](val)


@always_inline("nodebug")
def min(val: SIMD) -> Scalar[val.dtype]:
    """Computes the minimum value across all lanes in a warp.

    This is a convenience wrapper around lane_group_min that operates on the entire warp.
    It performs a parallel reduction using warp shuffle operations to find the global minimum
    value across all lanes in the warp.

    Args:
        val: The SIMD value to reduce. Each lane contributes its value to find the minimum.

    Returns:
        The scalar minimum value across all lanes in the warp.
    """
    return std._gpu.primitives.warp.min(val)


@always_inline("nodebug")
def broadcast[
    val_type: DType, simd_width: SIMDLength, //
](val: SIMD[val_type, simd_width]) -> SIMD[val_type, simd_width]:
    """Broadcasts a SIMD value from lane 0 to all lanes in the warp.

    This function takes a SIMD value from lane 0 and copies it to all other lanes in the warp,
    effectively broadcasting the value across the entire warp. This is useful for sharing data
    between threads in a warp without using shared memory.

    Parameters:
        val_type: The data type of the SIMD elements (e.g. float32, int32).
        simd_width: The number of elements in the SIMD vector.

    Args:
        val: The SIMD value to broadcast from lane 0.

    Returns:
        A SIMD value where all lanes contain a copy of the input value from lane 0.
    """
    return std._gpu.primitives.warp.broadcast(val)


@always_inline("nodebug")
def broadcast(val: Int) -> Int:
    """Broadcasts an integer value from lane 0 to all lanes in the warp.

    This function takes an integer value from lane 0 and copies it to all other lanes in the warp.
    It provides a convenient way to share scalar integer data between threads without using shared memory.

    Args:
        val: The integer value to broadcast from lane 0.

    Returns:
        The broadcast integer value, where all lanes receive a copy of the input from lane 0.
    """
    return std._gpu.primitives.warp.broadcast(val)


@always_inline("nodebug")
def broadcast(val: UInt) -> UInt:
    """Broadcasts an unsigned integer value from lane 0 to all lanes in the warp.

    This function takes an unsigned integer value from lane 0 and copies it to all other lanes in the warp.
    It provides a convenient way to share scalar unsigned integer data between threads without using shared memory.

    Args:
        val: The unsigned integer value to broadcast from lane 0.

    Returns:
        The broadcast unsigned integer value, where all lanes receive a copy of the input from lane 0.
    """
    return std._gpu.primitives.warp.broadcast(val)


@always_inline("nodebug")
def vote[ret_type: DType](val: Bool) -> Scalar[ret_type]:
    """Creates a 32 or 64 bit mask among all threads in the warp, where each bit is set to 1 if the
    corresponding thread voted True, and 0 otherwise.

    This function takes a boolean value which represents the corresponding threads vote.

    NVIDIA supports 32-bit masks; AMD supports 32- and 64-bit masks; Apple
    Silicon (a 32-lane SIMD-group) supports 32-bit masks, and also accepts a
    `DType.uint64` return whose upper 32 bits are always zero.

    Parameters:
        ret_type: Return type for the mask (must be `DType.uint32` or `DType.uint64`).

    Args:
        val: The boolean vote.

    Returns:
        A mask containing the vote of all threads in the warp.
    """
    return std._gpu.primitives.warp.vote[ret_type](val)


@always_inline("nodebug")
def match_any[
    dtype: DType,
    //,
    mask_type: DType = (DType.uint32 if WARP_SIZE <= 32 else DType.uint64),
](value: Scalar[dtype]) -> Scalar[mask_type]:
    """Finds, for each lane, the mask of warp lanes whose `value` bits match it.

    Returns a per-lane lane mask whose bit `l` is set for every active lane `l`
    whose `value` has the same bit pattern as the calling lane's. The comparison
    is on the bits (matching NVIDIA's `match.any.sync`), so `0.0` and `-0.0` do
    not match while two `NaN`s with equal bits do. This is the fold a warp uses
    to coalesce same-keyed lanes (a histogram or scatter leader handling a whole
    group in one non-atomic update) instead of one atomic per lane.

    All `WARP_SIZE` lanes must reach the call converged.

    Example:

        ```mojo
        from max.gpu.primitives.warp import match_any

        # If lanes 0, 3, 7 hold the same value, each of them gets a mask with
        # bits 0, 3, and 7 set; the remaining lanes get their own groups.
        var my_key = Int32(42)
        var group = match_any(my_key)
        ```

    Parameters:
        dtype: The element type of `value` (inferred from the argument).
        mask_type: The lane-mask return type, `DType.uint32` or `DType.uint64`
            (defaults to the type matching `WARP_SIZE`).

    Args:
        value: The calling lane's value to match against the rest of the warp.

    Returns:
        A `mask_type` lane mask with bit `l` set for each active lane `l` holding
        a bit-equal `value`.

    Constraints:
        Only NVIDIA, AMD, and Apple Silicon GPUs are supported. `dtype` must be
        a 32- or 64-bit type and `mask_type` must be `DType.uint32` or
        `DType.uint64` (NVIDIA returns a 32-bit mask, so `mask_type` must be
        `DType.uint32` there).
    """
    return std._gpu.primitives.warp.match_any[mask_type](value)


@always_inline("nodebug")
def match_all[
    dtype: DType,
    //,
    mask_type: DType = (DType.uint32 if WARP_SIZE <= 32 else DType.uint64),
](value: Scalar[dtype]) -> Scalar[mask_type]:
    """Returns the warp's active-lane mask if all lanes share `value`, else 0.

    When every active lane holds the same bits as the calling lane, returns the
    mask of those lanes (so a non-zero result is the "all agree" predicate that
    NVIDIA's `match.all.sync` also exposes); otherwise returns 0. The comparison
    is on the bits, so `0.0` and `-0.0` are treated as different. This is the
    dual of `match_any`: it reports warp-wide agreement on a key.

    All `WARP_SIZE` lanes must reach the call converged.

    Example:

        ```mojo
        from max.gpu.primitives.warp import match_all

        # `agreed` is non-zero (the active-lane mask) iff every lane passed the
        # same `key`.
        var key = Int32(42)
        var agreed = match_all(key)
        ```

    Parameters:
        dtype: The element type of `value` (inferred from the argument).
        mask_type: The lane-mask return type, `DType.uint32` or `DType.uint64`
            (defaults to the type matching `WARP_SIZE`).

    Args:
        value: The calling lane's value to compare against the rest of the warp.

    Returns:
        A `mask_type` lane mask of the active lanes when they all hold a
        bit-equal `value`, otherwise 0.

    Constraints:
        Only NVIDIA, AMD, and Apple Silicon GPUs are supported. `dtype` must be
        a 32- or 64-bit type and `mask_type` must be `DType.uint32` or
        `DType.uint64` (NVIDIA returns a 32-bit mask, so `mask_type` must be
        `DType.uint32` there).
    """
    return std._gpu.primitives.warp.match_all[mask_type](value)
