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
"""Provides low-level GPU intrinsic operations and memory access primitives.

Implements hardware-specific intrinsics that map directly to GPU assembly
instructions, focusing on NVIDIA GPU architectures. Includes:

- Global memory load/store operations with cache control
- Warp-level primitives and synchronization
- Memory fence and barrier operations
- Atomic operations and memory ordering primitives

These low-level primitives should be used carefully as they correspond
directly to hardware instructions and require understanding of the
underlying GPU architecture.
"""

import std._gpu.intrinsics
from std.memory import AddressSpace
from std.sys import align_of

# `std._gpu.intrinsics` makes this visible by importing it, and call sites import
# it from there rather than from `std.sys._assembly`. Keep that path working.
from std.sys._assembly import inlined_assembly

# `mojo doc` renders an aliased struct as a bare `comptime` entry, dropping the
# fields and methods that `std._gpu.intrinsics` documents on pages of their own.
# These three keep only their summary until the definitions themselves move.

comptime AMDBufferResource = std._gpu.intrinsics.AMDBufferResource
"""128-bit descriptor for a buffer resource on AMD GPUs.

Used for buffer_load/buffer_store instructions.
"""

comptime CacheOperation = std._gpu.intrinsics.CacheOperation
"""Represents different GPU cache operation policies.

This struct defines various caching behaviors for GPU memory operations,
controlling how data is cached and evicted at different cache levels.
The policies affect performance and memory coherency.
"""

comptime Scope = std._gpu.intrinsics.Scope
"""Represents memory synchronization scope levels for GPU memory operations.

Defines different scopes of memory visibility and synchronization, from
thread-local to system-wide. Each scope level determines how memory
operations are ordered and visible across different execution units.

The scope levels form a hierarchy, with each higher level providing
stronger ordering guarantees but potentially higher synchronization costs.
"""


@always_inline("nodebug")
def ldg[
    dtype: DType,
    //,
    width: Int = 1,
    *,
    alignment: Int = align_of[SIMD[dtype, width]](),
](x: Pointer[Scalar[dtype], address_space=AddressSpace.GENERIC, ...]) -> SIMD[
    dtype, width
] where dtype.is_numeric():
    """Load data from global memory through the non-coherent cache.

    This function provides a hardware-accelerated global memory load operation
    that uses the GPU's non-coherent cache (equivalent to CUDA's `__ldg` instruction).
    It optimizes for read-only data access patterns.

    Parameters:
        dtype: The data type to load (must be numeric).
        width: The SIMD vector width for vectorized loads.
        alignment: Memory alignment in bytes. Defaults to natural alignment
            of the SIMD vector dtype.

    Args:
        x: Pointer to global memory location to load from.

    Returns:
        SIMD vector containing the loaded data.

    Note:
        - Uses invariant loads which indicate the memory won't change during kernel execution.
        - Particularly beneficial for read-only texture-like access patterns.
        - May improve performance on memory-bound kernels.
    """
    return std._gpu.intrinsics.ldg[width, alignment=alignment](x)


@always_inline("nodebug")
def warpgroup_reg_alloc[count: Int]():
    """Allocates additional registers for the executing warp group.

    Hints to the system to increase per-thread registers owned by the
    executing warp. Requests additional registers to increase the absolute
    per-thread maximum register count from its current value to the specified
    count.

    Parameters:
        count: The desired number of registers per thread. Must be:
            - A multiple of 8
            - Between 24 and 256 (inclusive).

    Note:
        - Only supported on NVIDIA SM90+ GPUs
        - Performance optimization hint that may be ignored by the hardware
        - Pair with `warpgroup_reg_dealloc() when extra registers are no
          longer needed
    """
    std._gpu.intrinsics.warpgroup_reg_alloc[count]()


@always_inline("nodebug")
def warpgroup_reg_dealloc[count: Int]():
    """Deallocates additional registers for the executing warp group.

    Hints to the system to decrease per-thread registers owned by the
    executing warp. Releases extra registers to reduce the absolute per-thread
    maximum register count from its current value to the specified count.

    Parameters:
        count: The desired number of registers per thread. Must be:
            - A multiple of 8.
            - Between 24 and 256 (inclusive).

    Note:
        - Only supported on NVIDIA SM90+ GPUs.
        - Performance optimization hint that may be ignored by the hardware.
        - Pair with `warpgroup_reg_alloc()` when extra registers are needed.
    """
    std._gpu.intrinsics.warpgroup_reg_dealloc[count]()


@always_inline("nodebug")
def lop[lut: Int32](a: Int32, b: Int32, c: Int32) -> Int32:
    """Performs an arbitrary logical operation on 3 inputs using a lookup table.

    Implements a 3-input lookup table (LUT) operation. The result is
    determined by bits in the lookup table value for each input combination.

    Parameters:
        lut: 32-bit lookup table value that defines the logical operation.

    Args:
        a: First input value.
        b: Second input value.
        c: Third input value.

    Returns:
        Result of applying the lookup table operation to the inputs.

    Note:
        - Only supported on NVIDIA GPUs.
        - Maps to the LOP3.B32 PTX instruction.
        - Lookup table value determines output for each possible input combo.
    """
    return std._gpu.intrinsics.lop[lut](a, b, c)


@always_inline("nodebug")
def byte_permute(a: UInt32, b: UInt32, c: UInt32) -> UInt32:
    """Permutes bytes from two 32-bit integers based on a control mask.

    Selects and rearranges bytes from two source integers based on a control
    mask to create a new 32-bit value.

    Args:
        a: First source integer containing bytes to select from.
        b: Second source integer containing bytes to select from.
        c: Control mask that specifies which bytes to select and their
           positions. Each byte in the mask controls selection/placement of
           one output byte.

    Returns:
        A new 32-bit integer containing the selected and rearranged bytes

    Note:
        Byte selection behavior depends on the GPU architecture:
        - On NVIDIA: Maps to PRMT instruction
        - On AMD: Maps to PERM instruction.
    """
    return std._gpu.intrinsics.byte_permute(a, b, c)


@always_inline("nodebug")
def mulhi(a: UInt16, b: UInt16) -> UInt32:
    """Calculates the most significant 32 bits of the product of two 16-bit
    unsigned integers.

    Multiplies two 16-bit unsigned integers and returns the high 32 bits
    of their product. Useful for fixed-point arithmetic and overflow
    detection.

    Args:
        a: First 16-bit unsigned integer operand.
        b: Second 16-bit unsigned integer operand.

    Returns:
        The high 32 bits of the product a * b

    Note:
        On NVIDIA GPUs, this maps directly to the MULHI.U16 PTX instruction.
        On others, it performs multiplication using 32-bit arithmetic.
    """
    return std._gpu.intrinsics.mulhi(a, b)


@always_inline("nodebug")
def mulhi(a: Int16, b: Int16) -> Int32:
    """Calculates the most significant 32 bits of the product of two 16-bit
    signed integers.

    Multiplies two 16-bit signed integers and returns the high 32 bits
    of their product. Useful for fixed-point arithmetic and overflow detection.

    Args:
        a: First 16-bit signed integer operand.
        b: Second 16-bit signed integer operand.

    Returns:
        The high 32 bits of the product a * b

    Note:
        On NVIDIA GPUs, this maps directly to the MULHI.S16 PTX instruction.
        On others, it performs multiplication using 32-bit arithmetic.
    """
    return std._gpu.intrinsics.mulhi(a, b)


@always_inline("nodebug")
def mulhi(a: UInt32, b: UInt32) -> UInt32:
    """Calculates the most significant 32 bits of the product of two 32-bit
    unsigned integers.

    Multiplies two 32-bit unsigned integers and returns the high 32 bits
    of their product. Useful for fixed-point arithmetic and overflow detection.

    Args:
        a: First 32-bit unsigned integer operand.
        b: Second 32-bit unsigned integer operand.

    Returns:
        The high 32 bits of the product a * b

    Note:
        On NVIDIA GPUs, this maps directly to the MULHI.U32 PTX instruction.
        On others, it performs multiplication using 64-bit arithmetic.
    """
    return std._gpu.intrinsics.mulhi(a, b)


@always_inline("nodebug")
def mulhi(a: Int32, b: Int32) -> Int32:
    """Calculates the most significant 32 bits of the product of two 32-bit
    signed integers.

    Multiplies two 32-bit signed integers and returns the high 32 bits
    of their product. Useful for fixed-point arithmetic and overflow detection.

    Args:
        a: First 32-bit signed integer operand.
        b: Second 32-bit signed integer operand.

    Returns:
        The high 32 bits of the product a * b

    Note:
        On NVIDIA GPUs, this maps directly to the MULHI.S32 PTX instruction.
        On others, it performs multiplication using 64-bit arithmetic.
    """
    return std._gpu.intrinsics.mulhi(a, b)


@always_inline("nodebug")
def mulhi(a: UInt64, b: UInt64) -> UInt64:
    """Calculates the most significant 64 bits of the product of two 64-bit
    unsigned integers.

    Multiplies two 64-bit unsigned integers and returns the high 64 bits
    of their product. Useful for fixed-point arithmetic and overflow detection.

    Args:
        a: First 64-bit unsigned integer operand.
        b: Second 64-bit unsigned integer operand.

    Returns:
        The high 64 bits of the product a * b.

    Note:
        On NVIDIA GPUs, this maps directly to the MULHI.U64 PTX instruction.
        On others, it performs multiplication using 128-bit arithmetic.
    """
    return std._gpu.intrinsics.mulhi(a, b)


@always_inline("nodebug")
def mulhi(a: Int64, b: Int64) -> Int64:
    """Calculates the most significant 64 bits of the product of two 64-bit
    signed integers.

    Multiplies two 64-bit signed integers and returns the high 64 bits
    of their product. Useful for fixed-point arithmetic and overflow detection.

    Args:
        a: First 64-bit signed integer operand.
        b: Second 64-bit signed integer operand.

    Returns:
        The high 64 bits of the product a * b.

    Note:
        On NVIDIA GPUs, this maps directly to the MULHI.S64 PTX instruction.
        On others, it performs multiplication using 128-bit arithmetic.
    """
    return std._gpu.intrinsics.mulhi(a, b)


@always_inline("nodebug")
def mulwide(a: UInt32, b: UInt32) -> UInt64:
    """Performs a wide multiplication of two 32-bit unsigned integers.

    Multiplies two 32-bit unsigned integers and returns the full 64-bit result.
    Useful when the product may exceed 32 bits.

    Args:
        a: First 32-bit unsigned integer operand.
        b: Second 32-bit unsigned integer operand.

    Returns:
        The full 64-bit product of a * b

    Note:
        On NVIDIA GPUs, this maps directly to the MUL.WIDE.U32 PTX instruction.
        On others, it performs multiplication using 64-bit casts.
    """
    return std._gpu.intrinsics.mulwide(a, b)


@always_inline("nodebug")
def mulwide(a: Int32, b: Int32) -> Int64:
    """Performs a wide multiplication of two 32-bit signed integers.

    Multiplies two 32-bit signed integers and returns the full 64-bit result.
    Useful when the product may exceed 32 bits or be negative.

    Args:
        a: First 32-bit signed integer operand.
        b: Second 32-bit signed integer operand.

    Returns:
        The full 64-bit signed product of a * b

    Note:
        On NVIDIA GPUs, this maps directly to the MUL.WIDE.S32 PTX instruction.
        On others, it performs multiplication using 64-bit casts.
    """
    return std._gpu.intrinsics.mulwide(a, b)


@always_inline("nodebug")
def get_ib_sts() -> Int32:
    """Returns the IB status of the current thread.

    Returns:
        The IB status of the current thread.
    """
    return std._gpu.intrinsics.get_ib_sts()


@always_inline("nodebug")
def threadfence[scope: Scope = Scope.GPU]():
    """Enforces ordering of memory operations across threads.

    Acts as a memory fence/barrier that ensures all memory operations (both
    loads and stores) issued before the fence are visible to other threads
    within the specified scope before any memory operations after the fence.

    Parameters:
        scope: Memory scope level for the fence. Defaults to GPU-wide scope.
              Valid values are:
              - Scope.BLOCK: Orders memory within a thread block/CTA.
              - Scope.GPU: Orders memory across all threads on the GPU (default).
              - Scope.SYSTEM: Orders memory across the entire system.

    Note:
        - Maps directly to CUDA `__threadfence()` family of functions.
        - Critical for synchronizing memory access in parallel algorithms.
        - Performance impact increases with broader scopes.
    """
    std._gpu.intrinsics.threadfence[scope]()


@always_inline("nodebug")
def ds_read_tr16_b64[
    dtype: DType,
    //,
](
    shared_ptr: Pointer[Scalar[dtype], address_space=AddressSpace.SHARED, ...]
) -> SIMD[dtype, 4]:
    """Reads a 64-bit LDS transpose block using TR16 layout and returns SIMD[dtype, 4] of 16-bit types.

    Parameters:
        dtype: Data type of the elements (must be 16-bit type).

    Args:
        shared_ptr: Pointer to the LDS transpose block.

    Returns:
        SIMD[dtype, 4] of 16-bit types.

    Notes:
        - Only supported on AMD GPUs.
        - Maps directly to llvm.amdgcn.ds.read.tr16.b64 intrinsic.
        - Result width is fixed to 4 elements of dtype.
    """
    return std._gpu.intrinsics.ds_read_tr16_b64(shared_ptr)


@always_inline("nodebug")
def ds_read_tr8_b64[
    dtype: DType,
    //,
](
    shared_ptr: Pointer[Scalar[dtype], address_space=AddressSpace.SHARED, ...]
) -> SIMD[dtype, 8]:
    """Reads a 64-bit LDS transpose block using TR8 layout and returns SIMD[dtype, 8] of 8-bit types.

    Each 16-lane row reads 16x8 bytes from LDS and performs two interleaved
    8x8 byte transposes, producing 8 transposed bytes per lane.

    Parameters:
        dtype: Data type of the elements (must be 8-bit type).

    Args:
        shared_ptr: Pointer to the LDS transpose block.

    Returns:
        SIMD[dtype, 8] of 8-bit types.

    Notes:
        - Only supported on AMD GPUs (CDNA4+).
        - Maps directly to llvm.amdgcn.ds.read.tr8.b64 intrinsic.
        - Return type must use v2i32 intermediate to avoid LLVM type legalizer crash.
    """
    return std._gpu.intrinsics.ds_read_tr8_b64(shared_ptr)


@always_inline("nodebug")
def cvt_pk_fp8_f32_raw[
    dtype: DType,
](src: SIMD[DType.float32, 4]) -> SIMD[dtype, 4]:
    """Packs 4 f32 into 4 fp8 via 2 chained `v_cvt_pk_fp8_f32` ops.

    Unlike `SIMD.cast[fp8]()`, this bypasses the compiler's clamp + NaN
    scrub wrapper (`v_med3_f32` + `v_cmp_u_f32` + `v_cndmask_b32`) that the
    `pop.cast` lowering emits on AMDGPU. The caller is responsible for
    ensuring inputs are in the FP8 representable range; finite
    out-of-range values are NOT saturated by the hardware instruction.
    NaN/Inf inputs produce implementation-defined FP8 outputs.

    Parameters:
        dtype: The FP8 destination type, `float8_e4m3fn` or `float8_e5m2`.

    Args:
        src: Four f32 values to pack.

    Returns:
        SIMD of 4 fp8 values, bitcast from the packed i32 result.

    Notes:
        - Only supported on AMD CDNA4+ GPUs.
        - Maps to two `v_cvt_pk_fp8_f32` (or `.pk.bf8.f32`) instructions.
        - Use only when input domain is provably bounded (e.g. softmax
          output, where values are in (0, 1]).
    """
    return std._gpu.intrinsics.cvt_pk_fp8_f32_raw[dtype](src)


@always_inline("nodebug")
def permlane_swap[
    dtype: DType, //, stride: Int
](val1: Scalar[dtype], val2: Scalar[dtype]) -> SIMD[dtype, 2]:
    """Swaps values between lanes using AMD permlane swap instruction.

    Parameters:
        dtype: Data type of the values (must be 32-bit type).
        stride: Swap stride (must be 16 or 32).

    Args:
        val1: First value to swap.
        val2: Second value to swap.

    Returns:
        SIMD vector containing the swapped values.
    """
    return std._gpu.intrinsics.permlane_swap[stride](val1, val2)


@always_inline("nodebug")
def permlane_shuffle[
    dtype: DType, simd_width: SIMDLength, //, stride: Int
](val: SIMD[dtype, simd_width], out res: type_of(val)):
    """Shuffles SIMD values across lanes using AMD permlane operations.

    Parameters:
        dtype: Data type of the values.
        simd_width: Width of the SIMD vector.
        stride: Shuffle stride.

    Args:
        val: Input SIMD vector to shuffle.

    Returns:
        Shuffled SIMD vector in the `res` output parameter.
    """
    return std._gpu.intrinsics.permlane_shuffle[stride](val)


from std._gpu.intrinsics import _get_nvtx_register_constraint
