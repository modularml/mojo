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

from std.atomic import Ordering
from std.ffi import external_call
from std.sys import (
    is_amd_gpu,
    is_gpu,
    is_nvidia_gpu,
    is_apple_gpu,
    size_of,
    _RegisterPackType,
)
from std.sys._assembly import inlined_assembly
from std.sys.info import (
    CompilationTarget,
    _is_sm_9x_or_newer,
    align_of,
    bit_width_of,
    _is_amd_cdna,
    _cdna_4_or_newer,
    _is_amd_rdna1,
    _is_amd_rdna2,
    _is_amd_rdna3,
    _is_amd_rdna4,
)
from std.sys.intrinsics import llvm_intrinsic, readfirstlane
from std.math.uutils import ufloordiv, umod

from std._gpu._utils import to_i32
from std._gpu import lane_id

from std.memory.unsafe import bitcast

# ===-----------------------------------------------------------------------===#
# CacheOperation
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct CacheOperation(Equatable, TrivialRegisterPassable):
    """Represents different GPU cache operation policies.

    This struct defines various caching behaviors for GPU memory operations,
    controlling how data is cached and evicted at different cache levels.
    The policies affect performance and memory coherency.
    """

    var _value: Int

    comptime ALWAYS = Self(0)
    """Cache at all levels. This will be accessed again.

    Best for data that will be frequently reused across multiple threads.
    Provides fastest subsequent access but uses the most cache space.
    """

    comptime GLOBAL = Self(1)
    """Cache at global level.

    Caches data only in the L2 cache, bypassing L1.
    Good for data shared between different thread blocks.
    """

    comptime STREAMING = Self(2)
    """Streaming, this is likely to be accessed once.

    Optimizes for streaming access patterns where data is only read once.
    May bypass certain cache levels for better throughput.
    """

    comptime LAST_USE = Self(4)
    """Indicates the cache line will not be used again.

    Hints to the cache that this data can be evicted after this access.
    Helps optimize cache utilization.
    """

    comptime VOLATILE = Self(8)
    """Don't cache, and fetch again.

    Forces reads/writes to bypass cache and go directly to memory.
    Useful for memory-mapped I/O or when cache coherency is required.
    """

    comptime WRITE_BACK = Self(16)
    """Write back at all coherent levels.

    Updates all cache levels and eventually writes to memory.
    Most efficient for multiple writes to same location.
    """

    comptime WRITE_THROUGH = Self(32)
    """Write through to system memory.

    Immediately writes updates to memory while updating cache.
    Provides stronger consistency but lower performance than write-back.
    """

    comptime WORKGROUP = Self(64)
    """Workgroup level coherency.

    Caches data in the L1 cache and streams it to the wave.
    """

    def __eq__(self, other: Self) -> Bool:
        """Tests if two `CacheOperation` instances are equal.

        Args:
            other: The `CacheOperation` to compare against.

        Returns:
            True if the operations are equal, False otherwise.
        """
        return self._value == other._value

    def __or__(self, other: Self) -> Self:
        """Returns the bitwise OR of two `CacheOperation` instances.

        Args:
            other: The other `CacheOperation` to OR with.

        Returns:
            A new `CacheOperation` representing the bitwise OR of the two values.
        """
        return Self(self._value | other._value)

    @always_inline
    def mnemonic(self) -> StaticString:
        """Returns the PTX mnemonic string for this cache operation.

        Converts the cache operation into its corresponding PTX assembly
        mnemonic string used in GPU instructions.

        Returns:
            A string literal containing the PTX mnemonic for this operation.
        """
        if self == Self.ALWAYS:
            return "ca"
        if self == Self.GLOBAL:
            return "cg"
        if self == Self.STREAMING:
            return "cs"
        if self == Self.LAST_USE:
            return "lu"
        if self == Self.VOLATILE:
            return "cv"
        if self == Self.WRITE_BACK:
            return "wb"
        if self == Self.WRITE_THROUGH:
            return "wt"
        if self == Self.WORKGROUP:
            return "wg"

        return "unknown cache operation"


# ===-----------------------------------------------------------------------===#
# ldg
# ===-----------------------------------------------------------------------===#


@always_inline
def ldg[
    dtype: DType,
    //,
    width: Int = 1,
    *,
    alignment: Int = align_of[SIMD[dtype, width]](),
](x: Pointer[Scalar[dtype], address_space=.GENERIC, ...]) -> SIMD[
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
    return x.unsafe_load[width=width, alignment=alignment, invariant=True]()


# ===-----------------------------------------------------------------------===#
# warpgroup_reg
# ===-----------------------------------------------------------------------===#


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

    comptime assert (
        count % 8 == 0
    ), "count argument to warpgroup_reg_alloc must be in multiples of 8"

    comptime assert (
        24 <= count <= 256
    ), "count argument must be within 24 and 256"

    comptime if _is_sm_9x_or_newer():
        inlined_assembly[
            "setmaxnreg.inc.sync.aligned.u32 $0;",
            NoneType,
            constraints="n",
        ](Int32(count))


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

    comptime assert (
        count % 8 == 0
    ), "count argument to warpgroup_reg_dealloc must be in multiples of 8"

    comptime assert (
        24 <= count <= 256
    ), "count argument must be within 24 and 256"

    comptime if _is_sm_9x_or_newer():
        inlined_assembly[
            "setmaxnreg.dec.sync.aligned.u32 $0;",
            NoneType,
            constraints="n",
        ](Int32(count))


# ===-----------------------------------------------------------------------===#
# lop
# ===-----------------------------------------------------------------------===#


@always_inline
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

    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "lop3.b32 $0, $1, $2, $3, $4;",
            Int32,
            constraints="=r,r,n,n,n",
            has_side_effect=False,
        ](a, b, c, lut)
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name(),
            note="lop() is only supported when targeting NVIDIA GPUs.",
        ]()


# ===-----------------------------------------------------------------------===#
# permute
# ===-----------------------------------------------------------------------===#


@always_inline
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
    comptime asm = _byte_permute_inst()

    return llvm_intrinsic[asm, UInt32, has_side_effect=False](a, b, c)


def _byte_permute_inst() -> StaticString:
    comptime if is_nvidia_gpu():
        return "llvm.nvvm.prmt"
    elif is_amd_gpu():
        return "llvm.amdgcn.perm"
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name(),
        ]()


# ===-----------------------------------------------------------------------===#
# mulhi
# ===-----------------------------------------------------------------------===#


@always_inline
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

    comptime if is_nvidia_gpu():
        return llvm_intrinsic[
            "llvm.nvvm.mulhi.us", UInt32, has_side_effect=False
        ](a, b)

    var au32 = a.cast[.uint32]()
    var bu32 = b.cast[.uint32]()
    return au32 * bu32


@always_inline
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

    comptime if is_nvidia_gpu():
        return llvm_intrinsic[
            "llvm.nvvm.mulhi.s", Int32, has_side_effect=False
        ](a, b)

    var ai32 = a.cast[.int32]()
    var bi32 = b.cast[.int32]()
    return ai32 * bi32


@always_inline
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

    comptime if is_nvidia_gpu():
        return llvm_intrinsic[
            "llvm.nvvm.mulhi.ui", UInt32, has_side_effect=False
        ](a, b)

    var au64 = a.cast[.uint64]()
    var bu64 = b.cast[.uint64]()
    return ((au64 * bu64) >> 32).cast[.uint32]()


@always_inline
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

    comptime if is_nvidia_gpu():
        return llvm_intrinsic[
            "llvm.nvvm.mulhi.i", Int32, has_side_effect=False
        ](a, b)

    var ai64 = a.cast[.int64]()
    var bi64 = b.cast[.int64]()
    return ((ai64 * bi64) >> 32).cast[.int32]()


@always_inline
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

    comptime if is_nvidia_gpu():
        return llvm_intrinsic[
            "llvm.nvvm.mulhi.ull", UInt64, has_side_effect=False
        ](a, b)

    var au128 = a.cast[.uint128]()
    var bu128 = b.cast[.uint128]()
    return ((au128 * bu128) >> 64).cast[.uint64]()


@always_inline
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

    comptime if is_nvidia_gpu():
        return llvm_intrinsic[
            "llvm.nvvm.mulhi.ll", Int64, has_side_effect=False
        ](a, b)

    var ai128 = a.cast[.int128]()
    var bi128 = b.cast[.int128]()
    return ((ai128 * bi128) >> 64).cast[.int64]()


# ===-----------------------------------------------------------------------===#
# mulwide
# ===-----------------------------------------------------------------------===#


@always_inline
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

    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "mul.wide.u32 $0, $1, $2;",
            UInt64,
            constraints="=l,r,r",
            has_side_effect=False,
        ](a, b)

    var au64 = a.cast[.uint64]()
    var bu64 = b.cast[.uint64]()
    return au64 * bu64


@always_inline
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

    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "mul.wide.s32 $0, $1, $2;",
            Int64,
            constraints="=l,r,r",
            has_side_effect=False,
        ](a, b)

    var ai64 = a.cast[.int64]()
    var bi64 = b.cast[.int64]()
    return ai64 * bi64


@always_inline
def get_ib_sts() -> Int32:
    """Returns the IB status of the current thread.

    Returns:
        The IB status of the current thread.
    """
    if is_amd_gpu():
        return inlined_assembly[
            "s_getreg_b32 $0, hwreg(HW_REG_IB_STS);",
            Int32,
            constraints="=r",
            has_side_effect=False,
        ]()
    else:
        return 0


# ===-----------------------------------------------------------------------===#
# threadfence
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct Scope(Equatable, ImplicitlyCopyable, Writable):
    """Represents memory synchronization scope levels for GPU memory operations.

    Defines different scopes of memory visibility and synchronization, from
    thread-local to system-wide. Each scope level determines how memory
    operations are ordered and visible across different execution units.

    The scope levels form a hierarchy, with each higher level providing
    stronger ordering guarantees but potentially higher synchronization costs.
    """

    var _value: Int

    comptime NONE = Self(0)
    """No memory ordering guarantees. Operations may be reordered freely."""

    comptime THREAD = Self(1)
    """Thread-level scope. Memory operations are ordered within a single thread."""

    comptime WARP = Self(2)
    """Warp-level scope. Memory operations are ordered within a warp of threads."""

    comptime BLOCK = Self(3)
    """Block-level scope. Memory operations ordered within a thread block/CTA."""

    comptime CLUSTER = Self(4)
    """Cluster-level scope. Memory operations ordered within a thread block cluster."""

    comptime GPU = Self(5)
    """GPU-level scope. Memory operations are ordered across all threads on the GPU."""

    comptime SYSTEM = Self(6)
    """System-wide scope. Memory operations ordered across the entire system."""

    def __eq__(self, other: Self) -> Bool:
        """Checks if two `Scope` instances are equal.

        Uses pointer comparison for efficiency.

        Args:
            other: The other `Scope` instance to compare with.

        Returns:
            True if the instances are the same, False otherwise.
        """
        return self._value == other._value

    @no_inline
    def write_to(self, mut w: Some[Writer]):
        """Writes the string representation of the scope to a writer.

        Args:
            w: The writer to write to.
        """
        if self == Self.NONE:
            return w.write("none")
        if self == Self.THREAD:
            return w.write("thread")
        if self == Self.WARP:
            return w.write("warp")
        if self == Self.BLOCK:
            return w.write("block")
        if self == Self.CLUSTER:
            return w.write("cluster")
        if self == Self.GPU:
            return w.write("gpu")
        if self == Self.SYSTEM:
            return w.write("system")

        return w.write("<<unknown scope>>")

    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes the string representation of the scope to a writer.

        Args:
            writer: The writer to write to.
        """
        t"Scope({self})".write_to(writer)

    @always_inline("nodebug")
    def mnemonic(self) -> StaticString:
        """Returns the mnemonic string representation of the memory scope.

        Converts the memory scope level into a string mnemonic used by LLVM/NVVM
        intrinsics for memory operations.

        Returns:
            A string literal containing the mnemonic.
        """
        if self in (Self.NONE, Self.THREAD, Self.WARP):
            return ""
        if self == Self.BLOCK:
            return "cta"
        if self == Self.CLUSTER:
            return "cluster"
        if self == Self.GPU:
            return "gpu"
        if self == Self.SYSTEM:
            return "sys"
        return "<<invalid scope>>"


@always_inline
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
    comptime assert scope in (
        Scope.GPU,
        Scope.BLOCK,
        Scope.SYSTEM,
    ), "invalid threadfence scope"
    comptime assert (
        is_nvidia_gpu()
    ), "threadfence is only implemented on NVIDIA GPUs"

    comptime suffix = "gl" if scope == Scope.GPU else scope.mnemonic()
    llvm_intrinsic["llvm.nvvm.membar." + suffix, NoneType]()


# ===-----------------------------------------------------------------------===#
# release / acquire
# ===-----------------------------------------------------------------------===#


def _get_nvtx_register_constraint[dtype: DType]() -> StaticString:
    comptime assert is_nvidia_gpu(), (
        "the _get_nvtx_register_constraint function is currently restricted"
        " to only be defined on NVIDIA GPUs"
    )
    if dtype == .bool:
        return "b"
    if dtype.is_half_float():
        return "h"
    if dtype.is_integral():
        comptime width = bit_width_of[dtype]()
        if width == 16:
            return "c"
        if width == 32:
            return "r"
        if width == 64:
            return "l"
    if dtype == .float32:
        return "f"
    if dtype == .float64:
        return "d"

    return "<<unknown_register_constraint>>"


struct AMDBufferResource(TrivialRegisterPassable):
    """128-bit descriptor for a buffer resource on AMD GPUs.

    Used for buffer_load/buffer_store instructions.
    """

    var desc: SIMD[.uint32, 4]
    """The 128-bit buffer descriptor encoded as four 32-bit values."""

    @always_inline("nodebug")
    def __init__[
        dtype: DType
    ](
        out self,
        # TODO: This should propagate mutability correctly.
        # E.g. only allow AMDBufferResource.store when mutable.
        gds_ptr: Pointer[Scalar[dtype], ...],
        num_records: Int = Int(UInt32.MAX),
    ):
        """Constructs an AMD buffer resource descriptor.

        Parameters:
            dtype: Data type of the buffer elements.

        Args:
            gds_ptr: Pointer to the buffer in global memory.
            num_records: Number of records in the buffer.
        """
        comptime assert (
            is_amd_gpu()
        ), "The AMDBufferResource struct is only applicable on AMDGPU hardware."

        self.desc = SIMD[.uint32, 4](0)
        var address = bitcast[.uint32, 2](UInt64(Int(gds_ptr)))
        self.desc[0] = address[0]
        # assuming 0 stride currently
        self.desc[1] = address[1]
        self.desc[2] = UInt32(size_of[dtype]() * num_records)

        # Architecture-specific word 3 value for buffer resource.
        # https://github.com/ROCm/composable_kernel/blob/3b2302081eab4975370e29752343058392578bcb/include/ck/ck.hpp#L84
        comptime if _is_amd_rdna3() or _is_amd_rdna4():
            # GFX11/GFX12 (RDNA3/RDNA4)
            self.desc[3] = 0x31004000
        elif _is_amd_rdna1() or _is_amd_rdna2():
            # GFX10.x (RDNA1/RDNA2)
            self.desc[3] = 0x31014000
        else:
            comptime assert _is_amd_cdna(), (
                "The AMDBufferResource struct is only defined for CDNA"
                " and RDNA 1-4 GPUs."
            )
            # GFX9 (CDNA/GCN)
            self.desc[3] = 0x00020000

    @always_inline("nodebug")
    def __init__(out self):
        """Constructs a zeroed AMD buffer resource descriptor."""
        comptime assert (
            is_amd_gpu()
        ), "The AMDBufferResource struct is only applicable on AMDGPU hardware."
        self.desc = 0

    @always_inline("nodebug")
    def get_base_ptr(self) -> Int:
        """Gets the base pointer address from the buffer resource descriptor.

        Returns:
            The base pointer address as an integer.
        """
        return Int(
            bitcast[.int64, 1](SIMD[.uint32, 2](self.desc[0], self.desc[1]))
        )

    @always_inline("nodebug")
    def load[
        dtype: DType,
        width: Int,
        *,
        cache_policy: CacheOperation = CacheOperation.ALWAYS,
    ](
        self,
        vector_offset: Int32,
        *,
        scalar_offset: Int32 = 0,
    ) -> SIMD[
        dtype, width
    ]:
        """Loads data from the buffer using AMD buffer load intrinsic.

        Parameters:
            dtype: Data type to load.
            width: Number of elements to load.
            cache_policy: Cache operation policy.

        Args:
            vector_offset: Offset in elements from the base pointer.
            scalar_offset: Additional scalar offset in elements.

        Returns:
            SIMD vector containing the loaded data.
        """
        comptime assert (
            is_amd_gpu()
        ), "The buffer_load function is only applicable on AMDGPU hardware."

        comptime bytes = size_of[dtype]() * width
        comptime aux = _cache_operation_to_amd_aux[cache_policy]()

        var vector_offset_bytes = vector_offset * Int32(size_of[dtype]())
        var scalar_offset_bytes = scalar_offset * Int32(size_of[dtype]())

        var load_val = llvm_intrinsic[
            "llvm.amdgcn.raw.buffer.load",
            SIMD[
                _get_buffer_intrinsic_simd_dtype[bytes](),
                _get_buffer_intrinsic_simd_width[bytes](),
            ],
            has_side_effect=False,
        ](self.desc, vector_offset_bytes, scalar_offset_bytes, aux)

        return bitcast[dtype, width](load_val)

    @always_inline("nodebug")
    def load_to_lds[
        dtype: DType,
        *,
        width: Int = 1,
        cache_policy: CacheOperation = CacheOperation.ALWAYS,
        async_copies: Bool = False,
    ](
        self,
        vector_offset: Int32,
        shared_ptr: Pointer[Scalar[dtype], address_space=.SHARED, ...],
        *,
        scalar_offset: Int32 = 0,
    ):
        """Loads data from global memory and stores to shared memory.

        Copies from global memory to shared memory (aka LDS) bypassing storing to
        register.

        Uses the `.ptr.` form (descriptor as `ptr addrspace(8)`) over the
        legacy `<4 x i32>` form. Both lower to the same MUBUF
        `buffer_load_*_lds` instruction on gfx9x/CDNA, but the `.ptr.`
        form exposes the descriptor as a typed pointer so
        `ScopedNoAliasAA` / `SIInsertWaitcnts` can reason about it. The
        legacy form produced a 0.76-abs MLA decode regression at
        output[0,0,0,0] when used from attention DMAs — the `.ptr.`
        form is MLA-safe.

        Parameters:
            dtype: The dtype of the data to be loaded.
            width: The SIMD vector width.
            cache_policy: Cache operation policy controlling cache behavior at all levels.
            async_copies: If True, attach the `amdgpu.AsyncCopies` alias
                scope to the load — unlocks `ScopedNoAliasAA`-driven
                optimizations (LICM, CSE, reordering) and, on AMDGPU,
                the vmcnt relaxation in `SIInsertWaitcnts` (LLVM
                PR #74537): a later `ds_read` tagged `noalias` against
                the same scope can skip `s_waitcnt vmcnt(0)`. Only set
                True when (a) every LDS read of this data carries the
                matching `noalias` tag, AND (b) the kernel maintains an
                explicit runtime fence (`s_waitcnt vmcnt(0)` +
                `s_barrier`) — scheduling hints like
                `s_sched_group_barrier` do NOT qualify. Defaults to
                False (safe for all callers). Future extension: the
                backend can bucket up to 8 distinct scopes independently
                (LDSDMAStores slots), so more scope variants could be
                added here if a kernel wants multiple independent DMA
                streams.

        Args:
            vector_offset: Vector memory offset in elements (per thread).
            shared_ptr: Shared memory address.
            scalar_offset: Scalar memory offset in elements (shared across wave).
        """
        comptime assert (
            is_amd_gpu()
        ), "The buffer_load_lds function is only applicable on AMDGPU hardware."

        comptime bytes = size_of[dtype]() * width
        comptime aux = _cache_operation_to_amd_aux[cache_policy]()

        var vector_offset_bytes = vector_offset * Int32(size_of[dtype]())
        var scalar_offset_bytes = scalar_offset * Int32(size_of[dtype]())

        # Convert the SIMD[uint32, 4] descriptor to a `ptr addrspace(8)`
        # so the `.ptr.` form of the intrinsic accepts it.
        var desc_ptr = Pointer[
            BFloat16, MutAnyOrigin, address_space=.BUFFER_RESOURCE
        ].unsafe_dangling()
        var ptr_to_ptr = Pointer(to=desc_ptr)
        var ptr_to_simd = Pointer(to=self.desc)
        ptr_to_ptr[] = ptr_to_simd.unsafe_bitcast[
            Pointer[BFloat16, MutAnyOrigin, address_space=.BUFFER_RESOURCE]
        ]()[]

        comptime if not async_copies:
            # No alias-scope metadata — clean `llvm_intrinsic` call path.
            llvm_intrinsic[
                "llvm.amdgcn.raw.ptr.buffer.load.lds",
                NoneType,
                has_side_effect=True,
            ](
                desc_ptr,
                shared_ptr,
                Int32(bytes),
                vector_offset_bytes,
                scalar_offset_bytes,
                Int32(0),
                aux,
            )
        else:
            # Attach `amdgpu.AsyncCopies` alias scope via the
            # `rocdl.raw.ptr.buffer.load.lds` MLIR op (the `llvm_intrinsic`
            # wrapper can't attach arbitrary attrs).
            var desc_ptr_llvm = __mlir_op.`builtin.unrealized_conversion_cast`[
                _type=__mlir_type.`!llvm.ptr<8>`
            ](desc_ptr)
            var shared_ptr3 = __mlir_op.`builtin.unrealized_conversion_cast`[
                _type=__mlir_type.`!llvm.ptr<3>`
            ](shared_ptr)
            comptime async_copies_scope = __mlir_attr.`[#llvm.alias_scope<id= "amdgpu.AsyncCopies", domain=#llvm.alias_scope_domain<id = "amdgpu.AsyncOps">>]`
            _raw_ptr_buffer_load_lds[async_copies_scope, Int(aux)](
                desc_ptr_llvm,
                shared_ptr3,
                to_i32(Int32(bytes)),
                to_i32(vector_offset_bytes),
                to_i32(scalar_offset_bytes),
                to_i32(Int32(0)),
            )

    @always_inline("nodebug")
    def store[
        dtype: DType,
        width: SIMDLength,
        *,
        cache_policy: CacheOperation = CacheOperation.ALWAYS,
    ](
        self,
        vector_offset: Int32,
        val: SIMD[dtype, width],
        *,
        scalar_offset: Int32 = 0,
    ):
        """Stores a register variable to global memory with cache operation control.

        Writes to global memory from a register with high-level cache control.

        Parameters:
            dtype: The data type.
            width: The SIMD vector width.
            cache_policy: Cache operation policy controlling cache behavior at all levels.

        Args:
            vector_offset: Vector memory offset in elements (per thread).
            val: Value to write.
            scalar_offset: Scalar memory offset in elements (shared across wave).

        Note:
            - Only supported on AMD GPUs.
            - Provides high-level cache control via CacheOperation enum values.
            - Maps directly to llvm.amdgcn.raw.buffer.store intrinsics.
            - Cache control bits:
            - SC[1:0] controls coherency scope: 0=wave, 1=group, 2=device, 3=system.
            - nt=True: Use streaming-optimized cache policies (recommended for streaming data).
        """
        comptime assert (
            is_amd_gpu()
        ), "The buffer_store function is only applicable on AMDGPU hardware."

        comptime bytes = width * size_of[dtype]()
        comptime aux: Int32 = _cache_operation_to_amd_aux[cache_policy]()

        var vector_offset_bytes = vector_offset * Int32(size_of[dtype]())
        var scalar_offset_bytes = scalar_offset * Int32(size_of[dtype]())

        var store_val = bitcast[
            _get_buffer_intrinsic_simd_dtype[bytes](),
            _get_buffer_intrinsic_simd_width[bytes](),
        ](val)

        llvm_intrinsic[
            "llvm.amdgcn.raw.buffer.store", NoneType, has_side_effect=True
        ](
            store_val,
            self.desc,
            vector_offset_bytes,
            scalar_offset_bytes,
            aux,
        )


def _cache_operation_to_amd_aux[cache_policy: CacheOperation]() -> Int32:
    """Converts CacheOperation to AMD auxiliary parameter at compile time.

    Parameters:
        cache_policy: The cache operation policy.

    Returns:
        The auxiliary parameter value formatted for AMD buffer operations.
        Format: bit 0 = SC0, bit 1 = NT, bit 4 = SC1
    """

    comptime if cache_policy == CacheOperation.ALWAYS:
        return 0x00  # SC=00, NT=0
    elif cache_policy == CacheOperation.STREAMING:
        return 0x02  # SC=00, NT=1
    elif cache_policy == CacheOperation.GLOBAL:
        return 0x10  # SC=10, NT=0
    elif cache_policy == CacheOperation.VOLATILE:
        return 0x11  # SC=11, NT=0
    elif cache_policy == CacheOperation.WORKGROUP | CacheOperation.STREAMING:
        return 0x03  # SC=01, NT=1
    elif cache_policy == CacheOperation.GLOBAL | CacheOperation.STREAMING:
        return 0x12  # SC=10, NT=1
    else:
        # Default to ALWAYS for unknown/unsupported operations
        return 0x00

    # Additional cache operations for potential future support:
    # CacheOperation.WORKGROUP -> 0x01 (SC=01, NT=0) - Workgroup/CU-level coherency
    # CacheOperation.GLOBAL_STREAMING -> 0x12 (SC=10, NT=1) - Global + streaming
    # CacheOperation.VOLATILE_STREAMING -> 0x13 (SC=11, NT=1) - Volatile + streaming


@always_inline("nodebug")
def _raw_ptr_buffer_load_lds[
    scopes: __mlir_type.`!kgen.deferred`, aux: Int
](
    desc_ptr_llvm: __mlir_type.`!llvm.ptr<8>`,
    shared_ptr3: __mlir_type.`!llvm.ptr<3>`,
    size: __mlir_type.i32,
    voffset: __mlir_type.i32,
    soffset: __mlir_type.i32,
    offset: __mlir_type.i32,
):
    # The cache-policy `aux` slot is an `I32Attr`, i.e. a plain signless
    # builtin `i32` IntegerAttr literal (`N : i32`). It cannot be fed a value
    # expression (`to_i32(...)`) nor an `int_literal_convert` /
    # `cast_to_builtin` attribute — both produce non-builtin attrs the
    # `non-atomic AMDGPU buffer cache policy` verifier rejects. Splicing the
    # comptime-constant `aux` with a builtin type annotation
    # (`__mlir_attr[..., `: i32`]`) folds to the same builtin `N : i32`
    # IntegerAttr literal the verifier accepts. `scopes` (the `alias_scopes`
    # attribute) is threaded through unchanged so its identity is preserved.
    comptime assert aux in (
        0x00,
        0x02,
        0x03,
        0x10,
        0x11,
        0x12,
    ), "unsupported AMD buffer cache-policy aux bitmask"
    __mlir_op.`rocdl.raw.ptr.buffer.load.lds`[
        alias_scopes=scopes,
        aux=__mlir_attr[aux.__mlir_index__(), `: i32`],
        _type=None,
    ](desc_ptr_llvm, shared_ptr3, size, voffset, soffset, offset)


def _get_buffer_intrinsic_simd_dtype[bytes: Int]() -> DType:
    comptime if bytes == 1:
        return DType.uint8
    elif bytes == 2:
        return DType.uint16
    else:
        comptime assert bytes in (4, 8, 16), "Width not supported"
        return DType.uint32


def _get_buffer_intrinsic_simd_width[bytes: Int]() -> Int:
    return bytes // size_of[DType.uint32]() if bytes >= 4 else 1


# ===-----------------------------------------------------------------------===#
# AMD LDS transpose reads (ds.read.tr*)
# ===-----------------------------------------------------------------------===#


@always_inline
def ds_read_tr16_b64[
    dtype: DType,
    //,
](shared_ptr: Pointer[Scalar[dtype], address_space=.SHARED, ...]) -> SIMD[
    dtype, 4
]:
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

    comptime assert (
        is_amd_gpu()
    ), "The ds_read_tr16_b64 function is only applicable on AMDGPU hardware."

    comptime assert (
        size_of[dtype]() == 2
    ), "ds_read_tr16_b64 supports 16-bit dtypes."

    comptime assert (
        _cdna_4_or_newer()
    ), "ds_read_tr16_b64 is only supported on CDNA4+"

    return llvm_intrinsic[
        "llvm.amdgcn.ds.read.tr16.b64", SIMD[dtype, 4], has_side_effect=True
    ](shared_ptr)


@always_inline
def ds_read_tr8_b64[
    dtype: DType,
    //,
](shared_ptr: Pointer[Scalar[dtype], address_space=.SHARED, ...]) -> SIMD[
    dtype, 8
]:
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

    comptime assert (
        is_amd_gpu()
    ), "The ds_read_tr8_b64 function is only applicable on AMDGPU hardware."

    comptime assert (
        size_of[dtype]() == 1
    ), "ds_read_tr8_b64 supports 8-bit dtypes."

    comptime assert (
        _cdna_4_or_newer()
    ), "ds_read_tr8_b64 is only supported on CDNA4+"

    # Must use v2i32 return type; v8i8 crashes LLVM type legalizer.
    var raw = llvm_intrinsic[
        "llvm.amdgcn.ds.read.tr8.b64",
        SIMD[.uint32, 2],
        has_side_effect=True,
    ](shared_ptr)
    return bitcast[dtype, 8](raw)


# ===-----------------------------------------------------------------------===#
# AMD f32 -> fp8 raw packed conversion (no clamp / NaN scrub)
# ===-----------------------------------------------------------------------===#


@always_inline
def cvt_pk_fp8_f32_raw[
    dtype: DType,
](src: SIMD[.float32, 4]) -> SIMD[dtype, 4]:
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
    comptime assert (
        is_amd_gpu()
    ), "cvt_pk_fp8_f32_raw is only supported on AMDGPU hardware."
    comptime assert (
        _cdna_4_or_newer()
    ), "cvt_pk_fp8_f32_raw is only supported on CDNA4+"
    comptime assert (
        dtype == .float8_e4m3fn or dtype == .float8_e5m2
    ), "cvt_pk_fp8_f32_raw requires E4M3FN or E5M2 destination dtype."

    comptime intrinsic_name = (
        "llvm.amdgcn.cvt.pk.fp8.f32" if dtype
        == .float8_e4m3fn else "llvm.amdgcn.cvt.pk.bf8.f32"
    )

    var lo = llvm_intrinsic[intrinsic_name, UInt32, has_side_effect=False](
        src[0], src[1], UInt32(0), False
    )
    var packed = llvm_intrinsic[intrinsic_name, UInt32, has_side_effect=False](
        src[2], src[3], lo, True
    )
    return bitcast[dtype, 4](packed)


# ===-----------------------------------------------------------------------===#
# AMD permlane shuffle
# ===-----------------------------------------------------------------------===#


@always_inline
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
    comptime assert (
        is_amd_gpu()
    ), "The _amd_permlane_swap function is only applicable on AMDGPU hardware."
    comptime assert (
        _cdna_4_or_newer()
    ), "permlane swap is only supported on CDNA4+"
    comptime assert bit_width_of[dtype]() == 32, "Unsupported dtype"
    comptime assert stride in (16, 32), "Unsupported stride"

    comptime asm = "llvm.amdgcn.permlane" + String(stride) + ".swap"
    var result = llvm_intrinsic[
        asm,
        _RegisterPackType[Int32, Int32],
        has_side_effect=False,
    ](
        bitcast[.int32, 1](val1),
        bitcast[.int32, 1](val2),
        False,
        False,
    )
    return SIMD[dtype, 2](
        bitcast[dtype, 1](result[0]), bitcast[dtype, 1](result[1])
    )


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
    var lane_group = ufloordiv(lane_id(), stride)

    var out = type_of(res)()

    comptime for i in range(simd_width):
        out[i] = permlane_swap[stride](val[i], val[i])[umod(lane_group + 1, 2)]
    return out
