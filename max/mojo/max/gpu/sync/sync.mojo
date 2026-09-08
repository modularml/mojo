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
"""This module provides GPU synchronization primitives and barriers.

The module includes:
- Block-level synchronization barriers (barrier())
- Warp-level synchronization (syncwarp())
- Memory barriers (mbarrier) for NVIDIA GPUs
- Instruction scheduling controls for AMD GPUs
- Asynchronous copy and bulk transfer synchronization

The synchronization primitives help coordinate execution between threads within
thread blocks and warps, and manage memory consistency across different memory spaces.
"""

from std.os import abort
from std.atomic import Ordering, fence
from std.sys import is_amd_gpu, is_apple_gpu, is_nvidia_gpu, llvm_intrinsic
from std.sys._assembly import inlined_assembly
from std.sys.info import CompilationTarget, _is_amd_cdna
from std.sys.defines import get_defined_bool

from std._gpu.intrinsics import Scope

from std._gpu._utils import to_i32, to_llvm_shared_mem_ptr

# ===-----------------------------------------------------------------------===#
# barrier
# ===-----------------------------------------------------------------------===#

comptime _USE_EXPERIMENTAL_AMD_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM = get_defined_bool[
    "USE_EXPERIMENTAL_AMD_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM", False
]()

comptime MaxHardwareBarriers = 16
"""Maximum number of hardware barriers available per block."""


@always_inline("nodebug")
def named_barrier[
    num_threads: Int32,
](id: Int32 = 0):
    """Performs a named synchronization barrier at the block level.

    This function creates a synchronization point using a specific barrier ID, allowing
    for multiple independent barriers within a thread block. All threads in the block
    must execute this function with the same barrier ID and thread count before any
    thread can proceed past the barrier.

    Parameters:
        num_threads: The number of threads that must reach the barrier before any can proceed.

    Args:
        id: The barrier identifier (0-16). Default is 0.

    Notes:

        - Only supported on NVIDIA GPUs.
        - Maps directly to the `nvvm.barrier` instruction.
        - Useful for fine-grained synchronization when different subsets of threads
          need to synchronize independently.
        - The barrier ID must not exceed 16.
        - All threads participating in the barrier must specify the same num_threads value.
        - The number of threads value must be a multiple of the warp size.
    """

    assert id < MaxHardwareBarriers, "barrier id should not exceed 16"
    comptime assert (
        is_nvidia_gpu()
    ), "named barrier is only supported by NVIDIA GPUs"
    _ = __mlir_op.`nvvm.barrier`[
        _properties=__mlir_attr.`{operandSegmentSizes = array<i32: 1, 1>}`,
    ](to_i32(id), to_i32(num_threads))


@always_inline("nodebug")
def named_barrier_arrive[
    num_threads: Int32,
](id: Int32 = 0):
    """Arrives at a named synchronization barrier at the block level.

    This function marks the arrival of a named synchronization barrier point using a specific barrier ID.

    Parameters:
        num_threads: The number of threads that must reach the barrier before any can proceed.

    Args:
        id: The barrier identifier (0-16). Default is 0.

    Notes:
        - Only supported on NVIDIA GPUs.
        - Maps directly to the `nvvm.barrier.arrive` instruction.
        - Useful for fine-grained synchronization when different subsets of threads
          need to synchronize independently.
        - The barrier ID must not exceed 16.
        - All threads participating in the barrier must specify the same num_threads value.
    """
    assert id < MaxHardwareBarriers, "barrier id should not exceed 16"
    comptime assert (
        is_nvidia_gpu()
    ), "named barrier is only supported by NVIDIA GPUs"
    __mlir_op.`nvvm.barrier.arrive`(to_i32(id), to_i32(num_threads))


@always_inline("nodebug")
def barrier():
    """Performs a synchronization barrier at the block level.

    This is equivalent to __syncthreads() in CUDA. All threads in a thread block must
    execute this function before any thread can proceed past the barrier. This ensures
    memory operations before the barrier are visible to all threads after the barrier.
    """

    comptime if is_nvidia_gpu():
        _ = __mlir_op.`nvvm.barrier`[
            _properties=__mlir_attr.`{operandSegmentSizes = array<i32: 0, 0>}`,
        ]()
    elif _USE_EXPERIMENTAL_AMD_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM:
        comptime assert is_amd_gpu()
        llvm_intrinsic["llvm.amdgcn.s.waitcnt", NoneType](Int32(0xC07F))
        llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()
    elif is_amd_gpu():
        fence[Ordering.RELEASE, scope="workgroup"]()
        llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()
        fence[Ordering.ACQUIRE, scope="workgroup"]()
    elif is_apple_gpu():
        # threadgroup_barrier(mem_flags::mem_threadgroup)
        llvm_intrinsic["llvm.air.wg.barrier", NoneType](Int32(2), Int32(1))
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name()
        ]()


@fieldwise_init
struct AMDScheduleBarrierMask(
    Equatable, Intable, TrivialRegisterPassable, Writable
):
    """Represents different instruction scheduling masks for AMDGPU scheduling instructions.

    These masks control which types of instructions can be reordered across a barrier for
    performance optimization. When used with schedule_barrier(), the mask determines which
    instructions the compiler is allowed to move across the barrier point.
    """

    var _value: Int32
    """Internal value storage for the barrier mask."""

    comptime NONE = Self(0)
    """No instructions can cross the barrier. Most restrictive option."""

    comptime ALL_ALU = Self(1 << 0)
    """Allows reordering of all arithmetic and logic instructions that don't involve memory operations."""

    comptime VALU = Self(1 << 1)
    """Permits reordering of vector arithmetic/logic unit instructions only."""

    comptime SALU = Self(1 << 2)
    """Permits reordering of scalar arithmetic/logic unit instructions only."""

    comptime MFMA = Self(1 << 3)
    """Allows reordering of matrix multiplication and WMMA instructions."""

    comptime ALL_VMEM = Self(1 << 4)
    """Enables reordering of all vector memory operations (reads and writes)."""

    comptime VMEM_READ = Self(1 << 5)
    """Allows reordering of vector memory read operations only."""

    comptime VMEM_WRITE = Self(1 << 6)
    """Allows reordering of vector memory write operations only."""

    comptime ALL_DS = Self(1 << 7)
    """Permits reordering of all Local Data Share (LDS) operations."""

    comptime DS_READ = Self(1 << 8)
    """Enables reordering of LDS read operations only."""

    comptime DS_WRITE = Self(1 << 9)
    """Enables reordering of LDS write operations only."""

    comptime TRANS = Self(1 << 10)
    """Allows reordering of transcendental instructions (sin, cos, exp, etc)."""

    def __init__(out self, value: Int):
        """Initializes an `AMDScheduleBarrierMask` from an integer value.

        This implicit constructor allows creating a barrier mask directly from an integer,
        which is useful for combining multiple mask flags using bitwise operations.

        Args:
            value: The integer value to use for the barrier mask.
        """
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        """Compares two `AMDScheduleBarrierMask` instances for equality.

        Args:
            other: The other `AMDScheduleBarrierMask` to compare with.

        Returns:
            True if the masks have the same value, False otherwise.
        """
        return self._value == other._value

    def write_to(self, mut writer: Some[Writer]):
        """Writes a string representation of the `AMDScheduleBarrierMask`.

        Converts the mask to a human-readable string based on its value.

        Args:
            writer: The object to write to.
        """
        if self == Self.NONE:
            return writer.write_string("NONE")
        elif self == Self.ALL_ALU:
            return writer.write_string("ALL_ALU")
        elif self == Self.VALU:
            return writer.write_string("VALU")
        elif self == Self.SALU:
            return writer.write_string("SALU")
        elif self == Self.MFMA:
            return writer.write_string("MFMA")
        elif self == Self.ALL_VMEM:
            return writer.write_string("ALL_VMEM")
        elif self == Self.VMEM_READ:
            return writer.write_string("VMEM_READ")
        elif self == Self.VMEM_WRITE:
            return writer.write_string("VMEM_WRITE")
        elif self == Self.ALL_DS:
            return writer.write_string("ALL_DS")
        elif self == Self.DS_READ:
            return writer.write_string("DS_READ")
        elif self == Self.DS_WRITE:
            return writer.write_string("DS_WRITE")
        elif self == Self.TRANS:
            return writer.write_string("TRANS")
        else:
            abort("invalid AMDScheduleBarrierMask value")

    def __int__(self) -> Int:
        """Converts the `AMDScheduleBarrierMask` to an integer.

        Returns:
            The integer value of the mask, which can be used with low-level APIs.
        """
        return Int(self._value)


@always_inline("nodebug")
def schedule_barrier(
    mask: AMDScheduleBarrierMask = AMDScheduleBarrierMask.NONE,
):
    """Controls instruction scheduling across a barrier point in AMD GPU code.

    This function creates a scheduling barrier that controls which types of instructions
    can be reordered across it by the compiler. The mask parameter specifies which
    instruction categories (ALU, memory, etc) are allowed to cross the barrier during
    scheduling optimization.

    Args:
        mask: A bit mask of AMDScheduleBarrierMask flags indicating which instruction
            types can be scheduled across this barrier. Default is NONE, meaning no
            instructions can cross.

    Note:
        This function only has an effect on AMD GPUs. On other platforms it will
        raise a compile time error.
    """

    comptime assert (
        is_amd_gpu()
    ), "schedule_barrier is only supported on AMDGPU."
    llvm_intrinsic["llvm.amdgcn.sched.barrier", NoneType](Int32(Int(mask)))


@always_inline("nodebug")
def schedule_group_barrier(
    mask: AMDScheduleBarrierMask, size: Int32, sync_id: Int32
):
    """Controls instruction scheduling across a barrier point in AMD GPU code by creating schedule groups.

    This function creates a scheduling barrier that groups instructions into sequences with custom ordering.
    It affects the code that precedes the barrier. The barrier ensures instructions are scheduled according
    to the specified group parameters.

    Args:
        mask: A bit mask of AMDScheduleBarrierMask flags indicating which instruction types can be
            scheduled across this barrier. Similar to schedule_barrier masks.
        size: The number of times to repeat the instruction sequence in the schedule group.
        sync_id: A unique identifier for the group that determines the ordering of instructions
            within the same schedule group.

    Note:
        This function only has an effect on AMD GPUs. On other platforms it will raise a compile time error.
        The sync_id parameter allows creating multiple schedule groups that can be ordered relative to each other.
    """

    comptime assert (
        is_amd_gpu()
    ), "schedule_group_barrier is only supported on AMDGPU."
    llvm_intrinsic["llvm.amdgcn.sched.group.barrier", NoneType](
        Int32(Int(mask)), size, sync_id
    )


# reference for waitcnt_arg and related synchronization utilities:
# https://github.com/ROCm/rocm-libraries/blob/5ba64a9aa8557e465d1a5937609e23f34adb601e/projects/composablekernel/include/ck_tile/core/arch/arch.hpp#L140


struct _WaitCountArg:
    """
    Mojo struct to encapsulate waitcnt argument bitfields and helpers.
    """

    # [V]M [E]XP [L]GKM counters and [U]NUSED ---> VV'UU'LLLL'U'EEE'VVVV

    # Constants
    comptime MAX: UInt32 = 0b1100_1111_0111_1111
    comptime MAX_VM_CNT: UInt32 = 0b111111
    comptime MAX_EXP_CNT: UInt32 = 0b111
    comptime MAX_LGKM_CNT: UInt32 = 0b1111

    @staticmethod
    def from_vmcnt(cnt: UInt32) -> UInt32:
        comptime assert (
            _is_amd_cdna()
        ), "from_vmcnt is only supported on AMD CDNA GPUs"
        assert (
            cnt <= Self.MAX_VM_CNT
        ), "cnt should be less than or equal to MAX_VM_CNT = 63"
        return Self.MAX & ((cnt & 0b1111) | ((cnt & 0b110000) << 10))

    @staticmethod
    def from_expcnt(cnt: UInt32) -> UInt32:
        comptime assert (
            _is_amd_cdna()
        ), "from_expcnt is only supported on AMD CDNA GPUs"
        assert (
            cnt <= Self.MAX_EXP_CNT
        ), "cnt should be less than or equal to MAX_EXP_CNT = 7"
        return Self.MAX & (cnt << 4)

    @staticmethod
    def from_lgkmcnt(cnt: UInt32) -> UInt32:
        comptime assert (
            _is_amd_cdna()
        ), "from_lgkmcnt is only supported on AMD CDNA GPUs"
        assert (
            cnt <= Self.MAX_LGKM_CNT
        ), "cnt should be less than or equal to MAX_LGKM_CNT = 15"
        return Self.MAX & (cnt << 8)


@always_inline
def s_waitcnt[
    *,
    vmcnt: UInt32 = _WaitCountArg.MAX_VM_CNT,
    expcnt: UInt32 = _WaitCountArg.MAX_EXP_CNT,
    lgkmcnt: UInt32 = _WaitCountArg.MAX_LGKM_CNT,
]():
    """Performs an `s_waitcnt` instruction with the specified counters on AMD GPUs.

    This function waits until the number of *remaining* outstanding memory operations
    matches the given counter values: vector (vmcnt), export (expcnt), and LGKM
    (local/global/memory/constant, lgkmcnt). It does not wait for this many instructions
    to complete, but for the counters to have these exact values before proceeding.

    Only effective on AMD GPUs.

    Parameters:
        vmcnt: Waits until the remaining VMEM instructions is equal to this value (default: max).
        expcnt: Waits until the remaining export (GDS) instructions is equal to this value (default: max).
        lgkmcnt: Waits until the remaining LDS, GDS, constant-fetch, and message instructions is equal to this value (default: max).

    Note:
        - Only has an effect on AMD GPUs. On other platforms, raises a compile-time error if called.
        - The counters should be set carefully to avoid deadlocks or missed synchronization.
        - For example, s_waitcnt(vmcnt=0, expcnt=0, lgkmcnt=0) waits until all memory operations are complete.
    """
    comptime assert (
        _is_amd_cdna()
    ), "s_waitcnt is only supported on AMD CDNA GPUs"
    comptime waitcnt_val = (
        _WaitCountArg.from_vmcnt(vmcnt)
        | _WaitCountArg.from_expcnt(expcnt)
        | _WaitCountArg.from_lgkmcnt(lgkmcnt)
    )
    llvm_intrinsic["llvm.amdgcn.s.waitcnt", NoneType](waitcnt_val)


@always_inline
def s_waitcnt_barrier[
    *,
    vmcnt: UInt32 = _WaitCountArg.MAX_VM_CNT,
    expcnt: UInt32 = _WaitCountArg.MAX_EXP_CNT,
    lgkmcnt: UInt32 = _WaitCountArg.MAX_LGKM_CNT,
]():
    """Performs an `s_waitcnt` followed by a barrier on AMD GPUs.

    This function first waits until the number of outstanding memory operations
    matches the specified values (see s_waitcnt above), then forces all threads
    in the block to synchronize via a barrier. Only effective on AMD GPUs.

    Parameters:
        vmcnt: Waits until the remaining VMEM instructions is equal to this value (default: max).
        expcnt: Waits until the remaining export (GDS) instructions is equal to this value (default: max).
        lgkmcnt: Waits until the remaining LDS, GDS, constant-fetch, and message instructions is equal to this value (default: max).

    Note:
        - Only has an effect on AMD GPUs. On other platforms, raises a compile-time error if called.
        - Use this to guarantee memory visibility and thread ordering for precise synchronization.
        - For example, s_waitcnt_barrier(0,0,0) ensures all outstanding memory instructions are completed and then all threads are synchronized.
    """
    comptime assert (
        _is_amd_cdna()
    ), "s_waitcnt_barrier is only supported on AMD CDNA GPUs"
    s_waitcnt[vmcnt=vmcnt, expcnt=expcnt, lgkmcnt=lgkmcnt]()
    llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()


# ===-----------------------------------------------------------------------===#
# syncwarp
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def syncwarp(mask: Int = -1):
    """Synchronizes threads within a warp using a barrier.

    This function creates a synchronization point where threads in a warp must wait until all
    threads specified by the mask reach this point. On NVIDIA GPUs, it uses warp-level
    synchronization primitives. On AMD GPUs, this acts as a wave execution barrier.
    On Apple GPUs, this acts as a SIMDGROUP execution barrier. Lane masks are not supported,
    so the mask argument is ignored and all active lanes must reach this point.

    Args:
        mask: An integer bitmask specifying which lanes (threads) in the warp should be
            synchronized. Each bit corresponds to a lane, with bit i controlling lane i.
            A value of 1 means the lane participates in the sync, 0 means it does not.
            Default value of -1 (all bits set) synchronizes all lanes.

    Note:
        - On NVIDIA GPUs, this maps to the nvvm.bar.warp.sync intrinsic.
        - On AMD GPUs, this maps to the llvm.amdgcn.wave.barrier intrinsic.
        - On Apple GPUs, this provides *execution synchronization only* via a SIMDGROUP
          barrier with `mem_none` (no memory fence). Use `barrier()` for threadgroup
          memory ordering.
        - Threads not participating in the sync must still execute the instruction.
    """

    comptime if is_nvidia_gpu():
        __mlir_op.`nvvm.bar.warp.sync`(
            __mlir_op.`index.casts`[_type=__mlir_type.i32](
                mask.__mlir_index__()
            )
        )
    elif is_amd_gpu():
        llvm_intrinsic["llvm.amdgcn.wave.barrier", NoneType]()
    elif is_apple_gpu():
        # simdgroup_barrier(mem_flags::mem_none)
        llvm_intrinsic["llvm.air.simdgroup.barrier", NoneType](
            Int32(0), Int32(4)
        )
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name()
        ]()


# ===-----------------------------------------------------------------------===#
# mbarrier
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def _mbarrier_impl[
    type: AnyType, address_space: AddressSpace
](address: Pointer[mut=True, type, _, address_space=address_space]):
    """Internal implementation for making a memory barrier track async operations.

    This is an internal helper function that implements the core memory barrier tracking
    functionality for different address spaces.

    Args:
        address: Pointer to the memory barrier object location.
    """

    comptime if address_space == .SHARED:
        llvm_intrinsic["llvm.nvvm.cp.async.mbarrier.arrive.shared", NoneType](
            address
        )
    elif (address_space == .GLOBAL or address_space == .GENERIC):
        llvm_intrinsic["llvm.nvvm.cp.async.mbarrier.arrive", NoneType](
            address.unsafe_address_space_cast[.GENERIC]()._get_kgen_pointer()
        )
    else:
        comptime assert False, "invalid address space"


@always_inline("nodebug")
def _mbarrier_noinc_impl[
    type: AnyType, address_space: AddressSpace
](address: Pointer[mut=True, type, _, address_space=address_space]):
    """Internal noinc implementation for making a memory barrier track async operations.

    The noinc variant does not increment the expected transaction count.
    It still tracks the async copies for mbarrier completion, but avoids
    incrementing outstanding bytes.

    Args:
        address: Pointer to the memory barrier object location.
    """

    comptime if address_space == .SHARED:
        llvm_intrinsic[
            "llvm.nvvm.cp.async.mbarrier.arrive.noinc.shared", NoneType
        ](address)
    else:
        comptime assert False, "invalid address space"


@always_inline("nodebug")
def async_copy_arrive[
    type: AnyType, address_space: AddressSpace, *, noinc: Bool = False
](address: Pointer[mut=True, type, _, address_space=address_space]):
    """Makes a memory barrier track all prior async copy operations from this thread.

    This function ensures that all previously initiated asynchronous copy operations
    from the executing thread are tracked by the memory barrier at the specified location.
    Only supported on NVIDIA GPUs.

    When `noinc` is True, the increment to the pending count of the mbarrier
    object is not performed. The decrement of the pending count done by the
    asynchronous arrive-on operation must be accounted for in the
    initialization of the mbarrier object.

    Parameters:
        type: The data type stored at the barrier location.
        address_space: The memory address space where the barrier is located.
        noinc: If True, do not increment the pending count. Defaults to False.

    Args:
        address: Pointer to the memory barrier object location.
    """

    comptime if is_nvidia_gpu():
        comptime if noinc:
            _mbarrier_noinc_impl(address)
        else:
            _mbarrier_impl(address)
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name(),
            note="async_copy_arrive() is only supported when targeting NVIDIA GPUs",
        ]()


@always_inline("nodebug")
def mbarrier_init[
    type: AnyType
](
    shared_mem: Pointer[mut=True, type, _, address_space=.SHARED],
    num_threads: Int32,
):
    """Initialize a shared memory barrier for synchronizing multiple threads.

    Sets up a memory barrier in shared memory that will be used to synchronize
    the specified number of threads. Only supported on NVIDIA GPUs.

    Parameters:
        type: The data type stored at the barrier location.

    Args:
        shared_mem: Pointer to shared memory location for the barrier.
        num_threads: Number of threads that will synchronize on this barrier.
    """

    comptime if is_nvidia_gpu():
        llvm_intrinsic["llvm.nvvm.mbarrier.init.shared", NoneType](
            shared_mem, num_threads
        )
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name(),
            note="mbarrier_init() is only supported when targeting NVIDIA GPUs",
        ]()


@always_inline("nodebug")
def mbarrier_arrive[
    type: AnyType
](shared_mem: Pointer[mut=True, type, _, address_space=.SHARED]) -> Int:
    """Signal thread arrival at a shared memory barrier.

    Records that the calling thread has reached the barrier synchronization point.
    Only supported on NVIDIA GPUs.

    Parameters:
        type: The data type stored at the barrier location.

    Args:
        shared_mem: Pointer to the shared memory barrier.

    Returns:
        An integer representing the current state of the memory barrier.
    """

    comptime if is_nvidia_gpu():
        return llvm_intrinsic["llvm.nvvm.mbarrier.arrive.shared", Int](
            shared_mem
        )
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name(),
            note="mbarrier_arrive() is only supported when targeting NVIDIA GPUs",
        ]()


@always_inline("nodebug")
def mbarrier_test_wait[
    type: AnyType
](
    shared_mem: Pointer[mut=True, type, _, address_space=.SHARED],
    state: Int,
) -> Bool:
    """Test if all threads have arrived at the memory barrier.

    Non-blocking check to see if all participating threads have reached the barrier.
    Only supported on NVIDIA GPUs.

    Parameters:
        type: The data type stored at the barrier location.

    Args:
        shared_mem: Pointer to the shared memory barrier.
        state: Expected state of the memory barrier.

    Returns:
        True if all threads have arrived, False otherwise.
    """

    comptime if is_nvidia_gpu():
        return llvm_intrinsic["llvm.nvvm.mbarrier.test.wait.shared", Bool](
            shared_mem, state
        )
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name(),
            note="mbarrier_test_wair() is only supported when targeting NVIDIA GPUs",
        ]()


@always_inline("nodebug")
def mbarrier_arrive_expect_tx_shared[
    type: AnyType  # The type of the memory barrier
](addr: Pointer[mut=True, type, _, address_space=.SHARED], tx_count: Int32,):
    """Configure a shared memory barrier to expect additional async transactions.

    Updates the current phase of the memory barrier to track completion of
    additional asynchronous transactions. Only supported on NVIDIA GPUs.

    Parameters:
        type: The type of the memory barrier.

    Args:
        addr: Pointer to the shared memory barrier.
        tx_count: Number of expected transactions to track.
    """

    comptime if is_nvidia_gpu():
        _ = __mlir_op.`nvvm.mbarrier.arrive.expect_tx`[_type=__mlir_type.i64](
            to_llvm_shared_mem_ptr(addr), to_i32(tx_count)
        )
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name(),
            note="mbarrier_arrive_expect_tx_shared() is only supported when targeting NVIDIA GPUs",
        ]()


@always_inline("nodebug")
def mbarrier_arrive_expect_tx_relaxed[
    type: AnyType,  # The type of the memory barrier
    scope: Scope = Scope.BLOCK,
    space: Scope = Scope.BLOCK,
](
    addr: Pointer[mut=True, type, _, address_space=.SHARED],
    tx_count: Int32,
) -> UInt64:
    """Configure a shared memory barrier to expect additional async transactions.

    Updates the current phase of the memory barrier to track completion of
    additional asynchronous transactions. Only supported on NVIDIA GPUs.

    Parameters:
        type: The type of the memory barrier.
        scope: The scope of the memory barrier.
        space: The space of the memory barrier.

    Args:
        addr: Pointer to the shared memory barrier.
        tx_count: Number of expected transactions to track.

    Returns:
        The state.
    """

    comptime assert scope == Scope.BLOCK or scope == Scope.CLUSTER, (
        "mbarrier_arrive_expect_tx_relaxed scope is only supported for"
        " cluster or block/CTA scope."
    )

    comptime assert space == Scope.BLOCK or space == Scope.CLUSTER, (
        "mbarrier_arrive_expect_tx_relaxed space is only supported for"
        " cluster or block/CTA scope."
    )

    comptime if space == Scope.CLUSTER:
        comptime assert scope == Scope.CLUSTER, (
            "mbarrier_arrive_expect_tx_relaxed scope and space must be the"
            " same if space is cluster."
        )

    comptime if is_nvidia_gpu():
        comptime asm = (
            """mbarrier.arrive.expect_tx.relaxed."""
            + scope.mnemonic()
            + """.shared::"""
            + space.mnemonic()
            + """.b64 $0, [$1], $2;"""
        )
        return inlined_assembly[
            asm,
            UInt64,
            constraints="=l,r,r",
            has_side_effect=True,
        ](addr, tx_count)
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name(),
            note="mbarrier_arrive_expect_tx_relaxed() is only supported when targeting NVIDIA GPUs",
        ]()


@always_inline("nodebug")
def mbarrier_try_wait_parity_shared[
    type: AnyType  # The type of the memory barrier
](
    addr: Pointer[mut=True, type, _, address_space=.SHARED],
    phase: Int32,
    ticks: Int32,
):
    """Wait for completion of a barrier phase with timeout.

    Waits for the shared memory barrier to complete the specified phase,
    or until the timeout period expires. Only supported on NVIDIA GPUs.

    Parameters:
        type: The type of the memory barrier.

    Args:
        addr: Pointer to the shared memory barrier.
        phase: Phase number to wait for.
        ticks: Timeout period in nanoseconds.
    """

    comptime if is_nvidia_gpu():
        __mlir_op.`nvvm.mbarrier.try_wait.parity`(
            to_llvm_shared_mem_ptr(addr), to_i32(phase), to_i32(ticks)
        )
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name(),
            note="mbarrier_try_wait_parity_shared() is only supported when targeting NVIDIA GPUs",
        ]()


@always_inline("nodebug")
def umma_arrive_leader_cta[
    type: AnyType
](mbar_ptr: Pointer[mut=True, type, _, address_space=.SHARED]):
    """Signal arrival at the barrier to the leader CTA of the pair.

    This function signals arrival at the barrier to the leader CTA of the pair.
    It is used in the context of pair MMA operations.

    Parameters:
        type: The type of the memory barrier.

    Args:
        mbar_ptr: Pointer to the shared memory barrier.
    """

    inlined_assembly[
        "mbarrier.arrive.shared::cluster.b64 _, [$0];",
        NoneType,
        constraints="r",
        has_side_effect=True,
    ](Int32(Int(mbar_ptr)) & 0xFEFFFFFF)


@always_inline("nodebug")
def umma_arrive_peer_cta[
    type: AnyType
](mbar_ptr: Pointer[mut=True, type, _, address_space=.SHARED]):
    """Signal arrival at the barrier on the peer CTA of the pair.

    This is the mirror of `umma_arrive_leader_cta`. Where the leader variant
    clears bit 24 of the SMEM address to target the leader CTA's barrier,
    this function sets bit 24 to target the peer CTA's barrier within the
    same cluster. Used when the leader CTA needs to signal a CTA-private
    barrier on its partner.

    Parameters:
        type: The type of the memory barrier.

    Args:
        mbar_ptr: Pointer to the shared memory barrier.
    """

    inlined_assembly[
        "mbarrier.arrive.shared::cluster.b64 _, [$0];",
        NoneType,
        constraints="r",
        has_side_effect=True,
    ](Int32(Int(mbar_ptr)) | 0x01000000)


@always_inline
def cp_async_bulk_commit_group():
    """Commits all prior initiated but uncommitted cp.async.bulk instructions into a cp.async.bulk-group.

    This function commits all previously initiated but uncommitted cp.async.bulk instructions into a
    cp.async.bulk-group. The cp.async.bulk instructions are used for asynchronous bulk memory transfers
    on NVIDIA GPUs.

    The function creates a synchronization point for bulk memory transfers, allowing better control over
    memory movement and synchronization between different stages of computation.

    Note:
        This functionality is only available on NVIDIA GPUs. Attempting to use this function on
        non-NVIDIA GPUs will result in a compile time error.
    """

    comptime if is_nvidia_gpu():
        __mlir_op.`nvvm.cp.async.bulk.commit.group`[_type=None]()
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name(),
            note="cp_async_bulk_commit_group() is only supported when targeting NVIDIA GPUs",
        ]()


@always_inline
def cp_async_bulk_wait_group[n: Int32, read: Bool = True]():
    """Waits for completion of asynchronous bulk memory transfer groups.

    This function causes the executing thread to wait until a specified number of the most recent
    bulk async-groups are pending. It provides synchronization control for bulk memory transfers
    on NVIDIA GPUs.

    Parameters:
        n: The number of most recent bulk async-groups allowed to remain pending. When n=0,
           waits for all prior bulk async-groups to complete.
        read: If True, indicates that subsequent reads to the transferred memory are expected,
              enabling optimizations for read access patterns. Defaults to True.

    Note:
        This functionality is only available on NVIDIA GPUs. Attempting to use this function on
        non-NVIDIA GPUs will result in a compile time error.

    Example:
        ```mojo
        from max.gpu.sync.sync import cp_async_bulk_wait_group

        # Wait until at most 2 async groups are pending
        cp_async_bulk_wait_group[2]()

        # Wait for all async groups to complete
        cp_async_bulk_wait_group[0]()
        ```
    """

    @__parameter
    def get_asm() -> String:
        comptime base = "llvm.nvvm.cp.async.bulk.wait.group"
        if read:
            return base + ".read"
        return base

    comptime if is_nvidia_gpu():
        llvm_intrinsic[
            get_asm(),
            NoneType,
        ](n)

    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name(),
            note="cp_async_bulk_wait_group() is only supported when targeting NVIDIA GPUs",
        ]()
