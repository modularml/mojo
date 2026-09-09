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
"""
Tensor Memory Accelerator (TMA) Asynchronous Operations Module.

Provides high-performance abstractions for NVIDIA's Tensor Memory Accelerator (TMA),
enabling efficient asynchronous data movement between global and shared memory in GPU kernels.
It is designed for use with NVIDIA Hopper architecture and newer GPUs that support TMA instructions.

Key Components:
--------------
- `TMATensorTile`: Core struct that encapsulates a TMA descriptor for efficient data transfers
  between global and shared memory with various access patterns and optimizations.

- `SharedMemBarrier`: Synchronization primitive for coordinating asynchronous TMA operations,
  ensuring data transfers complete before dependent operations begin.

- `PipelineState`: Helper struct for managing multi-stage pipeline execution with circular
  buffer semantics, enabling efficient double or triple buffering techniques.

- `create_tma_tile`: Factory functions for creating optimized `TMATensorTile` instances with
  various configurations for different tensor shapes and memory access patterns.
"""

from std.math import ceildiv
from std.math.uutils import udivmod
from std.sys import align_of, llvm_intrinsic, simd_width_of, size_of
from std.sys._assembly import inlined_assembly

from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.host._tensormap import (
    SwizzleMode as _SwizzleMode,
    create_tensormap_im2col as _create_tensormap_im2col,
)
from max.gpu.host.nvidia.tma import (
    TensorMapL2Promotion,
    TensorMapSwizzle,
    TMADescriptor,
    create_tma_descriptor,
    prefetch_tma_descriptor,
)
from max.gpu.intrinsics import Scope
from max.gpu.memory import (
    ReduceOp,
    async_copy,
    cp_async_bulk_tensor_global_shared_cta,
    cp_async_bulk_tensor_global_shared_cta_elect,
    cp_async_bulk_tensor_reduce_global_shared_cta,
    cp_async_bulk_tensor_shared_cluster_global,
    cp_async_bulk_tensor_shared_cluster_global_elect,
    cp_async_bulk_tensor_shared_cluster_global_im2col,
    cp_async_bulk_tensor_shared_cluster_global_im2col_multicast,
    cp_async_bulk_tensor_shared_cluster_global_multicast,
    cp_async_bulk_tensor_2d_gather4,
    CacheEviction,
)
from max.gpu.sync import (
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
    mbarrier_arrive,
    mbarrier_arrive_expect_tx_relaxed,
    mbarrier_arrive_expect_tx_shared,
    mbarrier_init,
)
from layout import (
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
)
from layout.coord import Coord, DynamicCoord
from layout.runtime_tuple import (
    coalesce_nested_tuple,
    flatten,
    to_index_list as runtime_tuple_to_index_list,
)
from layout.tensor_core_async import tile_layout_k_major

from std.utils.index import Index, IndexList
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.utils.static_tuple import StaticTuple
from layout.layout_tensor import LayoutTensorIter


# Swizzle-atom / core-matrix row count. Mirrors `_CM_NUM_ROWS` in
# `layout/tensor_core_async.mojo` (module-private there): the canonical MMA
# core matrix is 8 rows tall and the SWIZZLE_128B 8-row swizzle tile is exactly
# one atom. Used by the rank-5 chunk-inner (row-major-atoms) fold box, which
# splits a page's `box_rows` into `box_rows / _SWIZZLE_ATOM_ROWS` atom-rows.
comptime _SWIZZLE_ATOM_ROWS = 8


def _default_desc_shape[
    rank: Int,
    dtype: DType,
    tile_shape: IndexList[rank],
    swizzle_mode: TensorMapSwizzle,
]() -> IndexList[rank]:
    """Compute the default descriptor shape: tile_shape with last dim = swizzle granularity.
    """
    comptime assert (
        size_of[dtype]() >= 1
    ), "Don't support sub-byte dtype in TMA yet."
    comptime assert (
        rank == 2 or rank == 3 or rank == 4 or rank == 5
    ), "Only support 2D/3D/4D/5D TMA descriptor for now."
    # TMA copies data in swizzle-width chunks along the innermost dimension,
    # so the descriptor's last dim is set to the swizzle granularity (in elements).
    comptime swizzle_bytes = swizzle_mode.bytes() // size_of[dtype]()
    var result = tile_shape
    result[rank - 1] = swizzle_bytes
    return result


@__parameter
def _idx_product[rank: Int, shape: IndexList[rank]]() -> Int:
    """Compute the total number of elements from an IndexList shape."""
    var result = 1
    comptime for i in range(rank):
        result *= shape[i]
    return result


@__parameter
def _idx_str[rank: Int, shape: IndexList[rank]]() -> String:
    """Build a debug string from an IndexList shape."""
    return String(shape)


@__parameter
def _desc_offset[
    rank: Int, dims: IndexList[rank], is_k_major: Bool
](coords: IndexList[rank]) -> Int:
    """Compute linear offset for descriptor layout.

    col_major (is_k_major=True): first dim varies fastest,
        strides (1, d0, d0*d1, ...).
    row_major (is_k_major=False): last dim varies fastest,
        strides (..., d2*d3, d3, 1).
    """
    var offset = 0

    comptime if is_k_major:
        comptime for i in range(rank):
            var stride = 1

            comptime for j in range(i):
                stride *= dims[j]
            offset += coords[i] * stride
    else:
        comptime for i in range(rank):
            var stride = 1

            comptime for j in range(i + 1, rank):
                stride *= dims[j]
            offset += coords[i] * stride
    return offset


def _tma_desc_tile_shape[
    dtype: DType,
    rank: Int,
    tile_shape: IndexList[rank],
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
]() -> IndexList[rank]:
    """Compute the TMA descriptor tile shape.

    Returns an IndexList with the tile shape, where the last dimension is
    replaced by the swizzle granularity (swizzle_mode.bytes() // size_of[dtype]()).
    """
    comptime assert (
        size_of[dtype]() >= 1
    ), "Don't support sub-byte dtype in TMA yet."

    comptime assert (
        rank == 2 or rank == 3 or rank == 4 or rank == 5
    ), "Only support 2D/3D/4D/5D TMA descriptor for now."

    comptime swizzle_bytes = swizzle_mode.bytes() // size_of[dtype]()
    var result = tile_shape
    result[rank - 1] = swizzle_bytes
    return result


struct SharedMemBarrier(TrivialRegisterPassable):
    """A hardware-accelerated synchronization primitive for GPU shared memory operations.

    This struct provides a barrier mechanism optimized for coordinating thread execution
    and memory transfers in GPU kernels, particularly for Tensor Memory Accelerator (TMA)
    operations. It enables efficient synchronization between threads and memory operations
    by leveraging hardware-specific barrier instructions.

    Key features:
    - Thread synchronization across thread blocks
    - Memory transfer completion tracking
    - Hardware-accelerated barrier operations
    - Support for phased synchronization

    This barrier is particularly useful for ensuring that shared memory operations
    complete before dependent computations begin, which is critical for maintaining
    data consistency in high-performance GPU kernels.
    """

    var mbar: Int64
    """Shared memory location used for the barrier state.

    This field stores an 8-byte aligned shared memory location that
    maintains the state of the barrier. The memory must be in shared address
    space to be accessible by all threads in a block.
    """

    @always_inline("nodebug")
    def init[
        o: MutOrigin
    ](ref[o, AddressSpace.SHARED] self, num_threads: Int32 = 1):
        """Initialize the barrier state with the expected number of threads.

        Sets up the barrier to expect arrivals from the specified number of threads
        before it can be satisfied. This is essential for coordinating thread
        synchronization in GPU kernels.

        Args:
            num_threads: Number of threads that must arrive at the barrier
                         before it is satisfied. Defaults to 1.

        Parameters:
            o: Origin of self.
        """
        mbarrier_init(self.unsafe_ptr(), num_threads)

    @always_inline("nodebug")
    def expect_bytes[
        o: MutOrigin
    ](ref[o, AddressSpace.SHARED] self, bytes: Int32):
        """Configure the barrier to expect a specific number of bytes to be transferred.

        Used with TMA operations to indicate the expected size of data transfer.
        The barrier will be satisfied when the specified number of bytes has been
        transferred, enabling efficient coordination of memory operations.

        Args:
            bytes: Number of bytes expected to be transferred.

        Parameters:
            o: Origin of self.
        """
        mbarrier_arrive_expect_tx_shared(self.unsafe_ptr(), bytes)

    @always_inline
    def expect_bytes_relaxed[
        o: MutOrigin
    ](ref[o, AddressSpace.SHARED] self, bytes: Int32) -> UInt64:
        """Configure the barrier to expect a specific number of bytes to be transferred.

        Used with TMA operations to indicate the expected size of data transfer.
        The barrier will be satisfied when the specified number of bytes has been
        transferred, enabling efficient coordination of memory operations.

        Args:
            bytes: Number of bytes expected to be transferred.

        Parameters:
            o: Origin of self.

        Returns:
            The state.
        """
        return mbarrier_arrive_expect_tx_relaxed(self.unsafe_ptr(), bytes)

    @always_inline
    def arrive_and_expect_bytes[
        o: MutOrigin
    ](
        ref[o, AddressSpace.SHARED] self,
        bytes: Int32,
        cta_id: UInt32,
        pred: UInt32,
    ):
        """Configure the barrier to expect a specific number to bytes to be transferred
        at a remote CTA.

         Used with TMA operations to indicate the expected size of data transfer.
         The barrier will be satisfied when the specified number of bytes has been
         transferred at the specified CTA in the cluster.

        Args:
            bytes: Number of bytes expected to be transferred.
            cta_id: The CTA ID in a cluster to configure an arrival.
            pred: Predication on the arrival configuration instruction. Use UInt32 to match `selp.u32` in ptx.

        Parameters:
            o: Origin of self.
        """

        comptime asm = """
        .reg .pred p;
        .reg .b32 remAddr32;
        setp.eq.u32 p, $2, 1;
        @p mapa.shared::cluster.u32  remAddr32, $0, $1;
        @p mbarrier.arrive.expect_tx.shared::cluster.b64  _, [remAddr32], $3;
        """

        inlined_assembly[asm, NoneType, constraints="r,r,r,r"](
            Int32(Int(self.unsafe_ptr())), cta_id, pred, bytes
        )

    @always_inline("nodebug")
    def wait[
        ticks: Optional[UInt32] = None
    ](ref[AddressSpace.SHARED] self, phase: UInt32 = 0):
        """Wait until the barrier is satisfied.

        Blocks the calling thread until the barrier is satisfied, either by
        the expected number of threads arriving or the expected data transfer
        completing. This method implements an efficient spin-wait mechanism
        optimized for GPU execution.

        Parameters:
            ticks: The number of ticks to wait before timing out in nanoseconds.
                   Defaults to None.

        Args:
            phase: The phase value to check against. Defaults to 0.

        Note:
            Minimizes thread divergence during synchronization by using
            hardware-accelerated barrier instructions.
        """
        # Based on cutlass
        # https://github.com/NVIDIA/cutlass/blob/d1ef0e87f2f3d68cf5ad7472cadc1152a8d3857c/include/cutlass/arch/barrier.h#L408

        comptime wait_asm = (
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [$0], $1"
            + (" , $2" if ticks else "")
            + ";"
        )
        comptime asm = """{
            .reg .pred P1;
            LAB_WAIT:
            """ + wait_asm + """
            @P1 bra DONE;
            bra LAB_WAIT;
            DONE:
        }"""

        comptime constraints = "r,r" + (",r" if ticks else "")

        comptime if ticks:
            inlined_assembly[asm, NoneType, constraints=constraints](
                Int32(Int(self.unsafe_ptr())), phase, ticks.value()
            )
        else:
            inlined_assembly[asm, NoneType, constraints=constraints](
                Int32(Int(self.unsafe_ptr())), phase
            )

    @always_inline("nodebug")
    def wait_acquire[
        scope: Scope
    ](ref[AddressSpace.SHARED] self, phase: UInt32 = 0):
        """Acquire and wait until the barrier is satisfied.

        Blocks the calling thread until the barrier is satisfied, either by
        the expected number of threads arriving or the expected data transfer
        completing. This method implements an efficient spin-wait mechanism
        optimized for GPU execution.

        Parameters:
            scope: The scope of the barrier.

        Args:
            phase: The phase value to check against. Defaults to 0.

        Note:
            Minimizes thread divergence during synchronization by using
            hardware-accelerated barrier instructions.
        """
        # Based on cccl
        # https://github.com/NVIDIA/cccl/blob/ba510b38e01dac5ab9b5faad9b9b1701d60d9980/libcudacxx/include/cuda/__ptx/instructions/generated/mbarrier_try_wait_parity.h#L94

        comptime assert (
            scope == Scope.CLUSTER or scope == Scope.BLOCK
        ), "wait_acquire is only supported for cluster or block/CTA scope."

        comptime asm = (
            """{
            .reg .pred P1;
            LAB_WAIT:
            mbarrier.try_wait.parity.acquire."""
            + scope.mnemonic()
            + """.shared::cta.b64 P1, [$0], $1;
            @P1 bra DONE;
            bra LAB_WAIT;
            DONE:
            }"""
        )
        inlined_assembly[asm, NoneType, constraints="r,r"](
            Int32(Int(self.unsafe_ptr())), phase
        )

    @always_inline("nodebug")
    def wait_relaxed[
        scope: Scope
    ](ref[AddressSpace.SHARED] self, phase: UInt32 = 0):
        """Wait until the barrier is satisfied with relaxed ordering.

        Blocks the calling thread until the barrier is satisfied, either by
        the expected number of threads arriving or the expected data transfer
        completing. This method implements an efficient spin-wait mechanism
        optimized for GPU execution.

        Parameters:
            scope: The scope of the barrier.

        Args:
            phase: The phase value to check against. Defaults to 0.

        Note:
            Minimizes thread divergence during synchronization by using
            hardware-accelerated barrier instructions.
        """
        # Based on cccl
        # https://github.com/NVIDIA/cccl/blob/ba510b38e01dac5ab9b5faad9b9b1701d60d9980/libcudacxx/include/cuda/__ptx/instructions/generated/mbarrier_try_wait_parity.h#L104

        comptime assert (
            scope == Scope.CLUSTER or scope == Scope.BLOCK
        ), "wait_relaxed is only supported for cluster or block/CTA scope."

        comptime asm = (
            """{
            .reg .pred P1;
            LAB_WAIT:
            mbarrier.try_wait.parity.relaxed."""
            + scope.mnemonic()
            + """.shared::cta.b64 P1, [$0], $1;
            @P1 bra DONE;
            bra LAB_WAIT;
            DONE:
            }"""
        )
        inlined_assembly[asm, NoneType, constraints="r,r"](
            Int32(Int(self.unsafe_ptr())), phase
        )

    @always_inline("nodebug")
    def try_wait(ref[AddressSpace.SHARED] self, phase: UInt32 = 0) -> Bool:
        """Non-blocking check if barrier phase is complete.

        Performs a single non-blocking check to see if the barrier has completed
        the specified phase. Returns immediately with the result without spinning.

        This is useful for implementing the try-acquire pattern where you want to
        overlap barrier checking with other useful work.

        Args:
            phase: The phase parity (0 or 1) to check for. Defaults to 0.

        Returns:
            True if the barrier phase is complete, False otherwise.

        Example:
            ```mojo
            # Try-acquire pattern for pipelined execution
            var ready = barrier.try_wait(phase)
            # Do other work while potentially waiting
            do_useful_work()
            # Now wait conditionally
            if not ready:
                barrier.wait(phase)
            ```
        """
        # PTX: mbarrier.try_wait.parity.shared::cta.b64 waitComplete, [addr], phaseParity;
        return inlined_assembly[
            "mbarrier.try_wait.parity.shared::cta.b64 $0, [$1], $2;",
            Bool,
            constraints="=b,r,r",
        ](Int32(Int(self.unsafe_ptr())), phase)

    @always_inline
    def unsafe_ptr[
        origin: Origin
    ](
        ref[origin, AddressSpace.SHARED] self,
    ) -> Pointer[
        Int64, origin=origin, address_space=.SHARED
    ]:
        """Get an unsafe pointer to the barrier's memory location.

        Provides low-level access to the shared memory location storing the barrier state.
        This method is primarily used internally by other barrier operations that need
        direct access to the underlying memory.

        Parameters:
            origin: Origin of self.

        Returns:
            An unsafe pointer to the barrier's memory location in shared memory,
            properly typed and aligned for barrier operations.
        """
        return Pointer(to=self.mbar).unsafe_origin_cast[origin]()

    @always_inline
    def arrive_cluster(
        ref[AddressSpace.SHARED] self, cta_id: UInt32, count: UInt32 = 1
    ):
        """Signal arrival at the barrier from a specific CTA (Cooperative Thread Array) in a cluster.

        This method is used in multi-CTA scenarios to coordinate barrier arrivals
        across different CTAs within a cluster. It enables efficient synchronization
        across thread blocks in clustered execution models.

        Args:
            cta_id: The ID of the CTA (Cooperative Thread Array) that is arriving.
            count: The number of arrivals to signal. Defaults to 1.
        """
        comptime asm = """{
            .reg .b32 remAddr32;
            mapa.shared::cluster.u32  remAddr32, $0, $1;
            mbarrier.arrive.shared::cluster.b64  _, [remAddr32], $2;
        }"""
        inlined_assembly[asm, NoneType, constraints="r,r,r"](
            Int32(Int(self.unsafe_ptr())), cta_id, count
        )

    @always_inline("nodebug")
    def arrive[o: MutOrigin](ref[o, AddressSpace.SHARED] self) -> Int:
        """Signal arrival at the barrier and return the arrival count.

        This method increments the arrival count at the barrier and returns
        the updated count. It's used to track how many threads have reached
        the synchronization point.

        Returns:
            The updated arrival count after this thread's arrival.

        Parameters:
            o: Origin of self.
        """
        return mbarrier_arrive(self.unsafe_ptr())

    @always_inline
    def complete_transaction(
        ref[AddressSpace.SHARED] self,
        dst_cta_id: UInt32,
        bytes: Int32,
        pred: UInt32,
    ):
        """Manually advances the barrier's expected-bytes count without performing a real transfer.

        Used to honor an outstanding `expect_bytes` on a barrier when the
        corresponding TMA load has been elided (for example, when all gathered
        indices in a sparse-gather are invalid and the load was skipped). The
        consumer still waits on the barrier; this satisfies that wait so the
        pipeline doesn't deadlock.

        Args:
            dst_cta_id: ID of the CTA whose barrier should be advanced.
            bytes: Number of bytes to credit toward the barrier's expected count.
            pred: Predicate (1 to apply the credit, 0 to skip). Used so only one
                lane in a warp issues the credit.
        """
        comptime asm = """{
            .reg .pred p;
            .reg .s32 remAddr32;
            setp.eq.u32 p, $3, 1;
            mapa.shared::cluster.u32  remAddr32, $0, $1;
            @p mbarrier.complete_tx.relaxed.cluster.shared::cluster.b64 [remAddr32], $2;
        }"""

        inlined_assembly[asm, NoneType, constraints="r,r,r,r"](
            Int32(Int(self.unsafe_ptr())), dst_cta_id, bytes, pred
        )


struct PipelineState[num_stages: Int](Defaultable, TrivialRegisterPassable):
    """Manages state for a multi-stage pipeline with circular buffer semantics.

    PipelineState provides a mechanism for tracking the current stage in a
    multi-stage pipeline, particularly useful for double or triple buffering
    in GPU tensor operations. It maintains an index that cycles through the
    available stages, a phase bit that toggles when the index wraps around,
    and a monotonically increasing count.

    This struct is commonly used with TMA operations to coordinate the use of
    multiple buffers in a pipeline fashion, allowing for overlapping computation
    and data transfer.

    Parameters:
        num_stages: The number of stages in the pipeline (e.g., 2 for double buffering,
                   3 for triple buffering).
    """

    var _index: UInt32
    """The current stage index in the pipeline.

    This field tracks which buffer in the circular pipeline is currently active.
    Values range from 0 to num_stages-1 and wrap around when incremented past
    the last stage.
    """

    var _phase: UInt32
    """The current phase bit of the pipeline.

    This field alternates between 0 and 1 each time the index completes a full cycle.
    It's used to detect when a full pipeline cycle has completed, particularly
    useful for synchronization in producer-consumer scenarios.
    """

    var _count: UInt32
    """A monotonically increasing counter tracking pipeline iterations.

    This counter increments with each pipeline advancement, providing a
    total count of how many times the pipeline has been advanced since
    initialization. Useful for tracking progress and debugging.
    """

    @always_inline
    def __init__(out self):
        """Initialize a PipelineState with default values.

        Creates a new PipelineState with index 0, phase 0, and count 0.
        """
        self._index = 0
        self._phase = 0
        self._count = 0

    @always_inline
    def __init__(out self, index: Int, phase: Int, count: Int):
        """Initialize a PipelineState with specific values.

        Creates a new PipelineState with the specified index, phase, and count.

        Args:
            index: The initial stage index.
            phase: The initial phase value (0 or 1).
            count: The initial count value.
        """
        self._index = UInt32(index)
        self._phase = UInt32(phase)
        self._count = UInt32(count)

    @always_inline
    def index(self) -> UInt32:
        """Get the current stage index.

        Returns:
            The current index value, which ranges from 0 to num_stages-1.
        """
        return self._index

    @always_inline
    def phase(self) -> UInt32:
        """Get the current phase bit.

        Returns:
            The current phase value (0 or 1), which toggles when the index wraps around.
        """
        return self._phase

    @always_inline
    def step(mut self):
        """Advance the pipeline state to the next stage.

        Increments the index and count. When the index reaches num_stages,
        it wraps around to 0 and toggles the phase bit.

        This function is used to move to the next buffer in a multi-buffer
        pipeline, implementing circular buffer semantics.
        """

        comptime if Self.num_stages > 1:
            self._index += 1
            self._count += 1
            if self._index == UInt32(Self.num_stages):
                self._index = 0
                self._phase ^= 1

        comptime if Self.num_stages == 1:
            self._count += 1
            self._phase ^= 1

    @always_inline
    def next(mut self) -> Self:
        """Advance the pipeline state to the next stage and return the new state.

        This function is used to move to the next buffer in a multi-buffer
        pipeline, implementing circular buffer semantics.

        Returns:
            The new pipeline state after advancing to the next stage.
        """
        self.step()
        return self

    @always_inline
    def __enter__(var self) -> Self:
        """Enter the context manager.

        Returns:
            The pipeline state instance for use in a `with` statement.
        """
        return self


# TMATensorTile is created on the host with specific memory and tile sizes.
# Each TMATensorTile provides an asynchronous load of a specific tile at specified tile coordinates.
#
struct TMATensorTile[
    dtype: DType,
    rank: Int,
    tile_shape: IndexList[rank],
    desc_shape: IndexList[rank] = tile_shape,
    is_k_major: Bool = True,
](DevicePassable, ImplicitlyCopyable):
    """
    A hardware-accelerated tensor memory access (TMA) tile for efficient asynchronous data movement.

    The TMATensorTile struct provides a high-performance interface for asynchronous data transfers
    between global memory and shared memory in GPU tensor operations. It encapsulates a TMA descriptor
    that defines the memory access pattern and provides methods for various asynchronous operations.

    Parameters:
        dtype: DType
            The data type of the tensor elements.
        rank: Int
            The dimensionality of the tile (2, 3, 4, or 5).
        tile_shape: IndexList[rank]
            The shape of the tile in shared memory.
        desc_shape: IndexList[rank] = tile_shape
            The shape of the descriptor, which can be different from the tile shape
            to accommodate hardware requirements like WGMMA.
        is_k_major: Bool = True
            Whether the shared memory is k-major.

    Performance:

        - Hardware-accelerated memory transfers using TMA instructions
        - Supports prefetching of descriptors for latency hiding
        - Enforces 128-byte alignment requirements for optimal memory access
    """

    var descriptor: TMADescriptor
    """The TMA descriptor that defines the memory access pattern.

    This field stores the hardware descriptor that encodes information about:
    - The source tensor's memory layout and dimensions
    - The tile shape and access pattern
    - Swizzling configuration for optimal memory access

    The descriptor is used by the GPU's Tensor Memory Accelerator hardware to
    efficiently transfer data between global and shared memory.
    """

    comptime device_type: AnyType = Self
    """The device-side type representation."""

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        """Device type mapping is the identity function."""
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        """
        Gets this type's name, for use in error messages when handing arguments
        to kernels.

        Returns:
            This type's name.
        """
        return String(
            "TMATensorTile[dtype = ",
            Self.dtype,
            ", rank = ",
            Self.rank,
            ", tile_shape = ",
            _idx_str[Self.rank, Self.tile_shape](),
            ", desc_shape = ",
            _idx_str[Self.rank, Self.desc_shape](),
            ", is_k_major = ",
            Self.is_k_major,
            "]",
        )

    @always_inline
    @implicit
    def __init__(out self, descriptor: TMADescriptor):
        """
        Initializes a new TMATensorTile with the provided TMA descriptor.

        Args:
            descriptor: The TMA descriptor that defines the memory access pattern.
        """
        self.descriptor = descriptor

    @always_inline
    def prefetch_descriptor(self):
        """
        Prefetches the TMA descriptor into cache to reduce latency.

        This method helps hide memory access latency by prefetching the descriptor
        before it's needed for actual data transfers.
        """
        var desc_ptr = Pointer(to=self.descriptor).bitcast[NoneType]()
        prefetch_tma_descriptor(desc_ptr)

    @always_inline
    def async_copy[
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: LayoutTensor[mut=True, _, _, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int],
    ):
        """
        Schedules an asynchronous copy from global memory to shared memory at specified coordinates.

        This method initiates a hardware-accelerated asynchronous transfer of data from global memory
        to the specified destination in shared memory. The transfer is tracked by the provided memory
        barrier.

        Parameters:
            cta_group: Int
                If the TMA is issued with cta_group == 2, only the leader CTA needs
                to be notified upon completion.
            eviction_policy: Optional cache eviction policy that controls how the data is handled
                in the cache hierarchy. Defaults to EVICT_NORMAL.

        Args:
            dst: The destination tensor in shared memory where data will be copied.
                 Must be 128-byte aligned.
            mem_barrier: The memory barrier used to track and synchronize the asynchronous transfer.
            coords: The 2D coordinates in the source tensor from which to copy data.

        Constraints:

            - The destination tensor must be 128-byte aligned in shared memory.
            - The descriptor layout may be smaller than the shared memory tile shape
              to accommodate hardware requirements.
        """
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#table-alignment-multi-dim-tma
        comptime assert (
            type_of(dst).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        comptime assert (
            type_of(dst).dtype == Self.dtype
        ), "Input tensor has a different type than the TMA op"

        # The descriptor layout i.e. data per copy can be smaller than the shared memory
        # tile shape due to WGMMA requirement. E.g. k-major no swizzle WGMMA BM x 16B to be
        # one continuous chunk in shared memory. We need to break down tile shape in K by 16B.
        #
        # dim0, dim1 are MN, K for K-major and K, MN for MN-major because our inputs are
        # row_major(K, MN) for the latter.
        #
        # TODO: use layout algebra here
        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = Self.tile_shape[
            Int(not Self.is_k_major)
        ] // copy_dim0
        comptime num_copies_dim1 = Self.tile_shape[
            Int(Self.is_k_major)
        ] // copy_dim1

        comptime for i in range(num_copies_dim0):
            comptime for j in range(num_copies_dim1):
                comptime copy_offset: UInt32 = UInt32(
                    (i * num_copies_dim1 + j) * copy_size
                )

                comptime assert (
                    copy_offset * UInt32(size_of[Self.dtype]())
                ) % 128 == 0, (
                    "TMA async_copy requires 128B-aligned copy offset (offset="
                    + String(copy_offset)
                    + ")"
                )
                cp_async_bulk_tensor_shared_cluster_global[
                    cta_group=cta_group,
                    eviction_policy=eviction_policy,
                ](
                    dst.ptr + copy_offset,
                    Pointer(to=self.descriptor).bitcast[NoneType](),
                    mem_barrier.unsafe_ptr(),
                    Index(
                        coords[0] + (j * copy_dim1),
                        coords[1] + (i * copy_dim0),
                    ),
                )

    @always_inline
    def async_copy_elect[
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: LayoutTensor[_, _, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int],
        elect: Int32,
    ):
        """Elect-predicated overload of `async_copy` (2D).

        Each unrolled `cp_async_bulk_tensor_shared_cluster_global` issue is
        predicated in-PTX on `elect`: the TMA fires only on the elected lane.
        All lanes follow the same PTX control flow; no warp-divergent
        `if elect != 0:` is needed at the call site.

        Parameters:
            cta_group: If the TMA is issued with `cta_group == 2`, only the
                leader CTA is notified on completion. Defaults to 1.
            eviction_policy: Cache eviction policy. Defaults to EVICT_NORMAL.

        Args:
            dst: The destination tensor in shared memory. Must be 128-byte
                aligned.
            mem_barrier: The memory barrier for synchronization.
            coords: The 2D coordinates in the source tensor.
            elect: `0` on non-elected lanes (skip the TMA), non-zero on the
                single elected lane (issue the TMA).
        """
        comptime assert (
            type_of(dst).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        comptime assert (
            type_of(dst).dtype == Self.dtype
        ), "Input tensor has a different type than the TMA op"

        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = Self.tile_shape[
            Int(not Self.is_k_major)
        ] // copy_dim0
        comptime num_copies_dim1 = Self.tile_shape[
            Int(Self.is_k_major)
        ] // copy_dim1

        comptime for i in range(num_copies_dim0):
            comptime for j in range(num_copies_dim1):
                comptime copy_offset: UInt32 = UInt32(
                    (i * num_copies_dim1 + j) * copy_size
                )

                comptime assert (
                    copy_offset * UInt32(size_of[Self.dtype]())
                ) % 128 == 0, (
                    "TMA async_copy requires 128B-aligned copy offset (offset="
                    + String(copy_offset)
                    + ")"
                )
                cp_async_bulk_tensor_shared_cluster_global_elect[
                    cta_group=cta_group,
                    eviction_policy=eviction_policy,
                ](
                    dst.ptr.unsafe_mut_cast[True]() + copy_offset,
                    Pointer(to=self.descriptor).bitcast[NoneType](),
                    mem_barrier.unsafe_ptr(),
                    Index(
                        coords[0] + (j * copy_dim1),
                        coords[1] + (i * copy_dim0),
                    ),
                    elect,
                )

    @always_inline
    def async_copy[
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: TileTensor[mut=True, Self.dtype, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int],
    ):
        """TileTensor overload for 2D async copy from global to shared memory.

        Parameters:
            cta_group: If the TMA is issued with cta_group == 2, only the
                leader CTA needs to be notified upon completion.
            eviction_policy: Cache eviction policy. Defaults to EVICT_NORMAL.

        Args:
            dst: TileTensor in shared memory where data will be copied.
            mem_barrier: The memory barrier for synchronization.
            coords: The 2D coordinates in the source tensor.
        """
        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = Self.tile_shape[
            Int(not Self.is_k_major)
        ] // copy_dim0
        comptime num_copies_dim1 = Self.tile_shape[
            Int(Self.is_k_major)
        ] // copy_dim1

        comptime for i in range(num_copies_dim0):
            comptime for j in range(num_copies_dim1):
                comptime copy_offset: UInt32 = UInt32(
                    (i * num_copies_dim1 + j) * copy_size
                )
                cp_async_bulk_tensor_shared_cluster_global[
                    cta_group=cta_group,
                    eviction_policy=eviction_policy,
                ](
                    dst.ptr + copy_offset,
                    Pointer(to=self.descriptor).bitcast[NoneType](),
                    mem_barrier.unsafe_ptr(),
                    Index(
                        coords[0] + (j * copy_dim1),
                        coords[1] + (i * copy_dim0),
                    ),
                )

    @always_inline
    def async_copy_elect[
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: TileTensor[mut=True, Self.dtype, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int],
        elect: Int32,
    ):
        """Elect-predicated TileTensor overload of `async_copy` (2D).

        See the `LayoutTensor` overload of `async_copy_elect` for semantics.

        Parameters:
            cta_group: If the TMA is issued with `cta_group == 2`, only the
                leader CTA is notified on completion. Defaults to 1.
            eviction_policy: Cache eviction policy. Defaults to EVICT_NORMAL.

        Args:
            dst: TileTensor in shared memory where data will be copied.
            mem_barrier: The memory barrier for synchronization.
            coords: The 2D coordinates in the source tensor.
            elect: `0` on non-elected lanes (skip the TMA), non-zero on the
                single elected lane (issue the TMA).
        """
        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = Self.tile_shape[
            Int(not Self.is_k_major)
        ] // copy_dim0
        comptime num_copies_dim1 = Self.tile_shape[
            Int(Self.is_k_major)
        ] // copy_dim1

        comptime for i in range(num_copies_dim0):
            comptime for j in range(num_copies_dim1):
                comptime copy_offset: UInt32 = UInt32(
                    (i * num_copies_dim1 + j) * copy_size
                )
                cp_async_bulk_tensor_shared_cluster_global_elect[
                    cta_group=cta_group,
                    eviction_policy=eviction_policy,
                ](
                    dst.ptr + copy_offset,
                    Pointer(to=self.descriptor).bitcast[NoneType](),
                    mem_barrier.unsafe_ptr(),
                    Index(
                        coords[0] + (j * copy_dim1),
                        coords[1] + (i * copy_dim0),
                    ),
                    elect,
                )

    @always_inline("nodebug")
    def async_copy_3d[
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: LayoutTensor[mut=True, Self.dtype, _, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int, Int],
    ):
        """
        Schedules an asynchronous copy from global memory to shared memory at specified 3D coordinates.

        This method initiates a hardware-accelerated asynchronous transfer of data from global memory
        to the specified destination in shared memory for 3D tensors. The transfer is tracked by the
        provided memory barrier.

        Args:
            dst: The destination tensor in shared memory where data will be copied.
                 Must be 128-byte aligned.
            mem_barrier: The memory barrier used to track and synchronize the asynchronous transfer.
            coords: The 3D coordinates in the source tensor from which to copy data.

        Parameters:
            cta_group: If set to 2, the TMA emits `cta_group::2` PTX so the
                mbarrier arrival routes to the leader CTA's barrier, required
                for pair-CTA kernels that share one barrier across the pair.
                Defaults to 1.
            eviction_policy: Optional cache eviction policy that controls how the data is handled
                in the cache hierarchy. Defaults to EVICT_NORMAL.

        Constraints:

            - The destination tensor must be 128-byte aligned in shared memory.
            - The descriptor layout may be smaller than the shared memory tile shape
              to accommodate hardware requirements.
        """
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#table-alignment-multi-dim-tma
        comptime assert (
            type_of(dst).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        # The descriptor layout i.e. data per copy can be smaller than the shared memory
        # tile shape due to WGMMA requirement. E.g. k-major no swizzle WGMMA BM x 16B to be
        # one continuous chunk in shared memory. We need to break down tile shape in K by 16B.
        #
        # dim0, dim1 are MN, K for K-major and K, MN for MN-major because our inputs are
        # row_major(K, MN) for the latter.
        #
        # TODO: use layout algebra here
        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_dim2 = Self.desc_shape[2]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = ceildiv(Self.tile_shape[0], copy_dim0)
        comptime num_copies_dim1 = ceildiv(Self.tile_shape[1], copy_dim1)
        comptime num_copies_dim2 = ceildiv(Self.tile_shape[2], copy_dim2)

        comptime for m in range(num_copies_dim0):
            comptime for i in range(num_copies_dim1):
                comptime for j in range(num_copies_dim2):
                    comptime copy_offset: UInt32 = UInt32(
                        _desc_offset[
                            3,
                            Index(
                                num_copies_dim0,
                                num_copies_dim1,
                                num_copies_dim2,
                            ),
                            Self.is_k_major,
                        ](Index(m, i, j))
                        * copy_size
                    )

                    cp_async_bulk_tensor_shared_cluster_global[
                        cta_group=cta_group,
                        eviction_policy=eviction_policy,
                    ](
                        dst.ptr + copy_offset,
                        Pointer(to=self.descriptor).bitcast[NoneType](),
                        mem_barrier.unsafe_ptr(),
                        Index(
                            coords[0] + (j * copy_dim2),
                            coords[1] + (i * copy_dim1),
                            coords[2] + (m * copy_dim0),
                        ),
                    )

    @always_inline("nodebug")
    def async_copy_3d_elect[
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: LayoutTensor[mut=True, Self.dtype, _, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int, Int],
        elect: Int32,
    ):
        """Elect-predicated overload of `async_copy_3d`.

        See `async_copy_elect` for semantics: each unrolled TMA issue is
        predicated in-PTX on `elect`.

        Parameters:
            cta_group: If set to 2, the TMA emits `cta_group::2` PTX so the
                mbarrier arrival routes to the leader CTA's barrier.
                Defaults to 1.
            eviction_policy: Cache eviction policy. Defaults to EVICT_NORMAL.

        Args:
            dst: The destination tensor in shared memory. Must be 128-byte
                aligned.
            mem_barrier: The memory barrier used to synchronize the
                transfer.
            coords: The 3D coordinates in the source tensor.
            elect: `0` on non-elected lanes, non-zero on the elected lane.
        """
        comptime assert (
            type_of(dst).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_dim2 = Self.desc_shape[2]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = ceildiv(Self.tile_shape[0], copy_dim0)
        comptime num_copies_dim1 = ceildiv(Self.tile_shape[1], copy_dim1)
        comptime num_copies_dim2 = ceildiv(Self.tile_shape[2], copy_dim2)

        comptime for m in range(num_copies_dim0):
            comptime for i in range(num_copies_dim1):
                comptime for j in range(num_copies_dim2):
                    comptime copy_offset: UInt32 = UInt32(
                        _desc_offset[
                            3,
                            Index(
                                num_copies_dim0,
                                num_copies_dim1,
                                num_copies_dim2,
                            ),
                            Self.is_k_major,
                        ](Index(m, i, j))
                        * copy_size
                    )

                    cp_async_bulk_tensor_shared_cluster_global_elect[
                        cta_group=cta_group,
                        eviction_policy=eviction_policy,
                    ](
                        dst.ptr + copy_offset,
                        Pointer(to=self.descriptor).bitcast[NoneType](),
                        mem_barrier.unsafe_ptr(),
                        Index(
                            coords[0] + (j * copy_dim2),
                            coords[1] + (i * copy_dim1),
                            coords[2] + (m * copy_dim0),
                        ),
                        elect,
                    )

    @always_inline
    def async_copy_3d[
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: TileTensor[mut=True, Self.dtype, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int, Int],
    ):
        """TileTensor overload for 3D async copy from global to shared memory.

        Assumes 128B alignment (TileTensor tiles are allocated with proper
        alignment by the caller's SMEM layout).

        Parameters:
            cta_group: If set to 2, the TMA emits `cta_group::2` PTX so the
                mbarrier arrival routes to the leader CTA's barrier, required
                for pair-CTA kernels that share one barrier across the pair.
                Defaults to 1.
            eviction_policy: Cache eviction policy. Defaults to EVICT_NORMAL.

        Args:
            dst: TileTensor in shared memory where data will be copied.
                 Must be 128-byte aligned.
            mem_barrier: The memory barrier for synchronization.
            coords: The 3D coordinates in the source tensor.
        """
        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_dim2 = Self.desc_shape[2]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = ceildiv(Self.tile_shape[0], copy_dim0)
        comptime num_copies_dim1 = ceildiv(Self.tile_shape[1], copy_dim1)
        comptime num_copies_dim2 = ceildiv(Self.tile_shape[2], copy_dim2)

        comptime for m in range(num_copies_dim0):
            comptime for i in range(num_copies_dim1):
                comptime for j in range(num_copies_dim2):
                    comptime copy_offset: UInt32 = UInt32(
                        _desc_offset[
                            3,
                            Index(
                                num_copies_dim0,
                                num_copies_dim1,
                                num_copies_dim2,
                            ),
                            Self.is_k_major,
                        ](Index(m, i, j))
                        * copy_size
                    )

                    cp_async_bulk_tensor_shared_cluster_global[
                        cta_group=cta_group,
                        eviction_policy=eviction_policy,
                    ](
                        dst.ptr + copy_offset,
                        Pointer(to=self.descriptor).bitcast[NoneType](),
                        mem_barrier.unsafe_ptr(),
                        Index(
                            coords[0] + (j * copy_dim2),
                            coords[1] + (i * copy_dim1),
                            coords[2] + (m * copy_dim0),
                        ),
                    )

    @always_inline
    def async_copy_3d_elect[
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: TileTensor[mut=True, Self.dtype, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int, Int],
        elect: Int32,
    ):
        """Elect-predicated TileTensor overload of `async_copy_3d`.

        See the `LayoutTensor` overload of `async_copy_3d_elect` for
        semantics.

        Parameters:
            cta_group: If set to 2, the TMA emits `cta_group::2` PTX.
                Defaults to 1.
            eviction_policy: Cache eviction policy. Defaults to EVICT_NORMAL.

        Args:
            dst: TileTensor in shared memory where data will be copied.
            mem_barrier: The memory barrier for synchronization.
            coords: The 3D coordinates in the source tensor.
            elect: `0` on non-elected lanes, non-zero on the elected lane.
        """
        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_dim2 = Self.desc_shape[2]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = ceildiv(Self.tile_shape[0], copy_dim0)
        comptime num_copies_dim1 = ceildiv(Self.tile_shape[1], copy_dim1)
        comptime num_copies_dim2 = ceildiv(Self.tile_shape[2], copy_dim2)

        comptime for m in range(num_copies_dim0):
            comptime for i in range(num_copies_dim1):
                comptime for j in range(num_copies_dim2):
                    comptime copy_offset: UInt32 = UInt32(
                        _desc_offset[
                            3,
                            Index(
                                num_copies_dim0,
                                num_copies_dim1,
                                num_copies_dim2,
                            ),
                            Self.is_k_major,
                        ](Index(m, i, j))
                        * copy_size
                    )

                    cp_async_bulk_tensor_shared_cluster_global_elect[
                        cta_group=cta_group,
                        eviction_policy=eviction_policy,
                    ](
                        dst.ptr + copy_offset,
                        Pointer(to=self.descriptor).bitcast[NoneType](),
                        mem_barrier.unsafe_ptr(),
                        Index(
                            coords[0] + (j * copy_dim2),
                            coords[1] + (i * copy_dim1),
                            coords[2] + (m * copy_dim0),
                        ),
                        elect,
                    )

    @always_inline
    def async_copy_4d[
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: LayoutTensor[mut=True, Self.dtype, _, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int, Int, Int],
    ):
        """
        Schedules an asynchronous copy from global memory to shared memory at specified 4D coordinates.

        This method initiates a hardware-accelerated asynchronous transfer of data from global memory
        to the specified destination in shared memory for 4D tensors. The transfer is tracked by the
        provided memory barrier.

        Parameters:
            cta_group: Int
                If the TMA is issued with cta_group == 2, only the leader CTA needs
                to be notified upon completion.
            eviction_policy: Optional cache eviction policy that controls how the data is handled
                in the cache hierarchy. Defaults to EVICT_NORMAL.

        Args:
            dst: The destination tensor in shared memory where data will be copied.
                 Must be 128-byte aligned.
            mem_barrier: The memory barrier used to track and synchronize the asynchronous transfer.
            coords: The 4D coordinates in the source tensor from which to copy data.

        Constraints:

            - The destination tensor must be 128-byte aligned in shared memory.
            - The descriptor layout may be smaller than the shared memory tile shape
              to accommodate hardware requirements.
        """
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#table-alignment-multi-dim-tma
        comptime assert (
            type_of(dst).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_dim2 = Self.desc_shape[2]
        comptime copy_dim3 = Self.desc_shape[3]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = ceildiv(Self.tile_shape[0], copy_dim0)
        comptime num_copies_dim1 = ceildiv(Self.tile_shape[1], copy_dim1)
        comptime num_copies_dim2 = ceildiv(Self.tile_shape[2], copy_dim2)
        comptime num_copies_dim3 = ceildiv(Self.tile_shape[3], copy_dim3)
        comptime for n in range(num_copies_dim0):
            comptime for m in range(num_copies_dim1):
                comptime for i in range(num_copies_dim2):
                    comptime for j in range(num_copies_dim3):
                        comptime copy_offset: UInt32 = UInt32(
                            _desc_offset[
                                4,
                                Index(
                                    num_copies_dim0,
                                    num_copies_dim1,
                                    num_copies_dim2,
                                    num_copies_dim3,
                                ),
                                Self.is_k_major,
                            ](Index(n, m, i, j))
                            * copy_size
                        )

                        cp_async_bulk_tensor_shared_cluster_global[
                            cta_group=cta_group,
                            eviction_policy=eviction_policy,
                        ](
                            dst.ptr + copy_offset,
                            Pointer(to=self.descriptor).bitcast[NoneType](),
                            mem_barrier.unsafe_ptr(),
                            Index(
                                coords[0] + (j * copy_dim3),
                                coords[1] + (i * copy_dim2),
                                coords[2] + (m * copy_dim1),
                                coords[3] + (n * copy_dim0),
                            ),
                        )

    @always_inline
    def async_copy_4d_elect[
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: LayoutTensor[mut=True, Self.dtype, _, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int, Int, Int],
        elect: Int32,
    ):
        """Elect-predicated overload of `async_copy_4d`.

        See `async_copy_elect` for semantics: each unrolled TMA issue is
        predicated in-PTX on `elect`.

        Parameters:
            cta_group: If set to 2, the TMA emits `cta_group::2` PTX.
                Defaults to 1.
            eviction_policy: Cache eviction policy. Defaults to EVICT_NORMAL.

        Args:
            dst: The destination tensor in shared memory. Must be 128-byte
                aligned.
            mem_barrier: The memory barrier for synchronization.
            coords: The 4D coordinates in the source tensor.
            elect: `0` on non-elected lanes, non-zero on the elected lane.
        """
        comptime assert (
            type_of(dst).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_dim2 = Self.desc_shape[2]
        comptime copy_dim3 = Self.desc_shape[3]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = ceildiv(Self.tile_shape[0], copy_dim0)
        comptime num_copies_dim1 = ceildiv(Self.tile_shape[1], copy_dim1)
        comptime num_copies_dim2 = ceildiv(Self.tile_shape[2], copy_dim2)
        comptime num_copies_dim3 = ceildiv(Self.tile_shape[3], copy_dim3)
        comptime for n in range(num_copies_dim0):
            comptime for m in range(num_copies_dim1):
                comptime for i in range(num_copies_dim2):
                    comptime for j in range(num_copies_dim3):
                        comptime copy_offset: UInt32 = UInt32(
                            _desc_offset[
                                4,
                                Index(
                                    num_copies_dim0,
                                    num_copies_dim1,
                                    num_copies_dim2,
                                    num_copies_dim3,
                                ),
                                Self.is_k_major,
                            ](Index(n, m, i, j))
                            * copy_size
                        )

                        cp_async_bulk_tensor_shared_cluster_global_elect[
                            cta_group=cta_group,
                            eviction_policy=eviction_policy,
                        ](
                            dst.ptr + copy_offset,
                            Pointer(to=self.descriptor).bitcast[NoneType](),
                            mem_barrier.unsafe_ptr(),
                            Index(
                                coords[0] + (j * copy_dim3),
                                coords[1] + (i * copy_dim2),
                                coords[2] + (m * copy_dim1),
                                coords[3] + (n * copy_dim0),
                            ),
                            elect,
                        )

    @always_inline
    def async_copy_4d[
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: TileTensor[mut=True, Self.dtype, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int, Int, Int],
    ):
        """
        Schedules an asynchronous copy from global memory to shared memory at specified 4D coordinates.

        TileTensor overload - accepts TileTensor instead of LayoutTensor.
        Assumes 128B alignment (TileTensor tiles are allocated with proper alignment).

        Parameters:
            cta_group: If the TMA is issued with cta_group == 2, only the leader CTA needs
                to be notified upon completion.
            eviction_policy: Optional cache eviction policy that controls how the data is handled
                in the cache hierarchy. Defaults to EVICT_NORMAL.

        Args:
            dst: TileTensor in shared memory where data will be copied.
            mem_barrier: The memory barrier for synchronization.
            coords: The 4D coordinates in the source tensor from which to copy data.
        """
        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_dim2 = Self.desc_shape[2]
        comptime copy_dim3 = Self.desc_shape[3]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = ceildiv(Self.tile_shape[0], copy_dim0)
        comptime num_copies_dim1 = ceildiv(Self.tile_shape[1], copy_dim1)
        comptime num_copies_dim2 = ceildiv(Self.tile_shape[2], copy_dim2)
        comptime num_copies_dim3 = ceildiv(Self.tile_shape[3], copy_dim3)
        comptime for n in range(num_copies_dim0):
            comptime for m in range(num_copies_dim1):
                comptime for i in range(num_copies_dim2):
                    comptime for j in range(num_copies_dim3):
                        comptime copy_offset: UInt32 = UInt32(
                            _desc_offset[
                                4,
                                Index(
                                    num_copies_dim0,
                                    num_copies_dim1,
                                    num_copies_dim2,
                                    num_copies_dim3,
                                ),
                                Self.is_k_major,
                            ](Index(n, m, i, j))
                            * copy_size
                        )

                        cp_async_bulk_tensor_shared_cluster_global[
                            cta_group=cta_group,
                            eviction_policy=eviction_policy,
                        ](
                            dst.ptr + copy_offset,
                            Pointer(to=self.descriptor).bitcast[NoneType](),
                            mem_barrier.unsafe_ptr(),
                            Index(
                                coords[0] + (j * copy_dim3),
                                coords[1] + (i * copy_dim2),
                                coords[2] + (m * copy_dim1),
                                coords[3] + (n * copy_dim0),
                            ),
                        )

    @always_inline
    def async_copy_4d_elect[
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: TileTensor[mut=True, Self.dtype, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int, Int, Int],
        elect: Int32,
    ):
        """Elect-predicated TileTensor overload of `async_copy_4d`.

        See the `LayoutTensor` overload of `async_copy_4d_elect` for
        semantics.

        Parameters:
            cta_group: If set to 2, the TMA emits `cta_group::2` PTX.
                Defaults to 1.
            eviction_policy: Cache eviction policy. Defaults to EVICT_NORMAL.

        Args:
            dst: TileTensor in shared memory where data will be copied.
            mem_barrier: The memory barrier for synchronization.
            coords: The 4D coordinates in the source tensor.
            elect: `0` on non-elected lanes, non-zero on the elected lane.
        """
        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_dim2 = Self.desc_shape[2]
        comptime copy_dim3 = Self.desc_shape[3]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = ceildiv(Self.tile_shape[0], copy_dim0)
        comptime num_copies_dim1 = ceildiv(Self.tile_shape[1], copy_dim1)
        comptime num_copies_dim2 = ceildiv(Self.tile_shape[2], copy_dim2)
        comptime num_copies_dim3 = ceildiv(Self.tile_shape[3], copy_dim3)
        comptime for n in range(num_copies_dim0):
            comptime for m in range(num_copies_dim1):
                comptime for i in range(num_copies_dim2):
                    comptime for j in range(num_copies_dim3):
                        comptime copy_offset: UInt32 = UInt32(
                            _desc_offset[
                                4,
                                Index(
                                    num_copies_dim0,
                                    num_copies_dim1,
                                    num_copies_dim2,
                                    num_copies_dim3,
                                ),
                                Self.is_k_major,
                            ](Index(n, m, i, j))
                            * copy_size
                        )

                        cp_async_bulk_tensor_shared_cluster_global_elect[
                            cta_group=cta_group,
                            eviction_policy=eviction_policy,
                        ](
                            dst.ptr + copy_offset,
                            Pointer(to=self.descriptor).bitcast[NoneType](),
                            mem_barrier.unsafe_ptr(),
                            Index(
                                coords[0] + (j * copy_dim3),
                                coords[1] + (i * copy_dim2),
                                coords[2] + (m * copy_dim1),
                                coords[3] + (n * copy_dim0),
                            ),
                            elect,
                        )

    @always_inline
    def async_copy_5d[
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: LayoutTensor[mut=True, Self.dtype, _, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int, Int, Int, Int],
    ):
        """
        Schedules an asynchronous copy from global memory to shared memory at specified 5D coordinates.

        This method initiates a hardware-accelerated asynchronous transfer of data from global memory
        to the specified destination in shared memory for 5D tensors. The transfer is tracked by the
        provided memory barrier.

        Parameters:
            cta_group: Int
                If the TMA is issued with cta_group == 2, only the leader CTA needs
                to be notified upon completion.
            eviction_policy: Optional cache eviction policy that controls how the data is handled
                in the cache hierarchy. Defaults to EVICT_NORMAL.

        Args:
            dst: The destination tensor in shared memory where data will be copied.
                 Must be 128-byte aligned.
            mem_barrier: The memory barrier used to track and synchronize the asynchronous transfer.
            coords: The 5D coordinates in the source tensor from which to copy data.

        Constraints:

            - The destination tensor must be 128-byte aligned in shared memory.
            - The descriptor layout may be smaller than the shared memory tile shape
              to accommodate hardware requirements.
        """
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#table-alignment-multi-dim-tma
        comptime assert (
            type_of(dst).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_dim2 = Self.desc_shape[2]
        comptime copy_dim3 = Self.desc_shape[3]
        comptime copy_dim4 = Self.desc_shape[4]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = ceildiv(Self.tile_shape[0], copy_dim0)
        comptime num_copies_dim1 = ceildiv(Self.tile_shape[1], copy_dim1)
        comptime num_copies_dim2 = ceildiv(Self.tile_shape[2], copy_dim2)
        comptime num_copies_dim3 = ceildiv(Self.tile_shape[3], copy_dim3)
        comptime num_copies_dim4 = ceildiv(Self.tile_shape[4], copy_dim4)
        comptime for o in range(num_copies_dim0):
            comptime for n in range(num_copies_dim1):
                comptime for m in range(num_copies_dim2):
                    comptime for i in range(num_copies_dim3):
                        comptime for j in range(num_copies_dim4):
                            comptime copy_offset: UInt32 = UInt32(
                                _desc_offset[
                                    5,
                                    Index(
                                        num_copies_dim0,
                                        num_copies_dim1,
                                        num_copies_dim2,
                                        num_copies_dim3,
                                        num_copies_dim4,
                                    ),
                                    Self.is_k_major,
                                ](Index(o, n, m, i, j))
                                * copy_size
                            )

                            cp_async_bulk_tensor_shared_cluster_global[
                                cta_group=cta_group,
                                eviction_policy=eviction_policy,
                            ](
                                dst.ptr + copy_offset,
                                Pointer(to=self.descriptor).bitcast[NoneType](),
                                mem_barrier.unsafe_ptr(),
                                Index(
                                    coords[0] + (j * copy_dim4),
                                    coords[1] + (i * copy_dim3),
                                    coords[2] + (m * copy_dim2),
                                    coords[3] + (n * copy_dim1),
                                    coords[4] + (o * copy_dim0),
                                ),
                            )

    @always_inline
    def async_copy_5d_elect[
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: LayoutTensor[mut=True, Self.dtype, _, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int, Int, Int, Int],
        elect: Int32,
    ):
        """Elect-predicated overload of `async_copy_5d`.

        See `async_copy_elect` for semantics: each unrolled TMA issue is
        predicated in-PTX on `elect`.

        Parameters:
            cta_group: If set to 2, the TMA emits `cta_group::2` PTX.
                Defaults to 1.
            eviction_policy: Cache eviction policy. Defaults to EVICT_NORMAL.

        Args:
            dst: The destination tensor in shared memory. Must be 128-byte
                aligned.
            mem_barrier: The memory barrier for synchronization.
            coords: The 5D coordinates in the source tensor.
            elect: `0` on non-elected lanes, non-zero on the elected lane.
        """
        comptime assert (
            type_of(dst).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_dim2 = Self.desc_shape[2]
        comptime copy_dim3 = Self.desc_shape[3]
        comptime copy_dim4 = Self.desc_shape[4]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = ceildiv(Self.tile_shape[0], copy_dim0)
        comptime num_copies_dim1 = ceildiv(Self.tile_shape[1], copy_dim1)
        comptime num_copies_dim2 = ceildiv(Self.tile_shape[2], copy_dim2)
        comptime num_copies_dim3 = ceildiv(Self.tile_shape[3], copy_dim3)
        comptime num_copies_dim4 = ceildiv(Self.tile_shape[4], copy_dim4)
        comptime for o in range(num_copies_dim0):
            comptime for n in range(num_copies_dim1):
                comptime for m in range(num_copies_dim2):
                    comptime for i in range(num_copies_dim3):
                        comptime for j in range(num_copies_dim4):
                            comptime copy_offset: UInt32 = UInt32(
                                _desc_offset[
                                    5,
                                    Index(
                                        num_copies_dim0,
                                        num_copies_dim1,
                                        num_copies_dim2,
                                        num_copies_dim3,
                                        num_copies_dim4,
                                    ),
                                    Self.is_k_major,
                                ](Index(o, n, m, i, j))
                                * copy_size
                            )

                            cp_async_bulk_tensor_shared_cluster_global_elect[
                                cta_group=cta_group,
                                eviction_policy=eviction_policy,
                            ](
                                dst.ptr + copy_offset,
                                Pointer(to=self.descriptor).bitcast[NoneType](),
                                mem_barrier.unsafe_ptr(),
                                Index(
                                    coords[0] + (j * copy_dim4),
                                    coords[1] + (i * copy_dim3),
                                    coords[2] + (m * copy_dim2),
                                    coords[3] + (n * copy_dim1),
                                    coords[4] + (o * copy_dim0),
                                ),
                                elect,
                            )

    @always_inline
    def async_copy_5d[
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: TileTensor[mut=True, Self.dtype, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int, Int, Int, Int],
    ):
        """
        Schedules an asynchronous copy from global memory to shared memory at specified 5D coordinates.

        TileTensor overload - accepts TileTensor instead of LayoutTensor.
        Assumes 128B alignment (TileTensor tiles are allocated with proper alignment).

        Parameters:
            cta_group: If the TMA is issued with cta_group == 2, only the leader CTA needs
                to be notified upon completion.
            eviction_policy: Optional cache eviction policy that controls how the data is handled
                in the cache hierarchy. Defaults to EVICT_NORMAL.

        Args:
            dst: TileTensor in shared memory where data will be copied.
            mem_barrier: The memory barrier for synchronization.
            coords: The 5D coordinates in the source tensor from which to copy data.
        """
        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_dim2 = Self.desc_shape[2]
        comptime copy_dim3 = Self.desc_shape[3]
        comptime copy_dim4 = Self.desc_shape[4]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = ceildiv(Self.tile_shape[0], copy_dim0)
        comptime num_copies_dim1 = ceildiv(Self.tile_shape[1], copy_dim1)
        comptime num_copies_dim2 = ceildiv(Self.tile_shape[2], copy_dim2)
        comptime num_copies_dim3 = ceildiv(Self.tile_shape[3], copy_dim3)
        comptime num_copies_dim4 = ceildiv(Self.tile_shape[4], copy_dim4)
        comptime for o in range(num_copies_dim0):
            comptime for n in range(num_copies_dim1):
                comptime for m in range(num_copies_dim2):
                    comptime for i in range(num_copies_dim3):
                        comptime for j in range(num_copies_dim4):
                            comptime copy_offset: UInt32 = UInt32(
                                _desc_offset[
                                    5,
                                    Index(
                                        num_copies_dim0,
                                        num_copies_dim1,
                                        num_copies_dim2,
                                        num_copies_dim3,
                                        num_copies_dim4,
                                    ),
                                    Self.is_k_major,
                                ](Index(o, n, m, i, j))
                                * copy_size
                            )

                            cp_async_bulk_tensor_shared_cluster_global[
                                cta_group=cta_group,
                                eviction_policy=eviction_policy,
                            ](
                                dst.ptr + copy_offset,
                                Pointer(to=self.descriptor).bitcast[NoneType](),
                                mem_barrier.unsafe_ptr(),
                                Index(
                                    coords[0] + (j * copy_dim4),
                                    coords[1] + (i * copy_dim3),
                                    coords[2] + (m * copy_dim2),
                                    coords[3] + (n * copy_dim1),
                                    coords[4] + (o * copy_dim0),
                                ),
                            )

    @always_inline
    def async_copy_5d_elect[
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: TileTensor[mut=True, Self.dtype, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int, Int, Int, Int],
        elect: Int32,
    ):
        """Elect-predicated TileTensor overload of `async_copy_5d`.

        See the `LayoutTensor` overload of `async_copy_5d_elect` for
        semantics.

        Parameters:
            cta_group: If set to 2, the TMA emits `cta_group::2` PTX.
                Defaults to 1.
            eviction_policy: Cache eviction policy. Defaults to EVICT_NORMAL.

        Args:
            dst: TileTensor in shared memory where data will be copied.
            mem_barrier: The memory barrier for synchronization.
            coords: The 5D coordinates in the source tensor.
            elect: `0` on non-elected lanes, non-zero on the elected lane.
        """
        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_dim2 = Self.desc_shape[2]
        comptime copy_dim3 = Self.desc_shape[3]
        comptime copy_dim4 = Self.desc_shape[4]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = ceildiv(Self.tile_shape[0], copy_dim0)
        comptime num_copies_dim1 = ceildiv(Self.tile_shape[1], copy_dim1)
        comptime num_copies_dim2 = ceildiv(Self.tile_shape[2], copy_dim2)
        comptime num_copies_dim3 = ceildiv(Self.tile_shape[3], copy_dim3)
        comptime num_copies_dim4 = ceildiv(Self.tile_shape[4], copy_dim4)
        comptime for o in range(num_copies_dim0):
            comptime for n in range(num_copies_dim1):
                comptime for m in range(num_copies_dim2):
                    comptime for i in range(num_copies_dim3):
                        comptime for j in range(num_copies_dim4):
                            comptime copy_offset: UInt32 = UInt32(
                                _desc_offset[
                                    5,
                                    Index(
                                        num_copies_dim0,
                                        num_copies_dim1,
                                        num_copies_dim2,
                                        num_copies_dim3,
                                        num_copies_dim4,
                                    ),
                                    Self.is_k_major,
                                ](Index(o, n, m, i, j))
                                * copy_size
                            )

                            cp_async_bulk_tensor_shared_cluster_global_elect[
                                cta_group=cta_group,
                                eviction_policy=eviction_policy,
                            ](
                                dst.ptr + copy_offset,
                                Pointer(to=self.descriptor).bitcast[NoneType](),
                                mem_barrier.unsafe_ptr(),
                                Index(
                                    coords[0] + (j * copy_dim4),
                                    coords[1] + (i * copy_dim3),
                                    coords[2] + (m * copy_dim2),
                                    coords[3] + (n * copy_dim1),
                                    coords[4] + (o * copy_dim0),
                                ),
                                elect,
                            )

    @always_inline("nodebug")
    def async_copy[
        coord_rank: Int,
        //,
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: LayoutTensor[mut=True, Self.dtype, _, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: StaticTuple[UInt32, coord_rank],
    ):
        """Schedules an asynchronous copy from global memory to shared memory for N-dimensional tensors.

        This is a generic dispatcher that selects the appropriate rank-specific async copy method
        based on the tensor rank. It provides a unified interface for initiating TMA transfers
        across 2D, 3D, 4D, and 5D tensors using `StaticTuple` coordinates.

        Parameters:
            coord_rank: The dimensionality of the tensor (must be 2, 3, 4, or 5).
            cta_group: If set to 2, only the leader CTA needs to be notified upon completion.
                Defaults to 1.
            eviction_policy: Optional cache eviction policy that controls how the data is handled
                in the cache hierarchy. Defaults to EVICT_NORMAL.

        Args:
            dst: The destination tensor in shared memory where data will be copied.
                Must be 128-byte aligned.
            mem_barrier: The memory barrier used to track and synchronize the asynchronous transfer.
            coords: The N-dimensional coordinates in the source tensor from which to copy data,
                provided as a `StaticTuple` of `UInt32` values.

        Constraints:
            - The coord_rank must be 2, 3, 4, or 5.
            - The destination tensor must be 128-byte aligned in shared memory.
        """
        comptime assert coord_rank in (2, 3, 4, 5)

        comptime if coord_rank == 2:
            self.async_copy[
                cta_group=cta_group, eviction_policy=eviction_policy
            ](dst, mem_barrier, (Int(coords[0]), Int(coords[1])))
        elif coord_rank == 3:
            self.async_copy_3d[
                cta_group=cta_group, eviction_policy=eviction_policy
            ](
                dst,
                mem_barrier,
                (Int(coords[0]), Int(coords[1]), Int(coords[2])),
            )
        elif coord_rank == 4:
            self.async_copy_4d[
                cta_group=cta_group, eviction_policy=eviction_policy
            ](
                dst,
                mem_barrier,
                (
                    Int(coords[0]),
                    Int(coords[1]),
                    Int(coords[2]),
                    Int(coords[3]),
                ),
            )
        elif coord_rank == 5:
            self.async_copy_5d[
                cta_group=cta_group, eviction_policy=eviction_policy
            ](
                dst,
                mem_barrier,
                (
                    Int(coords[0]),
                    Int(coords[1]),
                    Int(coords[2]),
                    Int(coords[3]),
                    Int(coords[4]),
                ),
            )

    @always_inline("nodebug")
    def async_copy_elect[
        coord_rank: Int,
        //,
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: LayoutTensor[mut=True, Self.dtype, _, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: StaticTuple[UInt32, coord_rank],
        elect: Int32,
    ):
        """Elect-predicated rank-dispatched overload of `async_copy`.

        Dispatches to the rank-specific `async_copy_*_elect` methods. Each
        underlying TMA issue is predicated in-PTX on `elect`.

        Parameters:
            coord_rank: The dimensionality (must be 2, 3, 4, or 5).
            cta_group: If set to 2, the TMA emits `cta_group::2` PTX.
                Defaults to 1.
            eviction_policy: Cache eviction policy. Defaults to EVICT_NORMAL.

        Args:
            dst: The destination tensor in shared memory. Must be 128-byte
                aligned.
            mem_barrier: The memory barrier for synchronization.
            coords: The N-dimensional coordinates as `StaticTuple`.
            elect: `0` on non-elected lanes, non-zero on the elected lane.
        """
        comptime assert coord_rank in (2, 3, 4, 5)

        comptime if coord_rank == 2:
            self.async_copy_elect[
                cta_group=cta_group, eviction_policy=eviction_policy
            ](dst, mem_barrier, (Int(coords[0]), Int(coords[1])), elect)
        elif coord_rank == 3:
            self.async_copy_3d_elect[
                cta_group=cta_group, eviction_policy=eviction_policy
            ](
                dst,
                mem_barrier,
                (Int(coords[0]), Int(coords[1]), Int(coords[2])),
                elect,
            )
        elif coord_rank == 4:
            self.async_copy_4d_elect[
                cta_group=cta_group, eviction_policy=eviction_policy
            ](
                dst,
                mem_barrier,
                (
                    Int(coords[0]),
                    Int(coords[1]),
                    Int(coords[2]),
                    Int(coords[3]),
                ),
                elect,
            )
        elif coord_rank == 5:
            self.async_copy_5d_elect[
                cta_group=cta_group, eviction_policy=eviction_policy
            ](
                dst,
                mem_barrier,
                (
                    Int(coords[0]),
                    Int(coords[1]),
                    Int(coords[2]),
                    Int(coords[3]),
                    Int(coords[4]),
                ),
                elect,
            )

    @always_inline("nodebug")
    def async_copy[
        coord_rank: Int,
        //,
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: TileTensor[mut=True, Self.dtype, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: StaticTuple[UInt32, coord_rank],
    ):
        """TileTensor overload of the generic rank-dispatched async_copy.
        Dispatches to the rank-specific TileTensor async_copy methods.

        Parameters:
            coord_rank: The dimensionality (must be >=2 and <= 5).
            cta_group: CTA group configuration. Defaults to 1.
            eviction_policy: Cache eviction policy. Defaults to EVICT_NORMAL.

        Args:
            dst: TileTensor in shared memory where data will be copied.
            mem_barrier: The memory barrier for synchronization.
            coords: The N-dimensional coordinates as StaticTuple.
        """
        comptime assert coord_rank in (2, 3, 4, 5)

        comptime if coord_rank == 2:
            self.async_copy[
                cta_group=cta_group, eviction_policy=eviction_policy
            ](dst, mem_barrier, (Int(coords[0]), Int(coords[1])))
        elif coord_rank == 3:
            self.async_copy_3d[
                cta_group=cta_group, eviction_policy=eviction_policy
            ](
                dst,
                mem_barrier,
                (Int(coords[0]), Int(coords[1]), Int(coords[2])),
            )
        elif coord_rank == 4:
            self.async_copy_4d[
                cta_group=cta_group, eviction_policy=eviction_policy
            ](
                dst,
                mem_barrier,
                (
                    Int(coords[0]),
                    Int(coords[1]),
                    Int(coords[2]),
                    Int(coords[3]),
                ),
            )
        else:
            self.async_copy_5d[
                cta_group=cta_group, eviction_policy=eviction_policy
            ](
                dst,
                mem_barrier,
                (
                    Int(coords[0]),
                    Int(coords[1]),
                    Int(coords[2]),
                    Int(coords[3]),
                    Int(coords[4]),
                ),
            )

    @always_inline("nodebug")
    def async_copy_elect[
        coord_rank: Int,
        //,
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: TileTensor[mut=True, Self.dtype, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: StaticTuple[UInt32, coord_rank],
        elect: Int32,
    ):
        """Elect-predicated TileTensor rank-dispatched overload of
        `async_copy`. Dispatches to the rank-specific `_elect` methods.

        Parameters:
            coord_rank: The dimensionality (must be 2, 3, 4, or 5).
            cta_group: If set to 2, the TMA emits `cta_group::2` PTX.
                Defaults to 1.
            eviction_policy: Cache eviction policy. Defaults to EVICT_NORMAL.

        Args:
            dst: TileTensor in shared memory where data will be copied.
            mem_barrier: The memory barrier for synchronization.
            coords: The N-dimensional coordinates as `StaticTuple`.
            elect: `0` on non-elected lanes, non-zero on the elected lane.
        """
        comptime assert coord_rank in (2, 3, 4, 5)

        comptime if coord_rank == 2:
            self.async_copy_elect[
                cta_group=cta_group, eviction_policy=eviction_policy
            ](dst, mem_barrier, (Int(coords[0]), Int(coords[1])), elect)
        elif coord_rank == 3:
            self.async_copy_3d_elect[
                cta_group=cta_group, eviction_policy=eviction_policy
            ](
                dst,
                mem_barrier,
                (Int(coords[0]), Int(coords[1]), Int(coords[2])),
                elect,
            )
        elif coord_rank == 4:
            self.async_copy_4d_elect[
                cta_group=cta_group, eviction_policy=eviction_policy
            ](
                dst,
                mem_barrier,
                (
                    Int(coords[0]),
                    Int(coords[1]),
                    Int(coords[2]),
                    Int(coords[3]),
                ),
                elect,
            )
        else:
            self.async_copy_5d_elect[
                cta_group=cta_group, eviction_policy=eviction_policy
            ](
                dst,
                mem_barrier,
                (
                    Int(coords[0]),
                    Int(coords[1]),
                    Int(coords[2]),
                    Int(coords[3]),
                    Int(coords[4]),
                ),
                elect,
            )

    @always_inline("nodebug")
    def async_copy_gather4[
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: LayoutTensor[mut=True, _, _, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        col_idx: Int32,
        row0: Int32,
        row1: Int32,
        row2: Int32,
        row3: Int32,
    ):
        """Schedules an asynchronous gather4 copy of 4 non-contiguous rows from global memory to shared memory.

        This method uses the TMA gather4 hardware instruction (SM100/Blackwell) to load 4 rows
        at arbitrary row indices from a 2D tensor in global memory, placing them contiguously
        in shared memory. The TMA descriptor must be configured with box dim1=1 (one row per tile).

        Parameters:
            cta_group: If the TMA is issued with cta_group == 2, only the leader CTA needs
                to be notified upon completion. Defaults to 1.
            eviction_policy: Cache eviction policy that controls how the data is handled
                in the cache hierarchy. Defaults to EVICT_NORMAL.

        Args:
            dst: The destination tensor in shared memory where data will be copied.
                Must be 128-byte aligned.
            mem_barrier: The memory barrier used to track and synchronize the asynchronous transfer.
            col_idx: Column offset in the source tensor (typically 0 for full-row loads).
            row0: Row index of the first row to gather.
            row1: Row index of the second row to gather.
            row2: Row index of the third row to gather.
            row3: Row index of the fourth row to gather.

        Constraints:
            - Requires rank == 2 (gather4 is 2D only).
            - Requires desc_shape[0] == 1 (gather4 hardware requirement: one row per tile).
            - The destination tensor must be 128-byte aligned in shared memory.
            - Requires SM100 (Blackwell) or newer GPU architecture.
        """
        comptime assert (
            Self.rank == 2
        ), "gather4 is only supported for 2D tensors (rank == 2)"
        comptime assert (
            Self.desc_shape[0] == 1
        ), "gather4 requires desc_shape row dimension == 1 (one row per tile)"
        comptime assert (
            type_of(dst).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        comptime assert (
            type_of(dst).dtype == Self.dtype
        ), "Input tensor has a different type than the TMA op"

        cp_async_bulk_tensor_2d_gather4[
            cta_group=cta_group,
            eviction_policy=eviction_policy,
        ](
            dst.ptr,
            Pointer(to=self.descriptor).bitcast[NoneType](),
            mem_barrier.unsafe_ptr(),
            col_idx,
            row0,
            row1,
            row2,
            row3,
        )

    @always_inline("nodebug")
    def async_copy_gather4[
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: TileTensor[mut=True, Self.dtype, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        col_idx: Int32,
        row0: Int32,
        row1: Int32,
        row2: Int32,
        row3: Int32,
    ):
        """Schedules an asynchronous gather4 copy of 4 non-contiguous rows from global memory to shared memory.

        This method uses the TMA gather4 hardware instruction (SM100/Blackwell) to load 4 rows
        at arbitrary row indices from a 2D tensor in global memory, placing them contiguously
        in shared memory. The TMA descriptor must be configured with box dim1=1 (one row per tile).

        Parameters:
            cta_group: If the TMA is issued with cta_group == 2, only the leader CTA needs
                to be notified upon completion. Defaults to 1.
            eviction_policy: Cache eviction policy that controls how the data is handled
                in the cache hierarchy. Defaults to EVICT_NORMAL.

        Args:
            dst: The destination tensor in shared memory where data will be copied.
                Must be 128-byte aligned.
            mem_barrier: The memory barrier used to track and synchronize the asynchronous transfer.
            col_idx: Column offset in the source tensor (typically 0 for full-row loads).
            row0: Row index of the first row to gather.
            row1: Row index of the second row to gather.
            row2: Row index of the third row to gather.
            row3: Row index of the fourth row to gather.

        Constraints:
            - Requires rank == 2 (gather4 is 2D only).
            - Requires desc_shape[0] == 1 (gather4 hardware requirement: one row per tile).
            - The destination tensor must be 128-byte aligned in shared memory.
            - Requires SM100 (Blackwell) or newer GPU architecture.
        """
        comptime assert (
            Self.rank == 2
        ), "gather4 is only supported for 2D tensors (rank == 2)"
        comptime assert (
            Self.desc_shape[0] == 1
        ), "gather4 requires desc_shape row dimension == 1 (one row per tile)"

        comptime assert (
            type_of(dst).dtype == Self.dtype
        ), "Input tensor has a different type than the TMA op"

        cp_async_bulk_tensor_2d_gather4[
            cta_group=cta_group,
            eviction_policy=eviction_policy,
        ](
            dst.ptr,
            Pointer(to=self.descriptor).bitcast[NoneType](),
            mem_barrier.unsafe_ptr(),
            col_idx,
            row0,
            row1,
            row2,
            row3,
        )

    @always_inline
    def gather4_tile_bytes[tile_width: Int](self) -> Int32:
        """Returns total expected bytes for a full gather4 tile load.

        Computes ``tile_height * tile_width * sizeof(dtype)`` which is
        the number of bytes that ``async_copy_gather4_tile`` will transfer
        into shared memory.  Pass this value to
        ``SharedMemBarrier.expect_bytes`` before issuing the tile load.

        Parameters:
            tile_width: Total number of elements per row in global
                memory.

        Returns:
            The total expected transfer size in bytes as ``Int32``.
        """
        comptime BN = Self.tile_shape[0]
        return Int32(BN * tile_width * size_of[Self.dtype]())

    @always_inline
    def async_copy_gather4_tile[
        tile_width: Int,
        cta_group: Int = 1,
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
        d_indices_addr_space: AddressSpace = .GENERIC,
    ](
        self,
        smem_base: MutPointer[Scalar[Self.dtype], _, address_space=.SHARED],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        d_indices: ImmPointer[Int32, _, address_space=d_indices_addr_space],
        start_idx: Int = 0,
    ):
        """Loads a full tile of ``tile_height`` rows via gather4 in 4-row chunks.

        Internally loops over column groups and 4-row chunks, issuing one
        ``async_copy_gather4`` call per chunk per column group.  The SMEM
        destination layout matches the bulk TMA async_copy ordering: column
        groups are stored contiguously (each group holds ``tile_height``
        rows of ``box_width`` elements), and within each group 4-row chunks
        are contiguous.

        The caller must call ``mem_barrier.expect_bytes(self.gather4_tile_bytes[tile_width]())``
        before invoking this method.

        Parameters:
            tile_width: Total number of elements per row in global
                memory.
            cta_group: CTA group configuration. Defaults to 1.
            eviction_policy: Cache eviction policy. Defaults to
                EVICT_NORMAL.
            d_indices_addr_space: Address space of the ``d_indices``
                pointer. Defaults to ``GENERIC``, but callers may
                pass ``SHARED`` pointers directly.

        Args:
            smem_base: Base pointer to the shared memory destination region.
            mem_barrier: The shared memory barrier tracking the transfers.
            d_indices: Pointer to ``tile_height`` Int32 row indices in shared
                or global memory.
            start_idx: Offset into ``d_indices`` for the first row index.
                Defaults to 0.
        """
        comptime BN = Self.tile_shape[0]
        comptime box_w = Self.tile_shape[1]
        comptime num_col_groups = ceildiv(tile_width, box_w)
        comptime num_4row_chunks = BN // 4

        var desc_ptr = Pointer(to=self.descriptor).bitcast[NoneType]()
        var mbar_ptr = mem_barrier.unsafe_ptr()

        comptime for cg in range(num_col_groups):
            comptime for c in range(num_4row_chunks):
                var idx = start_idx + c * 4
                var elem_off = cg * BN * box_w + c * 4 * box_w
                var dst_ptr = smem_base + elem_off
                cp_async_bulk_tensor_2d_gather4[
                    cta_group=cta_group,
                    eviction_policy=eviction_policy,
                ](
                    dst_ptr,
                    desc_ptr,
                    mbar_ptr,
                    Int32(cg * box_w),
                    d_indices[idx + 0],
                    d_indices[idx + 1],
                    d_indices[idx + 2],
                    d_indices[idx + 3],
                )

    @always_inline
    def async_store[
        coord_rank: Int, //, cta_group: Int = 1
    ](
        self,
        dst: LayoutTensor[Self.dtype, _, address_space=.SHARED, ...],
        coords: StaticTuple[UInt32, coord_rank],
    ):
        """Schedules an asynchronous store from shared memory to global memory for N-dimensional tensors.

        This is a generic dispatcher that selects the appropriate rank-specific async store method
        based on the tensor rank. It provides a unified interface for initiating TMA store operations
        across 2D, 3D, 4D, and 5D tensors using `StaticTuple` coordinates.

        Parameters:
            coord_rank: The dimensionality of the tensor (must be 2, 3, 4, or 5).
            cta_group: CTA group configuration for the store operation. Defaults to 1.

        Args:
            dst: The source tensor in shared memory from which data will be copied to global memory.
                Must be 128-byte aligned.
            coords: The N-dimensional coordinates in the destination global tensor where data
                will be stored, provided as a `StaticTuple` of `UInt32` values.

        Constraints:
            - The coord_rank must be 2, 3, 4, or 5.
            - The source tensor must be 128-byte aligned in shared memory.
        """
        comptime assert coord_rank in (2, 3, 4, 5)

        comptime if coord_rank == 2:
            self.async_store(dst, (Int(coords[0]), Int(coords[1])))
        elif coord_rank == 3:
            self.async_store_3d(
                dst,
                (Int(coords[0]), Int(coords[1]), Int(coords[2])),
            )
        elif coord_rank == 4:
            self.async_store_4d(
                dst,
                (
                    Int(coords[0]),
                    Int(coords[1]),
                    Int(coords[2]),
                    Int(coords[3]),
                ),
            )
        elif coord_rank == 5:
            self.async_store_5d(
                dst,
                (
                    Int(coords[0]),
                    Int(coords[1]),
                    Int(coords[2]),
                    Int(coords[3]),
                    Int(coords[4]),
                ),
            )

    @always_inline
    def async_store[
        coord_rank: Int, //, cta_group: Int = 1
    ](
        self,
        dst: TileTensor[Self.dtype, address_space=.SHARED, ...],
        coords: StaticTuple[UInt32, coord_rank],
    ):
        """Schedules an asynchronous store from shared memory to global memory.

        TileTensor overload of the generic rank-dispatched async_store.
        Dispatches to the rank-specific TileTensor async_store methods.

        Parameters:
            coord_rank: The dimensionality of the tensor (must be 2 or 3).
            cta_group: CTA group configuration. Defaults to 1.

        Args:
            dst: TileTensor in shared memory from which data will be copied.
            coords: The N-dimensional coordinates in the destination tensor.
        """
        comptime assert coord_rank in (2, 3, 4)

        comptime if coord_rank == 2:
            self.async_store(dst, (Int(coords[0]), Int(coords[1])))
        elif coord_rank == 3:
            self.async_store_3d(
                dst,
                (Int(coords[0]), Int(coords[1]), Int(coords[2])),
            )
        elif coord_rank == 4:
            self.async_store_4d(
                dst,
                (
                    Int(coords[0]),
                    Int(coords[1]),
                    Int(coords[2]),
                    Int(coords[3]),
                ),
            )

    @always_inline
    def async_multicast_load[
        cta_group: Int = 1
    ](
        self,
        dst: LayoutTensor[Self.dtype, _, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int],
        multicast_mask: UInt16,
    ):
        """
        Schedules an asynchronous multicast load from global memory to multiple shared memory locations.

        This method initiates a hardware-accelerated asynchronous transfer of data from global memory
        to multiple destination locations in shared memory across different CTAs (Cooperative Thread Arrays)
        as specified by the multicast mask.

        Parameters:
            cta_group: Int
                If the TMA is issued with cta_group == 2, only the leader CTA needs
                to be notified upon completion.

        Args:
            dst: LayoutTensor
                The destination tensor in shared memory where data will be copied.
                Must be 128-byte aligned.
            mem_barrier: SharedMemBarrierArray
                The memory barrier used to track and synchronize the asynchronous transfer.
            coords: Tuple[Int, Int]
                The 2D coordinates in the source tensor from which to copy data.
            multicast_mask: UInt16
                A bit mask specifying which CTAs should receive the data.

        Constraints:
            The destination tensor must be 128-byte aligned in shared memory.
        """
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#table-alignment-multi-dim-tma
        comptime assert (
            type_of(dst).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = Self.tile_shape[0] // copy_dim0
        comptime num_copies_dim1 = Self.tile_shape[1] // copy_dim1

        comptime for i in range(num_copies_dim0):
            comptime for j in range(num_copies_dim1):
                comptime copy_offset: UInt32 = UInt32(
                    (i * num_copies_dim1 + j) * copy_size
                )

                cp_async_bulk_tensor_shared_cluster_global_multicast[
                    cta_group=cta_group
                ](
                    dst.ptr.unsafe_mut_cast[True]() + copy_offset,
                    Pointer(to=self.descriptor).bitcast[NoneType](),
                    mem_barrier.unsafe_ptr(),
                    Index(
                        coords[0] + j * copy_dim1,
                        coords[1] + i * copy_dim0,
                    ),
                    multicast_mask,
                )

    @always_inline
    def async_multicast_load[
        cta_group: Int = 1,
    ](
        self,
        dst: TileTensor[mut=True, Self.dtype, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int],
        multicast_mask: UInt16,
    ):
        """
        Schedules an asynchronous 2D multicast load from global to shared memory.

        TileTensor overload - accepts TileTensor instead of LayoutTensor.
        Assumes 128B alignment (TileTensor tiles are allocated with proper alignment).

        Parameters:
            cta_group: If issued with cta_group == 2, only the leader CTA needs
                to be notified upon completion.

        Args:
            dst: TileTensor in shared memory where data will be copied.
            mem_barrier: The memory barrier for synchronization.
            coords: The 2D coordinates in the source tensor from which to copy.
            multicast_mask: Bit mask specifying which CTAs should receive the data.
        """
        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = Self.tile_shape[0] // copy_dim0
        comptime num_copies_dim1 = Self.tile_shape[1] // copy_dim1

        comptime for i in range(num_copies_dim0):
            comptime for j in range(num_copies_dim1):
                comptime copy_offset: UInt32 = UInt32(
                    (i * num_copies_dim1 + j) * copy_size
                )

                cp_async_bulk_tensor_shared_cluster_global_multicast[
                    cta_group=cta_group
                ](
                    dst.ptr + copy_offset,
                    Pointer(to=self.descriptor).bitcast[NoneType](),
                    mem_barrier.unsafe_ptr(),
                    Index(
                        coords[0] + j * copy_dim1,
                        coords[1] + i * copy_dim0,
                    ),
                    multicast_mask,
                )

    @always_inline
    def async_multicast_load_3d[
        cta_group: Int = 1
    ](
        self,
        dst: LayoutTensor[mut=True, Self.dtype, _, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int, Int],
        multicast_mask: UInt16,
    ):
        """
        Schedules an asynchronous 3D multicast load from global memory to multiple shared memory locations.

        This method initiates a hardware-accelerated asynchronous transfer of data from global memory
        to multiple destination locations in shared memory across different CTAs (Cooperative Thread Arrays)
        as specified by the multicast mask.

        Parameters:
            cta_group: Int
                If the TMA is issued with cta_group == 2, only the leader CTA needs
                to be notified upon completion.

        Args:
            dst: LayoutTensor
                The destination tensor in shared memory where data will be copied.
                Must be 128-byte aligned.
            mem_barrier: SharedMemBarrierArray
                The memory barrier used to track and synchronize the asynchronous transfer.
            coords: The 2D coordinates in the source tensor from which to copy data.
            multicast_mask: UInt16
                A bit mask specifying which CTAs should receive the data.

        Constraints:
            The destination tensor must be 128-byte aligned in shared memory.
        """
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#table-alignment-multi-dim-tma
        comptime assert (
            type_of(dst).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        # The descriptor layout i.e. data per copy can be smaller than the shared memory
        # tile shape due to WGMMA requirement. E.g. k-major no swizzle WGMMA BM x 16B to be
        # one continuous chunk in shared memory. We need to break down tile shape in K by 16B.
        #
        # dim0, dim1 are MN, K for K-major and K, MN for MN-major because our inputs are
        # row_major(K, MN) for the latter.
        #
        # TODO: use layout algebra here
        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_dim2 = Self.desc_shape[2]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = ceildiv(Self.tile_shape[0], copy_dim0)
        comptime num_copies_dim1 = ceildiv(Self.tile_shape[1], copy_dim1)
        comptime num_copies_dim2 = ceildiv(Self.tile_shape[2], copy_dim2)

        comptime for m in range(num_copies_dim0):
            comptime for i in range(num_copies_dim1):
                comptime for j in range(num_copies_dim2):
                    comptime copy_offset: UInt32 = UInt32(
                        _desc_offset[
                            3,
                            Index(
                                num_copies_dim0,
                                num_copies_dim1,
                                num_copies_dim2,
                            ),
                            Self.is_k_major,
                        ](Index(m, i, j))
                        * copy_size
                    )

                    cp_async_bulk_tensor_shared_cluster_global_multicast[
                        cta_group=cta_group
                    ](
                        dst.ptr + copy_offset,
                        Pointer(to=self.descriptor).bitcast[NoneType](),
                        mem_barrier.unsafe_ptr(),
                        Index(
                            coords[0] + j * copy_dim2,
                            coords[1] + i * copy_dim1,
                            coords[2] + m * copy_dim0,
                        ),
                        multicast_mask,
                    )

    @always_inline
    def async_multicast_load_3d[
        cta_group: Int = 1,
    ](
        self,
        dst: TileTensor[mut=True, Self.dtype, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int, Int],
        multicast_mask: UInt16,
    ):
        """
        Schedules an asynchronous 3D multicast load from global to shared memory.

        TileTensor overload - accepts TileTensor instead of LayoutTensor.
        Assumes 128B alignment (TileTensor tiles are allocated with proper alignment).

        Parameters:
            cta_group: If issued with cta_group == 2, only the leader CTA needs
                to be notified upon completion.

        Args:
            dst: TileTensor in shared memory where data will be copied.
            mem_barrier: The memory barrier for synchronization.
            coords: The 3D coordinates in the source tensor from which to copy.
            multicast_mask: Bit mask specifying which CTAs should receive the data.
        """
        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_dim2 = Self.desc_shape[2]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = ceildiv(Self.tile_shape[0], copy_dim0)
        comptime num_copies_dim1 = ceildiv(Self.tile_shape[1], copy_dim1)
        comptime num_copies_dim2 = ceildiv(Self.tile_shape[2], copy_dim2)

        comptime for m in range(num_copies_dim0):
            comptime for i in range(num_copies_dim1):
                comptime for j in range(num_copies_dim2):
                    comptime copy_offset: UInt32 = UInt32(
                        _desc_offset[
                            3,
                            Index(
                                num_copies_dim0,
                                num_copies_dim1,
                                num_copies_dim2,
                            ),
                            Self.is_k_major,
                        ](Index(m, i, j))
                        * copy_size
                    )

                    cp_async_bulk_tensor_shared_cluster_global_multicast[
                        cta_group=cta_group
                    ](
                        dst.ptr + copy_offset,
                        Pointer(to=self.descriptor).bitcast[NoneType](),
                        mem_barrier.unsafe_ptr(),
                        Index(
                            coords[0] + j * copy_dim2,
                            coords[1] + i * copy_dim1,
                            coords[2] + m * copy_dim0,
                        ),
                        multicast_mask,
                    )

    @always_inline
    def async_multicast_load_4d[
        cta_group: Int = 1,
    ](
        self,
        dst: TileTensor[mut=True, Self.dtype, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int, Int, Int],
        multicast_mask: UInt16,
    ):
        """
        Schedules an asynchronous 4D multicast load from global to shared memory.

        TileTensor overload - accepts TileTensor instead of LayoutTensor.
        Assumes 128B alignment (TileTensor tiles are allocated with proper alignment).

        Parameters:
            cta_group: If issued with cta_group == 2, only the leader CTA needs
                to be notified upon completion.

        Args:
            dst: TileTensor in shared memory where data will be copied.
            mem_barrier: The memory barrier for synchronization.
            coords: The 4D coordinates in the source tensor from which to copy.
            multicast_mask: Bit mask specifying which CTAs should receive the data.
        """
        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_dim2 = Self.desc_shape[2]
        comptime copy_dim3 = Self.desc_shape[3]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = ceildiv(Self.tile_shape[0], copy_dim0)
        comptime num_copies_dim1 = ceildiv(Self.tile_shape[1], copy_dim1)
        comptime num_copies_dim2 = ceildiv(Self.tile_shape[2], copy_dim2)
        comptime num_copies_dim3 = ceildiv(Self.tile_shape[3], copy_dim3)
        comptime for n in range(num_copies_dim0):
            comptime for m in range(num_copies_dim1):
                comptime for i in range(num_copies_dim2):
                    comptime for j in range(num_copies_dim3):
                        comptime copy_offset: UInt32 = UInt32(
                            _desc_offset[
                                4,
                                Index(
                                    num_copies_dim0,
                                    num_copies_dim1,
                                    num_copies_dim2,
                                    num_copies_dim3,
                                ),
                                Self.is_k_major,
                            ](Index(n, m, i, j))
                            * copy_size
                        )

                        cp_async_bulk_tensor_shared_cluster_global_multicast[
                            cta_group=cta_group
                        ](
                            dst.ptr + copy_offset,
                            Pointer(to=self.descriptor).bitcast[NoneType](),
                            mem_barrier.unsafe_ptr(),
                            Index(
                                coords[0] + (j * copy_dim3),
                                coords[1] + (i * copy_dim2),
                                coords[2] + (m * copy_dim1),
                                coords[3] + (n * copy_dim0),
                            ),
                            multicast_mask,
                        )

    @always_inline
    def async_multicast_load_partitioned[
        tma_rows: Int,
        tma_load_size: Int,
    ](
        self,
        dst: LayoutTensor[
            mut=True,
            Self.dtype,
            _,
            address_space=.SHARED,
            alignment=128,
            ...,
        ],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        cta_rank: Int,
        coords: Tuple[Int, Int],
        multicast_mask: UInt16,
    ):
        """
        Performs a partitioned multicast load where each rank loads a distinct slice of data.

        This method is designed for clustered execution where different ranks (CTAs) load
        different, contiguous slices of the source tensor. Each rank's slice is offset
        by `cta_rank * tma_rows` in the second dimension and stored at offset `cta_rank * tma_load_size`
        in shared memory.

        Parameters:
            tma_rows: The number of rows each rank is responsible for loading.
            tma_load_size: The size in elements of each rank's slice in shared memory.

        Args:
            dst: The destination tensor in shared memory where data will be copied.
                Must be 128-byte aligned.
            mem_barrier: The memory barrier used to track and synchronize the asynchronous transfer.
            cta_rank: The rank ID (0-based) that determines which slice to load.
            coords: The base 2D coordinates in the source tensor from which to copy data.
                   The second coordinate will be offset by `cta_rank * tma_rows`.
            multicast_mask: A bit mask specifying which CTAs should receive the data.

        Note:
            This is typically used in matrix multiplication kernels where the input matrices
            are partitioned across multiple CTAs for parallel processing.
        """
        var dst_slice = LayoutTensor[
            Self.dtype,
            dst.layout,
            address_space=.SHARED,
            alignment=128,
        ](dst.ptr + cta_rank * tma_load_size)

        self.async_multicast_load(
            dst_slice,
            mem_barrier,
            (coords[0], coords[1] + cta_rank * tma_rows),
            multicast_mask,
        )

    @always_inline
    def async_multicast_load_partitioned[
        tma_rows: Int,
        tma_load_size: Int,
    ](
        self,
        dst: TileTensor[mut=True, Self.dtype, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        cta_rank: Int,
        coords: Tuple[Int, Int],
        multicast_mask: UInt16,
    ):
        """Perform a partitioned multicast load into a TileTensor.

        Each CTA rank loads a distinct contiguous slice of the source tensor.
        The source coordinate in the second dimension is offset by
        `cta_rank * tma_rows`, and the destination pointer is offset by
        `cta_rank * tma_load_size` elements.

        Parameters:
            tma_rows: Number of source rows loaded by each CTA rank.
            tma_load_size: Number of elements in each destination slice.

        Args:
            dst: Destination shared-memory TileTensor for the multicast load.
            mem_barrier: Shared-memory barrier that tracks transfer completion.
            cta_rank: CTA rank that selects the source and destination slice.
            coords: Base 2D coordinates in the source tensor.
            multicast_mask: Bit mask specifying CTAs that receive the data.
        """
        # `_offset_storage` yields an offset-derived engine; storages
        # are copy-compatible, so reinterpret it as `dst`'s own storage type.
        var dst_slice = type_of(dst)(
            rebind[type_of(dst._storage)](
                dst._offset_storage(cta_rank * tma_load_size)
            ),
            dst.layout,
        )

        self.async_multicast_load(
            dst_slice,
            mem_barrier,
            (coords[0], coords[1] + cta_rank * tma_rows),
            multicast_mask,
        )

    @always_inline
    def async_store(
        self,
        src: LayoutTensor[Self.dtype, _, address_space=.SHARED, ...],
        coords: Tuple[Int, Int],
    ):
        """
        Schedules an asynchronous store from shared memory to global memory.

        This method initiates a hardware-accelerated asynchronous transfer of data from shared memory
        to global memory at the specified coordinates.

        Args:
            src: LayoutTensor
                The source tensor in shared memory from which data will be copied.
                Must be 128-byte aligned.
            coords: The 2D coordinates in the destination tensor where data will be stored.

        Constraints:
            The source tensor must be 128-byte aligned in shared memory.
        """
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#table-alignment-multi-dim-tma
        comptime assert (
            type_of(src).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = Self.tile_shape[
            Int(not Self.is_k_major)
        ] // copy_dim0
        comptime num_copies_dim1 = Self.tile_shape[
            Int(Self.is_k_major)
        ] // copy_dim1

        comptime for i in range(num_copies_dim0):
            comptime for j in range(num_copies_dim1):
                comptime copy_offset: UInt32 = UInt32(
                    (i * num_copies_dim1 + j) * copy_size
                )

                cp_async_bulk_tensor_global_shared_cta(
                    src.ptr + copy_offset,
                    Pointer(to=self.descriptor).bitcast[NoneType](),
                    Index(
                        coords[0] + j * copy_dim1,
                        coords[1] + i * copy_dim0,
                    ),
                )

    @always_inline
    def async_store(
        self,
        src: TileTensor[Self.dtype, address_space=.SHARED, ...],
        coords: Tuple[Int, Int],
    ):
        """
        Schedules an asynchronous store from shared memory to global memory.

        TileTensor overload - accepts TileTensor instead of LayoutTensor.
        Assumes 128B alignment (TileTensor tiles are allocated with proper alignment).

        Args:
            src: TileTensor in shared memory from which data will be copied.
            coords: The 2D coordinates in the destination tensor where data will be stored.
        """
        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = Self.tile_shape[
            Int(not Self.is_k_major)
        ] // copy_dim0
        comptime num_copies_dim1 = Self.tile_shape[
            Int(Self.is_k_major)
        ] // copy_dim1

        comptime for i in range(num_copies_dim0):
            comptime for j in range(num_copies_dim1):
                comptime copy_offset: UInt32 = UInt32(
                    (i * num_copies_dim1 + j) * copy_size
                )

                cp_async_bulk_tensor_global_shared_cta(
                    src.ptr + copy_offset,
                    Pointer(to=self.descriptor).bitcast[NoneType](),
                    Index(
                        coords[0] + j * copy_dim1,
                        coords[1] + i * copy_dim0,
                    ),
                )

    @always_inline
    def async_store_3d(
        self,
        src: LayoutTensor[Self.dtype, _, address_space=.SHARED, ...],
        coords: Tuple[Int, Int, Int],
    ):
        """
        Schedules an asynchronous store from shared memory to global memory at specified 3D coordinates.

        This method initiates a hardware-accelerated asynchronous transfer of data from shared memory
        to the specified destination in global memory for 3D tensors.

        Args:
            src: The source tensor in shared memory from which data will be copied.
                 Must be 128-byte aligned.
            coords: The 3D coordinates in the destination tensor where data will be stored.

        Constraints:

            - The source tensor must be 128-byte aligned in shared memory.
            - The descriptor layout may be smaller than the shared memory tile shape
              to accommodate hardware requirements.
        """
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#table-alignment-multi-dim-tma
        comptime assert (
            type_of(src).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        # The descriptor layout i.e. data per copy can be smaller than the shared memory
        # tile shape due to WGMMA requirement. E.g. k-major no swizzle WGMMA BM x 16B to be
        # one continuous chunk in shared memory. We need to break down tile shape in K by 16B.
        #
        # dim0, dim1 are MN, K for K-major and K, MN for MN-major because our inputs are
        # row_major(K, MN) for the latter.
        #
        # TODO: use layout algebra here
        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_dim2 = Self.desc_shape[2]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = ceildiv(Self.tile_shape[0], copy_dim0)
        comptime num_copies_dim1 = ceildiv(Self.tile_shape[1], copy_dim1)
        comptime num_copies_dim2 = ceildiv(Self.tile_shape[2], copy_dim2)

        comptime for m in range(num_copies_dim0):
            comptime for i in range(num_copies_dim1):
                comptime for j in range(num_copies_dim2):
                    comptime copy_offset: UInt32 = UInt32(
                        _desc_offset[
                            3,
                            Index(
                                num_copies_dim0,
                                num_copies_dim1,
                                num_copies_dim2,
                            ),
                            Self.is_k_major,
                        ](Index(m, i, j))
                        * copy_size
                    )

                    cp_async_bulk_tensor_global_shared_cta(
                        src.ptr + copy_offset,
                        Pointer(to=self.descriptor).bitcast[NoneType](),
                        Index(
                            coords[0] + j * copy_dim2,
                            coords[1] + i * copy_dim1,
                            coords[2] + m * copy_dim0,
                        ),
                    )

    @always_inline
    def async_store_3d(
        self,
        src: TileTensor[Self.dtype, address_space=.SHARED, ...],
        coords: Tuple[Int, Int, Int],
    ):
        """
        Schedules an asynchronous store from shared memory to global memory at 3D coordinates.

        TileTensor overload - accepts TileTensor instead of LayoutTensor.
        Assumes 128B alignment (TileTensor tiles are allocated with proper alignment).

        Args:
            src: TileTensor in shared memory from which data will be copied.
            coords: The 3D coordinates in the destination tensor.
        """
        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_dim2 = Self.desc_shape[2]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = ceildiv(Self.tile_shape[0], copy_dim0)
        comptime num_copies_dim1 = ceildiv(Self.tile_shape[1], copy_dim1)
        comptime num_copies_dim2 = ceildiv(Self.tile_shape[2], copy_dim2)

        comptime for m in range(num_copies_dim0):
            comptime for i in range(num_copies_dim1):
                comptime for j in range(num_copies_dim2):
                    comptime copy_offset: UInt32 = UInt32(
                        _desc_offset[
                            3,
                            Index(
                                num_copies_dim0,
                                num_copies_dim1,
                                num_copies_dim2,
                            ),
                            Self.is_k_major,
                        ](Index(m, i, j))
                        * copy_size
                    )

                    cp_async_bulk_tensor_global_shared_cta(
                        src.ptr + copy_offset,
                        Pointer(to=self.descriptor).bitcast[NoneType](),
                        Index(
                            coords[0] + j * copy_dim2,
                            coords[1] + i * copy_dim1,
                            coords[2] + m * copy_dim0,
                        ),
                    )

    @always_inline
    def async_store_4d(
        self,
        src: LayoutTensor[Self.dtype, _, address_space=.SHARED, ...],
        coords: Tuple[Int, Int, Int, Int],
    ):
        """
        Schedules an asynchronous store from shared memory to global memory at specified 4D coordinates.

        This method initiates a hardware-accelerated asynchronous transfer of data from shared memory
        to the specified destination in global memory for 4D tensors.

        Args:
            src: The source tensor in shared memory from which data will be copied.
                 Must be 128-byte aligned.
            coords: The 4D coordinates in the destination tensor where data will be stored.

        Constraints:

            - The source tensor must be 128-byte aligned in shared memory.
            - The descriptor layout may be smaller than the shared memory tile shape
              to accommodate hardware requirements.
        """
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#table-alignment-multi-dim-tma
        comptime assert (
            type_of(src).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_dim2 = Self.desc_shape[2]
        comptime copy_dim3 = Self.desc_shape[3]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = ceildiv(Self.tile_shape[0], copy_dim0)
        comptime num_copies_dim1 = ceildiv(Self.tile_shape[1], copy_dim1)
        comptime num_copies_dim2 = ceildiv(Self.tile_shape[2], copy_dim2)
        comptime num_copies_dim3 = ceildiv(Self.tile_shape[3], copy_dim3)
        comptime for n in range(num_copies_dim0):
            comptime for m in range(num_copies_dim1):
                comptime for i in range(num_copies_dim2):
                    comptime for j in range(num_copies_dim3):
                        comptime copy_offset: UInt32 = UInt32(
                            _desc_offset[
                                4,
                                Index(
                                    num_copies_dim0,
                                    num_copies_dim1,
                                    num_copies_dim2,
                                    num_copies_dim3,
                                ),
                                Self.is_k_major,
                            ](Index(n, m, i, j))
                            * copy_size
                        )

                        cp_async_bulk_tensor_global_shared_cta(
                            src.ptr + copy_offset,
                            Pointer(to=self.descriptor).bitcast[NoneType](),
                            Index(
                                coords[0] + j * copy_dim3,
                                coords[1] + i * copy_dim2,
                                coords[2] + m * copy_dim1,
                                coords[3] + n * copy_dim0,
                            ),
                        )

    @always_inline
    def async_store_4d(
        self,
        src: TileTensor[Self.dtype, address_space=.SHARED, ...],
        coords: Tuple[Int, Int, Int, Int],
    ):
        """
        Schedules an asynchronous store from shared memory to global memory at 4D coordinates.

        TileTensor overload - accepts TileTensor instead of LayoutTensor.
        Assumes 128B alignment (TileTensor tiles are allocated with proper alignment).

        Args:
            src: TileTensor in shared memory from which data will be copied.
            coords: The 4D coordinates in the destination tensor.
        """
        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_dim2 = Self.desc_shape[2]
        comptime copy_dim3 = Self.desc_shape[3]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = ceildiv(Self.tile_shape[0], copy_dim0)
        comptime num_copies_dim1 = ceildiv(Self.tile_shape[1], copy_dim1)
        comptime num_copies_dim2 = ceildiv(Self.tile_shape[2], copy_dim2)
        comptime num_copies_dim3 = ceildiv(Self.tile_shape[3], copy_dim3)
        comptime for n in range(num_copies_dim0):
            comptime for m in range(num_copies_dim1):
                comptime for i in range(num_copies_dim2):
                    comptime for j in range(num_copies_dim3):
                        comptime copy_offset: UInt32 = UInt32(
                            _desc_offset[
                                4,
                                Index(
                                    num_copies_dim0,
                                    num_copies_dim1,
                                    num_copies_dim2,
                                    num_copies_dim3,
                                ),
                                Self.is_k_major,
                            ](Index(n, m, i, j))
                            * copy_size
                        )

                        cp_async_bulk_tensor_global_shared_cta(
                            src.ptr + copy_offset,
                            Pointer(to=self.descriptor).bitcast[NoneType](),
                            Index(
                                coords[0] + j * copy_dim3,
                                coords[1] + i * copy_dim2,
                                coords[2] + m * copy_dim1,
                                coords[3] + n * copy_dim0,
                            ),
                        )

    @always_inline
    def async_store_5d(
        self,
        src: LayoutTensor[Self.dtype, _, address_space=.SHARED, ...],
        coords: Tuple[Int, Int, Int, Int, Int],
    ):
        """
        Schedules an asynchronous store from shared memory to global memory at specified 5D coordinates.

        This method initiates a hardware-accelerated asynchronous transfer of data from shared memory
        to the specified destination in global memory for 5D tensors.

        Args:
            src: The source tensor in shared memory from which data will be copied.
                 Must be 128-byte aligned.
            coords: The 5D coordinates in the destination tensor where data will be stored.

        Constraints:

            - The source tensor must be 128-byte aligned in shared memory.
            - The descriptor layout may be smaller than the shared memory tile shape
              to accommodate hardware requirements.
        """
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#table-alignment-multi-dim-tma
        comptime assert (
            type_of(src).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_dim2 = Self.desc_shape[2]
        comptime copy_dim3 = Self.desc_shape[3]
        comptime copy_dim4 = Self.desc_shape[4]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = ceildiv(Self.tile_shape[0], copy_dim0)
        comptime num_copies_dim1 = ceildiv(Self.tile_shape[1], copy_dim1)
        comptime num_copies_dim2 = ceildiv(Self.tile_shape[2], copy_dim2)
        comptime num_copies_dim3 = ceildiv(Self.tile_shape[3], copy_dim3)
        comptime num_copies_dim4 = ceildiv(Self.tile_shape[4], copy_dim4)
        comptime for o in range(num_copies_dim0):
            comptime for n in range(num_copies_dim1):
                comptime for m in range(num_copies_dim2):
                    comptime for i in range(num_copies_dim3):
                        comptime for j in range(num_copies_dim4):
                            comptime copy_offset: UInt32 = UInt32(
                                _desc_offset[
                                    5,
                                    Index(
                                        num_copies_dim0,
                                        num_copies_dim1,
                                        num_copies_dim2,
                                        num_copies_dim3,
                                        num_copies_dim4,
                                    ),
                                    Self.is_k_major,
                                ](Index(o, n, m, i, j))
                                * copy_size
                            )

                            cp_async_bulk_tensor_global_shared_cta(
                                src.ptr + copy_offset,
                                Pointer(to=self.descriptor).bitcast[NoneType](),
                                Index(
                                    coords[0] + j * copy_dim4,
                                    coords[1] + i * copy_dim3,
                                    coords[2] + m * copy_dim2,
                                    coords[3] + n * copy_dim1,
                                    coords[4] + o * copy_dim0,
                                ),
                            )

    @always_inline
    def async_reduce[
        reduction_kind: ReduceOp
    ](
        self,
        src: LayoutTensor[Self.dtype, _, address_space=.SHARED, ...],
        coords: Tuple[Int, Int],
    ):
        """
        Schedules an asynchronous reduction operation from shared memory to global memory.

        This method initiates a hardware-accelerated asynchronous reduction operation that combines
        data from shared memory with data in global memory using the specified reduction operation.
        The reduction is performed element-wise at the specified coordinates in the global tensor.

        Parameters:
            reduction_kind: The type of reduction operation to perform (e.g., ADD, MIN, MAX).
                           This determines how values are combined during the reduction.

        Args:
            src: The source tensor in shared memory containing the data to be reduced.
                 Must be 128-byte aligned.
            coords: The 2D coordinates in the destination tensor where the reduction will be applied.

        Constraints:
            The source tensor must be 128-byte aligned in shared memory.
        """
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#table-alignment-multi-dim-tma
        comptime assert (
            type_of(src).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"
        cp_async_bulk_tensor_reduce_global_shared_cta[
            reduction_kind=reduction_kind
        ](
            src.ptr,
            Pointer(to=self.descriptor).bitcast[NoneType](),
            Index(coords[0], coords[1]),
        )

    @always_inline
    def commit_group(self):
        """
        Commits all prior initiated but uncommitted TMA instructions into a group.

        This function behaves the same as `cp_async_bulk_commit_group`, which creates
        a synchronization point for bulk TMA transfer.
        """
        cp_async_bulk_commit_group()

    @always_inline
    def wait_group[n: Int = 0](self):
        """
        Wait for the completion of asynchronous copy until a specified number of groups are waiting.

        This function behaves the same as `cp_async_bulk_wait_group`, which causes the executing
        thread to wait until a specified number of the most recent TMA copy are pending.

        Parameters:
            n: The number of pending groups left.
        """
        cp_async_bulk_wait_group[Int32(n)]()

    @always_inline
    def smem_tensormap_init(
        self,
        smem_tma_descriptor_ptr: MutPointer[
            TMADescriptor, _, address_space=.SHARED
        ],
    ):
        """
        Initializes a TMA descriptor in shared memory from this tensor tile's descriptor.

        This method copies the TMA descriptor from global memory to shared memory, allowing
        for faster access during kernel execution. The descriptor is copied in 16-byte chunks
        using asynchronous copy operations for efficiency.

        Args:
            smem_tma_descriptor_ptr: Pointer to the location in shared memory where the
                                    descriptor will be stored. Must be properly aligned.

        Note:

            - Only one thread should call this method to avoid race conditions
            - The descriptor is copied in 8 chunks of 16 bytes each (total 128 bytes)
        """
        # NOTE: Only one thread should call this

        var src_desc = (
            Pointer(to=self.descriptor)
            .bitcast[UInt8]()
            .address_space_cast[.GLOBAL]()
        )
        var dst_desc = smem_tma_descriptor_ptr.bitcast[UInt8]().unsafe_mut_cast[
            True
        ]()

        comptime simd_width = simd_width_of[DType.uint8]()
        comptime src_align = align_of[SIMD[.uint8, simd_width]]()
        comptime dst_align = align_of[SIMD[.uint8, simd_width]]()

        comptime descriptor_bytes = 128

        comptime for src_idx in range(descriptor_bytes // simd_width):
            var src_vec = (src_desc).load[
                width=simd_width, alignment=src_align
            ](src_idx * simd_width)
            dst_desc.store[alignment=dst_align](src_idx * simd_width, src_vec)

    @always_inline
    def replace_tensormap_global_address_in_gmem[
        _dtype: DType,
    ](self, src_ptr: ImmPointer[Scalar[_dtype], _]):
        """
        Replaces the global memory address in the TMA descriptor stored in global memory.

        This method allows dynamically changing the source tensor for TMA operations without
        recreating the entire descriptor, which is useful for reusing descriptors with different
        data sources. The operation modifies the descriptor in global memory directly.


        Parameters:
            _dtype: The data type of the new source tensor.

        Args:
            src_ptr: The new source tensor whose address will replace the current one in the descriptor.
                    Must have compatible layout with the original tensor.

        Note:
            A memory fence may be required after this operation to ensure visibility
            of the changes to other threads.
        """

        comptime assert src_ptr.address_space in (
            AddressSpace.GENERIC,
            AddressSpace.GLOBAL,
        ), "src address space must be GENERIC or GLOBAL."

        var desc_ptr = Pointer(to=self.descriptor).bitcast[NoneType]()

        inlined_assembly[
            "tensormap.replace.tile.global_address.global.b1024.b64 [$0], $1;",
            NoneType,
            constraints="l,l",
            has_side_effect=True,
        ](desc_ptr, src_ptr.bitcast[NoneType]())

    @always_inline
    def tensormap_fence_acquire(self):
        """
        Establishes a memory fence for TMA operations with acquire semantics.

        This method ensures proper ordering of memory operations by creating a barrier
        that prevents subsequent TMA operations from executing before prior operations
        have completed. It is particularly important when reading from a descriptor
        that might have been modified by other threads or processes.

        The acquire semantics ensure that all memory operations after this fence
        will observe any modifications made to the descriptor before the fence.

        Notes:

            - The entire warp must call this function as the instruction is warp-aligned.
            - Typically used in pairs with `tensormap_fence_release` for proper synchronization.
        """
        # NOTE: Entire warp must call this function as the instruction is aligned
        llvm_intrinsic[
            "llvm.nvvm.fence.proxy.tensormap_generic.acquire.gpu", NoneType
        ](
            Pointer(to=self.descriptor).bitcast[NoneType](),
            Int32(128),
        )

    @always_inline
    def tensormap_fence_release(self):
        """
        Establishes a memory fence for TMA operations with release semantics.

        This method ensures proper ordering of memory operations by creating a barrier
        that ensures all prior memory operations are visible before subsequent operations
        can proceed. It is particularly important when modifying a TMA descriptor in
        global memory that might be read by other threads or processes.

        The release semantics ensure that all memory operations before this fence
        will be visible to any thread that observes operations after the fence.

        Notes:

            - Typically used after modifying a tensormap descriptor in global memory.
            - Often paired with `tensormap_fence_acquire` for proper synchronization.
        """
        # This fence is needed when modifying tensormap directly in GMEM
        llvm_intrinsic[
            "llvm.nvvm.fence.proxy.tensormap_generic.release.gpu", NoneType
        ]()

    @always_inline
    def replace_tensormap_global_address_in_shared_mem[
        _dtype: DType,
    ](
        self,
        smem_tma_descriptor_ptr: MutPointer[
            TMADescriptor, _, address_space=.SHARED
        ],
        src_ptr: ImmPointer[Scalar[_dtype], _],
    ):
        """
        Replaces the global memory address in the TMA descriptor stored in shared memory.

        This method allows dynamically changing the source tensor for TMA operations without
        recreating the entire descriptor, which is useful for reusing descriptors with different
        data sources. The operation modifies a descriptor that has been previously copied to
        shared memory.


        Parameters:
            _dtype: The data type of the new source tensor.

        Args:
            smem_tma_descriptor_ptr: Pointer to the TMA descriptor in shared memory that will be modified.
            src_ptr: The new source tensor whose address will replace the current one in the descriptor.

        Notes:

            - Only one thread should call this method to avoid race conditions.
            - A memory fence may be required after this operation to ensure visibility
              of the changes to other threads.
            - Typically used with descriptors previously initialized with `smem_tensormap_init`.
        """

        comptime assert src_ptr.address_space in (
            AddressSpace.GENERIC,
            AddressSpace.GLOBAL,
        ), "src address space must be GENERIC or GLOBAL."

        # NOTE: Only one thread should call this
        inlined_assembly[
            (
                "tensormap.replace.tile.global_address.shared::cta.b1024.b64"
                " [$0], $1;"
            ),
            NoneType,
            constraints="r,l",
            has_side_effect=True,
        ](
            smem_tma_descriptor_ptr.bitcast[NoneType](),
            src_ptr.bitcast[NoneType](),
        )

    @always_inline
    def tensormap_cp_fence_release(
        self,
        smem_tma_descriptor_ptr: ImmPointer[
            TMADescriptor, _, address_space=.SHARED
        ],
    ):
        """
        Establishes a memory fence for TMA operations with release semantics for shared memory descriptors.

        This method ensures proper ordering of memory operations by creating a barrier
        that ensures all prior memory operations are visible before subsequent operations
        can proceed. It is specifically designed for synchronizing between global memory and
        shared memory TMA descriptors.

        The release semantics ensure that all memory operations before this fence
        will be visible to any thread that observes operations after the fence.

        Args:
            smem_tma_descriptor_ptr: Pointer to the TMA descriptor in shared memory that
                                    is being synchronized with the global memory descriptor.

        Notes:

            - The entire warp must call this function as the instruction is warp-aligned
            - Typically used after modifying a tensormap descriptor in shared memory
            - More specialized than the general `tensormap_fence_release` for cross-memory space synchronization
        """
        # This fence is needed when modifying tensormap directly in SMEM
        # NOTE: Entire warp must call this function as the instruction is aligned
        var gmem_tma_descriptor_ptr = Pointer(to=self.descriptor).bitcast[
            NoneType
        ]()

        inlined_assembly[
            (
                "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned"
                " [$0], [$1], 128;"
            ),
            NoneType,
            constraints="l,r",
            has_side_effect=True,
        ](gmem_tma_descriptor_ptr, smem_tma_descriptor_ptr.bitcast[NoneType]())

    @always_inline
    def replace_tensormap_global_dim_strides_in_shared_mem[
        _dtype: DType,
        only_update_dim_0: Bool,
        /,
        *,
        tensor_rank: Int,
    ](
        self,
        smem_tma_descriptor_ptr: MutPointer[
            TMADescriptor, address_space=.SHARED, ...
        ],
        gmem_dims: IndexList[tensor_rank],
        gmem_strides: IndexList[tensor_rank],
    ):
        """
        Replaces dimensions and strides in a TMA descriptor stored in shared memory.
        Note: This function is only supported for CUDA versions >= 12.5.

        This function allows dynamically modifying the dimensions and strides of a TMA
        descriptor that has been previously initialized in shared memory. If only the first dimension (dim 0) is updated, then updating strides can be skipped.

        Parameters:
            _dtype: The data type of the new source tensor.
            only_update_dim_0: If true, only the first dimension (dim 0) is updated with updating strides.
            tensor_rank: The rank of the tensor.

        Args:
            smem_tma_descriptor_ptr: Pointer to the TMA descriptor in shared memory that will be modified.
            gmem_dims: The global dimensions of the tensor to be updated.
            gmem_strides: The global strides of the tensor to be updated.

        Notes:
            - Only one thread should call this method to avoid race conditions.
            - A memory fence may be required after this operation to ensure visibility
            of the changes to other threads.
        """

        var desc_ptr = smem_tma_descriptor_ptr.bitcast[UInt64]()

        comptime if only_update_dim_0:
            comptime temp = "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [$0], " + String(
                tensor_rank - 1
            ) + ", $1;"
            inlined_assembly[
                temp,
                NoneType,
                constraints="l,r",
                has_side_effect=True,
            ](desc_ptr, gmem_dims[0])

        else:
            # Replace dimensions
            comptime for i in range(tensor_rank):
                comptime temp = "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [$0], " + String(
                    i
                ) + ", $1;"
                inlined_assembly[
                    temp,
                    NoneType,
                    constraints="l,r",
                    has_side_effect=True,
                ](desc_ptr, gmem_dims[tensor_rank - i - 1])

            # Replace strides - note: stride for innermost dimension is implicitly 1
            # For CUDA versions >= 12.5, we use the full stride value. Note that this is not true for all CUDA versions and strides should be left shifted by 4 for CUDA versions < 12.5
            comptime for i in range(1, tensor_rank):
                comptime temp = "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [$0], " + String(
                    i - 1
                ) + ", $1;"
                inlined_assembly[
                    temp,
                    NoneType,
                    constraints="l,l",
                    has_side_effect=True,
                ](
                    desc_ptr,
                    gmem_strides[tensor_rank - i - 1] * size_of[Self.dtype](),
                )

    @always_inline
    def replace_tensormap_global_dim_strides_in_shared_mem[
        _dtype: DType,
        tensor_rank: Int,
        dim_idx: Int,
    ](
        self,
        smem_tma_descriptor_ptr: MutPointer[
            TMADescriptor, address_space=.SHARED, ...
        ],
        dim_value: UInt32,
        dim_stride: Optional[UInt64] = None,
    ):
        """
        Replaces dimensions and strides in a TMA descriptor stored in shared memory.
        Note: This function is only supported for CUDA versions >= 12.5.
        This function allows dynamically modifying the dimensions and strides of a TMA
        descriptor that has been previously initialized in shared memory. If only the first dimension is updated, then updating strides can be skipped.

        Parameters:
            _dtype: The data type of the source tensor in GMEM.
            tensor_rank: The rank of the source tensor in GMEM.
            dim_idx: The index of the dimension to be updated in the TMA descriptor with the provided dimension and stride values at runtime.

        Args:
            smem_tma_descriptor_ptr: Pointer to the TMA descriptor in shared memory that will be modified.
            dim_value: The new dimension value to be set.
            dim_stride: The new stride value to be set.

        Notes:
            - Only one thread should call this method to avoid race conditions.
            - A memory fence may be required after this operation to ensure visibility
            of the changes to other threads.
        """

        var desc_ptr = smem_tma_descriptor_ptr.bitcast[UInt64]()

        # Replace dimensions

        comptime temp = "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [$0], " + String(
            tensor_rank - dim_idx - 1
        ) + ", $1;"
        inlined_assembly[
            temp,
            NoneType,
            constraints="l,r",
            has_side_effect=True,
        ](desc_ptr, dim_value)

        # Replace strides - note: stride for innermost dimension is implicitly 1
        # For CUDA versions >= 12.5, we use the full stride value. Note that this is not true for all CUDA versions and strides should be left shifted by 4 for CUDA versions < 12.5
        comptime if dim_idx > 0:
            assert (
                dim_stride is not None
            ), " dim_stride must be provided if dim_idx > 0"
            comptime temp = "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [$0], " + String(
                tensor_rank - dim_idx - 1
            ) + ", $1;"
            inlined_assembly[
                temp,
                NoneType,
                constraints="l,l",
                has_side_effect=True,
            ](desc_ptr, dim_stride)


@always_inline
def create_tma_tile[
    *tile_sizes: Int,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](ctx: DeviceContext, tensor: LayoutTensor) raises -> TMATensorTile[
    tensor.dtype,
    2,
    IndexList[2](tile_sizes[0], tile_sizes[1]),
]:
    """
    Creates a `TMATensorTile` with specified tile dimensions and swizzle mode.

    This function creates a hardware-accelerated Tensor Memory Access (TMA) descriptor
    for efficient asynchronous data transfers between global memory and shared memory.
    It configures the tile dimensions and memory access patterns based on the provided
    parameters.

    Parameters:
        tile_sizes: The dimensions of the tile to be transferred. For 2D tensors, this should be
            [height, width]. The dimensions determine the shape of data transferred in each
            TMA operation.
        swizzle_mode:
            The swizzling mode to use for memory access optimization. Swizzling can improve
            memory access patterns for specific hardware configurations.

    Args:
        ctx:
            The CUDA device context used to create the TMA descriptor.
        tensor:
            The source tensor from which data will be transferred. This defines the
            global memory layout and data type.

    Returns:
        A `TMATensorTile` configured with the specified tile dimensions and swizzle mode,
        ready for use in asynchronous data transfer operations.

    Constraints:

        - The last dimension's size in bytes must not exceed the swizzle mode's byte limit
          (32B for SWIZZLE_32B, 64B for SWIZZLE_64B, 128B for SWIZZLE_128B).
        - Only supports 2D tensors in this overload.

    Raises:
        If TMA descriptor creation fails.
    """
    # the last dimension of smem shape has to be smaller or equals to the
    # swizzle bytes.
    comptime swizzle_rows_bytes = tile_sizes[tensor.rank - 1] * size_of[
        tensor.dtype
    ]()

    comptime if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE:
        comptime assert swizzle_rows_bytes <= swizzle_mode.bytes(), (
            "Current swizzle bytes is "
            + String(swizzle_rows_bytes)
            + " which exceeds "
            + String(swizzle_mode.bytes())
            + "B swizzle requirement."
        )

    return create_tma_descriptor[tensor.dtype, 2, swizzle_mode](
        DeviceBuffer(
            ctx,
            tensor.ptr.unsafe_mut_cast[True]().address_space_cast[.GENERIC](),
            1,
            owning=False,
        ),
        (tensor.dim(0), tensor.dim(1)),
        (tensor.stride(0), tensor.stride(1)),
        (tile_sizes[0], tile_sizes[1]),
    )


@__parameter
def _gather4_box_width[
    dtype: DType,
    tile_width: Int,
    swizzle_mode: TensorMapSwizzle,
]() -> Int:
    """Computes the TMA box width for gather4 based on the swizzle mode.

    For SWIZZLE_NONE, the box width equals the tile width (no chunking).
    For swizzle modes (32B, 64B, 128B), the box width is
    ``swizzle_bytes // sizeof(dtype)``, so that each gather4 call loads one
    swizzle-group-sized chunk of the row.

    The caller iterates over column groups using
    ``_gather4_num_col_groups`` which uses ``ceildiv`` to handle
    non-divisible widths (TMA hardware zero-fills out-of-bounds elements).

    Parameters:
        dtype: Element data type.
        tile_width: Total number of elements per row in the global tensor.
        swizzle_mode: TMA swizzle mode.

    Returns:
        The box width (number of elements per gather4 call along the column
        dimension).
    """
    comptime if swizzle_mode == TensorMapSwizzle.SWIZZLE_NONE:
        return tile_width
    else:
        return swizzle_mode.bytes() // size_of[dtype]()


@__parameter
def _gather4_num_col_groups[
    dtype: DType,
    tile_width: Int,
    swizzle_mode: TensorMapSwizzle,
]() -> Int:
    """Returns the number of column groups for a gather4 load of a wide row.

    Each column group loads ``box_width`` elements. The last group may extend
    past the end of the row when ``tile_width`` is not a multiple of
    ``box_width``; the TMA hardware zero-fills the out-of-bounds elements.

    Parameters:
        dtype: Element data type.
        tile_width: Total number of elements per row in the global tensor.
        swizzle_mode: TMA swizzle mode.

    Returns:
        ``ceildiv(tile_width, box_width)`` where ``box_width`` comes
        from ``_gather4_box_width``.
    """
    comptime bw = _gather4_box_width[dtype, tile_width, swizzle_mode]()
    return ceildiv(tile_width, bw)


@always_inline
def create_tma_tile_gather4[
    dtype: DType,
    *,
    tile_height: Int = 4,
    tile_width: Int,
    tile_stride: Int = tile_width,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
](
    ctx: DeviceContext,
    device_buf: DeviceBuffer[dtype],
    num_rows: Int,
) raises -> TMATensorTile[
    dtype,
    2,
    tile_shape=IndexList[2](
        tile_height,
        _gather4_box_width[dtype, tile_width, swizzle_mode](),
    ),
    desc_shape=IndexList[2](
        1,
        _gather4_box_width[dtype, tile_width, swizzle_mode](),
    ),
]:
    """Creates a TMATensorTile for gather4 with automatic box-width computation.

    The global tensor has ``tile_width`` elements per row.  The TMA box
    width is derived from the swizzle mode so that each gather4 call loads one
    swizzle-group-sized column chunk (for SWIZZLE_NONE the box equals the full
    row).  The caller iterates over column groups using the ``col_idx``
    parameter of ``async_copy_gather4``::

        for cg in range(_gather4_num_col_groups[dtype, tile_width, swizzle_mode]()):
            tile.async_copy_gather4(dst, bar, col_idx=Int32(cg * box_width),
                                    row0, row1, row2, row3)

    Alternatively, use ``async_copy_gather4_tile`` to load the full
    ``tile_height``-row tile in one call (it loops over 4-row chunks and
    column groups internally).

    Parameters:
        dtype: The element data type.
        tile_height: Number of rows in the tile. Must be a multiple of 4.
            Defaults to 4 for backward compatibility.
        tile_width: Number of elements per row to load (box width).
        tile_stride: Row stride in elements in global memory. Defaults to
            ``tile_width``. Use a larger value when the row in global memory
            is wider than the portion to load (e.g. loading only nope from a
            nope+rope row).
        swizzle_mode: TMA swizzle mode.
        l2_promotion: L2 cache promotion hint for TMA loads. Defaults to NONE.

    Args:
        ctx: CUDA device context for TMA descriptor creation.
        device_buf: Device buffer containing the 2D row-major tensor data.
        num_rows: Total number of rows in the tensor.

    Returns:
        A TMATensorTile configured for gather4 with the appropriate box width.

    Raises:
        If TMA descriptor creation fails.
    """
    comptime assert tile_width > 0, "tile_width must be positive"
    comptime assert (
        tile_stride >= tile_width
    ), "tile_stride must be >= tile_width"
    comptime assert tile_height > 0, "tile_height must be positive"
    comptime assert tile_height % 4 == 0, "tile_height must be a multiple of 4"

    comptime box_w = _gather4_box_width[dtype, tile_width, swizzle_mode]()
    return create_tma_descriptor[dtype, 2, swizzle_mode, l2_promotion](
        device_buf,
        IndexList[2](num_rows, tile_stride),
        IndexList[2](tile_stride, 1),
        IndexList[2](1, box_w),
    )


@always_inline
def create_tma_tile_gather4[
    dtype: DType,
    *,
    tile_height: Int = 4,
    tile_width: Int,
    tile_stride: Int = tile_width,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    l2_promotion: TensorMapL2Promotion = TensorMapL2Promotion.NONE,
](
    ctx: DeviceContext,
    ptr: ImmPointer[Scalar[dtype], _],
    num_rows: Int,
) raises -> TMATensorTile[
    dtype,
    2,
    tile_shape=IndexList[2](
        tile_height,
        _gather4_box_width[dtype, tile_width, swizzle_mode](),
    ),
    desc_shape=IndexList[2](
        1,
        _gather4_box_width[dtype, tile_width, swizzle_mode](),
    ),
]:
    """Creates a TMATensorTile for gather4 from a raw pointer with automatic
    box-width computation.

    The TMA box width is derived from the swizzle mode. For SWIZZLE_NONE the
    box width equals ``tile_width``.

    Parameters:
        dtype: The element data type.
        tile_height: Number of rows in the tile. Must be a multiple of 4.
            Defaults to 4 for backward compatibility.
        tile_width: Number of elements per row to load (box width).
        tile_stride: Row stride in elements in global memory. Defaults to
            ``tile_width``. Use a larger value when the row in global memory
            is wider than the portion to load.
        swizzle_mode: TMA swizzle mode.
        l2_promotion: L2 cache promotion hint for TMA loads. Defaults to NONE.

    Args:
        ctx: CUDA device context for TMA descriptor creation.
        ptr: Raw device pointer to the 2D row-major tensor data.
        num_rows: Total number of rows in the tensor.

    Returns:
        A TMATensorTile configured for gather4 with the appropriate box width.

    Raises:
        If TMA descriptor creation fails.
    """
    comptime assert tile_width > 0, "tile_width must be positive"
    comptime assert (
        tile_stride >= tile_width
    ), "tile_stride must be >= tile_width"
    comptime assert tile_height > 0, "tile_height must be positive"
    comptime assert tile_height % 4 == 0, "tile_height must be a multiple of 4"

    comptime box_w = _gather4_box_width[dtype, tile_width, swizzle_mode]()
    return create_tma_descriptor[dtype, 2, swizzle_mode, l2_promotion](
        DeviceBuffer(
            ctx,
            ptr.address_space_cast[.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](num_rows, tile_stride),
        IndexList[2](tile_stride, 1),
        IndexList[2](1, box_w),
    )


@always_inline
def _create_tma_descriptor_helper[
    dtype: DType,
    rank: Int,
    //,
    desc_index_list: IndexList[rank],
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](
    ctx: DeviceContext,
    tensor: LayoutTensor[mut=True, dtype, address_space=.GENERIC, ...],
) raises -> TMADescriptor:
    """
    Helper function to create a TMA descriptor from a global memory layout tensor.

    This internal function creates a hardware-accelerated Tensor Memory Access (TMA) descriptor
    for efficient asynchronous data transfers between global memory and shared memory.
    It validates the tensor rank, flattens the layout shape and strides, and ensures
    swizzle mode compatibility with the tile dimensions.

    Parameters:
        dtype: The data type of the tensor elements.
        rank: The rank (number of dimensions) of the tensor.
        desc_index_list:
            The dimensions of the tile descriptor in each dimension. This defines the shape
            of data transferred in each TMA operation.
        swizzle_mode:
            The swizzling mode to use for memory access optimization. Swizzling can improve
            memory access patterns for specific hardware configurations. Defaults to SWIZZLE_NONE.

    Args:
        ctx:
            The CUDA device context used to create the TMA descriptor.
        tensor:
            The source layout tensor from which data will be transferred. This defines the
            global memory layout and data type.

    Returns:
        A `TMADescriptor` configured with the specified tile dimensions and swizzle mode,
        ready for use in asynchronous data transfer operations.

    Constraints:
        - The tensor rank must match the specified rank parameter.
        - When swizzling is enabled, the last dimension's size in bytes (calculated as
          `desc_index_list[rank-1] * sizeof(dtype)`) must not exceed the swizzle mode's
          byte limit (32B for SWIZZLE_32B, 64B for SWIZZLE_64B, 128B for SWIZZLE_128B).
    """

    comptime assert rank == tensor.rank, "Rank mismatch"

    var global_shape = coalesce_nested_tuple(tensor.runtime_layout.shape)
    var global_strides = coalesce_nested_tuple(tensor.runtime_layout.stride)

    comptime swizzle_rows_bytes = desc_index_list[rank - 1] * size_of[
        tensor.dtype
    ]()

    var global_shape_list = runtime_tuple_to_index_list[rank](global_shape)
    var global_strides_list = runtime_tuple_to_index_list[rank](global_strides)

    comptime if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE:
        comptime assert swizzle_rows_bytes <= swizzle_mode.bytes(), (
            "Current swizzle bytes is "
            + String(swizzle_rows_bytes)
            + " which exceeds "
            + String(swizzle_mode.bytes())
            + "B swizzle requirement."
        )

    return create_tma_descriptor[tensor.dtype, rank, swizzle_mode](
        DeviceBuffer(
            ctx,
            tensor.ptr,
            1,
            owning=False,
        ),
        global_shape_list,
        global_strides_list,
        desc_index_list,
    )


@always_inline
def create_tensor_tile[
    dtype: DType,
    rank: Int,
    //,
    tile_shape: IndexList[rank],
    /,
    k_major_tma: Bool = True,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    *,
    __tile_shape: IndexList[rank] = tile_shape,
    __desc_shape: IndexList[rank] = _default_desc_shape[
        rank, dtype, tile_shape, swizzle_mode
    ](),
](ctx: DeviceContext, tensor: LayoutTensor[dtype, ...]) raises -> TMATensorTile[
    dtype,
    rank,
    __tile_shape,
    __desc_shape,
    is_k_major=k_major_tma,
]:
    """
    Creates a `TMATensorTile` with advanced configuration options for 2D, 3D, 4D, or 5D tensors.

    This overload provides more control over the TMA descriptor creation, allowing
    specification of data type, rank, and layout orientation. It supports 2D, 3D, 4D, and 5D
    tensors and provides fine-grained control over the memory access patterns.

    Parameters:
        dtype: DType
            The data type of the tensor elements.
        rank: Int
            The dimensionality of the tensor (must be 2, 3, 4, or 5).
        tile_shape: IndexList[rank]
            The shape of the tile to be transferred.
        k_major_tma: Bool = True
            Whether the tma should copy desc into shared memory following a
            column-major (if `True`) or row-major (if `False`) pattern.
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE
            The swizzling mode to use for memory access optimization.
        __tile_shape: IndexList[rank] = tile_shape
            Internal parameter for the tile shape in shared memory.
        __desc_shape: IndexList[rank] = _default_desc_shape[...]()
            Internal parameter for the descriptor shape, which may differ from the
            tile shape to accommodate hardware requirements.

    Args:
        ctx: DeviceContext
            The CUDA device context used to create the TMA descriptor.
        tensor: LayoutTensor[dtype, ...]
            The source tensor from which data will be transferred. This defines the
            global memory layout and must match the specified data type.

    Returns:
        A `TMATensorTile` configured with the specified parameters, ready for use in
        asynchronous data transfer operations.

    Constraints:

        - Only supports 2D, 3D, 4D, and 5D tensors (rank must be 2, 3, 4, or 5).
        - For non-SWIZZLE_NONE modes, the K dimension size in bytes must be a multiple
          of the swizzle mode's byte size.
        - For MN-major layout, only SWIZZLE_128B is supported.
        - For 3D, 4D, and 5D tensors, only K-major layout is supported.

    Raises:
        If TMA descriptor creation fails.
    """
    # Current impl limitations
    comptime assert (
        rank == 2 or rank == 3 or rank == 4 or rank == 5
    ), "Only support 2D/3D/4D/5D TMA"

    comptime desc_bytes_size = _idx_product[rank, __desc_shape]() * size_of[
        dtype
    ]()
    comptime layout_size = _idx_product[rank, __tile_shape]() * size_of[dtype]()

    comptime if desc_bytes_size < layout_size:
        # When we do multiple TMA copy, every address has to be align to 128.
        comptime assert desc_bytes_size % 128 == 0, (
            "desc shape byte size has to be aligned to 128 bytes for"
            " multiple TMA copies. desc_shape: "
            + String(__desc_shape[0])
            + " "
            + String(__desc_shape[1])
            + " tile_shape: "
            + String(__tile_shape[0])
            + " "
            + String(__tile_shape[1])
        )

    comptime if rank == 2:
        comptime if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE:
            comptime assert (
                tile_shape[1] * size_of[dtype]()
            ) % swizzle_mode.bytes() == 0, (
                String(swizzle_mode)
                + " mode requires K dim multiple of "
                + String(swizzle_mode.bytes())
                + "B. K dim is now "
                + String(tile_shape[1] * size_of[dtype]())
                + " bytes, tile_shape[1] = "
                + String(tile_shape[1])
                + "\ndtype ="
                + String(dtype)
            )

        return create_tma_descriptor[dtype, 2, swizzle_mode](
            DeviceBuffer(
                ctx,
                tensor.ptr.address_space_cast[.GENERIC](),
                1,
                owning=False,
            ),
            (tensor.dim(0), tensor.dim(1)),
            (tensor.stride(0), tensor.stride(1)),
            (__desc_shape[0], __desc_shape[1]),
        )

    elif rank == 3:
        comptime if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE:
            comptime assert (
                tile_shape[2] * size_of[dtype]()
            ) % swizzle_mode.bytes() == 0, (
                String(swizzle_mode)
                + " mode requires K dim multiple of "
                + String(swizzle_mode.bytes())
                + "B. K dim is now "
                + String(tile_shape[2] * size_of[dtype]())
                + "bytes."
            )

        return create_tma_descriptor[dtype, 3, swizzle_mode](
            DeviceBuffer(
                ctx,
                tensor.ptr.address_space_cast[.GENERIC](),
                1,
                owning=False,
            ),
            IndexList[3](tensor.dim(0), tensor.dim(1), tensor.dim(2)),
            IndexList[3](tensor.stride(0), tensor.stride(1), tensor.stride(2)),
            IndexList[3](
                __desc_shape[0],
                __desc_shape[1],
                __desc_shape[2],
            ),
        )

    elif rank == 4:
        comptime if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE:
            comptime assert (
                tile_shape[3] * size_of[dtype]()
            ) % swizzle_mode.bytes() == 0, (
                String(swizzle_mode)
                + " mode requires K dim multiple of "
                + String(swizzle_mode.bytes())
                + "B. K dim is now "
                + String(tile_shape[3] * size_of[dtype]())
                + "bytes."
            )

        return create_tma_descriptor[dtype, 4, swizzle_mode](
            DeviceBuffer(
                ctx,
                tensor.ptr.address_space_cast[.GENERIC](),
                1,
                owning=False,
            ),
            IndexList[4](
                tensor.dim(0), tensor.dim(1), tensor.dim(2), tensor.dim(3)
            ),
            IndexList[4](
                tensor.stride(0),
                tensor.stride(1),
                tensor.stride(2),
                tensor.stride(3),
            ),
            IndexList[4](
                __desc_shape[0],
                __desc_shape[1],
                __desc_shape[2],
                __desc_shape[3],
            ),
        )

    else:  # rank == 5
        comptime if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE:
            comptime assert (
                tile_shape[4] * size_of[dtype]()
            ) % swizzle_mode.bytes() == 0, (
                String(swizzle_mode)
                + " mode requires K dim multiple of "
                + String(swizzle_mode.bytes())
                + "B. K dim is now "
                + String(tile_shape[4] * size_of[dtype]())
                + "bytes."
            )

        return create_tma_descriptor[dtype, 5, swizzle_mode](
            DeviceBuffer(
                ctx,
                tensor.ptr.address_space_cast[.GENERIC](),
                1,
                owning=False,
            ),
            IndexList[5](
                tensor.dim(0),
                tensor.dim(1),
                tensor.dim(2),
                tensor.dim(3),
                tensor.dim(4),
            ),
            IndexList[5](
                tensor.stride(0),
                tensor.stride(1),
                tensor.stride(2),
                tensor.stride(3),
                tensor.stride(4),
            ),
            IndexList[5](
                __desc_shape[0],
                __desc_shape[1],
                __desc_shape[2],
                __desc_shape[3],
                __desc_shape[4],
            ),
        )


@always_inline
def create_tensor_tile[
    dtype: DType,
    rank: Int,
    //,
    tile_shape: IndexList[rank],
    /,
    k_major_tma: Bool = True,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    *,
    __tile_shape: IndexList[rank] = tile_shape,
    __desc_shape: IndexList[rank] = _default_desc_shape[
        rank, dtype, tile_shape, swizzle_mode
    ](),
    unpack_fp4: Bool = False,
](ctx: DeviceContext, tensor: TileTensor[dtype, ...]) raises -> TMATensorTile[
    dtype,
    rank,
    __tile_shape,
    __desc_shape,
    is_k_major=k_major_tma,
]:
    """
    Creates a `TMATensorTile` from a TileTensor.

    This overload accepts a TileTensor instead of LayoutTensor, enabling use
    with the new coordinate-based tensor abstraction.

    Parameters:
        dtype: The data type of the tensor elements.
        rank: The dimensionality of the tensor (must be 2, 3, 4, or 5).
        tile_shape: The shape of the tile to be transferred.
        k_major_tma: Whether the TMA should use column-major pattern.
        swizzle_mode: The swizzling mode for memory access optimization.
        __tile_shape: Internal parameter for the tile shape.
        __desc_shape: Internal parameter for the descriptor shape.
        unpack_fp4: When True, `tensor` holds nibble-packed E2M1 as `uint8`
            and the copy pads it into shared memory so a K extent spans one
            byte per element (the values themselves stay nibble-packed; see
            `PACKED_FP4_ALIGN16B`). The tile and descriptor shapes are then
            counted in FP4 elements, so they are twice the tensor's innermost
            extent per tile.

    Args:
        ctx: The CUDA device context.
        tensor: The source TileTensor.

    Returns:
        A `TMATensorTile` configured for the given tensor.

    Raises:
        If TMA descriptor creation fails.
    """
    comptime assert rank in (2, 3, 4, 5), "Only support 2D/3D/4D/5D TMA"

    comptime desc_bytes_size = _idx_product[rank, __desc_shape]() * size_of[
        dtype
    ]()
    comptime layout_size = _idx_product[rank, __tile_shape]() * size_of[dtype]()

    comptime if desc_bytes_size < layout_size:
        comptime assert desc_bytes_size % 128 == 0, (
            "desc shape byte size has to be aligned to 128 bytes for"
            " multiple TMA copies."
        )

    # Swizzle constraint applies to all ranks - check once here. A padded FP4
    # tile spans one shared-memory byte per element, so its shape already
    # counts shared-memory bytes and needs no element-size scaling.
    comptime if swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE:
        comptime tile_smem_bytes = tile_shape[rank - 1] * (
            1 if unpack_fp4 else size_of[dtype]()
        )
        comptime assert tile_smem_bytes % swizzle_mode.bytes() == 0, (
            String(swizzle_mode)
            + " mode requires K dim multiple of "
            + String(swizzle_mode.bytes())
            + "B."
        )

    comptime assert (
        rank == 2 or not unpack_fp4
    ), "packed FP4 TMA is only wired for rank 2"

    comptime if rank == 2:
        # The innermost extent reaches the descriptor in FP4 elements, which
        # is twice what the `uint8` view spells. Strides stay in `uint8`.
        return create_tma_descriptor[
            dtype, 2, swizzle_mode, unpack_fp4=unpack_fp4
        ](
            DeviceBuffer(
                ctx,
                tensor.ptr.address_space_cast[.GENERIC](),
                1,
                owning=False,
            ),
            (
                Int(tensor.layout.shape[0]().value()),
                Int(tensor.layout.shape[1]().value())
                * (2 if unpack_fp4 else 1),
            ),
            (
                Int(tensor.layout.stride[0]().value()),
                Int(tensor.layout.stride[1]().value()),
            ),
            (__desc_shape[0], __desc_shape[1]),
        )

    elif rank == 3:
        return create_tma_descriptor[dtype, 3, swizzle_mode](
            DeviceBuffer(
                ctx,
                tensor.ptr.address_space_cast[.GENERIC](),
                1,
                owning=False,
            ),
            IndexList[3](
                Int(tensor.layout.shape[0]().value()),
                Int(tensor.layout.shape[1]().value()),
                Int(tensor.layout.shape[2]().value()),
            ),
            IndexList[3](
                Int(tensor.layout.stride[0]().value()),
                Int(tensor.layout.stride[1]().value()),
                Int(tensor.layout.stride[2]().value()),
            ),
            IndexList[3](
                __desc_shape[0],
                __desc_shape[1],
                __desc_shape[2],
            ),
        )

    elif rank == 4:
        return create_tma_descriptor[dtype, 4, swizzle_mode](
            DeviceBuffer(
                ctx,
                tensor.ptr.address_space_cast[.GENERIC](),
                1,
                owning=False,
            ),
            IndexList[4](
                Int(tensor.layout.shape[0]().value()),
                Int(tensor.layout.shape[1]().value()),
                Int(tensor.layout.shape[2]().value()),
                Int(tensor.layout.shape[3]().value()),
            ),
            IndexList[4](
                Int(tensor.layout.stride[0]().value()),
                Int(tensor.layout.stride[1]().value()),
                Int(tensor.layout.stride[2]().value()),
                Int(tensor.layout.stride[3]().value()),
            ),
            IndexList[4](
                __desc_shape[0],
                __desc_shape[1],
                __desc_shape[2],
                __desc_shape[3],
            ),
        )

    else:  # rank == 5
        return create_tma_descriptor[dtype, 5, swizzle_mode](
            DeviceBuffer(
                ctx,
                tensor.ptr.address_space_cast[.GENERIC](),
                1,
                owning=False,
            ),
            IndexList[5](
                Int(tensor.layout.shape[0]().value()),
                Int(tensor.layout.shape[1]().value()),
                Int(tensor.layout.shape[2]().value()),
                Int(tensor.layout.shape[3]().value()),
                Int(tensor.layout.shape[4]().value()),
            ),
            IndexList[5](
                Int(tensor.layout.stride[0]().value()),
                Int(tensor.layout.stride[1]().value()),
                Int(tensor.layout.stride[2]().value()),
                Int(tensor.layout.stride[3]().value()),
                Int(tensor.layout.stride[4]().value()),
            ),
            IndexList[5](
                __desc_shape[0],
                __desc_shape[1],
                __desc_shape[2],
                __desc_shape[3],
                __desc_shape[4],
            ),
        )


def _padded_shape[
    rank: Int,
    dtype: DType,
    tile_shape: IndexList[rank],
    swizzle_mode: TensorMapSwizzle,
]() -> IndexList[rank]:
    """Compute the padded tile shape for SplitLastDimTMATensorTile."""
    comptime swizzle_granularity = swizzle_mode.bytes() // size_of[dtype]()
    comptime final_dim = tile_shape[rank - 1]
    comptime num_tma = ceildiv(final_dim, swizzle_granularity)
    var result: IndexList[rank] = {}
    comptime for i in range(rank - 1):
        result[i] = tile_shape[i]
    result[rank - 1] = num_tma * swizzle_granularity
    return result


def _ragged_shape[
    rank: Int,
    dtype: DType,
    tile_shape: IndexList[rank],
    swizzle_mode: TensorMapSwizzle,
]() -> IndexList[rank]:
    """Compute the ragged descriptor shape for SplitLastDimTMATensorTile."""
    comptime swizzle_granularity = swizzle_mode.bytes() // size_of[dtype]()
    var result: IndexList[rank] = {}
    comptime for i in range(rank - 1):
        comptime if tile_shape[i] != 1:
            result[i] = tile_shape[i]
        else:
            result[i] = 1
    result[rank - 1] = swizzle_granularity
    return result


comptime SplitLastDimTMATensorTile[
    rank: Int,
    //,
    dtype: DType,
    smem_shape: IndexList[rank],
    swizzle_mode: TensorMapSwizzle,
] = TMATensorTile[
    dtype,
    rank,
    _padded_shape[rank, dtype, smem_shape, swizzle_mode](),
    _ragged_shape[rank, dtype, smem_shape, swizzle_mode](),
]
"""A specialized TMA tensor tile type alias that handles layouts where the last
dimension is split based on swizzle granularity for optimal memory access patterns.
The current behavior is to not actually split the last dimension.

Parameters:
    rank: The number of dimensions of the tensor.
    dtype: The data type of the tensor elements.
    smem_shape: The shape of the tile in shared memory. The last dimension will be
        padded if necessary to align with the swizzle granularity.
    swizzle_mode: The swizzling mode for memory access optimization. Determines
        the granularity at which the last dimension is split or padded.
"""


@always_inline
def _split_tma_gmem_tensor[
    dtype: DType,
    rank: Int,
    //,
    shape: IndexList[rank],
    swizzle_mode: TensorMapSwizzle,
](
    ptr: Pointer[Scalar[dtype], _],
    dim0: Int,
    out ret: LayoutTensor[
        dtype,
        Layout.row_major(shape),
        ptr.origin,
    ],
):
    comptime split_rank = len(flatten(ret.layout.shape))
    var runtime_shape: IndexList[split_rank] = {}
    runtime_shape[0] = dim0

    comptime for i in range(1, split_rank):
        comptime dim_i: Int = ret.layout.shape[i].value()
        runtime_shape[i] = dim_i
    ret = {ptr, RuntimeLayout[ret.layout].row_major(runtime_shape)}


@always_inline
def _split_tma_gmem_tensor[
    dtype: DType,
    rank: Int,
    //,
    shape: IndexList[rank],
    swizzle_mode: TensorMapSwizzle,
](
    ptr: Pointer[Scalar[dtype], _],
    dim0: Int,
    dim1: Int,
    out ret: LayoutTensor[
        dtype,
        Layout.row_major(shape),
        ptr.origin,
    ],
):
    comptime swizzle_granularity = swizzle_mode.bytes() // size_of[dtype]()
    var runtime_shape: IndexList[rank] = {}
    runtime_shape[0] = dim0
    runtime_shape[1] = dim1

    comptime for i in range(2, rank):
        runtime_shape[i] = shape[i]

    comptime assert rank == len(flatten(ret.layout.shape)), (
        "rank = " + String(rank) + "\nlayout = " + String(ret.layout)
    )
    ret = {ptr, RuntimeLayout[ret.layout].row_major(runtime_shape)}


def _create_split_tma_folded[
    dtype: DType,
    rank: Int,
    //,
    smem_shape: IndexList[rank],
    gmem_shape: IndexList[rank],
    swizzle_mode: TensorMapSwizzle,
    fold_chunks: Int,
    row_major: Bool = False,
](
    ctx: DeviceContext,
    ptr: ImmPointer[Scalar[dtype], _],
    runtime_rows: Int,
    num_heads: Int,
    out res: SplitLastDimTMATensorTile[
        dtype,
        smem_shape,
        swizzle_mode,
    ],
) raises:
    """Builds the depth-chunk-folded K/V TMA descriptor (SM100 / B200).

    Shared by both `create_split_tma` overloads. `smem_shape` is the rank-3 K view
    `[box_rows, 1, BK]` and `gmem_shape` is `[rows, num_heads, head_size]`
    (`gmem_shape[2]` = head_size). `num_heads` is supplied at runtime by the caller
    (it may be a static or dynamic value depending on the overload). See
    `create_split_tma` for the byte-equivalence contract.

    `row_major=False` (default) builds the rank-4 **chunk-outer** box
    `[gran, box_rows, fold_chunks, 1]` (CUDA fast->slow): all `box_rows` of chunk 0,
    then chunk 1, ..., byte-equivalent to the per-chunk loop only for a single page
    (`box_rows == smem_j_stride_rows`, `pages_per_iter == 1`).

    `row_major=True` builds the rank-5 **chunk-inner** (row-major-atoms) box
    `[gran, CM, fold_chunks, box_rows/CM, 1]` (CUDA fast->slow): it splits `box_rows`
    into `(box_rows/CM)` atom-rows of `CM` rows each and nests the `fold_chunks`
    chunk axis BETWEEN the atom-row axis and the in-atom-row (`CM`) axis, so one TMA
    writes a whole multi-atom-row page in chunk-inner SMEM order
    `off(ar,c) = ar*(num_chunks*CM*gran) + c*(CM*gran)`. This is the layout that lets
    `page_size < BN` tiles fold to one TMA per page. Validated standalone by
    `max/kernels/test/gpu/kv_cache/test_kv_rowmajor_fold_spike.mojo`.
    """
    comptime assert fold_chunks >= 2, "folded builder needs fold_chunks >= 2"
    comptime assert rank == 3, "folded builder expects the rank-3 K view"
    comptime gran = swizzle_mode.bytes() // size_of[dtype]()
    comptime BK = smem_shape[2]
    comptime box_rows = smem_shape[0]
    comptime head_size = gmem_shape[2]
    comptime assert (
        gran * size_of[dtype]() == swizzle_mode.bytes()
    ), "swizzled innermost box must be exactly one swizzle atom"
    comptime assert (
        fold_chunks * gran == BK
    ), "fold_chunks * swizzle_granularity must equal BK"
    comptime assert (
        head_size % gran == 0
    ), "head_size must be a multiple of swizzle granularity"
    # The depth (head_size) axis is presented to the descriptor reshaped as
    # [num_depth_dim, gran] with num_depth_dim = head_size // gran. The chunk
    # (num_depth_dim) axis spans the FULL head_size so every per-stage window
    # `depth_offset` (= qk_stage * BK) is in-bounds; the BOX covers only
    # `fold_chunks` of those chunks per issue (one stage's BK worth of depth).
    comptime num_depth_dim = head_size // gran
    # rebind-free: the rank-4/rank-5 descriptor blob is wrapped into the rank-3
    # `SplitLastDimTMATensorTile` via `TMATensorTile.__init__(descriptor)`.
    # `TMADescriptor` is a fixed opaque 128 B blob independent of rank, so no
    # cross-rank `rebind` of the tile type is needed (and would be illegal:
    # rank-3/4/5 `TMATensorTile` are distinct nominal types).
    var device_buf = DeviceBuffer(
        ctx,
        ptr.address_space_cast[.GENERIC](),
        1,
        owning=False,
    )
    comptime if row_major:
        # Rank-5 chunk-inner (row-major-atoms) box. `CM` is the swizzle-atom /
        # core-matrix row count (== `_CM_NUM_ROWS` in tensor_core_async.mojo,
        # module-private there; the SWIZZLE_128B 8-row swizzle tile is exactly one
        # atom). Repo order (slowest-first) -> CUDA fast->slow box
        # [gran, CM, fold_chunks, box_rows/CM, 1]: the chunk axis (extent
        # fold_chunks, globalDim num_depth_dim) is nested BETWEEN the atom-row axis
        # (box_rows/CM) and the in-atom-row axis (CM), giving chunk-inner SMEM
        # order. The row axis is split (atom_row stride = CM*num_heads*head_size,
        # in-atom-row stride = num_heads*head_size); chunk stride stays `gran` so
        # the issue-site coord sets chunk-base = depth_offset // gran. Validated by
        # test_kv_rowmajor_fold_spike.mojo.
        comptime CM = _SWIZZLE_ATOM_ROWS
        comptime assert box_rows % CM == 0, (
            "row_major fold: box_rows must be a multiple of the swizzle-atom"
            " rows"
        )
        var desc = create_tma_descriptor[dtype, 5, swizzle_mode](
            device_buf,
            IndexList[5](
                num_heads, runtime_rows // CM, num_depth_dim, CM, gran
            ),
            IndexList[5](
                head_size,
                CM * num_heads * head_size,
                gran,
                num_heads * head_size,
                1,
            ),
            IndexList[5](1, box_rows // CM, fold_chunks, CM, gran),
        )
        res = SplitLastDimTMATensorTile[dtype, smem_shape, swizzle_mode](desc)
    else:
        # Rank-4 chunk-outer box (today's default). Repo order (slowest-first);
        # `create_tma_descriptor` reverses into CUDA order at tma.mojo:383-388 ->
        # CUDA boxDim[0]=gran (swizzled, 128 B), boxDim[3]=chunk (box extent =
        # fold_chunks, globalDim = num_depth_dim). The chunk axis stride is gran, so
        # chunk c covers depth elements [c*gran, c*gran+gran); the issue-site coord
        # sets chunk-base = depth_offset // gran and gran coord = 0.
        var desc = create_tma_descriptor[dtype, 4, swizzle_mode](
            device_buf,
            IndexList[4](num_heads, num_depth_dim, runtime_rows, gran),
            IndexList[4](head_size, gran, num_heads * head_size, 1),
            IndexList[4](1, fold_chunks, box_rows, gran),
        )
        res = SplitLastDimTMATensorTile[dtype, smem_shape, swizzle_mode](desc)


def create_split_tma[
    rank: Int,
    dtype: DType,
    //,
    smem_shape: IndexList[rank],
    gmem_shape: IndexList[rank],
    swizzle_mode: TensorMapSwizzle,
    fold_chunks: Int = 1,
    row_major: Bool = False,
](
    ctx: DeviceContext,
    ptr: ImmPointer[Scalar[dtype], _],
    runtime_dim0: Int,
    out res: SplitLastDimTMATensorTile[
        dtype,
        smem_shape,
        swizzle_mode,
    ],
) raises:
    """Creates a TMA tensor tile assuming that the first dimension in global memory has `UNKNOWN_VALUE`.

    This function creates a `TMATensorTile` that optionally splits the last dimension
    of the tensor into multiples of swizzle granularity. This functionality is currently
    disabled because it was not found to improve performance.

    When `fold_chunks >= 2`, the contiguous depth chunks are folded into a single
    rank-4/rank-5 TMA (see the 2-runtime-dim overload's docstring and
    `_create_split_tma_folded`). This overload is used by the cache-backed builders
    where `num_heads` is the static `gmem_shape[1]`.

    Parameters:
        rank: The number of dimensions of the tensor.
        dtype: The data type of the tensor elements.
        smem_shape: The shape of the tile in shared memory.
        gmem_shape: The shape of the global memory tensor.
        swizzle_mode: The swizzling mode for memory access optimization.
        fold_chunks: Number of depth chunks to fold into one rank-4 TMA (`1` =
            original 3D behavior).
        row_major: When `True` (and `fold_chunks >= 2`), build the rank-5
            chunk-inner (row-major-atoms) box so one TMA writes a whole
            multi-atom-row page; `False` (default) keeps the rank-4 chunk-outer box.

    Args:
        ctx: The CUDA device context used to create the TMA descriptor.
        ptr: Pointer to the global memory tensor data.
        runtime_dim0: The runtime size of the first dimension of the global tensor.

    Returns:
        The resulting TMA tensor tile with split layout.

    Raises:
        If TMA descriptor creation fails.
    """
    comptime if fold_chunks >= 2:
        comptime assert rank == 3, "fold path expects the rank-3 K view"
        # num_heads is the static second gmem dim for the cache-backed builders.
        res = _create_split_tma_folded[
            smem_shape, gmem_shape, swizzle_mode, fold_chunks, row_major
        ](ctx, ptr, runtime_dim0, gmem_shape[1])
    else:
        var tensor = _split_tma_gmem_tensor[gmem_shape, swizzle_mode](
            ptr, runtime_dim0
        )
        res = create_tensor_tile[
            res.tile_shape,
            swizzle_mode=swizzle_mode,
            __tile_shape=res.tile_shape,
            __desc_shape=res.desc_shape,
        ](ctx, tensor)


def create_split_tma[
    rank: Int,
    dtype: DType,
    //,
    smem_shape: IndexList[rank],
    gmem_shape: IndexList[rank],
    swizzle_mode: TensorMapSwizzle,
    fold_chunks: Int = 1,
    row_major: Bool = False,
](
    ctx: DeviceContext,
    ptr: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    runtime_dim0: Int,
    runtime_dim1: Int,
    out res: SplitLastDimTMATensorTile[
        dtype,
        smem_shape,
        swizzle_mode,
    ],
) raises:
    """Creates a TMA tensor tile assuming that the first two dimensions in global memory has `UNKNOWN_VALUE`.

    This function creates a `TMATensorTile` that optionally splits the last dimension
    of the tensor into multiples of swizzle granularity. This functionality is currently
    disabled because it was not found to improve performance.

    When `fold_chunks >= 2`, the contiguous innermost (depth) dimension, which the
    swizzle hardware forces to be split into `swizzle_granularity`-sized chunks, is
    folded into an extra, non-innermost box dimension so that a *single* rank-4
    `cp.async.bulk.tensor` copies all `fold_chunks` depth chunks at once instead of one
    TMA per chunk. The PUBLIC return type stays rank-3 (`SplitLastDimTMATensorTile`);
    the rank-4 CUDA descriptor is built internally and its opaque 128 B
    `TMADescriptor` blob (which is rank-agnostic; see `TMADescriptor`) is wrapped into
    the rank-3 tile via its `@implicit` constructor. The issue site
    (`PagedRowIndices._tma_copy_kv_impl`) must agree by issuing rank-4 coords; the
    shared `kv_tma_fold_chunks` predicate is the single source of truth that keeps the
    baked rank and the issue rank from drifting. `fold_chunks == 1` reproduces exactly
    the original 3D behavior.

    Folding is byte-equivalent to the per-chunk loop ONLY when the box's per-chunk SMEM
    stride (`box_rows * swizzle_granularity`) equals the consumer/producer chunk stride
    (`smem_j_stride_rows * swizzle_granularity`), i.e. `box_rows == smem_j_stride_rows`,
    and the tile occupies a single page (`pages_per_iter == 1`). The caller is
    responsible for only passing `fold_chunks >= 2` when those hold; here `box_rows`
    equals `smem_shape[0]`.

    Parameters:
        rank: The number of dimensions of the tensor.
        dtype: The data type of the tensor elements.
        smem_shape: The shape of the tile in shared memory.
        gmem_shape: The shape of the global memory tensor.
        swizzle_mode: The swizzling mode for memory access optimization.
        fold_chunks: Number of depth chunks to fold into one rank-4 TMA. `1`
            (default) is the original per-chunk 3D behavior; `>= 2` builds a rank-4
            descriptor.
        row_major: When `True` (and `fold_chunks >= 2`), build the rank-5
            chunk-inner (row-major-atoms) box so one TMA writes a whole
            multi-atom-row page; `False` (default) keeps the rank-4 chunk-outer box.

    Args:
        ctx: The CUDA device context used to create the TMA descriptor.
        ptr: Pointer to the global memory tensor data.
        runtime_dim0: The runtime size of the first dimension of the global tensor.
        runtime_dim1: The runtime size of the second dimension of the global tensor.

    Returns:
        The resulting TMA tensor tile with split layout.

    Raises:
        If TMA descriptor creation fails.
    """
    comptime if fold_chunks >= 2:
        # SM100 (B200) rank-4 depth-chunk fold. `gmem_shape` is the rank-3 view
        # `[rows, num_heads, head_size]` (`gmem_shape[0]`/`[1]` are UNKNOWN,
        # `gmem_shape[2]` = head_size); `smem_shape` is `[box_rows, 1, BK]`.
        # `num_heads` is the runtime second gmem dim here.
        comptime assert rank == 3, "fold path expects the rank-3 K view"
        res = _create_split_tma_folded[
            smem_shape, gmem_shape, swizzle_mode, fold_chunks, row_major
        ](ctx, ptr, runtime_dim0, runtime_dim1)
    else:
        var tensor = _split_tma_gmem_tensor[gmem_shape, swizzle_mode](
            ptr, runtime_dim0, runtime_dim1
        )
        res = create_tensor_tile[
            res.tile_shape,
            swizzle_mode=swizzle_mode,
            __tile_shape=res.tile_shape,
            __desc_shape=res.desc_shape,
        ](ctx, tensor)


@always_inline
def create_tma_tile_template[
    dtype: DType,
    rank: Int,
    tile_shape: IndexList[rank],
    /,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    *,
    __tile_shape: IndexList[rank] = tile_shape,
    __desc_shape: IndexList[rank] = _default_desc_shape[
        rank, dtype, tile_shape, swizzle_mode
    ](),
]() raises -> TMATensorTile[dtype, rank, __tile_shape, __desc_shape]:
    """
    Same as create_tma_tile expect the descriptor is only a placeholder or a template for later replacement.

    specification of data type, rank, and layout orientation. It supports both 2D and 3D
    tensors and provides fine-grained control over the memory access patterns.

    Parameters:
        dtype: DType
            The data type of the tensor elements.
        rank: Int
            The dimensionality of the tensor (must be 2 or 3).
        tile_shape: IndexList[rank]
            The shape of the tile to be transferred.
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE
            The swizzling mode to use for memory access optimization.
        __tile_shape: IndexList[rank] = tile_shape
            Internal parameter for the tile shape in shared memory.
        __desc_shape: IndexList[rank] = _default_desc_shape[...]()
            Internal parameter for the descriptor shape, which may differ from the
            tile shape to accommodate hardware requirements.

    Returns:
        A `TMATensorTile` configured with the specified parameters, ready for use in
        asynchronous data transfer operations.

    Constraints:

        - Only supports 2D and 3D tensors (rank must be 2 or 3).
        - For non-SWIZZLE_NONE modes, the K dimension size in bytes must be a multiple
          of the swizzle mode's byte size.
        - For MN-major layout, only SWIZZLE_128B is supported.
        - For 3D tensors, only K-major layout is supported.

    Raises:
        If TMA descriptor creation fails.
    """

    return TMATensorTile[dtype, rank, __tile_shape, __desc_shape](
        TMADescriptor()
    )


struct TMATensorTileArray[
    num_of_tensormaps: Int,
    dtype: DType,
    rank: Int,
    cta_tile_shape: IndexList[rank],
    desc_shape: IndexList[rank],
](DevicePassable, TrivialRegisterPassable):
    """An array of TMA descriptors.

    Parameters:
        num_of_tensormaps: Int
            The number of TMA descriptors aka tensor map.
        dtype: DType
            The data type of the tensor elements.
        rank: Int
            The dimensionality of the tile (2, 3, 4, or 5).
        cta_tile_shape: IndexList[rank]
            The shape of the CTA tile in shared memory.
        desc_shape: IndexList[rank]
            The shape of the descriptor, which can be different from the tile shape
            to accommodate hardware requirements like WGMMA.
    """

    var tensormaps_ptr: MutPointer[UInt8, MutUntrackedOrigin]
    """A static tuple of pointers to TMA descriptors.

    This field stores an array of pointers to `TMATensorTile` instances, where each pointer
    references a TMA descriptor in device memory. The array has a fixed size determined by
    the num_of_tensormaps parameter.

    The TMA descriptors are used by the GPU hardware to efficiently transfer data between
    global and shared memory with specific memory access patterns defined by the layouts.
    """

    comptime descriptor_bytes = 128
    """Size of the TMA descriptor in bytes.

    This is a constant value that represents the size of the TMA descriptor in bytes.
    It is used to calculate the offset of the TMA descriptor in the device memory.
    """

    comptime device_type: AnyType = Self
    """The device-side type representation."""

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        """Device type mapping is the identity function."""
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        """
        Gets this type's name, for use in error messages when handing arguments
        to kernels.

        Returns:
            This type's name.
        """
        return String(
            "TMATensorTileArray[num_of_tensormaps = ",
            Self.num_of_tensormaps,
            ", dtype = ",
            Self.dtype,
            ", cta_tile_shape = ",
            _idx_str[Self.rank, Self.cta_tile_shape](),
            ", desc_shape = ",
            _idx_str[Self.rank, Self.desc_shape](),
            "]",
        )

    @always_inline
    def __init__(
        out self,
        mut tensormaps_device: DeviceBuffer[.uint8],
    ):
        """
        Initializes a new TMATensorTileArray.

        Args:
            tensormaps_device: Device buffer to store TMA descriptors.
        """
        # TODO: this type should properly hold origins
        self.tensormaps_ptr = tensormaps_device.unsafe_ptr().unsafe_origin_cast[
            MutUntrackedOrigin
        ]()

    @always_inline
    def __getitem__(
        self, index: Int
    ) -> MutPointer[
        TMATensorTile[
            Self.dtype, Self.rank, Self.cta_tile_shape, Self.desc_shape
        ],
        MutAnyOrigin,
    ]:
        """
        Retrieve a TMA descriptor.

        Args:
            index: Index of the TMA descriptor.

        Returns:
            `Pointer` to the `TMATensorTile` at the specified index.
        """
        return (
            (self.tensormaps_ptr + index * self.descriptor_bytes)
            .bitcast[
                TMATensorTile[
                    Self.dtype, Self.rank, Self.cta_tile_shape, Self.desc_shape
                ]
            ]()
            .as_unsafe_any_origin()
        )


struct RaggedTMA3DTile[
    dtype: DType,
    swizzle_mode: TensorMapSwizzle,
    *,
    BM: Int,
    BN: Int,
    middle_dim: Int,
    group: Int = 1,
    tma_blocks_per_op: Int = 0,
](DevicePassable, ImplicitlyCopyable):
    """
    Creates a TMA descriptor for loading/storing from ragged 3D arrays with a
    ragged leading dimension. This loads 2D tiles, indexing into the middle dim.
    When using this loads, it is essential that at least `BM_seq * stride` space
    has been allocated in front of the gmem pointer, otherwise
    `CUDA_ERROR_ILLEGAL_ADDRESS` may result.

    The `(middle_dim, rows)` selector dims are always folded into one outermost
    descriptor dim (both are `box == 1` and GMEM-contiguous), dropping one rank:
    each copy issues coordinate `(ragged_idx + dynamic_dim) * middle_dim + middle_idx`
    on that dim. Fewer descriptor dims means fewer per-issue `UMOV`s into uniform
    registers (which `ptxas` allocates poorly), at no offsetting cost.

    When `group > 1`, the gmem is treated as 4D `(rows, middle_dim, group, depth)`.
    The smem tile has `BM_seq * group = BM` rows, where `BM_seq = BM // group` is the
    number of distinct sequence positions. The `dynamic_dim` parameter in copy
    methods represents valid sequence positions. The descriptor is rank-4
    (`merged, BM_seq, group, depth`), or rank-5 when `tma_blocks_per_op > 0`.

    When `tma_blocks_per_op > 0` (only valid for `swizzle_mode == SWIZZLE_NONE`),
    the contiguous `depth` dimension is split into
    `(depth // swizzle_granularity, swizzle_granularity)` and a *blocks* dimension is
    added to the descriptor box, so a single `async_copy_batched` copies
    `tma_blocks_per_op` swizzle-granularity blocks at once rather than one block per
    `async_copy_from_col`. With the selector merge this is rank-4 for `group == 1`
    and rank-5 for `group > 1`. The blocks dimension's global extent is the true
    block count, so a box that overhangs the end is masked off by the TMA.

    Parameters:
        dtype: The data type of the tensor.
        swizzle_mode: The swizzling mode to use for memory access.
        BM: The number of rows of the corresponding 2D shared memory tile.
        BN: The number of columns of the corresponding 2D shared memory tile.
        middle_dim: The middle (head) extent, folded into the ragged-selector
            coordinate. Each copy issues coordinate
            `(ragged_idx + dynamic_dim) * middle_dim + middle_idx` on the merged
            outermost descriptor dim.
        group: The number of heads fused into each sequence position (default 1).
        tma_blocks_per_op: Swizzle-granularity blocks copied per `async_copy_batched`
            (0 = disabled, use the per-block `async_copy_from_col` path).
    """

    comptime BM_seq: Int = Self.BM // Self.group
    """Number of distinct sequence positions per tile."""

    var descriptor: TMADescriptor
    """The TMA descriptor that will be used to store the ragged tensor."""

    comptime device_type: AnyType = Self
    """The device-side type representation."""

    comptime swizzle_granularity = Self.swizzle_mode.bytes() // size_of[
        Self.dtype
    ]()
    """The number of columns that must be copied at a time due to the swizzle size."""

    comptime layout: Layout = tile_layout_k_major[
        Self.dtype, Self.BM, Self.BN, Self.swizzle_mode
    ]()
    """The unswizzled-smem layout copied to/from by this tma op."""

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        """Device type mapping is the identity function."""
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        """
        Returns a string representation of the RaggedTMA3DTile type.

        Returns:
            A string containing the type name with all template parameters.
        """
        return String(
            "RaggedTMA3DTile[dtype = ",
            Self.dtype,
            ", swizzle_mode = ",
            Self.swizzle_mode,
            ", BM = ",
            Self.BM,
            ", BN = ",
            Self.BN,
            ", middle_dim = ",
            Self.middle_dim,
            ", group = ",
            Self.group,
            ", tma_blocks_per_op = ",
            Self.tma_blocks_per_op,
        )

    @always_inline
    def __init__(out self, descriptor: TMADescriptor):
        """
        Initializes a new RaggedTMA3DTile with the provided TMA descriptor.

        Args:
            descriptor: The TMA descriptor that defines the memory access pattern.
        """
        self.descriptor = descriptor

    @staticmethod
    @always_inline
    def create[
        *,
        depth: Int = Self.BN,
    ](
        ctx: DeviceContext,
        ptr: ImmPointer[Scalar[Self.dtype], _],
        *,
        rows: Int,
    ) raises -> Self:
        """
        Create a RaggedTMA3DTile.

        Parameters:
            depth: The size of the inner-most, contiguous, dimension.

        Args:
            ctx: The device context used to create the TMA descriptors.
            ptr: The global memory pointer.
            rows: The size of the ragged dimension.

        Returns:
            A RaggedTMA3DTile corresponding to the gmem.

        Raises:
            If TMA descriptor creation fails.
        """
        # The `(middle_dim, rows)` selector dims are folded into one outermost
        # `merged` dim: extent `middle_dim*(rows+1)`, stride = the head stride
        # (`depth` for group==1, `group*depth` for group>1), box 1. Each copy
        # issues coordinate `(ragged_idx + dynamic_dim)*middle_dim + middle_idx`
        # on it. This drops one descriptor rank everywhere.
        var merged_extent = Self.middle_dim * (rows + 1)
        comptime if Self.group > 1:
            # Fused GQA: gmem is 4D (rows, middle_dim, group, depth).
            var stride = Self.middle_dim * (Self.group * depth)
            comptime if Self.tma_blocks_per_op > 0:
                # Batched fused-GQA store: split `depth` into (n_blocks, K) and
                # add a *blocks* box dim. Rank-5 dims (inner->outer) =
                # (K, group, BM_seq, n_blocks, merged); the SMEM box traversal
                # (K, group, BM_seq, blocks) matches the blocked smem
                # `[blocks, BM_seq, group, K]` that `write_block` produces
                # (local_row = bm_seq*group + g).
                comptime assert (
                    Self.swizzle_mode == TensorMapSwizzle.SWIZZLE_NONE
                ), (
                    "tma_blocks_per_op requires SWIZZLE_NONE (identity smem"
                    " layout)."
                )
                comptime assert (
                    depth % Self.swizzle_granularity == 0
                ), "depth must be a multiple of the swizzle granularity."
                comptime K = Self.swizzle_granularity
                comptime n_blocks = depth // K
                return Self(
                    create_tma_descriptor[Self.dtype, 5, Self.swizzle_mode](
                        DeviceBuffer(
                            ctx, ptr - stride * Self.BM_seq, 1, owning=False
                        ),
                        IndexList[5](
                            merged_extent, n_blocks, Self.BM_seq, Self.group, K
                        ),
                        IndexList[5](Self.group * depth, K, stride, depth, 1),
                        IndexList[5](
                            1,
                            Self.tma_blocks_per_op,
                            Self.BM_seq,
                            Self.group,
                            K,
                        ),
                    ),
                )
            else:
                # Per-block / load fused-GQA: rank-4 dims (inner->outer) =
                # (depth, group, BM_seq, merged).
                return Self(
                    create_tma_descriptor[Self.dtype, 4, Self.swizzle_mode](
                        DeviceBuffer(
                            ctx, ptr - stride * Self.BM_seq, 1, owning=False
                        ),
                        IndexList[4](
                            merged_extent, Self.BM_seq, Self.group, depth
                        ),
                        IndexList[4](Self.group * depth, stride, depth, 1),
                        IndexList[4](
                            1, Self.BM_seq, Self.group, Self.swizzle_granularity
                        ),
                    ),
                )
        else:
            # group == 1: gmem is 3D (rows, middle_dim, depth).
            var stride = Self.middle_dim * depth
            comptime if Self.tma_blocks_per_op > 0:
                # Batched store: split `depth` into (n_blocks, K) + blocks box.
                # Rank-4 dims (inner->outer) = (K, BM, n_blocks, merged); the
                # SMEM box traversal (K, BM, blocks) == blocked smem
                # `[blocks, BM, K]` that `write_block` produces.
                comptime assert (
                    Self.swizzle_mode == TensorMapSwizzle.SWIZZLE_NONE
                ), (
                    "tma_blocks_per_op requires SWIZZLE_NONE (identity smem"
                    " layout)."
                )
                comptime assert (
                    depth % Self.swizzle_granularity == 0
                ), "depth must be a multiple of the swizzle granularity."
                comptime K = Self.swizzle_granularity
                comptime n_blocks = depth // K
                return Self(
                    create_tma_descriptor[Self.dtype, 4, Self.swizzle_mode](
                        DeviceBuffer(
                            ctx, ptr - stride * Self.BM, 1, owning=False
                        ),
                        IndexList[4](merged_extent, n_blocks, Self.BM, K),
                        IndexList[4](depth, K, stride, 1),
                        IndexList[4](1, Self.tma_blocks_per_op, Self.BM, K),
                    ),
                )
            else:
                # Per-block / load: rank-3 dims (inner->outer) = (depth, BM, merged).
                return Self(
                    create_tma_descriptor[Self.dtype, 3, Self.swizzle_mode](
                        DeviceBuffer(
                            ctx, ptr - stride * Self.BM, 1, owning=False
                        ),
                        IndexList[3](merged_extent, Self.BM, depth),
                        IndexList[3](depth, stride, 1),
                        IndexList[3](1, Self.BM, Self.swizzle_granularity),
                    ),
                )

    @always_inline
    def async_copy_from_col[
        col: Int,
        eviction_policy: CacheEviction = CacheEviction.EVICT_FIRST,
    ](
        self,
        src: ImmPointer[Scalar[Self.dtype], _, address_space=.SHARED],
        *,
        ragged_idx: UInt32,
        dynamic_dim: UInt32,
        middle_idx: UInt32,
        elect: Int32,
    ):
        """Copy a single swizzle_granularity-wide column chunk from smem to
        gmem.

        The TMA store is PTX-predicated on `elect`, so call this unconditionally
        from every lane (no warp-divergent `if elect != 0:` wrapper); only the
        elected lane issues the copy.

        Parameters:
            col: Which column chunk (0-indexed, each chunk is
                swizzle_granularity columns).
            eviction_policy: Optional cache eviction policy that controls how
                the data is handled in the cache hierarchy. Defaults to
                EVICT_FIRST.

        Args:
            src: Source shared memory pointer (base of the full tile).
            ragged_idx: Index into the ragged dimension.
            dynamic_dim: Number of rows (or seq positions when group > 1)
                to copy.
            middle_idx: Index into the middle (generally head) dimension.
            elect: `0` on non-elected lanes (skip the TMA), non-zero on the
                single elected lane (issue the TMA).
        """
        comptime copy_offset = col * Self.BM * Self.swizzle_granularity
        # `merged` folds (middle_dim, rows) into the outermost descriptor dim.
        var offset_ragged_idx = Int(ragged_idx + dynamic_dim)
        var merged = offset_ragged_idx * Self.middle_dim + Int(middle_idx)

        comptime if Self.group > 1:
            var box_idx = Int(UInt32(Self.BM_seq) - dynamic_dim)

            cp_async_bulk_tensor_global_shared_cta_elect[
                eviction_policy=eviction_policy
            ](
                src + copy_offset,
                Pointer(to=self.descriptor).bitcast[NoneType](),
                # dims (inner->outer): (depth, group, BM_seq, merged)
                Index(
                    col * Self.swizzle_granularity,
                    0,
                    box_idx,
                    merged,
                ),
                elect,
            )
        else:
            var box_idx = Int(UInt32(Self.BM) - dynamic_dim)

            cp_async_bulk_tensor_global_shared_cta_elect[
                eviction_policy=eviction_policy
            ](
                src + copy_offset,
                Pointer(to=self.descriptor).bitcast[NoneType](),
                # dims (inner->outer): (depth, BM, merged)
                Index(
                    col * Self.swizzle_granularity,
                    box_idx,
                    merged,
                ),
                elect,
            )

    @always_inline
    def async_copy_batched[
        col_start: Int,
        eviction_policy: CacheEviction = CacheEviction.EVICT_FIRST,
    ](
        self,
        src: ImmPointer[Scalar[Self.dtype], _, address_space=.SHARED],
        *,
        ragged_idx: UInt32,
        dynamic_dim: UInt32,
        middle_idx: UInt32,
        elect: Int32,
    ):
        """Copy `tma_blocks_per_op` swizzle_granularity-wide column blocks from
        smem to gmem in a single TMA, starting at block `col_start`.

        Only valid when `tma_blocks_per_op > 0` (SWIZZLE_NONE). The descriptor
        box covers `tma_blocks_per_op` blocks; if `col_start + tma_blocks_per_op`
        overruns the true block count, the TMA masks the overhang off (no gmem
        write). With the (middle_dim, rows) selector merge the descriptor is
        rank-4 for `group == 1` and rank-5 for `group > 1`.

        The TMA store is PTX-predicated on `elect`, so call this unconditionally
        from every lane (no warp-divergent `if elect != 0:` wrapper); only the
        elected lane issues the copy.

        Parameters:
            col_start: First block (0-indexed, each block is swizzle_granularity
                columns) copied by this op.
            eviction_policy: Optional cache eviction policy that controls how the
                data is handled in the cache hierarchy. Defaults to EVICT_FIRST.

        Args:
            src: Source shared memory pointer (base of the full blocked tile).
            ragged_idx: Index into the ragged dimension.
            dynamic_dim: Number of rows (or seq positions when group > 1) to copy.
            middle_idx: Index into the middle (generally head) dimension.
            elect: `0` on non-elected lanes (skip the TMA), non-zero on the
                single elected lane (issue the TMA).
        """
        comptime assert (
            Self.tma_blocks_per_op > 0
        ), "async_copy_batched requires tma_blocks_per_op > 0."
        comptime copy_offset = col_start * Self.BM * Self.swizzle_granularity
        # `merged` folds (middle_dim, rows) into the outermost descriptor dim.
        var offset_ragged_idx = Int(ragged_idx + dynamic_dim)
        var merged = offset_ragged_idx * Self.middle_dim + Int(middle_idx)

        comptime if Self.group > 1:
            var box_idx = Int(UInt32(Self.BM_seq) - dynamic_dim)
            # dims (inner->outer): (K, group, BM_seq, n_blocks, merged)
            cp_async_bulk_tensor_global_shared_cta_elect[
                eviction_policy=eviction_policy
            ](
                src + copy_offset,
                Pointer(to=self.descriptor).bitcast[NoneType](),
                Index(0, 0, box_idx, col_start, merged),
                elect,
            )
        else:
            var box_idx = Int(UInt32(Self.BM) - dynamic_dim)
            # dims (inner->outer): (K, BM, n_blocks, merged)
            cp_async_bulk_tensor_global_shared_cta_elect[
                eviction_policy=eviction_policy
            ](
                src + copy_offset,
                Pointer(to=self.descriptor).bitcast[NoneType](),
                Index(0, box_idx, col_start, merged),
                elect,
            )

    @always_inline
    def prefetch_descriptor(self):
        """
        Prefetches the TMA descriptor into cache.
        """

        prefetch_tma_descriptor(Pointer(to=self.descriptor).bitcast[NoneType]())


struct RaggedTensorMap[
    descriptor_rank: Int,
    //,
    dtype: DType,
    descriptor_shape: IndexList[descriptor_rank],
    remaining_global_dim_rank: Int,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](DevicePassable, ImplicitlyCopyable):

    """
    Creates a TMA descriptor that can handle stores with varying lengths. This struct is mainly used
    for MHA, where sequence lengths may vary between sample.

    This struct only supports one dimension being ragged. The continuous dimension (where stride is 1) cannot be ragged.

    Parameters:
        descriptor_rank:
            The rank of the descriptor shape (inferred).
        dtype:
            The data type of the tensor.
        descriptor_shape:
            The shape of the shared memory descriptor.
        remaining_global_dim_rank:
            The rank of the remaining global tensor dimensions.
        swizzle_mode:
            The swizzling mode to use for memory access optimization. Swizzling can improve
            memory access patterns for specific hardware configurations. Defaults to SWIZZLE_NONE.

    """

    var descriptor: TMADescriptor
    """The TMA descriptor that will be used to store the ragged tensor."""
    var max_length: Int
    """The maximum length present in the sequences of the ragged tensor."""
    var global_shape: DynamicCoord[.int64, Self.global_rank]
    """The shape of the global tensor."""
    var global_stride: DynamicCoord[.int64, Self.global_rank]
    """The stride of the global tensor."""

    comptime global_rank = Self.remaining_global_dim_rank + 3
    """The rank of the global tensor."""

    @staticmethod
    def _descriptor_shape() -> IndexList[Self.descriptor_rank + 1]:
        """
        Constructs a descriptor shape that can handle one ragged dimension for loads.

        Returns:
            A descriptor shape.
        """

        var idx_list = IndexList[Self.descriptor_rank + 1](fill=0)
        idx_list[0] = 1

        comptime for idx in range(Self.descriptor_rank):
            idx_list[idx + 1] = Self.descriptor_shape[idx]

        return idx_list

    @staticmethod
    @always_inline
    def _get_layout() -> Layout:
        var layout = Layout(
            IntTuple(num_elems=Self.global_rank),
            IntTuple(num_elems=Self.global_rank),
        )

        comptime for idx in range(Self.global_rank):
            layout.shape.replace_entry(idx, int_value=UNKNOWN_VALUE)
            layout.stride.replace_entry(idx, int_value=UNKNOWN_VALUE)

        return layout^

    comptime device_type: AnyType = Self
    """The TensorMapDescriptorArray type."""

    comptime ragged_descriptor_shape = Self._descriptor_shape()
    """The shape of the descriptor that will tile and load from shared -> global memory."""

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        """
        Copies this descriptor array to device memory.

        Args:
            encoder: The device specific type encoder.
            target: Opaque pointer to the target device memory location.
        """
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        """
        Returns a string representation of the TensorMapDescriptorArray type.

        Returns:
            A string containing the type name with all template parameters.
        """
        return String(
            "RaggedTensorMap[rank = ",
            Self.descriptor_rank,
            ", dtype = ",
            Self.dtype,
            ", descriptor_shape = ",
            Self.ragged_descriptor_shape,
            ", swizzle_mode = ",
            Self.swizzle_mode,
            ", max_descriptor_length = ",
            "]",
        )

    @staticmethod
    @always_inline
    def _create_global_stride(
        ragged_stride: Int,
        remaining_global_stride: IndexList[Self.remaining_global_dim_rank],
    ) -> IndexList[Self.global_rank]:
        var global_stride = IndexList[Self.global_rank](fill=0)
        global_stride[0] = ragged_stride
        global_stride[Self.global_rank - 2] = ragged_stride
        global_stride[Self.global_rank - 1] = 1

        comptime for idx in range(1, 1 + Self.remaining_global_dim_rank):
            global_stride[idx] = remaining_global_stride[idx - 1]

        return global_stride

    @staticmethod
    @always_inline
    def _create_global_shape(
        cumulative_length: Int,
        max_length: Int,
        global_last_dim: Int,
        remaining_global_shape: IndexList[Self.remaining_global_dim_rank],
    ) -> IndexList[Self.global_rank]:
        var global_shape = IndexList[Self.global_rank](fill=0)
        global_shape[0] = cumulative_length

        comptime for idx in range(1, 1 + Self.remaining_global_dim_rank):
            global_shape[idx] = remaining_global_shape[idx - 1]

        global_shape[Self.global_rank - 2] = max_length
        global_shape[Self.global_rank - 1] = global_last_dim

        return global_shape

    def __init__(
        out self,
        ctx: DeviceContext,
        global_ptr: ImmPointer[Scalar[Self.dtype], _],
        max_length: Int,
        ragged_stride: Int,
        batch_size: Int,
        global_last_dim: Int,
        remaining_global_dims: IndexList[Self.remaining_global_dim_rank],
        remaining_global_stride: IndexList[Self.remaining_global_dim_rank],
    ) raises:
        """
        Initializes a TensorMapDescriptorArray with descriptors for all power-of-2 lengths.

        This constructor creates a complete set of TMA descriptors, one for each power of 2
        from 1 up to max_descriptor_length. Each descriptor is configured to handle a different
        first dimension size (1, 2, 4, 8, ..., max_descriptor_length) while maintaining the
        same remaining tile shape specified by desc_remaining_tile_shape.

        Raises:
            If the operation fails.

        Args:
            ctx:
                The device context used to create the TMA descriptors.
            global_ptr:
                The source tensor in global memory that will be accessed using the descriptors.
            max_length:
                The maximum length present in the sequences of the ragged tensor.
            ragged_stride:
                The stride of the ragged dimension in the global tensor.
            batch_size:
                The total number of sequences in the ragged tensor.
            global_last_dim:
                The last dimension of the global tensor.
            remaining_global_dims:
                The dimensions of the remaining global tensor.
            remaining_global_stride:
                The stride of the remaining global tensor.
        Constraints:
            - max_descriptor_length must be a power of two.
            - max_descriptor_length must be less than or equal to 256.
        """

        comptime assert (
            Self.global_rank >= 2
        ), "global_rank must be at least 2 with one ragged dimension"

        var cumulative_length = (batch_size + 1) * max_length

        var global_shape = Self._create_global_shape(
            cumulative_length,
            max_length,
            global_last_dim,
            remaining_global_dims,
        )

        var global_stride = Self._create_global_stride(
            ragged_stride, remaining_global_stride
        )

        comptime global_layout = Self._get_layout()

        var global_runtime_layout = RuntimeLayout[global_layout](
            global_shape, global_stride
        )

        comptime GlobalTensorType = LayoutTensor[
            Self.dtype,
            global_layout,
            MutAnyOrigin,
        ]

        var decremented_ptr = global_ptr - (ragged_stride * max_length)
        var global_tensor = GlobalTensorType(
            decremented_ptr.unsafe_mut_cast[True]().unsafe_origin_cast[
                MutAnyOrigin
            ](),
            global_runtime_layout,
        )

        self.descriptor = _create_tma_descriptor_helper[
            Self.ragged_descriptor_shape, Self.swizzle_mode
        ](
            ctx,
            global_tensor,
        )

        self.max_length = max_length
        self.global_shape = Coord(global_shape)
        self.global_stride = Coord(global_stride)

    @always_inline
    def _get_descriptor_ptr(self) -> MutPointer[NoneType, MutAnyOrigin]:
        return (
            Pointer(to=self.descriptor)
            .bitcast[NoneType]()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )

    @always_inline
    def store_ragged_tile[
        rank: Int,
        //,
        using_max_descriptor_size: Bool = False,
    ](
        self,
        coordinates: IndexList[rank],
        preceding_cumulative_length: Int,
        store_length: Int,
        mut tile_iterator: LayoutTensorIter[
            Self.dtype,
            _,
            MutAnyOrigin,
            address_space=.SHARED,
            ...,
        ],
    ):
        """
        Stores a ragged tile from shared memory to global memory.

        Parameters:
            rank:
                The rank of the coordinates.
            using_max_descriptor_size:
                If True, optimizes the store around the max descriptor size.

        Args:
            coordinates:
                The starting coordinates of all dimensions except the ragged dimension.
            preceding_cumulative_length:
                The cumulative length of the preceding sequences.
            store_length:
                The length of the current sequence to be stored.
            tile_iterator:
                The iterator over the tile in shared memory.
        """

        comptime assert rank == Self.global_rank

        # Assume we have the following ragged tensor:

        # It has 16 heads, head depth of 128, and 4 sequences of length
        # [43, 32, 10, 64]

        # The overall shape will look like this with ? representing the 4 sequences:
        # [?, 16, 128]

        # When creating the TMA descriptor you pass in several values: max_length, ragged_stride,
        # batch_size, global_last_dim, remaining_global_dims, remaining_global_stride

        # In our case:

        # max_length = 64 (the max length of the sequences)
        # ragged_stride = 2048 (heads x head depth)
        # batch_size = 4 (the number of sequence batches)
        # global_last_dim = 128 (the last dimension of the global tensor, the head depth)
        # remaining_global_dims = [16] (the only value not supplied, the head dimension)
        # remaining_global_stride = [128] (the stride of the head dimension)

        # We also compute values such as the cumulative length using this formula:
        # cumulative_length = (batch_size + 1) * max_length = (4 + 1) * 64 = 320

        # With these values we create our descriptor with an artificial layout of:

        # (cumulative_length, remaining_global_dims..., max_length, global_last_dim) : (ragged_stride, remaining_global_stride..., max_length, global_last_dim)
        # (320, 16, 64, 128) : (2048, 128, 2048, 1)

        # (internally this layout gets reversed when passed into the descriptor)

        # Now lets say we have a descriptor of shape (1, 1, 24, 64), the 24 tells us that we
        # want to store 24 sequences at once and 64 tells us we want to store half the depth.

        # Now lets say we want to store the first depth chunk (64) of the first batch (43) at head 7.

        # We would need to do a total of 2 stores, with the global coordinates naively being:
        # [(0, 7, 0, 0), (24, 7, 0, 0)] || [(0, 7, 0, 0), (0, 7, 24, 0)]

        # Both cases will cause spillage since 24 * 2 = 48.

        # Instead we will utilize the cumulative_length dimension and max_length dimension to mask the out of bounds
        # segments in each ragged store.

        # One prerequisite for this to work is that the starting pointer must be negatively offset by ragged_stride * max_length.
        # Which in our case is 2048 * 64 or 64 sequences.

        # Now to get bounds checked store we set the
        # cumulative_length dimension to the cumulative length of the preceding sequences + this sequence's length.
        # And we set the max_length dimension to the max_length - this sequence's length.

        # This would make our new coordinate starting global coordinates: [(43, 7, 21, 0), (43, 7, 45, 0)]

        # When adding 43 + 21 we get a starting offset of 64, which is how much we offset our original pointer. This gives
        # us the correct starting offset for our store. Finally our max_length dimension is set to start at 21. It is hardbounded
        # by 64 (the max length) so this ensure that anything we load past 64 will be masked out. So when we end at 68 for the second store,
        # the last 5 sequences will be masked out.

        # Now lets say we want to try and store the second sequence (32)

        # Our new coordinates would be: [(75, 7, 32, 0), (75, 7, 56, 0)]

        # starting us at (75 + 32) - 64 = 43, and allowing us to only load 32 sequences

        comptime if using_max_descriptor_size:
            # if the max length is the same as the descriptor size we don't need to do
            # multiple stores and generate multiple coords so we can avoid unnecessary
            # branching in this case.
            var cumulative_length = preceding_cumulative_length + store_length

            var adjusted_coordinates = coordinates
            adjusted_coordinates[Self.global_rank - 1] = cumulative_length
            adjusted_coordinates[1] = self.max_length - store_length

            cp_async_bulk_tensor_global_shared_cta(
                tile_iterator[].ptr,
                self._get_descriptor_ptr(),
                adjusted_coordinates,
            )
        else:
            comptime descriptor_load_length = Self.ragged_descriptor_shape[
                Self.global_rank - 2
            ]

            var descriptor_iters = ceildiv(store_length, descriptor_load_length)

            var cumulative_length = preceding_cumulative_length + store_length

            var adjusted_coordinates = coordinates
            adjusted_coordinates[Self.global_rank - 1] = cumulative_length

            for i in range(descriptor_iters):
                var max_length_offset = (
                    self.max_length
                    - store_length
                    + (i * descriptor_load_length)
                )
                adjusted_coordinates[1] = max_length_offset

                cp_async_bulk_tensor_global_shared_cta(
                    tile_iterator[].ptr,
                    self._get_descriptor_ptr(),
                    adjusted_coordinates,
                )

                tile_iterator._incr()

    @always_inline
    def prefetch_descriptor(self):
        """
        Prefetches the TMA descriptor into cache.
        """

        prefetch_tma_descriptor(self._get_descriptor_ptr())


struct TMATensorTileIm2col[
    dtype: DType,
    rank: Int,
    tile_shape: IndexList[rank],
    desc_shape: IndexList[rank] = tile_shape,
](DevicePassable, ImplicitlyCopyable):
    """TMA tensor tile with im2col coordinate transformation for convolution.

    This struct enables hardware-accelerated im2col transformation during TMA loads,
    used for implicit GEMM convolution. The TMA descriptor encodes the convolution
    geometry (padding, stride, dilation) and performs coordinate transformation
    on-the-fly.

    The coordinate system uses GEMM-style 2D coordinates:
    - coords[0]: K coordinate (indexes into R * S * C reduction dimension)
    - coords[1]: M coordinate (indexes into batch * H_out * W_out spatial)

    Internally:
    - K is decomposed into (c, r, s) where K = r*S*C + s*C + c (filter-first, channel-last for NHWC)
    - M is decomposed into (n, h, w) where M = n*H_out*W_out + h*W_out + w
    - 4D coordinates (c, w, h, n) and filter offsets (s, r) are passed to the
      PTX im2col instruction.

    Parameters:
        dtype: The data type of tensor elements.
        rank: The dimensionality of the tile (2, 3, 4, or 5).
        tile_shape: The shape of the tile in shared memory.
        desc_shape: The shape of the descriptor (may differ for WGMMA compatibility).
    """

    var descriptor: TMADescriptor
    """The TMA descriptor encoding im2col transformation parameters."""

    var out_height: UInt32
    """Output height (H_out) for M coordinate decomposition."""

    var out_width: UInt32
    """Output width (W_out) for M coordinate decomposition."""

    var filter_h: UInt32
    """Filter height (R) for K coordinate decomposition."""

    var filter_w: UInt32
    """Filter width (S) for K coordinate decomposition."""

    var in_channels: UInt32
    """Input channels (C) for K coordinate decomposition."""

    var lower_corner_h: Int32
    """Lower corner offset for height (H dimension) - matches CUTLASS ArithmeticTupleIterator pattern."""

    var lower_corner_w: Int32
    """Lower corner offset for width (W dimension) - matches CUTLASS ArithmeticTupleIterator pattern."""

    comptime device_type: AnyType = Self
    """The device-side type representation."""

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        """Device type mapping is the identity function."""
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        """Gets this type's name for error messages.

        Returns:
            This type's name.
        """
        return String(
            "TMATensorTileIm2col[dtype = ",
            Self.dtype,
            ", tile_shape = ",
            _idx_str[Self.rank, Self.tile_shape](),
            ", desc_shape = ",
            _idx_str[Self.rank, Self.desc_shape](),
            "]",
        )

    @always_inline
    def __init__(
        out self,
        descriptor: TMADescriptor,
        out_height: UInt32,
        out_width: UInt32,
        filter_h: UInt32,
        filter_w: UInt32,
        in_channels: UInt32,
        lower_corner_h: Int32 = 0,
        lower_corner_w: Int32 = 0,
    ):
        """Initializes with the provided TMA im2col descriptor and dimensions.

        Args:
            descriptor: The TMA descriptor that encodes im2col transformation.
            out_height: Output height (H_out) for M coordinate decomposition.
            out_width: Output width (W_out) for M coordinate decomposition.
            filter_h: Filter height (R) for K coordinate decomposition.
            filter_w: Filter width (S) for K coordinate decomposition.
            in_channels: Input channels (C) for K coordinate decomposition.
            lower_corner_h: Lower corner offset for H dimension (matches CUTLASS pattern).
            lower_corner_w: Lower corner offset for W dimension (matches CUTLASS pattern).
        """
        self.descriptor = descriptor
        self.out_height = out_height
        self.out_width = out_width
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.in_channels = in_channels
        self.lower_corner_h = lower_corner_h
        self.lower_corner_w = lower_corner_w

    @always_inline
    def prefetch_descriptor(self):
        """Prefetches the TMA descriptor into cache."""
        var desc_ptr = Pointer(to=self.descriptor).bitcast[NoneType]()
        prefetch_tma_descriptor(desc_ptr)

    @always_inline
    def async_copy[
        cta_group: Int = 1,  # Use SM90-style TMA for cluster 1x1x1
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: LayoutTensor[mut=True, Self.dtype, _, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int],
    ):
        """Schedules an asynchronous im2col TMA load.

        Uses 2D GEMM-style coordinates:
        - coords[0]: K coordinate (indexes into C * R * S reduction dimension)
        - coords[1]: M coordinate (indexes into batch * H_out * W_out spatial)

        Internally:
        - K is decomposed into (c, r, s) where K = c*R*S + r*S + s
        - M is decomposed into (n, h, w) where M = n*H_out*W_out + h*W_out + w
        - 4D coordinates (c, w, h, n) and filter offsets (s, r) are passed to
          the PTX im2col instruction.

        Note: The cta_group parameter defaults to 2 because SM100/Blackwell
        im2col TMA with padding (negative corners) requires the cta_group::2
        PTX format. This is consistent with CUTLASS which only provides
        SM100_TMA_2SM_LOAD_IM2COL (no cta_group::1 variant for im2col).

        Parameters:
            cta_group: CTA group size for TMA operations.
            eviction_policy: Cache eviction policy for the TMA load.

        Args:
            dst: Destination tensor in shared memory.
            mem_barrier: Memory barrier for synchronization.
            coords: GEMM coordinates (k_coord, m_coord).
        """
        comptime assert (
            type_of(dst).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = Self.tile_shape[0] // copy_dim0
        comptime num_copies_dim1 = Self.tile_shape[1] // copy_dim1

        # Precompute spatial size for M decomposition
        var hw = Int(self.out_height) * Int(self.out_width)
        var out_w = Int(self.out_width)

        # Precompute filter window size for K decomposition
        # K = r * S * C + s * C + c (filter-first, channel-last ordering for NHWC)
        var num_channels = Int(self.in_channels)
        var filter_w = Int(self.filter_w)

        # OPTIMIZATION: Hoist K decomposition outside loop (constant when j=0).
        # For typical configs (num_copies_dim1=1), K coords don't change within tile.
        var k_coord = coords[0]
        var filter_idx, c = udivmod(k_coord, num_channels)
        var r, s = udivmod(filter_idx, filter_w)

        # Initial M decomposition (done once, then use iterator)
        var m_coord_init = coords[1]
        var n, m_remainder = udivmod(m_coord_init, hw)
        var h_out, w_out = udivmod(m_remainder, out_w)

        # Pre-add lower_corner offset
        var h = h_out + Int(self.lower_corner_h)
        var w = w_out + Int(self.lower_corner_w)

        # Cache bounds for iterator wraparound
        var out_h_int = Int(self.out_height)
        var lower_h = Int(self.lower_corner_h)
        var lower_w = Int(self.lower_corner_w)

        comptime for i in range(num_copies_dim0):
            comptime for j in range(num_copies_dim1):
                comptime copy_offset: UInt32 = UInt32(
                    (i * num_copies_dim1 + j) * copy_size
                )

                # K recomputation only needed when j > 0 (rare in practice)
                comptime if j > 0:
                    k_coord = coords[0] + j * copy_dim1
                    filter_idx, c = udivmod(k_coord, num_channels)
                    r, s = udivmod(filter_idx, filter_w)

                # Pass 4D coords (c, w, h, n) and filter offsets (s, r) to im2col PTX
                cp_async_bulk_tensor_shared_cluster_global_im2col[
                    cta_group=cta_group,
                ](
                    dst.ptr + copy_offset,
                    Pointer(to=self.descriptor).bitcast[NoneType](),
                    mem_barrier.unsafe_ptr(),
                    Index(c, w, h, n),
                    Index(s, r),
                )

            # Iterator pattern: advance M by copy_dim0 using addition (not division)
            # This avoids 4 divisions per sub-tile, reducing from O(n*8) to O(8+n*3)
            w += copy_dim0
            if w >= out_w + lower_w:
                w -= out_w
                h += 1
                if h >= out_h_int + lower_h:
                    h -= out_h_int
                    n += 1

    @always_inline
    def async_multicast_load[
        cta_group: Int = 1,  # Use SM90-style TMA for cluster 1x1x1
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: LayoutTensor[mut=True, Self.dtype, _, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int],
        multicast_mask: UInt16,
    ):
        """Schedules an asynchronous im2col TMA load with multicast.

        Uses 2D GEMM-style coordinates:
        - coords[0]: K coordinate (indexes into C * R * S reduction dimension)
        - coords[1]: M coordinate (indexes into batch * H_out * W_out spatial)

        Internally:
        - K is decomposed into (c, r, s) where K = c*R*S + r*S + s
        - M is decomposed into (n, h, w) where M = n*H_out*W_out + h*W_out + w
        - 4D coordinates (c, w, h, n) and filter offsets (s, r) are passed to
          the PTX im2col instruction with multicast.

        Note: The cta_group parameter defaults to 2 because SM100/Blackwell
        im2col TMA with padding (negative corners) requires the cta_group::2
        PTX format. This is consistent with CUTLASS which only provides
        SM100_TMA_2SM_LOAD_IM2COL_MULTICAST (no cta_group::1 variant).

        Parameters:
            cta_group: CTA group size for TMA operations.
            eviction_policy: Cache eviction policy for the TMA load.

        Args:
            dst: Destination tensor in shared memory.
            mem_barrier: Memory barrier for synchronization.
            coords: GEMM coordinates (k_coord, m_coord).
            multicast_mask: Bitmask specifying target CTAs for multicast.
        """
        comptime assert (
            type_of(dst).alignment % 128 == 0
        ), "TMA requires 128B alignment in shared memory"

        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = Self.tile_shape[0] // copy_dim0
        comptime num_copies_dim1 = Self.tile_shape[1] // copy_dim1

        # Precompute spatial size for M decomposition
        var hw = Int(self.out_height) * Int(self.out_width)
        var out_w = Int(self.out_width)

        # Precompute filter window size for K decomposition
        # K = r * S * C + s * C + c (filter-first, channel-last ordering for NHWC)
        var num_channels = Int(self.in_channels)
        var filter_w = Int(self.filter_w)

        # OPTIMIZATION: Hoist K decomposition outside loop (constant when j=0).
        var k_coord = coords[0]
        var filter_idx, c = udivmod(k_coord, num_channels)
        var r, s = udivmod(filter_idx, filter_w)

        # Initial M decomposition (done once, then use iterator)
        var m_coord_init = coords[1]
        var n, m_remainder = udivmod(m_coord_init, hw)
        var h_out, w_out = udivmod(m_remainder, out_w)

        # Pre-add lower_corner offset
        var h = h_out + Int(self.lower_corner_h)
        var w = w_out + Int(self.lower_corner_w)

        # Cache bounds for iterator wraparound
        var out_h_int = Int(self.out_height)
        var lower_h = Int(self.lower_corner_h)
        var lower_w = Int(self.lower_corner_w)

        comptime for i in range(num_copies_dim0):
            comptime for j in range(num_copies_dim1):
                comptime copy_offset: UInt32 = UInt32(
                    (i * num_copies_dim1 + j) * copy_size
                )

                # K recomputation only needed when j > 0
                comptime if j > 0:
                    k_coord = coords[0] + j * copy_dim1
                    filter_idx, c = udivmod(k_coord, num_channels)
                    r, s = udivmod(filter_idx, filter_w)

                # Pass 4D coords (c, w, h, n) and filter offsets (s, r) to im2col PTX
                cp_async_bulk_tensor_shared_cluster_global_im2col_multicast[
                    cta_group=cta_group,
                ](
                    dst.ptr + copy_offset,
                    Pointer(to=self.descriptor).bitcast[NoneType](),
                    mem_barrier.unsafe_ptr(),
                    Index(c, w, h, n),
                    Index(s, r),
                    multicast_mask,
                )

            # Iterator pattern: advance M by copy_dim0 using addition
            w += copy_dim0
            if w >= out_w + lower_w:
                w -= out_w
                h += 1
                if h >= out_h_int + lower_h:
                    h -= out_h_int
                    n += 1

    @always_inline
    def async_copy[
        cta_group: Int = 1,  # Use SM90-style TMA for cluster 1x1x1
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: TileTensor[mut=True, Self.dtype, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int],
    ):
        """Schedules an asynchronous im2col TMA load.

        TileTensor overload - accepts TileTensor instead of LayoutTensor.
        Assumes 128B alignment (TileTensor tiles are allocated with proper alignment).

        Uses 2D GEMM-style coordinates:
        - coords[0]: K coordinate (indexes into C * R * S reduction dimension)
        - coords[1]: M coordinate (indexes into batch * H_out * W_out spatial)

        Internally:
        - K is decomposed into (c, r, s) where K = c*R*S + r*S + s
        - M is decomposed into (n, h, w) where M = n*H_out*W_out + h*W_out + w
        - 4D coordinates (c, w, h, n) and filter offsets (s, r) are passed to
          the PTX im2col instruction.

        Note: Uses cta_group=1 (SM90-style TMA) for single-CTA clusters.

        Parameters:
            cta_group: CTA group size for TMA operations.
            eviction_policy: Cache eviction policy for the TMA load.

        Args:
            dst: TileTensor in shared memory where data will be copied.
            mem_barrier: Memory barrier for synchronization.
            coords: GEMM coordinates (k_coord, m_coord).
        """
        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = Self.tile_shape[0] // copy_dim0
        comptime num_copies_dim1 = Self.tile_shape[1] // copy_dim1

        # Precompute spatial size for M decomposition
        var hw = Int(self.out_height) * Int(self.out_width)
        var out_w = Int(self.out_width)

        # Precompute filter window size for K decomposition
        # K = r * S * C + s * C + c (filter-first, channel-last ordering for NHWC)
        var num_channels = Int(self.in_channels)
        var filter_w = Int(self.filter_w)

        # OPTIMIZATION: Hoist K decomposition outside loop (constant when j=0).
        var k_coord = coords[0]
        var filter_idx, c = udivmod(k_coord, num_channels)
        var r, s = udivmod(filter_idx, filter_w)

        # Initial M decomposition (done once, then use iterator)
        var m_coord_init = coords[1]
        var n, m_remainder = udivmod(m_coord_init, hw)
        var h_out, w_out = udivmod(m_remainder, out_w)

        # Pre-add lower_corner offset
        var h = h_out + Int(self.lower_corner_h)
        var w = w_out + Int(self.lower_corner_w)

        # Cache bounds for iterator wraparound
        var out_h_int = Int(self.out_height)
        var lower_h = Int(self.lower_corner_h)
        var lower_w = Int(self.lower_corner_w)

        comptime for i in range(num_copies_dim0):
            comptime for j in range(num_copies_dim1):
                comptime copy_offset: UInt32 = UInt32(
                    (i * num_copies_dim1 + j) * copy_size
                )

                # K recomputation only needed when j > 0
                comptime if j > 0:
                    k_coord = coords[0] + j * copy_dim1
                    filter_idx, c = udivmod(k_coord, num_channels)
                    r, s = udivmod(filter_idx, filter_w)

                # Pass 4D coords (c, w, h, n) and filter offsets (s, r) to im2col PTX
                cp_async_bulk_tensor_shared_cluster_global_im2col[
                    cta_group=cta_group,
                ](
                    dst.ptr + copy_offset,
                    Pointer(to=self.descriptor).bitcast[NoneType](),
                    mem_barrier.unsafe_ptr(),
                    Index(c, w, h, n),
                    Index(s, r),
                )

            # Iterator pattern: advance M by copy_dim0 using addition
            w += copy_dim0
            if w >= out_w + lower_w:
                w -= out_w
                h += 1
                if h >= out_h_int + lower_h:
                    h -= out_h_int
                    n += 1

    @always_inline
    def async_multicast_load[
        cta_group: Int = 1,  # Use SM90-style TMA for cluster 1x1x1
        eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    ](
        self,
        dst: TileTensor[mut=True, Self.dtype, address_space=.SHARED, ...],
        ref[AddressSpace.SHARED] mem_barrier: SharedMemBarrier,
        coords: Tuple[Int, Int],
        multicast_mask: UInt16,
    ):
        """Schedules an asynchronous im2col TMA load with multicast.

        TileTensor overload - accepts TileTensor instead of LayoutTensor.
        Assumes 128B alignment (TileTensor tiles are allocated with proper alignment).

        Uses 2D GEMM-style coordinates:
        - coords[0]: K coordinate (indexes into C * R * S reduction dimension)
        - coords[1]: M coordinate (indexes into batch * H_out * W_out spatial)

        Internally:
        - K is decomposed into (c, r, s) where K = c*R*S + r*S + s
        - M is decomposed into (n, h, w) where M = n*H_out*W_out + h*W_out + w
        - 4D coordinates (c, w, h, n) and filter offsets (s, r) are passed to
          the PTX im2col instruction with multicast.

        Note: Uses cta_group=1 (SM90-style TMA) for single-CTA clusters.

        Parameters:
            cta_group: CTA group size for TMA operations.
            eviction_policy: Cache eviction policy for the TMA load.

        Args:
            dst: TileTensor in shared memory where data will be copied.
            mem_barrier: Memory barrier for synchronization.
            coords: GEMM coordinates (k_coord, m_coord).
            multicast_mask: Bitmask specifying target CTAs for multicast.
        """
        comptime copy_dim0 = Self.desc_shape[0]
        comptime copy_dim1 = Self.desc_shape[1]
        comptime copy_size = _idx_product[Self.rank, Self.desc_shape]()
        comptime num_copies_dim0 = Self.tile_shape[0] // copy_dim0
        comptime num_copies_dim1 = Self.tile_shape[1] // copy_dim1

        # Precompute spatial size for M decomposition
        var hw = Int(self.out_height) * Int(self.out_width)
        var out_w = Int(self.out_width)

        # Precompute filter window size for K decomposition
        # K = r * S * C + s * C + c (filter-first, channel-last ordering for NHWC)
        var num_channels = Int(self.in_channels)
        var filter_w = Int(self.filter_w)

        # OPTIMIZATION: Hoist K decomposition outside loop (constant when j=0).
        var k_coord = coords[0]
        var filter_idx, c = udivmod(k_coord, num_channels)
        var r, s = udivmod(filter_idx, filter_w)

        # Initial M decomposition (done once, then use iterator)
        var m_coord_init = coords[1]
        var n, m_remainder = udivmod(m_coord_init, hw)
        var h_out, w_out = udivmod(m_remainder, out_w)

        # Pre-add lower_corner offset
        var h = h_out + Int(self.lower_corner_h)
        var w = w_out + Int(self.lower_corner_w)

        # Cache bounds for iterator wraparound
        var out_h_int = Int(self.out_height)
        var lower_h = Int(self.lower_corner_h)
        var lower_w = Int(self.lower_corner_w)

        comptime for i in range(num_copies_dim0):
            comptime for j in range(num_copies_dim1):
                comptime copy_offset: UInt32 = UInt32(
                    (i * num_copies_dim1 + j) * copy_size
                )

                # K recomputation only needed when j > 0
                comptime if j > 0:
                    k_coord = coords[0] + j * copy_dim1
                    filter_idx, c = udivmod(k_coord, num_channels)
                    r, s = udivmod(filter_idx, filter_w)

                # Pass 4D coords (c, w, h, n) and filter offsets (s, r) to im2col PTX
                cp_async_bulk_tensor_shared_cluster_global_im2col_multicast[
                    cta_group=cta_group,
                ](
                    dst.ptr + copy_offset,
                    Pointer(to=self.descriptor).bitcast[NoneType](),
                    mem_barrier.unsafe_ptr(),
                    Index(c, w, h, n),
                    Index(s, r),
                    multicast_mask,
                )

            # Iterator pattern: advance M by copy_dim0 using addition
            w += copy_dim0
            if w >= out_w + lower_w:
                w -= out_w
                h += 1
                if h >= out_h_int + lower_h:
                    h -= out_h_int
                    n += 1


def _im2col_desc_shape[
    dtype: DType,
    tile_shape: IndexList[2],
    swizzle_mode: TensorMapSwizzle,
]() -> IndexList[2]:
    """Compute the im2col descriptor shape as an IndexList[2]."""
    comptime swizzle_bytes = (
        16 if swizzle_mode
        == TensorMapSwizzle.SWIZZLE_NONE else (
            32 if swizzle_mode
            == TensorMapSwizzle.SWIZZLE_32B else (
                64 if swizzle_mode == TensorMapSwizzle.SWIZZLE_64B else 128
            )
        )
    )
    comptime element_size = size_of[dtype]()
    comptime swizzle_width = swizzle_bytes // element_size
    comptime k_tile = tile_shape[1]
    comptime channels_per_pixel = swizzle_width if swizzle_width < k_tile else k_tile
    comptime max_tma_box_elements = 256
    comptime m_tile = tile_shape[0]
    comptime max_pixels_from_box = max_tma_box_elements // channels_per_pixel
    comptime pixels_per_column = (
        m_tile if m_tile < max_pixels_from_box else max_pixels_from_box
    )
    return IndexList[2](pixels_per_column, channels_per_pixel)


@always_inline
def _build_im2col_descriptor[
    dtype: DType,
    swizzle_mode: TensorMapSwizzle,
    *,
    __tile_shape: IndexList[2],
    __desc_shape: IndexList[2],
](
    ctx: DeviceContext,
    ptr: ImmPointer[Scalar[dtype], _],
    batch: Int,
    height: Int,
    width: Int,
    channels: Int,
    lower_corner_h: Int,
    lower_corner_w: Int,
    upper_corner_h: Int,
    upper_corner_w: Int,
    out_height: Int,
    out_width: Int,
    filter_h: Int,
    filter_w: Int,
) raises -> TMATensorTileIm2col[dtype, 2, __tile_shape, __desc_shape]:
    """Shared implementation for building an im2col TMA descriptor.

    Both the LayoutTensor and TileTensor overloads of
    `create_tensor_tile_im2col` delegate here after extracting dimensions
    from their respective tensor types.
    """
    var global_buf = DeviceBuffer(
        ctx,
        ptr,
        1,
        owning=False,
    )

    var global_shape = IndexList[4](batch, height, width, channels)

    # Row-major NHWC strides: stride(i) = product of all dims after i
    var global_strides = IndexList[4](
        height * width * channels,
        width * channels,
        channels,
        1,
    )

    var lower_corner = IndexList[2](lower_corner_h, lower_corner_w)
    var upper_corner = IndexList[2](upper_corner_h, upper_corner_w)

    comptime pixels_per_column = __desc_shape[0]
    comptime channels_per_pixel = __desc_shape[1]

    var swizzle = _SwizzleMode(Int32(Int(swizzle_mode)))

    var tensormap = _create_tensormap_im2col[dtype, 4, 2](
        global_buf,
        global_shape,
        global_strides,
        lower_corner,
        upper_corner,
        channels_per_pixel,
        pixels_per_column,
        swizzle,
    )

    # TensorMap and TMADescriptor are both 128-byte aligned with the same layout
    var descriptor = TMADescriptor()
    descriptor.data = tensormap.data

    return TMATensorTileIm2col[dtype, 2, __tile_shape, __desc_shape](
        descriptor,
        UInt32(out_height),
        UInt32(out_width),
        UInt32(filter_h),
        UInt32(filter_w),
        UInt32(channels),
        Int32(lower_corner_h),
        Int32(lower_corner_w),
    )


@always_inline
def create_tensor_tile_im2col[
    dtype: DType,
    tile_shape: IndexList[2],  # [M_tile, K_tile] = [pixels, channels]
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    *,
    __tile_shape: IndexList[2] = tile_shape,
    __desc_shape: IndexList[2] = _im2col_desc_shape[
        dtype, tile_shape, swizzle_mode
    ](),
](
    ctx: DeviceContext,
    tensor: LayoutTensor[
        mut=True, dtype, address_space=.GENERIC, ...
    ],  # 4D NHWC tensor
    lower_corner_h: Int,
    lower_corner_w: Int,
    upper_corner_h: Int,
    upper_corner_w: Int,
    out_height: Int,
    out_width: Int,
    filter_h: Int,
    filter_w: Int,
) raises -> TMATensorTileIm2col[dtype, 2, __tile_shape, __desc_shape]:
    """Creates a TMA tensor tile with im2col transformation for 2D convolution.

    This factory function creates a TMA descriptor that performs hardware
    im2col transformation during loads. The descriptor encodes the convolution
    geometry and the TMA hardware computes addresses on-the-fly.

    For im2col TMA, each transaction loads one output pixel with multiple channels.
    This follows CUTLASS's approach where:
    - pixels_per_column = 1 (one pixel per TMA transaction)
    - channels_per_pixel = min(K_tile, swizzle_width) (contiguous channels)

    Parameters:
        dtype: The data type of tensor elements.
        tile_shape: Shape `[M_tile, K_tile]` for the GEMM tile.
            - M_tile: Number of output pixels (batch * H_out * W_out slice).
            - K_tile: Number of channels (C_in * R * S slice for filter).
        swizzle_mode: Memory swizzling pattern.
        __tile_shape: Internal parameter for the tile shape.
        __desc_shape: Internal parameter for the descriptor shape.

    Args:
        ctx: The CUDA device context.
        tensor: The 4D activation tensor in NHWC layout.
        lower_corner_h: Lower corner offset for height (negative for padding).
        lower_corner_w: Lower corner offset for width (negative for padding).
        upper_corner_h: Upper corner offset for height.
        upper_corner_w: Upper corner offset for width.
        out_height: Output height (H_out) for M coordinate decomposition.
        out_width: Output width (W_out) for M coordinate decomposition.
        filter_h: Filter height (R) for K coordinate decomposition.
        filter_w: Filter width (S) for K coordinate decomposition.

    Returns:
        A TMATensorTileIm2col configured for im2col loads.

    Raises:
        Error if TMA descriptor creation fails.

    Note:
        For stride=1, dilation=1 convolution with padding (following CUTLASS convention):
        - lower_corner_h = -pad_h
        - lower_corner_w = -pad_w
        - upper_corner_h = pad_h - (filter_h - 1)
        - upper_corner_w = pad_w - (filter_w - 1)

        The filter offsets passed to the PTX instruction range from 0 to (filter_size - 1)
        and are added to lower_corner to compute actual input coordinates.
    """
    comptime assert tensor.rank == 4, "Im2col TMA requires 4D NHWC tensor"

    # The helper hardcodes row-major strides from dims; verify the tensor
    # is actually contiguous so the strides match.
    var h = Int(tensor.dim(1))
    var w = Int(tensor.dim(2))
    var c = Int(tensor.dim(3))
    debug_assert(
        tensor.stride(3) == 1
        and tensor.stride(2) == c
        and tensor.stride(1) == w * c
        and tensor.stride(0) == h * w * c,
        "im2col TMA requires a contiguous NHWC tensor",
    )

    return _build_im2col_descriptor[
        swizzle_mode=swizzle_mode,
        __tile_shape=__tile_shape,
        __desc_shape=__desc_shape,
    ](
        ctx,
        tensor.ptr,
        Int(tensor.dim(0)),
        h,
        w,
        c,
        lower_corner_h,
        lower_corner_w,
        upper_corner_h,
        upper_corner_w,
        out_height,
        out_width,
        filter_h,
        filter_w,
    )


@always_inline
def create_tensor_tile_im2col[
    dtype: DType,
    tile_shape: IndexList[2],  # [M_tile, K_tile] = [pixels, channels]
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    *,
    __tile_shape: IndexList[2] = tile_shape,
    __desc_shape: IndexList[2] = _im2col_desc_shape[
        dtype, tile_shape, swizzle_mode
    ](),
](
    ctx: DeviceContext,
    tensor: TileTensor[
        mut=True, dtype, address_space=.GENERIC, ...
    ],  # 4D NHWC tensor
    lower_corner_h: Int,
    lower_corner_w: Int,
    upper_corner_h: Int,
    upper_corner_w: Int,
    out_height: Int,
    out_width: Int,
    filter_h: Int,
    filter_w: Int,
) raises -> TMATensorTileIm2col[dtype, 2, __tile_shape, __desc_shape]:
    """Creates a TMA tensor tile with im2col transformation for 2D convolution.

    TileTensor overload: delegates to the shared `_build_im2col_descriptor`
    helper. See the LayoutTensor overload for full background.

    Parameters:
        dtype: The data type of tensor elements.
        tile_shape: Shape `[M_tile, K_tile]` for the GEMM tile.
        swizzle_mode: Memory swizzling pattern.
        __tile_shape: Internal parameter for the tile shape.
        __desc_shape: Internal parameter for the descriptor shape.

    Args:
        ctx: The CUDA device context.
        tensor: The 4D activation tensor in NHWC layout.
        lower_corner_h: Lower corner offset for height (negative for padding).
        lower_corner_w: Lower corner offset for width (negative for padding).
        upper_corner_h: Upper corner offset for height.
        upper_corner_w: Upper corner offset for width.
        out_height: Output height (H_out) for M coordinate decomposition.
        out_width: Output width (W_out) for M coordinate decomposition.
        filter_h: Filter height (R) for K coordinate decomposition.
        filter_w: Filter width (S) for K coordinate decomposition.

    Returns:
        A TMATensorTileIm2col configured for im2col loads.

    Raises:
        Error if TMA descriptor creation fails.
    """
    comptime assert tensor.rank == 4, "Im2col TMA requires 4D NHWC tensor"

    return _build_im2col_descriptor[
        swizzle_mode=swizzle_mode,
        __tile_shape=__tile_shape,
        __desc_shape=__desc_shape,
    ](
        ctx,
        tensor.ptr,
        Int(tensor.dim[0]()),
        Int(tensor.dim[1]()),
        Int(tensor.dim[2]()),
        Int(tensor.dim[3]()),
        lower_corner_h,
        lower_corner_w,
        upper_corner_h,
        upper_corner_w,
        out_height,
        out_width,
        filter_h,
        filter_w,
    )
