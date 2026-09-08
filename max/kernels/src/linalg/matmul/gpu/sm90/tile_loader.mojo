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

"""TileLoader module for efficient tile loading in GPU matrix multiplication.

This module provides utilities for loading matrix tiles from global memory to
shared memory using two different mechanisms:

1. TMA (Tensor Memory Accelerator): Hardware-accelerated loads that can efficiently
   transfer 2D tiles with multicast support for multi-block clusters.

2. cp.async: Software-based asynchronous copy instructions with manual bounds
   checking and swizzling for optimal shared memory access patterns.

The TileLoader struct abstracts these loading mechanisms to provide a unified
interface for the matmul kernel's producer threads.
"""
from layout.tma_async import TMATensorTile, _idx_product
from layout import (
    Coord,
    Idx,
    MixedLayout,
    DefaultEngine,
    TensorLayout,
    TensorEngine,
    TileTensor,
)
from max.gpu.memory import (
    async_copy,
)
from ....structuring import SMemBarrier
from layout.swizzle import make_swizzle
from max.gpu import thread_idx
from max.gpu.globals import WARPGROUP_SIZE
from max.gpu.sync import async_copy_arrive
from structured_kernels.pipeline import (
    ProducerConsumerPipeline,
)
from std.sys import simd_width_of, size_of
from std.utils.index import IndexList
from max.gpu.host.nvidia.tma import TensorMapSwizzle


trait TileLoader(TrivialRegisterPassable):
    """Base trait for tile loading mechanisms in matrix multiplication.

    This trait defines the interface for loading tiles from global memory
    to shared memory, abstracting over different hardware mechanisms.
    """

    comptime _dtype: DType

    @always_inline
    def load_tile(
        self,
        dst: TileTensor[
            mut=True,
            address_space=.SHARED,
            Engine=DefaultEngine[element_width=1],
            ...,
        ],
        mem_barrier: SMemBarrier,
        coords: Tuple[Int, Int],
    ):
        """Load a tile from global memory to shared memory.

        Args:
            dst: Destination tile in shared memory (must be 128-byte aligned).
            mem_barrier: Memory barrier for synchronization.
            coords: Tile coordinates (row, column) in the source matrix.
        """
        ...


trait BarrierHandler(TrivialRegisterPassable):
    """Handles barrier lifecycle for different transfer mechanisms.

    Separates barrier management from tile loading:
    - prepare_stage: Called once before loading tiles for a stage.
    - complete_stage: Called once after all tiles for a stage are loaded.

    TMA: prepare sets expected bytes, complete is noop (hardware signals).
    cp.async: prepare is noop, complete commits copies and signals arrival.
    """

    @always_inline
    def prepare_stage(self, mem_barrier: SMemBarrier):
        """Prepare barrier for incoming transfers.

        For TMA: sets expected transaction bytes.
        For cp.async: noop.

        Args:
            mem_barrier: The stage's memory barrier.
        """
        ...

    @always_inline
    def complete_stage(self, mem_barrier: SMemBarrier):
        """Signal that all transfers for this stage are done.

        For TMA: noop (hardware auto-signals).
        For cp.async: commits pending copies and signals thread arrival.

        Args:
            mem_barrier: The stage's memory barrier.
        """
        ...


struct TMABarrierHandler[expected_bytes: Int](BarrierHandler):
    """TMA barrier handler: sets expected bytes on prepare, noop on complete.

    Initializes the pipeline on construction (phase=0, barrier counts).

    Parameters:
        expected_bytes: Total bytes expected per stage across all loaders.
    """

    def __init__[
        num_stages: Int
    ](
        out self,
        mut pipeline: ProducerConsumerPipeline[num_stages],
        num_consumers: Int,
        cluster_size: Int,
    ):
        pipeline._producer_phase = 0
        if thread_idx.x == 0:
            pipeline.init_mbars(
                producer_arrive_count=1,
                consumer_arrive_count=Int32(num_consumers * cluster_size),
            )

    @always_inline
    def prepare_stage(self, mem_barrier: SMemBarrier):
        mem_barrier[].expect_bytes(Int32(Self.expected_bytes))

    @always_inline
    def complete_stage(self, mem_barrier: SMemBarrier):
        pass


struct CPAsyncBarrierHandler(BarrierHandler):
    """The cp.async barrier handler: noop on prepare, arrives on complete.

    Initializes the pipeline on construction (phase=0, barrier counts).
    """

    def __init__[
        num_stages: Int
    ](
        out self,
        mut pipeline: ProducerConsumerPipeline[num_stages],
        num_consumers: Int,
        cluster_size: Int,
    ):
        pipeline._producer_phase = 0
        if thread_idx.x == 0:
            pipeline.init_mbars(
                producer_arrive_count=Int32(WARPGROUP_SIZE),
                consumer_arrive_count=Int32(num_consumers * cluster_size),
            )

    @always_inline
    def prepare_stage(self, mem_barrier: SMemBarrier):
        pass

    @always_inline
    def complete_stage(self, mem_barrier: SMemBarrier):
        async_copy_arrive(mem_barrier)
        _ = mem_barrier[].arrive()


struct TileLoaderTMA[
    tma_origin: ImmOrigin,
    dtype: DType,
    tma_rank: Int,
    tile_shape: IndexList[tma_rank],
    desc_shape: IndexList[tma_rank],
    /,
    *,
    BK: Int,
    cluster_size: Int32,
    use_partitioned_multicast: Bool,
](TileLoader):
    """TMA-based tile loader for hardware-accelerated memory transfers.

    This loader uses NVIDIA's Tensor Memory Accelerator (TMA) for efficient
    2D tile transfers from global to shared memory, with optional multicast
    support for multi-block clusters.

    Parameters:
        tma_origin: Origin type for the TMA operation.
        dtype: Data type of the elements being loaded.
        tma_rank: Rank of the TMA tile (number of dimensions).
        tile_shape: Shape of the complete tile in shared memory.
        desc_shape: Shape described by the TMA descriptor (may be smaller).
        BK: Block size in the K dimension (for coordinate conversion).
        cluster_size: Number of blocks in the cluster (1 for no clustering).
        use_partitioned_multicast: Whether to use partitioned multicast loading.
    """

    comptime _dtype = Self.dtype

    comptime TMATensorTilePtr = Pointer[
        TMATensorTile[
            Self.dtype, Self.tma_rank, Self.tile_shape, Self.desc_shape
        ],
        Self.tma_origin,
    ]
    var tma_op: Self.TMATensorTilePtr
    var rank: Int
    var multicast_mask: UInt16

    @always_inline
    def __init__(
        out self,
        tma_op: Self.TMATensorTilePtr,
        rank: Int,
        multicast_mask: UInt16,
    ):
        """Initialize the TMA tile loader.

        Args:
            tma_op: Pointer to the TMA tensor descriptor.
            rank: Rank of this block within the cluster.
            multicast_mask: Bit mask for multicast targets.
        """
        self.tma_op = tma_op
        self.rank = rank
        self.multicast_mask = multicast_mask

    @always_inline
    def load_tile(
        self,
        dst: TileTensor[
            mut=True,
            address_space=.SHARED,
            Engine=DefaultEngine[element_width=1],
            ...,
        ],
        mem_barrier: SMemBarrier,
        _coords: Tuple[Int, Int],
    ):
        """Load a tile using TMA hardware acceleration.

        Converts tile indices to element coordinates and initiates a TMA
        transfer. For clusters, uses multicast to share data across blocks.

        Args:
            dst: Destination tile in shared memory.
            mem_barrier: Memory barrier for synchronization.
            _coords: Tile coordinates (row_tile_idx, col_tile_idx).

        Note:
            Coordinates are converted from (row, col) tile indices to
            (k_elements, row/col_elements) for TMA's K-major ordering.
        """
        comptime assert type_of(dst).dtype == Self._dtype
        # Materialize the inferred destination as an exact TileTensor type for
        # TMA overload resolution. The trait method accepts any shared-memory
        # TileTensor, but TMATensorTile is parameterized on Self._dtype.
        var dst_exact = TileTensor[
            mut=True,
            Self._dtype,
            LayoutType=type_of(dst).LayoutType,
            origin=MutAnyOrigin,
            address_space=.SHARED,
            linear_idx_type=type_of(dst).linear_idx_type,
        ](
            dst._storage.as_unsafe_any_origin().bitcast[Scalar[Self._dtype]](),
            dst.layout,
        )

        # Switch coordinates to k-minor and multiply k by BK to match the CPAsync API.
        var coords = (
            _coords[1] * Self.BK,
            _coords[0],
        )  # (m/n, k) -> (k, m/n)

        comptime tma_load_size = _idx_product[Self.tma_rank, Self.desc_shape]()
        comptime tma_rows = Self.desc_shape[0]

        comptime if Self.cluster_size > 1:
            # Multi-block cluster: Use multicast to share data across blocks

            comptime if Self.use_partitioned_multicast:
                # Partitioned multicast: Each block loads a portion of the tile
                # This is more efficient for large tiles as it distributes the load
                self.tma_op[].async_multicast_load_partitioned[
                    tma_rows, tma_load_size
                ](
                    dst_exact,
                    mem_barrier[],
                    self.rank,
                    coords,
                    self.multicast_mask,
                )

            else:
                # Standard multicast: Only rank 0 loads and broadcasts to others
                # This is simpler but can create a bottleneck for large tiles
                if self.rank == 0:
                    self.tma_op[].async_multicast_load(
                        dst_exact,
                        mem_barrier[],
                        coords,
                        self.multicast_mask,
                    )

        else:
            # Single block: Direct TMA copy without multicast overhead
            self.tma_op[].async_copy(
                dst_exact,
                mem_barrier[],
                (coords[0], coords[1]),
            )


struct TileLoaderCPAsync[
    dtype: DType,
    src_layout: TensorLayout,
    thread_layout: MixedLayout,
    swizzle_mode: TensorMapSwizzle,
    vector_size: Int,
    src_engine: TensorEngine = DefaultEngine[element_width=1],
](TileLoader):
    """Software-based tile loader using cp.async instructions.

    This loader uses CUDA's cp.async instructions for asynchronous memory
    transfers with manual bounds checking and shared memory swizzling for
    optimal bank conflict avoidance.

    Parameters:
        dtype: Data type of the elements being loaded.
        src_layout: Layout of the source matrix in global memory.
        thread_layout: Thread arrangement for distributed copying.
        swizzle_mode: Swizzling pattern for shared memory access.
        vector_size: Number of elements loaded per thread.
        src_engine: Engine of the source tensor (defaults to
            `DefaultEngine`).
    """

    comptime _dtype = Self.dtype

    @__allow_legacy_any_origin_fields
    var src: TileTensor[
        mut=False,
        Self.dtype,
        LayoutType=Self.src_layout,
        origin=ImmutAnyOrigin,
        address_space=.GENERIC,
        Engine=Self.src_engine,
    ]

    @always_inline
    def __init__(
        out self,
        src: TileTensor[
            mut=False,
            Self.dtype,
            LayoutType=Self.src_layout,
            origin=ImmutAnyOrigin,
            address_space=.GENERIC,
            Engine=Self.src_engine,
        ],
    ):
        """Initialize the cp.async tile loader.

        Args:
            src: Source tensor in global memory.
        """
        self.src = src

    def load_tile(
        self,
        dst: TileTensor[
            mut=True,
            address_space=.SHARED,
            Engine=DefaultEngine[element_width=1],
            ...,
        ],
        mem_barrier: SMemBarrier,
        coords: Tuple[Int, Int],
    ):
        """Load a tile using cp.async instructions.

        Extracts a tile from the source tensor and performs an asynchronous
        copy to shared memory with bounds checking and swizzling.

        Args:
            dst: Destination tile in shared memory.
            mem_barrier: Memory barrier for synchronization (currently unused).
            coords: Tile indices (row_tile, col_tile) in the source matrix.

        Note:
            Unlike TMA, this method expects tile indices and handles the
            conversion to element offsets internally via the tile() method.
        """
        comptime assert type_of(dst).dtype == Self._dtype

        # Use the swizzle width as the contiguous copy width and derive rows
        # from the destination tile element count.
        comptime BN = Self.swizzle_mode.bytes() // size_of[Self.dtype]()
        comptime BM = type_of(dst).LayoutType.static_product // BN

        # Extract the requested tile from global memory and vectorize it
        var a_gmem_tile = self.src.tile[BM, BN](
            Coord(coords[0], coords[1])
        ).vectorize[1, Self.vector_size]()

        # Perform the async copy with bounds checking and swizzling. Rebind
        # through an exact destination type so the dtype parameter can unify
        # across the source and destination TileTensor arguments.
        # Materialize an exact, any-origin destination tile from the raw
        # pointer (non-vectorized `DefaultEngine[element_width=1]`), then vectorize it.
        # Vectorized tiles cannot be reconstructed directly from a pointer.
        var dst_exact = TileTensor[
            mut=True,
            Self._dtype,
            LayoutType=type_of(dst).LayoutType,
            origin=MutAnyOrigin,
            address_space=.SHARED,
            linear_idx_type=type_of(dst).linear_idx_type,
        ](
            dst._storage.as_unsafe_any_origin().bitcast[Scalar[Self._dtype]](),
            dst.layout,
        )
        var dst_vec = dst_exact.vectorize[1, Self.vector_size]()
        async_copy_with_bound_check[
            Self.thread_layout,
            Self.swizzle_mode,
        ](a_gmem_tile, dst_vec)


@always_inline
def async_copy_with_bound_check[
    dtype: DType,
    src_layout: TensorLayout,
    dst_layout: TensorLayout,
    src_element_width: Int,
    dst_element_width: Int,
    //,
    thread_layout: MixedLayout,
    swizzle_mode: TensorMapSwizzle,
](
    src: TileTensor[
        mut=False,
        dtype,
        LayoutType=src_layout,
        origin=ImmutAnyOrigin,
        address_space=.GENERIC,
        Engine=DefaultEngine[element_width=src_element_width],
        ...,
    ],
    dst: TileTensor[
        mut=True,
        dtype,
        LayoutType=dst_layout,
        origin=MutAnyOrigin,
        address_space=.SHARED,
        Engine=DefaultEngine[element_width=dst_element_width],
        ...,
    ],
):
    """Helper function for cp.async with boundary checking.

    This method performs element-wise async copies with per-element boundary
    checking. Out-of-bounds accesses are automatically zero-filled, ensuring
    safe operation near matrix edges.

    The method also handles shared memory swizzling to avoid bank conflicts
    and maximize memory bandwidth utilization.

    Parameters:
        dtype: Element type of the source and destination tiles (inferred).
        src_layout: Static layout of the source tile in global memory (inferred).
        dst_layout: Static layout of the destination tile in shared memory
            (inferred).
        thread_layout: Thread mapping that partitions the source and
            destination tiles across threads.
        swizzle_mode: Shared memory swizzle pattern applied to avoid bank
            conflicts.

    Args:
        src: Source tensor fragment in global memory.
        dst: Destination tensor fragment in shared memory.
    """
    comptime assert src.rank == 2, "Global memory tile must be rank 2."

    comptime assert (
        src_layout.static_product == dst_layout.static_product
    ), "Global memory tile must match source layout element count"

    # Validate swizzle pattern alignment with tile dimensions
    comptime src_shape1 = src_layout.static_shape[1]
    comptime swizzle_bytes = swizzle_mode.bytes()
    comptime assert (
        src_shape1 * src.element_size * size_of[src.dtype]() == swizzle_bytes
    ), String(
        "Global memory tile shape-1 ",
        src_shape1 * src.element_size,
        "must match swizzle bytes.",
        swizzle_bytes,
    )

    # Distribute work across threads according to thread_layout
    var src_frag = src.distribute[thread_layout](thread_idx.x)
    var dst_frag = dst.distribute[thread_layout](thread_idx.x)

    # Source matrix bounds for boundary checking
    comptime src_stride0 = src_layout.static_stride[0]
    var src_bound0 = Int32(src.dim[0]())
    var src_bound1 = Int32(src.dim[1]()) * Int32(dst.element_size)

    # Calculate base coordinates for this thread's destination fragment
    var dst_frag_offset = (
        Int(dst_frag._storage) - Int(dst._storage)
    ) // size_of[dtype]()
    comptime dst_stride0 = dst_layout.static_stride[0]
    var dst_frag_base_coord0, dst_frag_base_coord1 = divmod(
        Int32(dst_frag_offset), Int32(dst_stride0)
    )

    # Create swizzle pattern to avoid shared memory bank conflicts
    comptime swizzle = make_swizzle[
        8,
        swizzle_bytes // size_of[dst.dtype](),
        simd_width_of[dst.dtype](),
    ]()

    comptime num_vecs = type_of(dst_frag).LayoutType.static_product

    # Process each vector element assigned to this thread
    comptime for i in range(num_vecs):
        # Apply swizzling to the destination index to avoid bank conflicts
        var dst_idx = Int(type_of(dst_frag).LayoutType()(Coord(Idx[i], Idx[0])))
        var dst_idx_base = dst_idx % swizzle.size()
        var dst_idx_diff = dst_idx - dst_idx_base
        var dst_swizzled_idx = Int32(
            swizzle(Scalar[dst.linear_idx_type](dst_frag_offset + dst_idx_base))
            + Scalar[dst.linear_idx_type](dst_idx_diff)
        )
        var dst_ptr = dst._storage.bitcast[Scalar[dtype]]() + Int(
            dst_swizzled_idx
        )

        # Calculate the 2D coordinates for this element
        # TODO: we should be able to use idx2crd for this.
        var dst_shifted_coord0, dst_shifted_coord1 = divmod(
            dst_idx, dst_stride0
        )
        var dst_coord0 = Int32(dst_shifted_coord0) + dst_frag_base_coord0
        var dst_coord1 = Int32(dst_shifted_coord1) + dst_frag_base_coord1

        comptime size_bytes = dst.element_size * size_of[dst.dtype]()

        # Calculate source pointer based on 2D coordinates
        var src_ptr = (
            src._storage.bitcast[Scalar[dtype]]().address_space_cast[.GLOBAL]()
            + dst_coord1
            + dst_coord0 * Int32(src_stride0)
        )

        # Perform boundary check and issue appropriate async copy
        if dst_coord0 < src_bound0 and dst_coord1 < src_bound1:
            # In-bounds: copy actual data
            async_copy[
                size_bytes,
                bypass_L1_16B=False,
                fill=Scalar[dst.dtype](0),
            ](src_ptr, dst_ptr, src_size=Int32(size_bytes))
        else:
            # Out-of-bounds: zero-fill
            async_copy[
                size_bytes, bypass_L1_16B=False, fill=Scalar[dst.dtype](0)
            ](src_ptr, dst_ptr, src_size=0)
