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
"""Implements Hopper SM90 matrix multiplication kernels using TMA and WGMMA.

Provides the `HopperMatmulSM90Kernel` struct and its shared-memory companion
`HopperMatmulSM90Kernel_SMem`, together with producer-consumer pipeline
routines that overlap asynchronous TMA loads with warp-group matrix
multiply-accumulate (WGMMA) tensor-core operations. The module supplies
standard, split-K, and grouped (MoE) kernel entry points targeting NVIDIA
H100 GPUs.
"""
from std.collections import OptionalReg
from std.math import ceildiv
from std.math.uutils import udivmod
from std.sys import size_of

from max.gpu import MAX_THREADS_PER_BLOCK_METADATA
from max.gpu.sync import barrier
from max.gpu.primitives.cluster import (
    cluster_sync,
    cluster_sync_relaxed,
    elect_one_sync,
)
from max.gpu.globals import WARPGROUP_SIZE
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    wait_on_dependent_grids,
)
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu import (
    block_id_in_cluster,
    block_idx,
    grid_dim,
    thread_idx,
    warp_id,
)
from max.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from max.gpu.memory import (
    external_memory,
    fence_mbarrier_init,
)
from layout import (
    Coord,
    Idx,
    Layout,
    MixedLayout,
    DefaultEngine,
    TensorLayout,
    TensorEngine,
    TileTensor,
    row_major,
)
from layout.tensor_core_async import (
    TensorCoreAsync,
    tile_layout_k_major,
    warpgroup_fence,
)
from layout.tma_async import (
    TMATensorTile,
)
from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple

from ....utils import elementwise_compute_lambda_type, elementwise_epilogue_type
from ....utils_gpu import block_swizzle
from ..tile_scheduler import RasterOrder
from ..tile_scheduler_splitk import SplitKTileScheduler
from ....structuring import RegTile

# Shared types from SM100 tile_types
from structured_kernels.tile_types import (
    SMemTile,
    SMemTileArrayWithLayout,
    SMemTileArray2DRowMajor,
)
from layout.tile_layout import Layout as _Layout
from structured_kernels.pipeline import (
    ProducerConsumerPipeline,
)
from structured_kernels.pipeline_backend import MbarPtr
from structured_kernels.pipeline_storage import BarrierPair
from .tile_loader import (
    TileLoaderTMA,
    TileLoaderCPAsync,
    TileLoader,
    BarrierHandler,
    TMABarrierHandler,
)
from .matmul_output import MatmulTileWriter


# Shared memory structure for Hopper SM90 kernel (TileTensor-based)
struct HopperMatmulSM90Kernel_SMem[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    BM: Int,
    BN: Int,
    BK: Int,
    WG_BM: Int,  # C tile M dimension (warpgroup output)
    WG_BN: Int,  # C tile N dimension (warpgroup output)
    num_pipeline_stages: Int,
    k_group_size: Int,
    swizzle_bytes: Int = 128,
]:
    """Shared memory layout for Hopper SM90 matrix multiplication kernel.

    This struct manages the shared memory allocation for:
    - Input tiles (A and B matrices) with multi-stage pipelining
    - Output tile (C matrix) for accumulation
    - Synchronization barriers for producer-consumer coordination

    The memory is organized to support asynchronous loads and efficient
    bank-conflict-free access patterns for tensor core operations.

    All tiles use TileTensor-based types from tile_types.mojo.
    """

    # SM90 blocked_product ordering: ((8, tiles_m), ...) with non-zero K-tile stride.
    # The atom (8 x atom_k) is the inner dimension, M-tiles are outer.
    comptime _k_major[dtype: DType, BM_: Int, BK_: Int, sb: Int] = _Layout(
        Coord(
            Coord(Idx[8], Idx[BM_ // 8]),
            Coord(
                Idx[sb // size_of[dtype]()],
                Idx[BK_ * size_of[dtype]() // sb],
            ),
        ),
        Coord(
            Coord(
                Idx[sb // size_of[dtype]()],
                Idx[8 * (sb // size_of[dtype]())],
            ),
            Coord(
                Idx[1],
                Idx[
                    0 if BK_ * size_of[dtype]() // sb
                    == 1 else BM_ * (sb // size_of[dtype]())
                ],
            ),
        ),
    )

    # TileTensor-based tile array types for A/B (SM90 swizzled k-major layout)
    comptime ATileArray = SMemTileArrayWithLayout[
        Self.a_type,
        Self._k_major[Self.a_type, Self.BM, Self.BK, Self.swizzle_bytes],
        Self.num_pipeline_stages,
    ]
    comptime BTileArray = SMemTileArrayWithLayout[
        Self.b_type,
        Self._k_major[Self.b_type, Self.BN, Self.BK, Self.swizzle_bytes],
        Self.num_pipeline_stages,
    ]
    # TileTensor-based for C tile (row-major, no swizzle, single tile)
    comptime CTileArray = SMemTileArray2DRowMajor[
        Self.c_type, Self.WG_BM, Self.WG_BN, 1  # num_tiles=1
    ]
    comptime CTile = Self.CTileArray.Tile

    # Number of pipeline stages for barriers (adjusted for k_group_size)
    comptime _num_barrier_stages = Self.num_pipeline_stages // Self.k_group_size

    # Array storage fields (like SM100 pattern)
    var a_tiles_storage: Self.ATileArray.Storage
    var b_tiles_storage: Self.BTileArray.Storage
    var c_tile_storage: Self.CTileArray.Storage

    # Pipeline barriers using BarrierPair (explicit contiguous layout)
    # Contains full[0..n-1] + empty[0..n-1] in one array
    var barriers: BarrierPair[Self._num_barrier_stages]

    # Accessor functions (like SM100 pattern)
    @always_inline
    def a_tiles(ref[AddressSpace.SHARED] self) -> Self.ATileArray:
        """Get A tile array accessor (TileTensor-based)."""
        return Self.ATileArray(self.a_tiles_storage.unsafe_ptr())

    @always_inline
    def b_tiles(ref[AddressSpace.SHARED] self) -> Self.BTileArray:
        """Get B tile array accessor (TileTensor-based)."""
        return Self.BTileArray(self.b_tiles_storage.unsafe_ptr())

    @always_inline
    def c_tile(ref[AddressSpace.SHARED] self) -> Self.CTile:
        """Get C tile accessor (TileTensor-based)."""
        return Self.CTileArray(self.c_tile_storage.unsafe_ptr())[0]

    @always_inline
    def create_pipeline(
        ref[AddressSpace.SHARED] self,
    ) -> ProducerConsumerPipeline[Self._num_barrier_stages]:
        """Create producer-consumer pipeline from barrier storage."""
        return self.barriers.create_pipeline()

    # Barrier pair type alias for storage size calculation
    comptime _BarrierPair = BarrierPair[Self._num_barrier_stages]

    @staticmethod
    @always_inline
    def pipeline_storage_size() -> Int:
        """Calculate the memory size for all pipeline stages."""
        return (
            # A and B tile storage
            Self.ATileArray.storage_size
            + Self.BTileArray.storage_size
            # Pipeline barriers (full + empty in BarrierPair)
            + Self._BarrierPair.Array.storage_size
        )

    @staticmethod
    @always_inline
    def output_storage_size() -> Int:
        """Calculate the memory size for output tile."""
        return Self.CTileArray.storage_size

    @staticmethod
    @always_inline
    def storage_size() -> Int:
        """Calculate the total storage size."""
        return Self.pipeline_storage_size() + Self.output_storage_size()


struct HopperMatmulSM90Kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_layout: TensorLayout,
    c_smem_layout: Layout,
    block_tile_shape: IndexList[3],
    wgmma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    num_pipeline_stages: Int,
    num_threads: Int = 128,
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    partitioned_multicast: Bool = False,
    use_tma_store: Bool = False,
    promotion_frequency: Int = 1,
    pdl_level: PDLLevel = PDLLevel(),
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    hilbert_swizzle: Bool = False,
    k_group_size: Int = 1,
    swapAB: Bool = False,
    a_engine: TensorEngine = DefaultEngine[element_width=1],
    b_engine: TensorEngine = DefaultEngine[element_width=1],
    c_engine: TensorEngine = DefaultEngine[element_width=1],
    a_offsets_engine: TensorEngine = DefaultEngine[element_width=1],
    expert_ids_engine: TensorEngine = DefaultEngine[element_width=1],
]:
    """Hopper SM90 GEMM for NVIDIA H100 GPUs.

    Uses TMA loads, WGMMA tensor-core MMA, multi-stage pipelining, and a
    producer-consumer warp-group layout.

    Parameters:
        a_type: Data type of the A input matrix.
        b_type: Data type of the B input matrix.
        c_type: Data type of the C output matrix.
        a_layout: Memory layout of the A matrix.
        b_layout: Memory layout of the B matrix.
        c_layout: Memory layout of the C matrix.
        c_smem_layout: Shared memory layout for the output tile.
        block_tile_shape: Tile dimensions `[M, N, K]` processed by each
            thread block.
        wgmma_shape: Dimensions for each WGMMA instruction `[M, N, K]`.
        cluster_shape: Thread block cluster dimensions for distributed
            shared memory.
        num_pipeline_stages: Number of stages in the software pipeline
            (3-7 in most configs).
        num_threads: Number of threads per block (must be a multiple of
            128).
        transpose_b: Whether the B matrix is transposed (required to be
            `True`).
        a_swizzle: Memory swizzling for bank-conflict-free A tile access.
        b_swizzle: Memory swizzling for bank-conflict-free B tile access.
        c_swizzle: Swizzling mode for output writes.
        partitioned_multicast: Whether partitioned multicast is enabled
            for large tiles.
        use_tma_store: Whether TMA is used for storing output (versus
            regular stores).
        promotion_frequency: How often FP8 accumulation is promoted to
            higher precision.
        pdl_level: Programmatic Dependency Launch (PDL) level.
        elementwise_lambda_fn: Optional epilogue function.
        elementwise_compute_lambda_fn: Optional compute function.
        hilbert_swizzle: Whether Hilbert-curve thread block scheduling is
            used.
        k_group_size: Number of K-dimension tiles loaded and consumed per
            pipeline stage; both `num_pipeline_stages` and the total K
            extent must be multiples of this value (defaults to 1).
        swapAB: Whether to swap the A and B operand roles in the output
            writer for the small-M strategy, transposing the tile and
            block coordinate mapping (defaults to `False`).
        a_engine: Engine of the A operand (defaults to
            `DefaultEngine`).
        b_engine: Engine of the B operand (defaults to
            `DefaultEngine`).
        c_engine: Engine of the C output (defaults to
            `DefaultEngine`).
        a_offsets_engine: Engine of the `a_offsets` tensor
            (defaults to `DefaultEngine`).
        expert_ids_engine: Engine of the `expert_ids` tensor
            (defaults to `DefaultEngine`).
    """

    comptime BM = Self.block_tile_shape[0]
    comptime BN = Self.block_tile_shape[1]
    comptime BK = Self.block_tile_shape[2]

    comptime num_consumer = (Self.num_threads // 128) - 1
    comptime num_consumer_threads = Self.num_consumer * 128

    comptime num_m_mmas = Self.BM // Self.wgmma_shape[0] // Self.num_consumer
    comptime num_n_mmas = Self.BN // Self.wgmma_shape[1]

    comptime accum_type = get_accum_type[Self.a_type]()
    comptime c_frag_size = Self.wgmma_shape[0] * Self.wgmma_shape[1] // 128

    comptime a_smem_layout = tile_layout_k_major[
        Self.a_type, Self.BM, Self.BK, Self.a_swizzle
    ]()
    comptime b_smem_layout = tile_layout_k_major[
        Self.b_type, Self.BN, Self.BK, Self.b_swizzle
    ]()

    comptime adjusted_num_pipeline_stages = Self.num_pipeline_stages // Self.k_group_size

    # TMA expected bytes per pipeline stage (combined A + B tiles)
    comptime _tma_bytes_per_stage = (
        Self.SMem.ATileArray.storage_size + Self.SMem.BTileArray.storage_size
    ) // Self.adjusted_num_pipeline_stages
    comptime TMABarrier = TMABarrierHandler[Self._tma_bytes_per_stage]

    comptime AccumRegTile = RegTile[
        Self.accum_type,
        Layout.row_major(Self.num_m_mmas * Self.num_n_mmas, Self.c_frag_size),
    ]

    comptime cluster_size = Int(
        Self.cluster_shape[0] * Self.cluster_shape[1] * Self.cluster_shape[2]
    )

    comptime SMem = HopperMatmulSM90Kernel_SMem[
        Self.a_type,
        Self.b_type,
        Self.c_type,
        Self.BM,
        Self.BN,
        Self.BK,
        Self.c_smem_layout.shape[0].value(),  # WG_BM
        Self.c_smem_layout.shape[1].value(),  # WG_BN
        Self.num_pipeline_stages,
        Self.k_group_size,
        128,  # swizzle_bytes (matches TensorMapSwizzle.SWIZZLE_128B default)
    ]

    comptime WgmmaOp = TensorCoreAsync[
        Self.accum_type,
        Self.a_type,
        Self.b_type,
        Self.wgmma_shape,
        Self.a_swizzle,
        Self.b_swizzle,
        Self.transpose_b,
    ]

    @staticmethod
    @always_inline
    def validate_constraints():
        """Validate common constraints for all kernel variants."""
        comptime assert (
            Self.a_type == Self.b_type
        ), "A and B must have the same type"

        comptime assert Self.transpose_b, "Only support transposed B in layout"

        comptime assert (
            not Self.partitioned_multicast
            or Self.a_swizzle.bytes() // size_of[Self.a_type]() == Self.BK
        ), (
            "Currently partitioned multi-casting is only supported when BK"
            " == (a_swizzle.bytes // size_of[a_type])"
        )
        comptime assert (
            not Self.partitioned_multicast
            or Self.b_swizzle.bytes() // size_of[Self.b_type]() == Self.BK
        ), (
            "Currently partitioned multi-casting is only supported when BK"
            " == (b_swizzle.bytes // size_of[b_type])"
        )

        comptime assert (
            Self.num_pipeline_stages % Self.k_group_size == 0
        ), "num_pipeline_stages must be a multiple of k_group_size"
        comptime K = Self.b_layout.static_shape[1]
        comptime assert (
            K % Self.k_group_size == 0
        ), "K must be a multiple of k_group_size"

    @always_inline
    @staticmethod
    def pipeline_init():
        """Initialize pipeline synchronization barriers.

        This function ensures that all pipeline initialization (barriers, shared memory)
        is visible to all thread blocks in the cluster before proceeding. This is
        critical for correct producer-consumer synchronization.

        For multi-cluster configurations, uses fence and cluster sync.
        For single block, uses a simple barrier.
        """

        comptime if Self.cluster_size > 1:
            fence_mbarrier_init()
            cluster_sync_relaxed()
        else:
            barrier()

    @staticmethod
    @always_inline
    def finalize_kernel():
        """Common finalization for all kernel variants."""

        comptime if Self.pdl_level >= PDLLevel.OVERLAP_AT_END:
            launch_dependent_grids()

        # Synchronize all thread blocks in the cluster before kernel exit
        # to ensure shared memory isn't deallocated while other blocks are still using it
        comptime if Self.cluster_size > 1:
            cluster_sync()

    @staticmethod
    @always_inline
    def multicast_mask(rank_m: Int, rank_n: Int) -> Tuple[Int32, Int32]:
        comptime CLUSTER_N = Self.cluster_shape[0]
        comptime CLUSTER_M = Self.cluster_shape[1]

        # Setup multicast masks for cluster-wide data distribution
        var multicast_column_mask = 0

        comptime for i in range(CLUSTER_M):
            multicast_column_mask |= Int(1 << (i * CLUSTER_N))
        multicast_column_mask <<= rank_n

        var multicast_row_mask = ((1 << CLUSTER_N) - 1) << (
            Int32(rank_m) * CLUSTER_N
        )
        return (multicast_row_mask, Int32(multicast_column_mask))

    @staticmethod
    @always_inline
    def common_kernel_init() -> (
        Tuple[
            Int,
            Int,
            Int,
            Int,
            Int,
            Bool,
        ]
    ):
        """Common initialization for all kernel variants.

        Returns:
            Tuple of (warp_group_idx, warp_group_thread_idx,
                     rank_m, rank_n, warp_id, lane_predicate).
        """
        Self.validate_constraints()

        var warp_group_idx, warp_group_thread_idx = udivmod(
            thread_idx.x,
            WARPGROUP_SIZE,
        )

        var rank_m = block_id_in_cluster.y
        var rank_n = block_id_in_cluster.x

        var warp_id = warp_id()
        var lane_predicate = elect_one_sync()

        return (
            warp_group_idx,
            warp_group_thread_idx,
            rank_m,
            rank_n,
            warp_id,
            lane_predicate,
        )

    @staticmethod
    @always_inline
    def setup_producer() -> Int:
        """Setup producer warp group by deallocating registers.

        Returns:
            Number of registers deallocated.
        """
        comptime num_regs = 24 if Self.num_consumer <= 2 else 32
        warpgroup_reg_dealloc[num_regs]()
        return num_regs

    @staticmethod
    @always_inline
    def setup_consumer(
        warp_group_idx: Int,
    ) -> Tuple[Int, Self.AccumRegTile, Self.AccumRegTile]:
        """Setup consumer warp group.

        Returns:
            Tuple of (local_warp_group_idx, c_reg_tile, final_c_reg_tile).
        """

        @__parameter
        def num_regs() -> Int:
            if Self.num_consumer == 1:
                return 256
            if Self.num_consumer == 2:
                return 240
            return 160

        warpgroup_reg_alloc[num_regs()]()

        var local_warp_group_idx = warp_group_idx - 1
        var c_reg_tile = Self.AccumRegTile.stack_allocation()
        var final_c_reg_tile = Self.AccumRegTile.stack_allocation()

        return (local_warp_group_idx, c_reg_tile, final_c_reg_tile)

    @staticmethod
    @always_inline
    def consumer_arrive_empty_barriers(
        warp_group_thread_idx: Int,
        mut pipeline: ProducerConsumerPipeline[
            Self.adjusted_num_pipeline_stages
        ],
    ):
        """Signal initial empty barrier arrival for all pipeline stages.

        Must be called by consumer warp groups before the main loop so
        the producer knows it can start filling stages.

        Args:
            warp_group_thread_idx: Thread index within the warp group,
                used to select which threads signal the barrier.
            pipeline: Producer-consumer pipeline whose empty barriers are
                signaled (modified in place).
        """

        comptime for i in range(Self.adjusted_num_pipeline_stages):
            comptime if Self.cluster_size > 1:
                if warp_group_thread_idx < Self.cluster_size:
                    _ = rebind[MbarPtr](pipeline.consumer_mbar(UInt32(i)))[
                        0
                    ].arrive_cluster(UInt32(warp_group_thread_idx))
            else:
                if warp_group_thread_idx == 0:
                    _ = rebind[MbarPtr](pipeline.consumer_mbar(UInt32(i)))[
                        0
                    ].arrive()

    @staticmethod
    @always_inline
    def get_block_swizzle(
        lut_ptr: OptionalReg[UnsafePointer[UInt32, MutAnyOrigin]] = None,
    ) -> IndexList[2, element_type=.uint32]:
        """Calculate block swizzle for better L2 cache locality.

        Args:
            lut_ptr: Lookup table for Hilbert curve block scheduling (optional).

        Returns:
            Swizzled block indices.
        """
        comptime use_cluster = Self.cluster_size > 1

        comptime if not use_cluster:
            comptime if Self.hilbert_swizzle:
                # Hilbert curve ordering maximizes spatial locality
                var linear = UInt32(block_idx.y * grid_dim.x + block_idx.x)
                var packed = lut_ptr.unsafe_value()[linear]
                var new_x = packed & 0xFFFF
                var new_y = packed >> 16
                return Index[dtype=.uint32](new_x, new_y)
            else:
                # Default swizzling pattern for L2 cache optimization
                return block_swizzle(
                    Index[dtype=.uint32](block_idx.x, block_idx.y),
                    Index[dtype=.uint32](grid_dim.x, grid_dim.y),
                )
        else:
            # Multi-cluster mode: no swizzling (handled by hardware)
            return Index[dtype=.uint32](block_idx.x, block_idx.y)

    @staticmethod
    @always_inline
    def consumer_output[
        custom_elementwise_lambda_fn: Optional[
            elementwise_epilogue_type
        ] = Self.elementwise_lambda_fn
    ](
        c_tma_op: TMATensorTile[Self.c_type, _, _, _],
        c: TileTensor[mut=True, Self.c_type, address_space=.GENERIC, ...],
        c_tile: Self.SMem.CTile,
        output_reg_tile: Self.AccumRegTile,
        warp_group_thread_idx: Int,
        local_warp_group_idx: Int,
        local_thread_idx: Int,
        block_y: Int,
        block_x: Int,
    ):
        """Handle consumer output by writing GEMM results to global memory.

        Parameters:
            custom_elementwise_lambda_fn: Optional epilogue function applied
                to output elements (defaults to the struct's
                `elementwise_lambda_fn`).

        Args:
            c_tma_op: TMA descriptor for the output matrix C, used for TMA
                stores.
            c: Writable output matrix C tile tensor.
            c_tile: Shared memory tile staging the output before the global
                write.
            output_reg_tile: Register tile holding the accumulated GEMM
                result to write.
            warp_group_thread_idx: Thread index within the warp group.
            local_warp_group_idx: Index of this consumer warp group
                (0-based).
            local_thread_idx: Thread index within the consumer warp group.
            block_y: Block-level M coordinate (row) of the output tile.
            block_x: Block-level N coordinate (column) of the output tile.
        """
        var matmul_tile_writer = MatmulTileWriter[
            BM=Self.BM,
            BN=Self.BN,
            swizzle=Self.c_swizzle,
            wgmma_shape=Self.wgmma_shape,
            num_consumer=Self.num_consumer,
            use_tma_store=Self.use_tma_store,
            elementwise_lambda_fn=custom_elementwise_lambda_fn,
            elementwise_compute_lambda_fn=Self.elementwise_compute_lambda_fn,
            swapAB=Self.swapAB,
        ](
            c.as_unsafe_any_origin(),
            c_tile,
            warp_group_thread_idx,
            local_warp_group_idx,
            local_thread_idx,
            block_y,
            block_x,
        )
        matmul_tile_writer.write_tile(c_tma_op, output_reg_tile)

    @staticmethod
    @always_inline
    def build_tma_loaders[
        a_tma_rank: Int,
        b_tma_rank: Int,
        a_tile_shape: IndexList[a_tma_rank],
        b_tile_shape: IndexList[b_tma_rank],
        a_desc_shape: IndexList[a_tma_rank],
        b_desc_shape: IndexList[b_tma_rank],
        //,
    ](
        a_tma_op: TMATensorTile[
            Self.a_type, a_tma_rank, a_tile_shape, a_desc_shape
        ],
        b_tma_op: TMATensorTile[
            Self.b_type, b_tma_rank, b_tile_shape, b_desc_shape
        ],
        rank_m: Int,
        rank_n: Int,
    ) -> Tuple[
        TileLoaderTMA[
            origin_of(a_tma_op),
            Self.a_type,
            a_tma_rank,
            a_tile_shape,
            a_desc_shape,
            BK=Self.BK,
            cluster_size=Self.cluster_shape[0],
            use_partitioned_multicast=Self.partitioned_multicast,
        ],
        TileLoaderTMA[
            origin_of(b_tma_op),
            Self.b_type,
            b_tma_rank,
            b_tile_shape,
            b_desc_shape,
            BK=Self.BK,
            cluster_size=Self.cluster_shape[1],
            use_partitioned_multicast=Self.partitioned_multicast,
        ],
    ]:
        # Prefetch TMA descriptors if on thread 0.
        if thread_idx.x == 0:
            a_tma_op.prefetch_descriptor()
            b_tma_op.prefetch_descriptor()

        var a_multicast_mask, b_multicast_mask = Self.multicast_mask(
            rank_m, rank_n
        )
        var a_loader = TileLoaderTMA[
            BK=Self.BK,
            cluster_size=Self.cluster_shape[0],
            use_partitioned_multicast=Self.partitioned_multicast,
        ](Pointer(to=a_tma_op), rank_n, UInt16(a_multicast_mask))
        var b_loader = TileLoaderTMA[
            BK=Self.BK,
            cluster_size=Self.cluster_shape[1],
            use_partitioned_multicast=Self.partitioned_multicast,
        ](Pointer(to=b_tma_op), rank_m, UInt16(b_multicast_mask))
        return (a_loader, b_loader)

    @always_inline
    @staticmethod
    def build_cpasync_loaders[
        k_align: Int,
        vector_size: Int = k_align // size_of[Self.a_type](),
        num_threads_per_row: Int = Self.BK // vector_size,
        thread_layout: MixedLayout = row_major[
            WARPGROUP_SIZE // num_threads_per_row, num_threads_per_row
        ](),
    ](
        a: TileTensor[
            Self.a_type, Self.a_layout, ImmutAnyOrigin, Engine=Self.a_engine
        ],
        b: TileTensor[
            Self.b_type, Self.b_layout, ImmutAnyOrigin, Engine=Self.b_engine
        ],
    ) -> Tuple[
        TileLoaderCPAsync[
            Self.a_type,
            Self.a_layout,
            thread_layout,
            Self.a_swizzle,
            vector_size,
            Self.a_engine,
        ],
        TileLoaderCPAsync[
            Self.b_type,
            Self.b_layout,
            thread_layout,
            Self.b_swizzle,
            vector_size,
            Self.b_engine,
        ],
    ]:
        var a_loader = TileLoaderCPAsync[
            Self.a_type,
            Self.a_layout,
            thread_layout,
            Self.a_swizzle,
            vector_size,
            Self.a_engine,
        ](a)
        var b_loader = TileLoaderCPAsync[
            Self.b_type,
            Self.b_layout,
            thread_layout,
            Self.b_swizzle,
            vector_size,
            Self.b_engine,
        ](b)
        return (a_loader, b_loader)

    @staticmethod
    @always_inline
    def producer_main_loop_pipeline[
        a_loader_type: TileLoader,
        b_loader_type: TileLoader,
        barrier_handler_type: BarrierHandler,
        //,
        num_k_iters: Int,
    ](
        m_coord: Int,
        n_coord: Int,
        k_coord: Int,
        a_loader: a_loader_type,
        b_loader: b_loader_type,
        barrier_handler: barrier_handler_type,
        mut pipeline: ProducerConsumerPipeline[
            Self.adjusted_num_pipeline_stages
        ],
        a_tiles: Self.SMem.ATileArray,
        b_tiles: Self.SMem.BTileArray,
    ):
        @always_inline
        @__parameter
        def producer_loop[
            num_pipeline_stages_to_unroll: Int,
        ](k_iter: Int):
            comptime for j in range(num_pipeline_stages_to_unroll):
                var k_offset = k_coord + (
                    k_iter * Self.num_pipeline_stages + (j * Self.k_group_size)
                )

                # Acquire producer stage (waits for consumer)
                var stage = pipeline.acquire_producer()
                var slot = Int(stage.index())

                # Prepare barrier for this stage (TMA: expect_bytes, cp.async: noop)
                barrier_handler.prepare_stage(stage.mbar())

                # Get tile slices for this stage
                var a_tile_slice = a_tiles.slice[Self.k_group_size](
                    slot * Self.k_group_size
                )
                var b_tile_slice = b_tiles.slice[Self.k_group_size](
                    slot * Self.k_group_size
                )

                comptime for k in range(Self.k_group_size):
                    a_loader.load_tile(
                        a_tile_slice[k],
                        stage.mbar(),
                        (m_coord, k_offset),
                    )
                    b_loader.load_tile(
                        b_tile_slice[k],
                        stage.mbar(),
                        (n_coord, k_offset),
                    )

                    k_offset += 1

                # Complete stage (TMA: noop, cp.async: arrive + signal)
                barrier_handler.complete_stage(stage.mbar())
                stage^.release()

        # Calculate how many full pipeline iterations we need
        comptime num_full_k_iters = ceildiv(
            num_k_iters, Self.num_pipeline_stages
        )
        # Handle uneven division: the last iteration may have fewer stages
        comptime num_remaining_k_iters = num_k_iters % Self.num_pipeline_stages

        comptime if num_remaining_k_iters == 0:
            for k_iter in range(num_full_k_iters):
                producer_loop[Self.adjusted_num_pipeline_stages](k_iter)
        else:
            for k_iter in range(num_full_k_iters - 1):
                producer_loop[Self.adjusted_num_pipeline_stages](k_iter)
            producer_loop[num_remaining_k_iters // Self.k_group_size](
                num_full_k_iters - 1
            )

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.num_threads)
        ),
        `nvvm.cluster_dim`=Self.cluster_shape,
    )
    @__name(
        t"sm90_matmul_{Self.a_type}_{Self.b_type}_{Self.c_type}_{Self.transpose_b}",
    )
    @__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
    def run[
        a_tma_rank: Int,
        b_tma_rank: Int,
        c_tma_rank: Int,
        a_tile_shape: IndexList[a_tma_rank],
        b_tile_shape: IndexList[b_tma_rank],
        c_tile_shape: IndexList[c_tma_rank],
        a_desc_shape: IndexList[a_tma_rank],
        b_desc_shape: IndexList[b_tma_rank],
        c_desc_shape: IndexList[c_tma_rank],
        a_tensor_layout: TensorLayout,
        b_tensor_layout: TensorLayout,
        c_tensor_layout: TensorLayout,
    ](
        a_tma_op: TMATensorTile[
            Self.a_type, a_tma_rank, a_tile_shape, a_desc_shape
        ],
        b_tma_op: TMATensorTile[
            Self.b_type, b_tma_rank, b_tile_shape, b_desc_shape
        ],
        c_tma_op: TMATensorTile[
            Self.c_type, c_tma_rank, c_tile_shape, c_desc_shape
        ],
        a: TileTensor[
            Self.a_type, a_tensor_layout, ImmutAnyOrigin, Engine=Self.a_engine
        ],
        b: TileTensor[
            Self.b_type, b_tensor_layout, ImmutAnyOrigin, Engine=Self.b_engine
        ],
        c: TileTensor[
            Self.c_type, c_tensor_layout, MutAnyOrigin, Engine=Self.c_engine
        ],
        lut_ptr: UnsafePointer[UInt32, MutAnyOrigin],
    ):
        """Main kernel entry point for matrix multiplication.

        This kernel implements a producer-consumer pattern where:
        - One warp group (producer) loads tiles from global memory using TMA
        - Multiple warp groups (consumers) perform matrix multiplication using tensor cores

        The kernel uses software pipelining to overlap memory transfers with computation,
        achieving high throughput on Hopper GPUs.

        Parameters:
            a_tma_rank: Number of dimensions in the TMA descriptor for
                matrix A.
            b_tma_rank: Number of dimensions in the TMA descriptor for
                matrix B.
            c_tma_rank: Number of dimensions in the TMA descriptor for
                matrix C.
            a_tile_shape: Shape of each A tile loaded by TMA.
            b_tile_shape: Shape of each B tile loaded by TMA.
            c_tile_shape: Shape of each C tile stored by TMA.
            a_desc_shape: Full shape of matrix A as described by the TMA
                descriptor.
            b_desc_shape: Full shape of matrix B as described by the TMA
                descriptor.
            c_desc_shape: Full shape of matrix C as described by the TMA
                descriptor.
            a_tensor_layout: Memory layout of input matrix A.
            b_tensor_layout: Memory layout of input matrix B.
            c_tensor_layout: Memory layout of output matrix C.

        Args:
            a_tma_op: TMA descriptor for matrix A.
            b_tma_op: TMA descriptor for matrix B.
            c_tma_op: TMA descriptor for matrix C.
            a: Input matrix A.
            b: Input matrix B.
            c: Output matrix C.
            lut_ptr: Lookup table for Hilbert curve block scheduling (optional).
        """
        comptime K = Self.b_layout.static_shape[1]
        comptime num_k_iters = ceildiv(K, Self.BK)

        # Initialize WgmmaOp and SMem first
        var wgmma_op = Self.WgmmaOp()
        ref smem = external_memory[
            UInt8,
            address_space=.SHARED,
            alignment=128,
        ]().bitcast[Self.SMem]()[]

        # Common initialization
        var (
            warp_group_idx,
            warp_group_thread_idx,
            rank_m,
            rank_n,
            warp_id,
            lane_predicate,
        ) = Self.common_kernel_init()

        # Create pipeline and barrier handler (initializes phase + barrier counts)
        var pipeline = smem.create_pipeline()
        var barrier_handler = Self.TMABarrier(
            pipeline, Self.num_consumer, Self.cluster_size
        )

        # Create TileLoaderTMA loaders
        var a_loader, b_loader = Self.build_tma_loaders(
            a_tma_op, b_tma_op, rank_m, rank_n
        )

        Self.pipeline_init()

        # Calculate block swizzle
        var block_idx_swizzle = Self.get_block_swizzle(lut_ptr)
        var m_coord = block_idx_swizzle[1] * Self.BM
        var n_coord = block_idx_swizzle[0] * Self.BN

        # Split thread blocks into producer and consumer warp groups
        if warp_group_idx == 0:
            # Producer warp group

            # Check and wait for PDL grids if needed
            comptime if (
                Self.pdl_level > PDLLevel.OFF
                and Self.pdl_level != PDLLevel.NO_WAIT_OVERLAP_AT_END
            ):
                wait_on_dependent_grids()

            _ = Self.setup_producer()

            if warp_id == 0 and lane_predicate:
                Self.producer_main_loop_pipeline[num_k_iters=num_k_iters](
                    m_coord,
                    n_coord,
                    0,  # k_start,
                    a_loader,
                    b_loader,
                    barrier_handler,
                    pipeline,
                    smem.a_tiles(),
                    smem.b_tiles(),
                )
        else:
            # Consumer warp groups
            var local_warp_group_idx, c_reg_tile, final_c_reg_tile = (
                Self.setup_consumer(warp_group_idx)
            )

            Self.consumer_arrive_empty_barriers(warp_group_thread_idx, pipeline)

            Self.consumer_main_loop_pipeline[num_k_iters=num_k_iters](
                wgmma_op,
                local_warp_group_idx,
                final_c_reg_tile,
                c_reg_tile,
                pipeline,
                smem.a_tiles(),
                smem.b_tiles(),
                warp_group_thread_idx,
            )

            var output_reg_tile = (
                final_c_reg_tile if Self.a_type
                == .float8_e4m3fn else c_reg_tile
            )

            Self.consumer_output(
                c_tma_op,
                c,
                smem.c_tile(),
                output_reg_tile,
                warp_group_thread_idx,
                local_warp_group_idx,
                thread_idx.x - WARPGROUP_SIZE,
                block_idx_swizzle[1],
                block_idx_swizzle[0],
            )

        Self.finalize_kernel()

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.num_threads)
        ),
        `nvvm.cluster_dim`=Self.cluster_shape,
    )
    @__name(
        t"sm90_matmul_split_k_{Self.a_type}_{Self.b_type}_{Self.c_type}_{Self.transpose_b}",
    )
    @__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
    def run_splitk[
        a_tma_rank: Int,
        b_tma_rank: Int,
        c_tma_rank: Int,
        a_tile_shape: IndexList[a_tma_rank],
        b_tile_shape: IndexList[b_tma_rank],
        c_tile_shape: IndexList[c_tma_rank],
        a_desc_shape: IndexList[a_tma_rank],
        b_desc_shape: IndexList[b_tma_rank],
        c_desc_shape: IndexList[c_tma_rank],
        splits: Int,
        raster_order: RasterOrder,
        c_tensor_layout: TensorLayout,
    ](
        a_tma_op: TMATensorTile[
            Self.a_type, a_tma_rank, a_tile_shape, a_desc_shape
        ],
        b_tma_op: TMATensorTile[
            Self.b_type, b_tma_rank, b_tile_shape, b_desc_shape
        ],
        c_tma_op: TMATensorTile[
            Self.c_type, c_tma_rank, c_tile_shape, c_desc_shape
        ],
        c: TileTensor[
            Self.c_type, c_tensor_layout, MutAnyOrigin, Engine=Self.c_engine
        ],
        workspace_ptr: UnsafePointer[Scalar[Self.accum_type], MutAnyOrigin],
        locks_ptr: UnsafePointer[UInt8, MutAnyOrigin],
        problem_shape: IndexList[3],
    ):
        """Split-K variant of the kernel for better load balancing on small problems.

        Parameters:
            a_tma_rank: Number of dimensions in the TMA descriptor for
                matrix A.
            b_tma_rank: Number of dimensions in the TMA descriptor for
                matrix B.
            c_tma_rank: Number of dimensions in the TMA descriptor for
                matrix C.
            a_tile_shape: Shape of each A tile loaded by TMA.
            b_tile_shape: Shape of each B tile loaded by TMA.
            c_tile_shape: Shape of each C tile stored by TMA.
            a_desc_shape: Full shape of matrix A as described by the TMA
                descriptor.
            b_desc_shape: Full shape of matrix B as described by the TMA
                descriptor.
            c_desc_shape: Full shape of matrix C as described by the TMA
                descriptor.
            splits: Number of equal chunks the K dimension is divided
                into for parallel reduction. Each block processes one
                chunk per output tile.
            raster_order: Tile rasterization order used by the split-K
                scheduler to assign output tiles to blocks.
            c_tensor_layout: Memory layout of output matrix C.

        Args:
            a_tma_op: TMA descriptor for matrix A.
            b_tma_op: TMA descriptor for matrix B.
            c_tma_op: TMA descriptor for matrix C.
            c: Output matrix C.
            workspace_ptr: Pointer to the reduction workspace storing
                partial accumulations from each split.
            locks_ptr: Pointer to the lock array coordinating split-K
                synchronization across blocks.
            problem_shape: Full GEMM problem dimensions `[M, N, K]`.
        """
        comptime K = Self.b_layout.static_shape[1]
        comptime num_k_iters = K // Self.BK

        # FIXME: this seems to trip some logits tests
        # comptime assert (K % Self.BK) == 0, "K must be divisible by BK"

        # Initialize WgmmaOp and SMem first
        var wgmma_op = Self.WgmmaOp()
        ref smem = external_memory[
            UInt8,
            address_space=.SHARED,
            alignment=128,
        ]().bitcast[Self.SMem]()[]

        # Common initialization
        var (
            warp_group_idx,
            warp_group_thread_idx,
            rank_m,
            rank_n,
            warp_id,
            lane_predicate,
        ) = Self.common_kernel_init()

        # Create pipeline from barrier storage (uses BarrierPair for explicit layout)
        var pipeline = smem.create_pipeline()
        var barrier_handler = Self.TMABarrier(
            pipeline, Self.num_consumer, Self.cluster_size
        )

        # Create TileLoaderTMA loaders
        var a_loader, b_loader = Self.build_tma_loaders(
            a_tma_op, b_tma_op, rank_m, rank_n
        )

        Self.pipeline_init()

        comptime N = Self.b_layout.static_shape[0]
        comptime M = Self.a_layout.static_shape[0]
        comptime NUM_TILES = ceildiv(M, Self.BM) * ceildiv(N, Self.BN)

        comptime workspace_layout = row_major[NUM_TILES, Self.BM, Self.BN]()
        var reduction_workspace = TileTensor(workspace_ptr, workspace_layout)

        comptime CLUSTER_N = Self.cluster_shape[0]
        comptime CLUSTER_M = Self.cluster_shape[1]

        var scheduler = SplitKTileScheduler[
            Index(N, K),
            Self.block_tile_shape,
            UInt32(splits),
            UInt32(Self.num_consumer),
            UInt32(Self.num_pipeline_stages),
            Index(CLUSTER_M, CLUSTER_N),
            raster_order,
        ](
            problem_shape,
            Index(rank_m, rank_n),
            locks_ptr,
        )

        # Split thread blocks into producer and consumer warp groups
        if warp_group_idx == 0:
            # Producer warp group
            _ = Self.setup_producer()
            var work_tile_info = scheduler.initial_work_tile_info()

            if warp_id == 0 and lane_predicate:
                while work_tile_info.is_valid():
                    var m_coord = work_tile_info.m * UInt32(Self.BM)
                    var n_coord = work_tile_info.n * UInt32(Self.BN)

                    comptime work_k_tile_count = num_k_iters // splits
                    var work_k_tile_start = work_tile_info.get_k_start()

                    Self.producer_main_loop_pipeline[
                        num_k_iters=work_k_tile_count
                    ](
                        Int(m_coord),
                        Int(n_coord),
                        Int(work_k_tile_start),
                        a_loader,
                        b_loader,
                        barrier_handler,
                        pipeline,
                        smem.a_tiles(),
                        smem.b_tiles(),
                    )

                    # Get next work tile
                    work_tile_info = scheduler.fetch_next_work(work_tile_info)
        else:
            # Consumer warp groups
            var local_warp_group_idx, c_reg_tile, final_c_reg_tile = (
                Self.setup_consumer(warp_group_idx)
            )

            var work_tile_info = scheduler.initial_work_tile_info()

            Self.consumer_arrive_empty_barriers(warp_group_thread_idx, pipeline)

            while work_tile_info.is_valid():
                comptime work_k_tile_count = num_k_iters // splits

                Self.consumer_main_loop_pipeline[num_k_iters=work_k_tile_count](
                    wgmma_op,
                    local_warp_group_idx,
                    final_c_reg_tile,
                    c_reg_tile,
                    pipeline,
                    smem.a_tiles(),
                    smem.b_tiles(),
                    warp_group_thread_idx,
                )

                var output_reg_tile = (
                    final_c_reg_tile if Self.a_type
                    == .float8_e4m3fn else c_reg_tile
                )

                scheduler.reduction(
                    reduction_workspace,
                    output_reg_tile,
                    work_tile_info,
                    UInt32(Self.num_consumer),
                    UInt32(local_warp_group_idx),
                )

                # check if this is the reduction tile
                if scheduler.is_last_split(work_tile_info):
                    var block_y = Int(work_tile_info.m)
                    var block_x = Int(work_tile_info.n)

                    Self.consumer_output(
                        c_tma_op,
                        c,
                        smem.c_tile(),
                        output_reg_tile,
                        warp_group_thread_idx,
                        local_warp_group_idx,
                        thread_idx.x - WARPGROUP_SIZE,
                        block_y,
                        block_x,
                    )

                # Get next work tile
                work_tile_info = scheduler.fetch_next_work(work_tile_info)

        Self.finalize_kernel()

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.num_threads)
        ),
        `nvvm.cluster_dim`=Self.cluster_shape,
    )
    @__name(
        t"sm90_matmul_grouped_{Self.a_type}_{Self.b_type}_{Self.c_type}_{Self.transpose_b}",
    )
    @__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
    def run_grouped[
        a_tma_rank: Int,
        b_tma_rank: Int,
        c_tma_rank: Int,
        a_tile_shape: IndexList[a_tma_rank],
        b_tile_shape: IndexList[b_tma_rank],
        c_tile_shape: IndexList[c_tma_rank],
        a_desc_shape: IndexList[a_tma_rank],
        b_desc_shape: IndexList[b_tma_rank],
        c_desc_shape: IndexList[c_tma_rank],
        AOffsetsLayout: TensorLayout,
        ExpertIdsLayout: TensorLayout,
        c_tensor_layout: TensorLayout,
    ](
        a_tma_op: TMATensorTile[
            Self.a_type, a_tma_rank, a_tile_shape, a_desc_shape
        ],
        b_tma_op: TMATensorTile[
            Self.b_type, b_tma_rank, b_tile_shape, b_desc_shape
        ],
        c_tma_op: TMATensorTile[
            Self.c_type, c_tma_rank, c_tile_shape, c_desc_shape
        ],
        a_offsets: TileTensor[
            mut=False,
            .uint32,
            AOffsetsLayout,
            MutAnyOrigin,
            Engine=Self.a_offsets_engine,
        ],
        expert_ids: TileTensor[
            mut=False,
            .int32,
            ExpertIdsLayout,
            MutAnyOrigin,
            Engine=Self.expert_ids_engine,
        ],
        c: TileTensor[
            Self.c_type, c_tensor_layout, MutAnyOrigin, Engine=Self.c_engine
        ],
    ):
        """Grouped matmul variant for MoE (Mixture of Experts) models.

        This variant handles multiple experts where each expert processes a subset of tokens.
        The a_offsets array indicates token boundaries for each expert.

        Parameters:
            a_tma_rank: Number of dimensions in the TMA descriptor for
                matrix A.
            b_tma_rank: Number of dimensions in the TMA descriptor for
                matrix B.
            c_tma_rank: Number of dimensions in the TMA descriptor for
                matrix C.
            a_tile_shape: Shape of each A tile loaded by TMA.
            b_tile_shape: Shape of each B tile loaded by TMA.
            c_tile_shape: Shape of each C tile stored by TMA.
            a_desc_shape: Full shape of matrix A as described by the TMA
                descriptor.
            b_desc_shape: Full shape of matrix B as described by the TMA
                descriptor.
            c_desc_shape: Full shape of matrix C as described by the TMA
                descriptor.
            AOffsetsLayout: Memory layout of the `a_offsets` tensor.
            ExpertIdsLayout: Memory layout of the `expert_ids` tensor.
            c_tensor_layout: Memory layout of output matrix C.

        Args:
            a_tma_op: TMA descriptor for matrix A.
            b_tma_op: TMA descriptor for matrix B.
            c_tma_op: TMA descriptor for matrix C.
            a_offsets: Starting row offsets into matrix A for each
                expert. The token count for expert `i` is `a_offsets[i+1]
                - a_offsets[i]`, indexed by `block_idx.z`.
            expert_ids: Expert index selected by each block.
                `expert_ids[block_idx.z]` picks the row block in B for
                this block, and -1 marks an inactive block whose output
                is zeroed.
            c: Output matrix C.
        """
        comptime assert a_offsets.flat_rank == 1, "a_offsets must be rank 1"
        comptime assert expert_ids.flat_rank == 1, "expert_ids must be rank 1"

        comptime K = Self.b_layout.static_shape[1]
        comptime num_k_iters = ceildiv(K, Self.BK)

        # FIXME: this seems to trip some logits tests
        # comptime assert (K % Self.BK) == 0, "K must be divisible by BK"

        # Initialize WgmmaOp and SMem first
        var wgmma_op = Self.WgmmaOp()
        ref smem = external_memory[
            UInt8,
            address_space=.SHARED,
            alignment=128,
        ]().bitcast[Self.SMem]()[]

        # Common initialization
        var (
            warp_group_idx,
            warp_group_thread_idx,
            rank_m,
            rank_n,
            warp_id,
            lane_predicate,
        ) = Self.common_kernel_init()

        var pipeline = smem.create_pipeline()
        var barrier_handler = Self.TMABarrier(
            pipeline, Self.num_consumer, Self.cluster_size
        )

        # Create TileLoaderTMA loaders
        var a_loader, b_loader = Self.build_tma_loaders(
            a_tma_op, b_tma_op, rank_m, rank_n
        )

        Self.pipeline_init()

        # Calculate block swizzle
        var block_idx_swizzle = Self.get_block_swizzle()

        # The block may be OOB because we create blocks based the maximum
        # number of tokens per expert.
        var M = a_offsets[block_idx.z + 1][0] - a_offsets[block_idx.z][0]
        if UInt32(block_idx_swizzle[1] * Self.BM) >= M:
            return

        var a_start_row = a_offsets[block_idx.z][0]

        var expert = expert_ids[block_idx.z][0]
        # We use -1 to indicate that the block is not active for LoRA use cases.
        # but we still need to zero out the output for this case.
        var skip_matmul = expert < 0

        comptime N = Self.c_layout.static_shape[1]
        var b_start_row = expert * Int32(N)

        comptime CLUSTER_N = Self.cluster_shape[0]
        comptime CLUSTER_M = Self.cluster_shape[1]

        # Split thread blocks into producer and consumer warp groups
        if warp_group_idx == 0:
            # Producer warp group
            _ = Self.setup_producer()

            if warp_id == 0 and lane_predicate and not skip_matmul:
                var m_coord = (
                    block_idx.y * Self.BM if CLUSTER_N
                    > 1 else Int(a_start_row) + block_idx_swizzle[1] * Self.BM
                )
                var n_coord = (
                    block_idx.x * Self.BN if CLUSTER_M
                    > 1 else Int(b_start_row) + block_idx_swizzle[0] * Self.BN
                )

                if warp_id == 0 and lane_predicate:
                    Self.producer_main_loop_pipeline[num_k_iters=num_k_iters](
                        m_coord,
                        n_coord,
                        0,  # k_start,
                        a_loader,
                        b_loader,
                        barrier_handler,
                        pipeline,
                        smem.a_tiles(),
                        smem.b_tiles(),
                    )
        else:
            # Consumer warp groups
            var local_warp_group_idx, c_reg_tile, final_c_reg_tile = (
                Self.setup_consumer(warp_group_idx)
            )

            Self.consumer_arrive_empty_barriers(warp_group_thread_idx, pipeline)

            if not skip_matmul:
                Self.consumer_main_loop_pipeline[num_k_iters=num_k_iters](
                    wgmma_op,
                    local_warp_group_idx,
                    final_c_reg_tile,
                    c_reg_tile,
                    pipeline,
                    smem.a_tiles(),
                    smem.b_tiles(),
                    warp_group_thread_idx,
                )
            else:
                _ = c_reg_tile.fill(0.0)

            var output_reg_tile = (
                final_c_reg_tile if Self.a_type
                == .float8_e4m3fn else c_reg_tile
            )

            # C tile for current expert.
            var c_by_expert = TileTensor(
                c._offset_storage(Coord(a_start_row * UInt32(N))),
                row_major(Coord(Int(M), Idx[N])),
            )

            @__parameter
            def elementwise_epilogue_fn_wrapper[
                dtype: DType, width: SIMDLength, *, alignment: Int = 1
            ](idx: IndexList[2], val: SIMD[dtype, width]):
                comptime if Self.elementwise_lambda_fn:
                    comptime elementwise_epilogue = Self.elementwise_lambda_fn.value()
                    var batch_idx = IndexList[2](
                        Int(a_start_row + UInt32(idx[0])), idx[1]
                    )
                    elementwise_epilogue(batch_idx, val)

            Self.consumer_output[
                Optional[elementwise_epilogue_type](
                    elementwise_epilogue_fn_wrapper
                ) if Self.elementwise_lambda_fn else None
            ](
                c_tma_op,
                c_by_expert,
                smem.c_tile(),
                output_reg_tile,
                warp_group_thread_idx,
                local_warp_group_idx,
                thread_idx.x - WARPGROUP_SIZE,
                block_idx_swizzle[1],
                block_idx_swizzle[0],
            )

        Self.finalize_kernel()

    @staticmethod
    @always_inline
    def consumer_main_loop_pipeline[
        num_k_iters: Int,
    ](
        wgmma_op: Self.WgmmaOp,
        local_warp_group_idx: Int,
        final_c_reg_tile: Self.AccumRegTile,
        c_reg_tile: Self.AccumRegTile,
        mut pipeline: ProducerConsumerPipeline[
            Self.adjusted_num_pipeline_stages
        ],
        a_tiles: Self.SMem.ATileArray,
        b_tiles: Self.SMem.BTileArray,
        warp_group_thread_idx: Int,
    ):
        """Pipeline-based consumer loop using ProducerConsumerPipeline.

        This is an alternative implementation of consumer_main_loop that uses
        the SM100 ProducerConsumerPipeline for synchronization instead of RingBuffer.

        Parameters:
            num_k_iters: Number of K-dimension tiles the consumer processes
                in this loop.

        Args:
            wgmma_op: Tensor core operator for matrix multiplication.
            local_warp_group_idx: Index of this consumer warp group (0-based).
            final_c_reg_tile: Final accumulation register tile (for FP8 promotion).
            c_reg_tile: Working accumulation register tile.
            pipeline: ProducerConsumerPipeline for synchronized tile access.
            a_tiles: Tile array for A matrix in shared memory.
            b_tiles: Tile array for B matrix in shared memory.
            warp_group_thread_idx: Thread index within the warp group.
        """

        comptime if Self.a_type == .float8_e4m3fn:
            _ = final_c_reg_tile.fill(0.0)
        else:
            _ = c_reg_tile.fill(0.0)

        var fp8_promotion_iter = 0

        comptime num_full_k_iters = ceildiv(
            num_k_iters, Self.num_pipeline_stages
        )
        comptime num_remaining_k_iters = num_k_iters % Self.num_pipeline_stages

        @always_inline
        @__parameter
        def consumer_loop[
            num_pipeline_stages_to_unroll: Int,
        ]():
            comptime for _ in range(num_pipeline_stages_to_unroll):
                # Acquire consumer stage (waits for producer)
                var stage = pipeline.acquire_consumer()
                var slot = Int(stage.index())

                # Get tile slices for this stage
                var a_tile_slice = a_tiles.slice[Self.k_group_size](
                    slot * Self.k_group_size
                )
                var b_tile_slice = b_tiles.slice[Self.k_group_size](
                    slot * Self.k_group_size
                )

                comptime for k in range(Self.k_group_size):
                    var a_tile = a_tile_slice[k]
                    var b_tile = b_tile_slice[k]

                    Self.wgmma(
                        wgmma_op,
                        local_warp_group_idx,
                        a_tile,
                        b_tile,
                        c_reg_tile,
                    )

                # SM90-specific: cluster-aware barrier arrive
                comptime if Self.cluster_size > 1:
                    if warp_group_thread_idx < Self.cluster_size:
                        _ = stage.mbar()[].arrive_cluster(
                            UInt32(warp_group_thread_idx)
                        )
                else:
                    if warp_group_thread_idx == 0:
                        stage.arrive()

                # Release stage (advance to next) - signal already done above
                stage^.release_without_signal()

                comptime if Self.a_type == .float8_e4m3fn:
                    fp8_promotion_iter += 1
                    if fp8_promotion_iter == Self.promotion_frequency:
                        Self.promote_to_cuda_cores(c_reg_tile, final_c_reg_tile)
                        fp8_promotion_iter -= Self.promotion_frequency

        comptime if num_remaining_k_iters == 0:
            for k_iter in range(num_full_k_iters):
                consumer_loop[Self.adjusted_num_pipeline_stages]()
        else:
            for k_iter in range(num_full_k_iters - 1):
                consumer_loop[Self.adjusted_num_pipeline_stages]()
            consumer_loop[num_remaining_k_iters // Self.k_group_size]()

        # Final promotion for fp8 data type if num_k_iters % promotion_frequency != 0
        comptime if Self.a_type == .float8_e4m3fn:
            if fp8_promotion_iter != 0:
                Self.promote_to_cuda_cores(c_reg_tile, final_c_reg_tile)

    @staticmethod
    @always_inline
    def promote_to_cuda_cores(
        c_reg_tile: Self.AccumRegTile,
        final_c_reg_tile: Self.AccumRegTile,
    ):
        """Promote FP8 accumulation to higher precision using CUDA cores.

        When using FP8 data types, tensor cores accumulate in limited precision.
        To maintain accuracy over many accumulations, we periodically add the
        intermediate results to a higher-precision accumulator using CUDA cores.

        This technique is commonly used in production libraries like cuBLAS to
        achieve both high performance (from FP8 tensor cores) and good accuracy.

        Args:
            c_reg_tile: Current accumulation from tensor cores.
            final_c_reg_tile: Higher-precision accumulator (updated in place).
        """
        comptime assert c_reg_tile.dtype in (
            DType.float32,
            DType.float16,
        ), "Only support fp32 and fp16 data type in CUDA Core promotion"
        comptime assert (
            len(c_reg_tile.layout) == 2
        ), "Only support 2D layout in CUDA Core promotion"

        comptime num_mma = c_reg_tile.layout.shape[0].value()
        comptime c_frag_size = c_reg_tile.layout.shape[1].value()

        # Add tensor core results to higher-precision accumulator
        comptime for mma_id in range(num_mma):
            comptime for i in range(c_frag_size):
                final_c_reg_tile[mma_id, i] = rebind[Scalar[Self.accum_type]](
                    final_c_reg_tile[mma_id, i]
                ) + rebind[Scalar[Self.accum_type]](c_reg_tile[mma_id, i])

    @always_inline
    @staticmethod
    def wgmma(
        wgmma_op: Self.WgmmaOp,
        local_warp_group_idx: Int,
        a_tile: Self.SMem.ATileArray.Tile,
        b_tile: Self.SMem.BTileArray.Tile,
        c_reg_tile: Self.AccumRegTile,
    ):
        warpgroup_fence(c_reg_tile)
        wgmma_op.arrive()
        comptime scale_c = 0 if Self.a_type == DType.float8_e4m3fn else 1
        wgmma_op.wgmma[Self.num_consumer, scale_c=scale_c](
            a_tile,
            b_tile,
            c_reg_tile,
            local_warp_group_idx,
        )
        wgmma_op.commit_group()
        warpgroup_fence(c_reg_tile)
        wgmma_op.wait_group()


@always_inline
def find_K_alignment_upto_16B(row_bytes_arg: Int) -> Int:
    """Find alignment among 1B, 2B, 4B, 16B based on the row's bytes.

    This function determines the largest power-of-2 alignment (up to 16 bytes)
    that evenly divides the given row size. This is used to determine the
    optimal vector size for cp.async operations when K dimension alignment
    doesn't meet TMA requirements.

    Args:
        row_bytes_arg: Number of bytes in a row (K * sizeof(element)).

    Returns:
        Alignment in bytes (1, 2, 4, 8, or 16).
    """

    var row_bytes = row_bytes_arg
    var alignment = 1

    comptime for i in range(4):
        # Check if current alignment divides evenly
        if row_bytes & 1 == 1:
            return alignment
        row_bytes >>= 1
        alignment <<= 1

    return alignment
