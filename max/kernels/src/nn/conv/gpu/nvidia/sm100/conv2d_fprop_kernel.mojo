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

"""SM100 Conv2D Forward Propagation Kernel.

This module implements a high-performance Conv2D fprop kernel for NVIDIA
Blackwell (SM100) GPUs using the Structured Kernel architecture.

The kernel uses implicit GEMM to compute convolution:
- Maps conv to GEMM: C[M,N] = A[M,K] @ B[K,N]
- M = batch * out_h * out_w (output spatial)
- N = out_channels (filters)
- K = in_channels * filter_h * filter_w (reduction)

The implementation reuses matmul infrastructure:
- 8-warp specialization (scheduler, load, MMA, epilogue load, epilogue)
- TMA-based tile loading with im2col addressing
- TMEM accumulators
- Producer-consumer pipelining

Supported configurations (Flux VAE optimized):
- stride=1, dilation=1 (most common in VAE decoder)
- 3x3 and 1x1 kernels
- BF16/FP16 data types
"""

from std.collections import Optional
from std.math import ceildiv

from std.sys import size_of

from max.gpu import WARP_SIZE
from max.gpu.sync import barrier
from max.gpu.primitives.cluster import cluster_sync, elect_one_sync
from max.gpu.memory import external_memory, fence_mbarrier_init
from max.gpu.compute.arch.mma_nvidia_sm100 import *
from max.gpu.sync import syncwarp
from max.gpu.compute.arch.tcgen05 import *
from layout import Layout as LegacyLayout
from layout.tma_async import TMATensorTile
from structured_kernels.tile_types import (
    TmaOpType,
    TmaOpTypeIm2col,
    static_row_major,
)

from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple

from linalg.arch.sm100 import MmaOpSM100_SS
from structured_kernels.tile_types import (
    SMemTileArray2DRowMajor,
)

# Import shared components from matmul structured kernels
from structured_kernels.kernel_common import (
    WarpRole,
    KernelContext,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.tile_pipeline import (
    InputTilePipeline,
    StandardTilePayload,
    ProducerTiles,
    ConsumerTiles,
    OutputTilePipeline,
)
from structured_kernels.barriers import (
    WarpGroupBarrier,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    OutputPipelineConfig,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.tmem import (
    TmemAllocation,
    TmemDeallocBarrier,
    TmemTensor,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.warp_context import (
    MmaWarpContext,
    EpilogueWarpContext,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.tile_loader import (
    TileLoaderTMA,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.tile_scheduler import (
    TileScheduler,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.output_writer import (
    TileWriter,
)
from linalg.utils import (
    elementwise_compute_lambda_type,
    elementwise_epilogue_type,
)

from .conv_config import Conv2dConfig
from .conv_smem import Conv2dSmem
from .conv_tile_loader import TileLoaderTMAIm2col
from .epilogue_load_pipeline import EpiLoadPipeline, LoadOrderBarrier


# =============================================================================
# Conv2dFpropKernel - SM100 structured conv2d forward kernel
# =============================================================================


struct Conv2dFpropKernel[
    # Core types
    act_type: DType,
    filter_type: DType,
    out_type: DType,
    # Configuration
    config: Conv2dConfig[act_type, filter_type, out_type],
    # Cluster shape
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1),
    # Optional epilogue lambda for fusion (bias, activation, residual add)
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
]:
    """SM100 Conv2D forward propagation kernel.

    This kernel implements conv2d fprop using implicit GEMM with warp
    specialization. It reuses the matmul kernel architecture but with
    convolution-specific address calculation.

    The kernel structure:
    - Scheduler warp: CLC-based tile scheduling
    - Load warp: TMA loads with im2col transformation
    - MMA warp: Tensor core operations
    - Epilogue warps: Output from TMEM to GMEM

    Parameters:
        act_type: Activation data type.
        filter_type: Filter data type.
        out_type: Output data type.
        config: Kernel configuration.
        cluster_shape: CUDA cluster dimensions.
        elementwise_lambda_fn: Optional void epilogue lambda applied after
            output write. Signature: `def(IndexList[2], SIMD) -> None`.
        elementwise_compute_lambda_fn: Optional epilogue lambda for fusion
            (bias add, activation functions, residual connections).
        register_based_epilogue: Whether to apply the lambda in registers.
    """

    # ========== Derived Constants ==========
    comptime BM = Self.config.block_tile_shape[0]
    comptime BN = Self.config.block_tile_shape[1]
    comptime BK = Self.config.block_tile_shape[2]

    comptime MMA_M = Self.config.mma_shape[0]
    comptime MMA_N = Self.config.mma_shape[1]
    comptime MMA_K = Self.config.mma_shape[2]

    comptime OutputM = Self.config.output_tile_shape[0]
    comptime OutputN = Self.config.output_tile_shape[1]

    comptime accum_type = Self.config.accum_type()
    comptime cta_group = Self.config.cta_group

    comptime CLUSTER_M: Int = Self.config.cluster_shape[0]
    comptime CLUSTER_N: Int = Self.config.cluster_shape[1]
    comptime CLUSTER_SIZE = Self.CLUSTER_M * Self.CLUSTER_N

    # ========== Thread/Warp Organization ==========
    # Warp-specialized kernel structure (8 warps = 256 threads):
    # - Epilogue: 4 warps (warp IDs 0-3) - output writing
    # - Scheduler: 1 warp (warp ID 4) - work distribution
    # - MainLoad: 1 warp (warp ID 5) - TMA loads for activation/filter tiles
    # - MMA: 1 warp (warp ID 6) - tensor core operations
    # - EpilogueLoad: 1 warp (warp ID 7) - TMA loads for source C (residual)
    #
    # This matches CUTLASS's warp layout. When residual is not enabled,
    # the EpilogueLoad warp participates in barrier synchronization but
    # performs no actual work.
    comptime num_output_warps = 4
    comptime SCHEDULER_THREADS = WARP_SIZE
    comptime TMA_LOAD_THREADS = WARP_SIZE
    comptime MMA_THREADS = WARP_SIZE
    comptime EPILOGUE_LOAD_THREADS = WARP_SIZE
    comptime EPILOGUE_THREADS = Self.num_output_warps * WARP_SIZE

    comptime NUM_THREADS = (
        Self.SCHEDULER_THREADS
        + Self.TMA_LOAD_THREADS
        + Self.MMA_THREADS
        + Self.EPILOGUE_LOAD_THREADS
        + Self.EPILOGUE_THREADS
    )  # = 256 threads

    # ========== Pipeline Configuration ==========
    comptime num_pipeline_stages = Self.config.num_pipeline_stages
    comptime num_group_pipeline_stages = (
        Self.num_pipeline_stages // Self.config.k_group_size
    )
    comptime num_clc_pipeline_stages: Int = Self.config.num_clc_pipeline_stages
    comptime num_accum_pipeline_stages = Self.config.num_accum_pipeline_stages
    comptime num_output_stages: Int = Self.config.num_output_stages

    # TMEM configuration
    comptime NUM_TMEM_COLS = 512
    comptime stage_stride_cols = Self.NUM_TMEM_COLS // Self.num_accum_pipeline_stages

    # ========== Barrier Arrival Counts ==========
    comptime clc_producer_arv_count = 1
    # All warps except scheduler participate in CLC pipeline as consumers
    comptime clc_consumer_arv_count = Self.SCHEDULER_THREADS + Self.CLUSTER_SIZE * (
        Self.TMA_LOAD_THREADS
        + Self.MMA_THREADS
        + Self.EPILOGUE_LOAD_THREADS
        + Self.EPILOGUE_THREADS
    )
    comptime clc_throttle_producer_arv_count = Self.TMA_LOAD_THREADS
    comptime clc_throttle_consumer_arv_count = Self.SCHEDULER_THREADS
    comptime accum_pipeline_producer_arv_count = 1
    comptime accum_pipeline_consumer_arv_count = Self.cta_group * Self.EPILOGUE_THREADS

    # ========== Shared Memory Type ==========
    comptime SmemType = Conv2dSmem[
        Self.act_type,
        Self.filter_type,
        Self.out_type,
        config=Self.config,
    ]

    # ========== MMA Operation Type ==========
    comptime MmaOp = MmaOpSM100_SS[
        Self.out_type,
        Self.act_type,
        Self.filter_type,
        Self.config.block_tile_shape,
        Self.config.mma_shape,
        accum_type=Self.accum_type,
        cta_group=Self.cta_group,
        cluster_shape=Self.config.cluster_shape,
        a_swizzle=Self.config.a_swizzle,
        b_swizzle=Self.config.b_swizzle,
        transpose_b=True,  # Filter is transposed
    ]

    # ========== Tile Scheduler Type ==========
    comptime Scheduler = TileScheduler[
        num_stages=Self.num_clc_pipeline_stages,
        cluster_shape=Index[dtype=DType.uint32](
            Self.config.cluster_shape[0],
            Self.config.cluster_shape[1],
            Self.config.cluster_shape[2],
        ),
        block_swizzle_size=Self.config.block_swizzle_size,
    ]

    # ========== Tile Pipeline Type ==========
    comptime TilePayload = StandardTilePayload[
        Self.act_type,
        Self.filter_type,
        IndexList[2](Self.BM, Self.BK),
        IndexList[2](Self.BN, Self.BK),
        Self.SmemType.num_pipeline_stages,
    ]
    comptime InputTilePipelineType = InputTilePipeline[
        Self.TilePayload,
        Self.SmemType.num_group_pipeline_stages,
        Self.config.k_group_size,
    ]

    # ========== Tile Loader Types ==========
    comptime ActTileLoaderTypeIm2col = TileLoaderTMAIm2col[
        ..., cta_group=Self.cta_group
    ]
    comptime FilterTileLoaderType = TileLoaderTMA[..., cta_group=Self.cta_group]
    # Source C tile loader for residual (same structure as output)
    comptime SrcTileLoaderType = TileLoaderTMA[..., cta_group=1]

    # TMA expected bytes
    comptime act_expected_bytes = Self.SmemType.act_smem_elements * size_of[
        Self.act_type
    ]()
    comptime filter_expected_bytes = Self.SmemType.filter_smem_elements * size_of[
        Self.filter_type
    ]()
    comptime input_expected_bytes = Self.cta_group * (
        Self.act_expected_bytes + Self.filter_expected_bytes
    ) * Self.config.k_group_size
    # Source C TMA expected bytes (one output tile)
    comptime src_expected_bytes = Self.OutputM * Self.OutputN * size_of[
        Self.out_type
    ]()

    # TMA descriptor sizes
    # ========== TMA Layouts (computed from config, new Layout types) ==========

    comptime act_tile_dim0 = Self.BM // Self.CLUSTER_N
    comptime filter_tile_dim0 = Self.BN // (Self.CLUSTER_M // Self.cta_group)
    comptime act_swizzle_elems = Self.config.a_swizzle.bytes() // size_of[
        Self.act_type
    ]()
    comptime filter_swizzle_elems = Self.config.b_swizzle.bytes() // size_of[
        Self.filter_type
    ]()
    comptime out_swizzle_elems = Self.config.c_swizzle.bytes() // size_of[
        Self.out_type
    ]()

    # C tile shape -- same logic as matmul kernels
    comptime out_tile_dim0 = Self.OutputM if (
        Self.MMA_M == 256 or Self.cta_group == 1
    ) else 64

    # Activation: 2D im2col TMA layout
    comptime ActTileLayout = static_row_major[Self.act_tile_dim0, Self.BK]
    comptime ActDescLayout = static_row_major[
        Self.act_tile_dim0, Self.act_swizzle_elems
    ]

    # Filter: 2D standard TMA layout
    comptime FilterTileLayout = static_row_major[Self.filter_tile_dim0, Self.BK]
    comptime FilterDescLayout = static_row_major[
        Self.filter_tile_dim0, Self.filter_swizzle_elems
    ]

    # Output: 2D standard TMA layout
    comptime OutTileLayout = static_row_major[Self.out_tile_dim0, Self.OutputN]
    comptime OutDescLayout = static_row_major[
        Self.out_tile_dim0, Self.out_swizzle_elems
    ]

    # Source C (residual): same shape as output
    comptime SrcTileLayout = Self.OutTileLayout
    comptime SrcDescLayout = Self.OutDescLayout

    # TMA operation types
    comptime ActTmaOp = TmaOpTypeIm2col[
        Self.act_type, Self.ActTileLayout, Self.ActDescLayout
    ]
    comptime FilterTmaOp = TmaOpType[
        Self.filter_type, Self.FilterTileLayout, Self.FilterDescLayout
    ]
    comptime OutTmaOp = TmaOpType[
        Self.out_type, Self.OutTileLayout, Self.OutDescLayout
    ]
    comptime SrcTmaOp = TmaOpType[
        Self.out_type, Self.SrcTileLayout, Self.SrcDescLayout
    ]

    # TMA load size constants
    comptime act_tma_load_size = Self.act_tile_dim0 * Self.act_swizzle_elems
    comptime filter_tma_load_size = Self.filter_tile_dim0 * Self.filter_swizzle_elems
    comptime act_tma_rows = Self.act_tile_dim0
    comptime filter_tma_rows = Self.filter_tile_dim0

    # ========== Output Pipeline Configuration ==========
    comptime opc = OutputPipelineConfig(
        Self.num_accum_pipeline_stages,
        Self.stage_stride_cols,
        Self.cta_group,
    )

    # ========== Tensor Memory Type ==========
    comptime Tmem = TmemAllocation[Self.opc.cta_group]
    comptime accum_layout = LegacyLayout.row_major(Self.MMA_M, Self.MMA_N)
    comptime AccumTensor = TmemTensor[
        Self.accum_type, Self.accum_layout, cta_group=Self.cta_group
    ]

    # ========== Output Pipeline Type ==========
    comptime OutputPipeline = OutputTilePipeline[Self.opc]

    # ========== Epilogue Load Pipeline Type ==========
    # For source C loading (residual add: D = Conv + beta*C)
    comptime num_epi_load_stages = Self.SmemType.num_epi_load_stages
    comptime EpiLoadPipelineType = EpiLoadPipeline[Self.num_epi_load_stages]

    # Arrive counts for epilogue load pipeline
    comptime epi_load_producer_arv_count: Int32 = 1  # TMA transaction
    comptime epi_load_consumer_arv_count: Int32 = Int32(
        Self.EPILOGUE_THREADS
    )  # 128 epilogue threads

    # Warp synchronization
    comptime MmaEpilogueSync = WarpGroupBarrier[
        Self.MMA_THREADS + Self.EPILOGUE_THREADS, 1
    ]
    comptime TmemDealloc = TmemDeallocBarrier[Self.opc.cta_group]

    # Warp contexts
    comptime MmaCtx = MmaWarpContext[
        Self.opc,
        Self.MMA_THREADS,
        Self.EPILOGUE_THREADS,
    ]
    comptime EpilogueCtx = EpilogueWarpContext[
        Self.opc,
        Self.MMA_THREADS,
        Self.EPILOGUE_THREADS,
    ]

    # ========== Output Writer ==========
    comptime TileWriterType = TileWriter[
        a_type=Self.act_type,
        accum_type=Self.accum_type,
        block_tile_shape=Self.config.block_tile_shape,
        mma_shape=Self.config.mma_shape,
        opc=Self.opc,
        c_swizzle=Self.config.c_swizzle,
        transpose_c=False,
        c_smem_dim0=Self.SmemType.OutputM,
        c_smem_dim1=Self.SmemType.OutputN,
        num_output_stages=Self.SmemType.num_output_stages,
        num_output_warps=Self.num_output_warps,
        # Epilogue lambda for fusion (bias, activation, residual add)
        elementwise_lambda_fn=Self.elementwise_lambda_fn,
        elementwise_compute_lambda_fn=Self.elementwise_compute_lambda_fn,
        register_based_epilogue=Self.register_based_epilogue,
    ]

    # ========== Source C Tile Type (for write_with_residual) ==========
    # TileTensor-based source C tile array. num_stages must match the
    # underlying `SourceTileStorage` buffer count (= num_epi_load_stages,
    # one buffer per epilogue sub-stage); not `num_output_stages` (which is
    # the OUTPUT SMEM staging count, independent of source).
    comptime SrcCTileArray = SMemTileArray2DRowMajor[
        Self.out_type,
        Self.SmemType.OutputM,
        Self.SmemType.OutputN,
        Self.SmemType.num_epi_load_stages,
        128,
    ]

    # ========== Kernel Context ==========
    comptime Context = KernelContext[
        Self.num_clc_pipeline_stages,
        Self.cta_group,
        Self.CLUSTER_M,
        Self.CLUSTER_N,
    ]

    # ========== Static Methods ==========

    @staticmethod
    @always_inline
    def mma[
        tiles_origin: MutOrigin,
        //,
    ](
        tmem_stage: Self.OutputPipeline.Stage.Tmem,
        tiles: ConsumerTiles[
            tiles_origin,
            Self.TilePayload,
            Self.SmemType.num_group_pipeline_stages,
            Self.config.k_group_size,
        ],
        mma_op: MmaOpSM100_SS,
        elect_one_warp: Bool,
        iter_idx: UInt32,
        k_start: UInt32,
    ):
        """Execute MMA operations for one pipeline stage.

        Parameters:
            tiles_origin: Borrow origin of the consumer tile payload
                (inferred).

        Args:
            tmem_stage: TMEM accumulator stage to accumulate results
                into.
            tiles: Consumer tile payload and barrier handle holding the
                activation and filter tiles for the current stage.
            mma_op: SM100 tensor core MMA operation object.
            elect_one_warp: Whether this warp is elected for cluster-wide
                operations.
            iter_idx: K-dimension tile index of the current MMA iteration.
            k_start: K-dimension tile index at which accumulation begins,
                used to initialize the accumulator on the first tile.
        """
        var accum = tmem_stage.tensor[Self.accum_type, Self.accum_layout]()

        if elect_one_sync():
            comptime for j in range(Self.config.k_group_size):
                var act_tile, filter_tile = tiles.payload().get_tile[
                    Self.config.k_group_size
                ](tiles.stage(), j)
                var is_first_k = (iter_idx + UInt32(j)) == k_start
                mma_op.mma(
                    act_tile,
                    filter_tile,
                    UInt32(accum.offset()),
                    init_c=is_first_k,
                )
            mma_op.commit(tiles.mbar())

    # ========== Im2col TMA-based convolution ==========

    @staticmethod
    @always_inline
    def init_barriers(
        ctx: Self.Context,
        act_tma_op: Self.ActTmaOp,
        filter_tma_op: Self.FilterTmaOp,
        out_tma_op: Self.OutTmaOp,
        input_barriers: Self.SmemType.Pipelines.InputBarriers,
        accum_barriers: Self.SmemType.Pipelines.AccumBarriers,
        clc_throttle: Self.SmemType.Pipelines.ClcThrottleBarriers,
        clc_full: Self.SmemType.Pipelines.ClcBarriers,
        clc_empty: Self.SmemType.Pipelines.ClcBarriers,
        tmem_dealloc: Self.SmemType.Pipelines.TmemDealloc,
        epi_load_barriers: Self.SmemType.EpiLoadBarriers,
        load_order_barrier: Self.SmemType.LoadOrderBarriers,
    ):
        """Initialize barriers and prefetch TMA descriptors.

        Args:
            ctx: Kernel context holding per-CTA state and multicast masks.
            act_tma_op: Im2col TMA descriptor for the activation tensor.
            filter_tma_op: TMA descriptor for the filter tensor.
            out_tma_op: TMA descriptor for the output tensor.
            input_barriers: Producer-consumer barriers for the input tile
                pipeline.
            accum_barriers: Producer-consumer barriers for the TMEM
                accumulator pipeline.
            clc_throttle: Throttle barriers pacing the CLC scheduler against
                the load warps.
            clc_full: CLC barriers signalling tile slots are full, producer
                to consumer.
            clc_empty: CLC barriers signalling tile slots are empty,
                consumer to producer.
            tmem_dealloc: Barrier governing TMEM accumulator deallocation.
            epi_load_barriers: Barriers for the epilogue load pipeline
                (source C residual loading).
            load_order_barrier: Barrier coordinating MainLoad completion
                with EpilogueLoad.
        """
        if ctx.elect_one_warp and ctx.elect_one_thread:
            # Prefetch TMA descriptors
            act_tma_op.prefetch_descriptor()
            filter_tma_op.prefetch_descriptor()
            out_tma_op.prefetch_descriptor()

            # Initialize pipeline barriers
            Self.InputTilePipelineType.init_barriers(
                input_barriers.ptr,
                Int32(1),
                Int32(
                    Self.config.cluster_shape[0] // Self.config.cta_group
                    + Self.config.cluster_shape[1]
                    - 1
                ),
            )
            Self.OutputPipeline.init_barriers(
                accum_barriers.ptr,
                Self.accum_pipeline_producer_arv_count,
                Int32(Self.accum_pipeline_consumer_arv_count),
            )
            Self.Scheduler.init_throttle_barriers(
                clc_throttle.ptr,
                Int32(Self.clc_throttle_producer_arv_count),
                Int32(Self.clc_throttle_consumer_arv_count),
            )

            # Initialize TMEM deallocation barrier
            tmem_dealloc.ptr[].init(
                Int32(Self.EPILOGUE_THREADS * Self.config.cta_group)
            )

            # Initialize CLC barriers
            comptime for i in range(Self.config.num_clc_pipeline_stages):
                clc_full.ptr[i].init(Self.clc_producer_arv_count)
                clc_empty.ptr[i].init(Int32(Self.clc_consumer_arv_count))

            # Initialize epilogue load pipeline barriers (for residual C loading)
            var epi_load_pipeline = Self.EpiLoadPipelineType(
                epi_load_barriers.ptr
            )
            epi_load_pipeline.init_barriers(
                Self.epi_load_producer_arv_count,
                Self.epi_load_consumer_arv_count,
            )

            # Initialize load order barrier (MainLoad → EpilogueLoad coordination)
            var load_order = LoadOrderBarrier(load_order_barrier.ptr)
            load_order.init(arrive_count=1)

        fence_mbarrier_init()
        cluster_sync()

    @staticmethod
    @always_inline
    def load_input_tiles[
        act_tma_origin: ImmOrigin,
        filter_tma_origin: ImmOrigin,
        tiles_origin: MutOrigin,
        //,
    ](
        act_loader: TileLoaderTMAIm2col[
            act_tma_origin,
            Self.act_type,
            Self.ActTmaOp.rank,
            Self.ActTmaOp.tile_shape,
            Self.ActTmaOp.desc_shape,
            cta_group=Self.cta_group,
        ],
        filter_loader: TileLoaderTMA[
            filter_tma_origin,
            Self.filter_type,
            Self.FilterTmaOp.rank,
            Self.FilterTmaOp.tile_shape,
            Self.FilterTmaOp.desc_shape,
            cta_group=Self.cta_group,
        ],
        tiles: ProducerTiles[
            tiles_origin,
            Self.TilePayload,
            Self.SmemType.num_group_pipeline_stages,
            Self.config.k_group_size,
        ],
        iter_idx: UInt32,
        work_m_coord: Int,
        work_n_coord: Int,
        peer_cta_coord: Tuple[Int, Int, Int],
        elect_one_cta: Bool,
    ):
        """Load activation (via im2col TMA) and filter tiles.

        The im2col TMA descriptor handles coordinate transformation
        internally. Coordinates are in GEMM space.

        Parameters:
            act_tma_origin: Borrow origin of the activation TMA descriptor
                (inferred).
            filter_tma_origin: Borrow origin of the filter TMA descriptor
                (inferred).
            tiles_origin: Borrow origin of the producer tile payload
                (inferred).

        Args:
            act_loader: Im2col TMA tile loader for activation tiles.
            filter_loader: TMA tile loader for filter tiles.
            tiles: Producer tile payload and barrier handle for the current
                stage.
            iter_idx: K-dimension tile index (C * R * S) of the current
                load.
            work_m_coord: M coordinate in GEMM space (batch * H_out * W_out).
            work_n_coord: N coordinate in GEMM space (output channels).
            peer_cta_coord: Peer CTA slicing coordinates as
                `(peer_rank_n, peer_rank_m, peer_m_rank)` for
                cluster-distributed tiles.
            elect_one_cta: Whether this CTA is elected for cluster-wide
                operations.
        """
        var peer_rank_n = peer_cta_coord[0]
        var peer_rank_m = peer_cta_coord[1]
        var peer_m_rank = peer_cta_coord[2]

        # Coordinates for TMA
        var act_gmem_m_coord = (
            peer_m_rank * Self.act_tma_rows + work_m_coord * Self.BM
        )
        var filter_gmem_n_coord = (
            peer_rank_m * Self.filter_tma_rows
            + peer_rank_n * Self.BN
            + work_n_coord * Self.MMA_N
        )

        if elect_one_sync():
            if elect_one_cta:
                tiles.expect_bytes(Self.input_expected_bytes)

            var barrier = tiles.barrier()

            comptime for j in range(Self.config.k_group_size):
                var act_tile, filter_tile = tiles.payload().get_tile[
                    Self.config.k_group_size
                ](tiles.stage(), j)

                # Peer CTA slicing
                var act_peer_tile = type_of(act_tile)(
                    act_tile.ptr + peer_m_rank * Self.act_tma_load_size,
                    act_tile.layout,
                )
                var filter_peer_tile = type_of(filter_tile)(
                    filter_tile.ptr + peer_rank_m * Self.filter_tma_load_size,
                    filter_tile.layout,
                )

                var k_coord = Int(iter_idx + UInt32(j)) * Self.BK

                # Load tiles - act_loader uses im2col TMA
                act_loader.load(
                    act_peer_tile,
                    barrier[0],
                    k_coord,
                    act_gmem_m_coord,
                )
                filter_loader.load(
                    filter_peer_tile,
                    barrier[0],
                    k_coord,
                    filter_gmem_n_coord,
                )

    @staticmethod
    @always_inline
    @__llvm_metadata(`nvvm.cluster_dim`=Self.cluster_shape)
    @__llvm_arg_metadata(act_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(filter_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(out_tma_op, `nvvm.grid_constant`)
    @__name(
        t"sm100_conv2d_fprop_{Self.act_type}_{Self.filter_type}_{Self.out_type}",
    )
    def run(
        act_tma_op: Self.ActTmaOp,
        filter_tma_op: Self.FilterTmaOp,
        out_tma_op: Self.OutTmaOp,
        cluster_dim: StaticTuple[Int32, 3],
        mnk: StaticTuple[UInt32, 3],
    ):
        """Kernel entry point for Conv2D fprop (no residual).

        Args:
            act_tma_op: Im2col TMA descriptor for activation.
            filter_tma_op: TMA descriptor for filter.
            out_tma_op: TMA descriptor for output.
            cluster_dim: Cluster dimensions.
            mnk: GEMM dimensions (M, N, K).
        """
        Self._run_impl[
            has_residual=False,
            _src_rank=Self.OutTmaOp.rank,
            _src_tile_shape=Self.OutTmaOp.tile_shape,
            _src_desc_shape=Self.OutTmaOp.desc_shape,
        ](
            act_tma_op,
            filter_tma_op,
            out_tma_op,
            out_tma_op,  # Unused dummy for src_tma_op
            cluster_dim,
            mnk,
            Float32(0.0),
        )

    @staticmethod
    @always_inline
    @__llvm_metadata(`nvvm.cluster_dim`=Self.cluster_shape)
    @__llvm_arg_metadata(act_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(filter_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(out_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(src_tma_op, `nvvm.grid_constant`)
    @__name(
        t"sm100_conv2d_fprop_with_residual_{Self.act_type}_{Self.filter_type}_{Self.out_type}",
    )
    def run_with_residual(
        act_tma_op: Self.ActTmaOp,
        filter_tma_op: Self.FilterTmaOp,
        out_tma_op: Self.OutTmaOp,
        src_tma_op: Self.SrcTmaOp,
        cluster_dim: StaticTuple[Int32, 3],
        mnk: StaticTuple[UInt32, 3],
        beta: Float32,
    ):
        """Kernel entry point for Conv2D fprop with residual (D = Conv + beta*C).

        Args:
            act_tma_op: Im2col TMA descriptor for activation.
            filter_tma_op: TMA descriptor for filter.
            out_tma_op: TMA descriptor for output D.
            src_tma_op: TMA descriptor for source C (residual input).
            cluster_dim: Cluster dimensions.
            mnk: GEMM dimensions (M, N, K).
            beta: Residual scale factor.
        """
        Self._run_impl[
            has_residual=True,
            _src_rank=Self.SrcTmaOp.rank,
            _src_tile_shape=Self.SrcTmaOp.tile_shape,
            _src_desc_shape=Self.SrcTmaOp.desc_shape,
        ](
            act_tma_op,
            filter_tma_op,
            out_tma_op,
            src_tma_op,
            cluster_dim,
            mnk,
            beta,
        )

    # ========== Unified Kernel Implementation ==========

    @staticmethod
    @always_inline
    def _run_impl[
        has_residual: Bool,
        _src_rank: Int = Self.SrcTmaOp.rank,
        _src_tile_shape: IndexList[_src_rank] = Self.SrcTmaOp.tile_shape,
        _src_desc_shape: IndexList[_src_rank] = Self.SrcTmaOp.desc_shape,
    ](
        act_tma_op: Self.ActTmaOp,
        filter_tma_op: Self.FilterTmaOp,
        out_tma_op: Self.OutTmaOp,
        src_tma_op: TMATensorTile[
            Self.out_type, _src_rank, _src_tile_shape, _src_desc_shape
        ],
        cluster_dim: StaticTuple[Int32, 3],
        mnk: StaticTuple[UInt32, 3],
        beta: Float32,
    ):
        """Unified Conv2D fprop implementation with optional residual.

        When has_residual is False, the epilogue load warp is a no-op and
        the epilogue uses standard write. When True, the epilogue load warp
        pre-fetches source C via TMA and the epilogue applies D = Conv + beta*C
        in registers.

        Parameters:
            has_residual: Whether to load source C and apply residual add.
            _src_rank: Rank of source C TMA tile/descriptor shapes (internal,
                set by entry points).
            _src_tile_shape: Source C TMA tile shape (internal, set by
                entry points).
            _src_desc_shape: Source C TMA descriptor shape (internal, set
                by entry points).

        Args:
            act_tma_op: Im2col TMA descriptor for activation.
            filter_tma_op: TMA descriptor for filter.
            out_tma_op: TMA descriptor for output.
            src_tma_op: TMA descriptor for source C (only used when
                has_residual is True).
            cluster_dim: Cluster dimensions.
            mnk: GEMM dimensions (M, N, K).
            beta: Residual scale factor (only used when has_residual is True).
        """
        # Access shared memory
        ref smem = external_memory[
            UInt8,
            address_space=.SHARED,
            alignment=128,
        ]().bitcast[Self.SmemType]()[]

        # Create input pipeline with payload
        var tile_payload = Self.TilePayload(
            smem.act_tiles(), smem.filter_tiles()
        )
        var input_pipeline = Self.InputTilePipelineType(
            smem.pipelines.input_barriers(), tile_payload
        )

        # Create kernel context
        var ctx = Self.Context(smem.pipelines.tmem_addr())

        # Initialize barriers
        Self.init_barriers(
            ctx,
            act_tma_op,
            filter_tma_op,
            out_tma_op,
            smem.pipelines.input_barriers(),
            smem.pipelines.accum_barriers(),
            smem.pipelines.clc_throttle(),
            smem.pipelines.clc_full(),
            smem.pipelines.clc_empty(),
            smem.pipelines.tmem_dealloc(),
            smem.epi_load_barriers(),
            smem.get_load_order_barrier(),
        )

        # Create epilogue load pipeline and load order barrier
        var epi_load_pipeline = Self.EpiLoadPipelineType(
            smem.epi_load_barriers().ptr
        )
        var load_order_barrier = LoadOrderBarrier(
            smem.get_load_order_barrier().ptr
        )

        var mma_op = Self.MmaOp()

        # Create scheduler
        var scheduler = Self.Scheduler(
            cluster_dim,
            smem.pipelines.clc_response(),
            smem.pipelines.clc_full(),
            smem.pipelines.clc_empty(),
            smem.pipelines.clc_throttle(),
        )

        # Create tile loaders
        var act_loader = Self.ActTileLoaderTypeIm2col(
            Pointer(to=act_tma_op), ctx.a_multicast_mask
        )
        var filter_loader = Self.FilterTileLoaderType(
            Pointer(to=filter_tma_op), ctx.b_multicast_mask
        )

        var num_iters: UInt32 = ceildiv(mnk[2], UInt32(Self.BK))

        # ========== Warp-Specialized Execution ==========

        if WarpRole.is_main_load():
            var load_iter = scheduler.work_iterator()

            with input_pipeline.producer() as producer:
                for current in load_iter:
                    scheduler.throttle_signal(ctx.is_first_cta_in_cluster)

                    # Prologue/steady-state split for LoadOrderBarrier
                    var num_prologue = min(
                        UInt32(Self.num_pipeline_stages),
                        num_iters,
                    )

                    # PROLOGUE: Fill pipeline stages
                    for i in range(
                        0, Int(num_prologue), Self.config.k_group_size
                    ):
                        with producer.acquire() as tiles:
                            Self.load_input_tiles(
                                act_loader,
                                filter_loader,
                                tiles,
                                UInt32(i),
                                Int(current.m),
                                Int(current.n),
                                ctx.peer_cta_coord,
                                ctx.elect_one_cta,
                            )

                    # Signal LoadOrderBarrier after prologue
                    if elect_one_sync():
                        load_order_barrier.arrive_and_step()

                    # STEADY-STATE: Continue with remaining iterations
                    for i in range(
                        Int(num_prologue),
                        Int(num_iters),
                        Self.config.k_group_size,
                    ):
                        with producer.acquire() as tiles:
                            Self.load_input_tiles(
                                act_loader,
                                filter_loader,
                                tiles,
                                UInt32(i),
                                Int(current.m),
                                Int(current.n),
                                ctx.peer_cta_coord,
                                ctx.elect_one_cta,
                            )

                    syncwarp()

                producer.drain()

        if WarpRole.is_scheduler() and ctx.is_first_cta_in_cluster:
            comptime if Self.config.num_clc_pipeline_stages == 0:
                return

            var sched_iter = scheduler.scheduler_iterator()

            for _ in sched_iter:
                sched_iter.signal_and_advance()

            sched_iter.drain()

        if WarpRole.is_epilogue_load():
            var epi_load_iter = scheduler.work_iterator()

            # Epilogue load warp: participates in work loop for CLC barrier
            # counts. When has_residual is True, pre-fetches source C via TMA.
            for current in epi_load_iter:
                load_order_barrier.wait_and_step()

                comptime if has_residual:
                    # Produce C tile into SMEM. Match CUTLASS
                    # `sm100_epilogue_tma_warpspecialized`: one TMA load per
                    # epilogue sub-stage, each (OutputM, OutputN), pipelined
                    # across `num_epi_load_stages` SMEM buffers. Coords are
                    # `(work.m * BM, work.n * MMA_N + k * OutputN)` — note
                    # the unit is `MMA_N` (the full block stride in N), not
                    # `OutputN` as a previous revision had it (MODELS-1484).
                    comptime _num_epi_stages = (
                        Self.SmemType.num_epi_load_stages
                    )
                    comptime assert (
                        _num_epi_stages == Self.MMA_N // Self.OutputN
                    ), "num_epi_load_stages must equal MMA_N / OutputN"
                    comptime for k in range(_num_epi_stages):
                        epi_load_pipeline.wait_consumer()
                        if elect_one_sync():
                            var mbar = epi_load_pipeline.producer_mbar()
                            mbar[0].expect_bytes(Int32(Self.src_expected_bytes))
                            var _prod_stage = (
                                epi_load_pipeline.pipeline.producer_stage()
                            )
                            # TMA coords are (col, row) order — fast-varying
                            # dim first, matching CUDA's TMA convention used
                            # by the matching output TMA store at
                            # `epilogue_components.mojo:519`. A previous
                            # revision passed (row, col), which silently read
                            # the wrong GMEM region — masked by random-input
                            # tests (MODELS-1484).
                            var _row_coord = Int(current.m) * Self.BM
                            var _col_coord = (
                                Int(current.n) * Self.MMA_N + k * Self.OutputN
                            )
                            src_tma_op.async_copy[1](
                                smem.src_tiles()[Int(_prod_stage)],
                                mbar[0],
                                (_col_coord, _row_coord),
                            )
                        epi_load_pipeline.producer_step()

        if WarpRole.is_mma():
            var mma_iter = scheduler.work_iterator()

            var tmem = Self.Tmem.allocate(smem.pipelines.tmem_addr())
            var mma_ctx = Self.MmaCtx(
                tmem,
                Self.OutputPipeline(
                    smem.pipelines.accum_barriers().ptr,
                    tmem,
                    UInt16(ctx.mma_complete_mask),
                ),
                Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
            )

            with mma_ctx:
                for _ in mma_iter:
                    if ctx.elect_one_cta:
                        with mma_ctx.output_pipeline.producer() as output_stage:
                            with input_pipeline.consumer() as consumer:
                                for i in range(
                                    0,
                                    Int(num_iters),
                                    Self.config.k_group_size,
                                ):
                                    with consumer.acquire() as input_tiles:
                                        Self.mma(
                                            output_stage.tmem,
                                            input_tiles,
                                            mma_op,
                                            ctx.elect_one_warp,
                                            UInt32(i),
                                            0,
                                        )

        if WarpRole.is_epilogue():
            Self.EpilogueCtx.Sync.wait()

            var tmem = Self.Tmem.from_shared(smem.pipelines.tmem_addr())
            var epi_ctx = Self.EpilogueCtx(
                tmem,
                Self.OutputPipeline(
                    smem.pipelines.accum_barriers().ptr,
                    tmem,
                    UInt16(ctx.mma_complete_mask),
                ),
                Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
            )

            var tile_writer = Self.TileWriterType(Pointer(to=out_tma_op))

            var epi_iter = scheduler.work_iterator()

            with epi_ctx:
                for current in epi_iter:
                    with epi_ctx.output_pipeline.consumer() as output_stage:
                        comptime if has_residual:
                            # TileTensor view over the source C SMEM tile
                            # array (num_epi_load_stages buffers).
                            var src_tiles = Self.SrcCTileArray(
                                smem.src_tiles().ptr
                            )

                            # Pass the underlying ProducerConsumerPipeline by
                            # mutable pointer — write_with_residual drives one
                            # `wait_producer / use / consumer_release / step`
                            # cycle per inner epilogue stage, matching the
                            # CUTLASS `sm100_epilogue_tma_warpspecialized`
                            # lockstep pattern (MODELS-1484).
                            tile_writer.write_with_residual[
                                Self.SmemType.num_epi_load_stages
                            ](
                                smem.out_tiles(),
                                output_stage,
                                src_tiles,
                                Pointer(to=epi_load_pipeline.pipeline),
                                Scalar[Self.out_type](beta),
                                (current.m, current.n),
                                (mnk[0], mnk[1]),
                                ctx.elect_one_warp,
                            )
                        else:
                            tile_writer.write(
                                smem.out_tiles(),
                                output_stage,
                                (current.m, current.n),
                                (mnk[0], mnk[1]),
                                ctx.elect_one_warp,
                            )
