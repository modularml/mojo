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
"""Blockwise FP8 1D2D SM100 matmul kernel.

This kernel combines:
- Accumulation pattern from blockwise_fp8/ (register-based per-K scaling via
  BlockwiseFP8Accumulator, standard MMA, A-scales in SMEM, B-scales from GMEM)
- 1D2D work distribution from grouped_block_scaled_1d1d/ (GroupedWorkIterator1D1D,
  offset-based A tensor addressing, bounds-checked output, 3-warp specialization,
  SmemPipelineBundleNoClc)

Architecture:
- TMA warp: Loads A, B, A-scales tiles using grid-constant TMAs
- MMA warp: Standard MMA (partial results to TMEM, init_c=True every K iter)
- Epilogue warps: Per-K TMEM read → scale → register accumulate → final output
  with bounds checking
"""

from std.math import ceildiv
from std.math.uutils import ufloordiv, umod
from std.sys import size_of

from max.gpu import WARP_SIZE, thread_idx
from max.gpu.memory import external_memory, fence_mbarrier_init
from max.gpu.primitives.cluster import (
    block_rank_in_cluster,
    cluster_sync,
    elect_one_sync,
    elect_one_sync_with_mask,
)
from max.gpu.sync import named_barrier, syncwarp
from layout import DefaultEngine, TensorLayout, TensorEngine, TileTensor
from structured_kernels.tile_types import (
    TmaOpType,
    static_row_major,
)

from std.utils.index import IndexList
from std.utils.static_tuple import StaticTuple

from linalg.arch.sm100 import MmaOpSM100_SS

from ..structured_kernels.config import MatmulConfig, OutputPipelineConfig
from structured_kernels.kernel_common import (
    WarpRole1D1D,
    compute_tma_tile_dims,
    compute_accum_barrier_counts,
    compute_input_consumer_count,
    init_core_barriers,
)
from ..structured_kernels.tile_pipeline import (
    InputTilePipeline,
    ProducerTiles,
    InputConsumerStage,
    OutputTilePipeline,
    BlockwiseFP8TilePayload,
)
from ..structured_kernels.tmem import TmemAllocation, TmemDeallocBarrier
from structured_kernels.barriers import WarpGroupBarrier
from ..structured_kernels.warp_context import (
    MmaWarpContext,
    EpilogueWarpContext,
)
from .blockwise_fp8_1d2d_smem import BlockwiseFP8_1D2DSmem
from ..grouped_block_scaled_1d1d.grouped_1d1d_tile_scheduler import (
    GroupedWorkIterator1D1D,
    GroupedWorkContext1D1D,
)

# Blockwise FP8 specific components
from ..blockwise_fp8.blockwise_fp8_accumulator import (
    BlockwiseFP8Accumulator,
    get_accumulator_dims,
    is_lower_fragment_required,
)
from ..blockwise_fp8.blockwise_fp8_output_writer import BlockwiseFP8TileWriter


# =============================================================================
# BlockwiseFP8_1D2DMatmulKernel - Main kernel struct
# =============================================================================


struct BlockwiseFP8_1D2DMatmulKernel[
    # Core types
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    # B-scales and C device layouts (TensorLayout from caller's TileTensor)
    b_scales_layout: TensorLayout,
    c_device_layout: TensorLayout,
    # Configuration
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    # Static dimensions
    static_N: Int,
    static_K: Int,
    # Cluster shape
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1),
    b_scales_engine: TensorEngine = DefaultEngine[element_width=1],
    c_device_engine: TensorEngine = DefaultEngine[element_width=1],
    offsets_engine: TensorEngine = DefaultEngine[element_width=1],
    expert_ids_engine: TensorEngine = DefaultEngine[element_width=1],
    expert_scales_engine: TensorEngine = DefaultEngine[element_width=1],
]:
    """Blockwise FP8 1D2D matmul kernel with register-based accumulation.

    Combines blockwise FP8 scaling (per-K in CUDA cores) with 1D-1D offset-based
    work distribution for grouped GEMM in MoE layers.

    Uses 3-warp specialization (Load, MMA, Epilogue) with grid-constant TMAs.
    Work distribution via GroupedWorkIterator1D1D using offset-based addressing.

    Parameters:
        a_type: Element `DType` of the A input matrix.
        b_type: Element `DType` of the B input matrix.
        c_type: Element `DType` of the output C matrix.
        a_scales_type: Element `DType` of the A-side blockwise scales;
            must match `b_scales_type`.
        b_scales_type: Element `DType` of the B-side blockwise scales;
            must match `a_scales_type`.
        b_scales_layout: Device `TensorLayout` of the B-scales `TileTensor`,
            read from GMEM (not via TMA).
        c_device_layout: Device `TensorLayout` of the output C `TileTensor`
            used for bounds-checked stores.
        transpose_b: Whether B is stored transposed; must be `True`.
        config: Compile-time `MatmulConfig` carrying tile shapes, swizzles,
            pipeline stage counts, and cluster shape.
        static_N: Compile-time-known per-expert N dimension (output width,
            columns of B).
        static_K: Compile-time-known K contraction dimension; sets the
            B-scales row stride as `static_K // 128`.
        cluster_shape: Thread block cluster shape as a
            `StaticTuple[Int32, 3]` (defaults to `(1, 1, 1)`).
        b_scales_engine: Engine of the B-scales `TileTensor`.
        c_device_engine: Engine of the output C `TileTensor`.
        offsets_engine: Engine of the per-expert offsets
            `TileTensor`.
        expert_ids_engine: Engine of the expert-IDs `TileTensor`.
        expert_scales_engine: Engine of the expert-scales
            `TileTensor`.
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

    comptime accum_type = DType.float32
    comptime cta_group = Self.config.cta_group

    comptime CLUSTER_M: Int = Self.config.cluster_shape[0]
    comptime CLUSTER_N: Int = Self.config.cluster_shape[1]
    comptime CLUSTER_SIZE = Self.CLUSTER_M * Self.CLUSTER_N

    # ========== Thread/Warp Organization ==========

    comptime num_output_warps = 4
    comptime WarpRole = WarpRole1D1D[has_sfb=False, num_epi_warps=4]
    comptime NUM_THREADS = Self.WarpRole.TOTAL_THREADS

    # ========== Pipeline Configuration ==========

    comptime num_pipeline_stages = Self.config.num_pipeline_stages
    comptime num_group_pipeline_stages = (
        Self.num_pipeline_stages // Self.config.k_group_size
    )
    comptime num_accum_pipeline_stages = Self.config.num_accum_pipeline_stages
    comptime num_output_stages: Int = Self.config.num_output_stages

    # TMEM configuration — stride matches MMA output width (1D2D variant).
    comptime NUM_TMEM_COLS = 512
    comptime stage_stride_cols = Self.MMA_N

    # Output pipeline config (bundles accum stages, stride, and cta_group)
    comptime opc = OutputPipelineConfig(
        Self.num_accum_pipeline_stages,
        Self.stage_stride_cols,
        Self.cta_group,
    )

    # ========== Barrier Arrival Counts ==========

    comptime _accum_barrier_counts = compute_accum_barrier_counts[
        Self.WarpRole.NUM_EPILOGUE_THREADS, Self.cta_group
    ]()
    comptime accum_pipeline_producer_arv_count = Self._accum_barrier_counts[0]
    comptime accum_pipeline_consumer_arv_count = Self._accum_barrier_counts[1]

    # ========== Shared Memory Type ==========

    comptime SmemType = BlockwiseFP8_1D2DSmem[
        Self.a_type,
        Self.b_type,
        Self.c_type,
        Self.a_scales_type,
        Self.transpose_b,
        config=Self.config,
    ]

    # ========== MMA Operation Type ==========
    # Standard MMA (not block-scaled) - scaling applied in CUDA cores
    comptime MmaOp = MmaOpSM100_SS[
        Self.c_type,
        Self.a_type,
        Self.b_type,
        Self.config.block_tile_shape,
        Self.config.mma_shape,
        accum_type=Self.accum_type,
        cta_group=Self.cta_group,
        cluster_shape=Self.config.cluster_shape,
        a_swizzle=Self.config.a_swizzle,
        b_swizzle=Self.config.b_swizzle,
        transpose_b=Self.transpose_b,
    ]

    # ========== Tile Pipeline Types ==========
    comptime TilePayload = BlockwiseFP8TilePayload[
        Self.a_type,
        Self.b_type,
        Self.a_scales_type,
        IndexList[2](
            Self.SmemType.Core.BM, Self.SmemType.Core.BK
        ),  # A tile shape
        IndexList[2](
            Self.SmemType.Core.BN, Self.SmemType.Core.BK
        ),  # B tile shape
        IndexList[2](1, Self.SmemType.Core.BM),  # A-scales shape
        Self.SmemType.Core.num_pipeline_stages,
    ]

    comptime InputTilePipelineType = InputTilePipeline[
        Self.TilePayload,
        Self.SmemType.Core.num_group_pipeline_stages,
        Self.config.k_group_size,
    ]

    # ========== TMEM and Output Pipeline Types ==========
    comptime Tmem = TmemAllocation[Self.opc.cta_group]
    comptime TmemDealloc = TmemDeallocBarrier[Self.opc.cta_group]

    comptime OutputPipeline = OutputTilePipeline[Self.opc]

    # ========== Warp Context Types ==========
    comptime MmaEpilogueSync = WarpGroupBarrier[
        Self.WarpRole.NUM_MMA_THREADS + Self.WarpRole.NUM_EPILOGUE_THREADS, 1
    ]

    comptime MmaCtx = MmaWarpContext[
        Self.opc,
        Self.WarpRole.NUM_MMA_THREADS,
        Self.WarpRole.NUM_EPILOGUE_THREADS,
    ]

    comptime EpilogueCtx = EpilogueWarpContext[
        Self.opc,
        Self.WarpRole.NUM_MMA_THREADS,
        Self.WarpRole.NUM_EPILOGUE_THREADS,
    ]

    # ========== TMA Load Size Constants ==========
    comptime a_expected_bytes = Self.BM * Self.BK * size_of[Self.a_type]()
    comptime b_expected_bytes = Self.BN * Self.BK * size_of[Self.b_type]()
    comptime a_scales_expected_bytes = Self.BM * size_of[Self.a_scales_type]()
    comptime input_expected_bytes = Self.cta_group * (
        Self.a_expected_bytes
        + Self.b_expected_bytes
        + Self.a_scales_expected_bytes
    )

    # ========== TMA Layouts (computed from config, new Layout types) ==========

    comptime _tma_tile_dims = compute_tma_tile_dims[
        Self.BM,
        Self.BN,
        Self.MMA_M,
        Self.OutputM,
        Self.CLUSTER_M,
        Self.CLUSTER_N,
        Self.cta_group,
    ]()
    comptime a_tile_dim0 = Self._tma_tile_dims[0]
    comptime b_tile_dim0 = Self._tma_tile_dims[1]
    comptime a_swizzle_elems = Self.config.a_swizzle.bytes() // size_of[
        Self.a_type
    ]()
    comptime b_swizzle_elems = Self.config.b_swizzle.bytes() // size_of[
        Self.b_type
    ]()

    comptime ATileLayout = static_row_major[Self.a_tile_dim0, Self.BK]
    comptime ADescLayout = static_row_major[
        Self.a_tile_dim0, Self.a_swizzle_elems
    ]
    comptime BTileLayout = static_row_major[Self.b_tile_dim0, Self.BK]
    comptime BDescLayout = static_row_major[
        Self.b_tile_dim0, Self.b_swizzle_elems
    ]
    comptime AScalesLayout = static_row_major[1, Self.BM]

    # TMA operation types (derived from new Layout types)
    comptime ATmaOp = TmaOpType[Self.a_type, Self.ATileLayout, Self.ADescLayout]
    comptime BTmaOp = TmaOpType[Self.b_type, Self.BTileLayout, Self.BDescLayout]
    comptime AScalesTmaOp = TmaOpType[
        Self.a_scales_type, Self.AScalesLayout, Self.AScalesLayout
    ]

    # TMA load size constants (from desc layout dimensions)
    comptime a_tma_load_size = Self.a_tile_dim0 * Self.a_swizzle_elems
    comptime b_tma_load_size = Self.b_tile_dim0 * Self.b_swizzle_elems
    comptime a_tma_rows = Self.a_tile_dim0
    comptime b_tma_rows = Self.b_tile_dim0

    # ========== Accumulator Type ==========
    comptime is_lower_required = is_lower_fragment_required[
        Self.cta_group, Self.config.block_tile_shape
    ]()

    comptime accum_dims = get_accumulator_dims[
        c_smem_dim1=Self.OutputN,
        block_tile_shape=Self.config.block_tile_shape,
        mma_shape=Self.config.mma_shape,
        cta_group=Self.cta_group,
    ]()

    comptime Accumulator = BlockwiseFP8Accumulator[
        Self.accum_type,
        Self.accum_dims[0],
        Self.accum_dims[1],
        Self.is_lower_required,
        Self.config.block_tile_shape,
        Self.config.mma_shape,
        Self.CLUSTER_SIZE,
    ]

    # ========== Output Writer Type ==========
    comptime TileWriterType = BlockwiseFP8TileWriter[
        Self.c_type,
        Self.OutputM,
        Self.OutputN,
        Self.accum_type,
        Self.accum_dims[0],
        Self.accum_dims[1],
        block_tile_shape=Self.config.block_tile_shape,
        mma_shape=Self.config.mma_shape,
        is_lower_frag_required=Self.is_lower_required,
        cta_group=Self.cta_group,
        num_output_stages=Self.num_output_stages,
        num_output_warps=Self.num_output_warps,
        c_swizzle=Self.config.c_swizzle,
    ]

    # ========== Work Iterator Type ==========
    comptime WorkIterator = GroupedWorkIterator1D1D[
        static_N=Self.static_N,
        tile_shape=Self.config.block_tile_shape,
        cluster=Self.config.cluster_shape,
        cta_group=Self.cta_group,
        OffsetsEngine=Self.offsets_engine,
        ExpertIdsEngine=Self.expert_ids_engine,
        ExpertScalesEngine=Self.expert_scales_engine,
    ]

    # ========== Validation ==========

    @staticmethod
    def validate_config():
        """Compile-time validation of kernel configuration."""
        comptime assert Self.transpose_b, "Only support transposed B"
        comptime assert (
            Self.a_scales_type == Self.b_scales_type
        ), "a_scales_type and b_scales_type must match"
        comptime assert Self.cta_group in (
            1,
            2,
        ), "Only support cta_group == 1 or 2"
        comptime assert Self.BK in (64, 128), "Only support BK in (64, 128)"

    # ========== Computed Layouts (single source of truth) ==========

    # ========== Kernel Parameter TileTensor Types ==========

    comptime BScalesTile = TileTensor[
        Self.b_scales_type,
        Self.b_scales_layout,
        MutAnyOrigin,
        Engine=Self.b_scales_engine,
    ]

    # Same tile re-based on one expert's B-scale rows. Offsetting the storage
    # handle can change the engine, so name the result type through
    # `OffsetViewType` rather than assuming it is still `BScalesTile`.
    comptime BScalesExpertTile = Self.BScalesTile.OffsetViewType[
        TypeList.of[Scalar[Self.BScalesTile.linear_idx_type]]()
    ]

    comptime CDeviceTile = TileTensor[
        Self.c_type,
        Self.c_device_layout,
        MutAnyOrigin,
        Engine=Self.c_device_engine,
    ]

    # ========== Static Helper Methods ==========

    @staticmethod
    @always_inline
    def init_barriers(
        elect_one_warp: Bool,
        elect_one_thread: Bool,
        a_tma_op: Self.ATmaOp,
        b_tma_op: Self.BTmaOp,
        a_scales_tma_op: Self.AScalesTmaOp,
        input_barriers: Self.SmemType.Pipelines.InputBarriers,
        accum_barriers: Self.SmemType.Pipelines.AccumBarriers,
        tmem_dealloc: Self.SmemType.Pipelines.TmemDealloc,
    ):
        """Initialize barriers and prefetch TMA descriptors.

        Args:
            elect_one_warp: True if this thread is in the elected warp
                (warp 0) that prefetches TMA descriptors and inits barriers.
            elect_one_thread: True for the single elected thread within the
                elected warp; gates the one-time setup work.
            a_tma_op: TMA operation descriptor for the A matrix loads.
            b_tma_op: TMA operation descriptor for the B matrix loads.
            a_scales_tma_op: TMA operation descriptor for the A-scales loads.
            input_barriers: Input pipeline `mbarrier`s synchronizing
                producer and consumer access to A, B, and A-scales tiles.
            accum_barriers: Accumulator pipeline `mbarrier`s synchronizing
                MMA producers and epilogue consumers of TMEM stages.
            tmem_dealloc: TMEM deallocation barrier for releasing accumulator
                stages back to the allocator.
        """
        if elect_one_warp and elect_one_thread:
            a_tma_op.prefetch_descriptor()
            b_tma_op.prefetch_descriptor()
            a_scales_tma_op.prefetch_descriptor()

            # Epilogue warps also consume A-scales from input pipeline.
            init_core_barriers[
                Self.num_group_pipeline_stages,
                Self.num_accum_pipeline_stages,
            ](
                input_barriers.ptr,
                Int32(
                    compute_input_consumer_count[
                        Self.CLUSTER_M,
                        Self.CLUSTER_N,
                        Self.cta_group,
                        CLUSTER_SIZE=Self.CLUSTER_SIZE,
                        epilogue_threads=Self.WarpRole.NUM_EPILOGUE_THREADS,
                    ]()
                ),
                accum_barriers.ptr,
                Int32(Self.accum_pipeline_producer_arv_count),
                Int32(Self.accum_pipeline_consumer_arv_count),
                tmem_dealloc.ptr,
                Int32(Self.WarpRole.NUM_EPILOGUE_THREADS * Self.cta_group),
            )

        fence_mbarrier_init()
        cluster_sync()

    # ========== Kernel Entry Point ==========

    @staticmethod
    @always_inline
    @__llvm_metadata(`nvvm.cluster_dim`=Self.cluster_shape)
    @__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(a_scales_tma_op, `nvvm.grid_constant`)
    @__name(StaticString(Self.config.get_kernel_name()))
    def run(
        # Grid-constant TMA descriptors
        a_tma_op: Self.ATmaOp,
        b_tma_op: Self.BTmaOp,
        a_scales_tma_op: Self.AScalesTmaOp,
        # B-scales from GMEM (not TMA)
        b_scales: Self.BScalesTile,
        # Offset tensors for 1D-1D addressing
        a_offsets: Self.WorkIterator.OffsetsTile,
        expert_ids: Self.WorkIterator.ExpertIdsTile,
        expert_scales: Self.WorkIterator.ExpertScalesTile,
        # C tensor for bounds-checked stores
        c_device: Self.CDeviceTile,
        # Number of active experts
        num_active_experts: Int32,
        # K dimension for iteration
        K: UInt32,
    ):
        """Grouped 1D-1D blockwise FP8 GEMM kernel entry point.

        Uses grid-constant TMAs with offset-based addressing for 1D-1D layout.
        Accumulates in registers with per-K scaling in CUDA cores.

        Args:
            a_tma_op: Grid-constant TMA descriptor for the A matrix loads.
            b_tma_op: Grid-constant TMA descriptor for the B matrix loads.
            a_scales_tma_op: Grid-constant TMA descriptor for the A-side
                blockwise scales loads.
            b_scales: B-side blockwise scales `TileTensor` read from GMEM,
                with shape `(num_experts * N // 128, K // 128)`.
            a_offsets: Offset tensor for 1D-1D A tensor addressing, providing
                per-expert token offsets.
            expert_ids: Expert ID tensor mapping each work tile to its
                expert.
            expert_scales: Per-expert scale factors applied in the epilogue
                output write.
            c_device: Output C `TileTensor` used for bounds-checked stores.
            num_active_experts: Number of active experts in the grouped
                GEMM.
            K: K contraction dimension; the inner reduction axis length used
                to compute `ceildiv(K, BK)` K iterations.
        """
        var _num_active_experts = Int(num_active_experts)
        Self.validate_config()

        # ===== Shared Memory Setup =====
        ref smem = external_memory[
            UInt8,
            address_space=.SHARED,
            alignment=128,
        ]().bitcast[Self.SmemType]()[]

        # Get typed tile arrays from SMEM
        var a_tiles = smem.a_tiles()
        var b_tiles = smem.b_tiles()
        var c_tiles = smem.c_tiles()
        var a_scales_tiles = smem.a_scales_tiles()

        # Get typed barrier arrays
        var input_barriers = smem.pipelines.input_barriers()
        var accum_barriers = smem.pipelines.accum_barriers()
        var tmem_addr_storage = smem.pipelines.tmem_addr().ptr

        # Create input pipeline with tile payload
        var tile_payload = Self.TilePayload(a_tiles, b_tiles, a_scales_tiles)
        var input_pipeline = Self.InputTilePipelineType(
            input_barriers, tile_payload
        )

        # ===== Warp/Thread Election =====
        var elect_one_warp = ufloordiv(thread_idx.x, WARP_SIZE) == 0
        var elect_one_thread = elect_one_sync_with_mask()
        var elect_one_cta = (
            block_rank_in_cluster() % 2 == 0 if Self.cta_group == 2 else True
        )

        # Peer CTA coordinates for multicast
        var peer_rank_n = Int(block_rank_in_cluster() % UInt32(Self.CLUSTER_N))
        var peer_rank_m = Int(
            block_rank_in_cluster()
            // UInt32(Self.CLUSTER_N)
            % UInt32(Self.CLUSTER_M)
        )
        var peer_m_rank = umod(peer_rank_m, Self.cta_group)
        var peer_cta_coord = (peer_rank_n, peer_rank_m, peer_m_rank)

        # Multicast masks
        var a_multicast_mask = UInt16((1 << Self.CLUSTER_M) - 1)
        var b_multicast_mask = UInt16((1 << Self.CLUSTER_N) - 1)
        var mma_complete_mask = UInt16((1 << Self.cta_group) - 1)

        # K iteration count
        var num_k_iters = ceildiv(Int(K), Self.BK)

        # ===== Barrier Initialization =====
        Self.init_barriers(
            elect_one_warp,
            elect_one_thread,
            a_tma_op,
            b_tma_op,
            a_scales_tma_op,
            input_barriers,
            accum_barriers,
            smem.pipelines.tmem_dealloc(),
        )

        var mma_op = Self.MmaOp()

        # ===== TMA LOAD WARP =====
        if Self.WarpRole.is_load():
            var load_iter = Self.WorkIterator(
                _num_active_experts, a_offsets, expert_ids, expert_scales
            )

            with input_pipeline.producer() as producer:
                for ctx in load_iter:
                    var next_ready = True
                    if num_k_iters > 0:
                        next_ready = producer.try_acquire()

                    for k_tile in range(num_k_iters):
                        with producer.acquire_if_needed(next_ready) as tiles:
                            Self.load_input_tiles(
                                a_tma_op,
                                b_tma_op,
                                a_scales_tma_op,
                                tiles,
                                peer_cta_coord,
                                ctx,
                                k_tile,
                                elect_one_cta,
                            )
                        next_ready = True
                        if k_tile + 1 < num_k_iters:
                            next_ready = producer.try_acquire()

                    syncwarp()

                producer.drain()

        # ===== MMA WARP =====
        # Blockwise FP8: per-K synchronization (MMA writes fresh partial each K,
        # epilogue reads TMEM per-K to accumulate in registers).
        if Self.WarpRole.is_mma():
            var mma_iter = Self.WorkIterator(
                _num_active_experts, a_offsets, expert_ids, expert_scales
            )

            var tmem = Self.Tmem.allocate(smem.pipelines.tmem_addr())
            var mma_ctx = Self.MmaCtx(
                tmem,
                Self.OutputPipeline(
                    accum_barriers.ptr, tmem, mma_complete_mask
                ),
                Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
            )

            with mma_ctx:
                for _ in mma_iter:
                    if elect_one_cta:
                        for _ in range(num_k_iters):
                            # Per-K: acquire stage, MMA, release
                            with mma_ctx.per_k_stage() as mma_stage:
                                var tmem_offset = UInt32(
                                    mma_stage.tmem.offset()
                                )

                                var input_tiles = (
                                    input_pipeline.acquire_consumer()
                                )
                                Self.mma(input_tiles, mma_op, tmem_offset)
                                input_tiles^.release()

        # ===== EPILOGUE WARPS =====
        if Self.WarpRole.is_epilogue():
            var epi_iter = Self.WorkIterator(
                _num_active_experts, a_offsets, expert_ids, expert_scales
            )
            Self.MmaEpilogueSync.wait()

            var tmem = Self.Tmem.from_shared(smem.pipelines.tmem_addr())
            var epi_ctx = Self.EpilogueCtx(
                tmem,
                Self.OutputPipeline(
                    accum_barriers.ptr, tmem, mma_complete_mask
                ),
                Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
            )

            with epi_ctx:
                for ctx in epi_iter:
                    # Blockwise FP8: accumulate across K iterations in registers
                    var accum = Self.Accumulator()

                    # Offset b_scales to current expert's section.
                    # b_scales is (num_experts * N//128, K//128), we need
                    # to index with local tile index within the expert.
                    comptime b_scales_k = Self.static_K // 128
                    # b_scales shape is (num_experts * N//128, K//128).
                    # expert_id * (N//128) gives the row offset.
                    comptime n_scale_blocks = Self.static_N // Self.BK
                    var expert_b_scale_offset = (
                        Int(ctx.expert_id()) * n_scale_blocks
                    )
                    var b_scales_expert = Self.BScalesExpertTile(
                        b_scales._offset_storage(
                            Scalar[Self.BScalesTile.linear_idx_type](
                                expert_b_scale_offset * b_scales_k
                            )
                        ),
                        b_scales.layout,
                    )

                    # Convert absolute N to tile index for b_scales lookup
                    var n_tile = ufloordiv(Int(ctx.n()), Self.MMA_N)

                    for k_iter in range(num_k_iters):
                        with epi_ctx.per_k_stage(input_pipeline) as epi_stage:
                            accum.promote(
                                b_scales_expert,
                                a_scales_tiles,
                                epi_stage,
                                work_tile_coord=(
                                    Int(ctx.m()),
                                    n_tile,
                                ),
                                k_iter=k_iter,
                                problem_shape=StaticTuple[Int32, 3](
                                    Int32(0),
                                    Int32(Self.static_N),
                                    Int32(K),
                                ),
                            )

                    named_barrier[Int32(Self.num_output_warps * WARP_SIZE)]()

                    # Write with bounds checking and expert scale
                    Self.TileWriterType.write_absolute_with_bounds_check[
                        Self.c_device_layout,
                        Self.CLUSTER_SIZE,
                    ](
                        accum,
                        c_tiles,
                        ctx.m(),
                        ctx.n(),
                        ctx.m_end,
                        ctx.expert_scale,
                        c_device,
                    )

    # ========== Load Input Tiles ==========

    @staticmethod
    @always_inline
    def load_input_tiles[
        tiles_origin: MutOrigin,
        //,
    ](
        a_tma_op: Self.ATmaOp,
        b_tma_op: Self.BTmaOp,
        a_scales_tma_op: Self.AScalesTmaOp,
        tiles: ProducerTiles[
            tiles_origin,
            Self.TilePayload,
            Self.SmemType.Core.num_group_pipeline_stages,
            Self.config.k_group_size,
        ],
        peer_cta_coord: Tuple[Int, Int, Int],
        work_ctx: GroupedWorkContext1D1D,
        iter_idx: Int,
        elect_one_cta: Bool,
    ):
        """Load A, B, and A-scales tiles using TMA.

        Parameters:
            tiles_origin: Memory origin of the producer tile payload
                (inferred).

        Args:
            a_tma_op: Grid-constant TMA descriptor for the A matrix loads.
            b_tma_op: Grid-constant TMA descriptor for the B matrix loads.
            a_scales_tma_op: Grid-constant TMA descriptor for the A-side
                blockwise scales loads.
            tiles: Producer pipeline stage holding the A, B, and A-scales
                SMEM tiles and associated barrier.
            peer_cta_coord: Peer CTA coordinates `(rank_n, rank_m, m_rank)`
                used for multicast load addressing within the cluster.
            work_ctx: Work context providing the current tile's `m`, `n`,
                and `expert_id` coordinates.
            iter_idx: K-iteration index; the K coordinate is
                `iter_idx * BK`.
            elect_one_cta: True if this CTA is the elected CTA in a two-CTA
                cluster; gates `expect_bytes` setup.
        """
        var peer_rank_n = peer_cta_coord[0]
        var peer_rank_m = peer_cta_coord[1]
        var peer_m_rank = peer_cta_coord[2]

        # M coordinate in contiguous token space
        var m_coord = work_ctx.m()
        var n_coord = work_ctx.n()
        var expert_id = work_ctx.expert_id()

        # Int required at TMA coord boundary
        var a_gmem_m_coord = peer_m_rank * Self.a_tma_rows + Int(m_coord)
        var b_gmem_n_coord = (
            peer_rank_m * Self.b_tma_rows
            + peer_rank_n * Self.BN
            + Int(n_coord)
            + Int(expert_id) * Self.static_N
        )

        if elect_one_sync():
            if elect_one_cta:
                tiles.expect_bytes(Self.input_expected_bytes)

            var barrier = tiles.barrier()
            var stage = tiles.stage()

            # Get tiles as TileTensor (native SMEM storage)
            var a_tile, b_tile, a_scales_tile = tiles.payload().get_tile[
                Self.config.k_group_size
            ](stage, 0)

            # Peer CTA slicing using TileTensor pattern (ptr + layout)
            var a_peer_tile = type_of(a_tile)(
                a_tile._storage + peer_m_rank * Self.a_tma_load_size,
                a_tile.layout,
            )
            var b_peer_tile = type_of(b_tile)(
                b_tile._storage + peer_rank_m * Self.b_tma_load_size,
                b_tile.layout,
            )

            var k_coord = iter_idx * Self.BK

            # Load A and B using TileTensor overload
            a_tma_op.async_multicast_load[Self.cta_group](
                a_peer_tile,
                barrier[0],
                (k_coord, a_gmem_m_coord),
                UInt16((1 << Self.CLUSTER_M) - 1),
            )
            b_tma_op.async_multicast_load[Self.cta_group](
                b_peer_tile,
                barrier[0],
                (k_coord, b_gmem_n_coord),
                UInt16((1 << Self.CLUSTER_N) - 1),
            )

            # Load A-scales via TMA (TileTensor directly)
            a_scales_tma_op.async_copy[Self.cta_group](
                a_scales_tile,
                barrier[0],
                (Int(m_coord), iter_idx),
            )

    # ========== MMA Operation ==========

    @staticmethod
    @always_inline
    def mma[
        tiles_origin: MutOrigin,
        //,
    ](
        tiles: InputConsumerStage[
            tiles_origin,
            Self.TilePayload,
            Self.SmemType.Core.num_group_pipeline_stages,
            Self.config.k_group_size,
        ],
        mma_op: Self.MmaOp,
        tmem_addr: UInt32,
    ):
        """Execute standard MMA operations (partial results to TMEM).

        For blockwise FP8, each K iteration writes a fresh partial to TMEM.
        The epilogue accumulates across K in registers, not TMEM.
        Therefore init_c is always True.

        Parameters:
            tiles_origin: Memory origin of the input consumer tile payload
                (inferred).

        Args:
            tiles: Input consumer stage holding the A and B SMEM tiles and
                associated barrier.
            mma_op: MMA operation object performing the tensor core
                multiply-accumulate into TMEM.
            tmem_addr: TMEM address offset where MMA writes partial results.
        """
        if elect_one_sync():
            # Loop through k_group_size tiles (typically 1)
            for jj in range(Self.config.k_group_size):
                # Get tiles as TileTensor (native SMEM storage)
                var a_tile, b_tile, _ = tiles.payload().get_tile[
                    Self.config.k_group_size
                ](tiles.stage(), jj)

                # Blockwise FP8: always init_c=True since epilogue accumulates
                # in registers, not TMEM.
                mma_op.mma(
                    a_tile,
                    b_tile,
                    tmem_addr,
                    init_c=True,
                )

            mma_op.commit(tiles.mbar())
