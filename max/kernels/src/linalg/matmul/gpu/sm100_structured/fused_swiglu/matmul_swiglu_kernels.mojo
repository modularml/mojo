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

"""SM100 fused GEMM+SwiGLU kernel (monolithic warp-specialized).

Follows the same monolithic structure as the BF16/FP8 dense kernel: free
functions for each phase (load, MMA, epilogue) and a single top-level kernel
function that dispatches to them via ``WarpRole`` checks.

Two epilogue modes controlled by ``register_swiglu``:
  - False (default): TMEM → FP32 SwiGLU → half SMEM → TMA store to GMEM.
  - True:            TMEM → FP32 SwiGLU → BF16 stores directly to GMEM.

Weight layout: natural row-interleaved (gate, up) pairs in both swap modes.
- non-swap (AB_swapped=False): (gate, up) at adjacent N positions.
- swap (AB_swapped=True):      (gate, up) at adjacent M positions;
  warp.shuffle_xor(_, 4) brings the up value into the gate-owner lane.
"""

from std.math import ceildiv, exp, recip
from std.sys import size_of
from std.math.uutils import umod, ufloordiv

from max.gpu import WARP_SIZE, warp_id as get_warp_id
from max.gpu.sync import barrier
from max.gpu import block_id_in_cluster, lane_id
from max.gpu.primitives.cluster import (
    block_rank_in_cluster,
    elect_one_sync,
    elect_one_sync_with_mask,
    cluster_wait,
    cluster_arrive_relaxed,
)
from max.gpu.memory import (
    async_copy,
    external_memory,
    fence_mbarrier_init,
    fence_async_view_proxy,
    cp_async_bulk_tensor_global_shared_cta,
)
from max.gpu.sync import (
    async_copy_arrive,
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
    named_barrier,
    named_barrier_arrive,
    syncwarp,
)
from max.gpu.compute.arch.mma_nvidia_sm100 import *
from max.gpu.compute.arch.tcgen05 import *
from max.gpu.primitives.grid_controls import (
    launch_dependent_grids,
    PDLLevel,
    wait_on_dependent_grids,
)
import max.gpu.primitives.warp as warp
from max.gpu.host.nvidia.tma import TensorMapSwizzle

from layout import Layout, RowMajorLayout, TileTensor, row_major
from layout.tma_async import SharedMemBarrier, TMATensorTile, _idx_product
from structured_kernels.tile_types import (
    SMemTileArray2D,
    swizzle_mode_to_bytes,
    static_row_major,
    tma_desc_layout_3d,
)
from layout.tile_layout import _IntToComptimeInt

from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple

from linalg.arch.sm100 import MmaOpSM100_SS
from linalg.structuring import SMemArray, SMemPtr

from ..structured_kernels.config import OutputPipelineConfig
from .config import FusedSwiGLUMatmulConfig
from ..structured_kernels.tile_scheduler import TileScheduler
from linalg.matmul.gpu.tile_scheduler import RasterOrder
from ..structured_kernels.epilogue_components import (
    AccumBarrier,
    EpilogueApplier,
    EpilogueConfig,
    FragmentCoords,
)
from ..structured_kernels.tmem import TmemArrayType
from structured_kernels.pipeline import ProducerConsumerPipeline

from structured_kernels.kernel_common import (
    WarpRole,
    compute_input_consumer_count,
    init_clc_barriers,
)


# ===----------------------------------------------------------------------=== #
# _SwiGLUSmem - per-CTA shared memory layout for the fused kernel
# (Same as B200MatmulSmem but without the C output tile storage,
#  since SwiGLU reduces directly from TMEM.)
# ===----------------------------------------------------------------------=== #


struct _SwiGLUSmem[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool,
    *,
    config: FusedSwiGLUMatmulConfig[a_type, b_type, c_type, transpose_b],
]:
    comptime BM = Self.config.block_tile_shape[0]
    comptime BN = Self.config.block_tile_shape[1]
    comptime BK = Self.config.block_tile_shape[2]
    comptime MMA_N = Self.config.mma_shape[1]
    comptime bias_dim = Self.BM if Self.config.AB_swapped else Self.MMA_N
    comptime bias_smem_elems = 2 * Self.bias_dim if Self.config.use_bias else 0
    comptime num_group_pipeline_stages = (
        Self.config.num_pipeline_stages // Self.config.k_group_size
    )

    # A/B pipelines
    var a_smem: Array[
        Scalar[Self.a_type], Self.BM * Self.BK * Self.config.num_pipeline_stages
    ]
    var b_smem: Array[
        Scalar[Self.b_type], Self.BN * Self.BK * Self.config.num_pipeline_stages
    ]

    var tma_mma_mbars: Array[
        SharedMemBarrier, Self.num_group_pipeline_stages * 2
    ]
    var accum_mbars: Array[
        SharedMemBarrier, Self.config.num_accum_pipeline_stages * 2
    ]

    # CLC
    var clc_mbars_full: Array[
        SharedMemBarrier, Self.config.num_clc_pipeline_stages
    ]
    var clc_mbars_empty: Array[
        SharedMemBarrier, Self.config.num_clc_pipeline_stages
    ]
    var clc_throttle_mbars: Array[
        SharedMemBarrier, Self.config.num_clc_pipeline_stages * 2
    ]
    var clc_response: Array[UInt128, Self.config.num_clc_pipeline_stages]

    # TMEM dealloc barrier and address storage
    var tmem_dealloc_mbar: Array[SharedMemBarrier, 1]
    var tmem_addr: Array[UInt32, 1]
    # Epilogue load producer/consumer pipeline barriers (4 for 2-stage pipeline)
    var epilogue_load_mbars: Array[
        SharedMemBarrier, 4 if Self.config.use_bias else 0
    ]
    # 1D bias SMEM tile (2 * bias_dim elements for 2-stage double buffering)
    var bias_smem: Array[Scalar[Self.c_type], Self.bias_smem_elems]


# ===----------------------------------------------------------------------=== #
# load_AB - TMA producer phase
# ===----------------------------------------------------------------------=== #


@always_inline
def load_AB[
    a_type: DType,
    b_type: DType,
    a_rank: Int,
    a_tile_shape: IndexList[a_rank],
    a_desc_shape: IndexList[a_rank],
    b_rank: Int,
    b_tile_shape: IndexList[b_rank],
    b_desc_shape: IndexList[b_rank],
    a_dim0: Int,
    a_dim1: Int,
    a_num_tiles: Int,
    a_swizzle_bytes: Int,
    b_dim0: Int,
    b_dim1: Int,
    b_num_tiles: Int,
    b_swizzle_bytes: Int,
    num_pipeline_stages: Int,
    /,
    *,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int = 1,
    k_group_size: Int = 1,
](
    a_tma_op: TMATensorTile[a_type, a_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[b_type, b_rank, b_tile_shape, b_desc_shape],
    a_smem_tiles: SMemTileArray2D[
        a_type, a_dim0, a_dim1, a_num_tiles, a_swizzle_bytes
    ],
    b_smem_tiles: SMemTileArray2D[
        b_type, b_dim0, b_dim1, b_num_tiles, b_swizzle_bytes
    ],
    load_mma_pipeline: ProducerConsumerPipeline[num_pipeline_stages],
    peer_cta_coord: Tuple[Int, Int, Int],
    work_tile_coord: Tuple[Int, Int, Int],
    a_multicast_mask: UInt16,
    b_multicast_mask: UInt16,
    iter_idx: UInt32,
    elect_one_cta: Bool,
):
    """TMA producer phase that loads A and B tiles from GMEM into SMEM via multicast.

    Parameters:
        a_type: DType of the A operand elements.
        b_type: DType of the B operand elements.
        a_rank: Number of dimensions in the A TMA tensor map.
        a_tile_shape: Shape of each A tile loaded by TMA, in elements.
        a_desc_shape: Shape of the A TMA descriptor, in elements.
        b_rank: Number of dimensions in the B TMA tensor map.
        b_tile_shape: Shape of each B tile loaded by TMA, in elements.
        b_desc_shape: Shape of the B TMA descriptor, in elements.
        a_dim0: Row extent of each A SMEM tile, in elements.
        a_dim1: Column extent of each A SMEM tile, in elements.
        a_num_tiles: Number of A SMEM tiles in the pipeline buffer.
        a_swizzle_bytes: SMEM swizzle granularity for A tiles, in bytes.
        b_dim0: Row extent of each B SMEM tile, in elements.
        b_dim1: Column extent of each B SMEM tile, in elements.
        b_num_tiles: Number of B SMEM tiles in the pipeline buffer.
        b_swizzle_bytes: SMEM swizzle granularity for B tiles, in bytes.
        num_pipeline_stages: Number of stages in the load-to-MMA pipeline.
        block_tile_shape: Block tile shape as (BM, BN, BK).
        mma_shape: MMA instruction shape as (MMA_M, MMA_N, MMA_K).
        cta_group: Number of cooperating CTAs per multicast load
            (defaults to 1).
        k_group_size: Number of K tiles loaded per pipeline stage
            (defaults to 1).

    Args:
        a_tma_op: TMA tensor map descriptor for the A operand.
        b_tma_op: TMA tensor map descriptor for the B operand.
        a_smem_tiles: Pipeline buffer of A SMEM tiles.
        b_smem_tiles: Pipeline buffer of B SMEM tiles.
        load_mma_pipeline: Producer/consumer pipeline between TMA load and
            MMA stages.
        peer_cta_coord: Coordinate of the peer CTA within the cluster,
            used to offset GMEM read and SMEM write addresses.
        work_tile_coord: (M, N, batch) tile coordinate assigned to this
            CTA by the tile scheduler.
        a_multicast_mask: Multicast mask selecting CTAs that receive the
            A tile.
        b_multicast_mask: Multicast mask selecting CTAs that receive the
            B tile.
        iter_idx: Current K iteration index for this pipeline stage.
        elect_one_cta: Whether this CTA sets up the TMA barrier byte
            expectation.
    """
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]

    comptime a_expected_bytes = a_dim0 * a_dim1 * size_of[a_type]()
    comptime b_expected_bytes = b_dim0 * b_dim1 * size_of[b_type]()
    comptime expected_bytes = (
        cta_group * (a_expected_bytes + b_expected_bytes)
    ) * k_group_size

    comptime a_tma_load_size = _idx_product[a_rank, a_desc_shape]()
    comptime b_tma_load_size = _idx_product[b_rank, b_desc_shape]()
    comptime a_tma_rows = a_desc_shape[1]
    comptime b_tma_rows = b_desc_shape[1]

    var stage = load_mma_pipeline.producer_stage()
    var tma_mbar = load_mma_pipeline.producer_mbar(stage)
    var a_gmem_slice_coord = (
        peer_cta_coord[2] * a_tma_rows + work_tile_coord[0] * BM
    )
    var b_gmem_slice_coord = (
        peer_cta_coord[1] * b_tma_rows
        + peer_cta_coord[0] * BN
        + work_tile_coord[1] * MMA_N
    )
    var batch_coord = work_tile_coord[2]

    load_mma_pipeline.wait_consumer()

    if elect_one_sync():
        if elect_one_cta:
            tma_mbar[0].expect_bytes(Int32(expected_bytes))

        for jj in range(k_group_size):
            var j = UInt32(jj)
            var offset = stage * UInt32(k_group_size) + j
            var a_smem_tile = a_smem_tiles[offset]
            var b_smem_tile = b_smem_tiles[offset]

            var a_smem_slice = type_of(a_smem_tile)(
                a_smem_tile._storage + peer_cta_coord[2] * a_tma_load_size,
                a_smem_tile.layout,
            )
            var b_smem_slice = type_of(b_smem_tile)(
                b_smem_tile._storage + peer_cta_coord[1] * b_tma_load_size,
                b_smem_tile.layout,
            )

            a_tma_op.async_multicast_load_3d[cta_group](
                a_smem_slice,
                tma_mbar[0],
                (
                    Int(iter_idx + j) * BK,
                    a_gmem_slice_coord,
                    batch_coord,
                ),
                a_multicast_mask,
            )

            b_tma_op.async_multicast_load_3d[cta_group](
                b_smem_slice,
                tma_mbar[0],
                (
                    Int(iter_idx + j) * BK,
                    b_gmem_slice_coord,
                    batch_coord,
                ),
                b_multicast_mask,
            )


# ===----------------------------------------------------------------------=== #
# consumer_main_loop - MMA consumer phase
# ===----------------------------------------------------------------------=== #


@always_inline
def consumer_main_loop[
    accum_type: DType,
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_dim0: Int,
    a_dim1: Int,
    a_num_tiles: Int,
    a_swizzle_bytes: Int,
    b_dim0: Int,
    b_dim1: Int,
    b_num_tiles: Int,
    b_swizzle_bytes: Int,
    a_swizzle: TensorMapSwizzle,
    b_swizzle: TensorMapSwizzle,
    transpose_b: Bool,
    pipeline_stages: Int,
    /,
    *,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int = 1,
    cluster_shape: IndexList[3] = Index(1, 1, 1),
    k_group_size: Int = 1,
](
    tmem_addr: UInt32,
    a_smem_tiles: SMemTileArray2D[
        a_type, a_dim0, a_dim1, a_num_tiles, a_swizzle_bytes
    ],
    b_smem_tiles: SMemTileArray2D[
        b_type, b_dim0, b_dim1, b_num_tiles, b_swizzle_bytes
    ],
    load_mma_pipeline: ProducerConsumerPipeline[pipeline_stages],
    mma_op: MmaOpSM100_SS[
        c_type,
        a_type,
        b_type,
        block_tile_shape,
        mma_shape,
        accum_type=accum_type,
        cta_group=cta_group,
        cluster_shape=cluster_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        transpose_b=transpose_b,
    ],
    elect_one_warp: Bool,
    iter_idx: UInt32,
    k_start: UInt32,
):
    """Consumer main loop for BF16 MMA on SM100.

    Parameters:
        accum_type: DType of the MMA accumulator elements.
        c_type: DType of the C output elements.
        a_type: DType of the A operand elements.
        b_type: DType of the B operand elements.
        a_dim0: Row extent of each A SMEM tile, in elements.
        a_dim1: Column extent of each A SMEM tile, in elements.
        a_num_tiles: Number of A SMEM tiles in the pipeline buffer.
        a_swizzle_bytes: SMEM swizzle granularity for A tiles, in bytes.
        b_dim0: Row extent of each B SMEM tile, in elements.
        b_dim1: Column extent of each B SMEM tile, in elements.
        b_num_tiles: Number of B SMEM tiles in the pipeline buffer.
        b_swizzle_bytes: SMEM swizzle granularity for B tiles, in bytes.
        a_swizzle: TMA tensor map swizzle mode for the A operand.
        b_swizzle: TMA tensor map swizzle mode for the B operand.
        transpose_b: Whether the B operand is stored transposed.
        pipeline_stages: Number of stages in the load-to-MMA pipeline.
        block_tile_shape: Block tile shape as (BM, BN, BK).
        mma_shape: MMA instruction shape as (MMA_M, MMA_N, MMA_K).
        cta_group: Number of cooperating CTAs per MMA (defaults to 1).
        cluster_shape: CTA cluster shape as (M, N, K) (defaults to
            (1, 1, 1)).
        k_group_size: Number of K tiles consumed per pipeline stage
            (defaults to 1).

    Args:
        tmem_addr: TMEM accumulator address to write MMA results into.
        a_smem_tiles: Pipeline buffer of A SMEM tiles.
        b_smem_tiles: Pipeline buffer of B SMEM tiles.
        load_mma_pipeline: Producer/consumer pipeline between TMA load and
            MMA stages.
        mma_op: SM100 SS MMA operation descriptor.
        elect_one_warp: Whether this warp is the elected warp (warp 0).
        iter_idx: Current K iteration index for this pipeline stage.
        k_start: K iteration index at which to initialize the accumulator.
    """
    var stage = load_mma_pipeline.consumer_stage()

    load_mma_pipeline.wait_producer()

    if elect_one_sync():
        for jj in range(k_group_size):
            var j = UInt32(jj)
            var offset = stage * UInt32(k_group_size) + j
            var a_smem_tile = a_smem_tiles[offset]
            var b_smem_tile = b_smem_tiles[offset]

            mma_op.mma(
                a_smem_tile,
                b_smem_tile,
                tmem_addr,
                init_c=((iter_idx + j) == k_start),
            )
        mma_op.commit(load_mma_pipeline.consumer_mbar(stage))


# ===----------------------------------------------------------------------=== #
# _swiglu_epilogue_gmem - Register SwiGLU → direct GMEM stores
# ===----------------------------------------------------------------------=== #


@always_inline
def _swiglu_epilogue_gmem[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool,
    config: FusedSwiGLUMatmulConfig[a_type, b_type, c_type, transpose_b],
    num_accum_stages: Int,
](
    tmem_offset: UInt32,
    mma_output_pipeline: ProducerConsumerPipeline[num_accum_stages],
    output_stage_index: UInt32,
    c_gmem_ptr: UnsafePointer[Scalar[c_type], MutAnyOrigin],
    c_gmem_stride: UInt32,
    c_m_base: UInt32,
    c_h_base: UInt32,
    M_bound: UInt32,
    H_bound: UInt32,
    bias_smem_ptr: UnsafePointer[
        Scalar[c_type], MutAnyOrigin, address_space=.SHARED
    ],
):
    """TMEM → FP32 SwiGLU → BF16 GMEM epilogue. No SMEM, no TMA."""
    comptime AB_swapped = config.AB_swapped
    comptime cta_group = config.cta_group
    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]
    comptime OutputM = config.output_tile_shape[0]
    comptime OutputN = config.output_tile_shape[1]
    comptime stageN = OutputM if AB_swapped else OutputN
    comptime accum_type = config.accum_type

    comptime bits = 256
    comptime rep = stageN // (bits // 32)
    comptime fragment_size = (16 * (bits // 32)) // WARP_SIZE
    comptime rep_frag_size = fragment_size * rep

    comptime epc = EpilogueConfig.create(
        MMA_M=MMA_M,
        MMA_N=MMA_N,
        stageN=stageN,
        cta_group=cta_group,
        transpose_c=AB_swapped,
        BM=BM,
        BN=BN,
    )
    comptime is_lower_frag_required = epc.is_lower_frag_required
    comptime num_stages = epc.num_stages

    comptime accum_tile_layout = Layout.row_major(BM, stageN)
    comptime AccumTmemArray = TmemArrayType[
        accum_type, accum_tile_layout, num_stages, cta_group=cta_group
    ]
    comptime EpilogueApplierType = EpilogueApplier[
        MMA_M,
        stageN,
        num_stages,
        rep,
        cta_group,
        False,
    ]

    var accum_tiles = AccumTmemArray(Int(tmem_offset))
    var warp_id = get_warp_id()
    var lane = lane_id()

    var applier = EpilogueApplierType(
        UInt32(warp_id), UInt32(lane), (M_bound, H_bound * 2)
    )
    var frag_coords = EpilogueApplierType.Coords(UInt32(lane))

    var upper_frag_partial: Array[Scalar[accum_type], rep_frag_size]
    var lower_frag_partial = Array[Scalar[accum_type], rep_frag_size](
        uninitialized=True
    )

    comptime for stage in range(num_stages):
        var frags = accum_tiles[stage].load_fragments[rep]()
        AccumTmemArray.Tile.wait_load()

        comptime PartialType = Array[Scalar[accum_type], rep_frag_size]
        upper_frag_partial = rebind[PartialType](frags.upper).copy()

        comptime if is_lower_frag_required:
            lower_frag_partial = rebind[PartialType](frags.lower).copy()

        comptime if stage == num_stages - 1:
            AccumBarrier[cta_group].arrive(
                mma_output_pipeline, output_stage_index
            )

        comptime for r1 in range(rep):
            comptime if AB_swapped:
                var lane_row_is_even = (lane & 7) < 4

                var staged_u = applier.compute_staged_coords(
                    UInt32(stage),
                    frag_coords.top_upper[0],
                    frag_coords.top_upper[1] + UInt32(r1 * 8),
                )
                # Round staged_u[0] down to even so all bias[base+offset] stay
                # in [0, bias_dim-1]; staged_u[0] can be odd when the lane's
                # kernel-frame row is 2h+1 (up), giving h_row_top = h+½→h.
                var bias_base = (Int(staged_u[0]) >> 1) * 2
                var h_row_top = c_h_base + (staged_u[0] >> 1)
                var m_col_a = c_m_base + staged_u[1]
                var m_col_b = m_col_a + 1

                # u0 and u1 share the same H-row (h_row_top) → same bias values.
                var own_u0 = upper_frag_partial[r1 * 4 + 0]
                var prt_u0 = warp.shuffle_xor(own_u0, UInt32(4))
                var gate_u0 = Float32(own_u0) if lane_row_is_even else Float32(
                    prt_u0
                )
                var up_u0 = Float32(prt_u0) if lane_row_is_even else Float32(
                    own_u0
                )
                var own_u1 = upper_frag_partial[r1 * 4 + 1]
                var prt_u1 = warp.shuffle_xor(own_u1, UInt32(4))
                var gate_u1 = Float32(own_u1) if lane_row_is_even else Float32(
                    prt_u1
                )
                var up_u1 = Float32(prt_u1) if lane_row_is_even else Float32(
                    own_u1
                )
                comptime if config.use_bias:
                    var bias_gate_top = Float32(bias_smem_ptr[bias_base])
                    var bias_up_top = Float32(bias_smem_ptr[bias_base + 1])
                    gate_u0 += bias_gate_top
                    up_u0 += bias_up_top
                    gate_u1 += bias_gate_top
                    up_u1 += bias_up_top
                if (
                    lane_row_is_even
                    and m_col_a < M_bound
                    and h_row_top < H_bound
                ):
                    var sig_u0 = recip(Float32(1.0) + exp(-gate_u0))
                    c_gmem_ptr.store(
                        Int(m_col_a) * Int(c_gmem_stride) + Int(h_row_top),
                        (gate_u0 * sig_u0 * up_u0).cast[c_type](),
                    )
                if (
                    lane_row_is_even
                    and m_col_b < M_bound
                    and h_row_top < H_bound
                ):
                    var sig_u1 = recip(Float32(1.0) + exp(-gate_u1))
                    c_gmem_ptr.store(
                        Int(m_col_b) * Int(c_gmem_stride) + Int(h_row_top),
                        (gate_u1 * sig_u1 * up_u1).cast[c_type](),
                    )

                # u2 and u3 share h_row_bu → same bias values.
                var own_u2 = upper_frag_partial[r1 * 4 + 2]
                var prt_u2 = warp.shuffle_xor(own_u2, UInt32(4))
                var gate_u2 = Float32(own_u2) if lane_row_is_even else Float32(
                    prt_u2
                )
                var up_u2 = Float32(prt_u2) if lane_row_is_even else Float32(
                    own_u2
                )
                var h_row_bu = h_row_top + 4
                var own_u3 = upper_frag_partial[r1 * 4 + 3]
                var prt_u3 = warp.shuffle_xor(own_u3, UInt32(4))
                var gate_u3 = Float32(own_u3) if lane_row_is_even else Float32(
                    prt_u3
                )
                var up_u3 = Float32(prt_u3) if lane_row_is_even else Float32(
                    own_u3
                )
                comptime if config.use_bias:
                    var bias_gate_bu = Float32(bias_smem_ptr[bias_base + 8])
                    var bias_up_bu = Float32(bias_smem_ptr[bias_base + 9])
                    gate_u2 += bias_gate_bu
                    up_u2 += bias_up_bu
                    gate_u3 += bias_gate_bu
                    up_u3 += bias_up_bu
                if (
                    lane_row_is_even
                    and m_col_a < M_bound
                    and h_row_bu < H_bound
                ):
                    var sig_u2 = recip(Float32(1.0) + exp(-gate_u2))
                    c_gmem_ptr.store(
                        Int(m_col_a) * Int(c_gmem_stride) + Int(h_row_bu),
                        (gate_u2 * sig_u2 * up_u2).cast[c_type](),
                    )
                if (
                    lane_row_is_even
                    and m_col_b < M_bound
                    and h_row_bu < H_bound
                ):
                    var sig_u3 = recip(Float32(1.0) + exp(-gate_u3))
                    c_gmem_ptr.store(
                        Int(m_col_b) * Int(c_gmem_stride) + Int(h_row_bu),
                        (gate_u3 * sig_u3 * up_u3).cast[c_type](),
                    )

                comptime if is_lower_frag_required:
                    var h_row_tl = h_row_top + 8
                    var h_row_bl = h_row_top + 12

                    # l0 and l1 share h_row_tl → same bias values.
                    var own_l0 = lower_frag_partial[r1 * 4 + 0]
                    var prt_l0 = warp.shuffle_xor(own_l0, UInt32(4))
                    var gate_l0 = Float32(
                        own_l0
                    ) if lane_row_is_even else Float32(prt_l0)
                    var up_l0 = Float32(
                        prt_l0
                    ) if lane_row_is_even else Float32(own_l0)
                    var own_l1 = lower_frag_partial[r1 * 4 + 1]
                    var prt_l1 = warp.shuffle_xor(own_l1, UInt32(4))
                    var gate_l1 = Float32(
                        own_l1
                    ) if lane_row_is_even else Float32(prt_l1)
                    var up_l1 = Float32(
                        prt_l1
                    ) if lane_row_is_even else Float32(own_l1)
                    comptime if config.use_bias:
                        var bias_gate_tl = Float32(
                            bias_smem_ptr[bias_base + 16]
                        )
                        var bias_up_tl = Float32(bias_smem_ptr[bias_base + 17])
                        gate_l0 += bias_gate_tl
                        up_l0 += bias_up_tl
                        gate_l1 += bias_gate_tl
                        up_l1 += bias_up_tl
                    if (
                        lane_row_is_even
                        and m_col_a < M_bound
                        and h_row_tl < H_bound
                    ):
                        var sig_l0 = recip(Float32(1.0) + exp(-gate_l0))
                        c_gmem_ptr.store(
                            Int(m_col_a) * Int(c_gmem_stride) + Int(h_row_tl),
                            (gate_l0 * sig_l0 * up_l0).cast[c_type](),
                        )
                    if (
                        lane_row_is_even
                        and m_col_b < M_bound
                        and h_row_tl < H_bound
                    ):
                        var sig_l1 = recip(Float32(1.0) + exp(-gate_l1))
                        c_gmem_ptr.store(
                            Int(m_col_b) * Int(c_gmem_stride) + Int(h_row_tl),
                            (gate_l1 * sig_l1 * up_l1).cast[c_type](),
                        )

                    # l2 and l3 share h_row_bl → same bias values.
                    var own_l2 = lower_frag_partial[r1 * 4 + 2]
                    var prt_l2 = warp.shuffle_xor(own_l2, UInt32(4))
                    var gate_l2 = Float32(
                        own_l2
                    ) if lane_row_is_even else Float32(prt_l2)
                    var up_l2 = Float32(
                        prt_l2
                    ) if lane_row_is_even else Float32(own_l2)
                    var own_l3 = lower_frag_partial[r1 * 4 + 3]
                    var prt_l3 = warp.shuffle_xor(own_l3, UInt32(4))
                    var gate_l3 = Float32(
                        own_l3
                    ) if lane_row_is_even else Float32(prt_l3)
                    var up_l3 = Float32(
                        prt_l3
                    ) if lane_row_is_even else Float32(own_l3)
                    comptime if config.use_bias:
                        var bias_gate_bl = Float32(
                            bias_smem_ptr[bias_base + 24]
                        )
                        var bias_up_bl = Float32(bias_smem_ptr[bias_base + 25])
                        gate_l2 += bias_gate_bl
                        up_l2 += bias_up_bl
                        gate_l3 += bias_gate_bl
                        up_l3 += bias_up_bl
                    if (
                        lane_row_is_even
                        and m_col_a < M_bound
                        and h_row_bl < H_bound
                    ):
                        var sig_l2 = recip(Float32(1.0) + exp(-gate_l2))
                        c_gmem_ptr.store(
                            Int(m_col_a) * Int(c_gmem_stride) + Int(h_row_bl),
                            (gate_l2 * sig_l2 * up_l2).cast[c_type](),
                        )
                    if (
                        lane_row_is_even
                        and m_col_b < M_bound
                        and h_row_bl < H_bound
                    ):
                        var sig_l3 = recip(Float32(1.0) + exp(-gate_l3))
                        c_gmem_ptr.store(
                            Int(m_col_b) * Int(c_gmem_stride) + Int(h_row_bl),
                            (gate_l3 * sig_l3 * up_l3).cast[c_type](),
                        )
            else:
                # tu and bu share the same N-column → same bias values.
                var gate_tu = Float32(upper_frag_partial[r1 * 4 + 0])
                var up_tu = Float32(upper_frag_partial[r1 * 4 + 1])
                var staged_tu = applier.compute_staged_coords(
                    UInt32(stage),
                    frag_coords.top_upper[0],
                    frag_coords.top_upper[1] + UInt32(r1 * 8),
                )
                var grow_tu = c_m_base + staged_tu[0]
                var gcol_tu = c_h_base + (staged_tu[1] >> 1)

                var gate_bu = Float32(upper_frag_partial[r1 * 4 + 2])
                var up_bu = Float32(upper_frag_partial[r1 * 4 + 3])
                var staged_bu = applier.compute_staged_coords(
                    UInt32(stage),
                    frag_coords.bottom_upper[0],
                    frag_coords.bottom_upper[1] + UInt32(r1 * 8),
                )
                var grow_bu = c_m_base + staged_bu[0]
                var gcol_bu = c_h_base + (staged_bu[1] >> 1)

                comptime if config.use_bias:
                    var bias_gate = Float32(bias_smem_ptr[Int(staged_tu[1])])
                    var bias_up = Float32(bias_smem_ptr[Int(staged_tu[1]) + 1])
                    gate_tu += bias_gate
                    up_tu += bias_up
                    gate_bu += bias_gate
                    up_bu += bias_up

                if grow_tu < M_bound and gcol_tu < H_bound:
                    var sig_tu = recip(Float32(1.0) + exp(-gate_tu))
                    c_gmem_ptr.store(
                        Int(grow_tu) * Int(c_gmem_stride) + Int(gcol_tu),
                        (gate_tu * sig_tu * up_tu).cast[c_type](),
                    )

                if grow_bu < M_bound and gcol_bu < H_bound:
                    var sig_bu = recip(Float32(1.0) + exp(-gate_bu))
                    c_gmem_ptr.store(
                        Int(grow_bu) * Int(c_gmem_stride) + Int(gcol_bu),
                        (gate_bu * sig_bu * up_bu).cast[c_type](),
                    )

                comptime if is_lower_frag_required:
                    # tl and bl share the same N-column → same bias values.
                    var gate_tl = Float32(lower_frag_partial[r1 * 4 + 0])
                    var up_tl = Float32(lower_frag_partial[r1 * 4 + 1])
                    var staged_tl = applier.compute_staged_coords(
                        UInt32(stage),
                        frag_coords.top_lower[0],
                        frag_coords.top_lower[1] + UInt32(r1 * 8),
                    )
                    var grow_tl = c_m_base + staged_tl[0]
                    var gcol_tl = c_h_base + (staged_tl[1] >> 1)

                    var gate_bl = Float32(lower_frag_partial[r1 * 4 + 2])
                    var up_bl = Float32(lower_frag_partial[r1 * 4 + 3])
                    var staged_bl = applier.compute_staged_coords(
                        UInt32(stage),
                        frag_coords.bottom_lower[0],
                        frag_coords.bottom_lower[1] + UInt32(r1 * 8),
                    )
                    var grow_bl = c_m_base + staged_bl[0]
                    var gcol_bl = c_h_base + (staged_bl[1] >> 1)

                    comptime if config.use_bias:
                        var bias_gate_l = Float32(
                            bias_smem_ptr[Int(staged_tl[1])]
                        )
                        var bias_up_l = Float32(
                            bias_smem_ptr[Int(staged_tl[1]) + 1]
                        )
                        gate_tl += bias_gate_l
                        up_tl += bias_up_l
                        gate_bl += bias_gate_l
                        up_bl += bias_up_l

                    if grow_tl < M_bound and gcol_tl < H_bound:
                        var sig_tl = recip(Float32(1.0) + exp(-gate_tl))
                        c_gmem_ptr.store(
                            Int(grow_tl) * Int(c_gmem_stride) + Int(gcol_tl),
                            (gate_tl * sig_tl * up_tl).cast[c_type](),
                        )

                    if grow_bl < M_bound and gcol_bl < H_bound:
                        var sig_bl = recip(Float32(1.0) + exp(-gate_bl))
                        c_gmem_ptr.store(
                            Int(grow_bl) * Int(c_gmem_stride) + Int(gcol_bl),
                            (gate_bl * sig_bl * up_bl).cast[c_type](),
                        )


# ===----------------------------------------------------------------------=== #
# _swiglu_epilogue_smem_tma - Register SwiGLU → half SMEM → TMA store
# ===----------------------------------------------------------------------=== #


@always_inline
def _swiglu_epilogue_smem_tma[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool,
    config: FusedSwiGLUMatmulConfig[a_type, b_type, c_type, transpose_b],
    c_rank: Int,
    c_tile_shape: IndexList[c_rank],
    c_desc_shape: IndexList[c_rank],
    num_accum_stages: Int,
](
    tmem_offset: UInt32,
    mma_output_pipeline: ProducerConsumerPipeline[num_accum_stages],
    output_stage_index: UInt32,
    c_tma_op: TMATensorTile[c_type, c_rank, c_tile_shape, c_desc_shape],
    c_out_smem: UnsafePointer[
        Scalar[c_type], MutAnyOrigin, address_space=.SHARED
    ],
    c_m_base: UInt32,
    c_h_base: UInt32,
    M_bound: UInt32,
    H_bound: UInt32,
    bias_smem_ptr: UnsafePointer[
        Scalar[c_type], MutAnyOrigin, address_space=.SHARED
    ],
):
    """TMEM → FP32 SwiGLU → half SMEM → TMA store epilogue."""
    comptime AB_swapped = config.AB_swapped
    comptime cta_group = config.cta_group
    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]
    comptime OutputM = config.output_tile_shape[0]
    comptime OutputN = config.output_tile_shape[1]
    comptime HalfN = OutputN // 2
    comptime stageN = OutputM if AB_swapped else OutputN
    comptime accum_type = config.accum_type

    comptime bits = 256
    comptime rep = stageN // (bits // 32)
    comptime fragment_size = (16 * (bits // 32)) // WARP_SIZE
    comptime rep_frag_size = fragment_size * rep

    comptime epc = EpilogueConfig.create(
        MMA_M=MMA_M,
        MMA_N=MMA_N,
        stageN=stageN,
        cta_group=cta_group,
        transpose_c=AB_swapped,
        BM=BM,
        BN=BN,
    )
    comptime is_lower_frag_required = epc.is_lower_frag_required
    comptime num_stages = epc.num_stages

    comptime accum_tile_layout = Layout.row_major(BM, stageN)
    comptime AccumTmemArray = TmemArrayType[
        accum_type, accum_tile_layout, num_stages, cta_group=cta_group
    ]

    comptime _is_layout_b_swap = AB_swapped and cta_group == 2 and MMA_M == 128
    comptime c_out_smem_inner = BM // 2 if AB_swapped else HalfN
    comptime c_out_pair_tile_elems = OutputM * c_out_smem_inner
    comptime c_out_tile_elems = (
        2
        * c_out_pair_tile_elems if _is_layout_b_swap else c_out_pair_tile_elems
    )
    comptime num_output_warps = 4

    var accum_tiles = AccumTmemArray(Int(tmem_offset))
    var warp_id = get_warp_id()
    var lane = lane_id()

    var frag_coords = FragmentCoords[stageN, rep](UInt32(lane))

    var smem_row_base: UInt32
    var h_row_base: UInt32
    comptime if MMA_M == 256 or (MMA_M == 128 and cta_group == 1):
        smem_row_base = UInt32(warp_id) * 32
        h_row_base = UInt32(warp_id) * 16
    elif MMA_M == 64 and cta_group == 1:
        smem_row_base = UInt32(warp_id) * 16
        h_row_base = UInt32(warp_id) * 8
    else:
        smem_row_base = (
            UInt32(warp_id // 2) * UInt32(BM) + UInt32(warp_id % 2) * 32
        )
        comptime if AB_swapped:
            h_row_base = UInt32(warp_id % 2) * 16
        else:
            h_row_base = 0

    var lane_row = frag_coords.top_upper[0]
    var lane_col = frag_coords.top_upper[1]

    var upper_frag_partial: Array[Scalar[accum_type], rep_frag_size]
    var lower_frag_partial = Array[Scalar[accum_type], rep_frag_size](
        uninitialized=True
    )

    var store_idx = UInt32(0)

    comptime for stage in range(num_stages):
        var frags = accum_tiles[stage].load_fragments[rep]()
        AccumTmemArray.Tile.wait_load()

        comptime PartialType = Array[Scalar[accum_type], rep_frag_size]
        upper_frag_partial = rebind[PartialType](frags.upper).copy()

        comptime if is_lower_frag_required:
            lower_frag_partial = rebind[PartialType](frags.lower).copy()

        comptime if stage == num_stages - 1:
            AccumBarrier[cta_group].arrive(
                mma_output_pipeline, output_stage_index
            )

        var buf_base = c_out_smem + Int(store_idx & 1) * c_out_tile_elems
        var bias_stage_offset: Int = stage * stageN
        # For cg2 MMA_M=128 non-swap, warps 0-1 write the top SMEM half (→ H-cols
        # c_h_base .. c_h_base+stageN/2-1) while warps 2-3 write the bot SMEM half
        # (→ H-cols c_h_base+MMA_N/2 .. ). Fragment column indices are identical
        # across warp groups, so add the per-warp-group bias offset explicitly.
        # (For MMA_M=256 cg2, all warps write the same H-columns — no extra offset.)
        comptime if cta_group == 2 and MMA_M == 128 and not AB_swapped:
            bias_stage_offset += (Int(warp_id) // 2) * (MMA_N // 2)

        comptime for r in range(rep):
            comptime inc = r * 8

            comptime if AB_swapped:
                var lane_row_is_even = (lane & 7) < 4
                var h_row_top = h_row_base + (lane_row >> 1)
                var m_col_a = lane_col + UInt32(inc)
                var m_col_b = m_col_a + 1

                var pair_base = (
                    buf_base
                    + UInt32(warp_id // 2) * UInt32(c_out_pair_tile_elems)
                ) if _is_layout_b_swap else buf_base

                var own_ua = upper_frag_partial[r * 4 + 0]
                var prt_ua = warp.shuffle_xor(own_ua, UInt32(4))
                var gate_ua = Float32(own_ua) if lane_row_is_even else Float32(
                    prt_ua
                )
                var up_ua = Float32(prt_ua) if lane_row_is_even else Float32(
                    own_ua
                )
                comptime if config.use_bias:
                    if lane_row_is_even:
                        gate_ua += Float32(bias_smem_ptr[Int(h_row_top) * 2])
                        up_ua += Float32(bias_smem_ptr[Int(h_row_top) * 2 + 1])
                var sig_ua = recip(Float32(1.0) + exp(-gate_ua))
                if lane_row_is_even:
                    pair_base.store(
                        Int(m_col_a) * c_out_smem_inner + Int(h_row_top),
                        (gate_ua * sig_ua * up_ua).cast[c_type](),
                    )

                var own_ub = upper_frag_partial[r * 4 + 1]
                var prt_ub = warp.shuffle_xor(own_ub, UInt32(4))
                var gate_ub = Float32(own_ub) if lane_row_is_even else Float32(
                    prt_ub
                )
                var up_ub = Float32(prt_ub) if lane_row_is_even else Float32(
                    own_ub
                )
                comptime if config.use_bias:
                    if lane_row_is_even:
                        gate_ub += Float32(bias_smem_ptr[Int(h_row_top) * 2])
                        up_ub += Float32(bias_smem_ptr[Int(h_row_top) * 2 + 1])
                var sig_ub = recip(Float32(1.0) + exp(-gate_ub))
                if lane_row_is_even:
                    pair_base.store(
                        Int(m_col_b) * c_out_smem_inner + Int(h_row_top),
                        (gate_ub * sig_ub * up_ub).cast[c_type](),
                    )

                var own_uc = upper_frag_partial[r * 4 + 2]
                var prt_uc = warp.shuffle_xor(own_uc, UInt32(4))
                var gate_uc = Float32(own_uc) if lane_row_is_even else Float32(
                    prt_uc
                )
                var up_uc = Float32(prt_uc) if lane_row_is_even else Float32(
                    own_uc
                )
                comptime if config.use_bias:
                    if lane_row_is_even:
                        gate_uc += Float32(
                            bias_smem_ptr[Int(h_row_top) * 2 + 8]
                        )
                        up_uc += Float32(bias_smem_ptr[Int(h_row_top) * 2 + 9])
                var sig_uc = recip(Float32(1.0) + exp(-gate_uc))
                if lane_row_is_even:
                    pair_base.store(
                        Int(m_col_a) * c_out_smem_inner + Int(h_row_top + 4),
                        (gate_uc * sig_uc * up_uc).cast[c_type](),
                    )

                var own_ud = upper_frag_partial[r * 4 + 3]
                var prt_ud = warp.shuffle_xor(own_ud, UInt32(4))
                var gate_ud = Float32(own_ud) if lane_row_is_even else Float32(
                    prt_ud
                )
                var up_ud = Float32(prt_ud) if lane_row_is_even else Float32(
                    own_ud
                )
                comptime if config.use_bias:
                    if lane_row_is_even:
                        gate_ud += Float32(
                            bias_smem_ptr[Int(h_row_top) * 2 + 8]
                        )
                        up_ud += Float32(bias_smem_ptr[Int(h_row_top) * 2 + 9])
                var sig_ud = recip(Float32(1.0) + exp(-gate_ud))
                if lane_row_is_even:
                    pair_base.store(
                        Int(m_col_b) * c_out_smem_inner + Int(h_row_top + 4),
                        (gate_ud * sig_ud * up_ud).cast[c_type](),
                    )

                comptime if is_lower_frag_required:
                    var own_la = lower_frag_partial[r * 4 + 0]
                    var prt_la = warp.shuffle_xor(own_la, UInt32(4))
                    var gate_la = Float32(
                        own_la
                    ) if lane_row_is_even else Float32(prt_la)
                    var up_la = Float32(
                        prt_la
                    ) if lane_row_is_even else Float32(own_la)
                    comptime if config.use_bias:
                        if lane_row_is_even:
                            gate_la += Float32(
                                bias_smem_ptr[Int(h_row_top) * 2 + 16]
                            )
                            up_la += Float32(
                                bias_smem_ptr[Int(h_row_top) * 2 + 17]
                            )
                    var sig_la = recip(Float32(1.0) + exp(-gate_la))
                    if lane_row_is_even:
                        pair_base.store(
                            Int(m_col_a) * c_out_smem_inner
                            + Int(h_row_top + 8),
                            (gate_la * sig_la * up_la).cast[c_type](),
                        )

                    var own_lb = lower_frag_partial[r * 4 + 1]
                    var prt_lb = warp.shuffle_xor(own_lb, UInt32(4))
                    var gate_lb = Float32(
                        own_lb
                    ) if lane_row_is_even else Float32(prt_lb)
                    var up_lb = Float32(
                        prt_lb
                    ) if lane_row_is_even else Float32(own_lb)
                    comptime if config.use_bias:
                        if lane_row_is_even:
                            gate_lb += Float32(
                                bias_smem_ptr[Int(h_row_top) * 2 + 16]
                            )
                            up_lb += Float32(
                                bias_smem_ptr[Int(h_row_top) * 2 + 17]
                            )
                    var sig_lb = recip(Float32(1.0) + exp(-gate_lb))
                    if lane_row_is_even:
                        pair_base.store(
                            Int(m_col_b) * c_out_smem_inner
                            + Int(h_row_top + 8),
                            (gate_lb * sig_lb * up_lb).cast[c_type](),
                        )

                    var own_lc = lower_frag_partial[r * 4 + 2]
                    var prt_lc = warp.shuffle_xor(own_lc, UInt32(4))
                    var gate_lc = Float32(
                        own_lc
                    ) if lane_row_is_even else Float32(prt_lc)
                    var up_lc = Float32(
                        prt_lc
                    ) if lane_row_is_even else Float32(own_lc)
                    comptime if config.use_bias:
                        if lane_row_is_even:
                            gate_lc += Float32(
                                bias_smem_ptr[Int(h_row_top) * 2 + 24]
                            )
                            up_lc += Float32(
                                bias_smem_ptr[Int(h_row_top) * 2 + 25]
                            )
                    var sig_lc = recip(Float32(1.0) + exp(-gate_lc))
                    if lane_row_is_even:
                        pair_base.store(
                            Int(m_col_a) * c_out_smem_inner
                            + Int(h_row_top + 12),
                            (gate_lc * sig_lc * up_lc).cast[c_type](),
                        )

                    var own_ld = lower_frag_partial[r * 4 + 3]
                    var prt_ld = warp.shuffle_xor(own_ld, UInt32(4))
                    var gate_ld = Float32(
                        own_ld
                    ) if lane_row_is_even else Float32(prt_ld)
                    var up_ld = Float32(
                        prt_ld
                    ) if lane_row_is_even else Float32(own_ld)
                    comptime if config.use_bias:
                        if lane_row_is_even:
                            gate_ld += Float32(
                                bias_smem_ptr[Int(h_row_top) * 2 + 24]
                            )
                            up_ld += Float32(
                                bias_smem_ptr[Int(h_row_top) * 2 + 25]
                            )
                    var sig_ld = recip(Float32(1.0) + exp(-gate_ld))
                    if lane_row_is_even:
                        pair_base.store(
                            Int(m_col_b) * c_out_smem_inner
                            + Int(h_row_top + 12),
                            (gate_ld * sig_ld * up_ld).cast[c_type](),
                        )
            else:
                var gate_tu = Float32(upper_frag_partial[r * 4 + 0])
                var up_tu = Float32(upper_frag_partial[r * 4 + 1])
                comptime if config.use_bias:
                    var n_col_tu = (
                        Int(frag_coords.top_upper[1] + UInt32(inc))
                        + bias_stage_offset
                    )
                    gate_tu += Float32(bias_smem_ptr[n_col_tu])
                    up_tu += Float32(bias_smem_ptr[n_col_tu + 1])
                var sig_tu = recip(Float32(1.0) + exp(-gate_tu))
                var row_tu = smem_row_base + frag_coords.top_upper[0]
                var col_tu = (frag_coords.top_upper[1] + UInt32(inc)) >> 1
                buf_base.store(
                    Int(row_tu) * HalfN + Int(col_tu),
                    (gate_tu * sig_tu * up_tu).cast[c_type](),
                )

                var gate_bu = Float32(upper_frag_partial[r * 4 + 2])
                var up_bu = Float32(upper_frag_partial[r * 4 + 3])
                comptime if config.use_bias:
                    var n_col_bu = (
                        Int(frag_coords.bottom_upper[1] + UInt32(inc))
                        + bias_stage_offset
                    )
                    gate_bu += Float32(bias_smem_ptr[n_col_bu])
                    up_bu += Float32(bias_smem_ptr[n_col_bu + 1])
                var sig_bu = recip(Float32(1.0) + exp(-gate_bu))
                var row_bu = smem_row_base + frag_coords.bottom_upper[0]
                var col_bu = (frag_coords.bottom_upper[1] + UInt32(inc)) >> 1
                buf_base.store(
                    Int(row_bu) * HalfN + Int(col_bu),
                    (gate_bu * sig_bu * up_bu).cast[c_type](),
                )

                comptime if is_lower_frag_required:
                    var gate_tl = Float32(lower_frag_partial[r * 4 + 0])
                    var up_tl = Float32(lower_frag_partial[r * 4 + 1])
                    comptime if config.use_bias:
                        var n_col_tl = (
                            Int(frag_coords.top_lower[1] + UInt32(inc))
                            + bias_stage_offset
                        )
                        gate_tl += Float32(bias_smem_ptr[n_col_tl])
                        up_tl += Float32(bias_smem_ptr[n_col_tl + 1])
                    var sig_tl = recip(Float32(1.0) + exp(-gate_tl))
                    var row_tl = smem_row_base + frag_coords.top_lower[0]
                    var col_tl = (frag_coords.top_lower[1] + UInt32(inc)) >> 1
                    buf_base.store(
                        Int(row_tl) * HalfN + Int(col_tl),
                        (gate_tl * sig_tl * up_tl).cast[c_type](),
                    )

                    var gate_bl = Float32(lower_frag_partial[r * 4 + 2])
                    var up_bl = Float32(lower_frag_partial[r * 4 + 3])
                    comptime if config.use_bias:
                        var n_col_bl = (
                            Int(frag_coords.bottom_lower[1] + UInt32(inc))
                            + bias_stage_offset
                        )
                        gate_bl += Float32(bias_smem_ptr[n_col_bl])
                        up_bl += Float32(bias_smem_ptr[n_col_bl + 1])
                    var sig_bl = recip(Float32(1.0) + exp(-gate_bl))
                    var row_bl = smem_row_base + frag_coords.bottom_lower[0]
                    var col_bl = (
                        frag_coords.bottom_lower[1] + UInt32(inc)
                    ) >> 1
                    buf_base.store(
                        Int(row_bl) * HalfN + Int(col_bl),
                        (gate_bl * sig_bl * up_bl).cast[c_type](),
                    )

        named_barrier[Int32(num_output_warps * WARP_SIZE)](0)

        # TMA store: half SMEM → GMEM
        comptime if AB_swapped and _is_layout_b_swap:
            if warp_id == 0 and lane == 0:
                fence_async_view_proxy()

                var out_col = c_h_base
                var pair0_row = c_m_base + UInt32(stage * stageN)
                var pair1_row = c_m_base + UInt32((num_stages + stage) * stageN)

                if pair0_row < M_bound and out_col < H_bound:
                    cp_async_bulk_tensor_global_shared_cta(
                        buf_base,
                        UnsafePointer(to=c_tma_op.descriptor).bitcast[
                            NoneType
                        ](),
                        Index(Int(out_col), Int(pair0_row)),
                    )

                var pair1_buf = buf_base + c_out_pair_tile_elems
                if pair1_row < M_bound and out_col < H_bound:
                    cp_async_bulk_tensor_global_shared_cta(
                        pair1_buf,
                        UnsafePointer(to=c_tma_op.descriptor).bitcast[
                            NoneType
                        ](),
                        Index(Int(out_col), Int(pair1_row)),
                    )

                cp_async_bulk_commit_group()
        elif AB_swapped:
            if warp_id == 0 and lane == 0:
                fence_async_view_proxy()

                var out_col = c_h_base
                var out_row = c_m_base + UInt32(stage * stageN)

                if out_row < M_bound and out_col < H_bound:
                    cp_async_bulk_tensor_global_shared_cta(
                        buf_base,
                        UnsafePointer(to=c_tma_op.descriptor).bitcast[
                            NoneType
                        ](),
                        Index(Int(out_col), Int(out_row)),
                    )

                cp_async_bulk_commit_group()
        elif MMA_M == 256 or cta_group == 1:
            if warp_id == 0 and lane == 0:
                fence_async_view_proxy()

                var out_col = c_h_base + UInt32(stage * (stageN // 2))
                var out_row = c_m_base

                if out_row < M_bound and out_col < H_bound:
                    cp_async_bulk_tensor_global_shared_cta(
                        buf_base,
                        UnsafePointer(to=c_tma_op.descriptor).bitcast[
                            NoneType
                        ](),
                        Index(Int(out_col), Int(out_row)),
                    )

                cp_async_bulk_commit_group()
        else:
            # Layout B (2SM MMA_M=128, non-swap): two stores per stage.
            if warp_id == 0 and lane == 0:
                fence_async_view_proxy()

                var stage_h_off = UInt32(stage * (stageN // 2))
                var top_out_col = c_h_base + stage_h_off
                var bot_out_col = top_out_col + UInt32(BN // 2)
                var out_row = c_m_base

                if out_row < M_bound and top_out_col < H_bound:
                    cp_async_bulk_tensor_global_shared_cta(
                        buf_base,
                        UnsafePointer(to=c_tma_op.descriptor).bitcast[
                            NoneType
                        ](),
                        Index(Int(top_out_col), Int(out_row)),
                    )

                var bot_buf = buf_base + BM * HalfN
                if out_row < M_bound and bot_out_col < H_bound:
                    cp_async_bulk_tensor_global_shared_cta(
                        bot_buf,
                        UnsafePointer(to=c_tma_op.descriptor).bitcast[
                            NoneType
                        ](),
                        Index(Int(bot_out_col), Int(out_row)),
                    )

                cp_async_bulk_commit_group()

        store_idx += 1

        comptime if _is_layout_b_swap:
            cp_async_bulk_wait_group[Int32(2)]()
        elif AB_swapped or MMA_M == 256 or (cta_group == 1):
            cp_async_bulk_wait_group[Int32(1)]()
        else:
            cp_async_bulk_wait_group[Int32(2)]()

        comptime if stage > 0 or stage == num_stages - 1:
            named_barrier[Int32(num_output_warps * WARP_SIZE)](0)

    if lane == 0 and warp_id == 0:
        cp_async_bulk_wait_group[Int32(0)]()


# ===----------------------------------------------------------------------=== #
# SwiGLUKernelConstants - comptime constants for the launch site
# ===----------------------------------------------------------------------=== #


struct SwiGLUKernelConstants[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool,
    config: FusedSwiGLUMatmulConfig[a_type, b_type, c_type, transpose_b],
]:
    """Compile-time constants for TMA descriptor creation and kernel launch.

    Parameters:
        a_type: Element data type of the A operand.
        b_type: Element data type of the B operand.
        c_type: Element data type of the C output.
        transpose_b: Whether B is transposed (always True for SwiGLU).
        config: Fused SwiGLU matmul config carrying tile shapes and pipeline
            stages.
    """

    comptime BM = Self.config.block_tile_shape[0]
    comptime BN = Self.config.block_tile_shape[1]
    comptime BK = Self.config.block_tile_shape[2]
    comptime MMA_M = Self.config.mma_shape[0]
    comptime MMA_N = Self.config.mma_shape[1]
    comptime cta_group = Self.config.cta_group
    comptime AB_swapped = Self.config.AB_swapped
    comptime CLUSTER_M: Int = Self.config.cluster_shape[0]
    comptime CLUSTER_N: Int = Self.config.cluster_shape[1]

    # 3D TMA layouts for A (batch=1)
    comptime a_tile_dim0 = Self.BM // Self.CLUSTER_N
    comptime ATileLayout = RowMajorLayout[
        *_IntToComptimeInt[1, Self.a_tile_dim0, Self.BK]
    ]
    comptime ADescLayout = tma_desc_layout_3d[
        Self.a_type, 1, Self.a_tile_dim0, Self.config.a_swizzle
    ]

    # 3D TMA layouts for B (batch=1)
    comptime b_tile_dim0 = Self.BN // (Self.CLUSTER_M // Self.cta_group)
    comptime BTileLayout = RowMajorLayout[
        *_IntToComptimeInt[1, Self.b_tile_dim0, Self.BK]
    ]
    comptime BDescLayout = tma_desc_layout_3d[
        Self.b_type, 1, Self.b_tile_dim0, Self.config.b_swizzle
    ]

    # 2D TMA layout for C output (half-width, no swizzle)
    comptime OutputM = Self.config.output_tile_shape[0]
    comptime OutputN = Self.config.output_tile_shape[1]
    comptime HalfN = Self.OutputN // 2
    comptime c_store_m = Self.OutputM if (
        Self.MMA_M == 256 or Self.cta_group == 1 or Self.AB_swapped
    ) else Self.BM
    comptime _c_tma_inner_native = Self.BM // 2 if Self.AB_swapped else Self.HalfN
    comptime c_tma_inner = (
        max(
            Self._c_tma_inner_native, 8
        ) if Self.config.register_swiglu else Self._c_tma_inner_native
    )
    comptime CTileLayout = static_row_major[Self.c_store_m, Self.c_tma_inner]
    comptime CDescLayout = static_row_major[Self.c_store_m, Self.c_tma_inner]

    # SMEM sizes
    comptime SmemType = _SwiGLUSmem[
        Self.a_type,
        Self.b_type,
        Self.c_type,
        Self.transpose_b,
        config=Self.config,
    ]
    comptime _is_layout_b_swap = Self.AB_swapped and Self.cta_group == 2 and Self.MMA_M == 128
    comptime c_out_smem_inner = Self.BM // 2 if Self.AB_swapped else Self.HalfN
    comptime c_out_pair_tile_elems = Self.OutputM * Self.c_out_smem_inner
    comptime c_out_tile_elems = (
        2
        * Self.c_out_pair_tile_elems if Self._is_layout_b_swap else Self.c_out_pair_tile_elems
    )
    comptime c_out_smem_offset = (size_of[Self.SmemType]() + 127) & ~127
    comptime c_out_tile_bytes = Self.c_out_tile_elems * size_of[Self.c_type]()
    comptime total_smem_bytes = (
        size_of[
            Self.SmemType
        ]() if Self.config.register_swiglu else Self.c_out_smem_offset
        + 2 * Self.c_out_tile_bytes
    )

    # Launch parameters
    comptime EPILOGUE_LOAD_THREADS = WARP_SIZE if Self.config.use_bias else 0
    comptime NUM_THREADS = 224 + Self.EPILOGUE_LOAD_THREADS
    comptime Bias1DTileLayout = row_major[1, Self.MMA_N]()
    comptime Bias1DTile = TileTensor[
        Self.c_type, type_of(Self.Bias1DTileLayout), ImmutAnyOrigin
    ]


# ===----------------------------------------------------------------------=== #
# blackwell_swiglu_warp_specialized_kernel - monolithic GPU kernel
# ===----------------------------------------------------------------------=== #


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
@__name(
    StaticString("sm100_matmul_swiglu_")
    + StaticString(config.get_kernel_name())
)
def blackwell_swiglu_warp_specialized_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_rank: Int,
    a_tile_shape: IndexList[a_rank],
    a_desc_shape: IndexList[a_rank],
    b_rank: Int,
    b_tile_shape: IndexList[b_rank],
    b_desc_shape: IndexList[b_rank],
    c_rank: Int,
    c_tile_shape: IndexList[c_rank],
    c_desc_shape: IndexList[c_rank],
    transpose_b: Bool,
    config: FusedSwiGLUMatmulConfig[a_type, b_type, c_type, transpose_b],
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1),
    pdl_level: PDLLevel = PDLLevel(),
](
    a_tma_op: TMATensorTile[a_type, a_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[b_type, b_rank, b_tile_shape, b_desc_shape],
    c_tma_op: TMATensorTile[c_type, c_rank, c_tile_shape, c_desc_shape],
    c_gmem_ptr: UnsafePointer[Scalar[c_type], MutAnyOrigin],
    c_gmem_stride: UInt32,
    bias_1d_tile: SwiGLUKernelConstants[
        a_type,
        b_type,
        c_type,
        transpose_b,
        config=config,
    ].Bias1DTile,
    cluster_dim: StaticTuple[Int32, 3],
    mnk: StaticTuple[UInt32, 3],
    workspace: Span[UInt64, MutAnyOrigin],
):
    """Fused GEMM+SwiGLU warp-specialized kernel for SM100.

    Five warp roles when use_bias=True (four otherwise):
      - Scheduler:      issues CLC work requests.
      - Load:           TMA-loads A/B tiles into SMEM.
      - MMA:            executes UMMA and signals epilogue.
      - Epilogue:       reads TMEM, applies bias+SwiGLU, stores to GMEM.
      - EpilogueLoad:   async-loads the 1D bias tile from GMEM to SMEM (warp 7).

    Parameters:
        a_type: Element data type of the A operand.
        b_type: Element data type of the B operand.
        c_type: Element data type of the C output.
        a_rank: Rank of the A TMA tile and descriptor shapes.
        a_tile_shape: A TMA tile shape as an ``IndexList``.
        a_desc_shape: A TMA descriptor shape as an ``IndexList``.
        b_rank: Rank of the B TMA tile and descriptor shapes.
        b_tile_shape: B TMA tile shape as an ``IndexList``.
        b_desc_shape: B TMA descriptor shape as an ``IndexList``.
        c_rank: Rank of the C TMA tile and descriptor shapes.
        c_tile_shape: C TMA tile shape as an ``IndexList``.
        c_desc_shape: C TMA descriptor shape as an ``IndexList``.
        transpose_b: Whether B is transposed (always True for SwiGLU).
        config: Fused SwiGLU matmul config carrying tile shapes and pipeline
            stages.
        cluster_shape: CTA cluster shape as an (M, N, K) ``StaticTuple``
            (defaults to a singleton cluster).
        pdl_level: Programmatic dependency level for grid launch (defaults to
            ``PDLLevel.OFF``).

    Args:
        a_tma_op: TMA tensor tile for loading A from GMEM into SMEM.
        b_tma_op: TMA tensor tile for loading B from GMEM into SMEM.
        c_tma_op: TMA tensor tile for storing C to GMEM (SMEM to TMA path
            only).
        c_gmem_ptr: Base pointer to the C output tensor in GMEM (register to
            GMEM path).
        c_gmem_stride: Row stride of the C output tensor in GMEM, in elements.
        bias_1d_tile: 1D bias tile in GMEM loaded by the epilogue load warp.
        cluster_dim: CTA cluster dimensions used by the tile scheduler.
        mnk: Kernel-frame (M, N, K) dimensions; N is the pre-SwiGLU width,
            equal to twice the output H.
        workspace: Scratch workspace memory provided by the kernel launcher.
    """
    # ===== Compile-time constants =====
    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]
    comptime BK = config.block_tile_shape[2]
    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime CLUSTER_M = config.cluster_shape[0]
    comptime CLUSTER_N = config.cluster_shape[1]
    comptime CLUSTER_SIZE = CLUSTER_M * CLUSTER_N
    comptime cta_group = config.cta_group
    comptime AB_swapped = config.AB_swapped
    comptime k_group_size = config.k_group_size
    comptime num_pipeline_stages = config.num_pipeline_stages
    comptime num_group_pipeline_stages = num_pipeline_stages // k_group_size
    comptime num_clc_pipeline_stages = config.num_clc_pipeline_stages
    comptime num_accum_pipeline_stages = config.num_accum_pipeline_stages

    comptime NUM_TMEM_COLS = 512
    comptime stage_stride_cols = NUM_TMEM_COLS // num_accum_pipeline_stages

    comptime SCHEDULER_THREADS = WARP_SIZE
    comptime TMA_LOAD_THREADS = WARP_SIZE
    comptime MMA_THREADS = WARP_SIZE
    comptime num_output_warps = 4
    comptime EPILOGUE_THREADS = num_output_warps * WARP_SIZE
    comptime EPILOGUE_LOAD_THREADS = WARP_SIZE if config.use_bias else 0

    comptime clc_producer_arv_count = 1
    comptime clc_consumer_arv_count = SCHEDULER_THREADS + CLUSTER_SIZE * (
        TMA_LOAD_THREADS
        + MMA_THREADS
        + EPILOGUE_THREADS
        + EPILOGUE_LOAD_THREADS
    )
    comptime clc_throttle_producer_arv_count = TMA_LOAD_THREADS
    comptime clc_throttle_consumer_arv_count = SCHEDULER_THREADS
    comptime accum_pipeline_producer_arv_count = 1
    comptime accum_pipeline_consumer_arv_count = cta_group * EPILOGUE_THREADS

    comptime SmemType = _SwiGLUSmem[
        a_type, b_type, c_type, transpose_b, config=config
    ]
    comptime OutputM = config.output_tile_shape[0]
    comptime OutputN = config.output_tile_shape[1]
    comptime HalfN = OutputN // 2
    comptime _is_layout_b_swap = AB_swapped and cta_group == 2 and MMA_M == 128
    comptime c_out_smem_inner = BM // 2 if AB_swapped else HalfN
    comptime c_out_pair_tile_elems = OutputM * c_out_smem_inner
    comptime c_out_tile_elems = (
        2
        * c_out_pair_tile_elems if _is_layout_b_swap else c_out_pair_tile_elems
    )
    comptime c_out_smem_offset = (size_of[SmemType]() + 127) & ~127

    # ===== Shared memory =====
    var smem_bytes = external_memory[
        UInt8,
        address_space=.SHARED,
        alignment=128,
    ]()
    ref smem = smem_bytes.bitcast[SmemType]()[]
    var ptr_tmem_addr: UnsafePointer[
        UInt32, origin_of(smem.tmem_addr), address_space=.SHARED
    ] = smem.tmem_addr.unsafe_ptr()
    var tmem_dealloc_mbar = smem.tmem_dealloc_mbar.unsafe_ptr()
    var c_out_smem = (smem_bytes + c_out_smem_offset).bitcast[Scalar[c_type]]()

    # ===== A/B SMEM tile views =====
    var a_smem_tt = SMemTileArray2D[
        a_type,
        BM,
        BK,
        num_pipeline_stages,
        swizzle_mode_to_bytes[config.a_swizzle],
    ](smem.a_smem.unsafe_ptr())
    var b_smem_tt = SMemTileArray2D[
        b_type,
        BN,
        BK,
        num_pipeline_stages,
        swizzle_mode_to_bytes[config.b_swizzle],
    ](smem.b_smem.unsafe_ptr())

    # ===== Pipelines =====
    # TMA load → MMA dependency (A/B SMEM tiles)
    var load_mma_pipeline = ProducerConsumerPipeline[num_group_pipeline_stages](
        smem.tma_mma_mbars.unsafe_ptr()
    )
    # MMA → Epilogue dependency (TMEM accumulator stages)
    var mma_output_pipeline = ProducerConsumerPipeline[
        num_accum_pipeline_stages
    ](smem.accum_mbars.unsafe_ptr())

    # ===== Tile scheduler =====
    # Build SMemArray wrappers from the inline-array SMEM fields.
    # (SMemArray stores a raw pointer; building before barrier-init is safe.)
    var clc_resp = SMemArray[UInt128, num_clc_pipeline_stages](
        ref smem.clc_response
    )
    var clc_full = SMemArray[SharedMemBarrier, num_clc_pipeline_stages](
        ref smem.clc_mbars_full
    )
    var clc_empty = SMemArray[SharedMemBarrier, num_clc_pipeline_stages](
        ref smem.clc_mbars_empty
    )
    var clc_throttle = SMemArray[SharedMemBarrier, num_clc_pipeline_stages * 2](
        ref smem.clc_throttle_mbars
    )
    var scheduler = TileScheduler[
        num_stages=num_clc_pipeline_stages,
        cluster_shape=Index[dtype=DType.uint32](CLUSTER_M, CLUSTER_N, 1),
        block_swizzle_size=config.block_swizzle_size,
        rasterize_order=RasterOrder(config.raster_order._value),
    ](cluster_dim, clc_resp, clc_full, clc_empty, clc_throttle)

    # ===== Per-thread warp/CTA info =====
    var warp_id = get_warp_id()
    var elect_one_warp = warp_id == 0
    var elect_one_thread = elect_one_sync_with_mask()
    var elect_one_cta = (
        block_rank_in_cluster() % 2 == 0 if cta_group == 2 else True
    )
    var is_first_cta_in_cluster = block_rank_in_cluster() == 0

    var rank_m = block_id_in_cluster.x
    var rank_n = block_id_in_cluster.y
    var peer_cta_coord = (
        umod(rank_m, cta_group),
        ufloordiv(rank_m, cta_group),
        rank_n,
    )

    # ===== Multicast masks =====
    var a_multicast_mask: UInt16 = 0x0
    var b_multicast_mask: UInt16 = 0x0
    comptime for i in range(CLUSTER_N):
        a_multicast_mask |= UInt16(1 << (i * CLUSTER_M))
    comptime for i in range(CLUSTER_M // cta_group):
        b_multicast_mask |= UInt16(1 << (i * cta_group))
    a_multicast_mask <<= UInt16(rank_m)
    b_multicast_mask <<= UInt16(peer_cta_coord[0])
    b_multicast_mask <<= UInt16(rank_n * CLUSTER_M)

    var self_mask = 1 << Int(block_rank_in_cluster())
    var peer_mask = 1 << Int(block_rank_in_cluster() + 1)
    var mma_complete_mask = self_mask | peer_mask

    var num_iters: UInt32 = ceildiv(mnk[2], UInt32(BK))

    # ===== MMA operation =====
    var mma_op = MmaOpSM100_SS[
        c_type,
        a_type,
        b_type,
        config.block_tile_shape,
        config.mma_shape,
        accum_type=config.accum_type,
        cta_group=cta_group,
        cluster_shape=config.cluster_shape,
        a_swizzle=config.a_swizzle,
        b_swizzle=config.b_swizzle,
        transpose_b=True,
    ]()

    # ===== Barrier initialization (elect_one_warp && elect_one_thread) =====
    if elect_one_warp and elect_one_thread:
        a_tma_op.prefetch_descriptor()
        b_tma_op.prefetch_descriptor()
        c_tma_op.prefetch_descriptor()

        load_mma_pipeline.init_mbars(
            Int32(1),
            Int32(
                compute_input_consumer_count[CLUSTER_M, CLUSTER_N, cta_group]()
            ),
        )
        mma_output_pipeline.init_mbars(
            Int32(accum_pipeline_producer_arv_count),
            Int32(accum_pipeline_consumer_arv_count),
        )
        ProducerConsumerPipeline[num_clc_pipeline_stages](
            clc_throttle.ptr
        ).init_mbars(
            Int32(clc_throttle_producer_arv_count),
            Int32(clc_throttle_consumer_arv_count),
        )
        init_clc_barriers[num_clc_pipeline_stages](
            clc_full.ptr,
            clc_empty.ptr,
            Int32(clc_producer_arv_count),
            Int32(clc_consumer_arv_count),
        )
        tmem_dealloc_mbar[].init(Int32(EPILOGUE_THREADS * cta_group))
        comptime if config.use_bias:
            comptime bias_dim = BM if AB_swapped else MMA_N
            comptime epi_load_producer_arv_count = ceildiv(bias_dim, 8)
            ProducerConsumerPipeline[2](
                smem.epilogue_load_mbars.unsafe_ptr()
            ).init_mbars(
                Int32(epi_load_producer_arv_count),
                Int32(EPILOGUE_THREADS),
            )

    fence_mbarrier_init()

    comptime if CLUSTER_SIZE > 1:
        cluster_arrive_relaxed()

    comptime if CLUSTER_SIZE > 1:
        cluster_wait()
    else:
        barrier()

    # ===== TMA Load Warp =====
    if WarpRole.is_main_load():
        comptime if pdl_level > PDLLevel.OFF:
            wait_on_dependent_grids()

        for current in scheduler.work_iterator():
            scheduler.throttle_signal(is_first_cta_in_cluster)
            for i in range(0, Int(num_iters), k_group_size):
                load_AB[
                    block_tile_shape=config.block_tile_shape,
                    mma_shape=config.mma_shape,
                    cta_group=cta_group,
                    k_group_size=k_group_size,
                ](
                    a_tma_op,
                    b_tma_op,
                    a_smem_tt,
                    b_smem_tt,
                    load_mma_pipeline,
                    peer_cta_coord,
                    (Int(current.m), Int(current.n), Int(current.k_start)),
                    a_multicast_mask,
                    b_multicast_mask,
                    UInt32(i),
                    elect_one_cta,
                )
                load_mma_pipeline.producer_step()
            syncwarp()

        # Drain remaining pipeline stages so MMA warp can exit.
        for _ in range(num_group_pipeline_stages):
            load_mma_pipeline.wait_consumer()
            load_mma_pipeline.producer_step()

    # ===== Scheduler Warp =====
    if WarpRole.is_scheduler() and is_first_cta_in_cluster:
        comptime if num_clc_pipeline_stages == 0:
            return

        var sched_iter = scheduler.scheduler_iterator()
        for _ in sched_iter:
            sched_iter.signal_and_advance()
        sched_iter.drain()

    # ===== MMA Warp =====
    if WarpRole.is_mma():
        tcgen05_alloc[Int32(cta_group)](ptr_tmem_addr, NUM_TMEM_COLS)
        syncwarp()
        # Signal to epilogue warps that TMEM address is ready.
        named_barrier_arrive[Int32(MMA_THREADS + EPILOGUE_THREADS)](1)

        var tmem_addr = ptr_tmem_addr[0]

        for _ in scheduler.work_iterator():
            if elect_one_cta:
                var stage = mma_output_pipeline.producer_stage()
                mma_output_pipeline.wait_consumer()
                var tmem_offset = tmem_addr + stage * UInt32(stage_stride_cols)

                for i in range(0, Int(num_iters), k_group_size):
                    consumer_main_loop[
                        block_tile_shape=config.block_tile_shape,
                        mma_shape=config.mma_shape,
                        cta_group=cta_group,
                        cluster_shape=config.cluster_shape,
                        k_group_size=k_group_size,
                    ](
                        tmem_offset,
                        a_smem_tt,
                        b_smem_tt,
                        load_mma_pipeline,
                        mma_op,
                        elect_one_warp,
                        UInt32(i),
                        0,
                    )
                    load_mma_pipeline.consumer_step()

                if elect_one_sync():
                    comptime if cta_group == 1:
                        mma_arrive[cta_group](
                            mma_output_pipeline.producer_mbar(stage)
                        )
                    else:
                        mma_arrive_multicast[cta_group](
                            mma_output_pipeline.producer_mbar(stage),
                            UInt16(mma_complete_mask),
                        )
                mma_output_pipeline.producer_step()

        tcgen05_release_allocation_lock[Int32(cta_group)]()
        tmem_dealloc_mbar[].wait()

        comptime if pdl_level > PDLLevel.OFF:
            launch_dependent_grids()

        tcgen05_dealloc[Int32(cta_group)](tmem_addr, NUM_TMEM_COLS)

    # ===== Epilogue Load Producer Warp (warp 7, bias only) =====
    comptime if config.use_bias:
        if WarpRole.is_epilogue_load():
            var epi_load_pl = ProducerConsumerPipeline[2](
                smem.epilogue_load_mbars.unsafe_ptr()
            )
            comptime bias_dim = BM if AB_swapped else MMA_N
            comptime elems_per_lane = 8
            comptime bytes_per_lane = elems_per_lane * size_of[c_type]()
            var lane = Int(lane_id())
            var lane_start = lane * elems_per_lane
            var bias_total: Int
            comptime if AB_swapped:
                bias_total = Int(mnk[0])
            else:
                bias_total = Int(mnk[1])
            for current in scheduler.work_iterator():
                epi_load_pl.wait_consumer()
                var stage = epi_load_pl.producer_stage()
                var gmem_offset: Int
                comptime if AB_swapped:
                    gmem_offset = Int(current.m) * bias_dim
                else:
                    gmem_offset = Int(current.n) * MMA_N
                var valid_elems = min(bias_dim, bias_total - gmem_offset)
                if lane_start < bias_dim:
                    var src_bytes = Int32(
                        bytes_per_lane
                    ) if lane_start + elems_per_lane <= valid_elems else Int32(
                        0
                    )
                    var src_ptr = (
                        bias_1d_tile._storage + gmem_offset + lane_start
                    ).address_space_cast[.GLOBAL]()
                    var bias_smem_base: UnsafePointer[
                        Scalar[c_type],
                        origin_of(smem.bias_smem),
                        address_space=.SHARED,
                    ] = smem.bias_smem.unsafe_ptr()
                    var dst_ptr = (
                        bias_smem_base + Int(stage) * bias_dim + lane_start
                    )
                    async_copy[bytes_per_lane, fill=Scalar[c_type](0)](
                        src_ptr, dst_ptr, src_size=src_bytes
                    )
                var mbar = epi_load_pl.producer_mbar(stage)
                if lane_start < bias_dim:
                    async_copy_arrive(mbar[0].unsafe_ptr())
                    _ = mbar[0].arrive()
                epi_load_pl.producer_step()

    # ===== Epilogue Warps (SwiGLU) =====
    if WarpRole.is_epilogue():
        # Wait for MMA warp to allocate TMEM and write address.
        named_barrier[Int32(MMA_THREADS + EPILOGUE_THREADS)](1)
        var tmem_addr = ptr_tmem_addr[0]

        # User-frame (M, H) bounds from kernel-frame (M, N=2H).
        var out_M: UInt32
        var out_H: UInt32
        comptime if AB_swapped:
            out_M = mnk[1]
            out_H = mnk[0] >> 1
        else:
            out_M = mnk[0]
            out_H = mnk[1] >> 1

        comptime if config.use_bias:
            var epi_load_pl = ProducerConsumerPipeline[2](
                smem.epilogue_load_mbars.unsafe_ptr()
            )
            comptime bias_dim = BM if AB_swapped else MMA_N
            for current in scheduler.work_iterator():
                var stage_idx = mma_output_pipeline.consumer_stage()
                mma_output_pipeline.wait_producer()
                var tmem_offset = tmem_addr + stage_idx * UInt32(
                    stage_stride_cols
                )

                # User-frame base coords for this CTA's output tile.
                var c_m_base: UInt32
                var c_h_base: UInt32
                comptime if AB_swapped:
                    c_m_base = current.n * UInt32(MMA_N)
                    c_h_base = current.m * UInt32(BM // 2)
                else:
                    c_m_base = current.m * UInt32(BM)
                    c_h_base = current.n * UInt32(MMA_N // 2)

                var epi_stage = Int(epi_load_pl.consumer_stage())
                epi_load_pl.wait_producer()
                var bias_smem_base: UnsafePointer[
                    Scalar[c_type],
                    origin_of(smem.bias_smem),
                    address_space=.SHARED,
                ] = smem.bias_smem.unsafe_ptr()
                var bias_smem_ptr = bias_smem_base + epi_stage * bias_dim
                comptime if config.register_swiglu:
                    _swiglu_epilogue_gmem[
                        a_type,
                        b_type,
                        c_type,
                        transpose_b,
                        config,
                        num_accum_pipeline_stages,
                    ](
                        tmem_offset,
                        mma_output_pipeline,
                        stage_idx,
                        c_gmem_ptr,
                        c_gmem_stride,
                        c_m_base,
                        c_h_base,
                        out_M,
                        out_H,
                        bias_smem_ptr.as_unsafe_any_origin(),
                    )
                else:
                    _swiglu_epilogue_smem_tma[
                        a_type,
                        b_type,
                        c_type,
                        transpose_b,
                        config,
                        c_rank,
                        c_tile_shape,
                        c_desc_shape,
                        num_accum_pipeline_stages,
                    ](
                        tmem_offset,
                        mma_output_pipeline,
                        stage_idx,
                        c_tma_op,
                        c_out_smem.as_unsafe_any_origin(),
                        c_m_base,
                        c_h_base,
                        out_M,
                        out_H,
                        bias_smem_ptr.as_unsafe_any_origin(),
                    )
                _ = epi_load_pl.consumer_mbar(UInt32(epi_stage))[0].arrive()
                epi_load_pl.consumer_step()
                mma_output_pipeline.consumer_step()
        else:
            for current in scheduler.work_iterator():
                var stage_idx = mma_output_pipeline.consumer_stage()
                mma_output_pipeline.wait_producer()
                var tmem_offset = tmem_addr + stage_idx * UInt32(
                    stage_stride_cols
                )

                # User-frame base coords for this CTA's output tile.
                var c_m_base: UInt32
                var c_h_base: UInt32
                comptime if AB_swapped:
                    c_m_base = current.n * UInt32(MMA_N)
                    c_h_base = current.m * UInt32(BM // 2)
                else:
                    c_m_base = current.m * UInt32(BM)
                    c_h_base = current.n * UInt32(MMA_N // 2)

                var bias_smem_ptr = smem.bias_smem.unsafe_ptr()
                comptime if config.register_swiglu:
                    _swiglu_epilogue_gmem[
                        a_type,
                        b_type,
                        c_type,
                        transpose_b,
                        config,
                        num_accum_pipeline_stages,
                    ](
                        tmem_offset,
                        mma_output_pipeline,
                        stage_idx,
                        c_gmem_ptr,
                        c_gmem_stride,
                        c_m_base,
                        c_h_base,
                        out_M,
                        out_H,
                        bias_smem_ptr.as_unsafe_any_origin(),
                    )
                else:
                    _swiglu_epilogue_smem_tma[
                        a_type,
                        b_type,
                        c_type,
                        transpose_b,
                        config,
                        c_rank,
                        c_tile_shape,
                        c_desc_shape,
                        num_accum_pipeline_stages,
                    ](
                        tmem_offset,
                        mma_output_pipeline,
                        stage_idx,
                        c_tma_op,
                        c_out_smem.as_unsafe_any_origin(),
                        c_m_base,
                        c_h_base,
                        out_M,
                        out_H,
                        bias_smem_ptr.as_unsafe_any_origin(),
                    )
                mma_output_pipeline.consumer_step()

        # Signal MMA warp that TMEM can be freed.
        comptime if cta_group == 2:
            _ = tmem_dealloc_mbar[].arrive_cluster(block_rank_in_cluster() ^ 1)
        _ = tmem_dealloc_mbar[].arrive()
