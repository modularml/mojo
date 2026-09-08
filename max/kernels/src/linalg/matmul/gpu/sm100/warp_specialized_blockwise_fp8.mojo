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

"""Implements a warp-specialized blockwise FP8 matrix multiplication kernel for NVIDIA SM100 (Blackwell) GPUs."""

from std.math import align_up, ceildiv, gcd
from std.math.uutils import umod, ufloordiv
from std.sys import size_of

from max.gpu import WARP_SIZE
from max.gpu.sync import barrier
from max.gpu.primitives.cluster import (
    block_rank_in_cluster,
    cluster_sync,
    elect_one_sync,
    elect_one_sync_with_mask,
)
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.host.info import B200
from max.gpu import block_id_in_cluster, lane_id, warp_id as get_warp_id
from max.gpu.memory import (
    external_memory,
    fence_async_view_proxy,
    fence_mbarrier_init,
)
from max.gpu.compute.arch.mma_nvidia_sm100 import *
from max.gpu.sync import (
    named_barrier,
    named_barrier_arrive,
    syncwarp,
    umma_arrive_leader_cta,
    mbarrier_arrive,
)
from max.gpu.compute.arch.tcgen05 import *
from layout import (
    Coord,
    Layout,
    TensorLayout,
    TensorEngine,
    TileTensor,
    row_major,
    stack_allocation,
)
from layout.swizzle import make_swizzle
from layout.tensor_core_async import tile_layout_k_major_typed
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    TMATensorTile,
    _idx_product,
    create_tensor_tile,
)

from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple

from ....arch.sm100 import MmaOpSM100_SS
from .config import MatmulConfig
from .matmul import WarpRole, consumer_main_loop, stsm_helper
from .tile_scheduler import TileScheduler
from .pipeline import ProducerConsumerPipeline
from structured_kernels.tile_types import (
    SMemTileArray2D,
    SMemTileArray2DRowMajor,
    swizzle_mode_to_bytes,
)


@always_inline
def _get_accumulator_size[
    *,
    c_smem_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int,
]() -> IndexList[2]:
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]

    comptime num_m_mmas = BM // (mma_shape[0] // cta_group)
    comptime num_n_mmas = BN // (mma_shape[1] // cta_group)

    comptime assert num_m_mmas == 1 and num_n_mmas == 1

    comptime stageN = c_smem_layout.shape[1].value()
    comptime cg2_num_stages = MMA_N // stageN if MMA_M == 256 else MMA_N // stageN // 2
    comptime cg1_num_stages = MMA_N // stageN
    comptime num_stages = cg2_num_stages if cta_group == 2 else cg1_num_stages
    comptime data_paths = 16
    comptime bits = 256
    comptime repeats = stageN // (bits // 32)

    comptime num_elements_per_load = bits // 32  # each element in tmem is 4 bytes, 32 bits
    comptime fragment_size = (data_paths * num_elements_per_load) // WARP_SIZE
    comptime num_elements = repeats * fragment_size

    return Index(num_stages, num_elements)


@always_inline
def load_AB[
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    a_rank: Int,
    a_tile_shape: IndexList[a_rank],
    a_desc_shape: IndexList[a_rank],
    b_rank: Int,
    b_tile_shape: IndexList[b_rank],
    b_desc_shape: IndexList[b_rank],
    a_scales_rank: Int,
    a_scales_tile_shape: IndexList[a_scales_rank],
    a_scales_desc_shape: IndexList[a_scales_rank],
    a_dim0: Int,
    a_dim1: Int,
    a_num_tiles: Int,
    a_swizzle_bytes: Int,
    b_dim0: Int,
    b_dim1: Int,
    b_num_tiles: Int,
    b_swizzle_bytes: Int,
    a_scales_dim0: Int,
    a_scales_dim1: Int,
    a_scales_num_tiles: Int,
    num_pipeline_stages: Int,
    /,
    *,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int = 1,
](
    a_tma_op: TMATensorTile[a_type, a_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[b_type, b_rank, b_tile_shape, b_desc_shape],
    a_scales_tma_op: TMATensorTile[
        a_scales_type, a_scales_rank, a_scales_tile_shape, a_scales_desc_shape
    ],
    a_smem_tiles: SMemTileArray2D[
        a_type, a_dim0, a_dim1, a_num_tiles, a_swizzle_bytes
    ],
    b_smem_tiles: SMemTileArray2D[
        b_type, b_dim0, b_dim1, b_num_tiles, b_swizzle_bytes
    ],
    a_scales_smem_tiles: SMemTileArray2DRowMajor[
        a_scales_type, a_scales_dim0, a_scales_dim1, a_scales_num_tiles, 128
    ],
    load_mma_pipeline: ProducerConsumerPipeline[num_pipeline_stages],
    peer_cta_coord: Tuple[Int, Int, Int],
    work_tile_coord: Tuple[Int, Int],
    a_multicast_mask: UInt16,
    b_multicast_mask: UInt16,
    iter_idx: Int,
    elect_one_cta: Bool,
):
    """Loads A, B, and A-scales tiles into shared memory via multicast TMA for one pipeline stage.

    Parameters:
        a_type: FP8 element type of the A operand matrix.
        b_type: FP8 element type of the B operand matrix.
        a_scales_type: Element type of the A blockwise scales (`float32`).
        a_rank: Number of dimensions in the A TMA descriptor.
        a_tile_shape: Shared-memory tile shape for each A TMA load.
        a_desc_shape: Copy box shape governing bytes per A TMA load.
        b_rank: Number of dimensions in the B TMA descriptor.
        b_tile_shape: Shared-memory tile shape for each B TMA load.
        b_desc_shape: Copy box shape governing bytes per B TMA load.
        a_scales_rank: Number of dimensions in the A scales TMA descriptor.
        a_scales_tile_shape: Shared-memory tile shape for each A scales
            TMA load.
        a_scales_desc_shape: Copy box shape governing bytes per A scales
            TMA load.
        a_dim0: Row extent of each A shared-memory tile.
        a_dim1: Column extent of each A shared-memory tile.
        a_num_tiles: Number of A tiles in the shared-memory tile array.
        a_swizzle_bytes: Swizzle stride in bytes for the A shared-memory
            tile.
        b_dim0: Row extent of each B shared-memory tile.
        b_dim1: Column extent of each B shared-memory tile.
        b_num_tiles: Number of B tiles in the shared-memory tile array.
        b_swizzle_bytes: Swizzle stride in bytes for the B shared-memory
            tile.
        a_scales_dim0: Row extent of each A scales shared-memory tile.
        a_scales_dim1: Column extent of each A scales shared-memory tile.
        a_scales_num_tiles: Number of A scales tiles in the shared-memory
            tile array.
        num_pipeline_stages: Number of double-buffered stages for A, B,
            and A scales TMA loads.
        block_tile_shape: GEMM block tile shape `(BM, BN, BK)`.
        mma_shape: Tensor-core MMA shape `(MMA_M, MMA_N, MMA_K)`.
        cta_group: Number of CTAs cooperating per MMA group (defaults to
            1).

    Args:
        a_tma_op: TMA descriptor for async multicast loads of A tiles
            from global memory.
        b_tma_op: TMA descriptor for async multicast loads of B tiles
            from global memory.
        a_scales_tma_op: TMA descriptor for async loads of A blockwise
            scales from global memory.
        a_smem_tiles: Shared-memory tile array holding A tiles across
            pipeline stages.
        b_smem_tiles: Shared-memory tile array holding B tiles across
            pipeline stages.
        a_scales_smem_tiles: Shared-memory tile array holding A blockwise
            scales across pipeline stages.
        load_mma_pipeline: Producer/consumer pipeline synchronizing TMA
            loads with MMA consumption.
        peer_cta_coord: Peer CTA coordinate `(peer_id, m_coord, n_coord)`
            within the cluster.
        work_tile_coord: Work tile coordinate `(m, n)` of the current
            output tile.
        a_multicast_mask: Multicast mask selecting CTAs that receive A
            tile broadcasts.
        b_multicast_mask: Multicast mask selecting CTAs that receive B
            tile broadcasts.
        iter_idx: K-dimension iteration index for the current TMA load.
        elect_one_cta: Whether this CTA is the elected leader for
            multicast TMA setup.
    """
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]

    comptime a_expected_bytes = a_dim0 * a_dim1 * size_of[a_type]()
    comptime b_expected_bytes = b_dim0 * b_dim1 * size_of[b_type]()
    comptime a_scales_expected_bytes = (
        a_scales_dim0 * a_scales_dim1 * size_of[a_scales_type]()
    )
    # Leader CTAs expect SMEM from itself and their peers
    comptime expected_bytes = cta_group * (
        a_expected_bytes + b_expected_bytes + a_scales_expected_bytes
    )

    comptime a_tma_load_size = _idx_product[a_rank, a_desc_shape]()
    comptime b_tma_load_size = _idx_product[b_rank, b_desc_shape]()
    comptime a_scales_tma_load_size = _idx_product[
        a_scales_rank, a_scales_desc_shape
    ]()
    comptime a_tma_rows = a_desc_shape[0]
    comptime b_tma_rows = b_desc_shape[0]

    var stage = load_mma_pipeline.producer_stage()

    # Wait until MMA (consumer) has used the buffer.
    load_mma_pipeline.wait_consumer()

    var a_gmem_slice_coord = (
        peer_cta_coord[2] * a_tma_rows + work_tile_coord[0] * BM
    )
    var b_gmem_slice_coord = (
        peer_cta_coord[1] * b_tma_rows
        + peer_cta_coord[0] * BN
        + work_tile_coord[1] * MMA_N
    )

    var a_smem_tile = a_smem_tiles[stage]
    var b_smem_tile = b_smem_tiles[stage]
    var a_scales_smem_tile = a_scales_smem_tiles[stage]

    var a_smem_slice = type_of(a_smem_tile)(
        a_smem_tile.ptr + peer_cta_coord[2] * a_tma_load_size,
        a_smem_tile.layout,
    )
    var b_smem_slice = type_of(b_smem_tile)(
        b_smem_tile.ptr + peer_cta_coord[1] * b_tma_load_size,
        b_smem_tile.layout,
    )
    var tma_mbar = load_mma_pipeline.producer_mbar(stage)

    if elect_one_sync():
        if elect_one_cta:
            tma_mbar[0].expect_bytes(Int32(expected_bytes))

        a_tma_op.async_multicast_load[cta_group](
            a_smem_slice,
            tma_mbar[0],
            (iter_idx * BK, a_gmem_slice_coord),
            a_multicast_mask,
        )

        b_tma_op.async_multicast_load[cta_group](
            b_smem_slice,
            tma_mbar[0],
            (iter_idx * BK, b_gmem_slice_coord),
            b_multicast_mask,
        )

        a_scales_tma_op.async_copy[cta_group](
            a_scales_smem_tile,
            tma_mbar[0],
            (work_tile_coord[0] * BM, iter_idx),
        )


@always_inline
def multi_stage_reg_epilogue[
    c_rank: Int,
    c_tile_shape: IndexList[c_rank],
    c_desc_shape: IndexList[c_rank],
    accum_type: DType,
    accum_layout: TensorLayout,
    /,
    *,
    c_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    is_lower_frag_required: Bool,
    cta_group: Int,
    num_output_warps: Int,
    c_swizzle: TensorMapSwizzle,
](
    c_upper_main_tile: TileTensor[
        mut=True,
        accum_type,
        LayoutType=accum_layout,
        address_space=.LOCAL,
        ...,
    ],
    c_lower_main_tile: TileTensor[
        mut=True,
        accum_type,
        LayoutType=accum_layout,
        address_space=.LOCAL,
        ...,
    ],
    c_tiles: SMemTileArray2DRowMajor[c_type, ...],
    c_tma_op: TMATensorTile[c_type, c_rank, c_tile_shape, c_desc_shape],
    c_coord: Tuple[Int, Int],
    elect_one_warp: Bool,
):
    """Casts register accumulators to the output type and stores each stage to global memory via TMA.

    Parameters:
        c_rank: Number of dimensions in the C TMA descriptor.
        c_tile_shape: Shared-memory tile shape for each C TMA store.
        c_desc_shape: Copy box shape governing bytes per C TMA store.
        accum_type: Element type of the register accumulators before
            casting.
        accum_layout: Memory layout of the accumulator `TileTensor`.
        c_type: Element type of the C output matrix.
        block_tile_shape: GEMM block tile shape `(BM, BN, BK)`.
        mma_shape: Tensor-core MMA shape `(MMA_M, MMA_N, MMA_K)`.
        is_lower_frag_required: Whether the lower accumulator fragment must
            be processed and stored.
        cta_group: Number of CTAs cooperating per MMA group.
        num_output_warps: Number of warps participating in the epilogue
            store.
        c_swizzle: Swizzle mode for the C shared-memory tile.

    Args:
        c_upper_main_tile: Upper accumulator fragment holding scaled MMA
            results to be cast and stored.
        c_lower_main_tile: Lower accumulator fragment holding scaled MMA
            results to be cast and stored.
        c_tiles: Shared-memory tile array for staging C output tiles before
            TMA store.
        c_tma_op: TMA descriptor for async stores of C tiles to global
            memory.
        c_coord: Work tile coordinate `(m, n)` of the current output tile.
        elect_one_warp: Whether this warp is the elected leader for issuing
            TMA stores.
    """
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]

    comptime num_m_mmas = BM // (mma_shape[0] // cta_group)
    comptime num_n_mmas = BN // (mma_shape[1] // cta_group)

    comptime assert num_m_mmas == 1 and num_n_mmas == 1

    comptime num_stages = accum_layout.static_shape[0]
    comptime num_elements = accum_layout.static_shape[1]

    comptime data_paths = 16
    comptime bits = 256
    comptime num_elements_per_load = bits // 32  # each element in tmem is 4 bytes, 32 bits
    comptime fragment_size = (data_paths * num_elements_per_load) // WARP_SIZE
    comptime repeats = num_elements // fragment_size
    comptime stageN = repeats * (bits // 32)
    comptime fragments_per_stage = fragment_size * repeats

    comptime swizzle = make_swizzle[c_type, c_swizzle]()

    var warp_id = get_warp_id()

    comptime for stage in range(num_stages):
        var upper_frag = c_upper_main_tile.load[fragments_per_stage](
            Coord(stage, 0)
        )
        var lower_frag = c_lower_main_tile.load[fragments_per_stage](
            Coord(stage, 0)
        )

        # Assume double-buffer for shared memory packing
        var c_smem_tile = c_tiles[stage % 2]
        comptime c_smem_tile_m = 32 if cta_group == 2 else BM // num_output_warps
        var c_smem_warp_tile = c_smem_tile.tile[c_smem_tile_m, stageN](
            warp_id, 0
        )

        var c_smem_warp_tile_upper = c_smem_warp_tile.tile[data_paths, stageN](
            0, 0
        )
        var upper_st = Array[Scalar[c_type], fragments_per_stage](
            uninitialized=True
        )

        comptime cast_width = 4 // size_of[Scalar[c_type]]()
        comptime for _i in range(fragments_per_stage // cast_width):
            comptime offset = _i * cast_width
            var src = SIMD[accum_type, cast_width]()
            comptime for _j in range(cast_width):
                src[_j] = upper_frag[offset + _j]
            var casted = src.cast[c_type]()
            comptime for _j in range(cast_width):
                upper_st[offset + _j] = casted[_j]
        stsm_helper[swizzle, stageN](upper_st, c_smem_warp_tile_upper)

        var c_smem_warp_tile_lower = c_smem_warp_tile.tile[data_paths, stageN](
            1, 0
        )

        comptime if is_lower_frag_required:
            var lower_st = Array[Scalar[c_type], fragments_per_stage](
                uninitialized=True
            )

            comptime for _i in range(fragments_per_stage // cast_width):
                comptime offset = _i * cast_width
                var src = SIMD[accum_type, cast_width]()
                comptime for _j in range(cast_width):
                    src[_j] = lower_frag[offset + _j]
                var casted = src.cast[c_type]()
                comptime for _j in range(cast_width):
                    lower_st[offset + _j] = casted[_j]
            stsm_helper[swizzle, stageN](lower_st, c_smem_warp_tile_lower)

        # Guard the write to shared memory is done.
        named_barrier[Int32(num_output_warps * WARP_SIZE)]()

        var lane = lane_id()

        comptime CG2_TMA_BM = type_of(c_smem_tile).static_shape[
            0
        ] if MMA_M == 256 else BM
        comptime CG1_TMA_BM = type_of(c_smem_tile).static_shape[0]
        comptime TMA_BM = CG2_TMA_BM if cta_group == 2 else CG1_TMA_BM

        var cg2_elect_one_warp = (
            warp_id == 0 if MMA_M == 256 else umod(warp_id, 2) == 0
        )
        var cg1_elect_one_warp = warp_id == 0
        var elect_one_warp = (
            cg2_elect_one_warp if cta_group == 2 else cg1_elect_one_warp
        )

        var coord_n_mma_m256 = c_coord[1] * MMA_N + stage * stageN
        var coord_n_mma_m128 = (
            c_coord[1] * MMA_N + (stage * stageN) + (BN * ufloordiv(warp_id, 2))
        )

        var cg2_coord_n = coord_n_mma_m256 if MMA_M == 256 else coord_n_mma_m128
        var cg1_coord_n = coord_n_mma_m256
        var coord_n = cg2_coord_n if cta_group == 2 else cg1_coord_n
        var coord_m = c_coord[0] * BM

        var cg2_c_smem_coord_m = 0 if MMA_M == 256 else ufloordiv(warp_id, 2)
        var cg1_c_smem_coord_m = 0
        var c_smem_coord_m = (
            cg2_c_smem_coord_m if cta_group == 2 else cg1_c_smem_coord_m
        )
        var c_smem_split = c_smem_tile.tile[TMA_BM, stageN](c_smem_coord_m, 0)

        if elect_one_warp and lane == 0:
            fence_async_view_proxy()
            c_tma_op.async_store(
                c_smem_split,
                (
                    coord_n,
                    coord_m,
                ),
            )
            c_tma_op.commit_group()

        # Keep one tma store in fly
        comptime if stage < num_stages - 1:
            c_tma_op.wait_group[1]()
        # Last stage guard all tma store to finish
        else:
            c_tma_op.wait_group[0]()

        comptime if stage > 0 and stage < num_stages - 1:
            # Guard the tma read from shared memory is done.
            named_barrier[Int32(num_output_warps * WARP_SIZE)]()


@always_inline
def promote_accumulators[
    pipeline_stages: Int,
    num_accum_pipeline_stages: Int,
    accum_type: DType,
    accum_layout: TensorLayout,
    a_scales_type: DType,
    b_scales_type: DType,
    b_scales_layout: TensorLayout,
    /,
    *,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int,
    CLUSTER_SIZE: Int32,
    is_lower_frag_required: Bool,
    num_output_warps: Int,
](
    b_scales: TileTensor[mut=False, b_scales_type, b_scales_layout, ...],
    a_scales_smem_tiles: SMemTileArray2DRowMajor[a_scales_type, ...],
    c_upper_main_tile: TileTensor[
        mut=True,
        accum_type,
        LayoutType=accum_layout,
        address_space=.LOCAL,
        ...,
    ],
    c_lower_main_tile: TileTensor[
        mut=True,
        accum_type,
        LayoutType=accum_layout,
        address_space=.LOCAL,
        ...,
    ],
    mma_output_pipeline: ProducerConsumerPipeline[num_accum_pipeline_stages],
    tmem_addr: UInt32,
    load_mma_pipeline: ProducerConsumerPipeline[pipeline_stages],
    work_tile_coord: Tuple[Int, Int],
    elect_one_warp: Bool,
    stage_stride_cols: Int,
    k_iter: Int,
    problem_shape: StaticTuple[Int32, 3],
):
    """Promotes FP8 MMA partial products by applying blockwise A and B scales to the tensor-memory accumulators.

    Parameters:
        pipeline_stages: Number of double-buffered stages in the load-MMA
            pipeline.
        num_accum_pipeline_stages: Number of double-buffered stages in the
            MMA-output pipeline.
        accum_type: Element type of the accumulators (`float32`).
        accum_layout: Memory layout of the accumulator `TileTensor`.
        a_scales_type: Element type of the A blockwise scales (`float32`).
        b_scales_type: Element type of the B blockwise scales (`float32`).
        b_scales_layout: Memory layout of the B scales `TileTensor`.
        block_tile_shape: GEMM block tile shape `(BM, BN, BK)`.
        mma_shape: Tensor-core MMA shape `(MMA_M, MMA_N, MMA_K)`.
        cta_group: Number of CTAs cooperating per MMA group.
        CLUSTER_SIZE: Total number of CTAs in the cluster.
        is_lower_frag_required: Whether the lower accumulator fragment must
            be processed.
        num_output_warps: Number of warps participating in the epilogue.

    Args:
        b_scales: B blockwise scales used to rescale FP8 partial products.
        a_scales_smem_tiles: Shared-memory tile array holding A blockwise
            scales loaded for the current pipeline stage.
        c_upper_main_tile: Upper accumulator fragment to accumulate scaled
            results into.
        c_lower_main_tile: Lower accumulator fragment to accumulate scaled
            results into.
        mma_output_pipeline: Producer/consumer pipeline synchronizing MMA
            production with accumulator consumption.
        tmem_addr: Base tensor-memory address for loading accumulator
            fragments.
        load_mma_pipeline: Producer/consumer pipeline synchronizing TMA
            loads with MMA consumption.
        work_tile_coord: Work tile coordinate `(m, n)` of the current
            output tile.
        elect_one_warp: Whether this warp is the elected leader for
            barrier signaling.
        stage_stride_cols: Column stride between accumulator pipeline
            stages in tensor memory.
        k_iter: K-dimension iteration index for the current scale
            application.
        problem_shape: Full GEMM problem dimensions `(M, N, K)`.
    """
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]

    comptime num_m_mmas = BM // (mma_shape[0] // cta_group)
    comptime num_n_mmas = BN // (mma_shape[1] // cta_group)

    comptime assert num_m_mmas == 1 and num_n_mmas == 1

    comptime assert (
        a_scales_type == b_scales_type and accum_type == .float32
    ), "Only support float32 for a_scales, b_scales, and accum_type"
    # Rows each warp is responsible for:
    # warp_id 0 -> 0-15 upper, 16-31 lower
    # warp_id 1 -> 32-47 upper, 48-63 lower
    # warp_id 2 -> 64-79 upper, 80-95 lower
    # warp_id 3 -> 96-111 upper, 112-127 lower

    var M = problem_shape[0]
    var N = problem_shape[1]
    var K = problem_shape[2]

    comptime num_stages = accum_layout.static_shape[0]
    comptime num_elements = accum_layout.static_shape[1]
    comptime data_paths = 16
    comptime bits = 256
    comptime num_elements_per_load = bits // 32  # each element in tmem is 4 bytes, 32 bits
    comptime fragment_size = (data_paths * num_elements_per_load) // WARP_SIZE
    comptime assert fragment_size == 4, "fragment_size must be 4"
    comptime repeats = num_elements // fragment_size
    comptime stageN = repeats * (bits // 32)
    comptime load_width = 2

    var bm = work_tile_coord[0]
    var bn = work_tile_coord[1]

    var tma_load_stage_index = load_mma_pipeline.consumer_stage()

    # scale_b index calculation when MMA_N != BK(128)
    var b_scale_idx0 = 0
    var b_scale_next_n = 0
    var b_scale_0: Scalar[accum_type]
    var b_scale_1: Scalar[accum_type]

    comptime if MMA_N != BK:
        comptime assert stageN <= gcd(MMA_N, BK) and (
            gcd(MMA_N, BK) % stageN == 0
        ), (
            "gcd(MMA_N, BK) must be divisible by stageN. If not then this"
            " step should be updated to support non-divisible case"
            " accordingly"
        )

        var global_bn_start = bn * MMA_N
        var begin_n = min(
            Int32(BK) - Int32(umod(global_bn_start, BK)), Int32(MMA_N)
        )
        var end_n = min(N - Int32(global_bn_start), Int32(MMA_N))

        # find the first b_scale index just by dividing by block size (128)
        # we use `b_scale_next_n` to find the second b_scale index later
        b_scale_idx0 = ufloordiv(global_bn_start, BK)
        # If MMA_N > BK (128) then we should use two scales_b in each block. `next_n` determines the border between the two scales_b.
        # Example: N = 960, MMA_N = 192, num_of_b_scales: ceildiv(960, BK) = 8
        # <------------------------------------ MMA_N (192) ------------------------------------>
        # <-------------------------128------------------------------>|<----------64------------>
        # <-------------------------block_scales[idx0]--------------->|<--block_scales[idx0+1]-->
        #                                                           next_n(128)

        # this condition determines the border between the two scale_b and whether we have two scale_b in this block or one
        b_scale_next_n = Int(begin_n) if begin_n < end_n else MMA_N
        # Example 1: N = 896, MMA_N = 192, num_of_b_scales: ceildiv(896, BK) = 7
        # This will be the last block on the horizontal axis i.e., work_tile_block[1] == 4
        # <------------------------------------ MMA_N (192) ------------------------------------>
        # <------------------------------------------------------------------------------------->|<
        # <-----------------------------------block_scales[6]----------------------------------->|<
        #                                                                                     next_n (192)

        # Example 2: N = 904, MMA_N = 192, num_of_b_scales: ceildiv(N, BK) = 8
        # This will be the last block on the horizontal axis i.e., work_tile_block[1] == 4
        # <------------------------------------ MMA_N (192) ------------------------------------>
        # <-------------------------128------------------------------>|<----------64------------>
        # <-------------------------block_scales[6]------------------>|<-----block_scales[7]---->
        #                                                           next_n(128)

        # prefetch b scales
        b_scale_0 = rebind[Scalar[accum_type]](
            b_scales[b_scale_idx0, k_iter].cast[accum_type]()
        )
        # this mean in this block we have two scale_b
        if b_scale_next_n < MMA_N:
            b_scale_1 = rebind[Scalar[accum_type]](
                b_scales[b_scale_idx0 + 1, k_iter].cast[accum_type]()
            )
        else:
            b_scale_1 = 0.0

    else:
        # when MMA_N == BK == 128 we only have one scale_b per block
        b_scale_0 = rebind[Scalar[accum_type]](
            b_scales[bn, k_iter].cast[accum_type]()
        )
        b_scale_1 = 0.0

    var warp_id = get_warp_id()

    # we update the column offset to include the current stage
    var staged_c_row: Int
    var staged_c_col: Int

    comptime if MMA_M == 256 or (MMA_M == 128 and cta_group == 1):
        # based on layout A/D (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-data-path-layout-a)
        staged_c_row = warp_id * WARP_SIZE
        staged_c_col = 0
    elif MMA_M == 64 and cta_group == 1:
        # based on layout F (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-data-path-layout-f)
        staged_c_row = warp_id * (WARP_SIZE // 2)
        staged_c_col = 0
    else:
        # based on layout B (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-data-path-layout-b)
        staged_c_row = umod(warp_id, 2) * WARP_SIZE
        staged_c_col = BN * ufloordiv(warp_id, 2)

    # this is the tensor memory layout
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-matrix-fragments-shape-16256b
    # we use it to figure out the starting coordinate
    comptime threads_per_row = (
        stageN // repeats // load_width
    )  # 4 threads per row
    var top_frag_upper_coord = StaticTuple[UInt32, 2](
        UInt32(ufloordiv(lane_id(), threads_per_row)),
        UInt32(umod(lane_id(), threads_per_row * load_width)),
    )

    # getting the other 3 coordinates is straightforward. Each fragment is spaced out by 16 rows
    # and within each fragment the elements are spaced out by 8 rows(this can be seen by the tv layout).
    var bottom_frag_upper_coord = StaticTuple[UInt32, 2](
        top_frag_upper_coord[0] + 8, top_frag_upper_coord[1]
    )

    var top_frag_lower_coord = StaticTuple[UInt32, 2](
        top_frag_upper_coord[0] + 16, top_frag_upper_coord[1]
    )

    var bottom_frag_lower_coord = StaticTuple[UInt32, 2](
        top_frag_lower_coord[0] + 8, top_frag_lower_coord[1]
    )

    var mma_output_stage = mma_output_pipeline.consumer_stage()
    var tmem_offset = mma_output_stage * UInt32(stage_stride_cols) + tmem_addr
    mma_output_pipeline.wait_producer()

    var a_scales_smem = a_scales_smem_tiles[tma_load_stage_index]
    # load a_scales from SMEM
    var upper_sfa0_smem = a_scales_smem[
        0, UInt32(staged_c_row) + top_frag_upper_coord[0]
    ].cast[accum_type]()
    var upper_sfa1_smem = a_scales_smem[
        0, UInt32(staged_c_row) + bottom_frag_upper_coord[0]
    ].cast[accum_type]()

    var lower_sfa0_smem = Scalar[accum_type]()
    var lower_sfa1_smem = Scalar[accum_type]()

    comptime if is_lower_frag_required:
        lower_sfa0_smem = rebind[Scalar[accum_type]](
            a_scales_smem[
                0, UInt32(staged_c_row) + top_frag_lower_coord[0]
            ].cast[accum_type]()
        )
        lower_sfa1_smem = rebind[Scalar[accum_type]](
            a_scales_smem[
                0, UInt32(staged_c_row) + bottom_frag_lower_coord[0]
            ].cast[accum_type]()
        )

    syncwarp()
    if lane_id() < Int(CLUSTER_SIZE):
        _ = load_mma_pipeline.consumer_mbar(tma_load_stage_index)[0].arrive()
    syncwarp()

    comptime rep_frag_size = repeats * fragment_size
    var upper_frag: Array[Scalar[accum_type], rep_frag_size]
    var lower_frag = Array[Scalar[accum_type], rep_frag_size](
        uninitialized=True
    )

    comptime for stage in range(num_stages):
        var stage_tmem_addr = tmem_offset + UInt32(stage * stageN)
        upper_frag = tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=repeats,
            dtype=accum_type,
            pack=False,
            width=rep_frag_size,
        ](stage_tmem_addr)

        comptime if is_lower_frag_required:
            lower_frag = tcgen05_ld[
                datapaths=data_paths,
                bits=bits,
                repeat=repeats,
                dtype=accum_type,
                pack=False,
                width=rep_frag_size,
            ](stage_tmem_addr + (16 << 16))

        tcgen05_load_wait()

        comptime if stage == num_stages - 1:
            comptime if cta_group == 1:
                _ = mbarrier_arrive(
                    mma_output_pipeline.consumer_mbar(mma_output_stage)
                )
            else:
                umma_arrive_leader_cta(
                    mma_output_pipeline.consumer_mbar(mma_output_stage)
                )

        var b_scale: Scalar[accum_type]

        comptime if MMA_N != BK:
            # check if we cross the border between the two scale_b
            b_scale = (
                b_scale_0 if (stage * stageN + staged_c_col)
                < b_scale_next_n else b_scale_1
            )
        else:
            b_scale = b_scale_0

        comptime for ld_iter in range(repeats):
            comptime for j in range(fragment_size // 2):
                comptime offset = ld_iter * fragment_size + j * 2

                var upper_a_scale = (
                    upper_sfa0_smem if j == 0 else upper_sfa1_smem
                )
                var lower_a_scale = (
                    lower_sfa0_smem if j == 0 else lower_sfa1_smem
                )

                var upper_scale = upper_a_scale * b_scale
                var lower_scale = lower_a_scale * b_scale

                c_upper_main_tile[Coord(stage, offset)] = c_upper_main_tile[
                    Coord(stage, offset)
                ] + rebind[Scalar[accum_type]](upper_frag[offset]) * rebind[
                    Scalar[accum_type]
                ](
                    upper_scale
                )
                c_upper_main_tile[Coord(stage, offset + 1)] = c_upper_main_tile[
                    Coord(stage, offset + 1)
                ] + rebind[Scalar[accum_type]](upper_frag[offset + 1]) * rebind[
                    Scalar[accum_type]
                ](
                    upper_scale
                )

                comptime if is_lower_frag_required:
                    c_lower_main_tile[Coord(stage, offset)] = c_lower_main_tile[
                        Coord(stage, offset)
                    ] + rebind[Scalar[accum_type]](lower_frag[offset]) * rebind[
                        Scalar[accum_type]
                    ](
                        lower_scale
                    )
                    c_lower_main_tile[
                        Coord(stage, offset + 1)
                    ] = c_lower_main_tile[Coord(stage, offset + 1)] + rebind[
                        Scalar[accum_type]
                    ](
                        lower_frag[offset + 1]
                    ) * rebind[
                        Scalar[accum_type]
                    ](
                        lower_scale
                    )


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(a_scales_tma_op, `nvvm.grid_constant`)
@__name(
    t"blackwell_warp_specialized_blockwise_fp8_{a_type}_{b_type}_{c_type}_{transpose_b}_s{num_pipeline_stages}",
)
def blackwell_tma_umma_warp_specialized_blockwise_fp8_kernel[
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
    a_scales_rank: Int,
    a_scales_tile_shape: IndexList[a_scales_rank],
    a_scales_desc_shape: IndexList[a_scales_rank],
    a_scales_type: DType,
    b_scales_type: DType,
    b_scales_layout: TensorLayout,
    b_scales_engine: TensorEngine,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    num_pipeline_stages: Int,
    cluster_shape: StaticTuple[Int32, 3],
](
    a_tma_op: TMATensorTile[a_type, a_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[b_type, b_rank, b_tile_shape, b_desc_shape],
    c_tma_op: TMATensorTile[c_type, c_rank, c_tile_shape, c_desc_shape],
    a_scales_tma_op: TMATensorTile[
        a_scales_type, a_scales_rank, a_scales_tile_shape, a_scales_desc_shape
    ],
    cluster_dim: StaticTuple[Int32, 3],
    num_iters: Int32,
    b_scales: TileTensor[
        b_scales_type, b_scales_layout, ImmutAnyOrigin, Engine=b_scales_engine
    ],
    problem_shape: StaticTuple[Int32, 3],
):
    """Implements the warp-specialized blockwise FP8 GEMM kernel for SM100 using TMA loads, UMMA tensor-core MMA, and a CLC tile scheduler.

    Parameters:
        a_type: FP8 element type of the A operand matrix.
        b_type: FP8 element type of the B operand matrix.
        c_type: Element type of the C output matrix.
        a_rank: Number of dimensions in the A TMA descriptor.
        a_tile_shape: Shared-memory tile shape for each A TMA load.
        a_desc_shape: Copy box shape governing bytes per A TMA load.
        b_rank: Number of dimensions in the B TMA descriptor.
        b_tile_shape: Shared-memory tile shape for each B TMA load.
        b_desc_shape: Copy box shape governing bytes per B TMA load.
        c_rank: Number of dimensions in the C TMA descriptor.
        c_tile_shape: Shared-memory tile shape for each C TMA store.
        c_desc_shape: Copy box shape governing bytes per C TMA store.
        a_scales_rank: Number of dimensions in the A scales TMA descriptor.
        a_scales_tile_shape: Shared-memory tile shape for each A scales TMA
            load.
        a_scales_desc_shape: Copy box shape governing bytes per A scales TMA
            load.
        a_scales_type: Element type of the A blockwise scales (`float32`).
        b_scales_type: Element type of the B blockwise scales (`float32`).
        b_scales_layout: Memory layout of the B scales `TileTensor`.
        b_scales_engine: Engine of the B scales `TileTensor`.
        transpose_b: Whether B is k-major (transposed); must be `True`.
        config: Static GEMM configuration holding tile, MMA, cluster,
            pipeline, and swizzle parameters.
        num_pipeline_stages: Number of double-buffered stages for A, B, and
            A scales TMA loads.
        cluster_shape: Thread cluster shape `(x, y, z)` for the launch.

    Args:
        a_tma_op: TMA descriptor for async multicast loads of A tiles from
            global memory.
        b_tma_op: TMA descriptor for async multicast loads of B tiles from
            global memory.
        c_tma_op: TMA descriptor for async stores of C tiles to global
            memory.
        a_scales_tma_op: TMA descriptor for async loads of A blockwise
            scales from global memory.
        cluster_dim: Number of clusters along each grid axis `(x, y, z)`.
        num_iters: Number of K-dimension iterations, equal to `ceil(K, BK)`.
        b_scales: B blockwise scales used to rescale FP8 partial products.
        problem_shape: Full GEMM problem dimensions `(M, N, K)`.
    """
    var _num_iters = Int(num_iters)
    comptime num_output_warps = 4

    comptime accum_type = get_accum_type[a_type]()

    comptime assert (
        b_scales_type == a_scales_type and accum_type == .float32
    ), "Only support float32 for a_scales and b_scales"
    comptime assert transpose_b, "only support k-major B"

    comptime SCHEDULER_THREADS = WARP_SIZE
    comptime TMA_LOAD_THREADS = WARP_SIZE
    comptime MMA_THREADS = WARP_SIZE
    comptime EPILOGUE_THREADS = num_output_warps * WARP_SIZE
    comptime CLUSTER_SIZE = config.cluster_shape[0] * config.cluster_shape[1]
    comptime clc_producer_arv_count = 1
    comptime clc_consumer_arv_count = SCHEDULER_THREADS + CLUSTER_SIZE * (
        TMA_LOAD_THREADS + MMA_THREADS + EPILOGUE_THREADS
    )

    # For ld from TMEM, use same per-stage stride in column field.
    comptime NUM_TMEM_COLS = 512
    comptime stage_stride_cols = NUM_TMEM_COLS // config.num_accum_pipeline_stages

    comptime clc_throttle_producer_arv_count = TMA_LOAD_THREADS
    comptime clc_throttle_consumer_arv_count = SCHEDULER_THREADS

    comptime accum_pipeline_producer_arv_count = 1
    comptime accum_pipeline_consumer_arv_count = config.cta_group * EPILOGUE_THREADS

    comptime BM = config.block_tile_shape[0]
    comptime BN = config.block_tile_shape[1]
    comptime BK = config.block_tile_shape[2]
    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    comptime assert BK in (64, 128), "Only support BK in (64, 128)"
    comptime assert MMA_N <= BK or gcd(MMA_N, BK) == MMA_N - BK, (
        "MMA_N <= BK or gcd(MMA_N, BK) == MMA_N - BK. MMA_N="
        + String(MMA_N)
        + ", GCD="
        + String(gcd(MMA_N, BK))
    )

    comptime num_m_mmas = BM // (config.mma_shape[0] // config.cta_group)
    comptime num_n_mmas = BN // (config.mma_shape[1] // config.cta_group)
    comptime num_k_mmas = BK // config.mma_shape[2]

    comptime CLUSTER_M: Int = config.cluster_shape[0]
    comptime CLUSTER_N: Int = config.cluster_shape[1]

    comptime a_tma_load_size = _idx_product[a_rank, a_desc_shape]()
    comptime b_tma_load_size = _idx_product[b_rank, b_desc_shape]()
    comptime a_tma_rows = a_desc_shape[0]
    comptime b_tma_rows = b_desc_shape[0]
    comptime c_smem_layout = Layout.row_major(BM, MMA_N)

    comptime a_scales_smem_layout = Layout.row_major(1, BM)

    var base_ptr_smem = external_memory[
        Scalar[a_type],
        address_space=.SHARED,
        alignment=128,
    ]()

    comptime a_smem_size = tile_layout_k_major_typed[
        a_type, BM, BK, swizzle_mode=config.a_swizzle
    ].static_product * num_pipeline_stages
    comptime b_smem_size = tile_layout_k_major_typed[
        b_type, BN, BK, swizzle_mode=config.b_swizzle
    ].static_product * num_pipeline_stages
    comptime c_smem_size = config.output_tile_shape[
        0
    ] * config.output_tile_shape[1] * config.num_output_stages

    comptime a_scales_smem_size = a_scales_smem_layout.size() * num_pipeline_stages

    var a_smem_base = base_ptr_smem
    var b_smem_base = (a_smem_base + a_smem_size).bitcast[Scalar[b_type]]()
    var c_smem_base = (b_smem_base + b_smem_size).bitcast[Scalar[c_type]]()
    var a_scales_smem_base = (c_smem_base + c_smem_size).bitcast[
        Scalar[a_scales_type]
    ]()

    # TileTensor views of shared memory for both TMA producer and MMA consumer.
    # The size guards make the TileTensor array boundary explicit: pointer
    # partitioning still follows tile_layout_k_major_typed, while consumers use
    # the matching SMemTileArray2D TileTensor view.
    comptime ASMemTiles = SMemTileArray2D[
        a_type,
        BM,
        BK,
        num_pipeline_stages,
        swizzle_mode_to_bytes[config.a_swizzle],
    ]
    comptime BSMemTiles = SMemTileArray2D[
        b_type,
        BN,
        BK,
        num_pipeline_stages,
        swizzle_mode_to_bytes[config.b_swizzle],
    ]
    comptime assert (
        ASMemTiles.tile_size * num_pipeline_stages == a_smem_size
    ), "A SMEM tile array size mismatch"
    comptime assert (
        BSMemTiles.tile_size * num_pipeline_stages == b_smem_size
    ), "B SMEM tile array size mismatch"
    var a_smem_tt = ASMemTiles(a_smem_base)
    var b_smem_tt = BSMemTiles(b_smem_base)

    var c_tiles = SMemTileArray2DRowMajor[
        c_type,
        config.output_tile_shape[0],
        config.output_tile_shape[1],
        config.num_output_stages,
        128,
    ](c_smem_base)

    var a_scales_smem = SMemTileArray2DRowMajor[
        a_scales_type,
        1,
        BM,
        num_pipeline_stages,
        128,
    ](a_scales_smem_base)
    var load_mma_mbar_ptr = (a_scales_smem_base + a_scales_smem_size).bitcast[
        SharedMemBarrier
    ]()

    # Load warp as producer and mma warp as consumer
    var load_mma_pipeline = ProducerConsumerPipeline[num_pipeline_stages](
        load_mma_mbar_ptr
    )

    var mma_output_mbar_ptr = load_mma_mbar_ptr + 2 * num_pipeline_stages
    var mma_output_pipeline = ProducerConsumerPipeline[
        config.num_accum_pipeline_stages
    ](mma_output_mbar_ptr)

    var clc_full_mbar_ptr = (
        mma_output_mbar_ptr + 2 * config.num_accum_pipeline_stages
    )
    var clc_empty_mbar_ptr = clc_full_mbar_ptr + config.num_clc_pipeline_stages

    # Load warp as producer and scheduler warp as consumer.
    # No data dependence. Introduce dependence to prevent CLC goes too ahead.
    # In the extreme case, all ctas keep querying next work simultaneously,
    # there will be no guarantee they get balanced number of tiles.
    var load_clc_pipeline = ProducerConsumerPipeline[
        config.num_clc_pipeline_stages
    ](clc_empty_mbar_ptr + config.num_clc_pipeline_stages)

    var clc_response_ptr = (
        clc_empty_mbar_ptr + 3 * config.num_clc_pipeline_stages
    ).bitcast[Int128]()

    var tmem_dealloc_mbar_ptr = (
        clc_response_ptr + config.num_clc_pipeline_stages
    ).bitcast[Int64]()

    var ptr_tmem_addr = (tmem_dealloc_mbar_ptr + 1).bitcast[UInt32]()

    var clc_response = clc_response_ptr.bitcast[UInt128]()
    var clc_full_mbar = clc_full_mbar_ptr.bitcast[SharedMemBarrier]()
    var clc_empty_mbar = clc_empty_mbar_ptr.bitcast[SharedMemBarrier]()
    var tmem_dealloc_mbar = tmem_dealloc_mbar_ptr.bitcast[SharedMemBarrier]()

    var warp_id = get_warp_id()
    var elect_one_warp = warp_id == 0
    var elect_one_thread = elect_one_sync_with_mask()
    var elect_one_cta = (
        block_rank_in_cluster() % 2 == 0 if config.cta_group == 2 else True
    )
    var is_first_cta_in_cluster = block_rank_in_cluster() == 0
    comptime max_tmem_cols = 512

    if elect_one_warp and elect_one_thread:
        a_tma_op.prefetch_descriptor()
        b_tma_op.prefetch_descriptor()
        c_tma_op.prefetch_descriptor()
        a_scales_tma_op.prefetch_descriptor()

        load_mma_pipeline.init_mbars(
            Int32(1),
            Int32(
                config.cluster_shape[0] // config.cta_group
                + config.cluster_shape[1]
                - 1
                + CLUSTER_SIZE * (EPILOGUE_THREADS // 32)
            ),
        )

        mma_output_pipeline.init_mbars(
            accum_pipeline_producer_arv_count,
            Int32(accum_pipeline_consumer_arv_count),
        )
        load_clc_pipeline.init_mbars(
            Int32(clc_throttle_producer_arv_count),
            Int32(clc_throttle_consumer_arv_count),
        )

        tmem_dealloc_mbar[].init(Int32(EPILOGUE_THREADS * config.cta_group))

    comptime for i in range(config.num_clc_pipeline_stages):
        clc_full_mbar[i].init(clc_producer_arv_count)
        clc_empty_mbar[i].init(Int32(clc_consumer_arv_count))

    fence_mbarrier_init()
    cluster_sync()

    var clc_pipe_producer_state = PipelineState[config.num_clc_pipeline_stages](
        0, 1, 0
    )
    var clc_pipe_consumer_state = PipelineState[
        config.num_clc_pipeline_stages
    ]()

    var mma_op = MmaOpSM100_SS[
        c_type,
        a_type,
        b_type,
        config.block_tile_shape,
        config.mma_shape,
        accum_type=accum_type,
        cta_group=config.cta_group,
        cluster_shape=config.cluster_shape,
        a_swizzle=config.a_swizzle,
        b_swizzle=config.b_swizzle,
        transpose_b=transpose_b,
    ]()

    var scheduler = TileScheduler[
        num_stages=config.num_clc_pipeline_stages,
        cluster_shape=Index[dtype=DType.uint32](
            config.cluster_shape[0],
            config.cluster_shape[1],
            config.cluster_shape[2],
        ),
        block_swizzle_size=config.block_swizzle_size,
        rasterize_order=config.raster_order,
    ](cluster_dim, clc_response, clc_full_mbar, clc_empty_mbar)

    var work_info = scheduler.initial_work_info()

    var rank_m = block_id_in_cluster.x
    var rank_n = block_id_in_cluster.y

    # (peer_id, mma_coord_m, mma_coord_n)
    var peer_cta_coord = (
        umod(rank_m, config.cta_group),
        ufloordiv(rank_m, config.cta_group),
        rank_n,
    )  # v,m,n

    var a_multicast_mask: UInt16 = 0x0
    var b_multicast_mask: UInt16 = 0x0

    # TODO: find a generic way to calculate multicast mask
    comptime for i in range(CLUSTER_N):
        a_multicast_mask |= UInt16(1 << (i * CLUSTER_M))
    # they all have the same v and m, but different n,

    comptime for i in range(CLUSTER_M // config.cta_group):
        b_multicast_mask |= UInt16(1 << (i * config.cta_group))

    a_multicast_mask <<= UInt16(rank_m)
    b_multicast_mask <<= UInt16(peer_cta_coord[0])
    b_multicast_mask <<= UInt16(rank_n * CLUSTER_M)

    var self_mask = 1 << Int(block_rank_in_cluster())
    var peer_mask = 1 << Int(block_rank_in_cluster() + 1)
    var mma_complete_mask = self_mask | peer_mask

    if WarpRole.is_main_load():
        var required_clc_query = True

        while work_info.is_valid():
            # CLC throuttle prevents each CTA from going a few waves ahead.
            if is_first_cta_in_cluster and required_clc_query:
                load_clc_pipeline.wait_consumer()
                var load_clc_producer_state = load_clc_pipeline.producer_stage()
                _ = load_clc_pipeline.producer_mbar(load_clc_producer_state)[
                    0
                ].arrive()
                load_clc_pipeline.producer_step()

            # DO TMA LOAD

            for i in range(_num_iters):
                load_AB[
                    block_tile_shape=config.block_tile_shape,
                    mma_shape=config.mma_shape,
                    cta_group=config.cta_group,
                ](
                    a_tma_op,
                    b_tma_op,
                    a_scales_tma_op,
                    a_smem_tt,
                    b_smem_tt,
                    a_scales_smem,
                    load_mma_pipeline,
                    peer_cta_coord,
                    (Int(work_info.m), Int(work_info.n)),
                    a_multicast_mask,
                    b_multicast_mask,
                    i,
                    elect_one_cta,
                )
                load_mma_pipeline.producer_step()

            syncwarp()
            var next_work_info = scheduler.fetch_next_work(
                work_info, clc_pipe_consumer_state
            )
            work_info = next_work_info
            clc_pipe_consumer_state.step()

        comptime for i in range(num_pipeline_stages):
            load_mma_pipeline.wait_consumer()
            load_mma_pipeline.producer_step()

    if WarpRole.is_scheduler() and is_first_cta_in_cluster:
        var required_clc_query = True

        while work_info.is_valid():
            if required_clc_query:
                load_clc_pipeline.wait_producer()
                var load_clc_consumer_stage = load_clc_pipeline.consumer_stage()
                _ = load_clc_pipeline.consumer_mbar(load_clc_consumer_stage)[
                    0
                ].arrive()
                load_clc_pipeline.consumer_step()

                # advance to next work
                clc_pipe_producer_state = scheduler.advance_to_next_work(
                    clc_pipe_producer_state
                )

            # scheduler fetch next work
            var next_work_info = scheduler.fetch_next_work(
                work_info, clc_pipe_consumer_state
            )

            work_info = next_work_info
            clc_pipe_consumer_state.step()

        # make sure all pipes are empty before kernel exit
        comptime for i in range(config.num_clc_pipeline_stages):
            clc_empty_mbar[clc_pipe_producer_state.index()].wait(
                clc_pipe_producer_state.phase()
            )
            clc_pipe_producer_state.step()

    if WarpRole.is_mma():
        tcgen05_alloc[Int32(config.cta_group)](ptr_tmem_addr, max_tmem_cols)
        syncwarp()
        # non blocking, arrives and proceeds
        named_barrier_arrive[Int32(MMA_THREADS + EPILOGUE_THREADS)](1)

        var tmem_addr = ptr_tmem_addr[0]

        while work_info.is_valid():
            # scheduler fetch next work
            var next_work_info = scheduler.fetch_next_work(
                work_info, clc_pipe_consumer_state
            )
            clc_pipe_consumer_state.step()
            # DO MMA
            if elect_one_cta:
                for _ in range(_num_iters):
                    var mma_output_mma_stage = (
                        mma_output_pipeline.producer_stage()
                    )
                    mma_output_pipeline.wait_consumer()
                    var tmem_offset = tmem_addr + (
                        mma_output_mma_stage * UInt32(stage_stride_cols)
                    )

                    consumer_main_loop[
                        block_tile_shape=config.block_tile_shape,
                        mma_shape=config.mma_shape,
                        cta_group=config.cta_group,
                        cluster_shape=config.cluster_shape,
                    ](
                        tmem_offset,
                        a_smem_tt,
                        b_smem_tt,
                        load_mma_pipeline,
                        mma_op,
                        elect_one_warp,
                        0,
                        0,
                    )
                    load_mma_pipeline.consumer_step()

                    # mma arrive multicast will track completion of all mma prior to this barrier.
                    if elect_one_sync():
                        comptime if config.cta_group == 1:
                            mma_arrive[config.cta_group](
                                mma_output_pipeline.producer_mbar(
                                    mma_output_mma_stage
                                )
                            )
                        else:
                            mma_arrive_multicast[config.cta_group](
                                mma_output_pipeline.producer_mbar(
                                    mma_output_mma_stage
                                ),
                                UInt16(mma_complete_mask),
                            )
                    mma_output_pipeline.producer_step()

            work_info = next_work_info

        tcgen05_release_allocation_lock[Int32(config.cta_group)]()

        # wait for epilogue to finish
        tmem_dealloc_mbar[].wait()

        tcgen05_dealloc[Int32(config.cta_group)](tmem_addr, max_tmem_cols)

    if WarpRole.is_epilogue():
        named_barrier[Int32(MMA_THREADS + EPILOGUE_THREADS)](1)
        var tmem_addr = ptr_tmem_addr[0]

        while work_info.is_valid():
            comptime reg_info = _get_accumulator_size[
                c_smem_layout=Layout.row_major(
                    config.output_tile_shape[0], config.output_tile_shape[1]
                ),
                block_tile_shape=config.block_tile_shape,
                mma_shape=config.mma_shape,
                cta_group=config.cta_group,
            ]()

            comptime is_lower_frag_required = not (
                config.cta_group == 1 and config.block_tile_shape[0] == 64
            )
            # final results accumulator regs for C
            var c_upper_main_tile = stack_allocation[
                dtype=accum_type, address_space=.LOCAL
            ](row_major[reg_info[0], reg_info[1]]())

            var c_lower_main_tile = stack_allocation[
                dtype=accum_type, address_space=.LOCAL
            ](row_major[reg_info[0], reg_info[1]]())

            _ = c_upper_main_tile.fill(0.0)

            comptime if is_lower_frag_required:
                _ = c_lower_main_tile.fill(0.0)

            for k_iter in range(_num_iters):
                promote_accumulators[
                    block_tile_shape=config.block_tile_shape,
                    mma_shape=config.mma_shape,
                    cta_group=config.cta_group,
                    CLUSTER_SIZE=Int32(CLUSTER_SIZE),
                    is_lower_frag_required=is_lower_frag_required,
                    num_output_warps=num_output_warps,
                ](
                    b_scales,
                    a_scales_smem,
                    c_upper_main_tile,
                    c_lower_main_tile,
                    # accum_pipeline_consumer_state,
                    mma_output_pipeline,
                    tmem_addr,
                    load_mma_pipeline,
                    work_tile_coord=(Int(work_info.m), Int(work_info.n)),
                    elect_one_warp=elect_one_warp,
                    stage_stride_cols=stage_stride_cols,
                    k_iter=k_iter,
                    problem_shape=problem_shape,
                )
                load_mma_pipeline.consumer_step()
                mma_output_pipeline.consumer_step()

            # TODO (KERN-2081): investigate why this barrier is needed and if we can move/remove it
            named_barrier[Int32(num_output_warps * WARP_SIZE)]()

            # wait for CUDA core promotion to finish and store result
            # scheduler fetch next work
            multi_stage_reg_epilogue[
                block_tile_shape=config.block_tile_shape,
                mma_shape=config.mma_shape,
                is_lower_frag_required=is_lower_frag_required,
                cta_group=config.cta_group,
                num_output_warps=num_output_warps,
                c_swizzle=config.c_swizzle,
            ](
                c_upper_main_tile,
                c_lower_main_tile,
                c_tiles,
                c_tma_op,
                c_coord=(Int(work_info.m), Int(work_info.n)),
                elect_one_warp=elect_one_warp,
            )

            var next_work_info = scheduler.fetch_next_work(
                work_info, clc_pipe_consumer_state
            )
            work_info = next_work_info
            clc_pipe_consumer_state.step()

        comptime if config.cta_group == 2:
            _ = tmem_dealloc_mbar[].arrive_cluster(block_rank_in_cluster() ^ 1)
        _ = tmem_dealloc_mbar[].arrive()


def sm100_warp_specialized_blockwise_fp8[
    transpose_b: Bool,
    a_scales_type: DType,
    b_scales_type: DType,
    *,
    config: MatmulConfig[_, _, _, transpose_b],
](
    c: TileTensor,
    a: TileTensor[mut=False, ...],
    b: TileTensor[mut=False, ...],
    a_scales: TileTensor[mut=False, ...],
    b_scales: TileTensor[mut=False, ...],
    ctx: DeviceContext,
) raises:
    """Sets up TMA descriptors, shared-memory layouts, and pipeline barriers, then launches the blockwise FP8 GEMM kernel on SM100.
    """
    comptime a_type = config.a_type
    comptime b_type = config.b_type
    comptime c_type = config.c_type

    comptime assert transpose_b, "Only support transposed B"

    comptime assert (
        a_type == b_type and a_type == .float8_e4m3fn
    ), "Only support float8_e4m3fn"

    comptime assert (
        a_scales_type == b_scales_type
    ), "Only support float32 for scales"

    if (Int(a_scales.dim[1]()) * size_of[a_scales_type]()) % 16 != 0:
        raise Error(
            "a_scales should be a multiple of 16 bytes on the M dimension"
        )

    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    comptime BM = MMA_M // config.cta_group
    comptime BN = MMA_N // config.cta_group
    comptime BK = config.block_tile_shape[2]

    comptime assert config.cta_group in (1, 2), "Only support cta_group == 2"
    comptime assert not config.AB_swapped, "Swapped AB is not supported"

    var M = Int(c.dim[0]())
    var N = Int(c.dim[1]())
    var K = Int(a.dim[1]())

    var a_tma_op = create_tensor_tile[
        Index(BM // config.cluster_shape[1], BK),
        swizzle_mode=config.a_swizzle,
    ](ctx, a)

    var b_tma_op = create_tensor_tile[
        Index(
            BN // (config.cluster_shape[0] // config.cta_group), BK
        ) if transpose_b else Index(
            BK, BN // (config.cluster_shape[0] // config.cta_group)
        ),
        swizzle_mode=config.b_swizzle,
    ](ctx, b)

    var a_scales_tma_op = create_tensor_tile[
        Index(1, BM),
        __desc_shape=Index(1, BM),
    ](ctx, a_scales)

    # For MMA_M=128, output tile has 128 rows and each 64 rows belongs to one c tile.
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-b
    comptime c_tma_tile_shape_mma128 = Index(64, config.output_tile_shape[1])
    comptime c_tma_tile_shape = config.output_tile_shape if (
        MMA_M == 256 or config.cta_group == 1
    ) else c_tma_tile_shape_mma128

    var c_tma_op = create_tensor_tile[
        c_tma_tile_shape,
        swizzle_mode=config.c_swizzle,
    ](ctx, c)

    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024
    comptime a_smem_bytes_per_stage = BM * BK * size_of[a_type]()
    comptime b_smem_bytes_per_stage = BN * BK * size_of[b_type]()
    comptime a_scales_smem_bytes_per_stage = BM * size_of[a_scales_type]()
    comptime AB_smem_per_stage = a_smem_bytes_per_stage + b_smem_bytes_per_stage

    comptime c_smem_bytes = config.output_tile_shape[
        0
    ] * config.output_tile_shape[1] * config.num_output_stages * size_of[
        c_type
    ]()

    comptime MBAR_BYTES = size_of[Int64]()  # 8 bytes per barrier
    comptime CLC_RESPONSE_BYTES = size_of[Int128]()  # 16 bytes per response
    comptime TMEM_ADDR_BYTES = size_of[
        Int32
    ]()  # 4 bytes or 32 bits for tensor memory address

    comptime accum_full_mbar_bytes = MBAR_BYTES * config.num_accum_pipeline_stages
    comptime accum_empty_mbar_bytes = MBAR_BYTES * config.num_accum_pipeline_stages

    comptime clc_response_bytes = CLC_RESPONSE_BYTES * config.num_clc_pipeline_stages
    comptime clc_full_mbar_bytes = MBAR_BYTES * config.num_clc_pipeline_stages
    comptime clc_empty_mbar_bytes = MBAR_BYTES * config.num_clc_pipeline_stages
    comptime clc_throttle_full_mbar_bytes = MBAR_BYTES * config.num_clc_pipeline_stages
    comptime clc_throttle_empty_mbar_bytes = MBAR_BYTES * config.num_clc_pipeline_stages

    comptime tmem_addr_bytes = TMEM_ADDR_BYTES
    comptime tmem_dealloc_mbar_bytes = MBAR_BYTES

    comptime tmem_writeout_smem = c_smem_bytes + tmem_addr_bytes + tmem_dealloc_mbar_bytes
    comptime accum_smem = accum_full_mbar_bytes + accum_empty_mbar_bytes
    comptime clc_smem = (
        clc_response_bytes
        + clc_full_mbar_bytes
        + clc_empty_mbar_bytes
        + clc_throttle_full_mbar_bytes
        + clc_throttle_empty_mbar_bytes
    )
    comptime smem_leftover = (b200_smem) - (
        clc_smem + accum_smem + tmem_writeout_smem
    )

    comptime tma_mbar_bytes_per_stage = MBAR_BYTES
    comptime mma_mbar_bytes_per_stage = MBAR_BYTES

    comptime producer_consumer_smem_per_stage = (
        AB_smem_per_stage
        + a_scales_smem_bytes_per_stage
        + tma_mbar_bytes_per_stage
        + mma_mbar_bytes_per_stage
    )

    comptime max_pipeline_stages: Int = (
        smem_leftover // producer_consumer_smem_per_stage
    )

    comptime assert (
        max_pipeline_stages >= 1
    ), "not enough smem even for one pipeline stage!"

    comptime producer_consumer_smem = producer_consumer_smem_per_stage * max_pipeline_stages

    comptime smem_size = (
        clc_smem + accum_smem + producer_consumer_smem + tmem_writeout_smem
    )

    comptime kernel = blackwell_tma_umma_warp_specialized_blockwise_fp8_kernel[
        a_type,
        b_type,
        c_type,
        type_of(a_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).rank,
        type_of(b_tma_op).tile_shape,
        type_of(b_tma_op).desc_shape,
        type_of(c_tma_op).rank,
        type_of(c_tma_op).tile_shape,
        type_of(c_tma_op).desc_shape,
        type_of(a_scales_tma_op).rank,
        type_of(a_scales_tma_op).tile_shape,
        type_of(a_scales_tma_op).desc_shape,
        a_scales_type,
        b_scales_type,
        type_of(b_scales).LayoutType,
        type_of(b_scales).Engine,
        transpose_b=transpose_b,
        config=config,
        num_pipeline_stages=max_pipeline_stages,
        cluster_shape=StaticTuple[Int32, 3](
            Int32(config.cluster_shape[0]),
            Int32(config.cluster_shape[1]),
            Int32(config.cluster_shape[2]),
        ),
    ]

    var grid_dim = (
        align_up(ceildiv(M, BM), config.cluster_shape[0]),
        align_up(ceildiv(N, MMA_N), config.cluster_shape[1]),
        1,
    )

    var cluster_dim = StaticTuple[Int32, 3](
        Int32(ceildiv(grid_dim[0], config.cluster_shape[0])),
        Int32(ceildiv(grid_dim[1], config.cluster_shape[1])),
        1,
    )

    var problem_shape = StaticTuple[Int32, 3](Int32(M), Int32(N), Int32(K))

    ctx.enqueue_function[kernel, dump_asm=False](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        a_scales_tma_op,
        cluster_dim,
        Int32(ceildiv(K, BK)),
        b_scales,
        problem_shape,
        grid_dim=grid_dim,
        # 1 TMA, 1 MMA, 1 Scheduler, 4 EPILOGUE warps
        block_dim=(32 * 7),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_size)
        ),
    )
