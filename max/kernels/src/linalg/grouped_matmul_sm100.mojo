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

"""Provides warp-specialized persistent grouped GEMM kernels for SM100 (B200) GPUs."""

from std.collections import Optional
from std.math import ceildiv
from std.math.uutils import umod, ufloordiv
from std.memory import bitcast
from std.sys import align_of, simd_width_of, size_of
from std.bit import next_power_of_two
from max.gpu import WARP_SIZE
from max.gpu.sync import barrier
from max.gpu.primitives.cluster import (
    block_rank_in_cluster,
    cluster_sync,
    elect_one_sync,
    elect_one_sync_with_mask,
)
from max.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.host.info import B200
from max.gpu import (
    block_id_in_cluster,
    thread_idx,
    lane_id,
    warp_id as get_warp_id,
)
from max.gpu.memory import (
    external_memory,
    fence_async_view_proxy,
    fence_mbarrier_init,
)
from max.gpu.compute.mma import st_matrix
from max.gpu.compute.arch.mma_nvidia_sm100 import *
from max.gpu.sync import (
    named_barrier,
    named_barrier_arrive,
    syncwarp,
    umma_arrive_leader_cta,
)
from max.gpu.compute.arch.tcgen05 import *
from layout import (
    Coord,
    Idx,
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    UNKNOWN_VALUE,
)
from layout.tile_tensor import TileTensor

from .utils import partial_simd_load
from layout.swizzle import Swizzle, make_swizzle
from layout.tile_layout import col_major as tl_col_major
from layout.tensor_core_async import tile_layout_k_major_typed
from structured_kernels.tile_types import (
    SMemTileArray2D,
    swizzle_mode_to_bytes,
)
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

from .arch.sm100 import MmaOpSM100_SS
from .utils import elementwise_epilogue_type, lora_qkv_plane_row_offset
from .utils_gpu import MatmulConfig
from .grouped_matmul_tile_scheduler import TileScheduler


@fieldwise_init
struct WarpRole(TrivialRegisterPassable):
    """Enumerates the role each warp plays in the warp-specialized kernel.

    The main-load, MMA, and epilogue warps select their code path by
    comparing the runtime warp id against these roles.
    """

    var _role: Int32

    comptime Mma = Self(5)
    comptime MainLoad = Self(4)
    comptime Epilogue = Self(3)

    @always_inline
    def __eq__(self, other: Int) -> Bool:
        return self._role == Int32(other)

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self._role == other._role

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        return self._role != other._role

    @always_inline
    def __ge__(self, other: Int) -> Bool:
        return self._role >= Int32(other)

    @staticmethod
    @always_inline
    def is_main_load() -> Bool:
        return Self.MainLoad == get_warp_id()

    @staticmethod
    @always_inline
    def is_mma() -> Bool:
        return Self.Mma == get_warp_id()

    @staticmethod
    @always_inline
    def is_epilogue() -> Bool:
        return Self.Epilogue >= get_warp_id()


@always_inline
def load_AB[
    a_type: DType,
    b_type: DType,
    a_tile_rank: Int,
    a_tile_shape: IndexList[a_tile_rank],
    a_desc_shape: IndexList[a_tile_rank],
    b_tile_rank: Int,
    b_tile_shape: IndexList[b_tile_rank],
    b_desc_shape: IndexList[b_tile_rank],
    a_dim0: Int,
    a_dim1: Int,
    a_num_tiles: Int,
    a_swizzle_bytes: Int,
    b_dim0: Int,
    b_dim1: Int,
    b_num_tiles: Int,
    b_swizzle_bytes: Int,
    num_pipeline_stages: Int,
    //,
    *,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int = 1,
    a_plane_splits: IndexList[2] = Index(0, 0),
](
    expert_ids: UnsafePointer[mut=False, Int32, _],
    a_tma_op: TMATensorTile[a_type, a_tile_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[b_type, b_tile_rank, b_tile_shape, b_desc_shape],
    a_smem_tiles: SMemTileArray2D[
        a_type, a_dim0, a_dim1, a_num_tiles, a_swizzle_bytes
    ],
    b_smem_tiles: SMemTileArray2D[
        b_type, b_dim0, b_dim1, b_num_tiles, b_swizzle_bytes
    ],
    mma_mbar: UnsafePointer[
        mut=True, SharedMemBarrier, _, address_space=.SHARED
    ],
    tma_mbar: UnsafePointer[
        mut=True, SharedMemBarrier, _, address_space=.SHARED
    ],
    producer_phase: PipelineState[num_pipeline_stages],
    peer_cta_coord: Tuple[Int, Int, Int],
    work_tile_coord: Tuple[Int, Int],
    a_multicast_mask: UInt16,
    b_multicast_mask: UInt16,
    iter_idx: UInt32,
    elect_one_cta: Bool,
    scheduler: TileScheduler,
    qkv_plane_stride: Int = 0,
):
    """Loads A and B tiles from global memory into shared memory via TMA multicast.

    Issues asynchronous multicast TMA loads for the current pipeline stage,
    addressing the expert-local slice of A and the shared B slice, then
    signals completion through the TMA mbarrier.

    Parameters:
        a_type: Element type of the A operand tiles (inferred).
        b_type: Element type of the B operand tiles (inferred).
        a_tile_rank: Rank of the A TMA tile descriptor (inferred).
        a_tile_shape: Element shape of one A TMA tile (inferred).
        a_desc_shape: Descriptor shape of the A TMA tile, used to compute
            per-load element counts and row stride (inferred).
        b_tile_rank: Rank of the B TMA tile descriptor (inferred).
        b_tile_shape: Element shape of one B TMA tile (inferred).
        b_desc_shape: Descriptor shape of the B TMA tile, used to compute
            per-load element counts and row stride (inferred).
        a_dim0: Number of rows in one A shared-memory tile (inferred).
        a_dim1: Number of columns in one A shared-memory tile (inferred).
        a_num_tiles: Number of pipeline stages in the A shared-memory
            tile array (inferred).
        a_swizzle_bytes: Swizzle granularity in bytes for A shared-memory
            tiles (inferred).
        b_dim0: Number of rows in one B shared-memory tile (inferred).
        b_dim1: Number of columns in one B shared-memory tile (inferred).
        b_num_tiles: Number of pipeline stages in the B shared-memory
            tile array (inferred).
        b_swizzle_bytes: Swizzle granularity in bytes for B shared-memory
            tiles (inferred).
        num_pipeline_stages: Number of double-buffered pipeline stages
            for A and B shared-memory tiles (inferred).
        block_tile_shape: Block tile shape `[BM, BN, BK]` partitioning
            the GEMM into work tiles.
        mma_shape: MMA instruction shape `[MMA_M, MMA_N, MMA_K]`.
        cta_group: Number of CTAs cooperating per MMA along the M
            dimension (defaults to 1).
        a_plane_splits: Per-plane split sizes for fused LoRA QKV A-plane
            row offsetting; `(0, 0)` disables it (defaults to `(0, 0)`).

    Args:
        expert_ids: Pointer to the per-group expert id array; used to
            offset the A global-memory slice by the expert's local M base.
        a_tma_op: TMA tile descriptor for loading A tiles from global
            to shared memory.
        b_tma_op: TMA tile descriptor for loading B tiles from global
            to shared memory.
        a_smem_tiles: Shared-memory tile array of staged A tiles, one
            per pipeline stage.
        b_smem_tiles: Shared-memory tile array of staged B tiles, one
            per pipeline stage.
        mma_mbar: Pointer to the MMA mbarrier array, one per pipeline
            stage, waited on to confirm the consumer has freed the prior
            stage's shared memory.
        tma_mbar: Pointer to the TMA mbarrier array, one per pipeline
            stage, signaled when the TMA load for a stage is complete.
        producer_phase: Producer pipeline state tracking the stage index
            and phase bit.
        peer_cta_coord: `(peer_id, mma_coord_m, mma_coord_n)` tuple
            giving the peer CTA's coordinates within the cluster.
        work_tile_coord: `(m, n)` element coordinates of the current
            work tile returned by the scheduler.
        a_multicast_mask: Bitmask of CTAs participating in the A TMA
            multicast load.
        b_multicast_mask: Bitmask of CTAs participating in the B TMA
            multicast load.
        iter_idx: Current K-dimension iteration index within the work
            tile.
        elect_one_cta: Whether this CTA is the elected leader that sets
            the expected bytes on the TMA mbarrier.
        scheduler: Tile scheduler providing the current group index and
            static M/N bounds.
        qkv_plane_stride: Row stride between fused QKV planes used to
            compute the A-plane row offset (defaults to 0).
    """
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]

    comptime a_expected_bytes = a_dim0 * a_dim1 * size_of[a_type]()
    comptime b_expected_bytes = b_dim0 * b_dim1 * size_of[b_type]()
    # Leader CTAs expect SMEM from itself and their peers
    comptime expected_bytes = cta_group * (a_expected_bytes + b_expected_bytes)

    comptime a_tma_load_size = _idx_product[a_tile_rank, a_desc_shape]()
    comptime b_tma_load_size = _idx_product[b_tile_rank, b_desc_shape]()
    comptime a_tma_rows = a_desc_shape[0]
    comptime b_tma_rows = b_desc_shape[0]

    var stage = producer_phase.index()
    var phase = producer_phase.phase()
    mma_mbar[stage].wait(phase)

    var a_gmem_slice_coord = Int32(
        peer_cta_coord[2] * a_tma_rows + work_tile_coord[0]
    ) + expert_ids[Int(scheduler.current_group_idx)] * Int32(
        scheduler.static_MN
    )
    var b_gmem_slice_coord = (
        peer_cta_coord[1] * b_tma_rows
        + peer_cta_coord[0] * BN
        + work_tile_coord[1]
    )

    # Activation-operand (post-swapAB `b`) plane select: shift the gmem row by a
    # function of the output-column tile base (`work_tile_coord[0]`, the kernel's
    # M dim after swapAB) and the per-plane row stride. Lets one launch read a
    # different slice of the activations per output-column region (LoRA-B QKV
    # expand: pick the Q/K/V plane of the planar shrink output). The descriptor
    # must cover the full activation extent (see `b_desc_rows` in the launcher)
    # so the shifted rows are in bounds. `a_plane_splits == (0, 0)` disables it.
    b_gmem_slice_coord += lora_qkv_plane_row_offset[a_plane_splits](
        work_tile_coord[0], qkv_plane_stride
    )

    var a_smem_tile = a_smem_tiles[stage]
    var b_smem_tile = b_smem_tiles[stage]

    var a_smem_slice = type_of(a_smem_tile)(
        a_smem_tile.ptr + peer_cta_coord[2] * a_tma_load_size,
        a_smem_tile.layout,
    )
    var b_smem_slice = type_of(b_smem_tile)(
        b_smem_tile.ptr + peer_cta_coord[1] * b_tma_load_size,
        b_smem_tile.layout,
    )

    if elect_one_sync():
        if elect_one_cta:
            tma_mbar[stage].expect_bytes(Int32(expected_bytes))

        a_tma_op.async_multicast_load[cta_group](
            a_smem_slice,
            tma_mbar[stage],
            (Int(iter_idx) * BK, Int(a_gmem_slice_coord)),
            a_multicast_mask,
        )

        b_tma_op.async_multicast_load[cta_group](
            b_smem_slice,
            tma_mbar[stage],
            (Int(iter_idx) * BK, b_gmem_slice_coord),
            b_multicast_mask,
        )


@always_inline
def load_AB_cuda_core[
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
    num_pipeline_stages: Int,
    //,
    *,
    K_actual: Int,
    cta_group: Int = 1,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_32B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_32B,
    a_gmem_layout: Layout = Layout.row_major(1, 1),
    b_gmem_layout: Layout = Layout.row_major(1, 1),
    a_plane_splits: IndexList[2] = Index(0, 0),
](
    a_gmem: LayoutTensor[a_type, a_gmem_layout, ImmutAnyOrigin],
    b_gmem: LayoutTensor[b_type, b_gmem_layout, ImmutAnyOrigin],
    expert_ids: UnsafePointer[mut=False, Int32, _],
    a_smem_tiles: SMemTileArray2D[
        a_type, a_dim0, a_dim1, a_num_tiles, a_swizzle_bytes
    ],
    b_smem_tiles: SMemTileArray2D[
        b_type, b_dim0, b_dim1, b_num_tiles, b_swizzle_bytes
    ],
    mma_mbar: UnsafePointer[
        mut=True, SharedMemBarrier, _, address_space=.SHARED
    ],
    tma_mbar: UnsafePointer[
        mut=True, SharedMemBarrier, _, address_space=.SHARED
    ],
    producer_phase: PipelineState[num_pipeline_stages],
    peer_cta_coord: Tuple[Int, Int, Int],
    work_tile_coord: Tuple[Int, Int],
    iter_idx: UInt32,
    scheduler: TileScheduler,
    qkv_plane_stride: Int = 0,
):
    """CUDA core fallback for load_AB when K*sizeof < 16 bytes.

    Copies [BM, BK] and [BN, BK] tiles from gmem LayoutTensors into
    swizzled smem, zero-filling columns where k >= K_actual.

    Parameters:
        a_type: Element type of the A operand tiles (inferred).
        b_type: Element type of the B operand tiles (inferred).
        a_dim0: Number of rows in one A shared-memory tile (inferred).
        a_dim1: Number of columns in one A shared-memory tile (inferred).
        a_num_tiles: Number of pipeline stages in the A shared-memory
            tile array (inferred).
        a_swizzle_bytes: Swizzle granularity in bytes for A shared-memory
            tiles (inferred).
        b_dim0: Number of rows in one B shared-memory tile (inferred).
        b_dim1: Number of columns in one B shared-memory tile (inferred).
        b_num_tiles: Number of pipeline stages in the B shared-memory
            tile array (inferred).
        b_swizzle_bytes: Swizzle granularity in bytes for B shared-memory
            tiles (inferred).
        num_pipeline_stages: Number of double-buffered pipeline stages
            for A and B shared-memory tiles (inferred).
        K_actual: Actual K dimension in elements; columns where
            `k >= K_actual` are zero-filled.
        cta_group: Number of CTAs cooperating per MMA along the M
            dimension (defaults to 1).
        a_swizzle: TMA swizzle mode applied to A shared-memory tiles
            (defaults to `SWIZZLE_32B`).
        b_swizzle: TMA swizzle mode applied to B shared-memory tiles
            (defaults to `SWIZZLE_32B`).
        a_gmem_layout: Layout of the A global-memory tensor (defaults to
            `Layout.row_major(1, 1)`).
        b_gmem_layout: Layout of the B global-memory tensor (defaults to
            `Layout.row_major(1, 1)`).
        a_plane_splits: Per-plane split sizes for fused LoRA QKV A-plane
            row offsetting; `(0, 0)` disables it (defaults to `(0, 0)`).

    Args:
        a_gmem: A operand tensor in global memory, source of the
            `[BM, BK]` A tiles.
        b_gmem: B operand tensor in global memory, source of the
            `[BN, BK]` B tiles.
        expert_ids: Pointer to the per-group expert id array; used to
            offset the A global-memory slice by the expert's local M base.
        a_smem_tiles: Shared-memory tile array of staged A tiles, one
            per pipeline stage.
        b_smem_tiles: Shared-memory tile array of staged B tiles, one
            per pipeline stage.
        mma_mbar: Pointer to the MMA mbarrier array, one per pipeline
            stage, waited on to confirm the consumer has freed the prior
            stage's shared memory.
        tma_mbar: Pointer to the TMA mbarrier array, one per pipeline
            stage, signaled when the copy for a stage is complete.
        producer_phase: Producer pipeline state tracking the stage index
            and phase bit.
        peer_cta_coord: `(peer_id, mma_coord_m, mma_coord_n)` tuple
            giving the peer CTA's coordinates within the cluster.
        work_tile_coord: `(m, n)` element coordinates of the current
            work tile returned by the scheduler.
        iter_idx: Current K-dimension iteration index within the work
            tile.
        scheduler: Tile scheduler providing the current group index and
            static M/N bounds.
        qkv_plane_stride: Row stride between fused QKV planes used to
            compute the A-plane row offset (defaults to 0).
    """
    comptime BM = a_dim0
    comptime BN = b_dim0
    comptime BK = a_dim1

    var stage = producer_phase.index()
    mma_mbar[stage].wait(producer_phase.phase())

    # Tile base coordinates (Int32, same as TMA path).
    var a_row0 = Int32(
        peer_cta_coord[2] * BM + work_tile_coord[0]
    ) + expert_ids[Int(scheduler.current_group_idx)] * Int32(
        scheduler.static_MN
    )
    var a_col0 = Int32(iter_idx) * Int32(BK)

    var b_row0 = Int32(
        peer_cta_coord[1] * (BN // cta_group)
        + peer_cta_coord[0] * BN
        + work_tile_coord[1]
    )
    # Activation-operand plane select (see `load_AB`).
    b_row0 += Int32(
        lora_qkv_plane_row_offset[a_plane_splits](
            work_tile_coord[0], qkv_plane_stride
        )
    )
    var b_col0 = Int32(iter_idx) * Int32(BK)

    var tid = Int32(lane_id())

    # TV layout maps threads to rows for vectorized gmem loads.
    # col_major(WARP_SIZE, rows_per_thread) gives row_idx = tid + v * WARP_SIZE
    # (coalesced across threads). Per row: vector-load K_actual elements,
    # scatter-store BK to swizzled smem with zero-padding.

    # Row alignment for vectorized gmem loads: largest power of 2
    # dividing K_actual * sizeof(dtype).
    comptime a_row_align = (K_actual * size_of[a_type]()) & (
        -(K_actual * size_of[a_type]())
    )
    comptime b_row_align = (K_actual * size_of[b_type]()) & (
        -(K_actual * size_of[b_type]())
    )

    # A tile [BM, BK]
    comptime a_sw = make_swizzle[a_type, a_swizzle]()
    var a_smem_ptr = a_smem_tiles[stage].ptr
    comptime a_rows_per_thread = BM // WARP_SIZE
    comptime a_tv = tl_col_major(Coord(Idx[WARP_SIZE], Idx[a_rows_per_thread]))

    # SIMD widths must be powers of two: odd K rows are gathered with a
    # masked load into the next power-of-two width, whose padded lanes are
    # never read by the scatter below.
    comptime K_padded = next_power_of_two(K_actual)

    comptime for v in range(a_rows_per_thread):
        var m = a_tv[linear_idx_type=DType.int32](Coord(Int32(tid), Idx[v]))
        var vec: SIMD[a_type, K_padded]
        comptime if K_actual == K_padded:
            vec = a_gmem.load[K_padded, a_row_align](
                Int(a_row0 + m), Int(a_col0)
            )
        else:
            vec = partial_simd_load[K_padded](
                a_gmem.ptr_at_offset(Index(Int(a_row0 + m), Int(a_col0))),
                0,
                K_actual,
                0,
            )
        comptime for k in range(BK):
            var smem_off = a_sw(m * Int32(BK) + Int32(k))
            comptime if k < K_actual:
                a_smem_ptr[smem_off] = vec[k]
            else:
                a_smem_ptr[smem_off] = Scalar[a_type](0)

    # B tile [BN, BK]
    comptime b_sw = make_swizzle[b_type, b_swizzle]()
    var b_smem_ptr = b_smem_tiles[stage].ptr
    comptime b_rows_per_thread = BN // WARP_SIZE
    comptime b_tv = tl_col_major(Coord(Idx[WARP_SIZE], Idx[b_rows_per_thread]))

    comptime for v in range(b_rows_per_thread):
        var n = b_tv[linear_idx_type=DType.int32](Coord(Int32(tid), Idx[v]))
        var vec: SIMD[b_type, K_padded]
        comptime if K_actual == K_padded:
            vec = b_gmem.load[K_padded, b_row_align](
                Int(b_row0 + n), Int(b_col0)
            )
        else:
            vec = partial_simd_load[K_padded](
                b_gmem.ptr_at_offset(Index(Int(b_row0 + n), Int(b_col0))),
                0,
                K_actual,
                0,
            )
        comptime for k in range(BK):
            var smem_off = b_sw(n * Int32(BK) + Int32(k))
            comptime if k < K_actual:
                b_smem_ptr[smem_off] = vec[k]
            else:
                b_smem_ptr[smem_off] = Scalar[b_type](0)

    # fence.proxy.async makes smem stores visible through the
    # mbarrier async proxy before signaling completion.
    fence_async_view_proxy()
    if elect_one_sync():
        _ = tma_mbar[stage].arrive()


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
    //,
    *,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int = 1,
    cluster_shape: IndexList[3] = Index(1, 1, 1),
](
    tmem_addr: UInt32,
    a_smem_tiles: SMemTileArray2D[
        a_type, a_dim0, a_dim1, a_num_tiles, a_swizzle_bytes
    ],
    b_smem_tiles: SMemTileArray2D[
        b_type, b_dim0, b_dim1, b_num_tiles, b_swizzle_bytes
    ],
    mma_mbar: UnsafePointer[
        mut=True, SharedMemBarrier, _, address_space=.SHARED
    ],
    tma_mbar: UnsafePointer[
        mut=True, SharedMemBarrier, _, address_space=.SHARED
    ],
    consumer_phase: PipelineState[pipeline_stages],
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
):
    """Performs the MMA consumer step for one pipeline stage.

    Waits on the TMA mbarrier for the current stage, issues the SM100 SS MMA
    against the staged shared-memory tiles into tensor memory, and commits
    the result to the MMA mbarrier.

    Parameters:
        accum_type: Accumulator dtype used for the MMA result in tensor
            memory (inferred).
        c_type: Element type of the C output matrix (inferred).
        a_type: Element type of the A operand tiles (inferred).
        b_type: Element type of the B operand tiles (inferred).
        a_dim0: Number of rows in one A shared-memory tile (inferred).
        a_dim1: Number of columns in one A shared-memory tile (inferred).
        a_num_tiles: Number of pipeline stages in the A shared-memory
            tile array (inferred).
        a_swizzle_bytes: Swizzle granularity in bytes for A shared-memory
            tiles (inferred).
        b_dim0: Number of rows in one B shared-memory tile (inferred).
        b_dim1: Number of columns in one B shared-memory tile (inferred).
        b_num_tiles: Number of pipeline stages in the B shared-memory
            tile array (inferred).
        b_swizzle_bytes: Swizzle granularity in bytes for B shared-memory
            tiles (inferred).
        a_swizzle: TMA swizzle mode applied to A shared-memory tiles
            (inferred).
        b_swizzle: TMA swizzle mode applied to B shared-memory tiles
            (inferred).
        transpose_b: Whether B is stored transposed in global memory
            (inferred).
        pipeline_stages: Number of double-buffered pipeline stages for A
            and B shared-memory tiles (inferred).
        block_tile_shape: Block tile shape `[BM, BN, BK]` partitioning the
            GEMM into work tiles.
        mma_shape: MMA instruction shape `[MMA_M, MMA_N, MMA_K]`.
        cta_group: Number of CTAs cooperating per MMA along the M
            dimension (defaults to 1).
        cluster_shape: Thread block cluster shape `[CLUSTER_M, CLUSTER_N,
            CLUSTER_D]` (defaults to `Index(1, 1, 1)`).

    Args:
        tmem_addr: Tensor-memory base address where the MMA accumulates
            results for this accumulator stage.
        a_smem_tiles: Shared-memory tile array of staged A tiles, one
            per pipeline stage.
        b_smem_tiles: Shared-memory tile array of staged B tiles, one
            per pipeline stage.
        mma_mbar: Pointer to the MMA mbarrier array, one per pipeline
            stage, signaled when the MMA for a stage completes.
        tma_mbar: Pointer to the TMA mbarrier array, one per pipeline
            stage, waited on to confirm the TMA load for a stage is
            complete.
        consumer_phase: Consumer pipeline state tracking the stage index
            and phase bit.
        mma_op: SM100 SS MMA operation object used to issue and commit the
            tensor-memory multiply-accumulate.
        elect_one_warp: Whether this warp is the elected MMA warp for the
            current work tile.
        iter_idx: Current K-dimension iteration index; when zero, the MMA
            initializes the accumulator instead of accumulating.
    """
    var stage = consumer_phase.index()
    var phase = consumer_phase.phase()

    tma_mbar[stage].wait(phase)

    var a_smem_tile = a_smem_tiles[stage]
    var b_smem_tile = b_smem_tiles[stage]
    if elect_one_sync():
        mma_op.mma(
            a_smem_tile,
            b_smem_tile,
            tmem_addr,
            init_c=(iter_idx == 0),  # Initialize C on first iteration
        )

        mma_op.commit(mma_mbar + stage)


@always_inline
def stsm_helper[
    swizzle: Swizzle,
    vec_dtype: DType,
    vec_size: Int,
    transpose_c: Bool = False,
](
    vec: Array[Scalar[vec_dtype], vec_size],
    dst: LayoutTensor[_, _, address_space=.SHARED, ...],
):
    """Stores a register fragment to shared memory using the stmatrix instruction.

    Casts the loaded fragment to the destination dtype, applies the shared
    memory swizzle, and emits `st_matrix` stores so the epilogue can drain
    tensor memory through shared memory before the TMA store.

    Parameters:
        swizzle: Shared-memory swizzle applied to compute the store offset
            from the lane offset.
        vec_dtype: Element type of the input register fragment `vec`.
        vec_size: Number of scalar elements in the input register fragment
            `vec`.
        transpose_c: Whether to transpose the fragment layout when computing
            the store offset (defaults to False).

    Args:
        vec: Register fragment loaded from tensor memory to store to shared
            memory.
        dst: Destination shared-memory `LayoutTensor` where the fragment is
            written via `st_matrix`.
    """
    # Number of elements in one row per stsmx4 tile, a row is 32B.
    comptime stsmx4_row_size = 32 // size_of[dst.dtype]()
    # Number of elements owned by each lane, each lane has 16B
    comptime stsmx4_lane_size = 16 // size_of[dst.dtype]()
    # TODO: constrain the shared memory layout to be 2D row-major.
    # E.g. dst layout can be (16, 16) : (32, 1), which is tiled from
    # row-major(16, 32). The map should use tile's stride to calculate
    # the dst row offset.
    comptime stride0 = dst.layout.stride[0].value()
    comptime stride1 = dst.layout.stride[1].value()
    comptime assert stride1 == 1, "stride1 must be 1. Got: " + String(stride1)
    comptime shape0 = dst.layout.shape[
        1
    ].value() if not transpose_c else dst.layout.shape[0].value()
    # the layout looks like
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-fragments-shape-16256b
    # but transposed and coalesced by 8 elements.
    comptime trans_st_matrix_layout = Layout(
        IntTuple(8, 2, 2), IntTuple(stride0, 8 * stride1, 8 * stride0)
    )
    comptime stsmx4_tile_offset = (
        stride0 if transpose_c else stride1
    ) * stsmx4_row_size

    var lane = lane_id()
    comptime RLayout32Bits[layout: Layout] = RuntimeLayout[
        layout, element_type=.uint32, linear_idx_type=.uint32
    ]
    var stsm_lane_offset = UInt32(
        (lane & 15) * stride0 + (lane >> 4)
    ) * 8 if not transpose_c else RLayout32Bits[trans_st_matrix_layout]()(lane)

    # Assume the dst tile has 16 rows and only use stsm in N dim.
    comptime for i in range(shape0 // stsmx4_row_size):
        comptime n_offset = i * stsmx4_tile_offset
        var offset = swizzle(stsm_lane_offset + UInt32(n_offset))
        var v = SIMD[dst.dtype, stsmx4_lane_size]()
        comptime cast_width = 4 // size_of[Scalar[dst.dtype]]()
        comptime for k in range(stsmx4_lane_size // cast_width):
            var src = SIMD[vec_dtype, cast_width]()
            comptime for _j in range(cast_width):
                src[_j] = vec[i * stsmx4_lane_size + k * cast_width + _j]
            var casted = src.cast[dst.dtype]()
            comptime for _j in range(cast_width):
                v[k * cast_width + _j] = casted[_j]
        st_matrix[simd_width=4, transpose=transpose_c](
            dst.ptr.unsafe_mut_cast[True]() + offset,
            bitcast[.float32, 4](v),
        )


@always_inline
def multi_stage_store_C[
    c_type: DType,
    c_tile_rank: Int,
    c_tile_shape: IndexList[c_tile_rank],
    c_desc_shape: IndexList[c_tile_rank],
    num_accum_pipeline_stages: Int,
    /,
    *,
    c_smem_layout: Layout,
    accum_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    stage_stride_cols: Int,
    c_static_N: Int,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    cta_group: Int = 1,
    num_output_warps: Int = 4,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    transpose_c: Bool = False,
](
    c_smem_base: UnsafePointer[
        mut=True, Scalar[c_type], _, address_space=.SHARED
    ],
    c_tma_op: TMATensorTile[c_type, c_tile_rank, c_tile_shape, c_desc_shape],
    c_ptr: UnsafePointer[mut=True, Scalar[c_type], _],
    accum_pipeline_consumer_state: PipelineState[num_accum_pipeline_stages],
    accum_full_mbar: UnsafePointer[
        mut=True, SharedMemBarrier, _, address_space=.SHARED
    ],
    accum_empty_mbar: UnsafePointer[
        mut=True, SharedMemBarrier, _, address_space=.SHARED
    ],
    tmem_addr: UInt32,
    work_tile_coord: Tuple[Int, Int],
    group_end_idx: UInt32,
    elect_one_warp: Bool,
    M: UInt32,
    N: UInt32,
):
    """Drains accumulated results from tensor memory and stores them to global memory.

    Waits on the accumulator-full mbarrier, loads fragments from tensor
    memory in stages, packs them into swizzled shared memory via
    `stsm_helper`, and issues TMA async stores (or a scalar fallback for
    unaligned tails) to write the output tile, optionally applying an
    elementwise epilogue.

    Parameters:
        c_type: Element type of the C output matrix.
        c_tile_rank: Rank of the C TMA tile descriptor.
        c_tile_shape: Element shape of one C TMA tile.
        c_desc_shape: Descriptor shape of the C TMA tile, used to compute
            per-store element counts and row stride.
        num_accum_pipeline_stages: Number of accumulator pipeline stages
            used to double-buffer TMEM-to-global-memory C stores.
        c_smem_layout: Layout of one C shared-memory output tile stage;
            its shape determines the per-stage column width and tile size.
        accum_type: Accumulator dtype loaded from tensor memory before
            packing into shared memory.
        block_tile_shape: Block tile shape `[BM, BN, BK]` partitioning the
            GEMM into work tiles.
        mma_shape: MMA instruction shape `[MMA_M, MMA_N, MMA_K]`.
        stage_stride_cols: Per-stage column stride in tensor memory
            between accumulator pipeline stages.
        c_static_N: Static N dimension of the C output matrix; used as
            the row stride and M-coordinate bound in the scalar fallback
            store path.
        c_swizzle: TMA swizzle mode applied to C shared-memory tiles
            (defaults to `SWIZZLE_128B`).
        cta_group: Number of CTAs cooperating per MMA along the M
            dimension (defaults to 1).
        num_output_warps: Number of warps participating in the epilogue
            store (defaults to 4).
        elementwise_lambda_fn: Optional elementwise epilogue applied to
            stored C fragments (defaults to None).
        transpose_c: Whether to transpose the C output tile in shared
            memory before the TMA store (defaults to False).

    Args:
        c_smem_base: Base pointer to the C shared-memory buffer used for
            double-buffered output tile staging.
        c_tma_op: TMA tile descriptor for storing C tiles from shared to
            global memory.
        c_ptr: Base pointer to the row-major C output tensor in global
            memory, used by the scalar fallback store path.
        accum_pipeline_consumer_state: Consumer pipeline state tracking
            the accumulator stage index and phase bit.
        accum_full_mbar: Pointer to the accumulator-full mbarrier array,
            one per pipeline stage, waited on to confirm the MMA has
            filled the accumulator stage.
        accum_empty_mbar: Pointer to the accumulator-empty mbarrier
            array, arrived on to signal the consumer has drained an
            accumulator stage.
        tmem_addr: Tensor-memory base address where the MMA accumulated
            results for this work tile reside.
        work_tile_coord: `(m, n)` element coordinates of the current work
            tile returned by the scheduler.
        group_end_idx: Exclusive end of the current group's valid N
            range, used to mask out-of-bound TMA stores and select the
            scalar fallback for unaligned tails.
        elect_one_warp: Whether this warp is the elected epilogue leader
            for issuing TMA stores.
        M: M dimension of the GEMM output in elements.
        N: N dimension of the GEMM output in elements.
    """
    # WAIT FOR MMA TO FINISH AND STORE RESULT
    # scheduler fetch next work
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]

    comptime num_m_mmas = BM // (mma_shape[0] // cta_group)
    comptime num_n_mmas = BN // (mma_shape[1] // cta_group)

    comptime assert num_m_mmas == 1 and num_n_mmas == 1

    # assume N dimension is static
    comptime simd_size = simd_width_of[c_type]()

    # we break down the output tile BM x MMA_N to BM x stageN tiles
    # and output one tile per stage.
    # stage N is 32
    comptime N_dim = 0 if transpose_c else 1
    comptime stageN = c_smem_layout.shape[N_dim].value()
    comptime stage_contiguous_size = c_smem_layout.shape[1].value()
    # so num stages is usually 256 by 32 is 8
    # MMA Size will be larger than output tile shape. E.G. MMA_MxMMA_N = (128, 256); OUT_MxOUT_N = (128, 32)

    comptime num_stages = MMA_N // stageN if (
        MMA_M == 256 or cta_group == 1
    ) else MMA_N // stageN // 2
    comptime data_paths = 16  # same as lanes
    comptime bits = 256
    # every element in tmem is 4 bytes, so bits being 256 means 8 elements stored across N
    # repeated 4 times is 8*4 = 32, enough to move elements into the width of our 128x32 tile
    comptime rep = stageN // (bits // 32)  # repetitions per stage
    # typically repeated 4 times to get the desired 32 elements

    # stageN is how many elements we want to load at once

    # stmatrix related
    comptime st_matrix_swizzle = c_swizzle
    comptime swizzle = make_swizzle[c_type, st_matrix_swizzle]()

    var warp_id = get_warp_id()

    # before i start the process of transferring over num_stages * stageN= MMA_N from tensor memory to global, i should wait
    # on the accum_full_mbar barrier
    var index = accum_pipeline_consumer_state.index()
    var phase = accum_pipeline_consumer_state.phase()
    accum_full_mbar[index].wait(phase)
    # this is the column offset for all the stages of THIS load, where one load takes (num_stages iterations)
    var tmem_offset = index * UInt32(stage_stride_cols) + tmem_addr

    comptime frag_width = rep * data_paths * (bits // 32) // WARP_SIZE

    comptime for stage in range(num_stages):
        # column offset, moving right by 32 columns each time, since each num_stage stores two, 16 column submatrices
        # MMA has result in 32 rows per warp's data paths.
        # upper_frag is for rows 0-15, lower is for 16-31.
        var stage_tmem_addr = tmem_offset + UInt32(stage * stageN)
        var upper_frag = tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=rep,
            dtype=accum_type,
            pack=False,
        ](stage_tmem_addr)

        var lower_frag = tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=rep,
            dtype=accum_type,
            pack=False,
        ](stage_tmem_addr + (16 << 16))

        tcgen05_load_wait()

        comptime if stage == num_stages - 1:
            umma_arrive_leader_cta(accum_empty_mbar + index)

        # Assume double-buffer for shared memory packing
        comptime c_smem_tile_size = c_smem_layout.size()
        var c_smem_tile = LayoutTensor[
            c_type,
            c_smem_layout,
            address_space=.SHARED,
            alignment=128,
        ](c_smem_base + (stage % 2) * c_smem_tile_size)

        comptime if transpose_c:
            # if stage_contiguous_size is 128, we need to split the shared memory
            # into two stageNx64 row-major tiles due to the limitation of 128B TMA
            # swizzle. However, for easier programming, we reshape the tile
            # contiguous row_major(stageN, 16) chunks.
            var c_smem_warp_tile_upper = c_smem_tile.tile[
                stageN * 16 // stage_contiguous_size, stage_contiguous_size
            ](2 * warp_id, 0).reshape[Layout.row_major(stageN, 16)]()
            var c_smem_warp_tile_lower = c_smem_tile.tile[
                stageN * 16 // stage_contiguous_size, stage_contiguous_size
            ](2 * warp_id + 1, 0).reshape[Layout.row_major(stageN, 16)]()

            # Pack the upper frag to shared memory
            stsm_helper[swizzle, transpose_c=transpose_c](
                upper_frag, c_smem_warp_tile_upper
            )
            stsm_helper[swizzle, transpose_c=transpose_c](
                lower_frag, c_smem_warp_tile_lower
            )

            # Guard the write to shared memory is done.
            named_barrier[Int32(num_output_warps * WARP_SIZE)]()

        else:
            var c_smem_warp_tile = c_smem_tile.tile[32, stageN](warp_id, 0)

            var c_smem_warp_tile_upper = c_smem_warp_tile.tile[
                data_paths, stageN
            ](0, 0)
            var c_smem_warp_tile_lower = c_smem_warp_tile.tile[
                data_paths, stageN
            ](1, 0)
            stsm_helper[swizzle, transpose_c=transpose_c](
                upper_frag, c_smem_warp_tile_upper
            )
            stsm_helper[swizzle, transpose_c=transpose_c](
                lower_frag, c_smem_warp_tile_lower
            )

            # Guard the write to shared memory is done.
            named_barrier[Int32(num_output_warps * WARP_SIZE)]()

        var lane = lane_id()

        comptime TMA_BM = c_smem_tile.layout.shape[
            0
        ].value() if MMA_M == 256 or cta_group == 1 else BM

        var elect_one_warp = (
            warp_id == 0 if MMA_M == 256
            or cta_group == 1 else umod(warp_id, 2) == 0
        )
        # var coord_n_mma_m256 = work_tile_coord[1] * MMA_N + stage * stageN
        # var coord_n_mma_m128 = (
        #     work_tile_coord[1] * MMA_N + stage * stageN + BN * (warp_id // 2)
        # )
        var coord_n_mma_m256 = work_tile_coord[1] + stage * stageN
        var coord_n_mma_m128 = (
            work_tile_coord[1] + stage * stageN + BN * ufloordiv(warp_id, 2)
        )

        var coord_n = (
            coord_n_mma_m256 if MMA_M == 256
            or cta_group == 1 else coord_n_mma_m128
        )

        var n_inbound_size = group_end_idx - UInt32(coord_n)

        comptime M = c_smem_tile.layout.shape[1].value()

        comptime has_elementwise_lambda = Bool(elementwise_lambda_fn)

        if not has_elementwise_lambda and n_inbound_size >= UInt32(stageN):
            if elect_one_warp and lane == 0:
                fence_async_view_proxy()

                comptime if transpose_c:
                    comptime for i in range(M // 16):
                        var c_smem_warp_tile = c_smem_tile.tile[
                            stageN * 16 // stage_contiguous_size,
                            stage_contiguous_size,
                        ](i, 0).reshape[Layout.row_major(stageN, 16)]()
                        c_tma_op.async_store(
                            c_smem_warp_tile,
                            (
                                work_tile_coord[0] + i * 16,
                                coord_n,
                            ),
                        )
                else:
                    var c_smem_coord_m = 0 if MMA_M == 256 else ufloordiv(
                        warp_id, 2
                    )
                    var c_smem_split = c_smem_tile.tile[TMA_BM, stageN](
                        c_smem_coord_m, 0
                    )
                    c_tma_op.async_store(
                        c_smem_split,
                        (
                            coord_n,
                            # work_tile_coord[0] * BM,
                            work_tile_coord[0],
                        ),
                    )
                c_tma_op.commit_group()

            # Keep one tma store in fly
            comptime if stage < num_stages - 1:
                c_tma_op.wait_group[1]()
            # Last stage guard all tma store to finish
            else:
                c_tma_op.wait_group[0]()
        else:
            comptime assert (
                transpose_c
            ), "Unaligned handling only supports transpose_c"
            comptime assert MMA_M == 256 or cta_group == 1, (
                "Unaligned handling only supports MMA_M == 256 or cta_group =="
                " 1. Got "
                + String(MMA_M)
                + " and "
                + String(cta_group)
            )

            comptime chunkM = st_matrix_swizzle.bytes() // size_of[c_type]()
            comptime vec_chunkM = chunkM // simd_size
            comptime chunk_num = M // chunkM
            comptime logical_c_layout = Layout.row_major(
                chunk_num, stageN, vec_chunkM
            )
            comptime thread_num = num_output_warps * WARP_SIZE
            comptime assert logical_c_layout.size() % thread_num == 0, (
                "logical_c_layout.size() must be a multiple of thread_num. Got "
                + String(logical_c_layout.size())
                + "."
            )
            comptime value_shape = logical_c_layout.size() // thread_num
            comptime cN = c_static_N

            comptime for v in range(value_shape):
                comptime thread_offset = v * thread_num
                var thread_index = UInt32(thread_idx.x) + UInt32(thread_offset)
                # idx2crd but RuntimeTuple.idx2crd is too hard to use
                var vec_chunkM_idx = thread_index % UInt32(vec_chunkM)
                var rest = thread_index // UInt32(vec_chunkM)
                var n_idx = rest % UInt32(stageN)
                if n_idx >= min(n_inbound_size, UInt32(stageN)):
                    continue
                var src_idx = UInt32(simd_size) * thread_index
                var c_smem_idx = swizzle(src_idx)
                comptime alignment = align_of[SIMD[c_type, simd_size]]()
                var val_vec = (c_smem_tile.ptr + c_smem_idx).load[
                    width=simd_size, alignment=alignment
                ]()
                var chunk_idx = rest // UInt32(stageN)
                var n = UInt32(coord_n) + n_idx
                var m = UInt32(work_tile_coord[0]) + (
                    chunk_idx * UInt32(vec_chunkM) + vec_chunkM_idx
                ) * UInt32(simd_size)
                if m < UInt32(cN):
                    comptime if elementwise_lambda_fn:
                        comptime elementwise_lambda = elementwise_lambda_fn.value()
                        elementwise_lambda[
                            c_type, simd_size, alignment=alignment
                        ](Index(n, m), val_vec)
                    else:
                        (c_ptr + n * UInt32(cN) + m).store[alignment=alignment](
                            val_vec
                        )

        comptime if stage > 0 or stage == num_stages - 1:
            # Guard the tma read from shared memory is done.
            named_barrier[Int32(num_output_warps * WARP_SIZE)]()


def zero_output[
    c_type: DType,
    *,
    output_tile_shape: IndexList[2],
    c_stride: Int,
    c_N: Int,
](
    c_ptr: UnsafePointer[Scalar[c_type], MutAnyOrigin],
    coord: Tuple[UInt32, UInt32],
    group_end_idx: UInt32,
):
    """Zero-fills an output tile for skipped or invalid expert assignments.

    Each thread stores a SIMD vector of zeros per row across the output
    tile width, masking out threads and rows that fall outside the valid
    tile or group boundary.

    Parameters:
        c_type: Element type of the C output tensor.
        output_tile_shape: Tile shape `[rows, cols]` of the output region to
            zero-fill, where index 0 bounds the row count and index 1 bounds
            the column count.
        c_stride: Row stride of the C output tensor in elements; the store
            pointer advances by this amount per row.
        c_N: Total width of the C output tensor in elements; bounds the
            column offset `coord[0]` to mask out-of-bound stores.

    Args:
        c_ptr: Base pointer to the C output tensor in global memory.
        coord: `(m, n)` element coordinates of the output tile origin, where
            `coord[0]` is the column offset and `coord[1]` is the row offset.
        group_end_idx: Exclusive end of the current group's valid row range;
            bounds the number of rows zeroed.
    """
    comptime thread_num = 4 * WARP_SIZE
    comptime simd_size = min(2, simd_width_of[c_type]())

    # This is an easy to implement but quite inefficient way to zero out the
    # output. Note that we are simply filling the output row by row and mask out
    # out of bound thread and the thread layout is simply one dimensional.

    # Note that output_tile_shape is always the proper C tile shape independent of transpose_c.
    comptime output_N = output_tile_shape[1]
    var ptr = c_ptr + coord[1] * UInt32(c_stride) + coord[0]
    comptime assert thread_num * simd_size >= output_N, (
        "output_N must be less than thread_num * simd_size. Got "
        + String(output_N)
        + "."
    )
    comptime row_thread_num = UInt32(output_N // simd_size)
    var row_boundary = (UInt32(c_N) - coord[0]) // UInt32(simd_size)
    comptime alignment = align_of[SIMD[c_type, simd_size]]()
    var zero_vec = SIMD[c_type, simd_size](0.0)
    var M = group_end_idx - coord[1]
    if UInt32(thread_idx.x) < min(row_thread_num, row_boundary):
        for i in range(min(M, UInt32(output_tile_shape[0]))):
            (ptr + thread_idx.x * simd_size).store[alignment=alignment](
                zero_vec
            )
            ptr += c_stride


@always_inline
def zero_output_epilogue[
    c_type: DType,
    /,
    *,
    c_smem_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    c_static_N: Int,
    c_swizzle: TensorMapSwizzle,
    cta_group: Int,
    num_output_warps: Int,
    elementwise_lambda_fn: elementwise_epilogue_type,
    transpose_c: Bool,
](work_tile_coord: Tuple[Int, Int], group_end_idx: UInt32,):
    """Zero an inactive group's output THROUGH the elementwise epilogue.

    Mirror of the epilogue (unaligned/`elementwise_lambda_fn`) branch of
    `multi_stage_store_C`, but with a zero accumulator: for an inactive group
    (`expert_id < 0`) there is no MMA, so this walks the same output-tile
    coordinate mapping and calls `elementwise_lambda_fn` with a zero vector at
    each `(n, m)`. It deliberately does NOT wait on `accum_full_mbar` or read
    TMEM: the main-load and MMA warps skip inactive groups entirely (they never
    arrive on that barrier), so waiting would deadlock. Matches the naive
    grouped matmul, which likewise runs the epilogue with `accum == 0` for
    `expert == -1` instead of storing zeros directly to C -- so callers whose
    real output lives behind the epilogue (e.g. the LoRA-B QKV expand's
    `route_qkv`) get their `-1` tokens zeroed without a separate memset.
    """
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime simd_size = simd_width_of[c_type]()

    comptime N_dim = 0 if transpose_c else 1
    comptime stageN = c_smem_layout.shape[N_dim].value()
    comptime num_stages = MMA_N // stageN if (
        MMA_M == 256 or cta_group == 1
    ) else MMA_N // stageN // 2

    comptime assert (
        transpose_c
    ), "zero_output_epilogue only supports transpose_c (epilogue path)"
    comptime assert (
        MMA_M == 256 or cta_group == 1
    ), "zero_output_epilogue only supports MMA_M == 256 or cta_group == 1"

    # `M` here is the tile's contiguous (row) extent, matching
    # `c_smem_tile.layout.shape[1]` in `multi_stage_store_C`.
    comptime M = c_smem_layout.shape[1].value()
    comptime chunkM = c_swizzle.bytes() // size_of[c_type]()
    comptime vec_chunkM = chunkM // simd_size
    comptime chunk_num = M // chunkM
    comptime logical_c_layout = Layout.row_major(chunk_num, stageN, vec_chunkM)
    comptime thread_num = num_output_warps * WARP_SIZE
    comptime assert logical_c_layout.size() % thread_num == 0
    comptime value_shape = logical_c_layout.size() // thread_num
    comptime cN = c_static_N
    comptime alignment = align_of[SIMD[c_type, simd_size]]()
    comptime elementwise_lambda = elementwise_lambda_fn
    var zero_vec = SIMD[c_type, simd_size](0)

    comptime for stage in range(num_stages):
        # Only the M==256/cta_group==1 coord form is reachable here (asserted
        # above); `coord_n` matches `coord_n_mma_m256` in `multi_stage_store_C`.
        var coord_n = work_tile_coord[1] + stage * stageN
        var n_inbound_size = group_end_idx - UInt32(coord_n)

        comptime for v in range(value_shape):
            comptime thread_offset = v * thread_num
            var thread_index = UInt32(thread_idx.x) + UInt32(thread_offset)
            var vec_chunkM_idx = thread_index % UInt32(vec_chunkM)
            var rest = thread_index // UInt32(vec_chunkM)
            var n_idx = rest % UInt32(stageN)
            if n_idx >= min(n_inbound_size, UInt32(stageN)):
                continue
            var chunk_idx = rest // UInt32(stageN)
            var n = UInt32(coord_n) + n_idx
            var m = UInt32(work_tile_coord[0]) + (
                chunk_idx * UInt32(vec_chunkM) + vec_chunkM_idx
            ) * UInt32(simd_size)
            if m < UInt32(cN):
                elementwise_lambda[c_type, simd_size, alignment=alignment](
                    Index(n, m), zero_vec
                )


# Important deviation from the normal SM100 matmul: The coordinate returned by
# the tile scheduler is not necessarily aligned to `MMA_N` because of group
# splitting. Thus, we simply take the `work_tile_coord` without scaling it
# in the callee; the scheduler returns element coordinates instead of tile
# coordinates.
@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
@__name(t"grouped_matmul_sm100_{a_type}_{b_type}_{c_type}_em{expert_m}")
def blackwell_tma_umma_warp_specialized_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    expert_m: Int,
    a_tile_rank: Int,
    a_tile_shape: IndexList[a_tile_rank],
    a_desc_shape: IndexList[a_tile_rank],
    b_tile_rank: Int,
    b_tile_shape: IndexList[b_tile_rank],
    b_desc_shape: IndexList[b_tile_rank],
    c_tile_rank: Int,
    c_tile_shape_param: IndexList[c_tile_rank],
    c_desc_shape: IndexList[c_tile_rank],
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    num_pipeline_stages: Int,
    num_accum_pipeline_stages: Int,
    num_output_stages: Int = 2,
    output_tile_shape: IndexList[2] = Index(128, 32),
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    cta_group: Int = 2,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    a_plane_splits: IndexList[2] = Index(0, 0),
    transpose_c: Bool = False,
    use_tma: Bool = True,
    K_actual: Int = 0,
    a_gmem_layout: Layout = Layout.row_major(1, 1),
    b_gmem_layout: Layout = Layout.row_major(1, 1),
](
    expert_usage_stats: UnsafePointer[UInt32, ImmutAnyOrigin],
    a_tma_op: TMATensorTile[a_type, a_tile_rank, a_tile_shape, a_desc_shape],
    expert_ids: UnsafePointer[Int32, ImmutAnyOrigin],
    b_tma_op: TMATensorTile[b_type, b_tile_rank, b_tile_shape, b_desc_shape],
    b_offsets: UnsafePointer[UInt32, ImmutAnyOrigin],
    c_tma_op: TMATensorTile[
        c_type, c_tile_rank, c_tile_shape_param, c_desc_shape
    ],
    c_ptr: UnsafePointer[Scalar[c_type], MutAnyOrigin],
    mnk: StaticTuple[UInt32, 3],
    a_gmem: LayoutTensor[a_type, a_gmem_layout, ImmutAnyOrigin],
    b_gmem: LayoutTensor[b_type, b_gmem_layout, ImmutAnyOrigin],
):
    """Implements the warp-specialized persistent grouped GEMM kernel for SM100.

    Splits the CTA warps into main-load, MMA, and epilogue roles that
    cooperate through mbarriers and a tile scheduler to iterate over
    grouped GEMM work tiles, streaming A and B through shared memory via
    TMA and accumulating into tensor memory before storing C back to
    global memory.

    Parameters:
        a_type: Element type of the A operand matrix.
        b_type: Element type of the B operand matrix.
        c_type: Element type of the C output matrix.
        expert_m: Static M dimension of each expert's local slice, used as the
            C stride and row bound for contiguous row-major C output.
        a_tile_rank: Rank of the A TMA tile descriptor.
        a_tile_shape: Element shape of one A TMA tile.
        a_desc_shape: Descriptor shape of the A TMA tile, used to compute per-
            load element counts and row stride.
        b_tile_rank: Rank of the B TMA tile descriptor.
        b_tile_shape: Element shape of one B TMA tile.
        b_desc_shape: Descriptor shape of the B TMA tile, used to compute per-
            load element counts and row stride.
        c_tile_rank: Rank of the C TMA tile descriptor.
        c_tile_shape_param: Element shape of one C TMA tile.
        c_desc_shape: Descriptor shape of the C TMA tile.
        block_tile_shape: Block tile shape `[BM, BN, BK]` partitioning the GEMM
            into work tiles.
        mma_shape: MMA instruction shape `[MMA_M, MMA_N, MMA_K]`.
        cluster_shape: Thread block cluster shape `[CLUSTER_M, CLUSTER_N,
            CLUSTER_D]`.
        num_pipeline_stages: Number of TMA producer and consumer pipeline stages
            used to double-buffer A and B shared-memory tiles.
        num_accum_pipeline_stages: Number of accumulator pipeline stages
            used to double-buffer TMEM-to-global-memory C stores.
        num_output_stages: Number of output shared-memory buffer stages for C
            (defaults to 2).
        output_tile_shape: Output tile shape `[OUT_M, OUT_N]` for one C shared-
            memory stage (defaults to `Index(128, 32)`).
        transpose_b: Whether B is stored transposed in global memory (defaults
            to True).
        a_swizzle: TMA swizzle mode applied to A shared-memory tiles (defaults
            to `SWIZZLE_128B`).
        b_swizzle: TMA swizzle mode applied to B shared-memory tiles (defaults
            to `SWIZZLE_128B`).
        c_swizzle: TMA swizzle mode applied to C shared-memory tiles (defaults
            to `SWIZZLE_128B`).
        cta_group: Number of CTAs cooperating per MMA along the M dimension
            (defaults to 2).
        elementwise_lambda_fn: Optional elementwise epilogue applied to stored
            C fragments (defaults to None).
        a_plane_splits: Per-plane split sizes for fused LoRA QKV A-plane row
            offsetting; `(0, 0)` disables it (defaults to `(0, 0)`).
        transpose_c: Whether to transpose the C output tile in shared memory
            before the TMA store (defaults to False).
        use_tma: Whether to use TMA for A and B loads; when False, use the
            CUDA-core fallback (defaults to True).
        K_actual: Actual K dimension in elements for the CUDA-core fallback
            when `use_tma` is False (defaults to 0).
        a_gmem_layout: Layout of the A global-memory tensor, used only by the
            CUDA-core fallback (defaults to `Layout.row_major(1, 1)`).
        b_gmem_layout: Layout of the B global-memory tensor, used only by the
            CUDA-core fallback (defaults to `Layout.row_major(1, 1)`).

    Args:
        expert_usage_stats: Pointer to per-expert usage stats; index 1 holds
            the active expert count.
        a_tma_op: TMA tile descriptor for loading A tiles from global to shared
            memory.
        expert_ids: Pointer to the per-group expert id array; negative entries
            skip the tile.
        b_tma_op: TMA tile descriptor for loading B tiles from global to shared
            memory.
        b_offsets: Pointer to the per-expert B offset prefix sum, of length
            `num_active_experts + 1`.
        c_tma_op: TMA tile descriptor for storing C tiles from shared to global
            memory.
        c_ptr: Base pointer to the row-major C output tensor in global memory.
        mnk: `[M, N, K]` dimensions of the grouped GEMM as unsigned 32-bit
            integers.
        a_gmem: A operand tensor used only by the CUDA-core fallback when
            `use_tma` is False.
        b_gmem: B operand tensor used only by the CUDA-core fallback when
            `use_tma` is False.
    """
    comptime assert c_type != .float32, "c_type cannot be float32"
    comptime if not use_tma:
        comptime assert (
            K_actual > 0
        ), "K_actual must be positive when use_tma is False"

    comptime num_output_warps = 4

    comptime SCHEDULER_THREADS = WARP_SIZE
    comptime TMA_LOAD_THREADS = WARP_SIZE
    comptime MMA_THREADS = WARP_SIZE
    comptime EPILOGUE_THREADS = num_output_warps * WARP_SIZE
    comptime CLUSTER_SIZE = Int(cluster_shape[0]) * Int(cluster_shape[1])
    comptime clc_producer_arv_count = 1
    comptime clc_consumer_arv_count = SCHEDULER_THREADS + CLUSTER_SIZE * (
        TMA_LOAD_THREADS + MMA_THREADS + EPILOGUE_THREADS
    )

    # For ld from TMEM, use same per-stage stride in column field.
    comptime NUM_TMEM_COLS = 512
    comptime stage_stride_cols = NUM_TMEM_COLS // num_accum_pipeline_stages

    comptime clc_throttle_producer_arv_count = TMA_LOAD_THREADS
    comptime clc_throttle_consumer_arv_count = SCHEDULER_THREADS

    comptime accum_pipeline_producer_arv_count = 1
    comptime accum_pipeline_consumer_arv_count = cta_group * EPILOGUE_THREADS

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]

    comptime num_m_mmas = BM // (mma_shape[0] // cta_group)
    comptime num_n_mmas = BN // (mma_shape[1] // cta_group)
    comptime num_k_mmas = BK // mma_shape[2]

    comptime CLUSTER_M = Int(cluster_shape[0])
    comptime CLUSTER_N = Int(cluster_shape[1])

    comptime a_tma_load_size = _idx_product[a_tile_rank, a_desc_shape]()
    comptime b_tma_load_size = _idx_product[b_tile_rank, b_desc_shape]()
    comptime a_tma_rows = a_desc_shape[0]
    comptime b_tma_rows = b_desc_shape[0]

    var base_ptr_smem = external_memory[
        Scalar[a_type],
        address_space=.SHARED,
        alignment=128,
    ]()

    comptime a_smem_size = tile_layout_k_major_typed[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ].static_product * num_pipeline_stages
    comptime b_smem_size = tile_layout_k_major_typed[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ].static_product * num_pipeline_stages
    comptime c_smem_size = output_tile_shape[0] * output_tile_shape[
        1
    ] * num_output_stages

    var a_smem_base = base_ptr_smem
    var b_smem_base = (a_smem_base + a_smem_size).bitcast[Scalar[b_type]]()
    var c_smem_base = (b_smem_base + b_smem_size).bitcast[Scalar[c_type]]()

    # TileTensor views of shared memory for both TMA producer and MMA consumer.
    var a_smem_tt = SMemTileArray2D[
        a_type,
        BM,
        BK,
        num_pipeline_stages,
        swizzle_mode_to_bytes[a_swizzle],
    ](a_smem_base)
    var b_smem_tt = SMemTileArray2D[
        b_type,
        BN,
        BK,
        num_pipeline_stages,
        swizzle_mode_to_bytes[b_swizzle],
    ](b_smem_base)

    var smem_pool = (c_smem_base + c_smem_size).bitcast[Int64]()

    var tma_mbar_ptr = smem_pool
    var mma_mbar_ptr = tma_mbar_ptr + num_pipeline_stages
    var accum_full_mbar_ptr = mma_mbar_ptr + num_pipeline_stages
    var accum_empty_mbar_ptr = accum_full_mbar_ptr + num_accum_pipeline_stages

    var tmem_dealloc_mbar_ptr = (
        accum_empty_mbar_ptr + num_accum_pipeline_stages
    ).bitcast[Int64]()

    var ptr_tmem_addr = (tmem_dealloc_mbar_ptr + 1).bitcast[UInt32]()

    var tma_mbar = tma_mbar_ptr.bitcast[SharedMemBarrier]()
    var mma_mbar = mma_mbar_ptr.bitcast[SharedMemBarrier]()
    var accum_full_mbar = accum_full_mbar_ptr.bitcast[SharedMemBarrier]()
    var accum_empty_mbar = accum_empty_mbar_ptr.bitcast[SharedMemBarrier]()
    var tmem_dealloc_mbar = tmem_dealloc_mbar_ptr.bitcast[SharedMemBarrier]()

    comptime accum_type = get_accum_type[a_type]()

    var warp_id = get_warp_id()
    var elect_one_warp = warp_id == 0
    var elect_one_thread = elect_one_sync_with_mask()
    var elect_one_cta = block_rank_in_cluster() % 2 == 0
    var is_first_cta_in_cluster = block_rank_in_cluster() == 0
    comptime max_tmem_cols = 512

    if elect_one_warp and elect_one_thread:
        comptime if use_tma:
            a_tma_op.prefetch_descriptor()
            b_tma_op.prefetch_descriptor()
        c_tma_op.prefetch_descriptor()

        comptime for i in range(num_pipeline_stages):
            tma_mbar[i].init()
            # we need to have 5 arrivals, 2 M, 4 N, top left M/N is shared
            mma_mbar[i].init(
                cluster_shape[0] // Int32(cta_group) + cluster_shape[1] - 1
            )

        comptime for i in range(num_accum_pipeline_stages):
            accum_full_mbar[i].init(accum_pipeline_producer_arv_count)
            accum_empty_mbar[i].init(Int32(accum_pipeline_consumer_arv_count))

        tmem_dealloc_mbar[].init(Int32(EPILOGUE_THREADS * cta_group))

    fence_mbarrier_init()
    cluster_sync()

    var consumer_phase = PipelineState[num_pipeline_stages]()
    var producer_phase = PipelineState[num_pipeline_stages](0, 1, 0)

    var accum_pipeline_producer_state = PipelineState[
        num_accum_pipeline_stages
    ](0, 1, 0)
    var accum_pipeline_consumer_state = PipelineState[
        num_accum_pipeline_stages
    ]()

    var mma_op = MmaOpSM100_SS[
        c_type,
        a_type,
        b_type,
        block_tile_shape,
        mma_shape,
        accum_type=accum_type,
        cta_group=cta_group,
        cluster_shape=Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        transpose_b=transpose_b,
    ]()

    # num_active_experts is a TileScheduler loop bound, and the persistent grid
    # is fixed at one CTA per SM, so it can be read from the device tensor
    # rather than taken as a host scalar. (`[0]` = max tokens, unused here.)
    var num_active_experts = Int(expert_usage_stats[1])

    comptime _offsets_layout = Layout.row_major(UNKNOWN_VALUE)
    var b_offsets_tensor = LayoutTensor[
        .uint32,
        _offsets_layout,
        ImmutAnyOrigin,
    ](
        b_offsets,
        RuntimeLayout[_offsets_layout].row_major(Index(num_active_experts + 1)),
    )
    var scheduler = TileScheduler[
        static_MN=expert_m,
        cluster=Index(cluster_shape[0], cluster_shape[1], cluster_shape[2]),
        cta_group=cta_group,
        tile_shape=Index(
            block_tile_shape[0], block_tile_shape[1], block_tile_shape[2]
        ),
        swapAB=transpose_c,
    ](num_active_experts, b_offsets_tensor)

    var work_info = scheduler.fetch_next_work()

    var rank_m = block_id_in_cluster.x
    var rank_n = block_id_in_cluster.y

    # (peer_id, mma_coord_m, mma_coord_n)
    var peer_cta_coord = (
        umod(rank_m, cta_group),
        ufloordiv(rank_m, cta_group),
        rank_n,
    )  # v,m,n

    var a_multicast_mask: UInt16 = 0x0
    var b_multicast_mask: UInt16 = 0x0

    # TODO: find a generic way to calculate multicast mask
    comptime for i in range(CLUSTER_N):
        a_multicast_mask |= UInt16(1 << (i * CLUSTER_M))
    # they all have the same v and m, but different n,

    comptime for i in range(CLUSTER_M // cta_group):
        b_multicast_mask |= UInt16(1 << (i * cta_group))

    a_multicast_mask <<= UInt16(rank_m)
    b_multicast_mask <<= UInt16(peer_cta_coord[0])
    b_multicast_mask <<= UInt16(rank_n * CLUSTER_M)

    var self_mask = 1 << Int(block_rank_in_cluster())
    var peer_mask = 1 << Int(block_rank_in_cluster() + 1)
    var mma_complete_mask = self_mask | peer_mask

    var num_iters: UInt32 = ceildiv(mnk[2], UInt32(BK))

    if WarpRole.is_main_load():
        while not work_info.is_done():
            if (
                not work_info.is_valid()
                or expert_ids[Int(scheduler.current_group_idx)] < 0
            ):
                work_info = scheduler.fetch_next_work()
                continue

            for i in range(num_iters):
                comptime if use_tma:
                    load_AB[
                        block_tile_shape=block_tile_shape,
                        mma_shape=mma_shape,
                        cta_group=cta_group,
                        a_plane_splits=a_plane_splits,
                    ](
                        expert_ids,
                        a_tma_op,
                        b_tma_op,
                        a_smem_tt,
                        b_smem_tt,
                        mma_mbar,
                        tma_mbar,
                        producer_phase,
                        peer_cta_coord,
                        (Int(work_info.m), Int(work_info.n)),
                        a_multicast_mask,
                        b_multicast_mask,
                        i,
                        elect_one_cta,
                        scheduler,
                        Int(mnk[0]),
                    )
                else:
                    load_AB_cuda_core[
                        K_actual=K_actual,
                        cta_group=cta_group,
                        a_swizzle=a_swizzle,
                        b_swizzle=b_swizzle,
                        a_gmem_layout=a_gmem_layout,
                        b_gmem_layout=b_gmem_layout,
                        a_plane_splits=a_plane_splits,
                    ](
                        a_gmem,
                        b_gmem,
                        expert_ids,
                        a_smem_tt,
                        b_smem_tt,
                        mma_mbar,
                        tma_mbar,
                        producer_phase,
                        peer_cta_coord,
                        (Int(work_info.m), Int(work_info.n)),
                        i,
                        scheduler,
                        Int(mnk[0]),
                    )
                producer_phase.step()

            syncwarp()
            var next_work_info = scheduler.fetch_next_work()
            work_info = next_work_info

        comptime for i in range(num_pipeline_stages):
            mma_mbar[producer_phase.index()].wait(producer_phase.phase())
            producer_phase.step()

    if WarpRole.is_mma():
        tcgen05_alloc[Int32(cta_group)](ptr_tmem_addr, max_tmem_cols)
        syncwarp()
        # non blocking, arrives and proceeds
        named_barrier_arrive[Int32(MMA_THREADS + EPILOGUE_THREADS)](1)

        var tmem_addr = ptr_tmem_addr[0]

        while not work_info.is_done():
            if (
                not work_info.is_valid()
                or expert_ids[Int(scheduler.current_group_idx)] < 0
            ):
                work_info = scheduler.fetch_next_work()
                continue
            # scheduler fetch next work
            var next_work_info = scheduler.fetch_next_work()
            # DO MMA
            if elect_one_cta:
                var accum_index = accum_pipeline_producer_state.index()
                var accum_phase = accum_pipeline_producer_state.phase()

                accum_empty_mbar[accum_index].wait(accum_phase)
                var tmem_offset = tmem_addr + (
                    accum_index * UInt32(stage_stride_cols)
                )

                for i in range(num_iters):
                    consumer_main_loop[
                        block_tile_shape=block_tile_shape,
                        mma_shape=mma_shape,
                        cta_group=cta_group,
                        cluster_shape=Index(
                            cluster_shape[0], cluster_shape[1], cluster_shape[2]
                        ),
                    ](
                        tmem_offset,
                        a_smem_tt,
                        b_smem_tt,
                        mma_mbar,
                        tma_mbar,
                        consumer_phase,
                        mma_op,
                        elect_one_warp,
                        i,
                    )
                    consumer_phase.step()

                # mma arrive multicast will track completion of all mma prior to this barrier.
                if elect_one_sync():
                    comptime if cta_group == 2:
                        mma_arrive_multicast[cta_group](
                            accum_full_mbar + accum_index,
                            UInt16(mma_complete_mask),
                        )
                    else:
                        mma_arrive(accum_full_mbar + accum_index)

                accum_pipeline_producer_state.step()
            work_info = next_work_info

        tcgen05_release_allocation_lock[Int32(cta_group)]()

        # wait for epilogue to finish
        tmem_dealloc_mbar[].wait()

        tcgen05_dealloc[Int32(cta_group)](tmem_addr, max_tmem_cols)

    if WarpRole.is_epilogue():
        named_barrier[Int32(MMA_THREADS + EPILOGUE_THREADS)](1)
        var tmem_addr = ptr_tmem_addr[0]

        # while work_info.is_valid():
        while not work_info.is_done():
            if not work_info.is_valid():
                work_info = scheduler.fetch_next_work()
                continue

            if expert_ids[Int(scheduler.current_group_idx)] < 0:
                comptime if elementwise_lambda_fn:
                    # An epilogue owns every store and `c_ptr` may be dangling,
                    # so zero the inactive group THROUGH the epilogue (accum==0)
                    # rather than storing zeros straight to C. Matches the naive
                    # path; see `zero_output_epilogue`.
                    zero_output_epilogue[
                        c_type,
                        c_smem_layout=Layout.row_major(
                            output_tile_shape[0], output_tile_shape[1]
                        ),
                        block_tile_shape=block_tile_shape,
                        mma_shape=mma_shape,
                        c_static_N=expert_m,
                        c_swizzle=c_swizzle,
                        cta_group=cta_group,
                        num_output_warps=num_output_warps,
                        elementwise_lambda_fn=elementwise_lambda_fn.value(),
                        transpose_c=transpose_c,
                    ](
                        work_tile_coord=(Int(work_info.m), Int(work_info.n)),
                        group_end_idx=rebind[UInt32](
                            scheduler.group_offsets[
                                Int(scheduler.current_group_idx + 1)
                            ]
                        ),
                    )
                else:
                    # c_stride == c_N == expert_m for contiguous row-major C.
                    zero_output[
                        output_tile_shape=output_tile_shape,
                        c_stride=expert_m,
                        c_N=expert_m,
                    ](
                        c_ptr,
                        (work_info.m, work_info.n),
                        rebind[UInt32](
                            scheduler.group_offsets[
                                Int(scheduler.current_group_idx + 1)
                            ]
                        ),
                    )
                work_info = scheduler.fetch_next_work()
                continue
            # WAIT FOR MMA TO FINISH AND STORE RESULT
            # scheduler fetch next work
            multi_stage_store_C[
                c_smem_layout=Layout.row_major(
                    output_tile_shape[0], output_tile_shape[1]
                ),
                accum_type=accum_type,
                block_tile_shape=block_tile_shape,
                mma_shape=mma_shape,
                stage_stride_cols=stage_stride_cols,
                c_static_N=expert_m,
                c_swizzle=c_swizzle,
                cta_group=cta_group,
                num_output_warps=num_output_warps,
                elementwise_lambda_fn=elementwise_lambda_fn,
                transpose_c=transpose_c,
            ](
                c_smem_base,
                c_tma_op,
                c_ptr,
                accum_pipeline_consumer_state,
                accum_full_mbar,
                accum_empty_mbar,
                tmem_addr,
                work_tile_coord=(Int(work_info.m), Int(work_info.n)),
                group_end_idx=rebind[UInt32](
                    scheduler.group_offsets[
                        Int(scheduler.current_group_idx + 1)
                    ]
                ),
                elect_one_warp=elect_one_warp,
                M=mnk[0],
                N=mnk[1],
            )
            accum_pipeline_consumer_state.step()

            var next_work_info = scheduler.fetch_next_work()
            work_info = next_work_info

        comptime if cta_group == 2:
            _ = tmem_dealloc_mbar[].arrive_cluster(block_rank_in_cluster() ^ 1)

        _ = tmem_dealloc_mbar[].arrive()


def grouped_matmul_sm100_persistent[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    transpose_b: Bool,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    cta_group: Int = 1,
    num_pipeline_stages: Optional[Int] = None,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    a_plane_splits: IndexList[2] = Index(0, 0),
](
    c: TileTensor[mut=True, c_type, address_space=.GENERIC, ...],
    a: TileTensor[mut=False, a_type, address_space=.GENERIC, ...],
    a_offsets: TileTensor[mut=False, .uint32, address_space=.GENERIC, ...],
    b: TileTensor[mut=False, b_type, address_space=.GENERIC, ...],
    expert_ids: TileTensor[mut=False, .int32, address_space=.GENERIC, ...],
    expert_usage_stats: TileTensor[
        mut=False, .uint32, address_space=.GENERIC, ...
    ],
    ctx: DeviceContext,
) raises:
    """Launches the persistent grouped GEMM kernel for SM100 from host tensors.

    Swaps A and B to match the kernel's transposed-B convention, delegates
    to `_grouped_matmul_sm100_persistent` which builds TMA descriptors and
    shared-memory layouts from the matmul config, selects the pipeline
    depth from available shared memory, and enqueues the warp-specialized
    kernel on the device context.

    Parameters:
        c_type: Element type of the C output matrix (inferred).
        a_type: Element type of the A operand matrix (inferred).
        b_type: Element type of the B operand matrix (inferred).
        transpose_b: Whether B is stored transposed in global memory
            (inferred).
        config: Matmul config carrying the block tile shape, MMA shape,
            and cluster shape.
        cta_group: Number of CTAs cooperating per MMA along the M
            dimension (defaults to 1).
        num_pipeline_stages: Number of TMA producer and consumer
            pipeline stages; when None, auto-selected from available
            shared memory (defaults to None).
        a_swizzle: TMA swizzle mode applied to A shared-memory tiles
            (defaults to `SWIZZLE_128B`).
        b_swizzle: TMA swizzle mode applied to B shared-memory tiles
            (defaults to `SWIZZLE_128B`).
        elementwise_lambda_fn: Optional elementwise epilogue applied to
            stored C fragments (defaults to None).
        a_plane_splits: Per-plane split sizes for fused LoRA QKV A-plane
            row offsetting; `(0, 0)` disables it (defaults to `(0, 0)`).

    Args:
        c: Output C tile tensor in generic address space.
        a: A operand tile tensor (activations); swapped to B by the
            kernel's transposed-B convention.
        a_offsets: Per-expert A offset prefix sum tile tensor.
        b: B operand tile tensor (weights); swapped to A by the
            kernel's transposed-B convention.
        expert_ids: Per-group expert id tile tensor; negative entries
            skip the tile.
        expert_usage_stats: Per-expert usage stats tile tensor; index 1
            holds the active expert count.
        ctx: Device context used to enqueue the kernel.
    """
    # swapAB by default
    comptime num_experts = b.static_shape[0]
    comptime M = b.static_shape[1]
    comptime K = b.static_shape[2]

    comptime new_config = config.swapAB()

    _grouped_matmul_sm100_persistent[
        c_type=c_type,
        a_type=b_type,
        b_type=a_type,
        transpose_b=transpose_b,
        config=new_config,
        num_experts=num_experts,
        expert_m=M,
        K=K,
        cta_group=cta_group,
        num_pipeline_stages=num_pipeline_stages,
        transpose_c=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        elementwise_lambda_fn=elementwise_lambda_fn,
        # Plane-split boundaries (output-column element offsets) for the optional
        # activation plane select, forwarded unchanged across the swapAB boundary.
        a_plane_splits=a_plane_splits,
    ](
        c.ptr.as_unsafe_any_origin(),
        b.ptr.as_unsafe_any_origin(),  # weights (a after swapAB)
        expert_ids.ptr.as_unsafe_any_origin(),
        a.ptr.as_unsafe_any_origin(),  # activations (b after swapAB)
        a_offsets.ptr.as_unsafe_any_origin(),
        expert_usage_stats.ptr.as_unsafe_any_origin(),
        Int(c.dim[0]()),
        # Activation descriptor row extent (`b_desc_rows`): the activation's own
        # row count. Equals `c.dim[0]` for a normal grouped matmul; the LoRA-B
        # QKV expand passes a `[3M, R]` activation so this is `3M`.
        Int(a.dim[0]()),
        ctx,
    )


def _grouped_matmul_sm100_persistent[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    transpose_b: Bool,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    num_experts: Int,
    expert_m: Int,
    K: Int,
    cta_group: Int = 1,
    num_pipeline_stages: Optional[Int] = None,
    transpose_c: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    a_plane_splits: IndexList[2] = Index(0, 0),
](
    c_ptr: UnsafePointer[Scalar[c_type], MutAnyOrigin],
    a_ptr: UnsafePointer[Scalar[a_type], ImmutAnyOrigin],
    expert_ids: UnsafePointer[Int32, ImmutAnyOrigin],
    b_ptr: UnsafePointer[Scalar[b_type], ImmutAnyOrigin],
    b_offsets: UnsafePointer[UInt32, ImmutAnyOrigin],
    expert_usage_stats: UnsafePointer[UInt32, ImmutAnyOrigin],
    M_runtime: Int,
    b_desc_rows: Int,
    ctx: DeviceContext,
) raises:
    comptime assert transpose_b, "Only support transposed B"

    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    comptime BM = MMA_M // cta_group
    comptime BN = MMA_N // cta_group
    comptime BK = config.block_tile_shape[2]

    comptime assert (MMA_M != 128) or (
        MMA_N % 32 == 0
    ), "if MMA_M is 128, then MMA_N must be a multiple of 32"

    comptime cluster_shape = config.cluster_shape

    # Validate compile-time shape params.
    comptime assert expert_m != 0 and K != 0, "expert_m and K must be non-zero"

    # TMA requires the global stride (K * sizeof) to be a multiple of
    # 16 bytes. When it is not, the kernel uses CUDA core copies
    # instead. We still create TMA descriptors (to satisfy the type
    # system) but use BK as the fake K so the stride is large enough.
    comptime use_tma = (K * size_of[a_type]()) % 16 == 0
    comptime tma_K = K if use_tma else BK

    # Real gmem layouts with actual K (used by kernel for CUDA core path).
    comptime a_gmem_layout = Layout(
        IntTuple(num_experts * expert_m, K), IntTuple(K, 1)
    )
    comptime b_gmem_layout = Layout(IntTuple(UNKNOWN_VALUE, K), IntTuple(K, 1))
    var a_gmem = LayoutTensor[a_type, a_gmem_layout, ImmutAnyOrigin](a_ptr)
    var b_gmem = LayoutTensor[b_type, b_gmem_layout, ImmutAnyOrigin](
        b_ptr,
        RuntimeLayout[b_gmem_layout](Index(b_desc_rows, K), Index(K, 1)),
    )

    # TMA layouts with tma_K (may be padded when use_tma=False).
    comptime a_tma_layout = Layout(
        IntTuple(num_experts * expert_m, tma_K), IntTuple(tma_K, 1)
    )
    comptime b_tma_layout = Layout(
        IntTuple(UNKNOWN_VALUE, tma_K), IntTuple(tma_K, 1)
    )
    comptime c_layout = Layout(
        IntTuple(UNKNOWN_VALUE, expert_m), IntTuple(expert_m, 1)
    )

    # TMA descriptor creation uses tma_K layouts.
    var a_device = LayoutTensor[a_type, a_tma_layout, ImmutAnyOrigin](a_ptr)
    # The activation (post-swapAB `b`) descriptor is sized from the activation's
    # own row extent `b_desc_rows`, not `M_runtime`. For a normal grouped matmul
    # the two are equal; the LoRA-B QKV expand passes a `[3M, R]` planar activation
    # (`b_desc_rows == 3M`) so the `a_plane_splits` plane shifts stay in bounds.
    var b_device = LayoutTensor[b_type, b_tma_layout, ImmutAnyOrigin](
        b_ptr,
        RuntimeLayout[b_tma_layout](Index(b_desc_rows, tma_K), Index(tma_K, 1)),
    )
    # When an elementwise epilogue owns every store, the kernel never writes to
    # `c_ptr` through the C TMA descriptor (see `multi_stage_store_C` /
    # `zero_output_epilogue`), so the caller is allowed to pass a dangling
    # `c_ptr`. But `create_tensor_tile` below still calls `cuTensorMapEncodeTiled`
    # on the descriptor's base address, which rejects a dangling global pointer
    # with `CUDA_ERROR_INVALID_VALUE`. Point the descriptor at a valid 1-element
    # scratch allocation in that case: the descriptor only needs a valid,
    # aligned base to encode (its extent is never dereferenced). The descriptor
    # type is identical in both branches (only the runtime base pointer differs),
    # so this does not create the scoped-reference issue that branching the
    # descriptor's static type would.
    # No-epilogue callers (e.g. MoE `grouped_matmul_ragged`) emit NO extra host
    # op here: `c_desc_scratch` stays `None` and the descriptor uses `c_ptr`
    # directly, exactly as before this change.
    comptime has_epilogue = Bool(elementwise_lambda_fn)
    var c_desc_scratch = Optional[DeviceBuffer[c_type]](None)
    var c_desc_ptr = c_ptr
    comptime if has_epilogue:
        c_desc_scratch = ctx.enqueue_create_buffer[c_type](1)
        c_desc_ptr = c_desc_scratch.value().unsafe_ptr().as_unsafe_any_origin()
    var c_device = LayoutTensor[
        c_type,
        c_layout,
        MutAnyOrigin,
    ](
        c_desc_ptr,
        RuntimeLayout[c_layout](Index(M_runtime, expert_m), Index(expert_m, 1)),
    )

    var M = M_runtime
    var N = expert_m
    if M == 0:
        return

    var a_tma_op = create_tensor_tile[
        Index(BM // cluster_shape[1], BK), swizzle_mode=a_swizzle
    ](ctx, a_device)

    var b_tma_op = create_tensor_tile[
        Index(
            BN // (cluster_shape[0] // cta_group), BK
        ) if transpose_b else Index(BK, BN // (cluster_shape[0] // cta_group)),
        swizzle_mode=b_swizzle,
    ](ctx, b_device)

    # If MMA_M is 256, the warps read the entire MMA_N.
    # That MMA_N to be multiple of 32 for me to use large N dim on C buf write out
    # If MMA_M is 128, the warps read 1/2 of MMA_N (BN), so now *that* has to be multiple of 32
    # Otherwise, we just use 16
    comptime width = 32 if (MMA_M == 256 and MMA_N % 32 == 0) or (
        MMA_M == 128 and BN % 32 == 0
    ) else 16
    comptime output_tile_shape = Index(
        128, width
    ) if not transpose_c else Index(width, 128)
    comptime split_tile_shape = Index(64, width) if not transpose_c else Index(
        width, 64
    )
    comptime c_tma_tile_shape = output_tile_shape if MMA_M == 256 else split_tile_shape
    comptime c_swizzle = TensorMapSwizzle.SWIZZLE_32B if transpose_c else (
        TensorMapSwizzle.SWIZZLE_64B if width
        == 32 else TensorMapSwizzle.SWIZZLE_32B
    )
    # transpose_c => MMA_M == 256 is the same as (not transpose_c) or MMA_M == 256
    comptime assert (
        not (transpose_c and cta_group == 2)
    ) or MMA_M == 256, "swapAB is only supported for MMA_M == 256"
    var c_tma_op = create_tensor_tile[
        c_tma_tile_shape if not transpose_c else Index(
            c_tma_tile_shape[0], c_swizzle.bytes() // size_of[c_type]()
        ),
        swizzle_mode=c_swizzle,
    ](ctx, c_device)

    # ctx.default_device_info.shared_memory_per_multiprocessor gives this magic number on B200
    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024
    comptime a_smem_bytes_per_stage = BM * BK * size_of[a_type]()
    comptime b_smem_bytes_per_stage = BN * BK * size_of[b_type]()
    # A and B per pipeline stage
    comptime AB_smem_per_stage = a_smem_bytes_per_stage + b_smem_bytes_per_stage
    # Support double-buffer for output stages.
    comptime num_output_stages = 2

    comptime c_smem_bytes = output_tile_shape[0] * output_tile_shape[
        1
    ] * num_output_stages * size_of[c_type]()

    comptime MBAR_BYTES = size_of[Int64]()  # 8 bytes per barrier
    comptime CLC_RESPONSE_BYTES = size_of[Int128]()  # 16 bytes per response
    comptime TMEM_ADDR_BYTES = size_of[
        Int32
    ]()  # 4 bytes or 32 bits for tensor memory address
    # the 'N' dimension of tensor memory is 512
    comptime TMEM_N = 512
    # the maximum different number of mma's that can be run in parallel is TMEM_N/MMA_N
    comptime max_accum_pipeline_stages = TMEM_N // next_power_of_two(MMA_N)
    # Mainloop barrier
    comptime accum_full_mbar_bytes = MBAR_BYTES * max_accum_pipeline_stages
    comptime accum_empty_mbar_bytes = MBAR_BYTES * max_accum_pipeline_stages

    comptime tmem_addr_bytes = TMEM_ADDR_BYTES
    comptime tmem_dealloc_mbar_bytes = MBAR_BYTES

    comptime tmem_writeout_smem = c_smem_bytes + tmem_addr_bytes + tmem_dealloc_mbar_bytes
    comptime accum_smem = accum_full_mbar_bytes + accum_empty_mbar_bytes
    comptime smem_leftover = (b200_smem) - (accum_smem + tmem_writeout_smem)

    comptime tma_mbar_bytes_per_stage = MBAR_BYTES
    comptime mma_mbar_bytes_per_stage = MBAR_BYTES

    comptime producer_consumer_smem_per_stage = (
        AB_smem_per_stage + tma_mbar_bytes_per_stage + mma_mbar_bytes_per_stage
    )

    comptime max_pipeline_stages: Int = (
        smem_leftover // producer_consumer_smem_per_stage
    )

    comptime assert (
        max_pipeline_stages >= 1
    ), "Max pipeline stages must be at least 1"

    comptime if num_pipeline_stages:
        comptime assert (
            num_pipeline_stages.value() <= max_pipeline_stages
        ), "Pipeline stage must be less than or equal to max pipeline stages"

    comptime pipeline_stage = num_pipeline_stages.value() if num_pipeline_stages else max_pipeline_stages
    comptime producer_consumer_smem = producer_consumer_smem_per_stage * pipeline_stage

    comptime smem_size = (
        # clc_smem + accum_smem + producer_consumer_smem + tmem_writeout_smem
        accum_smem
        + producer_consumer_smem
        + tmem_writeout_smem
    )

    comptime kernel = blackwell_tma_umma_warp_specialized_kernel[
        a_type,
        b_type,
        c_type,
        expert_m,
        type_of(a_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).rank,
        type_of(b_tma_op).tile_shape,
        type_of(b_tma_op).desc_shape,
        type_of(c_tma_op).rank,
        type_of(c_tma_op).tile_shape,
        type_of(c_tma_op).desc_shape,
        config.block_tile_shape,
        config.mma_shape,
        transpose_b=transpose_b,
        cluster_shape=StaticTuple[Int32, 3](
            Int32(cluster_shape[0]),
            Int32(cluster_shape[1]),
            Int32(cluster_shape[2]),
        ),
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        c_swizzle=c_swizzle,
        cta_group=cta_group,
        num_pipeline_stages=pipeline_stage,
        num_accum_pipeline_stages=max_accum_pipeline_stages,
        num_output_stages=num_output_stages,
        output_tile_shape=output_tile_shape,
        transpose_c=transpose_c,
        use_tma=use_tma,
        K_actual=K,
        a_gmem_layout=a_gmem_layout,
        b_gmem_layout=b_gmem_layout,
        elementwise_lambda_fn=elementwise_lambda_fn,
        a_plane_splits=a_plane_splits,
    ]

    comptime assert (
        cluster_shape[1] == 1
    ), "cluster_shape[1] must be 1. Got " + String(cluster_shape[1])

    var grid_dim = (
        B200.sm_count,
        1,
        1,
    )

    var mnk = StaticTuple[UInt32, 3](UInt32(M), UInt32(N), UInt32(K))

    ctx.enqueue_function[kernel](
        expert_usage_stats,
        a_tma_op,
        expert_ids,
        b_tma_op,
        b_offsets,
        c_tma_op,
        c_ptr,
        mnk,
        a_gmem,
        b_gmem,
        grid_dim=grid_dim,
        # 1 TMA, 1 MMA, 4 EPILOGUE warps
        block_dim=(32 * 6),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_size)
        ),
    )

    # Keep the C-descriptor scratch buffer alive until the kernel launch above
    # has been enqueued (`c_device`/`c_tma_op` alias its pointer when an
    # epilogue owns all stores). Only exists in the epilogue case.
    comptime if has_epilogue:
        _ = c_desc_scratch^
