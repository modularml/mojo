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

from std.builtin._closure import __ownership_keepalive
from std.hashlib import default_comp_time_hasher
from std.math import align_up, ceildiv
from std.math.uutils import udivmod
from std.memory import bitcast
from std.sys import argv, size_of

import linalg.matmul.vendor.blas as vendor_blas
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
from max.gpu import block_id_in_cluster, lane_id
from max.gpu import warp_id as get_warp_id
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
from internal_utils import assert_almost_equal
from layout import (
    Layout,
    LayoutTensor,
    lt_to_tt,
)
from layout._utils import ManagedLayoutTensor
from layout.layout_tensor import LayoutTensorIter
from layout.swizzle import Swizzle, make_swizzle
from layout.tensor_core_async import tile_layout_k_major, tile_layout_mn_major
from layout.tma_async import (
    _idx_product,
    create_tensor_tile,
    PipelineState,
    SharedMemBarrier,
    TMATensorTile,
)
from linalg.arch.sm100 import MmaOpSM100_SS
from linalg.matmul.gpu.sm100.tile_scheduler import TileScheduler

from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple


def is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark":
            return True
    return False


@fieldwise_init
struct WarpRole(TrivialRegisterPassable):
    var _role: Int32

    comptime Mma = Self(6)
    comptime MainLoad = Self(5)
    comptime Scheduler = Self(4)
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

    @staticmethod
    @always_inline
    def is_scheduler() -> Bool:
        return Self.Scheduler == get_warp_id()


@always_inline
def load_AB[
    a_type: DType,
    b_type: DType,
    a_tma_rank: Int,
    b_tma_rank: Int,
    a_tile_shape: IndexList[a_tma_rank],
    b_tile_shape: IndexList[b_tma_rank],
    a_desc_shape: IndexList[a_tma_rank],
    b_desc_shape: IndexList[b_tma_rank],
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    num_pipeline_stages: Int,
    /,
    *,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int = 1,
](
    a_tma_op: TMATensorTile[a_type, a_tma_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[b_type, b_tma_rank, b_tile_shape, b_desc_shape],
    a_smem: LayoutTensorIter[
        a_type,
        a_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ],
    b_smem: LayoutTensorIter[
        b_type,
        b_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ],
    mma_mbar: MutPointer[SharedMemBarrier, address_space=.SHARED, _],
    tma_mbar: MutPointer[SharedMemBarrier, address_space=.SHARED, _],
    producer_phase: PipelineState[num_pipeline_stages],
    peer_cta_coord: Tuple[Int, Int, Int],
    work_tile_coord: Tuple[Int, Int],
    a_multicast_mask: UInt16,
    b_multicast_mask: UInt16,
    iter_idx: Int,
    elect_one_cta: Bool,
):
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]

    comptime a_expected_bytes = a_smem_layout.size() * size_of[a_type]()
    comptime b_expected_bytes = b_smem_layout.size() * size_of[b_type]()
    # Leader CTAs expect SMEM from itself and their peers
    comptime expected_bytes = cta_group * (a_expected_bytes + b_expected_bytes)

    comptime a_tma_load_size = _idx_product[a_tma_rank, a_desc_shape]()
    comptime b_tma_load_size = _idx_product[b_tma_rank, b_desc_shape]()
    comptime a_tma_rows = a_desc_shape[0]
    comptime b_tma_rows = b_desc_shape[0]

    var stage = producer_phase.index()
    var phase = producer_phase.phase()
    mma_mbar[stage].wait(phase)

    var a_gmem_slice_coord = (
        peer_cta_coord[2] * a_tma_rows + work_tile_coord[0] * BM
    )
    var b_gmem_slice_coord = (
        peer_cta_coord[1] * b_tma_rows
        + peer_cta_coord[0] * BN
        + work_tile_coord[1] * MMA_N
    )

    var a_smem_tile = a_smem.next(stage)[]
    var b_smem_tile = b_smem.next(stage)[]

    var a_smem_slice = type_of(a_smem_tile)(
        a_smem_tile.ptr + peer_cta_coord[2] * a_tma_load_size
    )
    var b_smem_slice = type_of(b_smem_tile)(
        b_smem_tile.ptr + peer_cta_coord[1] * b_tma_load_size
    )

    if elect_one_sync():
        if elect_one_cta:
            tma_mbar[stage].expect_bytes(Int32(expected_bytes))

        a_tma_op.async_multicast_load[cta_group](
            a_smem_slice,
            tma_mbar[stage],
            (iter_idx * BK, a_gmem_slice_coord),
            a_multicast_mask,
        )

        b_tma_op.async_multicast_load[cta_group](
            b_smem_slice,
            tma_mbar[stage],
            (iter_idx * BK, b_gmem_slice_coord),
            b_multicast_mask,
        )


@always_inline
def consumer_main_loop[
    accum_type: DType,
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_smem_layout: Layout,
    b_smem_layout: Layout,
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
](
    tmem_addr: UInt32,
    a_smem_iter: LayoutTensorIter[
        a_type,
        a_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ],
    b_smem_iter: LayoutTensorIter[
        b_type,
        b_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ],
    mma_mbar: MutPointer[SharedMemBarrier, address_space=.SHARED, _],
    tma_mbar: MutPointer[SharedMemBarrier, address_space=.SHARED, _],
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
    iter_idx: Int,
):
    var stage = consumer_phase.index()
    var phase = consumer_phase.phase()

    tma_mbar[stage].wait(phase)

    var a_smem_tile = a_smem_iter.next(stage)[]
    var b_smem_tile = b_smem_iter.next(stage)[]
    # Compose TMEM address: accum stage encoded in column field with stride in columns.
    if elect_one_sync():
        mma_op.mma(
            lt_to_tt(a_smem_tile),
            lt_to_tt(b_smem_tile),
            tmem_addr,
            init_c=(iter_idx == 0),  # Initialize C on first iteration
        )

        mma_op.commit(mma_mbar + stage)


@always_inline
def stsm_helper[
    swizzle: Swizzle,
    vec_dtype: DType,
    vec_size: Int,
](
    vec: Array[Scalar[vec_dtype], vec_size],
    dst: LayoutTensor[mut=True, _, _, address_space=.SHARED, ...],
):
    # Number of elements in one row per stsmx4 tile, a row is 32B.
    comptime stsmx4_row_size = 32 // size_of[dst.dtype]()
    # Number of elements owned by each lane, each lane has 16B
    comptime stsmx4_lane_size = 16 // size_of[dst.dtype]()
    # TODO: constrain the shared memory layout to be 2D row-major.
    # E.g. dst layout can be (16, 16) : (32, 1), which is tiled from
    # row-major(16, 32). The map should use tile's stride to calculate
    # the dst row offset.
    comptime stride0 = dst.layout.stride[0].value()
    comptime shape0 = dst.layout.shape[1].value()

    var lane = lane_id()
    var stsm_lane_offset = (lane & 15) * stride0 + (lane >> 4) * 8

    # Assume the dst tile has 16 rows and only use stsm in N dim.
    comptime for i in range(shape0 // stsmx4_row_size):
        comptime n_offset = i * stsmx4_row_size
        var offset = swizzle(stsm_lane_offset + n_offset)
        var v = SIMD[dst.dtype, stsmx4_lane_size]()

        comptime for k in range(stsmx4_lane_size // 2):
            var pair = SIMD[vec_dtype, 2](
                vec[i * stsmx4_lane_size + 2 * k],
                vec[i * stsmx4_lane_size + 2 * k + 1],
            )
            var casted = pair.cast[dst.dtype]()
            v[2 * k] = casted[0]
            v[2 * k + 1] = casted[1]
        st_matrix[simd_width=4](dst.ptr + offset, bitcast[.float32, 4](v))


@always_inline
def multi_stage_store_C[
    c_type: DType,
    c_smem_layout: Layout,
    c_tma_rank: Int,
    c_tile_shape: IndexList[c_tma_rank],
    c_desc_shape: IndexList[c_tma_rank],
    num_accum_pipeline_stages: Int,
    /,
    *,
    accum_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    stage_stride_cols: Int,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    cta_group: Int = 1,
    num_output_warps: Int = 4,
    max_tmem_cols: Int = 512,
](
    c_iter: LayoutTensorIter[
        c_type,
        c_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ],
    c_tma_op: TMATensorTile[c_type, c_tma_rank, c_tile_shape, c_desc_shape],
    accum_pipeline_consumer_state: PipelineState[num_accum_pipeline_stages],
    accum_full_mbar: MutPointer[SharedMemBarrier, address_space=.SHARED, _],
    accum_empty_mbar: MutPointer[SharedMemBarrier, address_space=.SHARED, _],
    tmem_addr: UInt32,
    work_tile_coord: Tuple[Int, Int],
    elect_one_warp: Bool,
):
    # WAIT FOR MMA TO FINISH AND STORE RESULT
    # scheduler fetch next work
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]

    comptime num_m_mmas = BM // (mma_shape[0] // cta_group)
    comptime num_n_mmas = BN // (mma_shape[1] // cta_group)

    comptime assert num_m_mmas == 1 and num_n_mmas == 1

    # we break down the output tile BM x MMA_N to BM x stageN tiles
    # and output one tile per stage.
    # stage N is 32
    comptime stageN = c_smem_layout.shape[1].value()
    # so num stages is usually 256 by 32 is 8
    comptime num_stages = MMA_N // stageN
    comptime tmem_cell_bytes = 4
    comptime data_paths = 16
    comptime bits = 256
    # every element in tmem is 4 bytes, so bits being 256 means 8 elements stored across N
    # repeated 4 times is 8*4 = 32, enough to move elements into the width of our 128x32 tile
    comptime rep = stageN // (bits // 32)

    # stmatrix related
    comptime stsmx4N_bytes = 32
    comptime stsmx4N = stsmx4N_bytes // size_of[c_type]()  # 16
    comptime stsmx4_size_per_lane = (16 * stsmx4N) // WARP_SIZE  # 8
    comptime swizzle = make_swizzle[c_type, TensorMapSwizzle.SWIZZLE_64B]()

    var warp_id = get_warp_id()

    # before i start the process of transferring over num_stages * stageN= MMA_N from tensor memory to global, i should wait
    # on the accum_full_mbar barrier
    var index = accum_pipeline_consumer_state.index()
    var phase = accum_pipeline_consumer_state.phase()
    accum_full_mbar[index].wait(phase)
    # this is the column offset for all the stages of THIS load, where one load takes (num_stages iterations)
    var tmem_offset = index * UInt32(stage_stride_cols) + tmem_addr

    comptime for stage in range(num_stages):
        # column offset, moving right by 32 columns each time, since each num_stage stores two, 16 column submatrices
        # MMA has result in 32 rows per warp's data paths.
        # upper_frag is for rows 0-15, lower is for 16-31.
        var stage_tmem_addr = tmem_offset + UInt32((stage * stageN))
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
        var c_smem_tile = c_iter.next(stage % 2)[]
        var c_smem_warp_tile = c_smem_tile.tile[32, stageN](warp_id, 0)

        # Pack the upper frag to shared memory
        comptime frag_width = rep * data_paths * (bits // 32) // WARP_SIZE
        stsm_helper[swizzle](
            rebind[Array[Scalar[accum_type], frag_width]](upper_frag),
            c_smem_warp_tile.tile[16, stageN](0, 0),
        )
        stsm_helper[swizzle](
            rebind[Array[Scalar[accum_type], frag_width]](lower_frag),
            c_smem_warp_tile.tile[16, stageN](1, 0),
        )

        # Guard the write to shared memory is done.
        named_barrier[Int32(num_output_warps * WARP_SIZE)]()
        var lane = lane_id()
        if elect_one_warp and lane == 0:
            fence_async_view_proxy()
            c_tma_op.async_store(
                c_smem_tile,
                (
                    work_tile_coord[1] * MMA_N + stage * stageN,
                    work_tile_coord[0] * BM,
                ),
            )
            c_tma_op.commit_group()
            # c_tma_op.wait_group[0]()

            # Keep one tma store in fly
            comptime if stage < num_stages - 1:
                c_tma_op.wait_group[1]()
            # Last stage guard all tma store to finish
            else:
                c_tma_op.wait_group[0]()

        comptime if stage > 0 and stage < num_stages - 1:
            # Guard the tma read from shared memory is done.
            # E.g. stage = 1, this guards the TMA store using buffer 0 is done.
            named_barrier[Int32(num_output_warps * WARP_SIZE)]()


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
def kernel_8[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_tma_rank: Int,
    b_tma_rank: Int,
    c_tma_rank: Int,
    a_tile_shape: IndexList[a_tma_rank],
    b_tile_shape: IndexList[b_tma_rank],
    c_tile_shape: IndexList[c_tma_rank],
    a_desc_shape: IndexList[a_tma_rank],
    b_desc_shape: IndexList[b_tma_rank],
    c_desc_shape: IndexList[c_tma_rank],
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    num_pipeline_stages: Int,
    num_clc_pipeline_stages: Int,
    num_accum_pipeline_stages: Int,
    num_output_stages: Int = 2,
    output_tile_shape: IndexList[2] = Index(128, 32),
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    cta_group: Int = 2,
](
    a_tma_op: TMATensorTile[a_type, a_tma_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[b_type, b_tma_rank, b_tile_shape, b_desc_shape],
    c_tma_op: TMATensorTile[c_type, c_tma_rank, c_tile_shape, c_desc_shape],
    cluster_dim: StaticTuple[Int32, 3],
    num_iters_dev: Int32,
):
    var num_iters = Int(num_iters_dev)
    comptime num_output_warps = 4

    comptime SCHEDULER_THREADS = WARP_SIZE
    comptime TMA_LOAD_THREADS = WARP_SIZE
    comptime MMA_THREADS = WARP_SIZE
    comptime EPILOGUE_THREADS = num_output_warps * WARP_SIZE
    comptime CLUSTER_SIZE = cluster_shape[0] * cluster_shape[1]
    comptime clc_producer_arv_count = 1
    comptime clc_consumer_arv_count = Int32(
        SCHEDULER_THREADS
    ) + CLUSTER_SIZE * Int32(
        (TMA_LOAD_THREADS + MMA_THREADS + EPILOGUE_THREADS)
    )

    # For ld from TMEM, use same per-stage stride in column field.
    comptime TMEM_N = 512
    comptime stage_stride_cols = TMEM_N // num_accum_pipeline_stages

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

    comptime a_tma_load_size = _idx_product[a_tma_rank, a_desc_shape]()
    comptime b_tma_load_size = _idx_product[b_tma_rank, b_desc_shape]()
    comptime a_tma_rows = a_desc_shape[0]
    comptime b_tma_rows = b_desc_shape[0]
    comptime c_smem_layout = Layout.row_major(BM, MMA_N)

    # keep the physical SMEM buffer BM x MMA_N

    comptime a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    comptime b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()

    var base_ptr_smem = external_memory[
        Scalar[a_type],
        address_space=.SHARED,
        alignment=128,
    ]()

    comptime a_smem_size = a_smem_layout.size() * num_pipeline_stages
    comptime b_smem_size = b_smem_layout.size() * num_pipeline_stages
    comptime c_smem_size = output_tile_shape[0] * output_tile_shape[
        1
    ] * num_output_stages

    var a_smem_base = base_ptr_smem
    var b_smem_base = (a_smem_base + a_smem_size).bitcast[Scalar[b_type]]()
    var c_smem_base = (b_smem_base + b_smem_size).bitcast[Scalar[c_type]]()

    var a_smem = LayoutTensorIter[
        a_type,
        a_smem_layout,
        address_space=.SHARED,
        alignment=128,
    ](
        a_smem_base.as_unsafe_any_origin(),
        a_smem_size,
    )

    var b_smem = LayoutTensorIter[
        b_type,
        b_smem_layout,
        address_space=.SHARED,
        alignment=128,
    ](
        b_smem_base.as_unsafe_any_origin(),
        b_smem_size,
    )

    var c_smem_iter = LayoutTensorIter[
        c_type,
        Layout.row_major(output_tile_shape[0], output_tile_shape[1]),
        address_space=.SHARED,
        alignment=128,
    ](c_smem_base.as_unsafe_any_origin(), c_smem_size)

    var smem_pool = (c_smem_base + c_smem_size).bitcast[Int64]()

    var tma_mbar_ptr = smem_pool
    var mma_mbar_ptr = tma_mbar_ptr + num_pipeline_stages
    var accum_full_mbar_ptr = mma_mbar_ptr + num_pipeline_stages
    var accum_empty_mbar_ptr = accum_full_mbar_ptr + num_accum_pipeline_stages

    var clc_full_mbar_ptr = accum_empty_mbar_ptr + num_accum_pipeline_stages
    var clc_empty_mbar_ptr = clc_full_mbar_ptr + num_clc_pipeline_stages
    var clc_throttle_full_mbar_ptr = (
        clc_empty_mbar_ptr + num_clc_pipeline_stages
    )
    var clc_throttle_empty_mbar_ptr = (
        clc_throttle_full_mbar_ptr + num_clc_pipeline_stages
    )

    var clc_response_ptr = (
        clc_throttle_empty_mbar_ptr + num_clc_pipeline_stages
    ).bitcast[Int128]()

    var tmem_dealloc_mbar_ptr = (
        clc_response_ptr + num_clc_pipeline_stages
    ).bitcast[Int64]()

    var ptr_tmem_addr = (tmem_dealloc_mbar_ptr + 1).bitcast[UInt32]()

    var tma_mbar = tma_mbar_ptr.bitcast[SharedMemBarrier]()
    var mma_mbar = mma_mbar_ptr.bitcast[SharedMemBarrier]()
    var accum_full_mbar = accum_full_mbar_ptr.bitcast[SharedMemBarrier]()
    var accum_empty_mbar = accum_empty_mbar_ptr.bitcast[SharedMemBarrier]()
    var clc_response = clc_response_ptr.bitcast[UInt128]()
    var clc_full_mbar = clc_full_mbar_ptr.bitcast[SharedMemBarrier]()
    var clc_empty_mbar = clc_empty_mbar_ptr.bitcast[SharedMemBarrier]()
    var tmem_dealloc_mbar = tmem_dealloc_mbar_ptr.bitcast[SharedMemBarrier]()
    var clc_throttle_full_mbar = clc_throttle_full_mbar_ptr.bitcast[
        SharedMemBarrier
    ]()
    var clc_throttle_empty_mbar = clc_throttle_empty_mbar_ptr.bitcast[
        SharedMemBarrier
    ]()

    comptime accum_type = get_accum_type[a_type]()

    var warp_id = get_warp_id()
    var elect_one_warp = warp_id == 0
    var elect_one_thread = elect_one_sync_with_mask()
    var elect_one_cta = block_rank_in_cluster() % 2 == 0
    var is_first_cta_in_cluster = block_rank_in_cluster() == 0
    comptime max_tmem_cols = 512

    if elect_one_warp and elect_one_thread:
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

    comptime for i in range(num_clc_pipeline_stages):
        clc_full_mbar[i].init(clc_producer_arv_count)
        clc_empty_mbar[i].init(clc_consumer_arv_count)
        clc_throttle_full_mbar[i].init(Int32(clc_throttle_producer_arv_count))
        clc_throttle_empty_mbar[i].init(Int32(clc_throttle_consumer_arv_count))

    fence_mbarrier_init()
    cluster_sync()

    var consumer_phase = PipelineState[num_pipeline_stages]()
    var producer_phase = PipelineState[num_pipeline_stages](0, 1, 0)

    var clc_pipe_producer_state = PipelineState[num_clc_pipeline_stages](
        0, 1, 0
    )
    var clc_pipe_consumer_state = PipelineState[num_clc_pipeline_stages]()

    var clc_throttle_producer_state = PipelineState[num_clc_pipeline_stages](
        0, 1, 0
    )
    var clc_throttle_consumer_state = PipelineState[num_clc_pipeline_stages]()

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

    var scheduler = TileScheduler[
        num_stages=num_clc_pipeline_stages,
        cluster_shape=Index[dtype=.uint32](
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        block_swizzle_size=1,
    ](cluster_dim, clc_response, clc_full_mbar, clc_empty_mbar)

    var work_info = scheduler.initial_work_info()

    var rank_m = block_id_in_cluster.x
    var rank_n = block_id_in_cluster.y

    # (peer_id, mma_coord_m, mma_coord_n)
    var peer_cta_quot, peer_cta_rem = udivmod(rank_m, cta_group)
    var peer_cta_coord = (
        peer_cta_rem,
        peer_cta_quot,
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

    if WarpRole.is_main_load():
        var required_clc_query = True

        while work_info.is_valid():
            # CLC throuttle prevents each CTA from going a few waves ahead.
            if is_first_cta_in_cluster and required_clc_query:
                var index = clc_throttle_producer_state.index()
                var phase = clc_throttle_producer_state.phase()
                clc_throttle_empty_mbar[index].wait(phase)
                _ = clc_throttle_full_mbar[index].arrive()

                clc_throttle_producer_state.step()

            # DO TMA LOAD

            for i in range(num_iters):
                load_AB[
                    block_tile_shape=block_tile_shape,
                    mma_shape=mma_shape,
                    cta_group=cta_group,
                ](
                    a_tma_op,
                    b_tma_op,
                    a_smem,
                    b_smem,
                    mma_mbar,
                    tma_mbar,
                    producer_phase,
                    peer_cta_coord,
                    (Int(work_info.m), Int(work_info.n)),
                    a_multicast_mask,
                    b_multicast_mask,
                    i,
                    elect_one_cta,
                )
                producer_phase.step()

            syncwarp()
            var next_work_info = scheduler.fetch_next_work(
                work_info, clc_pipe_consumer_state
            )
            work_info = next_work_info
            clc_pipe_consumer_state.step()

        comptime for i in range(num_pipeline_stages):
            mma_mbar[producer_phase.index()].wait(producer_phase.phase())
            producer_phase.step()

    if WarpRole.is_scheduler() and is_first_cta_in_cluster:
        var required_clc_query = True

        while work_info.is_valid():
            if required_clc_query:
                var index = clc_throttle_consumer_state.index()
                var phase = clc_throttle_consumer_state.phase()
                clc_throttle_full_mbar[index].wait(phase)
                _ = clc_throttle_empty_mbar[index].arrive()

                clc_throttle_consumer_state.step()

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
        comptime for i in range(num_clc_pipeline_stages):
            clc_empty_mbar[clc_pipe_producer_state.index()].wait(
                clc_pipe_producer_state.phase()
            )
            clc_pipe_producer_state.step()

    if WarpRole.is_mma():
        tcgen05_alloc[Int32(cta_group)](ptr_tmem_addr, max_tmem_cols)
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
                        a_smem,
                        b_smem,
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
                    mma_arrive_multicast[cta_group](
                        accum_full_mbar + accum_index,
                        UInt16(mma_complete_mask),
                    )
                accum_pipeline_producer_state.step()
            work_info = next_work_info

        tcgen05_release_allocation_lock[Int32(cta_group)]()

        # wait for epilogue to finish
        tmem_dealloc_mbar[].wait()

        tcgen05_dealloc[Int32(cta_group)](tmem_addr, max_tmem_cols)

    if WarpRole.is_epilogue():
        named_barrier[Int32(MMA_THREADS + EPILOGUE_THREADS)](1)
        var tmem_addr = ptr_tmem_addr[0]

        while work_info.is_valid():
            # WAIT FOR MMA TO FINISH AND STORE RESULT
            # scheduler fetch next work
            multi_stage_store_C[
                accum_type=accum_type,
                block_tile_shape=block_tile_shape,
                mma_shape=mma_shape,
                stage_stride_cols=stage_stride_cols,
                c_swizzle=c_swizzle,
                cta_group=cta_group,
                num_output_warps=num_output_warps,
                max_tmem_cols=max_tmem_cols,
            ](
                c_smem_iter,
                c_tma_op,
                accum_pipeline_consumer_state,
                accum_full_mbar,
                accum_empty_mbar,
                tmem_addr,
                work_tile_coord=(Int(work_info.m), Int(work_info.n)),
                elect_one_warp=elect_one_warp,
            )
            accum_pipeline_consumer_state.step()

            var next_work_info = scheduler.fetch_next_work(
                work_info, clc_pipe_consumer_state
            )
            work_info = next_work_info
            clc_pipe_consumer_state.step()

        _ = tmem_dealloc_mbar[].arrive_cluster(block_rank_in_cluster() ^ 1)
        _ = tmem_dealloc_mbar[].arrive()


def blackwell_kernel_8[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    *,
    transpose_b: Bool,
    umma_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    cta_group: Int = 1,
    num_clc_pipeline_stages: Int = 2,
](
    c: LayoutTensor[c_type, c_layout, _],
    a: LayoutTensor[a_type, a_layout, _],
    b: LayoutTensor[b_type, b_layout, _],
    ctx: DeviceContext,
) raises:
    var M = c.dim[0]()
    var N = c.dim[1]()
    var K = a.dim[1]()

    comptime assert transpose_b, "Only support transposed B"

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]

    comptime MMA_M = umma_shape[0]
    comptime MMA_N = umma_shape[1]
    comptime MMA_K = umma_shape[2]

    var a_tma_op = create_tensor_tile[
        Index(Int32(BM) // cluster_shape[1], BK), swizzle_mode=a_swizzle
    ](ctx, a)

    var b_tma_op = create_tensor_tile[
        Index(
            Int32(BN) // (cluster_shape[0] // Int32(cta_group)), BK
        ) if transpose_b else Index(
            BK, Int32(BN) // (cluster_shape[0] // Int32(cta_group))
        ),
        swizzle_mode=b_swizzle,
    ](ctx, b)

    comptime output_tile_shape = Index(BM, 32)
    comptime c_swizzle = TensorMapSwizzle.SWIZZLE_64B
    var c_tma_op = create_tensor_tile[
        output_tile_shape, swizzle_mode=c_swizzle
    ](ctx, c)

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
    comptime max_accum_pipeline_stages = TMEM_N // MMA_N
    # Mainloop barrier
    comptime accum_full_mbar_bytes = MBAR_BYTES * max_accum_pipeline_stages
    comptime accum_empty_mbar_bytes = MBAR_BYTES * max_accum_pipeline_stages

    comptime clc_response_bytes = CLC_RESPONSE_BYTES * num_clc_pipeline_stages
    comptime clc_full_mbar_bytes = MBAR_BYTES * num_clc_pipeline_stages
    comptime clc_empty_mbar_bytes = MBAR_BYTES * num_clc_pipeline_stages
    comptime clc_throttle_full_mbar_bytes = MBAR_BYTES * num_clc_pipeline_stages
    comptime clc_throttle_empty_mbar_bytes = MBAR_BYTES * num_clc_pipeline_stages

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
        AB_smem_per_stage + tma_mbar_bytes_per_stage + mma_mbar_bytes_per_stage
    )

    comptime max_pipeline_stages = smem_leftover // producer_consumer_smem_per_stage

    comptime producer_consumer_smem = producer_consumer_smem_per_stage * max_pipeline_stages

    comptime smem_size = (
        clc_smem + accum_smem + producer_consumer_smem + tmem_writeout_smem
    )

    comptime kernel = kernel_8[
        a_type,
        b_type,
        c_type,
        type_of(a_tma_op).rank,
        type_of(b_tma_op).rank,
        type_of(c_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(b_tma_op).tile_shape,
        type_of(c_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).desc_shape,
        type_of(c_tma_op).desc_shape,
        block_tile_shape,
        umma_shape,
        transpose_b=transpose_b,
        cluster_shape=cluster_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        c_swizzle=c_swizzle,
        cta_group=cta_group,
        num_pipeline_stages=max_pipeline_stages,
        num_clc_pipeline_stages=num_clc_pipeline_stages,
        num_accum_pipeline_stages=max_accum_pipeline_stages,
        num_output_stages=num_output_stages,
        output_tile_shape=output_tile_shape,
    ]

    var grid_dim = (
        align_up(ceildiv(M, BM), Int(cluster_shape[0])),
        align_up(ceildiv(N, MMA_N), Int(cluster_shape[1])),
        1,
    )

    var cluster_dim = StaticTuple[Int32, 3](
        Int32(grid_dim[0]) // cluster_shape[0],
        Int32(grid_dim[1]) // cluster_shape[1],
        1,
    )

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        cluster_dim,
        Int32(K // BK),
        grid_dim=grid_dim,
        # 1 TMA, 1 MMA, 1 Scheduler, 4 EPILOGUE warps
        block_dim=(32 * 7),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_size)
        ),
    )


def test_blackwell_kernel_8[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    benchmark: Bool = False,
    M: Int = 4096,
    N: Int = 4096,
    K: Int = 4096,
](ctx: DeviceContext) raises:
    if not benchmark:
        print(
            t"in/out dtypes=({a_type}, {b_type}, {c_type})  problem shape=({M},"
            t" {N}, {K})"
            t" mma_shape={mma_shape} block_tile_shape={block_tile_shape}"
        )

    comptime a_layout = Layout.row_major(M, K)
    comptime b_layout = Layout.row_major(
        N, K
    ) if transpose_b else Layout.row_major(K, N)
    comptime c_layout = Layout.row_major(M, N)

    # Host memory allocation
    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](M * K)
    var a_host = LayoutTensor[a_type, a_layout](a_host_ptr)
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](N * K)
    var b_host = LayoutTensor[b_type, b_layout](b_host_ptr)
    var c_host_managed = ManagedLayoutTensor[c_type, c_layout](ctx)
    var c_host = c_host_managed.tensor[update=False]()
    var c_host_ref_managed = ManagedLayoutTensor[c_type, c_layout](ctx)
    var c_host_ref = c_host_ref_managed.tensor[update=False]()

    # Device memory allocation
    var a_device = ctx.enqueue_create_buffer[a_type](M * K)
    var a_device_lt = LayoutTensor[a_type, a_layout](a_device.unsafe_ptr())
    var b_device = ctx.enqueue_create_buffer[b_type](N * K)
    var b_device_lt = LayoutTensor[b_type, b_layout](b_device.unsafe_ptr())
    var c_device = ctx.enqueue_create_buffer[c_type](M * N)
    var c_device_lt = LayoutTensor[c_type, c_layout](c_device.unsafe_ptr())
    var c_device_ref = ctx.enqueue_create_buffer[c_type](M * N)
    var c_device_ref_lt = LayoutTensor[c_type, c_layout](
        c_device_ref.unsafe_ptr()
    )

    # Initialize matmul operands
    for m_idx in range(M):
        for k_idx in range(K):
            a_host[m_idx, k_idx] = Float32(k_idx).cast[a_type]()
    for n_idx in range(N):
        for k_idx in range(K):
            b_host[n_idx, k_idx] = Float32(1 if n_idx == k_idx else 0).cast[
                b_type
            ]()

    # Move operands to the Device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    ctx.enqueue_copy(c_device, c_host.ptr)
    ctx.enqueue_copy(c_device_ref, c_host_ref.ptr)

    blackwell_kernel_8[
        transpose_b=transpose_b,
        umma_shape=mma_shape,
        block_tile_shape=block_tile_shape,
        cluster_shape=cluster_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        cta_group=2,
    ](
        c_device_lt,
        a_device_lt,
        b_device_lt,
        ctx,
    )

    comptime if benchmark:
        comptime num_runs = 10000
        comptime num_warmup = 100

        @always_inline
        def run_kernel(ctx: DeviceContext) raises {imm}:
            blackwell_kernel_8[
                transpose_b=transpose_b,
                umma_shape=mma_shape,
                block_tile_shape=block_tile_shape,
                cluster_shape=cluster_shape,
                a_swizzle=a_swizzle,
                b_swizzle=b_swizzle,
                cta_group=2,
            ](
                c_device_lt,
                a_device_lt,
                b_device_lt,
                ctx,
            )

        # Warmup
        for _ in range(num_warmup):
            run_kernel(ctx)
        ctx.synchronize()
        # print("finished warmup")

        var nstime = (
            Float64(ctx.execution_time(run_kernel, num_runs)) / num_runs
        )
        var sectime = nstime * 1e-9
        var TFlop = 2.0 * Float64(M) * Float64(N) * Float64(K) * 1e-12
        # Round TFLOPS to two decimal places for cleaner output
        var tflops = TFlop / sectime
        var tflops_rounded = round(tflops, 2)
        print(
            t"{a_type}x{M}x{N}x{K}",
            sectime * 1000,
            tflops_rounded,
        )
    else:
        comptime assert a_type != .float8_e4m3fn or transpose_b, (
            "Testing is only supported for transposed_b==True when"
            " a_type==float8_e4m3fn. Add the non-transposed case if needed."
        )

        vendor_blas.matmul(
            ctx,
            c_device_ref_lt,
            a_device_lt,
            b_device_lt,
            c_row_major=True,
            transpose_b=transpose_b,
        )

        ctx.synchronize()

        ctx.enqueue_copy(c_host.ptr, c_device)
        ctx.enqueue_copy(c_host_ref.ptr, c_device_ref)
        ctx.synchronize()

        comptime rtol = 1e-2
        assert_almost_equal(
            c_host.ptr,
            c_host_ref.ptr,
            M * N,
            atol=0.0001,
            rtol=rtol,
        )
        print("\n=== TEST PASSED ===\n")

    # `a_device_lt` / `b_device_lt` are raw pointer views, so past the copies
    # above nothing else refers to the buffers backing them.
    __ownership_keepalive(a_device, b_device)


def get_dic_of_shapes(
    index: Int, dic_bro: Dict[Int, Tuple[Int, Int, Int], ...]
) -> Tuple[Int, Int, Int]:
    try:
        return dic_bro[index]
    except error:
        print("error")
        return (128, 128, 128)


def make_dic_of_shapes() -> (
    Dict[Int, Tuple[Int, Int, Int], default_comp_time_hasher]
):
    var dic = Dict[Int, Tuple[Int, Int, Int], default_comp_time_hasher]()
    dic[0] = (4096, 4096, 4096)
    return dic^


def benchmark_blackwell_matmul(ctx: DeviceContext) raises:
    comptime for swizzle in [TensorMapSwizzle.SWIZZLE_128B]:
        print("Benchmarking blackwell_matmul_tma_umma_kernel")
        print("============================================")
        print("dtype, M, N, K, time(ms), TFLOPS")

        comptime BK = 64
        comptime block_tile_shape = Index(128, 64, BK)
        comptime MMA_K = 16
        comptime umma_shape = Index(
            block_tile_shape[0] * 2, block_tile_shape[1] * 2, MMA_K
        )

        comptime c_type = DType.bfloat16
        comptime dic_of_shapes = make_dic_of_shapes()

        comptime for i in range(len(dic_of_shapes)):
            comptime shape = get_dic_of_shapes(i, dic_of_shapes)
            try:
                test_blackwell_kernel_8[
                    .bfloat16,
                    .bfloat16,
                    c_type,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    benchmark=True,
                    M=shape[0],
                    N=shape[1],
                    K=shape[2],
                ](ctx)
            except error:
                print("error")


def main() raises:
    with DeviceContext() as ctx:
        if is_benchmark():
            benchmark_blackwell_matmul(ctx)
            return

        comptime block_tile_shape = Index(128, 64, 64)
        comptime umma_shape = Index(256, 128, 16)

        test_blackwell_kernel_8[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            block_tile_shape,
            umma_shape,
            cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
            a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            M=4096,
            N=4096,
            K=4096,
        ](ctx)
