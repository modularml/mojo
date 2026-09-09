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
from max.gpu import block_id_in_cluster, block_idx, lane_id
from max.gpu import warp_id as get_warp_id
from max.gpu.memory import fence_async_view_proxy, external_memory
from max.gpu.compute.mma import st_matrix
from max.gpu.compute.arch.mma_nvidia_sm100 import *
from max.gpu.sync import named_barrier
from max.gpu.compute.arch.tcgen05 import *
from internal_utils import assert_almost_equal
from layout import (
    CoordLike,
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    TileTensor,
    row_major,
    lt_to_tt,
)

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

    comptime MainLoad = Self(4)
    comptime Mma = Self(5)
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
        circular=False,
    ],
    b_smem: LayoutTensorIter[
        b_type,
        b_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
        circular=False,
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

    if elect_one_cta:
        tma_mbar[stage].expect_bytes(Int32(expected_bytes))

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
        circular=False,
    ],
    b_smem_iter: LayoutTensorIter[
        b_type,
        b_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
        circular=False,
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

    var a_smem_tile = a_smem_iter.next_unsafe(
        rebind[a_smem_iter.linear_uint_type](stage)
    )[]
    var b_smem_tile = b_smem_iter.next_unsafe(
        rebind[b_smem_iter.linear_uint_type](stage)
    )[]

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
    /,
    *,
    accum_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
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
    tmem_addr: UInt32,
    work_tile_coord: Tuple[Int, Int],
    elect_one_warp: Bool,
):
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]

    comptime num_m_mmas = BM // (mma_shape[0] // cta_group)
    comptime num_n_mmas = BN // (mma_shape[1] // cta_group)

    comptime assert num_m_mmas == 1 and num_n_mmas == 1

    # We break down the output tile BM x MMA_N to BM x stageN tiles
    # and output one tile per stage.
    # For MMA_M=128, we use 64x16 tiles, otherwise 64x32
    comptime stageM = c_smem_layout.shape[0].value()  # 128
    comptime stageN = c_smem_layout.shape[1].value()  # 32
    # For MMA_M=128, we have 8 logical stages (128/16) but process left and right separately
    comptime num_stages = MMA_N // stageN if MMA_M == 256 else MMA_N // stageN // 2
    comptime tmem_cell_bytes = 4
    comptime data_paths = 16
    comptime bits = 256
    comptime rep = stageN // (bits // 32)

    # stmatrix related
    comptime stsmx4N_bytes = 32
    comptime stsmx4N = stsmx4N_bytes // size_of[c_type]()
    comptime stsmx4_size_per_lane = (16 * stsmx4N) // WARP_SIZE
    # if the tile is not 32, it is assumed to be 16
    comptime st_matrix_swizzle = TensorMapSwizzle.SWIZZLE_64B if stageN == 32 else TensorMapSwizzle.SWIZZLE_32B
    comptime swizzle = make_swizzle[c_type, st_matrix_swizzle]()

    var warp_id = get_warp_id()

    comptime for stage in range(num_stages):
        # MMA has result in 32 rows per warp's data paths.
        # upper_frag is for rows 0-15, lower is for 16-31.
        var stage_tmem_addr = tmem_addr + UInt32(stage * stageN)

        # MMA has result in 32 rows per warp's data paths.
        # upper_frag is for rows 0-15, lower is for 16-31.
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

    if elect_one_warp:
        tcgen05_release_allocation_lock[Int32(cta_group)]()
        tcgen05_dealloc[Int32(cta_group)](tmem_addr, UInt32(max_tmem_cols))


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
def kernel_7[
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
    num_iters_dev: Int32,
):
    var num_iters = Int(num_iters_dev)
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]

    comptime num_m_mmas = BM // (mma_shape[0] // cta_group)
    comptime num_n_mmas = BN // (mma_shape[1] // cta_group)
    comptime num_k_mmas = BK // mma_shape[2]
    comptime num_output_warps = 4

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

    var base_ptr_smem = rebind[
        MutPointer[Scalar[a_type], address_space=.SHARED, MutUntrackedOrigin]
    ](
        external_memory[
            Scalar[a_type],
            address_space=.SHARED,
            alignment=128,
        ]()
    )  # pointer to first byte of scratchpad

    comptime a_smem_size = a_smem_layout.size()
    comptime b_smem_size = b_smem_layout.size()
    comptime c_smem_size = output_tile_shape[0] * output_tile_shape[
        1
    ] * num_output_stages

    var a_smem_base = base_ptr_smem  # need space for 4096 (64 x 64) elements by 2 bytes or 8192 total, which is 0x2000
    var b_smem_base = (a_smem_base + a_smem_size * num_pipeline_stages).bitcast[
        Scalar[b_type]
    ]()

    var a_smem = LayoutTensorIter[
        a_type,
        a_smem_layout,
        address_space=.SHARED,
        alignment=128,
        circular=False,
    ](
        a_smem_base,
        a_smem_size * num_pipeline_stages,
    )

    var b_smem = LayoutTensorIter[
        b_type,
        b_smem_layout,
        address_space=.SHARED,
        alignment=128,
        circular=False,
    ](
        b_smem_base,
        b_smem_size * num_pipeline_stages,
    )

    var c_smem_base = (b_smem_base + b_smem_size * num_pipeline_stages).bitcast[
        Scalar[c_type]
    ]()
    var c_smem_iter = LayoutTensorIter[
        c_type,
        Layout.row_major(output_tile_shape[0], output_tile_shape[1]),
        address_space=.SHARED,
        alignment=128,
    ](c_smem_base, c_smem_size)

    var smem_pool = (c_smem_base + c_smem_size).bitcast[Int64]()

    comptime accum_type = get_accum_type[a_type]()

    # this gets 8 bytes of space
    # var ptr_tmem_addr = smem_pool.bitcast[UInt32]()
    # adding 8 bytes for ptr_tmem_addr (smem poll is 8 byte casted)
    var tma_mbar_ptr = smem_pool.bitcast[Int64]()
    # + num_pipeline_stages is 1 * num_pipeline_stage so 8 bytes for each barrier at each stage
    var mma_mbar_ptr = tma_mbar_ptr + num_pipeline_stages
    var compute_barrier_base = mma_mbar_ptr + num_pipeline_stages
    var ptr_tmem_addr = (compute_barrier_base + 1).bitcast[UInt32]()

    var tma_mbar = tma_mbar_ptr.bitcast[SharedMemBarrier]()
    var mma_mbar = mma_mbar_ptr.bitcast[SharedMemBarrier]()
    var compute_barrier = compute_barrier_base.bitcast[SharedMemBarrier]()

    var warp_id = get_warp_id()
    var elect_one_warp = warp_id == 0
    var elect_one_thread = elect_one_sync_with_mask()
    var elect_one_cta = block_rank_in_cluster() % 2 == 0
    comptime max_tmem_cols = 512

    if elect_one_warp:
        tcgen05_alloc[Int32(cta_group)](ptr_tmem_addr, max_tmem_cols)

    # Ensure all threads sees initialized mbarrier and
    # tensor memory allocation
    barrier()

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
        compute_barrier[].init()

    cluster_sync()

    var consumer_phase = PipelineState[num_pipeline_stages]()
    var producer_phase = PipelineState[num_pipeline_stages](0, 1, 0)

    var tmem_addr = ptr_tmem_addr[0]

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
        if elect_one_sync():
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
                    (block_idx.x, block_idx.y),
                    a_multicast_mask,
                    b_multicast_mask,
                    i,
                    elect_one_cta,
                )
                producer_phase.step()

    if elect_one_cta and WarpRole.is_mma():
        for i in range(num_iters):
            consumer_main_loop[
                block_tile_shape=block_tile_shape,
                mma_shape=mma_shape,
                cta_group=cta_group,
                cluster_shape=Index(
                    cluster_shape[0], cluster_shape[1], cluster_shape[2]
                ),
            ](
                tmem_addr,
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
                compute_barrier, UInt16(mma_complete_mask)
            )

    if WarpRole.is_epilogue():
        compute_barrier[].wait()

        multi_stage_store_C[
            accum_type=accum_type,
            block_tile_shape=block_tile_shape,
            mma_shape=mma_shape,
            c_swizzle=c_swizzle,
            cta_group=cta_group,
            num_output_warps=num_output_warps,
            max_tmem_cols=max_tmem_cols,
        ](
            c_smem_iter,
            c_tma_op,
            tmem_addr,
            (block_idx.x, block_idx.y),
            elect_one_warp,
        )


def blackwell_kernel_7[
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

    # Configure shared memory usage
    # Total size = capacity - 1KB_reserved_by_L1
    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024
    # A and B per pipeline stage
    comptime AB_smem_per_stage = BM * BK * size_of[
        a_type
    ]() + BN * BK * size_of[b_type]()
    # Support double-buffer for output stages.
    comptime num_output_stages = 2
    comptime C_smem = output_tile_shape[0] * output_tile_shape[
        1
    ] * num_output_stages * size_of[c_type]()
    # Usage reserved for mbar and others
    # - tma_mbar_ptr: 8 bytes per pipeline stage
    # - mma_mbar_ptr: 8 bytes per pipeline stage
    # - compute_barrier: 8 bytes
    # - tmem addr in shared memory: 4B
    comptime mbar_per_stage = 16
    comptime other_usage = 8 + 4
    comptime per_stage_usage = AB_smem_per_stage + mbar_per_stage
    # Compute the max number of pipeline stages supported
    comptime num_pipeline_stages = (
        b200_smem - C_smem - other_usage
    ) // per_stage_usage
    # Total smem usage
    comptime smem_size = per_stage_usage * num_pipeline_stages + C_smem + other_usage

    comptime kernel = kernel_7[
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
        num_pipeline_stages=num_pipeline_stages,
        num_output_stages=num_output_stages,
        output_tile_shape=output_tile_shape,
    ]

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        Int32(K // BK),
        grid_dim=(
            align_up(ceildiv(M, BM), Int(cluster_shape[0])),
            align_up(ceildiv(N, MMA_N), Int(cluster_shape[1])),
            1,
        ),
        # 1 TMA, 1 MMA, 4 EPILOGUE warps
        block_dim=(32 * 6),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_size)
        ),
    )


def test_blackwell_kernel_7[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    benchmark: Bool = False,
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    var M = Int(m.value())
    var N = Int(n.value())
    var K = Int(k.value())

    if not benchmark:
        print(
            String(
                M,
                "x",
                N,
                "x",
                K,
                " mma_shape=",
                mma_shape,
                " block_tile_shape=",
                block_tile_shape,
            )
        )

    var a_shape = Coord(m, k)
    var b_shape = Coord(
        Idx[NType.static_value if transpose_b else KType.static_value],
        Idx[KType.static_value if transpose_b else NType.static_value],
    )
    var c_shape = Coord(m, n)

    # Host memory allocation
    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](M * K)
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](N * K)
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](M * N)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[c_type](M * N)

    # Device memory allocation
    var a_device = ctx.enqueue_create_buffer[a_type](M * K)
    var b_device = ctx.enqueue_create_buffer[b_type](N * K)
    var c_device = ctx.enqueue_create_buffer[c_type](M * N)
    var c_device_ref = ctx.enqueue_create_buffer[c_type](M * N)

    var a_tt = TileTensor(a_device, row_major(a_shape))
    var b_tt = TileTensor(b_device, row_major(b_shape))
    var c_tt = TileTensor(c_device, row_major(c_shape))
    var c_ref_tt = TileTensor(c_device_ref, row_major(c_shape))

    # Perf varies with initial values. Simple values have lower noise for
    # the current benchmark comparing to random initial values.
    var a_host_tt = TileTensor(a_host_ptr, row_major(a_shape))
    var b_host_tt = TileTensor(b_host_ptr, row_major(b_shape))
    comptime assert a_host_tt.flat_rank == 2
    for m_idx in range(M):
        for k_idx in range(K):
            a_host_tt[m_idx, k_idx] = Float32(k_idx).cast[a_type]()
    for n_idx in range(N):
        for k_idx in range(K):
            b_host_tt[n_idx, k_idx] = Float32(1 if n_idx == k_idx else 0).cast[
                b_type
            ]()

    # Move operands to the Device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    blackwell_kernel_7[
        transpose_b=transpose_b,
        umma_shape=mma_shape,
        block_tile_shape=block_tile_shape,
        cluster_shape=cluster_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        cta_group=2,
    ](
        c_tt.to_layout_tensor(),
        a_tt.to_layout_tensor(),
        b_tt.to_layout_tensor(),
        ctx,
    )

    if benchmark:
        comptime num_runs = 100
        comptime num_warmup = 100

        @always_inline
        def run_kernel(ctx: DeviceContext) raises {imm}:
            blackwell_kernel_7[
                transpose_b=transpose_b,
                umma_shape=mma_shape,
                block_tile_shape=block_tile_shape,
                cluster_shape=cluster_shape,
                a_swizzle=a_swizzle,
                b_swizzle=b_swizzle,
                cta_group=2,
            ](
                c_tt.to_layout_tensor(),
                a_tt.to_layout_tensor(),
                b_tt.to_layout_tensor(),
                ctx,
            )

        # Warmup
        for _ in range(num_warmup):
            run_kernel(ctx)
        ctx.synchronize()

        var nstime = (
            Float64(ctx.execution_time(run_kernel, num_runs)) / num_runs
        )
        var sectime = nstime * 1e-9
        var TFlop = 2.0 * Float64(M) * Float64(N) * Float64(K) * 1e-12
        # Round TFLOPS to two decimal places for cleaner output
        var tflops = TFlop / sectime
        var tflops_rounded = round(tflops, 2)
        print(t"{M}x{N}x{K}", tflops_rounded)
    else:
        vendor_blas.matmul(
            ctx,
            c_ref_tt,
            a_tt,
            b_tt,
            c_row_major=True,
            transpose_b=transpose_b,
        )

        ctx.synchronize()

        ctx.enqueue_copy(c_host_ptr, c_device)
        ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
        ctx.synchronize()

        comptime rtol = 1e-2
        assert_almost_equal(
            c_host_ptr.unsafe_ptr(),
            c_host_ref_ptr.unsafe_ptr(),
            M * N,
            atol=0.0001,
            rtol=rtol,
        )
        print("\n=== TEST PASSED ===\n")


def get_shapes_dict(
    index: Int, shapes_dict: Dict[Int, Tuple[Int, Int, Int], ...]
) -> Tuple[Int, Int, Int]:
    try:
        return shapes_dict[index]
    except error:
        print("error")
        return (128, 128, 128)


def make_shapes_dict() -> (
    Dict[Int, Tuple[Int, Int, Int], default_comp_time_hasher]
):
    var dic: Dict[Int, Tuple[Int, Int, Int], default_comp_time_hasher] = {
        0: (4096, 4096, 4096),
    }
    return dic^


def benchmark_blackwell_matmul(ctx: DeviceContext) raises:
    comptime a_type = DType.bfloat16
    comptime b_type = DType.bfloat16
    comptime c_type = DType.bfloat16
    comptime block_tile_shape = Index(128, 128, 64)
    comptime umma_shape = Index(
        block_tile_shape[0] * 2, block_tile_shape[1] * 2, 16
    )
    comptime shapes_dict = make_shapes_dict()

    print("Benchmarking blackwell_matmul_tma_umma_kernel")
    print("============================================")
    print("M, N, K, time(ms), TFLOPS")

    comptime for i in range(len(shapes_dict)):
        comptime shape = get_shapes_dict(i, shapes_dict)
        try:
            test_blackwell_kernel_7[
                a_type,
                b_type,
                c_type,
                block_tile_shape,
                umma_shape,
                cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
                a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                benchmark=True,
            ](ctx, shape[0], Idx[shape[1]], Idx[shape[2]])
        except error:
            print("error")


def main() raises:
    with DeviceContext() as ctx:
        # Run the benchmark
        if is_benchmark():
            benchmark_blackwell_matmul(ctx)
            return

        comptime block_tile_shape = Index(128, 64, 64)
        comptime umma_shape = Index(256, 128, 16)

        test_blackwell_kernel_7[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            block_tile_shape,
            umma_shape,
            cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
            a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
        ](ctx, Idx[4096], Idx[4096], Idx[4096])
