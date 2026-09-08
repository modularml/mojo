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
from std.math import align_up
from std.math.uutils import umod, ufloordiv, udivmod
from std.memory import bitcast
from std.sys import argv, size_of

import linalg.matmul.vendor.blas as vendor_blas
from std.bit import prev_power_of_two
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
from max.gpu import (
    block_id_in_cluster,
    block_idx,
    lane_id,
    thread_idx,
    warp_id as get_warp_id,
)
from max.gpu.memory import fence_async_view_proxy, external_memory
from max.gpu.compute.mma import st_matrix
from max.gpu.compute.arch.mma_nvidia_sm100 import *
from max.gpu.sync import named_barrier
from max.gpu.compute.arch.tcgen05 import *
from internal_utils import assert_almost_equal
from layout import (
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
    UNKNOWN_VALUE,
    lt_to_tt,
)
from layout._utils import ManagedLayoutTensor
from layout.layout_tensor import LayoutTensorIter
from layout.swizzle import make_swizzle
from layout.tensor_core_async import (
    st_matrix_n_layout,
    tile_layout_k_major,
    tile_layout_mn_major,
)
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    TMATensorTile,
    _idx_product,
    create_tensor_tile,
    create_tma_tile,
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
def store_C[
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
    c_smem_tile: LayoutTensor[
        c_type,
        c_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ],
    c_tma_op: TMATensorTile[c_type, c_tma_rank, c_tile_shape, c_desc_shape],
    tmem_addr: UInt32,
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

    comptime TMA_BN = c_tile_shape[1]
    var warp_id = get_warp_id()

    # Rows each warp is responsible for:
    # warp_id 0 -> 0-15 upper, 16-31 lower
    # warp_id 1 -> 32-47 upper, 48-63 lower
    # warp_id 2 -> 64-79 upper, 80-95 lower
    # warp_id 3 -> 96-111 upper, 112-127 lower

    # Calculate how many elements we need to load based on MMA_N
    comptime elements_per_row = BN if MMA_M == 128 else MMA_N
    # this is the main load, most cases use this, power of 2
    comptime main_load_elements = prev_power_of_two(elements_per_row)
    # this is remainder load, executed only if MMA_N is not power of 2
    comptime remainder_elements = elements_per_row - main_load_elements

    # if i do have non-power of 2, then remainder_elements must be divisible by 32 (can extend to support more values later)
    comptime assert (
        remainder_elements % 32 == 0
    ), "remainder_elements must be divisible by 32"

    comptime main_repetition = main_load_elements // 8
    comptime remainder_repetitions = remainder_elements // 8

    comptime data_paths = 16
    comptime bits = 256
    comptime num_elements_per_load = bits // 32  # each element in tmem is 4 bytes, 32 bits
    comptime num_regs_per_thread = (
        data_paths * num_elements_per_load
    ) // WARP_SIZE

    comptime NUM_TMA_TILES = MMA_N // TMA_BN
    comptime NUM_ST_MATRIX = BN // TMA_BN if MMA_M == 128 else MMA_N // TMA_BN
    comptime C_SPLIT_ROWS = BM * NUM_TMA_TILES // 2 if MMA_M == 128 else BM * NUM_TMA_TILES

    # NOTE: Every load is 8 elements (256 bits), repetitions is row size / 8
    # We load 16 lanes by 8 elements so 128 elements total
    # 1 warp or 32 threads does this, each thread storing 128/32=4 elements on every load
    # and total number of register usage is num_regs_per_thread * main_repetition

    # Load c_frag_upper
    # Load once if MMA_N is power of 2, otherwise load twice

    comptime main_frag_size = main_repetition * num_regs_per_thread
    comptime remainder_frag_size = max(
        1, remainder_repetitions * num_regs_per_thread
    )

    var c_upper_pow_2_main: Array[Scalar[accum_type], main_frag_size]

    var c_lower_pow_2_main: Array[Scalar[accum_type], main_frag_size]

    var c_upper_pow_2_rem = Array[Scalar[accum_type], remainder_frag_size](
        fill={}
    )
    var c_lower_pow_2_rem = Array[Scalar[accum_type], remainder_frag_size](
        fill={}
    )

    # Primary Load
    c_upper_pow_2_main = tcgen05_ld[
        datapaths=data_paths,
        bits=bits,
        repeat=main_repetition,
        dtype=accum_type,
        pack=False,
        width=main_frag_size,
    ](tmem_addr | UInt32((warp_id * 32) << 16))

    # Load c_frag_lower
    # Primary load
    c_lower_pow_2_main = tcgen05_ld[
        datapaths=data_paths,
        bits=bits,
        repeat=main_repetition,
        dtype=accum_type,
        pack=False,
        width=main_frag_size,
    ](tmem_addr | UInt32((warp_id * 32 + 16) << 16))

    comptime if MMA_N != prev_power_of_two(MMA_N):
        # no mma_n can be larger than 256, so if there's a remainder,
        # we've loaded the smallest power of 2, 128, and the rem is after
        # 128. this is why tmem address is offset by 128
        c_upper_pow_2_rem = tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=remainder_repetitions,
            dtype=accum_type,
            pack=False,
            width=remainder_frag_size,
        ](tmem_addr + 128 | UInt32((warp_id * WARP_SIZE) << 16))

        c_lower_pow_2_rem = tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=remainder_repetitions,
            dtype=accum_type,
            pack=False,
            width=remainder_frag_size,
        ](tmem_addr + 128 | UInt32((warp_id * WARP_SIZE + 16) << 16))

    # Remainder load happens later, only if needed
    tcgen05_load_wait()

    # Create a layout for everything
    var st_matrix_rt_layout = RuntimeLayout[
        st_matrix_n_layout[c_type, TMA_BN, num_m_mmas, 1](),
        element_type=.int32,
        linear_idx_type=.int32,
    ]()

    # For 32-column tiles, we need a different swizzle pattern
    comptime st_matrix_swizzle = make_swizzle[
        c_type,
        TensorMapSwizzle.SWIZZLE_64B if TMA_BN
        == 32 else TensorMapSwizzle.SWIZZLE_128B,
    ]()

    # 128*160 = 20,480 and is same as (128 * 5) * 32 = 20,480
    var c_smem_tile_reshaped = c_smem_tile.reshape[
        Layout.row_major(BM * NUM_TMA_TILES, TMA_BN)
    ]()

    var split_coord_x = ufloordiv(warp_id, 2) if MMA_M == 128 else 0
    var c_smem_split = c_smem_tile_reshaped.tile[C_SPLIT_ROWS, TMA_BN](
        split_coord_x, 0
    )

    comptime for tma_n in range(NUM_ST_MATRIX):
        var c_smem_iter = c_smem_split.tile[BM, TMA_BN](tma_n, 0)
        var c_smem_warp_tile = c_smem_iter.tile[32, TMA_BN](
            umod(warp_id, 2) if MMA_M == 128 else warp_id, 0
        )
        var upper = c_smem_warp_tile.tile[16, TMA_BN](0, 0)
        var lower = c_smem_warp_tile.tile[16, TMA_BN](1, 0)

        var d_reg_upper = SIMD[.bfloat16, 8]()
        var d_reg_lower = SIMD[.bfloat16, 8]()

        comptime for m_mma in range(num_m_mmas):
            comptime for i in range((TMA_BN // 16)):
                var st_matrix_args = RuntimeTuple[
                    IntTuple(
                        UNKNOWN_VALUE,
                        IntTuple(
                            i,
                            m_mma,
                            UNKNOWN_VALUE,
                        ),
                    )
                ](lane_id(), i, m_mma, 0)
                # i,0,0

                var d_reg_upper = SIMD[.bfloat16, 8]()
                var d_reg_lower = SIMD[.bfloat16, 8]()

                # if MMA_N is a power of 2, then just use the main load for all iterations
                # if it's not a power of 2, then go till NUM_ST_MATRIX -1 using the main regists
                # and for last iteration we load remainder registers (for the remainder 32 )
                comptime if (
                    MMA_N == prev_power_of_two(MMA_N)
                    or tma_n < NUM_ST_MATRIX - 1
                ):
                    # every iteration of tma_n is a motion across BM * 32 elements
                    # and we agree that each of those has 2 rows * 8 elements in the register
                    comptime for _ei in range(4):
                        comptime _src_offset = (i * 8) + tma_n * (
                            TMA_BN // 16
                        ) * 8 + 2 * _ei
                        var upper_pair = SIMD[accum_type, 2](
                            rebind[Scalar[accum_type]](
                                c_upper_pow_2_main[_src_offset]
                            ),
                            rebind[Scalar[accum_type]](
                                c_upper_pow_2_main[_src_offset + 1]
                            ),
                        )
                        var lower_pair = SIMD[accum_type, 2](
                            rebind[Scalar[accum_type]](
                                c_lower_pow_2_main[_src_offset]
                            ),
                            rebind[Scalar[accum_type]](
                                c_lower_pow_2_main[_src_offset + 1]
                            ),
                        )
                        var upper_casted = upper_pair.cast[.bfloat16]()
                        var lower_casted = lower_pair.cast[.bfloat16]()
                        d_reg_upper[2 * _ei] = upper_casted[0]
                        d_reg_upper[2 * _ei + 1] = upper_casted[1]
                        d_reg_lower[2 * _ei] = lower_casted[0]
                        d_reg_lower[2 * _ei + 1] = lower_casted[1]
                else:
                    comptime for _ei in range(4):
                        comptime _src_offset = (i * 8) + 2 * _ei
                        var upper_pair = SIMD[accum_type, 2](
                            rebind[Scalar[accum_type]](
                                c_upper_pow_2_rem[_src_offset]
                            ),
                            rebind[Scalar[accum_type]](
                                c_upper_pow_2_rem[_src_offset + 1]
                            ),
                        )
                        var lower_pair = SIMD[accum_type, 2](
                            rebind[Scalar[accum_type]](
                                c_lower_pow_2_rem[_src_offset]
                            ),
                            rebind[Scalar[accum_type]](
                                c_lower_pow_2_rem[_src_offset + 1]
                            ),
                        )
                        var upper_casted = upper_pair.cast[.bfloat16]()
                        var lower_casted = lower_pair.cast[.bfloat16]()
                        d_reg_upper[2 * _ei] = upper_casted[0]
                        d_reg_upper[2 * _ei + 1] = upper_casted[1]
                        d_reg_lower[2 * _ei] = lower_casted[0]
                        d_reg_lower[2 * _ei + 1] = lower_casted[1]

                var d_reg_upper_packed = bitcast[.float32, 4](d_reg_upper)
                var d_reg_lower_packed = bitcast[.float32, 4](d_reg_lower)

                st_matrix[simd_width=4](
                    upper.ptr
                    + st_matrix_swizzle(st_matrix_rt_layout(st_matrix_args)),
                    d_reg_upper_packed,
                )
                st_matrix[simd_width=4](
                    lower.ptr
                    + st_matrix_swizzle(st_matrix_rt_layout(st_matrix_args)),
                    d_reg_lower_packed,
                )

    named_barrier[Int32(num_output_warps * WARP_SIZE)]()

    # SMEM -> GMEM: Direct TMA store
    # UMMA (tensor memory) → registers → shared memory → global memory
    # #           c_frag                   c_smem_tile      c_tma_op

    if elect_one_warp and thread_idx.x < NUM_TMA_TILES:
        var row_start = block_idx.x * BM
        var col_start = block_idx.y * MMA_N + thread_idx.x * TMA_BN

        fence_async_view_proxy()
        var c_smem_offset = c_smem_tile.ptr + BM * TMA_BN * thread_idx.x

        var c_tma_tile = LayoutTensor[
            c_type,
            Layout.row_major(c_tile_shape[0], c_tile_shape[1]),
            MutAnyOrigin,
            address_space=.SHARED,
            alignment=128,
        ](c_smem_offset)

        c_tma_op.async_store(c_tma_tile, (col_start, row_start))
        c_tma_op.commit_group()
        c_tma_op.wait_group[0]()

    if elect_one_warp:
        tcgen05_release_allocation_lock[Int32(cta_group)]()
        tcgen05_dealloc[Int32(cta_group)](tmem_addr, UInt32(max_tmem_cols))


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
def kernel_6[
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

    comptime c_smem_tile_t = LayoutTensor[
        c_type,
        c_smem_layout,
        _,
        address_space=.SHARED,
        alignment=128,
    ]

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
    comptime c_smem_size = c_smem_layout.size()

    var a_smem_base = base_ptr_smem  # need space for 4096 (64 x 64) elements by 2 bytes or 8192 total, which is 0x2000
    var b_smem_base = (a_smem_base + a_smem_size * num_pipeline_stages).bitcast[
        Scalar[b_type]
    ]()
    var c_smem_base = (b_smem_base + b_smem_size * num_pipeline_stages).bitcast[
        Scalar[c_type]
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

    var c_smem_tile = c_smem_tile_t(c_smem_base)

    var smem_pool = (c_smem_base + c_smem_size).bitcast[Int64]()

    comptime accum_type = get_accum_type[a_type]()

    # adding 8 bytes for ptr_tmem_addr (smem poll is 8 byte casted)
    var tma_mbar_ptr = smem_pool
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

        store_C[
            accum_type=accum_type,
            block_tile_shape=block_tile_shape,
            mma_shape=mma_shape,
            c_swizzle=c_swizzle,
            cta_group=cta_group,
            num_output_warps=num_output_warps,
            max_tmem_cols=max_tmem_cols,
        ](
            c_smem_tile,
            c_tma_op,
            tmem_addr,
            elect_one_warp,
        )


def blackwell_kernel_6[
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
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    cta_group: Int = 1,
](
    c: LayoutTensor[mut=True, c_type, c_layout, _],
    a: LayoutTensor[mut=True, a_type, a_layout, _],
    b: LayoutTensor[mut=True, b_type, b_layout, _],
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

    # Create a separate TMA descriptor for the 32-column leftover tile
    # Using SWIZZLE_64B to match the swizzle pattern used in st_matrix for leftover
    var c_tma_op = create_tma_tile[
        BM,
        64 if MMA_N == prev_power_of_two(MMA_N) else 32,
        swizzle_mode=TensorMapSwizzle.SWIZZLE_128B if MMA_N
        == prev_power_of_two(MMA_N) else TensorMapSwizzle.SWIZZLE_64B,
    ](ctx, c)

    # ctx.default_device_info.shared_memory_per_multiprocessor gives this magic number on B200
    comptime total_smem_size_available = 233472
    comptime smem_available_after_c = total_smem_size_available - (
        BM * MMA_N * size_of[c_type]()
    )
    comptime smem_per_stage_no_c = (BM * BK * size_of[a_type]()) + (
        BN * BK * size_of[b_type]()
    ) + (32)
    comptime max_pipeline_stages = smem_available_after_c // smem_per_stage_no_c

    # - ptr_tmem_addr: 4 bytes → 8 bytes (padded)
    # - tma_mbar_ptr: 8 bytes
    # - mma_mbar_ptr: 8 bytes
    # - compute_barrier: 8 bytes (padded)
    # Total with alignment: 32 bytes
    # This is why we pad 32 bytes * num_pipeline_stages to the smem size

    comptime smem_size = (
        (BM * BK * size_of[a_type]()) * max_pipeline_stages
        + (BN * BK * size_of[b_type]()) * max_pipeline_stages
        + (BM * MMA_N * size_of[c_type]())
    ) + (32) * max_pipeline_stages

    comptime kernel = kernel_6[
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
    ]

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        Int32(K // BK),
        grid_dim=(
            align_up(M // BM, Int(cluster_shape[0])),
            align_up(N // BN // cta_group, Int(cluster_shape[1])),
            1,
        ),
        # 1 TMA, 1 MMA, 4 EPILOGUE warps
        block_dim=(32 * 6),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_size)
        ),
    )


def test_blackwell_kernel_6[
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
    M: Int = 4096,
    N: Int = 4096,
    K: Int = 4096,
](ctx: DeviceContext) raises:
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
    # Zero out c buffers
    for i in range(M * N):
        c_host.ptr[i] = Scalar[c_type](0)
        c_host_ref.ptr[i] = Scalar[c_type](0)

    # Move operands to the Device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    ctx.enqueue_copy(c_device, c_host.ptr)
    ctx.enqueue_copy(c_device_ref, c_host_ref.ptr)

    blackwell_kernel_6[
        transpose_b=transpose_b,
        umma_shape=mma_shape,
        block_tile_shape=block_tile_shape,
        cluster_shape=cluster_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        c_swizzle=c_swizzle,
        cta_group=2,
    ](
        c_device_lt,
        a_device_lt,
        b_device_lt,
        ctx,
    )

    if benchmark:
        comptime num_runs = 50
        comptime num_warmup = 20

        @always_inline
        def run_kernel(ctx: DeviceContext) raises {imm}:
            blackwell_kernel_6[
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
        print(t"{M}x{N}x{K}", sectime * 1000, tflops_rounded)
    else:
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

        # print(ndbuffer_to_str(c_host))
        # print(ndbuffer_to_str(c_host_ref))

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
    comptime a_type = DType.bfloat16
    comptime b_type = DType.bfloat16
    comptime c_type = DType.bfloat16
    comptime block_tile_shape = Index(128, 128, 64)
    comptime umma_shape = Index(
        block_tile_shape[0] * 2, block_tile_shape[1] * 2, 16
    )
    comptime dic_of_shapes = make_dic_of_shapes()

    print("Benchmarking blackwell_matmul_tma_umma_kernel")
    print("============================================")
    print("M, N, K, time(ms), TFLOPS")

    comptime for i in range(len(dic_of_shapes)):
        comptime shape = get_dic_of_shapes(i, dic_of_shapes)
        try:
            test_blackwell_kernel_6[
                a_type,
                b_type,
                c_type,
                block_tile_shape,
                umma_shape,
                cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
                a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
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
            # Run the benchmark
            print("\n\n========== Running Benchmarks ==========\n")
            benchmark_blackwell_matmul(ctx)
            return

        comptime block_tile_shape = Index(128, 128, 64)
        comptime umma_shape = Index(
            block_tile_shape[0] * 2, block_tile_shape[1] * 2, 16
        )
        test_blackwell_kernel_6[
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
