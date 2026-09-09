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
from max.gpu.sync import barrier
from max.gpu.primitives.cluster import (
    block_rank_in_cluster,
    cluster_sync,
    elect_one_sync,
)
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu import block_id_in_cluster, block_idx, lane_id, thread_idx, warp_id
from max.gpu.memory import fence_async_view_proxy, external_memory
from max.gpu.compute.mma import st_matrix
from max.gpu.compute.arch.mma_nvidia_sm100 import *
from max.gpu.compute.arch.tcgen05 import *
from internal_utils import assert_almost_equal
from layout import (
    IntTuple,
    Layout,
    LayoutTensor,
    LTToTTLayout,
    RuntimeLayout,
    RuntimeTuple,
    TileTensor,
    UNKNOWN_VALUE,
)
from layout._utils import ManagedLayoutTensor
from layout.swizzle import make_swizzle
from layout.tensor_core_async import (
    st_matrix_n_layout,
    tile_layout_k_major,
    tile_layout_mn_major,
)
from layout.tma_async import (
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


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
def kernel_5[
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
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    cta_group: Int = 1,
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

    comptime CLUSTER_M = Int(cluster_shape[0])
    comptime CLUSTER_N = Int(cluster_shape[1])

    comptime TMA_BN = c_tile_shape[1]
    comptime a_tma_load_size = _idx_product[a_tma_rank, a_desc_shape]()
    comptime b_tma_load_size = _idx_product[b_tma_rank, b_desc_shape]()
    comptime a_tma_rows = a_desc_shape[0]
    comptime b_tma_rows = b_desc_shape[0]
    comptime c_smem_layout = Layout.row_major(BM, MMA_N)

    comptime a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    comptime b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()
    comptime sub_a_smem_layout = tile_layout_k_major[
        a_type, BM, 64, swizzle_mode=a_swizzle
    ]()
    comptime sub_b_smem_layout = tile_layout_k_major[
        b_type, BN, 64, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, 64, swizzle_mode=b_swizzle
    ]()

    comptime sub_a_smem_tile_t = LayoutTensor[
        a_type,
        sub_a_smem_layout,
        _,
        address_space=.SHARED,
        alignment=128,
    ]
    comptime sub_b_smem_tile_t = LayoutTensor[
        b_type,
        sub_b_smem_layout,
        _,
        address_space=.SHARED,
        alignment=128,
    ]
    comptime c_smem_tile_t = LayoutTensor[
        c_type,
        c_smem_layout,
        _,
        address_space=.SHARED,
        alignment=128,
    ]

    var smem = external_memory[UInt8, address_space=.SHARED, alignment=8]()

    comptime a_smem_bytes = a_smem_layout.size() * size_of[a_type]()
    comptime b_smem_bytes = b_smem_layout.size() * size_of[b_type]()
    comptime c_smem_bytes = c_smem_layout.size() * size_of[c_type]()

    var a_smem = smem.bitcast[Scalar[a_type]]()
    var b_smem = (smem + a_smem_bytes).bitcast[Scalar[b_type]]()
    var c_smem = (smem + a_smem_bytes + b_smem_bytes).bitcast[Scalar[c_type]]()

    var c_smem_tile = c_smem_tile_t(c_smem)

    var smem_pool = (smem + a_smem_bytes + b_smem_bytes + c_smem_bytes).bitcast[
        Int64
    ]()

    var a_smem_tile = TileTensor(a_smem, LTToTTLayout[a_smem_layout]())
    var b_smem_tile = TileTensor(b_smem, LTToTTLayout[b_smem_layout]())

    comptime accum_type = get_accum_type[a_type]()

    comptime c_frag_size = MMA_M * MMA_N // 128 // cta_group
    comptime c_half_frag_size = c_frag_size // 2

    comptime a_expected_bytes = a_smem_layout.size() * size_of[a_type]()
    comptime b_expected_bytes = b_smem_layout.size() * size_of[b_type]()
    # Leader CTAs expect SMEM from itself and their peers
    comptime expected_bytes = cta_group * (a_expected_bytes + b_expected_bytes)

    var tma_mbar_ptr = smem_pool.bitcast[Int64]()
    var mma_mbar_ptr = tma_mbar_ptr + 2
    # Shared memory pointer to hold tensor memory address
    var ptr_tmem_addr = (mma_mbar_ptr + 2).bitcast[UInt32]()

    var tma_mbar = tma_mbar_ptr.bitcast[SharedMemBarrier]()
    var mma_mbar = mma_mbar_ptr.bitcast[SharedMemBarrier]()

    var elect_one_warp = warp_id() == 0
    var elect_one_thread = elect_one_sync()
    var elect_one_cta = block_rank_in_cluster() % 2 == 0
    comptime max_tmem_cols = 512

    if elect_one_warp:
        tcgen05_alloc[Int32(cta_group)](ptr_tmem_addr, max_tmem_cols)

    # Ensure all threads sees initialized mbarrier and
    # tensor memory allocation
    barrier()

    if elect_one_warp and elect_one_thread:
        tma_mbar[0].init()
        mma_mbar[0].init(
            cluster_shape[0] // Int32(cta_group) + cluster_shape[1] - 1
        )

    cluster_sync()

    var tma_phase: UInt32 = 0
    var mma_phase: UInt32 = 0

    var tmem_addr = ptr_tmem_addr[0]

    var rank_m = block_id_in_cluster.x
    var rank_n = block_id_in_cluster.y

    # (peer_id, mma_coord_m, mma_coord_n)
    var peer_cta_quot, peer_cta_rem = udivmod(rank_m, cta_group)
    var peer_cta_coord = (
        peer_cta_rem,
        peer_cta_quot,
        rank_n,
    )

    var a_multicast_mask: UInt16 = 0x0
    var b_multicast_mask: UInt16 = 0x0

    # TODO: find a generic way to calculate multicast mask
    comptime for i in range(CLUSTER_N):
        a_multicast_mask |= UInt16(1 << (i * CLUSTER_M))

    comptime for i in range(CLUSTER_M // cta_group):
        b_multicast_mask |= UInt16(1 << (i * cta_group))

    a_multicast_mask <<= UInt16(rank_m)
    b_multicast_mask <<= UInt16(peer_cta_coord[0])
    b_multicast_mask <<= UInt16(rank_n * CLUSTER_M)

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

    for i in range(Int(num_iters)):
        if elect_one_warp and elect_one_thread:
            if elect_one_cta:
                tma_mbar[0].expect_bytes(Int32(expected_bytes))

            var a_gmem_slice_coord = (
                peer_cta_coord[2] * a_tma_rows + block_idx.x * BM
            )
            var b_gmem_slice_coord = (
                peer_cta_coord[1] * b_tma_rows
                + peer_cta_coord[0] * BN
                + block_idx.y * MMA_N
            )

            comptime for j in range(BK // 64):
                comptime k = 64 * j
                comptime a_offset = a_smem_layout(IntTuple(0, k))
                comptime b_offset = b_smem_layout(IntTuple(0, k))
                comptime assert ((a_offset * size_of[a_type]()) % 128) == 0
                comptime assert ((b_offset * size_of[b_type]()) % 128) == 0
                var sub_a_smem_tile = sub_a_smem_tile_t(a_smem + a_offset)
                var sub_b_smem_tile = sub_b_smem_tile_t(b_smem + b_offset)

                var a_smem_slice = type_of(sub_a_smem_tile)(
                    sub_a_smem_tile.ptr + peer_cta_coord[2] * a_tma_load_size
                )
                var b_smem_slice = type_of(sub_b_smem_tile)(
                    sub_b_smem_tile.ptr + peer_cta_coord[1] * b_tma_load_size
                )
                a_tma_op.async_multicast_load[cta_group](
                    a_smem_slice,
                    tma_mbar[0],
                    (i * BK + k, a_gmem_slice_coord),
                    a_multicast_mask,
                )

                b_tma_op.async_multicast_load[cta_group](
                    b_smem_slice,
                    tma_mbar[0],
                    (i * BK + k, b_gmem_slice_coord),
                    b_multicast_mask,
                )

        if elect_one_cta:
            tma_mbar[0].wait(tma_phase)
            tma_phase ^= 1

            if elect_one_warp and elect_one_thread:
                mma_op.mma(
                    a_smem_tile,
                    b_smem_tile,
                    tmem_addr,
                    init_c=(i == 0),  # Initialize C on first iteration
                )

                mma_op.commit(mma_mbar)

        mma_mbar[0].wait(mma_phase)
        mma_phase ^= 1

    # For tcgen05.ld 16x256, we need to split the register to deal with
    # loading 32 lanes for each warp.

    # warp_id 0 -> 0, 16
    # warp_id 1 -> 32, 48
    # warp_id 2 -> 64, 80
    # warp_id 3 -> 96, 112
    var c_frag_upper = tcgen05_ld[
        datapaths=16,
        bits=256,
        repeat=BN // 8 if MMA_M == 128 else MMA_N // 8,
        dtype=accum_type,
        pack=False,
        width=c_half_frag_size,
    ](tmem_addr | UInt32((warp_id() * 32) << 16))

    var c_frag_lower = tcgen05_ld[
        datapaths=16,
        bits=256,
        repeat=BN // 8 if MMA_M == 128 else MMA_N // 8,
        dtype=accum_type,
        pack=False,
        width=c_half_frag_size,
    ](tmem_addr | UInt32((warp_id() * 32 + 16) << 16))
    tcgen05_load_wait()

    comptime C_WBM = BM // 2 if MMA_M == 128 else BM // 4
    comptime C_WBN = BN if MMA_M == 128 else MMA_N
    var c_coord_x = umod(warp_id(), 2) if MMA_M == 128 else warp_id()
    var c_coord_y = ufloordiv(warp_id(), 2) if MMA_M == 128 else 0

    # 32 x BN
    var _c_warp_tile = c_smem_tile.tile[C_WBM, C_WBN](c_coord_x, c_coord_y)

    var st_matrix_rt_layout = RuntimeLayout[
        st_matrix_n_layout[c_type, TMA_BN, num_m_mmas, 1](),
        element_type=.int32,
        linear_idx_type=.int32,
    ]()

    comptime st_matrix_swizzle = make_swizzle[c_type, c_swizzle]()
    comptime NUM_TMA_TILES = MMA_N // TMA_BN
    comptime NUM_ST_MATRIX = BN // TMA_BN if MMA_M == 128 else MMA_N // TMA_BN
    comptime C_SPLIT_ROWS = BM * NUM_TMA_TILES // 2 if MMA_M == 128 else BM * NUM_TMA_TILES

    var c_smem_tile_reshaped = c_smem_tile.reshape[
        Layout.row_major(BM * NUM_TMA_TILES, TMA_BN)
    ]()

    var split_coord_x = ufloordiv(warp_id(), 2) if MMA_M == 128 else 0
    var c_smem_split = c_smem_tile_reshaped.tile[C_SPLIT_ROWS, TMA_BN](
        split_coord_x, 0
    )

    comptime for tma_n in range(NUM_ST_MATRIX):
        var c_smem_iter = c_smem_split.tile[BM, TMA_BN](tma_n, 0)
        var c_smem_warp_tile = c_smem_iter.tile[32, TMA_BN](
            umod(warp_id(), 2) if MMA_M == 128 else warp_id(), 0
        )
        var upper = c_smem_warp_tile.tile[16, TMA_BN](0, 0)
        var lower = c_smem_warp_tile.tile[16, TMA_BN](1, 0)

        comptime for i in range(TMA_BN // 16):
            var d_reg_upper = SIMD[.bfloat16, 8]()
            var d_reg_lower = SIMD[.bfloat16, 8]()

            comptime for _ei in range(4):
                comptime _src_offset = (
                    i + tma_n * (TMA_BN // 16)
                ) * 8 + 2 * _ei
                var upper_pair = SIMD[.float32, 2](
                    rebind[Float32](c_frag_upper[_src_offset]),
                    rebind[Float32](c_frag_upper[_src_offset + 1]),
                )
                var lower_pair = SIMD[.float32, 2](
                    rebind[Float32](c_frag_lower[_src_offset]),
                    rebind[Float32](c_frag_lower[_src_offset + 1]),
                )
                var upper_casted = upper_pair.cast[.bfloat16]()
                var lower_casted = lower_pair.cast[.bfloat16]()
                d_reg_upper[2 * _ei] = upper_casted[0]
                d_reg_upper[2 * _ei + 1] = upper_casted[1]
                d_reg_lower[2 * _ei] = lower_casted[0]
                d_reg_lower[2 * _ei + 1] = lower_casted[1]

            var st_matrix_args = RuntimeTuple[
                IntTuple(
                    UNKNOWN_VALUE,
                    IntTuple(
                        i,
                        0,
                        UNKNOWN_VALUE,
                    ),
                )
            ](lane_id(), i, 0, 0)

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

    barrier()

    # SMEM -> GMEM: Direct TMA store
    # UMMA (tensor memory) → registers → shared memory → global memory
    #           c_frag                   c_smem_tile      c_tma_op
    if elect_one_warp and thread_idx.x < NUM_TMA_TILES:
        var row_start = block_idx.x * BM

        var col_start = block_idx.y * MMA_N + thread_idx.x * TMA_BN

        fence_async_view_proxy()
        var c_smem_offset = c_smem_tile.ptr + BM * TMA_BN * thread_idx.x

        var c_tma_tile = LayoutTensor[
            c_type,
            Layout.row_major(c_tile_shape[0], c_tile_shape[1]),
            address_space=.SHARED,
            alignment=128,
        ](c_smem_offset)

        c_tma_op.async_store(c_tma_tile, (col_start, row_start))
        c_tma_op.commit_group()
        c_tma_op.wait_group[0]()

    if elect_one_warp:
        tcgen05_release_allocation_lock[Int32(cta_group)]()
        tcgen05_dealloc[Int32(cta_group)](tmem_addr, max_tmem_cols)

    cluster_sync()


def blackwell_kernel_5[
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
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    cta_group: Int = 1,
](
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    a: LayoutTensor[a_type, a_layout, MutAnyOrigin],
    b: LayoutTensor[b_type, b_layout, MutAnyOrigin],
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
        Index(Int32(BM) // cluster_shape[1], 64), swizzle_mode=a_swizzle
    ](ctx, a)

    var b_tma_op = create_tensor_tile[
        Index(
            Int32(BN) // (cluster_shape[0] // Int32(cta_group)), 64
        ) if transpose_b else Index(
            64, Int32(BN) // (cluster_shape[0] // Int32(cta_group))
        ),
        swizzle_mode=b_swizzle,
    ](ctx, b)

    # TODO: 64 satisfies 128B swizzle, we need set TMA_BN according to swizzle mode
    var c_tma_op = create_tma_tile[BM, 64, swizzle_mode=c_swizzle](ctx, c)

    comptime smem_size = (
        BM * BK * size_of[a_type]()
        + BN * BK * size_of[b_type]()
        + BM * MMA_N * size_of[c_type]()
    ) + 16 + 16 + 16 + 16

    comptime kernel = kernel_5[
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
        block_dim=(128),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_size)
        ),
    )


def test_blackwell_kernel_5[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    benchmark: Bool = False,
    M: Int = 4096,
    N: Int = 4096,
    K: Int = 4096,
](ctx: DeviceContext) raises:
    print(
        "mma_"
        + "s"
        + "s_bf16_bf16_f32 block tile "
        + String(block_tile_shape)
        + " transb="
        + String(transpose_b)
        + "; inst shape "
        + String(mma_shape)
        + " A "
        + String(a_swizzle)
        + " B "
        + String(b_swizzle)
        + "\nMNK="
        + String(M)
        + "x"
        + String(N)
        + "x"
        + String(K)
        + " cluster_shape=("
        + String(cluster_shape[0])
        + ", "
        + String(cluster_shape[1])
        + ", "
        + String(cluster_shape[2])
        + ")"
    )

    comptime a_layout = Layout.row_major(M, K)
    comptime b_layout = Layout.row_major(
        N, K
    ) if transpose_b else Layout.row_major(K, N)
    comptime c_layout = Layout.row_major(M, N)

    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](M * K)
    var a_host = LayoutTensor[a_type, a_layout](a_host_ptr)
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](N * K)
    var b_host = LayoutTensor[b_type, b_layout](b_host_ptr)
    var c_host = ManagedLayoutTensor[c_type, c_layout](ctx)
    var c_host_ref = ManagedLayoutTensor[c_type, c_layout](ctx)

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
    for i in range(M * N):
        c_host.tensor[update=False]().ptr[i] = Scalar[c_type](0)
        c_host_ref.tensor[update=False]().ptr[i] = Scalar[c_type](0)

    # Move operands to the Device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    ctx.enqueue_copy(c_device, c_host.tensor[update=False]().ptr)
    ctx.enqueue_copy(c_device_ref, c_host_ref.tensor[update=False]().ptr)

    blackwell_kernel_5[
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
            blackwell_kernel_5[
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

        # Warmup
        for _ in range(num_warmup):
            run_kernel(ctx)
        ctx.synchronize()
        print("finished warmup")

        var nstime = (
            Float64(ctx.execution_time(run_kernel, num_runs)) / num_runs
        )
        var sectime = nstime * 1e-9
        var TFlop = 2.0 * Float64(M) * Float64(N) * Float64(K) * 1e-12

        print("  Average time: ", sectime * 1000, " ms")
        print("  Performance: ", TFlop / sectime, " TFLOPS")
        print()
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

        ctx.enqueue_copy(c_host.tensor[update=False]().ptr, c_device)
        ctx.enqueue_copy(c_host_ref.tensor[update=False]().ptr, c_device_ref)
        ctx.synchronize()

        comptime rtol = 1e-2

        assert_almost_equal(
            c_host.tensor[update=False]().ptr,
            c_host_ref.tensor[update=False]().ptr,
            M * N,
            atol=0.0001,
            rtol=rtol,
        )

    # `a_device_lt` / `b_device_lt` are raw pointer views, so past the copies
    # above nothing else refers to the buffers backing them.
    __ownership_keepalive(a_device, b_device)

    print("\n=== TEST PASSED ===")


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
    comptime transpose_b = True

    comptime dic_of_shapes = make_dic_of_shapes()

    print("Shapes: [M, N, K]")

    comptime block_tile_shape = Index(128, 128, 64)
    comptime umma_shape = Index(256, 256, 16)

    comptime for i in range(len(dic_of_shapes)):
        comptime shape = get_dic_of_shapes(i, dic_of_shapes)
        print(
            "Benchmarking shape: [",
            shape[0],
            ",",
            shape[1],
            ",",
            shape[2],
            "]",
        )
        test_blackwell_kernel_5[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            block_tile_shape,
            umma_shape,
            cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
            a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            c_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            benchmark=True,
            M=4096,
            N=2560,
            K=8192,
        ](ctx)


def main() raises:
    with DeviceContext() as ctx:
        if is_benchmark():
            # Run the benchmark
            print("\n\n========== Running Benchmarks ==========\n")
            benchmark_blackwell_matmul(ctx)
            return

        comptime block_tile_shape = Index(128, 128, 64)
        comptime umma_shape = Index(256, 256, 16)

        test_blackwell_kernel_5[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            block_tile_shape,
            umma_shape,
            cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
            a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            c_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            M=4096,
            N=4096,
            K=4096,
        ](ctx)
