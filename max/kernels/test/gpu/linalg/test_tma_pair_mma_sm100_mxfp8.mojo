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

from std.math import align_up
from std.math.uutils import umod, ufloordiv
from std.sys import size_of
import linalg.matmul.vendor.blas as vendor_blas
from max.gpu import (
    warp_id as get_warp_id,
    block_id_in_cluster,
    block_idx,
    lane_id,
)
from max.gpu.sync import barrier
from max.gpu.primitives.cluster import (
    block_rank_in_cluster,
    cluster_sync,
    elect_one_sync_with_mask,
)
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.memory import external_memory
from max.gpu.compute.arch.mma_nvidia_sm100 import *
from max.gpu.compute.arch.tcgen05 import *
from layout import (
    CoordLike,
    Coord,
    Idx,
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    row_major,
)
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_to_descriptor,
    tile_sf_layout_k_major,
)
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    _idx_product,
    create_tensor_tile,
)
from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple
from internal_utils import assert_almost_equal
from std.random import rand
from std.math import ceildiv
from max.gpu.sync import syncwarp
from std.sys import argv
from std.random import random_ui64
from linalg.fp4_utils import (
    convert_ref_scales_to_mxfp8_format,
    MXFP8_SF_VECTOR_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    MXFP8_SF_DTYPE,
)
from linalg.matmul.vendor.blas import matmul


def simple_init() -> Bool:
    for arg in argv():
        if arg == "--simple-init":
            return True
    return False


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(a_scales_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_scales_tma_op, `nvvm.grid_constant`)
def blockscaled_pair_cta_mxfp8[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    a_tile_rank: Int,
    a_tile_shape: IndexList[a_tile_rank],
    a_desc_shape: IndexList[a_tile_rank],
    b_tile_rank: Int,
    b_tile_shape: IndexList[b_tile_rank],
    b_desc_shape: IndexList[b_tile_rank],
    a_scales_tile_rank: Int,
    a_scales_tile_shape: IndexList[a_scales_tile_rank],
    a_scales_desc_shape: IndexList[a_scales_tile_rank],
    b_scales_tile_rank: Int,
    b_scales_tile_shape: IndexList[b_scales_tile_rank],
    b_scales_desc_shape: IndexList[b_scales_tile_rank],
    c_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    cta_group: Int = 1,
](
    a_tma_op: TMATensorTile[a_type, a_tile_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[b_type, b_tile_rank, b_tile_shape, b_desc_shape],
    a_scales_tma_op: TMATensorTile[
        a_scales_type,
        a_scales_tile_rank,
        a_scales_tile_shape,
        a_scales_desc_shape,
    ],
    b_scales_tma_op: TMATensorTile[
        b_scales_type,
        b_scales_tile_rank,
        b_scales_tile_shape,
        b_scales_desc_shape,
    ],
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    num_iters_dev: Int32,
):
    var num_iters = Int(num_iters_dev)
    comptime assert (
        a_type == b_type == .float8_e4m3fn
    ), "a_type and b_type must be the same and either float8_e4m3fn"

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]

    comptime CLUSTER_M = Int(cluster_shape[0])
    comptime CLUSTER_N = Int(cluster_shape[1])

    comptime a_tma_load_size = _idx_product[a_tile_rank, a_desc_shape]()
    comptime b_tma_load_size = _idx_product[b_tile_rank, b_desc_shape]()
    comptime a_tma_rows = a_desc_shape[0]
    comptime b_tma_rows = b_desc_shape[0]

    comptime a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    comptime b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()

    comptime assert (
        BK == 128 and BM == 128 and MMA_N == 128
    ), "Only support 128x128x128, 128x256x128, and 256x128x1128 block size"

    comptime a_scales_smem_layout = tile_sf_layout_k_major[
        BM, BK, MXFP8_SF_VECTOR_SIZE
    ]()
    comptime b_scales_smem_layout = tile_sf_layout_k_major[
        MMA_N, BK, MXFP8_SF_VECTOR_SIZE
    ]()

    var smem = external_memory[UInt8, address_space=.SHARED, alignment=8]()

    comptime a_smem_bytes = a_smem_layout.size() * size_of[a_type]()
    comptime b_smem_bytes = b_smem_layout.size() * size_of[b_type]()
    comptime a_scales_bytes = a_scales_smem_layout.size() * size_of[
        a_scales_type
    ]()
    comptime b_scales_bytes = b_scales_smem_layout.size() * size_of[
        b_scales_type
    ]()

    var a_smem = smem.bitcast[Scalar[a_type]]()
    var b_smem = (smem + a_smem_bytes).bitcast[Scalar[b_type]]()
    var a_scales_smem = (smem + a_smem_bytes + b_smem_bytes).bitcast[
        Scalar[a_scales_type]
    ]()
    var b_scales_smem = (
        smem + a_smem_bytes + b_smem_bytes + a_scales_bytes
    ).bitcast[Scalar[b_scales_type]]()

    var smem_pool = (
        smem + a_smem_bytes + b_smem_bytes + a_scales_bytes + b_scales_bytes
    ).bitcast[Int64]()

    var a_smem_tile = LayoutTensor[
        a_type,
        a_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ](a_smem.as_unsafe_any_origin())

    var b_smem_tile = LayoutTensor[
        b_type,
        b_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ](b_smem.as_unsafe_any_origin())

    var a_scales_smem_tile = LayoutTensor[
        a_scales_type,
        a_scales_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ](a_scales_smem.as_unsafe_any_origin())

    var b_scales_smem_tile = LayoutTensor[
        b_scales_type,
        b_scales_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ](b_scales_smem.as_unsafe_any_origin())

    comptime accum_type = get_accum_type[a_type]()

    # Shared memory pointer to hold tensor memory address
    var ptr_tmem_addr = (smem_pool.bitcast[Int64]() + 4).bitcast[UInt32]()

    comptime c_frag_size = MMA_M * MMA_N // 128 // cta_group
    var c_frag: Array[Scalar[accum_type], c_frag_size]

    comptime a_expected_bytes = a_smem_layout.size() * size_of[a_type]()
    comptime b_expected_bytes = b_smem_layout.size() * size_of[b_type]()
    # Leader CTAs expect SMEM from itself and their peers
    comptime expected_bytes = cta_group * (
        a_expected_bytes + b_expected_bytes
    ) + a_scales_bytes + b_scales_bytes

    var tma_mbar_ptr = smem_pool.bitcast[Int64]()
    var mma_mbar_ptr = smem_pool.bitcast[Int64]() + 2

    var tma_mbar = tma_mbar_ptr.bitcast[SharedMemBarrier]()
    var mma_mbar = mma_mbar_ptr.bitcast[SharedMemBarrier]()

    var elect_one_warp = get_warp_id() == 0
    var elect_one_thread = elect_one_sync_with_mask()
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

    comptime SFA_NUM_COLS = BM // 32
    comptime SFB_NUM_COLS = MMA_N // 32
    var a_scales_tmem_addr_start = tmem_addr + UInt32(MMA_N)
    var b_scales_tmem_addr_start = a_scales_tmem_addr_start + UInt32(
        SFA_NUM_COLS
    )

    comptime a_canonical_layout = tile_to_descriptor[a_type, a_smem_layout]()
    comptime b_canonical_layout = tile_to_descriptor[
        b_type, b_smem_layout, is_k_major=transpose_b
    ]()
    comptime aSBO = a_canonical_layout[0].stride[1].value() * size_of[a_type]()
    comptime aLBO = a_canonical_layout[1].stride[1].value() * size_of[a_type]()
    comptime b_stride01 = b_canonical_layout[0].stride[1].value()
    comptime b_stride11 = b_canonical_layout[1].stride[1].value()
    comptime b_k_stride = b_stride11 * 2 * size_of[b_type]()
    comptime bSBO = (b_stride01 if transpose_b else b_stride11) * size_of[
        b_type
    ]()
    comptime bLBO = (b_stride11 if transpose_b else b_stride01) * size_of[
        b_type
    ]()

    var adesc_base = MMASmemDescriptor.create[aSBO, aLBO, a_swizzle](
        a_smem_tile.ptr
    )
    var bdesc_base = MMASmemDescriptor.create[bSBO, bLBO, b_swizzle](
        b_smem_tile.ptr
    )

    var idesc = UMMAInsDescriptor[UMMAKind.KIND_MXF8F6F4].create[
        accum_type,
        a_type,
        b_type,
        a_scales_type,
        Index[dtype=.uint32](mma_shape[0], mma_shape[1]),
        transpose_b=transpose_b,
    ]()

    var rank_m = block_id_in_cluster.x
    var rank_n = block_id_in_cluster.y

    # (peer_id, mma_coord_m, mma_coord_n)
    var peer_cta_coord = (
        umod(rank_m, cta_group),
        ufloordiv(rank_m, cta_group),
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

    var a_mma_mask = a_multicast_mask >> UInt16(peer_cta_coord[0])
    var b_mma_mask = b_multicast_mask >> UInt16(peer_cta_coord[0])
    var c_mma_mask: UInt16 = (a_mma_mask | a_mma_mask << 1) | (
        b_mma_mask | b_mma_mask << 1
    )

    for k_iter in range(num_iters):
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

            var a_smem_reshape = a_smem_tile.reshape[Layout.row_major(BM, BK)]()
            var b_smem_reshape = b_smem_tile.reshape[Layout.row_major(BN, BK)]()

            a_tma_op.async_multicast_load[cta_group](
                a_smem_reshape.split[CLUSTER_N]()[peer_cta_coord[2]],
                tma_mbar[0],
                (k_iter * BK, a_gmem_slice_coord),
                a_multicast_mask,
            )

            b_tma_op.async_multicast_load[cta_group](
                b_smem_reshape.split[CLUSTER_M // cta_group]()[
                    peer_cta_coord[1]
                ],
                tma_mbar[0],
                (k_iter * BK, b_gmem_slice_coord),
                b_multicast_mask,
            )

            a_scales_tma_op.async_copy_4d(
                a_scales_smem_tile,
                tma_mbar[0],
                (
                    0,
                    0,
                    k_iter,
                    block_idx.x * (BM // SF_MN_GROUP_SIZE),
                ),
            )

            b_scales_tma_op.async_copy_4d(
                b_scales_smem_tile,
                tma_mbar[0],
                (
                    0,
                    0,
                    k_iter,
                    block_idx.y * (MMA_N // SF_MN_GROUP_SIZE),
                ),
            )

        if elect_one_cta:
            tma_mbar[0].wait(tma_phase)
            tma_phase ^= 1

            if elect_one_warp:
                if elect_one_thread:
                    comptime for i in range(BM // SF_MN_GROUP_SIZE):
                        comptime idx = IntTuple(i * SF_ATOM_M[0], 0)
                        comptime a_scales_offset = a_scales_smem_layout(
                            idx
                        ) * size_of[a_scales_type]()
                        var a_scales_tmem_addr = (
                            a_scales_tmem_addr_start
                            + UInt32(i * (SF_MN_GROUP_SIZE // 32))
                        )
                        var a_scales_desc = MMASmemDescriptor.create[
                            8 * 16, 0, TensorMapSwizzle.SWIZZLE_NONE
                        ](a_scales_smem_tile.ptr + a_scales_offset)
                        tcgen05_cp[
                            cta_group=2,
                            datapaths=32,
                            bits=128,
                            multicast="warpx4",
                        ](a_scales_tmem_addr, a_scales_desc)

                    comptime for i in range(MMA_N // SF_MN_GROUP_SIZE):
                        comptime idx = IntTuple(i * SF_ATOM_M[0], 0)
                        comptime b_scales_offset = b_scales_smem_layout(
                            idx
                        ) * size_of[b_scales_type]()
                        var b_scales_tmem_addr = (
                            b_scales_tmem_addr_start
                            + UInt32(i * (SF_MN_GROUP_SIZE // 32))
                        )
                        var b_scales_desc = MMASmemDescriptor.create[
                            8 * 16, 0, TensorMapSwizzle.SWIZZLE_NONE
                        ](b_scales_smem_tile.ptr + b_scales_offset)
                        tcgen05_cp[
                            cta_group=2,
                            datapaths=32,
                            bits=128,
                            multicast="warpx4",
                        ](b_scales_tmem_addr, b_scales_desc)

                syncwarp()

            barrier()

            if elect_one_warp:
                var adesc = adesc_base
                var bdesc = bdesc_base

                if k_iter == 0:
                    if elect_one_thread:
                        var runtime_desc = UMMAInsDescriptor[
                            UMMAKind.KIND_MXF8F6F4
                        ].update_desc_with_sf_id[0](
                            idesc,
                        )
                        mma[cta_group,](
                            adesc,
                            bdesc,
                            tmem_addr,
                            runtime_desc,
                            a_scales_tmem_addr_start,
                            b_scales_tmem_addr_start,
                            c_scale=0,
                        )

                    comptime for j in range(1, BK // mma_shape[2]):
                        adesc += mma_shape[2] * size_of[a_type]()
                        bdesc += b_k_stride
                        if elect_one_thread:
                            var runtime_desc = UMMAInsDescriptor[
                                UMMAKind.KIND_MXF8F6F4
                            ].update_desc_with_sf_id[UInt32(j)](
                                idesc,
                            )
                            mma[cta_group](
                                adesc,
                                bdesc,
                                tmem_addr,
                                runtime_desc,
                                a_scales_tmem_addr_start,
                                b_scales_tmem_addr_start,
                                c_scale=1,
                            )
                else:
                    comptime for j in range(BK // mma_shape[2]):
                        if elect_one_thread:
                            var runtime_desc = UMMAInsDescriptor[
                                UMMAKind.KIND_MXF8F6F4
                            ].update_desc_with_sf_id[UInt32(j)](
                                idesc,
                            )
                            mma[cta_group](
                                adesc,
                                bdesc,
                                tmem_addr,
                                runtime_desc,
                                a_scales_tmem_addr_start,
                                b_scales_tmem_addr_start,
                                c_scale=1,
                            )
                        adesc += mma_shape[2] * size_of[a_type]()
                        bdesc += b_k_stride

                if elect_one_thread:
                    mma_arrive_multicast[cta_group](mma_mbar, c_mma_mask)
        mma_mbar[0].wait(mma_phase)
        mma_phase ^= 1

    c_frag = tcgen05_ld[
        datapaths=32,
        bits=32,
        repeat=BN if MMA_M == 128 else MMA_N,
        dtype=accum_type,
        pack=False,
        width=c_frag_size,
    ](tmem_addr)
    tcgen05_load_wait()

    if elect_one_warp:
        tcgen05_release_allocation_lock[Int32(cta_group)]()
        tcgen05_dealloc[Int32(cta_group)](tmem_addr, max_tmem_cols)

    var warp_id = get_warp_id()

    var c_gmem_block = c.tile[MMA_M, MMA_N](
        peer_cta_coord[1], peer_cta_coord[2]
    )
    var c_gmem_slice = c_gmem_block.tile[BM, MMA_N](peer_cta_coord[0], 0)

    comptime if MMA_M == 128:
        var c_gmem_frag = c_gmem_slice.tile[BM // 2, BN](
            warp_id % 2, warp_id // 2
        ).vectorize[1, 2]()

        comptime for i in range(c_frag_size // 2):
            c_gmem_frag[lane_id(), i] = rebind[c_gmem_frag.element_type](
                SIMD[accum_type, 2](
                    rebind[Scalar[accum_type]](c_frag[2 * i]),
                    rebind[Scalar[accum_type]](c_frag[2 * i + 1]),
                ).cast[c_type]()
            )
    else:
        var c_gmem_frag = c_gmem_slice.tile[BM // 4, MMA_N](
            warp_id, 0
        ).vectorize[1, 2]()

        comptime for i in range(c_frag_size // 2):
            c_gmem_frag[lane_id(), i] = rebind[c_gmem_frag.element_type](
                SIMD[accum_type, 2](
                    rebind[Scalar[accum_type]](c_frag[2 * i]),
                    rebind[Scalar[accum_type]](c_frag[2 * i + 1]),
                ).cast[c_type]()
            )


def sm100_blockscaled_mxfp8_cta_pair[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    a_scales_layout: Layout,
    b_scales_layout: Layout,
    //,
    *,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int,
    cluster_shape: StaticTuple[Int32, 3],
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
](
    c: LayoutTensor[mut=True, c_type, c_layout, _],
    a: LayoutTensor[mut=True, a_type, a_layout, _],
    b: LayoutTensor[mut=True, b_type, b_layout, _],
    a_scales: LayoutTensor[mut=True, a_scales_type, a_scales_layout, _],
    b_scales: LayoutTensor[mut=True, b_scales_type, b_scales_layout, _],
    ctx: DeviceContext,
) raises:
    comptime assert transpose_b, "Only support transposed B"

    comptime assert (
        a_type == b_type and a_type == .float8_e4m3fn
    ), "Only support float8_e4m3fn"

    var M = c.dim(0)
    comptime N = c_layout.shape[1].value()
    comptime K = a_layout.shape[1].value()

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]

    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]

    comptime assert MMA_M == 256 and MMA_N in (
        128,
        256,
    ), "MMA_M and MMA_N must be divisible by 128"

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

    comptime assert (
        a_scales_type == b_scales_type and a_scales_type == MXFP8_SF_DTYPE
    ), "Only support F8-UE8M0 scales"
    comptime assert (
        a_scales.rank == b_scales.rank == 5
    ), "a_scales and b_scales must be 5D tensors"
    comptime assert (
        a_scales_layout.shape[2].value()
        == b_scales_layout.shape[2].value()
        == SF_ATOM_M[0]
    ), ""
    comptime assert (
        a_scales_layout.shape[3].value()
        == b_scales_layout.shape[3].value()
        == SF_ATOM_M[1]
    ), ""
    comptime assert (
        a_scales_layout.shape[4].value()
        == b_scales_layout.shape[4].value()
        == SF_ATOM_K
    ), ""

    comptime scales_4d_layout[layout: Layout] = Layout.row_major(
        layout.shape[0].value(),
        layout.shape[1].value(),
        SF_ATOM_M[0],
        SF_ATOM_M[1] * SF_ATOM_K,
    )
    comptime a_scales_4d_layout = scales_4d_layout[a_scales_layout]
    comptime b_scales_4d_layout = scales_4d_layout[b_scales_layout]

    var a_scales_4d = LayoutTensor[
        a_scales_type,
        a_scales_4d_layout,
    ](
        a_scales.ptr,
        RuntimeLayout[a_scales_4d_layout].row_major(
            IndexList[4](
                a_scales.dim(0),
                a_scales.dim(1),
                a_scales.dim(2),
                a_scales.dim(3) * a_scales.dim(4),
            ),
        ),
    )
    var b_scales_4d = LayoutTensor[
        b_scales_type,
        b_scales_4d_layout,
    ](
        b_scales.ptr,
        RuntimeLayout[b_scales_4d_layout].row_major(
            IndexList[4](
                b_scales.dim(0),
                b_scales.dim(1),
                b_scales.dim(2),
                b_scales.dim(3) * b_scales.dim(4),
            ),
        ),
    )

    var a_scales_tma_op = create_tensor_tile[
        Index(
            BM // SF_MN_GROUP_SIZE, 1, SF_ATOM_M[0], SF_ATOM_M[1] * SF_ATOM_K
        ),
        swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
        __tile_shape=Index(
            BM // SF_MN_GROUP_SIZE, 1, SF_ATOM_M[0], SF_ATOM_M[1] * SF_ATOM_K
        ),
    ](ctx, a_scales_4d)

    var b_scales_tma_op = create_tensor_tile[
        Index(
            MMA_N // SF_MN_GROUP_SIZE, 1, SF_ATOM_M[0], SF_ATOM_M[1] * SF_ATOM_K
        ),
        swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
        __tile_shape=Index(
            MMA_N // SF_MN_GROUP_SIZE, 1, SF_ATOM_M[0], SF_ATOM_M[1] * SF_ATOM_K
        ),
    ](ctx, b_scales_4d)

    comptime sf_block_atom_size = SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K

    comptime smem_size = BM * BK * size_of[a_type]() + BN * BK * size_of[
        b_type
    ]() + (
        (BM // SF_MN_GROUP_SIZE + MMA_N // SF_MN_GROUP_SIZE)
        * sf_block_atom_size
        * size_of[a_scales_type]()
    ) + 16 + 16 + 16

    comptime kernel = blockscaled_pair_cta_mxfp8[
        a_type,
        b_type,
        c_type,
        a_scales_type,
        b_scales_type,
        type_of(a_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).rank,
        type_of(b_tma_op).tile_shape,
        type_of(b_tma_op).desc_shape,
        type_of(a_scales_tma_op).rank,
        type_of(a_scales_tma_op).tile_shape,
        type_of(a_scales_tma_op).desc_shape,
        type_of(b_scales_tma_op).rank,
        type_of(b_scales_tma_op).tile_shape,
        type_of(b_scales_tma_op).desc_shape,
        c_layout,
        block_tile_shape,
        mma_shape,
        transpose_b=transpose_b,
        cluster_shape=cluster_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        cta_group=cta_group,
    ]

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        a_scales_tma_op,
        b_scales_tma_op,
        c,
        Int32(ceildiv(K, BK)),
        grid_dim=(
            align_up(ceildiv(M, BM), Int(cluster_shape[0])),
            align_up(ceildiv(N, BN) // cta_group, Int(cluster_shape[1])),
            1,
        ),
        block_dim=(128),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_size)
        ),
    )


def test_blockscaled_pair_cta_mxfp8[
    MType: CoordLike,
    NType: CoordLike,
    //,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    block_tile_shape: IndexList[3],
    umma_shape: IndexList[3],
    cta_group: Int,
    cluster_shape: StaticTuple[Int32, 3],
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    *,
    k: Int,
](ctx: DeviceContext, m: MType, n: NType) raises:
    comptime assert transpose_b, "transpose_b must be true"

    var M = Int(m.value())
    var N = Int(n.value())
    var K = k

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]

    if N % BN != 0:
        raise Error("N must be divisible by BN")

    comptime scales_type = MXFP8_SF_DTYPE
    comptime ref_scales_type = DType.float32

    # Initialize reference scales
    comptime REF_BLOCK_SCALE = 128

    var ref_a_scales_shape = Coord(Idx[ceildiv(k, REF_BLOCK_SCALE)], m)
    var ref_b_scales_shape = Coord(
        ceildiv(N, REF_BLOCK_SCALE),
        Idx[ceildiv(k, REF_BLOCK_SCALE)],
    )

    var ref_a_scales_size = ceildiv(k, REF_BLOCK_SCALE) * M
    var ref_b_scales_size = ceildiv(N, REF_BLOCK_SCALE) * ceildiv(
        k, REF_BLOCK_SCALE
    )

    var a_scales_host_ref_ptr = ctx.enqueue_create_host_buffer[ref_scales_type](
        ref_a_scales_size
    )
    var a_scales_host_ref = TileTensor(
        a_scales_host_ref_ptr, row_major(ref_a_scales_shape)
    )
    var b_scales_host_ref_ptr = ctx.enqueue_create_host_buffer[ref_scales_type](
        ref_b_scales_size
    )
    var b_scales_host_ref = TileTensor(
        b_scales_host_ref_ptr, row_major(ref_b_scales_shape)
    )

    var a_scales_device_ref = ctx.enqueue_create_buffer[ref_scales_type](
        ref_a_scales_size
    )
    var b_scales_device_ref = ctx.enqueue_create_buffer[ref_scales_type](
        ref_b_scales_size
    )

    for i in range(ref_a_scales_size):
        a_scales_host_ref_ptr[i] = Scalar[ref_scales_type](1.0)
    for i in range(ref_b_scales_size):
        b_scales_host_ref_ptr[i] = Scalar[ref_scales_type](1.0)

    comptime assert a_scales_host_ref.flat_rank == 2
    for i in range(ceildiv(k, REF_BLOCK_SCALE)):
        for j in range(M // 32):
            for k in range(32):
                a_scales_host_ref[i, j * 32 + k] = (
                    1 << random_ui64(0, 3)
                ).cast[ref_scales_type]()

    for i in range(ceildiv(N, REF_BLOCK_SCALE)):
        for j in range(ceildiv(K, REF_BLOCK_SCALE)):
            b_scales_host_ref[Coord(i, j)] = (1 << random_ui64(0, 3)).cast[
                ref_scales_type
            ]()

    ctx.enqueue_copy(a_scales_device_ref, a_scales_host_ref_ptr)
    ctx.enqueue_copy(b_scales_device_ref, b_scales_host_ref_ptr)

    print(
        String(a_type)
        + "_"
        + String(b_type)
        + "_"
        + String(c_type)
        + " Problem Shape: "
        + String(Index(M, N, K))
        + " Block Tile Shape: "
        + String(block_tile_shape)
        + " UMMA Shape: "
        + String(umma_shape)
        + " Cluster Shape: "
        + String(cluster_shape[0])
        + " "
        + String(cluster_shape[1])
        + " "
        + String(cluster_shape[2])
        + " CTA Group: "
        + String(cta_group)
    )

    var a_shape = Coord(m, Idx[k])
    var b_shape = Coord(n, Idx[k])
    var c_shape = Coord(m, n)

    comptime SF_VECTOR_SIZE = 32
    comptime atom_m = (32, 4)
    comptime SF_ATOM_K = 4
    comptime sf_k = ceildiv(k, SF_VECTOR_SIZE)

    var a_scales_shape = Coord(
        ceildiv(M, SF_ATOM_M[0] * SF_ATOM_M[1]),
        Idx[ceildiv(sf_k, SF_ATOM_K)],
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )
    var b_scales_shape = Coord(
        ceildiv(N, SF_ATOM_M[0] * SF_ATOM_M[1]),
        Idx[ceildiv(sf_k, SF_ATOM_K)],
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )

    var a_scales_total = (
        ceildiv(M, SF_ATOM_M[0] * SF_ATOM_M[1])
        * Int(ceildiv(sf_k, SF_ATOM_K))
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )
    var b_scales_total = (
        ceildiv(N, SF_ATOM_M[0] * SF_ATOM_M[1])
        * Int(ceildiv(sf_k, SF_ATOM_K))
        * SF_ATOM_M[0]
        * SF_ATOM_M[1]
        * SF_ATOM_K
    )

    var a_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_type](
        a_scales_total
    )
    var a_scales_host = TileTensor(a_scales_host_ptr, row_major(a_scales_shape))
    var b_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_type](
        b_scales_total
    )
    var b_scales_host = TileTensor(b_scales_host_ptr, row_major(b_scales_shape))

    var a_scales_device = ctx.enqueue_create_buffer[scales_type](a_scales_total)
    var b_scales_device = ctx.enqueue_create_buffer[scales_type](b_scales_total)

    var a_size = M * k
    var b_size = N * k
    var c_size = M * N

    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)

    convert_ref_scales_to_mxfp8_format[
        REF_BLOCK_SIZE=REF_BLOCK_SCALE, SF_VECTOR_SIZE=MXFP8_SF_VECTOR_SIZE
    ](
        m,
        n,
        Idx[k],
        a_scales_host_ref.to_layout_tensor(),
        b_scales_host_ref.to_layout_tensor(),
        a_scales_host.to_layout_tensor(),
        b_scales_host.to_layout_tensor(),
    )
    # Initialize matmul operands
    if simple_init():
        var a_host_tt = TileTensor(a_host_ptr, row_major(a_shape))
        var b_host_tt = TileTensor(b_host_ptr, row_major(b_shape))
        comptime assert a_host_tt.flat_rank == 2
        comptime assert b_host_tt.flat_rank == 2

        for m in range(M):
            for k in range(K):
                a_host_tt[m, k] = Float32(k).cast[a_type]()
        for n in range(N):
            for k in range(K):
                b_host_tt[n, k] = Float32(1 if n == k else 0).cast[b_type]()
    else:
        rand(a_host_ptr.unsafe_ptr(), a_size)
        rand(b_host_ptr.unsafe_ptr(), b_size)

    # Move operands to the Device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)

    var a = TileTensor(a_device, row_major(a_shape))
    var b = TileTensor(b_device, row_major(b_shape))
    var c = TileTensor(c_device, row_major(c_shape))
    var a_scales = TileTensor(a_scales_device, row_major(a_scales_shape))
    var b_scales = TileTensor(b_scales_device, row_major(b_scales_shape))

    var c_ref = TileTensor(c_device_ref, row_major(c_shape))

    sm100_blockscaled_mxfp8_cta_pair[
        transpose_b=transpose_b,
        block_tile_shape=block_tile_shape,
        mma_shape=umma_shape,
        cta_group=cta_group,
        cluster_shape=cluster_shape,
    ](
        c.to_layout_tensor(),
        a.to_layout_tensor(),
        b.to_layout_tensor(),
        a_scales.to_layout_tensor(),
        b_scales.to_layout_tensor(),
        ctx,
    )

    matmul[scales_type=scales_type](
        ctx,
        c_ref,
        a,
        b,
        a_scales=a_scales.as_immut(),
        b_scales=b_scales.as_immut(),
        transpose_b=True,
        c_row_major=True,
    )

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)

    ctx.synchronize()

    assert_almost_equal(
        c_host_ptr.unsafe_ptr(),
        c_host_ref_ptr.unsafe_ptr(),
        c_size,
        atol=1e-3,
        rtol=1e-4,
    )


def main() raises:
    with DeviceContext() as ctx:
        comptime dtype = DType.float8_e4m3fn
        comptime MMA_K = 32
        comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
        comptime BK = (swizzle.bytes() // size_of[dtype]())

        test_blockscaled_pair_cta_mxfp8[
            dtype,
            dtype,
            .bfloat16,
            Index(128, 64, BK),
            Index(256, 128, MMA_K),
            cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            cta_group=2,
            k=3 * BK,
        ](ctx, Idx[256], Idx[128])

        test_blockscaled_pair_cta_mxfp8[
            dtype,
            dtype,
            .bfloat16,
            Index(128, 64, BK),
            Index(256, 128, MMA_K),
            cluster_shape=StaticTuple[Int32, 3](2, 2, 1),
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            cta_group=2,
            k=2 * BK,
        ](ctx, Idx[256], Idx[256])

        test_blockscaled_pair_cta_mxfp8[
            dtype,
            dtype,
            .bfloat16,
            Index(128, 64, BK),
            Index(256, 128, MMA_K),
            cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            cta_group=2,
            k=2 * BK,
        ](ctx, Idx[512], Idx[512])
