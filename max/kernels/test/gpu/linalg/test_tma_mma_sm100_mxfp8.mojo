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

from std.sys import size_of, argv
from max.gpu import (
    WARP_SIZE,
    warp_id as get_warp_id,
    block_idx,
    lane_id,
    thread_idx,
)
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.memory import external_memory
from max.gpu.compute.arch.mma_nvidia_sm100 import *
from max.gpu.compute.arch.tcgen05 import *
from layout import IntTuple, Layout, LayoutTensor, RuntimeLayout
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_to_descriptor,
    tile_sf_layout_k_major,
)
from max.gpu.primitives.cluster import block_rank_in_cluster
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    create_tensor_tile,
)
from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type
from std.math import ceildiv
from std.math.uutils import udivmod
from layout import CoordLike, Coord, Idx, TileTensor, row_major
from internal_utils import assert_almost_equal
from std.random import rand
from std.collections import Optional
from linalg.utils import elementwise_epilogue_type
from max.gpu.sync import syncwarp
from std.random import random_ui64
from linalg.fp4_utils import (
    convert_ref_scales_to_mxfp8_format,
    MXFP8_SF_VECTOR_SIZE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    MXFP8_SF_DTYPE,
)
from linalg.matmul.vendor.blas import matmul


def simple_init() -> Bool:
    for arg in argv():
        if arg == "--simple-init":
            return True
    return False


@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(a_scales_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_scales_tma_op, `nvvm.grid_constant`)
def block_scaled_mxfp8_kernel[
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
    umma_shape: IndexList[3],
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    num_threads: Int = 256,
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
    comptime assert num_threads == 256
    comptime assert (
        a_type == b_type and a_type == .float8_e4m3fn
    ), "Only support float8_e4m3fn"

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = umma_shape[0]
    comptime MMA_N = umma_shape[1]
    comptime MMA_K = umma_shape[2]
    comptime num_m_mmas = BM // MMA_M
    comptime num_n_mmas = BN // MMA_N
    comptime num_k_mmas = BK // MMA_K

    comptime a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    comptime b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()

    var smem = external_memory[UInt8, address_space=.SHARED, alignment=8]()
    var a_smem = smem.bitcast[Scalar[a_type]]()

    comptime a_smem_tile_t = LayoutTensor[
        a_type,
        a_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ]
    comptime b_smem_tile_t = LayoutTensor[
        b_type,
        b_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ]

    comptime assert BM == BK == 128 and BN in (
        128,
        256,
    ), "Only support 128x128x128 or 128x256x128 block size"

    comptime a_scales_smem_layout = tile_sf_layout_k_major[
        BM, BK, MXFP8_SF_VECTOR_SIZE
    ]()
    comptime b_scales_smem_layout = tile_sf_layout_k_major[
        BN, BK, MXFP8_SF_VECTOR_SIZE
    ]()

    comptime a_scales_smem_tile_t = LayoutTensor[
        a_scales_type,
        a_scales_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ]
    comptime b_scales_smem_tile_t = LayoutTensor[
        b_scales_type,
        b_scales_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ]

    comptime a_size = a_smem_layout.size()
    comptime b_size = b_smem_layout.size()
    comptime a_scales_size = a_scales_smem_layout.size()
    comptime b_scales_size = b_scales_smem_layout.size()

    comptime assert (
        (a_size * size_of[a_type]()) % 128
    ) == 0, "preserve alignment"
    comptime assert (
        (b_size * size_of[b_type]()) % 128
    ) == 0, "preserve alignment"
    comptime assert (
        (a_scales_size * size_of[a_scales_type]()) % 128
    ) == 0, "preserve alignment"
    comptime assert (
        (b_scales_size * size_of[b_scales_type]()) % 16
    ) == 0, "preserve alignment"

    var b_smem = (a_smem + a_size).bitcast[Scalar[b_type]]()
    var a_scales_smem = (b_smem + b_size).bitcast[Scalar[a_scales_type]]()
    var b_scales_smem = (a_scales_smem + a_scales_size).bitcast[
        Scalar[b_scales_type]
    ]()

    var a_smem_tile = a_smem_tile_t(a_smem.as_unsafe_any_origin())
    var b_smem_tile = b_smem_tile_t(b_smem.as_unsafe_any_origin())
    var a_scales_smem_tile = a_scales_smem_tile_t(
        a_scales_smem.as_unsafe_any_origin()
    )
    var b_scales_smem_tile = b_scales_smem_tile_t(
        b_scales_smem.as_unsafe_any_origin()
    )

    # Shared memory pointer to hold tensor memory address
    var ptr_tmem_addr = (b_scales_smem + b_scales_size).bitcast[UInt32]()

    comptime accum_type = get_accum_type[a_type]()

    comptime c_frag_size = MMA_M * MMA_N // num_threads
    var c_frag: Array[Scalar[accum_type], c_frag_size]

    comptime a_expected_bytes = a_size * size_of[a_type]()
    comptime b_expected_bytes = b_size * size_of[b_type]()
    comptime a_scales_expected_bytes = a_scales_size * size_of[a_scales_type]()
    comptime b_scales_expected_bytes = b_scales_size * size_of[b_scales_type]()
    comptime expected_bytes = a_expected_bytes + b_expected_bytes + a_scales_expected_bytes + b_scales_expected_bytes

    var tma_mbar = (ptr_tmem_addr + 2).bitcast[SharedMemBarrier]()
    var mma_mbar = tma_mbar + 1

    if thread_idx.x == 0:
        tma_mbar[0].init()
        mma_mbar[0].init()

    var tma_phase: UInt32 = 0
    var mma_phase: UInt32 = 0

    var elect_one_warp = get_warp_id() == 0
    var elect_one_thread = thread_idx.x == 0
    var elect_one_cta = block_rank_in_cluster() % 2 == 0
    comptime max_tmem_cols = 512

    var tmem_addr_ptr = (mma_mbar + 1).bitcast[UInt32]()
    if elect_one_warp:
        tcgen05_alloc[1](tmem_addr_ptr, max_tmem_cols)

    # Ensure all threads sees initialized mbarrier and
    # tensor memory allocation
    barrier()

    var tmem_addr = tmem_addr_ptr[0]

    comptime SFA_NUM_COLS = BM // 32
    comptime SFB_NUM_COLS = BN // 32
    var a_scales_tmem_addr_start = tmem_addr + UInt32(BN)
    var b_scales_tmem_addr_start = a_scales_tmem_addr_start + UInt32(
        SFA_NUM_COLS
    )

    if thread_idx.x >= 128:
        tmem_addr += 16 << 16  # offset for lane 16

    comptime a_canonical_layout = tile_to_descriptor[a_type, a_smem_layout]()
    comptime b_canonical_layout = tile_to_descriptor[
        b_type, b_smem_layout, is_k_major=transpose_b
    ]()
    comptime aSBO = a_canonical_layout[0].stride[1].value() * size_of[a_type]()
    comptime aLBO = a_canonical_layout[1].stride[1].value() * size_of[a_type]()
    comptime b_stride01 = b_canonical_layout[0].stride[1].value()
    comptime b_stride11 = b_canonical_layout[1].stride[1].value()
    comptime bSBO = (b_stride01 if transpose_b else b_stride11) * size_of[
        b_type
    ]()
    comptime bLBO = (b_stride11 if transpose_b else b_stride01) * size_of[
        b_type
    ]()

    var adesc = MMASmemDescriptor.create[aSBO, aLBO, a_swizzle](a_smem_tile.ptr)
    var bdesc = MMASmemDescriptor.create[bSBO, bLBO, b_swizzle](b_smem_tile.ptr)

    var idesc = UMMAInsDescriptor[UMMAKind.KIND_MXF8F6F4].create[
        accum_type,
        a_type,
        b_type,
        a_scales_type,
        Index[dtype=.uint32](umma_shape[0], umma_shape[1]),
        transpose_b=transpose_b,
    ]()

    for k_iter in range(num_iters):
        if elect_one_thread:
            tma_mbar[0].expect_bytes(Int32(expected_bytes))

            a_tma_op.async_copy(
                a_smem_tile,
                tma_mbar[0],
                (k_iter * BK, block_idx.y * BM),
            )
            b_tma_op.async_copy(
                b_smem_tile,
                tma_mbar[0],
                (
                    k_iter * BK,
                    block_idx.x * BN,
                ) if transpose_b else (
                    block_idx.x * BN,
                    k_iter * BK,
                ),
            )
            a_scales_tma_op.async_copy_4d(
                a_scales_smem_tile,
                tma_mbar[0],
                (
                    0,
                    0,
                    k_iter,
                    block_idx.y * (BM // SF_MN_GROUP_SIZE),
                ),
            )
            b_scales_tma_op.async_copy_4d(
                b_scales_smem_tile,
                tma_mbar[0],
                (
                    0,
                    0,
                    k_iter,
                    block_idx.x * (BN // SF_MN_GROUP_SIZE),
                ),
            )

        tma_mbar[0].wait(tma_phase)
        tma_phase ^= 1

        if elect_one_thread:
            comptime for i in range(BM // SF_MN_GROUP_SIZE):
                comptime idx = IntTuple(i * SF_ATOM_M[0], 0)
                comptime a_scales_offset = a_scales_smem_layout(idx) * size_of[
                    a_scales_type
                ]()
                var a_scales_tmem_addr = a_scales_tmem_addr_start + UInt32(
                    i * (SF_MN_GROUP_SIZE // 32)
                )
                var a_scales_desc = MMASmemDescriptor.create[
                    8 * 16, 0, TensorMapSwizzle.SWIZZLE_NONE
                ](a_scales_smem_tile.ptr + a_scales_offset)
                tcgen05_cp[
                    cta_group=1, datapaths=32, bits=128, multicast="warpx4"
                ](a_scales_tmem_addr, a_scales_desc)

            comptime for i in range(BN // SF_MN_GROUP_SIZE):
                comptime idx = IntTuple(i * SF_ATOM_M[0], 0)
                comptime b_scales_offset = b_scales_smem_layout(idx) * size_of[
                    b_scales_type
                ]()
                var b_scales_tmem_addr = b_scales_tmem_addr_start + UInt32(
                    i * (SF_MN_GROUP_SIZE // 32)
                )
                var b_scales_desc = MMASmemDescriptor.create[
                    8 * 16, 0, TensorMapSwizzle.SWIZZLE_NONE
                ](b_scales_smem_tile.ptr + b_scales_offset)
                tcgen05_cp[
                    cta_group=1, datapaths=32, bits=128, multicast="warpx4"
                ](b_scales_tmem_addr, b_scales_desc)

        syncwarp()

        barrier()

        if elect_one_thread:
            if k_iter == 0:
                var runtime_desc = UMMAInsDescriptor[
                    UMMAKind.KIND_MXF8F6F4
                ].update_desc_with_sf_id[0](
                    idesc,
                )
                mma(
                    adesc,
                    bdesc,
                    tmem_addr,
                    runtime_desc,
                    a_scales_tmem_addr_start,
                    b_scales_tmem_addr_start,
                    c_scale=0,
                )

                comptime for j in range(1, num_k_mmas):
                    runtime_desc = UMMAInsDescriptor[
                        UMMAKind.KIND_MXF8F6F4
                    ].update_desc_with_sf_id[UInt32(j)](
                        idesc,
                    )
                    comptime idx = IntTuple(0, MMA_K * j)
                    comptime a_offset = a_smem_layout(idx) * size_of[a_type]()
                    comptime b_offset = b_smem_layout(idx) * size_of[b_type]()
                    mma(
                        adesc + a_offset,
                        bdesc + b_offset,
                        tmem_addr,
                        runtime_desc,
                        a_scales_tmem_addr_start,
                        b_scales_tmem_addr_start,
                        c_scale=1,
                    )
            else:
                comptime for j in range(num_k_mmas):
                    var runtime_desc = UMMAInsDescriptor[
                        UMMAKind.KIND_MXF8F6F4
                    ].update_desc_with_sf_id[UInt32(j)](
                        idesc,
                    )
                    comptime idx = IntTuple(0, MMA_K * j)
                    comptime a_offset = a_smem_layout(idx) * size_of[a_type]()
                    comptime b_offset = b_smem_layout(idx) * size_of[b_type]()
                    mma(
                        adesc + a_offset,
                        bdesc + b_offset,
                        tmem_addr,
                        runtime_desc,
                        a_scales_tmem_addr_start,
                        b_scales_tmem_addr_start,
                        c_scale=1,
                    )

            mma_arrive(mma_mbar)

        mma_mbar[0].wait(mma_phase)
        mma_phase ^= 1

    c_frag = tcgen05_ld[
        datapaths=16,
        bits=256,
        repeat=BN // 8,
        dtype=accum_type,
        pack=False,
        width=c_frag_size,
    ](tmem_addr)

    tcgen05_load_wait()

    if elect_one_warp:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, max_tmem_cols)

    comptime num_warps = num_threads // WARP_SIZE
    var warp_id = get_warp_id()
    var warp_id_q, warp_id_r = udivmod(warp_id, 4)
    warp_id = 2 * warp_id_r + warp_id_q

    var ctile = c.tile[BM, BN](block_idx.y, block_idx.x)

    comptime for m_mma in range(num_m_mmas):
        comptime for n_mma in range(num_n_mmas):
            comptime mma_id = n_mma * num_m_mmas + m_mma

            var c_gmem_warp_tile = ctile.tile[MMA_M // num_warps, MMA_N](
                4 * m_mma + warp_id, n_mma
            )

            var c_gmem_frag = c_gmem_warp_tile.vectorize[1, 2]().distribute[
                Layout.row_major(8, 4)
            ](lane_id())

            comptime num_vecs_m = c_gmem_frag.layout.shape[0].value()
            comptime num_vecs_n = c_gmem_frag.layout.shape[1].value()

            comptime for n_vec in range(num_vecs_n):
                comptime for m_vec in range(num_vecs_m):
                    comptime i_vec = n_vec * num_vecs_m + m_vec

                    c_gmem_frag[m_vec, n_vec] = rebind[
                        c_gmem_frag.element_type
                    ](
                        SIMD[accum_type, 2](
                            c_frag[2 * i_vec], c_frag[2 * i_vec + 1]
                        ).cast[c_type]()
                    )


def sm100_block_scaled_mxfp8[
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
    *,
    transpose_b: Bool,
    umma_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    SF_VECTOR_SIZE: Int,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    accum_type: DType = get_accum_type[c_type](),
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

    comptime assert BM == BK == 128 and BN in (
        128,
        256,
    ), "Only support 128x128x128 or 128x256x128 block size"

    var a_tma_op = create_tensor_tile[Index(BM, BK), swizzle_mode=a_swizzle](
        ctx, a
    )
    var b_tma_op = create_tensor_tile[
        Index(BN, BK),
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

    var a_scales_4d = LayoutTensor[a_scales_type, a_scales_4d_layout](
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
            BN // SF_MN_GROUP_SIZE, 1, SF_ATOM_M[0], SF_ATOM_M[1] * SF_ATOM_K
        ),
        swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
        __tile_shape=Index(
            BN // SF_MN_GROUP_SIZE, 1, SF_ATOM_M[0], SF_ATOM_M[1] * SF_ATOM_K
        ),
    ](ctx, b_scales_4d)

    comptime block_dim = 256
    comptime sf_block_atom_size = SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K

    comptime smem_use = (BM + BN) * size_of[a_type]() * BK + (
        (BM // SF_MN_GROUP_SIZE + BN // SF_MN_GROUP_SIZE)
        * sf_block_atom_size
        * size_of[a_scales_type]()
    ) + 24
    comptime kernel = block_scaled_mxfp8_kernel[
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
        umma_shape,
        transpose_b=transpose_b,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        num_threads=block_dim,
    ]
    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        a_scales_tma_op,
        b_scales_tma_op,
        c,
        Int32(ceildiv(K, BK)),
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(block_dim),
        shared_mem_bytes=smem_use,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_use)
        ),
    )


def test_block_scaled_mxfp8[
    MType: CoordLike,
    NType: CoordLike,
    //,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    block_tile_shape: IndexList[3],
    umma_shape: IndexList[3],
    transpose_b: Bool,
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

    comptime MMA_M = umma_shape[0]
    comptime MMA_N = umma_shape[1]
    comptime MMA_K = umma_shape[2]

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
    )

    var a_shape = Coord(m, Idx[k])
    var b_shape = Coord(n, Idx[k])
    var c_shape = Coord(m, n)

    comptime SF_VECTOR_SIZE = 32
    comptime atom_m = (32, 4)
    comptime atom_k = 4
    comptime sf_k = ceildiv(k, SF_VECTOR_SIZE)

    var a_scales_shape = Coord(
        ceildiv(M, atom_m[0] * atom_m[1]),
        Idx[ceildiv(sf_k, atom_k)],
        Idx[atom_m[0]],
        Idx[atom_m[1]],
        Idx[atom_k],
    )
    var b_scales_shape = Coord(
        ceildiv(N, atom_m[0] * atom_m[1]),
        Idx[ceildiv(sf_k, atom_k)],
        Idx[atom_m[0]],
        Idx[atom_m[1]],
        Idx[atom_k],
    )

    var a_scales_total = (
        ceildiv(M, atom_m[0] * atom_m[1])
        * ceildiv(sf_k, atom_k)
        * atom_m[0]
        * atom_m[1]
        * atom_k
    )
    var b_scales_total = (
        ceildiv(N, atom_m[0] * atom_m[1])
        * ceildiv(sf_k, atom_k)
        * atom_m[0]
        * atom_m[1]
        * atom_k
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

    sm100_block_scaled_mxfp8[
        transpose_b=transpose_b,
        umma_shape=umma_shape,
        block_tile_shape=block_tile_shape,
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
    ](
        c.to_layout_tensor().as_unsafe_any_origin(),
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

    comptime rtol = 1e-2
    assert_almost_equal(
        c_host_ptr.unsafe_ptr(),
        c_host_ref_ptr.unsafe_ptr(),
        c_size,
        atol=0.0001,
        rtol=rtol,
    )


def main() raises:
    with DeviceContext() as ctx:
        comptime dtype = DType.float8_e4m3fn
        comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
        comptime BK = (swizzle.bytes() // size_of[dtype]())
        comptime MMA_M = 128
        comptime MMA_K = 32

        test_block_scaled_mxfp8[
            dtype,
            dtype,
            .bfloat16,
            Index(MMA_M, 256, BK),
            Index(MMA_M, 256, MMA_K),
            transpose_b=True,
            k=BK * 3,
        ](ctx, Idx[256], Idx[256])
        test_block_scaled_mxfp8[
            dtype,
            dtype,
            .bfloat16,
            Index(MMA_M, 256, BK),
            Index(MMA_M, 256, MMA_K),
            transpose_b=True,
            k=BK * 3,
        ](ctx, Idx[256], Idx[256 * 2])
        test_block_scaled_mxfp8[
            dtype,
            dtype,
            .bfloat16,
            Index(MMA_M, 256, BK),
            Index(MMA_M, 256, MMA_K),
            transpose_b=True,
            k=BK * 3,
        ](ctx, Idx[1000], Idx[256 * 4])

        test_block_scaled_mxfp8[
            dtype,
            dtype,
            .bfloat16,
            Index(MMA_M, 128, BK),
            Index(MMA_M, 128, MMA_K),
            transpose_b=True,
            k=BK * 3,
        ](ctx, Idx[256], Idx[2 * 128])
        test_block_scaled_mxfp8[
            dtype,
            dtype,
            .bfloat16,
            Index(MMA_M, 128, BK),
            Index(MMA_M, 128, MMA_K),
            transpose_b=True,
            k=BK * 2,
        ](ctx, Idx[256], Idx[3 * 128])
        test_block_scaled_mxfp8[
            dtype,
            dtype,
            .bfloat16,
            Index(MMA_M, 128, BK),
            Index(MMA_M, 128, MMA_K),
            transpose_b=True,
            k=BK * 3,
        ](ctx, Idx[1000], Idx[3 * 128])
