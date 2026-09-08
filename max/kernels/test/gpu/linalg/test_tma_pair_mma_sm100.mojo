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
from std.math.uutils import ufloordiv, umod
from std.sys import size_of

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.sync import barrier
from max.gpu.primitives.cluster import (
    block_rank_in_cluster,
    cluster_sync,
    elect_one_sync_with_mask,
)
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu import (
    block_id_in_cluster,
    block_idx,
    lane_id,
    warp_id,
)
from max.gpu.memory import external_memory
from max.gpu.compute.arch.mma_nvidia_sm100 import *
from max.gpu.compute.arch.tcgen05 import *
from layout import IntTuple, Layout, LayoutTensor
from layout._fillers import random
from layout._utils import ManagedLayoutTensor
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_to_descriptor,
)
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    _idx_product,
    create_tensor_tile,
)
from std.testing import assert_almost_equal

from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type, max_finite, min_finite
from std.utils.static_tuple import StaticTuple


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
def tma_umma_kernel_pair_cta[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_tma_rank: Int,
    b_tma_rank: Int,
    a_tile_shape: IndexList[a_tma_rank],
    b_tile_shape: IndexList[b_tma_rank],
    c_layout: Layout,
    a_desc_shape: IndexList[a_tma_rank],
    b_desc_shape: IndexList[b_tma_rank],
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    cta_group: Int = 1,
](
    a_tma_op: TMATensorTile[a_type, a_tma_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[b_type, b_tma_rank, b_tile_shape, b_desc_shape],
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    num_iters_dev: Int32,
):
    var num_iters = Int(num_iters_dev)
    comptime assert a_type == b_type and a_type in (
        DType.float8_e4m3fn,
        DType.bfloat16,
    ), "a_type and b_type must be the same and either float8_e4m3fn or bfloat16"

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime num_m_mmas = BM // mma_shape[0]
    comptime num_n_mmas = BN // mma_shape[1]

    comptime CLUSTER_M = Int(cluster_shape[0])
    comptime CLUSTER_N = Int(cluster_shape[1])

    comptime a_tma_load_size = _idx_product[a_tma_rank, a_desc_shape]()
    comptime b_tma_load_size = _idx_product[b_tma_rank, b_desc_shape]()
    comptime a_tma_rows = a_desc_shape[0]
    comptime b_tma_rows = b_desc_shape[0] if transpose_b else b_desc_shape[1]

    comptime a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    comptime b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()

    var smem = external_memory[UInt8, address_space=.SHARED, alignment=8]()

    comptime a_smem_bytes = a_smem_layout.size() * size_of[a_type]()
    comptime b_smem_bytes = b_smem_layout.size() * size_of[b_type]()

    var a_smem = smem.bitcast[Scalar[a_type]]()
    var b_smem = (smem + a_smem_bytes).bitcast[Scalar[b_type]]()

    var smem_pool = (smem + a_smem_bytes + b_smem_bytes).bitcast[Int64]()

    var a_smem_tile = LayoutTensor[
        a_type,
        a_smem_layout,
        address_space=.SHARED,
        alignment=128,
    ](a_smem.as_unsafe_any_origin())

    var b_smem_tile = LayoutTensor[
        b_type,
        b_smem_layout,
        address_space=.SHARED,
        alignment=128,
    ](b_smem.as_unsafe_any_origin())

    comptime accum_type = get_accum_type[a_type]()

    # Shared memory pointer to hold tensor memory address
    var ptr_tmem_addr = (smem_pool.bitcast[Int64]() + 4).bitcast[UInt32]()

    comptime c_frag_size = MMA_M * MMA_N // 128 // cta_group

    comptime a_expected_bytes = a_smem_layout.size() * size_of[a_type]()
    comptime b_expected_bytes = b_smem_layout.size() * size_of[b_type]()
    # Leader CTAs expect SMEM from itself and their peers
    comptime expected_bytes = cta_group * (a_expected_bytes + b_expected_bytes)

    var tma_mbar_ptr = smem_pool.bitcast[Int64]()
    var mma_mbar_ptr = smem_pool.bitcast[Int64]() + 2

    var tma_mbar = tma_mbar_ptr.bitcast[SharedMemBarrier]()
    var mma_mbar = mma_mbar_ptr.bitcast[SharedMemBarrier]()

    var elect_one_warp = warp_id() == 0
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

    var adesc_base = MMASmemDescriptor.create[aSBO, aLBO, a_swizzle](
        a_smem_tile.ptr
    )
    var bdesc_base = MMASmemDescriptor.create[bSBO, bLBO, b_swizzle](
        b_smem_tile.ptr
    )

    comptime mma_kind = UMMAKind.KIND_F8F6F4 if a_type == DType.float8_e4m3fn else UMMAKind.KIND_F16
    comptime idesc = UMMAInsDescriptor[mma_kind].create[
        accum_type,
        a_type,
        b_type,
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

    for i in range(num_iters):
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
                (i * BK, a_gmem_slice_coord),
                a_multicast_mask,
            )

            b_tma_op.async_multicast_load[cta_group](
                b_smem_reshape.split[CLUSTER_M // cta_group]()[
                    peer_cta_coord[1]
                ],
                tma_mbar[0],
                (i * BK, b_gmem_slice_coord) if transpose_b else (
                    b_gmem_slice_coord,
                    i * BK,
                ),
                b_multicast_mask,
            )

        if elect_one_cta:
            tma_mbar[0].wait(tma_phase)
            tma_phase ^= 1

            if elect_one_warp:
                if i == 0:
                    if elect_one_thread:
                        mma[cta_group, c_scale=0](
                            adesc_base, bdesc_base, tmem_addr, idesc
                        )

                    comptime for j in range(1, BK // mma_shape[2]):
                        comptime a_k_offset = j * mma_shape[2] * size_of[
                            a_type
                        ]()
                        comptime b_k_offset = b_smem_layout(
                            IntTuple(0, mma_shape[2] * j)
                        ) * size_of[b_type]()
                        if elect_one_thread:
                            mma[cta_group, c_scale=1](
                                adesc_base + a_k_offset,
                                bdesc_base + b_k_offset,
                                tmem_addr,
                                idesc,
                            )
                else:
                    comptime for j in range(BK // mma_shape[2]):
                        comptime a_k_offset = j * mma_shape[2] * size_of[
                            a_type
                        ]()
                        comptime b_k_offset = b_smem_layout(
                            IntTuple(0, mma_shape[2] * j)
                        ) * size_of[b_type]()
                        if elect_one_thread:
                            mma[cta_group, c_scale=1](
                                adesc_base + a_k_offset,
                                bdesc_base + b_k_offset,
                                tmem_addr,
                                idesc,
                            )

                if elect_one_thread:
                    mma_arrive_multicast[cta_group](mma_mbar, c_mma_mask)
        mma_mbar[0].wait(mma_phase)
        mma_phase ^= 1

    comptime total_repeat = BN if MMA_M == 128 else MMA_N
    comptime ld_repeat = min(total_repeat, 128)
    comptime num_ld_iters = total_repeat // ld_repeat
    comptime ld_width = c_frag_size // num_ld_iters

    var cluster_idx_m = ufloordiv(block_idx.x, CLUSTER_M)
    var cluster_idx_n = ufloordiv(block_idx.y, CLUSTER_N)
    var global_mma_m = (
        cluster_idx_m * (CLUSTER_M // cta_group) + peer_cta_coord[1]
    )
    var global_mma_n = cluster_idx_n * CLUSTER_N + peer_cta_coord[2]

    var c_gmem_block = c.tile[MMA_M, MMA_N](global_mma_m, global_mma_n)
    var c_gmem_slice = c_gmem_block.tile[BM, MMA_N](peer_cta_coord[0], 0)

    comptime for ld_i in range(num_ld_iters):
        var c_frag = tcgen05_ld[
            datapaths=32,
            bits=32,
            repeat=ld_repeat,
            dtype=accum_type,
            pack=False,
            width=ld_width,
        ](tmem_addr + UInt32(ld_i * ld_repeat))
        tcgen05_load_wait()

        comptime if ld_i == num_ld_iters - 1:
            if elect_one_warp:
                tcgen05_release_allocation_lock[Int32(cta_group)]()
                tcgen05_dealloc[Int32(cta_group)](tmem_addr, max_tmem_cols)

        comptime if MMA_M == 128:
            var c_gmem_frag = c_gmem_slice.tile[BM // 2, BN](
                umod(warp_id(), 2), ufloordiv(warp_id(), 2)
            ).vectorize[1, 2]()

            comptime for i in range(ld_width // 2):
                c_gmem_frag[lane_id(), i] = rebind[c_gmem_frag.element_type](
                    SIMD[accum_type, 2](
                        rebind[Scalar[accum_type]](c_frag[2 * i]),
                        rebind[Scalar[accum_type]](c_frag[2 * i + 1]),
                    ).cast[c_type]()
                )
        else:
            var c_gmem_frag = c_gmem_slice.tile[BM // 4, ld_repeat](
                warp_id(), ld_i
            ).vectorize[1, 2]()

            comptime for i in range(ld_width // 2):
                c_gmem_frag[lane_id(), i] = rebind[c_gmem_frag.element_type](
                    SIMD[accum_type, 2](
                        rebind[Scalar[accum_type]](c_frag[2 * i]),
                        rebind[Scalar[accum_type]](c_frag[2 * i + 1]),
                    ).cast[c_type]()
                )


def test_tma_umma_pair_cta[
    *,
    ab_type: DType,
    c_type: DType,
    prob_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    cta_group: Int = 1,
](ctx: DeviceContext) raises:
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]

    comptime MMA_M = 2 * BM
    comptime MMA_N = 2 * BN
    comptime MMA_K = 32 if ab_type == DType.float8_e4m3fn else 16
    comptime mma_shape = Index(MMA_M, MMA_N, MMA_K)
    comptime assert (BN if MMA_M == 128 else MMA_N) <= 256, String(
        "MMA_M = ",
        MMA_M,
        "\nBN = ",
        BN,
        "\nMMA_N = ",
        MMA_N,
        "\nab_type = ",
        ab_type,
        "\nswizzle = ",
        a_swizzle,
        "\nBK = ",
        BK,
    )

    print(
        "mma_"
        + "s"
        + "s_"
        + String(ab_type)
        + "_"
        + String(ab_type)
        + "_"
        + String(c_type)
        + " block tile "
        + String(block_tile_shape)
        + " transb="
        + String(transpose_b)
        + "; inst shape "
        + String(mma_shape)
        + " A "
        + String(a_swizzle)
        + " B "
        + String(b_swizzle)
    )

    comptime M = prob_shape[0]
    comptime N = prob_shape[1]
    comptime K = prob_shape[2]

    var a = ManagedLayoutTensor[
        ab_type,
        Layout.row_major(M, K),
    ](ctx)

    random(
        a.tensor[update=False](),
        min=min_finite[ab_type](),
        max=max_finite[ab_type](),
    )

    comptime b_layout = Layout.row_major(
        N, K
    ) if transpose_b else Layout.row_major(K, N)
    var b = ManagedLayoutTensor[ab_type, b_layout](ctx)
    var b_col_major = ManagedLayoutTensor[ab_type, Layout.row_major(N, K)](ctx)

    random(
        b.tensor[update=False](),
        min=min_finite[ab_type](),
        max=max_finite[ab_type](),
    )

    var c = ManagedLayoutTensor[
        c_type,
        Layout.row_major(M, N),
    ](ctx)

    var c_ref = ManagedLayoutTensor[
        c_type,
        Layout.row_major(M, N),
    ](ctx)

    var a_tma_op = create_tensor_tile[
        Index(Int32(BM) // cluster_shape[1], BK), swizzle_mode=a_swizzle
    ](ctx, a.device_tensor())
    var b_tma_op = create_tensor_tile[
        Index(
            Int32(BN) // (cluster_shape[0] // Int32(cta_group)), BK
        ) if transpose_b else Index(
            BK, Int32(BN) // (cluster_shape[0] // Int32(cta_group))
        ),
        swizzle_mode=b_swizzle,
    ](ctx, b.device_tensor())

    comptime smem_size = BM * BK * size_of[ab_type]() + BN * BK * size_of[
        ab_type
    ]() + 16 + 16 + 16

    comptime kernel = tma_umma_kernel_pair_cta[
        ab_type,
        ab_type,
        c_type,
        type_of(a_tma_op).rank,
        type_of(b_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(b_tma_op).tile_shape,
        Layout.row_major(M, N),
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).desc_shape,
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
        c.device_tensor(),
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

    comptime if ab_type == .float8_e4m3fn and (not transpose_b):
        # NOTE: Matrix B should always be in col-major layout for cublasLt to work
        var b_host_col_major = b_col_major.tensor()
        var b_tensor = b.tensor()
        for i in range(N):
            for j in range(K):
                b_host_col_major[i, j] = b_tensor[j, i]

        vendor_blas.matmul(
            ctx,
            c_ref.device_tensor[update=False](),
            a.device_tensor[update=False](),
            b_col_major.device_tensor[update=True](),
            c_row_major=True,
            transpose_b=True,
        )
    else:
        vendor_blas.matmul(
            ctx,
            c_ref.device_tensor[update=False](),
            a.device_tensor[update=False](),
            b.device_tensor[update=False](),
            c_row_major=True,
            transpose_b=transpose_b,
        )

    ctx.synchronize()

    var c_host = c.tensor()
    var c_host_ref = c_ref.tensor()
    for m in range(M):
        for n in range(N):
            # Increased tolerance for FP8/bfloat16 accumulation errors
            # FP8/bf16 matrix multiplication can have larger numerical errors
            # due to reduced precision in intermediate accumulations
            assert_almost_equal(
                c_host[m, n],
                c_host_ref[m, n],
                atol=0.01,
                rtol=0.01,
                msg=String(m) + ", " + String(n),
            )

    _ = a^
    _ = b^
    _ = b_col_major^
    _ = c^
    _ = c_ref^


def main() raises:
    with DeviceContext() as ctx:
        comptime for dtype in [DType.bfloat16, DType.float8_e4m3fn]:
            comptime for swizzle in [
                TensorMapSwizzle.SWIZZLE_32B,
                TensorMapSwizzle.SWIZZLE_64B,
                TensorMapSwizzle.SWIZZLE_128B,
            ]:
                comptime BK = (swizzle.bytes() // size_of[dtype]())

                test_tma_umma_pair_cta[
                    ab_type=dtype,
                    c_type=.bfloat16,
                    prob_shape=Index(512, 1024, 2 * BK),
                    block_tile_shape=Index(64, 64, BK),
                    transpose_b=True,
                    cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    cta_group=2,
                ](ctx)
                test_tma_umma_pair_cta[
                    ab_type=dtype,
                    c_type=.bfloat16,
                    prob_shape=Index(256, 1024, 2 * BK),
                    block_tile_shape=Index(64, 128, BK),
                    transpose_b=True,
                    cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    cta_group=2,
                ](ctx)

                # we skip for fp8 !transpose_b to avoid excessive BN
                test_tma_umma_pair_cta[
                    ab_type=dtype,
                    c_type=.bfloat16,
                    prob_shape=Index(512, 512, 2 * BK),
                    block_tile_shape=Index(128, 64, BK),
                    transpose_b=True,
                    cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    cta_group=2,
                ](ctx)

                comptime for transpose_b in [True, False]:
                    # BK is swizzle granularity
                    # if transpose_b, BN also gets divided by the cluster shape
                    # 2 * cluster_shape[0] // cta_group
                    comptime BN_BM64 = 128 if transpose_b else BK

                    test_tma_umma_pair_cta[
                        ab_type=dtype,
                        c_type=.bfloat16,
                        prob_shape=Index(128, 4 * BN_BM64, 2 * BK),
                        block_tile_shape=Index(64, BN_BM64, BK),
                        transpose_b=transpose_b,
                        cluster_shape=StaticTuple[Int32, 3](2, 2, 1),
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        cta_group=2,
                    ](ctx)

                    test_tma_umma_pair_cta[
                        ab_type=dtype,
                        c_type=.bfloat16,
                        prob_shape=Index(128, 2 * BN_BM64, 2 * BK),
                        block_tile_shape=Index(64, BN_BM64, BK),
                        transpose_b=transpose_b,
                        cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        cta_group=2,
                    ](ctx)
                    comptime BN_BM128 = 64 if transpose_b else BK

                    test_tma_umma_pair_cta[
                        ab_type=dtype,
                        c_type=.bfloat16,
                        prob_shape=Index(256, 2 * BN_BM128, 2 * BK),
                        block_tile_shape=Index(128, BN_BM128, BK),
                        transpose_b=transpose_b,
                        cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        cta_group=2,
                    ](ctx)

                    test_tma_umma_pair_cta[
                        ab_type=dtype,
                        c_type=.bfloat16,
                        prob_shape=Index(256, 4 * BN_BM128, 2 * BK),
                        block_tile_shape=Index(128, BN_BM128, BK),
                        transpose_b=transpose_b,
                        cluster_shape=StaticTuple[Int32, 3](2, 2, 1),
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        cta_group=2,
                    ](ctx)
