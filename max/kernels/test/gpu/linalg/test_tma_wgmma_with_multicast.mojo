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

from std.math import ceildiv
from std.sys import size_of

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.sync import barrier
from max.gpu.primitives.cluster import block_rank_in_cluster, cluster_sync
from max.gpu.host import DeviceContext, Dim
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu import block_idx, thread_idx
from max.gpu import warp_id as get_warp_id
from max.gpu.memory import fence_mbarrier_init
from layout import Layout, LayoutTensor
from layout._fillers import arange
from layout._utils import ManagedLayoutTensor
from layout.layout_tensor import copy_local_to_dram
from layout.tensor_core_async import (
    TensorCoreAsync,
    tile_layout_k_major,
    tile_layout_mn_major,
    warpgroup_fence,
)
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    _idx_product,
    create_tma_tile,
)
from std.memory import unsafe_stack_allocation
from std.testing import assert_almost_equal

from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type


@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
def multicast_tma_wgmma_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_tile_rank: Int,
    a_tile_shape: IndexList[a_tile_rank],
    a_desc_shape: IndexList[a_tile_rank],
    b_tile_rank: Int,
    b_tile_shape: IndexList[b_tile_rank],
    b_desc_shape: IndexList[b_tile_rank],
    c_layout: Layout,
    block_tile_shape: IndexList[3],
    wgmma_shape: IndexList[3],
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    cluster_shape: IndexList[3],
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    partitioned_multicast: Bool = False,
](
    a_tma_op: TMATensorTile[a_type, a_tile_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[b_type, b_tile_rank, b_tile_shape, b_desc_shape],
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    num_iters: Int,
):
    var a_smem_tile = LayoutTensor[
        a_type,
        a_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ].stack_allocation()

    var b_smem_tile = LayoutTensor[
        b_type,
        b_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ].stack_allocation()

    comptime accum_type = get_accum_type[a_type]()
    var wgmma_op = TensorCoreAsync[
        accum_type,
        a_type,
        b_type,
        wgmma_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        transpose_b=transpose_b,
    ]()

    comptime CLUSTER_M = cluster_shape[0]
    comptime CLUSTER_N = cluster_shape[1]

    comptime a_tma_load_size = _idx_product[a_tile_rank, a_desc_shape]()
    comptime b_tma_load_size = _idx_product[b_tile_rank, b_desc_shape]()
    comptime a_tma_rows = a_desc_shape[0]
    comptime b_tma_rows = b_desc_shape[0]

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime num_m_mmas = BM // wgmma_shape[0]
    comptime num_n_mmas = BN // wgmma_shape[1]

    comptime c_frag_size = wgmma_shape[0] * wgmma_shape[1] // 128
    var c_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, c_frag_size),
        MutAnyOrigin,
        address_space=.LOCAL,
    ].stack_allocation()

    _ = c_reg_tile.fill(0.0)

    comptime a_expected_bytes = a_smem_layout.size() * size_of[a_type]()
    comptime b_expected_bytes = b_smem_layout.size() * size_of[b_type]()
    comptime expected_bytes = a_expected_bytes + b_expected_bytes

    comptime CLUSTER_SIZE = CLUSTER_M * CLUSTER_N

    var block_rank = block_rank_in_cluster()
    var rank_m = block_rank / CLUSTER_N
    var rank_n = block_rank % CLUSTER_N

    var mbar = unsafe_stack_allocation[
        1,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()
    if thread_idx.x == 0:
        mbar[0].init()

    var phase: UInt32 = 0

    cluster_sync()
    fence_mbarrier_init()

    for i in range(num_iters):
        if thread_idx.x == 0:
            mbar[0].expect_bytes(expected_bytes)

            comptime if CLUSTER_N > 1:
                var multicast_mask = ((1 << CLUSTER_N) - 1) << (
                    rank_m * CLUSTER_N
                )

                comptime if partitioned_multicast:
                    var a_gmem_slice_coord = (
                        block_idx.y * BM + rank_n * a_tma_rows
                    )
                    var a_smem_slice = type_of(a_smem_tile)(
                        a_smem_tile.ptr + rank_n * a_tma_load_size
                    )

                    a_tma_op.async_multicast_load(
                        a_smem_slice,
                        mbar[0],
                        (i * BK, a_gmem_slice_coord),
                        multicast_mask.cast[.uint16](),
                    )

                else:
                    if rank_n == 0:
                        a_tma_op.async_multicast_load(
                            a_smem_tile,
                            mbar[0],
                            (i * BK, block_idx.y * BM),
                            multicast_mask.cast[.uint16](),
                        )

            else:
                a_tma_op.async_copy(
                    a_smem_tile,
                    mbar[0],
                    (i * BK, block_idx.y * BM),
                )

            comptime if CLUSTER_M > 1:
                var multicast_mask = 0
                for i in range(CLUSTER_M):
                    multicast_mask |= 1 << (i * CLUSTER_N)

                comptime if partitioned_multicast:
                    var b_gmem_slice_coord = (
                        block_idx.x * BN + rank_m * b_tma_rows
                    )
                    var b_smem_slice = type_of(b_smem_tile)(
                        b_smem_tile.ptr + rank_m * b_tma_load_size
                    )

                    b_tma_op.async_multicast_load(
                        b_smem_slice,
                        mbar[0],
                        (
                            i * BK,
                            b_gmem_slice_coord,
                        ) if transpose_b else (
                            block_idx.x * BN,
                            i * BK,
                        ),
                        (multicast_mask << rank_n).cast[.uint16](),
                    )

                else:
                    if rank_m == 0:
                        b_tma_op.async_multicast_load(
                            b_smem_tile,
                            mbar[0],
                            (
                                i * BK,
                                block_idx.x * BN,
                            ) if transpose_b else (
                                block_idx.x * BN,
                                i * BK,
                            ),
                            (multicast_mask << rank_n).cast[.uint16](),
                        )

            else:
                b_tma_op.async_copy(
                    b_smem_tile,
                    mbar[0],
                    (
                        i * BK,
                        block_idx.x * BN,
                    ) if transpose_b else (
                        block_idx.x * BN,
                        i * BK,
                    ),
                )

        # Ensure all threads sees initialized mbarrier
        barrier()

        mbar[0].wait(phase)
        phase ^= 1

        warpgroup_fence(c_reg_tile)
        wgmma_op.arrive()
        wgmma_op.wgmma(a_smem_tile, b_smem_tile, c_reg_tile)
        wgmma_op.commit_group()
        warpgroup_fence(c_reg_tile)
        wgmma_op.wait_group()

        barrier()

    var c_gmem_tile = c.tile[BM, BN](block_idx.y, block_idx.x)
    var warp_id = get_warp_id()

    comptime for m_mma in range(num_m_mmas):
        comptime for n_mma in range(num_n_mmas):
            comptime mma_id = n_mma * num_m_mmas + m_mma

            # (m_mma, n_mma) is coordinates for a warp group's tile.
            # A warp group is 4x1 warps.
            var warp_tile = c_gmem_tile.tile[
                wgmma_shape[0] // 4, wgmma_shape[1]
            ](m_mma * 4 + warp_id, n_mma)

            # Tile at (mma_id, 0) is a long vector containing all fragments
            # for this warp.
            var c_frag = c_reg_tile.tile[1, c_frag_size](mma_id, 0)

            # A warp is organized as row_major(8, 4) and each thread owns 2 contiguous
            # elementwise. This pattern repeats to fill the warp tile.
            copy_local_to_dram[Layout.row_major(8, 4)](
                warp_tile.vectorize[1, 2](), c_frag.vectorize[1, 2]()
            )

    cluster_sync()


def test_multicast_tma_wgmma[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    prob_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    wgmma_shape: IndexList[3],
    cluster_shape: IndexList[3],
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    partitioned_multicast: Bool = False,
](ctx: DeviceContext) raises:
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]

    comptime assert (
        transpose_b
    ), "multicasting is only supported for K-Major transposed B"

    comptime assert (
        not partitioned_multicast
        or a_swizzle.bytes() // size_of[a_type]() == BK
    ), (
        "Currently partitioned multi-casting is only supported when BK =="
        " (a_swizzle.bytes // size_of[a_type]"
    )
    comptime assert (
        not partitioned_multicast
        or b_swizzle.bytes() // size_of[b_type]() == BK
    ), (
        "Currently partitioned multi-casting is only supported when BK =="
        " (b_swizzle.bytes // size_of[b_type]"
    )

    comptime WGMMA_M = wgmma_shape[0]
    comptime WGMMA_N = wgmma_shape[1]
    comptime WGMMA_K = wgmma_shape[2]

    print(
        "wgmma_bf16_bf16_f32 cluster shape "
        + String(cluster_shape)
        + " block tile "
        + String(block_tile_shape)
        + " transb inst shape "
        + String(wgmma_shape)
        + " A "
        + String(a_swizzle)
        + " B "
        + String(b_swizzle)
        + " partitioned multicast: "
        + String(partitioned_multicast)
    )

    comptime M = prob_shape[0]
    comptime N = prob_shape[1]
    comptime K = prob_shape[2]

    comptime CLUSTER_M = cluster_shape[0]
    comptime CLUSTER_N = cluster_shape[1]

    var a = ManagedLayoutTensor[
        a_type,
        Layout.row_major(M, K),
    ](ctx)
    arange(a.tensor[update=False]())

    comptime b_layout = Layout.row_major(
        N, K
    ) if transpose_b else Layout.row_major(K, N)
    var b = ManagedLayoutTensor[b_type, b_layout](ctx)
    arange(b.tensor[update=False]())

    var c = ManagedLayoutTensor[
        c_type,
        Layout.row_major(M, N),
    ](ctx)

    var c_ref = ManagedLayoutTensor[
        c_type,
        Layout.row_major(M, N),
    ](ctx)

    # Shared memory tile layouts
    comptime a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    comptime b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()

    var a_tma_op = create_tma_tile[
        Index(BM // CLUSTER_N, BK) if partitioned_multicast else Index(BM, BK),
        swizzle_mode=a_swizzle,
    ](ctx, a.device_tensor())

    comptime b_tma_op_shape = Index(
        BN // CLUSTER_M, BK
    ) if partitioned_multicast else Index(BN, BK)
    var b_tma_op = create_tma_tile[
        b_tma_op_shape if transpose_b else Index(BK, BN),
        is_k_major=transpose_b,
        swizzle_mode=b_swizzle,
    ](ctx, b.device_tensor())

    comptime kernel = multicast_tma_wgmma_kernel[
        a_type,
        b_type,
        c_type,
        type_of(a_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).rank,
        type_of(b_tma_op).tile_shape,
        type_of(b_tma_op).desc_shape,
        Layout.row_major(M, N),
        block_tile_shape,
        wgmma_shape,
        a_smem_layout,
        b_smem_layout,
        cluster_shape=cluster_shape,
        transpose_b=transpose_b,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        partitioned_multicast=partitioned_multicast,
    ]

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c.device_tensor(),
        K // BK,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(128),
        cluster_dim=Dim(CLUSTER_N, CLUSTER_M, 1),
    )

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
            assert_almost_equal(
                c_host[m, n], c_host_ref[m, n], atol=1e-3, rtol=1e-4
            )

    # print(c.tensor())
    _ = a^
    _ = b^
    _ = c^
    _ = c_ref^


def main() raises:
    with DeviceContext() as ctx:
        # 2x1 cluster tests
        test_multicast_tma_wgmma[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(128, 16, 128),
            Index(64, 16, 64),
            Index(64, 8, 16),
            Index(2, 1, 1),
            transpose_b=True,
        ](ctx)

        test_multicast_tma_wgmma[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(128, 160, 128),
            Index(64, 160, 64),
            Index(64, 80, 16),
            Index(2, 1, 1),
            transpose_b=True,
        ](ctx)

        comptime for multicast_mode in range(2):
            test_multicast_tma_wgmma[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(128, 16, 128),
                Index(64, 16, 64),
                Index(64, 8, 16),
                Index(2, 1, 1),
                a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                transpose_b=True,
                partitioned_multicast=Bool(multicast_mode),
            ](ctx)

            test_multicast_tma_wgmma[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(128, 160, 128),
                Index(64, 160, 64),
                Index(64, 80, 16),
                Index(2, 1, 1),
                a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                transpose_b=True,
                partitioned_multicast=Bool(multicast_mode),
            ](ctx)

            test_multicast_tma_wgmma[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(128, 16, 64),
                Index(64, 16, 32),
                Index(64, 8, 16),
                Index(2, 1, 1),
                a_swizzle=TensorMapSwizzle.SWIZZLE_64B,
                b_swizzle=TensorMapSwizzle.SWIZZLE_64B,
                transpose_b=True,
                partitioned_multicast=Bool(multicast_mode),
            ](ctx)

            test_multicast_tma_wgmma[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(128, 160, 64),
                Index(64, 160, 32),
                Index(64, 80, 16),
                Index(2, 1, 1),
                a_swizzle=TensorMapSwizzle.SWIZZLE_64B,
                b_swizzle=TensorMapSwizzle.SWIZZLE_64B,
                transpose_b=True,
                partitioned_multicast=Bool(multicast_mode),
            ](ctx)

            test_multicast_tma_wgmma[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(128, 16, 32),
                Index(64, 16, 16),
                Index(64, 8, 16),
                Index(2, 1, 1),
                a_swizzle=TensorMapSwizzle.SWIZZLE_32B,
                b_swizzle=TensorMapSwizzle.SWIZZLE_32B,
                transpose_b=True,
                partitioned_multicast=Bool(multicast_mode),
            ](ctx)

            test_multicast_tma_wgmma[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(128, 160, 32),
                Index(64, 160, 16),
                Index(64, 80, 16),
                Index(2, 1, 1),
                a_swizzle=TensorMapSwizzle.SWIZZLE_32B,
                b_swizzle=TensorMapSwizzle.SWIZZLE_32B,
                transpose_b=True,
                partitioned_multicast=Bool(multicast_mode),
            ](ctx)

        # 2x2 cluster tests
        test_multicast_tma_wgmma[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(128, 16, 128),
            Index(64, 8, 64),
            Index(64, 8, 16),
            Index(2, 2, 1),
            transpose_b=True,
        ](ctx)

        test_multicast_tma_wgmma[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(128, 160, 128),
            Index(64, 80, 64),
            Index(64, 80, 16),
            Index(2, 2, 1),
            transpose_b=True,
        ](ctx)

        comptime for multicast_mode in range(2):
            test_multicast_tma_wgmma[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(128, 32, 128),
                Index(64, 16, 64),
                Index(64, 8, 16),
                Index(2, 2, 1),
                a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                transpose_b=True,
                partitioned_multicast=Bool(multicast_mode),
            ](ctx)

            test_multicast_tma_wgmma[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(128, 320, 128),
                Index(64, 160, 64),
                Index(64, 80, 16),
                Index(2, 2, 1),
                a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                transpose_b=True,
                partitioned_multicast=Bool(multicast_mode),
            ](ctx)

            test_multicast_tma_wgmma[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(128, 32, 64),
                Index(64, 16, 32),
                Index(64, 8, 16),
                Index(2, 2, 1),
                a_swizzle=TensorMapSwizzle.SWIZZLE_64B,
                b_swizzle=TensorMapSwizzle.SWIZZLE_64B,
                transpose_b=True,
                partitioned_multicast=Bool(multicast_mode),
            ](ctx)

            test_multicast_tma_wgmma[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(128, 320, 64),
                Index(64, 160, 32),
                Index(64, 80, 16),
                Index(2, 2, 1),
                a_swizzle=TensorMapSwizzle.SWIZZLE_64B,
                b_swizzle=TensorMapSwizzle.SWIZZLE_64B,
                transpose_b=True,
                partitioned_multicast=Bool(multicast_mode),
            ](ctx)

            test_multicast_tma_wgmma[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(128, 32, 32),
                Index(64, 16, 16),
                Index(64, 8, 16),
                Index(2, 2, 1),
                a_swizzle=TensorMapSwizzle.SWIZZLE_32B,
                b_swizzle=TensorMapSwizzle.SWIZZLE_32B,
                transpose_b=True,
                partitioned_multicast=Bool(multicast_mode),
            ](ctx)

            test_multicast_tma_wgmma[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(128, 320, 32),
                Index(64, 160, 16),
                Index(64, 80, 16),
                Index(2, 2, 1),
                a_swizzle=TensorMapSwizzle.SWIZZLE_32B,
                b_swizzle=TensorMapSwizzle.SWIZZLE_32B,
                transpose_b=True,
                partitioned_multicast=Bool(multicast_mode),
            ](ctx)

        # 1x2 cluster tests
        test_multicast_tma_wgmma[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(64, 16, 128),
            Index(64, 8, 64),
            Index(64, 8, 16),
            Index(1, 2, 1),
            transpose_b=True,
        ](ctx)

        test_multicast_tma_wgmma[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(64, 160, 128),
            Index(64, 80, 64),
            Index(64, 80, 16),
            Index(1, 2, 1),
            transpose_b=True,
        ](ctx)

        comptime for multicast_mode in range(2):
            test_multicast_tma_wgmma[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(64, 32, 128),
                Index(64, 16, 64),
                Index(64, 8, 16),
                Index(1, 2, 1),
                a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                transpose_b=True,
                partitioned_multicast=Bool(multicast_mode),
            ](ctx)

            test_multicast_tma_wgmma[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(64, 320, 128),
                Index(64, 160, 64),
                Index(64, 80, 16),
                Index(1, 2, 1),
                a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                transpose_b=True,
                partitioned_multicast=Bool(multicast_mode),
            ](ctx)

            test_multicast_tma_wgmma[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(64, 32, 64),
                Index(64, 16, 32),
                Index(64, 8, 16),
                Index(1, 2, 1),
                a_swizzle=TensorMapSwizzle.SWIZZLE_64B,
                b_swizzle=TensorMapSwizzle.SWIZZLE_64B,
                transpose_b=True,
                partitioned_multicast=Bool(multicast_mode),
            ](ctx)

            test_multicast_tma_wgmma[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(64, 320, 64),
                Index(64, 160, 32),
                Index(64, 80, 16),
                Index(1, 2, 1),
                a_swizzle=TensorMapSwizzle.SWIZZLE_64B,
                b_swizzle=TensorMapSwizzle.SWIZZLE_64B,
                transpose_b=True,
                partitioned_multicast=Bool(multicast_mode),
            ](ctx)

            test_multicast_tma_wgmma[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(64, 32, 32),
                Index(64, 16, 16),
                Index(64, 8, 16),
                Index(1, 2, 1),
                a_swizzle=TensorMapSwizzle.SWIZZLE_32B,
                b_swizzle=TensorMapSwizzle.SWIZZLE_32B,
                transpose_b=True,
                partitioned_multicast=Bool(multicast_mode),
            ](ctx)

            test_multicast_tma_wgmma[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(64, 320, 32),
                Index(64, 160, 16),
                Index(64, 80, 16),
                Index(1, 2, 1),
                a_swizzle=TensorMapSwizzle.SWIZZLE_32B,
                b_swizzle=TensorMapSwizzle.SWIZZLE_32B,
                transpose_b=True,
                partitioned_multicast=Bool(multicast_mode),
            ](ctx)
