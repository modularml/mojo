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
from max.gpu import warp_id, lane_id
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu import block_idx, thread_idx
from layout import Layout, LayoutTensor
from layout._fillers import arange
from layout._utils import ManagedLayoutTensor
from layout.int_tuple import product
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
    create_tensor_tile,
)
from std.memory import unsafe_stack_allocation
from std.testing import assert_almost_equal

from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type


def _compute_reg_tile_layout(layout: Layout, frag_size: Int) -> Layout:
    var local_size = layout.size() // 128
    return Layout.row_major(local_size // frag_size, frag_size)


@always_inline
def _load_a_reg_tile[
    dtype: DType,
    layout: Layout,
    //,
    wgmma_shape: IndexList[3],
](
    out ret: LayoutTensor[
        dtype,
        _compute_reg_tile_layout(layout, 16 // size_of[dtype]()),
        MutAnyOrigin,
        address_space=.LOCAL,
    ],
    smem_tile: LayoutTensor[
        dtype,
        layout,
        MutAnyOrigin,
        address_space=.SHARED,
        ...,
    ],
):
    comptime assert ret.layout[0].shape[0].value() > 0
    ret = type_of(ret).stack_allocation()

    comptime WGMMA_M = wgmma_shape[0]
    comptime WGMMA_K = wgmma_shape[2]

    comptime rows = product(layout[0].shape)
    comptime cols = product(layout[1].shape)

    comptime num_wgmma_m = ceildiv(rows, WGMMA_M)
    comptime num_wgmma_k = ceildiv(cols, WGMMA_K)
    comptime assert num_wgmma_m * num_wgmma_k == ret.layout[0].shape[0].value()

    comptime simd_size = 4 // size_of[dtype]()
    var vret = ret.vectorize[1, simd_size]()

    comptime for m_mma in range(num_wgmma_m):
        comptime for k_mma in range(num_wgmma_k):
            comptime r_id = m_mma + k_mma * num_wgmma_m
            var smem_wg = (
                smem_tile.tile[WGMMA_M, WGMMA_K](m_mma, k_mma)
                .tile[WGMMA_M // 4, WGMMA_K](warp_id(), 0)
                .vectorize[1, simd_size]()
                .distribute[Layout.row_major(8, 4)](lane_id())
            )
            vret.tile[1, 4](r_id, 0).copy_from(smem_wg)


@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
def tma_wgmma_kernel[
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
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    a_smem: Bool = True,
](
    a_tma_op: TMATensorTile[a_type, a_tile_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[b_type, b_tile_rank, b_tile_shape, b_desc_shape],
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    num_iters_dev: Int32,
):
    var num_iters = Int(num_iters_dev)
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]
    comptime num_m_mmas = BM // wgmma_shape[0]
    comptime num_n_mmas = BN // wgmma_shape[1]

    comptime a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    comptime b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()

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

    var mbar = unsafe_stack_allocation[
        1,
        SharedMemBarrier,
        address_space=.SHARED,
        alignment=8,
    ]()
    if thread_idx.x == 0:
        mbar[0].init()

    var phase: UInt32 = 0

    for i in range(num_iters):
        if thread_idx.x == 0:
            mbar[0].expect_bytes(Int32(expected_bytes))
            a_tma_op.async_copy(
                a_smem_tile,
                mbar[0],
                (i * BK, block_idx.y * BM),
            )
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

        comptime if a_smem:
            wgmma_op.wgmma(a_smem_tile, b_smem_tile, c_reg_tile)
        else:
            var a_reg_tile = _load_a_reg_tile[wgmma_shape](a_smem_tile)
            wgmma_op.wgmma(a_reg_tile, b_smem_tile, c_reg_tile)
        wgmma_op.commit_group()
        warpgroup_fence(c_reg_tile)
        wgmma_op.wait_group()

        barrier()

    var c_gmem_tile = c.tile[BM, BN](block_idx.y, block_idx.x)

    comptime for m_mma in range(num_m_mmas):
        comptime for n_mma in range(num_n_mmas):
            comptime mma_id = n_mma * num_m_mmas + m_mma

            # (m_mma, n_mma) is coordinates for a warp group's tile.
            # A warp group is 4x1 warps.
            var warp_tile = c_gmem_tile.tile[
                wgmma_shape[0] // 4, wgmma_shape[1]
            ](m_mma * 4 + warp_id(), n_mma)

            # Tile at (mma_id, 0) is a long vector containing all fragments
            # for this warp.
            var c_frag = c_reg_tile.tile[1, c_frag_size](mma_id, 0)

            # A warp is organized as row_major(8, 4) and each thread owns 2 contiguous
            # elementwise. This pattern repeats to fill the warp tile.
            copy_local_to_dram[Layout.row_major(8, 4)](
                warp_tile.vectorize[1, 2](), c_frag.vectorize[1, 2]()
            )


def test_tma_wgmma[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    prob_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    wgmma_shape: IndexList[3],
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    a_smem: Bool = True,
](ctx: DeviceContext) raises:
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]

    comptime WGMMA_M = wgmma_shape[0]
    comptime WGMMA_N = wgmma_shape[1]
    comptime WGMMA_K = wgmma_shape[2]

    print(
        "wgmma_"
        + ("s" if a_smem else "r")
        + "s_bf16_bf16_f32 block tile "
        + String(block_tile_shape)
        + " transb="
        + String(transpose_b)
        + "; inst shape "
        + String(wgmma_shape)
        + " A "
        + String(a_swizzle)
        + " B "
        + String(b_swizzle)
    )

    comptime M = prob_shape[0]
    comptime N = prob_shape[1]
    comptime K = prob_shape[2]

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

    var a_tma_op = create_tensor_tile[Index(BM, BK), swizzle_mode=a_swizzle](
        ctx, a.device_tensor()
    )
    var b_tma_op = create_tensor_tile[
        Index(BN, BK) if transpose_b else Index(BK, BN),
        swizzle_mode=b_swizzle,
    ](ctx, b.device_tensor())

    comptime kernel = tma_wgmma_kernel[
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
        transpose_b=transpose_b,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        a_smem=a_smem,
    ]

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c.device_tensor(),
        Int32(K // BK),
        grid_dim=(N // BN, M // BM),
        block_dim=(128),
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
        test_tma_wgmma[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(128, 16, 32),
            Index(128, 16, 32),
            Index(64, 8, 16),
        ](ctx)

        test_tma_wgmma[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(64, 8, 64),
            Index(64, 8, 64),
            Index(64, 8, 16),
        ](ctx)

        test_tma_wgmma[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(64, 8, 64),
            Index(64, 8, 64),
            Index(64, 8, 16),
            a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
        ](ctx)

        test_tma_wgmma[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(128, 16, 32),
            Index(128, 16, 32),
            Index(64, 8, 16),
            a_swizzle=TensorMapSwizzle.SWIZZLE_64B,
            b_swizzle=TensorMapSwizzle.SWIZZLE_64B,
        ](ctx)

        test_tma_wgmma[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(128, 16, 16),
            Index(128, 16, 16),
            Index(64, 8, 16),
            a_swizzle=TensorMapSwizzle.SWIZZLE_32B,
            b_swizzle=TensorMapSwizzle.SWIZZLE_32B,
        ](ctx)

        comptime for log2BN in range(6, 8):
            comptime BN = 1 << log2BN
            test_tma_wgmma[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(128, 256, 64),
                Index(64, BN, 64),
                Index(64, 64, 16),
                a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                transpose_b=True,
            ](ctx)

            test_tma_wgmma[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(128, 256, 64),
                Index(64, BN, 64),
                Index(64, 64, 16),
                a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                transpose_b=False,
            ](ctx)

            test_tma_wgmma[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(128, 256, 16),
                Index(64, BN, 16),
                Index(64, 64, 16),
                a_swizzle=TensorMapSwizzle.SWIZZLE_NONE,
                b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                transpose_b=False,
            ](ctx)

            test_tma_wgmma[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(128, 256, 16),
                Index(64, BN, 16),
                Index(64, 64, 16),
                a_swizzle=TensorMapSwizzle.SWIZZLE_NONE,
                b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                a_smem=False,
                transpose_b=False,
            ](ctx)

        test_tma_wgmma[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(64, 8, 64),
            Index(64, 8, 64),
            Index(64, 8, 16),
            a_swizzle=TensorMapSwizzle.SWIZZLE_NONE,
            b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            a_smem=False,
            transpose_b=True,
        ](ctx)
