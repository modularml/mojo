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

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu import warp_id, lane_id
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu import thread_idx
from max.gpu.compute.mma import (
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_fence_aligned,
    wgmma_wait_group_sync,
)
from layout import Layout, LayoutTensor, TileTensor, row_major
from layout._fillers import arange
from layout._utils import ManagedLayoutTensor
from layout.tensor_core_async import (
    _lhs_descriptor,
    _rhs_descriptor,
    tile_layout_k_major,
)
from std.testing import assert_almost_equal

from std.utils import StaticTuple


def wgmma_kernel_ss[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    WMMA_M: Int,
    WMMA_N: Int,
    WMMA_K: Int,
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    transpose_b: Bool = False,
](
    a_gmem: LayoutTensor[a_type, a_layout, MutAnyOrigin],
    b_gmem: LayoutTensor[b_type, b_layout, MutAnyOrigin],
    c_gmem: LayoutTensor[c_type, c_layout, MutAnyOrigin],
):
    var a_smem_tile = LayoutTensor[
        .bfloat16,
        a_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
    ].stack_allocation()

    var b_smem_tile = LayoutTensor[
        .bfloat16,
        b_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
    ].stack_allocation()

    comptime num_output_regs = WMMA_M * WMMA_N // 128
    var c_reg = StaticTuple[Float32, num_output_regs](0)

    comptime M = a_layout.shape[0].value()
    comptime K = a_layout.shape[1].value()
    comptime N = c_layout.shape[1].value()

    comptime b_tile_dim0 = N if transpose_b else WMMA_K
    comptime b_tile_dim1 = WMMA_K if transpose_b else N

    for k_i in range(K // WMMA_K):
        var a_gmem_tile = a_gmem.tile[M, WMMA_K](0, k_i)

        var b_tile_coord0 = 0 if transpose_b else k_i
        var b_tile_coord1 = k_i if transpose_b else 0
        var b_gmem_tile = b_gmem.tile[b_tile_dim0, b_tile_dim1](
            b_tile_coord0, b_tile_coord1
        )

        if thread_idx.x == 0:
            a_smem_tile.copy_from(a_gmem_tile)
            b_smem_tile.copy_from(b_gmem_tile)

        barrier()

        var mat_a_desc = _lhs_descriptor(a_smem_tile)
        var mat_b_desc = _rhs_descriptor[transpose_b](b_smem_tile)

        wgmma_fence_aligned()

        c_reg = wgmma_async[
            WMMA_M,
            WMMA_N,
            WMMA_K,
            a_type=DType.bfloat16,
            b_type=DType.bfloat16,
        ](mat_a_desc, mat_b_desc, c_reg)
        wgmma_commit_group_sync()
        wgmma_wait_group_sync()

    var th_local_res = (
        c_gmem.tile[16, WMMA_N](warp_id(), 0)
        .vectorize[1, 2]()
        .distribute[Layout.row_major(8, 4)](lane_id())
    )

    for i in range(num_output_regs):
        th_local_res[(i // 2) % 2, i // 4][i % 2] = c_reg[i].cast[
            c_gmem.dtype
        ]()


def wgmma_bf16_bf16_f32[
    M: Int, N: Int, K: Int, transpose_b: Bool = False, a_reg: Bool = False
](ctx: DeviceContext) raises:
    print(
        "== wgmma_bf16_bf16_f32_64xNx16(N, r/s) => ",
        N,
        ", r" if a_reg else ", s",
        sep="",
    )

    var a = ManagedLayoutTensor[.bfloat16, Layout.row_major(M, K)](ctx)
    arange(a.tensor[update=False]())

    var b = ManagedLayoutTensor[.bfloat16, Layout.row_major(N, K)](ctx)
    arange(b.tensor[update=False]())

    var c = ManagedLayoutTensor[.bfloat16, Layout.row_major(M, N)](ctx)
    var c_ref = ManagedLayoutTensor[.bfloat16, Layout.row_major(M, N)](ctx)

    comptime a_smem_layout = tile_layout_k_major[.bfloat16, BM=M, BK=16]()

    comptime b_smem_layout = tile_layout_k_major[.bfloat16, BM=N, BK=16]()

    comptime kernel = wgmma_kernel_ss[
        DType.bfloat16,
        DType.bfloat16,
        DType.bfloat16,
        Layout.row_major(M, K),
        Layout.row_major(N, K),
        Layout.row_major(M, N),
        M,
        N,
        K,
        a_smem_layout,
        b_smem_layout,
        transpose_b=transpose_b,
    ]

    ctx.enqueue_function[kernel](
        a.device_tensor(),
        b.device_tensor(),
        c.device_tensor(),
        grid_dim=(1, 1),
        block_dim=(128),
    )
    ctx.synchronize()

    var a_buf = TileTensor(a.device_tensor().ptr, row_major[M, K]())
    var b_buf = TileTensor(b.device_tensor().ptr, row_major[N, K]())
    var c_ref_buf = TileTensor(c_ref.device_tensor().ptr, row_major[M, N]())

    vendor_blas.matmul(
        ctx,
        c_ref_buf,
        a_buf,
        b_buf,
        c_row_major=True,
        transpose_b=transpose_b,
    )

    for m in range(M):
        for n in range(N):
            assert_almost_equal(
                c_ref.tensor()[m, n], c.tensor()[m, n], atol=1e-3, rtol=1e-3
            )

    _ = a^
    _ = b^
    _ = c^
    _ = c_ref^


def main() raises:
    with DeviceContext() as ctx:
        comptime for n in range(8, 264, 8):
            wgmma_bf16_bf16_f32[64, n, 16, True](ctx)
