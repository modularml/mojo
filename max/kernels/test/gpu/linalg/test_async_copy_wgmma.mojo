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

from std.sys import align_of

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu import block_idx, thread_idx
from max.gpu.memory import (
    async_copy_commit_group,
    async_copy_wait_group,
)
from layout import Layout, LayoutTensor, RuntimeLayout
from layout._fillers import arange
from layout._utils import ManagedLayoutTensor
from layout.layout_tensor import cp_async_k_major
from layout.tensor_core_async import (
    TensorCoreAsync,
    tile_layout_mn_major,
    warpgroup_fence,
    wgmma_c_layout,
)
from std.testing import assert_almost_equal

from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type


def cpasync_wgmma_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    block_tile_shape: IndexList[3],
    wgmma_shape: IndexList[3],
    transpose_b: Bool = False,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
](
    a: LayoutTensor[a_type, a_layout, MutAnyOrigin],
    b: LayoutTensor[b_type, b_layout, MutAnyOrigin],
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    num_iters_dev: Int32,
):
    """Test k_major @ mn_major with cp.async to simulate the 2nd matmul in mha.
    """
    # `Int` is not device-passable; widen the fixed-width arg.
    var num_iters = Int(num_iters_dev)
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]

    comptime a_smem_layout = Layout.row_major(BM, BK)
    var a_smem_tile = LayoutTensor[
        a_type,
        a_smem_layout,
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ].stack_allocation()

    comptime b_smem_layout = Layout.row_major(
        BN, BK
    ) if transpose_b else tile_layout_mn_major[b_type, BN, BK, b_swizzle]()
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

    comptime num_m_mmas = BM // wgmma_shape[0]
    comptime num_n_mmas = BN // wgmma_shape[1]

    var a_gmem_iter = a.tiled_iterator[BM, BK, axis=1](block_idx.y, 0)

    comptime b_dim0 = BN if transpose_b else BK
    comptime b_dim1 = BK if transpose_b else BN
    comptime b_tile_axis = 1 if transpose_b else 0
    var b_tile_coords = (block_idx.x, 0) if transpose_b else (0, block_idx.y)
    var b_gmem_iter = b.tiled_iterator[b_dim0, b_dim1, axis=b_tile_axis](
        b_tile_coords[0], b_tile_coords[1]
    )

    comptime c_frag_size = wgmma_shape[0] * wgmma_shape[1] // 128
    var c_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, c_frag_size),
        MutAnyOrigin,
        address_space=.LOCAL,
    ].stack_allocation()

    _ = c_reg_tile.fill(0.0)

    for i in range(num_iters):
        cp_async_k_major(a_smem_tile, a_gmem_iter[])
        cp_async_k_major(b_smem_tile, b_gmem_iter[])

        async_copy_commit_group()
        async_copy_wait_group(0)

        barrier()

        warpgroup_fence(c_reg_tile)
        wgmma_op.arrive()
        wgmma_op.wgmma(a_smem_tile, b_smem_tile, c_reg_tile)
        wgmma_op.commit_group()
        warpgroup_fence(c_reg_tile)
        wgmma_op.wait_group()

        barrier()

        a_gmem_iter._incr()
        b_gmem_iter._incr()

    var c_gmem_tile = c.tile[BM, BN](block_idx.y, block_idx.x)
    comptime c_layouts = wgmma_c_layout[
        wgmma_shape[0], wgmma_shape[1], c_gmem_tile.layout
    ]()
    comptime tv_tile_to_idx_const = c_layouts[2]
    comptime tv_to_idx = tv_tile_to_idx_const[0]
    comptime tile_to_idx = tv_tile_to_idx_const[1]
    comptime t_to_idx_const = tv_to_idx[0]
    comptime v_to_idx = tv_to_idx[1]
    var t_to_idx = RuntimeLayout[t_to_idx_const]()
    var t_idx = t_to_idx(thread_idx.x)

    var c_reg_tile_vec2 = c_reg_tile.vectorize[1, 2]()
    comptime T = c_reg_tile_vec2.element_type
    var c_gmem_ptr = c_gmem_tile.ptr + t_idx

    comptime for mma_id in range(tile_to_idx.size()):
        comptime mma_idx = tile_to_idx(mma_id)

        comptime for local_idx_v2 in range(c_reg_tile_vec2.layout[1].size()):
            comptime local_idx = local_idx_v2 * 2
            comptime v_idx = v_to_idx(local_idx)
            comptime c_idx = v_idx + mma_idx
            var casted_vec = c_reg_tile_vec2[mma_id, local_idx_v2].cast[
                c_type
            ]()
            (c_gmem_ptr + c_idx).store[alignment=align_of[T]()](casted_vec)


def test_cpasync_wgmma[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    prob_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    wgmma_shape: IndexList[3],
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
](ctx: DeviceContext) raises:
    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]

    comptime WGMMA_M = wgmma_shape[0]
    comptime WGMMA_N = wgmma_shape[1]
    comptime WGMMA_K = wgmma_shape[2]

    print(
        "wgmma_ss_bf16_bf16_f32 block tile "
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

    comptime kernel = cpasync_wgmma_kernel[
        a_type,
        b_type,
        c_type,
        type_of(a).layout,
        type_of(b).layout,
        Layout.row_major(M, N),
        block_tile_shape,
        wgmma_shape,
        transpose_b=transpose_b,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
    ]

    ctx.enqueue_function[kernel](
        a.device_tensor(),
        b.device_tensor(),
        c.device_tensor(),
        Int32(K // BK),
        grid_dim=(1, 1),
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


def main() raises:
    with DeviceContext() as ctx:
        test_cpasync_wgmma[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(64, 64, 64),
            Index(64, 64, 64),
            Index(64, 64, 16),
            a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=False,
        ](ctx)

        test_cpasync_wgmma[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(64, 128, 128),
            Index(64, 128, 128),
            Index(64, 128, 16),
            a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=False,
        ](ctx)

        test_cpasync_wgmma[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(64, 64, 64),
            Index(64, 64, 64),
            Index(64, 64, 16),
            a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](ctx)

        test_cpasync_wgmma[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(64, 128, 128),
            Index(64, 128, 128),
            Index(64, 128, 16),
            a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](ctx)

        test_cpasync_wgmma[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(128, 64, 128),
            Index(128, 64, 128),
            Index(64, 64, 16),
            a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](ctx)
