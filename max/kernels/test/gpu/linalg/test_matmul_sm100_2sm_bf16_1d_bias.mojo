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

from std.sys import size_of

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.memory import alloc
from internal_utils import assert_almost_equal
from std.random import rand
from layout import TileTensor, Coord, CoordLike, row_major, Idx
from linalg.matmul.gpu.sm100_structured.default.matmul import (
    blackwell_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    MatmulConfig,
)

from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple


def test_blackwell_matmul_with_1d_bias[
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
    block_swizzle_size: Int = 0,
    cta_group: Int = 2,
    k_group_size: Int = 1,
    num_accum_pipeline_stages: Int = 2,
    num_clc_pipeline_stages: Int = 2,
    swapAB: Bool = False,
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    var M = Int(m.value())
    var N = Int(n.value())
    var K = Int(k.value())

    print(
        t"[1D bias] in/out dtypes=({a_type}, {b_type}, {c_type})  problem"
        t" shape=({M}, {N}, {K})"
        t" mma_shape={mma_shape} block_tile_shape={block_tile_shape}"
    )

    var a_shape = row_major(Coord(m, Idx[KType.static_value]))
    var b_shape = row_major(
        Coord(
            Idx[NType.static_value if transpose_b else KType.static_value],
            Idx[KType.static_value if transpose_b else NType.static_value],
        )
    )
    var c_shape = row_major(Coord(m, Idx[NType.static_value]))
    # 1D bias: shape [N] — broadcast across all M rows.
    var bias_shape = row_major(Coord(Idx[NType.static_value]))

    var a_size = M * K
    var b_size = N * K if transpose_b else K * N
    var c_size = M * N

    # Host allocations
    var a_host_ptr = alloc[Scalar[a_type]](a_size)
    var a_host = TileTensor(a_host_ptr, a_shape)
    var b_host_ptr = alloc[Scalar[b_type]](b_size)
    var b_host = TileTensor(b_host_ptr, b_shape)
    var c_host_ptr = alloc[Scalar[c_type]](c_size)
    var c_host = TileTensor(c_host_ptr, c_shape)
    var c_host_ref_ptr = alloc[Scalar[c_type]](c_size)
    var c_host_ref = TileTensor(c_host_ref_ptr, c_shape)
    var bias_host_ptr = alloc[Scalar[c_type]](N)
    var bias_host = TileTensor(bias_host_ptr, bias_shape)

    # Device allocations
    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_tensor = TileTensor(a_device, a_shape)
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_tensor = TileTensor(b_device, b_shape)
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_tensor = TileTensor(c_device, c_shape)
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ref_tensor = TileTensor(c_device_ref, c_shape)
    var bias_device = ctx.enqueue_create_buffer[c_type](N)
    var bias_tile = TileTensor(bias_device, bias_shape)

    # Initialize
    rand(a_host._storage, a_host.num_elements())
    rand(b_host._storage, b_host.num_elements())
    rand(bias_host._storage, bias_host.num_elements(), min=-10, max=10)

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(bias_device, bias_host_ptr)

    comptime matmul_config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        cluster_shape=Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        mma_shape=mma_shape,
        block_swizzle_size=block_swizzle_size,
        cta_group=cta_group,
        use_tma_epilogue_load=True,
        AB_swapped=swapAB,
        epilogue_is_1d=True,
        num_accum_pipeline_stages=num_accum_pipeline_stages,
        num_clc_pipeline_stages=num_clc_pipeline_stages,
    )

    comptime EpilogueType = TileTensor[
        matmul_config.c_type,
        type_of(bias_shape),
        ImmutAnyOrigin,
        Engine=bias_tile.Engine,
    ]
    blackwell_matmul_tma_umma_warp_specialized[
        transpose_b=transpose_b,
        config=matmul_config,
    ](
        c_tensor,
        a_tensor,
        b_tensor,
        ctx,
        epilogue_tensor=rebind[EpilogueType](bias_tile),
    )

    # Reference: cuBLAS matmul
    var a_lt = a_tensor.to_layout_tensor()
    var b_lt = b_tensor.to_layout_tensor()
    var c_ref_tensor_lt = c_ref_tensor.to_layout_tensor()

    vendor_blas.matmul(
        ctx,
        c_ref_tensor_lt,
        a_lt,
        b_lt,
        c_row_major=True,
        transpose_b=transpose_b,
    )

    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    # Add 1D bias to reference: C_ref[i, j] += bias[j] for all i (broadcast).
    for i in range(M):
        for j in range(N):
            var bias_val = bias_host_ptr[j].cast[.float32]()
            var idx = c_host_ref.layout(Coord(i, j))
            c_host_ref_ptr[idx] = (
                c_host_ref_ptr[idx].cast[.float32]() + bias_val
            ).cast[c_type]()

    comptime rtol = 1e-2
    assert_almost_equal(
        c_host._storage,
        c_host_ref._storage,
        c_host.num_elements(),
        atol=0.0001,
        rtol=rtol,
    )
    print("\n=== TEST PASSED ===\n")

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_host_ref_ptr.free()
    bias_host_ptr.free()
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^
    _ = bias_device^


def main() raises:
    with DeviceContext() as ctx:
        comptime dtype = DType.bfloat16

        comptime for swizzle in [TensorMapSwizzle.SWIZZLE_128B]:
            comptime BK = (swizzle.bytes() // size_of[dtype]())
            comptime MMA_K = 16

            comptime for bm in [64, 128]:
                comptime for bn in [
                    8,
                    16,
                    32,
                    40,
                    64,
                    72,
                    80,
                    88,
                    104,
                    112,
                    128,
                ]:
                    comptime block_tile_shape = Index(bm, bn, BK)
                    comptime umma_shape = Index(2 * bm, 2 * bn, MMA_K)

                    test_blackwell_matmul_with_1d_bias[
                        dtype,
                        dtype,
                        .bfloat16,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        block_swizzle_size=4,
                    ](
                        ctx,
                        Int(233),
                        Idx[4096],
                        Idx[1024 + 16],
                    )

                    test_blackwell_matmul_with_1d_bias[
                        dtype,
                        dtype,
                        .bfloat16,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        block_swizzle_size=0,
                        k_group_size=2,
                        num_accum_pipeline_stages=1,
                        num_clc_pipeline_stages=0,
                        swapAB=True,
                    ](
                        ctx,
                        Int(500),
                        Idx[2048],
                        Idx[4096],
                    )

                    test_blackwell_matmul_with_1d_bias[
                        dtype,
                        dtype,
                        .bfloat16,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](8, 2, 1),
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        block_swizzle_size=2,
                    ](
                        ctx,
                        Int(999),
                        Idx[256],
                        Idx[128],
                    )

                    test_blackwell_matmul_with_1d_bias[
                        dtype,
                        dtype,
                        .bfloat16,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        block_swizzle_size=1,
                    ](
                        ctx,
                        Int(777),
                        Idx[2560],
                        Idx[8192],
                    )

            # AB_swapped + 1D bias (2SM)
            comptime for bm in [64, 128]:
                comptime for bn in [16, 32, 64, 80, 112, 128]:
                    comptime block_tile_shape = Index(bm, bn, BK)
                    comptime umma_shape = Index(2 * bm, 2 * bn, MMA_K)

                    test_blackwell_matmul_with_1d_bias[
                        dtype,
                        dtype,
                        .bfloat16,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        block_swizzle_size=4,
                        swapAB=True,
                    ](
                        ctx,
                        Int(1),
                        Idx[4096],
                        Idx[1024 + 16],
                    )

                    test_blackwell_matmul_with_1d_bias[
                        dtype,
                        dtype,
                        .bfloat16,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        block_swizzle_size=0,
                        k_group_size=2,
                        num_accum_pipeline_stages=1,
                        num_clc_pipeline_stages=0,
                        swapAB=True,
                    ](
                        ctx,
                        Int(500),
                        Idx[2048],
                        Idx[4096],
                    )

            # Edge cases: small/odd M with swapAB (transpose bias path, 2SM)
            comptime for bm in [64, 128]:
                comptime for bn in [16, 64, 128]:
                    comptime block_tile_shape = Index(bm, bn, BK)
                    comptime umma_shape = Index(2 * bm, 2 * bn, MMA_K)

                    # M=3: odd, smaller than warp size
                    test_blackwell_matmul_with_1d_bias[
                        dtype,
                        dtype,
                        .bfloat16,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        block_swizzle_size=4,
                        swapAB=True,
                    ](
                        ctx,
                        Int(3),
                        Idx[256],
                        Idx[128],
                    )

                    # M=17: prime, not power-of-2
                    test_blackwell_matmul_with_1d_bias[
                        dtype,
                        dtype,
                        .bfloat16,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](4, 2, 1),
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        block_swizzle_size=0,
                        swapAB=True,
                    ](
                        ctx,
                        Int(17),
                        Idx[512],
                        Idx[128],
                    )

            # Non-swapAB with M=1 (single-row, non-transpose bias path, 2SM)
            comptime for bn in [16, 64, 128]:
                comptime block_tile_shape = Index(128, bn, BK)
                comptime umma_shape = Index(256, 2 * bn, MMA_K)

                test_blackwell_matmul_with_1d_bias[
                    dtype,
                    dtype,
                    .bfloat16,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=4,
                ](
                    ctx,
                    Int(1),
                    Idx[256],
                    Idx[128],
                )
