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


def test_blackwell_matmul_with_epilogue_tensor[
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
    cta_group: Int = 1,
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    block_swizzle_size: Int = 0,
    k_group_size: Int = 1,
    swapAB: Bool = False,
    num_accum_pipeline_stages: Int = 2,
    num_clc_pipeline_stages: Int = 2,
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    var M = Int(m.value())
    var N = Int(n.value())
    var K = Int(k.value())

    print(
        t"in/out dtypes=({a_type}, {b_type}, {c_type})  problem shape=({M},"
        t" {N}, {K})"
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
    var epilogue_shape = row_major(Coord(m, Idx[NType.static_value]))

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
    var epilogue_host_ptr = alloc[Scalar[c_type]](c_size)
    var epilogue_host = TileTensor(epilogue_host_ptr, epilogue_shape)

    # Device allocations
    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_tensor = TileTensor(a_device, a_shape)
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_tensor = TileTensor(b_device, b_shape)
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_tensor = TileTensor(c_device, c_shape)
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ref_tensor = TileTensor(c_device_ref, c_shape)
    var epilogue_device = ctx.enqueue_create_buffer[c_type](c_size)
    var epilogue_tile = TileTensor(epilogue_device, epilogue_shape)

    # Initialize
    rand(a_host._storage, a_host.num_elements())
    rand(b_host._storage, b_host.num_elements())
    rand(epilogue_host._storage, epilogue_host.num_elements(), min=-10, max=10)

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(epilogue_device, epilogue_host_ptr)

    comptime matmul_config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        cluster_shape=Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        mma_shape=mma_shape,
        block_swizzle_size=block_swizzle_size,
        cta_group=1,
        use_tma_epilogue_load=True,
        AB_swapped=swapAB,
        num_accum_pipeline_stages=num_accum_pipeline_stages,
        num_clc_pipeline_stages=num_clc_pipeline_stages,
    )

    comptime EpilogueType = TileTensor[
        matmul_config.c_type,
        type_of(epilogue_shape),
        ImmutAnyOrigin,
        Engine=epilogue_tile.Engine,
    ]
    blackwell_matmul_tma_umma_warp_specialized[
        transpose_b=transpose_b,
        config=matmul_config,
    ](
        c_tensor,
        a_tensor,
        b_tensor,
        ctx,
        epilogue_tensor=rebind[EpilogueType](epilogue_tile),
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

    # Print all elements of epilogue tensor, kernel output, and cuBLAS ref
    # print("epilogue_host  :", end=" ")
    # for i in range(M):
    #     for j in range(N):
    #         var idx = epilogue_host.layout(Coord(i, j))
    #         print(epilogue_host.ptr[idx], end=" ")
    #     print()
    # print("c_host(ker):", end=" ")
    # for i in range(M):
    #     for j in range(N):
    #         var idx = c_host.layout(Coord(i, j))
    #         print(c_host.ptr[idx], end=" ")
    #     print()
    # print("c_ref(blas):", end=" ")
    # for i in range(M):
    #     for j in range(N):
    #         var idx = c_host_ref.layout(Coord(i, j))
    #         print(c_host_ref.ptr[idx], end=" ")
    #     print()

    # print("diff(ker-ref):", end=" ")
    # for i in range(M):
    #     for j in range(N):
    #         var idx = c_host.layout(Coord(i, j))
    #         print(
    #             c_host.ptr[idx].cast[.float32]()
    #             - c_host_ref.ptr[idx].cast[.float32](),
    #         end=" ",
    #     )
    #     print()

    # Add epilogue tensor to reference on host: C_ref[m, n] += epilogue[m, n]
    for i in range(M):
        for j in range(N):
            c_host_ref[Coord(i, j)] += epilogue_host[Coord(i, j)]

    # print("c_ref(blas+epilogue):", end=" ")
    # for i in range(M):
    #     for j in range(N):
    #         var idx = c_host_ref.layout(Coord(i, j))
    #         print(c_host_ref.ptr[idx], end=" ")
    #     print()

    # print("diff(ker-ref+epilogue):", end=" ")
    # for i in range(M):
    #     for j in range(N):
    #         var idx = c_host.layout(Coord(i, j))
    #         print(
    #             c_host.ptr[idx].cast[.float32]()
    #             - c_host_ref.ptr[idx].cast[.float32](),
    #         end=" ",
    #     )
    #     print()

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
    epilogue_host_ptr.free()
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^
    _ = epilogue_device^


def main() raises:
    with DeviceContext() as ctx:
        comptime dtype = DType.bfloat16
        comptime cta_group = 1
        comptime for swizzle in [TensorMapSwizzle.SWIZZLE_128B]:
            comptime BK = (swizzle.bytes() // size_of[dtype]())
            comptime MMA_K = 16

            # we support all range of mma_n_scales in range(1, 33) but the test will time out so we only test a subset
            comptime for mma_m in [64, 128]:
                comptime for mma_n in [
                    8,
                    16,
                    32,
                    48,
                    64,
                    80,
                    88,
                    96,
                    112,
                    128,
                    144,
                    152,
                    184,
                    192,
                    256,
                ]:
                    comptime block_tile_shape = Index(mma_m, mma_n, BK)
                    comptime umma_shape = Index(mma_m, mma_n, MMA_K)

                    test_blackwell_matmul_with_epilogue_tensor[
                        dtype,
                        dtype,
                        .bfloat16,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
                        cta_group=cta_group,
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        block_swizzle_size=8,
                        swapAB=True,
                        num_accum_pipeline_stages=1,
                        num_clc_pipeline_stages=0,
                    ](
                        ctx,
                        Int(1000),
                        Idx[1024],
                        Idx[1024 + 16],
                    )

                    test_blackwell_matmul_with_epilogue_tensor[
                        dtype,
                        dtype,
                        .bfloat16,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                        cta_group=cta_group,
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        block_swizzle_size=4,
                    ](
                        ctx,
                        Int(1),
                        Idx[4096],
                        Idx[1024 + 16],
                    )

                    comptime for swapAB in [False, True]:
                        test_blackwell_matmul_with_epilogue_tensor[
                            dtype,
                            dtype,
                            .bfloat16,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape=StaticTuple[Int32, 3](4, 2, 1),
                            cta_group=cta_group,
                            a_swizzle=swizzle,
                            b_swizzle=swizzle,
                            block_swizzle_size=0,
                            k_group_size=2,
                            swapAB=swapAB,
                            num_accum_pipeline_stages=1,
                            num_clc_pipeline_stages=0,
                        ](
                            ctx,
                            Int(500),
                            Idx[2048],
                            Idx[4096],
                        )

                        test_blackwell_matmul_with_epilogue_tensor[
                            dtype,
                            dtype,
                            .bfloat16,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape=StaticTuple[Int32, 3](8, 2, 1),
                            cta_group=cta_group,
                            a_swizzle=swizzle,
                            b_swizzle=swizzle,
                            block_swizzle_size=2,
                            k_group_size=1,
                            swapAB=swapAB,
                        ](
                            ctx,
                            Int(999),
                            Idx[256],
                            Idx[128],
                        )

                    test_blackwell_matmul_with_epilogue_tensor[
                        dtype,
                        dtype,
                        .bfloat16,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                        cta_group=cta_group,
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        block_swizzle_size=1,
                        k_group_size=2,
                        swapAB=False,
                    ](
                        ctx,
                        Int(777),
                        Idx[2560],
                        Idx[8192],
                    )
