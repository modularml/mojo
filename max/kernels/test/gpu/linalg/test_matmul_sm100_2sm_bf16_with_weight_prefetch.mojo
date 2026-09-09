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
#
# Test: bf16 matmul (sm100, 2SM) with weight prefetch.
#
# Validates weight prefetching (prefetch_tiles_n > 0) for both swapAB=False
# and swapAB=True across all supported 2SM tile shapes, using PDL overlap.
# ===----------------------------------------------------------------------=== #

from std.sys import size_of

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.primitives.grid_controls import PDLLevel
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


def test_blackwell_matmul_with_weight_prefetch[
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
    swapAB: Bool = False,
    k_group_size: Int = 1,
    pdl_level: PDLLevel = PDLLevel.OVERLAP_AT_END,
    prefetch_tiles_n: Int = 2,
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    var M = Int(m.value())
    var N = Int(n.value())
    var K = Int(k.value())

    print(
        t"weight prefetch 2SM: dtype={a_type} shape=({M}, {N}, {K})"
        t" mma_shape={mma_shape} block_tile_shape={block_tile_shape}"
        t" swapAB={swapAB} prefetch_tiles_n={prefetch_tiles_n}"
    )

    var a_shape = row_major(Coord(m, Idx[KType.static_value]))
    var b_shape = row_major(
        Coord(
            Idx[NType.static_value if transpose_b else KType.static_value],
            Idx[KType.static_value if transpose_b else NType.static_value],
        )
    )
    var c_shape = row_major(Coord(m, Idx[NType.static_value]))

    var a_size = Int(m.value()) * Int(k.value())
    var b_size = (
        Int(n.value())
        * Int(k.value()) if transpose_b else Int(k.value())
        * Int(n.value())
    )
    var c_size = Int(m.value()) * Int(n.value())

    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_size)
    var a_host = TileTensor(a_host_ptr, a_shape)
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](b_size)
    var b_host = TileTensor(b_host_ptr, b_shape)
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_host = TileTensor(c_host_ptr, c_shape)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_host_ref = TileTensor(c_host_ref_ptr, c_shape)

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_tensor = TileTensor(a_device, a_shape)
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_tensor = TileTensor(b_device, b_shape)
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_tensor = TileTensor(c_device, c_shape)
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ref_tensor = TileTensor(c_device_ref, c_shape)

    rand(a_host._storage, a_host.num_elements())
    rand(b_host._storage, b_host.num_elements())

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    comptime matmul_config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        cluster_shape=Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        mma_shape=mma_shape,
        block_swizzle_size=block_swizzle_size,
        cta_group=2,
        AB_swapped=swapAB,
        k_group_size=k_group_size,
        prefetch_tiles_n=prefetch_tiles_n,
    )

    blackwell_matmul_tma_umma_warp_specialized[
        transpose_b=transpose_b,
        config=matmul_config,
        pdl_level=pdl_level,
    ](
        c_tensor,
        a_tensor,
        b_tensor,
        ctx,
    )

    comptime assert a_type != .float8_e4m3fn or transpose_b, (
        "Testing is only supported for transposed_b==True when"
        " a_type==float8_e4m3fn. Add the non-transposed case if needed."
    )

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

    comptime rtol = 1e-2
    assert_almost_equal(
        c_host._storage,
        c_host_ref._storage,
        c_host.num_elements(),
        atol=0.0001,
        rtol=rtol,
    )
    print("\n=== TEST PASSED ===\n")

    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^


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

                    test_blackwell_matmul_with_weight_prefetch[
                        dtype,
                        dtype,
                        .bfloat16,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        block_swizzle_size=8,
                        swapAB=False,
                        prefetch_tiles_n=2,
                    ](
                        ctx,
                        Int(1000),
                        Idx[1024],
                        Idx[1024 + 16],
                    )

                    comptime for swapAB in [False, True]:
                        test_blackwell_matmul_with_weight_prefetch[
                            dtype,
                            dtype,
                            .bfloat16,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                            a_swizzle=swizzle,
                            b_swizzle=swizzle,
                            block_swizzle_size=4,
                            swapAB=swapAB,
                            prefetch_tiles_n=2,
                        ](
                            ctx,
                            Int(512),
                            Idx[4096],
                            Idx[1024 + 16],
                        )

                        test_blackwell_matmul_with_weight_prefetch[
                            dtype,
                            dtype,
                            .bfloat16,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                            a_swizzle=swizzle,
                            b_swizzle=swizzle,
                            block_swizzle_size=0,
                            swapAB=swapAB,
                            k_group_size=2,
                            prefetch_tiles_n=2,
                        ](
                            ctx,
                            Int(500),
                            Idx[2048],
                            Idx[4096],
                        )

                    test_blackwell_matmul_with_weight_prefetch[
                        dtype,
                        dtype,
                        .bfloat16,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](8, 2, 1),
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        block_swizzle_size=2,
                        swapAB=False,
                        prefetch_tiles_n=2,
                    ](
                        ctx,
                        Int(999),
                        Idx[256],
                        Idx[128],
                    )

                    test_blackwell_matmul_with_weight_prefetch[
                        dtype,
                        dtype,
                        .bfloat16,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        block_swizzle_size=1,
                        swapAB=False,
                        prefetch_tiles_n=2,
                    ](
                        ctx,
                        Int(777),
                        Idx[2560],
                        Idx[8192],
                    )

        # -----------------------------------------------------------------------
        # Part 2: small-K boundary conditions for weight prefetch (2SM)
        #
        # Exercises cases where prefetch_tiles_n >= ceildiv(K, BK) so the
        # runtime guard in Phase 1 / Phase 2 must clamp the prefetch loop.
        # BK=64 (SWIZZLE_128B / sizeof[bf16]).  N must be a multiple of the
        # 2SM umma shape N (2*bn=256), so N=4096 is used throughout.
        # Minimum valid K for 2SM TMA is 128 (2 K-tiles), so Scenarios E and A
        # (which need K=64) are covered only in the 1SM test.
        #
        #   C) K=128, k_group_size=1, pf=2  → num_iters == pf            (safe)
        #   D) K=256, k_group_size=2, pf=2  → num_group_iters == pf      (safe)
        #   B) K=128, k_group_size=2, pf=2  → num_group_iters < pf, guard (safe)
        # -----------------------------------------------------------------------
        comptime for swapAB in [False, True]:
            # C: equality, pf=2 k_group=1
            test_blackwell_matmul_with_weight_prefetch[
                dtype,
                dtype,
                .bfloat16,
                block_tile_shape=Index(64, 128, 64),
                mma_shape=Index(128, 256, 16),
                cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                block_swizzle_size=8,
                swapAB=swapAB,
                k_group_size=1,
                prefetch_tiles_n=2,
            ](ctx, Int(256), Idx[4096], Idx[128])

            # D: equality, pf=2 k_group=2
            test_blackwell_matmul_with_weight_prefetch[
                dtype,
                dtype,
                .bfloat16,
                block_tile_shape=Index(64, 128, 64),
                mma_shape=Index(128, 256, 16),
                cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                block_swizzle_size=8,
                swapAB=swapAB,
                k_group_size=2,
                prefetch_tiles_n=2,
            ](ctx, Int(256), Idx[4096], Idx[256])

            # B: pf > num_group_iters, guard fires (K=128, k_group=2: num_group_iters=1 < pf=2)
            test_blackwell_matmul_with_weight_prefetch[
                dtype,
                dtype,
                .bfloat16,
                block_tile_shape=Index(64, 128, 64),
                mma_shape=Index(128, 256, 16),
                cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                block_swizzle_size=8,
                swapAB=swapAB,
                k_group_size=2,
                prefetch_tiles_n=2,
            ](ctx, Int(256), Idx[4096], Idx[128])
