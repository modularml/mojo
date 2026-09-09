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
# Test: NVFP4 block-scaled matmul (sm100) with weight prefetching.
#
# Validates that prefetch_tiles_n > 0 produces correct results for both
# swapAB=False and swapAB=True, across 1SM (cta_group=1) and 2SM
# (cta_group=2) tile configurations.
# ===----------------------------------------------------------------------=== #

from std.math import align_up, ceildiv
from std.sys import size_of
import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.primitives.grid_controls import PDLLevel
from internal_utils import assert_almost_equal
from std.random import rand
from layout import (
    TileTensor,
    Coord,
    CoordLike,
    row_major,
    Idx,
)
from linalg.matmul.gpu.sm100_structured.block_scaled.block_scaled_matmul import (
    blackwell_block_scaled_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100.config import BlockScaledMatmulConfig
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple
from linalg.fp4_utils import (
    NVFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    set_scale_factor,
)
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind


def test_block_scaled_prefetch[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    scales_dtype: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    cta_group: Int,
    transpose_b: Bool = True,
    block_swizzle_size: Int = 0,
    swapAB: Bool = False,
    prefetch_tiles_n: Int = 2,
    SF_VECTOR_SIZE: Int = NVFP4_SF_VECTOR_SIZE,
](
    ctx: DeviceContext, m: MType, n: NType, k: KType, alpha: Float32 = 1.0
) raises:
    print(
        t"NVFP4 prefetch: dtypes=({a_type},{b_type},{c_type},{scales_dtype})"
        t" shape=({Int(m.value())},{Int(n.value())},{Int(k.value())})"
        t" mma_shape={mma_shape} cta_group={cta_group}"
        t" cluster=({cluster_shape[0]},{cluster_shape[1]},{cluster_shape[2]})"
        t" swapAB={swapAB} prefetch_tiles_n={prefetch_tiles_n}"
    )

    # NVFP4: each uint8 stores 2 FP4 values; halve K for buffer sizing
    var a_shape = row_major(Coord(m, Idx[KType.static_value // 2]))
    var b_shape = row_major(
        Coord(Idx[NType.static_value], Idx[KType.static_value // 2])
    )
    var c_shape = row_major(Coord(m, Idx[NType.static_value]))

    var a_size = Int(m.value()) * (KType.static_value // 2)
    var b_size = Int(n.value()) * (KType.static_value // 2)
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

    var a_scales_shape = row_major(
        Coord(
            ceildiv(Int(m.value()), SF_MN_GROUP_SIZE),
            Idx[ceildiv(KType.static_value, SF_VECTOR_SIZE * SF_ATOM_K)],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    var b_scales_shape = row_major(
        Coord(
            Idx[ceildiv(NType.static_value, SF_MN_GROUP_SIZE)],
            Idx[ceildiv(KType.static_value, SF_VECTOR_SIZE * SF_ATOM_K)],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )

    var a_scales_total = a_scales_shape.product()
    var b_scales_total = b_scales_shape.product()

    var a_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_dtype](
        a_scales_total
    )
    var a_scales_host = TileTensor(a_scales_host_ptr, a_scales_shape)
    var b_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_dtype](
        b_scales_total
    )
    var b_scales_host = TileTensor(b_scales_host_ptr, b_scales_shape)

    var a_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        a_scales_total
    )
    var a_scales_tensor = TileTensor(a_scales_device, a_scales_shape)
    var b_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        b_scales_total
    )
    var b_scales_tensor = TileTensor(b_scales_device, b_scales_shape)

    var a_lt = a_tensor.to_layout_tensor()
    var b_lt = b_tensor.to_layout_tensor()
    var a_scales_lt = a_scales_tensor.to_layout_tensor()
    var b_scales_lt = b_scales_tensor.to_layout_tensor()
    var c_ref_tensor_lt = c_ref_tensor.to_layout_tensor()

    rand(a_host._storage, a_host.num_elements(), min=0, max=255)
    rand(b_host._storage, b_host.num_elements(), min=0, max=255)
    rand(a_scales_host._storage, a_scales_host.num_elements())
    rand(b_scales_host._storage, b_scales_host.num_elements())

    # Zero out unused scale entries to avoid accuracy issues
    for idx0 in range(align_up(Int(m.value()), SF_MN_GROUP_SIZE)):
        for idx1 in range(
            0,
            align_up(Int(k.value()), SF_VECTOR_SIZE * SF_ATOM_K),
            SF_VECTOR_SIZE,
        ):
            if idx0 >= Int(m.value()) or idx1 >= Int(k.value()):
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    a_scales_host, idx0, idx1, Scalar[scales_dtype](0.0)
                )

    for idx0 in range(align_up(Int(n.value()), SF_MN_GROUP_SIZE)):
        for idx1 in range(
            0,
            align_up(Int(k.value()), SF_VECTOR_SIZE * SF_ATOM_K),
            SF_VECTOR_SIZE,
        ):
            if idx0 >= Int(n.value()) or idx1 >= Int(k.value()):
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    b_scales_host, idx0, idx1, Scalar[scales_dtype](0.0)
                )

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)

    comptime matmul_config = BlockScaledMatmulConfig[
        a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
    ](
        scaling_kind=UMMAKind.KIND_MXF4NVF4,
        cluster_shape=Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        mma_shape=mma_shape,
        block_swizzle_size=block_swizzle_size,
        cta_group=cta_group,
        AB_swapped=swapAB,
        num_accum_pipeline_stages=1 if mma_shape[1] == 256 else 2,
        prefetch_tiles_n=prefetch_tiles_n,
    )

    blackwell_block_scaled_matmul_tma_umma_warp_specialized[
        transpose_b=transpose_b,
        config=matmul_config,
        pdl_level=PDLLevel.ON,
    ](
        c_tensor,
        a_tensor,
        b_tensor,
        a_scales_tensor,
        b_scales_tensor,
        ctx,
        alpha,
    )

    vendor_blas.matmul(
        ctx,
        c_ref_tensor_lt.as_unsafe_any_origin(),
        a_lt,
        b_lt,
        a_scales=a_scales_lt.as_imm().as_unsafe_any_origin(),
        b_scales=b_scales_lt.as_imm().as_unsafe_any_origin(),
        transpose_b=transpose_b,
        c_row_major=True,
        alpha=alpha,
    )

    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    assert_almost_equal(
        c_host._storage,
        c_host_ref._storage,
        c_host.num_elements(),
        atol=1e-2,
        rtol=1e-2,
    )
    print("\n=== TEST PASSED ===\n")

    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^
    _ = a_scales_device^
    _ = b_scales_device^


def main() raises:
    with DeviceContext() as ctx:
        comptime dtype = DType.uint8  # two FP4 values per uint8
        comptime out_dtype = DType.bfloat16
        comptime scales_dtype = NVFP4_SF_DTYPE
        comptime SF_VECTOR_SIZE = NVFP4_SF_VECTOR_SIZE
        comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
        comptime BK = swizzle.bytes() // size_of[dtype]()  # 128 uint8
        comptime MMA_K = 32

        # ------------------------------------------------------------------
        # 1SM (cta_group=1): mma_shape == block_tile_shape in M and N
        # ------------------------------------------------------------------
        comptime for bm in [128]:
            comptime for bn in [128, 256]:
                comptime block_tile_shape = Index(bm, bn, BK)
                comptime umma_shape = Index(bm, bn, MMA_K)

                # swapAB=False, cluster 1×1
                test_block_scaled_prefetch[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
                    cta_group=1,
                    block_swizzle_size=8,
                    swapAB=False,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(1000),
                    Idx[1024],
                    Idx[1024 + 32],
                )

                # swapAB=True, cluster 1×1
                test_block_scaled_prefetch[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
                    cta_group=1,
                    block_swizzle_size=8,
                    swapAB=True,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(16),
                    Idx[1024],
                    Idx[1024 + 32],
                )

                # swapAB=False, cluster 4×4 (N must be multiple of 4*bn)
                test_block_scaled_prefetch[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                    cta_group=1,
                    block_swizzle_size=4,
                    swapAB=False,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(512),
                    Idx[4096],
                    Idx[1024 + 32],
                )

                # swapAB=True, cluster 4×4
                test_block_scaled_prefetch[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                    cta_group=1,
                    block_swizzle_size=4,
                    swapAB=True,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(100),
                    Idx[2560],
                    Idx[8192],
                )

        # ------------------------------------------------------------------
        # 2SM (cta_group=2): mma_shape is doubled vs block_tile_shape in M/N
        # ------------------------------------------------------------------
        comptime for bm in [128]:
            comptime for bn in [64, 128]:
                comptime block_tile_shape = Index(bm, bn, BK)
                comptime umma_shape = Index(2 * bm, 2 * bn, MMA_K)

                # swapAB=False
                test_block_scaled_prefetch[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                    cta_group=2,
                    block_swizzle_size=8,
                    swapAB=False,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(1000),
                    Idx[1024],
                    Idx[1024 + 32],
                )

                # swapAB=True
                test_block_scaled_prefetch[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                    cta_group=2,
                    block_swizzle_size=4,
                    swapAB=True,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(100),
                    Idx[2560],
                    Idx[8192],
                )

        # ------------------------------------------------------------------
        # Small-K boundary: guard prevents deadlock when pf >= num_iters.
        #
        # BK=128 uint8; K in FP4 units; K_stored_uint8 = K_fp4 // 2.
        # num_iters = ceildiv(K_stored_uint8, BK).
        #
        #   A) K_fp4=2*BK=256 → K_stored=128 → num_iters=1 < pf=2
        #      Phase 1: pf=0 fires (0 < 1), pf=1 skipped (1 < 1 False)
        #   B) K_fp4=4*BK=512 → K_stored=256 → num_iters=2 == pf=2
        #      Phase 1: both pf=0 and pf=1 fire; while loop is empty
        # ------------------------------------------------------------------
        comptime for swapAB in [False, True]:
            # A: only pf=0 fires; pf=1 guard clamps
            test_block_scaled_prefetch[
                dtype,
                dtype,
                out_dtype,
                scales_dtype,
                block_tile_shape=Index(128, 128, BK),
                mma_shape=Index(128, 128, MMA_K),
                cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                cta_group=1,
                block_swizzle_size=8,
                swapAB=swapAB,
                prefetch_tiles_n=2,
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            ](ctx, Int(256), Idx[1024], Idx[2 * BK])

            # B: pf=2 equals num_iters=2; while loop is entirely skipped
            test_block_scaled_prefetch[
                dtype,
                dtype,
                out_dtype,
                scales_dtype,
                block_tile_shape=Index(128, 128, BK),
                mma_shape=Index(128, 128, MMA_K),
                cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                cta_group=1,
                block_swizzle_size=8,
                swapAB=swapAB,
                prefetch_tiles_n=2,
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            ](ctx, Int(256), Idx[1024], Idx[4 * BK])
