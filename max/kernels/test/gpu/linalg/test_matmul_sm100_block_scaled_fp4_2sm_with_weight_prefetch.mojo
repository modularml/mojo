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
# Test: block-scaled FP4 matmul (sm100, 2SM) with weight prefetching.
#
# Mirrors test_matmul_sm100_block_scaled_fp4_2sm.mojo with prefetch_tiles_n=2
# added to BlockScaledMatmulConfig.  All tile shapes (bn in [64, 128]), both
# swapAB variants, and all 8 cluster shapes from the reference test are
# exercised.  NVFP4 scaling only (2SM MXFP4 is not exercised here).
# ===----------------------------------------------------------------------=== #
from std.math import align_up, ceildiv
from std.sys import size_of
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.memory import alloc
from std.random import rand

from internal_utils import assert_almost_equal
from layout import (
    TileTensor,
    Coord,
    CoordLike,
    row_major,
    Idx,
)
from linalg.matmul.gpu.sm100.block_scaled_matmul import (
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
import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind


def _test_impl[
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
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    block_swizzle_size: Int = 0,
    swapAB: Bool = False,
    k_group_size: Int = 1,
    SF_VECTOR_SIZE: Int = NVFP4_SF_VECTOR_SIZE,
    num_accum_pipeline_stages: Int = 0,
    num_clc_pipeline_stages: Int = 2,
    scaling_kind: UMMAKind = UMMAKind.KIND_MXF4NVF4,
    prefetch_tiles_n: Int = 2,
](
    ctx: DeviceContext,
    m: MType,
    n: NType,
    k: KType,
    alpha: Float32 = 1.0,
) raises:
    print(
        t"[prefetch-2sm] in/out dtypes=({a_type}, {b_type}, {c_type},"
        t" {scales_dtype})"
        t" shape=({Int(m.value())}, {Int(n.value())}, {Int(k.value())})"
        t" mma_shape={mma_shape} block_tile_shape={block_tile_shape}"
        t" cta_group={cta_group}"
        t" cluster_shape=({cluster_shape[0]}, {cluster_shape[1]},"
        t" {cluster_shape[2]})"
        t" swapAB={swapAB} k_group_size={k_group_size}"
        t" SF_VECTOR_SIZE={SF_VECTOR_SIZE} alpha={alpha}"
        t" prefetch_tiles_n={prefetch_tiles_n}"
    )

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

    rand(a_host._storage, a_host.num_elements(), min=0, max=255)
    rand(b_host._storage, b_host.num_elements(), min=0, max=255)
    rand(a_scales_host._storage, a_scales_host.num_elements())
    rand(b_scales_host._storage, b_scales_host.num_elements())

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

    var a_lt = a_tensor.to_layout_tensor()
    var b_lt = b_tensor.to_layout_tensor()
    var a_scales_lt = a_scales_tensor.to_layout_tensor()
    var b_scales_lt = b_scales_tensor.to_layout_tensor()
    var c_ref_tensor_lt = c_ref_tensor.to_layout_tensor()

    comptime matmul_config = BlockScaledMatmulConfig[
        a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
    ](
        scaling_kind=scaling_kind,
        cluster_shape=Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        mma_shape=mma_shape,
        block_swizzle_size=block_swizzle_size,
        cta_group=cta_group,
        AB_swapped=swapAB,
        k_group_size=k_group_size,
        num_accum_pipeline_stages=num_accum_pipeline_stages if num_accum_pipeline_stages
        > 0 else (1 if mma_shape[1] in (192, 256) else 2),
        num_clc_pipeline_stages=num_clc_pipeline_stages,
        prefetch_tiles_n=prefetch_tiles_n,
    )

    comptime K_phys = KType.static_value
    blackwell_block_scaled_matmul_tma_umma_warp_specialized[
        transpose_b=transpose_b,
        K=K_phys,
        config=matmul_config,
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


def run_matmul_sm100_block_scaled_fp4_2sm_prefetch_suite[
    suite_scales_dtype: DType,
    suite_sf_vector_size: Int,
    suite_scaling_kind: UMMAKind,
]() raises:
    with DeviceContext() as ctx:
        comptime dtype = DType.uint8
        comptime out_dtype = DType.bfloat16
        comptime scales_dtype = suite_scales_dtype
        comptime SF_VECTOR_SIZE = suite_sf_vector_size
        comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
        comptime BK = (swizzle.bytes() // size_of[dtype]())
        comptime MMA_K = 32

        @__parameter
        @always_inline
        def run[
            MType: CoordLike,
            NType: CoordLike,
            KType: CoordLike,
            //,
            a_type: DType,
            b_type: DType,
            c_type: DType,
            _scales_dtype: DType,
            block_tile_shape: IndexList[3],
            mma_shape: IndexList[3],
            cluster_shape: StaticTuple[Int32, 3],
            cta_group: Int,
            transpose_b: Bool = True,
            a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
            c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
            block_swizzle_size: Int = 0,
            swapAB: Bool = False,
            k_group_size: Int = 1,
            SF_VECTOR_SIZE: Int = NVFP4_SF_VECTOR_SIZE,
            num_accum_pipeline_stages: Int = 0,
            num_clc_pipeline_stages: Int = 2,
        ](
            ctx: DeviceContext,
            m: MType,
            n: NType,
            k: KType,
            alpha: Float32 = 1.0,
        ) raises:
            _test_impl[
                a_type,
                b_type,
                c_type,
                scales_dtype,
                block_tile_shape,
                mma_shape,
                cluster_shape,
                cta_group,
                transpose_b,
                a_swizzle,
                b_swizzle,
                c_swizzle,
                block_swizzle_size,
                swapAB,
                k_group_size,
                SF_VECTOR_SIZE=suite_sf_vector_size,
                num_accum_pipeline_stages=num_accum_pipeline_stages,
                num_clc_pipeline_stages=num_clc_pipeline_stages,
                scaling_kind=suite_scaling_kind,
                prefetch_tiles_n=2,
            ](ctx, m, n, k, alpha)

        # ------------------------------------------------------------------
        # 2SM tile sweep: cta_group=2, bm=128, bn in [64, 128]
        # umma_shape = Index(2*bm, 2*bn, MMA_K) — reflects 2SM layout
        # ------------------------------------------------------------------
        comptime for bm in [128]:
            comptime for bn in [64, 128]:
                comptime block_tile_shape = Index(bm, bn, BK)
                comptime umma_shape = Index(2 * bm, 2 * bn, MMA_K)

                run[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=8,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(1000),
                    Idx[1024],
                    Idx[1024 + 32],
                    alpha=0.5,
                )

                run[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=4,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(512),
                    Idx[4096],
                    Idx[1024 + 32],
                    alpha=0.135,
                )

                run[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](4, 2, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=0,
                    k_group_size=1,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(500),
                    Idx[2048],
                    Idx[4096],
                )

                run[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](8, 2, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=2,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(999),
                    Idx[256],
                    Idx[128],
                )

                run[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=1,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(777),
                    Idx[2560],
                    Idx[8192],
                )

                run[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=1,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(1),
                    Idx[576],
                    Idx[7168],
                )

                # swapAB tests
                run[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    swapAB=True,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(16),
                    Idx[1024],
                    Idx[1024 + 32],
                )

                run[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    swapAB=True,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(100),
                    Idx[2560],
                    Idx[8192],
                )


def main() raises:
    run_matmul_sm100_block_scaled_fp4_2sm_prefetch_suite[
        suite_scales_dtype=NVFP4_SF_DTYPE,
        suite_sf_vector_size=NVFP4_SF_VECTOR_SIZE,
        suite_scaling_kind=UMMAKind.KIND_MXF4NVF4,
    ]()
