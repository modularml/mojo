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
from std.sys import argv, size_of
import std.itertools
import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.memory import alloc

from internal_utils import assert_almost_equal
from std.random import rand
from layout import (
    Coord,
    CoordLike,
    Idx,
    TileTensor,
    row_major,
)
from linalg.matmul.gpu.sm100_structured.block_scaled.block_scaled_matmul import (
    blackwell_block_scaled_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100.config import BlockScaledMatmulConfig, GEMMKind
from std.math import ceildiv, align_up
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple
from linalg.fp4_utils import (
    NVFP4_SF_DTYPE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    NVFP4_SF_VECTOR_SIZE,
    set_batched_scale_factor,
)
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind


def simple_init() -> Bool:
    for arg in argv():
        if arg == "--simple-init":
            return True
    return False


def test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
    BatchType: CoordLike,
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
    benchmark: Bool = False,
    swapAB: Bool = False,
    k_group_size: Int = 1,
    SF_VECTOR_SIZE: Int = NVFP4_SF_VECTOR_SIZE,
](ctx: DeviceContext, batch: BatchType, m: MType, n: NType, k: KType,) raises:
    print(
        t"in/out dtypes=({a_type}, {b_type}, {c_type}, {scales_dtype})  problem"
        t" shape=({Int(batch.value())}, {Int(m.value())}, {Int(n.value())},"
        t" {Int(k.value())})"
        t" mma_shape={mma_shape} block_tile_shape={block_tile_shape} cta_group={cta_group} cluster_shape=({cluster_shape[0]},"
        t" {cluster_shape[1]}, {cluster_shape[2]})"
        t" swapAB={swapAB} k_group_size={k_group_size} SF_VECTOR_SIZE={SF_VECTOR_SIZE}"
    )

    # Shape info is derived from the TileLayout types below.
    var a_shape = row_major(Coord(batch, m, Idx[KType.static_value // 2]))
    var b_shape = row_major(Coord(batch, n, Idx[KType.static_value // 2]))
    var c_shape = row_major(Coord(batch, m, n))

    var a_size = Int(batch.value()) * Int(m.value()) * (KType.static_value // 2)
    var b_size = Int(batch.value()) * Int(n.value()) * (KType.static_value // 2)
    var c_size = Int(batch.value()) * Int(m.value()) * Int(n.value())

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
            Int(batch.value()),
            ceildiv(Int(m.value()), SF_MN_GROUP_SIZE),
            Idx[ceildiv(KType.static_value, SF_VECTOR_SIZE * SF_ATOM_K)],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    var b_scales_shape = row_major(
        Coord(
            Int(batch.value()),
            ceildiv(Int(n.value()), SF_MN_GROUP_SIZE),
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

    # Initialize matmul operands
    if simple_init():
        for b in range(Int(batch.value())):
            for m in range(Int(m.value())):
                for k in range(Int(k.value()) // 2):
                    comptime assert a_host.flat_rank == 3
                    a_host[b, m, k] = UInt8(m).cast[a_type]()
        for b in range(Int(batch.value())):
            for n in range(Int(n.value())):
                for k in range(Int(k.value()) // 2):
                    comptime assert b_host.flat_rank == 3
                    b_host[b, n, k] = UInt8(n).cast[b_type]()
    else:
        rand(a_host._storage, a_host.num_elements(), min=0, max=255)
        rand(b_host._storage, b_host.num_elements(), min=0, max=255)

    rand(a_scales_host._storage, a_scales_host.num_elements())
    rand(b_scales_host._storage, b_scales_host.num_elements())
    # NOTE: It is very important that we set unused scales to 0.0 otherwise we will hit accuracy issues
    for batch_idx in range(Int(batch.value())):
        for row_idx in range(align_up(Int(m.value()), SF_MN_GROUP_SIZE)):
            for col_idx in range(
                0,
                align_up(Int(k.value()), SF_VECTOR_SIZE * SF_ATOM_K),
                SF_VECTOR_SIZE,
            ):
                if row_idx >= Int(m.value()) or col_idx >= Int(k.value()):
                    set_batched_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                        a_scales_host,
                        batch_idx,
                        row_idx,
                        col_idx,
                        Scalar[scales_dtype](0.0),
                    )

    for batch_idx in range(Int(batch.value())):
        for row_idx in range(align_up(Int(n.value()), SF_MN_GROUP_SIZE)):
            for col_idx in range(
                0,
                align_up(Int(k.value()), SF_VECTOR_SIZE * SF_ATOM_K),
                SF_VECTOR_SIZE,
            ):
                if row_idx >= Int(n.value()) or col_idx >= Int(k.value()):
                    set_batched_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                        b_scales_host,
                        batch_idx,
                        row_idx,
                        col_idx,
                        Scalar[scales_dtype](0.0),
                    )

    # Move operands to the Device
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
        k_group_size=k_group_size,
        num_accum_pipeline_stages=1 if mma_shape[1] == 256 else 2,
        gemm_kind=GEMMKind.BMM,
    )

    blackwell_block_scaled_matmul_tma_umma_warp_specialized[
        transpose_b=transpose_b,
        config=matmul_config,
    ](
        c_tensor,
        a_tensor,
        b_tensor,
        a_scales_tensor,
        b_scales_tensor,
        ctx,
    )

    var a_2d_shape = row_major(Coord(m, Idx[KType.static_value // 2]))
    var b_2d_shape = row_major(Coord(n, Idx[KType.static_value // 2]))
    var c_2d_shape = row_major(Coord(m, n))
    var a_scales_5d_shape = row_major(
        Coord(
            ceildiv(Int(m.value()), SF_MN_GROUP_SIZE),
            Idx[ceildiv(KType.static_value, SF_VECTOR_SIZE * SF_ATOM_K)],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    var b_scales_5d_shape = row_major(
        Coord(
            ceildiv(Int(n.value()), SF_MN_GROUP_SIZE),
            Idx[ceildiv(KType.static_value, SF_VECTOR_SIZE * SF_ATOM_K)],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )

    var a_batch_stride = Int(m.value()) * (KType.static_value // 2)
    var b_batch_stride = Int(n.value()) * (KType.static_value // 2)
    var c_batch_stride = Int(m.value()) * Int(n.value())
    var a_scales_batch_stride = a_scales_5d_shape.product()
    var b_scales_batch_stride = b_scales_5d_shape.product()

    for b in range(Int(batch.value())):
        var a_2d = TileTensor(
            a_tensor._storage + b * a_batch_stride, a_2d_shape
        )
        var b_2d = TileTensor(
            b_tensor._storage + b * b_batch_stride, b_2d_shape
        )
        var c_ref_2d = TileTensor(
            c_ref_tensor._storage + b * c_batch_stride, c_2d_shape
        )
        var a_scales_5d = TileTensor(
            a_scales_tensor._storage + b * a_scales_batch_stride,
            a_scales_5d_shape,
        )
        var b_scales_5d = TileTensor(
            b_scales_tensor._storage + b * b_scales_batch_stride,
            b_scales_5d_shape,
        )

        vendor_blas.matmul(
            ctx,
            c_ref_2d,
            a_2d,
            b_2d,
            a_scales=a_scales_5d,
            b_scales=b_scales_5d,
            transpose_b=transpose_b,
            c_row_major=True,
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

    # Cleanup
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^
    _ = a_scales_device^
    _ = b_scales_device^


def main() raises:
    with DeviceContext() as ctx:
        comptime dtype = DType.uint8  # TODO: (KERN-2238): Replace with float4-e2m1fn
        comptime out_dtype = DType.bfloat16
        comptime scales_dtype = NVFP4_SF_DTYPE
        comptime SF_VECTOR_SIZE = NVFP4_SF_VECTOR_SIZE
        comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
        comptime BK = (swizzle.bytes() // size_of[dtype]())
        comptime MMA_K = 32
        comptime cta_group = 2

        comptime for bm in [128]:
            comptime for bn in [64, 128]:
                comptime block_tile_shape = Index(bm, bn, BK)
                comptime umma_shape = Index(
                    cta_group * bm, cta_group * bn, MMA_K
                )

                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
                    cta_group=cta_group,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=8,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(2),
                    Int(1000),
                    Idx[1024],
                    Idx[1024 + 32],
                )

                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                    cta_group=cta_group,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=4,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(2),
                    Int(512),
                    Idx[4096],
                    Idx[1024 + 32],
                )

                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](4, 2, 1),
                    cta_group=cta_group,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=0,
                    k_group_size=1,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(3),
                    Int(500),
                    Idx[2048],
                    Idx[4096],
                )

                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](8, 2, 1),
                    cta_group=cta_group,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=2,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(16),
                    Int(999),
                    Idx[256],
                    Idx[128],
                )

                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                    cta_group=cta_group,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=1,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(17),
                    Int(777),
                    Idx[2560],
                    Idx[8192],
                )

                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                    cta_group=cta_group,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=1,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(23),
                    Int(1),
                    Idx[576],
                    Idx[7168],
                )

                # swapAB tests
                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
                    cta_group=cta_group,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    swapAB=True,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(2),
                    Int(16),
                    Idx[1024],
                    Idx[1024 + 32],
                )

                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                    cta_group=cta_group,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    swapAB=True,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(3),
                    Int(100),
                    Idx[2560],
                    Idx[8192],
                )
