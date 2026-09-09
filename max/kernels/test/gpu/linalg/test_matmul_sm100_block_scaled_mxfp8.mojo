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
from std.math import ceildiv, align_up
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    MXFP8_SF_VECTOR_SIZE,
    set_scale_factor,
)
from std.random import random_ui64
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind


def simple_init() -> Bool:
    for arg in argv():
        if arg == "--simple-init":
            return True
    return False


def test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
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
    SF_VECTOR_SIZE: Int = MXFP8_SF_VECTOR_SIZE,
](ctx: DeviceContext, m: MType, n: NType, k: KType,) raises:
    print(
        t"in/out dtypes=({a_type}, {b_type}, {c_type}, {scales_dtype})"
        t"  problem shape=({Int(m.value())}, {Int(n.value())},"
        t" {Int(k.value())})"
        t" mma_shape={mma_shape} block_tile_shape={block_tile_shape}"
        t" cta_group={cta_group} cluster_shape=({cluster_shape[0]},"
        t" {cluster_shape[1]}, {cluster_shape[2]})"
        t" swapAB={swapAB} k_group_size={k_group_size}"
        t" SF_VECTOR_SIZE={SF_VECTOR_SIZE}"
    )

    # Shape info is derived from the TileLayout types below.
    var a_shape = row_major(Coord(m, Idx[KType.static_value]))
    comptime assert (
        transpose_b
    ), "TileTensor migration only supports transpose_b=True for now"
    var b_shape = row_major(
        Coord(Idx[NType.static_value], Idx[KType.static_value])
    )
    var c_shape = row_major(Coord(m, Idx[NType.static_value]))

    var a_size = Int(m.value()) * Int(k.value())
    var b_size = Int(n.value()) * Int(k.value())
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

    # LayoutTensors for reference matmul (vendor_blas)
    var a_lt = a_tensor.to_layout_tensor()
    var b_lt = b_tensor.to_layout_tensor()
    var a_scales_lt = a_scales_tensor.to_layout_tensor()
    var b_scales_lt = b_scales_tensor.to_layout_tensor()
    var c_ref_tensor_lt = c_ref_tensor.to_layout_tensor()

    # Initialize matmul operands
    if simple_init():
        for m in range(Int(m.value())):
            for k in range(Int(k.value())):
                comptime assert a_host.flat_rank == 2
                a_host[m, k] = random_ui64(0, 1).cast[a_type]()
        for n in range(Int(n.value())):
            for k in range(Int(k.value())):
                comptime assert b_host.flat_rank == 2
                b_host[n, k] = random_ui64(0, 1).cast[b_type]()
    else:
        rand(a_host._storage, a_host.num_elements())
        rand(b_host._storage, b_host.num_elements())

    # NOTE: It is very important that we set unused scales to 0.0 otherwise we will hit accuracy issues
    for idx0 in range(align_up(Int(m.value()), SF_MN_GROUP_SIZE)):
        for idx1 in range(
            0,
            align_up(Int(k.value()), SF_VECTOR_SIZE * SF_ATOM_K),
            SF_VECTOR_SIZE,
        ):
            if idx0 < Int(m.value()) and idx1 < Int(k.value()):
                var scale_value = (
                    (1 << random_ui64(0, 3))
                    .cast[.float32]()
                    .cast[scales_dtype]()
                )
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    a_scales_host, idx0, idx1, scale_value
                )
            else:
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    a_scales_host, idx0, idx1, Scalar[scales_dtype](0.0)
                )

    for idx0 in range(align_up(Int(n.value()), SF_MN_GROUP_SIZE)):
        for idx1 in range(
            0,
            align_up(Int(k.value()), SF_VECTOR_SIZE * SF_ATOM_K),
            SF_VECTOR_SIZE,
        ):
            if idx0 < Int(n.value()) and idx1 < Int(k.value()):
                var scale_value = (
                    (1 << random_ui64(0, 3))
                    .cast[.float32]()
                    .cast[scales_dtype]()
                )
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    b_scales_host, idx0, idx1, scale_value
                )
            else:
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    b_scales_host, idx0, idx1, Scalar[scales_dtype](0.0)
                )

    # Move operands to the Device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)

    comptime matmul_config = BlockScaledMatmulConfig[
        a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
    ](
        scaling_kind=UMMAKind.KIND_MXF8F6F4,
        cluster_shape=Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        mma_shape=mma_shape,
        block_swizzle_size=block_swizzle_size,
        cta_group=cta_group,
        AB_swapped=swapAB,
        k_group_size=k_group_size,
        num_accum_pipeline_stages=1 if mma_shape[1] in (192, 256) else 2,
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
    )

    comptime assert a_type != .float8_e4m3fn or transpose_b, (
        "Testing is only supported for transposed_b==True when"
        " a_type==float8_e4m3fn. Add the non-transposed case if needed."
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
        comptime dtype = DType.float8_e4m3fn
        comptime out_dtype = DType.bfloat16
        comptime scale_dtype = MXFP8_SF_DTYPE
        comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE
        comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
        comptime BK = (swizzle.bytes() // size_of[dtype]())
        comptime MMA_K = 32

        comptime for cta_group in [1, 2]:
            comptime for bm in [128]:
                comptime for mma_n in [64, 128, 192, 256]:
                    comptime block_tile_shape = Index(
                        bm, mma_n // cta_group, BK
                    )
                    comptime umma_shape = Index(cta_group * bm, mma_n, MMA_K)

                    test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                        dtype,
                        dtype,
                        out_dtype,
                        scale_dtype,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](
                            Int32(cta_group), 1, 1
                        ),
                        cta_group=cta_group,
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        block_swizzle_size=8,
                    ](
                        ctx,
                        Int(1000),
                        Idx[1024],
                        Idx[1024 + 16],
                    )

                    test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                        dtype,
                        dtype,
                        out_dtype,
                        scale_dtype,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](4, 2, 1),
                        cta_group=cta_group,
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        block_swizzle_size=0,
                        k_group_size=2,
                    ](
                        ctx,
                        Int(500),
                        Idx[2048],
                        Idx[4096],
                    )

                    test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                        dtype,
                        dtype,
                        out_dtype,
                        scale_dtype,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](8, 2, 1),
                        cta_group=cta_group,
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        block_swizzle_size=2,
                    ](
                        ctx,
                        Int(999),
                        Idx[256],
                        Idx[128],
                    )

                    test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                        dtype,
                        dtype,
                        out_dtype,
                        scale_dtype,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                        cta_group=cta_group,
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        block_swizzle_size=1,
                    ](
                        ctx,
                        Int(777),
                        Idx[2560],
                        Idx[8192],
                    )

                    test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                        dtype,
                        dtype,
                        out_dtype,
                        scale_dtype,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                        cta_group=cta_group,
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        block_swizzle_size=1,
                    ](
                        ctx,
                        Int(1),
                        Idx[576],
                        Idx[7168],
                    )

                    # swapAB tests
                    test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                        dtype,
                        dtype,
                        out_dtype,
                        scale_dtype,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](
                            Int32(cta_group), 1, 1
                        ),
                        cta_group=cta_group,
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        swapAB=True,
                    ](
                        ctx,
                        Int(16),
                        Idx[1024],
                        Idx[1024 + 16],
                    )

                    test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                        dtype,
                        dtype,
                        out_dtype,
                        scale_dtype,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                        cta_group=cta_group,
                        a_swizzle=swizzle,
                        b_swizzle=swizzle,
                        swapAB=True,
                        k_group_size=2,
                    ](
                        ctx,
                        Int(100),
                        Idx[2560],
                        Idx[8192],
                    )
