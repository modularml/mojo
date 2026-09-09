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
"""Blockwise FP8 1SM tests - Part B (mma_m_scale=2, 128xN MMA shapes)."""

from std.math import ceildiv
from std.sys import argv, size_of
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    MatmulConfig,
    GEMMKind,
)
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.memory import unsafe_memset_zero
from internal_utils import (
    assert_almost_equal,
    assert_with_measure,
)
from std.random import rand
from internal_utils._measure import relative_difference
from layout import (
    TileTensor,
    Coord,
    CoordLike,
    row_major,
    Idx,
)
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_matmul
from linalg.matmul.gpu.sm100_structured.blockwise_fp8.blockwise_fp8_matmul import (
    blockwise_fp8_matmul,
)

from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple


def simple_init() -> Bool:
    for arg in argv():
        if arg == "--simple-init":
            return True
    return False


def test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
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
    cta_group: Int,
    scales_type: DType = .float32,
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    comptime BLOCK_SCALE_K = 128

    if Int(m.value()) * size_of[DType.float32]() % 16 != 0:
        raise Error("TMA expects M to be divisible by 16 bytes")

    print(
        "in/out dtypes=(",
        a_type,
        ", ",
        b_type,
        ", ",
        c_type,
        ") ",
        " problem shape=(",
        Int(m.value()),
        ", ",
        Int(n.value()),
        ", ",
        Int(k.value()),
        ") ",
        "mma_shape=",
        mma_shape,
        " block_tile_shape=",
        block_tile_shape,
        " cta_group=",
        cta_group,
        " cluster_shape=(",
        cluster_shape[0],
        ", ",
        cluster_shape[1],
        ", ",
        cluster_shape[2],
        ")",
        sep="",
    )

    var a_shape = row_major(Coord(m, Idx[KType.static_value]))
    var b_shape = row_major(
        Coord(
            Idx[NType.static_value if transpose_b else KType.static_value],
            Idx[KType.static_value if transpose_b else NType.static_value],
        )
    )
    var c_shape = row_major(Coord(m, Idx[NType.static_value]))

    var a_scales_shape = row_major(
        Coord(ceildiv(Int(k.value()), BLOCK_SCALE_K), m)
    )
    var b_scales_shape = row_major(
        Coord(
            ceildiv(Int(n.value()), BLOCK_SCALE_K),
            ceildiv(Int(k.value()), BLOCK_SCALE_K),
        )
    )

    var a_size = Int(m.value()) * Int(k.value())
    var b_size = (
        Int(n.value())
        * Int(k.value()) if transpose_b else Int(k.value())
        * Int(n.value())
    )
    var c_size = Int(m.value()) * Int(n.value())
    var a_scales_size = ceildiv(Int(k.value()), BLOCK_SCALE_K) * Int(m.value())
    var b_scales_size = ceildiv(Int(n.value()), BLOCK_SCALE_K) * ceildiv(
        Int(k.value()), BLOCK_SCALE_K
    )

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

    var a_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_type](
        a_scales_size
    )
    var a_scales_host = TileTensor(a_scales_host_ptr, a_scales_shape)
    var b_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_type](
        b_scales_size
    )
    var b_scales_host = TileTensor(b_scales_host_ptr, b_scales_shape)

    var a_scales_device = ctx.enqueue_create_buffer[scales_type](a_scales_size)
    var a_scales_tensor = TileTensor(a_scales_device, a_scales_shape)
    var b_scales_device = ctx.enqueue_create_buffer[scales_type](b_scales_size)
    var b_scales_tensor = TileTensor(b_scales_device, b_scales_shape)

    unsafe_memset_zero(c_host_ptr.unsafe_ptr(), c_size)
    unsafe_memset_zero(c_host_ref_ptr.unsafe_ptr(), c_size)

    # Initialize matmul operands
    if simple_init():
        for m in range(Int(m.value())):
            for k in range(Int(k.value())):
                comptime assert a_host.flat_rank == 2
                a_host[m, k] = Scalar[a_type](1.0)
        for n in range(Int(n.value())):
            for k in range(Int(k.value())):
                b_host[n, k] = Scalar[b_type](1.0)

        for m in range(Int(m.value())):
            for k in range(Int(k.value())):
                comptime assert a_scales_host.flat_rank == 2
                a_scales_host[k // BLOCK_SCALE_K, m] = Scalar[scales_type](0.5)
        for n in range(Int(n.value())):
            for k in range(Int(k.value())):
                comptime assert b_scales_host.flat_rank >= 2
                b_scales_host[n // BLOCK_SCALE_K, k // BLOCK_SCALE_K] = Scalar[
                    scales_type
                ](0.5)

    else:
        rand(a_host._storage, a_host.num_elements())
        rand(b_host._storage, b_host.num_elements())
        rand(a_scales_host._storage, a_scales_host.num_elements())
        rand(b_scales_host._storage, b_scales_host.num_elements())

    # Move operands to the Device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)

    # LayoutTensors for reference matmul
    var a_lt = a_tensor.to_layout_tensor()
    var b_lt = b_tensor.to_layout_tensor()
    var a_scales_lt = a_scales_tensor.to_layout_tensor()
    var b_scales_lt = b_scales_tensor.to_layout_tensor()
    var c_ref_tensor_lt = c_ref_tensor.to_layout_tensor()

    comptime matmul_config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        cluster_shape=Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        mma_shape=mma_shape,
        block_swizzle_size=0,
        cta_group=cta_group,
        gemm_kind=GEMMKind.BLOCK_SCALED_1D2D_FP8,
    )

    blockwise_fp8_matmul[
        transpose_b=transpose_b,
        a_scales_type=scales_type,
        b_scales_type=scales_type,
        config=matmul_config,
    ](
        c_tensor,
        a_tensor,
        b_tensor,
        a_scales_tensor,
        b_scales_tensor,
        ctx,
    )

    naive_blockwise_scaled_fp8_matmul[
        BLOCK_DIM=16,
        transpose_b=transpose_b,
        scales_granularity_mnk=Index(1, BLOCK_SCALE_K, BLOCK_SCALE_K),
    ](
        c_ref_tensor_lt,
        a_lt,
        b_lt,
        a_scales_lt.as_imm(),
        b_scales_lt.as_imm(),
        ctx,
    )

    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    assert_with_measure[relative_difference](
        c_host._storage,
        c_host_ref._storage,
        c_host.num_elements(),
        threshold=0.001,
    )

    assert_almost_equal(
        c_host._storage,
        c_host_ref._storage,
        c_host.num_elements(),
        atol=1e-2,
        rtol=1e-2,
    )

    # Cleanup
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^
    _ = a_scales_device^
    _ = b_scales_device^


def main() raises:
    with DeviceContext() as ctx:
        comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
        comptime in_dtype = DType.float8_e4m3fn
        comptime BK = (swizzle.bytes() // size_of[in_dtype]())
        comptime MMA_K = 32
        comptime out_dtype = DType.bfloat16

        # Part B: mma_m_scale = 2 only (128xN MMA shapes)
        comptime for mma_n_scale in [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            20,
            24,
            32,
        ]:
            comptime block_tile_shape = Index(128, 8 * mma_n_scale, BK)
            comptime umma_shape = Index(128, 8 * mma_n_scale, MMA_K)

            print(
                "block_tile_shape",
                block_tile_shape,
                "umma_shape",
                umma_shape,
            )

            test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
                in_dtype,
                in_dtype,
                out_dtype,
                block_tile_shape,
                umma_shape,
                cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                cta_group=1,
            ](
                ctx,
                Int(1000),
                Idx[576],
                Idx[7168],
            )

            test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
                in_dtype,
                in_dtype,
                out_dtype,
                block_tile_shape,
                umma_shape,
                cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                cta_group=1,
            ](
                ctx,
                Int(1000),
                Idx[576],
                Idx[256 + 64],
            )

            test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
                in_dtype,
                in_dtype,
                out_dtype,
                block_tile_shape,
                umma_shape,
                cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                scales_type=.bfloat16,
                cta_group=1,
            ](
                ctx,
                Int(1000),
                Idx[32768],
                Idx[512],
            )

            test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
                in_dtype,
                in_dtype,
                out_dtype,
                block_tile_shape,
                umma_shape,
                cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                cta_group=1,
            ](
                ctx,
                Int(512),
                Idx[4096],
                Idx[1024],
            )

            test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
                in_dtype,
                in_dtype,
                out_dtype,
                block_tile_shape,
                umma_shape,
                cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                cta_group=1,
            ](
                ctx,
                Int(500),
                Idx[24576],
                Idx[1536],
            )

            test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
                in_dtype,
                in_dtype,
                out_dtype,
                block_tile_shape,
                umma_shape,
                cluster_shape=StaticTuple[Int32, 3](8, 2, 1),
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                cta_group=1,
            ](
                ctx,
                Int(1024),
                Idx[1536],
                Idx[7168],
            )

            test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
                in_dtype,
                in_dtype,
                out_dtype,
                block_tile_shape,
                umma_shape,
                cluster_shape=StaticTuple[Int32, 3](2, 2, 1),
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                scales_type=.bfloat16,
                cta_group=1,
            ](
                ctx,
                Idx[1024],
                Idx[1024],
                Idx[2048],
            )

            test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
                in_dtype,
                in_dtype,
                out_dtype,
                block_tile_shape,
                umma_shape,
                cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                cta_group=1,
            ](
                ctx,
                Int(8192),
                Idx[2560],
                Idx[8192],
            )
