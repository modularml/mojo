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

from std.collections import Optional
from std.sys import align_of, size_of
from std.math import ceildiv
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle

# Additional imports for testing
from internal_utils import (
    assert_almost_equal,
    assert_with_measure,
)
from std.random import rand
from internal_utils._measure import relative_difference
from layout import TileTensor, Coord, CoordLike, Idx, row_major
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_matmul
from linalg.matmul.gpu.sm100.blockwise_fp8 import (
    matmul_sm100_blockwise_scaled_fp8,
)
from linalg.utils import elementwise_epilogue_type

from std.utils.index import Index, IndexList


def test_matmul_sm100_blockwise_scaled_fp8[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    umma_shape: IndexList[3],
    swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    transpose_b: Bool = True,
    use_epilogue: Bool = False,
](ctx: DeviceContext, m: MType, n: NType, k: KType,) raises:
    comptime BLOCK_SCALE_K = 128
    comptime block_tile_shape = Index(umma_shape[0], umma_shape[1], 128)

    comptime assert transpose_b, "transpose_b must be true"

    var M = Int(m.value())
    var N = Int(n.value())
    var K = Int(k.value())

    assert (
        M * size_of[DType.float32]() % 16 == 0
    ), "TMA expects M to be divisible by 16 bytes"

    print(
        "== test_sm100_blockwise_scaled_fp8_matmul",
        a_type,
        "problem shape: (",
        M,
        "x",
        N,
        "x",
        K,
        ")",
        "block_tile_shape: (",
        block_tile_shape[0],
        "x",
        block_tile_shape[1],
        "x",
        block_tile_shape[2],
        ")",
        "transpose_b:",
        transpose_b,
    )

    assert K % BLOCK_SCALE_K == 0, "K must be divisible by BLOCK_SCALE_K"

    var a_shape = row_major(Coord(m, k))
    var b_shape = row_major(
        Coord(
            Idx[NType.static_value if transpose_b else KType.static_value],
            Idx[KType.static_value if transpose_b else NType.static_value],
        )
    )
    var c_shape = row_major(Coord(m, n))
    var a_scales_shape = row_major(
        Coord(Idx[ceildiv(KType.static_value, BLOCK_SCALE_K)], m)
    )
    var b_scales_shape = row_major(
        Coord(
            Idx[ceildiv(NType.static_value, BLOCK_SCALE_K)],
            Idx[ceildiv(KType.static_value, BLOCK_SCALE_K)],
        )
    )

    var a_size = M * K
    var b_size = N * K if transpose_b else K * N
    var c_size = M * N
    var a_scales_size = ceildiv(K, BLOCK_SCALE_K) * M
    var b_scales_size = ceildiv(N, BLOCK_SCALE_K) * ceildiv(K, BLOCK_SCALE_K)

    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_size)
    var a_host = TileTensor(a_host_ptr, a_shape)
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](b_size)
    var b_host = TileTensor(b_host_ptr, b_shape)
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_host = TileTensor(c_host_ptr, c_shape)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_host_ref = TileTensor(c_host_ref_ptr, c_shape)

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_device_nd = TileTensor(a_device, a_shape)
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_device_nd = TileTensor(b_device, b_shape)
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_nd = TileTensor(c_device, c_shape)
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_ref_nd = TileTensor(c_device_ref, c_shape)

    var a_scales_host_ptr = ctx.enqueue_create_host_buffer[.float32](
        a_scales_size
    )
    var a_scales_host = TileTensor(a_scales_host_ptr, a_scales_shape)
    var b_scales_host_ptr = ctx.enqueue_create_host_buffer[.float32](
        b_scales_size
    )
    var b_scales_host = TileTensor(b_scales_host_ptr, b_scales_shape)

    var a_scales_device = ctx.enqueue_create_buffer[.float32](a_scales_size)
    var a_scales_device_nd = TileTensor(a_scales_device, a_scales_shape)
    var b_scales_device = ctx.enqueue_create_buffer[.float32](b_scales_size)
    var b_scales_device_nd = TileTensor(b_scales_device, b_scales_shape)

    var c_tensor = c_device_nd

    @__parameter
    @always_inline
    @__copy_capture(c_tensor)
    def epilogue_fn[
        _dtype: DType,
        width: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> None:
        comptime assert c_tensor.flat_rank >= 2
        c_tensor.store[alignment=alignment](
            Coord(idx[0], idx[1]),
            rebind[SIMD[c_type, width]](val),
        )

    rand(a_host._storage, a_host.num_elements())
    rand(b_host._storage, b_host.num_elements())
    _ = c_host.fill(0)
    _ = c_host_ref.fill(0)

    rand(a_scales_host._storage, a_scales_host.num_elements())
    rand(b_scales_host._storage, b_scales_host.num_elements())

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    ctx.enqueue_copy(c_device, c_host_ptr)
    ctx.enqueue_copy(c_device_ref, c_host_ref_ptr)

    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)

    matmul_sm100_blockwise_scaled_fp8[
        transpose_b=transpose_b,
        umma_shape=umma_shape,
        block_tile_shape=block_tile_shape,
        a_swizzle=swizzle,
        b_swizzle=swizzle,
        elementwise_lambda_fn=Optional[elementwise_epilogue_type](
            epilogue_fn
        ) if use_epilogue else None,
    ](
        c_device_nd,
        a_device_nd,
        b_device_nd,
        a_scales_device_nd,
        b_scales_device_nd,
        ctx,
    )

    ctx.synchronize()

    var a_lt = a_device_nd.to_layout_tensor()
    var b_lt = b_device_nd.to_layout_tensor()
    var c_ref_lt = c_device_ref_nd.to_layout_tensor()
    var a_scales_lt = a_scales_device_nd.to_layout_tensor()
    var b_scales_lt = b_scales_device_nd.to_layout_tensor()

    naive_blockwise_scaled_fp8_matmul[
        BLOCK_DIM=16,
        transpose_b=transpose_b,
        scales_granularity_mnk=Index(1, BLOCK_SCALE_K, BLOCK_SCALE_K),
    ](
        c_ref_lt,
        a_lt,
        b_lt,
        a_scales_lt,
        b_scales_lt,
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


def main() raises:
    with DeviceContext() as ctx:
        test_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            umma_shape=Index(64, 64, 32),
            swizzle=TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](
            ctx,
            Int(120),
            Idx[1536],
            Idx[7168],
        )
        test_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            umma_shape=Index(64, 64, 32),
            swizzle=TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](
            ctx,
            Int(120),
            Idx[24576],
            Idx[1536],
        )
        test_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            umma_shape=Index(64, 64, 32),
            swizzle=TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
            use_epilogue=True,
        ](
            ctx,
            Int(128),
            Idx[576],
            Idx[7168],
        )

        test_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            umma_shape=Index(64, 64, 32),
            swizzle=TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](
            ctx,
            Int(400),
            Idx[32768],
            Idx[512],
        )

        test_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            umma_shape=Index(64, 128, 32),
            swizzle=TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](
            ctx,
            Int(1024),
            Idx[2048],
            Idx[2048],
        )

        test_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            umma_shape=Index(64, 64, 32),
            swizzle=TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](
            ctx,
            Int(1024),
            Idx[2048],
            Idx[2048],
        )

        test_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            umma_shape=Index(64, 16, 32),
            swizzle=TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](
            ctx,
            Int(100),
            Idx[512],
            Idx[256],
        )

        test_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            umma_shape=Index(64, 8, 32),
            swizzle=TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](
            ctx,
            Int(96),
            Idx[1024],
            Idx[1024],
        )

        test_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            umma_shape=Index(64, 256, 32),
            swizzle=TensorMapSwizzle.SWIZZLE_128B,
            transpose_b=True,
        ](
            ctx,
            Int(208),
            Idx[2048],
            Idx[256],
        )
