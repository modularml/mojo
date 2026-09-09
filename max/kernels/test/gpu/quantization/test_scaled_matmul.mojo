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

from max.gpu.host import DeviceContext
from layout import (
    CoordLike,
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from layout._fillers import random
from linalg.fp8_quantization import matmul_dynamic_scaled_fp8
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_matmul
from std.testing import assert_almost_equal
from std.utils.index import Index


def test_matmul_dynamic_scaled_fp8[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    transpose_b: Bool,
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    var a_size = Int(m.value()) * Int(k.value())
    var b_size = (
        Int(n.value())
        * Int(k.value()) if transpose_b else Int(k.value())
        * Int(n.value())
    )
    var c_size = Int(m.value()) * Int(n.value())
    var a_scales_size = 1 * Int(m.value())
    var b_scales_size = (
        Int(n.value()) * 1 if transpose_b else 1 * Int(n.value())
    )

    # Host allocations
    var a_host_ptr = alloc[Scalar[in_dtype]](a_size)
    var b_host_ptr = alloc[Scalar[in_dtype]](b_size)
    var c_host_ptr = alloc[Scalar[out_dtype]](c_size)
    var a_scales_host_ptr = alloc[Scalar[scales_dtype]](a_scales_size)
    var b_scales_host_ptr = alloc[Scalar[scales_dtype]](b_scales_size)
    var c_host_ref_ptr = alloc[Float32](c_size)

    var a_layout = row_major(Coord(m, k))
    var b_layout = row_major(
        Coord(
            Idx[NType.static_value if transpose_b else KType.static_value],
            Idx[KType.static_value if transpose_b else NType.static_value],
        )
    )
    var c_layout = row_major(Coord(m, n))
    var a_scales_layout = row_major(Coord(Idx[1], m))
    var b_scales_layout = row_major(
        Coord(
            Idx[NType.static_value if transpose_b else 1],
            Idx[1 if transpose_b else NType.static_value],
        )
    )

    var a_host = TileTensor(a_host_ptr, a_layout)
    var b_host = TileTensor(b_host_ptr, b_layout)
    var c_host = TileTensor(c_host_ptr, c_layout)
    var a_scales_host = TileTensor(a_scales_host_ptr, a_scales_layout)
    var b_scales_host = TileTensor(b_scales_host_ptr, b_scales_layout)
    var c_host_ref = TileTensor(c_host_ref_ptr, c_layout)

    # Device allocations
    var a_device = ctx.enqueue_create_buffer[in_dtype](a_size)
    var b_device = ctx.enqueue_create_buffer[in_dtype](b_size)
    var c_device = ctx.enqueue_create_buffer[out_dtype](c_size)
    var a_scales_device = ctx.enqueue_create_buffer[scales_dtype](a_scales_size)
    var b_scales_device = ctx.enqueue_create_buffer[scales_dtype](b_scales_size)
    var c_device_ref = ctx.enqueue_create_buffer[.float32](c_size)

    comptime k_dim = KType.static_value

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)

    var a_tile = TileTensor(a_device, a_layout)
    var b_tile = TileTensor(b_device, b_layout)
    var c_tile = TileTensor(c_device, c_layout)
    var a_scales_tile = TileTensor(a_scales_device, a_scales_layout)
    var b_scales_tile = TileTensor(b_scales_device, b_scales_layout)
    var c_ref_tile = TileTensor(c_device_ref, row_major(Coord(m, n)))

    random(a_host)
    random(b_host)
    random(a_scales_host)
    random(b_scales_host)

    matmul_dynamic_scaled_fp8[
        input_scale_granularity="colwise",
        weight_scale_granularity="rowwise",
        m_scale_granularity=1,
        n_scale_granularity=1,
        k_scale_granularity=k_dim,
        transpose_b=transpose_b,
        target="gpu",
    ](
        c_tile,
        a_tile,
        b_tile,
        a_scales_tile,
        b_scales_tile,
        ctx,
    )
    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.synchronize()

    naive_blockwise_scaled_fp8_matmul[
        BLOCK_DIM=16,
        transpose_b=transpose_b,
        scales_granularity_mnk=Index(1, 1, k_dim),
    ](
        c_ref_tile.to_layout_tensor(),
        a_tile.to_layout_tensor().as_imm(),
        b_tile.to_layout_tensor().as_imm(),
        a_scales_tile.to_layout_tensor().as_imm(),
        b_scales_tile.to_layout_tensor().as_imm(),
        ctx,
    )

    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    comptime assert c_host.flat_rank == 2

    for i in range(Int(m.value())):
        for j in range(Int(n.value())):
            assert_almost_equal(
                c_host[i, j].cast[.float32](),
                c_host_ref[i, j][0],
                msg="At [" + String(i) + ", " + String(j) + "]",
                atol=1.5e-2,
                rtol=1.5e-2,
            )

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    a_scales_host_ptr.free()
    b_scales_host_ptr.free()
    c_host_ref_ptr.free()


def test_matmul_dynamic_scaled_fp8_tensor[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    transpose_b: Bool,
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    """Test tensor-granularity (per-tensor) scaling where a_scales and b_scales
    are both shape [1, 1]."""
    var a_size = Int(m.value()) * Int(k.value())
    var b_size = (
        Int(n.value())
        * Int(k.value()) if transpose_b else Int(k.value())
        * Int(n.value())
    )
    var c_size = Int(m.value()) * Int(n.value())

    # Host allocations
    var a_host_ptr = alloc[Scalar[in_dtype]](a_size)
    var b_host_ptr = alloc[Scalar[in_dtype]](b_size)
    var c_host_ptr = alloc[Scalar[out_dtype]](c_size)
    var a_scales_host_ptr = alloc[Scalar[scales_dtype]](1)
    var b_scales_host_ptr = alloc[Scalar[scales_dtype]](1)
    var c_host_ref_ptr = alloc[Float32](c_size)

    var a_layout = row_major(Coord(m, k))
    var b_layout = row_major(
        Coord(
            Idx[NType.static_value if transpose_b else KType.static_value],
            Idx[KType.static_value if transpose_b else NType.static_value],
        )
    )
    var c_layout = row_major(Coord(m, n))
    var a_scales_layout = row_major(Coord(Idx[1], Idx[1]))
    var b_scales_layout = row_major(Coord(Idx[1], Idx[1]))

    var a_host = TileTensor(a_host_ptr, a_layout)
    var b_host = TileTensor(b_host_ptr, b_layout)
    var c_host = TileTensor(c_host_ptr, c_layout)
    var a_scales_host = TileTensor(a_scales_host_ptr, a_scales_layout)
    var b_scales_host = TileTensor(b_scales_host_ptr, b_scales_layout)
    var c_host_ref = TileTensor(c_host_ref_ptr, c_layout)

    # Device allocations
    var a_device = ctx.enqueue_create_buffer[in_dtype](a_size)
    var b_device = ctx.enqueue_create_buffer[in_dtype](b_size)
    var c_device = ctx.enqueue_create_buffer[out_dtype](c_size)
    var a_scales_device = ctx.enqueue_create_buffer[scales_dtype](1)
    var b_scales_device = ctx.enqueue_create_buffer[scales_dtype](1)
    var c_device_ref = ctx.enqueue_create_buffer[.float32](c_size)

    comptime k_dim = KType.static_value

    random(a_host)
    random(b_host)
    random(a_scales_host)
    random(b_scales_host)

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)

    var a_tile = TileTensor(a_device, a_layout)
    var b_tile = TileTensor(b_device, b_layout)
    var c_tile = TileTensor(c_device, c_layout)
    var a_scales_tile = TileTensor(a_scales_device, a_scales_layout)
    var b_scales_tile = TileTensor(b_scales_device, b_scales_layout)
    var c_ref_tile = TileTensor(c_device_ref, row_major(Coord(m, n)))

    matmul_dynamic_scaled_fp8[
        input_scale_granularity="tensor",
        weight_scale_granularity="tensor",
        m_scale_granularity=1,
        n_scale_granularity=1,
        k_scale_granularity=k_dim,
        transpose_b=transpose_b,
        target="gpu",
    ](
        c_tile,
        a_tile,
        b_tile,
        a_scales_tile,
        b_scales_tile,
        ctx,
    )
    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.synchronize()

    # Build colwise/rowwise reference scales by broadcasting the tensor scalar.
    # a_ref_scales shape [1, M], b_ref_scales shape [N, 1] (when transpose_b).
    var a_ref_scales_size = 1 * Int(m.value())
    var b_ref_scales_size = (
        Int(n.value()) * 1 if transpose_b else 1 * Int(n.value())
    )
    var a_ref_scales_host_ptr = alloc[Scalar[scales_dtype]](a_ref_scales_size)
    var b_ref_scales_host_ptr = alloc[Scalar[scales_dtype]](b_ref_scales_size)

    var a_scalar = a_scales_host_ptr[0]
    var b_scalar = b_scales_host_ptr[0]
    for i in range(a_ref_scales_size):
        a_ref_scales_host_ptr[i] = a_scalar
    for i in range(b_ref_scales_size):
        b_ref_scales_host_ptr[i] = b_scalar

    var a_ref_scales_layout = row_major(Coord(Idx[1], m))
    var b_ref_scales_layout = row_major(
        Coord(
            Idx[NType.static_value if transpose_b else 1],
            Idx[1 if transpose_b else NType.static_value],
        )
    )

    var a_ref_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        a_ref_scales_size
    )
    var b_ref_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        b_ref_scales_size
    )
    ctx.enqueue_copy(a_ref_scales_device, a_ref_scales_host_ptr)
    ctx.enqueue_copy(b_ref_scales_device, b_ref_scales_host_ptr)

    var a_ref_scales_tile = TileTensor(a_ref_scales_device, a_ref_scales_layout)
    var b_ref_scales_tile = TileTensor(b_ref_scales_device, b_ref_scales_layout)

    naive_blockwise_scaled_fp8_matmul[
        BLOCK_DIM=16,
        transpose_b=transpose_b,
        scales_granularity_mnk=Index(1, 1, k_dim),
    ](
        c_ref_tile.to_layout_tensor(),
        a_tile.to_layout_tensor().as_imm(),
        b_tile.to_layout_tensor().as_imm(),
        a_ref_scales_tile.to_layout_tensor().as_imm(),
        b_ref_scales_tile.to_layout_tensor().as_imm(),
        ctx,
    )

    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    comptime assert c_host.flat_rank == 2

    for i in range(Int(m.value())):
        for j in range(Int(n.value())):
            assert_almost_equal(
                c_host[i, j].cast[.float32](),
                c_host_ref[i, j][0],
                msg="At [" + String(i) + ", " + String(j) + "]",
                atol=1.5e-2,
                rtol=1.5e-2,
            )

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    a_scales_host_ptr.free()
    b_scales_host_ptr.free()
    c_host_ref_ptr.free()
    a_ref_scales_host_ptr.free()
    b_ref_scales_host_ptr.free()


def main() raises:
    with DeviceContext() as ctx:
        test_matmul_dynamic_scaled_fp8[
            in_dtype=DType.float8_e4m3fn,
            out_dtype=DType.bfloat16,
            scales_dtype=DType.bfloat16,
            transpose_b=True,
        ](ctx, Int(17), Idx[256 + 256], Idx[256])

        test_matmul_dynamic_scaled_fp8[
            in_dtype=DType.float8_e4m3fn,
            out_dtype=DType.bfloat16,
            scales_dtype=DType.bfloat16,
            transpose_b=True,
        ](ctx, Int(124), Idx[512], Idx[512])

        # these tests are guaranteed to hit a mojo fp8 kernel in the dispatch table.
        # if the fp8 kernel is not registered, these tests will fail.
        test_matmul_dynamic_scaled_fp8[
            in_dtype=DType.float8_e4m3fn,
            out_dtype=DType.bfloat16,
            scales_dtype=DType.bfloat16,
            transpose_b=True,
        ](ctx, Int(3000), Idx[5376], Idx[4096])

        test_matmul_dynamic_scaled_fp8[
            in_dtype=DType.float8_e4m3fn,
            out_dtype=DType.bfloat16,
            scales_dtype=DType.bfloat16,
            transpose_b=True,
        ](ctx, Int(224), Idx[43008], Idx[5376])

        # Tensor-granularity (per-tensor) scaling tests
        test_matmul_dynamic_scaled_fp8_tensor[
            in_dtype=DType.float8_e4m3fn,
            out_dtype=DType.bfloat16,
            scales_dtype=DType.bfloat16,
            transpose_b=True,
        ](ctx, Int(17), Idx[512], Idx[256])

        test_matmul_dynamic_scaled_fp8_tensor[
            in_dtype=DType.float8_e4m3fn,
            out_dtype=DType.bfloat16,
            scales_dtype=DType.bfloat16,
            transpose_b=True,
        ](ctx, Int(124), Idx[512], Idx[512])
