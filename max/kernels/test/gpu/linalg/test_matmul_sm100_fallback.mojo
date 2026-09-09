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

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.memory import alloc

# Additional imports for testing
from internal_utils import assert_almost_equal
from std.random import rand
from layout import (
    TileTensor,
    Coord,
    CoordLike,
    row_major,
    Idx,
)
from linalg.matmul.gpu.sm100_structured.default.matmul import (
    matmul_sm100_fallback,
)
from linalg.utils import elementwise_epilogue_type

from std.utils.index import Index, IndexList


def test_matmul_sm100_fallback[
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
    BK: Int = 64,
    use_epilogue: Bool = False,
](ctx: DeviceContext, m: MType, n: NType, k: KType,) raises:
    var a_shape = row_major(Coord(m, Idx[KType.static_value]))
    var b_shape = row_major(
        Coord(
            Idx[NType.static_value if transpose_b else KType.static_value],
            Idx[KType.static_value if transpose_b else NType.static_value],
        )
    )
    var c_shape = row_major(Coord(m, Idx[NType.static_value]))

    var a_size = Int(m.value()) * Int(k.value())
    var b_size = Int(n.value()) * Int(k.value())
    var c_size = Int(m.value()) * Int(n.value())

    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)

    var a_host = TileTensor(a_host_ptr, a_shape)
    var b_host = TileTensor(b_host_ptr, b_shape)
    var c_host = TileTensor(c_host_ptr, c_shape)
    var c_host_ref = TileTensor(c_host_ref_ptr, c_shape)

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)

    var a_tensor = TileTensor(a_device, a_shape)
    var b_tensor = TileTensor(b_device, b_shape)
    var c_tensor = TileTensor(c_device, c_shape)
    var c_ref_tensor = TileTensor(c_device_ref, c_shape)

    print(
        "umma_shape",
        umma_shape,
        a_type,
        "x",
        b_type,
        "x",
        c_type,
        "transpose_b:",
        transpose_b,
        "use_epilogue:",
        use_epilogue,
        " : PROBLEM SHAPE (M,N,K): (",
        Int(m.value()),
        "x",
        Int(n.value()),
        "x",
        Int(k.value()),
        ") - ",
        "BLOCKS SHAPE (BM,BN,BK): (",
        umma_shape[0],
        "x",
        umma_shape[1],
        "x",
        BK,
        ")",
    )

    var c_tensor_lt = c_tensor.to_layout_tensor()

    @__parameter
    @always_inline
    @__copy_capture(c_tensor_lt)
    def epilogue_fn[
        _dtype: DType,
        width: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> None:
        c_tensor_lt.store[store_alignment=alignment](
            idx, rebind[SIMD[c_type, width]](val)
        )

    # Initialize matmul operands
    rand(a_host._storage, a_host.num_elements())
    rand(b_host._storage, b_host.num_elements())
    _ = c_host.fill(0)
    _ = c_host_ref.fill(0)

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    ctx.enqueue_copy(c_device, c_host_ptr)
    ctx.enqueue_copy(c_device_ref, c_host_ref_ptr)

    comptime block_tile_shape = Index(umma_shape[0], umma_shape[1], BK)

    matmul_sm100_fallback[
        c_type,
        a_type,
        b_type,
        transpose_b=transpose_b,
        umma_shape=umma_shape,
        block_tile_shape=block_tile_shape,
        a_swizzle=swizzle,
        b_swizzle=swizzle,
        elementwise_lambda_fn=Optional[elementwise_epilogue_type](
            epilogue_fn
        ) if use_epilogue else None,
    ](c_tensor, a_tensor, b_tensor, ctx)

    ctx.synchronize()

    comptime assert a_type != .float8_e4m3fn or transpose_b, (
        "Testing is only supported for transposed_b==True when"
        " a_type==float8_e4m3fn. Add the non-transposed case if needed."
    )

    var c_ref_tensor_lt = c_ref_tensor.to_layout_tensor()
    var a_lt = a_tensor.to_layout_tensor()
    var b_lt = b_tensor.to_layout_tensor()

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

    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^

    _ = a_tensor
    _ = b_tensor
    _ = c_tensor


def main() raises:
    with DeviceContext() as ctx:
        comptime for dtype in [DType.float8_e4m3fn, DType.bfloat16]:
            comptime for swizzle in [TensorMapSwizzle.SWIZZLE_128B]:
                comptime MMA_K = 32 if dtype == DType.float8_e4m3fn else 16
                comptime BK = (swizzle.bytes() // size_of[dtype]())

                test_matmul_sm100_fallback[
                    dtype,
                    dtype,
                    .bfloat16,
                    umma_shape=Index(64, 128, MMA_K),
                    swizzle=swizzle,
                    transpose_b=True,
                    BK=BK,
                ](
                    ctx,
                    Int(200),
                    Idx[128],
                    Idx[128],
                )
                test_matmul_sm100_fallback[
                    dtype,
                    dtype,
                    .bfloat16,
                    umma_shape=Index(64, 128, MMA_K),
                    swizzle=swizzle,
                    transpose_b=True,
                    BK=BK,
                    use_epilogue=True,
                ](
                    ctx,
                    Int(128),
                    Idx[128],
                    Idx[128],
                )

                test_matmul_sm100_fallback[
                    dtype,
                    dtype,
                    .bfloat16,
                    umma_shape=Index(64, 128, MMA_K),
                    swizzle=swizzle,
                    transpose_b=True,
                    BK=BK,
                ](
                    ctx,
                    Int(400),
                    Idx[128],
                    Idx[128],
                )

                test_matmul_sm100_fallback[
                    dtype,
                    dtype,
                    .bfloat16,
                    umma_shape=Index(64, 128, MMA_K),
                    swizzle=swizzle,
                    transpose_b=True,
                    BK=BK,
                ](
                    ctx,
                    Int(1024),
                    Idx[2048],
                    Idx[2048],
                )

                comptime BK_list: List[Int] = [BK, BK * 2]

                comptime for _BK in BK_list:
                    test_matmul_sm100_fallback[
                        dtype,
                        dtype,
                        .bfloat16,
                        umma_shape=Index(64, 128, MMA_K),
                        transpose_b=True,
                        BK=_BK,
                    ](
                        ctx,
                        Int(1024),
                        Idx[2048],
                        Idx[2048],
                    )

                    test_matmul_sm100_fallback[
                        dtype,
                        dtype,
                        .bfloat16,
                        umma_shape=Index(64, 128, MMA_K),
                        transpose_b=True,
                        BK=_BK,
                    ](
                        ctx,
                        Idx[1024],
                        Idx[2048],
                        Idx[2048],
                    )

                    test_matmul_sm100_fallback[
                        dtype,
                        dtype,
                        .bfloat16,
                        umma_shape=Index(64, 128, MMA_K),
                        transpose_b=True,
                        BK=_BK,
                    ](
                        ctx,
                        Int(100),
                        Idx[512],
                        Idx[256],
                    )

                    test_matmul_sm100_fallback[
                        dtype,
                        dtype,
                        .bfloat16,
                        umma_shape=Index(64, 128, MMA_K),
                        transpose_b=True,
                        BK=_BK,
                    ](
                        ctx,
                        Int(99),
                        Idx[1024],
                        Idx[1024],
                    )

                    test_matmul_sm100_fallback[
                        dtype,
                        dtype,
                        .bfloat16,
                        umma_shape=Index(64, 128, MMA_K),
                        transpose_b=True,
                        BK=_BK,
                    ](
                        ctx,
                        Int(201),
                        Idx[2048],
                        Idx[256],
                    )
