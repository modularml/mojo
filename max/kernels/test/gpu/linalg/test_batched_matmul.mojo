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


from std.sys import has_nvidia_gpu_accelerator, simd_width_of

import linalg.matmul.vendor.blas as vendor_blas
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext, get_gpu_target
from layout import Coord, Idx, DefaultEngine, TileTensor, row_major
from layout.tile_layout import Layout
from linalg.bmm import _batched_matmul_gpu

from std.random import rand
from std.testing import assert_almost_equal
from std.utils import IndexList

comptime epilogue_func_type = def[
    dtype: DType, width: SIMDLength, *, alignment: Int = 1
](SIMD[dtype, width]) capturing -> SIMD[dtype, width]


@always_inline
@__parameter
def elementwise_epilogue_fn[
    dtype: DType,
    width: SIMDLength,
    *,
    alignment: Int = 1,
](val: SIMD[dtype, width],) -> SIMD[dtype, width]:
    return val + 2


def run_bmm_and_check_result[
    dtype: DType,
    //,
    transpose_b: Bool,
    lambda_fn: Optional[epilogue_func_type] = None,
    check_against_naive_kernel: Bool = False,
](
    a_host: TileTensor[
        mut=True,
        dtype,
        address_space=.GENERIC,
        ...,
        Engine=DefaultEngine[element_width=1],
    ],
    b_host: TileTensor[
        mut=True,
        dtype,
        address_space=.GENERIC,
        ...,
        Engine=DefaultEngine[element_width=1],
    ],
    c_host: TileTensor[
        mut=True,
        dtype,
        address_space=.GENERIC,
        ...,
        Engine=DefaultEngine[element_width=1],
    ],
    c_host_ref: TileTensor[
        mut=True,
        dtype,
        address_space=.GENERIC,
        ...,
        Engine=DefaultEngine[element_width=1],
    ],
    ctx: DeviceContext,
    rtol: Float64 = 1e-3 if dtype == .float32 else 1e-2,
) raises:
    comptime assert c_host.flat_rank == 3, "c_device must have rank 3"
    comptime assert c_host_ref.flat_rank == 3, "c_device_ref must have rank 3"
    var a_size = a_host.num_elements()
    var b_size = b_host.num_elements()
    var c_size = c_host.num_elements()

    # allocate device buffers
    var a_device_buffer = ctx.enqueue_create_buffer[dtype](a_size)
    var b_device_buffer = ctx.enqueue_create_buffer[dtype](b_size)
    var c_device_buffer = ctx.enqueue_create_buffer[dtype](c_size)
    var c_device_ref_buffer = ctx.enqueue_create_buffer[dtype](c_size)

    var a_device = TileTensor[dtype](a_device_buffer, a_host.layout)
    var b_device = TileTensor[dtype](b_device_buffer, b_host.layout)
    var c_device = TileTensor[dtype](c_device_buffer, c_host.layout)
    var c_device_ref = TileTensor[dtype](c_device_ref_buffer, c_host_ref.layout)

    rand(a_host._storage, a_size)
    rand(b_host._storage, b_size)
    c_device_buffer.enqueue_fill(0)
    c_device_ref_buffer.enqueue_fill(0)

    # Copy operands to the Device
    ctx.enqueue_copy(a_device_buffer, a_host._storage)
    ctx.enqueue_copy(b_device_buffer, b_host._storage)

    # Run BMM
    @__parameter
    @always_inline
    @__copy_capture(c_device)
    def epilogue_fn[
        dtype: DType,
        width: SIMDLength,
        rank: Int,
        *,
        alignment: Int = 1,
    ](idx0: IndexList[rank], val: SIMD[dtype, width],) capturing -> None:
        var idx = rebind[IndexList[3]](idx0)
        comptime func = lambda_fn.value()
        var update_val = func(val)
        var coord = Coord((idx[0], idx[1], idx[2]))
        comptime assert c_device.flat_rank >= 3
        c_device.store(coord, update_val.cast[c_device.dtype]())

    comptime if lambda_fn:
        _batched_matmul_gpu[
            transpose_b=transpose_b,
            elementwise_epilogue_fn=epilogue_fn,
        ](c_device, a_device, b_device, ctx)
    else:
        _batched_matmul_gpu[transpose_b=transpose_b](
            c_device, a_device, b_device, ctx
        )

    ctx.synchronize()

    var b = Int(c_device_ref.dim(0))
    var m = Int(c_device_ref.dim(1))
    var n = Int(c_device_ref.dim(2))
    var k = Int(a_device.dim(2))

    # Skip equality check if N or K are 0 (causes error in vendor_blas).
    if n == 0 or k == 0:
        return
    if not has_nvidia_gpu_accelerator() and m == 0:
        # AMD doesn't support matmul with M=0
        return

    comptime if check_against_naive_kernel:
        # erase static dimensions so that the naive kernel can be used
        _batched_matmul_gpu[transpose_b=transpose_b](
            c_device_ref.make_dynamic[.int64](),
            a_device.make_dynamic[.int64](),
            b_device.make_dynamic[.int64](),
            ctx,
        )
    else:
        for i in range(a_host.dim(0)):
            var c_ptr = c_device_ref._storage + i * Scalar[
                a_host.linear_idx_type
            ](c_device_ref.layout.stride[0]().value())
            var a_ptr = a_device._storage + i * Scalar[a_host.linear_idx_type](
                a_device.layout.stride[0]().value()
            )
            var b_ptr = b_device._storage + i * Scalar[a_host.linear_idx_type](
                b_device.layout.stride[0]().value()
            )

            var c_buffer = TileTensor(c_ptr, row_major(m, n))
            var a_buffer = TileTensor(a_ptr, row_major(m, k))
            var b_buffer = TileTensor(
                b_ptr,
                row_major((n if transpose_b else k, k if transpose_b else n)),
            )

            vendor_blas.matmul(
                ctx,
                c_buffer,
                a_buffer,
                b_buffer,
                c_row_major=True,
                transpose_b=transpose_b,
            )

    comptime pack_size = simd_width_of[dtype, target=get_gpu_target()]()

    @always_inline
    def func[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        comptime assert c_device_ref.flat_rank >= 3
        var val = c_device_ref.load[width=simd_width](coord)
        comptime element_lambda = lambda_fn.value()
        var update_val = element_lambda(val)

        c_device_ref.store(
            coord,
            update_val,
        )

    comptime if lambda_fn:
        elementwise[pack_size, target="gpu"](
            func,
            (b, m, n),
            ctx,
        )

    ctx.enqueue_copy(c_host._storage, c_device_buffer)
    ctx.enqueue_copy(c_host_ref._storage, c_device_ref_buffer)
    ctx.synchronize()

    for batch_idx in range(b):
        for m_idx in range(m):
            for n_idx in range(n):
                var expect = c_host_ref[batch_idx, m_idx, n_idx][0]
                var actual = c_host[batch_idx, m_idx, n_idx][0]

                assert_almost_equal(actual, expect, rtol=rtol)


def test_dynamic_shapes[
    dtype: DType,
    /,
    *,
    transpose_b: Bool,
    lambda_fn: Optional[epilogue_func_type] = None,
](
    ctx: DeviceContext,
    b: Int,
    m: Int,
    n: Int,
    k: Int,
    rtol: Float64 = 1e-3 if dtype == .float32 else 1e-2,
) raises:
    # fmt: off
    print(
        "test_dynamic_shapes", b, "x", m, "x", n, "x", k, "transpose_b", transpose_b,
    )
    # fmt: on

    var a_size = b * m * k
    var b_size = b * n * k
    var c_size = b * m * n

    # Host allocations
    var a_host_ptr = ctx.enqueue_create_host_buffer[dtype](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[dtype](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[dtype](c_size)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[dtype](c_size)

    var a_host = TileTensor(a_host_ptr, row_major(b, m, k))
    var c_host = TileTensor(c_host_ptr, row_major(b, m, n))
    var c_host_ref = TileTensor(c_host_ref_ptr, row_major(b, m, n))

    var b_host = TileTensor(
        b_host_ptr, row_major(b, n, k)
    ) if transpose_b else TileTensor(b_host_ptr, row_major(b, k, n))
    run_bmm_and_check_result[transpose_b=transpose_b, lambda_fn=lambda_fn](
        a_host, b_host, c_host, c_host_ref, ctx, rtol
    )


def test_static_NK[
    dtype: DType,
    /,
    *,
    N: Int,
    K: Int,
    transpose_b: Bool,
    lambda_fn: Optional[epilogue_func_type] = None,
](
    ctx: DeviceContext,
    b: Int,
    m: Int,
    rtol: Float64 = 1e-3 if dtype == .float32 else 1e-2,
) raises:
    print(
        "test_static_NK", b, "x", m, "x", N, "x", K, "transpose_b", transpose_b
    )

    var a_size = b * m * K
    var b_size = b * N * K
    var c_size = b * m * N

    # Host allocations
    var a_host_ptr = ctx.enqueue_create_host_buffer[dtype](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[dtype](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[dtype](c_size)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[dtype](c_size)

    var a_host = TileTensor(a_host_ptr, row_major(b, m, Idx[K]))
    var c_host = TileTensor(c_host_ptr, row_major(b, m, Idx[N]))
    var c_host_ref = TileTensor(c_host_ref_ptr, row_major(b, m, Idx[N]))

    comptime if transpose_b:
        var b_host = TileTensor(b_host_ptr, row_major(b, Idx[N], Idx[K]))
        run_bmm_and_check_result[transpose_b=transpose_b, lambda_fn=lambda_fn](
            a_host, b_host, c_host, c_host_ref, ctx, rtol
        )

    else:
        var b_host = TileTensor(b_host_ptr, row_major(b, Idx[K], Idx[N]))
        run_bmm_and_check_result[transpose_b=transpose_b, lambda_fn=lambda_fn](
            a_host, b_host, c_host, c_host_ref, ctx, rtol
        )


def test_non_row_major_layout[
    dtype: DType,
    /,
    *,
    B: Int,
    N: Int,
    K: Int,
    lambda_fn: Optional[epilogue_func_type] = None,
](
    ctx: DeviceContext,
    m: Int,
    rtol: Float64 = 1e-3 if dtype == .float32 else 1e-2,
) raises:
    """
    This function tests bacthed matmul with non-row major inputs.
    For example, A's layout could be (B, m, K):(K, B*K, 1).
    """
    print("test_non_row_major_layout", B, "x", m, "x", N, "x", K)

    var a_size = B * m * K
    var b_size = B * N * K
    var c_size = B * m * N

    # Host allocations
    var a_host_ptr = ctx.enqueue_create_host_buffer[dtype](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[dtype](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[dtype](c_size)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[dtype](c_size)

    var a_layout = Layout((Idx[B], m, Idx[K]), (Idx[K], Idx[B * K], Idx[1]))
    var c_layout = Layout((Idx[B], m, Idx[N]), (Idx[N], Idx[B * N], Idx[1]))

    var a_host = TileTensor(a_host_ptr, a_layout)
    var c_host = TileTensor(c_host_ptr, c_layout)
    var c_host_ref = TileTensor(c_host_ref_ptr, c_layout)

    var b_host = TileTensor(b_host_ptr, row_major(Idx[B], Idx[N], Idx[K]))
    run_bmm_and_check_result[
        transpose_b=True, lambda_fn=lambda_fn, check_against_naive_kernel=True
    ](a_host, b_host, c_host, c_host_ref, ctx, rtol)


def main() raises:
    with DeviceContext() as ctx:
        # Test zero-dimension edge cases
        test_dynamic_shapes[
            .bfloat16,
            transpose_b=False,
        ](ctx, 0, 2, 2, 2)

        # Test non-batch dispatch logic
        test_dynamic_shapes[
            .bfloat16,
            transpose_b=False,
        ](ctx, 1, 2, 2, 2)

        test_dynamic_shapes[
            .bfloat16,
            transpose_b=False,
        ](ctx, 2, 0, 2, 2)

        test_dynamic_shapes[
            .bfloat16,
            transpose_b=False,
        ](ctx, 2, 2, 0, 2)

        test_dynamic_shapes[
            .bfloat16,
            transpose_b=False,
        ](ctx, 2, 2, 2, 0)

        # tests naive kernels
        test_dynamic_shapes[
            .bfloat16,
            transpose_b=False,
        ](ctx, 2, 2, 2, 2)

        test_dynamic_shapes[
            .float32,
            transpose_b=False,
            lambda_fn=elementwise_epilogue_fn,
        ](ctx, 2, 2, 2, 2)

        test_dynamic_shapes[
            .float32,
            transpose_b=False,
            lambda_fn=elementwise_epilogue_fn,
        ](ctx, 64, 256, 512, 128)

        comptime if has_nvidia_gpu_accelerator():
            # NOTE: these tests should be run on a100 and above

            # tests kernels.ampere_128x128_4
            test_static_NK[
                .bfloat16,
                transpose_b=True,
                lambda_fn=elementwise_epilogue_fn,
                N=Int(128256),
                K=Int(4096),
            ](ctx, 2, 600)

            # tests kernels.ampere_256x64_4
            test_static_NK[
                .bfloat16,
                transpose_b=True,
                lambda_fn=elementwise_epilogue_fn,
                N=Int(3072),
                K=Int(12288),
            ](ctx, 4, 14, rtol=2e-2)

            # tests DeepSeek Case
            test_static_NK[
                .bfloat16,
                transpose_b=True,
                lambda_fn=elementwise_epilogue_fn,
                N=Int(128),
                K=Int(512),
            ](ctx, 128, 256)

            test_static_NK[
                .bfloat16,
                transpose_b=True,
                lambda_fn=elementwise_epilogue_fn,
                N=Int(512),
                K=Int(128),
            ](ctx, 128, 256)

            test_static_NK[
                .bfloat16,
                transpose_b=False,
                lambda_fn=elementwise_epilogue_fn,
                N=Int(3072),
                K=Int(12288),
            ](ctx, 4, 14, rtol=2e-2)

            # test non-row major layout
            test_non_row_major_layout[
                .bfloat16,
                B=Int(128),
                N=Int(128),
                K=Int(512),
            ](ctx, 22)

            test_non_row_major_layout[
                .bfloat16,
                B=Int(128),
                N=Int(512),
                K=Int(128),
            ](ctx, 22)
