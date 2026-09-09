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

from std.math import ceildiv, isclose
from std.random import random_float64

from max.gpu.host import DeviceContext
from max.gpu.host.info import A100
from layout import Coord, Idx, TileTensor, row_major
from linalg.bmm import _batched_matmul_gpu
from linalg.matmul.gpu import _matmul_gpu, matmul_kernel_naive, multistage_gemm
from linalg.utils_gpu import MatmulConfig, MatmulKernels, select_config
from std.testing import assert_almost_equal

from std.utils import IndexList


def run_matmul_naive(ctx: DeviceContext, M: Int, N: Int, K: Int) raises:
    print("== run_matmul naive kernel")

    var a_host = alloc[BFloat16](M * K)
    var b_host = alloc[BFloat16](K * N)
    var c_host = alloc[BFloat16](M * N)
    var a_host_n = alloc[Float32](M * K)
    var b_host_n = alloc[Float32](K * N)
    var c_host_n = alloc[Float32](M * N)

    var rand_min = -1.0
    var rand_max = 1.0

    for i in range(M * K):
        var val = random_float64(rand_min, rand_max).cast[.float32]()
        a_host[i] = val.cast[.bfloat16]()
        a_host_n[i] = a_host[i].cast[.float32]()

    for i in range(K * N):
        var val = random_float64(rand_min, rand_max).cast[.float32]()
        b_host[i] = val.cast[.bfloat16]()
        b_host_n[i] = b_host[i].cast[.float32]()

    for i in range(M * N):
        c_host[i] = 0
        c_host_n[i] = 0

    var a_device = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var b_device = ctx.enqueue_create_buffer[.bfloat16](K * N)
    var c_device = ctx.enqueue_create_buffer[.bfloat16](M * N)
    var a_device_n = ctx.enqueue_create_buffer[.float32](M * K)
    var b_device_n = ctx.enqueue_create_buffer[.float32](K * N)
    var c_device_n = ctx.enqueue_create_buffer[.float32](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    comptime BLOCK_DIM = 16

    # Create TileTensors for bf16 kernel.
    # a/b are constructed as immutable to match the ImmutAnyOrigin
    # parameters that matmul_kernel_naive expects.

    var c_tt_bf16 = TileTensor(
        c_device,
        row_major(Coord(M, N)),
    )
    var a_tt_bf16 = TileTensor(
        ImmPointer[BFloat16, ImmutAnyOrigin](
            unsafe_from_address=Int(a_device.unsafe_ptr())
        ),
        row_major(Coord(M, K)),
    )
    var b_tt_bf16 = TileTensor(
        ImmPointer[BFloat16, ImmutAnyOrigin](
            unsafe_from_address=Int(b_device.unsafe_ptr())
        ),
        row_major(Coord(K, N)),
    )

    @always_inline
    @__parameter
    def run_func_bf16() raises:
        comptime kernel = matmul_kernel_naive[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            type_of(c_tt_bf16).LayoutType,
            type_of(a_tt_bf16).LayoutType,
            type_of(b_tt_bf16).LayoutType,
            BLOCK_DIM,
        ]
        ctx.enqueue_function[kernel](
            c_tt_bf16,
            a_tt_bf16,
            b_tt_bf16,
            Int32(M),
            Int32(N),
            Int32(K),
            grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )

    run_func_bf16()

    ctx.enqueue_copy(c_host, c_device)

    # running naive
    ctx.enqueue_copy(a_device_n, a_host_n)
    ctx.enqueue_copy(b_device_n, b_host_n)

    # Create TileTensors for fp32 kernel.
    var c_tt_fp32 = TileTensor(
        c_device_n,
        row_major(Coord(M, N)),
    )
    var a_tt_fp32 = TileTensor(
        ImmPointer[Float32, ImmutAnyOrigin](
            unsafe_from_address=Int(a_device_n.unsafe_ptr())
        ),
        row_major(Coord(M, K)),
    )
    var b_tt_fp32 = TileTensor(
        ImmPointer[Float32, ImmutAnyOrigin](
            unsafe_from_address=Int(b_device_n.unsafe_ptr())
        ),
        row_major(Coord(K, N)),
    )

    @always_inline
    @__parameter
    def run_func_fp32() raises:
        comptime kernel = matmul_kernel_naive[
            .float32,
            .float32,
            .float32,
            type_of(c_tt_fp32).LayoutType,
            type_of(a_tt_fp32).LayoutType,
            type_of(b_tt_fp32).LayoutType,
            BLOCK_DIM,
        ]
        ctx.enqueue_function[kernel](
            c_tt_fp32,
            a_tt_fp32,
            b_tt_fp32,
            Int32(M),
            Int32(N),
            Int32(K),
            grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )

    run_func_fp32()

    ctx.enqueue_copy(c_host_n, c_device_n)
    ctx.synchronize()

    for i in range(M * N):
        var out_val = c_host[i]
        var out_ref = c_host_n[i].cast[.bfloat16]()
        assert_almost_equal(out_val, out_ref)

    _ = a_device
    _ = b_device
    _ = c_device

    _ = a_device_n
    _ = b_device_n
    _ = c_device_n

    _ = a_host
    _ = b_host
    _ = c_host

    _ = a_host_n
    _ = b_host_n
    _ = c_host_n


def run_matmul[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
](
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
    atol: Float64 = 1.0,
    rng_width: Float64 = Float64(100.0),
    debug: Bool = True,
) raises:
    print("== run_matmul kernel => ", dtype, M, N, K)

    var a_host = alloc[Scalar[dtype]](M * K)
    var b_host = alloc[Scalar[dtype]](K * N)
    var c_host = alloc[Scalar[dtype]](M * N)
    var a_host_n = alloc[Scalar[dtype]](M * K)
    var b_host_n = alloc[Scalar[dtype]](K * N)
    var c_host_n = alloc[Scalar[dtype]](M * N)

    var rand_min = -1 * rng_width
    var rand_max = rng_width

    for i in range(M * K):
        var val = random_float64(rand_min, rand_max).cast[.float32]()
        a_host[i] = val.cast[dtype]()
        a_host_n[i] = a_host[i]

    for i in range(K * N):
        var val = random_float64(rand_min, rand_max).cast[.float32]()
        b_host[i] = val.cast[dtype]()
        b_host_n[i] = b_host[i]

    for i in range(M * N):
        var val = Float32(0)
        c_host[i] = val.cast[dtype]()
        c_host_n[i] = c_host[i]

    var a_device = ctx.enqueue_create_buffer[dtype](M * K)
    var b_device = ctx.enqueue_create_buffer[dtype](K * N)
    var c_device = ctx.enqueue_create_buffer[dtype](M * N)
    var a_tensor = TileTensor(a_device, row_major(Coord(Idx[M], Idx[K])))
    var b_tensor = TileTensor(b_device, row_major(Coord(Idx[K], Idx[N])))
    var c_tensor = TileTensor(c_device, row_major(Coord(Idx[M], Idx[N])))

    var a_device_n = ctx.enqueue_create_buffer[dtype](M * K)
    var b_device_n = ctx.enqueue_create_buffer[dtype](K * N)
    var c_device_n = ctx.enqueue_create_buffer[dtype](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    _matmul_gpu(c_tensor, a_tensor.as_immut(), b_tensor.as_immut(), ctx)
    ctx.enqueue_copy(c_host, c_device)

    # running naive
    ctx.enqueue_copy(a_device_n, a_host_n)
    ctx.enqueue_copy(b_device_n, b_host_n)

    comptime BLOCK_DIM = 16

    # Create TileTensors for naive kernel.
    # a/b are constructed as immutable to match the ImmutAnyOrigin
    # parameters that matmul_kernel_naive expects.

    var c_tt = TileTensor(
        c_device_n,
        row_major(Coord(M, N)),
    )
    var a_tt = TileTensor(
        ImmPointer[Scalar[dtype], ImmutAnyOrigin](
            unsafe_from_address=Int(a_device_n.unsafe_ptr())
        ),
        row_major(Coord(M, K)),
    )
    var b_tt = TileTensor(
        ImmPointer[Scalar[dtype], ImmutAnyOrigin](
            unsafe_from_address=Int(b_device_n.unsafe_ptr())
        ),
        row_major(Coord(K, N)),
    )

    @always_inline
    @__parameter
    def run_func_naive() raises:
        comptime kernel = matmul_kernel_naive[
            dtype,
            dtype,
            dtype,
            type_of(c_tt).LayoutType,
            type_of(a_tt).LayoutType,
            type_of(b_tt).LayoutType,
            BLOCK_DIM,
        ]
        ctx.enqueue_function[kernel](
            c_tt,
            a_tt,
            b_tt,
            Int32(M),
            Int32(N),
            Int32(K),
            grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )

    run_func_naive()

    ctx.enqueue_copy(c_host_n, c_device_n)
    ctx.synchronize()

    for i in range(M * N):
        var out_val = c_host[i]
        var out_ref = c_host_n[i]
        if debug:
            if not isclose(out_val, out_ref, rtol=rtol, atol=atol):
                print(i, out_val, out_ref)
        assert_almost_equal(out_val, out_ref, rtol=rtol, atol=atol)

    _ = a_device
    _ = b_device
    _ = c_device

    _ = a_device_n
    _ = b_device_n
    _ = c_device_n

    _ = a_host
    _ = b_host
    _ = c_host

    _ = a_host_n
    _ = b_host_n
    _ = c_host_n


def run_matmul_split_k[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
    config: MatmulConfig[dtype, dtype, dtype, False],
](
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
    atol: Float64 = 1.0,
    rng_width: Float64 = Float64(100.0),
    debug: Bool = True,
) raises:
    print(
        "== run_matmul kernel split_k => ",
        String(dtype),
        M,
        N,
        K,
    )

    var a_host = alloc[Scalar[dtype]](M * K)
    var b_host = alloc[Scalar[dtype]](K * N)
    var c_host = alloc[Scalar[dtype]](M * N)
    var c_host_n = alloc[Scalar[dtype]](M * N)

    var rand_min = -1 * rng_width
    var rand_max = rng_width

    for i in range(M * K):
        var val = random_float64(rand_min, rand_max).cast[.float32]()
        a_host[i] = val.cast[dtype]()

    for i in range(K * N):
        var val = random_float64(rand_min, rand_max).cast[.float32]()
        b_host[i] = val.cast[dtype]()

    for i in range(M * N):
        var val = Float32(0)
        c_host[i] = val.cast[dtype]()
        c_host_n[i] = c_host[i]

    var a_device = ctx.enqueue_create_buffer[dtype](M * K)
    var b_device = ctx.enqueue_create_buffer[dtype](K * N)
    var c_device = ctx.enqueue_create_buffer[dtype](M * N)
    var a_tensor = TileTensor(a_device, row_major(Coord(Idx[M], Idx[K])))
    var b_tensor = TileTensor(b_device, row_major(Coord(Idx[K], Idx[N])))
    var c_tensor = TileTensor(c_device, row_major(Coord(Idx[M], Idx[N])))

    var a_device_n = ctx.enqueue_create_buffer[dtype](M * K)
    var b_device_n = ctx.enqueue_create_buffer[dtype](K * N)
    var c_device_n = ctx.enqueue_create_buffer[dtype](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    var best_config = select_config[dtype, dtype, dtype, False](M, N, K, ctx)

    multistage_gemm[transpose_b=False, config=config](
        c_tensor,
        a_tensor.as_immut(),
        b_tensor.as_immut(),
        best_config,
        ctx,
    )

    ctx.enqueue_copy(c_host, c_device)
    ctx.synchronize()

    # running naive
    ctx.enqueue_copy(a_device_n, a_host)
    ctx.enqueue_copy(b_device_n, b_host)

    comptime BLOCK_DIM = 16

    # Create TileTensors for naive kernel.
    # a/b are constructed as immutable to match the ImmutAnyOrigin
    # parameters that matmul_kernel_naive expects.

    var c_tt = TileTensor(
        c_device_n,
        row_major(Coord(M, N)),
    )
    var a_tt = TileTensor(
        ImmPointer[Scalar[dtype], ImmutAnyOrigin](
            unsafe_from_address=Int(a_device_n.unsafe_ptr())
        ),
        row_major(Coord(M, K)),
    )
    var b_tt = TileTensor(
        ImmPointer[Scalar[dtype], ImmutAnyOrigin](
            unsafe_from_address=Int(b_device_n.unsafe_ptr())
        ),
        row_major(Coord(K, N)),
    )

    comptime kernel = matmul_kernel_naive[
        dtype,
        dtype,
        dtype,
        type_of(c_tt).LayoutType,
        type_of(a_tt).LayoutType,
        type_of(b_tt).LayoutType,
        BLOCK_DIM,
    ]

    ctx.enqueue_function[kernel](
        c_tt,
        a_tt,
        b_tt,
        Int32(M),
        Int32(N),
        Int32(K),
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )

    ctx.enqueue_copy(c_host_n, c_device_n)
    ctx.synchronize()

    for i in range(M * N):
        var out_val = c_host[i]
        var out_ref = c_host_n[i]
        if debug:
            if not isclose(out_val, out_ref, rtol=rtol, atol=atol):
                print(i, out_val, out_ref)
        assert_almost_equal(out_val, out_ref, rtol=rtol, atol=atol)

    _ = a_device
    _ = b_device
    _ = c_device

    _ = a_device_n
    _ = b_device_n
    _ = c_device_n

    _ = a_host
    _ = b_host
    _ = c_host
    _ = c_host_n


def run_matmul_transpose[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
](
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
    atol: Float64 = 1.0,
    rng_width: Float64 = Float64(100.0),
    debug: Bool = True,
) raises:
    print("== run_matmul kernel transpose => ", String(dtype), M, N, K)

    comptime transpose_b = True
    var a_host = alloc[Scalar[dtype]](M * K)
    var b_host = alloc[Scalar[dtype]](K * N)
    var c_host = alloc[Scalar[dtype]](M * N)
    var a_host_n = alloc[Scalar[dtype]](M * K)
    var b_host_n = alloc[Scalar[dtype]](K * N)
    var c_host_n = alloc[Scalar[dtype]](M * N)

    var rand_min = -1 * rng_width
    var rand_max = rng_width

    for i in range(M * K):
        var val = random_float64(rand_min, rand_max).cast[.float32]()
        a_host[i] = val.cast[dtype]()
        a_host_n[i] = a_host[i]

    for i in range(K * N):
        var val = random_float64(rand_min, rand_max).cast[.float32]()
        b_host[i] = val.cast[dtype]()
        b_host_n[i] = b_host[i]

    for i in range(M * N):
        var val = Float32(0)
        c_host[i] = val.cast[dtype]()
        c_host_n[i] = c_host[i]

    var a_device = ctx.enqueue_create_buffer[dtype](M * K)
    var b_device = ctx.enqueue_create_buffer[dtype](N * K)
    var c_device = ctx.enqueue_create_buffer[dtype](M * N)
    var a_tensor = TileTensor(a_device, row_major(Coord(Idx[M], Idx[K])))
    var b_tensor = TileTensor(b_device, row_major(Coord(Idx[N], Idx[K])))
    var c_tensor = TileTensor(c_device, row_major(Coord(Idx[M], Idx[N])))

    var a_device_n = ctx.enqueue_create_buffer[dtype](M * K)
    var b_device_n = ctx.enqueue_create_buffer[dtype](N * K)
    var c_device_n = ctx.enqueue_create_buffer[dtype](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    _matmul_gpu[transpose_b=transpose_b, use_tensor_core=True](
        c_tensor, a_tensor.as_immut(), b_tensor.as_immut(), ctx
    )
    ctx.enqueue_copy(c_host, c_device)

    # running naive
    ctx.enqueue_copy(a_device_n, a_host_n)
    ctx.enqueue_copy(b_device_n, b_host_n)

    comptime BLOCK_DIM = 16

    # Create TileTensors for naive kernel.
    # a/b are constructed as immutable to match the ImmutAnyOrigin
    # parameters that matmul_kernel_naive expects.

    var c_tt = TileTensor(
        c_device_n,
        row_major(Coord(M, N)),
    )
    var a_tt = TileTensor(
        ImmPointer[Scalar[dtype], ImmutAnyOrigin](
            unsafe_from_address=Int(a_device_n.unsafe_ptr())
        ),
        row_major(Coord(M, K)),
    )
    var b_tt = TileTensor(
        ImmPointer[Scalar[dtype], ImmutAnyOrigin](
            unsafe_from_address=Int(b_device_n.unsafe_ptr())
        ),
        row_major(Coord(N, K)),
    )

    @always_inline
    @__parameter
    def run_func_naive() raises:
        comptime kernel = matmul_kernel_naive[
            dtype,
            dtype,
            dtype,
            type_of(c_tt).LayoutType,
            type_of(a_tt).LayoutType,
            type_of(b_tt).LayoutType,
            BLOCK_DIM,
            transpose_b,
        ]
        ctx.enqueue_function[kernel](
            c_tt,
            a_tt,
            b_tt,
            Int32(M),
            Int32(N),
            Int32(K),
            grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )

    run_func_naive()

    ctx.enqueue_copy(c_host_n, c_device_n)
    ctx.synchronize()

    for i in range(M * N):
        var out_val = c_host[i]
        var out_ref = c_host_n[i]
        if debug:
            if not isclose(out_val, out_ref, rtol=rtol, atol=atol):
                print(i, out_val, out_ref)
        assert_almost_equal(out_val, out_ref, rtol=rtol, atol=atol)

    _ = a_device
    _ = b_device
    _ = c_device

    _ = a_device_n
    _ = b_device_n
    _ = c_device_n

    _ = a_host
    _ = b_host
    _ = c_host

    _ = a_host_n
    _ = b_host_n
    _ = c_host_n


def run_batched_matmul(
    ctx: DeviceContext, B: Int, M: Int, N: Int, K: Int
) raises:
    print("== test_batched_matmul")

    var a_host = alloc[BFloat16](B * M * K)
    var b_host = alloc[BFloat16](B * K * N)
    var c_host = alloc[BFloat16](B * M * N)
    var a_host_n = alloc[Float32](B * M * K)
    var b_host_n = alloc[Float32](B * K * N)
    var c_host_n = alloc[Float32](B * M * N)

    var rand_min = -100.0
    var rand_max = 100.0

    for i in range(B * M * K):
        var val = random_float64(rand_min, rand_max)
        a_host[i] = val.cast[.bfloat16]()
        a_host_n[i] = a_host[i].cast[.float32]()

    for i in range(B * K * N):
        var val = random_float64(rand_min, rand_max)
        b_host[i] = val.cast[.bfloat16]()
        b_host_n[i] = b_host[i].cast[.float32]()

    for i in range(B * M * N):
        c_host[i] = 0
        c_host_n[i] = 0

    var a_device = ctx.enqueue_create_buffer[.bfloat16](B * M * K)
    var b_device = ctx.enqueue_create_buffer[.bfloat16](B * K * N)
    var c_device = ctx.enqueue_create_buffer[.bfloat16](B * M * N)
    var a_tensor = TileTensor(
        a_device,
        row_major(Coord(B, M, K)),
    )
    var b_tensor = TileTensor(
        b_device,
        row_major(Coord(B, K, N)),
    )
    var c_tensor = TileTensor(
        c_device,
        row_major(Coord(B, M, N)),
    )

    var a_device_n = ctx.enqueue_create_buffer[.float32](B * M * K)
    var b_device_n = ctx.enqueue_create_buffer[.float32](B * K * N)
    var c_device_n = ctx.enqueue_create_buffer[.float32](B * M * N)
    var a_tensor_n = TileTensor(
        a_device_n,
        row_major(Coord(B, M, K)),
    )
    var b_tensor_n = TileTensor(
        b_device_n,
        row_major(Coord(B, K, N)),
    )

    var c_tensor_n = TileTensor(
        c_device_n,
        row_major(Coord(B, M, N)),
    )

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    @always_inline
    @__copy_capture(c_tensor)
    @__parameter
    def elementwise_epilogue_fn1[
        c_type: DType,
        width: SIMDLength,
        rank: Int,
        *,
        alignment: Int = 1,
    ](idx: IndexList[rank], val: SIMD[c_type, width]) -> None:
        var coord = Coord(idx)
        c_tensor.store(coord, val.cast[c_tensor.dtype]() + 2)

    _batched_matmul_gpu[elementwise_epilogue_fn=elementwise_epilogue_fn1](
        c_tensor, a_tensor.as_immut(), b_tensor.as_immut(), ctx
    )

    ctx.enqueue_copy(c_host, c_device)

    ctx.enqueue_copy(a_device_n, a_host_n)
    ctx.enqueue_copy(b_device_n, b_host_n)

    @always_inline
    @__copy_capture(c_tensor_n)
    @__parameter
    def elementwise_epilogue_fn2[
        c_type: DType,
        width: SIMDLength,
        rank: Int,
        *,
        alignment: Int = 1,
    ](idx: IndexList[rank], val: SIMD[c_type, width]) -> None:
        var coord = Coord(idx)
        c_tensor_n.store(coord, val.cast[c_tensor_n.dtype]() + 2)

    _batched_matmul_gpu[elementwise_epilogue_fn=elementwise_epilogue_fn2](
        c_tensor_n, a_tensor_n.as_immut(), b_tensor_n.as_immut(), ctx
    )

    ctx.enqueue_copy(c_host_n, c_device_n)
    ctx.synchronize()

    for i in range(B * M * N):
        var out_val = c_host[i]
        var out_ref = c_host_n[i].cast[.bfloat16]()
        assert_almost_equal(out_val, out_ref, rtol=1e-02)

    _ = a_device
    _ = b_device
    _ = c_device

    _ = a_device_n
    _ = b_device_n
    _ = c_device_n

    _ = a_host
    _ = b_host
    _ = c_host

    _ = a_host_n
    _ = b_host_n
    _ = c_host_n


def main() raises:
    with DeviceContext() as ctx:
        comptime kernels = MatmulKernels[
            .bfloat16, .bfloat16, .bfloat16, False
        ]()
        comptime config = kernels.ampere_256x128_3 if ctx.default_device_info == A100 else kernels.ampere_128x128_4
        run_matmul_split_k[.bfloat16, 512, 4096, 14336, config](
            ctx, atol=1.0, rng_width=1.0
        )

        run_matmul_split_k[.bfloat16, 128, 128, 4096, kernels.ampere_128x128_4](
            ctx, atol=0.5, rng_width=1.0
        )

        run_matmul_transpose[.bfloat16, 1, 200, 300](
            ctx, atol=0.25, rng_width=1.0
        )
        run_matmul_transpose[.bfloat16, 1, 300, 200](
            ctx, atol=0.25, rng_width=1.0
        )
        run_matmul_transpose[.bfloat16, 1, 5120, 3072](
            ctx, atol=0.25, rng_width=1.0
        )
        run_matmul_transpose[.bfloat16, 1, 12288, 3072](
            ctx, atol=0.5, rng_width=1.0
        )
        run_matmul_transpose[.bfloat16, 1, 5120, 12288](
            ctx, atol=0.5, rng_width=1.0
        )
        run_matmul_transpose[.bfloat16, 1, 131072, 5120](
            ctx, atol=0.5, rng_width=1.0
        )
        run_matmul_transpose[.bfloat16, 1, 3072, 12288](
            ctx, atol=0.5, rng_width=1.0
        )

        run_matmul[.bfloat16, 128, 128, 128](ctx)
        run_matmul[.bfloat16, 32, 32, 32](ctx)
        run_matmul[.bfloat16, 1024, 1, 1024](ctx, atol=0.2, rng_width=1.0)
        run_matmul[.bfloat16, 1, 1024, 1024](ctx)

        # KERN-1807 We need to systematically test the float16 kernels.
        # run_matmul[.float16, 128, 128, 128](ctx, rng_width=10.0)
        # run_matmul[.float16, 32, 32, 32](ctx, rng_width=10.0)
        # run_matmul[.float16, 1024, 1, 1024](ctx, 1e-03, rng_width=10.0)
        # run_matmul[.float16, 1, 1024, 1024](ctx, 1e-01, rng_width=10.0)

        run_batched_matmul(ctx, 1, 32, 32, 32)
        run_batched_matmul(ctx, 3, 32, 32, 32)
