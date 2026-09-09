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

from std.math import ceildiv
from std.random import random_si64

from max.gpu import WARP_SIZE, block_idx
from max.gpu.host import DeviceContext
from max.gpu.compute.mma import mma
from max.gpu.compute.mma_util import (
    load_matrix_a,
    load_matrix_b,
    store_matrix_d,
)
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu import matmul_kernel_naive
from std.testing import assert_false

from std.utils.numerics import isnan


# TF32 Tensor core Matmul with shape m16n8k8
def mma_kernel_fp32_tf32(
    c_ptr: MutPointer[Float32, MutAnyOrigin],
    a_ptr: ImmPointer[Float32, ImmutAnyOrigin],
    b_ptr: ImmPointer[Float32, ImmutAnyOrigin],
    m_dev: Int32,
    n_dev: Int32,
    k_dev: Int32,
):
    var m = Int(m_dev)
    var n = Int(n_dev)
    var k = Int(k_dev)
    comptime mma_m = 16
    comptime mma_n = 8
    comptime mma_k = 8

    var d_reg = SIMD[.float32, 4](0)
    var tile_loops = k // mma_k

    for i in range(tile_loops):
        var a_tile_row = block_idx.x * mma_m
        var a_tile_col = i * mma_k
        var b_tile_row = i * mma_k
        var b_tile_col = block_idx.y * mma_n

        var a_reg = load_matrix_a[mma_m, mma_n, mma_k](
            a_ptr, a_tile_row, a_tile_col, k
        )
        var b_reg = load_matrix_b[mma_m, mma_n, mma_k](
            b_ptr, b_tile_row, b_tile_col, n
        )

        # Perform mma (d = a * b + d)
        mma(d_reg, a_reg, b_reg, d_reg)

    var c_tile_row = block_idx.x * mma_m
    var c_tile_col = block_idx.y * mma_n
    store_matrix_d[mma_m, mma_n, mma_k](c_ptr, d_reg, c_tile_row, c_tile_col, n)


# FP32-BF16 (mixed precision) Tensor core Matmul with shape m16n8k8
def mma_kernel_fp32_bf16(
    c_ptr: MutPointer[Float32, MutAnyOrigin],
    a_ptr: ImmPointer[BFloat16, ImmutAnyOrigin],
    b_ptr: ImmPointer[BFloat16, ImmutAnyOrigin],
    m_dev: Int32,
    n_dev: Int32,
    k_dev: Int32,
):
    var m = Int(m_dev)
    var n = Int(n_dev)
    var k = Int(k_dev)
    comptime mma_m = 16
    comptime mma_n = 8
    comptime mma_k = 8

    var d_reg = SIMD[.float32, 4](0)
    var tile_loops = k // mma_k

    for i in range(tile_loops):
        var a_tile_row = block_idx.x * mma_m
        var a_tile_col = i * mma_k
        var b_tile_row = i * mma_k
        var b_tile_col = block_idx.y * mma_n

        var a_reg = load_matrix_a[mma_m, mma_n, mma_k](
            a_ptr, a_tile_row, a_tile_col, k
        )
        var b_reg = load_matrix_b[mma_m, mma_n, mma_k](
            b_ptr, b_tile_row, b_tile_col, n
        )

        # Perform mma (d = a * b + d)
        mma(d_reg, a_reg, b_reg, d_reg)

    var c_tile_row = block_idx.x * mma_m
    var c_tile_col = block_idx.y * mma_n
    store_matrix_d[mma_m, mma_n, mma_k](c_ptr, d_reg, c_tile_row, c_tile_col, n)


# FP32-BF16 (mixed precision) Tensor core Matmul with shape m16n8k16
def mma_kernel_fp32_bf16_2(
    c_ptr: MutPointer[Float32, MutAnyOrigin],
    a_ptr: ImmPointer[BFloat16, ImmutAnyOrigin],
    b_ptr: ImmPointer[BFloat16, ImmutAnyOrigin],
    m_dev: Int32,
    n_dev: Int32,
    k_dev: Int32,
):
    var m = Int(m_dev)
    var n = Int(n_dev)
    var k = Int(k_dev)
    comptime mma_m = 16
    comptime mma_n = 8
    comptime mma_k = 16

    var d_reg = SIMD[.float32, 4](0)
    var tile_loops = k // mma_k

    for i in range(tile_loops):
        var a_tile_row = block_idx.x * mma_m
        var a_tile_col = i * mma_k
        var b_tile_row = i * mma_k
        var b_tile_col = block_idx.y * mma_n

        var a_reg = load_matrix_a[mma_m, mma_n, mma_k](
            a_ptr, a_tile_row, a_tile_col, k
        )
        var b_reg = load_matrix_b[mma_m, mma_n, mma_k](
            b_ptr, b_tile_row, b_tile_col, n
        )

        # Perform mma (d = a * b + d)
        mma(d_reg, a_reg, b_reg, d_reg)

    var c_tile_row = block_idx.x * mma_m
    var c_tile_col = block_idx.y * mma_n
    store_matrix_d[mma_m, mma_n, mma_k](c_ptr, d_reg, c_tile_row, c_tile_col, n)


# FP32-FP16 (mixed precision) Tensor core Matmul with shape m16n8k8
def mma_kernel_fp32_fp16(
    c_ptr: MutPointer[Float32, MutAnyOrigin],
    a_ptr: ImmPointer[Float16, ImmutAnyOrigin],
    b_ptr: ImmPointer[Float16, ImmutAnyOrigin],
    m_dev: Int32,
    n_dev: Int32,
    k_dev: Int32,
):
    var m = Int(m_dev)
    var n = Int(n_dev)
    var k = Int(k_dev)
    comptime mma_m = 16
    comptime mma_n = 8
    comptime mma_k = 8

    var d_reg = SIMD[.float32, 4](0)
    var tile_loops = k // mma_k

    for i in range(tile_loops):
        var a_tile_row = block_idx.x * mma_m
        var a_tile_col = i * mma_k
        var b_tile_row = i * mma_k
        var b_tile_col = block_idx.y * mma_n

        var a_reg = load_matrix_a[mma_m, mma_n, mma_k](
            a_ptr, a_tile_row, a_tile_col, k
        )
        var b_reg = load_matrix_b[mma_m, mma_n, mma_k](
            b_ptr, b_tile_row, b_tile_col, n
        )

        # Perform mma (d = a * b + d)
        mma(d_reg, a_reg, b_reg, d_reg)

    var c_tile_row = block_idx.x * mma_m
    var c_tile_col = block_idx.y * mma_n
    store_matrix_d[mma_m, mma_n, mma_k](c_ptr, d_reg, c_tile_row, c_tile_col, n)


# FP16 Tensor core Matmul with shape m16n8k8
def mma_kernel_fp16_fp16(
    c_ptr: MutPointer[Float16, MutAnyOrigin],
    a_ptr: ImmPointer[Float16, ImmutAnyOrigin],
    b_ptr: ImmPointer[Float16, ImmutAnyOrigin],
    m_dev: Int32,
    n_dev: Int32,
    k_dev: Int32,
):
    var m = Int(m_dev)
    var n = Int(n_dev)
    var k = Int(k_dev)
    comptime mma_m = 16
    comptime mma_n = 8
    comptime mma_k = 8

    var d_reg = SIMD[.float16, 4](0)
    var tile_loops = k // mma_k

    for i in range(tile_loops):
        var a_tile_row = block_idx.x * mma_m
        var a_tile_col = i * mma_k
        var b_tile_row = i * mma_k
        var b_tile_col = block_idx.y * mma_n

        var a_reg = load_matrix_a[mma_m, mma_n, mma_k](
            a_ptr, a_tile_row, a_tile_col, k
        )
        var b_reg = load_matrix_b[mma_m, mma_n, mma_k](
            b_ptr, b_tile_row, b_tile_col, n
        )

        # Perform mma (d = a * b + d)
        mma(d_reg, a_reg, b_reg, d_reg)

    var c_tile_row = block_idx.x * mma_m
    var c_tile_col = block_idx.y * mma_n
    store_matrix_d[mma_m, mma_n, mma_k](c_ptr, d_reg, c_tile_row, c_tile_col, n)


def run_mma_fp32_tf32(
    M: Int,
    N: Int,
    K: Int,
    rand_min: Int64,
    rand_max: Int64,
    iterations: Int,
    errorTolerance: Float32,
    ctx: DeviceContext,
) raises:
    print("== run_matmul fp32.tf32 tensor core kernel")

    var a_host = alloc[Float32](M * K)
    var b_host = alloc[Float32](K * N)
    var c_host = alloc[Float32](M * N)
    var a_host_ref = alloc[Float32](M * K)
    var b_host_ref = alloc[Float32](K * N)
    var c_host_ref = alloc[Float32](M * N)

    for i in range(M * K):
        var val = random_si64(rand_min, rand_max)
        a_host[i] = val.cast[.float32]()
        a_host_ref[i] = val.cast[.float32]()

    for i in range(K * N):
        var val = random_si64(rand_min, rand_max)
        b_host[i] = val.cast[.float32]()
        b_host_ref[i] = val.cast[.float32]()

    for i in range(M * N):
        c_host[i] = 0
        c_host_ref[i] = 0

    var a_device = ctx.enqueue_create_buffer[.float32](M * K)
    var b_device = ctx.enqueue_create_buffer[.float32](K * N)
    var c_device = ctx.enqueue_create_buffer[.float32](M * N)
    var a_device_ref = ctx.enqueue_create_buffer[.float32](M * K)
    var b_device_ref = ctx.enqueue_create_buffer[.float32](K * N)
    var c_device_ref = ctx.enqueue_create_buffer[.float32](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    comptime WARP_PER_BLOCK = 1
    comptime MMA_M = 16
    comptime MMA_N = 8
    comptime MMA_K = 8

    @always_inline
    def run_func_mma(ctx: DeviceContext) raises {imm}:
        comptime kernel = mma_kernel_fp32_tf32
        ctx.enqueue_function[kernel](
            c_device,
            a_device,
            b_device,
            Int32(M),
            Int32(N),
            Int32(K),
            grid_dim=(ceildiv(M, MMA_M), ceildiv(N, MMA_N)),
            block_dim=WARP_PER_BLOCK * WARP_SIZE,
        )

    var nstime = ctx.execution_time(run_func_mma, iterations)
    var flops = 2 * M * N * K
    var sectime = Float64(nstime) / Float64(iterations) / 1000000000
    print("Basic Tensor core kernel:")
    print(sectime, "sec")
    print(Float64(flops) * 1e-9 / sectime, " GFLOPS")
    print()

    ctx.enqueue_copy(c_host, c_device)

    # Run naive matmul.
    ctx.enqueue_copy(a_device_ref, a_host_ref)
    ctx.enqueue_copy(b_device_ref, b_host_ref)

    comptime BLOCK_DIM = 16

    # Create TileTensors for the naive kernel.
    # a/b are constructed as immutable to match the ImmutAnyOrigin
    # parameters that matmul_kernel_naive expects (enqueue_function
    # requires exact type matches).
    var c_tt = TileTensor(
        c_device_ref,
        row_major(Coord(M, N)),
    )
    var a_tt = TileTensor(
        ImmPointer[Float32, ImmutAnyOrigin](
            unsafe_from_address=Int(a_device_ref.unsafe_ptr())
        ),
        row_major(Coord(M, K)),
    )
    var b_tt = TileTensor(
        ImmPointer[Float32, ImmutAnyOrigin](
            unsafe_from_address=Int(b_device_ref.unsafe_ptr())
        ),
        row_major(Coord(K, N)),
    )

    @always_inline
    def run_func_naive(ctx: DeviceContext) raises {imm}:
        comptime kernel = matmul_kernel_naive[
            DType.float32,
            DType.float32,
            DType.float32,
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

    nstime = ctx.execution_time(run_func_naive, iterations)
    var sectime2 = Float64(nstime) / Float64(iterations) / 1000000000
    print("Naive matmul kernel:")
    print(sectime2, "sec")
    print(Float64(flops) * 1e-9 / sectime2, " GFLOPS")
    print()

    ctx.enqueue_copy(c_host_ref, c_device_ref)

    ctx.synchronize()

    # Check correctness.
    var failed = False
    for i in range(M * N):
        var outVal = c_host[i]
        var outRef = c_host_ref[i]
        var relDiff = (max(outVal, outRef) / min(outVal, outRef)) - 1.0
        if (relDiff > errorTolerance) or isnan(outVal) or isnan(outRef):
            failed = True
            print(i, outVal, outRef)

    if not failed:
        print("Success 🎉: Results match.")
        print(
            "Performance basic tensor core matmul vs. naive matmul: ",
            sectime2 / sectime,
            "x",
        )
    else:
        print("Failed ❌: results mismatch.")

    assert_false(failed)

    _ = a_device
    _ = b_device
    _ = c_device
    _ = a_device_ref
    _ = b_device_ref
    _ = c_device_ref

    _ = a_host
    _ = b_host
    _ = c_host
    _ = a_host_ref
    _ = a_host_ref
    _ = c_host_ref


def run_mma_fp32_bf16(
    M: Int,
    N: Int,
    K: Int,
    rand_min: Int64,
    rand_max: Int64,
    iterations: Int,
    errorTolerance: Float32,
    ctx: DeviceContext,
) raises:
    print("== run_matmul fp32.bf16 1688 tensor core kernel")

    var a_host = alloc[BFloat16](M * K)
    var b_host = alloc[BFloat16](K * N)
    var c_host = alloc[Float32](M * N)
    var a_host_ref = alloc[Float32](M * K)
    var b_host_ref = alloc[Float32](K * N)
    var c_host_ref = alloc[Float32](M * N)

    for i in range(M * K):
        var val = random_si64(rand_min, rand_max)
        a_host[i] = val.cast[.bfloat16]()
        a_host_ref[i] = val.cast[.float32]()

    for i in range(K * N):
        var val = random_si64(rand_min, rand_max)
        b_host[i] = val.cast[.bfloat16]()
        b_host_ref[i] = val.cast[.float32]()

    for i in range(M * N):
        c_host[i] = 0
        c_host_ref[i] = 0

    var a_device = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var b_device = ctx.enqueue_create_buffer[.bfloat16](K * N)
    var c_device = ctx.enqueue_create_buffer[.float32](M * N)
    var a_device_ref = ctx.enqueue_create_buffer[.float32](M * K)
    var b_device_ref = ctx.enqueue_create_buffer[.float32](K * N)
    var c_device_ref = ctx.enqueue_create_buffer[.float32](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    comptime WARP_PER_BLOCK = 1
    comptime MMA_M = 16
    comptime MMA_N = 8
    comptime MMA_K = 8

    @always_inline
    def run_func_mma(ctx: DeviceContext) raises {imm}:
        comptime kernel = mma_kernel_fp32_bf16
        ctx.enqueue_function[kernel](
            c_device,
            a_device,
            b_device,
            Int32(M),
            Int32(N),
            Int32(K),
            grid_dim=(ceildiv(M, MMA_M), ceildiv(N, MMA_N)),
            block_dim=WARP_PER_BLOCK * WARP_SIZE,
        )

    var nstime = ctx.execution_time(run_func_mma, iterations)
    var flops = 2 * M * N * K
    var sectime = Float64(nstime) / Float64(iterations) / 1000000000
    print("Basic Tensor core kernel:")
    print(sectime, "sec")
    print(Float64(flops) * 1e-9 / sectime, " GFLOPS")
    print()

    ctx.enqueue_copy(c_host, c_device)

    # Run naive matmul.
    ctx.enqueue_copy(a_device_ref, a_host_ref)
    ctx.enqueue_copy(b_device_ref, b_host_ref)

    comptime BLOCK_DIM = 16

    # Create TileTensors for the naive kernel.
    var c_tt = TileTensor(
        c_device_ref,
        row_major(Coord(M, N)),
    )
    var a_tt = TileTensor(
        ImmPointer[Float32, ImmutAnyOrigin](
            unsafe_from_address=Int(a_device_ref.unsafe_ptr())
        ),
        row_major(Coord(M, K)),
    )
    var b_tt = TileTensor(
        ImmPointer[Float32, ImmutAnyOrigin](
            unsafe_from_address=Int(b_device_ref.unsafe_ptr())
        ),
        row_major(Coord(K, N)),
    )

    @always_inline
    def run_func_naive(ctx: DeviceContext) raises {imm}:
        comptime kernel = matmul_kernel_naive[
            DType.float32,
            DType.float32,
            DType.float32,
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

    nstime = ctx.execution_time(run_func_naive, iterations)
    var sectime2 = Float64(nstime) / Float64(iterations) / 1000000000
    print("Naive matmul kernel:")
    print(sectime2, "sec")
    print(Float64(flops) * 1e-9 / sectime2, " GFLOPS")
    print()

    ctx.enqueue_copy(c_host_ref, c_device_ref)
    ctx.synchronize()

    # Check correctness.
    var failed = False
    for i in range(M * N):
        var outVal = c_host[i]
        var outRef = c_host_ref[i]
        var relDiff = (max(outVal, outRef) / min(outVal, outRef)) - 1.0
        if (relDiff > errorTolerance) or isnan(outVal) or isnan(outRef):
            failed = True
            print(i, outVal, outRef)

    if not failed:
        print("Success 🎉: Results match.")
        print(
            "Performance basic tensor core matmul vs. naive matmul: ",
            sectime2 / sectime,
            "x",
        )
    else:
        print("Failed ❌: results mismatch.")

    assert_false(failed)

    _ = a_device
    _ = b_device
    _ = c_device
    _ = a_device_ref
    _ = b_device_ref
    _ = c_device_ref

    _ = a_host
    _ = b_host
    _ = c_host
    _ = a_host_ref
    _ = a_host_ref
    _ = c_host_ref


def run_mma_fp32_bf16_2(
    M: Int,
    N: Int,
    K: Int,
    rand_min: Int64,
    rand_max: Int64,
    iterations: Int,
    errorTolerance: Float32,
    ctx: DeviceContext,
) raises:
    print("== run_matmul fp32.bf16 16816 tensor core kernel")

    var a_host = alloc[BFloat16](M * K)
    var b_host = alloc[BFloat16](K * N)
    var c_host = alloc[Float32](M * N)
    var a_host_ref = alloc[Float32](M * K)
    var b_host_ref = alloc[Float32](K * N)
    var c_host_ref = alloc[Float32](M * N)

    for i in range(M * K):
        var val = random_si64(rand_min, rand_max)
        a_host[i] = val.cast[.bfloat16]()
        a_host_ref[i] = val.cast[.float32]()

    for i in range(K * N):
        var val = random_si64(rand_min, rand_max)
        b_host[i] = val.cast[.bfloat16]()
        b_host_ref[i] = val.cast[.float32]()

    for i in range(M * N):
        c_host[i] = 0
        c_host_ref[i] = 0

    var a_device = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var b_device = ctx.enqueue_create_buffer[.bfloat16](K * N)
    var c_device = ctx.enqueue_create_buffer[.float32](M * N)
    var a_device_ref = ctx.enqueue_create_buffer[.float32](M * K)
    var b_device_ref = ctx.enqueue_create_buffer[.float32](K * N)
    var c_device_ref = ctx.enqueue_create_buffer[.float32](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    comptime WARP_PER_BLOCK = 1
    comptime MMA_M = 16
    comptime MMA_N = 8
    comptime MMA_K = 8

    @always_inline
    def run_func_mma(ctx: DeviceContext) raises {imm}:
        comptime kernel = mma_kernel_fp32_bf16_2
        ctx.enqueue_function[kernel](
            c_device,
            a_device,
            b_device,
            Int32(M),
            Int32(N),
            Int32(K),
            grid_dim=(ceildiv(M, MMA_M), ceildiv(N, MMA_N)),
            block_dim=WARP_PER_BLOCK * WARP_SIZE,
        )

    var nstime = ctx.execution_time(run_func_mma, iterations)
    var flops = 2 * M * N * K
    var sectime = Float64(nstime) / Float64(iterations) / 1000000000
    print("Basic Tensor core kernel:")
    print(sectime, "sec")
    print(Float64(flops) * 1e-9 / sectime, " GFLOPS")
    print()

    ctx.enqueue_copy(c_host, c_device)

    # Run naive matmul.
    ctx.enqueue_copy(a_device_ref, a_host_ref)
    ctx.enqueue_copy(b_device_ref, b_host_ref)

    comptime BLOCK_DIM = 16

    # Create TileTensors for the naive kernel.
    var c_tt = TileTensor(
        c_device_ref,
        row_major(Coord(M, N)),
    )
    var a_tt = TileTensor(
        ImmPointer[Float32, ImmutAnyOrigin](
            unsafe_from_address=Int(a_device_ref.unsafe_ptr())
        ),
        row_major(Coord(M, K)),
    )
    var b_tt = TileTensor(
        ImmPointer[Float32, ImmutAnyOrigin](
            unsafe_from_address=Int(b_device_ref.unsafe_ptr())
        ),
        row_major(Coord(K, N)),
    )

    @always_inline
    def run_func_naive(ctx: DeviceContext) raises {imm}:
        comptime kernel = matmul_kernel_naive[
            DType.float32,
            DType.float32,
            DType.float32,
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

    nstime = ctx.execution_time(run_func_naive, iterations)
    var sectime2 = Float64(nstime) / Float64(iterations) / 1000000000
    print("Naive matmul kernel:")
    print(sectime2, "sec")
    print(Float64(flops) * 1e-9 / sectime2, " GFLOPS")
    print()

    ctx.enqueue_copy(c_host_ref, c_device_ref)
    ctx.synchronize()

    # Check correctness.
    var failed = False
    for i in range(M * N):
        var outVal = c_host[i]
        var outRef = c_host_ref[i]
        var relDiff = (max(outVal, outRef) / min(outVal, outRef)) - 1.0
        if (relDiff > errorTolerance) or isnan(outVal) or isnan(outRef):
            failed = True
            print(i, outVal, outRef)

    if not failed:
        print("Success 🎉: Results match.")
        print(
            "Performance basic tensor core matmul vs. naive matmul: ",
            sectime2 / sectime,
            "x",
        )
    else:
        print("Failed ❌: results mismatch.")

    assert_false(failed)

    _ = a_device
    _ = b_device
    _ = c_device
    _ = a_device_ref
    _ = b_device_ref
    _ = c_device_ref

    _ = a_host
    _ = b_host
    _ = c_host
    _ = a_host_ref
    _ = a_host_ref
    _ = c_host_ref


def run_mma_fp32_fp16(
    M: Int,
    N: Int,
    K: Int,
    rand_min: Int64,
    rand_max: Int64,
    iterations: Int,
    errorTolerance: Float32,
    ctx: DeviceContext,
) raises:
    print("== run_matmul fp32.fp16 tensor core kernel")

    var a_host = alloc[Float16](M * K)
    var b_host = alloc[Float16](K * N)
    var c_host = alloc[Float32](M * N)
    var a_host_ref = alloc[Float32](M * K)
    var b_host_ref = alloc[Float32](K * N)
    var c_host_ref = alloc[Float32](M * N)

    for i in range(M * K):
        var val = random_si64(rand_min, rand_max)
        a_host[i] = val.cast[.float16]()
        a_host_ref[i] = val.cast[.float32]()

    for i in range(K * N):
        var val = random_si64(rand_min, rand_max)
        b_host[i] = val.cast[.float16]()
        b_host_ref[i] = val.cast[.float32]()

    for i in range(M * N):
        c_host[i] = 0
        c_host_ref[i] = 0

    var a_device = ctx.enqueue_create_buffer[.float16](M * K)
    var b_device = ctx.enqueue_create_buffer[.float16](K * N)
    var c_device = ctx.enqueue_create_buffer[.float32](M * N)
    var a_device_ref = ctx.enqueue_create_buffer[.float32](M * K)
    var b_device_ref = ctx.enqueue_create_buffer[.float32](K * N)
    var c_device_ref = ctx.enqueue_create_buffer[.float32](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    comptime WARP_PER_BLOCK = 1
    comptime MMA_M = 16
    comptime MMA_N = 8
    comptime MMA_K = 8

    @always_inline
    def run_func_mma(ctx: DeviceContext) raises {imm}:
        comptime kernel = mma_kernel_fp32_fp16
        ctx.enqueue_function[kernel](
            c_device,
            a_device,
            b_device,
            Int32(M),
            Int32(N),
            Int32(K),
            grid_dim=(ceildiv(M, MMA_M), ceildiv(N, MMA_N)),
            block_dim=WARP_PER_BLOCK * WARP_SIZE,
        )

    var nstime = ctx.execution_time(run_func_mma, iterations)
    var flops = 2 * M * N * K
    var sectime = Float64(nstime) / Float64(iterations) / 1000000000
    print("Basic Tensor core kernel:")
    print(sectime, "sec")
    print(Float64(flops) * 1e-9 / sectime, " GFLOPS")
    print()

    ctx.enqueue_copy(c_host, c_device)

    # Run naive matmul.
    ctx.enqueue_copy(a_device_ref, a_host_ref)
    ctx.enqueue_copy(b_device_ref, b_host_ref)

    comptime BLOCK_DIM = 16

    # Create TileTensors for the naive kernel.
    var c_tt = TileTensor(
        c_device_ref,
        row_major(Coord(M, N)),
    )
    var a_tt = TileTensor(
        ImmPointer[Float32, ImmutAnyOrigin](
            unsafe_from_address=Int(a_device_ref.unsafe_ptr())
        ),
        row_major(Coord(M, K)),
    )
    var b_tt = TileTensor(
        ImmPointer[Float32, ImmutAnyOrigin](
            unsafe_from_address=Int(b_device_ref.unsafe_ptr())
        ),
        row_major(Coord(K, N)),
    )

    @always_inline
    def run_func_naive(ctx: DeviceContext) raises {imm}:
        comptime kernel = matmul_kernel_naive[
            DType.float32,
            DType.float32,
            DType.float32,
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

    nstime = ctx.execution_time(run_func_naive, iterations)
    var sectime2 = Float64(nstime) / Float64(iterations) / 1000000000
    print("Naive matmul kernel:")
    print(sectime2, "sec")
    print(Float64(flops) * 1e-9 / sectime2, " GFLOPS")
    print()

    ctx.enqueue_copy(c_host_ref, c_device_ref)
    ctx.synchronize()

    # Check correctness.
    var failed = False
    for i in range(M * N):
        var outVal = c_host[i]
        var outRef = c_host_ref[i]
        var relDiff = (max(outVal, outRef) / min(outVal, outRef)) - 1.0
        if (relDiff > errorTolerance) or isnan(outVal) or isnan(outRef):
            failed = True
            print(i, outVal, outRef)

    if not failed:
        print("Success 🎉: Results match.")
        print(
            "Performance basic tensor core matmul vs. naive matmul: ",
            sectime2 / sectime,
            "x",
        )
    else:
        print("Failed ❌: results mismatch.")

    assert_false(failed)

    _ = a_device
    _ = b_device
    _ = c_device
    _ = a_device_ref
    _ = b_device_ref
    _ = c_device_ref

    _ = a_host
    _ = b_host
    _ = c_host
    _ = a_host_ref
    _ = a_host_ref
    _ = c_host_ref


def run_mma_fp16_fp16(
    M: Int,
    N: Int,
    K: Int,
    rand_min: Int64,
    rand_max: Int64,
    iterations: Int,
    errorTolerance: Float32,
    ctx: DeviceContext,
) raises:
    print("== run_matmul fp16.fp16 tensor core kernel")

    var a_host = alloc[Float16](M * K)
    var b_host = alloc[Float16](K * N)
    var c_host = alloc[Float16](M * N)
    var a_host_ref = alloc[Float32](M * K)
    var b_host_ref = alloc[Float32](K * N)
    var c_host_ref = alloc[Float32](M * N)

    for i in range(M * K):
        var val = random_si64(rand_min, rand_max)
        a_host[i] = val.cast[.float16]()
        a_host_ref[i] = val.cast[.float32]()

    for i in range(K * N):
        var val = random_si64(rand_min, rand_max)
        b_host[i] = val.cast[.float16]()
        b_host_ref[i] = val.cast[.float32]()

    for i in range(M * N):
        c_host[i] = 0
        c_host_ref[i] = 0

    var a_device = ctx.enqueue_create_buffer[.float16](M * K)
    var b_device = ctx.enqueue_create_buffer[.float16](K * N)
    var c_device = ctx.enqueue_create_buffer[.float16](M * N)
    var a_device_ref = ctx.enqueue_create_buffer[.float32](M * K)
    var b_device_ref = ctx.enqueue_create_buffer[.float32](K * N)
    var c_device_ref = ctx.enqueue_create_buffer[.float32](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    comptime WARP_PER_BLOCK = 1
    comptime MMA_M = 16
    comptime MMA_N = 8
    comptime MMA_K = 8

    @always_inline
    def run_func_mma(ctx: DeviceContext) raises {imm}:
        comptime kernel = mma_kernel_fp16_fp16
        ctx.enqueue_function[kernel](
            c_device,
            a_device,
            b_device,
            Int32(M),
            Int32(N),
            Int32(K),
            grid_dim=(ceildiv(M, MMA_M), ceildiv(N, MMA_N)),
            block_dim=WARP_PER_BLOCK * WARP_SIZE,
        )

    var nstime = ctx.execution_time(run_func_mma, iterations)
    var flops = 2 * M * N * K
    var sectime = Float64(nstime) / Float64(iterations) / 1000000000
    print("Basic Tensor core kernel:")
    print(sectime, "sec")
    print(Float64(flops) * 1e-9 / sectime, " GFLOPS")
    print()

    ctx.enqueue_copy(c_host, c_device)

    # Run naive matmul.
    ctx.enqueue_copy(a_device_ref, a_host_ref)
    ctx.enqueue_copy(b_device_ref, b_host_ref)

    comptime BLOCK_DIM = 16

    # Create TileTensors for the naive kernel.
    var c_tt = TileTensor(
        c_device_ref,
        row_major(Coord(M, N)),
    )
    var a_tt = TileTensor(
        ImmPointer[Float32, ImmutAnyOrigin](
            unsafe_from_address=Int(a_device_ref.unsafe_ptr())
        ),
        row_major(Coord(M, K)),
    )
    var b_tt = TileTensor(
        ImmPointer[Float32, ImmutAnyOrigin](
            unsafe_from_address=Int(b_device_ref.unsafe_ptr())
        ),
        row_major(Coord(K, N)),
    )

    @always_inline
    def run_func_naive(ctx: DeviceContext) raises {imm}:
        comptime kernel = matmul_kernel_naive[
            DType.float32,
            DType.float32,
            DType.float32,
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

    nstime = ctx.execution_time(run_func_naive, iterations)
    var sectime2 = Float64(nstime) / Float64(iterations) / 1000000000
    print("Naive matmul kernel:")
    print(sectime2, "sec")
    print(Float64(flops) * 1e-9 / sectime2, " GFLOPS")
    print()

    ctx.enqueue_copy(c_host_ref, c_device_ref)
    ctx.synchronize()

    # Check correctness.
    var failed = False
    for i in range(M * N):
        var outVal = c_host[i].cast[.float32]()
        # var outVal = c_host[i]
        var outRef = c_host_ref[i]
        var relDiff = (max(outVal, outRef) / min(outVal, outRef)) - 1.0
        if (relDiff > errorTolerance) or isnan(outVal) or isnan(outRef):
            failed = True
            print(i, outVal, outRef)

    if not failed:
        print("Success 🎉: Results match.")
        print(
            "Performance basic tensor core matmul vs. naive matmul: ",
            sectime2 / sectime,
            "x",
        )
    else:
        print("Failed ❌: results mismatch.")

    assert_false(failed)

    _ = a_device
    _ = b_device
    _ = c_device
    _ = a_device_ref
    _ = b_device_ref
    _ = c_device_ref

    _ = a_host
    _ = b_host
    _ = c_host
    _ = a_host_ref
    _ = a_host_ref
    _ = c_host_ref


def main() raises:
    with DeviceContext() as ctx:
        # Run tensor core versions of matmul, verify correctness & compare to naive.
        run_mma_fp32_fp16(16, 8, 8, -100, 100, 10, 0.01, ctx)
        run_mma_fp32_fp16(1024, 1024, 1024, -100, 100, 10, 0.01, ctx)
        run_mma_fp32_fp16(1024, 4096, 2048, -100, 100, 10, 0.01, ctx)

        run_mma_fp32_bf16(16, 8, 8, -100, 100, 10, 0.01, ctx)
        run_mma_fp32_bf16(1024, 1024, 1024, -100, 100, 10, 0.01, ctx)
        run_mma_fp32_bf16(1024, 4096, 2048, -100, 100, 10, 0.01, ctx)

        run_mma_fp32_bf16_2(16, 8, 16, -100, 100, 10, 0.01, ctx)
        run_mma_fp32_bf16_2(2048, 1024, 2048, -100, 100, 10, 0.01, ctx)
        run_mma_fp32_bf16_2(2048, 4096, 2048, -100, 100, 10, 0.01, ctx)

        run_mma_fp32_tf32(16, 8, 8, -100, 100, 10, 0.01, ctx)
        run_mma_fp32_tf32(1024, 1024, 1024, -100, 100, 10, 0.01, ctx)
        run_mma_fp32_tf32(1024, 4096, 2048, -100, 100, 10, 0.01, ctx)

        run_mma_fp16_fp16(16, 8, 8, -100, 100, 10, 0.01, ctx)
        run_mma_fp16_fp16(512, 128, 32, -10, 10, 10, 0.01, ctx)
        run_mma_fp16_fp16(128, 256, 64, -10, 10, 10, 0.01, ctx)
