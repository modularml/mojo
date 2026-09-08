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
from max.gpu.compute.mma_util import load_matrix_a_amd as load_matrix_a
from max.gpu.compute.mma_util import load_matrix_b_amd as load_matrix_b
from max.gpu.compute.mma_util import store_matrix_d
from std.testing import assert_equal


def matmul_naive[
    a_type: DType, b_type: DType, c_type: DType, //, mma_n_blocks: Int = 1
](
    a: Pointer[Scalar[a_type], _],
    b: Pointer[Scalar[b_type], _],
    c: MutPointer[Scalar[c_type], _],
    m: Int,
    n: Int,
    k: Int,
):
    for bl in range(mma_n_blocks):
        for i in range(m):
            for l in range(k):
                for j in range(n):
                    var av = a[bl * m * k + k * i + l].cast[c_type]()
                    var bv = b[bl * k * n + n * l + j].cast[c_type]()
                    c[bl * m * n + n * i + j] += av * bv


def mma_kernel_fp32_fp32(
    a_ptr: ImmPointer[Float32, ImmutAnyOrigin],
    b_ptr: ImmPointer[Float32, ImmutAnyOrigin],
    c_ptr: MutPointer[Float32, MutAnyOrigin],
    m_dev: Int32,
    n_dev: Int32,
    k_dev: Int32,
):
    var m = Int(m_dev)
    var n = Int(n_dev)
    var k = Int(k_dev)
    comptime mma_m = 16
    comptime mma_n = 16
    comptime mma_k = 4

    var d_reg: SIMD[.float32, 4] = 0
    var tile_loops = k // (4 * mma_k)

    for l in range(tile_loops):
        for i in range(4):
            var a_tile_row = block_idx.x * mma_m
            var a_tile_col = 4 * (l * mma_k + i)
            var b_tile_row = 4 * (l * mma_k + i)
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


def mma_kernel_fp32_fp16[
    mma_n_blocks: Int
](
    a_ptr: ImmPointer[Float16, ImmutAnyOrigin],
    b_ptr: ImmPointer[Float16, ImmutAnyOrigin],
    c_ptr: MutPointer[Float32, MutAnyOrigin],
    m_dev: Int32,
    n_dev: Int32,
    k_dev: Int32,
):
    var m = Int(m_dev)
    var n = Int(n_dev)
    var k = Int(k_dev)
    comptime mma_m = 4 if mma_n_blocks == 16 else 16
    comptime mma_n = 4 if mma_n_blocks == 16 else 16
    comptime mma_k = 4 if mma_n_blocks == 16 else 16

    var d_reg: SIMD[.float32, 4] = 0
    var tile_loops = k // mma_k

    for l in range(tile_loops):
        var a_tile_row = block_idx.x * mma_m
        var a_tile_col = l * mma_k
        var b_tile_row = l * mma_k
        var b_tile_col = block_idx.y * mma_n

        var a_reg = load_matrix_a[mma_m, mma_n, mma_k, mma_n_blocks](
            a_ptr, a_tile_row, a_tile_col, k
        )
        var b_reg = load_matrix_b[mma_m, mma_n, mma_k, mma_n_blocks](
            b_ptr, b_tile_row, b_tile_col, n, tile_loops
        )
        mma[mma_n_blocks](d_reg, a_reg, b_reg, d_reg)

    var c_tile_row = block_idx.x * mma_m
    var c_tile_col = block_idx.y * mma_n
    store_matrix_d[mma_m, mma_n, mma_k, mma_n_blocks](
        c_ptr, d_reg, c_tile_row, c_tile_col, n
    )


def mma_kernel_fp32_bf16[
    mma_n_blocks: Int
](
    a_ptr: ImmPointer[BFloat16, ImmutAnyOrigin],
    b_ptr: ImmPointer[BFloat16, ImmutAnyOrigin],
    c_ptr: MutPointer[Float32, MutAnyOrigin],
    m_dev: Int32,
    n_dev: Int32,
    k_dev: Int32,
):
    var m = Int(m_dev)
    var n = Int(n_dev)
    var k = Int(k_dev)
    comptime mma_m = 4 if mma_n_blocks == 16 else 16
    comptime mma_n = 4 if mma_n_blocks == 16 else 16
    comptime mma_k = 4 if mma_n_blocks == 16 else 16

    var d_reg: SIMD[.float32, 4] = 0
    var tile_loops = k // mma_k

    for l in range(tile_loops):
        var a_tile_row = block_idx.x * mma_m
        var a_tile_col = l * mma_k
        var b_tile_row = l * mma_k
        var b_tile_col = block_idx.y * mma_n

        var a_reg = load_matrix_a[mma_m, mma_n, mma_k, mma_n_blocks](
            a_ptr, a_tile_row, a_tile_col, k
        )
        var b_reg = load_matrix_b[mma_m, mma_n, mma_k, mma_n_blocks](
            b_ptr, b_tile_row, b_tile_col, n, tile_loops
        )
        mma[mma_n_blocks](d_reg, a_reg, b_reg, d_reg)

    var c_tile_row = block_idx.x * mma_m
    var c_tile_col = block_idx.y * mma_n
    store_matrix_d[mma_m, mma_n, mma_k, mma_n_blocks](
        c_ptr, d_reg, c_tile_row, c_tile_col, n
    )


def run_mma_fp32_fp32(
    M: Int,
    N: Int,
    K: Int,
    rand_min: Int64,
    rand_max: Int64,
    ctx: DeviceContext,
) raises:
    print("== run_matmul fp32.fp32 matrix core kernel")

    var a_host = ctx.enqueue_create_host_buffer[.float32](M * K)
    var b_host = ctx.enqueue_create_host_buffer[.float32](K * N)
    var c_host = ctx.enqueue_create_host_buffer[.float32](M * N)
    var c_host_ref = ctx.enqueue_create_host_buffer[.float32](M * N)
    # Zero-init c_host (copied to device) and c_host_ref (matmul accumulator).
    for i in range(M * N):
        c_host[i] = 0
        c_host_ref[i] = 0

    for i in range(M * K):
        var val = random_si64(rand_min, rand_max)
        a_host[i] = val.cast[.float32]()

    for i in range(K * N):
        var val = random_si64(rand_min, rand_max)
        b_host[i] = val.cast[.float32]()

    var a_device = ctx.enqueue_create_buffer[.float32](M * K)
    var b_device = ctx.enqueue_create_buffer[.float32](K * N)
    var c_device = ctx.enqueue_create_buffer[.float32](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)
    ctx.enqueue_copy(c_device, c_host)
    ctx.synchronize()

    comptime WARP_PER_BLOCK = 1
    comptime MMA_M = 16
    comptime MMA_N = 16
    comptime MMA_K = 4

    comptime kernel = mma_kernel_fp32_fp32

    ctx.enqueue_function[kernel](
        a_device,
        b_device,
        c_device,
        Int32(M),
        Int32(N),
        Int32(K),
        grid_dim=(ceildiv(M, MMA_M), ceildiv(N, MMA_N)),
        block_dim=WARP_PER_BLOCK * WARP_SIZE,
    )

    ctx.enqueue_copy(c_host, c_device)
    ctx.synchronize()

    matmul_naive(
        a_host.unsafe_ptr(),
        b_host.unsafe_ptr(),
        c_host_ref.unsafe_ptr(),
        M,
        N,
        K,
    )

    var errors = 0
    for i in range(M * N):
        if c_host[i] != c_host_ref[i]:
            errors += 1

    _ = a_device
    _ = b_device
    _ = c_device

    if errors == 0:
        print("Success 🎉: Results match.")
    else:
        print("Failed ❌: results mismatch.")

    assert_equal(errors, 0)


def run_mma_fp32_fp16[
    mma_n_blocks: Int = 1
](
    M: Int, N: Int, K: Int, rand_min: Int64, rand_max: Int64, ctx: DeviceContext
) raises:
    print("== run_matmul fp32.fp16 matrix core kernel")

    var a_host = ctx.enqueue_create_host_buffer[.float16](M * K * mma_n_blocks)
    var b_host = ctx.enqueue_create_host_buffer[.float16](K * N * mma_n_blocks)
    var c_host = ctx.enqueue_create_host_buffer[.float32](M * N * mma_n_blocks)
    var c_host_ref = ctx.enqueue_create_host_buffer[.float32](
        M * N * mma_n_blocks
    )

    for b in range(mma_n_blocks):
        for i in range(M * K):
            var val = random_si64(rand_min, rand_max)
            a_host[b * M * K + i] = val.cast[.float16]()

    for b in range(mma_n_blocks):
        for i in range(K * N):
            var val = random_si64(rand_min, rand_max)
            b_host[b * K * N + i] = val.cast[.float16]()

    for b in range(mma_n_blocks):
        for i in range(M * N):
            c_host[b * M * N + i] = 0
            c_host_ref[b * M * N + i] = 0

    var a_device = ctx.enqueue_create_buffer[.float16](M * K * mma_n_blocks)
    var b_device = ctx.enqueue_create_buffer[.float16](K * N * mma_n_blocks)
    var c_device = ctx.enqueue_create_buffer[.float32](M * N * mma_n_blocks)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)
    ctx.enqueue_copy(c_device, c_host)
    ctx.synchronize()

    comptime WARP_PER_BLOCK = 1
    comptime MMA_M = 4 if mma_n_blocks == 16 else 16
    comptime MMA_N = 4 if mma_n_blocks == 16 else 16
    comptime MMA_K = 4 if mma_n_blocks == 16 else 16

    comptime kernel = mma_kernel_fp32_fp16[mma_n_blocks]

    ctx.enqueue_function[kernel](
        a_device,
        b_device,
        c_device,
        Int32(M),
        Int32(N),
        Int32(K),
        grid_dim=(ceildiv(M, MMA_M), ceildiv(N, MMA_N)),
        block_dim=WARP_PER_BLOCK * WARP_SIZE,
    )

    ctx.enqueue_copy(c_host, c_device)
    ctx.synchronize()

    matmul_naive[mma_n_blocks](
        a_host.unsafe_ptr(),
        b_host.unsafe_ptr(),
        c_host_ref.unsafe_ptr(),
        M,
        N,
        K,
    )

    var errors = 0
    for b in range(mma_n_blocks):
        for i in range(M * N):
            if c_host[b * M * N + i] != c_host_ref[b * M * N + i]:
                errors += 1

    _ = a_device
    _ = b_device
    _ = c_device

    if errors == 0:
        print("Success 🎉: Results match.")
    else:
        print("Failed ❌: results mismatch.")

    assert_equal(errors, 0)


def run_mma_fp32_bf16[
    mma_n_blocks: Int = 1
](
    M: Int,
    N: Int,
    K: Int,
    rand_min: Int64,
    rand_max: Int64,
    ctx: DeviceContext,
) raises:
    print("== run_matmul fp32.bf16 matrix core kernel")

    var a_host = ctx.enqueue_create_host_buffer[.bfloat16](M * K * mma_n_blocks)
    var b_host = ctx.enqueue_create_host_buffer[.bfloat16](K * N * mma_n_blocks)
    var c_host = ctx.enqueue_create_host_buffer[.float32](M * N * mma_n_blocks)
    var c_host_ref = ctx.enqueue_create_host_buffer[.float32](
        M * N * mma_n_blocks
    )

    for b in range(mma_n_blocks):
        for i in range(M * K):
            var val = random_si64(rand_min, rand_max)
            a_host[b * M * K + i] = val.cast[.bfloat16]()

    for b in range(mma_n_blocks):
        for i in range(K * N):
            var val = random_si64(rand_min, rand_max)
            b_host[b * K * N + i] = val.cast[.bfloat16]()

    for b in range(mma_n_blocks):
        for i in range(M * N):
            c_host[b * M * N + i] = 0
            c_host_ref[b * M * N + i] = 0

    var a_device = ctx.enqueue_create_buffer[.bfloat16](M * K * mma_n_blocks)
    var b_device = ctx.enqueue_create_buffer[.bfloat16](K * N * mma_n_blocks)
    var c_device = ctx.enqueue_create_buffer[.float32](M * N * mma_n_blocks)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)
    ctx.enqueue_copy(c_device, c_host)
    ctx.synchronize()

    comptime WARP_PER_BLOCK = 1
    comptime MMA_M = 4 if mma_n_blocks == 16 else 16
    comptime MMA_N = 4 if mma_n_blocks == 16 else 16
    comptime MMA_K = 4 if mma_n_blocks == 16 else 16

    comptime kernel = mma_kernel_fp32_bf16[mma_n_blocks]

    ctx.enqueue_function[kernel](
        a_device,
        b_device,
        c_device,
        Int32(M),
        Int32(N),
        Int32(K),
        grid_dim=(ceildiv(M, MMA_M), ceildiv(N, MMA_N)),
        block_dim=WARP_PER_BLOCK * WARP_SIZE,
    )

    ctx.enqueue_copy(c_host, c_device)
    ctx.synchronize()

    matmul_naive[mma_n_blocks](
        a_host.unsafe_ptr(),
        b_host.unsafe_ptr(),
        c_host_ref.unsafe_ptr(),
        M,
        N,
        K,
    )

    var errors = 0
    for b in range(mma_n_blocks):
        for i in range(M * N):
            if c_host[b * M * N + i] != c_host_ref[b * M * N + i]:
                errors += 1

    _ = a_device
    _ = b_device
    _ = c_device

    if errors == 0:
        print("Success 🎉: Results match.")
    else:
        print("Failed ❌: results mismatch.")

    assert_equal(errors, 0)


def main() raises:
    with DeviceContext() as ctx:
        run_mma_fp32_fp32(16, 16, 16, -100, 100, ctx)
        run_mma_fp32_fp32(1024, 1024, 1024, -100, 100, ctx)
        run_mma_fp32_fp32(1024, 4096, 2048, -100, 100, ctx)

        run_mma_fp32_fp16(16, 16, 16, -100, 100, ctx)
        run_mma_fp32_fp16(1024, 1024, 1024, -100, 100, ctx)
        run_mma_fp32_fp16(1024, 4096, 2048, -100, 100, ctx)

        run_mma_fp32_bf16(16, 16, 16, -100, 100, ctx)
        run_mma_fp32_bf16(1024, 1024, 1024, -100, 100, ctx)
        run_mma_fp32_bf16(1024, 4096, 2048, -100, 100, ctx)

        # The below tests are for 4x4x4x_16B MFMA instructions.

        run_mma_fp32_fp16[16](4, 4, 4, -100, 100, ctx)
        run_mma_fp32_bf16[16](4, 4, 4, -100, 100, ctx)

        # Since these are effectively 16 * (M * N * K) we use smaller test matrices
        # than in the 16x16x16 MFMA test cases above.
        run_mma_fp32_fp16[16](256, 256, 256, -100, 100, ctx)
        run_mma_fp32_fp16[16](64, 256, 128, -100, 100, ctx)

        run_mma_fp32_bf16[16](256, 256, 256, -100, 100, ctx)
        run_mma_fp32_bf16[16](64, 256, 128, -100, 100, ctx)
