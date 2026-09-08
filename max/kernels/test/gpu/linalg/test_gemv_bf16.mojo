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

import max.gpu.primitives.warp as warp
from max.gpu import WARP_SIZE
from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from linalg.gemv import gemv_kernel
from linalg.matmul.gpu import matmul_kernel_naive
from std.testing import assert_false

from std.utils.numerics import isnan


def run_matvec(M: Int, N: Int, K: Int, *, ctx: DeviceContext) raises:
    print("== run_matvec kernel")

    var iterations = 100
    var a_host = alloc[BFloat16](M * K)
    var b_host = alloc[BFloat16](K * N)
    var c_host = alloc[Float32](M * N)
    var a_host_n = alloc[Float32](M * K)
    var b_host_n = alloc[Float32](K * N)
    var c_host_n = alloc[Float32](M * N)

    for i in range(M * K):
        a_host[i] = BFloat16(i)
        a_host_n[i] = Float32(i)

    for i in range(K * N):
        b_host[i] = BFloat16(i + 1)
        b_host_n[i] = Float32(i + 1)

    for i in range(M * N):
        c_host[i] = 0

    for i in range(M * N):
        c_host_n[i] = 0

    var a_device = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var b_device = ctx.enqueue_create_buffer[.bfloat16](K * N)
    var c_device = ctx.enqueue_create_buffer[.float32](M * N)
    var a_device_n = ctx.enqueue_create_buffer[.float32](M * K)
    var b_device_n = ctx.enqueue_create_buffer[.float32](K * N)
    var c_device_n = ctx.enqueue_create_buffer[.float32](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    comptime WARPS_PER_BLOCK = 32
    comptime kernel = gemv_kernel[.float32, .bfloat16, .bfloat16]

    @always_inline
    def run_func_gemv(ctx: DeviceContext) raises {imm}:
        ctx.enqueue_function[kernel](
            c_device,
            a_device,
            b_device,
            Int32(M),
            Int32(N),
            Int32(K),
            grid_dim=ceildiv(M, WARPS_PER_BLOCK),
            block_dim=WARP_SIZE * WARPS_PER_BLOCK,
        )

    var kernelType = "GEMV"
    var nstime = ctx.execution_time(run_func_gemv, iterations)
    var flops = 2 * M * N * K
    var sectime = Float64(nstime) / Float64(iterations) / 1000000000
    print(kernelType, "KERNEL:")
    print(sectime, "sec")
    print(Float64(flops) * 1e-9 / sectime, " GFLOPS")
    print()

    ctx.enqueue_copy(c_host, c_device)

    # running naive
    ctx.enqueue_copy(a_device_n, a_host_n)
    ctx.enqueue_copy(b_device_n, b_host_n)

    comptime BLOCK_DIM = 16

    # Create TileTensors for the naive kernel.
    # a/b are constructed as immutable to match the ImmutAnyOrigin
    # parameters that matmul_kernel_naive expects (enqueue_function
    # requires exact type matches).

    var c_tt = TileTensor(
        c_device_n,
        row_major(Coord(M, N)),
    )
    var a_tt = TileTensor(
        ImmPointer[Float32, ImmutAnyOrigin](
            unsafe_from_address=Int(a_device_n.unsafe_ptr())
        ),
        row_major(Coord(M, K)),
    )
    var b_tt = TileTensor(
        ImmPointer[Float32, ImmutAnyOrigin](
            unsafe_from_address=Int(b_device_n.unsafe_ptr())
        ),
        row_major(Coord(K, N)),
    )

    @always_inline
    def run_func_naive(ctx: DeviceContext) raises {imm}:
        comptime kernel = matmul_kernel_naive[
            .float32,
            .float32,
            .float32,
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
    print("SHMEM MATMUL:")
    print(sectime2, "sec")
    print(Float64(flops) * 1e-9 / sectime2, " GFLOPS")
    print()

    ctx.enqueue_copy(c_host_n, c_device_n)
    ctx.synchronize()

    # Due to varied pattern of FP arith the accumulated sum isn't exactly
    # accurate. Hence relative tolerance needs to be checked.
    comptime errorTolerance = 0.1
    var failed = False
    for i in range(M * N):
        var outVal = c_host[i]
        var outRef = c_host_n[i]
        var relDiff = (max(outVal, outRef) / min(outVal, outRef)) - 1.0
        if (relDiff > errorTolerance) or isnan(outVal) or isnan(outRef):
            failed = True

    if not failed:
        print("Success 🎉: results match")
        print(
            "Performance warp-shuffle matvec vs. shmem matmul: ",
            sectime2 / sectime,
            "x",
        )
    else:
        print("Failed ❌: results mismatch")

    assert_false(failed)

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
        run_matvec(4096, 1, 4096, ctx=ctx)
