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
from std.random import random_float64

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu import block_dim
from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu import matmul_kernel_naive
from std.testing import assert_almost_equal


def test_vendor_blas[
    dtype: DType, transpose_b: Bool
](*, M: Int, N: Int, K: Int, ctx: DeviceContext) raises:
    var a_host = ctx.enqueue_create_host_buffer[dtype](M * K)
    var b_host = ctx.enqueue_create_host_buffer[dtype](K * N)
    var c_host = ctx.enqueue_create_host_buffer[dtype](M * N)
    var c_host_ref = ctx.enqueue_create_host_buffer[dtype](M * N)

    for m in range(M):
        for k in range(K):
            a_host[m * K + k] = random_float64(-0.1, 0.1).cast[dtype]()

    for k in range(K):
        for n in range(N):
            b_host[k * N + n] = random_float64(-0.1, 0.1).cast[dtype]()

    var a_device = ctx.enqueue_create_buffer[dtype](M * K)
    var b_device = ctx.enqueue_create_buffer[dtype](K * N)
    var c_device = ctx.enqueue_create_buffer[dtype](M * N)
    var c_device_ref = ctx.enqueue_create_buffer[dtype](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    var a = TileTensor(
        a_device,
        row_major(Coord(M, K)),
    )
    var b = TileTensor(
        b_device,
        row_major(Coord(N, K)) if transpose_b else row_major(Coord(K, N)),
    )
    var c = TileTensor(
        c_device,
        row_major(Coord(M, N)),
    )

    vendor_blas.matmul(ctx, c, a, b, c_row_major=True, transpose_b=transpose_b)

    ctx.enqueue_copy(c_host, c_device)

    comptime BLOCK_DIM = 16

    # Create TileTensors for the naive kernel.
    # a/b are constructed as immutable to match the ImmutAnyOrigin
    # parameters that matmul_kernel_naive expects.

    var c_ref_tt = TileTensor(
        c_device_ref,
        row_major(Coord(M, N)),
    )
    var a_tt = TileTensor(
        ImmPointer[Scalar[dtype], ImmutAnyOrigin](
            unsafe_from_address=Int(a_device.unsafe_ptr())
        ),
        row_major(Coord(M, K)),
    )
    var b_tt = TileTensor(
        ImmPointer[Scalar[dtype], ImmutAnyOrigin](
            unsafe_from_address=Int(b_device.unsafe_ptr())
        ),
        row_major(Coord(N, K)) if transpose_b else row_major(Coord(K, N)),
    )

    comptime kernel = matmul_kernel_naive[
        dtype,
        dtype,
        dtype,
        type_of(c_ref_tt).LayoutType,
        type_of(a_tt).LayoutType,
        type_of(b_tt).LayoutType,
        BLOCK_DIM,
        transpose_b=transpose_b,
    ]

    ctx.enqueue_function[kernel](
        c_ref_tt,
        a_tt,
        b_tt,
        Int32(M),
        Int32(N),
        Int32(K),
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )

    ctx.enqueue_copy(c_host_ref, c_device_ref)

    ctx.synchronize()

    for i in range(M * N):
        assert_almost_equal(
            c_host[i],
            c_host_ref[i],
            atol=1e-2 if dtype.is_half_float() else 1e-3,
            rtol=1e-2 if dtype.is_half_float() else 1e-3,
        )

    _ = a_device
    _ = b_device
    _ = c_device
    _ = c_device_ref


def dispatch_test_vendor_blas[
    transpose_b: Bool
](*, M: Int, N: Int, K: Int, ctx: DeviceContext) raises:
    test_vendor_blas[dtype=.bfloat16, transpose_b=transpose_b](
        M=M, N=N, K=K, ctx=ctx
    )
    test_vendor_blas[dtype=.float32, transpose_b=transpose_b](
        M=M, N=N, K=K, ctx=ctx
    )


def test_vendor_blas_multi_gpu() raises:
    """Test vendor BLAS on multiple GPUs to ensure device contexts work correctly.
    """

    # Test on default device (GPU 0)
    with DeviceContext() as ctx0:
        # Test the specific failing shapes from the logs
        dispatch_test_vendor_blas[transpose_b=True](
            M=78, N=75837, K=5120, ctx=ctx0
        )
        dispatch_test_vendor_blas[transpose_b=True](
            M=31, N=75837, K=5120, ctx=ctx0
        )
        # Also test a smaller shape
        dispatch_test_vendor_blas[transpose_b=True](
            M=64, N=128, K=256, ctx=ctx0
        )

    # Test on GPU 1 if available.
    if DeviceContext.number_of_devices() >= 2:
        with DeviceContext(device_id=1) as ctx1:
            dispatch_test_vendor_blas[transpose_b=True](
                M=78, N=75837, K=5120, ctx=ctx1
            )
            dispatch_test_vendor_blas[transpose_b=True](
                M=31, N=75837, K=5120, ctx=ctx1
            )
            dispatch_test_vendor_blas[transpose_b=True](
                M=64, N=128, K=256, ctx=ctx1
            )

        # Test alternating between GPUs to ensure handle management works
        # correctly.
        with DeviceContext(device_id=0) as ctx0:
            dispatch_test_vendor_blas[transpose_b=True](
                M=32, N=64, K=128, ctx=ctx0
            )

        with DeviceContext(device_id=1) as ctx1:
            dispatch_test_vendor_blas[transpose_b=True](
                M=32, N=64, K=128, ctx=ctx1
            )

        with DeviceContext(device_id=0) as ctx0:
            dispatch_test_vendor_blas[transpose_b=True](
                M=32, N=64, K=128, ctx=ctx0
            )


def main() raises:
    with DeviceContext() as ctx:
        dispatch_test_vendor_blas[transpose_b=True](M=550, N=2048, K=8, ctx=ctx)
        dispatch_test_vendor_blas[transpose_b=False](M=63, N=65, K=66, ctx=ctx)
        dispatch_test_vendor_blas[transpose_b=False](
            M=7, N=6144, K=4096, ctx=ctx
        )
        dispatch_test_vendor_blas[transpose_b=False](
            M=1024, N=1024, K=1024, ctx=ctx
        )
        dispatch_test_vendor_blas[transpose_b=False](
            M=1, N=1024, K=1024, ctx=ctx
        )

    # Run multi-GPU test
    test_vendor_blas_multi_gpu()
