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
from std.os import abort
from std.sys import has_amd_gpu_accelerator, has_nvidia_gpu_accelerator

from max.gpu.host import DeviceContext
from internal_utils import assert_almost_equal
from std.random import rand
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu import matmul_kernel_naive
from linalg.matmul.vendor.blas import matmul


def test_matmul[
    input_type: DType, M: Int, N: Int, K: Int
](ctx: DeviceContext) raises:
    print("== test_vendor_blas", input_type, "x", M, "x", N, "x", K)

    var a_host_ptr = ctx.enqueue_create_host_buffer[input_type](M * K)
    var b_host_ptr = ctx.enqueue_create_host_buffer[input_type](N * K)
    var c_host_ptr = ctx.enqueue_create_host_buffer[.float32](M * N)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[.float32](M * N)

    rand(a_host_ptr.unsafe_ptr(), M * K)
    rand(b_host_ptr.unsafe_ptr(), N * K)

    var a_device = ctx.enqueue_create_buffer[input_type](M * K)
    var b_device = ctx.enqueue_create_buffer[input_type](N * K)
    var c_device = ctx.enqueue_create_buffer[.float32](M * N)
    var c_device_ref = ctx.enqueue_create_buffer[.float32](M * N)

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    var a_tt = TileTensor(
        a_device,
        row_major(Coord(M, K)),
    )
    var b_tt = TileTensor(
        b_device,
        row_major(Coord(N, K)),
    )
    var c_tt = TileTensor(
        c_device,
        row_major(Coord(M, N)),
    )

    matmul(
        ctx,
        c_tt,
        a_tt,
        b_tt,
        transpose_b=True,
        c_row_major=True,
    )

    ctx.enqueue_copy(c_host_ptr, c_device)

    # Create immutable TileTensors for the naive kernel reference.
    var c_ref_tt = TileTensor(
        c_device_ref,
        row_major(Coord(M, N)),
    )
    var a_immut_tt = TileTensor(
        ImmPointer[Scalar[input_type], ImmutAnyOrigin](
            unsafe_from_address=Int(a_device.unsafe_ptr())
        ),
        row_major(Coord(M, K)),
    )
    var b_immut_tt = TileTensor(
        ImmPointer[Scalar[input_type], ImmutAnyOrigin](
            unsafe_from_address=Int(b_device.unsafe_ptr())
        ),
        row_major(Coord(N, K)),
    )

    # Run naive matmul.
    comptime BLOCK_DIM = 16
    comptime kernel = matmul_kernel_naive[
        DType.float32,
        input_type,
        input_type,
        type_of(c_ref_tt).LayoutType,
        type_of(a_immut_tt).LayoutType,
        type_of(b_immut_tt).LayoutType,
        BLOCK_DIM,
        transpose_b=True,
    ]
    ctx.enqueue_function[kernel](
        c_ref_tt,
        a_immut_tt,
        b_immut_tt,
        Int32(M),
        Int32(N),
        Int32(K),
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )

    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)

    ctx.synchronize()

    assert_almost_equal(
        c_host_ptr.unsafe_ptr(),
        c_host_ref_ptr.unsafe_ptr(),
        M * N,
        atol=0.01,
        rtol=0.01,
    )

    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^


def test_matmul[input_types: List[DType]]() raises:
    with DeviceContext() as ctx:
        comptime for input_type in input_types:
            test_matmul[input_type, 64, 16, 32](ctx)
            test_matmul[input_type, 512, 2560, 512](ctx)


def main() raises:
    comptime if has_amd_gpu_accelerator():
        test_matmul[[DType.float8_e4m3fnuz, DType.bfloat16]]()
    elif has_nvidia_gpu_accelerator():
        test_matmul[[DType.float8_e4m3fn, DType.bfloat16]]()
    else:
        abort("Unknown GPU Accelerator.")
