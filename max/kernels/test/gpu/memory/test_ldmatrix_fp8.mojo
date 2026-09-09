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
from std.math.uutils import umod, ufloordiv

from max.gpu import WARP_SIZE, lane_id
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.compute.mma import ld_matrix, mma
from max.gpu.compute.mma_util import store_matrix_d
from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from layout.tensor_core import get_fragment_size, get_mma_shape
from linalg.matmul.gpu import matmul_kernel_naive
from std.memory import unsafe_stack_allocation
from std.testing import assert_almost_equal

from std.utils.numerics import get_accum_type


def test_ldmatrix_fp8[
    input_type: DType,
](
    c_ptr: MutPointer[Float32, MutAnyOrigin],
    a_ptr: ImmPointer[Scalar[input_type], ImmutAnyOrigin],
    b_ptr: ImmPointer[Scalar[input_type], ImmutAnyOrigin],
):
    comptime accum_type = get_accum_type[input_type]()
    comptime mma_shape = get_mma_shape[input_type, accum_type]()
    comptime M = mma_shape[0]
    comptime N = mma_shape[1]
    comptime K = mma_shape[2]
    comptime frag_size = get_fragment_size[mma_shape]()
    comptime a_frag_size = frag_size[0]
    comptime b_frag_size = frag_size[1]
    comptime c_frag_size = frag_size[2]

    var d = SIMD[accum_type, c_frag_size](0)

    var a_shared = unsafe_stack_allocation[
        M * K, input_type, alignment=32, address_space=.SHARED
    ]()
    var b_shared = unsafe_stack_allocation[
        N * K, input_type, alignment=32, address_space=.SHARED
    ]()

    for i in range(lane_id(), M * K, WARP_SIZE):
        a_shared[i] = a_ptr[i]

    for i in range(lane_id(), N * K, WARP_SIZE):
        b_shared[i] = b_ptr[i]

    barrier()

    var a_reg = ld_matrix[a_frag_size](
        a_shared + umod(lane_id(), 16) * 32 + ufloordiv(lane_id(), 16) * 16
    )

    var b_reg = ld_matrix[b_frag_size](
        b_shared + umod(lane_id(), 8) * 32 + ufloordiv(lane_id(), 8) * 16
    )

    mma(d, a_reg, b_reg, d)
    store_matrix_d[M, N, K](
        c_ptr,
        # Store matrix is hardcoded to store 4 elements.
        rebind[SIMD[.float32, 4]](d),
        0,
        0,
        mma_shape[1],
    )


def check_ldmatrix_fp8[
    input_type: DType,
](ctx: DeviceContext) raises:
    print("== test ldmatrix transposed fp8")

    # Shape for a single mma.
    comptime accum_type = get_accum_type[input_type]()
    comptime mma_shape = get_mma_shape[input_type, accum_type]()
    comptime M = mma_shape[0]
    comptime N = mma_shape[1]
    comptime K = mma_shape[2]

    var a_host = alloc[Scalar[input_type]](M * K)
    var b_host = alloc[Scalar[input_type]](K * N)
    var c_host = alloc[Float32](M * N)
    var c_host_ref = alloc[Float32](M * N)

    comptime for m in range(M):
        comptime for k in range(K):
            a_host[m * K + k] = Scalar[input_type](m + k)

    comptime for k in range(K):
        comptime for n in range(N):
            b_host[n * K + k] = Scalar[input_type](k + n)

    for i in range(M * N):
        c_host[i] = 0
        c_host_ref[i] = 0

    var a_device = ctx.enqueue_create_buffer[input_type](M * K)
    var b_device = ctx.enqueue_create_buffer[input_type](K * N)
    var c_device = ctx.enqueue_create_buffer[.float32](M * N)
    var c_device_ref = ctx.enqueue_create_buffer[.float32](M * N)

    ctx.enqueue_copy(a_device, a_host)
    ctx.enqueue_copy(b_device, b_host)

    comptime kernel_func = test_ldmatrix_fp8[input_type]
    ctx.enqueue_function[kernel_func](
        c_device,
        a_device,
        b_device,
        grid_dim=1,
        block_dim=WARP_SIZE,
    )

    ctx.enqueue_copy(c_host, c_device)

    # Run naive matmul.
    comptime BLOCK_DIM = 16

    # Create TileTensors for the naive kernel.
    # a/b are constructed as immutable to match the ImmutAnyOrigin
    # parameters that matmul_kernel_naive expects (enqueue_function
    # requires exact type matches).
    var c_ref_tt = TileTensor(
        c_device_ref,
        row_major(Coord(M, N)),
    )
    var a_tt = TileTensor(
        ImmPointer[Scalar[input_type], ImmutAnyOrigin](
            unsafe_from_address=Int(a_device.unsafe_ptr())
        ),
        row_major(Coord(M, K)),
    )
    var b_tt = TileTensor(
        ImmPointer[Scalar[input_type], ImmutAnyOrigin](
            unsafe_from_address=Int(b_device.unsafe_ptr())
        ),
        row_major(Coord(N, K)),
    )  # N x K for transpose_b=True

    comptime kernel = matmul_kernel_naive[
        DType.float32,
        input_type,
        input_type,
        type_of(c_ref_tt).LayoutType,
        type_of(a_tt).LayoutType,
        type_of(b_tt).LayoutType,
        BLOCK_DIM,
        transpose_b=True,
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
        var out_val = c_host.load(i)
        var out_ref = c_host_ref.load(i)
        assert_almost_equal(out_val, out_ref)

    _ = a_device
    _ = b_device
    _ = c_device
    _ = c_device_ref

    _ = a_host
    _ = b_host
    _ = c_host
    _ = c_host_ref


def main() raises:
    with DeviceContext() as ctx:
        check_ldmatrix_fp8[.float8_e4m3fn](ctx)
        check_ldmatrix_fp8[.float8_e5m2](ctx)
