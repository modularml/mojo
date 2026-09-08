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

from max.gpu import global_idx
from max.gpu.primitives import block
from max.gpu.primitives import warp
from max.gpu.globals import WARP_SIZE
from max.gpu.host import DeviceContext
from std.testing import assert_equal

comptime dtype = DType.uint64


def warp_prefix_sum_kernel[
    dtype: DType,
    exclusive: Bool,
](
    output: MutPointer[Scalar[dtype], MutAnyOrigin],
    input: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    size_dev: Int32,
):
    var size = Int(size_dev)
    var tid = global_idx.x
    if tid >= size:
        return
    output[tid] = warp.prefix_sum[exclusive=exclusive](input[tid])


def test_warp_prefix_sum[exclusive: Bool](ctx: DeviceContext) raises:
    comptime size = WARP_SIZE
    comptime BLOCK_SIZE = WARP_SIZE

    # Allocate and initialize host memory
    var in_host = ctx.enqueue_create_host_buffer[dtype](size)
    var out_host = ctx.enqueue_create_host_buffer[dtype](size)
    for i in range(size):
        in_host[i] = UInt64(i)

    # Create device buffers and copy input data
    var in_device = ctx.enqueue_create_buffer[dtype](size)
    var out_device = ctx.enqueue_create_buffer[dtype](size)
    ctx.enqueue_copy(in_device, in_host)

    # Launch kernel
    var grid_dim = ceildiv(size, BLOCK_SIZE)
    comptime kernel = warp_prefix_sum_kernel[dtype=dtype, exclusive=exclusive]
    ctx.enqueue_function[kernel](
        out_device,
        in_device,
        Int32(size),
        block_dim=BLOCK_SIZE,
        grid_dim=grid_dim,
    )

    # Copy results back and verify
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    for i in range(size):
        var expected: Scalar[dtype]

        comptime if exclusive:
            expected = UInt64(i * (i - 1) // 2)
        else:
            expected = UInt64(i * (i + 1) // 2)

        assert_equal(
            out_host[i],
            expected,
            msg=String(t"out_host[{i}] = {out_host[i]} expected = {expected}"),
        )


def block_prefix_sum_kernel[
    dtype: DType,
    block_size: Int,
    exclusive: Bool,
](
    output: MutPointer[Scalar[dtype], MutAnyOrigin],
    input: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    size_dev: Int32,
):
    var size = Int(size_dev)
    var tid = global_idx.x
    if tid >= size:
        return
    output[tid] = block.prefix_sum[exclusive=exclusive, block_size=block_size](
        input[tid]
    )


def test_block_prefix_sum[exclusive: Bool](ctx: DeviceContext) raises:
    # Initialize a block with several warps. The prefix sum for each warp is
    # tested above.
    comptime BLOCK_SIZE = WARP_SIZE * 13
    comptime size = BLOCK_SIZE

    # Allocate and initialize host memory
    var in_host = ctx.enqueue_create_host_buffer[dtype](size)
    var out_host = ctx.enqueue_create_host_buffer[dtype](size)
    for i in range(size):
        in_host[i] = UInt64(i)

    # Create device buffers and copy input data
    var in_device = ctx.enqueue_create_buffer[dtype](size)
    var out_device = ctx.enqueue_create_buffer[dtype](size)
    ctx.enqueue_copy(in_device, in_host)

    # Launch kernel
    var grid_dim = ceildiv(size, BLOCK_SIZE)
    comptime kernel = block_prefix_sum_kernel[
        dtype=dtype, block_size=BLOCK_SIZE, exclusive=exclusive
    ]
    ctx.enqueue_function[kernel](
        out_device,
        in_device,
        Int32(size),
        block_dim=BLOCK_SIZE,
        grid_dim=grid_dim,
    )

    # Copy results back and verify
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    for i in range(size):
        var expected: Scalar[dtype]

        comptime if exclusive:
            expected = UInt64(i * (i - 1) // 2)
        else:
            expected = UInt64(i * (i + 1) // 2)

        assert_equal(
            out_host[i],
            expected,
            msg=String(t"out_host[{i}] = {out_host[i]} expected = {expected}"),
        )


def main() raises:
    with DeviceContext() as ctx:
        test_warp_prefix_sum[exclusive=True](ctx)
        test_warp_prefix_sum[exclusive=False](ctx)

        test_block_prefix_sum[exclusive=True](ctx)
        test_block_prefix_sum[exclusive=False](ctx)
