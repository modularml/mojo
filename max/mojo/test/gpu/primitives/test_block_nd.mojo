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
"""Tests for 1D, 2D, and 3D block-level GPU operations.

These tests verify that block.sum, block.max, block.min, block.broadcast, and
block.prefix_sum produce correct results when launched with 1D, 2D and 3D
thread blocks. The key invariant is that every thread sees the same broadcasted
result, and that result matches the expected value computed over all linearized
thread IDs in the block.
"""

from max.gpu import thread_idx
from max.gpu.globals import WARP_SIZE
from std.testing import assert_equal, TestSuite

from max.gpu.host import DeviceContext
from max.gpu.primitives import block

# ===-----------------------------------------------------------------------===#
# 1D block sum
# ===-----------------------------------------------------------------------===#


def block_sum_1d_kernel[
    block_size: Int,
](output: Pointer[Float32, MutAnyOrigin]):
    output[unsafe_offset=thread_idx.x] = block.sum[block_size=block_size](
        Float32(thread_idx.x)
    )


def test_block_sum_1d() raises:
    comptime N = WARP_SIZE * 4

    with DeviceContext() as ctx:
        var buf = ctx.enqueue_create_buffer[.float32](N)
        ctx.enqueue_function[block_sum_1d_kernel[N]](
            buf, grid_dim=1, block_dim=N
        )
        var result = ctx.enqueue_create_host_buffer[.float32](N)
        ctx.enqueue_copy(result, buf)
        ctx.synchronize()

        var expected = Float32(N * (N - 1) // 2)
        for i in range(N):
            assert_equal(result[i], expected)


# ===-----------------------------------------------------------------------===#
# 1D block max
# ===-----------------------------------------------------------------------===#


def block_max_1d_kernel[
    block_size: Int,
](output: Pointer[Float32, MutAnyOrigin]):
    output[unsafe_offset=thread_idx.x] = block.max[block_size=block_size](
        Float32(thread_idx.x)
    )


def test_block_max_1d() raises:
    comptime N = WARP_SIZE * 4

    with DeviceContext() as ctx:
        var buf = ctx.enqueue_create_buffer[.float32](N)
        ctx.enqueue_function[block_max_1d_kernel[N]](
            buf, grid_dim=1, block_dim=N
        )
        var result = ctx.enqueue_create_host_buffer[.float32](N)
        ctx.enqueue_copy(result, buf)
        ctx.synchronize()

        var expected = Float32(N - 1)
        for i in range(N):
            assert_equal(result[i], expected)


# ===-----------------------------------------------------------------------===#
# 1D block min
# ===-----------------------------------------------------------------------===#


def block_min_1d_kernel[
    block_size: Int,
](output: Pointer[Float32, MutAnyOrigin]):
    # Offset by 1 so the minimum is 1, avoiding confusion with the neutral 0.
    output[unsafe_offset=thread_idx.x] = block.min[block_size=block_size](
        Float32(thread_idx.x + 1)
    )


def test_block_min_1d() raises:
    comptime N = WARP_SIZE * 4

    with DeviceContext() as ctx:
        var buf = ctx.enqueue_create_buffer[.float32](N)
        ctx.enqueue_function[block_min_1d_kernel[N]](
            buf, grid_dim=1, block_dim=N
        )
        var result = ctx.enqueue_create_host_buffer[.float32](N)
        ctx.enqueue_copy(result, buf)
        ctx.synchronize()

        for i in range(N):
            assert_equal(result[i], Float32(1))


# ===-----------------------------------------------------------------------===#
# 1D block broadcast
# ===-----------------------------------------------------------------------===#


def block_broadcast_1d_kernel[
    block_size: Int,
    src_thread: Int,
](output: Pointer[Float32, MutAnyOrigin]):
    output[unsafe_offset=thread_idx.x] = block.broadcast[block_size=block_size](
        Float32(thread_idx.x), src_thread
    )


def test_block_broadcast_1d() raises:
    comptime N = WARP_SIZE * 4
    # Source thread in the second warp.
    comptime src = WARP_SIZE + 1

    with DeviceContext() as ctx:
        var buf = ctx.enqueue_create_buffer[.float32](N)
        ctx.enqueue_function[block_broadcast_1d_kernel[N, src]](
            buf, grid_dim=1, block_dim=N
        )
        var result = ctx.enqueue_create_host_buffer[.float32](N)
        ctx.enqueue_copy(result, buf)
        ctx.synchronize()

        for i in range(N):
            assert_equal(result[i], Float32(src))


# ===-----------------------------------------------------------------------===#
# 1D block prefix_sum
# ===-----------------------------------------------------------------------===#


def block_prefix_sum_1d_kernel[
    block_size: Int,
](output: Pointer[Float32, MutAnyOrigin]):
    # All threads contribute 1, so the inclusive prefix sum at position k
    # equals k+1.
    output[unsafe_offset=thread_idx.x] = block.prefix_sum[
        block_size=block_size
    ](Float32(1))


def test_block_prefix_sum_1d() raises:
    comptime N = WARP_SIZE * 4

    with DeviceContext() as ctx:
        var buf = ctx.enqueue_create_buffer[.float32](N)
        ctx.enqueue_function[block_prefix_sum_1d_kernel[N]](
            buf, grid_dim=1, block_dim=N
        )
        var result = ctx.enqueue_create_host_buffer[.float32](N)
        ctx.enqueue_copy(result, buf)
        ctx.synchronize()

        for i in range(N):
            assert_equal(result[i], Float32(i + 1))


# ===-----------------------------------------------------------------------===#
# 2D block sum
# ===-----------------------------------------------------------------------===#


def block_sum_2d_kernel[
    block_dim_x: Int,
    block_dim_y: Int,
](output: Pointer[Float32, MutAnyOrigin]):
    """Each thread holds its linearized ID; the block sum is broadcast back."""
    var linear_tid = thread_idx.x + thread_idx.y * block_dim_x
    output[unsafe_offset=linear_tid] = block.sum[
        block_dim_x=block_dim_x, block_dim_y=block_dim_y
    ](Float32(linear_tid))


def test_block_sum_2d() raises:
    comptime BX = WARP_SIZE
    comptime BY = 4
    comptime total = BX * BY

    with DeviceContext() as ctx:
        var buf = ctx.enqueue_create_buffer[.float32](total)
        ctx.enqueue_function[block_sum_2d_kernel[BX, BY]](
            buf, grid_dim=1, block_dim=(BX, BY)
        )
        var result = ctx.enqueue_create_host_buffer[.float32](total)
        ctx.enqueue_copy(result, buf)
        ctx.synchronize()

        var expected = Float32(total * (total - 1) // 2)
        for i in range(total):
            assert_equal(result[i], expected)


# ===-----------------------------------------------------------------------===#
# 3D block sum
# ===-----------------------------------------------------------------------===#


def block_sum_3d_kernel[
    block_dim_x: Int,
    block_dim_y: Int,
    block_dim_z: Int,
](output: Pointer[Float32, MutAnyOrigin]):
    var linear_tid = (
        thread_idx.x
        + thread_idx.y * block_dim_x
        + thread_idx.z * block_dim_x * block_dim_y
    )
    output[unsafe_offset=linear_tid] = block.sum[
        block_dim_x=block_dim_x,
        block_dim_y=block_dim_y,
        block_dim_z=block_dim_z,
    ](Float32(linear_tid))


def test_block_sum_3d() raises:
    comptime BX = WARP_SIZE
    comptime BY = 2
    comptime BZ = 2
    comptime total = BX * BY * BZ

    with DeviceContext() as ctx:
        var buf = ctx.enqueue_create_buffer[.float32](total)
        ctx.enqueue_function[block_sum_3d_kernel[BX, BY, BZ]](
            buf, grid_dim=1, block_dim=(BX, BY, BZ)
        )
        var result = ctx.enqueue_create_host_buffer[.float32](total)
        ctx.enqueue_copy(result, buf)
        ctx.synchronize()

        var expected = Float32(total * (total - 1) // 2)
        for i in range(total):
            assert_equal(result[i], expected)


# ===-----------------------------------------------------------------------===#
# 2D block max
# ===-----------------------------------------------------------------------===#


def block_max_2d_kernel[
    block_dim_x: Int,
    block_dim_y: Int,
](output: Pointer[Float32, MutAnyOrigin]):
    var linear_tid = thread_idx.x + thread_idx.y * block_dim_x
    output[unsafe_offset=linear_tid] = block.max[
        block_dim_x=block_dim_x, block_dim_y=block_dim_y
    ](Float32(linear_tid))


def test_block_max_2d() raises:
    comptime BX = WARP_SIZE
    comptime BY = 4
    comptime total = BX * BY

    with DeviceContext() as ctx:
        var buf = ctx.enqueue_create_buffer[.float32](total)
        ctx.enqueue_function[block_max_2d_kernel[BX, BY]](
            buf, grid_dim=1, block_dim=(BX, BY)
        )
        var result = ctx.enqueue_create_host_buffer[.float32](total)
        ctx.enqueue_copy(result, buf)
        ctx.synchronize()

        var expected = Float32(total - 1)
        for i in range(total):
            assert_equal(result[i], expected)


# ===-----------------------------------------------------------------------===#
# 2D block min
# ===-----------------------------------------------------------------------===#


def block_min_2d_kernel[
    block_dim_x: Int,
    block_dim_y: Int,
](output: Pointer[Float32, MutAnyOrigin]):
    # Offset by 1 so the minimum is 1, avoiding confusion with the neutral 0.
    var linear_tid = thread_idx.x + thread_idx.y * block_dim_x
    output[unsafe_offset=linear_tid] = block.min[
        block_dim_x=block_dim_x, block_dim_y=block_dim_y
    ](Float32(linear_tid + 1))


def test_block_min_2d() raises:
    comptime BX = WARP_SIZE
    comptime BY = 4
    comptime total = BX * BY

    with DeviceContext() as ctx:
        var buf = ctx.enqueue_create_buffer[.float32](total)
        ctx.enqueue_function[block_min_2d_kernel[BX, BY]](
            buf, grid_dim=1, block_dim=(BX, BY)
        )
        var result = ctx.enqueue_create_host_buffer[.float32](total)
        ctx.enqueue_copy(result, buf)
        ctx.synchronize()

        for i in range(total):
            assert_equal(result[i], Float32(1))


# ===-----------------------------------------------------------------------===#
# 2D block broadcast
# ===-----------------------------------------------------------------------===#


def block_broadcast_2d_kernel[
    block_dim_x: Int,
    block_dim_y: Int,
    src_thread: Int,
](output: Pointer[Float32, MutAnyOrigin]):
    var linear_tid = thread_idx.x + thread_idx.y * block_dim_x
    # Each thread offers its own linearized ID; only src_thread's value
    # should be broadcast to everyone.
    output[unsafe_offset=linear_tid] = block.broadcast[
        block_dim_x=block_dim_x, block_dim_y=block_dim_y
    ](Float32(linear_tid), src_thread)


def test_block_broadcast_2d() raises:
    comptime BX = WARP_SIZE
    comptime BY = 4
    comptime total = BX * BY
    # Source thread in the second row, second lane (linearized ID = 33).
    comptime src: Int = 33

    with DeviceContext() as ctx:
        var buf = ctx.enqueue_create_buffer[.float32](total)
        ctx.enqueue_function[block_broadcast_2d_kernel[BX, BY, src]](
            buf, grid_dim=1, block_dim=(BX, BY)
        )
        var result = ctx.enqueue_create_host_buffer[.float32](total)
        ctx.enqueue_copy(result, buf)
        ctx.synchronize()

        for i in range(total):
            assert_equal(result[i], Float32(src))


# ===-----------------------------------------------------------------------===#
# 2D block prefix_sum
# ===-----------------------------------------------------------------------===#


def block_prefix_sum_2d_kernel[
    block_dim_x: Int,
    block_dim_y: Int,
](output: Pointer[Float32, MutAnyOrigin]):
    var linear_tid = thread_idx.x + thread_idx.y * block_dim_x
    # All threads contribute 1, so the inclusive prefix sum at linearized
    # position k equals k+1.
    output[unsafe_offset=linear_tid] = block.prefix_sum[
        block_dim_x=block_dim_x, block_dim_y=block_dim_y
    ](Float32(1))


def test_block_prefix_sum_2d() raises:
    comptime BX = WARP_SIZE
    comptime BY = 4
    comptime total = BX * BY

    with DeviceContext() as ctx:
        var buf = ctx.enqueue_create_buffer[.float32](total)
        ctx.enqueue_function[block_prefix_sum_2d_kernel[BX, BY]](
            buf, grid_dim=1, block_dim=(BX, BY)
        )
        var result = ctx.enqueue_create_host_buffer[.float32](total)
        ctx.enqueue_copy(result, buf)
        ctx.synchronize()

        for i in range(total):
            assert_equal(result[i], Float32(i + 1))


# ===-----------------------------------------------------------------------===#
# Main
# ===-----------------------------------------------------------------------===#


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
