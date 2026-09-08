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
from std.sys import has_amd_gpu_accelerator
import max.gpu.primitives.warp as warp
from max.gpu import thread_idx
from max.gpu.sync import barrier
from max.gpu.globals import WARP_SIZE
from max.gpu.host import DeviceContext
from max.gpu.primitives.warp import (
    shuffle_down,
    shuffle_idx,
    shuffle_up,
    shuffle_xor,
)
from std.testing import assert_equal


def kernel_wrapper[
    dtype: DType,
    simd_width: Int,
    kernel_fn: def(SIMD[dtype, simd_width]) capturing -> SIMD[
        dtype, simd_width
    ],
](device_ptr: MutPointer[Scalar[dtype], MutAnyOrigin]):
    var val = device_ptr.load[width=simd_width](thread_idx.x * simd_width)
    var result = kernel_fn(val)
    barrier()

    device_ptr.store(thread_idx.x * simd_width, result)


def _kernel_launch_helper[
    dtype: DType,
    simd_width: Int,
    kernel_fn: def(SIMD[dtype, simd_width]) capturing -> SIMD[
        dtype, simd_width
    ],
](
    host_ptr: MutPointer[Scalar[dtype], _],
    buffer_size: Int,
    block_size: Int,
    ctx: DeviceContext,
) raises:
    var device_ptr = ctx.enqueue_create_buffer[dtype](buffer_size)
    ctx.enqueue_copy(device_ptr, host_ptr)

    comptime kernel = kernel_wrapper[dtype, simd_width, kernel_fn]
    ctx.enqueue_function[kernel](device_ptr, grid_dim=1, block_dim=block_size)

    ctx.enqueue_copy(host_ptr, device_ptr)
    ctx.synchronize()
    _ = device_ptr


def _shuffle_idx_launch_helper[
    dtype: DType, simd_width: Int
](ctx: DeviceContext) raises:
    comptime block_size = WARP_SIZE
    comptime buffer_size = block_size * simd_width
    comptime constant_add: Scalar[dtype] = 42
    var host_ptr = ctx.enqueue_create_host_buffer[dtype](buffer_size)

    for i in range(buffer_size):
        host_ptr[i] = Scalar[dtype](i) + constant_add

    @__parameter
    def do_shuffle(val: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
        comptime src_lane = 0
        return shuffle_idx(val, src_lane)

    _kernel_launch_helper[dtype, simd_width, do_shuffle](
        host_ptr.unsafe_ptr(), buffer_size, block_size, ctx
    )

    for i in range(block_size):
        for j in range(simd_width):
            assert_equal(
                host_ptr[i * simd_width + j], Scalar[dtype](j) + constant_add
            )


def test_shuffle_idx_fp32(ctx: DeviceContext) raises:
    _shuffle_idx_launch_helper[.float32, 1](ctx)


def test_shuffle_idx_bf16(ctx: DeviceContext) raises:
    _shuffle_idx_launch_helper[.bfloat16, 1](ctx)


def test_shuffle_idx_bf16_packed(ctx: DeviceContext) raises:
    _shuffle_idx_launch_helper[.bfloat16, 2](ctx)


def test_shuffle_idx_fp16(ctx: DeviceContext) raises:
    _shuffle_idx_launch_helper[.float16, 1](ctx)


def test_shuffle_idx_fp16_packed(ctx: DeviceContext) raises:
    _shuffle_idx_launch_helper[.float16, 2](ctx)


def test_shuffle_idx_int64(ctx: DeviceContext) raises:
    _shuffle_idx_launch_helper[.int64, 1](ctx)


def test_shuffle_idx_int(ctx: DeviceContext) raises:
    _shuffle_idx_launch_helper[.int, 1](ctx)


def test_shuffle_idx_uint(ctx: DeviceContext) raises:
    _shuffle_idx_launch_helper[.uint, 1](ctx)


def _shuffle_up_launch_helper[
    dtype: DType, simd_width: Int
](ctx: DeviceContext) raises:
    comptime block_size = WARP_SIZE
    comptime buffer_size = block_size * simd_width
    comptime constant_add: Scalar[dtype] = 42
    comptime offset = WARP_SIZE // 2

    var host_ptr = ctx.enqueue_create_host_buffer[dtype](buffer_size)

    for i in range(buffer_size):
        host_ptr[i] = Scalar[dtype](i) + constant_add

    @__parameter
    def do_shuffle(val: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
        return shuffle_up(val, UInt32(offset))

    _kernel_launch_helper[dtype, simd_width, do_shuffle](
        host_ptr.unsafe_ptr(), buffer_size, block_size, ctx
    )

    for i in range(block_size):
        for j in range(simd_width):
            var idx = i * simd_width + j
            if i < offset:
                assert_equal(
                    host_ptr[idx],
                    Scalar[dtype](idx) + constant_add,
                )
            else:
                assert_equal(
                    host_ptr[idx],
                    Scalar[dtype](idx)
                    + constant_add
                    - Scalar[dtype]((offset * simd_width)),
                )


def test_shuffle_up_fp32(ctx: DeviceContext) raises:
    _shuffle_up_launch_helper[.float32, 1](ctx)


def test_shuffle_up_bf16(ctx: DeviceContext) raises:
    _shuffle_up_launch_helper[.bfloat16, 1](ctx)


def test_shuffle_up_bf16_packed(ctx: DeviceContext) raises:
    _shuffle_up_launch_helper[.bfloat16, 2](ctx)


def test_shuffle_up_fp16(ctx: DeviceContext) raises:
    _shuffle_up_launch_helper[.float16, 1](ctx)


def test_shuffle_up_fp16_packed(ctx: DeviceContext) raises:
    _shuffle_up_launch_helper[.float16, 2](ctx)


def test_shuffle_up_int64(ctx: DeviceContext) raises:
    _shuffle_up_launch_helper[.int64, 1](ctx)


def test_shuffle_up_int(ctx: DeviceContext) raises:
    _shuffle_up_launch_helper[.int, 1](ctx)


def test_shuffle_up_uint(ctx: DeviceContext) raises:
    _shuffle_up_launch_helper[.uint, 1](ctx)


def _shuffle_down_launch_helper[
    dtype: DType, simd_width: Int
](ctx: DeviceContext) raises:
    comptime block_size = WARP_SIZE
    comptime buffer_size = block_size * simd_width
    comptime constant_add: Scalar[dtype] = 42
    comptime offset = WARP_SIZE // 2

    var host_ptr = ctx.enqueue_create_host_buffer[dtype](buffer_size)

    for i in range(buffer_size):
        host_ptr[i] = Scalar[dtype](i) + constant_add

    @__parameter
    def do_shuffle(val: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
        return shuffle_down(val, UInt32(offset))

    _kernel_launch_helper[dtype, simd_width, do_shuffle](
        host_ptr.unsafe_ptr(), buffer_size, block_size, ctx
    )

    for i in range(block_size):
        for j in range(simd_width):
            var idx = i * simd_width + j
            if i < offset:
                assert_equal(
                    host_ptr[idx],
                    Scalar[dtype](idx)
                    + constant_add
                    + Scalar[dtype]((offset * simd_width)),
                )
            else:
                assert_equal(
                    host_ptr[idx],
                    Scalar[dtype](idx) + constant_add,
                )


def test_shuffle_down_fp32(ctx: DeviceContext) raises:
    _shuffle_down_launch_helper[.float32, 1](ctx)


def test_shuffle_down_bf16(ctx: DeviceContext) raises:
    _shuffle_down_launch_helper[.bfloat16, 1](ctx)


def test_shuffle_down_bf16_packed(ctx: DeviceContext) raises:
    _shuffle_down_launch_helper[.bfloat16, 2](ctx)


def test_shuffle_down_fp16(ctx: DeviceContext) raises:
    _shuffle_down_launch_helper[.float16, 1](ctx)


def test_shuffle_down_fp16_packed(ctx: DeviceContext) raises:
    _shuffle_down_launch_helper[.float16, 2](ctx)


def test_shuffle_down_int64(ctx: DeviceContext) raises:
    _shuffle_down_launch_helper[.int64, 1](ctx)


def test_shuffle_down_int(ctx: DeviceContext) raises:
    _shuffle_down_launch_helper[.int, 1](ctx)


def test_shuffle_down_uint(ctx: DeviceContext) raises:
    _shuffle_down_launch_helper[.uint, 1](ctx)


def _shuffle_xor_launch_helper[
    dtype: DType, simd_width: Int
](ctx: DeviceContext) raises:
    comptime block_size = WARP_SIZE
    comptime buffer_size = block_size * simd_width
    comptime constant_add: Scalar[dtype] = 42
    comptime offset = WARP_SIZE // 2

    var host_ptr = ctx.enqueue_create_host_buffer[dtype](buffer_size)

    for i in range(buffer_size):
        host_ptr[i] = Scalar[dtype](i) + constant_add

    @__parameter
    def do_shuffle(val: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
        return shuffle_xor(val, UInt32(offset))

    _kernel_launch_helper[dtype, simd_width, do_shuffle](
        host_ptr.unsafe_ptr(), buffer_size, block_size, ctx
    )

    for i in range(block_size):
        for j in range(simd_width):
            var xor_mask = (UInt32(i) ^ UInt32(offset)).cast[dtype]()
            var val = (
                xor_mask * Scalar[dtype](simd_width)
                + Scalar[dtype](j)
                + constant_add
            )
            assert_equal(host_ptr[i * simd_width + j], val)


def test_shuffle_xor_fp32(ctx: DeviceContext) raises:
    _shuffle_xor_launch_helper[.float32, 1](ctx)


def test_shuffle_xor_bf16(ctx: DeviceContext) raises:
    _shuffle_xor_launch_helper[.bfloat16, 1](ctx)


def test_shuffle_xor_bf16_packed(ctx: DeviceContext) raises:
    _shuffle_xor_launch_helper[.bfloat16, 2](ctx)


def test_shuffle_xor_fp16(ctx: DeviceContext) raises:
    _shuffle_xor_launch_helper[.float16, 1](ctx)


def test_shuffle_xor_fp16_packed(ctx: DeviceContext) raises:
    _shuffle_xor_launch_helper[.float16, 2](ctx)


def test_shuffle_xor_int64(ctx: DeviceContext) raises:
    _shuffle_xor_launch_helper[.int64, 1](ctx)


def test_shuffle_xor_int(ctx: DeviceContext) raises:
    _shuffle_xor_launch_helper[.int, 1](ctx)


def test_shuffle_xor_uint(ctx: DeviceContext) raises:
    _shuffle_xor_launch_helper[.uint, 1](ctx)


def _warp_reduce_launch_helper[
    dtype: DType,
    simd_width: Int,
](ctx: DeviceContext) raises:
    comptime block_size = WARP_SIZE
    comptime buffer_size = block_size * simd_width
    comptime offset = 1

    var host_ptr = ctx.enqueue_create_host_buffer[dtype](buffer_size)
    for i in range(buffer_size):
        host_ptr[i] = 1

    @__parameter
    def reduce_add[
        dtype: DType,
        width: SIMDLength,
    ](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return x + y

    @__parameter
    def do_warp_reduce(val: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
        return warp.reduce[shuffle_down, reduce_add](val)

    _kernel_launch_helper[dtype, simd_width, do_warp_reduce](
        host_ptr.unsafe_ptr(), buffer_size, block_size, ctx
    )

    for i in range(simd_width):
        assert_equal(host_ptr[i], Scalar[dtype](block_size))


def test_warp_reduce_fp32(ctx: DeviceContext) raises:
    _warp_reduce_launch_helper[.float32, 1](ctx)


def test_warp_reduce_bf16(ctx: DeviceContext) raises:
    _warp_reduce_launch_helper[.bfloat16, 1](ctx)


def test_warp_reduce_bf16_packed(ctx: DeviceContext) raises:
    _warp_reduce_launch_helper[.bfloat16, 2](ctx)


def test_warp_reduce_fp16(ctx: DeviceContext) raises:
    _warp_reduce_launch_helper[.float16, 1](ctx)


def test_warp_reduce_fp16_packed(ctx: DeviceContext) raises:
    _warp_reduce_launch_helper[.float16, 2](ctx)


def _warp_sum_launch_helper[
    dtype: DType,
](ctx: DeviceContext) raises:
    comptime block_size = WARP_SIZE
    var host_ptr = ctx.enqueue_create_host_buffer[dtype](block_size)
    for i in range(block_size):
        host_ptr[i] = Scalar[dtype](i)

    @__parameter
    def do_warp_sum(val: SIMD[dtype, 1]) -> SIMD[dtype, 1]:
        return warp.sum(val)

    _kernel_launch_helper[dtype, 1, do_warp_sum](
        host_ptr.unsafe_ptr(), block_size, block_size, ctx
    )

    # All lanes should have the full warp sum
    for i in range(block_size):
        assert_equal(
            host_ptr[i],
            Scalar[dtype](WARP_SIZE * (WARP_SIZE - 1) // 2),
        )


def test_warp_sum(ctx: DeviceContext) raises:
    _warp_sum_launch_helper[.float32](ctx)
    _warp_sum_launch_helper[.bfloat16](ctx)
    _warp_sum_launch_helper[.float16](ctx)
    comptime if has_amd_gpu_accelerator():
        _warp_sum_launch_helper[.float64](ctx)


def _lane_group_sum_broadcast_stride1_helper[
    dtype: DType,
    simd_width: Int,
    num_lanes: Int,
](ctx: DeviceContext) raises:
    comptime block_size = WARP_SIZE
    comptime buffer_size = block_size * simd_width

    var host_ptr = ctx.enqueue_create_host_buffer[dtype](buffer_size)
    for i in range(buffer_size):
        host_ptr[i] = Scalar[dtype](i // simd_width)

    @__parameter
    def do_reduce(
        val: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        return warp.lane_group_sum[num_lanes=num_lanes, stride=1](val)

    _kernel_launch_helper[dtype, simd_width, do_reduce](
        host_ptr.unsafe_ptr(), buffer_size, block_size, ctx
    )

    # For stride=1, thread t's group = {(t & ~(num_lanes-1)) + k : k in
    # 0..num_lanes-1}. Group sum = num_lanes*group_base +
    # num_lanes*(num_lanes-1)/2. All threads in group get broadcast result.
    for t in range(block_size):
        var group_base = Int(t) & ~(num_lanes - 1)
        var expected = Scalar[dtype](
            num_lanes * group_base + num_lanes * (num_lanes - 1) // 2
        )
        for i in range(simd_width):
            assert_equal(host_ptr[t * simd_width + i], expected)


def test_lane_group_sum_stride1(ctx: DeviceContext) raises:
    # Full warp
    _lane_group_sum_broadcast_stride1_helper[.float32, 1, WARP_SIZE](ctx)
    # Sub-warp sizes
    _lane_group_sum_broadcast_stride1_helper[.float32, 1, 2](ctx)
    _lane_group_sum_broadcast_stride1_helper[.float32, 1, 4](ctx)
    _lane_group_sum_broadcast_stride1_helper[.float32, 1, 8](ctx)
    _lane_group_sum_broadcast_stride1_helper[.float32, 1, 16](ctx)
    _lane_group_sum_broadcast_stride1_helper[.float32, 1, 32](ctx)
    # 64-bit (NVIDIA shuffle doesn't support float64)
    comptime if has_amd_gpu_accelerator():
        _lane_group_sum_broadcast_stride1_helper[.float64, 1, 4](ctx)
        _lane_group_sum_broadcast_stride1_helper[.float64, 1, WARP_SIZE](ctx)


def test_lane_group_sum_stride1_half(
    ctx: DeviceContext,
) raises:
    _lane_group_sum_broadcast_stride1_helper[.bfloat16, 1, 4](ctx)
    _lane_group_sum_broadcast_stride1_helper[.float16, 1, 4](ctx)
    _lane_group_sum_broadcast_stride1_helper[.bfloat16, 2, 4](ctx)
    _lane_group_sum_broadcast_stride1_helper[.float16, 2, 4](ctx)


def _lane_group_max_broadcast_stride1_helper[
    dtype: DType,
    simd_width: Int,
    num_lanes: Int,
](ctx: DeviceContext) raises:
    comptime block_size = WARP_SIZE
    comptime buffer_size = block_size * simd_width

    var host_ptr = ctx.enqueue_create_host_buffer[dtype](buffer_size)
    for i in range(buffer_size):
        host_ptr[i] = Scalar[dtype](i // simd_width)

    @__parameter
    def do_reduce(
        val: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        return warp.lane_group_max[num_lanes=num_lanes, stride=1](val)

    _kernel_launch_helper[dtype, simd_width, do_reduce](
        host_ptr.unsafe_ptr(), buffer_size, block_size, ctx
    )

    # For stride=1, thread t's group max = group_base + num_lanes - 1
    for t in range(block_size):
        var group_base = Int(t) & ~(num_lanes - 1)
        var expected = Scalar[dtype](group_base + num_lanes - 1)
        for i in range(simd_width):
            assert_equal(host_ptr[t * simd_width + i], expected)


def test_lane_group_max(ctx: DeviceContext) raises:
    # Full warp
    _lane_group_max_broadcast_stride1_helper[.float32, 1, WARP_SIZE](ctx)
    # Sub-warp sizes
    _lane_group_max_broadcast_stride1_helper[.float32, 1, 2](ctx)
    _lane_group_max_broadcast_stride1_helper[.float32, 1, 4](ctx)
    _lane_group_max_broadcast_stride1_helper[.float32, 1, 8](ctx)
    _lane_group_max_broadcast_stride1_helper[.float32, 1, 16](ctx)
    _lane_group_max_broadcast_stride1_helper[.float32, 1, 32](ctx)
    # Half precision
    _lane_group_max_broadcast_stride1_helper[.bfloat16, 1, 4](ctx)
    _lane_group_max_broadcast_stride1_helper[.float16, 1, 4](ctx)
    # 64-bit (NVIDIA shuffle doesn't support float64)
    comptime if has_amd_gpu_accelerator():
        _lane_group_max_broadcast_stride1_helper[.float64, 1, 4](ctx)
        _lane_group_max_broadcast_stride1_helper[.float64, 1, WARP_SIZE](ctx)


def _lane_group_reduce_launch_helper[
    dtype: DType,
    simd_width: Int,
    num_lanes: Int,
    stride: Int,
    broadcast: Bool = False,
](ctx: DeviceContext) raises:
    comptime block_size = WARP_SIZE
    comptime buffer_size = block_size * simd_width

    var host_ptr = ctx.enqueue_create_host_buffer[dtype](buffer_size)
    for i in range(buffer_size):
        host_ptr[i] = Scalar[dtype](i // simd_width)

    @__parameter
    def reduce_add[
        dtype: DType,
        width: SIMDLength,
    ](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return x + y

    @__parameter
    def do_lane_group_reduce(
        val: SIMD[dtype, simd_width]
    ) -> SIMD[dtype, simd_width]:
        comptime if broadcast:
            return warp.lane_group_sum[num_lanes=num_lanes, stride=stride](val)
        else:
            return warp.lane_group_reduce[
                shuffle_down, reduce_add, num_lanes=num_lanes, stride=stride
            ](val)

    _kernel_launch_helper[dtype, simd_width, do_lane_group_reduce](
        host_ptr.unsafe_ptr(), buffer_size, block_size, ctx
    )

    for lane in range(block_size // num_lanes):
        var lane_ = lane if not broadcast else lane % stride
        for i in range(simd_width):
            assert_equal(
                host_ptr[lane * simd_width + i],
                Scalar[dtype](
                    (num_lanes // 2) * (2 * lane_ + (num_lanes - 1) * stride)
                ),
            )


def test_lane_group_reduce_fp32(ctx: DeviceContext) raises:
    _lane_group_reduce_launch_helper[.float32, 1, 4, 8](ctx)
    _lane_group_reduce_launch_helper[.float32, 1, 4, 8, broadcast=True](ctx)

    comptime if has_amd_gpu_accelerator():
        # these two use permlane_shuffle on CDNA4+
        _lane_group_reduce_launch_helper[
            DType.float32, 1, 2, 32, broadcast=True
        ](ctx)
        _lane_group_reduce_launch_helper[
            DType.float32, 1, 4, 16, broadcast=True
        ](ctx)


def test_lane_group_reduce_bf16(ctx: DeviceContext) raises:
    _lane_group_reduce_launch_helper[.bfloat16, 1, 4, 8](ctx)


def test_lane_group_reduce_bf16_packed(ctx: DeviceContext) raises:
    _lane_group_reduce_launch_helper[.bfloat16, 2, 4, 8](ctx)


def test_lane_group_reduce_fp16(ctx: DeviceContext) raises:
    _lane_group_reduce_launch_helper[.float16, 1, 4, 8](ctx)


def test_lane_group_reduce_fp16_packed(ctx: DeviceContext) raises:
    _lane_group_reduce_launch_helper[.float16, 2, 4, 8](ctx)


def _lane_group_min_broadcast_helper[
    dtype: DType,
    simd_width: Int,
    num_lanes: Int,
    stride: Int = 1,
](ctx: DeviceContext) raises:
    comptime block_size = WARP_SIZE
    comptime buffer_size = block_size * simd_width

    var host_ptr = ctx.enqueue_create_host_buffer[dtype](buffer_size)
    for i in range(buffer_size):
        host_ptr[i] = Scalar[dtype](i // simd_width)

    @__parameter
    def do_reduce(
        val: SIMD[dtype, simd_width],
    ) -> SIMD[dtype, simd_width]:
        return warp.lane_group_min[num_lanes=num_lanes, stride=stride](val)

    _kernel_launch_helper[dtype, simd_width, do_reduce](
        host_ptr.unsafe_ptr(), buffer_size, block_size, ctx
    )

    comptime if stride == 1:
        # For stride=1, thread t's group min = group_base
        for t in range(block_size):
            var group_base = Int(t) & ~(num_lanes - 1)
            var expected = Scalar[dtype](group_base)
            for i in range(simd_width):
                assert_equal(host_ptr[t * simd_width + i], expected)
    else:
        # For stride>1, thread t's group base (= min) is:
        # (t // (num_lanes * stride)) * (num_lanes * stride) + t % stride
        for t in range(block_size):
            var group_base = (Int(t) // (num_lanes * stride)) * (
                num_lanes * stride
            ) + Int(t) % stride
            var expected = Scalar[dtype](group_base)
            for i in range(simd_width):
                assert_equal(host_ptr[t * simd_width + i], expected)


def test_lane_group_min(ctx: DeviceContext) raises:
    # Full warp
    _lane_group_min_broadcast_helper[.float32, 1, WARP_SIZE](ctx)
    # Sub-warp sizes
    _lane_group_min_broadcast_helper[.float32, 1, 2](ctx)
    _lane_group_min_broadcast_helper[.float32, 1, 4](ctx)
    _lane_group_min_broadcast_helper[.float32, 1, 8](ctx)
    _lane_group_min_broadcast_helper[.float32, 1, 16](ctx)
    _lane_group_min_broadcast_helper[.float32, 1, 32](ctx)
    # Half precision
    _lane_group_min_broadcast_helper[.bfloat16, 1, 4](ctx)
    _lane_group_min_broadcast_helper[.float16, 1, 4](ctx)
    # 64-bit (NVIDIA shuffle doesn't support float64)
    comptime if has_amd_gpu_accelerator():
        _lane_group_min_broadcast_helper[.float64, 1, 4](ctx)
        _lane_group_min_broadcast_helper[.float64, 1, WARP_SIZE](ctx)
    # Stride > 1 (exercises shuffle_xor fallback path)
    _lane_group_min_broadcast_helper[.float32, 1, 4, stride=8](ctx)
    # CDNA4 permlane path (stride=16 and stride=32)
    comptime if has_amd_gpu_accelerator():
        _lane_group_min_broadcast_helper[.float32, 1, 2, stride=32](ctx)
        _lane_group_min_broadcast_helper[.float32, 1, 4, stride=16](ctx)


def main() raises:
    with DeviceContext() as ctx:
        test_shuffle_idx_fp32(ctx)
        test_shuffle_idx_bf16(ctx)
        test_shuffle_idx_bf16_packed(ctx)
        test_shuffle_idx_fp16(ctx)
        test_shuffle_idx_fp16_packed(ctx)
        test_shuffle_idx_int64(ctx)
        test_shuffle_idx_int(ctx)
        test_shuffle_idx_uint(ctx)
        test_shuffle_up_fp32(ctx)
        test_shuffle_up_bf16(ctx)
        test_shuffle_up_bf16_packed(ctx)
        test_shuffle_up_fp16(ctx)
        test_shuffle_up_fp16_packed(ctx)
        test_shuffle_up_int64(ctx)
        test_shuffle_up_int(ctx)
        test_shuffle_up_uint(ctx)
        test_shuffle_down_fp32(ctx)
        test_shuffle_down_bf16(ctx)
        test_shuffle_down_bf16_packed(ctx)
        test_shuffle_down_fp16(ctx)
        test_shuffle_down_fp16_packed(ctx)
        test_shuffle_down_int64(ctx)
        test_shuffle_down_int(ctx)
        test_shuffle_down_uint(ctx)
        test_shuffle_xor_fp32(ctx)
        test_shuffle_xor_bf16(ctx)
        test_shuffle_xor_bf16_packed(ctx)
        test_shuffle_xor_fp16(ctx)
        test_shuffle_xor_fp16_packed(ctx)
        test_shuffle_xor_int64(ctx)
        test_shuffle_xor_int(ctx)
        test_shuffle_xor_uint(ctx)
        test_warp_reduce_fp32(ctx)
        test_warp_reduce_bf16(ctx)
        test_warp_reduce_bf16_packed(ctx)
        test_warp_reduce_fp16(ctx)
        test_warp_reduce_fp16_packed(ctx)
        test_warp_sum(ctx)
        test_lane_group_sum_stride1(ctx)
        test_lane_group_sum_stride1_half(ctx)
        test_lane_group_max(ctx)
        test_lane_group_min(ctx)
        test_lane_group_reduce_fp32(ctx)
        test_lane_group_reduce_bf16(ctx)
        test_lane_group_reduce_bf16_packed(ctx)
        test_lane_group_reduce_fp16(ctx)
        test_lane_group_reduce_fp16_packed(ctx)
