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

"""Test edge cases for grouped_matmul_dynamic_scaled_fp8.

This test focuses on the zero edge cases where num_active_experts or
max_num_tokens_per_expert are zero. The function should return early without
error in these cases.

There are more comprehensive non-zero test cases for
grouped_matmul_sm100_blockwise_scaled_fp8 in
test_grouped_matmul_sm100_blockwise_fp8.mojo.
"""

from max.gpu.host import DeviceContext
from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from layout._fillers import random
from linalg.grouped_matmul_sm100_blockwise_fp8 import (
    grouped_matmul_dynamic_scaled_fp8,
)


def test_grouped_matmul_dynamic_scaled_fp8_zero_edge_case[
    num_experts: Int = 4,
    N: Int = 256,
    K: Int = 256,
](
    num_active_experts: Int,
    max_num_tokens_per_expert: Int,
    ctx: DeviceContext,
) raises:
    """Test grouped_matmul_dynamic_scaled_fp8 with zero edge cases.

    This test verifies that the function returns early without errors when
    either num_active_experts or max_num_tokens_per_expert (or both) are 0.

    Args:
        num_active_experts: Number of active experts (can be 0).
        max_num_tokens_per_expert: Maximum tokens per expert (can be 0).
        ctx: Device context for GPU operations.
    """
    comptime in_type = DType.float8_e4m3fn
    comptime out_type = DType.bfloat16
    comptime BLOCK_SCALE_K = 128

    print(
        "== test_grouped_matmul_dynamic_scaled_fp8_zero_edge_case",
        "num_experts:",
        num_experts,
        "N:",
        N,
        "K:",
        K,
        "num_active_experts:",
        num_active_experts,
        "max_num_tokens_per_expert:",
        max_num_tokens_per_expert,
    )

    # Use minimal buffer size for efficiency since nothing will execute
    var total_tokens = max(max_num_tokens_per_expert, 16)
    var num_offsets = max(num_active_experts + 1, 1)
    var num_expert_ids = max(num_active_experts, 0)

    # Create host buffers
    var a_size = total_tokens * K

    var b_size = num_experts * N * K

    var c_size = total_tokens * N

    var a_host_ptr = ctx.enqueue_create_host_buffer[in_type](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[in_type](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[out_type](c_size)

    var a_host = TileTensor(
        a_host_ptr,
        row_major(Coord(total_tokens, Idx[K])),
    )
    var b_host = TileTensor(
        b_host_ptr,
        row_major[num_experts, N, K](),
    )

    # Create offsets and expert_ids
    var a_offsets_host_ptr = ctx.enqueue_create_host_buffer[.uint32](
        num_offsets
    )
    var expert_ids_host_ptr = ctx.enqueue_create_host_buffer[.int32](
        num_expert_ids
    )

    # Set up offsets
    for i in range(num_offsets):
        a_offsets_host_ptr[i] = 0

    # Set up expert_ids
    for i in range(num_expert_ids):
        expert_ids_host_ptr[i] = Int32(i % num_experts)

    # Create scale buffers
    var a_scales_size = (K // BLOCK_SCALE_K) * total_tokens

    var b_scales_size = (
        num_experts * (N // BLOCK_SCALE_K) * (K // BLOCK_SCALE_K)
    )

    var a_scales_host_ptr = ctx.enqueue_create_host_buffer[.float32](
        a_scales_size
    )
    var b_scales_host_ptr = ctx.enqueue_create_host_buffer[.float32](
        b_scales_size
    )

    var a_scales_host = TileTensor(
        a_scales_host_ptr,
        row_major(Coord(Idx[K // BLOCK_SCALE_K], total_tokens)),
    )
    var b_scales_host = TileTensor(
        b_scales_host_ptr,
        row_major[num_experts, N // BLOCK_SCALE_K, K // BLOCK_SCALE_K](),
    )

    # Initialize with random data
    random(a_host)
    random(b_host)
    random(a_scales_host)
    random(b_scales_host)

    # Create device buffers
    var a_device_buffer = ctx.enqueue_create_buffer[in_type](a_size)
    var b_device_buffer = ctx.enqueue_create_buffer[in_type](b_size)
    var c_device_buffer = ctx.enqueue_create_buffer[out_type](c_size)
    var a_offsets_device_buffer = ctx.enqueue_create_buffer[.uint32](
        num_offsets
    )
    var expert_ids_device_buffer = ctx.enqueue_create_buffer[.int32](
        num_expert_ids
    )
    var a_scales_device_buffer = ctx.enqueue_create_buffer[.float32](
        a_scales_size
    )
    var b_scales_device_buffer = ctx.enqueue_create_buffer[.float32](
        b_scales_size
    )

    var a_device = TileTensor(
        a_device_buffer,
        row_major(Coord(total_tokens, Idx[K])),
    )
    var b_device = TileTensor(
        b_device_buffer,
        row_major(Coord(Idx[num_experts], Idx[N], Idx[K])),
    )
    var c_device = TileTensor(
        c_device_buffer,
        row_major(Coord(total_tokens, Idx[N])),
    )
    var a_offsets_device = TileTensor(
        a_offsets_device_buffer,
        row_major(Coord(num_offsets)),
    )
    var expert_ids_device = TileTensor(
        expert_ids_device_buffer,
        row_major(Coord(num_expert_ids)),
    )
    var a_scales_device = TileTensor(
        a_scales_device_buffer,
        row_major(Coord(Idx[K // BLOCK_SCALE_K], total_tokens)),
    )
    var b_scales_device = TileTensor(
        b_scales_device_buffer,
        row_major(
            Coord(
                Idx[num_experts],
                Idx[N // BLOCK_SCALE_K],
                Idx[K // BLOCK_SCALE_K],
            )
        ),
    )

    # Copy to device
    ctx.enqueue_copy(a_device_buffer, a_host_ptr)
    ctx.enqueue_copy(b_device_buffer, b_host_ptr)
    ctx.enqueue_copy(a_offsets_device_buffer, a_offsets_host_ptr)
    if num_expert_ids > 0:
        ctx.enqueue_copy(expert_ids_device_buffer, expert_ids_host_ptr)
    ctx.enqueue_copy(a_scales_device_buffer, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device_buffer, b_scales_host_ptr)

    # Call with the specified zero edge case parameters
    # This should return early without error
    grouped_matmul_dynamic_scaled_fp8[
        input_scale_granularity="block",
        weight_scale_granularity="block",
        m_scale_granularity=1,
        n_scale_granularity=BLOCK_SCALE_K,
        k_scale_granularity=BLOCK_SCALE_K,
        transpose_b=True,
    ](
        c_device,
        a_device,
        b_device,
        a_scales_device,
        b_scales_device,
        a_offsets_device,
        expert_ids_device,
        max_num_tokens_per_expert=max_num_tokens_per_expert,
        num_active_experts=num_active_experts,
        ctx=ctx,
    )

    ctx.synchronize()
    print("  ✓ Successfully handled edge case")

    # Cleanup
    _ = a_device_buffer^
    _ = b_device_buffer^
    _ = c_device_buffer^
    _ = a_offsets_device_buffer^
    _ = expert_ids_device_buffer^
    _ = a_scales_device_buffer^
    _ = b_scales_device_buffer^


def main() raises:
    """Run all edge case tests for grouped_matmul_dynamic_scaled_fp8."""
    with DeviceContext() as ctx:
        # Test zero num_active_experts (with non-zero max_num_tokens_per_expert)
        test_grouped_matmul_dynamic_scaled_fp8_zero_edge_case[
            num_experts=4,
            N=256,
            K=256,
        ](num_active_experts=0, max_num_tokens_per_expert=64, ctx=ctx)

        # Test zero max_num_tokens_per_expert (with non-zero num_active_experts)
        test_grouped_matmul_dynamic_scaled_fp8_zero_edge_case[
            num_experts=4,
            N=256,
            K=256,
        ](num_active_experts=2, max_num_tokens_per_expert=0, ctx=ctx)

        # Test both zero
        test_grouped_matmul_dynamic_scaled_fp8_zero_edge_case[
            num_experts=4,
            N=256,
            K=256,
        ](num_active_experts=0, max_num_tokens_per_expert=0, ctx=ctx)

    print("\n✓ All edge case tests passed!")
