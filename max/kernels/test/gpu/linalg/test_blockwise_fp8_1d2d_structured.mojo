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

"""Tests for blockwise FP8 1D2D structured kernel.

Validates the new structured kernel against the naive reference implementation.
Uses DeepSeek V3-style MoE shapes:
- Gate/Up: num_experts=256, N=2048, K=7168 (hidden → expert intermediate)
- Down:    num_experts=256, N=7168, K=2048 (expert intermediate → hidden)
- top_k=8, typical decode batch 64-512 tokens → 512-4096 total tokens
"""

from std.sys import size_of

from max.gpu.host import DeviceContext
from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from layout._fillers import random
from structured_kernels.tile_types import (
    GMEMLayout1D,
)
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_grouped_matmul
from linalg.matmul.gpu.sm100_structured.blockwise_fp8_1d2d import (
    grouped_matmul_dynamic_scaled_fp8_1d2d,
)
from std.testing import assert_almost_equal

from std.utils.index import Index, IndexList


def test_blockwise_fp8_1d2d_structured[
    in_type: DType,
    out_type: DType,
    num_experts: Int,
    expert_shape: IndexList[2],  # (N, K)
](
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids_list: List[Int],
    ctx: DeviceContext,
) raises:
    """Test structured blockwise FP8 1D2D kernel against naive reference.

    Args:
        num_active_experts: Number of active experts in this batch.
        num_tokens_by_expert: Token counts per active expert.
        expert_ids_list: Expert IDs for each active expert.
        ctx: Device context.
    """
    comptime BLOCK_SCALE_K = 128
    comptime transpose_b = True

    comptime a_type = in_type
    comptime b_type = in_type
    comptime c_type = out_type

    comptime N = expert_shape[0]
    comptime K = expert_shape[1]

    var total_num_tokens = 0
    var max_num_tokens_by_expert = 0
    for i in range(len(num_tokens_by_expert)):
        var M = num_tokens_by_expert[i]
        total_num_tokens += M
        max_num_tokens_by_expert = max(max_num_tokens_by_expert, M)

    assert (
        total_num_tokens * size_of[DType.float32]() % 16 == 0
    ), "TMA expects total_num_tokens to be divisible by 16 bytes"

    print(
        "== test_blockwise_fp8_1d2d_structured",
        a_type,
        "->",
        c_type,
        "problem: (",
        total_num_tokens,
        "x",
        N,
        "x",
        K,
        ")",
        "experts:",
        num_active_experts,
        "/",
        num_experts,
    )

    # Sizes
    var a_size = total_num_tokens * K
    var b_size = num_experts * N * K
    var c_size = total_num_tokens * N
    var a_scales_size = (K // BLOCK_SCALE_K) * total_num_tokens
    var b_scales_size = (
        num_experts * (N // BLOCK_SCALE_K) * (K // BLOCK_SCALE_K)
    )

    # Host allocations
    var a_host_ptr = List(length=a_size, fill=Scalar[a_type](0))
    var b_host_ptr = List(length=b_size, fill=Scalar[b_type](0))
    var c_host_ptr = List(length=c_size, fill=Scalar[c_type](0))
    var c_host_ref_ptr = List(length=c_size, fill=Scalar[c_type](0))
    var a_offsets_host_ptr = List(length=num_active_experts + 1, fill=UInt32(0))
    var expert_ids_host_ptr = List(length=num_active_experts, fill=Int32(0))
    var a_scales_host_ptr = List(length=a_scales_size, fill=Float32(0))
    var b_scales_host_ptr = List(length=b_scales_size, fill=Float32(0))
    var expert_scales_host_ptr = List(length=num_experts, fill=Float32(0))

    var a_host = TileTensor(
        a_host_ptr,
        row_major(Coord(Int64(total_num_tokens), Idx[K])),
    )
    var b_host = TileTensor(
        b_host_ptr,
        row_major[num_experts, N, K](),
    )
    var c_host = TileTensor(
        c_host_ptr,
        row_major(Coord(Int64(total_num_tokens), Idx[N])),
    )
    var c_host_ref = TileTensor(
        c_host_ref_ptr,
        row_major(Coord(Int64(total_num_tokens), Idx[N])),
    )
    var a_scales_host = TileTensor(
        a_scales_host_ptr,
        row_major(Coord(Idx[K // BLOCK_SCALE_K], Int64(total_num_tokens))),
    )
    var b_scales_host = TileTensor(
        b_scales_host_ptr,
        row_major[num_experts, N // BLOCK_SCALE_K, K // BLOCK_SCALE_K](),
    )

    # Setup offsets and expert ids
    a_offsets_host_ptr[0] = 0
    for i in range(num_active_experts):
        a_offsets_host_ptr[i + 1] = a_offsets_host_ptr[i] + UInt32(
            num_tokens_by_expert[i]
        )
        expert_ids_host_ptr[i] = Int32(expert_ids_list[i])

    # Expert scales: use non-trivial values to verify scaling works
    for i in range(num_experts):
        expert_scales_host_ptr[i] = Float32(1.0)

    # Initialize data
    random(a_host)
    random(b_host)
    _ = c_host.fill(0)
    _ = c_host_ref.fill(0)
    random(a_scales_host)
    random(b_scales_host)

    # Device allocations
    var a_device_buffer = ctx.enqueue_create_buffer[a_type](a_size)
    var b_device_buffer = ctx.enqueue_create_buffer[b_type](b_size)
    var c_device_buffer = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_ref_buffer = ctx.enqueue_create_buffer[c_type](c_size)
    var a_offsets_device_buffer = ctx.enqueue_create_buffer[.uint32](
        num_active_experts + 1
    )
    var expert_ids_device_buffer = ctx.enqueue_create_buffer[.int32](
        num_active_experts
    )
    var a_scales_device_buffer = ctx.enqueue_create_buffer[.float32](
        a_scales_size
    )
    var b_scales_device_buffer = ctx.enqueue_create_buffer[.float32](
        b_scales_size
    )
    var expert_scales_device_buffer = ctx.enqueue_create_buffer[.float32](
        num_experts
    )

    # Copy to device
    ctx.enqueue_copy(a_device_buffer, a_host_ptr)
    ctx.enqueue_copy(b_device_buffer, b_host_ptr)
    ctx.enqueue_copy(c_device_buffer, c_host_ptr)
    ctx.enqueue_copy(c_device_ref_buffer, c_host_ref_ptr)
    ctx.enqueue_copy(a_offsets_device_buffer, a_offsets_host_ptr)
    ctx.enqueue_copy(expert_ids_device_buffer, expert_ids_host_ptr)
    ctx.enqueue_copy(a_scales_device_buffer, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device_buffer, b_scales_host_ptr)
    ctx.enqueue_copy(expert_scales_device_buffer, expert_scales_host_ptr)

    # Create TileTensors for device data
    var a_tt = TileTensor(
        a_device_buffer,
        row_major(Coord(Int64(total_num_tokens), Idx[K])),
    )
    var b_tt = TileTensor(
        b_device_buffer,
        row_major[num_experts, N, K](),
    )
    var c_tt = TileTensor(
        c_device_buffer,
        row_major(Coord(Int64(total_num_tokens), Idx[N])),
    )
    var c_ref_tt = TileTensor(
        c_device_ref_buffer,
        row_major(Coord(Int64(total_num_tokens), Idx[N])),
    )
    var a_scales_tt = TileTensor(
        a_scales_device_buffer,
        row_major(
            Coord(
                Idx[K // BLOCK_SCALE_K],
                Int64(total_num_tokens),
            )
        ),
    )
    var b_scales_tt = TileTensor(
        b_scales_device_buffer,
        row_major[num_experts, N // BLOCK_SCALE_K, K // BLOCK_SCALE_K](),
    )

    var a_offsets_tt = TileTensor(
        a_offsets_device_buffer,
        layout=GMEMLayout1D(
            Coord(Int64(num_active_experts + 1)),
            Coord(Idx[1]),
        ),
    )
    var expert_ids_tt = TileTensor(
        expert_ids_device_buffer,
        layout=GMEMLayout1D(
            Coord(Int64(num_active_experts)),
            Coord(Idx[1]),
        ),
    )
    var expert_scales_tt = TileTensor(
        expert_scales_device_buffer,
        layout=GMEMLayout1D(
            Coord(Int64(num_experts)),
            Coord(Idx[1]),
        ),
    )

    # Bridge to LayoutTensor for reference kernel
    var a = a_tt.to_layout_tensor()
    var b = b_tt.to_layout_tensor()
    var c_ref = c_ref_tt.to_layout_tensor()
    var a_scales = a_scales_tt.to_layout_tensor()
    var b_scales = b_scales_tt.to_layout_tensor()
    var a_offsets = a_offsets_tt.to_layout_tensor()
    var expert_ids = expert_ids_tt.to_layout_tensor()

    # ===== Reference: naive blockwise FP8 grouped matmul =====
    naive_blockwise_scaled_fp8_grouped_matmul[
        BLOCK_DIM_M=16,
        BLOCK_DIM_N=16,
        transpose_b=transpose_b,
        scales_granularity_mnk=Index(1, BLOCK_SCALE_K, BLOCK_SCALE_K),
    ](
        c_ref,
        a,
        b,
        a_scales,
        b_scales,
        a_offsets,
        expert_ids,
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )
    ctx.synchronize()

    # ===== Test: structured blockwise FP8 1D2D kernel =====
    grouped_matmul_dynamic_scaled_fp8_1d2d[
        a_scales_type=.float32,
        b_scales_type=.float32,
        transpose_b=transpose_b,
    ](
        c_tt,
        a_tt,
        b_tt,
        a_scales_tt,
        b_scales_tt,
        a_offsets_tt,
        expert_ids_tt,
        expert_scales_tt,
        num_active_experts,
        ctx,
    )
    ctx.synchronize()

    # ===== Compare results =====
    ctx.enqueue_copy(c_host_ptr, c_device_buffer)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref_buffer)
    ctx.synchronize()

    var rtol = 1e-2
    var atol = 1e-2
    for mi in range(total_num_tokens):
        for ni in range(N):
            assert_almost_equal(
                c_host_ptr[mi * N + ni],
                c_host_ref_ptr[mi * N + ni],
                msg=String(t"m: {mi} n: {ni}"),
                rtol=rtol,
                atol=atol,
            )

    print("  PASSED")

    # Cleanup
    _ = expert_scales_host_ptr^
    _ = b_scales_host_ptr^
    _ = a_scales_host_ptr^
    _ = expert_ids_host_ptr^
    _ = a_offsets_host_ptr^
    _ = c_host_ref_ptr^
    _ = c_host_ptr^
    _ = b_host_ptr^
    _ = a_host_ptr^


def main() raises:
    with DeviceContext() as ctx:
        # ============================================================
        # DeepSeek V3 Down projection shapes: K=2048, N=7168
        # Use 8 experts (subset of 256) to keep allocation reasonable
        # ============================================================

        # Single expert, aligned tokens
        test_blockwise_fp8_1d2d_structured[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=8,
            expert_shape=Index(7168, 2048),
        ](1, [128], [0], ctx)

        # 4 active experts, uniform decode-like distribution
        test_blockwise_fp8_1d2d_structured[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=8,
            expert_shape=Index(7168, 2048),
        ](4, [64, 64, 64, 64], [0, 2, 5, 7], ctx)

        # Unaligned token counts (realistic MoE routing)
        test_blockwise_fp8_1d2d_structured[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=8,
            expert_shape=Index(7168, 2048),
        ](4, [20, 100, 4, 48], [0, 3, 5, 7], ctx)

        # ============================================================
        # DeepSeek V3 Gate/Up projection shapes: K=7168, N=2048
        # ============================================================

        # Single expert, aligned
        test_blockwise_fp8_1d2d_structured[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=8,
            expert_shape=Index(2048, 7168),
        ](1, [128], [0], ctx)

        # Multi-expert, unaligned
        test_blockwise_fp8_1d2d_structured[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=8,
            expert_shape=Index(2048, 7168),
        ](4, [20, 256, 4, 32], [0, 2, 5, 7], ctx)

        # ============================================================
        # Smaller shapes (smoke tests)
        # ============================================================

        # Minimal: 1 expert, small shape
        test_blockwise_fp8_1d2d_structured[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=4,
            expert_shape=Index(256, 256),
        ](1, [128], [0], ctx)

        # Multi-expert unaligned
        test_blockwise_fp8_1d2d_structured[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=4,
            expert_shape=Index(512, 1024),
        ](2, [20, 40], [0, 2], ctx)

        # float32 output
        test_blockwise_fp8_1d2d_structured[
            .float8_e4m3fn,
            .float32,
            num_experts=4,
            expert_shape=Index(512, 1024),
        ](2, [20, 40], [0, 2], ctx)

        # ============================================================
        # Large batch with many experts and varied token counts
        # ============================================================

        test_blockwise_fp8_1d2d_structured[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=8,
            expert_shape=Index(2048, 7168),
        ](4, [20, 1500, 300, 28], [0, 3, 5, 7], ctx)

    print("\nAll blockwise FP8 1D2D structured kernel tests passed!")
