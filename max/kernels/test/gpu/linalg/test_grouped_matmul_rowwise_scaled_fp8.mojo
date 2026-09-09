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
"""Numerical-equivalence test for the rowwise / per-token grouped FP8 matmul.

Target: NVIDIA SM100 (B200). Validates
``grouped_matmul_rowwise_dynamic_scaled_fp8`` (the kernel under test, KUT)
against the in-tree naive grouped FP8 reference
``naive_blockwise_scaled_fp8_grouped_matmul`` configured with
``scales_granularity_mnk = (1, 1, K)`` - i.e. one activation scale per token
and one weight scale per (expert, output-channel). The two paths are distinct
code: the reference reloads + multiplies the (constant-over-K) scales inside
the K-loop, while the KUT applies them once after the full-K fp32 reduction.

Note the scale-tensor *layout* convention differs between the two, which is
the point of the cross-check:
  * reference ``a_scales``: ``[K // K = 1, total_tokens]`` (K-block x token)
  * KUT ``a_scales``:       ``[total_tokens, 1]``           (token x 1)
  * both ``b_scales``:      ``[num_experts, N, 1]``          ((expert,n) x 1)

Covered cases: single expert, partial last M-tile, a zero-token expert in a
multi-expert ragged batch, sparse expert-id routing, and a production-ish
Llama-4-Scout gate/up shape.
"""

from std.collections import Optional
from std.sys import align_of

from max.gpu.host import DeviceContext
from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from layout._fillers import random
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_grouped_matmul
from linalg.grouped_matmul import grouped_matmul_rowwise_dynamic_scaled_fp8
from std.testing import assert_almost_equal

from std.utils.index import Index, IndexList


def test_grouped_matmul_rowwise_scaled_fp8[
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
    comptime transpose_b = True
    comptime scales_type = DType.float32

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

    print(
        "== test_grouped_matmul_rowwise_scaled_fp8",
        a_type,
        "->",
        c_type,
        "problem shape: (",
        total_num_tokens,
        "x",
        N,
        "x",
        K,
        ") num_active_experts:",
        num_active_experts,
    )

    # Sizes.
    var a_size = total_num_tokens * K
    var b_size = num_experts * N * K
    var c_size = total_num_tokens * N
    # KUT layout: per-token, [T, 1].
    var a_scales_kut_size = total_num_tokens
    # Reference layout: [1, T] (K-block count is 1).
    var a_scales_ref_size = total_num_tokens
    # Both b_scales layouts are [E, N, 1].
    var b_scales_size = num_experts * N

    # KUT TileTensor shapes.
    var a_tt_shape = row_major(Coord(Int(total_num_tokens), Idx[K]))
    var b_tt_shape = row_major(Coord(Idx[num_experts], Idx[N], Idx[K]))
    var c_tt_shape = row_major(Coord(Int(total_num_tokens), Idx[N]))
    var a_scales_kut_shape = row_major(Coord(Int(total_num_tokens), Idx[1]))
    var a_scales_ref_shape = row_major(Coord(Idx[1], Int(total_num_tokens)))
    var b_scales_shape = row_major(Coord(Idx[num_experts], Idx[N], Idx[1]))

    # Host allocations.
    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var a_offsets_host_ptr = ctx.enqueue_create_host_buffer[.uint32](
        num_active_experts + 1
    )
    var expert_ids_host_ptr = ctx.enqueue_create_host_buffer[.int32](
        num_active_experts
    )
    var a_scales_kut_host_ptr = ctx.enqueue_create_host_buffer[scales_type](
        a_scales_kut_size
    )
    var a_scales_ref_host_ptr = ctx.enqueue_create_host_buffer[scales_type](
        a_scales_ref_size
    )
    var b_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_type](
        b_scales_size
    )

    var a_host = TileTensor(a_host_ptr, a_tt_shape)
    var b_host = TileTensor(b_host_ptr, b_tt_shape)
    var c_host = TileTensor(c_host_ptr, c_tt_shape)
    var c_host_ref = TileTensor(c_host_ref_ptr, c_tt_shape)

    # Offsets and expert ids.
    a_offsets_host_ptr[0] = 0
    for i in range(num_active_experts):
        a_offsets_host_ptr[i + 1] = a_offsets_host_ptr[i] + UInt32(
            num_tokens_by_expert[i]
        )
        expert_ids_host_ptr[i] = Int32(expert_ids_list[i])

    random(a_host)
    random(b_host)
    _ = c_host.fill(0)
    _ = c_host_ref.fill(0)

    # Fill per-token activation scales once, then mirror into both layouts.
    # KUT a_scales is [T, 1]; reference a_scales is [1, T]. Same values.
    var a_scales_kut_host = TileTensor(
        a_scales_kut_host_ptr, a_scales_kut_shape
    )
    random(a_scales_kut_host)
    for t in range(total_num_tokens):
        a_scales_ref_host_ptr[t] = a_scales_kut_host_ptr[t]

    # b_scales is [E, N, 1] for both paths.
    var b_scales_host = TileTensor(b_scales_host_ptr, b_scales_shape)
    random(b_scales_host)

    # Device allocations.
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
    var a_scales_kut_device_buffer = ctx.enqueue_create_buffer[scales_type](
        a_scales_kut_size
    )
    var a_scales_ref_device_buffer = ctx.enqueue_create_buffer[scales_type](
        a_scales_ref_size
    )
    var b_scales_device_buffer = ctx.enqueue_create_buffer[scales_type](
        b_scales_size
    )

    var a_device_tt = TileTensor(a_device_buffer, a_tt_shape)
    var b_device_tt = TileTensor(b_device_buffer, b_tt_shape)
    var c_device_tt = TileTensor(c_device_buffer, c_tt_shape)
    var c_device_ref_tt = TileTensor(c_device_ref_buffer, c_tt_shape)
    var a_offsets_device_tt = TileTensor(
        a_offsets_device_buffer,
        row_major(Coord(Int(num_active_experts + 1))),
    )
    var expert_ids_device_tt = TileTensor(
        expert_ids_device_buffer,
        row_major(Coord(Int(num_active_experts))),
    )
    var a_scales_kut_device_tt = TileTensor(
        a_scales_kut_device_buffer, a_scales_kut_shape
    )
    var a_scales_ref_device_tt = TileTensor(
        a_scales_ref_device_buffer, a_scales_ref_shape
    )
    var b_scales_device_tt = TileTensor(b_scales_device_buffer, b_scales_shape)

    ctx.enqueue_copy(a_device_buffer, a_host_ptr)
    ctx.enqueue_copy(b_device_buffer, b_host_ptr)
    ctx.enqueue_copy(c_device_buffer, c_host_ptr)
    ctx.enqueue_copy(c_device_ref_buffer, c_host_ref_ptr)
    ctx.enqueue_copy(a_offsets_device_buffer, a_offsets_host_ptr)
    ctx.enqueue_copy(expert_ids_device_buffer, expert_ids_host_ptr)
    ctx.enqueue_copy(a_scales_kut_device_buffer, a_scales_kut_host_ptr)
    ctx.enqueue_copy(a_scales_ref_device_buffer, a_scales_ref_host_ptr)
    ctx.enqueue_copy(b_scales_device_buffer, b_scales_host_ptr)

    # Reference path (LayoutTensor) - distinct code from the KUT.
    var a_ref = a_device_tt.to_layout_tensor()
    var b_ref = b_device_tt.to_layout_tensor()
    var c_ref = c_device_ref_tt.to_layout_tensor()
    var a_scales_ref = a_scales_ref_device_tt.to_layout_tensor()
    var b_scales_ref = b_scales_device_tt.to_layout_tensor()
    var a_offsets_ref = a_offsets_device_tt.to_layout_tensor()
    var expert_ids_ref = expert_ids_device_tt.to_layout_tensor()

    naive_blockwise_scaled_fp8_grouped_matmul[
        BLOCK_DIM_M=16,
        BLOCK_DIM_N=16,
        transpose_b=transpose_b,
        scales_granularity_mnk=Index(1, 1, K),
    ](
        c_ref,
        a_ref,
        b_ref,
        a_scales_ref,
        b_scales_ref,
        a_offsets_ref,
        expert_ids_ref,
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )

    ctx.synchronize()

    # Kernel under test (TileTensor).
    grouped_matmul_rowwise_dynamic_scaled_fp8[
        transpose_b=transpose_b,
        target="gpu",
    ](
        c_device_tt,
        a_device_tt,
        b_device_tt,
        a_scales_kut_device_tt,
        b_scales_device_tt,
        a_offsets_device_tt,
        expert_ids_device_tt,
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )

    ctx.synchronize()

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


def main() raises:
    with DeviceContext() as ctx:
        # Single expert, M aligned.
        test_grouped_matmul_rowwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=1,
            expert_shape=Index(256, 256),
        ](1, [128], [0], ctx)

        # Single expert, partial last M-tile (100 mod 16 != 0).
        test_grouped_matmul_rowwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=1,
            expert_shape=Index(256, 256),
        ](1, [100], [0], ctx)

        # Sparse routing: a single active expert that is NOT id 0.
        test_grouped_matmul_rowwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=4,
            expert_shape=Index(512, 1024),
        ](1, [256], [2], ctx)

        # Multi-expert ragged, sparse ids.
        test_grouped_matmul_rowwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=4,
            expert_shape=Index(512, 1024),
        ](2, [20, 40], [0, 2], ctx)

        # Zero-token expert in the middle of a ragged batch (M == 0 group).
        test_grouped_matmul_rowwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=6,
            expert_shape=Index(512, 1024),
        ](4, [20, 0, 300, 28], [0, 3, 2, 4], ctx)

        # Several small + medium experts, sparse ids.
        test_grouped_matmul_rowwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=6,
            expert_shape=Index(7168, 2048),
        ](4, [20, 1500, 300, 28], [0, 3, 2, 4], ctx)

        # fp32 output.
        test_grouped_matmul_rowwise_scaled_fp8[
            .float8_e4m3fn,
            .float32,
            num_experts=4,
            expert_shape=Index(512, 1024),
        ](2, [20, 40], [0, 2], ctx)

        # Llama-4-Scout gate/up shape (N x K = 8192 x 5120) with unaligned
        # per-expert token counts. ``num_experts`` is kept at 2 here so the B
        # weight allocation fits the bazel test sandbox's per-chunk GPU memory
        # cap; the full 16-expert Scout shape with sparse ids [4, 9] is
        # validated by a direct ``mojo run`` on an unconstrained B200, and
        # sparse-id-against-many-experts routing is covered by the 7168x2048
        # ``num_experts=6`` case above.
        test_grouped_matmul_rowwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=2,
            expert_shape=Index(8192, 5120),
        ](2, [13, 51], [0, 1], ctx)

        print("All rowwise grouped FP8 tests passed.")
