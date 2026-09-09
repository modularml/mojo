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
"""Tests for native MXFP4 grouped matmul on AMD CDNA4.

Validates block_scaled_grouped_matmul_amd against a per-element GPU reference
that runs one ungrouped block_scaled_matmul_amd per expert and
compares the concatenated output.

Usage:
  br test_mxfp4_grouped_matmul_amd.mojo.test
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.memory import bitcast
from std.random import random_ui64
from std.sys.intrinsics import llvm_intrinsic

from internal_utils import assert_almost_equal
from layout import Coord, Idx, TileTensor, row_major
from linalg.fp4_utils import MXFP4_SF_VECTOR_SIZE
from linalg.matmul.gpu.amd import (
    block_scaled_matmul_amd,
    block_scaled_grouped_matmul_amd,
)


# ===----------------------------------------------------------------------=== #
# Test harness
# ===----------------------------------------------------------------------=== #


def test_mxfp4_grouped_matmul[
    num_experts: Int, N: Int, K: Int
](
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids_list: List[Int],
    ctx: DeviceContext,
) raises:
    """Test block_scaled_grouped_matmul_amd against per-expert ungrouped matmul.

    Runs block_scaled_matmul_amd independently for each expert
    slice, then compares with the grouped kernel's output.

    Parameters:
        num_experts: Total number of expert weight matrices.
        N: Output columns / weight rows.
        K: Logical K dimension (FP4 elements, must be multiple of 128).
    """
    comptime assert (
        K % 128 == 0
    ), "K must be a multiple of 128 (MFMA K dimension)"

    comptime packed_K = K // 2
    comptime scale_K = K // MXFP4_SF_VECTOR_SIZE

    # Compute total tokens and max tokens per expert.
    var total_tokens = 0
    var max_tokens = 0
    for i in range(len(num_tokens_by_expert)):
        total_tokens += num_tokens_by_expert[i]
        max_tokens = max(max_tokens, num_tokens_by_expert[i])

    print(
        "  grouped matmul: num_experts=",
        num_experts,
        " N=",
        N,
        " K=",
        K,
        " num_active=",
        num_active_experts,
        " total_tokens=",
        total_tokens,
    )

    # --- Host allocations ---
    var a_host = ctx.enqueue_create_host_buffer[.uint8](total_tokens * packed_K)
    var b_host = ctx.enqueue_create_host_buffer[.uint8](
        num_experts * N * packed_K
    )
    var a_scales_host = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        total_tokens * scale_K
    )
    var b_scales_host = ctx.enqueue_create_host_buffer[.float8_e8m0fnu](
        num_experts * N * scale_K
    )
    var a_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        num_active_experts + 1
    )
    var expert_ids_host = ctx.enqueue_create_host_buffer[.int32](
        num_active_experts
    )

    # Random packed FP4 data.
    for i in range(total_tokens * packed_K):
        a_host[i] = UInt8(random_ui64(0, 255))
    for i in range(num_experts * N * packed_K):
        b_host[i] = UInt8(random_ui64(0, 255))

    # Scales: exponent range [125..129] for reasonable magnitudes.
    for i in range(total_tokens * scale_K):
        a_scales_host[i] = bitcast[.float8_e8m0fnu](
            UInt8(random_ui64(125, 129))
        )
    for i in range(num_experts * N * scale_K):
        b_scales_host[i] = bitcast[.float8_e8m0fnu](
            UInt8(random_ui64(125, 129))
        )

    # Setup offsets and expert ids.
    a_offsets_host[0] = UInt32(0)
    for i in range(num_active_experts):
        a_offsets_host[i + 1] = a_offsets_host[i] + UInt32(
            num_tokens_by_expert[i]
        )
        expert_ids_host[i] = Int32(expert_ids_list[i])

    # --- Device allocations ---
    var a_dev = ctx.enqueue_create_buffer[.uint8](total_tokens * packed_K)
    var b_dev = ctx.enqueue_create_buffer[.uint8](num_experts * N * packed_K)
    var a_scales_dev = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        total_tokens * scale_K
    )
    var b_scales_dev = ctx.enqueue_create_buffer[.float8_e8m0fnu](
        num_experts * N * scale_K
    )
    var a_offsets_dev = ctx.enqueue_create_buffer[.uint32](
        num_active_experts + 1
    )
    var expert_ids_dev = ctx.enqueue_create_buffer[.int32](num_active_experts)
    var c_dev = ctx.enqueue_create_buffer[.float32](total_tokens * N)

    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(a_scales_dev, a_scales_host)
    ctx.enqueue_copy(b_scales_dev, b_scales_host)
    ctx.enqueue_copy(a_offsets_dev, a_offsets_host)
    ctx.enqueue_copy(expert_ids_dev, expert_ids_host)

    # --- Compute reference: per-expert ungrouped matmul ---
    var c_ref_dev = ctx.enqueue_create_buffer[.float32](total_tokens * N)
    for i in range(num_active_experts):
        var token_start = Int(a_offsets_host[i])
        var token_end = Int(a_offsets_host[i + 1])
        var num_tokens = token_end - token_start
        var expert_id = Int(expert_ids_host[i])

        if num_tokens <= 0 or expert_id < 0:
            continue

        var a_expert_tt = TileTensor(
            a_dev.unsafe_ptr() + token_start * packed_K,
            row_major(Coord(num_tokens, Idx[packed_K])),
        )
        var b_expert_tt = TileTensor(
            b_dev.unsafe_ptr() + expert_id * N * packed_K,
            row_major[N, packed_K](),
        ).as_immut()
        var sfa_expert_tt = TileTensor(
            a_scales_dev.unsafe_ptr() + token_start * scale_K,
            row_major(Coord(num_tokens, Idx[scale_K])),
        ).as_immut()
        var sfb_expert_tt = TileTensor(
            b_scales_dev.unsafe_ptr() + expert_id * N * scale_K,
            row_major[N, scale_K](),
        ).as_immut()
        var c_expert_tt = TileTensor(
            c_ref_dev.unsafe_ptr() + token_start * N,
            row_major(Coord(num_tokens, Idx[N])),
        )

        block_scaled_matmul_amd(
            c_expert_tt,
            a_expert_tt,
            b_expert_tt,
            sfa_expert_tt,
            sfb_expert_tt,
            ctx,
        )

    ctx.synchronize()

    # --- Run grouped kernel under test ---
    var a_tt = TileTensor(
        a_dev, row_major(Coord(total_tokens, Idx[packed_K]))
    ).as_immut()
    var b_tt = TileTensor(
        b_dev, row_major[num_experts, N, packed_K]()
    ).as_immut()
    var a_scales_tt = TileTensor(
        a_scales_dev, row_major(Coord(total_tokens, Idx[scale_K]))
    ).as_immut()
    var b_scales_tt = TileTensor(
        b_scales_dev, row_major[num_experts, N, scale_K]()
    ).as_immut()
    var a_offsets_tt = TileTensor(
        a_offsets_dev, row_major(Coord(num_active_experts + 1))
    )
    var expert_ids_tt = TileTensor(
        expert_ids_dev, row_major(Coord(num_active_experts))
    )
    var c_tt = TileTensor(c_dev, row_major(Coord(total_tokens, Idx[N])))

    block_scaled_grouped_matmul_amd(
        c_tt,
        a_tt,
        b_tt,
        a_scales_tt,
        b_scales_tt,
        a_offsets_tt,
        expert_ids_tt,
        max_tokens,
        num_active_experts,
        ctx,
    )
    ctx.synchronize()

    # --- Compare ---
    var c_host = ctx.enqueue_create_host_buffer[.float32](total_tokens * N)
    var c_ref_host = ctx.enqueue_create_host_buffer[.float32](total_tokens * N)
    ctx.enqueue_copy(c_host, c_dev)
    ctx.enqueue_copy(c_ref_host, c_ref_dev)
    ctx.synchronize()

    assert_almost_equal(
        c_host.unsafe_ptr(),
        c_ref_host.unsafe_ptr(),
        total_tokens * N,
        atol=0.05,
        rtol=0.05,
    )

    print("    PASS")

    # Cleanup
    _ = a_dev^
    _ = b_dev^
    _ = a_scales_dev^
    _ = b_scales_dev^
    _ = a_offsets_dev^
    _ = expert_ids_dev^
    _ = c_dev^
    _ = c_ref_dev^


def main() raises:
    with DeviceContext() as ctx:
        print("===> MXFP4 grouped matmul (native CDNA4 MFMA)")

        # Single expert (degenerates to regular matmul)
        print("-- Single expert --")
        test_mxfp4_grouped_matmul[1, 128, 128](1, [128], [0], ctx)
        test_mxfp4_grouped_matmul[1, 256, 256](1, [128], [0], ctx)

        # Multiple experts, simple routing
        print("-- Multiple experts --")
        test_mxfp4_grouped_matmul[4, 128, 128](2, [128, 128], [0, 2], ctx)
        test_mxfp4_grouped_matmul[4, 256, 256](
            3, [128, 128, 128], [0, 1, 3], ctx
        )

        # Unequal token counts
        print("-- Unequal token counts --")
        test_mxfp4_grouped_matmul[4, 128, 256](2, [128, 256], [0, 2], ctx)

        # Larger dimensions
        print("-- Larger dimensions --")
        test_mxfp4_grouped_matmul[4, 256, 512](2, [128, 256], [1, 3], ctx)

        # Decode tile path: max_tokens_per_expert <= 64 with K_BYTES >= 256.
        print("-- Decode tile (small max tokens, large K) --")
        test_mxfp4_grouped_matmul[1, 128, 512](1, [32], [0], ctx)
        test_mxfp4_grouped_matmul[4, 128, 512](2, [16, 64], [0, 2], ctx)

        print("==== All MXFP4 grouped matmul tests passed ====")
