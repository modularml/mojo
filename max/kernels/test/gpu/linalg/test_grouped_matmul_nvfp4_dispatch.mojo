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
"""Accuracy test for grouped_matmul_nvfp4_dispatch.

Tests the dispatch function that selects optimal kernel configuration based
on (N, K) shape. Verifies correctness against vendor_blas reference for:
- Dispatch-tuned shapes: N=4096,K=7168 and N=7168,K=2048
- Fallback path (auto-computed config for unknown shapes)
- Various active expert counts, token patterns, and -1 expert IDs
"""
from std.math import align_up, ceildiv

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.host import DeviceContext
from std.memory import alloc
from internal_utils import assert_almost_equal
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d import (
    grouped_matmul_nvfp4_dispatch,
)
from std.random import random_ui64, seed, rand
from std.builtin.simd import _convert_f32_to_float8_scalar
from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from linalg.fp4_utils import (
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    NVFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    set_scale_factor,
)


def _test_dispatch[
    num_experts: Int,
    N: Int,
    K: Int,
](
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids: List[Int],
    ctx: DeviceContext,
) raises:
    """Test grouped_matmul_nvfp4_dispatch against vendor_blas reference.

    Follows the same reference-computation pattern as
    test_grouped_matmul_sm100_block_fp4.mojo but calls the dispatch function
    which selects all config parameters based on (N, K).
    """
    seed(1234)
    comptime a_type = DType.uint8
    comptime b_type = DType.uint8
    comptime c_type = DType.bfloat16
    comptime scales_dtype = NVFP4_SF_DTYPE
    comptime packed_K = K // 2
    comptime SF_VECTOR_SIZE = NVFP4_SF_VECTOR_SIZE
    comptime transpose_b = True

    var total_num_tokens = 0
    for i in range(len(num_tokens_by_expert)):
        total_num_tokens += num_tokens_by_expert[i]
    var M = total_num_tokens

    print(
        "  N=",
        N,
        " K=",
        K,
        " M=",
        M,
        " experts=",
        num_active_experts,
        "/",
        num_experts,
        sep="",
    )

    # --- Host allocations ---
    var a_shape = row_major(Coord(Int(M), Idx[packed_K]))
    var b_shape = row_major(Coord(Idx[num_experts], Idx[N], Idx[packed_K]))
    var c_shape = row_major(Coord(Int(M), Idx[N]))

    var a_size = M * packed_K
    var b_size = num_experts * N * packed_K
    var c_size = M * N

    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_size)
    var a_host = TileTensor(a_host_ptr, a_shape)
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](b_size)
    var b_host = TileTensor(b_host_ptr, b_shape)
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_host = TileTensor(c_host_ptr, c_shape)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_host_ref = TileTensor(c_host_ref_ptr, c_shape)

    # --- Device allocations ---
    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_tensor = TileTensor(a_device, a_shape)
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_tensor = TileTensor(b_device, b_shape)
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_tensor = TileTensor(c_device, c_shape)
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ref_tensor = TileTensor(c_device_ref, c_shape)

    var a_offsets_device = ctx.enqueue_create_buffer[.uint32](
        num_active_experts + 1
    )
    var a_offsets_tensor = TileTensor(
        a_offsets_device,
        row_major(Coord(Int(num_active_experts + 1))),
    )
    var a_scale_offsets_device = ctx.enqueue_create_buffer[.uint32](
        num_active_experts
    )
    var a_scale_offsets_tensor = TileTensor(
        a_scale_offsets_device,
        row_major(Coord(Int(num_active_experts))),
    )
    var expert_ids_device = ctx.enqueue_create_buffer[.int32](
        num_active_experts
    )
    var expert_ids_tensor = TileTensor(
        expert_ids_device,
        row_major(Coord(Int(num_active_experts))),
    )
    var expert_scales_device = ctx.enqueue_create_buffer[.float32](num_experts)
    var expert_scales_tensor = TileTensor(
        expert_scales_device,
        row_major(Coord(Idx[num_experts])),
    )

    # --- Offsets & expert IDs ---
    var a_offsets_host_ptr = ctx.enqueue_create_host_buffer[.uint32](
        num_active_experts + 1
    )
    var a_scale_offsets_ptr = ctx.enqueue_create_host_buffer[.uint32](
        num_active_experts
    )
    var expert_ids_host_ptr = ctx.enqueue_create_host_buffer[.int32](
        num_experts
    )
    var expert_scales_host_ptr = ctx.enqueue_create_host_buffer[.float32](
        num_experts
    )

    # Non-trivial expert scales: 1 + (i+1)/num_experts
    for i in range(num_experts):
        expert_scales_host_ptr[i] = 1.0 + Float32(i + 1) / Float32(num_experts)

    var a_scale_dim0 = 0
    a_offsets_host_ptr[0] = 0
    for i in range(num_active_experts):
        a_scale_offsets_ptr[i] = UInt32(
            a_scale_dim0
            - Int(a_offsets_host_ptr[i] // UInt32(SF_MN_GROUP_SIZE))
        )
        var local_m = num_tokens_by_expert[i]
        a_offsets_host_ptr[i + 1] = a_offsets_host_ptr[i] + UInt32(local_m)
        a_scale_dim0 += ceildiv(local_m, SF_MN_GROUP_SIZE)
        expert_ids_host_ptr[i] = Int32(expert_ids[i])

    # --- Scale factor dimensions ---
    comptime k_groups = ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)
    comptime n_groups = ceildiv(N, SF_MN_GROUP_SIZE)

    var a_scales_shape = row_major(
        Coord(
            Int(a_scale_dim0),
            Idx[k_groups],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    var b_scales_shape = row_major(
        Coord(
            Idx[num_experts],
            Idx[n_groups],
            Idx[k_groups],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )

    var a_scales_total = a_scales_shape.product()
    var b_scales_total = b_scales_shape.product()

    var a_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_dtype](
        a_scales_total
    )
    var a_scales_host = TileTensor(a_scales_host_ptr, a_scales_shape)
    var b_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_dtype](
        b_scales_total
    )
    var b_scales_host = TileTensor(b_scales_host_ptr, b_scales_shape)

    var a_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        a_scales_total
    )
    var a_scales_tensor = TileTensor(a_scales_device, a_scales_shape)
    var b_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        b_scales_total
    )
    var b_scales_tensor = TileTensor(b_scales_device, b_scales_shape)

    # --- Initialize data ---
    rand(a_host._storage, a_host.num_elements(), min=0, max=255)
    rand(b_host._storage, b_host.num_elements(), min=0, max=255)

    var a_scales_tensor_host = TileTensor(a_scales_host_ptr, a_scales_shape)
    var b_scales_tensor_host = TileTensor(b_scales_host_ptr, b_scales_shape)

    # Initialize a_scales to 0, then set valid regions to power-of-2 values
    for i in range(a_scales_host.num_elements()):
        a_scales_host._storage[i] = Scalar[scales_dtype](0.0)
    rand(b_scales_host._storage, b_scales_host.num_elements())

    # Set a_scales for active expert regions
    for i in range(num_active_experts):
        var start = Int(a_offsets_host_ptr[i])
        var end = Int(a_offsets_host_ptr[i + 1])
        var local_m = end - start
        var actual_start = (
            start // SF_MN_GROUP_SIZE + Int(a_scale_offsets_ptr[i])
        ) * SF_MN_GROUP_SIZE
        var actual_end = actual_start + local_m
        for idx0 in range(actual_start, actual_end):
            for idx1 in range(
                0,
                align_up(K, SF_VECTOR_SIZE * SF_ATOM_K),
                SF_VECTOR_SIZE,
            ):
                if idx1 < K:
                    var scale_value = _convert_f32_to_float8_scalar[
                        scales_dtype
                    ]((1 << random_ui64(0, 2)).cast[.float32]())
                    set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                        a_scales_tensor_host, idx0, idx1, scale_value
                    )

    # Zero unused b_scales regions
    for e in range(num_experts):
        var expert_slice_size = (
            Int(b_scales_host.dim(1))
            * Int(b_scales_host.dim(2))
            * Int(b_scales_host.dim(3))
            * Int(b_scales_host.dim(4))
            * Int(b_scales_host.dim(5))
        )
        var b_scales_tensor_expert_slice = TileTensor(
            b_scales_host_ptr.unsafe_ptr() + e * expert_slice_size,
            row_major(
                Coord(
                    Idx[n_groups],
                    Idx[k_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        )
        for idx0 in range(align_up(N, SF_MN_GROUP_SIZE)):
            for idx1 in range(
                0,
                align_up(K, SF_VECTOR_SIZE * SF_ATOM_K),
                SF_VECTOR_SIZE,
            ):
                if idx0 >= N or idx1 >= K:
                    set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                        b_scales_tensor_expert_slice,
                        idx0,
                        idx1,
                        Scalar[scales_dtype](0.0),
                    )

    # --- Copy to device ---
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(a_offsets_device, a_offsets_host_ptr)
    ctx.enqueue_copy(a_scale_offsets_device, a_scale_offsets_ptr)
    ctx.enqueue_copy(expert_ids_device, expert_ids_host_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)
    ctx.enqueue_copy(expert_scales_device, expert_scales_host_ptr)

    # --- Build TileTensors (5D/6D scales need explicit row_major layout) ---
    var a_scales_tt = TileTensor(
        a_scales_device,
        row_major(
            Coord(
                Int64(a_scale_dim0),
                Idx[k_groups],
                Idx[SF_ATOM_M[0]],
                Idx[SF_ATOM_M[1]],
                Idx[SF_ATOM_K],
            )
        ),
    ).as_unsafe_any_origin()
    var b_scales_tt = TileTensor(
        b_scales_device,
        row_major(
            Coord(
                Idx[num_experts],
                Idx[n_groups],
                Idx[k_groups],
                Idx[SF_ATOM_M[0]],
                Idx[SF_ATOM_M[1]],
                Idx[SF_ATOM_K],
            )
        ),
    ).as_unsafe_any_origin()
    var expert_scales_tt = TileTensor(
        expert_scales_device,
        row_major(Coord(Int64(num_experts))),
    ).as_unsafe_any_origin()

    # --- Call dispatch function (MOGG argument order) ---
    grouped_matmul_nvfp4_dispatch[transpose_b=transpose_b](
        c_tensor,
        a_tensor,
        b_tensor,
        a_scales_tt,
        b_scales_tt,
        a_offsets_tensor,
        a_scale_offsets_tensor,
        expert_ids_tensor,
        expert_scales_tt,
        num_active_experts,
        total_num_tokens,
        ctx,
    )
    ctx.synchronize()

    # --- Reference computation via vendor_blas (per-expert) ---
    var c_row_stride = N
    var a_row_stride = packed_K
    comptime b_expert_stride = N * packed_K
    comptime b_scales_expert_stride = n_groups * k_groups * SF_ATOM_M[
        0
    ] * SF_ATOM_M[1] * SF_ATOM_K
    comptime a_scales_row_stride = k_groups * SF_ATOM_M[0] * SF_ATOM_M[
        1
    ] * SF_ATOM_K

    for i in range(num_active_experts):
        var start = Int(a_offsets_host_ptr[i])
        var end = Int(a_offsets_host_ptr[i + 1])
        var expert_id = expert_ids_host_ptr[i]

        if expert_id < 0 or end - start == 0:
            continue

        var c_slice = TileTensor(
            c_ref_tensor.ptr + start * c_row_stride,
            row_major((end - start, Idx[N])),
        )

        var new_a_tensor = TileTensor(
            a_tensor.ptr + start * a_row_stride,
            row_major((end - start, Idx[packed_K])),
        )

        var new_b_tensor = TileTensor(
            b_tensor.ptr + Int(expert_id) * b_expert_stride,
            row_major((Idx[N], Idx[packed_K])),
        )

        var new_b_scales_tensor = TileTensor(
            b_scales_tensor.ptr + Int(expert_id) * b_scales_expert_stride,
            row_major(
                Coord(
                    Idx[n_groups],
                    Idx[k_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        )

        var a_scales_start = start // SF_MN_GROUP_SIZE + Int(
            a_scale_offsets_ptr[i]
        )
        var new_a_scales_tensor = TileTensor(
            a_scales_tensor.ptr + a_scales_start * a_scales_row_stride,
            row_major(
                Coord(
                    ceildiv(end - start, SF_MN_GROUP_SIZE),
                    Idx[k_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        )

        var expert_scale = expert_scales_host_ptr[Int(expert_id)]
        vendor_blas.matmul(
            ctx,
            c_slice,
            new_a_tensor,
            new_b_tensor,
            a_scales=new_a_scales_tensor,
            b_scales=new_b_scales_tensor,
            transpose_b=transpose_b,
            c_row_major=True,
            alpha=expert_scale,
        )

    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    # Zero output regions for skipped experts (expert_id == -1 or 0 tokens)
    for i in range(num_active_experts):
        var start = Int(a_offsets_host_ptr[i])
        var end = Int(a_offsets_host_ptr[i + 1])
        if expert_ids_host_ptr[i] < 0 or end - start == 0:
            for j in range(start * N, end * N):
                c_host_ptr[j] = Scalar[c_type](0)
                c_host_ref_ptr[j] = Scalar[c_type](0)

    assert_almost_equal(
        c_host._storage,
        c_host_ref._storage,
        c_host.num_elements(),
        atol=1e-2,
        rtol=1e-2,
    )
    print("    PASSED")

    # --- Cleanup ---
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^
    _ = a_scales_device^
    _ = b_scales_device^
    _ = a_offsets_device^
    _ = a_scale_offsets_device^
    _ = expert_ids_device^
    _ = expert_scales_device^


def main() raises:
    with DeviceContext() as ctx:
        # ============================================================
        # 1. Dispatch-tuned shape: N=4096, K=7168 (DeepSeek V3 up-proj)
        #    Should select stages=6 automatically
        # ============================================================
        print("=== Tuned shape: N=4096, K=7168 (auto stages=6) ===")

        # 1a: Multiple experts, various token counts
        print("  1a: 4 experts, mixed token counts")
        _test_dispatch[6, 4096, 7168](
            4,
            [128, 256, 512, 1024],
            [0, 3, 2, 4],
            ctx,
        )

        # 1b: Small token counts (MoE decode regime)
        print("  1b: 4 experts, very small tokens (2 each)")
        _test_dispatch[6, 4096, 7168](
            4,
            [2, 2, 2, 2],
            [0, 1, 2, 3],
            ctx,
        )

        # 1c: Single expert
        print("  1c: 1 active expert")
        _test_dispatch[6, 4096, 7168](
            1,
            [64],
            [3],
            ctx,
        )

        # 1d: Unaligned token counts
        print("  1d: unaligned tokens")
        _test_dispatch[6, 4096, 7168](
            3,
            [65, 129, 257],
            [2, 0, 1],
            ctx,
        )

        # 1e: -1 expert IDs (inactive experts skipped by kernel)
        print("  1e: -1 expert IDs")
        _test_dispatch[6, 4096, 7168](
            3,
            [128, 256, 512],
            [-1, 0, 2],
            ctx,
        )

        # ============================================================
        # 2. Dispatch-tuned shape: N=7168, K=2048 (DeepSeek V3 down-proj)
        #    Should select stages=4 automatically
        # ============================================================
        print("\n=== Tuned shape: N=7168, K=2048 (auto stages=4) ===")

        # 2a: Multiple experts
        print("  2a: 4 experts, mixed tokens")
        _test_dispatch[6, 7168, 2048](
            4,
            [128, 256, 512, 1024],
            [0, 3, 2, 4],
            ctx,
        )

        # 2b: Small tokens
        print("  2b: small tokens")
        _test_dispatch[6, 7168, 2048](
            4,
            [2, 2, 2, 2],
            [0, 1, 2, 3],
            ctx,
        )

        # 2c: Single expert, unaligned
        print("  2c: single expert, unaligned")
        _test_dispatch[6, 7168, 2048](
            1,
            [129],
            [3],
            ctx,
        )

        # 2d: -1 expert IDs mixed with valid
        print("  2d: -1 expert IDs mixed")
        _test_dispatch[6, 7168, 2048](
            4,
            [0, 3, 1, 2],
            [-1, 2, -1, 0],
            ctx,
        )

        # ============================================================
        # 3. Fallback path (unknown shapes, auto-computed stages)
        # ============================================================
        print("\n=== Fallback: unknown shapes (auto-computed stages) ===")

        # 3a: Standard shape, no dispatch match
        print("  3a: N=2048, K=1024 (auto)")
        _test_dispatch[6, 2048, 1024](
            4,
            [512, 1000, 2000, 3000],
            [0, 3, 2, 4],
            ctx,
        )

        # 3b: Unaligned tokens
        print("  3b: N=2048, K=1024, unaligned tokens")
        _test_dispatch[4, 2048, 1024](
            3,
            [64 + 1, 1024 + 3, 128 * 3 + 2],
            [2, 0, 1],
            ctx,
        )

        # 3c: Small tokens
        print("  3c: N=2048, K=1024, small tokens")
        _test_dispatch[4, 2048, 1024](
            3,
            [31, 97, 63],
            [2, 0, 1],
            ctx,
        )

        # 3d: Very small MoE-like
        print("  3d: N=2048, K=1024, very small tokens")
        _test_dispatch[6, 2048, 1024](
            4,
            [0, 1, 2, 3],
            [0, 3, 2, 4],
            ctx,
        )

        # ============================================================
        # 4. Expert activity sweep (1-8 active, important regime)
        # ============================================================
        print("\n=== Expert activity sweep (1-8 active) ===")

        # Use tuned shape to exercise dispatch path
        print("  4a: 1 active expert")
        _test_dispatch[8, 4096, 7168](
            1,
            [64],
            [5],
            ctx,
        )

        print("  4b: 2 active experts")
        _test_dispatch[8, 4096, 7168](
            2,
            [64, 128],
            [5, 2],
            ctx,
        )

        print("  4c: 4 active experts")
        _test_dispatch[8, 4096, 7168](
            4,
            [64, 128, 32, 96],
            [5, 2, 7, 0],
            ctx,
        )

        print("  4d: 8 active experts")
        _test_dispatch[8, 4096, 7168](
            8,
            [32, 64, 48, 96, 16, 80, 64, 128],
            [0, 1, 2, 3, 4, 5, 6, 7],
            ctx,
        )

        # ============================================================
        # 5. Edge cases
        # ============================================================
        print("\n=== Edge cases ===")

        # 5a: All -1 expert IDs (no work)
        print("  5a: all -1 expert IDs")
        _test_dispatch[4, 2048, 1024](
            3,
            [128, 256, 512],
            [-1, -1, -1],
            ctx,
        )

        # 5b: Zero tokens for some experts
        print("  5b: zero tokens mixed")
        _test_dispatch[6, 2048, 1024](
            4,
            [0, 256, 0, 512],
            [0, 1, 2, 3],
            ctx,
        )

        # 5c: Mixed -1 IDs and zero tokens on tuned shape
        print("  5c: -1 IDs + zero tokens on tuned shape")
        _test_dispatch[6, 4096, 7168](
            4,
            [0, 64, 128, 0],
            [-1, 2, 4, -1],
            ctx,
        )

        # 5d: Just-off-alignment tokens on tuned shape
        print("  5d: just-off-alignment on N=7168,K=2048")
        _test_dispatch[6, 7168, 2048](
            4,
            [127, 257, 513, 1025],
            [0, 3, 2, 4],
            ctx,
        )

        # ============================================================
        # 6. Prefill path (estimated_total_m >= num_active_experts * 8)
        #    Uses mma_bn=128, cta_group=2 via _dispatch_prefill
        # ============================================================
        print(
            "\n=== Prefill path (estimated_total_m >= num_active_experts *"
            " 8) ==="
        )

        # 6a: Tuned shape, moderate tokens
        print("  6a: N=4096, K=7168, 4 experts, 128 tok each")
        _test_dispatch[6, 4096, 7168](
            4,
            [128, 128, 128, 128],
            [0, 1, 2, 3],
            ctx,
        )

        # 6b: Tuned shape, larger tokens (prefill regime)
        print("  6b: N=4096, K=7168, 4 experts, 256 tok each")
        _test_dispatch[6, 4096, 7168](
            4,
            [256, 256, 256, 256],
            [0, 1, 2, 3],
            ctx,
        )

        # 6c: Down-proj shape
        print("  6c: N=7168, K=2048, 4 experts, mixed tokens")
        _test_dispatch[6, 7168, 2048](
            4,
            [128, 256, 192, 96],
            [0, 3, 2, 4],
            ctx,
        )

        # 6d: Unaligned tokens on prefill path
        print("  6d: N=4096, K=7168, unaligned tokens")
        _test_dispatch[6, 4096, 7168](
            3,
            [129, 257, 193],
            [2, 0, 1],
            ctx,
        )

        # 6e: Single expert, large M
        print("  6e: N=7168, K=2048, 1 expert, 512 tokens")
        _test_dispatch[6, 7168, 2048](
            1,
            [512],
            [3],
            ctx,
        )

        # 6f: Mixed -1 IDs on prefill path
        print("  6f: N=4096, K=7168, -1 IDs mixed")
        _test_dispatch[6, 4096, 7168](
            4,
            [128, 256, 128, 256],
            [-1, 2, -1, 4],
            ctx,
        )

        # 6g: Many experts (Kimi K2.5 like)
        print("  6g: N=4096, K=7168, 8 experts, 96 tok each")
        _test_dispatch[8, 4096, 7168](
            8,
            [96, 96, 96, 96, 96, 96, 96, 96],
            [0, 1, 2, 3, 4, 5, 6, 7],
            ctx,
        )

        # 6h: Fallback shape on prefill path
        print("  6h: N=2048, K=1024, fallback prefill")
        _test_dispatch[4, 2048, 1024](
            3,
            [256, 128, 192],
            [2, 0, 1],
            ctx,
        )

        # ============================================================
        # 7. cp.async SFB path: group_size < 128 (SF_MN_GROUP_SIZE)
        #    When tokens_per_expert < 128 for an expert, the SfbTMALoad
        #    warp uses cp.async instead of TMA to avoid wasting BW on
        #    OOB zero-fill in full 128-row SF atoms.
        # ============================================================
        print("\n=== cp.async SFB: small group_size < 128 ===")

        # 7a: All experts have very small groups (cp.async for all)
        print("  7a: N=4096, K=7168, all tiny groups")
        _test_dispatch[6, 4096, 7168](
            4,
            [1, 2, 3, 4],
            [0, 1, 2, 3],
            ctx,
        )

        # 7b: Mixed cp.async and TMA within one launch
        print("  7b: N=4096, K=7168, mixed cp.async(4,1) + TMA(256,200)")
        _test_dispatch[6, 4096, 7168](
            4,
            [4, 256, 1, 200],
            [0, 1, 2, 3],
            ctx,
        )

        # 7c: Boundary: gs=127 (cp.async) and gs=128 (TMA)
        print("  7c: N=4096, K=7168, boundary 127/128")
        _test_dispatch[6, 4096, 7168](
            4,
            [127, 128, 64, 256],
            [0, 1, 2, 3],
            ctx,
        )

        # 7d: Same tests on down-proj shape
        print("  7d: N=7168, K=2048, mixed small/large")
        _test_dispatch[6, 7168, 2048](
            4,
            [8, 512, 16, 300],
            [0, 3, 2, 4],
            ctx,
        )

        # 7e: Single expert, single token
        print("  7e: N=4096, K=7168, 1 expert, 1 token")
        _test_dispatch[6, 4096, 7168](
            1,
            [1],
            [3],
            ctx,
        )

        # 7f: Many experts, all small (Kimi K2.5 decode-like)
        print("  7f: N=4096, K=7168, 8 experts, 1 tok each")
        _test_dispatch[8, 4096, 7168](
            8,
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 2, 3, 4, 5, 6, 7],
            ctx,
        )

        # ============================================================
        # 8. Kimi K2.5 TP=8 shapes
        #    N=512, K=7168 (gate+up, TP-sharded on N)
        #    N=7168, K=256 (down-proj, TP-sharded on K)
        # ============================================================
        print("\n=== Kimi K2.5 TP=8: N=512, K=7168 (gate+up) ===")

        # 8a: Decode regime (avg_m = 1): 9 experts at 1 tok each
        print("  8a: N=512, K=7168, decode 9 experts @ 1 tok")
        _test_dispatch[9, 512, 7168](
            9,
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            ctx,
        )

        # 8b: Decode with -1 masking (id<0 experts skipped by kernel)
        print("  8b: N=512, K=7168, decode with -1 IDs mixed")
        _test_dispatch[8, 512, 7168](
            8,
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, -1, 2, -1, 4, -1, 6, -1],
            ctx,
        )

        # 8c: Decode b8 pattern: shared (8 tok) + 8 routed (1 tok each)
        print("  8c: N=512, K=7168, decode shared + 8 routed")
        _test_dispatch[9, 512, 7168](
            9,
            [8, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            ctx,
        )

        # 8d: Small prefill regime (8 < avg_m <= 64): 8 experts @ 40 tok
        print("  8d: N=512, K=7168, small prefill 8 experts @ 40 tok")
        _test_dispatch[8, 512, 7168](
            8,
            [40, 40, 40, 40, 40, 40, 40, 40],
            [0, 1, 2, 3, 4, 5, 6, 7],
            ctx,
        )

        # 8e: Large prefill regime (avg_m > 64): 4 experts @ 128 tok
        print("  8e: N=512, K=7168, large prefill 4 experts @ 128 tok")
        _test_dispatch[4, 512, 7168](
            4,
            [128, 128, 128, 128],
            [0, 1, 2, 3],
            ctx,
        )

        print("\n=== Kimi K2.5 TP=8: N=7168, K=256 (down-proj) ===")

        # 8f: Decode regime: 9 experts at 1 tok each
        print("  8f: N=7168, K=256, decode 9 experts @ 1 tok")
        _test_dispatch[9, 7168, 256](
            9,
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            ctx,
        )

        # 8g: Decode with -1 masking
        print("  8g: N=7168, K=256, decode with -1 IDs mixed")
        _test_dispatch[8, 7168, 256](
            8,
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, -1, 2, -1, 4, -1, 6, -1],
            ctx,
        )

        # 8h: Decode b8 pattern: shared + 8 routed
        print("  8h: N=7168, K=256, decode shared + 8 routed")
        _test_dispatch[9, 7168, 256](
            9,
            [8, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            ctx,
        )

        # 8i: Small prefill: 8 experts @ 40 tok
        print("  8i: N=7168, K=256, small prefill 8 experts @ 40 tok")
        _test_dispatch[8, 7168, 256](
            8,
            [40, 40, 40, 40, 40, 40, 40, 40],
            [0, 1, 2, 3, 4, 5, 6, 7],
            ctx,
        )

        # 8j: Large prefill: 4 experts @ 128 tok
        print("  8j: N=7168, K=256, large prefill 4 experts @ 128 tok")
        _test_dispatch[4, 7168, 256](
            4,
            [128, 128, 128, 128],
            [0, 1, 2, 3],
            ctx,
        )

        print("\n========================================")
        print("ALL DISPATCH TESTS PASSED!")
        print("========================================")
