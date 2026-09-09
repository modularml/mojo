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
# Numeric correctness test for the boundary-aware single-launch LoRA-B expand
# grouped GEMM on SM100 (`expand_qkv_sm100` in `linalg.lora`).
#
# Oracle: three independent `naive_grouped_matmul` calls (Q, K, V), each using
# the matching plane of the planar shrink output P as its A operand and the
# matching slice of the fused LoRA-B weight as its B operand. The kernel under
# test fuses these into ONE launch over the concatenated weight and routes the
# results into the (q_out, row-stacked kv_out) layout the KV-cache write
# consumes downstream.

import std.itertools

from max.gpu.host import DeviceContext
from max.gpu.host.info import _is_sm10x_gpu
from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from layout._fillers import random
from linalg.grouped_matmul import naive_grouped_matmul
from linalg.lora import expand_qkv_sm100
from std.testing import assert_almost_equal


def test[
    in_type: DType,
    out_type: DType,
    num_experts: Int,
    q_dim: Int,
    kv_dim: Int,
    R: Int,
](
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids: List[Int],
    ctx: DeviceContext,
) raises:
    print(
        num_active_experts,
        "active of",
        num_experts,
        "experts; q_dim",
        q_dim,
        "kv_dim",
        kv_dim,
        "R",
        R,
    )
    print("tokens:", end="")
    for i in range(len(num_tokens_by_expert)):
        print(num_tokens_by_expert[i], end=" ")
    print("expert ids:", end="")
    for i in range(len(expert_ids)):
        print(expert_ids[i], end=" ")
    print()

    comptime p_type = in_type
    comptime b_type = in_type
    comptime out_q_type = out_type
    comptime out_kv_type = out_type

    comptime D_total = q_dim + 2 * kv_dim

    # Total and max number of tokens.
    var M = 0
    var max_num_tokens_by_expert = 0
    for i in range(len(num_tokens_by_expert)):
        M += num_tokens_by_expert[i]
        max_num_tokens_by_expert = max(
            max_num_tokens_by_expert, num_tokens_by_expert[i]
        )

    # ---- Host allocations -------------------------------------------------
    # P: planar [3, M, R].
    var p_size = 3 * M * R
    # Fused B: [G, D_total, R].
    var b_size = num_experts * D_total * R
    # Outputs.
    var q_size = M * q_dim
    var kv_size = 2 * M * kv_dim

    var p_host_ptr = ctx.enqueue_create_host_buffer[p_type](p_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](b_size)
    var q_host_ptr = ctx.enqueue_create_host_buffer[out_q_type](q_size)
    var kv_host_ptr = ctx.enqueue_create_host_buffer[out_kv_type](kv_size)
    var a_offsets_host_ptr = ctx.enqueue_create_host_buffer[.uint32](
        num_experts + 1
    )
    var expert_ids_host_ptr = ctx.enqueue_create_host_buffer[.int32](
        num_experts
    )

    # Host views (only used to fill P and B and read back).
    var p_host = TileTensor(p_host_ptr, row_major(Coord(Idx[3], M, Idx[R])))
    var b_host = TileTensor(b_host_ptr, row_major[num_experts, D_total, R]())

    # Offsets / ids.
    a_offsets_host_ptr[0] = 0
    for i in range(num_active_experts):
        a_offsets_host_ptr[i + 1] = a_offsets_host_ptr[i] + UInt32(
            num_tokens_by_expert[i]
        )
        expert_ids_host_ptr[i] = Int32(expert_ids[i])

    random(p_host)
    random(b_host)

    # ---- Device allocations ----------------------------------------------
    var p_dev_buffer = ctx.enqueue_create_buffer[p_type](p_size)
    var b_dev_buffer = ctx.enqueue_create_buffer[b_type](b_size)
    var q_dev_buffer = ctx.enqueue_create_buffer[out_q_type](q_size)
    var kv_dev_buffer = ctx.enqueue_create_buffer[out_kv_type](kv_size)
    var a_offsets_dev_buffer = ctx.enqueue_create_buffer[.uint32](
        num_experts + 1
    )
    var expert_ids_dev_buffer = ctx.enqueue_create_buffer[.int32](num_experts)

    # The planar P, viewed both as [3, M, R] and as three [M, R] planes for the
    # oracle (planes are contiguous so plane `t` starts at offset t*M*R).
    var p_dev = TileTensor(p_dev_buffer, row_major(Coord(Idx[3], M, Idx[R])))
    var p_plane_q = TileTensor(
        p_dev_buffer.unsafe_ptr(),
        row_major(Coord(M, Idx[R])),
    )
    var p_plane_k = TileTensor(
        p_dev_buffer.unsafe_ptr() + M * R,
        row_major(Coord(M, Idx[R])),
    )
    var p_plane_v = TileTensor(
        p_dev_buffer.unsafe_ptr() + 2 * M * R,
        row_major(Coord(M, Idx[R])),
    )

    # Fused weight [G, D_total, R]. The oracle runs naive_grouped_matmul three
    # times against this *same* fused weight (N = D_total), once per A-plane, and
    # we read only each region's columns from the result. This avoids copying out
    # Q/K/V weight slices and is exact: ref[A=plane p][m, j] is precisely the dot
    # the kernel computes for any column j whose region selects plane p.
    var b_dev = TileTensor(b_dev_buffer, row_major[num_experts, D_total, R]())

    # Oracle outputs, sized to the full D_total per region's A-plane, then we
    # read only the region's columns.
    var ref_q_full_buffer = ctx.enqueue_create_buffer[out_q_type](M * D_total)
    var ref_k_full_buffer = ctx.enqueue_create_buffer[out_kv_type](M * D_total)
    var ref_v_full_buffer = ctx.enqueue_create_buffer[out_kv_type](M * D_total)
    var ref_q_full = TileTensor(
        ref_q_full_buffer, row_major(Coord(M, Idx[D_total]))
    )
    var ref_k_full = TileTensor(
        ref_k_full_buffer, row_major(Coord(M, Idx[D_total]))
    )
    var ref_v_full = TileTensor(
        ref_v_full_buffer, row_major(Coord(M, Idx[D_total]))
    )

    var q_dev = TileTensor(q_dev_buffer, row_major(Coord(M, Idx[q_dim])))
    var kv_dev = TileTensor(kv_dev_buffer, row_major(Coord(2 * M, Idx[kv_dim])))

    var a_offsets_dev = TileTensor(
        a_offsets_dev_buffer, row_major(Coord(num_experts + 1))
    )
    var expert_ids_dev = TileTensor(
        expert_ids_dev_buffer, row_major(Coord(num_experts))
    )

    ctx.enqueue_copy(p_dev_buffer, p_host_ptr)
    ctx.enqueue_copy(b_dev_buffer, b_host_ptr)
    ctx.enqueue_copy(a_offsets_dev_buffer, a_offsets_host_ptr)
    ctx.enqueue_copy(expert_ids_dev_buffer, expert_ids_host_ptr)
    ctx.synchronize()

    # ---- Oracle: three naive grouped matmuls over the fused weight --------
    # A = plane 0 -> ref_q_full[m, j] = dot(B[g, j, :], P[0, m, :]).
    naive_grouped_matmul(
        ref_q_full,
        p_plane_q.as_immut(),
        b_dev.as_immut(),
        a_offsets_dev.as_immut(),
        expert_ids_dev.as_immut(),
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )
    # A = plane 1.
    naive_grouped_matmul(
        ref_k_full,
        p_plane_k.as_immut(),
        b_dev.as_immut(),
        a_offsets_dev.as_immut(),
        expert_ids_dev.as_immut(),
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )
    # A = plane 2.
    naive_grouped_matmul(
        ref_v_full,
        p_plane_v.as_immut(),
        b_dev.as_immut(),
        a_offsets_dev.as_immut(),
        expert_ids_dev.as_immut(),
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )
    ctx.synchronize()

    # ---- Kernel under test: ONE launch -----------------------------------
    expand_qkv_sm100(
        q_dev,
        kv_dev,
        p_dev,
        b_dev,
        a_offsets_dev,
        expert_ids_dev,
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )
    ctx.synchronize()

    ctx.enqueue_copy(q_host_ptr, q_dev_buffer)
    ctx.enqueue_copy(kv_host_ptr, kv_dev_buffer)
    var ref_q_host_full = ctx.enqueue_create_host_buffer[out_q_type](
        M * D_total
    )
    var ref_k_host_full = ctx.enqueue_create_host_buffer[out_kv_type](
        M * D_total
    )
    var ref_v_host_full = ctx.enqueue_create_host_buffer[out_kv_type](
        M * D_total
    )
    ctx.enqueue_copy(ref_q_host_full, ref_q_full_buffer)
    ctx.enqueue_copy(ref_k_host_full, ref_k_full_buffer)
    ctx.enqueue_copy(ref_v_host_full, ref_v_full_buffer)
    ctx.synchronize()

    var rtol = 1e-2

    # q_out[m, j] vs ref_q_full[m, j] for j in [0, q_dim).
    for m, j in std.itertools.product(range(M), range(q_dim)):
        var expect = ref_q_host_full[m * D_total + j]
        var actual = q_host_ptr[m * q_dim + j]
        assert_almost_equal(
            actual,
            expect,
            msg=String(t"Q m: {m} j: {j} ref: {expect} actual: {actual}"),
            rtol=rtol,
        )

    # kv_out[m, c] (K region) vs ref_k_full[m, q_dim + c] for c in [0, kv_dim).
    for m, c in std.itertools.product(range(M), range(kv_dim)):
        var expect = ref_k_host_full[m * D_total + q_dim + c]
        var actual = kv_host_ptr[m * kv_dim + c]
        assert_almost_equal(
            actual,
            expect,
            msg=String(t"K m: {m} c: {c} ref: {expect} actual: {actual}"),
            rtol=rtol,
        )

    # kv_out[M + m, c] (V region) vs ref_v_full[m, q_dim + kv_dim + c].
    for m, c in std.itertools.product(range(M), range(kv_dim)):
        var expect = ref_v_host_full[m * D_total + q_dim + kv_dim + c]
        var actual = kv_host_ptr[(M + m) * kv_dim + c]
        assert_almost_equal(
            actual,
            expect,
            msg=String(t"V m: {m} c: {c} ref: {expect} actual: {actual}"),
            rtol=rtol,
        )

    # Cleanup.
    _ = p_dev_buffer^
    _ = b_dev_buffer^
    _ = q_dev_buffer^
    _ = kv_dev_buffer^
    _ = ref_q_full_buffer^
    _ = ref_k_full_buffer^
    _ = ref_v_full_buffer^
    _ = a_offsets_dev_buffer^
    _ = expert_ids_dev_buffer^


def main() raises:
    with DeviceContext() as ctx:
        comptime is_sm100_kernel_applicable = _is_sm10x_gpu(
            ctx.default_device_info
        )

        comptime if not is_sm100_kernel_applicable:
            return

        # Single group, GQA (q_dim != kv_dim), small R, aligned tokens.
        test[
            .bfloat16,
            .bfloat16,
            num_experts=1,
            q_dim=256,
            kv_dim=64,
            R=16,
        ](1, [128], [0], ctx)

        # Single group, GQA, unaligned token count, R=32.
        test[
            .bfloat16,
            .bfloat16,
            num_experts=1,
            q_dim=256,
            kv_dim=64,
            R=32,
        ](1, [200], [0], ctx)

        # q_dim == kv_dim case, R=64.
        test[
            .bfloat16,
            .bfloat16,
            num_experts=1,
            q_dim=128,
            kv_dim=128,
            R=64,
        ](1, [128], [0], ctx)

        # Multiple active adapters with routing (expert_ids select a subset),
        # GQA, mixed aligned/unaligned token counts.
        test[
            .bfloat16,
            .bfloat16,
            num_experts=6,
            q_dim=512,
            kv_dim=128,
            R=32,
        ](4, [27, 300, 150, 64], [0, 3, 2, 4], ctx)

        # Routing with an inactive adapter in the middle (expert id -1): output
        # for that group must be zero. Place -1 as the second active group.
        test[
            .bfloat16,
            .bfloat16,
            num_experts=4,
            q_dim=256,
            kv_dim=64,
            R=16,
        ](3, [32, 48, 80], [1, -1, 2], ctx)

        # Larger token counts to exercise multiple M-tiles per group.
        test[
            .bfloat16,
            .bfloat16,
            num_experts=2,
            q_dim=512,
            kv_dim=128,
            R=64,
        ](2, [1000, 1500], [0, 1], ctx)

        # ---- SM100 tensor-core path: q_dim and kv_dim both multiples of 256 ----
        # The cases above have kv_dim not a multiple of 256, so they take the
        # naive fallback. These exercise the SM100 grouped-matmul specialization
        # (`a_row_offset_fn` plane select + epilogue routing), the perf path real
        # GQA models (e.g. 4096 / 1024) hit.

        # Single group, GQA, aligned dims, small R.
        test[
            .bfloat16,
            .bfloat16,
            num_experts=1,
            q_dim=512,
            kv_dim=256,
            R=16,
        ](1, [128], [0], ctx)

        # Multiple active adapters with routing (subset) + an inactive id (-1),
        # aligned dims, R=32, mixed aligned/unaligned token counts.
        test[
            .bfloat16,
            .bfloat16,
            num_experts=6,
            q_dim=768,
            kv_dim=256,
            R=32,
        ](4, [27, 300, 150, 64], [0, -1, 2, 4], ctx)

        # Larger token counts to exercise multiple token-tiles per group on the
        # SM100 path, aligned dims, R=64.
        test[
            .bfloat16,
            .bfloat16,
            num_experts=2,
            q_dim=512,
            kv_dim=256,
            R=64,
        ](2, [1000, 1500], [0, 1], ctx)

        # `D_total = 640` is not a multiple of 256, so cta_group == 1 and the
        # output-D tile is 128: dims that are 128-aligned (but not 256-aligned)
        # still take the SM100 path. Exercises the tighter `128 * cta_group` gate.
        test[
            .bfloat16,
            .bfloat16,
            num_experts=1,
            q_dim=384,
            kv_dim=128,
            R=16,
        ](1, [128], [0], ctx)

        # fp32 output, tile-aligned dims: the un-quantized projection path this PR is
        # motivated by. Tile-aligned dims take the grouped_matmul[...] branch, where the
        # c_is_fp32 gate must divert fp32 off the SM100 tensor-core kernel (which asserts
        # c_type != float32) to the naive fallback. Direct kernel-level regression for
        # that dispatch fix; also checks route_qkv + a_plane_splits under fp32.
        test[
            .float32,
            .float32,
            num_experts=1,
            q_dim=512,
            kv_dim=256,
            R=16,
        ](1, [128], [0], ctx)

        print("All lora expand qkv sm100 tests passed.")
