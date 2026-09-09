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
"""Unit tests for the Apple M5 weight-only FP8 (W8A16) grouped (MoE) matmul.

Target: Apple M5 (`compute_capability() == 5`, Metal 4). The grouped analog of
`test_fp8_gemv.mojo`: `bf16` activation x `float8_e4m3fn` expert-weight stack ->
`bf16`, one expert group per `block_idx.z`, produced RAW (the per-expert
`weight_scale` is a graph-level post-matmul fold, not part of the kernel -- see
`nemotron_h.NemotronHMoE`).

The reference is `naive_grouped_matmul` (the dtype-generic per-`(m, n)`-thread
kernel this path runs today on Apple): it widens the SAME `float8_e4m3fn` weight
bytes to fp32 and accumulates in fp32, so the only slack vs the tiled simdgroup
MMA is the fp32 reduction ORDER (and the bf16 activation rounding, shared by
both). Every case asserts BOTH:

- `grouped_matmul(...)` (the production dispatch, which on M5 routes W8A16 to the
  tiled kernel) == naive, AND
- `enqueue_grouped_matmul2d_fp8(...)` (the tiled kernel called directly, isolating
  it from the dispatch) == naive.

Coverage: aligned + tile-unaligned N/K, 0-token experts, an inactive
`expert_ids == -1` LoRA group (must produce zeros, matching naive), a
c32-decode-like many-tiny-group pattern, and the real Nemotron-3-Nano-30B-A3B
MoE expert dims (up-proj N=1856/K=2688, down-proj N=2688/K=1856) at a reduced
expert count.
"""

from std.collections import Optional

from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from layout._fillers import random
from linalg.grouped_matmul import grouped_matmul, naive_grouped_matmul
from linalg.matmul.gpu.apple.matmul2d_fp8 import enqueue_grouped_matmul2d_fp8
from std.testing import assert_almost_equal

from std.utils import IndexList
from std.utils.index import Index
import std.itertools


def _run_case[
    out_type: DType,
    num_experts: Int,
    expert_shape: IndexList[2],
](
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids: List[Int],
    ctx: DeviceContext,
    rtol: Float64 = 1e-2,
) raises:
    """Assert tiled W8A16 grouped matmul == `naive_grouped_matmul` at these dims.

    `expert_shape` is `(N, K)` (weight stack `[num_experts, N, K]`). Checks the
    production dispatch AND the direct tiled launcher against the naive reference.
    W8A16 fixes the input dtypes: `bf16` activation, `float8_e4m3fn` weight.
    """
    comptime in_type = DType.bfloat16
    comptime weight_type = DType.float8_e4m3fn
    comptime N = expert_shape[0]
    comptime K = expert_shape[1]

    print(
        num_active_experts,
        "active of",
        num_experts,
        "experts, (N, K) =",
        N,
        K,
        " tokens:",
        end="",
    )
    for i in range(len(num_tokens_by_expert)):
        print(num_tokens_by_expert[i], end=" ")
    print(" ids:", end="")
    for i in range(len(expert_ids)):
        print(expert_ids[i], end=" ")
    print()

    var total_num_tokens = 0
    var max_num_tokens_by_expert = 0
    for i in range(num_active_experts):
        total_num_tokens += num_tokens_by_expert[i]
        max_num_tokens_by_expert = max(
            max_num_tokens_by_expert, num_tokens_by_expert[i]
        )

    var a_size = total_num_tokens * K
    var c_size = total_num_tokens * N
    var b_size = num_experts * N * K

    # Host buffers. `b` uses runtime `Idx` dims in the filler view so `random`
    # loops at runtime instead of hitting the comptime element cap on the real
    # (large) expert dims; the DEVICE view below keeps N/K static (dispatch needs
    # a static expert shape to select the tiled kernel).
    var a_host_ptr = ctx.enqueue_create_host_buffer[in_type](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[weight_type](b_size)
    var c_disp_host_ptr = ctx.enqueue_create_host_buffer[out_type](c_size)
    var c_direct_host_ptr = ctx.enqueue_create_host_buffer[out_type](c_size)
    var c_ref_host_ptr = ctx.enqueue_create_host_buffer[out_type](c_size)
    var a_offsets_host_ptr = ctx.enqueue_create_host_buffer[.uint32](
        num_experts + 1
    )
    var expert_ids_host_ptr = ctx.enqueue_create_host_buffer[.int32](
        num_experts
    )

    var a_host = TileTensor(
        a_host_ptr, row_major(Coord(total_num_tokens, Idx[K]))
    )
    var b_host = TileTensor(
        b_host_ptr, row_major(Coord(num_experts, Idx[N], Idx[K]))
    )
    random(a_host)
    random(b_host)

    for i in range(num_experts + 1):
        a_offsets_host_ptr[i] = 0
    for i in range(num_experts):
        expert_ids_host_ptr[i] = 0
    for i in range(num_active_experts):
        a_offsets_host_ptr[i + 1] = a_offsets_host_ptr[i] + UInt32(
            num_tokens_by_expert[i]
        )
        expert_ids_host_ptr[i] = Int32(expert_ids[i])

    # Device buffers (static N/K on b_dev -> dispatch picks the tiled kernel).
    var a_dev_buf = ctx.enqueue_create_buffer[in_type](a_size)
    var b_dev_buf = ctx.enqueue_create_buffer[weight_type](b_size)
    var c_disp_dev_buf = ctx.enqueue_create_buffer[out_type](c_size)
    var c_direct_dev_buf = ctx.enqueue_create_buffer[out_type](c_size)
    var c_ref_dev_buf = ctx.enqueue_create_buffer[out_type](c_size)
    var off_dev_buf = ctx.enqueue_create_buffer[.uint32](num_experts + 1)
    var eid_dev_buf = ctx.enqueue_create_buffer[.int32](num_experts)

    var a_dev = TileTensor[in_type](
        a_dev_buf, row_major(Coord(total_num_tokens, Idx[K]))
    )
    var b_dev = TileTensor[weight_type](
        b_dev_buf, row_major[num_experts, N, K]()
    )
    var c_disp_dev = TileTensor[out_type](
        c_disp_dev_buf, row_major(Coord(total_num_tokens, Idx[N]))
    )
    var c_direct_dev = TileTensor[out_type](
        c_direct_dev_buf, row_major(Coord(total_num_tokens, Idx[N]))
    )
    var c_ref_dev = TileTensor[out_type](
        c_ref_dev_buf, row_major(Coord(total_num_tokens, Idx[N]))
    )
    var off_dev = TileTensor[.uint32](
        off_dev_buf, row_major(Coord(num_experts + 1))
    )
    var eid_dev = TileTensor[.int32](
        eid_dev_buf, row_major(Coord(Idx[num_experts]))
    )

    ctx.enqueue_copy(a_dev_buf, a_host_ptr)
    ctx.enqueue_copy(b_dev_buf, b_host_ptr)
    ctx.enqueue_copy(off_dev_buf, a_offsets_host_ptr)
    ctx.enqueue_copy(eid_dev_buf, expert_ids_host_ptr)
    ctx.synchronize()

    # Reference: dtype-generic naive grouped matmul (fp32 accumulate).
    naive_grouped_matmul(
        c_ref_dev,
        a_dev,
        b_dev,
        off_dev,
        eid_dev,
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )

    # Under test 1: the production dispatch (routes W8A16 -> tiled on M5).
    grouped_matmul(
        c_disp_dev,
        a_dev,
        b_dev,
        off_dev,
        eid_dev,
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )

    # Under test 2: the tiled launcher called directly (isolates the kernel).
    enqueue_grouped_matmul2d_fp8[c_type=out_type](
        c_direct_dev,
        a_dev,
        b_dev,
        off_dev,
        eid_dev,
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )
    ctx.synchronize()

    ctx.enqueue_copy(c_ref_host_ptr, c_ref_dev_buf)
    ctx.enqueue_copy(c_disp_host_ptr, c_disp_dev_buf)
    ctx.enqueue_copy(c_direct_host_ptr, c_direct_dev_buf)
    ctx.synchronize()

    for m, n in std.itertools.product(range(total_num_tokens), range(N)):
        var expect = c_ref_host_ptr[m * N + n]
        assert_almost_equal(
            c_disp_host_ptr[m * N + n],
            expect,
            msg=String(t"dispatch m: {m} n: {n}"),
            rtol=rtol,
        )
        assert_almost_equal(
            c_direct_host_ptr[m * N + n],
            expect,
            msg=String(t"direct m: {m} n: {n}"),
            rtol=rtol,
        )

    _ = a_dev_buf^
    _ = b_dev_buf^
    _ = c_disp_dev_buf^
    _ = c_direct_dev_buf^
    _ = c_ref_dev_buf^
    _ = off_dev_buf^
    _ = eid_dev_buf^
    print("PASS")


def main() raises:
    with DeviceContext() as ctx:
        # Apple M5 only (the native fp8-operand simdgroup MMA); skip elsewhere so
        # the target stays runnable on non-M5 Apple CI without a hard failure.
        if ctx.compute_capability() != 5:
            print("skip: grouped W8A16 tiled matmul requires Apple M5 (cc==5)")
            return

        comptime BF16 = DType.bfloat16

        # --- Small aligned: single group, tile-aligned N/K. ---
        _run_case[BF16, num_experts=1, expert_shape=Index(64, 32)](
            1, [8], [0], ctx
        )

        # --- Small routing: multiple groups, incl. a 0-token group. ---
        _run_case[BF16, num_experts=4, expert_shape=Index(128, 64)](
            4, [3, 0, 5, 2], [2, 0, 3, 1], ctx
        )

        # --- Tile-unaligned N and K (ragged store guard + K tail). ---
        _run_case[BF16, num_experts=4, expert_shape=Index(70, 40)](
            3, [7, 0, 20], [1, 0, 3], ctx
        )

        # --- Inactive LoRA group (expert_ids == -1) must produce zeros. ---
        _run_case[BF16, num_experts=2, expert_shape=Index(128, 64)](
            2, [16, 24], [0, -1], ctx
        )

        # --- c32-decode-like: many tiny (1-3 token) groups over many experts. ---
        _run_case[BF16, num_experts=16, expert_shape=Index(128, 96)](
            8, [1, 2, 1, 3, 1, 1, 2, 1], [0, 5, 2, 9, 1, 12, 7, 3], ctx
        )

        # --- Real Nemotron-30B-A3B MoE up-proj dims (N=1856, K=2688). ---
        # Reduced expert count keeps the weight buffer small; per-group M is
        # tiny (batch-1 / c32 decode) incl. a 0-token group.
        _run_case[BF16, num_experts=8, expert_shape=Index(1856, 2688)](
            6, [1, 1, 2, 0, 1, 1], [0, 3, 1, 4, 5, 2], ctx, rtol=2e-2
        )

        # --- Real Nemotron-30B-A3B MoE down-proj dims (N=2688, K=1856). ---
        _run_case[BF16, num_experts=8, expert_shape=Index(2688, 1856)](
            6, [2, 1, 1, 0, 3, 1], [1, 0, 4, 2, 5, 3], ctx, rtol=2e-2
        )

        print("all grouped W8A16 tiled matmul tests passed")
