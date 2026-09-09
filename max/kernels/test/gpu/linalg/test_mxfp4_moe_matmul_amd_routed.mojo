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
"""Exhaustive correctness tests for `mxfp4_moe_matmul_amd_routed`.

Three coverage layers, all using a host-side CPU reference (so we can
scale to full kimi shapes without paying GPU per-element-thread cost):

  Layer 1: structural — small shapes that exercise the kernel's
    bookkeeping (sort-block padding, sentinels, multi-block experts,
    inactive-expert skip, topk=1/2/4/8 fan-out, non-identity expert
    IDs).

  Layer 2: kimi-shape regimes — proportional to kimi's MoE projections
    (decode 1-user, 16-user, 64-user; prefill scaled; aspect ratios for
    gate+up vs down) at scaled-down N/K to keep CPU ref fast.

  Layer 3: full kimi shapes — gate+up (N=14336, K=4096) and down
    (N=4096, K=7168) with realistic decode batch.

Reference is `cpu_routed_reference`: pure host-side iteration with
manual FP4 / E8M0 dequant. Slow per-op but trivial to scale.

Test data uses a SplitMix-style hash to populate FP4 nibbles, ensuring
no token/col/expert collisions and that `c[m,n] != c[n,m]` (so
transpose bugs are detectable). Full nibble range 0..15 covers
negative FP4 values.
"""

from max.gpu.host import DeviceContext, HostBuffer
from max.gpu.host.info import MI355X
from std.math import ceildiv
from std.random import rand
from std.testing import assert_almost_equal

from layout import Coord, Idx, TileTensor, row_major

from linalg.matmul.gpu.amd import (
    InputRowMode,
    Shuffler,
    mxfp4_moe_matmul_amd_routed,
)


# ===----------------------------------------------------------------------=== #
# FP4 E2M1 / E8M0 dequant lookup for CPU reference.
# ===----------------------------------------------------------------------=== #


@always_inline
def _fp4_nibble_to_fp32(nibble: UInt8) -> Float32:
    """Decodes a 4-bit FP4 E2M1 nibble (lower 4 bits) to FP32."""
    var n = Int(nibble) & 0xF
    var sign = (n >> 3) & 1
    var exp = (n >> 1) & 0x3
    var mant = n & 0x1
    var mag: Float32
    if exp == 0:
        mag = 0.5 * Float32(mant)  # 0 or 0.5
    else:
        var frac = 1.0 + 0.5 * Float32(mant)
        mag = frac * Float32(1 << (exp - 1))
    return -mag if sign == 1 else mag


@always_inline
def _fp4_byte_to_fp32_pair(byte: UInt8) -> Tuple[Float32, Float32]:
    """Returns the (low, high) FP32 values stored in a packed FP4 byte."""
    var low = _fp4_nibble_to_fp32(byte & UInt8(0xF))
    var high = _fp4_nibble_to_fp32((byte >> 4) & UInt8(0xF))
    return (low, high)


@always_inline
def _e8m0_to_fp32(byte: UInt8) -> Float32:
    """Decodes an E8M0 (exponent-only) byte to FP32. Byte 127 = 1.0."""
    var e = Int(byte)
    if e == 0:
        return 0.0  # E8M0 has no zero rep, but treat 0 byte as 0
    # 2^(e - 127). Safe approach: bit-construct an FP32.
    return Float32(2.0) ** Float32(e - 127)


# ===----------------------------------------------------------------------=== #
# CPU reference — pure host iteration over the routed layout.
# ===----------------------------------------------------------------------=== #


comptime _MXFP4_GROUP: Int = 32  # elements per scale group


def cpu_routed_reference(
    a: TileTensor[.uint8, ...],
    b: TileTensor[.uint8, ...],
    sfa: TileTensor[.uint8, ...],
    sfb: TileTensor[.uint8, ...],
    pair_to_expert: List[Int],
    num_tokens: Int,
    topk: Int,
    N: Int,
    K: Int,
    is_token_slot: Bool,
    mut c_out: TileTensor[mut=True, .float32, ...],
):
    """Computes the routed MoE matmul output on the CPU as one GEMV per
    (token, slot) pair — independent of the kernel's sort-block structure.

    For each `(t, s)` pair, looks up the routed expert via
    `pair_to_expert[t * topk + s]` (-1 = inactive → output stays zero).
    Computes `acc[n] = dequant(A[a_row]) · dequant(B[expert][n])` for each
    n, where `a_row = t` (TOKEN_ID, up) or `t*topk+s` (TOKEN_SLOT, down).

    Decoupled from `sti`/`ei` so a bug in `build_routing_metadata` or the
    kernel's sti consumption surfaces as a mismatch rather than agreeing
    silently with itself.

    Logical shapes:
      a              [num_input_rows, k_bytes]   (FP4 packed)
      b              [num_experts, N, k_bytes]   (FP4 packed)
      sfa            [num_input_rows, k_groups]  (E8M0)
      sfb            [num_experts, N, k_groups]  (E8M0)
      pair_to_expert [num_tokens * topk]         (-1 = inactive)
      c_out          [num_tokens * topk, N]      (float32 output)
    """
    var k_groups = K // _MXFP4_GROUP

    # Zero-fill output (inactive pairs / out-of-range t,s leave zeros here).
    for c_row in range(num_tokens * topk):
        for n in range(N):
            c_out[Coord(c_row, n)] = Float32(0.0)

    for t in range(num_tokens):
        for s in range(topk):
            var c_row = t * topk + s
            var expert = pair_to_expert[c_row]
            if expert < 0:
                continue  # inactive (or no pair routed here)

            var a_row = (t * topk + s) if is_token_slot else t

            for n in range(N):
                var acc: Float32 = 0.0
                for kg in range(k_groups):
                    var a_scale = _e8m0_to_fp32(sfa[Coord(a_row, kg)][0])
                    var b_scale = _e8m0_to_fp32(sfb[Coord(expert, n, kg)][0])
                    for ki in range(_MXFP4_GROUP // 2):
                        var k_byte_idx = kg * (_MXFP4_GROUP // 2) + ki
                        var a_byte = a[Coord(a_row, k_byte_idx)][0]
                        var b_byte = b[Coord(expert, n, k_byte_idx)][0]
                        var a_pair = _fp4_byte_to_fp32_pair(a_byte)
                        var b_pair = _fp4_byte_to_fp32_pair(b_byte)
                        acc += a_pair[0] * a_scale * b_pair[0] * b_scale
                        acc += a_pair[1] * a_scale * b_pair[1] * b_scale

                c_out[Coord(c_row, n)] = Float32(acc)


# ===----------------------------------------------------------------------=== #
# Routing fixture builder — builds (sti, ei) for a given (n_e, expert_ids).
# ===----------------------------------------------------------------------=== #


def build_pair_to_expert(
    n_e: List[Int],
    expert_ids_input: List[Int],
    num_tokens: Int,
    topk: Int,
) -> List[Int]:
    """Maps each (t, s) pair (flat index = t*topk+s) to its routed expert.

    Mirrors `build_routing_metadata`'s pair-assignment policy but emits a
    direct (t, s) → expert table for the CPU reference. Pairs whose slot
    has expert_id == -1 (inactive) keep their default value of -1.
    """
    var p2e = List[Int]()
    for _ in range(num_tokens * topk):
        p2e.append(-1)

    var pair_idx = 0
    for slot in range(len(n_e)):
        var ne = n_e[slot]
        var expert = expert_ids_input[slot]
        for _ in range(ne):
            if expert >= 0:
                p2e[pair_idx] = expert
            pair_idx += 1
    return p2e^


def build_routing_metadata(
    n_e: List[Int],
    expert_ids_input: List[Int],
    topk: Int,
    sort_block_m: Int,
    mut sti_out: HostBuffer[.uint32],  # length size_expert_ids * sort_block_m
    mut ei_out: HostBuffer[.int32],  # length size_expert_ids
):
    """Builds `sorted_token_ids` and per-block `expert_ids` arrays.

    Assigns (t, s) pairs sequentially: pair `i` becomes (i // topk, i % topk).
    Expert e gets the next n_e[e] consecutive (t, s) pairs.
    """
    var pad = UInt32(0xFFFFFF)
    var num_active = len(n_e)
    var pair_idx = 0  # global running index into the (t, s) pair sequence
    var blk_idx = 0  # global running sort-block index

    for e in range(num_active):
        var n = n_e[e]
        var blocks_for_expert = ceildiv(n, sort_block_m)
        for blk in range(blocks_for_expert):
            ei_out[blk_idx + blk] = Int32(expert_ids_input[e])
            for r in range(sort_block_m):
                var sti_offset = (blk_idx + blk) * sort_block_m + r
                var local_idx = blk * sort_block_m + r
                if local_idx < n:
                    var t = pair_idx // topk
                    var s = pair_idx % topk
                    sti_out[sti_offset] = UInt32(t) | (UInt32(s) << 24)
                    pair_idx += 1
                else:
                    sti_out[sti_offset] = pad
        blk_idx += blocks_for_expert


# ===----------------------------------------------------------------------=== #
# Test runner — does setup, kernel launch, CPU reference, comparison.
# ===----------------------------------------------------------------------=== #


def run_routed_test_case[
    topk: Int,
    num_experts: Int,
    N: Int,
    K: Int,
    input_row_mode: InputRowMode = InputRowMode.TOKEN_ID,
](
    ctx: DeviceContext,
    name: String,
    n_e: List[Int],
    expert_ids_input: List[Int],
    rtol: Float64 = 1e-3,
    atol: Float64 = 1e-3,
) raises:
    """Runs one routed-kernel correctness test against a CPU reference.

    `n_e[i]` is how many tokens are routed to expert `expert_ids_input[i]`.
    Use `-1` in `expert_ids_input` for inactive experts.

    Example — `n_e=[100, 50, 30, 20]`, `expert_ids_input=[10, 5, 20, 0]`:
    100 tokens to expert 10, 50 to expert 5, 30 to expert 20, 20 to
    expert 0. With `sort_block_m=64`, expert 10 fills 2 sort blocks
    (64+36 padded), expert 5 fills 1 (50+14 padded), etc.

    `topk` controls how (t, s) pairs are encoded. `input_row_mode`
    selects:
      - TOKEN_ID  (default): A row = t        (stage 1 / up-projection)
      - TOKEN_SLOT:          A row = t*topk+s (stage 2 / down-projection,
                              A is `[num_tokens*topk, K_BYTES]`).
    Shapes (N, K) parameterize the matmul; BM=BN=64 fixed.
    """
    print("==", name)

    comptime BM = 64
    comptime sort_block_m = BM
    comptime k_bytes = K // 2
    comptime k_scales = K // 32
    comptime n_padded_scale = Shuffler[num_experts].scale_padded_mn(N)
    comptime sfb_per_expert_bytes = n_padded_scale * k_scales
    comptime sfa_per_block_bytes = sort_block_m * k_scales
    comptime is_token_slot = (
        input_row_mode._value == InputRowMode.TOKEN_SLOT._value
    )

    var num_active_experts = len(n_e)

    # the number of expert_ids blocks
    var size_expert_ids = 0
    for ne in n_e:
        size_expert_ids += ceildiv(ne, sort_block_m)

    var total_pairs = 0
    for ne in n_e:
        total_pairs += ne
    debug_assert(
        total_pairs % topk == 0,
        "sum(n_e) must be a multiple of topk — every token contributes",
        " exactly topk pairs in real routing.",
    )
    var num_tokens = total_pairs // topk

    # A's row count depends on the projection stage:
    #   TOKEN_ID  (up):   A = hidden state [num_tokens, D] — one row per token,
    #                     multiple slots of the same token read the same row.
    #   TOKEN_SLOT (down): A = activated intermediate [num_tokens*topk, I] —
    #                     each (token, slot) has its own row because up applied
    #                     per-expert weights + SwiGLU (different per slot).
    var num_input_rows = (num_tokens * topk) if is_token_slot else num_tokens

    # ---- Host buffers ----
    var a_h = ctx.enqueue_create_host_buffer[.uint8](num_input_rows * k_bytes)
    var b_h = ctx.enqueue_create_host_buffer[.uint8](num_experts * N * k_bytes)
    var sfa_h = ctx.enqueue_create_host_buffer[.uint8](
        num_input_rows * k_scales
    )
    var sfb_h = ctx.enqueue_create_host_buffer[.uint8](
        num_experts * N * k_scales
    )
    ctx.synchronize()

    # FP4-packed bytes: full byte range. E8M0 scales clamped to bytes
    # [125, 129] → dequant range 2^-2..2^2 = [0.25, 4.0] (matches the
    # convention in test_mxfp4_matmul_amd.mojo) so accumulators stay in
    # f32 range across large K while still exercising scale-dequant.
    rand(a_h.unsafe_ptr(), num_input_rows * k_bytes, min=0, max=255)
    rand(b_h.unsafe_ptr(), num_experts * N * k_bytes, min=0, max=255)
    rand(sfa_h.unsafe_ptr(), num_input_rows * k_scales, min=125, max=129)
    rand(
        sfb_h.unsafe_ptr(),
        num_experts * N * k_scales,
        min=125,
        max=129,
    )

    # ---- Routing tables ----
    var sti_h = ctx.enqueue_create_host_buffer[.uint32](
        size_expert_ids * sort_block_m
    )
    var ei_h = ctx.enqueue_create_host_buffer[.int32](size_expert_ids)
    ctx.synchronize()
    build_routing_metadata(
        n_e, expert_ids_input, topk, sort_block_m, sti_h, ei_h
    )

    # ---- Host preshuffle (SFB per expert, SFA per chunk; B is preshuffled on-GPU below) ----
    var sfb_pre_hb = ctx.enqueue_create_host_buffer[.uint8](
        num_experts * sfb_per_expert_bytes
    )
    var sfa_pre_hb = ctx.enqueue_create_host_buffer[.uint8](
        size_expert_ids * sfa_per_block_bytes
    )
    var sfa_scratch_hb = ctx.enqueue_create_host_buffer[.uint8](
        sfa_per_block_bytes
    )
    ctx.synchronize()

    var sfb_h_tt = TileTensor(
        sfb_h,
        row_major(Coord(Idx[num_experts], Idx[N], Idx[k_scales])),
    )
    _ = Shuffler[num_experts].preshuffle_scale_4d[MN=N, K_SCALES=k_scales](
        sfb_h_tt, sfb_pre_hb
    )

    # SFA: gather per sort_block then preshuffle. In TOKEN_SLOT mode the
    # source row is `t*topk+s` (per-slot scales); in TOKEN_ID mode it's
    # just `t`. `size_expert_ids` is runtime, so each block is preshuffled
    # individually with E=1 into a scratch HostBuffer, then copied into
    # the right offset of the main `sfa_pre_hb`.
    var sfa_block_gathered = ctx.enqueue_create_host_buffer[.uint8](
        sort_block_m * k_scales
    )
    ctx.synchronize()
    for blk in range(size_expert_ids):
        for r in range(sort_block_m):
            var fused = sti_h[blk * sort_block_m + r]
            var t = Int(fused & UInt32(0xFFFFFF))
            var s = Int(fused >> UInt32(24))
            var src_row = (t * topk + s) if is_token_slot else t
            for kg in range(k_scales):
                if t < num_tokens:
                    sfa_block_gathered[r * k_scales + kg] = sfa_h[
                        src_row * k_scales + kg
                    ]
                else:
                    sfa_block_gathered[r * k_scales + kg] = UInt8(0)
        var sfa_block_tt = TileTensor(
            sfa_block_gathered,
            row_major(Coord(Idx[1], Idx[sort_block_m], Idx[k_scales])),
        )
        _ = Shuffler[1].preshuffle_scale_4d[MN=sort_block_m, K_SCALES=k_scales](
            sfa_block_tt, sfa_scratch_hb
        )
        for i in range(sfa_per_block_bytes):
            sfa_pre_hb[blk * sfa_per_block_bytes + i] = sfa_scratch_hb[i]

    # ---- Device buffers + copy ----
    var a_dev = ctx.enqueue_create_buffer[.uint8](num_input_rows * k_bytes)
    var b_raw_dev = ctx.enqueue_create_buffer[.uint8](num_experts * N * k_bytes)
    var b_pre_dev = ctx.enqueue_create_buffer[.uint8](num_experts * N * k_bytes)
    var sfa_pre_dev = ctx.enqueue_create_buffer[.uint8](
        size_expert_ids * sfa_per_block_bytes
    )
    var sfb_pre_dev = ctx.enqueue_create_buffer[.uint8](
        num_experts * sfb_per_expert_bytes
    )
    var sti_dev = ctx.enqueue_create_buffer[.uint32](
        size_expert_ids * sort_block_m
    )
    var ei_dev = ctx.enqueue_create_buffer[.int32](size_expert_ids)
    var c_dev = ctx.enqueue_create_buffer[.float32](num_tokens * topk * N)

    ctx.enqueue_copy(a_dev, a_h)
    ctx.enqueue_copy(b_raw_dev, b_h)
    ctx.enqueue_copy(sfa_pre_dev, sfa_pre_hb)
    ctx.enqueue_copy(sfb_pre_dev, sfb_pre_hb)
    ctx.enqueue_copy(sti_dev, sti_h)
    ctx.enqueue_copy(ei_dev, ei_h)

    # GPU-side preshuffle b_raw_dev → b_pre_dev.
    var b_raw_dev_tt = TileTensor[mut=False, ...](
        b_raw_dev, row_major[num_experts, N, k_bytes]()
    )
    var b_pre_dev_tt = TileTensor[mut=True, ...](
        b_pre_dev,
        Shuffler[num_experts].b_5d_grouped_layout[N=N, K_BYTES=k_bytes],
    )
    Shuffler[num_experts].preshuffle_b_5d[N=N, K_BYTES=k_bytes](
        b_raw_dev_tt, b_pre_dev_tt, ctx
    )

    # Zero output buffer (sentinel rows / inactive blocks won't write).
    c_dev.enqueue_fill(Float32(0.0))

    # ---- TileTensors ----
    var a_tt = TileTensor[mut=False, ...](
        a_dev, row_major(Coord(num_input_rows, Idx[k_bytes]))
    )
    var b_pre_tt = TileTensor[mut=False, ...](
        b_pre_dev,
        row_major(Coord(Idx[1], Idx[num_experts * N * k_bytes])),
    )
    var sfa_pre_tt = TileTensor[mut=False, ...](
        sfa_pre_dev,
        row_major(Coord(Idx[1], size_expert_ids * sfa_per_block_bytes)),
    )
    var sfb_pre_tt = TileTensor[mut=False, ...](
        sfb_pre_dev,
        row_major(Coord(Idx[1], Idx[num_experts * sfb_per_expert_bytes])),
    )
    var sti_tt = TileTensor[mut=False, ...](
        sti_dev, row_major(Coord(size_expert_ids * sort_block_m))
    )
    var ei_tt = TileTensor[mut=False, ...](
        ei_dev, row_major(Coord(size_expert_ids))
    )
    var c_tt = TileTensor[mut=True, ...](
        c_dev, row_major(Coord(num_tokens * topk, Idx[N]))
    )

    # ---- Launch kernel ----
    mxfp4_moe_matmul_amd_routed[topk=topk, INPUT_ROW_MODE=input_row_mode](
        c_tt,
        a_tt,
        b_pre_tt,
        sfa_pre_tt,
        sfb_pre_tt,
        sti_tt,
        ei_tt,
        num_input_rows,
        size_expert_ids,
        ctx,
    )

    # ---- Compute CPU reference + compare ----
    var c_kernel_h = ctx.enqueue_create_host_buffer[.float32](
        num_tokens * topk * N
    )
    var c_ref_h = ctx.enqueue_create_host_buffer[.float32](
        num_tokens * topk * N
    )
    ctx.enqueue_copy(c_kernel_h, c_dev)
    ctx.synchronize()

    var a_cpu_tt = TileTensor(
        a_h, row_major(Coord(num_input_rows, Idx[k_bytes]))
    )
    var b_cpu_tt = TileTensor(
        b_h,
        row_major(Coord(Idx[num_experts], Idx[N], Idx[k_bytes])),
    )
    var sfa_cpu_tt = TileTensor(
        sfa_h, row_major(Coord(num_input_rows, Idx[k_scales]))
    )
    var sfb_cpu_tt = TileTensor(
        sfb_h,
        row_major(Coord(Idx[num_experts], Idx[N], Idx[k_scales])),
    )
    var c_ref_tt = TileTensor(
        c_ref_h, row_major(Coord(num_tokens * topk, Idx[N]))
    )
    var pair_to_expert = build_pair_to_expert(
        n_e, expert_ids_input, num_tokens, topk
    )
    cpu_routed_reference(
        a_cpu_tt,
        b_cpu_tt,
        sfa_cpu_tt,
        sfb_cpu_tt,
        pair_to_expert,
        num_tokens,
        topk,
        N,
        K,
        is_token_slot,
        c_ref_tt,
    )

    for r in range(num_tokens * topk):
        for n in range(N):
            assert_almost_equal(
                c_kernel_h[r * N + n],
                c_ref_h[r * N + n],
                rtol=rtol,
                atol=atol,
            )

    # ---- Cleanup ----

    print("PASS")


# ===----------------------------------------------------------------------=== #
# Helpers to build common n_e / expert_ids patterns.
# ===----------------------------------------------------------------------=== #


def uniform_active(num_active: Int, n_per: Int) -> Tuple[List[Int], List[Int]]:
    """Each of the first `num_active` experts gets `n_per` (t, s) pairs."""
    var n_e = List[Int]()
    var ei = List[Int]()
    for e in range(num_active):
        n_e.append(n_per)
        ei.append(e)
    return (n_e^, ei^)


def imbalanced_active(counts: List[Int]) -> Tuple[List[Int], List[Int]]:
    """Active experts at id 0..len(counts)-1 with the given per-expert counts.
    """
    var ei = List[Int]()
    for e in range(len(counts)):
        ei.append(e)
    return (counts.copy(), ei^)


# ===----------------------------------------------------------------------=== #
# Layer 1: structural correctness (small shapes).
# ===----------------------------------------------------------------------=== #


def test_l1_single_active_expert(ctx: DeviceContext) raises:
    var pair = uniform_active(num_active=1, n_per=64)
    run_routed_test_case[topk=1, num_experts=1, N=64, K=512](
        ctx,
        "L1.1 single-active-expert",
        pair[0],
        pair[1],
    )


def test_l1_underfilled_BM(ctx: DeviceContext) raises:
    var pair = uniform_active(num_active=8, n_per=1)
    run_routed_test_case[topk=1, num_experts=8, N=64, K=512](
        ctx,
        "L1.2 underfilled-BM (1 real + 63 pad per block)",
        pair[0],
        pair[1],
    )


def test_l1_just_over_BM(ctx: DeviceContext) raises:
    var pair = uniform_active(num_active=2, n_per=65)
    run_routed_test_case[topk=1, num_experts=2, N=64, K=512](
        ctx,
        "L1.3 just-over-BM (each expert spills to 2 blocks)",
        pair[0],
        pair[1],
    )


def test_l1_large_multi_block(ctx: DeviceContext) raises:
    var pair = uniform_active(num_active=1, n_per=256)
    run_routed_test_case[topk=1, num_experts=1, N=64, K=512](
        ctx,
        "L1.4 large-multi-block (1 expert, 4 blocks)",
        pair[0],
        pair[1],
    )


def test_l1_topk_8_fanout(ctx: DeviceContext) raises:
    var pair = uniform_active(num_active=4, n_per=8)
    run_routed_test_case[topk=8, num_experts=4, N=64, K=512](
        ctx,
        "L1.5 topk-8 fan-out",
        pair[0],
        pair[1],
    )


def test_l1_imbalanced(ctx: DeviceContext) raises:
    var counts = List[Int]()
    counts.append(100)
    counts.append(50)
    counts.append(30)
    counts.append(20)
    var pair = imbalanced_active(counts)
    run_routed_test_case[topk=1, num_experts=4, N=64, K=512](
        ctx,
        "L1.6 imbalanced (100/50/30/20)",
        pair[0],
        pair[1],
    )


def test_l1_all_inactive(ctx: DeviceContext) raises:
    # Three sort blocks, all expert_ids=-1. Kernel should produce zero output.
    var n_e = List[Int]()
    var ei = List[Int]()
    n_e.append(64)
    n_e.append(64)
    n_e.append(64)
    ei.append(-1)
    ei.append(-1)
    ei.append(-1)
    run_routed_test_case[topk=1, num_experts=1, N=64, K=512](
        ctx,
        "L1.7 all-inactive (-1 sentinels)",
        n_e,
        ei,
    )


def test_l1_mostly_inactive(ctx: DeviceContext) raises:
    # 8 sort blocks, only the middle one is active.
    var n_e = List[Int]()
    var ei = List[Int]()
    for i in range(8):
        n_e.append(32)
        ei.append(0 if i == 3 else -1)
    run_routed_test_case[topk=1, num_experts=1, N=64, K=512](
        ctx,
        "L1.8 mostly-inactive (1 active out of 8)",
        n_e,
        ei,
    )


def test_l1_topk_2_fanout(ctx: DeviceContext) raises:
    # topk=2 fan-out: each token visits 2 experts. With n_per=8 per active
    # slot and 4 active slots, total pairs = 32; num_tokens = 32/2 = 16.
    # Catches topk-specific bit-packing or c_row=t*topk+s arithmetic bugs
    # that don't appear at topk=1 or topk=8.
    var pair = uniform_active(num_active=4, n_per=8)
    run_routed_test_case[topk=2, num_experts=4, N=64, K=512](
        ctx,
        "L1.10 topk-2 fan-out",
        pair[0],
        pair[1],
    )


def test_l1_topk_4_fanout(ctx: DeviceContext) raises:
    # topk=4 fan-out: each token visits 4 experts. 4 active × n_per=8 = 32
    # pairs, num_tokens = 32/4 = 8.
    var pair = uniform_active(num_active=4, n_per=8)
    run_routed_test_case[topk=4, num_experts=4, N=64, K=512](
        ctx,
        "L1.11 topk-4 fan-out",
        pair[0],
        pair[1],
    )


def test_l1_token_slot_minimal(ctx: DeviceContext) raises:
    # TOKEN_SLOT (stage 2 / down): A is `[num_tokens*topk, K_BYTES]` with
    # per-slot rows. With topk=2 and 64 (t, s) pairs, num_tokens=32 and
    # A has 64 rows. Each slot must read a distinct A row.
    var pair = uniform_active(num_active=1, n_per=64)
    run_routed_test_case[
        topk=2,
        num_experts=1,
        N=64,
        K=512,
        input_row_mode=InputRowMode.TOKEN_SLOT,
    ](
        ctx,
        "L1.12 TOKEN_SLOT minimal (topk=2, 1 expert, 1 full block)",
        pair[0],
        pair[1],
    )


def test_l1_token_slot_partial(ctx: DeviceContext) raises:
    # TOKEN_SLOT with sentinel rows: 2 experts × 65 (t, s) pairs spills
    # each expert to 2 sort blocks with 63 padded sentinels. The kernel
    # must skip the sentinels per-row in TOKEN_SLOT mode too.
    var pair = uniform_active(num_active=2, n_per=65)
    run_routed_test_case[
        topk=2,
        num_experts=2,
        N=64,
        K=512,
        input_row_mode=InputRowMode.TOKEN_SLOT,
    ](
        ctx,
        "L1.13 TOKEN_SLOT partial-fill (topk=2, sentinel rows)",
        pair[0],
        pair[1],
    )


def test_l1_token_slot_topk4(ctx: DeviceContext) raises:
    # TOKEN_SLOT with topk=4. 4 active × 8 = 32 pairs, num_tokens=8,
    # A has 32 rows. Catches t*topk+s arithmetic for higher topk.
    var pair = uniform_active(num_active=4, n_per=8)
    run_routed_test_case[
        topk=4,
        num_experts=4,
        N=64,
        K=512,
        input_row_mode=InputRowMode.TOKEN_SLOT,
    ](
        ctx,
        "L1.14 TOKEN_SLOT topk=4",
        pair[0],
        pair[1],
    )


def test_l1_non_identity_expert_ids(ctx: DeviceContext) raises:
    # 4 active slots mapping to physical experts [10, 5, 20, 0].
    # Catches a kernel bug where it reads B from b_pre[slot] instead
    # of b_pre[expert_ids[slot]] (would only be invisible with identity
    # mappings — every prior test).
    var n_e = List[Int]()
    var ei = List[Int]()
    n_e.append(64)
    ei.append(10)
    n_e.append(64)
    ei.append(5)
    n_e.append(64)
    ei.append(20)
    n_e.append(64)
    ei.append(0)
    run_routed_test_case[topk=1, num_experts=24, N=64, K=512](
        ctx,
        "L1.9 non-identity expert_ids ([10, 5, 20, 0])",
        n_e,
        ei,
    )


# ===----------------------------------------------------------------------=== #
# Layer 2: kimi-shape regimes (scaled N/K to keep CPU ref fast).
# ===----------------------------------------------------------------------=== #


def test_l2_kimi_decode_1user(ctx: DeviceContext) raises:
    var pair = uniform_active(num_active=8, n_per=1)
    run_routed_test_case[topk=8, num_experts=8, N=512, K=2048](
        ctx,
        "L2.1 kimi-decode 1-user (k=8)",
        pair[0],
        pair[1],
    )


def test_l2_kimi_decode_16user(ctx: DeviceContext) raises:
    var n_e = List[Int]()
    var ei = List[Int]()
    for e in range(49):
        n_e.append(3 if e < 30 else 2)
        ei.append(e)
    run_routed_test_case[topk=8, num_experts=49, N=512, K=2048](
        ctx,
        "L2.2 kimi-decode 16-user (k=8, all 49 active)",
        n_e,
        ei,
    )


def test_l2_kimi_decode_64user(ctx: DeviceContext) raises:
    var n_e = List[Int]()
    var ei = List[Int]()
    for e in range(49):
        n_e.append(11 if e < 22 else 10)
        ei.append(e)
    run_routed_test_case[topk=8, num_experts=49, N=512, K=2048](
        ctx,
        "L2.3 kimi-decode 64-user (k=8, ~10 per expert)",
        n_e,
        ei,
    )


def test_l2_kimi_prefill_scaled(ctx: DeviceContext) raises:
    var pair = uniform_active(num_active=49, n_per=40)
    run_routed_test_case[topk=8, num_experts=49, N=512, K=2048](
        ctx,
        "L2.4 kimi-prefill scaled (49 active, 40/expert)",
        pair[0],
        pair[1],
    )


def test_l2_gate_up_aspect(ctx: DeviceContext) raises:
    var pair = uniform_active(num_active=8, n_per=8)
    run_routed_test_case[topk=8, num_experts=8, N=1024, K=512](
        ctx,
        "L2.5 gate+up aspect (N>>K)",
        pair[0],
        pair[1],
    )


def test_l2_down_aspect(ctx: DeviceContext) raises:
    var pair = uniform_active(num_active=8, n_per=8)
    run_routed_test_case[topk=8, num_experts=8, N=512, K=1024](
        ctx,
        "L2.6 down aspect (N<K)",
        pair[0],
        pair[1],
    )


# ===----------------------------------------------------------------------=== #
# Layer 3: full kimi shapes.
# ===----------------------------------------------------------------------=== #


def test_l3_kimi_gate_up_real(ctx: DeviceContext) raises:
    var pair = uniform_active(num_active=8, n_per=1)
    run_routed_test_case[topk=8, num_experts=8, N=14336, K=4096](
        ctx,
        "L3.1 kimi gate+up real (N=14336, K=4096)",
        pair[0],
        pair[1],
    )


def test_l3_kimi_down_real(ctx: DeviceContext) raises:
    var pair = uniform_active(num_active=8, n_per=1)
    run_routed_test_case[topk=8, num_experts=8, N=4096, K=7168](
        ctx,
        "L3.2 kimi down real (N=4096, K=7168)",
        pair[0],
        pair[1],
    )


# ===----------------------------------------------------------------------=== #
# main
# ===----------------------------------------------------------------------=== #


def main() raises:
    var ctx = DeviceContext()
    comptime assert (
        ctx.default_device_info == MI355X
    ), "MoE-routed MXFP4 matmul exhaustive test requires MI355X"

    # Layer 1 — structural
    test_l1_single_active_expert(ctx)
    test_l1_underfilled_BM(ctx)
    test_l1_just_over_BM(ctx)
    test_l1_large_multi_block(ctx)
    test_l1_topk_8_fanout(ctx)
    test_l1_imbalanced(ctx)
    test_l1_all_inactive(ctx)
    test_l1_mostly_inactive(ctx)
    test_l1_topk_2_fanout(ctx)
    test_l1_topk_4_fanout(ctx)
    test_l1_token_slot_minimal(ctx)
    test_l1_token_slot_partial(ctx)
    test_l1_token_slot_topk4(ctx)
    test_l1_non_identity_expert_ids(ctx)

    # Layer 2 — kimi-shape regimes
    test_l2_kimi_decode_1user(ctx)
    test_l2_kimi_decode_16user(ctx)
    test_l2_kimi_decode_64user(ctx)
    test_l2_kimi_prefill_scaled(ctx)
    test_l2_gate_up_aspect(ctx)
    test_l2_down_aspect(ctx)

    # Layer 3 — full kimi shapes
    test_l3_kimi_gate_up_real(ctx)
    test_l3_kimi_down_real(ctx)

    print("ALL EXHAUSTIVE TESTS PASSED")
