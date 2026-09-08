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
"""Unit tests for the `OnlineSoftmax` struct used by `MhaPrefillV2`
and `MlaPrefillV2Core`.

Runs the full FlashAttention-2 online-softmax recurrence
(`col_max_acc` -> `update_scale_unconditional` -> `rescale_output` ->
`sub_max` -> exp2 -> `apply_unconditional_norm_rescale` -> `col_sum_acc`,
then a final `normalize_output`) over a synthetic `att_block` register
tile across multiple "tiles" simulated by refilling `att_block` between
iterations. The struct is shared by the MHA and MLA prefill kernels but
never tested in isolation — these tests cover edge cases that the
end-to-end prefill
correctness tests cannot exercise individually (sub-normal exp2 outputs,
lazy-vs-eager equivalence, all-negative numerical stability).

Config: FP32 softmax path only. The FP16 softmax path is gated on
`FP8 + KV_BLOCK >= 128`; the BF16 attention path always uses FP32
softmax.

All tests launch a single 64-thread block (1 warp) and each lane
emits a per-lane diag tuple (o_avg, max_vec, norm_vec, scale_vec,
nan_count). Host-side checks read those diags.

Pipeline per tile (matches `_full_softmax_unconditional` semantics):

  1. att_block <- new tile's logits (synthetic; pre-scaled to log2 units)
  2. col_max_acc(att_block)
  3. update_scale_unconditional()
  4. rescale_output(o_reg)
  5. sub_max(att_block)
  6. exp2_inplace_range[0, ATT_PER_LANE](att_block)
  7. apply_unconditional_norm_rescale()
  8. col_sum_acc(att_block)
  9. (o_reg += sum(att_block) * v_value)  -- emulated PV with unit V.

After all tiles:
  10. normalize_output(o_reg)
"""

from max.gpu import lane_id
from max.gpu.host import DeviceContext
from std.math import exp2 as math_exp2
from std.testing import assert_almost_equal, assert_equal

from layout.tile_layout import row_major

from structured_kernels.amd_tile_io import RegTile, reg_alloc

from nn.attention.gpu.amd_structured.mha_softmax import OnlineSoftmax


# ----------------------------------------------------------------------------
# Comptime config. Use a 1-warp shape so 1 block = 1 warp = 64 lanes and the
# per-lane state covers all 32 cols of the col_l rt_32x32 stripe (each col
# replicated across the two half-warps). KV_BLOCK=64, Q_BLOCK_SIZE=32, with
# MMA_M=MMA_N=32 gives ATT_LAYOUT = row_major[2, 1, 16] -> 32 FP32/lane.
# ----------------------------------------------------------------------------
comptime Q_BLOCK_SIZE = 32
comptime KV_BLOCK = 64
comptime MMA_M = 32  # BF16/FP8-KV64 path
comptime MMA_N = 32
comptime ATT_H = KV_BLOCK // MMA_M  # 2
comptime ATT_W = Q_BLOCK_SIZE // MMA_N  # 1
comptime ATT_FRAG = (MMA_M * MMA_N) // 64  # 16
comptime ATT_PER_LANE = ATT_H * ATT_W * ATT_FRAG  # 32

# O_LAYOUT for the output accumulator. DEPTH=64.
comptime DEPTH = 64
comptime O_H = DEPTH // MMA_M  # 2
comptime O_W = Q_BLOCK_SIZE // MMA_N  # 1
comptime O_FRAG = (MMA_M * MMA_N) // 64  # 16
comptime O_PER_LANE = O_H * O_W * O_FRAG  # 32

comptime ATT_LAYOUT = row_major[ATT_H, ATT_W, ATT_FRAG]()
comptime O_LAYOUT = row_major[O_H, O_W, O_FRAG]()

comptime NUM_LANES = 64
comptime DIAG_PER_LANE = 5  # o_avg, max_vec, norm_vec, scale_vec, nan_count


# ----------------------------------------------------------------------------
# Tile helpers — local, kernel-only.
# ----------------------------------------------------------------------------


@always_inline
def _fill_att_block(
    mut att_block: RegTile[.float32, _, MutUntrackedOrigin],
    value: Float32,
):
    """Sets every per-lane element of `att_block` to `value`."""
    var att_v = att_block.vectorize[1, 1, ATT_FRAG]()
    comptime assert att_v.flat_rank == 3 and att_v.origin.mut
    var v_simd = att_v.ElementType(value)
    comptime for h in range(ATT_H):
        comptime for w in range(ATT_W):
            att_v[h, w, 0] = v_simd


@always_inline
def _fill_att_block_one_hot(
    mut att_block: RegTile[.float32, _, MutUntrackedOrigin],
    base_value: Float32,
    hot_value: Float32,
    hot_lane: Int,
):
    """Fills `att_block` with `base_value`; the lane whose `lane_id() ==
    hot_lane` overwrites its first per-lane element with `hot_value`.
    Used for the one-hot and sub-normal cases."""
    var att_v = att_block.vectorize[1, 1, ATT_FRAG]()
    comptime assert att_v.flat_rank == 3 and att_v.origin.mut
    var v_simd = att_v.ElementType(base_value)
    comptime for h in range(ATT_H):
        comptime for w in range(ATT_W):
            att_v[h, w, 0] = v_simd
    if Int(lane_id()) == hot_lane:
        var v0 = att_v[0, 0, 0]
        v0[0] = hot_value
        att_v[0, 0, 0] = v0


@always_inline
def _exp2_inplace(
    mut att_block: RegTile[.float32, _, MutUntrackedOrigin],
):
    """Local copy of `MhaMmaOp.exp2_inplace_range[0, ATT_PER_LANE]` —
    avoids pulling in a full `MhaMmaOp` instantiation for the unit
    tests."""
    var att_v = att_block.vectorize[1, 1, ATT_FRAG]()
    comptime assert att_v.flat_rank == 3 and att_v.origin.mut
    comptime for h in range(ATT_H):
        comptime for w in range(ATT_W):
            att_v[h, w, 0] = math_exp2(att_v[h, w, 0])


@always_inline
def _accumulate_o_from_att(
    mut o_reg: RegTile[.float32, _, MutUntrackedOrigin],
    att_block: RegTile[.float32, _, MutUntrackedOrigin],
    v_value: Float32,
):
    """`o_reg[*] += sum(att_block) * v_value * 2`. Emulates PV MFMA
    with a synthetic unit V so the row-state recurrence is visible at
    the o_reg level. The `*2` factor accounts for the fact that
    `col_sum_acc` accumulates BOTH half-warps' partial sums (via
    `permlane_swap[32]`) into `norm_vec`, so the synthetic PV needs
    the same cross-warp doubling for the final `o_reg / norm_vec` to
    settle at the expected uniform value (1.0 in the uniform-input
    case)."""
    var att_sum: Float32 = 0
    var att_v = att_block.vectorize[1, 1, ATT_FRAG]()
    comptime assert att_v.flat_rank == 3
    comptime for h in range(ATT_H):
        comptime for w in range(ATT_W):
            var elt = att_v[h, w, 0]
            comptime for k in range(ATT_FRAG):
                att_sum = att_sum + elt[k]
    # `* 2` matches the cross-warp doubling that `col_sum_acc` applies
    # to `norm_vec` via `permlane_swap[32]`.
    var contrib = att_sum * v_value * Float32(2)
    var o_v = o_reg.vectorize[1, 1, O_FRAG]()
    comptime assert o_v.flat_rank == 3 and o_v.origin.mut
    comptime for h in range(O_H):
        comptime for w in range(O_W):
            var elt = o_v[h, w, 0]
            comptime for k in range(O_FRAG):
                elt[k] = elt[k] + contrib
            o_v[h, w, 0] = elt


@always_inline
def _emit_diag(
    out_ptr: MutPointer[Float32, MutAnyOrigin],
    o_reg: RegTile[.float32, _, MutUntrackedOrigin],
    softmax: OnlineSoftmax,
):
    """Writes 5 diag scalars (o_avg, max_vec, norm_vec, scale_vec,
    nan_count) for this lane."""
    var lid = Int(lane_id())
    var base = lid * DIAG_PER_LANE
    var nan_count: Float32 = 0
    var o_sum: Float32 = 0
    var o_v = o_reg.vectorize[1, 1, O_FRAG]()
    comptime assert o_v.flat_rank == 3
    comptime for h in range(O_H):
        comptime for w in range(O_W):
            var elt = o_v[h, w, 0]
            comptime for k in range(O_FRAG):
                var x = elt[k]
                # x != x detects NaN (NaN is the only float != itself).
                if x != x:
                    nan_count = nan_count + Float32(1)
                else:
                    o_sum = o_sum + x
    out_ptr[base + 0] = o_sum / Float32(O_PER_LANE)
    out_ptr[base + 1] = softmax.max_vec
    out_ptr[base + 2] = softmax.norm_vec
    out_ptr[base + 3] = softmax.scale_vec
    out_ptr[base + 4] = nan_count


# ----------------------------------------------------------------------------
# Case 1: Uniform positive logits (all 1.0) -> uniform softmax.
# ----------------------------------------------------------------------------


def kernel_case1_uniform(
    out_ptr: MutPointer[Float32, MutAnyOrigin],
):
    var att_block = reg_alloc[.float32](ATT_LAYOUT)
    var o_reg = reg_alloc[.float32](O_LAYOUT)
    _ = att_block.fill(0)
    _ = o_reg.fill(0)
    var softmax = OnlineSoftmax()

    # Tile 0: seed.
    _fill_att_block(att_block, Float32(1.0))
    softmax.seed_tile0(att_block)
    _exp2_inplace(att_block)
    softmax.col_sum_acc(att_block)
    _accumulate_o_from_att(o_reg, att_block, Float32(1.0))

    # Tiles 1..3: full recurrence.
    comptime for t in range(1, 4):
        _fill_att_block(att_block, Float32(1.0))
        softmax.col_max_acc(att_block)
        softmax.update_scale_unconditional()
        softmax.rescale_output(o_reg)
        softmax.sub_max(att_block)
        _exp2_inplace(att_block)
        softmax.apply_unconditional_norm_rescale()
        softmax.col_sum_acc(att_block)
        _accumulate_o_from_att(o_reg, att_block, Float32(1.0))

    softmax.normalize_output(o_reg)
    _emit_diag(out_ptr, o_reg, softmax)


# ----------------------------------------------------------------------------
# Case 2: One dominant logit on a single lane in tile 1.
# ----------------------------------------------------------------------------


def kernel_case2_one_hot(
    out_ptr: MutPointer[Float32, MutAnyOrigin],
):
    var att_block = reg_alloc[.float32](ATT_LAYOUT)
    var o_reg = reg_alloc[.float32](O_LAYOUT)
    _ = att_block.fill(0)
    _ = o_reg.fill(0)
    var softmax = OnlineSoftmax()

    # Tile 0: all zeros.
    _fill_att_block(att_block, Float32(0.0))
    softmax.seed_tile0(att_block)
    _exp2_inplace(att_block)
    softmax.col_sum_acc(att_block)
    _accumulate_o_from_att(o_reg, att_block, Float32(1.0))

    # Tile 1: lane 5 gets a huge hot value at element 0; others stay at 0.
    _fill_att_block_one_hot(att_block, Float32(0.0), Float32(20.0), 5)
    softmax.col_max_acc(att_block)
    softmax.update_scale_unconditional()
    softmax.rescale_output(o_reg)
    softmax.sub_max(att_block)
    _exp2_inplace(att_block)
    softmax.apply_unconditional_norm_rescale()
    softmax.col_sum_acc(att_block)
    _accumulate_o_from_att(o_reg, att_block, Float32(1.0))

    softmax.normalize_output(o_reg)
    _emit_diag(out_ptr, o_reg, softmax)


# ----------------------------------------------------------------------------
# Case 3: All negative mean-shifted (-10). After subtract-max -> 0; should
# match case 1.
# ----------------------------------------------------------------------------


def kernel_case3_negative(
    out_ptr: MutPointer[Float32, MutAnyOrigin],
):
    var att_block = reg_alloc[.float32](ATT_LAYOUT)
    var o_reg = reg_alloc[.float32](O_LAYOUT)
    _ = att_block.fill(0)
    _ = o_reg.fill(0)
    var softmax = OnlineSoftmax()

    _fill_att_block(att_block, Float32(-10.0))
    softmax.seed_tile0(att_block)
    _exp2_inplace(att_block)
    softmax.col_sum_acc(att_block)
    _accumulate_o_from_att(o_reg, att_block, Float32(1.0))

    comptime for t in range(1, 4):
        _fill_att_block(att_block, Float32(-10.0))
        softmax.col_max_acc(att_block)
        softmax.update_scale_unconditional()
        softmax.rescale_output(o_reg)
        softmax.sub_max(att_block)
        _exp2_inplace(att_block)
        softmax.apply_unconditional_norm_rescale()
        softmax.col_sum_acc(att_block)
        _accumulate_o_from_att(o_reg, att_block, Float32(1.0))

    softmax.normalize_output(o_reg)
    _emit_diag(out_ptr, o_reg, softmax)


# ----------------------------------------------------------------------------
# Case 4: Sub-normal boundary. Pattern: tile 0 has max=120 in lane 0
# (others 0); tile 1 all zero. In tile 1 on lane 0: max stays at 120,
# `att - max_vec = -120` -> exp2(-120) ~= 7.5e-37 (in the sub-normal-emitting
# region of gfx950 v_exp_f32). gfx950 does not flush these to zero.
# Verify the rescale chain handles them without NaN in o_reg.
# ----------------------------------------------------------------------------


def kernel_case4_subnormal(
    out_ptr: MutPointer[Float32, MutAnyOrigin],
):
    var att_block = reg_alloc[.float32](ATT_LAYOUT)
    var o_reg = reg_alloc[.float32](O_LAYOUT)
    _ = att_block.fill(0)
    _ = o_reg.fill(0)
    var softmax = OnlineSoftmax()

    _fill_att_block_one_hot(att_block, Float32(0.0), Float32(120.0), 0)
    softmax.seed_tile0(att_block)
    _exp2_inplace(att_block)
    softmax.col_sum_acc(att_block)
    _accumulate_o_from_att(o_reg, att_block, Float32(1.0))

    _fill_att_block(att_block, Float32(0.0))
    softmax.col_max_acc(att_block)
    softmax.update_scale_unconditional()
    softmax.rescale_output(o_reg)
    softmax.sub_max(att_block)
    _exp2_inplace(att_block)
    softmax.apply_unconditional_norm_rescale()
    softmax.col_sum_acc(att_block)
    _accumulate_o_from_att(o_reg, att_block, Float32(1.0))

    softmax.normalize_output(o_reg)
    _emit_diag(out_ptr, o_reg, softmax)


# ----------------------------------------------------------------------------
# Case 5: Lazy vs eager rescale.
#
# Eager: always run `update_scale_unconditional` + `rescale_output`.
# Lazy:  use `lazy_rescale_decision` (skips rescale when max growth is
#        small) + `apply_norm_rescale_if_pending`.
#
# Logits chosen so the rescale fires at tile 3:
#   tile 0: 0.0
#   tile 1: 0.5   (small bump)
#   tile 2: 1.0   (small)
#   tile 3: 10.0  (jump > RESCALE_THRESHOLD=8 -> lazy path fires)
# ----------------------------------------------------------------------------


@always_inline
def _logit_for_tile(t: Int) -> Float32:
    """Logit pattern for case 5; selected at runtime since the index is
    iteration-driven. Using a flat function instead of an Array
    sidesteps a constructor-overload picker mismatch in the test env."""
    if t == 0:
        return Float32(0.0)
    if t == 1:
        return Float32(0.5)
    if t == 2:
        return Float32(1.0)
    return Float32(10.0)


def kernel_case5_eager(
    out_ptr: MutPointer[Float32, MutAnyOrigin],
):
    var att_block = reg_alloc[.float32](ATT_LAYOUT)
    var o_reg = reg_alloc[.float32](O_LAYOUT)
    _ = att_block.fill(0)
    _ = o_reg.fill(0)
    var softmax = OnlineSoftmax()

    _fill_att_block(att_block, _logit_for_tile(0))
    softmax.seed_tile0(att_block)
    _exp2_inplace(att_block)
    softmax.col_sum_acc(att_block)
    _accumulate_o_from_att(o_reg, att_block, Float32(1.0))

    comptime for t in range(1, 4):
        _fill_att_block(att_block, _logit_for_tile(t))
        softmax.col_max_acc(att_block)
        softmax.update_scale_unconditional()
        softmax.rescale_output(o_reg)
        softmax.sub_max(att_block)
        _exp2_inplace(att_block)
        softmax.apply_unconditional_norm_rescale()
        softmax.col_sum_acc(att_block)
        _accumulate_o_from_att(o_reg, att_block, Float32(1.0))

    softmax.normalize_output(o_reg)
    _emit_diag(out_ptr, o_reg, softmax)


def kernel_case5_lazy(
    out_ptr: MutPointer[Float32, MutAnyOrigin],
):
    var att_block = reg_alloc[.float32](ATT_LAYOUT)
    var o_reg = reg_alloc[.float32](O_LAYOUT)
    # `att_bf16_full` plays the role of the previous tile's already-exp'd
    # attention values still pending in the PV pipeline. Production's
    # `lazy_rescale_decision` rescales it by `scale_vec` alongside `o_reg`
    # (#87284) so strips 1..3 consume it at the post-rescale scale. This
    # single-`att_block`-per-tile test carries NO pending att across tiles
    # (each tile's att is fully consumed by `_accumulate_o_from_att` before
    # the next fill), so rescaling this scratch tile is inert on the
    # validated row-state (o_reg / max / norm / scale) — it exists only to
    # exercise the real 3-arg signature and keep lazy == eager exact.
    var att_scratch = reg_alloc[.float32](ATT_LAYOUT)
    _ = att_block.fill(0)
    _ = o_reg.fill(0)
    _ = att_scratch.fill(0)
    var softmax = OnlineSoftmax()

    var threshold = Float32(8.0)  # same as MhaConfigV2 default

    _fill_att_block(att_block, _logit_for_tile(0))
    softmax.seed_tile0(att_block)
    _exp2_inplace(att_block)
    softmax.col_sum_acc(att_block)
    _accumulate_o_from_att(o_reg, att_block, Float32(1.0))

    comptime for t in range(1, 4):
        _fill_att_block(att_block, _logit_for_tile(t))
        softmax.col_max_acc(att_block)
        var pending = softmax.lazy_rescale_decision(
            o_reg, att_scratch, threshold
        )
        softmax.sub_max(att_block)
        _exp2_inplace(att_block)
        softmax.apply_norm_rescale_if_pending(pending)
        softmax.col_sum_acc(att_block)
        _accumulate_o_from_att(o_reg, att_block, Float32(1.0))

    softmax.normalize_output(o_reg)
    _emit_diag(out_ptr, o_reg, softmax)


# ----------------------------------------------------------------------------
# Host drivers.
# ----------------------------------------------------------------------------


def test_case1_uniform(ctx: DeviceContext) raises:
    print("--- Case 1: Uniform positive logits ---")
    var size = NUM_LANES * DIAG_PER_LANE
    var dev_out = ctx.enqueue_create_buffer[.float32](size)
    _ = dev_out.enqueue_fill(Float32(-1))
    ctx.enqueue_function[kernel_case1_uniform](
        dev_out, grid_dim=1, block_dim=NUM_LANES
    )
    with dev_out.map_to_host() as host_out:
        var total_nans: Int = 0
        for lane in range(NUM_LANES):
            total_nans += Int(host_out[lane * DIAG_PER_LANE + 4])
        print(
            "  lane 0: o_avg=",
            host_out[0],
            " max=",
            host_out[1],
            " norm=",
            host_out[2],
            " scale=",
            host_out[3],
            " total_nans=",
            total_nans,
        )
        assert_equal(total_nans, 0)
        # Uniform inputs: every tile contributes ATT_PER_LANE elements of
        # value exp2(0)=1. Numerator after 4 tiles = 4 * ATT_PER_LANE; norm
        # = 4 * ATT_PER_LANE. Final o_reg / norm = 1.
        for lane in range(NUM_LANES):
            var o_avg = host_out[lane * DIAG_PER_LANE + 0]
            assert_almost_equal(o_avg, Float32(1.0), atol=1e-4)
    _ = dev_out^
    print("  PASS")


def test_case2_one_hot(ctx: DeviceContext) raises:
    print("--- Case 2: One dominant logit ---")
    var size = NUM_LANES * DIAG_PER_LANE
    var dev_out = ctx.enqueue_create_buffer[.float32](size)
    _ = dev_out.enqueue_fill(Float32(-1))
    ctx.enqueue_function[kernel_case2_one_hot](
        dev_out, grid_dim=1, block_dim=NUM_LANES
    )
    with dev_out.map_to_host() as host_out:
        var total_nans: Int = 0
        for lane in range(NUM_LANES):
            total_nans += Int(host_out[lane * DIAG_PER_LANE + 4])
        print(
            "  hot lane 5: o_avg=",
            host_out[5 * DIAG_PER_LANE + 0],
            " max=",
            host_out[5 * DIAG_PER_LANE + 1],
            " norm=",
            host_out[5 * DIAG_PER_LANE + 2],
            " scale=",
            host_out[5 * DIAG_PER_LANE + 3],
        )
        print(
            "  cold lane 0: o_avg=",
            host_out[0],
            " max=",
            host_out[1],
            " norm=",
            host_out[2],
            " scale=",
            host_out[3],
            " total_nans=",
            total_nans,
        )
        assert_equal(total_nans, 0)
        # Cold lanes: 2 tiles of all-zero -> uniform o = 1.0 (same logic
        # as case 1 with 2 tiles).
        # Lane 5 sees the hot value directly; lane 37 sees it via
        # `permlane_swap[32]` cross-warp reduction (col_max_acc /
        # col_sum_acc combine both half-warps' partial sums into the
        # row state). Skip both.
        for lane in range(NUM_LANES):
            if lane != 5 and lane != 37:
                var o_avg = host_out[lane * DIAG_PER_LANE + 0]
                assert_almost_equal(o_avg, Float32(1.0), atol=1e-3)
        # Hot lanes (5, 37): just check no NaN (already asserted via total_nans).
    _ = dev_out^
    print("  PASS")


def test_case3_negative(ctx: DeviceContext) raises:
    print("--- Case 3: All negative mean-shifted (-10) ---")
    var size = NUM_LANES * DIAG_PER_LANE
    var dev_out = ctx.enqueue_create_buffer[.float32](size)
    _ = dev_out.enqueue_fill(Float32(-1))
    ctx.enqueue_function[kernel_case3_negative](
        dev_out, grid_dim=1, block_dim=NUM_LANES
    )
    with dev_out.map_to_host() as host_out:
        var total_nans: Int = 0
        for lane in range(NUM_LANES):
            total_nans += Int(host_out[lane * DIAG_PER_LANE + 4])
        print(
            "  lane 0: o_avg=",
            host_out[0],
            " max=",
            host_out[1],
            " norm=",
            host_out[2],
            " scale=",
            host_out[3],
            " total_nans=",
            total_nans,
        )
        assert_equal(total_nans, 0)
        # Mean-shifted -10: subtract-max -> all zeros -> identical to case 1.
        for lane in range(NUM_LANES):
            var o_avg = host_out[lane * DIAG_PER_LANE + 0]
            assert_almost_equal(o_avg, Float32(1.0), atol=1e-4)
    _ = dev_out^
    print("  PASS")


def test_case4_subnormal(ctx: DeviceContext) raises:
    print("--- Case 4: Sub-normal boundary (lane 0 max=120 in tile 0) ---")
    var size = NUM_LANES * DIAG_PER_LANE
    var dev_out = ctx.enqueue_create_buffer[.float32](size)
    _ = dev_out.enqueue_fill(Float32(-1))
    ctx.enqueue_function[kernel_case4_subnormal](
        dev_out, grid_dim=1, block_dim=NUM_LANES
    )
    with dev_out.map_to_host() as host_out:
        var total_nans: Int = 0
        for lane in range(NUM_LANES):
            total_nans += Int(host_out[lane * DIAG_PER_LANE + 4])
        print(
            "  hot lane 0: o_avg=",
            host_out[0],
            " max=",
            host_out[1],
            " norm=",
            host_out[2],
            " scale=",
            host_out[3],
        )
        print("  cold lane 1: o_avg=", host_out[DIAG_PER_LANE + 0])
        print("  total_nans=", total_nans)
        # Load-bearing assertion: no NaN anywhere. The sub-normal exp2
        # values must propagate cleanly through the rescale chain.
        assert_equal(total_nans, 0)
        # On lane 0, max_vec should be 120 (tile-0 max preserved through
        # tile-1's col_max_acc since prev > new).
        var lane0_max = host_out[1]
        assert_almost_equal(lane0_max, Float32(120.0), atol=1e-4)
    _ = dev_out^
    print("  PASS")


def test_case5_lazy_vs_eager(ctx: DeviceContext) raises:
    print("--- Case 5: Lazy vs eager rescale equivalence ---")
    var size = NUM_LANES * DIAG_PER_LANE
    var dev_eager = ctx.enqueue_create_buffer[.float32](size)
    var dev_lazy = ctx.enqueue_create_buffer[.float32](size)
    _ = dev_eager.enqueue_fill(Float32(-1))
    _ = dev_lazy.enqueue_fill(Float32(-1))
    ctx.enqueue_function[kernel_case5_eager](
        dev_eager, grid_dim=1, block_dim=NUM_LANES
    )
    ctx.enqueue_function[kernel_case5_lazy](
        dev_lazy, grid_dim=1, block_dim=NUM_LANES
    )
    with dev_eager.map_to_host() as host_e, dev_lazy.map_to_host() as host_l:
        var nans_e: Int = 0
        var nans_l: Int = 0
        for lane in range(NUM_LANES):
            nans_e += Int(host_e[lane * DIAG_PER_LANE + 4])
            nans_l += Int(host_l[lane * DIAG_PER_LANE + 4])
        var max_diff: Float32 = 0
        for lane in range(NUM_LANES):
            var d = abs(
                host_e[lane * DIAG_PER_LANE + 0]
                - host_l[lane * DIAG_PER_LANE + 0]
            )
            if d > max_diff:
                max_diff = d
        print(
            "  eager lane 0: o_avg=",
            host_e[0],
            " max=",
            host_e[1],
            " norm=",
            host_e[2],
            " scale=",
            host_e[3],
        )
        print(
            "  lazy  lane 0: o_avg=",
            host_l[0],
            " max=",
            host_l[1],
            " norm=",
            host_l[2],
            " scale=",
            host_l[3],
        )
        print("  max o_avg divergence across lanes:", max_diff)
        print("  eager nans=", nans_e, " lazy nans=", nans_l)
        assert_equal(nans_e, 0)
        assert_equal(nans_l, 0)
        # FP-rounding tolerance: 1e-3 on average o_reg per lane.
        for lane in range(NUM_LANES):
            var a = host_e[lane * DIAG_PER_LANE + 0]
            var b = host_l[lane * DIAG_PER_LANE + 0]
            assert_almost_equal(a, b, atol=1e-3)
    _ = dev_eager^
    _ = dev_lazy^
    print("  PASS")


def main() raises:
    print("=" * 60)
    print("OnlineSoftmax UNIT TESTS")
    print("=" * 60)
    with DeviceContext() as ctx:
        test_case1_uniform(ctx)
        test_case2_one_hot(ctx)
        test_case3_negative(ctx)
        test_case4_subnormal(ctx)
        test_case5_lazy_vs_eager(ctx)
    print("=" * 60)
    print("ALL UNIT TESTS PASSED")
    print("=" * 60)
