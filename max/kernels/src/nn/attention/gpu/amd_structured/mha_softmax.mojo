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
"""Online softmax row-state bundle for `MhaPrefillV2`.

`OnlineSoftmax` (below) owns the four row-state scalars maintained by
the FlashAttention-2 online-softmax recurrence (`max_vec`,
`max_vec_prev`, `norm_vec`, `scale_vec`) as `Float32` fields and exposes
the recurrence steps + the cross-lane reductions and column-broadcast
ops as static methods. Cluster fns in `mha_prefill_v2.mojo` thread a
single `OnlineSoftmax` reference instead of four `RegTile` parameters.

The col_l rt_32x32 accumulator topology gives each lane ownership of
one column of the 32x32 fragment, which corresponds to one Q row in the
warp's stripe. The online-softmax recurrence therefore tracks one
running max + one running norm + one pending scale **per lane**: each
of the four pieces of state is a single FP32 scalar in a VGPR.

Each column of an rt_32x32 is held redundantly across two half-warps
(lanes `[0, 32)` and `[32, 64)`). Per-column reductions combine in-lane
via `SIMD.reduce_*` and then across half-warps via `permlane_swap[32]`,
a single-cycle DPP-style swap. Using stdlib's `lane_group_reduce`
here would lower to `ds_bpermute_b32` (LDS-routed), so we go through
`permlane_swap` directly.
"""

from max.gpu.intrinsics import permlane_swap
from std.math import exp2 as math_exp2, recip
from std.sys.intrinsics import unlikely

from layout import TensorLayout

from structured_kernels.amd_tile_io import RegTile


# ===----------------------------------------------------------------------=== #
# OnlineSoftmax
# ===----------------------------------------------------------------------=== #


struct OnlineSoftmax[att_dtype: DType = .float32](ImplicitlyCopyable):
    """Online softmax row-state bundle for `MhaPrefillV2`.

    Parametrized on `att_dtype`: the dtype of the `att_block`
    `RegTile` this state operates against (`FP32` for the BF16 prefill
    path and FP8 + KV<128; `FP16` for FP8 + KV>=128 per
    `_SOFTMAX_DTYPE` in `MhaPrefillV2` / `MlaPrefillV2`). All
    `att_block`-touching
    methods bind to `Self.att_dtype` so the type checker rejects
    mismatched dtypes at the call site instead of silently coercing.
    The four state scalars stay `Float32` regardless, see
    "Accumulator dtype rationale" below.

    Owns the four row-state scalars maintained by the FlashAttention-2
    online-softmax recurrence as direct `Float32` fields:

    - `max_vec`: running rowmax (in log2 units; the reference prescales
      Q by `scale * log2(e)` so att values are already in log2 units).
    - `max_vec_prev`: rowmax from the previous tile. Used by the
      lazy-rescale comparison and the unconditional rescale's
      `exp2(prev - new)`. Shadow-updated to `max_vec` after each
      consumed cluster.
    - `norm_vec`: running denominator (exp-sum so far). Consumed at
      Epi-C12's `o_reg /= norm_vec`.
    - `scale_vec`: pending rescale factor `exp2(max_prev - max_new)`.
      Conditionally applied to `o_reg` during lazy-rescale (main loop)
      or unconditionally during the epilogue tails. Reset to 1 when no
      rescale fired so `norm_vec *= scale_vec` is a safe identity.

    Each field is a per-lane FP32 scalar (1 VGPR/lane). The col_l
    rt_32x32 topology gives each lane ownership of one column of the
    fragment = one Q row in the warp's stripe, so per-lane scalar state
    is the natural representation. (Each column is held redundantly
    across the two half-warps sharing it; both lanes store their own
    copy of the identical reduced value.)

    Lifetime (MhaPrefillV2):
    - `max_vec`, `max_vec_prev`, `scale_vec`: prologue → Epi-C10
      (last touched by `_full_softmax_unconditional` + the final
      `rescale_output(o_reg)`).
    - `norm_vec`: prologue → Epi-C12 (consumed by
      `normalize_output(o_reg)`, three clusters after the other three
      die).

    `norm_vec` outliving the other three by two clusters costs 1 VGPR
    over Epi-C11/C12; not register-pressure-constrained there.

    Accumulator dtype rationale (why state stays `Float32` regardless
    of `att_dtype`): the four scalars are per-lane (1 VGPR/lane each),
    so narrowing to FP16 saves zero VGPRs (registers are dword-granular;
    a half-VGPR scalar costs pack/unpack ALU on every read). `norm_vec`
    sums up to ~seq_len terms of `exp2(att - max)` ∈ [0, 1]; at
    seq=8192 the sum reaches ~2^13, leaving little FP16 headroom.
    Hardware `v_exp_f32` / `v_rcp_f32` is the FP32 path. Narrowing
    `att_dtype` to FP16 only pays off on the larger `att_block` tile
    storage, not on the recurrence scalars.

    Parameters:
        att_dtype: DType of the `att_block` `RegTile` this state operates
            against. `FP32` for the BF16 prefill path and FP8 KV<128;
            `FP16` for FP8 KV>=128. The four state scalars stay `Float32`
            regardless of this parameter.
    """

    var max_vec: Float32
    var max_vec_prev: Float32
    var norm_vec: Float32
    var scale_vec: Float32

    # ---- Lifecycle ----

    @always_inline
    def __init__(out self):
        """No-sink init: `max_vec`/`max_vec_prev`/`norm_vec` to 0,
        `scale_vec` to 1 so the epilogue's unconditional
        `norm_vec *= scale_vec` is a safe no-op when no rescale fired.
        """
        self.scale_vec = Float32(1.0)
        self.norm_vec = Float32(0.0)
        self.max_vec = Float32(0.0)
        self.max_vec_prev = Float32(0.0)

    @always_inline
    def reseed_with_sink(mut self, sw_log2: Float32):
        """Re-init for the sink path: pre-seed the recurrence with the
        virtual sink token's contribution.

        `max_vec = max_vec_prev = log2e * sink_weight` keeps the rowmax
        in log2 units (the reference prescales Q by `scale * log2e`, so
        att values are in log2 units). `norm_vec = 1` reflects the
        virtual sink's `exp2(score - max) = exp2(0) = 1` contribution.
        Subsequent tiles update through the normal recurrence; the sink
        is rescaled implicitly as the running max grows. `scale_vec`
        stays at the 1 set by `__init__`.

        Args:
            sw_log2: The virtual sink token's weight in log2 units
                (`log2e * sink_weight`). Seeds `max_vec` and
                `max_vec_prev`.
        """
        self.max_vec = sw_log2
        self.max_vec_prev = sw_log2
        self.norm_vec = Float32(1.0)

    # ---- Reduction helpers (static methods) ----

    @staticmethod
    @always_inline
    def _col_reduce_at_j[
        layout: TensorLayout,
        //,
        op: StaticString,
        j: Int,
    ](src: RegTile[Self.att_dtype, layout, _],) -> Float32:
        """Reduces base-tile column `j` of `src` to a single `Float32`.
        `op` is `"max"` or `"add"`; selected at comptime so the
        unrolled body emits the chosen intrinsic directly.

        `src.dtype == Self.att_dtype` (FP32 for the BF16 attention path
        and FP8 KV<128; FP16 for the FP8 path's KV>=128 softmax,
        sub-step 9). In-lane reduction stays in `Self.att_dtype`;
        cross-warp DPP-swap runs in FP32 regardless of `att_dtype` so
        the row-state stays in FP32 throughout the online-softmax
        recurrence.

        SHARED PRIMITIVE: called by BOTH the FP16-default softmax path
        (`col_max_acc` / `seed_tile0` / `col_sum_acc`) and the gated
        FP32-scores path's `col_sum_acc`. The body here is the
        byte-for-byte HEAD codegen: the FP32-scores `v_max3_f32`-folded
        column-max variant lives in the separate `_col_max_scalar_v3max`
        so this primitive's codegen for the shipping default never
        moves."""
        comptime assert (
            op == "max" or op == "add"
        ), "_col_reduce_at_j: op must be 'max' or 'add'"

        comptime src_height = layout.static_shape[0]
        comptime base_tile_elts = layout.static_shape[2]
        var src_v = src.vectorize[1, 1, base_tile_elts]()
        comptime assert src_v.flat_rank == 3

        var simd_accum = src_v[0, j, 0]
        comptime for i in range(1, src_height):
            comptime if op == "max":
                simd_accum = max(simd_accum, src_v[i, j, 0])
            else:
                simd_accum = simd_accum + src_v[i, j, 0]

        var lane_val: Float32
        comptime if op == "max":
            lane_val = simd_accum.reduce_max().cast[.float32]()
        else:
            lane_val = simd_accum.reduce_add().cast[.float32]()

        # Combine the two half-warps sharing each column.
        var swapped = permlane_swap[32](lane_val, lane_val)
        comptime if op == "max":
            return max(swapped[0], swapped[1])
        else:
            return swapped[0] + swapped[1]

    @staticmethod
    @always_inline
    def _col_max_scalar_v3max[
        layout: TensorLayout,
        //,
    ](src: RegTile[Self.att_dtype, layout, _]) -> Float32:
        """`max(src[*, 0, *])` via a flat pair-folded running max so
        ISel fuses each pair into `v_max3_f32`. FP32-scores-only column
        max (used by `seed_tile0_scaled` / `col_max_acc_scaled`).

        WHY: the default `_col_reduce_at_j` folds the cross-`src_height`
        rows SIMD-wide (`v_max_f32 t, x, x` to canonicalize each lane +
        `v_max_f32 acc, t, acc`), so ISel emits ~2 plain `v_max_f32` per
        element-pair. Folding the flattened `(src_height x
        base_tile_elts)` per-lane element set two-at-a-time as
        `acc = max(max(s[p], s[p+1]), acc)` lets ISel collapse each pair
        into one `v_max3_f32`. The flat pair-fold lowers each pair to one
        `v_max3_f32` instead of two `v_max_f32`, VGPR-neutral.

        MATH: identical to `_col_max_scalar(src)`; `max` is associative
        + commutative so re-grouping the fold order is bit-exact (no
        NaN-payload concern; the FP32-scores tile holds finite scores).

        SEED: from the first per-lane element (`simd[0]`), so the
        leading pair is one plain `v_max_f32` and the rest fold into
        `v_max3_f32` (matches `_col_max_scalar`'s `simd_accum =
        src_v[0, j, 0]` seed; no `-inf` sentinel needed).

        TAIL: the cross-warp `permlane_swap[32]` combine is UNCHANGED:
        each half-warp holds the full reduced lane value, so the
        combine stays `max(swapped[0], swapped[1])`; each half-warp must
        contribute, so `max(swapped[0], lane)` would drop half the warp.
        Kept OUT of `_col_reduce_at_j` so the shared primitive's
        shipping FP16-default codegen stays byte-identical to HEAD."""
        comptime assert (
            src.static_shape[1] == 1
        ), "_col_max_scalar_v3max: requires width==1 col_l stripe"
        comptime src_height = layout.static_shape[0]
        comptime base_tile_elts = layout.static_shape[2]
        comptime assert (
            base_tile_elts >= 2 and base_tile_elts % 2 == 0
        ), "_col_max_scalar_v3max: needs an even per-lane element width"
        var src_v = src.vectorize[1, 1, base_tile_elts]()
        comptime assert src_v.flat_rank == 3

        # Seed from the first pair of row 0 (one plain `v_max_f32`),
        # then fold every remaining pair as `max3(s[p], s[p+1], acc)`.
        var simd_0 = src_v[0, 0, 0]
        var acc = max(
            simd_0[0].cast[.float32](),
            simd_0[1].cast[.float32](),
        )
        comptime for i in range(src_height):
            var simd_row = src_v[i, 0, 0]
            comptime start = 2 if i == 0 else 0
            comptime for p in range(start, base_tile_elts, 2):
                acc = max(
                    max(
                        simd_row[p].cast[.float32](),
                        simd_row[p + 1].cast[.float32](),
                    ),
                    acc,
                )

        var swapped = permlane_swap[32](acc, acc)
        return max(swapped[0], swapped[1])

    @staticmethod
    @always_inline
    def _col_max_scalar[
        layout: TensorLayout,
        //,
    ](src: RegTile[Self.att_dtype, layout, _]) -> Float32:
        """Returns `max(src[*, 0, *])`, single-column collapse since
        the col_l rt_32x32 has 1 column per warp's stripe."""
        comptime assert (
            src.static_shape[1] == 1
        ), "_col_max_scalar: requires width==1 col_l rt_32x32 stripe"
        return Self._col_reduce_at_j["max", j=0](src)

    @staticmethod
    @always_inline
    def _col_sum_scalar[
        layout: TensorLayout,
        //,
    ](src: RegTile[Self.att_dtype, layout, _]) -> Float32:
        """Returns `sum(src[*, 0, *])`, single-column collapse for the
        additive reduction in the online-softmax denominator update."""
        comptime assert (
            src.static_shape[1] == 1
        ), "_col_sum_scalar: requires width==1 col_l rt_32x32 stripe"
        return Self._col_reduce_at_j["add", j=0](src)

    @staticmethod
    @always_inline
    def _sub_scalar_inplace[
        layout: TensorLayout,
        //,
    ](
        mut dst: RegTile[Self.att_dtype, layout, MutUntrackedOrigin],
        scalar: Float32,
    ):
        """`dst -= scalar` per element: broadcast scalar across the
        per-lane fragment.

        For FP16 `dst`, the cast is exact for the common case (the
        value is a max over FP16-precision elements held in FP32
        storage, so re-casting to FP16 is a no-op)."""
        comptime src_height = dst.static_shape[0]
        comptime src_width = dst.static_shape[1]
        comptime base_tile_elts = dst.static_shape[2]
        comptime assert (
            src_width == 1
        ), "_sub_scalar_inplace: requires width==1 col_l rt_32x32 stripe"

        var dst_v = dst.vectorize[1, 1, base_tile_elts]()
        comptime assert dst_v.flat_rank == 3 and dst_v.origin.mut
        var v_simd = dst_v.ElementType(scalar.cast[Self.att_dtype]())
        comptime for i in range(src_height):
            dst_v[i, 0, 0] = dst_v[i, 0, 0] - v_simd

    @staticmethod
    @always_inline
    def _fms_scalar_inplace[
        layout: TensorLayout,
        //,
    ](
        mut dst: RegTile[Self.att_dtype, layout, MutUntrackedOrigin],
        mul: Float32,
        sub: Float32,
    ):
        """Fused `dst = mul * dst - sub` per element: broadcast both
        scalars across the per-lane fragment.

        Used by the FP32-in-place sequential softmax (`_FP32_SOFTMAX_SCORES`)
        to fold the QK `scale * log2(e)` factor INTO the max-subtract
        instead of into the QK epilogue (`att *= scale`) or the per-tile
        `exp2`. `dst` arrives RAW (un-scaled QK); `mul = scale * log2(e)`
        and `sub = max_vec` (already in the scaled log2 domain). Lowers to
        the same one `v_pk_fma_f32` per fragment as the plain
        `_sub_scalar_inplace`'s subtract (no extra op), so:
        - `col_max` / `sub_max` run in place on ONE 64-VGPR att copy (no
          transient scaled-att copy `v_pk_mul v[200:255]`), AND
        - `exp2` stays PLAIN `exp2(dst)`: no scale multiply on the
          transcendental critical path."""
        comptime src_height = dst.static_shape[0]
        comptime src_width = dst.static_shape[1]
        comptime base_tile_elts = dst.static_shape[2]
        comptime assert (
            src_width == 1
        ), "_fms_scalar_inplace: requires width==1 col_l rt_32x32 stripe"

        var dst_v = dst.vectorize[1, 1, base_tile_elts]()
        comptime assert dst_v.flat_rank == 3 and dst_v.origin.mut
        var mul_simd = dst_v.ElementType(mul.cast[Self.att_dtype]())
        var sub_simd = dst_v.ElementType(sub.cast[Self.att_dtype]())
        comptime for i in range(src_height):
            dst_v[i, 0, 0] = dst_v[i, 0, 0] * mul_simd - sub_simd

    @staticmethod
    @always_inline
    def _mul_scalar_inplace[
        dtype: DType
    ](mut dst: RegTile[dtype, ...], scalar: Float32,):
        """`dst *= scalar` per element: rescale step of online softmax
        (`o_reg *= exp2(max_prev - max_new)` when the running max
        grows). `dst` may be FP32 (the `o_reg` accumulator) or the att
        fragment dtype (`config.dtype`: BF16, or FP8 in the FP8 path;
        the pre-cast `att_bf16_full` consumed by the PV MFMA strips that
        need the same scale flip as `o_reg`; see #87284)."""
        comptime src_height = dst.static_shape[0]
        comptime src_width = dst.static_shape[1]
        comptime base_tile_elts = dst.static_shape[2]
        comptime assert (
            src_width == 1
        ), "_mul_scalar_inplace: requires width==1 col_l rt_32x32 stripe"

        var dst_v = dst.vectorize[1, 1, base_tile_elts]()
        comptime assert dst_v.flat_rank == 3 and dst_v.origin.mut
        var v_simd = dst_v.ElementType(scalar.cast[dtype]())
        comptime for i in range(src_height):
            dst_v[i, 0, 0] = dst_v[i, 0, 0] * v_simd

    @staticmethod
    @always_inline
    def _div_scalar_inplace(
        mut dst: RegTile[.float32, ...],
        scalar: Float32,
    ):
        """`dst /= scalar` per element: final `o_reg /= norm_vec`.

        Hand-lowered to `recip(scalar) * dst` per element to avoid the
        IEEE-correct fdiv sequence (`v_div_scale_f32` + `v_rcp_f32` +
        `v_div_fmas_f32` + `v_div_fixup_f32`). Used at the kernel
        epilogue so the FP32 output normalization overwrites `o_reg`
        instead of materializing a second O_LAYOUT FP32 tile (64
        VGPRs/lane, pushed MhaPrefillV2's FP8 KV=128 path over the
        128 VGPR/thread cap)."""
        comptime src_height = dst.static_shape[0]
        comptime src_width = dst.static_shape[1]
        comptime base_tile_elts = dst.static_shape[2]
        comptime assert (
            src_width == 1
        ), "_div_scalar_inplace: requires width==1 col_l rt_32x32 stripe"

        var dst_v = dst.vectorize[1, 1, base_tile_elts]()
        comptime assert dst_v.flat_rank == 3 and dst_v.origin.mut
        var v_inv = dst_v.ElementType(recip(scalar))
        comptime for i in range(src_height):
            dst_v[i, 0, 0] = dst_v[i, 0, 0] * v_inv

    # ---- Recurrence steps ----

    @always_inline
    def seed_tile0[
        layout: TensorLayout,
        //,
    ](
        mut self,
        mut att_block: RegTile[Self.att_dtype, layout, MutUntrackedOrigin],
    ):
        """Prologue tile-0 partial softmax setup.

        Three-step composite: `max_vec = col_max(att_block)`,
        `max_vec_prev = max_vec`, `att_block -= max_vec`. Called once
        after the first QK + mask, before the prologue's first-half
        `exp2`. There's no `col_max_acc` because there's no prior
        rowmax yet (no-sink path) or the sink contribution is already
        seeded into `max_vec` (sink path).

        Parameters:
            layout: `TensorLayout` of `att_block`. Drives the column
                reduction vectorization.

        Args:
            att_block: The first-tile attention score `RegTile` of
                `Self.att_dtype`. Reduced in place to subtract the
                rowmax.
        """
        self.max_vec = Self._col_max_scalar(att_block)
        self.max_vec_prev = self.max_vec
        Self._sub_scalar_inplace(att_block, self.max_vec)

    @always_inline
    def col_max_acc[
        layout: TensorLayout,
        //,
    ](
        mut self,
        att_block: RegTile[Self.att_dtype, layout, MutUntrackedOrigin],
    ):
        """Running rowmax: `max_vec = max(max_vec_prev, col_max(att_block))`.

        Caller maintains the `max_vec_prev = max_vec` shadow-write
        separately (via `lazy_rescale_decision` or
        `update_scale_unconditional`), splitting the shadow-write out
        lets the cluster fns interpose IGLP barriers between the
        rowmax update and the scale update without dragging
        `max_vec_prev` along.

        Parameters:
            layout: `TensorLayout` of `att_block`. Drives the column
                reduction vectorization.

        Args:
            att_block: The attention score `RegTile` of
                `Self.att_dtype` whose column maximum is folded into
                the running rowmax.
        """
        self.max_vec = max(self.max_vec_prev, Self._col_max_scalar(att_block))

    @always_inline
    def sub_max[
        layout: TensorLayout,
        //,
    ](
        mut self,
        mut att_block: RegTile[Self.att_dtype, layout, MutUntrackedOrigin],
    ):
        """`att_block -= max_vec` per element. Prepares `att_block` for
        the subsequent `exp2_inplace_range` call.

        Parameters:
            layout: `TensorLayout` of `att_block`. Drives the column
                reduction vectorization.

        Args:
            att_block: The attention score `RegTile` of `Self.att_dtype`
                whose elements are reduced in place by subtracting the
                running rowmax `max_vec`.
        """
        Self._sub_scalar_inplace(att_block, self.max_vec)

    @always_inline
    def seed_tile0_scaled[
        layout: TensorLayout,
        //,
    ](
        mut self,
        mut att_block: RegTile[Self.att_dtype, layout, MutUntrackedOrigin],
        log2_scale: Float32,
    ):
        """Scale-folded `seed_tile0` for the FP32-in-place sequential
        cadence (`_FP32_SOFTMAX_SCORES`). `att_block` arrives RAW (un-scaled
        QK); `log2_scale = scale * log2(e)`.

        `max_vec = log2_scale * col_max(raw)` (running max kept in the
        SCALED log2 domain, so `update_scale_unconditional`'s
        `exp2(prev - new)` rescale is byte-identical to the non-folded
        path), `max_vec_prev = max_vec`, then `att = log2_scale*att -
        max_vec` via one fused `v_pk_fma` per fragment. The `col_max`
        runs in place on the raw 64-VGPR att (no transient scaled copy)
        and the scale never touches the `exp2` critical path. Math
        identical: `scale > 0` ⇒ `max(scale·x) = scale·max(x)`.

        Uses `_col_max_scalar_v3max` (`v_max3_f32`-folded, FP32-scores-
        only) instead of `_col_max_scalar`; same column max, only the
        in-lane fold grouping changes so ISel emits `v_max3_f32`."""
        self.max_vec = log2_scale * Self._col_max_scalar_v3max(att_block)
        self.max_vec_prev = self.max_vec
        Self._fms_scalar_inplace(att_block, log2_scale, self.max_vec)

    @always_inline
    def col_max_acc_scaled[
        layout: TensorLayout,
        //,
    ](
        mut self,
        att_block: RegTile[Self.att_dtype, layout, MutUntrackedOrigin],
        log2_scale: Float32,
    ):
        """Scale-folded `col_max_acc`: `max_vec = max(max_prev,
        log2_scale * col_max(raw att))`. Keeps the running max in the
        scaled log2 domain while `att_block` stays RAW. See
        `seed_tile0_scaled`.

        Uses `_col_max_scalar_v3max` (`v_max3_f32`-folded) for the raw
        column max. The running-max fold-in stays the OUTER `max(...)`
        here rather than a seed into the chain: `max_prev` is in the
        SCALED log2 domain but the chain reduces RAW scores, so seeding
        the chain with `max_prev` would compare across domains. The
        `log2_scale * raw_max` then `max(max_prev, ...)` ordering keeps
        the rescale `exp2(prev - new)` byte-identical to the non-folded
        path; the outer `max` is one residual `v_max_f32` per tile.

        Parameters:
            layout: `TensorLayout` of `att_block`. Drives the column
                reduction vectorization.

        Args:
            att_block: The RAW (un-scaled) attention score `RegTile` of
                `Self.att_dtype` whose column maximum is scaled and
                folded into the running rowmax.
            log2_scale: The QK scale in log2 units (`scale * log2(e)`).
                Multiplies the raw column max to bring it into the
                scaled log2 domain before the fold-in.
        """
        self.max_vec = max(
            self.max_vec_prev,
            log2_scale * Self._col_max_scalar_v3max(att_block),
        )

    @always_inline
    def sub_max_scaled[
        layout: TensorLayout,
        //,
    ](
        mut self,
        mut att_block: RegTile[Self.att_dtype, layout, MutUntrackedOrigin],
        log2_scale: Float32,
    ):
        """Scale-folded `sub_max`: `att = log2_scale*att - max_vec` via one
        fused `v_pk_fma` per fragment (`max_vec` already scaled). Folds
        the QK scale into the max-subtract so `exp2` stays plain and
        `col_max` ran in place on raw att. See `seed_tile0_scaled`.

        Parameters:
            layout: `TensorLayout` of `att_block`. Drives the column
                reduction vectorization.

        Args:
            att_block: The RAW (un-scaled) attention score `RegTile` of
                `Self.att_dtype` reduced in place by the fused
                `log2_scale * att - max_vec` per element.
            log2_scale: The QK scale in log2 units (`scale * log2(e)`).
                Multiplies `att_block` element-wise before the
                `max_vec` subtract so `exp2` stays plain.
        """
        Self._fms_scalar_inplace(att_block, log2_scale, self.max_vec)

    @always_inline
    def lazy_rescale_decision[
        att_full_dtype: DType
    ](
        mut self,
        mut o_reg: RegTile[.float32, _, MutUntrackedOrigin],
        mut att_bf16_full: RegTile[att_full_dtype, _, MutUntrackedOrigin],
        threshold: Float32,
    ) -> Bool:
        """Lazy-rescale decision for main-loop C2/C6.

        Returns `True` iff the running max grew by more than
        `threshold` log2 units in this cluster, in which case
        `scale_vec = exp2(prev - new)` was applied to `o_reg` and
        `max_vec_prev` was shadow-updated. On the skip path (most
        clusters), rolls back `max_vec` to `max_vec_prev` and resets
        `scale_vec` to 1.

        Per-lane decision (each lane's column of the col_l rt_32x32
        fragment is an independent Q row; see this struct's docstring),
        not a wave-wide AND -- a warp-vote here would let one row's
        growth force a rescale onto a sibling row whose own growth
        didn't warrant it.

        SCALE_VEC INVARIANT: `scale_vec` is exactly 1 whenever no
        rescale fired in the most recent C2/C6. The else-branch reset
        is load-bearing: without it, a stale `scale_vec ≈ 1e-38`
        (from a non-Causal mask's sentinel-driven huge initial growth,
        where `math_exp2(-10_000) = 1.18e-38` is the smallest float32
        normal and does NOT flush to 0) gets re-applied 3× in the
        epilogue tail clusters → `norm_vec` flushes to 0 → final
        divide produces `Inf`.

        Parameters:
            att_full_dtype: DType of the `att_bf16_full` tile. Used to
                dispatch the element-wise rescale on the att fragment
                by the same `scale_vec` as `o_reg`.

        Args:
            o_reg: The FP32 output accumulator `RegTile`. Rescaled in
                place by `scale_vec` when the running max grew.
            att_bf16_full: The full attention score `RegTile` of
                `att_full_dtype`. Rescaled by the same `scale_vec` as
                `o_reg` so subsequent PV MFMA strips consume it at the
                post-rescale scale.
            threshold: Maximum allowed log2 growth in the running max
                before a rescale fires. When `max_vec - max_vec_prev`
                exceeds this, the rescale is applied.
        """
        # Per-lane predicate, not a warp vote: the col_l rt_32x32 fragment
        # gives each lane an INDEPENDENT Q row (one column of the 32x32
        # tile, per this struct's own docstring), so an AND across the
        # wavefront would let one row's growth force a rescale (a real,
        # non-identity `exp2` multiply of `o_reg`/`att_bf16_full`, plus a
        # `max_vec` state change that persists into later clusters) onto a
        # sibling row whose own growth didn't warrant it.
        var lane_ok = (self.max_vec - self.max_vec_prev) <= threshold
        var all_ok = lane_ok
        var pending_scale = False
        if unlikely(not all_ok):
            self.scale_vec = math_exp2(self.max_vec_prev - self.max_vec)
            Self._mul_scalar_inplace(o_reg, self.scale_vec)
            # #87284: rescale `att_bf16_full` by the SAME `scale_vec` as
            # `o_reg`. Strips 1..3 (issued after this in the caller)
            # consume `att_bf16_full` at the post-rescale scale,
            # consistent with the rescaled `o_reg`. Without this they
            # over-contribute at the old scale — a bounded artifact that
            # corrupts wide-dynamic-range attention (FLUX NullMask
            # no-QK-norm prefill).
            Self._mul_scalar_inplace(att_bf16_full, self.scale_vec)
            self.max_vec_prev = self.max_vec
            pending_scale = True
        else:
            self.max_vec = self.max_vec_prev
            self.scale_vec = Float32(1.0)
        return pending_scale

    @always_inline
    def update_scale_unconditional(mut self):
        """UNCONDITIONAL rescale: `scale_vec = exp2(max_prev - max_new)`,
        then `max_vec_prev = max_vec`.

        Used by epilogue tail/full softmax and by
        `_pv_whole_with_partial_softmax` where rescale always fires
        (no lazy-rescale skip). Caller is responsible for the
        `rescale_output(o_reg)` step AFTER any IGLP barriers; this
        method does not touch `o_reg` so the cluster fn can interleave
        the rescale with PV MFMAs."""
        self.scale_vec = math_exp2(self.max_vec_prev - self.max_vec)
        self.max_vec_prev = self.max_vec

    @always_inline
    def rescale_output(
        mut self,
        mut o_reg: RegTile[.float32, _, MutUntrackedOrigin],
    ):
        """`o_reg *= scale_vec` per element.

        Used by `_pv_whole_with_partial_softmax` (after the IGLP
        barrier separating PV MFMAs from VALU work) and the final
        Epi-C10 step before the divide.

        Args:
            o_reg: The FP32 output accumulator `RegTile`. Multiplied in
                place by the pending `scale_vec`.
        """
        Self._mul_scalar_inplace(o_reg, self.scale_vec)

    @always_inline
    def apply_norm_rescale_if_pending(mut self, pending_scale: Bool):
        """`if pending_scale: norm_vec *= scale_vec`.

        Used by the C0/C4 tail softmax to roll forward a lazy rescale
        that fired in the previous C2/C6.

        Args:
            pending_scale: Whether a lazy rescale fired in the previous
                C2/C6 cluster. When `True`, `norm_vec` is multiplied by
                the pending `scale_vec`.
        """
        if pending_scale:
            self.norm_vec = self.norm_vec * self.scale_vec

    @always_inline
    def apply_unconditional_norm_rescale(mut self):
        """UNCONDITIONAL `norm_vec *= scale_vec`.

        Used by the epilogue tail/full softmax. Relies on the
        `scale_vec == 1` invariant when no rescale fired
        (`lazy_rescale_decision` maintains this on the skip branch)."""
        self.norm_vec = self.norm_vec * self.scale_vec

    @always_inline
    def col_sum_acc[
        layout: TensorLayout,
        //,
    ](
        mut self,
        att_block: RegTile[Self.att_dtype, layout, MutUntrackedOrigin],
    ):
        """Running denominator: `norm_vec += sum(att_block, axis=row)`.

        Parameters:
            layout: `TensorLayout` of `att_block`. Drives the column
                reduction vectorization.

        Args:
            att_block: The attention score `RegTile` of
                `Self.att_dtype` (already max-subtracted and `exp2`'d)
                whose column sum is accumulated into `norm_vec`.
        """
        self.norm_vec = self.norm_vec + Self._col_sum_scalar(att_block)

    @always_inline
    def normalize_output(
        mut self,
        mut o_reg: RegTile[.float32, _, MutUntrackedOrigin],
    ):
        """Final `o_reg /= norm_vec` in place.

        Avoids materializing a second FP32 O_LAYOUT tile (64 VGPRs/lane,
        the combined live set with `o_reg` pushed FP8 KV=128 over
        the 128 VGPR/thread cap and spilled 9 VGPR-equivalents to
        scratch in Epi-C12). Used at the very end of the kernel.

        Args:
            o_reg: The FP32 output accumulator `RegTile`. Divided in
                place by `norm_vec` to produce the final normalized
                output.
        """
        Self._div_scalar_inplace(o_reg, self.norm_vec)
