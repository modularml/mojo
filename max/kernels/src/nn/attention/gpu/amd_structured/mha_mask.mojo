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
"""Mask functor application for the `MhaPrefillV2` `att_block`.

`MaskApplier[mask_t, Q_BLOCK_SIZE, KV_BLOCK_SIZE]` (below) bundles the
mask functor with the comptime block sizes and exposes a single
`apply()` entry that comptime-dispatches over the mask type:

- `NullMask`: comptime-elided no-op. The mask trait would always
  report `NO_MASK`, so the entry is statically dead.
- `CausalMask`: 16-wide SIMD fast path (one `v_cmp` + one
  `v_cndmask` per stripe), generalized for `start_pos` so the causal
  cap moves with the cache start position. Gated on the runtime
  `q_start_pos < kv_end_pos` shortcut so fully-unmasked tiles bypass
  the work entirely.
- Anything else (`SlidingWindowCausalMask`, `ChunkedCausalMask`,
  `MaterializedMask`, fused combinations): runtime
  `mask_functor.status(...)` dispatch over `NO_MASK` (return),
  `FULL_MASK` (fill `-inf`), `PARTIAL` (per-element loop calling
  `mask_functor.mask(coord, score)` over the 16 fragment slots).

Per-element row-within-tile mapping comes from the
`v_mfma_f32_32x32x16_bf16` accumulator fragment geometry; see
`MhaMmaOp.ACC_ROW_OFFSETS_32x32`.
"""

from std.sys.intrinsics import unlikely
from std.utils.index import IndexList

from layout import TensorLayout
from nn.attention.mha_mask import CausalMask, MHAMask, NullMask, TileMaskStatus
from structured_kernels.amd_tile_io import RegTile

from .mha_mma_op import ACC_ROW_OFFSETS_32x32


@always_inline
def _apply_causal_mask_fast[
    T_dst: DType,
    layout: TensorLayout,
    //,
    Q_BLOCK_SIZE: Int,
    KV_BLOCK_SIZE: Int,
](
    mut dst: RegTile[T_dst, layout, MutUntrackedOrigin],
    q_tile_idx: Int32,
    k_tile_idx: Int32,
    start_pos: Int32,
    lane: Int32,
):
    """Causal mask fast path. One `v_cmp` + one `v_cndmask` per stripe.

    Identical structure to the prior `mask_kv_tile` but with
    `start_pos` baked into the Q position so the causal cap moves
    with the cache start position for prefill-with-cache.

    `T_dst` is FP32 for the BF16 attention path and BF16 for the FP8
    attention path's BF16 softmax (sub-step 8). The `-inf` sentinel
    is cast to `T_dst` once at function entry; the per-stripe
    `v_cndmask` then runs in `T_dst` (one mask register reused
    across the unrolled stripes)."""
    comptime dst_height = layout.static_shape[0]
    comptime dst_width = layout.static_shape[1]
    comptime base_tile_elts = layout.static_shape[2]

    comptime assert base_tile_elts == 16, (
        "_apply_causal_mask_fast: requires base_tile_elts == 16 (col_l"
        " rt_32x32 fragment)"
    )

    var _NEG_INF_VEC = SIMD[T_dst, 16](Float32(-3.4028235e38).cast[T_dst]())
    comptime _ROW_OFFSETS = ACC_ROW_OFFSETS_32x32

    var col_in_tile = lane & Int32(31)
    var row_extra = (lane >> Int32(5)) << Int32(2)
    var q_pos = q_tile_idx * Int32(Q_BLOCK_SIZE) + col_in_tile + start_pos
    var k_base = k_tile_idx * Int32(KV_BLOCK_SIZE)

    var dst_vec = dst.vectorize[1, 1, 16]()
    comptime assert dst_vec.flat_rank == 3

    comptime for i in range(dst_height):
        # `rel < _ROW_OFFSETS[p]` is the per-element form of `k_pos > q_pos`
        # once the per-element row offset is folded in.
        var rel = q_pos - (k_base + Int32(i * 32) + row_extra)
        var mask = SIMD[.int32, 16](rel).lt(_ROW_OFFSETS)
        comptime for j in range(dst_width):
            dst_vec[i, j, 0] = mask.select(_NEG_INF_VEC, dst_vec[i, j, 0])


@always_inline
def _apply_kbound_mask_fast[
    T_dst: DType,
    layout: TensorLayout,
    //,
    KV_BLOCK_SIZE: Int,
](
    mut dst: RegTile[T_dst, layout, MutUntrackedOrigin],
    k_tile_idx: Int32,
    num_keys: Int32,
    lane: Int32,
):
    """Key-bound mask fast path: excludes columns with `k_pos >= num_keys`.

    Used by `NullMask` to exclude a partial last K tile's OOB columns
    AND the phantom even-parity padding tile (see the `MhaPrefillV2`
    main loop's `max_num_tiles_local` even round-up). Both carry score
    `Q@0 = 0` from the SRD-clamp-zeroed K (competitive for normalized
    inputs) and would steal softmax denominator mass with a zero
    numerator. This sends them to `-inf` so the subsequent `exp2` zeros
    them.

    Structural analog of `_apply_causal_mask_fast` (one `v_cmp` + one
    `v_cndmask` per stripe), but the bound depends only on `k_pos`, not
    the q position:
        `k_pos >= num_keys`
        <=> `ROW_OFFSETS[p] >= num_keys - (k_base + i*32 + row_extra)`.

    `T_dst` is FP32 for the BF16 attention path and BF16 for the FP8
    attention path's BF16 softmax."""
    comptime dst_height = layout.static_shape[0]
    comptime dst_width = layout.static_shape[1]
    comptime base_tile_elts = layout.static_shape[2]

    comptime assert base_tile_elts == 16, (
        "_apply_kbound_mask_fast: requires base_tile_elts == 16 (col_l"
        " rt_32x32 fragment)"
    )

    var _NEG_INF_VEC = SIMD[T_dst, 16](Float32(-3.4028235e38).cast[T_dst]())
    comptime _ROW_OFFSETS = ACC_ROW_OFFSETS_32x32

    var row_extra = (lane >> Int32(5)) << Int32(2)
    var k_base = k_tile_idx * Int32(KV_BLOCK_SIZE)

    var dst_vec = dst.vectorize[1, 1, 16]()
    comptime assert dst_vec.flat_rank == 3

    comptime for i in range(dst_height):
        # `bound <= _ROW_OFFSETS[p]` is the per-element form of
        # `k_pos >= num_keys` once the per-element row offset is folded in.
        var bound = num_keys - (k_base + Int32(i * 32) + row_extra)
        var mask = SIMD[.int32, 16](bound).le(_ROW_OFFSETS)
        comptime for j in range(dst_width):
            dst_vec[i, j, 0] = mask.select(_NEG_INF_VEC, dst_vec[i, j, 0])


@always_inline
def _fill_dst_neg_inf[
    T_dst: DType,
    layout: TensorLayout,
    //,
](mut dst: RegTile[T_dst, layout, MutUntrackedOrigin]):
    """FULL_MASK path: every fragment slot to `-inf`. Subsequent
    `exp2` sends them all to zero. Cheaper than `mask_functor.mask`
    when the entire tile is masked out.

    `T_dst` is FP32 for the BF16 attention path and BF16 for the FP8
    attention path's BF16 softmax (sub-step 8)."""
    comptime dst_height = layout.static_shape[0]
    comptime dst_width = layout.static_shape[1]
    var _NEG_INF_VEC = SIMD[T_dst, 16](Float32(-3.4028235e38).cast[T_dst]())
    var dst_vec = dst.vectorize[1, 1, 16]()
    comptime assert dst_vec.flat_rank == 3
    comptime for i in range(dst_height):
        comptime for j in range(dst_width):
            dst_vec[i, j, 0] = _NEG_INF_VEC


@always_inline
def _apply_mask_generic[
    mask_t: MHAMask,
    T_dst: DType,
    layout: TensorLayout,
    //,
    Q_BLOCK_SIZE: Int,
    KV_BLOCK_SIZE: Int,
](
    mask_functor: mask_t,
    mut dst: RegTile[T_dst, layout, MutUntrackedOrigin],
    q_tile_idx: Int32,
    k_tile_idx: Int32,
    start_pos: Int32,
    head_idx: UInt32,
    batch_idx: UInt32,
    lane: Int32,
):
    """Generic per-element mask path. Calls `mask_functor.mask(coord,
    score)` 16 times per fragment stripe with width-1 SIMD. Used
    when the mask is not `CausalMask` / `NullMask`; the compiler
    inlines `mask` and vectorizes back where possible.

    `T_dst` is FP32 for the BF16 attention path and BF16 for the FP8
    attention path's BF16 softmax (sub-step 8). The mask functor
    operates on FP32 scores; per-element values are promoted to FP32
    on read and cast back to `T_dst` on write."""
    comptime dst_height = layout.static_shape[0]
    comptime dst_width = layout.static_shape[1]

    var col_in_tile = lane & Int32(31)
    var row_extra = (lane >> Int32(5)) << Int32(2)
    var q_pos = q_tile_idx * Int32(Q_BLOCK_SIZE) + col_in_tile + start_pos
    var k_base = k_tile_idx * Int32(KV_BLOCK_SIZE)

    var dst_vec = dst.vectorize[1, 1, 16]()
    comptime assert dst_vec.flat_rank == 3
    comptime for i in range(dst_height):
        comptime for j in range(dst_width):
            var frag = dst_vec[i, j, 0]
            comptime for k_local in range(16):
                var k_pos = (
                    k_base
                    + Int32(i * 32)
                    + row_extra
                    + Int32(ACC_ROW_OFFSETS_32x32[k_local])
                )
                var score = Float32(frag[k_local].cast[.float32]())
                var masked = mask_functor.mask(
                    IndexList[4, element_type=.uint32](
                        Int(batch_idx),
                        Int(head_idx),
                        Int(q_pos),
                        Int(k_pos),
                    ),
                    score,
                )
                frag[k_local] = masked[0].cast[T_dst]()
            dst_vec[i, j, 0] = frag


# ===----------------------------------------------------------------------=== #
# MaskApplier
# ===----------------------------------------------------------------------=== #


struct MaskApplier[
    mask_t: MHAMask,
    //,
    Q_BLOCK_SIZE: Int,
    KV_BLOCK_SIZE: Int,
]:
    """Mask functor + comptime block-size bundle for `MhaPrefillV2`.

    Owns the runtime mask functor and exposes a single `apply()` entry
    that comptime-dispatches over `mask_t`. The dispatch consolidates
    what was previously a two-level hop (`MhaPrefillV2._maybe_apply_mask`
    → `apply_mask_to_att_block`); after `@always_inline` both layers
    fold into the same set of branches, so the consolidated form is
    codegen-identical while being one level less to follow.

    The struct is light: a single `mask_functor` field. The comptime
    block sizes are parameters (not fields) so the dispatch arithmetic
    folds to literal Ints at instantiation.

    Parameters:
        mask_t: The mask functor type (any `MHAMask`).
        Q_BLOCK_SIZE: Q rows per tile (`MhaConfigV2.q_block_size`).
        KV_BLOCK_SIZE: K rows per tile (`MhaConfigV2.kv_block`).
    """

    var mask_functor: Self.mask_t

    @always_inline
    def __init__(out self, mask_functor: Self.mask_t):
        """Bundle the mask functor. Comptime block sizes come from the
        struct's parameters.

        Args:
            mask_functor: Mask functor instance to bundle; its type
                drives the comptime dispatch in `apply`.
        """
        self.mask_functor = mask_functor

    @always_inline
    def apply[
        T_dst: DType,
        layout: TensorLayout,
        //,
    ](
        self,
        mut att_block: RegTile[T_dst, layout, MutUntrackedOrigin],
        q_tile_idx: Int,
        k_tile_idx: Int,
        start_pos: Int,
        head_idx: UInt32,
        batch_idx: UInt32,
        lane: Int,
        num_keys: Int = -1,
    ):
        """Mask an `att_block` tile, comptime-dispatching on `mask_t`.

        - `NullMask` → key-bound fast path (`_apply_kbound_mask_fast`)
          on tiles whose K range extends past `num_keys` (partial last
          tile + phantom even-parity tile); fully-valid tiles are a
          branch-cheap no-op.
        - `CausalMask` → runtime `q_start_pos < kv_end_pos` shortcut
          (most non-trailing tiles bypass mask work entirely), then
          the 16-wide SIMD fast path with `start_pos` shift.
        - Any other `MHAMask` (`SlidingWindowCausalMask`,
          `ChunkedCausalMask`, `MaterializedMask`, fused
          combinations) → runtime `mask_functor.status(...)`
          dispatch over `NO_MASK` (return), `FULL_MASK` (fill `-inf`,
          subsequent `exp2` zeros every entry), `PARTIAL` (per-element
          `mask_functor.mask(coord, score)` loop over the 16 fragment
          slots). The runtime `status()` call + enum branching adds a
          few SGPR ops per tile; the per-tile cost is acceptable for
          masks without a comptime-specialized fast path. Production
          callers: Gemma-3 (sliding window), Gemma-4 (chunked).

        Parameters:
            T_dst: Destination dtype of the `att_block` tile; FP32 for
                the BF16 attention path and BF16 for the FP8 path's
                BF16 softmax.
            layout: `TensorLayout` of the `att_block` register tile;
                its static shape drives the per-stripe unroll counts.

        Args:
            att_block: Attention block tile (mutated in place).
            q_tile_idx: Absolute Q tile index (this warp's first row /
                `Q_BLOCK_SIZE`; usually `block_tile_idx * NUM_WARPS + w_id`).
            k_tile_idx: Absolute K tile index.
            start_pos: KV-cache start position. Shifts the Q absolute
                position by `start_pos` to account for cache reuse.
            head_idx: Q head index (`block_idx.x` in the kernel).
            batch_idx: Batch index (`block_idx.z` in the kernel).
            lane: `lane_id()` cast to `Int`.
            num_keys: Runtime K/V sequence length for the NullMask
                kbound; `-1` (the default) disables it; MLA callers
                whose masks don't need the bound omit it.
        """
        comptime if Self.mask_t == NullMask:
            # #87603: exclude any tile whose K range extends past
            # `num_keys` — a partial last tile (`num_keys % KV_BLOCK !=
            # 0`) AND the phantom even-parity padding tile (see the main
            # loop's `max_num_tiles_local` round-up). Both carry
            # SRD-clamp-zeroed columns scoring `Q@0 = 0`; excluding them
            # keeps the softmax denominator from being diluted. Guarded
            # so fully-valid tiles stay branch-cheap.
            var kv_end_pos = (k_tile_idx + 1) * Self.KV_BLOCK_SIZE
            if num_keys >= 0 and unlikely(kv_end_pos > num_keys):
                _apply_kbound_mask_fast[KV_BLOCK_SIZE=Self.KV_BLOCK_SIZE](
                    att_block,
                    Int32(k_tile_idx),
                    Int32(num_keys),
                    Int32(lane),
                )
        elif Self.mask_t == CausalMask:
            var q_start_pos = q_tile_idx * Self.Q_BLOCK_SIZE + start_pos
            var kv_end_pos = (k_tile_idx + 1) * Self.KV_BLOCK_SIZE
            if unlikely(q_start_pos < kv_end_pos):
                _apply_causal_mask_fast[
                    Q_BLOCK_SIZE=Self.Q_BLOCK_SIZE,
                    KV_BLOCK_SIZE=Self.KV_BLOCK_SIZE,
                ](
                    att_block,
                    Int32(q_tile_idx),
                    Int32(k_tile_idx),
                    Int32(start_pos),
                    Int32(lane),
                )
        else:
            var status = self.mask_functor.status(
                batch_idx,
                IndexList[2, element_type=.uint32](
                    Int(q_tile_idx * Self.Q_BLOCK_SIZE + start_pos),
                    Int(k_tile_idx * Self.KV_BLOCK_SIZE),
                ),
                IndexList[2, element_type=.uint32](
                    Int(Self.Q_BLOCK_SIZE), Int(Self.KV_BLOCK_SIZE)
                ),
            )
            if status == TileMaskStatus.FULL_MASK:
                _fill_dst_neg_inf(att_block)
                return
            if status == TileMaskStatus.NO_MASK:
                return
            _apply_mask_generic[
                Q_BLOCK_SIZE=Self.Q_BLOCK_SIZE,
                KV_BLOCK_SIZE=Self.KV_BLOCK_SIZE,
            ](
                self.mask_functor,
                att_block,
                Int32(q_tile_idx),
                Int32(k_tile_idx),
                Int32(start_pos),
                head_idx,
                batch_idx,
                Int32(lane),
            )
